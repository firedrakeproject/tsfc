"""Utilities for building an Impero AST from an ordered list of
terminal Impero operations, and for building any additional data
required for straightforward C code generation.

What this module does is independent of the generated code target.
"""

import collections
from functools import singledispatch
from itertools import chain, groupby

from gem.node import traversal, collect_refcount
from gem import gem, impero as imp, optimise, scheduling


# ImperoC is named tuple for C code generation.
#
# Attributes:
#     tree        - Impero AST describing the loop structure and operations
#     temporaries - List of GEM expressions which have assigned temporaries
#     declare     - Where to declare temporaries to get correct C code
#     indices     - Indices for declarations and referencing values
ImperoC = collections.namedtuple('ImperoC', ['tree', 'temporaries', 'declare', 'indices'])


class NoopError(Exception):
    """No operations in the kernel."""
    pass


def preprocess_gem(expressions, replace_delta=True, remove_componenttensors=True):
    """Lower GEM nodes that cannot be translated to C directly."""
    if remove_componenttensors:
        expressions = optimise.remove_componenttensors(expressions)
    if replace_delta:
        expressions = optimise.replace_delta(expressions)
    return expressions


def compile_gem(assignments, prefix_ordering, remove_zeros=False,
                emit_return_accumulate=True):
    """Compiles GEM to Impero.

    :arg assignments: list of (return variable, expression DAG root) pairs
    :arg prefix_ordering: outermost loop indices
    :arg remove_zeros: remove zero assignment to return variables
    :arg emit_return_accumulate: emit ReturnAccumulate nodes (see
         :func:`~.scheduling.emit_operations`)? If False,
         split into Accumulate/Return pairs. Set to False if the
         output tensor of kernels is not guaranteed to be zero on entry.
    """
    # Remove zeros
    if remove_zeros:
        def nonzero(assignment):
            variable, expression = assignment
            return not isinstance(expression, gem.Zero)
        assignments = list(filter(nonzero, assignments))

    # Just the expressions
    expressions = [expression for variable, expression in assignments]

    # Collect indices in a deterministic order
    indices = list(collections.OrderedDict.fromkeys(chain.from_iterable(
        node.index_ordering()
        for node in traversal(expressions)
        if isinstance(node, (gem.Indexed, gem.FlexiblyIndexed))
    )))

    # Build ordered index map
    index_ordering = make_prefix_ordering(indices, prefix_ordering)
    apply_ordering = make_index_orderer(index_ordering)

    get_indices = lambda expr: apply_ordering(expr.free_indices)

    # Build operation ordering
    ops = scheduling.emit_operations(assignments, get_indices, emit_return_accumulate)

    # Empty kernel
    if len(ops) == 0:
        raise NoopError()

    # Drop unnecessary temporaries
    ops = inline_temporaries(expressions, ops)

    # Build Impero AST
    tree = make_loop_tree(ops, get_indices)

    # Collect temporaries
    temporaries = collect_temporaries(tree)

    # Determine declarations
    declare, indices = place_declarations(tree, temporaries, get_indices)

    # Prepare ImperoC (Impero AST + other data for code generation)
    return ImperoC(tree, temporaries, declare, indices)


def make_prefix_ordering(indices, prefix_ordering):
    """Creates an ordering of ``indices`` which starts with those
    indices in ``prefix_ordering``."""
    # Need to return deterministically ordered indices
    return tuple(prefix_ordering) + tuple(k for k in indices if k not in prefix_ordering)


def make_index_orderer(index_ordering):
    """Returns a function which given a set of indices returns those
    indices in the order as they appear in ``index_ordering``."""
    idx2pos = {idx: pos for pos, idx in enumerate(index_ordering)}

    def apply_ordering(indices):
        return tuple(sorted(indices, key=lambda i: idx2pos[i]))
    return apply_ordering


def inline_temporaries(expressions, ops):
    """Inline temporaries which could be inlined without blowing up
    the code.

    :arg expressions: a multi-root GEM expression DAG, used for
                      reference counting
    :arg ops: ordered list of Impero terminals
    :returns: a filtered ``ops``, without the unnecessary
              :class:`impero.Evaluate`s
    """
    refcount = collect_refcount(expressions)

    candidates = set()  # candidates for inlining
    for op in ops:
        if isinstance(op, imp.Evaluate):
            expr = op.expression
            if expr.shape == () and refcount[expr] == 1:
                candidates.add(expr)

    # Prevent inlining that pulls expressions into inner loops
    for node in traversal(expressions):
        for child in node.children:
            if child in candidates and set(child.free_indices) < set(node.free_indices):
                candidates.remove(child)

    # Filter out candidates
    return [op for op in ops if not (isinstance(op, imp.Evaluate) and op.expression in candidates)]


def collect_temporaries(tree):
    """Collects GEM expressions to assign to temporaries from a list
    of Impero terminals."""
    result = []
    for node in traversal((tree,)):
        # IndexSum temporaries should be added either at Initialise or
        # at Accumulate.  The difference is only in ordering
        # (numbering).  We chose Accumulate here.
        if isinstance(node, imp.Accumulate):
            result.append(node.indexsum)
        elif isinstance(node, imp.Evaluate):
            result.append(node.expression)
    return result


def make_loop_tree(ops, get_indices, level=0):
    """Creates an Impero AST with loops from a list of operations and
    their respective free indices.

    :arg ops: a list of Impero terminal nodes
    :arg get_indices: callable mapping from GEM nodes to an ordering
                      of free indices
    :arg level: depth of loop nesting
    :returns: Impero AST with loops, without declarations
    """
    keyfunc = lambda op: op.loop_shape(get_indices)[level:level+1]
    statements = []
    for first_index, op_group in groupby(ops, keyfunc):
        if first_index:
            inner_block = make_loop_tree(op_group, get_indices, level+1)
            statements.append(imp.For(first_index[0], inner_block))
        else:
            statements.extend(op_group)
    # Remove no-op terminals from the tree
    statements = [s for s in statements if not isinstance(s, imp.Noop)]
    return imp.Block(statements)


def place_declarations(tree, temporaries, get_indices):
    """Determines where and how to declare temporaries for an Impero AST.

    :arg tree: Impero AST to determine the declarations for
    :arg temporaries: list of GEM expressions which are assigned to
                      temporaries
    :arg get_indices: callable mapping from GEM nodes to an ordering
                      of free indices
    """
    numbering = {t: n for n, t in enumerate(temporaries)}
    assert len(numbering) == len(temporaries)

    # Collect the total number of temporary references
    total_refcount = collections.Counter()
    for node in traversal((tree,)):
        if isinstance(node, imp.Terminal):
            total_refcount.update(temp_refcount(numbering, node))
    assert set(total_refcount) == set(temporaries)

    # Result
    declare = {}
    indices = {}

    @singledispatch
    def recurse(expr, loop_indices):
        """Visit an Impero AST to collect declarations.

        :arg expr: Impero tree node
        :arg loop_indices: loop indices (in order) from the outer
                           loops surrounding ``expr``
        :returns: :class:`collections.Counter` with the reference
                  counts for each temporary in the subtree whose root
                  is ``expr``
        """
        return AssertionError("unsupported expression type %s" % type(expr))

    @recurse.register(imp.Terminal)
    def recurse_terminal(expr, loop_indices):
        return temp_refcount(numbering, expr)

    @recurse.register(imp.For)
    def recurse_for(expr, loop_indices):
        return recurse(expr.children[0], loop_indices + (expr.index,))

    @recurse.register(imp.Block)
    def recurse_block(expr, loop_indices):
        # Temporaries declared at the beginning of the block are
        # collected here
        declare[expr] = []

        # Collect reference counts for the block
        refcount = collections.Counter()
        for statement in expr.children:
            refcount.update(recurse(statement, loop_indices))

        # Visit :class:`collections.Counter` in deterministic order
        for e in sorted(refcount.keys(), key=lambda t: numbering[t]):
            if refcount[e] == total_refcount[e]:
                # If all references are within this block, then this
                # block is the right place to declare the temporary.
                assert loop_indices == get_indices(e)[:len(loop_indices)]
                indices[e] = get_indices(e)[len(loop_indices):]
                if indices[e]:
                    # Scalar-valued temporaries are not declared until
                    # their value is assigned.  This does not really
                    # matter, but produces a more compact and nicer to
                    # read C code.
                    declare[expr].append(e)
                # Remove expression from the ``refcount`` so it will
                # not be declared again.
                del refcount[e]
        return refcount

    # Populate result
    remainder = recurse(tree, ())
    assert not remainder

    # Set in ``declare`` for Impero terminals whether they should
    # declare the temporary that they are writing to.
    for node in traversal((tree,)):
        if isinstance(node, imp.Terminal):
            declare[node] = False
            if isinstance(node, imp.Evaluate):
                e = node.expression
            elif isinstance(node, imp.Initialise):
                e = node.indexsum
            else:
                continue

            if len(indices[e]) == 0:
                declare[node] = True

    return declare, indices


def temp_refcount(temporaries, op):
    """Collects the number of times temporaries are referenced when
    generating code for an Impero terminal.

    :arg temporaries: set of temporaries
    :arg op: Impero terminal
    :returns: :class:`collections.Counter` object mapping some of
               elements from ``temporaries`` to the number of times
               they will referenced from ``op``
    """
    counter = collections.Counter()

    def recurse(o):
        """Traverses expression until reaching temporaries, counting
        temporary references."""
        if o in temporaries:
            counter[o] += 1
        else:
            for c in o.children:
                recurse(c)

    def recurse_top(o):
        """Traverses expression until reaching temporaries, counting
        temporary references. Always descends into children at least
        once, even when the root is a temporary."""
        if o in temporaries:
            counter[o] += 1
        for c in o.children:
            recurse(c)

    if isinstance(op, imp.Initialise):
        counter[op.indexsum] += 1
    elif isinstance(op, imp.Accumulate):
        recurse_top(op.indexsum)
    elif isinstance(op, imp.Evaluate):
        recurse_top(op.expression)
    elif isinstance(op, imp.Return):
        recurse(op.expression)
    elif isinstance(op, imp.ReturnAccumulate):
        recurse(op.indexsum.children[0])
    elif isinstance(op, imp.Noop):
        pass
    else:
        raise AssertionError("unhandled operation: %s" % type(op))

    return counter
