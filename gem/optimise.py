"""A set of routines implementing various transformations on GEM
expressions."""

from __future__ import absolute_import, print_function, division
from six import itervalues, iteritems, iterkeys
from six.moves import map, zip

from collections import OrderedDict, deque
from functools import reduce
from itertools import permutations, count
from cached_property import cached_property

import numpy
from singledispatch import singledispatch

from gem.node import (Memoizer, MemoizerArg, reuse_if_untouched, reuse_if_untouched_arg,
                      traversal)

from gem.gem import (Node, Terminal, Failure, Identity, Literal, Zero, Power,
                     Product, Sum, Comparison, Conditional, Index, Constant,
                     VariableIndex, Indexed, FlexiblyIndexed, Variable,
                     IndexSum, ComponentTensor, ListTensor, Delta,
                     partial_indexed, one, Division, MathFunction, LogicalAnd,
                     LogicalNot, LogicalOr)


@singledispatch
def literal_rounding(node, self):
    """Perform FFC rounding of FIAT tabulation matrices on the literals of
    a GEM expression.

    :arg node: root of the expression
    :arg self: function for recursive calls
    """
    raise AssertionError("cannot handle type %s" % type(node))


literal_rounding.register(Node)(reuse_if_untouched)


@literal_rounding.register(Literal)
def literal_rounding_literal(node, self):
    table = node.array
    epsilon = self.epsilon
    # Mimic the rounding applied at COFFEE formatting, which in turn
    # mimics FFC formatting.
    one_decimal = numpy.round(table, 1)
    one_decimal[numpy.logical_not(one_decimal)] = 0  # no minus zeros
    return Literal(numpy.where(abs(table - one_decimal) < epsilon, one_decimal, table))


def ffc_rounding(expression, epsilon):
    """Perform FFC rounding of FIAT tabulation matrices on the literals of
    a GEM expression.

    :arg expression: GEM expression
    :arg epsilon: tolerance limit for rounding
    """
    mapper = Memoizer(literal_rounding)
    mapper.epsilon = epsilon
    return mapper(expression)


@singledispatch
def _replace_div(node, self):
    """Replace division with multiplication

    :param node: root of expression
    :param self: function for recursive calls
    """
    raise AssertionError("cannot handle type %s" % type(node))


_replace_div.register(Node)(reuse_if_untouched)


@_replace_div.register(Division)
def _replace_div_division(node, self):
    a, b = node.children
    return Product(self(a), Division(one, self(b)))


def replace_division(expressions):
    """Replace divisions with multiplications in expressions"""
    mapper = Memoizer(_replace_div)
    return list(map(mapper, expressions))


@singledispatch
def _reassociate_product(node, self):
    """Rearrange sequence of chain of products in increasing order of node rank.
     For example, the product ::

        a*b[i]*c[i][j]*d

    are reordered as ::

        a*d*b[i]*c[i][j]

    :param node: root of expression
    :return: reassociated product node
    """
    raise AssertionError("cannot handle type %s" % type(node))


_reassociate_product.register(Node)(reuse_if_untouched)


@_reassociate_product.register(Product)
def _reassociate_product_prod(node, self):
    # collect all factors of product, sort by rank
    comp_func = lambda x: len(x.free_indices)
    factors = sorted(collect_terms(node, Product), key=comp_func)
    return reduce(Product, map(self, factors))


def reassociate_product(expressions):
    mapper = Memoizer(_reassociate_product)
    return list(map(mapper, expressions))


@singledispatch
def replace_indices(node, self, subst):
    """Replace free indices in a GEM expression.

    :arg node: root of the expression
    :arg self: function for recursive calls
    :arg subst: tuple of pairs; each pair is a substitution
                rule with a free index to replace and an index to
                replace with.
    """
    raise AssertionError("cannot handle type %s" % type(node))


replace_indices.register(Node)(reuse_if_untouched_arg)


@replace_indices.register(Delta)
def replace_indices_delta(node, self, subst):
    substitute = dict(subst)
    i = substitute.get(node.i, node.i)
    j = substitute.get(node.j, node.j)
    if i == node.i and j == node.j:
        return node
    else:
        return Delta(i, j)


@replace_indices.register(Indexed)
def replace_indices_indexed(node, self, subst):
    child, = node.children
    substitute = dict(subst)
    multiindex = tuple(substitute.get(i, i) for i in node.multiindex)
    if isinstance(child, ComponentTensor):
        # Indexing into ComponentTensor
        # Inline ComponentTensor and augment the substitution rules
        substitute.update(zip(child.multiindex, multiindex))
        return self(child.children[0], tuple(sorted(substitute.items())))
    else:
        # Replace indices
        new_child = self(child, subst)
        if new_child == child and multiindex == node.multiindex:
            return node
        else:
            return Indexed(new_child, multiindex)


@replace_indices.register(FlexiblyIndexed)
def replace_indices_flexiblyindexed(node, self, subst):
    child, = node.children
    assert isinstance(child, Terminal)
    assert not child.free_indices

    substitute = dict(subst)
    dim2idxs = tuple(
        (offset, tuple((substitute.get(i, i), s) for i, s in idxs))
        for offset, idxs in node.dim2idxs
    )

    if dim2idxs == node.dim2idxs:
        return node
    else:
        return FlexiblyIndexed(child, dim2idxs)


def filtered_replace_indices(node, self, subst):
    """Wrapper for :func:`replace_indices`.  At each call removes
    substitution rules that do not apply."""
    filtered_subst = tuple((k, v) for k, v in subst if k in node.free_indices)
    return replace_indices(node, self, filtered_subst)


def remove_componenttensors(expressions):
    """Removes all ComponentTensors in multi-root expression DAG."""
    mapper = MemoizerArg(filtered_replace_indices)
    return [mapper(expression, ()) for expression in expressions]


def _select_expression(expressions, index):
    """Helper function to select an expression from a list of
    expressions with an index.  This function expect sanitised input,
    one should normally call :py:func:`select_expression` instead.

    :arg expressions: a list of expressions
    :arg index: an index (free, fixed or variable)
    :returns: an expression
    """
    expr = expressions[0]
    if all(e == expr for e in expressions):
        return expr

    types = set(map(type, expressions))
    if types <= {Indexed, Zero}:
        multiindex, = set(e.multiindex for e in expressions if isinstance(e, Indexed))
        shape = tuple(i.extent for i in multiindex)

        def child(expression):
            if isinstance(expression, Indexed):
                return expression.children[0]
            elif isinstance(expression, Zero):
                return Zero(shape)
        return Indexed(_select_expression(list(map(child, expressions)), index), multiindex)

    if types <= {Literal, Zero, Failure}:
        return partial_indexed(ListTensor(expressions), (index,))

    if len(types) == 1:
        cls, = types
        if cls.__front__ or cls.__back__:
            raise NotImplementedError("How to factorise {} expressions?".format(cls.__name__))
        assert all(len(e.children) == len(expr.children) for e in expressions)
        assert len(expr.children) > 0

        return expr.reconstruct(*[_select_expression(nth_children, index)
                                  for nth_children in zip(*[e.children
                                                            for e in expressions])])

    raise NotImplementedError("No rule for factorising expressions of this kind.")


def select_expression(expressions, index):
    """Select an expression from a list of expressions with an index.
    Semantically equivalent to

        partial_indexed(ListTensor(expressions), (index,))

    but has a much more optimised implementation.

    :arg expressions: a list of expressions of the same shape
    :arg index: an index (free, fixed or variable)
    :returns: an expression of the same shape as the given expressions
    """
    # Check arguments
    shape = expressions[0].shape
    assert all(e.shape == shape for e in expressions)

    # Sanitise input expressions
    alpha = tuple(Index() for s in shape)
    exprs = remove_componenttensors([Indexed(e, alpha) for e in expressions])

    # Factor the expressions recursively and convert result
    selected = _select_expression(exprs, index)
    return ComponentTensor(selected, alpha)


def delta_elimination(sum_indices, factors):
    """IndexSum-Delta cancellation.

    :arg sum_indices: free indices for contractions
    :arg factors: product factors
    :returns: optimised (sum_indices, factors)
    """
    sum_indices = list(sum_indices)  # copy for modification

    delta_queue = [(f, index)
                   for f in factors if isinstance(f, Delta)
                   for index in (f.i, f.j) if index in sum_indices]
    while delta_queue:
        delta, from_ = delta_queue[0]
        to_, = list({delta.i, delta.j} - {from_})

        sum_indices.remove(from_)

        mapper = MemoizerArg(filtered_replace_indices)
        factors = [mapper(e, ((from_, to_),)) for e in factors]

        delta_queue = [(f, index)
                       for f in factors if isinstance(f, Delta)
                       for index in (f.i, f.j) if index in sum_indices]

    # Drop ones
    return sum_indices, [e for e in factors if e != one]


def sum_factorise(sum_indices, factors):
    """Optimise a tensor product through sum factorisation.

    :arg sum_indices: free indices for contractions
    :arg factors: product factors
    :returns: optimised GEM expression
    """
    if len(sum_indices) > 5:
        raise NotImplementedError("Too many indices for sum factorisation!")

    # Form groups by free indices
    groups = OrderedDict()
    for factor in factors:
        groups.setdefault(factor.free_indices, []).append(factor)
    groups = [reduce(Product, terms) for terms in itervalues(groups)]

    # Sum factorisation
    expression = None
    best_flops = numpy.inf

    # Consider all orderings of contraction indices
    for ordering in permutations(sum_indices):
        terms = groups[:]
        flops = 0
        # Apply contraction index by index
        for sum_index in ordering:
            # Select terms that need to be part of the contraction
            contract = [t for t in terms if sum_index in t.free_indices]
            deferred = [t for t in terms if sum_index not in t.free_indices]

            # A further optimisation opportunity is to consider
            # various ways of building the product tree.
            product = reduce(Product, contract)
            term = IndexSum(product, (sum_index,))
            # For the operation count estimation we assume that no
            # operations were saved with the particular product tree
            # that we built above.
            flops += len(contract) * numpy.prod([i.extent for i in product.free_indices], dtype=int)

            # Replace the contracted terms with the result of the
            # contraction.
            terms = deferred + [term]

        # If some contraction indices were independent, then we may
        # still have several terms at this point.
        expr = reduce(Product, terms)
        flops += (len(terms) - 1) * numpy.prod([i.extent for i in expr.free_indices], dtype=int)

        if flops < best_flops:
            expression = expr
            best_flops = flops

    return expression


def contraction(expression):
    """Optimise the contractions of the tensor product at the root of
    the expression, including:

    - IndexSum-Delta cancellation
    - Sum factorisation

    This routine was designed with finite element coefficient
    evaluation in mind.
    """
    # Eliminate annoying ComponentTensors
    expression, = remove_componenttensors([expression])

    # Flatten a product tree
    sum_indices = []
    factors = []

    queue = deque([expression])
    while queue:
        expr = queue.popleft()
        if isinstance(expr, IndexSum):
            queue.append(expr.children[0])
            sum_indices.extend(expr.multiindex)
        elif isinstance(expr, Product):
            queue.extend(expr.children)
        else:
            factors.append(expr)

    return sum_factorise(*delta_elimination(sum_indices, factors))


@singledispatch
def _replace_delta(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_replace_delta.register(Node)(reuse_if_untouched)


@_replace_delta.register(Delta)
def _replace_delta_delta(node, self):
    i, j = node.i, node.j

    if isinstance(i, Index) or isinstance(j, Index):
        if isinstance(i, Index) and isinstance(j, Index):
            assert i.extent == j.extent
        if isinstance(i, Index):
            assert i.extent is not None
            size = i.extent
        if isinstance(j, Index):
            assert j.extent is not None
            size = j.extent
        return Indexed(Identity(size), (i, j))
    else:
        def expression(index):
            if isinstance(index, int):
                return Literal(index)
            elif isinstance(index, VariableIndex):
                return index.expression
            else:
                raise ValueError("Cannot convert running index to expression.")
        e_i = expression(i)
        e_j = expression(j)
        return Conditional(Comparison("==", e_i, e_j), one, Zero())


def replace_delta(expressions):
    """Lowers all Deltas in a multi-root expression DAG."""
    mapper = Memoizer(_replace_delta)
    return list(map(mapper, expressions))


@singledispatch
def _unroll_indexsum(node, self):
    """Unrolls IndexSums below a certain extent.

    :arg node: root of the expression
    :arg self: function for recursive calls
    """
    raise AssertionError("cannot handle type %s" % type(node))


_unroll_indexsum.register(Node)(reuse_if_untouched)


@_unroll_indexsum.register(IndexSum)  # noqa
def _(node, self):
    unroll = tuple(index for index in node.multiindex
                   if index.extent <= self.max_extent)
    if unroll:
        # Unrolling
        summand = self(node.children[0])
        shape = tuple(index.extent for index in unroll)
        unrolled = reduce(Sum,
                          (Indexed(ComponentTensor(summand, unroll), alpha)
                           for alpha in numpy.ndindex(shape)),
                          Zero())
        return IndexSum(unrolled, tuple(index for index in node.multiindex
                                        if index not in unroll))
    else:
        return reuse_if_untouched(node, self)


def unroll_indexsum(expressions, max_extent):
    """Unrolls IndexSums below a specified extent.

    :arg expressions: list of expression DAGs
    :arg max_extent: maximum extent for which IndexSums are unrolled
    :returns: list of expression DAGs with some unrolled IndexSums
    """
    mapper = Memoizer(_unroll_indexsum)
    mapper.max_extent = max_extent
    return list(map(mapper, expressions))


def aggressive_unroll(expression):
    """Aggressively unrolls all loop structures."""
    # Unroll expression shape
    if expression.shape:
        tensor = numpy.empty(expression.shape, dtype=object)
        for alpha in numpy.ndindex(expression.shape):
            tensor[alpha] = Indexed(expression, alpha)
        expression, = remove_componenttensors((ListTensor(tensor),))

    # Unroll summation
    expression, = unroll_indexsum((expression,), max_extent=numpy.inf)
    expression, = remove_componenttensors((expression,))
    return expression


@singledispatch
def count_flop_node(node):
    """Count number of flops at a particular gem node, without recursing
    into childrens"""
    raise AssertionError("cannot handle type %s" % type(node))


@count_flop_node.register(Constant)
@count_flop_node.register(Terminal)
@count_flop_node.register(Indexed)
@count_flop_node.register(Variable)
@count_flop_node.register(ListTensor)
@count_flop_node.register(FlexiblyIndexed)
@count_flop_node.register(LogicalNot)
@count_flop_node.register(LogicalAnd)
@count_flop_node.register(LogicalOr)
@count_flop_node.register(Conditional)
def count_flop_node_zero(node):
    return 0


@count_flop_node.register(Power)
@count_flop_node.register(Comparison)
@count_flop_node.register(Sum)
@count_flop_node.register(Product)
@count_flop_node.register(Division)
@count_flop_node.register(MathFunction)
def count_flop_node_single(node):
    return numpy.prod([idx.extent for idx in node.free_indices])


@count_flop_node.register(IndexSum)
def count_flop_node_index_sum(node):
    return numpy.prod([idx.extent for idx in node.multiindex + node.free_indices])


def count_flop(node):
    """
    Count the total floating point operations required to compute a gem node.
    This function assumes that all subnodes that occur more than once induce a
    temporary, and are therefore only computed once.
    """
    # TODO: Add tests for this function (number still somewhat diffferent from COFFEE visitor)
    return sum(map(count_flop_node, traversal([node])))


@singledispatch
def _expand_products(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_expand_products.register(Node)(reuse_if_untouched)


@_expand_products.register(Product)
def _expand_products_prod(node, self):
    a = self(node.children[0])
    b = self(node.children[1])
    if isinstance(b, Sum) and set(b.free_indices) & self.index_set:
        return Sum(self(Product(a, b.children[0])),
                   self(Product(a, b.children[1])))
    elif isinstance(a, Sum) and set(a.free_indices) & self.index_set:
        return Sum(self(Product(a.children[0], b)),
                   self(Product(a.children[1], b)))
    else:
        return Product(a, b)


# TODO: arguablly should move this inside LoopOptimiser
def expand_products(node, indices):
    """
    Expand products recursively if free indices of the node contains index
    from :param indices
    e.g (a+(b+c)d)e = ae + bde + cde
    :param node: gem expression
    :param indices: tuple of indices
    :return: gem expression with products expanded
    """
    mapper = Memoizer(_expand_products)
    mapper.index_set = set(indices)
    return mapper(node)


def collect_terms(node, node_type):
    """Recursively collect all children into a list from :param:`node`
    and its children of class :param:`node_type`.

    :param node: root of expression
    :param node_type: class of node (e.g. Sum or Product)
    :return: list of all terms
    """
    terms = []
    queue = [node]
    while queue:
        child = queue.pop()
        if isinstance(child, node_type):
            queue.extend(child.children)
        else:
            terms.append(child)
    return tuple(terms)


@singledispatch
def _hoist_indexsum(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_hoist_indexsum.register(Node)(reuse_if_untouched)


@_hoist_indexsum.register(IndexSum)
def _hoist_indexsum_indexsum(node, self):
    child = self(node.children[0])
    if isinstance(child, Product):
        mi_set = set(node.multiindex)
        factors = collect_terms(child, Product)
        hoisted_factors = []
        remaining_factors = []
        for factor in factors:
            if set(factor.free_indices) & mi_set:
                remaining_factors.append(factor)
            else:
                hoisted_factors.append(factor)
        if hoisted_factors:
            new_indexsum = IndexSum(reduce(Product, remaining_factors, one), node.multiindex)
            return Product(reduce(Product, hoisted_factors, one), new_indexsum)
    return node


def hoist_indexsum(node):
    """
    lifting factors invariant in the contraction indices out of an IndexSum
    e.g. IndexSum(A*B, (i,)) = A * IndexSum(B), if A is independent of i
    :param node:
    :return:
    """
    mapper = Memoizer(_hoist_indexsum)
    return mapper(node)


def decide_key(factor, arg_ind_flat, arg_ind_set):
    """
    Helper function to decide the appropriate key of a factor
    :param factor: gem node
    :param arg_ind_flat: set of argument indices
    """
    fi = factor.free_indices
    if not fi:
        return 'const'
    else:
        ind_set = set(fi) & arg_ind_set
        if len(ind_set) > 1:
            return tuple(i for i in arg_ind_flat if i in ind_set)
        elif len(ind_set) == 0:
            return 'other'
        else:
            return ind_set.pop()


def build_repr(node, arg_ind):
    """
    Build representation from self.node
    """
    arg_ind_flat = tuple([i for indices in arg_ind for i in indices])
    multiindex = ()
    if isinstance(node, IndexSum):
        multiindex = node.multiindex
        node = node.children[0]
    node = expand_products(node, arg_ind_flat)
    summands = collect_terms(node, Sum)
    rep = []
    for summand in summands:
        d = Summand(arg_ind_flat=arg_ind_flat)
        for factor in collect_terms(summand, Product):
            key = decide_key(factor, arg_ind_flat, set(arg_ind_flat))
            d.setdefault(key, []).append(factor)
        rep.append(d)
    return (rep, multiindex)


def optimise(node, quad_ind, arg_ind):
    if isinstance(node, Constant):
        return node
    rep, multiindex = build_repr(node, arg_ind)
    lo = LoopOptimiser(rep=rep, multiindex=multiindex, arg_ind=arg_ind)
    optimal_arg = lo.find_optimal_arg_factors()
    lo.factorise_arg(optimal_arg[0] + optimal_arg[1], no_arg_factor=True)
    node1 = lo.generate_node()

    rep, multiindex = build_repr(node, arg_ind)
    lo = LoopOptimiser(rep=rep, multiindex=multiindex, arg_ind=arg_ind)
    lo.factorise_arg(optimal_arg[0] + optimal_arg[1], no_arg_factor=False)
    node2 = lo.generate_node()

    if count_flop(node1) < count_flop(node2):
        return node1
    else:
        return node2


def optimise_expressions(expressions, quadrature_indices, argument_indices):
    if propagate_failure(expressions):
        return expressions
    return [optimise(node, quadrature_indices, argument_indices) for node in expressions]


def propagate_failure(expressions):
    """
    Check if any gem nodes is Failure. In that case there is no need for subsequent optimisation.
    """
    for n in traversal(expressions):
        if isinstance(n, Failure):
            return True
    return False


def _list_2_node(children, self):
    if not self.balanced:
        return reduce(self.func, children, self.base)
    if len(children) < 3:
        return reduce(self.func, children, self.base)
    else:
        mid = len(children) // 2
        return self((self(children[:mid]), self(children[mid:])))


def list_2_node(function, children, balanced=True, sort=False):
    """
    generate gem node from list of children
    use recursion so that each term is not too long
    i.e. (a+b) + (c+d) instead of (((a+b)+c)+d
    :param children: list of children nodes
    :return: gem node
    """
    # TODO: DAG awareness. Hashing with tuple is probably slow here.
    if sort:
        children = sorted(children, key=lambda x: numpy.product([i.extent for i in x.free_indices]), reverse=True)
    if function == Sum:
        base = Zero()
    elif function == Product:
        base = one
    else:
        raise AssertionError('Cannot combine unless Sum or Product')
    mapper = Memoizer(_list_2_node)
    mapper.func = function
    mapper.base = base
    mapper.balanced = balanced
    return mapper(tuple(children))


def sort_keys(keys, arg_ind_flat):
    """
    sort keys of :class: `Summand` by lexicographical order, according to the order they
    appear in :param arg_ind_flat.
    e.g. [j, k, (j,k)] => [j, (j,k), k]
    """
    # Convert keys to string and sort by lexicographical order
    index_2_letter = dict().fromkeys(arg_ind_flat)
    start_number = ord('a')
    for number, arg_ind in enumerate(arg_ind_flat, start=start_number):
        index_2_letter[arg_ind] = chr(number)
    strings = dict().fromkeys(keys)
    for key in keys:
        if isinstance(key, Index):
            strings[key] = [index_2_letter[key]]
        else:
            strings[key] = [index_2_letter[i] for i in key]
    return sorted(keys, key=lambda k: strings[k])


class Summand(OrderedDict):
    """
    An object wrapped around an OrderedDict to represent product of gem nodes (factors)
    The factors are indexed with keys:
        1. One of the argument indices (e.g. j):
            factors which depend on j but not other argument indices
            All summands will have this item (to avoid checking existence),
            with the values being an empty list possibly.
        2. Tuple of more than one argument indices (e.g. (j, k)):
            factors which depend on j and k, but not other argument indices
            These are rarer and are added as factors are encourtered
        3. string 'const':
            factors with no free indices
        4. string 'other':
            factors depend on indices (typically quadrature indices for reduction)
            other than argument indices
    """
    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        if 'arg_ind_flat' in kwargs:
            self.pop('arg_ind_flat')
            for i in ['const', 'other'] + list(kwargs['arg_ind_flat']):
                self[i] = list()

    def arg_keys(self):
        return [k for k in iterkeys(self) if k != 'const' and k != 'other']

    def contains_arg_factor(self):
        return any([self[k] != [] for k in self.arg_keys()])

    def sorted_arg_keys(self, arg_ind_flat):
        return sort_keys(self.arg_keys(), arg_ind_flat)

    def keys_contain_indices(self, indices):
        """
        Returns list of keys which contains any of the index in indices
        """
        assert len(indices) > 0
        indices_set = set(indices)
        result = list()
        for key, factors in iteritems(self):
            if key == "const":
                continue
            for factor in factors:
                if set(factor.free_indices) & indices_set:
                    result.append(key)
                    break
        return result

    def arg_keys_not_contain_indices(self, indices):
        return [key for key in self.arg_keys() if key not in self.keys_contain_indices(indices)]


# TODO: Better naming for LoopOptimiser and Summand, to reflect their mathematical identity (Product and Sum)
class LoopOptimiser(object):
    """
    An object wrapping around a representation (as sum of products) of a gem node and
    perform optimisations which preserve the sematics of the gem node.

    Attributes:
        rep: a list of :class: `Summand`s, the sum of which gives the gem node represented
    """
    def __init__(self, rep, multiindex, arg_ind):
        """
        Constructor
        :param node: representation of a gem node as summation of products
        :param arg_ind: tuples of tuples of argument (linear) indices
        """
        self.rep = rep
        self.arg_ind = arg_ind
        self.multiindex = multiindex
        self.nodes = None

    @cached_property
    def arg_ind_flat(self):
        return tuple([i for indices in self.arg_ind for i in indices])

    @cached_property
    def arg_ind_set(self):
        return set(self.arg_ind_flat)

    def copy(self):
        new_rep = list(self.rep)
        for new_summand, summand in zip(new_rep, self.rep):
            new_summand = Summand(summand)
            for k, v in iteritems(summand):
                new_summand[k] = list(v)
        return LoopOptimiser(rep=new_rep, multiindex=self.multiindex, arg_ind=self.arg_ind)

    def factor_extent(self, factor):
        """
        Compute the product of extents of all argument indices of :param factor
        """
        return numpy.product([i.extent for i in set(factor.free_indices) & self.arg_ind_set])

    def _decide_key(self, node):
        return decide_key(node, self.arg_ind_flat, self.arg_ind_set)

    def all_arg_keys(self):
        _keys = OrderedDict()
        for summand in self.rep:
            for key in summand.arg_keys():
                _keys[key] = None
        return tuple(_keys.keys())

    def find_optimal_arg_factors(self):
        index = count()
        factor_index = OrderedDict()  # Gem node -> int
        connections = []
        # add connections (list of lists)
        for summand in self.rep:
            connection = []
            for key in summand.arg_keys():
                for factor in summand[key]:
                    if factor not in factor_index:
                        factor_index[factor] = next(index)
                    connection.append(factor_index[factor])
            connections.append(tuple(connection))

        if len(factor_index) == 0:
            return ((), ())

        # set up the ILP
        import pulp as ilp
        ilp_prob = ilp.LpProblem('gem factorise', ilp.LpMinimize)
        ilp_var = ilp.LpVariable.dicts('node', range(len(factor_index)), 0, 1, ilp.LpBinary)

        # Objective function
        # Minimise number of factors to pull. If same number, favour factor with larger extent
        big = 10000000  # some arbitrary big number
        ilp_prob += ilp.lpSum(
            ilp_var[index] * (big - self.factor_extent(factor)) for factor, index in iteritems(factor_index))

        # constraints
        for connection in connections:
            ilp_prob += ilp.lpSum(ilp_var[index] for index in connection) >= 1

        ilp_prob.solve()
        if ilp_prob.status != 1:
            raise AssertionError("Something bad happened during ILP")

        optimal_factors = [factor for factor, _index in iteritems(factor_index) if ilp_var[_index].value() == 1]
        other_factors = [factor for factor, _index in iteritems(factor_index) if ilp_var[_index].value() == 0]
        # TODO: investigate effects of sorting these two lists of factors
        optimal_factors = sorted(optimal_factors, key=lambda f: self.factor_extent(f), reverse=True)
        other_factors = sorted(other_factors, key=lambda f: self.factor_extent(f), reverse=True)
        # Sequence dictating order of factorisation
        return (tuple(optimal_factors), tuple(other_factors))

    def factorise_atom(self, atom):
        """
        factorise this LoopOptimiser (this) into two LoopOptimiser, lo1 and lo2, such that
        this = lo1 + lo2 * atom
        """
        key = self._decide_key(atom)
        # TODO: consider make a copy of rep so that self.rep is untouched
        to_add_2 = list()
        for idx, summand in enumerate(self.rep):
            if key not in summand:
                continue
            if atom not in summand[key]:
                continue
            to_add_2.append(idx)
        if len(to_add_2) <= 1:
            return (self, None)
        rep1 = list()
        rep2 = list()
        for idx, summand in enumerate(self.rep):
            if idx in to_add_2:
                summand[key].remove(atom)
                rep2.append(summand)
            else:
                rep1.append(summand)
        lo1 = LoopOptimiser(rep=rep1, multiindex=self.multiindex, arg_ind=self.arg_ind)
        lo2 = LoopOptimiser(rep=rep2, multiindex=(), arg_ind=self.arg_ind)
        return (lo1, lo2)

    def factorise_key(self, keys, no_arg_factor=True, sort_func=None):
        """
        Factorise common factors that have a particular key from :param keys
        :param no_arg_factor: do not factorise if Summand contains argument
        factors, since this might destroy subsequent factorisation passes.
        """
        if len(self.rep) < 2:
            return
        counter = OrderedDict()
        for summand in self.rep:
            if no_arg_factor and summand.contains_arg_factor():
                continue
            for _key in keys:
                if _key not in summand:
                    continue
                # One factor could appear multiple times in a Summand
                for factor in OrderedDict().fromkeys(summand[_key]):
                    counter.setdefault(factor, 0)
                    counter[factor] += 1
        if not counter:
            return
        if max(counter.values()) < 2:
            return
        if sort_func is None:
            sort_func = lambda x: x[1]
        common_factors = sorted([(k, v) for k, v in iteritems(counter) if v > 1],
                                key=sort_func,
                                reverse=True)
        cf, count = common_factors[0]
        cf_key = self._decide_key(cf)
        lo1, lo2 = self.factorise_atom(cf)
        assert(len(lo2.rep) == count)
        # continue to factorise this new node
        lo2.factorise_key(keys, no_arg_factor, sort_func)
        if len(lo2.rep) == 1:
            # result of further factorisation is a Product
            new_summand = lo2.rep[0]
            new_summand[cf_key].insert(0, cf)
            lo1.rep.append(new_summand)
        else:
            # result of further factorisation is a Sum
            node = lo2.generate_node()
            new_summand = Summand(arg_ind_flat=self.arg_ind_flat)
            new_summand[self._decide_key(node)] = [node]
            new_summand[cf_key].insert(0, cf)
            lo1.rep.append(new_summand)
        self.rep = lo1.rep
        self.factorise_key(keys, no_arg_factor, sort_func)  # Continue factorising

    def factorise_arg(self, factors_seq, no_arg_factor=True):
        """
        Factorise sequentially with common factors from :param factors_seq
        :param factors_seq: sequence of factors used to fa  ctorise
        """
        if len(self.rep) < 2:
            return
        if not factors_seq:
            # sort_func = lambda x: (count_flop(x[0]) + 1) * (x[1] - 1)
            sort_func = None
            self.factorise_key(('other', 'const'), no_arg_factor=no_arg_factor, sort_func=sort_func)
            return
        cf = factors_seq[0]  # pick the first common factor
        key = self._decide_key(cf)

        # this = lo1 + cf * lo2
        lo1, lo2 = self.factorise_atom(cf)
        if lo2 is None:
            self.factorise_arg(factors_seq[1:], no_arg_factor)
            return

        # Proceed with the next common factor for the factorised part
        lo2.factorise_arg(factors_seq[1:], no_arg_factor)
        if len(lo2.rep) == 1:
            # result is a product
            new_summand = lo2.rep[0]
            new_summand[key].insert(0, cf)
            lo1.rep.append(new_summand)
        else:
            node = lo2.generate_node()
            # Create new line in rep
            new_summand = Summand(arg_ind_flat=self.arg_ind_flat)
            new_summand[self._decide_key(node)] = [node]
            new_summand[key].insert(0, cf)
            lo1.rep.append(new_summand)
        self.rep = lo1.rep
        # Proceed with the next common factor
        self.factorise_arg(factors_seq[1:], no_arg_factor)
        return

    def generate_node(self):
        if self.multiindex and len(self.rep) == 1:
            # If argument factors do not contain reduction indices, do the reduction without such factors
            hoist_keys = self.rep[0].arg_keys_not_contain_indices(self.multiindex)
            if hoist_keys:
                hoisted_summand = Summand(arg_ind_flat=self.arg_ind_flat)
                unhoisted_summand = Summand(self.rep[0])
                for key in hoist_keys:
                    hoisted_summand[key] = self.rep[0][key]
                    unhoisted_summand[key] = []
                hoisted_node = LoopOptimiser(rep=[hoisted_summand], multiindex=(), arg_ind=self.arg_ind).generate_node()
                new_summand_node = LoopOptimiser(rep=[unhoisted_summand], multiindex=(), arg_ind=self.arg_ind).generate_node()
                return Product(hoisted_node, IndexSum(new_summand_node, self.multiindex))

        _summands = Summand(arg_ind_flat=self.arg_ind_flat)
        for summand in self.rep:
            _factors = list()
            for key in ['const', 'other'] + summand.sorted_arg_keys(self.arg_ind_flat):
                _factors.append(list_2_node(Product, summand[key], sort=False))
            _summand_node = list_2_node(Product, _factors, balanced=False, sort=False)
            _key = self._decide_key(_summand_node)
            _summands.setdefault(_key, []).append(_summand_node)

        _combined_summands = list()
        for _key in ['const', 'other'] + _summands.sorted_arg_keys(self.arg_ind_flat):
            _combined_summands.append(list_2_node(Sum, _summands[_key], sort=False))
        node = list_2_node(Sum, _combined_summands, balanced=False, sort=False)
        if self.multiindex:
            node = IndexSum(node, self.multiindex)
        return node
