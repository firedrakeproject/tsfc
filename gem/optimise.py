"""A set of routines implementing various transformations on GEM
expressions."""

from __future__ import absolute_import, print_function, division
from six.moves import filter, map, zip

from collections import namedtuple
from functools import reduce
from itertools import combinations, permutations

import numpy
from singledispatch import singledispatch

from gem.node import (Memoizer, MemoizerArg, reuse_if_untouched,
                      reuse_if_untouched_arg, collect_refcount)
from gem.gem import (Node, Terminal, Failure, Identity, Literal, Zero,
                     Product, Sum, Comparison, Conditional, Division,
                     Index, VariableIndex, Indexed, FlexiblyIndexed,
                     IndexSum, ComponentTensor, ListTensor, Delta,
                     partial_indexed, one)
from gem.utils import groupby


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

    return sum_indices, factors


ExpressionAndFlopCount = namedtuple('ExpressionAndFlopCount', ['expression', 'flops'])


def associate(gem_type, operands):
    """Apply associativity rules to associative operations (e.g. summation and
    product) to construct an operation-minimal operation tree.

    :arg gem_type: GEM class which is associative, e.g. Sum, Product
    :arg operands: list of operands

    :returns: named tuple ExpressionAndFlopCount, with expression holds the
              resultant expression, and flops holds its number of flops
    """

    if len(operands) > 32:
        # O(N^3) algorithm
        raise NotImplementedError("Not expected such a complicated expression!")

    def count(pair):
        """Operation count to operate on a pair of GEM expressions"""
        a, b = pair
        extents = [i.extent for i in set().union(a.free_indices, b.free_indices)]
        return numpy.prod(extents, dtype=int)

    flops = 0
    while len(operands) > 1:
        # Greedy algorithm: choose a pair of operands that are the
        # cheapest to do.
        a, b = min(combinations(operands, 2), key=count)
        flops += count((a, b))
        # Remove chosen factors, append their product
        operands.remove(a)
        operands.remove(b)
        operands.append(gem_type(a, b))
    node, = operands
    return ExpressionAndFlopCount(node, flops)


@singledispatch
def _reassociate_product(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_reassociate_product.register(Node)(reuse_if_untouched)


@_reassociate_product.register(Product)
def _reassociate_product_prod(node, self):
    # collect all factors of product
    _, factors = traverse_product(node, self.stop_at)
    factors = list(map(self, factors))
    return make_product(factors)


def reassociate_product(expressions):
    """Refactor a Product expression recursively by calling associate_product().

    :arg expressions: list of GEM expressions

    :returns: list of GEM expressions with Product nodes reassociated
    """
    refcount = collect_refcount(expressions)

    # Do not break down common subexpression
    def stop_at(node):
        if not isinstance(node, Product):
            return True
        if refcount[node] > 1:
            return True
        return False
    mapper = Memoizer(_reassociate_product)
    mapper.stop_at = stop_at
    return list(map(mapper, expressions))


def sum_factorise(sum_indices, factors):
    """Optimise a tensor product through sum factorisation.

    :arg sum_indices: free indices for contractions
    :arg factors: product factors
    :returns: optimised GEM expression
    """
    if len(factors) == 0 and len(sum_indices) == 0:
        # Empty product
        return one

    if len(sum_indices) > 5:
        raise NotImplementedError("Too many indices for sum factorisation!")

    # Form groups by free indices
    groups = groupby(factors, key=lambda f: f.free_indices)
    groups = [reduce(Product, terms) for _, terms in groups]

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

            # Optimise associativity
            product, flops_ = associate(Product, contract)
            term = IndexSum(product, (sum_index,))
            flops += flops_ + numpy.prod([i.extent for i in product.free_indices], dtype=int)

            # Replace the contracted terms with the result of the
            # contraction.
            terms = deferred + [term]

        # If some contraction indices were independent, then we may
        # still have several terms at this point.
        expr, flops_ = associate(Product, terms)
        flops += flops_

        if flops < best_flops:
            expression = expr
            best_flops = flops

    return expression


def make_sum(summands):
    """Create a Sum from collection of summands. The summands are grouped by
    their free indices first. Associativity rule is applied to construct an
    operation-minimal summation tree.
    :arg summands: A iterable collection of summands
    :return: Sum expression which represents summation of summands
    """
    groups = groupby(summands, key=lambda f: f.free_indices)
    summands = [reduce(Sum, terms) for _, terms in groups]
    return associate(Sum, summands).expression


def make_product(factors, sum_indices=()):
    """Create a Product from collection of factors. Calls sum_factorise to
    group factors based on their free indices and apply associativity rule to
    construct an operation-minimal product tree.

    :arg factors: A iterable collection of factors
    :arg sum_indices: sum indices of the product
    :return: Product expression which represents product of factors
    """
    return sum_factorise(sum_indices, factors)


def traverse_product(expression, stop_at=None):
    """Traverses a product tree and collects factors, also descending into
    tensor contractions (IndexSum).  The nominators of divisions are
    also broken up, but not the denominators.

    :arg expression: a GEM expression
    :arg stop_at: Optional predicate on GEM expressions.  If specified
                  and returns true for some subexpression, that
                  subexpression is not broken into further factors
                  even if it is a product-like expression.
    :returns: (sum_indices, terms)
              - sum_indices: list of indices to sum over
              - terms: list of product terms
    """
    sum_indices = []
    terms = []

    stack = [expression]
    while stack:
        expr = stack.pop()
        if stop_at is not None and stop_at(expr):
            terms.append(expr)
        elif isinstance(expr, IndexSum):
            stack.append(expr.children[0])
            sum_indices.extend(expr.multiindex)
        elif isinstance(expr, Product):
            stack.extend(reversed(expr.children))
        elif isinstance(expr, Division):
            # Break up products in the dividend, but not in divisor.
            dividend, divisor = expr.children
            if dividend == one:
                terms.append(expr)
            else:
                stack.append(Division(one, divisor))
                stack.append(dividend)
        else:
            terms.append(expr)

    return sum_indices, terms


def traverse_sum(expression, stop_at=None):
    """Traverses a summation tree and collects summands.

    :arg expression: a GEM expression
    :arg stop_at: Optional predicate on GEM expressions.  If specified
                  and returns true for some subexpression, that
                  subexpression is not broken into further summands
                  even if it is an addition.
    :returns: list of summand expressions
    """
    stack = [expression]
    result = []
    while stack:
        expr = stack.pop()
        if stop_at is not None and stop_at(expr):
            result.append(expr)
        elif isinstance(expr, Sum):
            stack.extend(reversed(expr.children))
        else:
            result.append(expr)
    return result


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

    # Flatten product tree, eliminate deltas, sum factorise
    return sum_factorise(*delta_elimination(*traverse_product(expression)))


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
    unroll = tuple(filter(self.predicate, node.multiindex))
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


def unroll_indexsum(expressions, predicate):
    """Unrolls IndexSums below a specified extent.

    :arg expressions: list of expression DAGs
    :arg predicate: a predicate function on :py:class:`Index` objects
                    that tells whether to unroll a particular index
    :returns: list of expression DAGs with some unrolled IndexSums
    """
    mapper = Memoizer(_unroll_indexsum)
    mapper.predicate = predicate
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
    expression, = unroll_indexsum((expression,), predicate=lambda index: True)
    expression, = remove_componenttensors((expression,))
    return expression
