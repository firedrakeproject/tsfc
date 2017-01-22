"""A set of routines implementing various transformations on GEM
expressions."""

from __future__ import absolute_import, print_function, division
from six import itervalues
from six.moves import map, zip

from collections import OrderedDict, deque, Counter
from functools import reduce
from itertools import permutations

import numpy
from singledispatch import singledispatch

from gem.node import Memoizer, MemoizerArg, reuse_if_untouched, reuse_if_untouched_arg
from gem.gem import (Node, Terminal, Failure, Identity, Literal, Zero,
                     Product, Sum, Comparison, Conditional, Index, Constant,
                     VariableIndex, Indexed, FlexiblyIndexed,
                     IndexSum, ComponentTensor, ListTensor, Delta,
                     partial_indexed, one, Division, MathFunction)


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
    if isinstance(b, Literal):
        return Product(self(a), Literal(1.0/b.array))
    else:
        return Product(self(a), Division(Literal(1.0), self(b)))


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
    # should use more sophisticated method later on for optimal result
    sort_func = lambda node: len(node.free_indices)
    factors = sorted(collect_terms(node, Product), key=sort_func)
    # need to optimise away iterator <==> list
    new_factors = list(map(self, factors))  # recursion
    return reduce(Product, new_factors)


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


# ---------------------------------------
# Count flop of expression, "as it is", no reordering etc
@singledispatch
def count_flop(node):
    raise AssertionError("cannot handle type %s" % type(node))


@count_flop.register(IndexSum)
def _count_flop_single(node):
    return count_flop(node.children[0])


@count_flop.register(MathFunction)
def _count_flop_func(node):
    return count_flop(node.children[0])*2


@count_flop.register(Sum)
@count_flop.register(Product)
@count_flop.register(Division)
@count_flop.register(FlexiblyIndexed)
def _count_flop_common(node):
    flop = numpy.product([i.extent for i in node.free_indices])
    # Hoisting the factors
    for child in node.children:
        flop += count_flop(child)
    return flop


@count_flop.register(Constant)
@count_flop.register(Terminal)
@count_flop.register(Indexed)
def _count_flop_const(node):
    return 0


# def count_flop(expression):
#     mapper = Memoizer(_count_flop)
#     return mapper(expression)


# ---------------------------------------
# Expand all products recursively
# e.g (a+(b+c)d)e = ae + bde + cde

@singledispatch
def _expand_all_product(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_expand_all_product.register(Node)(reuse_if_untouched)


@_expand_all_product.register(Product)
def _expand_all_product_common(node, self):
    a, b = map(self, node.children)
    if isinstance(b, Sum):
        return Sum(self(Product(a, b.children[0])), self(Product(a, b.children[1])))
    elif isinstance(a, Sum):
        return Sum(self(Product(a.children[0], b)), self(Product(a.children[1], b)))
    else:
        return node


def collect_terms(node, node_type):
    from collections import deque
    terms = []  # collected terms
    queue = deque([node])  # queue of children nodes to process
    while queue:
        child = queue.popleft()
        if isinstance(child, node_type):
            queue.extendleft(reversed(child.children))
        else:
            terms.append(child)
    return terms


# def collect_terms(node, node_type):
#     """Recursively collect all children into a list from :param:`node`
#     and its children of class :param:`node_type`.
#
#     :param node: root of expression
#     :param node_type: class of node (e.g. Sum or Product)
#     :return: list of all terms
#     """
#
#     mapper = MemoizerArg(_collect_terms)
#     return mapper(node, node_type)


def _flatten_sum(node, self):
    """
    factorise :param:`node` into sum of products, group factors of each product
    based on its dependency on index:
    1) nothing (key = 0)
    2) i (key = 1)
    3) (i)j (key = j)
    4) (i)k (key = k)
    ...
    :param node: root of expression
    :param index: tuple of argument (linear) indices
    :return: dictionary to list of factors
    """
    index = self.index
    sums = collect_terms(node, Sum)
    result = []
    for sum in sums:
        d = OrderedDict()
        for i in [0, 1] + list(index):
            d[i] = list()
        for factor in collect_terms(sum, Product):
            fi = factor.free_indices
            if fi == ():
                d[0].append(factor)
            else:
                flag = True
                for i in index:
                    if i in fi:
                        flag = False
                        d[i].append(factor)
                        break
                if flag:
                    d[1].append(factor)
        for i in [0, 1] + list(index):
            d[i] = tuple(d[i])
        result.append(d)
    return tuple(result)


# def flatten_sum(node, index):

#     mapper = MemoizerArg(_flatten_sum)
#     return mapper(node, index)


def _find_common_factor(node, self):
    """
    find common factors of :param `node`
    :param node: root of expression, usually sum of products
    :param index: tuple (free indices, current index)
    :return: list of common factors categorized by current index
    """
    fi, i = self.index  # free index and current index
    self.flatten_sum.index = fi
    self.flatten_sum.context = fi
    sumproduct = self.flatten_sum(node)
    return tuple(sorted(list(reduce(lambda a, b: a & b,
                                    [Counter(f[i]) for f in sumproduct]))))


def _factorise_i(node, self):
    """
    factorise :param `node` using factors with current index as common factor
    :param node: root of expression
    :param self: Memoizer object
    :return: factorised new node
    """
    fi, i = self.index  # free index, current index
    self.flatten_sum.index = fi
    self.flatten_sum.context = fi
    sumproduct = self.flatten_sum(node)
    # collect all factors with correct index
    factors = OrderedDict()
    for p in sumproduct:
        if p[i]:
            factors[p[i][0]] = 0
    factors = list(factors.iterkeys())
    # only 1 element per list due to linearity, thus p[i][0]
    # sort to ensure deterministic result
    sums = OrderedDict()
    for f in factors:
        sums[f] = []
    sums[0] = []
    for p in sumproduct:
        # extract out common factor
        p_const = reduce(Product, p[0], one)  # constants
        p_i = reduce(Product, p[1], one)  # quadrature index
        # argument index
        p_jk = reduce(Product, [p[j][0] for j in fi if j != i and p[j]], one)
        new_node = reduce(Product, [p_const, p_i, p_jk], one)
        if p[i]:
            # add to corresponding factor list if product contains
            # factor of index i
            sums[p[i][0]].append(new_node)
        else:
            # add to list of the rest
            sums[0].append(new_node)
    sum_i = []
    # create tuple of free indices with the current index removed
    # new_index = list(fi)
    # new_index.remove(i)
    # new_index = tuple(new_index)
    for f in factors:
        # factor * subexpression
        # recursively factorise newly creately subexpression (a sumproduct)
        sum_i.append(Product(
            f,
            self.factorise(reduce(Sum, sums[f], Zero()))))
    return reduce(Sum, sum_i + sums[0], Zero())


@singledispatch
def _factorise(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_factorise.register(Node)(reuse_if_untouched)


@_factorise.register(Sum)
@_factorise.register(Product)
def _factorise_common(node, self):
    # sort indices to ensure deterministic result
    index = tuple(sorted(list(node.free_indices), key=lambda x: x.count))
    flop = count_flop(node)
    optimal_i = None
    node_expand = self.expand_all_product(node)
    self.flatten_sum.index = index
    self.flatten_sum.context = index
    sumproduct = self.flatten_sum(node_expand)
    # find common factors that are constants or dependent on quadrature index
    self.find_common_factor.index = (index, 0)
    self.find_common_factor.context = (index, 0)
    factor_const = self.find_common_factor(node_expand)
    self.find_common_factor.index = (index, 1)
    self.find_common_factor.context = (index, 1)
    factor_1 = self.find_common_factor(node_expand)
    # node = factor_const * factor_1 * Sum(child_sum)
    child_sum = []
    for p in sumproduct:
        p0_list = list(p[0])
        p1_list = list(p[1])
        # extract common factors
        if factor_const or factor_1:
            for x in factor_const:
                p0_list.remove(x)
            for x in factor_1:
                p1_list.remove(x)
        child_sum.append(reduce(
            Product, p0_list + p1_list + [x for i in index for x in p[i]], one))
    p_const = reduce(Product, factor_const, one)
    p_1 = reduce(Product, factor_1, one)
    # new child node
    child = reduce(Sum, child_sum, Zero())
    if index:
        # try factorisation on each argument dimension
        for i in index:
            self.factorise_i.index = (index, i)
            self.factorise_i.context = i
            child = self.factorise_i(child)
            new_node = Product(Product(p_const, p_1), child)
            new_flop = count_flop(new_node)
            if new_flop < flop:
                optimal_i = i
                flop = new_flop
            child = self.expand_all_product(child)
    if optimal_i:
        self.factorise_i.index = (index, optimal_i)
        self.factorise_i.context = i
        child = self.factorise_i(child)
        return Product(Product(p_const, p_1), child)
    else:
        return node


def factorise(node):
    m1 = Memoizer(_factorise)
    m2 = Memoizer(_factorise_i)
    m3 = Memoizer(_expand_all_product)
    m4 = Memoizer(_flatten_sum)
    m5 = Memoizer(_find_common_factor)
    m1.factorise_i = m2
    m2.factorise = m1
    m1.expand_all_product = m3
    m1.flatten_sum = m4
    m2.flatten_sum = m4
    m5.flatten_sum = m4
    m1.find_common_factor = m5
    return m1(node)


def factorise_list(expressions):
    return list(map(factorise, expressions))
