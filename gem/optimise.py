"""A set of routines implementing various transformations on GEM
expressions."""

from __future__ import absolute_import, print_function, division
from six import itervalues
from six.moves import map, zip

from collections import OrderedDict, deque
from functools import reduce
from itertools import permutations

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
    comp_func = lambda x: len(x.free_indices)
    factors = sorted(collect_terms(node, Product), key=comp_func)
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
@count_flop_node.register(IndexSum)
def count_flop_node_zero(node):
    return 0


@count_flop_node.register(Power)
@count_flop_node.register(Comparison)
@count_flop_node.register(Sum)
@count_flop_node.register(Product)
@count_flop_node.register(Division)
@count_flop_node.register(MathFunction)
@count_flop_node.register(LogicalNot)
@count_flop_node.register(LogicalAnd)
@count_flop_node.register(LogicalOr)
@count_flop_node.register(Conditional)
def count_flop_node_single(node):
    return numpy.prod([idx.extent for idx in node.free_indices])


def count_flop(node):
    """Count total number of flops of a gem expression, assuming hoisting and
    reuse"""
    flops = sum(map(count_flop_node, traversal([node])))

    return flops


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
        return node


def expand_products(node, indices):
    """
    Expand products recursively if free indices of the node contains index from :param indices
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


def flatten_sum(node, argument_indices):
    """
    factorise :param:`node` into sum of products, group factors of each product
    based on its dependency on index:
    1) nothing (key = 0)
    2) i (key = 1)
    3) (i)j (key = j)
    4) (i)k (key = k)
    ...
    5) (i)jk (key = 2)
    :param node: root of expression
    :param index: tuple of argument (linear) indices
    :return: dictionary to list of factors
    """
    monos = collect_terms(node, Sum)
    result = []
    arg_ind_set = set(argument_indices)
    for mono in monos:
        d = OrderedDict()
        for i in [0, 1, 2] + list(argument_indices):
            d[i] = list()
        for factor in collect_terms(mono, Product):
            fi = factor.free_indices
            if not fi:
                d[0].append(factor)
            else:
                ind_set = set(fi) & arg_ind_set
                if len(ind_set) > 1:
                    # Aijk
                    d[2].append(factor)
                elif len(ind_set) == 0:
                    # Ai
                    d[1].append(factor)
                else:
                    # Aij
                    d[ind_set.pop()].append(factor)
            # elif j in fi and k in fi:
            #     d[2].append(factor)
            # elif j in fi:
            #     d[j].append(factor)
            # elif k in fi:
            #     d[k].append(factor)
            # else:
            #     d[1].append(factor)
        for i in d:
            d[i] = tuple(d[i])
        result.append(d)
    return tuple(result)


def sumproduct_2_node(factor_lists):
    """
    generate gem node from list of factors
    use recursion so that each term is not too long
    i.e. (a+b) + (c+d) instead of (((a+b)+c)+d
    :param factor_lists: list of factors representing sums of products
    :return: gem node
    """
    if len(factor_lists) < 3:
        sums = []
        for fl in factor_lists:
            sums.append(reduce(Product, fl, one))
        return reduce(Sum, sums, Zero())
    else:
        mid = int(len(factor_lists) / 2)
        return Sum(sumproduct_2_node(factor_lists[:mid]), sumproduct_2_node(factor_lists[mid:]))


def factorise(factor_lists, quad_ind, arg_ind_flat):
    """
    recursively pick the most common factor
    maybe can Memoize this one
    :param factors: list of list of factors, representing sum of products
    :return: optimised list of list of factors
    """
    if len(factor_lists) <= 1:
        return factor_lists
    # count number of common factors
    counter = OrderedDict()
    for fl in factor_lists:
        fl_set = OrderedDict.fromkeys(fl)
        if len(fl_set) > 1:
            # at least two terms in product
            for factor in fl_set:
                if factor in counter:
                    counter[factor] += 1
                else:
                    counter[factor] = 1

    if not counter:
        return factor_lists
    # find which factor to factorise out first
    mcf_value = [0, 1]  # (num arg indices, count)
    mcf = None
    arg_ind_set = set(arg_ind_flat)
    for factor, count in counter.items():
        if count > 1:
            num_arg_ind = len(set(factor.free_indices) & arg_ind_set)
            if (num_arg_ind > mcf_value[0]) or (num_arg_ind == mcf_value[0] and count > mcf_value[1]):
                # prioritize factors with more argument indices
                mcf = factor
                mcf_value[0] = num_arg_ind
                mcf_value[1] = count

    if not mcf:
        # no common factors
        return factor_lists

    new_list = []
    rest = []  # new_list = mcf * [rest] + rest
    # keep original factor list
    # this might not be economical as most of the times factorise should help,
    # and the copying slows down the optimisation quite a bit
    # before_rest = []
    for fl in factor_lists:
        if mcf in fl and len(fl) > 1:
            # at least two terms in the product
            # before_rest.append(fl)
            # new_fl = list(fl)
            # new_fl.remove(mcf)
            # rest.append(new_fl)
            fl.remove(mcf)
            rest.append(fl)
        else:
            new_list.append(fl)

    # before_node = reorder(sumproduct_2_node(before_rest), (quad_ind + arg_ind_flat))
    # before_flop = count_flop(before_node)
    # after_node = reorder(sumproduct_2_node(factorise(rest, quad_ind, arg_ind_flat)), (quad_ind + arg_ind_flat))
    # after_flop = count_flop(Product(mcf, after_node))
    # if after_flop > before_flop:
    #     # only factorise if reduces flops
    #     return factor_lists
    # new_list.append([mcf, after_node])
    new_list.append([mcf, sumproduct_2_node(factorise(rest, quad_ind, arg_ind_flat))])
    return factorise(new_list, quad_ind, arg_ind_flat)  # recursion


def get_array(tensor, subarray=None):
    if isinstance(tensor.children[0], Identity):
        dim = tensor.children[0].dim
        if subarray:
            return eval('numpy.identity({0})'.format(dim) + subarray)
        else:
            return numpy.identity(dim)
    if subarray:
        return eval('tensor.children[0].array' + subarray)
    else:
        return tensor.children[0].array


def contract(tensors, free_indices, indices):
    """
    :param tensors: (A, B, C, ...)
    :param free_indices: contract over (i, ...)
    :param indices: (j, k, ...)
    :return:
    """
    index_map = {}
    letter = ord('a')
    for i in indices + free_indices:
        index_map[i] = chr(letter)
        letter += 1
    subscripts = []
    arrays = []
    for t in tensors:
        if any(isinstance(i, int) for i in t.multiindex):
            subarray = [str(i) if isinstance(i, int) else ':' for i in t.multiindex]
            subarray = '[' + ','.join(subarray) + ']'  # e.g. [:,:,0]
            # this bit is a bit ugly
            arrays.append(get_array(t, subarray))
        else:
            arrays.append(get_array(t))
        # ['ij', 'jk', ...]
        subscripts.append(''.join([index_map[i] for i in t.multiindex
                                   if not isinstance(i, int)]))
    # this is used as the parameter for contraction with einsum
    subscripts = ','.join(subscripts) + ' -> ' +\
                 ''.join(''.join(index_map[i] for i in indices))
    return numpy.einsum(subscripts, *arrays)


def can_pre_evaluate(node, quad_ind, arg_ind):
    if not quad_ind:
        # point evaluation, not integral
        return False

    if not isinstance(node, IndexSum):
        return False

    quad_ind_set = set(quad_ind)
    for n in traversal([node]):
        if isinstance(n, Indexed):
            if quad_ind_set & set(n.free_indices):
                if not isinstance(n.children[0], Constant):
                    return False
                if any([isinstance(i, VariableIndex) for i in n.multiindex]):
                    return False
        if isinstance(n, IndexSum) and n != node:
            # cannot handle unexpanded IndexSum, need to correct in the future
            return False
        if isinstance(n, Power):
            if quad_ind_set & set(n.free_indices):
                # cannot do power yet, but arguably some power should be expanded
                return False
        if isinstance(n, MathFunction):
            # e.g. test if |Jacobian| depend on quadrature points (non-affine)
            # if n.name == 'abs':
            if quad_ind_set & set(n.free_indices):
                return False
    return True


def pre_evaluate(node, quad_ind, arg_ind_flat):
    quad_ind_set = set(quad_ind)
    # extents = dict().fromkeys(quad_ind + arg_ind_flat)
    expand_pe = expand_products(node.children[0], quad_ind + arg_ind_flat)
    monos_pe = flatten_sum(expand_pe, arg_ind_flat)
    # identify number of distinct pre-evaluate tensors
    pe_results = dict()
    for mono in monos_pe:
        tensors = list(mono[1])  # to be pre-evaluated
        rest = list(mono[0])  # does not contain i
        for t in [term for j in arg_ind_flat + (2,) for term in mono[j]]:
            if set(t.free_indices) & quad_ind_set:
                tensors.append(t)
            else:
                rest.append(t)
        mono['pe_tensors'] = tensors = tuple(tensors)
        mono['rest'] = tuple(rest)  # not pre-evaluated
        # there could be duplicated pre-evaluated tensors
        if not tensors:
            mono['pe_result'] = tuple()
        elif tensors in pe_results:
            mono['pe_result'] = pe_results[tensors]
        else:
            # can put in a temporary tensor here
            # mono['pe_result'] = pe_results[tensors] = (Indexed(Variable('PE'+str(len(pe_results)), (J,K)), (j,k)),)
            # sometimes not all argument indices are present in the contraction
            all_ind = set([fi for t1 in tensors for fi in t1.free_indices])
            result_ind = tuple([i for i in arg_ind_flat if i in all_ind])
            array = contract(tensors, quad_ind, result_ind)
            pe_tensor = Indexed(Literal(array, 'PE'+str(len(pe_results))), result_ind)
            mono['pe_result'] = pe_results[tensors] = (pe_tensor, )  # this is a tuple
        # construct list of factor lists after pre-evaluation
    factor_lists = []
    for mono in monos_pe:
        factor_lists.append(list(mono['rest'] + mono['pe_result']))

    node_pe = sumproduct_2_node(factorise(factor_lists, quad_ind, arg_ind_flat))
    node_pe = reorder(node_pe, arg_ind_flat)
    theta_pe = count_flop(node_pe)  # flop count for pre-evaluation method
    return (node_pe, theta_pe)


def find_optimal_factors(monos, arg_ind_flat):
    gem_int = OrderedDict()  # Gem node -> int
    int_gem = OrderedDict()  # int -> Gem node
    counter = 0
    for mono in monos:
        for j in arg_ind_flat:
            # really this should just have 1 element
            for n in mono[j]:
                if n not in gem_int:
                    gem_int[n] = counter
                    int_gem[counter] = n
                    counter += 1
    if counter == 0:
        return tuple()
    # add connections (list of tuples)
    edges_sets_list = []
    # num_edges = dict.fromkeys(int_gem.keys(), 0)
    for mono in monos:
        # this double loop should be optimised further
        edges_sets_list.append(tuple(
            [gem_int[n] for j in arg_ind_flat for n in mono[j]]))

    # product of extents of argument indices of nodes
    extent = dict().fromkeys(int_gem.keys())
    for key in extent:
        arg_ind = set(int_gem[key].free_indices) & set(arg_ind_flat)
        extent[key] = numpy.product([i.extent for i in arg_ind])
    # set up the ILP
    import pulp as ilp
    prob = ilp.LpProblem('factorise', ilp.LpMinimize)
    nodes = ilp.LpVariable.dicts('node', int_gem.keys(), 0, 1, ilp.LpBinary)

    # objective function
    big = 1000000  # some arbitrary big number
    prob += ilp.lpSum(nodes[i]*(big - extent[i]) for i in int_gem)

    # constraints (need to account for >2 argument indices)
    for edges_set in edges_sets_list:
        prob += ilp.lpSum(nodes[i] for i in edges_set) >= 1

    prob.solve()
    if prob.status != 1:
        raise AssertionError("Something bad happened during ILP")

    nodes_to_pull = tuple([n for n, node_number in gem_int.items()
                           if nodes[node_number].value() == 1])
    return nodes_to_pull


def cse(node, quad_ind, arg_ind_flat):
    """
    common subexpression elimination
    :param node:
    :param quad_ind:
    :param arg_ind:
    :return:
    """
    # do not expand quadrature terms all the way
    expand_cse = expand_products(node, arg_ind_flat)
    monos_cse = flatten_sum(expand_cse, arg_ind_flat)
    # need a firt pass of monos to combine terms which have same nodes for all arg_ind
    # this should be in a loop if >2 arg_ind
    if len(arg_ind_flat) > 1:
        optimal_factors = find_optimal_factors(monos_cse, arg_ind_flat)
    else:
        optimal_factors = list()
    # pull out the optimal factors and form factor_list
    factor_lists = list()
    factor_dict = OrderedDict().fromkeys(optimal_factors)
    for of in factor_dict:
        factor_dict[of] = list()
    for mono in monos_cse:
        all_factors = [f for g in (0, 1, 2) + arg_ind_flat for f in mono[g]]
        for of in optimal_factors:
            if of in all_factors:
                all_factors.remove(of)
                factor_dict[of].append(all_factors)
                break
        else:
            factor_lists.append(all_factors)
    for of, factors in factor_dict.items():
        # factors = list of lists representing sum of products
        if len(factors) == 1:
            # just one product
            factor_lists.append([of] + factors[0])
        else:
            if len(arg_ind_flat) > 2:
                # more argument indices to process
                if len(set(of.free_indices) & set(arg_ind_flat)) != 1:
                    raise AssertionError("this should not happen")
                ind = (set(of.free_indices) & set(arg_ind_flat)).pop()
                new_arg_ind_flat = tuple([i for i in arg_ind_flat if i != ind])
                if len(arg_ind_flat) - len(new_arg_ind_flat) != 1:
                    raise AssertionError("this should not happen")
                factor_lists.append([of, cse(sumproduct_2_node(factors),
                                             quad_ind, new_arg_ind_flat)])
            else:
                # remember to factorise factor list here
                factor_lists.append([of, sumproduct_2_node(factorise(factors, quad_ind, arg_ind_flat))])
    return sumproduct_2_node(factorise(factor_lists, quad_ind, arg_ind_flat))


def gen_lex_sequence(items):
    """
    generate lexicographical order of permutations.
    e.g. (i,j,k) -> [(i,), (i,j), (i,j,k), (j), (j,k), (k)]
    :param items:
    :return:
    """
    if not items:
        return list()
    result = gen_lex_sequence(items[1:])
    return [(items[0],)] + [(items[0],) + x for x in result] + result


@singledispatch
def _reorder(node, self, indices):
    """
    Reorder Sum and Product to promote hoisting
    :param node:
    :param self:
    :param indices:
    :return:
    """
    raise AssertionError("cannot handle type %s" % type(node))


_reorder.register(Node)(reuse_if_untouched_arg)


@_reorder.register(IndexSum)
def _reorder_indexsum(node, self, indices):
    new_indices = tuple(indices)
    for i in node.multiindex:
        if i not in indices:
            new_indices = (i,) + new_indices  # insert quadrature indice in front
    new_children = [self(child, new_indices) for child in node.children]
    return node.reconstruct(*new_children)


@_reorder.register(Product)
@_reorder.register(Sum)
def _reorder_product_sum(node, self, indices):
    _class = type(node)
    all_factors = list(collect_terms(node, _class))
    factor_lists = list()
    for curr_indices in [()] + gen_lex_sequence(indices):
        fl = list()
        for f in all_factors:
            if set(f.free_indices) == set(curr_indices):
                fl.append(self(f, indices))
        if fl:
            factor_lists.append(fl)
    if isinstance(node, Sum):
        root = Zero()
    elif isinstance(node, Product):
        root = one
    else:
        raise AssertionError("wrong class")
    return reduce(_class, [reduce(_class, _fl, root) for _fl in factor_lists], root)


def reorder(node, indices=None):
    if not indices:
        indices = node.free_indices
    mapper = MemoizerArg(_reorder)
    return mapper(node, indices)


def optimise(node, quad_ind, arg_ind):
    # there are Zero() sometimes
    if isinstance(node, Constant):
        return node
    arg_ind_flat = tuple([i for id in arg_ind for i in id])
    # do not expand quadrature terms all the way
    if isinstance(node, IndexSum):
        node_cse = IndexSum(cse(node.children[0], quad_ind, arg_ind_flat), node.multiindex)
    else:
        node_cse = cse(node, quad_ind, arg_ind_flat)
    node_cse = reorder(node_cse, arg_ind_flat)
    theta_cse = count_flop(node_cse)
    if can_pre_evaluate(node, quad_ind, arg_ind):
        node_pe, theta_pe = pre_evaluate(node, quad_ind, arg_ind_flat)
        if theta_cse > theta_pe:
            return node_pe
    return node_cse

    # now we need to really pre-evaluate the tensors
    # for tensors in pe_results:
    #     array = contract(tensors, (i,), (j, k))
    #     pe_results[tensors] = Indexed(Literal(array, '')
    #
    # return pe_results


def optimise_expressions(expressions, quadrature_indices, argument_indices):
    if propogate_failure(expressions):
        return expressions
    return [optimise(node, quadrature_indices, argument_indices) for node in expressions]


def propogate_failure(expressions):
    for n in traversal(expressions):
        if isinstance(n, Failure):
            return True
    return False
