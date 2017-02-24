"""A set of routines implementing various transformations on GEM
expressions."""

from __future__ import absolute_import, print_function, division
from six import itervalues, iteritems
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
@count_flop_node.register(IndexSum)
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


# TODO: arguablly should move this inside Kernel_Optimiser
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


def optimise(node, quad_ind, arg_ind):
    if isinstance(node, Constant):
        return node
    lo = LoopOptimiser(node, arg_ind)
    optimal_arg = lo.find_optimal_arg_factors()
    lo.factorise_arg(optimal_arg)
    return lo.generate_node()


def optimise_expressions(expressions, quadrature_indices, argument_indices):
    if propagate_failure(expressions):
        return expressions
    return [optimise(node, quadrature_indices, argument_indices) for node in expressions]


def propagate_failure(expressions):
    for n in traversal(expressions):
        if isinstance(n, Failure):
            return True
    return False


def _list_2_node(children, self):
    if len(children) < 3:
        return reduce(self.func, children, self.base)
    else:
        mid = len(children) // 2
        return self((self(children[:mid]), self(children[mid:])))


def list_2_node(function, children):
    """
    generate gem node from list of children
    use recursion so that each term is not too long
    i.e. (a+b) + (c+d) instead of (((a+b)+c)+d
    :param children: list of children nodes
    :return: gem node
    """
    # TODO: DAG awareness. Hashing with tuple is probably slow here.
    # possibly need to be able to pass in a sequence (of free_indices) dictating the order of grouping children nodes
    if function == Sum:
        base = Zero()
    elif function == Product:
        base = one
    else:
        raise AssertionError('Cannot combine unless Sum or Product')
    mapper = Memoizer(_list_2_node)
    mapper.func = function
    mapper.base = base
    return mapper(tuple(children))


class LoopOptimiser(object):
    """
    An object holding a representation of a gem IR (as sum of products) and
    perform optimisations which preserve the sematics of the IR.
    Fields:
    1. node
    2. multiindex
    3. arg_ind
    4. rep: a list of dictionaries (summands) of list of factors, such that
    node = Sum(summands), where a summand is Product(factors), possibly contracted
    with multiindex
    Factors are indexed with keys:
        1. One of the argument indices (e.g. j):
            factors depend on j but not other argument indices
            All summands will have this item (to avoid checking existence),
            with the values possibly as empty list
        2. Combination of >1 argument indices (e.g. j, k):
            factors depend on j and k, but not other argument indices
            These are rarer and are added as factors are encourtered
        3. string 'const':
            factors with no free indices
        4. string 'quad':
            factors depend on indices (typically quadrature indices for reduction)
            other than argument indices
    """
    def __init__(self, node, arg_ind, rep=None):
        """
        Constructor
        :param node: gem node
        :param arg_ind: tuples of tuples of argument (linear) indices
        """
        self.node = node
        self.arg_ind = arg_ind
        # TODO: make these two properties of the object
        self.arg_ind_flat = tuple([i for id in self.arg_ind for i in id])
        self.arg_ind_set = set(self.arg_ind_flat)
        if isinstance(node, IndexSum):
            self.multiindex = node.multiindex
            self.node = node.children[0]
        else:
            self.multiindex = ()
        if rep:
            self.rep = rep
        else:
            self._build_repr()
        self.opt_node = None  # optimised gem node

    def _decide_key(self, factor):
        """
        Helper function to decide the appropriate key of a factor
        :param factor: gem node
        """
        fi = factor.free_indices
        if not fi:
            return 'const'
        else:
            ind_set = set(fi) & self.arg_ind_set
            if len(ind_set) > 1:
                return tuple(i for i in self.arg_ind_flat if i in ind_set)
            elif len(ind_set) == 0:
                return 'other'
            else:
                return ind_set.pop()

    def _build_repr(self):
        """
        Build self.repr from self.node
        """
        node = expand_products(self.node, self.arg_ind_flat)
        summands = collect_terms(node, Sum)
        rep = []
        for summand in summands:
            d = OrderedDict()
            for i in ['const', 'other'] + list(self.arg_ind_flat):
                d[i] = list()
            for factor in collect_terms(summand, Product):
                key = self._decide_key(factor)
                d.setdefault(key, []).append(factor)
            rep.append(d)
        self.rep = rep

    def factor_extent(self, factor):
        """
        Compute the product of extents of all argument indices of :param factor
        """
        return numpy.product([i.extent for i in set(factor.free_indices) & self.arg_ind_set])

    def find_optimal_arg_factors(self):
        gem_int = OrderedDict()  # Gem node -> int
        int_gem = OrderedDict()  # int -> Gem node
        counter = 0
        # TODO: perhaps should keep a list of all nodes in the object
        # assign number to all factors
        for summand in self.rep:
            for key, factors in iteritems(summand):
                # TODO: this pattern appears multiple times, need to rewrite
                if key == 'const' or key == 'other':
                    continue
                for factor in factors:
                    if factor not in gem_int:
                        gem_int[factor] = counter
                        int_gem[counter] = factor
                        counter += 1
        if counter == 0:
            return tuple()
        # add connections (list of tuples)
        connections = []
        for summand in self.rep:
            connection = []
            for key, factors in iteritems(summand):
                if key == 'const' or key == 'other':
                    continue
                connection.extend([gem_int[factor] for factor in factors])
            connections.append(tuple(connection))

        # set up the ILP
        import pulp as ilp
        ilp_prob = ilp.LpProblem('gem factorise', ilp.LpMinimize)
        ilp_var = ilp.LpVariable.dicts('node', int_gem.keys(), 0, 1, ilp.LpBinary)

        # Objective function
        # Minimise number of factors to pull. If same number, favour factor with larger extent
        big = 10000000  # some arbitrary big number
        ilp_prob += ilp.lpSum(ilp_var[i] * (big - self.factor_extent(int_gem[i])) for i in int_gem)

        # constraints
        for connection in connections:
            ilp_prob += ilp.lpSum(ilp_var[i] for i in connection) >= 1

        ilp_prob.solve()
        if ilp_prob.status != 1:
            raise AssertionError("Something bad happened during ILP")

        optimal_factors = [factor for factor, number in iteritems(gem_int) if ilp_var[number].value() == 1]
        other_factors = [factor for factor, number in iteritems(gem_int) if ilp_var[number].value() == 0]
        # TODO: investigate effects of sorting these two lists of factors
        optimal_factors = sorted(optimal_factors, key=lambda f: self.factor_extent(f), reverse=True)
        other_factors = sorted(other_factors, key=lambda f: self.factor_extent(f), reverse=True)
        # Sequence dictating order of factorisation
        factors_seq = optimal_factors + other_factors
        return factors_seq

    def factorise_key(self, key):
        """
        Factorise common factors that have a particular :param key
        """
        if len(self.rep) < 2:
            return
        counter = OrderedDict()
        for summand in self.rep:
            if key not in summand:
                continue
            for factor in summand[key]:
                counter.setdefault(factor, 0)
                counter[factor] += 1
        if not counter:
            return
        if max(counter.values()) < 2:
            return

        mcf = None  # most common factor
        mcf_value = (0, 1)  # (number of free indices, count)
        for factor, count in iteritems(counter):
            if count > 1:
                nfi = len(factor.free_indices)
                if nfi > mcf_value[0] or (nfi == mcf_value[0] and count > mcf_value[1]):
                    # TODO: Possibly other heurstics than prioritizing factors with most free indices
                    mcf = factor
                    mcf_value = (nfi, count)
        if not mcf:
            return
        self.opt_node = None
        _summands = list()
        factored_out = list()
        for summand in self.rep:
            # TODO: This pattern of checking keys is repeated several times
            if key not in summand:
                continue
            if mcf in summand[key]:
                factored_out.append(summand)  # mark for deleting later
                _factors = OrderedDict([(_k, _v) for (_k, _v) in iteritems(summand) if _k != key])
                _factors[key] = [factor for factor in summand[key] if factor != mcf]
                _summands.append(_factors)
        # TODO: Create a method for this
        lo = LoopOptimiser(node=None, arg_ind=self.arg_ind, rep=_summands)
        # TODO: Maybe can continue to factorise this new node
        node = lo.generate_node()
        for to_delete in factored_out:
            self.rep.remove(to_delete)
        new_summand = OrderedDict()
        if self._decide_key(node) == key:
            new_summand[key] = [mcf, node]
        else:
            new_summand[key] = [mcf]
            new_summand[self._decide_key(node)] = [node]
        self.rep.append(new_summand)
        self.factorise_key(key)  # Continue factorising

    def factorise_arg(self, factors_seq):
        """
        Factorise sequentially with common factors from :param factors_seq
        :param factors_seq: sequence of factors used to factorise
        """
        if not factors_seq:
            # TODO: investigate if worthwhile to do a pass to check const common factors here
            # self.factorise_key('other')
            # self.factorise_key('const')
            return self.generate_node()
        cf = factors_seq[0]  # pick the first common factor
        key = self._decide_key(cf)
        factored_out = list()
        # cf * Sum(_summands), where summand = Product(_factors)
        _summands = list()
        for summand in self.rep:
            if key not in summand:
                continue
            factors = summand[key]
            if len(factors) > 1:
                raise AssertionError("There should be only one argument factor")
            if cf not in factors:
                continue
            factored_out.append(summand)  # mark for deleting later
            # TODO: default fields might be missing from this OrderedDict(), consider wrap an object around it
            _summands.append(OrderedDict([(_k, _v) for (_k, _v) in iteritems(summand) if _k != key]))
        self.opt_node = None
        if len(_summands) > 1:
            # Proceed with the next common factor for the factorised part
            lo = LoopOptimiser(node=None, arg_ind=self.arg_ind, rep=_summands)
            lo.factorise_arg(factors_seq[1:])
            node = lo.generate_node()
            # Delete factored out lines in rep
            for to_delete in factored_out:
                self.rep.remove(to_delete)
            # Create new line in rep
            # TODO: Need an object to wrap around summands
            new_summand = OrderedDict()
            for i in ['const', 'other'] + list(self.arg_ind_flat):
                new_summand[i] = list()
            new_summand[self._decide_key(cf)] = [cf]
            new_summand[self._decide_key(node)] = [node]
            self.rep.append(new_summand)
        # Proceed with the next common factor
        lo = LoopOptimiser(node=None, arg_ind=self.arg_ind, rep=self.rep)
        lo.factorise_arg(factors_seq[1:])
        return lo.generate_node()

    def generate_node(self):
        # TODO: Here need to consider the order of forming products, currently this only ensures same group gets multiplied first. Consider lexico order between groups.
        if self.opt_node:
            return self.opt_node
        _summands = list()
        for summands in self.rep:
            _factors = list()
            for factors in summands.values():
                _factors.append(list_2_node(Product, factors))
            _summands.append(list_2_node(Product, _factors))
        self.opt_node = list_2_node(Sum, _summands)
        if self.multiindex:
            self.opt_node = IndexSum(self.opt_node, self.multiindex)
        return self.opt_node
