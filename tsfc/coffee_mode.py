from __future__ import absolute_import, print_function, division

import numpy
import itertools
from functools import partial
from six import iteritems, iterkeys
from six.moves import filter
from collections import OrderedDict
from gem.optimise import (replace_division, make_sum, make_product,
                          unroll_indexsum, replace_delta, remove_componenttensors)
from gem.refactorise import (MonomialSum, ATOMIC, COMPOUND, OTHER,
                             collect_monomials)
from gem.node import traversal
from gem.gem import (Product, Sum, Comparison, Conditional, Division, Indexed,
                     IndexSum, MathFunction, Power, Failure, one, index_sum,
                     Terminal, ListTensor, FlexiblyIndexed, LogicalAnd,
                     LogicalNot, LogicalOr)
from gem.utils import groupby


import tsfc.vanilla as vanilla

flatten = vanilla.flatten

finalise_options = dict(replace_delta=False, remove_componenttensors=False)


def Integrals(expressions, quadrature_multiindex, argument_multiindices, parameters):
    """Constructs an integral representation for each GEM integrand
    expression.

    :arg expressions: integrand multiplied with quadrature weight;
                      multi-root GEM expression DAG
    :arg quadrature_multiindex: quadrature multiindex (tuple)
    :arg argument_multiindices: tuple of argument multiindices,
                                one multiindex for each argument
    :arg parameters: parameters dictionary

    :returns: list of integral representations
    """
    # Unroll
    max_extent = parameters["unroll_indexsum"]
    if max_extent:
        def predicate(index):
            return index.extent <= max_extent
        expressions = unroll_indexsum(expressions, predicate=predicate)
    # Choose GEM expression as the integral representation
    expressions = [index_sum(e, quadrature_multiindex) for e in expressions]
    expressions = replace_delta(expressions)
    expressions = remove_componenttensors(expressions)
    expressions = replace_division(expressions)
    return optimise_expressions(expressions, argument_multiindices)


def optimise_expressions(expressions, argument_multiindices):
    """Perform loop optimisations on GEM DAGs

    :arg expressions: list of GEM DAGs
    :arg argument_multiindices: tuple of argument multiindices,
                                one multiindex for each argument

    :returns: list of optimised GEM DAGs
    """
    # Propagate Failure nodes
    for n in traversal(expressions):
        if isinstance(n, Failure):
            return expressions

    def classify(argument_indices, expression):
        n = len(argument_indices.intersection(expression.free_indices))
        if n == 0:
            return OTHER
        elif n == 1:
            if isinstance(expression, (Indexed, Conditional)):
                return ATOMIC
            else:
                return COMPOUND
        else:
            return COMPOUND

    argument_indices = tuple(itertools.chain(*argument_multiindices))
    # Apply argument factorisation unconditionally
    classifier = partial(classify, set(argument_indices))
    monomial_sums = collect_monomials(expressions, classifier)
    monomial_sums = [optimise_monomial_sum(ms, argument_indices) for ms in monomial_sums]
    return list(map(monomial_sum_to_expression, monomial_sums))


def index_extent(factor, argument_indices):
    """Compute the product of the extents of argument indices of a GEM expression

    :arg factor: GEM expression
    :arg argument_indices: set of argument indices

    :returns: product of extents of argument indices
    """
    return numpy.product([i.extent for i in set(factor.free_indices).intersection(argument_indices)])


def monomial_sum_to_expression(monomial_sum):
    """Convert a monomial sum to a GEM expression.

    :arg monomial_sum: :class:`MonomialSum` object

    :returns: GEM expression
    """
    indexsums = []  # The result is summation of indexsums
    # Group monomials according to their sum indices
    groups = groupby(monomial_sum, key=lambda m: frozenset(m.sum_indices))
    # Create IndexSum's from each monomial group
    for _, monomials in groups:
        # Pick sum indices from the first monomial
        sum_indices = monomials[0].sum_indices
        # Create one product for each monomial
        products = [make_product(monomial.atomics + (monomial.rest,)) for monomial in monomials]
        indexsums.append(IndexSum(make_sum(products), sum_indices))
    return make_sum(indexsums)


def find_optimal_atomics(monomial_sum, argument_indices):
    """Find optimal atomic common subexpressions, which produce least number of
    terms in the resultant IndexSum when factorised.

    :arg monomial_sum: A :class:`MonomialSum` object, or a iterable collection
                       of :class:`Monomial`s, all of the monomials should have
                       the same sum indices
    :arg argument_indices: tuple of argument indices

    :returns: list of atomic GEM expressions
    """
    index = itertools.count()  # counter for variables used in ILP
    atomic_index = OrderedDict()  # Atomic GEM node -> int
    connections = []
    # add connections (list of tuples, items in each tuple form a product)
    for monomial in monomial_sum:
        connection = []
        for atomic in monomial.atomics:
            if atomic not in atomic_index:
                atomic_index[atomic] = next(index)
            connection.append(atomic_index[atomic])
        connections.append(tuple(connection))

    if len(atomic_index) == 0:
        return ()
    if len(atomic_index) == 1:
        optimal_atomics, = iterkeys(atomic_index)
        return (optimal_atomics, )

    # set up the ILP
    import pulp as ilp
    ilp_prob = ilp.LpProblem('gem factorise', ilp.LpMinimize)
    ilp_var = ilp.LpVariable.dicts('node', range(len(atomic_index)), 0, 1, ilp.LpBinary)

    # Objective function
    # Minimise number of factors to pull. If same number, favour factor with larger extent
    big = 1e20  # some arbitrary big number
    ilp_prob += ilp.lpSum(ilp_var[index] * (big - index_extent(atomic, argument_indices))
                          for atomic, index in iteritems(atomic_index))

    # constraints
    for connection in connections:
        ilp_prob += ilp.lpSum(ilp_var[index] for index in connection) >= 1

    ilp_prob.solve()
    if ilp_prob.status != 1:
        raise RuntimeError("Something bad happened during ILP")

    def optimal(atomic):
        return ilp_var[atomic_index[atomic]].value() == 1

    optimal_atomics = filter(optimal, iterkeys(atomic_index))
    return tuple(optimal_atomics)


def factorise_atomics(monomial_sum, optimal_atomics, argument_indices):
    """Group and factorise monomials using a list of atomics as common
    subexpressions. Create new monomials for each group and optimise them recursively.

    :arg monomial_sum: A :class:`MonomialSum` object, or a iterable collection
                       of monomials, all of the monomials should have the
                       same sum indices
    :arg optimal_atomics: list of tuples of atomics to be used as common subexpression
    :arg argument_indices: tuple of argument indices

    :returns: a factorised :class:`MonomialSum` object, or the original object
    if no changes are made. If original input is an iterable of :class`Monomial`s,
    convert it to a :class:`MonomialSum` object.
    """
    #TODO: decide on input/output choice, iterable of monomials or MonomialSum
    if not optimal_atomics or len(monomial_sum) < 2:
        # Nothing to do
        if isinstance(monomial_sum, MonomialSum):
            return monomial_sum
        else:
            # input is list of monomials
            new_monomial_sum = MonomialSum()
            for m in monomial_sum:
                new_monomial_sum.add(*m)
            return new_monomial_sum

    # Group monomials with respect to each optimal atomic
    def group_key(monomial):
        for oa in optimal_atomics:
            if oa in monomial.atomics:
                return oa
        assert False, "Expect at least one optimal atomic per monomial."
    factor_group = groupby(monomial_sum, key=group_key)

    # We should not drop monomials
    assert sum(len(ms) for _, ms in factor_group) == len(monomial_sum)

    sum_indices = next(iter(monomial_sum)).sum_indices
    new_monomial_sum = MonomialSum()
    for oa, monomials in factor_group:
        # Create new MonomialSum for the factorised out terms
        sub_monomial_sum = MonomialSum()
        for monomial in monomials:
            atomics = list(monomial.atomics)
            atomics.remove(oa)  # remove common factor
            sub_monomial_sum.add((), atomics, monomial.rest)
        sub_monomial_sum = optimise_monomials(sub_monomial_sum, argument_indices)
        assert len(sub_monomial_sum) > 0
        if len(sub_monomial_sum) == 1:
            # result is a product, add back the common atomics then add to
            # new MonomialSum directly
            sub_monomial, = sub_monomial_sum
            new_monomial_sum.add(sum_indices, sub_monomial.atomics + (oa,), sub_monomial.rest)
        else:
            # result is a sum, we need to create a new node
            node = monomial_sum_to_expression(sub_monomial_sum)
            if set(argument_indices) & set(node.free_indices):
                new_monomial_sum.add(sum_indices, (oa, node), one)
            else:
                new_monomial_sum.add(sum_indices, (oa, ), node)
    return new_monomial_sum


def optimise_monomial_sum(monomial_sum, argument_indices):
    """Choose optimal common atomic subexpressions and factorise a
    :class:`MonomialSum` object.

    :arg monomial_sum: a :class:`MonomialSum` object
    :arg argument_indices: tuple of argument indices

    :returns: factorised `MonomialSum` object
    """
    # Group monomials by their sum indices
    groups = groupby(monomial_sum, key=lambda m: frozenset(m.sum_indices))
    new_monomial_sums = []
    for _, monomials in groups:
        new_monomial_sums.append(optimise_monomials(monomials, argument_indices))

    return MonomialSum.sum(*new_monomial_sums)


def optimise_monomials(monomials, argument_indices):
    """Choose optimal common atomic subexpressions and factorise an iterable
    of monomials.

    :arg monomials: an iterable of monomials, all of which should have the same
    sum indices
    :arg argument_indices: tuple of argument indices

    :returns: factorised `MonomialSum` object
    """
    # Get the optimal atomics to factorise
    optimal_atomics = find_optimal_atomics(monomials, argument_indices)
    # Factorise with the optimal atomics and collect the results
    return factorise_atomics(monomials, optimal_atomics, argument_indices)


def count_flop(expression):
    """Count the total floating point operations required to compute a GEM node.
    This function assumes that all subnodes that occur more than once induce a
    temporary, and are therefore only computed once.

    :arg expression: GEM expression

    :returns: total number of FLOPs to compute the GEM expression
    """
    flop = 0
    for node in traversal([expression]):
        if isinstance(node, (Sum, Product, Division, MathFunction, Comparison, Power)):
            flop += numpy.prod([idx.extent for idx in node.free_indices])
        elif isinstance(node, IndexSum):
            flop += numpy.prod([idx.extent for idx in node.multiindex + node.free_indices])
        elif isinstance(node, (Terminal, Indexed, ListTensor, FlexiblyIndexed,
                               LogicalOr, LogicalNot, LogicalAnd, Conditional)):
            pass
        else:
            raise NotImplementedError("Do not know how to count flops of type {0}".format(type(node)))
    return flop
