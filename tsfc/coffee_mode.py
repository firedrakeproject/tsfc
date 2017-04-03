from __future__ import absolute_import, print_function, division

import numpy
from functools import partial
from itertools import count
from six import iteritems, iterkeys, itervalues
from six.moves import filter, filterfalse
from collections import OrderedDict
from singledispatch import singledispatch
from gem.optimise import (replace_division, fast_sum_factorise,
                          associate_product, associate_sum)
from gem.refactorise import MonomialSum
from gem.impero_utils import preprocess_gem
from gem.node import traversal
from gem.gem import (Terminal, Product, Sum, Comparison, Conditional,
                     Division, Indexed, FlexiblyIndexed, IndexSum, ListTensor,
                     MathFunction, LogicalAnd, LogicalNot, LogicalOr,
                     Constant, Variable, Power, Failure, one)


import tsfc.vanilla as vanilla

flatten = vanilla.flatten

finalise_options = {'replace_delta': False, 'remove_componenttensors': False}


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
    # Need optimised roots for COFFEE
    expressions = vanilla.Integrals(expressions, quadrature_multiindex, argument_multiindices, parameters)
    expressions = preprocess_gem(expressions)
    expressions = replace_division(expressions)
    return optimise_expressions(expressions, quadrature_multiindex, argument_multiindices)


def optimise(node, quadrature_multiindex, argument_multiindices):
    from gem.refactorise import ATOMIC, COMPOUND, OTHER, collect_monomials
    argument_indices = tuple([i for indices in argument_multiindices for i in indices])

    def classify(argument_indices, expression):
        if isinstance(expression, Conditional):
            return ATOMIC
        n = len(argument_indices.intersection(expression.free_indices))
        if n == 0:
            return OTHER
        elif n == 1:
            if isinstance(expression, Indexed):
                return ATOMIC
            else:
                return COMPOUND
        else:
            return COMPOUND
    classifier = partial(classify, set(argument_indices))

    monomial_sum, = collect_monomials([node], classifier)
    monomial_sum = optimise_monomial_sum(monomial_sum, argument_indices)

    return monomial_sum_to_expression(monomial_sum)


def optimise_expressions(expressions, quadrature_multiindices, argument_multiindices):
    """
    perform loop optimisations on gem DAGs
    :param expressions: list of gem DAGs
    :param quadrature_multiindices: quadrature multiindices, tuple of tuples
    :param argument_multiindices: argument multiindices, tuple of tuples
    :return: list of optimised gem DAGs
    """
    if propagate_failure(expressions):
        return expressions
    return [optimise(node, quadrature_multiindices, argument_multiindices) for node in expressions]


def propagate_failure(expressions):
    """
    Check if any gem nodes is Failure. In that case there is no need for subsequent optimisation.
    """
    for n in traversal(expressions):
        if isinstance(n, Failure):
            return True
    return False


def index_extent(factor, argument_indices):
    """Compute the product of the indices of factor that appear in argument
    indices"""
    return numpy.product([i.extent for i in set(factor.free_indices).intersection(argument_indices)])


def unique_sum_indices(monomial_sum):
    """Returnes a generator of unique sum indices, together with their original
    ordering of :param: monomial_sum
    """
    seen = set()
    for (sum_indices_set, _), (sum_indices, _) in iteritems(monomial_sum.ordering):
        if sum_indices_set not in seen:
            seen.add(sum_indices_set)
            yield (sum_indices_set, sum_indices)


def monomial_sum_to_expression(monomial_sum):
    """
    Convert MonomialSum object to gem node. Use associate_product() and
    associate_sum() to promote hoisting in subsequent code generation.
    ordering ensures deterministic code generation.
    :return: gem node represented by :param: monomial_sum
    """
    indexsums = []  # The result is summation of indexsums
    monomial_group = OrderedDict()  # (sum_indices_set, sum_indices) -> [(atomics, rest)]
    # Group monomials according to their sum indices
    for key, (sum_indices, atomics) in iteritems(monomial_sum.ordering):
        sum_indices_set, _ = key
        rest = monomial_sum.monomials[key]
        if not sum_indices_set:
            indexsums.append(fast_sum_factorise(sum_indices, atomics + (rest,)))
        else:
            monomial_group.setdefault((sum_indices_set, sum_indices), []).append((atomics, rest))

    # Form IndexSum's from each monomial group
    for (_, sum_indices), monomials in iteritems(monomial_group):
        all_atomics, all_rest = zip(*monomials)
        if len(all_atomics) == 1:
            # Just one term, add to indexsums directly
            indexsums.append(fast_sum_factorise(sum_indices, all_atomics[0] + (all_rest[0],)))
        else:
            # Form products for each monomial
            products = [associate_product(atomics + (_rest,))[0] for atomics, _rest in zip(all_atomics, all_rest)]
            indexsums.append(IndexSum(associate_sum(products)[0], sum_indices))

    return associate_sum(indexsums)[0]


def find_optimal_atomics(monomial_sum, sum_indices_set, argument_indices):
    """Find list of optimal atomics which when factorised gives least number of
    terms in the indexed sum"""
    index = count()
    atomic_index = OrderedDict()  # Atomic gem node -> int
    connections = []
    # add connections (list of lists)
    for (_sum_indices, _), (_, atomics) in iteritems(monomial_sum.ordering):
        if _sum_indices == sum_indices_set:
            connection = []
            for atomic in atomics:
                if atomic not in atomic_index:
                    atomic_index[atomic] = next(index)
                connection.append(atomic_index[atomic])
            connections.append(tuple(connection))

    if len(atomic_index) == 0:
        return ((), ())
    if len(atomic_index) == 1:
        return ((list(atomic_index.keys())[0], ), ())

    # set up the ILP
    import pulp as ilp
    ilp_prob = ilp.LpProblem('gem factorise', ilp.LpMinimize)
    ilp_var = ilp.LpVariable.dicts('node', range(len(atomic_index)), 0, 1, ilp.LpBinary)

    # Objective function
    # Minimise number of factors to pull. If same number, favour factor with larger extent
    big = 10000000  # some arbitrary big number
    ilp_prob += ilp.lpSum(ilp_var[index] * (big - index_extent(atomic, argument_indices)) for atomic, index in iteritems(atomic_index))

    # constraints
    for connection in connections:
        ilp_prob += ilp.lpSum(ilp_var[index] for index in connection) >= 1

    ilp_prob.solve()
    if ilp_prob.status != 1:
        raise AssertionError("Something bad happened during ILP")

    def optimal(atomic):
        return ilp_var[atomic_index[atomic]].value() == 1

    optimal_atomics = filter(optimal, iterkeys(atomic_index))
    other_atomics = filterfalse(optimal, iterkeys(atomic_index))

    return (tuple(optimal_atomics), tuple(other_atomics))


def factorise_atomics(monomial_sum, optimal_atomics, argument_indices):
    """
    Group and factorise monomials based on a list of atomics. Create new
    monomials for each group and optimise them recursively.
    :param optimal_atomics: list of tuples of optimal atomics and their sum indices
    :return: new MonomialSum object with atomics factorised, or the original
     object if no changes are made
    """
    if not optimal_atomics:
        return monomial_sum
    if len(monomial_sum.ordering) < 2:
        return monomial_sum
    new_monomial_sum = MonomialSum()
    # Group monomials according to each optimal atomic
    factor_group = OrderedDict()
    for key, (_sum_indices, _atomics) in iteritems(monomial_sum.ordering):
        for (sum_indices_set, sum_indices), oa in optimal_atomics:
            if key[0] == sum_indices_set and oa in _atomics:
                # Add monomial key to the list of corresponding optimal atomic
                factor_group.setdefault(((sum_indices_set, sum_indices), oa), []).append(key)
                break
        else:
            # Add monomials that do no have argument factors to new MonomialSum
            new_monomial_sum.add(_sum_indices, _atomics, monomial_sum.monomials[key])
    # We should not drop monomials
    assert sum(map(len, itervalues(factor_group))) + len(new_monomial_sum.ordering) == len(monomial_sum.ordering)

    for ((sum_indices_set, sum_indices), oa), keys in iteritems(factor_group):
        if len(keys) == 1:
            # Just one monomials with this atomic, add to new MonomialSum straightaway
            _, _atomics = monomial_sum.ordering[keys[0]]
            _rest = monomial_sum.monomials[keys[0]]
            new_monomial_sum.add(sum_indices, _atomics, _rest)
            continue
        all_atomics = []  # collect all atomics from monomials
        all_rest = []  # collect all rest from monomials
        for key in keys:
            _, _atomics = monomial_sum.ordering[key]
            _atomics = list(_atomics)
            _atomics.remove(oa)  # remove common factor
            all_atomics.append(_atomics)
            all_rest.append(monomial_sum.monomials[key])
        # Create new MonomialSum for the factorised out terms
        sub_monomial_sum = MonomialSum()
        for _atomics, _rest in zip(all_atomics, all_rest):
            sub_monomial_sum.add((), _atomics, _rest)
        sub_monomial_sum = optimise_monomial_sum(sub_monomial_sum, argument_indices)
        assert len(sub_monomial_sum.ordering) != 0
        if len(sub_monomial_sum.ordering) == 1:
            # result is a product, add to new MonomialSum directly
            (_, new_atomics), = itervalues(sub_monomial_sum.ordering)
            new_atomics += (oa,)
            new_rest, = itervalues(sub_monomial_sum.monomials)
        else:
            # result is a sum, need to form new node
            new_node = monomial_sum_to_expression(sub_monomial_sum)
            new_atomics = [oa]
            new_rest = one
            if set(argument_indices) & set(new_node.free_indices):
                new_atomics.append(new_node)
            else:
                new_rest = new_node
        new_monomial_sum.add(sum_indices, new_atomics, new_rest)
    return new_monomial_sum


def optimise_monomial_sum(monomial_sum, argument_indices):
    all_optimal_atomics = []  # [((sum_indices_set, sum_indces), optimal_atomics))]
    for sum_indices_set, sum_indices in unique_sum_indices(monomial_sum):
        # throw away other atomics here
        optimal_atomics, _ = find_optimal_atomics(monomial_sum, sum_indices_set, argument_indices)
        all_optimal_atomics.extend([((sum_indices_set, sum_indices), _atomic) for _atomic in optimal_atomics])
    # This algorithm is O(N!), where N = len(optimal_atomics)
    # we could truncate the optimal_atomics list at say 10
    return factorise_atomics(monomial_sum, all_optimal_atomics, argument_indices)


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
    return sum(map(count_flop_node, traversal([node])))
