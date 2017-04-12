from __future__ import absolute_import, print_function, division

import numpy
import itertools
from functools import partial
from six import iteritems, iterkeys, itervalues
from six.moves import filter, filterfalse
from collections import OrderedDict
from gem.optimise import (replace_division, associate_sum, associate_product,
                          unroll_indexsum, replace_delta, remove_componenttensors)
from gem.refactorise import (MonomialSum, ATOMIC, COMPOUND, OTHER,
                             collect_monomials)
from gem.node import traversal
from gem.gem import (Product, Sum, Comparison, Conditional, Division, Indexed,
                     IndexSum, MathFunction, Power, Failure, one, index_sum)


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
    return optimise_expressions(expressions, quadrature_multiindex, argument_multiindices)


def optimise(node, quadrature_multiindex, argument_multiindices):
    """Optimise a GEM expression through factorisation.

    :arg node: GEM expression
    :arg quadrature_multiindex: quadrature multiindex (tuple)
    :arg argument_multiindices: tuple of argument multiindices,
                                one multiindex for each argument

    :returns: factorised GEM expression
    """
    argument_indices = tuple([i for indices in argument_multiindices for i in indices])

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
    # Apply argument factorisation unconditionally
    classifier = partial(classify, set(argument_indices))
    monomial_sum, = collect_monomials([node], classifier)
    monomial_sum = optimise_monomial_sum(monomial_sum, argument_indices)
    return monomial_sum_to_expression(monomial_sum)


def optimise_expressions(expressions, quadrature_multiindex, argument_multiindices):
    """Perform loop optimisations on GEM DAGs

    :arg expressions: list of GEM DAGs
    :arg quadrature_multiindex: quadrature multiindex (tuple)
    :arg argument_multiindices: tuple of argument multiindices,
                                one multiindex for each argumen

    :returns: list of optimised GEM DAGs
    """
    # Propagate Failure nodes
    for n in traversal(expressions):
        if isinstance(n, Failure):
            return expressions
    return [optimise(node, quadrature_multiindex, argument_multiindices) for node in expressions]


def index_extent(factor, argument_indices):
    """Compute the product of the extents of argument indices of a GEM expression

    :arg factor: GEM expression
    :arg argument_indices: set of argument indices

    :returns: product of extents of argument indices
    """
    return numpy.product([i.extent for i in set(factor.free_indices).intersection(argument_indices)])


def unique_sum_indices(monomial_sum):
    """Create a generator of unique sum indices of monomials in a monomial sum.

    :arg monomial_sum: :class:`MonomialSum` object

    :returns: a generator of unique sum indices
    """
    seen = set()
    for monomial in monomial_sum:
        fs = frozenset(monomial.sum_indices)
        if fs not in seen:
            seen.add(fs)
            yield monomial.sum_indices


def monomial_sum_to_expression(monomial_sum):
    """Convert a monomial sum to a GEM expression. Uses associate_product() and
    associate_sum() to promote hoisting in the subsequent code generation.

    :arg monomial_sum: :class:`MonomialSum` object

    :returns: GEM expression
    """
    indexsums = []  # The result is summation of indexsums
    sum_indices_set_map = {}  # fronzenset(sum_indices) -> sum_indices
    monomial_groups = OrderedDict()  # frozonset(sum_indices) -> [(atomics, rest)]
    # Group monomials according to their sum indices
    for monomial in monomial_sum:
        if not monomial.sum_indices:
            # IndexSum(reduce(Product, atomics, rest), sum_indices)
            product, _ = associate_product(monomial.atomics + (monomial.rest,))
            indexsums.append(product)
        else:
            fs = frozenset(monomial.sum_indices)
            sum_indices_set_map.setdefault(fs, monomial.sum_indices)
            monomial_groups.setdefault(fs, []).append((monomial.atomics, monomial.rest))

    # Create IndexSum's from each monomial group
    for sum_indices_set, list_atomics_rest in iteritems(monomial_groups):
        sum_indices = sum_indices_set_map[sum_indices_set]
        all_atomics, all_rest = zip(*list_atomics_rest)
        if len(all_atomics) == 1:
            # Just one term, add to indexsums directly
            atomics, = all_atomics
            rest, = all_rest
            product, _ = associate_product(atomics + (rest,))
            indexsums.append(IndexSum(product, sum_indices))
        else:
            # Create one product for each monomial
            products = [associate_product(atomics + (rest,))[0] for atomics, rest in zip(all_atomics, all_rest)]
            indexsums.append(IndexSum(associate_sum(products)[0], sum_indices))

    return associate_sum(indexsums)[0]


def find_optimal_atomics(monomial_sum, sum_indices_set, argument_indices):
    """Find optimal atomic common subexpressions, which produce least number of
    terms in the resultant IndexSum when factorised.

    :arg monomial_sum: A :class:`MonomialSum` object
    :arg sum_indices_set: frozenset of sum indices to match the monomials
    :arg argument_indices: tuple of argument indices

    :returns: list of atomic GEM expressions
    """
    index = itertools.count()  # counter for variables used in ILP
    atomic_index = OrderedDict()  # Atomic GEM node -> int
    connections = []
    # add connections (list of tuples, items in each tuple form a product)
    for monomial in monomial_sum:
        if frozenset(monomial.sum_indices) == sum_indices_set:
            connection = []
            for atomic in monomial.atomics:
                if atomic not in atomic_index:
                    atomic_index[atomic] = next(index)
                connection.append(atomic_index[atomic])
            connections.append(tuple(connection))

    if len(atomic_index) == 0:
        return ((), ())
    if len(atomic_index) == 1:
        return ((next(iterkeys(atomic_index)), ), ())

    # set up the ILP
    import pulp as ilp
    ilp_prob = ilp.LpProblem('gem factorise', ilp.LpMinimize)
    ilp_var = ilp.LpVariable.dicts('node', range(len(atomic_index)), 0, 1, ilp.LpBinary)

    # Objective function
    # Minimise number of factors to pull. If same number, favour factor with larger extent
    big = 1e20  # some arbitrary big number
    ilp_prob += ilp.lpSum(ilp_var[index] * (big - index_extent(atomic, argument_indices)) for atomic, index in iteritems(atomic_index))

    # constraints
    for connection in connections:
        ilp_prob += ilp.lpSum(ilp_var[index] for index in connection) >= 1

    ilp_prob.solve()
    if ilp_prob.status != 1:
        raise RuntimeError("Something bad happened during ILP")

    def optimal(atomic):
        return ilp_var[atomic_index[atomic]].value() == 1

    optimal_atomics = filter(optimal, iterkeys(atomic_index))
    other_atomics = filterfalse(optimal, iterkeys(atomic_index))

    return (tuple(optimal_atomics), tuple(other_atomics))


def factorise_atomics(monomial_sum, optimal_atomics, argument_indices):
    """Group and factorise monomials using a list of atomics as common
    subexpressions. Create new monomials for each group and optimise them recursively.

    :arg monomial_sum: a :class:`MonomialSum` object
    :arg optimal_atomics: list of tuples of atomics to be used as common subexpression
                          and the frozenset of their sum indices
    :arg argument_indices: tuple of argument indices

    :returns: a factorised :class:`MonomialSum` object, or the original object
    if no changes are made
    """
    if not optimal_atomics:
        return monomial_sum
    if len(monomial_sum.ordering) < 2:
        return monomial_sum
    new_monomial_sum = MonomialSum()
    # Group monomials with respect to each optimal atomic
    factor_group = OrderedDict()
    for monomial in monomial_sum:
        for sum_indices, oa in optimal_atomics:
            if frozenset(monomial.sum_indices) == frozenset(sum_indices) and oa in monomial.atomics:
                # Add monomial to the list of corresponding optimal atomic
                factor_group.setdefault((sum_indices, oa), []).append(monomial)
                break
        else:
            # Add monomials that do no have argument factors to new MonomialSum
            new_monomial_sum.add(monomial.sum_indices, monomial.atomics, monomial.rest)
    # We should not drop monomials
    assert sum(map(len, itervalues(factor_group))) + len(list(new_monomial_sum)) == len(list(monomial_sum))

    for (sum_indices, oa), monomials in iteritems(factor_group):
        if len(monomials) == 1:
            # Just one monomial with this group, add to new MonomialSum straightaway
            monomial, = monomials
            new_monomial_sum.add(monomial.sum_indices, monomial.atomics, monomial.rest)
            continue
        all_atomics = []  # collect all atomics from monomials
        all_rest = []  # collect all rest from monomials
        for monomial in monomials:
            _atomics = list(monomial.atomics)
            _atomics.remove(oa)  # remove common factor
            all_atomics.append(_atomics)
            all_rest.append(monomial.rest)
        # Create new MonomialSum for the factorised out terms
        sub_monomial_sum = MonomialSum()
        for _atomics, _rest in zip(all_atomics, all_rest):
            sub_monomial_sum.add((), _atomics, _rest)
        sub_monomial_sum = optimise_monomial_sum(sub_monomial_sum, argument_indices)
        assert len(list(sub_monomial_sum)) > 0
        if len(list(sub_monomial_sum)) == 1:
            # result is a product, add to new MonomialSum directly
            sub_monomial, = sub_monomial_sum
            new_atomics = sub_monomial.atomics
            new_atomics += (oa,)  # add back common factor
            new_rest = sub_monomial.rest
        else:
            # result is a sum, we need to create new node
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
    """Choose optimal common atomic subexpressions and factorise a
    :class:`MonomialSum` object.

    :arg monomial_sum: a :class:`MonomialSum` object
    :arg argument_indices: tuple of argument indices

    :returns: factorised `MonomialSum` object
    """
    all_optimal_atomics = []  # [(sum_indces, optimal_atomics)]
    for sum_indices in unique_sum_indices(monomial_sum):
        # throw away other atomics here
        optimal_atomics, _ = find_optimal_atomics(monomial_sum, frozenset(sum_indices), argument_indices)
        all_optimal_atomics.extend([(sum_indices, atomic) for atomic in optimal_atomics])
    # This algorithm is O(N!), where N = len(optimal_atomics)
    # we could truncate the optimal_atomics list at say 10
    return factorise_atomics(monomial_sum, all_optimal_atomics, argument_indices)


def count_flop_node(node):
    """Count number of FLOPs at a particular GEM node, without recursing
    into childrens

    :arg node: GEM expression

    :returns: number of FLOPs to compute this node, assuming the children have
              been computed already
    """
    if isinstance(node, (Sum, Product, Division, MathFunction, Comparison, Power)):
        return numpy.prod([idx.extent for idx in node.free_indices])
    elif isinstance(node, IndexSum):
        return numpy.prod([idx.extent for idx in node.multiindex + node.free_indices])
    else:
        return 0


def count_flop(node):
    """Count the total floating point operations required to compute a GEM node.
    This function assumes that all subnodes that occur more than once induce a
    temporary, and are therefore only computed once.

    :arg node: GEM expression

    :returns: total number of FLOPs to compute the GEM expression
    """
    return sum(map(count_flop_node, traversal([node])))