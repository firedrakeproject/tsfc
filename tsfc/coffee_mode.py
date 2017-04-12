from __future__ import absolute_import, print_function, division

import numpy
import itertools
from functools import partial
from six import iteritems, iterkeys, itervalues
from six.moves import filter, filterfalse

from collections import OrderedDict, defaultdict, Counter
from gem.optimise import (replace_division, associate_sum, associate_product,
                          unroll_indexsum, replace_delta, traverse_product,
                          remove_componenttensors, traverse_sum)
from gem.refactorise import (MonomialSum, ATOMIC, COMPOUND, OTHER,
                             collect_monomials)
from gem.node import traversal, Memoizer
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
    print('Building sharing graph...')
    sharing_graph = build_sharing_graph(node, {}, set(), set(argument_indices))
    print('Finished building sharing graph')
    levels, start, end = find_optimal_expansion_levels(node, sharing_graph, set(argument_indices))
    expanded_node_ms = expand_node(node, argument_indices, levels, start, end)
    expanded_node_ms = optimise_monomial_sum(expanded_node_ms, argument_indices)
    return monomial_sum_to_expression(expanded_node_ms)


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


def _substitute_node(node, self):
    try:
        return self.replace[node]
    except KeyError:
        new_children = list(map(self, node.children))
        if all(nc == c for nc, c in zip(new_children, node.children)):
            return node
        else:
            return node.reconstruct(*new_children)

def substitute_node(nodes, replace):
    mapper = Memoizer(_substitute_node)
    mapper.replace = replace
    return list(map(mapper, nodes))


def expand_node(node, argument_indices, levels, start, end):
    def classify(argument_indices, atomic_set, expression):
        if isinstance(expression, Conditional):
            return ATOMIC
        n = len(argument_indices.intersection(expression.free_indices))
        if n == 0:
            return OTHER
        elif n == 1 and isinstance(expression, Indexed):
            # Terminal
            return ATOMIC
        elif isinstance(expression, Sum) and expression in atomic_set:
            return ATOMIC
        else:
            return COMPOUND
    max_level = max(iterkeys(levels))
    atomic_set = set()
    for l in range(end, max_level+1):
        atomic_set.update(set(levels[l]))
    classifier = partial(classify, set(argument_indices), atomic_set)
    start_nodes_ms = collect_monomials(levels[start], classifier)
    start_nodes_ms = [optimise_monomial_sum(ms, argument_indices) for ms in start_nodes_ms]
    new_start_nodes = map(monomial_sum_to_expression, start_nodes_ms)
    # substitute node with new start nodes as terminal node
    substituted_node, = substitute_node([node], dict(zip(levels[start], new_start_nodes)))
    # Atomic sets are new start nodes plus previous terminal node
    atomic_set.update(set(new_start_nodes))
    classifier = partial(classify, set(argument_indices), atomic_set)
    monomialsum, = collect_monomials([node], classifier)
    return monomialsum


def build_sharing_graph(node, sharing_graph, seen, argument_indices):
    """ Create sharing graph of Sum and Indexed nodes based on their levels of
    nesting from root node.
    sharing_graph: node -> [products]
    """
    # Do not need to revisit nodes
    if node in seen:
        return sharing_graph
    # Do not need to look into constants
    if not argument_indices.intersection(set(node.free_indices)):
        return sharing_graph
    if isinstance(node, Indexed):
        # Terminal node
        sharing_graph[node] = []
    elif isinstance(node, Sum):
        sums = traverse_sum(node)
        for s in sums:
            _, products = traverse_product(s)
            product = frozenset([p for p in products if argument_indices.intersection(p.free_indices)])
            sharing_graph.setdefault(node, []).append(product)
            for child in product:
                sharing_graph = build_sharing_graph(child, sharing_graph, seen, argument_indices)
    elif isinstance(node, Product):
        _, products = traverse_product(node)
        product = frozenset([p for p in products if argument_indices.intersection(p.free_indices)])
        sharing_graph.setdefault(node, []).append(product)
        for child in product:
            sharing_graph = build_sharing_graph(child, sharing_graph, seen, argument_indices)
    elif isinstance(node, IndexSum):
        sharing_graph[node] = [frozenset(node.children)]
        sharing_graph = build_sharing_graph(node.children[0], sharing_graph, seen, argument_indices)
    else:
        raise NotImplementedError("Don't know how to do this yet.")

    seen.add(node)
    return sharing_graph


def find_optimal_expansion_levels(root, sharing_graph, argument_indices):
    def extents(indices):
        return numpy.prod([i.extent for i in indices])
    def assign_level(node, node_level, sharing_graph, current_level):
        if node in node_level:
            old_level = node_level[node]
            if old_level >= current_level:
                return node_level
        node_level[node] = current_level
        if node in sharing_graph:
            for product in sharing_graph[node]:
                for child in product:
                    node_level = assign_level(child, node_level, sharing_graph, current_level+1)
        return node_level
    node_level = {}
    node_level = assign_level(root, node_level, sharing_graph, 0)
    levels = {}
    for node, level in iteritems(node_level):
        levels.setdefault(level, []).append(node)
    min_level, max_level = min(iterkeys(levels)), max(iterkeys(levels))
    node_cost = {}
    for node in iterkeys(sharing_graph):
        # Aij*Aik*z + Bij*Bik*z ...
        cost = 0
        products = sharing_graph[node]
        for product in products:
            # Number of multiplications
            niter = extents(set().union(*[f.free_indices for f in product]).intersection(argument_indices))
            cost += len(product) * niter
        # Number of additions
        cost += extents(argument_indices.intersection(node.free_indices)) * (len(products) - 1)
        node_cost[node] = cost
    level_cost = {}
    for l in range(min_level, max_level):
        cost = 0
        for node in levels[l]:
            cost += node_cost[node]
        level_cost[l] = cost
    min_parent_level = {}
    for l in range(max_level, min_level-1, -1):
        for node in levels[l]:
            if node in sharing_graph:
                for product in sharing_graph[node]:
                    for child in product:
                        min_parent_level[child] = l
    min_parent_level[root] = -1
    best_cost = 0
    best_start, best_end = min_level, max_level
    for start in range(min_level, max_level):
        for end in range(start+1, max_level + 1):
            new_sharing_graph = {}  # hold modified dependencies
            total_cost, outer_cost = 0, 0
            for node, products in iteritems(sharing_graph):
                new_sharing_graph[node] = list(products)
            for l in range(end-1, start-1, -1):
                for node in levels[l]:
                    all_factors = []
                    for product in new_sharing_graph[node]:
                        # product is a tuple, (b1, b2) means b1*b2*z, (b1,) means b1*z
                        all_sub_factors = list(itertools.product(*[new_sharing_graph[child]
                                if node_level[child] < end else [frozenset([child])] for child in product]))
                        # this gives ((c1,), (c2,)), we want (c1, c2)
                        all_sub_factors = list(map(lambda x:frozenset.union(*x), all_sub_factors))
                        all_factors.extend(all_sub_factors)
                    unique_factors = list(set(all_factors))
                    outer_cost += len(all_factors) - len(unique_factors)
                    new_sharing_graph[node] = unique_factors
            # temporaries at start level
            inner_cost = 0
            for node in levels[start]:
                products = new_sharing_graph[node]
                for product in products:
                    niter = extents(set().union(*[f.free_indices for f in product]).intersection(argument_indices))
                    inner_cost += len(product) * niter
                inner_cost += extents(argument_indices.intersection(node.free_indices)) * (len(products) - 1)
            for l in range(start, end):
                for node in levels[l]:
                    if min_parent_level[node] >= start-1:
                        total_cost -= node_cost[node]
            total_cost += inner_cost + outer_cost
            print('start: {0}, end:{1}, flops:{2}'.format(start, end, total_cost))
            if total_cost < best_cost:
                best_start = start
                best_end = end
                best_cost = total_cost
    print('best cost: {0:,}, best levels: {1} -> {2},'.format(best_cost, best_start, best_end))
    return levels, best_start, best_end



def find_optimal_expansion_levels_old(sharing_graph, argument_indices):
    def extents(indices):
        return numpy.prod([i.extent for i in indices])
    dependencies = {}  # node -> [children]
    levels = {}
    min_parent_level = {}  # node -> minimum level of parents of node
    for node, (level, parents) in iteritems(sharing_graph):
        levels.setdefault(level, []).append(node)
        for parent in parents:
            dependencies.setdefault(parent, []).append(node)
        min_parent_level[node] = min(map(lambda p: sharing_graph[p][0], parents) or [0])

    min_level, max_level = min(iterkeys(levels)), max(iterkeys(levels))
    level_flops = {}
    for l in range(min_level, max_level):
        flops = 0
        for node in levels[l]:
            if isinstance(node, Sum):
                cost = 2 * len(dependencies[node]) - 1
                flops += extents(node.free_indices) * cost
            if isinstance(node, IndexSum):
                flops += extents(node.children[0].free_indices) * 2
        level_flops[l] = flops
        print('level {0}, flops {1}'.format(l, flops))

    best_flops = 1e20
    best_level = max_level
    for end in range(min_level + 1, max_level):
        new_dependencies = {}  # hold modified dependencies
        for node, parents in iteritems(dependencies):
            new_dependencies[node] = list(parents)
        # pre-start flops
        post_end_flops = 0
        outer_flops = 0
        for l in range(end, max_level):
            post_end_flops += level_flops[l]
        for start in range(end - 1, min_level - 1, -1):
            inner_flops = 0
            for node in levels[end]:
                if isinstance(node, Indexed):
                    continue
                all_factors = Counter()
                for child in new_dependencies[node]:
                    child_level, _ = sharing_graph[child]
                    if child_level <= end and isinstance(child, Sum):
                        # child need to be inlined
                        for child_child in new_dependencies[child]:
                            # Outer loop costs (hoisted)
                            # 1 product in the linear loop outside of child argument indices
                            outer_flops += extents(set(node.free_indices).difference(argument_indices))
                            all_factors[child_child] += 1
                    else:
                        # child should be kept as ATOMIC
                        all_factors[child] += 1
                unique_factors = list(all_factors)
                # Additions in outer loop
                for factor in unique_factors:
                    niter = extents(set(node.free_indices).difference(argument_indices))
                    outer_flops += (all_factors[factor] - 1) * niter
                # Inner loop costs, N multiplications and N-1 additions
                inner_flops += (2 * len(unique_factors) - 1) * extents(node.free_indices)
                # Update dependencies
                new_dependencies[node] = unique_factors
            for l in range(max_level, end, -1):
                for node in levels[l]:
                    if min_parent_level[node] < start and isinstance(node, Sum):
                        inner_flops += (len(new_dependencies[node]) * 2 - 1) * extents(node.free_indices)
            # pre-start cost
            pre_start_flops = 0
            for l in range(min_level, start):
                pre_start_flops += level_flops[l]

            total_flops = pre_start_flops + inner_flops + outer_flops + post_end_flops
            print('start: {0}, end:{1}, flops:{2}'.format(start, end, total_flops))
            if total_flops < best_flops:
                best_flops = total_flops
                best_start = start
                best_end = end
    print('best flops: {0:,}, best level: {1} -> {2},'.format(best_flops, best_start, best_end))
    return levels, best_start, best_end




def _test(root, node_level_parents, argument_indices):
    if not node_level_parents:
        return root
    levels = defaultdict()
    for node, (level, parents) in iteritems(node_level_parents):
        levels.setdefault(level, []).append(node)
    min_level, max_level = min(iterkeys(levels)), max(iterkeys(levels))
    best_flops = int(count_flop(root))
    best_node = root
    best_level = (min_level, min_level)
    print('original flops: {:,}'.format(best_flops))
    print('min level: {0}, max level: {1}'.format(min_level, max_level))
    for start in range(min_level, max_level+1):
        for end in range(start+1, max_level+1):
            current_node_ms = expand_node(root, argument_indices, levels, start, end)
            current_node_ms = optimise_monomial_sum(current_node_ms, argument_indices)
            current_node = monomial_sum_to_expression(current_node_ms)
            current_flops = int(count_flop(current_node))
            print('start: {0}, end: {1}, currentflops: {2:,}'.format(start, end, current_flops))
            if current_flops < best_flops:
                best_flops = current_flops
                best_node = current_node
                best_level = (start, end)
    print('best flops: {0:,}, best level: {1} -> {2},'.format(best_flops, *best_level))
    return best_node
