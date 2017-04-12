from __future__ import absolute_import, print_function, division

import numpy
import itertools
from functools import partial
from six import iteritems, iterkeys, itervalues
from six.moves import filter, filterfalse
from collections import OrderedDict, defaultdict, Counter
from singledispatch import singledispatch
from gem.optimise import (replace_division, fast_sum_factorise,
                          associate_product, associate_sum, traverse_product, traverse_sum)
from gem.refactorise import MonomialSum, ATOMIC, COMPOUND, OTHER, collect_monomials
from gem.impero_utils import preprocess_gem
from gem.node import traversal, Memoizer
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
    argument_indices = tuple([i for indices in argument_multiindices for i in indices])
    print('Building sharing graph...')
    sharing_graph = build_sharing_graph(node, {}, set(), set(argument_indices))
    print('Finished building sharing graph')
    # return _test(node, sharing_graph, argument_indices)
    levels, start, end = find_optimal_expansion_levels(node, sharing_graph, set(argument_indices))
    expanded_node_ms = expand_node(node, argument_indices, levels, start, end)
    expanded_node_ms = optimise_monomial_sum(expanded_node_ms, argument_indices)
    return monomial_sum_to_expression(expanded_node_ms)


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
    index = itertools.count()
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
    sharing_graph: node -> [dependencies]
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
            dependency = frozenset([p for p in products if argument_indices.intersection(p.free_indices)])
            sharing_graph.setdefault(node, []).append(dependency)
            for child in dependency:
                sharing_graph = build_sharing_graph(child, sharing_graph, seen, argument_indices)
    elif isinstance(node, Product):
        _, products = traverse_product(node)
        dependency = frozenset([p for p in products if argument_indices.intersection(p.free_indices)])
        sharing_graph.setdefault(node, []).append(dependency)
        for child in dependency:
            sharing_graph = build_sharing_graph(child, sharing_graph, seen, argument_indices)
    elif isinstance(node, IndexSum):
        sharing_graph[node] = list(node.children)
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
        for dependencies in sharing_graph[node]:
            for child in dependencies:
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
            for product in sharing_graph[node]:
                for child in product:
                    min_parent_level[child] = l
    min_parent_level[root] = -1
    best_cost = 0
    best_start, best_end = 0, 0
    for start in range(min_level, max_level):
        for end in range(start+1, max_level + 1):
            new_sharing_graph = {}  # hold modified dependencies
            total_cost, outer_cost = 0, 0
            for node, products in iteritems(sharing_graph):
                new_sharing_graph[node] = list(products)
            for l in range(end-2, start-1, -1):
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
