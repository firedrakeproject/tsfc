from __future__ import absolute_import, print_function, division
from singledispatch import singledispatch
from gem.gem import Product, Sum, Node, Terminal, Constant, Scalar, Division, Zero
from gem.node import Memoizer, MemoizerArg, reuse_if_untouched, reuse_if_untouched_arg
from gem.optimise import one

# ---------------------------------------
# Count flop of expression, "as it is", no reordering etc
@singledispatch
def _count_flop(node, self, index):
    raise AssertionError("cannot handle type %s" % type(node))

@_count_flop.register(Sum)
@_count_flop.register(Product)
@_count_flop.register(Division)
def _count_flop_common(node, self, index):
    # The sum/product itself
    flop = 1
    for i in node.free_indices:
        flop *= i.extent
    # Hoisting the factors
    for child in node.children:
        flop += self(child, child.free_indices)
    return flop

@_count_flop.register(Scalar)
@_count_flop.register(Constant)
def _count_flop_const(node, self, index):
    return 0

def count_flop(expression, index):
    mapper = MemoizerArg(_count_flop)
    return mapper(expression, index)


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

def expand_all_product(node):
    mapper = Memoizer(_expand_all_product)
    return mapper(node)


# ---------------------------------------
# Collect terms

def _collect_terms(node, self, node_type):
    """Helper function to recursively collect all children into a list from
    :param:`node` and its children of class :param:`node_type`.

    :param node: root of expression
    :param node_type: class of node (e.g. Sum or Product)
    :param sort_func: function to sort the returned list
    :return: list of all terms
    """
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

def collect_terms(node, node_type):
    mapper = MemoizerArg(_collect_terms)
    return mapper(node, node_type)

def _flatten_sum(node, self, index):
    """
    factorise into sum of products, group product factors based on dependency
    on index:
    1) nothing (key = 0)
    2) i (key = 1)
    3) (i)j (key = j)
    4) (i)k (key = k)
    ...
    :param node:
    :param self:
    :return:
    """
    sums = collect_terms(node, Sum)
    result = []
    for sum in sums:
        d = {}
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
        result.append(d)
    return result

def flatten_sum(node, index):
    mapper = MemoizerArg(_flatten_sum)
    return mapper(node, index)


# ---------------------------------------
# Find common factor
def _find_common_factor(node, self, index):
    # index = (free index, i)
    factors = flatten_sum(node, index[0])
    from collections import Counter
    return list(reduce(lambda a,b : a&b, [Counter(f[index[1]]) for f in factors]))

def find_common_factor(node, index):
    mapper = MemoizerArg(_find_common_factor)
    return mapper(node, index)


# ---------------------------------------
# Factorisation

def _factorise_i(node, self, index):
    # index = (free index, i)
    sumproduct = flatten_sum(node, index[0])
    # collect all factors with correct index
    factors = set([p[index[1]][0] for p in sumproduct])  # only 1 element due to linearity
    sums = {}
    for f in factors:
        sums[f] = []
    sums[0] = []
    for p in sumproduct:
        p_const = reduce(Product, p[0], one)
        p_i = reduce(Product, p[1], one)
        p_jk = reduce(Product, [p[i][0] for i in index[0] if i != index[1]], one)
        new_node = reduce(Product, [p_const, p_i, p_jk], one)
        if p[index[1]]:
            sums[p[index[1]][0]].append(new_node)
        else:
            sums[0].append(new_node)
    sum_i = []
    new_index = list(index[0])
    new_index.remove(index[1])
    new_index = tuple(new_index)
    for f in factors:
        # recursion
        sum_i.append(Product(f, factorise(reduce(Sum, sums[f], Zero()), new_index)))
    return reduce(Sum, sum_i + sums[0], Zero())

def factorise_i(node, index):
    mapper = MemoizerArg(_factorise_i)
    return mapper(node, index)

def factorise(node, index):
    flop = count_flop(node, index)
    optimal_i = None
    node = expand_all_product(node)
    sumproduct = flatten_sum(node, index)
    factor_const = find_common_factor(node, (index, 0))
    factor_1 = find_common_factor(node, (index, 1))
    if factor_const or factor_1:
        for p in sumproduct:
            for x in factor_const:
                p[0].remove(x)
            for x in factor_1:
                p[1].remove(x)
    child_sum = []
    for p in sumproduct:
        child_sum.append(reduce(Product, p[0] + p[1] + [x for i in index for x in p[i]], one))
    p_const = reduce(Product, factor_const, one)
    p_1 = reduce(Product, factor_1, one)
    child = reduce(Sum, child_sum, Zero())
    if index:
        for i in index:
            child = factorise_i(child, (index, i))
        new_node = Product(Product(p_const,p_1), child)
        new_flop = count_flop(new_node, index)
        if new_flop<flop:
            optimal_i = i
            flop = new_flop
    if optimal_i:
        child = factorise_i(child, (index, optimal_i))
    return Product(Product(p_const, p_1), child)

# ---------------------------------------
# Collect factors
# assuming only + and *, all expanded out

# currently not memoized yet
def collect_factors(node, result):
    if isinstance(node, Terminal):
        return
    a, b = node.children
    if isinstance(node, Product):
        if a in result:
            result[a] += [b]
        else:
            result[a] = [b]
        if b in result:
            result[b] += [a]
        else:
            result[b] = [a]
        return
    collect_factors(a, result)
    collect_factors(b, result)
    return

# ---------------------------------------
# Factorisation

def expand_factor(expression, factor):
    # Need to be memoized
    if isinstance(expression, Sum):
        a, b = expression.children
        return Sum(expand_factor(a, factor), expand_factor(b, factor))
    elif isinstance(expression, Product):
        a, b = expression.children
        if a == factor or b == factor:
            if a == factor:
                common = _collect_terms(b, Sum)
            else:
                common = _collect_terms(a, Sum)
            if not common:
                return expression
            else:
                products = [Product(factor, x) for x in common]
                return reduce(Sum, products)
        else:
            return expression
    else:
        raise AssertionError("only Sum and Product")



def extract_factor_old(expression, factor):
    queue = [expression.children[0], expression.children[1]]
    common = []
    rest = []
    while len(queue)>0:
        node = queue.pop(0)
        a, b = node.children
        if isinstance(node, Product):
            if a == factor:
                common.append(b)
            elif b == factor:
                common.append(a)
            else:
                rest.append(node)
        elif isinstance(node, Sum):
            queue.extend(node.children)
    if not rest:
        return Product(factor, reduce(Sum, common))
    return Sum(Product(factor, reduce(Sum, common)), reduce(Sum, rest))

def _factorise(expression, index):
    nodes = {}
    collect_factors(expression, nodes)
    return factorise(expression, index, nodes)

def factorise_old(expression, index, nodes):
    # find optimal factorised expression, return (optimal expr, flops)
    flops = count_flop(expression, index)
    if not isinstance(expression, Sum):
        # nothing to factorise
        return (expression, flops)
    # collect all possible factors
    queue = []
    for node, factors in nodes.iteritems():
        if len(factors)>1:
            queue.append(node)
    optimal_factor = None
    for factor in queue:
        new_node = reduce(Sum, nodes[factor])
        old_list = list(nodes[factor])  # remember old list
        for node in nodes[factor]:
            nodes[node].remove(factor)  # remove common factor
        nodes[factor] = [new_node]
        if new_node in nodes:
            nodes[new_node].append(factor)
        else:
            nodes[new_node] = [factor]
        expression = extract_factor(expression, factor)  # do the factorisation
        _, new_flops = factorise(expression, index, nodes)
        if new_flops < flops:
            flops = new_flops
            optimal_factor = factor
        # restore expression
        expression = expand_factor(expression, factor)
        # restore node dictionary
        for node in old_list:
            nodes[node].append(factor)
        nodes[new_node].remove(factor)
        nodes[factor] = old_list
    if not optimal_factor:
        return (expression, flops)
    else:
        return (extract_factor(expression, optimal_factor), flops)
