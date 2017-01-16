from __future__ import absolute_import, print_function, division
from singledispatch import singledispatch
from gem.gem import Product, Sum, Node, Terminal, Constant, Scalar, Division
from gem.node import Memoizer, MemoizerArg, reuse_if_untouched, reuse_if_untouched_arg
from gem.optimise import _collect_terms

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



def extract_factor(expression, factor):
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

def factorise(expression, index, nodes):
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
