from __future__ import absolute_import, print_function, division
from singledispatch import singledispatch
from gem.gem import Product, Sum, Node, Terminal, Constant, Scalar, Division
from gem.node import Memoizer, MemoizerArg, reuse_if_untouched, reuse_if_untouched_arg
from gem.optimise import _collect_terms

# ---------------------------------------
# Count flop
@singledispatch
def _count_flop(node, self):
    raise AssertionError("cannot handle type %s" % type(node))


_count_flop.register(Node)(reuse_if_untouched)


@_count_flop.register(Sum)
def _count_flop_sum(node, self):
    a, b = node.children
    return self(a) + self(b) + 1

@_count_flop.register(Product)
def _count_flop_product(node, self):
    a, b = node.children
    return self(a) + self(b) + 1

@_count_flop.register(Division)
def _count_flop_division(node, self):
    a, b = node.children
    return self(a) + self(b) + 1

@_count_flop.register(Constant)
def _count_flop_const(node, self):
    return 0

@_count_flop.register(Scalar)
def _count_flop_scalar(node, self):
    return 0

def count_flop(expression):
    """Replace divisions with multiplications in expressions"""
    mapper = Memoizer(_count_flop)
    return mapper(expression)


# ---------------------------------------
# Collect factors
# assuming only + and *, all expanded out

# corrently not memoized yet
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

def _factorise(expression):
    nodes = {}
    collect_factors(expression, nodes)
    return factorise(expression, nodes)

def factorise(expression, nodes):
    # find optimal factorised expression, return (optimal expr, flops)
    flops = count_flop(expression)
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
        _, new_flops = factorise(expression, nodes)
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
