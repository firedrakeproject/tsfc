"""Pretty-printing GEM expressions."""

from __future__ import absolute_import, print_function, division

from collections import defaultdict
import itertools

from singledispatch import singledispatch

from gem import gem
from gem.node import collect_refcount, post_traversal


class Context(object):
    def __init__(self):
        expr_counter = itertools.count(1)
        self.expr_name = defaultdict(lambda: "${}".format(next(expr_counter)))
        index_counter = itertools.count(1)
        self.index_name = defaultdict(lambda: "i_{}".format(next(index_counter)))
        self.index_names = set()

    def force_expression(self, expr):
        assert isinstance(expr, gem.Node)
        return self.expr_name[expr]

    def expression(self, expr):
        assert isinstance(expr, gem.Node)
        return self.expr_name.get(expr)

    def index(self, index):
        assert isinstance(index, gem.Index)
        if index.name is None:
            name = self.index_name[index]
        elif index.name not in self.index_names:
            name = index.name
            self.index_name[index] = name
        else:
            name_ = index.name
            for i in itertools.count(1):
                name = "{}~{}".format(name_, i)
                if name not in self.index_names:
                    break
        self.index_names.add(name)
        return name


global_context = Context()


def pprint(expression_dags, context=global_context):
    refcount = collect_refcount(expression_dags)

    def force(node):
        if isinstance(node, gem.Variable):
            return False
        if node.shape:
            return True
        if isinstance(node, (gem.Constant, gem.Indexed, gem.FlexiblyIndexed)):
            return False
        return refcount[node] > 1

    for node in post_traversal(expression_dags):
        if force(node):
            context.force_expression(node)

        name = context.expression(node)
        if name is not None:
            print(make_decl(node, name, context), '=', to_str(node, context, top=True))

    for i, root in enumerate(expression_dags):
        print(make_decl(root, "#%d" % (i + 1), context), '=', to_str(root, context))


def make_decl(node, name, ctx):
    result = name
    if node.shape:
        result += '[' + ','.join(map(repr, node.shape)) + ']'
    if node.free_indices:
        result += '{' + ','.join(map(ctx.index, node.free_indices)) + '}'
    return result


def to_str(expr, ctx, top=False):
    if not top and ctx.expression(expr):
        result = ctx.expression(expr)
        if expr.free_indices:
            result += '{' + ','.join(map(ctx.index, expr.free_indices)) + '}'
        return result
    else:
        return _to_str(expr, ctx)


@singledispatch
def _to_str(node, ctx):
    raise AssertionError("GEM node expected")


@_to_str.register(gem.Node)
def _to_str_node(node, ctx):
    front_args = [repr(getattr(node, name)) for name in node.__front__]
    back_args = [repr(getattr(node, name)) for name in node.__back__]
    children = [to_str(child, ctx) for child in node.children]
    return "%s(%s)" % (type(node).__name__, ", ".join(front_args + children + back_args))


@_to_str.register(gem.Zero)
def _to_str_zero(node, ctx):
    assert not node.shape
    return "%g" % node.value


@_to_str.register(gem.Literal)
def _to_str_literal(node, ctx):
    if node.shape:
        return repr(node.array.tolist())
    else:
        return "%g" % node.value


@_to_str.register(gem.Variable)
def _to_str_variable(node, ctx):
    return node.name


@_to_str.register(gem.ListTensor)
def _to_str_listtensor(node, ctx):
    def recurse_rank(array):
        if len(array.shape) > 1:
            return '[' + ', '.join(map(recurse_rank, array)) + ']'
        else:
            return '[' + ', '.join(to_str(item, ctx) for item in array) + ']'

    return recurse_rank(node.array)


@_to_str.register(gem.Indexed)
def _to_str_indexed(node, ctx):
    child, = node.children
    result = to_str(child, ctx)
    # if child.free_indices:
    #     result += '{' + ','.join(index_names[i] for i in child.free_indices) + '}'
    dimensions = []
    for index in node.multiindex:
        if isinstance(index, gem.Index):
            dimensions.append(ctx.index(index))
        elif isinstance(index, int):
            dimensions.append(str(index))
        else:
            assert False
    result += '[' + ','.join(dimensions) + ']'
    return result


@_to_str.register(gem.FlexiblyIndexed)
def _to_str_flexiblyindexed(node, ctx):
    child, = node.children
    result = to_str(child, ctx)
    dimensions = []
    for offset, idxs in node.dim2idxs:
        parts = []
        if offset:
            parts.append(str(offset))
        for index, stride in idxs:
            index_name = ctx.index(index)
            assert stride
            if stride == 1:
                parts.append(index_name)
            else:
                parts.append(index_name + "*" + str(stride))
        if parts:
            dimensions.append(' + '.join(parts))
        else:
            dimensions.append('0')
    if dimensions:
        result += '[' + ','.join(dimensions) + ']'
    return result


@_to_str.register(gem.IndexSum)
def _to_str_indexsum(node, ctx):
    index, = node.multiindex
    return u'\u03A3_{' + ctx.index(index) + '}(' + to_str(node.children[0], ctx) + ')'


@_to_str.register(gem.ComponentTensor)
def _to_str_componenttensor(node, ctx):
    return to_str(node.children[0], ctx) + '|' + ','.join(ctx.index(i) for i in node.multiindex)


@_to_str.register(gem.Sum)
def _to_str_sum(node, ctx):
    children = [to_str(child, ctx) for child in node.children]
    return "(" + " + ".join(children) + ")"


@_to_str.register(gem.Product)
def _to_str_product(node, ctx):
    children = [to_str(child, ctx) for child in node.children]
    return "(" + "*".join(children) + ")"


@_to_str.register(gem.MathFunction)
def _to_str_mathfunction(node, ctx):
    child, = node.children
    return node.name + "(" + to_str(child, ctx) + ")"
