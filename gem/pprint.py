"""Pretty-printing GEM expressions."""

from __future__ import absolute_import
from __future__ import print_function

import collections
import itertools

from singledispatch import singledispatch

from gem import gem
from gem.node import collect_refcount, post_traversal


def pprint(expression_dags):
    refcount = collect_refcount(expression_dags)

    index_counter = itertools.count()
    index_names = collections.defaultdict(lambda: "i_%d" % (1 + next(index_counter)))

    def inlinable(node):
        return not (not isinstance(node, gem.Variable) and (node.shape or (refcount[node] > 1 and not isinstance(node, (gem.Literal, gem.Zero, gem.Indexed)))))

    i = 1
    stringified = {}
    for node in post_traversal(expression_dags):
        string = stringify(node, stringified, index_names)
        if inlinable(node):
            stringified[node] = string
        else:
            name = "$%d" % i
            print(make_decl(node, name, index_names) + ' = ' + string)
            stringified[node] = make_ref(node, name, index_names)
            i += 1

    for i, root in enumerate(expression_dags):
        print(make_decl(root, "#%d" % (i + 1), index_names) + ' = ' + stringified[root])


def make_decl(node, name, index_names):
    if node.shape:
        name += '[' + ','.join(map(repr, node.shape)) + ']'
    if node.free_indices:
        name += '{' + ','.join(index_names[i] for i in node.free_indices) + '}'
    return name


def make_ref(node, name, index_names):
    if node.free_indices:
        name += '{' + ','.join(index_names[i] for i in node.free_indices) + '}'
    return name


@singledispatch
def stringify(node, stringified, index_names):
    raise AssertionError("GEM node expected")


@stringify.register(gem.Node)
def stringify_node(node, stringified, index_names):
    front_args = [repr(getattr(node, name)) for name in node.__front__]
    back_args = [repr(getattr(node, name)) for name in node.__back__]
    children = [stringified[child] for child in node.children]
    return "%s(%s)" % (type(node).__name__, ", ".join(front_args + children + back_args))


@stringify.register(gem.Zero)
def stringify_zero(node, stringified, index_names):
    assert not node.shape
    return repr(0)


@stringify.register(gem.Literal)
def stringify_literal(node, stringified, index_names):
    if node.shape:
        return repr(node.array.tolist())
    else:
        return "%g" % node.value


@stringify.register(gem.Variable)
def stringify_variable(node, stringified, index_names):
    return node.name


@stringify.register(gem.ListTensor)
def stringify_listtensor(node, stringified, index_names):
    def recurse_rank(array):
        if len(array.shape) > 1:
            return '[' + ', '.join(map(recurse_rank, array)) + ']'
        else:
            return '[' + ', '.join(stringified[item] for item in array) + ']'

    return recurse_rank(node.array)


@stringify.register(gem.Indexed)
def stringify_indexed(node, stringified, index_names):
    child, = node.children
    result = stringified[child]
    if child.free_indices:
        result += '{' + ','.join(index_names[i] for i in child.free_indices) + '}'
    result += '[' + ','.join(index_names[i] for i in node.multiindex) + ']'
    return result


@stringify.register(gem.IndexSum)
def stringify_indexsum(node, stringified, index_names):
    return u'\u03A3_{' + index_names[node.index] + '}(' + stringified[node.children[0]] + ')'


@stringify.register(gem.ComponentTensor)
def stringify_componenttensor(node, stringified, index_names):
    return stringified[node.children[0]] + '|' + ','.join(index_names[i] for i in node.multiindex)
