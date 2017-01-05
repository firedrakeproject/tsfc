from __future__ import division, absolute_import, print_function

from functools import partial
from singledispatch import singledispatch
import numpy as np
import six

import islpy as isl
import loopy as lp

import gem.gem as g
import pymbolic.primitives as p

from pytools import UniqueNameGenerator


# {{{ conversion context

class ConversionContext(object):
    def __init__(self, expr_use_count):
        self.expr_use_count = expr_use_count

        self.name_gen = UniqueNameGenerator()
        self.index_to_iname_and_length = {}
        self.var_to_name_and_shape = {}
        self.literal_to_name_and_array = {}
        self.node_to_var_name = {}
        self.assignments = []
        self.subst_rules = []

    def variable_to_name(self, node):
        # deals with both Variable and VariableIndex nodes
        try:
            name, shape = self.var_to_name_and_shape[node]
        except KeyError:
            name = self.name_gen(node.name)
            self.var_to_name_and_shape[node] = (name, node.shape)

        else:
            assert node.shape == shape

        return name

    def literal_to_name(self, literal):
        try:
            name, array = self.literal_to_name_and_array[literal]
        except KeyError:
            name = self.name_gen("cnst")
            self.literal_to_name_and_array[literal] = (name, literal.array)

        else:
            assert np.array_equal(array, literal.array)

        return name

    def index_to_iname(self, index):
        try:
            iname, length = self.index_to_iname_and_length[index]
        except KeyError:
            if index.name is None:
                iname = self.name_gen("i%d" % index.count)
            else:
                iname = self.name_gen(index.name)

            self.index_to_iname_and_length[index] = (iname, index.extent)

        else:
            assert index.extent == length

        return iname

    @staticmethod
    def _is_cse_eligible(node):
        return not (
            (isinstance(node, g.Literal) and node.array.shape == ()) or

            not isinstance(node, g.Indexed) or

            not isinstance(node, g.Variable))

    def rec_gem(self, node, parent):
        if (
                self.expr_use_count.get(node, 0) > 1 and
                self._is_cse_eligible(node)):
            try:
                var_name = self.node_to_var_name[node]
            except KeyError:
                result = expr_to_loopy(node, self)
                var_name = self.name_gen("cse")
                self.node_to_var_name[node] = var_name
                free_inames = tuple(
                    self.index_to_iname(i) for i in node.free_indices)
                self.assignments.append((var_name, free_inames, result))

            return p.Variable(var_name)

        else:
            return expr_to_loopy(node, self)

# }}}


# {{{ index conversion

@singledispatch
def index_to_loopy(node, ctx):
    raise NotImplementedError(
        "ran into index type '%s', no conversion known"
        % type(node).__name__)


@index_to_loopy.register(g.Index)
def map_index(node, ctx):
    return p.Variable(ctx.index_to_iname(node))


@index_to_loopy.register(g.VariableIndex)
def map_varindex(node, ctx):
    return ctx.rec_gem(node.expression, None)


@index_to_loopy.register(int)
def map_int(node, ctx):
    return node

# }}}


# {{{ expression conversion

@singledispatch
def expr_to_loopy(node, ctx):
    raise NotImplementedError(
        "ran into node type '%s', no conversion known"
        % type(node).__name__)


@expr_to_loopy.register(g.Identity)
def map_identity(node, ctx):
    # no clear mapping of vectorial quantity into loopy
    raise NotImplementedError(type(node).__name__)


@expr_to_loopy.register(g.Literal)
def map_literal(node, ctx):
    if node.array.shape == ():
        return node.array[()]
    else:
        return p.Variable(g.literal_to_name(node))


@expr_to_loopy.register(g.Zero)
def map_zero(node, ctx):
    # no clear mapping of vectorial quantity into loopy
    raise NotImplementedError(type(node).__name__)


@expr_to_loopy.register(g.Variable)
def map_variable(node, ctx):
    return p.Variable(ctx.variable_to_name(node))


def convert_multichild(pymbolic_cls, node, ctx):
    return pymbolic_cls(tuple(ctx.rec_gem(c, node) for c in node.children))


expr_to_loopy.register(g.Sum)(partial(convert_multichild, p.Sum))
expr_to_loopy.register(g.Product)(
    partial(convert_multichild, p.Product))


@expr_to_loopy.register(g.Division)
def _(node, ctx):
    num, denom = node.children
    return p.Quotient(ctx.rec_gem(num, node), ctx.rec_gem(denom, node))


@expr_to_loopy.register(g.Power)
def map_power(node, ctx):
    base, exponent = node.children
    return p.Power(ctx.rec_gem(base, node), ctx.rec_gem(exponent, node))


@expr_to_loopy.register(g.MathFunction)
def map_function(node, ctx):
    return p.Variable(node.name)(
        *tuple(ctx.rec_gem(c, node) for c in node.children))


expr_to_loopy.register(g.MinValue)(partial(convert_multichild, p.Min))
expr_to_loopy.register(g.MaxValue)(partial(convert_multichild, p.Max))


@expr_to_loopy.register(g.Comparison)
def map_comparison(node, ctx):
    left, right = node.children
    return p.Comparison(
        ctx.rec_gem(left, node),
        node.operator,
        ctx.rec_gem(right, node))


def index_aggregate_to_name(c, ctx):
    if isinstance(c, g.Variable):
        return ctx.variable_to_name(c)

    elif isinstance(c, g.Literal):
        return ctx.literal_to_name(c)

    else:
        raise NotImplementedError(
            "indexing into %s" % type(c).__name__)


@expr_to_loopy.register(g.Indexed)
def map_indexed(node, ctx):
    c, = node.children

    return p.Subscript(
        p.Variable(index_aggregate_to_name(c, ctx)),
        tuple(index_to_loopy(i, ctx) for i in node.multiindex))


def cumulative_strides(strides):
    """Calculate cumulative strides from per-dimension capacities.

    For example:

        [2, 3, 4] ==> [12, 4, 1]

    """
    temp = np.flipud(np.cumprod(np.flipud(list(strides)[1:])))
    return tuple(temp) + (1,)


@expr_to_loopy.register(g.FlexiblyIndexed)
def map_flexibly_indexed(node, ctx):
    c, = node.children

    def flex_idx_to_loopy(f):
        off, idxs = f

        result = off
        for i, s in idxs:
            result += index_to_loopy(i, ctx)*s

        return result

    return p.Subscript(
        p.Variable(index_aggregate_to_name(c, ctx)),
        tuple(flex_idx_to_loopy(i) for i in node.dim2idxs))


@expr_to_loopy.register(g.IndexSum)
def map_index_sum(node, ctx):
    c, = node.children

    subexpr = ctx.rec_gem(c, None)

    name = ctx.name_gen("sum_tmp")
    arg_names = tuple(
        ctx.index_to_iname(fi)
        for fi in node.free_indices)

    # new_arg_names = tuple(ctx.name_gen(an) for an in arg_names)

    # from pymbolic import substitute
    # subexpr = substitute(
    #     subexpr,
    #     dict(
    #         (an, p.Variable(nan))
    #         for an, nan in zip(arg_names, new_arg_names)))

    ctx.subst_rules.append(
        lp.SubstitutionRule(
            name,
            arg_names,
            lp.Reduction(
                "sum",
                (ctx.index_to_iname(node.index),),
                subexpr)))

    return p.Variable(name)(*tuple(p.Variable(n) for n in arg_names))

# }}}


def count_subexpression_uses(node, expr_use_count):
    expr_use_count[node] = expr_use_count.get(node, 0) + 1
    for c in node.children:
        count_subexpression_uses(c, expr_use_count)


# {{{ main entrypoint

def tsfc_to_loopy(ir, output_names="A", kernel_name="tsfc_kernel"):

    if isinstance(output_names, str):
        output_names = tuple(
            output_names + str(i)
            for i in range(len(ir)))
    elif not isinstance(output_names, tuple):
        raise TypeError("output_names must be a string or a tuple")

    expr_use_count = {}
    for expr in ir:
        count_subexpression_uses(expr, expr_use_count)

    ctx = ConversionContext(expr_use_count)

    exprs_and_free_inames = [
        (ctx.rec_gem(node, None),
            tuple(ctx.index_to_iname(i) for i in node.free_indices))
        for node in ir]

    def subscr(name, indices):
        return (
            p.Variable(name)[
                tuple(p.Variable(i) for i in indices)]
            if indices else
            p.Variable(name))

    instructions = (
        [
            lp.Assignment(
                subscr(var_name, free_indices),
                rhs,
                forced_iname_deps=frozenset(free_indices),
                forced_iname_deps_is_final=True)
            for var_name, free_indices, rhs in ctx.assignments] +
        [
            lp.Assignment(
                subscr(var_name, free_indices),
                subscr(var_name, free_indices) + rhs,
                forced_iname_deps=frozenset(free_indices),
                forced_iname_deps_is_final=True)
            for var_name, (rhs, free_indices) in zip(
                output_names, exprs_and_free_inames)])

    inames = isl.make_zero_and_vars([
        iname
        for iname, length in six.itervalues(ctx.index_to_iname_and_length)])

    domain = None
    for iname, length in six.itervalues(ctx.index_to_iname_and_length):
        axis = (
            (inames[0].le_set(inames[iname])) &
            (inames[iname].lt_set(inames[0] + length)))

        if domain is None:
            domain = axis
        else:
            domain = domain & axis

    domain = domain.get_basic_sets()[0]

    data = [
        lp.TemporaryVariable(
            name, shape=lp.auto, initializer=val,
            scope=lp.temp_var_scope.GLOBAL,
            read_only=True)
        for name, val in six.itervalues(ctx.literal_to_name_and_array)] + ["..."]

    knl = lp.make_kernel(
        [domain],
        instructions + ctx.subst_rules,
        data,
        name=kernel_name)

    if 1:  # Add an option to turn this off later if we want.
        print("hithere")
        A0writes = [ins for ins in knl.instructions
                    if ins.assignee.aggregate.name == "A0"]
        assert len(A0writes) == 1

        # turn x = x + y into x = y
        insn = A0writes[0]
        rvalue = insn.expression
        newrvalue = rvalue.children[1]
        insn.expression = newrvalue

    return knl

# }}}

# vim: foldmethod=marker
