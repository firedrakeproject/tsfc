"""Generate loopy Loop from ImperoC tuple data.

This is the final stage of code generation in TSFC."""

from __future__ import absolute_import, print_function, division

from math import isnan
import itertools

import numpy
from singledispatch import singledispatch

from gem import gem, impero as imp

import islpy as isl
import loopy as lp

import pymbolic.primitives as p

from pytools import UniqueNameGenerator

lp.CACHING_ENABLED = False


class LoopyContext(object):
    def __init__(self):
        self.domain = None
        self.active_indices = {}  # gem index -> pymbolic variable
        self.index_extent = {}  # pymbolic variable for indices - > extent
        self.gem_to_pymbolic = {}  # gem node -> pymbolic variable
        self.counter = itertools.count()
        self.name_gen = UniqueNameGenerator()

    def next_index_name(self):
        return "i_{0}".format(next(self.counter))

    def pym_multiindex(self, multiindex):
        rank = []
        for index in multiindex:
            if isinstance(index, gem.Index):
                rank.append(self.active_indices[index])
            elif isinstance(index, gem.VariableIndex):
                rank.append(expression(index.expression, self))
            else:
                assert isinstance(index, int)
                rank.append(index)
        return tuple(rank)

    def pymbolic_variable(self, node):
        try:
            pym = self.gem_to_pymbolic[node]
        except KeyError:
            name = self.name_gen(node.name)
            pym = p.Variable(name)
            self.gem_to_pymbolic[node] = pym
        if node in self.indices:
            rank = self.pym_multiindex(self.indices[node])
            if rank:
                return p.Subscript(pym, rank)
            else:
                return pym
        else:
            return pym

    def active_inames(self):
        # Return all active indices
        return frozenset([i.name for i in self.active_indices.values()])


def generate(impero_c, args, precision, kernel_name="loopy_kernel"):
    """Generates loopy code.

    :arg impero_c: ImperoC tuple with Impero AST and other data
    :arg args: list of loopy.GlobalArgs
    :arg index_names: pre-assigned index names
    :arg precision: floating-point precision for printing
    :returns: loopy kernel
    """
    ctx = LoopyContext()
    ctx.indices = impero_c.indices
    ctx.precision = precision
    ctx.epsilon = 10.0 ** (-precision)

    data = list(args)
    for i, temp in enumerate(impero_c.temporaries):
        name = "t%d" % i
        if isinstance(temp, gem.Constant):
            data.append(lp.TemporaryVariable(name, shape=temp.shape, dtype=temp.array.dtype, initializer=temp.array, scope=lp.temp_var_scope.LOCAL, read_only=True))
        else:
            shape = tuple([i.extent for i in ctx.indices[temp]]) + temp.shape
            data.append(lp.TemporaryVariable(name, shape=shape, dtype=numpy.float64, initializer=None, scope=lp.temp_var_scope.LOCAL, read_only=False))
        ctx.gem_to_pymbolic[temp] = p.Variable(name)

    instructions = statement(impero_c.tree, ctx)

    domain = None
    inames = isl.make_zero_and_vars(list(ctx.index_extent.keys()))
    for idx, extent in ctx.index_extent.items():
        axis = ((inames[0].le_set(inames[idx])) & (inames[idx].lt_set(inames[0] + extent)))
        if not domain:
            domain = axis
        else:
            domain = domain & axis
    if not domain:
        domain = isl.BasicSet("[] -> {[]}")

    knl = lp.make_kernel([domain], instructions, data, name=kernel_name, target=lp.CTarget(), seq_dependencies=True)

    def f_mangler(target, name, arg_dtypes):
        if name == "fmin":
            return lp.CallMangleInfo("fmin", (lp.types.to_loopy_type(numpy.float64), ), arg_dtypes)
        return None

    knl = lp.register_function_manglers(knl, [f_mangler])

    def s_mangler(target, name):
        if name == "NAN":
            return lp.types.to_loopy_type(numpy.float64), name
        return None

    knl = lp.register_symbol_manglers(knl, [s_mangler])

    print(knl)
    # iname_tag = dict((i, 'ord') for i in knl.all_inames())
    # knl = lp.tag_inames(knl, iname_tag)
    return knl


@singledispatch
def statement(tree, ctx):
    """Translates an Impero (sub)tree into a loopy instructions corresponding
    to a C statement.

    :arg tree: Impero (sub)tree
    :arg ctx: miscellaneous code generation data
    :returns: list of loopy instructions
    """
    raise AssertionError("cannot generate loopy from %s" % type(tree))


@statement.register(imp.Block)
def statement_block(tree, ctx):
    statements = []
    for child in tree.children:
        statements.extend(statement(child, ctx))
    return statements


@statement.register(imp.For)
def statement_for(tree, ctx):
    extent = tree.index.extent
    assert extent
    idx = ctx.next_index_name()
    ctx.active_indices[tree.index] = p.Variable(idx)
    ctx.index_extent[idx] = extent

    statements = statement(tree.children[0], ctx)

    ctx.active_indices.pop(tree.index)
    return statements


@statement.register(imp.Initialise)
def statement_initialise(leaf, ctx):
    return [lp.Assignment(expression(leaf.indexsum, ctx), 0.0, within_inames=ctx.active_inames())]


@statement.register(imp.Accumulate)
def statement_accumulate(leaf, ctx):
    lhs = expression(leaf.indexsum, ctx)
    rhs = lhs + expression(leaf.indexsum.children[0], ctx)
    return [lp.Assignment(lhs, rhs, within_inames=ctx.active_inames())]


@statement.register(imp.Return)
def statement_return(leaf, ctx):
    lhs = expression(leaf.variable, ctx)
    rhs = lhs + expression(leaf.expression, ctx)
    return [lp.Assignment(lhs, rhs, within_inames=ctx.active_inames())]


@statement.register(imp.ReturnAccumulate)
def statement_returnaccumulate(leaf, ctx):
    lhs = expression(leaf.variable, ctx)
    rhs = lhs + expression(leaf.indexsum.children[0], ctx)
    return [lp.Assignment(lhs, rhs, within_inames=ctx.active_inames())]


@statement.register(imp.Evaluate)
def statement_evaluate(leaf, ctx):
    expr = leaf.expression
    if isinstance(expr, gem.ListTensor):
        ops = []
        var = ctx.pymbolic_variable(expr)
        index = ()
        if isinstance(var, p.Subscript):
            # Probably can do this better
            var, index = var.aggregate, var.index_tuple
        for multiindex, value in numpy.ndenumerate(expr.array):
            ops.append(lp.Assignment(p.Subscript(var, index + multiindex), expression(value, ctx), within_inames=ctx.active_inames()))
        return ops
    elif isinstance(expr, gem.Constant):
        return []
    else:
        return [lp.Assignment(ctx.pymbolic_variable(expr), expression(expr, ctx, top=True), within_inames=ctx.active_inames())]


def expression(expr, ctx, top=False):
    """Translates GEM expression into a pymbolic expression

    :arg expr: GEM expression
    :arg ctx: miscellaneous code generation data
    :arg top: do not generate temporary reference for the root node
    :returns: pymbolic expression
    """
    # TODO: fix top
    if not top and expr in ctx.gem_to_pymbolic:
        return ctx.pymbolic_variable(expr)
    else:
        return _expression(expr, ctx)


@singledispatch
def _expression(expr, parameters):
    raise AssertionError("cannot generate expression from %s" % type(expr))


@_expression.register(gem.Failure)
def _expression_failure(expr, parameters):
    raise expr.exception


@_expression.register(gem.Product)
def _expression_product(expr, ctx):
    return p.Product(tuple([expression(c, ctx) for c in expr.children]))


@_expression.register(gem.Sum)
def _expression_sum(expr, ctx):
    return p.Sum(tuple([expression(c, ctx) for c in expr.children]))


@_expression.register(gem.Division)
def _expression_division(expr, ctx):
    return p.Quotient(*[expression(c, ctx) for c in expr.children])


@_expression.register(gem.Power)
def _expression_power(expr, ctx):
    base, exponent = expr.children
    return p.Power(expression(base, ctx), expression(exponent, ctx))


@_expression.register(gem.MathFunction)
def _expression_mathfunction(expr, ctx):
    name_map = {
        'abs': 'fabs',
        'ln': 'log',

        # Bessel functions
        'cyl_bessel_j': 'jn',
        'cyl_bessel_y': 'yn',

        # Modified Bessel functions (C++ only)
        #
        # These mappings work for FEniCS only, and fail with Firedrake
        # since no Boost available.
        'cyl_bessel_i': 'boost::math::cyl_bessel_i',
        'cyl_bessel_k': 'boost::math::cyl_bessel_k',
    }
    name = name_map.get(expr.name, expr.name)
    if name == 'jn':
        assert False
        # nu, arg = expr.children
        # if nu == gem.Zero():
        #     return coffee.FunCall('j0', expression(arg, parameters))
        # elif nu == gem.one:
        #     return coffee.FunCall('j1', expression(arg, parameters))
    if name == 'yn':
        assert False
        # nu, arg = expr.children
        # if nu == gem.Zero():
        #     return coffee.FunCall('y0', expression(arg, parameters))
        # elif nu == gem.one:
        #     return coffee.FunCall('y1', expression(arg, parameters))
    return p.Variable(name)(*[expression(c, ctx) for c in expr.children])


@_expression.register(gem.MinValue)
def _expression_minvalue(expr, ctx):
    # loopy will translate p.Min to min() rather than fmin()
    return p.Variable("fmin")(*[expression(c, ctx) for c in expr.children])


@_expression.register(gem.MaxValue)
def _expression_maxvalue(expr, ctx):
    return p.Max(tuple([expression(c, ctx) for c in expr.children]))


@_expression.register(gem.Comparison)
def _expression_comparison(expr, ctx):
    left, right = [expression(c, ctx) for c in expr.children]
    return p.Comparison(left, expr.operator, right)


@_expression.register(gem.LogicalNot)
def _expression_logicalnot(expr, ctx):
    return p.LogicalNot(tuple([expression(c, ctx) for c in expr.children]))


@_expression.register(gem.LogicalAnd)
def _expression_logicaland(expr, ctx):
    return p.LogicalAnd(tuple([expression(c, ctx) for c in expr.children]))


@_expression.register(gem.LogicalOr)
def _expression_logicalor(expr, ctx):
    return p.LogicalOr(tuple([expression(c, ctx) for c in expr.children]))


@_expression.register(gem.Conditional)
def _expression_conditional(expr, ctx):
    return p.If(*[expression(c, ctx) for c in expr.children])


@_expression.register(gem.Constant)
def _expression_scalar(expr, parameters):
    assert not expr.shape
    if isnan(expr.value):
        return p.Variable("NAN")  # TODO: is this right?
    else:
        v = expr.value
        r = round(v, 1)
        if r and abs(v - r) < parameters.epsilon:
            v = r  # round to nonzero
        return v


@_expression.register(gem.Variable)
def _expression_variable(expr, ctx):
    return ctx.pymbolic_variable(expr)


@_expression.register(gem.Indexed)
def _expression_indexed(expr, ctx):
    rank = ctx.pym_multiindex(expr.multiindex)
    var = expression(expr.children[0], ctx)
    if isinstance(var, p.Subscript):
        rank += var.index
        var = var.aggregate
    return p.Subscript(var, rank)


@_expression.register(gem.FlexiblyIndexed)
def _expression_flexiblyindexed(expr, ctx):
    var = expression(expr.children[0], ctx)

    rank = []
    for off, idxs in expr.dim2idxs:
        for index, stride in idxs:
            assert isinstance(index, gem.Index)

        rank_ = [off]
        for index, stride in idxs:
            rank_.append(p.Product((ctx.active_indices[index], stride)))
        rank.append(p.Sum(tuple(rank_)))

    return p.Subscript(var, tuple(rank))
