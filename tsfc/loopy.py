"""Generate loopy Loop from ImperoC tuple data.

This is the final stage of code generation in TSFC."""

from __future__ import absolute_import, print_function, division

from collections import defaultdict
from functools import reduce
from math import isnan
import itertools

import numpy
from singledispatch import singledispatch

import coffee.base as coffee

from gem import gem, impero as imp

from tsfc.parameters import SCALAR_TYPE

import islpy as isl
import loopy as lp
import pymbolic.primitives as p

from pytools import UniqueNameGenerator

class LoopyContext(object):
    def __init__(self):
        self.domain = None
        # gem index -> [pymbolic variables]
        # use a stack to model the scope
        self.index_variables = defaultdict(list)
        self.index_extent = {}
        self.variable_to_pymbolic_and_shape = {}
        self.counter = itertools.count()
        self.name_gen = UniqueNameGenerator()

    def next_index_name(self):
        return "i_{0}".format(next(self.counter))

    def pymbolic_variable(self, var):
        try:
            pym, shape = self.variable_to_pymbolic_and_shape[var]
        except KeyError:
            pym = p.Variable(self.name_gen(var.name))
            self.variable_to_pymbolic_and_shape[var] = (pym, var.shape)
        return pym


def generate(impero_c, precision):
    """Generates COFFEE code.

    :arg impero_c: ImperoC tuple with Impero AST and other data
    :arg index_names: pre-assigned index names
    :arg precision: floating-point precision for printing
    :returns: loopy kernel
    """
    ctx = LoopyContext()
    ctx.precision = precision
    ctx.epsilon = 10.0 * eval("1e-%d" % precision)

    ctx.names = {}
    for i, temp in enumerate(impero_c.temporaries):
        ctx.names[temp] = "t%d" % i

    instructions = statement(impero_c.tree, ctx)

    data = ["..."]
    # data = [
    #     lp.TemporaryVariable(
    #         name, shape=lp.auto, initializer=val,
    #         scope=lp.temp_var_scope.GLOBAL,
    #         read_only=True)
    #     for name, val in six.itervalues(ctx.literal_to_name_and_array)] + ["..."]
    domain = None
    inames = isl.make_zero_and_vars(list(ctx.index_extent.keys()))
    for idx, extent in ctx.index_extent.items():
        axis = ((inames[0].le_set(inames[idx])) & (inames[idx].lt_set(inames[0] + extent)))
        if domain is None:
            domain = axis
        else:
            domain = domain & axis


    return lp.make_kernel([domain], instructions, data, name="test_loopy")


def _coffee_symbol(symbol, rank=()):
    """Build a coffee Symbol, concatenating rank.

    :arg symbol: Either a symbol name, or else an existing coffee Symbol.
    :arg rank: The ``rank`` argument to the coffee Symbol constructor.

    If symbol is a symbol, then the returned symbol has rank
    ``symbol.rank + rank``."""
    if isinstance(symbol, coffee.Symbol):
        rank = symbol.rank + rank
        symbol = symbol.symbol
    else:
        assert isinstance(symbol, str)
    return coffee.Symbol(symbol, rank=rank)


def _decl_symbol(expr, parameters):
    """Build a COFFEE Symbol for declaration."""
    multiindex = parameters.indices[expr]
    rank = tuple(index.extent for index in multiindex) + expr.shape
    return _coffee_symbol(parameters.names[expr], rank=rank)


def _ref_symbol(expr, parameters):
    """Build a COFFEE Symbol for referencing a value."""
    multiindex = parameters.indices[expr]
    rank = tuple(parameters.index_names[index] for index in multiindex)
    return _coffee_symbol(parameters.names[expr], rank=tuple(rank))


def _root_pragma(expr, parameters):
    """Decides whether to annonate the expression with
    #pragma coffee expression"""
    if expr in parameters.roots:
        return "#pragma coffee expression"
    else:
        return None


@singledispatch
def statement(tree, parameters):
    """Translates an Impero (sub)tree into a COFFEE AST corresponding
    to a C statement.

    :arg tree: Impero (sub)tree
    :arg parameters: miscellaneous code generation data
    :returns: COFFEE AST
    """
    raise AssertionError("cannot generate COFFEE from %s" % type(tree))


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
    ctx.index_variables[tree.index].append(p.Variable(idx))
    ctx.index_extent[idx] = extent

    statements = statement(tree.children[0], ctx)
    ctx.index_variables[tree.index].pop()
    return statements


@statement.register(imp.Initialise)
def statement_initialise(leaf, parameters):
    if parameters.declare[leaf]:
        return coffee.Decl(SCALAR_TYPE, _decl_symbol(leaf.indexsum, parameters), 0.0)
    else:
        return coffee.Assign(_ref_symbol(leaf.indexsum, parameters), 0.0)


@statement.register(imp.Accumulate)
def statement_accumulate(leaf, parameters):
    pragma = _root_pragma(leaf.indexsum, parameters)
    return coffee.Incr(_ref_symbol(leaf.indexsum, parameters),
                       expression(leaf.indexsum.children[0], parameters),
                       pragma=pragma)


@statement.register(imp.Return)
def statement_return(leaf, ctx):
    return [lp.Assignment(expression(leaf.variable, ctx), expression(leaf.expression, ctx))]


@statement.register(imp.ReturnAccumulate)
def statement_returnaccumulate(leaf, ctx):
    lhs = expression(leaf.variable, ctx)
    rhs = lhs + expression(leaf.indexsum.children[0], ctx)
    return [lp.Assignment(lhs, rhs)]


@statement.register(imp.Evaluate)
def statement_evaluate(leaf, parameters):
    expr = leaf.expression
    if isinstance(expr, gem.ListTensor):
        if parameters.declare[leaf]:
            array_expression = numpy.vectorize(lambda v: expression(v, parameters))
            return coffee.Decl(SCALAR_TYPE,
                               _decl_symbol(expr, parameters),
                               coffee.ArrayInit(array_expression(expr.array),
                                                precision=parameters.precision))
        else:
            ops = []
            for multiindex, value in numpy.ndenumerate(expr.array):
                coffee_sym = _coffee_symbol(_ref_symbol(expr, parameters), rank=multiindex)
                ops.append(coffee.Assign(coffee_sym, expression(value, parameters)))
            return coffee.Block(ops, open_scope=False)
    elif isinstance(expr, gem.Constant):
        assert parameters.declare[leaf]
        return coffee.Decl(SCALAR_TYPE,
                           _decl_symbol(expr, parameters),
                           coffee.ArrayInit(expr.array, parameters.precision),
                           qualifiers=["static", "const"])
    else:
        code = expression(expr, parameters, top=True)
        if parameters.declare[leaf]:
            return coffee.Decl(SCALAR_TYPE, _decl_symbol(expr, parameters), code)
        else:
            return coffee.Assign(_ref_symbol(expr, parameters), code)


def expression(expr, parameters, top=False):
    """Translates GEM expression into a COFFEE snippet, stopping at
    temporaries.

    :arg expr: GEM expression
    :arg parameters: miscellaneous code generation data
    :arg top: do not generate temporary reference for the root node
    :returns: COFFEE expression
    """
    if not top and expr in parameters.names:
        return _ref_symbol(expr, parameters)
    else:
        return _expression(expr, parameters)


@singledispatch
def _expression(expr, parameters):
    raise AssertionError("cannot generate COFFEE from %s" % type(expr))


@_expression.register(gem.Failure)
def _expression_failure(expr, parameters):
    raise expr.exception


@_expression.register(gem.Product)
def _expression_product(expr, ctx):
    return p.Product(tuple([expression(c, ctx) for c in expr.children]))


@_expression.register(gem.Sum)
def _expression_sum(expr, parameters):
    return coffee.Sum(*[expression(c, parameters)
                        for c in expr.children])


@_expression.register(gem.Division)
def _expression_division(expr, parameters):
    return coffee.Div(*[expression(c, parameters)
                        for c in expr.children])


@_expression.register(gem.Power)
def _expression_power(expr, parameters):
    base, exponent = expr.children
    return coffee.FunCall("pow", expression(base, parameters), expression(exponent, parameters))


@_expression.register(gem.MathFunction)
def _expression_mathfunction(expr, parameters):
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
        nu, arg = expr.children
        if nu == gem.Zero():
            return coffee.FunCall('j0', expression(arg, parameters))
        elif nu == gem.one:
            return coffee.FunCall('j1', expression(arg, parameters))
    if name == 'yn':
        nu, arg = expr.children
        if nu == gem.Zero():
            return coffee.FunCall('y0', expression(arg, parameters))
        elif nu == gem.one:
            return coffee.FunCall('y1', expression(arg, parameters))
    return coffee.FunCall(name, *[expression(c, parameters) for c in expr.children])


@_expression.register(gem.MinValue)
def _expression_minvalue(expr, parameters):
    return coffee.FunCall('fmin', *[expression(c, parameters) for c in expr.children])


@_expression.register(gem.MaxValue)
def _expression_maxvalue(expr, parameters):
    return coffee.FunCall('fmax', *[expression(c, parameters) for c in expr.children])


@_expression.register(gem.Comparison)
def _expression_comparison(expr, parameters):
    type_map = {">": coffee.Greater,
                ">=": coffee.GreaterEq,
                "==": coffee.Eq,
                "!=": coffee.NEq,
                "<": coffee.Less,
                "<=": coffee.LessEq}
    return type_map[expr.operator](*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.LogicalNot)
def _expression_logicalnot(expr, parameters):
    return coffee.Not(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.LogicalAnd)
def _expression_logicaland(expr, parameters):
    return coffee.And(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.LogicalOr)
def _expression_logicalor(expr, parameters):
    return coffee.Or(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.Conditional)
def _expression_conditional(expr, parameters):
    return coffee.Ternary(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.Constant)
def _expression_scalar(expr, parameters):
    assert not expr.shape
    if isnan(expr.value):
        return coffee.Symbol("NAN")
    else:
        v = expr.value
        r = round(v, 1)
        if r and abs(v - r) < parameters.epsilon:
            v = r  # round to nonzero
        return coffee.Symbol(("%%.%dg" % parameters.precision) % v)


@_expression.register(gem.Variable)
def _expression_variable(expr, ctx):
    return ctx.pymbolic_variable(expr)


@_expression.register(gem.Indexed)
def _expression_indexed(expr, ctx):
    rank = []
    for index in expr.multiindex:
        if isinstance(index, gem.Index):
            rank.append(ctx.index_variables[index][-1])
        elif isinstance(index, gem.VariableIndex):
            assert False
            rank.append(expression(index.expression, ctx).gencode())
        else:
            assert False  # what's here?
            rank.append(index)
    var = expression(expr.children[0], ctx)
    return p.Subscript(var, tuple(rank))


@_expression.register(gem.FlexiblyIndexed)
def _expression_flexiblyindexed(expr, parameters):
    var = expression(expr.children[0], parameters)
    assert isinstance(var, coffee.Symbol)
    assert not var.rank
    assert not var.offset

    rank = []
    offset = []
    for off, idxs in expr.dim2idxs:
        for index, stride in idxs:
            assert isinstance(index, gem.Index)

        if len(idxs) == 0:
            rank.append(off)
            offset.append((1, 0))
        elif len(idxs) == 1:
            (index, stride), = idxs
            rank.append(parameters.index_names[index])
            offset.append((stride, off))
        else:
            parts = []
            if off:
                parts += [coffee.Symbol(str(off))]
            for index, stride in idxs:
                index_sym = coffee.Symbol(parameters.index_names[index])
                assert stride
                if stride == 1:
                    parts += [index_sym]
                else:
                    parts += [coffee.Prod(index_sym, coffee.Symbol(str(stride)))]
            assert parts
            rank.append(reduce(coffee.Sum, parts))
            offset.append((1, 0))

    return coffee.Symbol(var.symbol, rank=tuple(rank), offset=tuple(offset))
