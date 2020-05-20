"""Generate COFFEE AST from ImperoC tuple data.

This is the final stage of code generation in TSFC."""

from collections import defaultdict
from cmath import isnan
from functools import singledispatch, reduce
import itertools

import numpy

import coffee.base as coffee

from gem import gem, impero as imp

from tsfc.parameters import is_complex


class Bunch(object):
    pass


def generate(impero_c, index_names, scalar_type):
    """Generates COFFEE code.

    :arg impero_c: ImperoC tuple with Impero AST and other data
    :arg index_names: pre-assigned index names
    :arg scalar_type: type of scalars as C typename string
    :returns: COFFEE function body
    """
    params = Bunch()
    params.declare = impero_c.declare
    params.indices = impero_c.indices
    finfo = numpy.finfo(scalar_type)
    params.precision = finfo.precision
    params.epsilon = finfo.resolution
    params.scalar_type = scalar_type

    params.names = {}
    for i, temp in enumerate(impero_c.temporaries):
        params.names[temp] = "t%d" % i

    counter = itertools.count()
    params.index_names = defaultdict(lambda: "i_%d" % next(counter))
    params.index_names.update(index_names)

    return statement(impero_c.tree, params)


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
def statement_block(tree, parameters):
    statements = [statement(child, parameters) for child in tree.children]
    declares = []
    for expr in parameters.declare[tree]:
        declares.append(coffee.Decl(parameters.scalar_type, _decl_symbol(expr, parameters)))
    return coffee.Block(declares + statements, open_scope=True)


@statement.register(imp.For)
def statement_for(tree, parameters):
    extent = tree.index.extent
    assert extent
    i = _coffee_symbol(parameters.index_names[tree.index])
    # TODO: symbolic constant for "int"
    return coffee.For(coffee.Decl("int", i, init=0),
                      coffee.Less(i, extent),
                      coffee.Incr(i, 1),
                      statement(tree.children[0], parameters))


@statement.register(imp.Initialise)
def statement_initialise(leaf, parameters):
    if parameters.declare[leaf]:
        return coffee.Decl(parameters.scalar_type, _decl_symbol(leaf.indexsum, parameters), 0.0)
    else:
        return coffee.Assign(_ref_symbol(leaf.indexsum, parameters), 0.0)


@statement.register(imp.Accumulate)
def statement_accumulate(leaf, parameters):
    return coffee.Incr(_ref_symbol(leaf.indexsum, parameters),
                       expression(leaf.indexsum.children[0], parameters))


@statement.register(imp.Return)
def statement_return(leaf, parameters):
    return coffee.Incr(expression(leaf.variable, parameters),
                       expression(leaf.expression, parameters))


@statement.register(imp.ReturnAccumulate)
def statement_returnaccumulate(leaf, parameters):
    return coffee.Incr(expression(leaf.variable, parameters),
                       expression(leaf.indexsum.children[0], parameters))


@statement.register(imp.Evaluate)
def statement_evaluate(leaf, parameters):
    expr = leaf.expression
    if isinstance(expr, gem.ListTensor):
        if parameters.declare[leaf]:
            array_expression = numpy.vectorize(lambda v: expression(v, parameters))
            return coffee.Decl(parameters.scalar_type,
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
        return coffee.Decl(parameters.scalar_type,
                           _decl_symbol(expr, parameters),
                           coffee.ArrayInit(expr.array, parameters.precision),
                           qualifiers=["static", "const"])
    else:
        code = expression(expr, parameters, top=True)
        if parameters.declare[leaf]:
            return coffee.Decl(parameters.scalar_type, _decl_symbol(expr, parameters), code)
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
def _expression_product(expr, parameters):
    return coffee.Prod(*[expression(c, parameters)
                         for c in expr.children])


@_expression.register(gem.Sum)
def _expression_sum(expr, parameters):
    return coffee.Sum(*[expression(c, parameters)
                        for c in expr.children])


@_expression.register(gem.Division)
def _expression_division(expr, parameters):
    return coffee.Div(*[expression(c, parameters)
                        for c in expr.children])


# Table of handled math functions in real and complex modes
# Copied from FFCX (ffc/language/ufl_to_cnodes.py)
math_table = {
    'sqrt': ('sqrt', 'csqrt'),
    'abs': ('fabs', 'cabs'),
    'cos': ('cos', 'ccos'),
    'sin': ('sin', 'csin'),
    'tan': ('tan', 'ctan'),
    'acos': ('acos', 'cacos'),
    'asin': ('asin', 'casin'),
    'atan': ('atan', 'catan'),
    'cosh': ('cosh', 'ccosh'),
    'sinh': ('sinh', 'csinh'),
    'tanh': ('tanh', 'ctanh'),
    'acosh': ('acosh', 'cacosh'),
    'asinh': ('asinh', 'casinh'),
    'atanh': ('atanh', 'catanh'),
    'power': ('pow', 'cpow'),
    'exp': ('exp', 'cexp'),
    'ln': ('log', 'clog'),
    'real': (None, 'creal'),
    'imag': (None, 'cimag'),
    'conj': (None, 'conj'),
    'erf': ('erf', None),
    'atan_2': ('atan2', None),
    'atan2': ('atan2', None),
    'min_value': ('fmin', None),
    'max_value': ('fmax', None)
}


@_expression.register(gem.Power)
def _expression_power(expr, parameters):
    base, exponent = expr.children
    complex_mode = int(is_complex(parameters.scalar_type))
    return coffee.FunCall(math_table['power'][complex_mode],
                          expression(base, parameters),
                          expression(exponent, parameters))


@_expression.register(gem.MathFunction)
def _expression_mathfunction(expr, parameters):
    complex_mode = int(is_complex(parameters.scalar_type))

    # Bessel functions
    if expr.name.startswith('cyl_bessel_'):
        if complex_mode:
            msg = "Bessel functions for complex numbers: missing implementation"
            raise NotImplementedError(msg)
        nu, arg = expr.children
        nu_thunk = lambda: expression(nu, parameters)
        arg_coffee = expression(arg, parameters)
        if expr.name == 'cyl_bessel_j':
            if nu == gem.Zero():
                return coffee.FunCall('j0', arg_coffee)
            elif nu == gem.one:
                return coffee.FunCall('j1', arg_coffee)
            else:
                return coffee.FunCall('jn', nu_thunk(), arg_coffee)
        if expr.name == 'cyl_bessel_y':
            if nu == gem.Zero():
                return coffee.FunCall('y0', arg_coffee)
            elif nu == gem.one:
                return coffee.FunCall('y1', arg_coffee)
            else:
                return coffee.FunCall('yn', nu_thunk(), arg_coffee)

        # Modified Bessel functions (C++ only)
        #
        # These mappings work for FEniCS only, and fail with Firedrake
        # since no Boost available.
        if expr.name in ['cyl_bessel_i', 'cyl_bessel_k']:
            name = 'boost::math::' + expr.name
            return coffee.FunCall(name, nu_thunk(), arg_coffee)

        assert False, "Unknown Bessel function: {}".format(expr.name)

    # Other math functions
    name = math_table[expr.name][complex_mode]
    if name is None:
        raise RuntimeError("{} not supported in complex mode".format(expr.name))
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
        vr = expr.value.real
        rr = round(vr, 1)
        if rr and abs(vr - rr) < parameters.epsilon:
            vr = rr  # round to nonzero

        vi = expr.value.imag  # also checks if v is purely real
        if vi == 0.0:
            return coffee.Symbol(("%%.%dg" % parameters.precision) % vr)
        ri = round(vi, 1)

        if ri and abs(vi - ri) < parameters.epsilon:
            vi = ri
        return coffee.Symbol("({real:.{prec}g} + {imag:.{prec}g} * I)".format(
            real=vr, imag=vi, prec=parameters.precision))


@_expression.register(gem.Variable)
def _expression_variable(expr, parameters):
    return _coffee_symbol(expr.name)


@_expression.register(gem.Indexed)
def _expression_indexed(expr, parameters):
    rank = []
    for index in expr.multiindex:
        if isinstance(index, gem.Index):
            rank.append(parameters.index_names[index])
        elif isinstance(index, gem.VariableIndex):
            rank.append(expression(index.expression, parameters).gencode())
        else:
            rank.append(index)
    return _coffee_symbol(expression(expr.children[0], parameters),
                          rank=tuple(rank))


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
