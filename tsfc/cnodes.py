"""Generate UFLACS CNode AST from ImperoC tuple data.

This is the final stage of code generation in TSFC."""

from __future__ import absolute_import, print_function, division

from collections import defaultdict
from math import isnan
import itertools

import numpy
from singledispatch import singledispatch

import ffc.uflacs.language.cnodes as cnodes

from gem import gem, impero as imp

from tsfc.parameters import SCALAR_TYPE


class Bunch(object):
    pass


def generate(impero_c, index_names):
    """Generates CNode AST.

    :arg impero_c: ImperoC tuple with Impero AST and other data
    :arg index_names: pre-assigned index names
    :returns: CNode function body
    """
    params = Bunch()
    params.declare = impero_c.declare
    params.indices = impero_c.indices

    params.names = {}
    for i, temp in enumerate(impero_c.temporaries):
        params.names[temp] = "t%d" % i

    counter = itertools.count()
    params.index_names = defaultdict(lambda: "i_%d" % next(counter))
    params.index_names.update(index_names)

    return statement(impero_c.tree, params)


def _ref_symbol(expr, parameters):
    """Build a CNode expression for referencing a value."""
    symbol = cnodes.Symbol(parameters.names[expr])
    indices = tuple(parameters.index_names[index]
                    for index in parameters.indices[expr])
    if indices:
        return cnodes.ArrayAccess(symbol, indices)
    else:
        return symbol


@singledispatch
def statement(tree, parameters):
    """Translates an Impero (sub)tree into a CNode AST corresponding
    to a C statement.

    :arg tree: Impero (sub)tree
    :arg parameters: miscellaneous code generation data
    :returns: CNode AST
    """
    raise AssertionError("cannot generate CNode from %s" % type(tree))


@statement.register(imp.Block)
def statement_block(tree, parameters):
    statements = [statement(child, parameters) for child in tree.children]
    declares = []
    for expr in parameters.declare[tree]:
        multiindex = parameters.indices[expr]
        sizes = tuple(index.extent for index in multiindex) + expr.shape
        name = parameters.names[expr]
        declares.append(cnodes.ArrayDecl(SCALAR_TYPE, name, sizes))
    return cnodes.StatementList(declares + statements)


@statement.register(imp.For)
def statement_for(tree, parameters):
    extent = tree.index.extent
    assert extent
    i = parameters.index_names[tree.index]
    return cnodes.ForRange(i, 0, extent, statement(tree.children[0], parameters))


@statement.register(imp.Initialise)
def statement_initialise(leaf, parameters):
    if parameters.declare[leaf]:
        return cnodes.VariableDecl(SCALAR_TYPE,
                                   parameters.names[leaf.indexsum],
                                   cnodes.LiteralFloat(0))
    else:
        return cnodes.Assign(_ref_symbol(leaf.indexsum, parameters),
                             cnodes.LiteralFloat(0))


@statement.register(imp.Accumulate)
def statement_accumulate(leaf, parameters):
    return cnodes.AssignAdd(
        _ref_symbol(leaf.indexsum, parameters),
        expression(leaf.indexsum.children[0], parameters)
    )


@statement.register(imp.Return)
def statement_return(leaf, parameters):
    return cnodes.AssignAdd(
        expression(leaf.variable, parameters),
        expression(leaf.expression, parameters)
    )


@statement.register(imp.ReturnAccumulate)
def statement_returnaccumulate(leaf, parameters):
    return cnodes.AssignAdd(
        expression(leaf.variable, parameters),
        expression(leaf.indexsum.children[0], parameters)
    )


@statement.register(imp.Evaluate)
def statement_evaluate(leaf, parameters):
    expr = leaf.expression
    if isinstance(expr, gem.ListTensor):
        if parameters.declare[leaf]:
            array_expression = numpy.vectorize(lambda v: expression(v, parameters))
            name = parameters.names[expr]
            sizes = tuple(index.extent for index in parameters.indices[expr]) + expr.shape
            return cnodes.ArrayDecl(SCALAR_TYPE, name, sizes, array_expression(expr.array))
        else:
            ops = []
            for multiindex, value in numpy.ndenumerate(expr.array):
                temp_ref = _ref_symbol(expr, parameters)
                if isinstance(temp_ref, cnodes.ArrayAccess):
                    elem_ref = cnodes.ArrayAccess(temp_ref.array, temp_ref.indices + multiindex)
                else:
                    elem_ref = cnodes.ArrayAccess(temp_ref, multiindex)
                ops.append(cnodes.Assign(elem_ref, expression(value, parameters)))
            return cnodes.StatementList(ops)
    elif isinstance(expr, gem.Constant):
        assert parameters.declare[leaf]
        name = parameters.names[expr]
        sizes = tuple(index.extent for index in parameters.indices[expr]) + expr.shape
        # qualifiers=["static", "const"])
        return cnodes.ArrayDecl(SCALAR_TYPE, name, sizes, expr.array)
    else:
        code = expression(expr, parameters, top=True)
        if parameters.declare[leaf]:
            name = parameters.names[expr]
            return cnodes.VariableDecl(SCALAR_TYPE, name, code)
        else:
            return cnodes.Assign(_ref_symbol(expr, parameters), code)


def expression(expr, parameters, top=False):
    """Translates GEM expression into a CNode expression, stopping at
    temporaries.

    :arg expr: GEM expression
    :arg parameters: miscellaneous code generation data
    :arg top: do not generate temporary reference for the root node
    :returns: CNode expression
    """
    if not top and expr in parameters.names:
        return _ref_symbol(expr, parameters)
    else:
        return _expression(expr, parameters)


@singledispatch
def _expression(expr, parameters):
    raise AssertionError("cannot generate CNode from %s" % type(expr))


@_expression.register(gem.Failure)
def _expression_failure(expr, parameters):
    raise expr.exception


@_expression.register(gem.Product)
def _expression_product(expr, parameters):
    return cnodes.Product([expression(c, parameters)
                           for c in expr.children])


@_expression.register(gem.Sum)
def _expression_sum(expr, parameters):
    return cnodes.Sum([expression(c, parameters)
                       for c in expr.children])


@_expression.register(gem.Division)
def _expression_division(expr, parameters):
    return cnodes.Div(*[expression(c, parameters)
                        for c in expr.children])


@_expression.register(gem.Power)
def _expression_power(expr, parameters):
    base, exponent = expr.children
    return cnodes.Call("pow", [expression(base, parameters), expression(exponent, parameters)])


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
            return cnodes.Call('j0', [expression(arg, parameters)])
        elif nu == gem.one:
            return cnodes.Call('j1', [expression(arg, parameters)])
    if name == 'yn':
        nu, arg = expr.children
        if nu == gem.Zero():
            return cnodes.Call('y0', [expression(arg, parameters)])
        elif nu == gem.one:
            return cnodes.Call('y1', [expression(arg, parameters)])
    return cnodes.Call(name, [expression(c, parameters) for c in expr.children])


@_expression.register(gem.MinValue)
def _expression_minvalue(expr, parameters):
    return cnodes.Call('fmin', [expression(c, parameters) for c in expr.children])


@_expression.register(gem.MaxValue)
def _expression_maxvalue(expr, parameters):
    return cnodes.Call('fmax', [expression(c, parameters) for c in expr.children])


@_expression.register(gem.Comparison)
def _expression_comparison(expr, parameters):
    type_map = {">": cnodes.GT,
                ">=": cnodes.GE,
                "==": cnodes.EQ,
                "!=": cnodes.NE,
                "<": cnodes.LT,
                "<=": cnodes.LE}
    return type_map[expr.operator](*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.LogicalNot)
def _expression_logicalnot(expr, parameters):
    return cnodes.Not(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.LogicalAnd)
def _expression_logicaland(expr, parameters):
    return cnodes.And(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.LogicalOr)
def _expression_logicalor(expr, parameters):
    return cnodes.Or(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.Conditional)
def _expression_conditional(expr, parameters):
    return cnodes.Conditional(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.Constant)
def _expression_scalar(expr, parameters):
    assert not expr.shape
    if isnan(expr.value):
        return cnodes.Symbol("NAN")
    else:
        return cnodes.LiteralFloat(expr.value)


@_expression.register(gem.Variable)
def _expression_variable(expr, parameters):
    return cnodes.Symbol(expr.name)


@_expression.register(gem.Indexed)
def _expression_indexed(expr, parameters):
    indices = []
    for index in expr.multiindex:
        if isinstance(index, gem.Index):
            indices.append(parameters.index_names[index])
        elif isinstance(index, gem.VariableIndex):
            indices.append(expression(index.expression, parameters))
        else:
            indices.append(index)
    temp_ref = expression(expr.children[0], parameters)
    if isinstance(temp_ref, cnodes.ArrayAccess):
        return cnodes.ArrayAccess(temp_ref.array, temp_ref.indices + tuple(indices))
    else:
        return cnodes.ArrayAccess(temp_ref, indices)


@_expression.register(gem.FlexiblyIndexed)
def _expression_flexiblyindexed(expr, parameters):
    var = expression(expr.children[0], parameters)
    assert isinstance(var, cnodes.Symbol)

    indices = []
    for off, idxs in expr.dim2idxs:
        for index, stride in idxs:
            assert isinstance(index, gem.Index)

        if len(idxs) == 0:
            indices.append(cnodes.LiteralInt(off))
        else:
            parts = []
            if off:
                parts += [cnodes.LiteralInt(off)]
            for index, stride in idxs:
                index_sym = cnodes.Symbol(parameters.index_names[index])
                assert stride
                if stride == 1:
                    parts += [index_sym]
                else:
                    parts += [index_sym * stride]
            assert parts
            indices.append(cnodes.Sum(parts))

    return cnodes.ArrayAccess(var, indices)
