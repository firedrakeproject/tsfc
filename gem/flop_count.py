"""
This file contains all the necessary functions to accurately count the
total number of floating point operations for a given script.
"""

import gem.gem as gem
import gem.impero as imp
from functools import singledispatch
import numpy
import math


@singledispatch
def statement(tree, parameters):
    raise NotImplementedError


@statement.register(imp.Block)
def statement_block(tree, parameters):
    flops = sum(statement(child, parameters) for child in tree.children)
    return flops


@statement.register(imp.For)
def statement_for(tree, parameters):
    extent = tree.index.extent
    assert extent is not None
    child, = tree.children
    flops = statement(child, parameters)
    return flops * extent


@statement.register(imp.Initialise)
def statement_initialise(tree, parameters):
    return 0


@statement.register(imp.Accumulate)
def statement_accumulate(tree, parameters):
    flops = expression_flops(tree.indexsum.children[0], parameters)
    return flops + 1


@statement.register(imp.Return)
def statement_return(tree, parameters):
    flops = expression_flops(tree.expression, parameters)
    return flops + 1


@statement.register(imp.ReturnAccumulate)
def statement_returnaccumulate(tree, parameters):
    flops = expression_flops(tree.indexsum.children[0], parameters)
    return flops + 1


@statement.register(imp.Evaluate)
def statement_evaluate(tree, parameters):
    flops = expression_flops(tree.expression, parameters, top=True)
    return flops


@singledispatch
def flops(expr, parameters):
    raise NotImplementedError(f"Don't know how to count flops of {type(expr)}")


@flops.register(gem.Failure)
def flops_failure(expr, parameters):
    raise ValueError("Not expecting a Failure node")


@flops.register(gem.Variable)
@flops.register(gem.Identity)
@flops.register(gem.Delta)
@flops.register(gem.Zero)
@flops.register(gem.Literal)
@flops.register(gem.Index)
@flops.register(gem.VariableIndex)
def flops_zero(expr, parameters):
    # Initial set up of these Gem nodes are of 0 floating point operations.
    return 0


@flops.register(gem.LogicalNot)
@flops.register(gem.LogicalAnd)
@flops.register(gem.LogicalOr)
@flops.register(gem.ListTensor)
def flops_zeroplus(expr, parameters):
    # These nodes contribute 0 floating point operations, but their children may not.
    return 0 + sum(expression_flops(child, parameters)
                   for child in expr.children)


@flops.register(gem.Product)
def flops_product(expr, parameters):
    # Multiplication by -1 is not a flop.
    a, b = expr.children
    if isinstance(a, gem.Literal) and a.value == -1:
        return expression_flops(b, parameters)
    elif isinstance(b, gem.Literal) and b.value == -1:
        return expression_flops(a, parameters)
    else:
        return 1 + sum(expression_flops(child, parameters)
                       for child in expr.children)


@flops.register(gem.Sum)
@flops.register(gem.Division)
@flops.register(gem.Comparison)
@flops.register(gem.MathFunction)
@flops.register(gem.MinValue)
@flops.register(gem.MaxValue)
def flops_oneplus(expr, parameters):
    return 1 + sum(expression_flops(child, parameters)
                   for child in expr.children)


@flops.register(gem.Power)
def flops_power(expr, parameters):
    base, exponent = expr.children
    base_flops = expression_flops(base, parameters)
    if isinstance(exponent, gem.Literal):
        exponent = exponent.value
        if exponent > 0 and exponent == math.floor(exponent):
            return base_flops + int(math.ceil(math.log2(exponent)))
        else:
            return base_flops + 5  # heuristic
    else:
        return base_flops + 5   # heuristic


@flops.register(gem.Conditional)
def flops_conditional(expr, parameters):
    condition, then, else_ = (expression_flops(child, parameters)
                              for child in expr.children)
    return condition + max(then, else_)


@flops.register(gem.Indexed)
@flops.register(gem.FlexiblyIndexed)
def flops_indexed(expr, parameters):
    aggregate = sum(expression_flops(child, parameters)
                    for child in expr.children)
    # Average flops per entry
    return aggregate / numpy.product(expr.children[0].shape, dtype=int)


@flops.register(gem.IndexSum)
def flops_indexsum(expr, parameters):
    raise ValueError("Not expecting IndexSum")


@flops.register(gem.Inverse)
def flops_inverse(expr, parameters):
    n, _ = expr.shape
    # 2n^3 + child flop count
    return 2*n**3 + sum(expression_flops(child, parameters)
                        for child in expr.children)


@flops.register(gem.Solve)
def flops_solve(expr, parameters):
    n, m = expr.shape
    # 2mn + inversion cost of A + children flop count
    return 2*n*m + 2*n**3 + sum(expression_flops(child, parameters)
                                for child in expr.children)


@flops.register(gem.ComponentTensor)
def flops_componenttensor(expr, parameters):
    raise ValueError("Not expecting ComponentTensor")


def expression_flops(expression, parameters, top=False):
    """An approximation to flops required for each expression.

    :arg expression: GEM expression.
    :arg parameters: Useful miscellaneous information.
    :arg top: are we at the root?
    :returns: flop count for the expression
    """
    if not top and expression in parameters.temporaries:
        return 0
    else:
        return flops(expression, parameters)


def count_flops(impero_c):
    """An approximation to flops required for a scheduled impero_c tree.

    :arg impero_c: a :class:`~.Impero_C` object.
    :returns: approximate flop count for the tree.
    """
    try:
        return statement(impero_c.tree, impero_c)
    except (ValueError, NotImplementedError):
        return 0
