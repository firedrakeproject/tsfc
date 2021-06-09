import gem.gem as gem
from gem.node import Memoizer
from functools import singledispatch
import numpy
import math


@singledispatch
def flops(expr, self):
    raise NotImplementedError(f"Don't know how to count flops of {type(expr)}")


@flops.register(gem.Failure)
def flops_failure(expr, self):
    raise ValueError("Not expecting a Failure node")


@flops.register(gem.Variable)
@flops.register(gem.Identity)
@flops.register(gem.Delta)
@flops.register(gem.Zero)
@flops.register(gem.Literal)
@flops.register(gem.Index)
@flops.register(gem.VariableIndex)
def flops_zero(expr, self):
    return 0


@flops.register(gem.LogicalNot)
@flops.register(gem.LogicalAnd)
@flops.register(gem.LogicalOr)
@flops.register(gem.ListTensor)
def flops_zeroplus(expr, self):
    return 0 + sum(map(self, expr.children))


@flops.register(gem.Sum)
@flops.register(gem.Product)
@flops.register(gem.Division)
@flops.register(gem.Comparison)
@flops.register(gem.MathFunction)
@flops.register(gem.MinValue)
@flops.register(gem.MaxValue)
def flops_oneplus(expr, self):
    return 1 + sum(map(self, expr.children))


@flops.register(gem.Power)
def flops_power(expr, self):
    base, exponent = expr.children
    base_flops = self(base)
    if isinstance(exponent, gem.Literal):
        exponent = exponent.value
        if exponent > 0 and exponent == math.floor(exponent):
            return base_flops + int(math.ceil(math.log2(exponent)))
    else:
        return base_flops + 5 # heuristic


@flops.register(gem.Conditional)
def flops_conditional(expr, self):
    condition, then, else_ = map(self, expr.children)
    return condition + max(then, else_)


@flops.register(gem.Indexed)
@flops.register(gem.FlexiblyIndexed)
def flops_indexed(expr, self):
    # Average flops per entry
    return sum(map(self, expr.children)) / numpy.product(expr.children[0].shape, dtype=int)


@flops.register(gem.IndexSum)
def flops_indexsum(expr, self):
    # Sum of the child flops multiplied by the extent of the indices being summed over
    return (sum(map(self, expr.children)) * numpy.product([i.extent for i in expr.multiindex], dtype=int))


@flops.register(gem.Inverse)
def flops_inverse(expr, self):
    n, _ = expr.shape
    # 2n^3 + child flop count
    return 2*n**3 + sum(map(self, expr.children))

@flops.register(gem.Solve)
def flops_solve(expr, self):
    n, m = expr.shape
    # 2mn + inversion cost of A + children flop count
    return 2*n*m + 2*n**3 + sum(map(self, expr.children))


@flops.register(gem.ComponentTensor)
def flops_componenttensor(expr, self):
    # Sum of the child flops multiplied by the extent of the indices being turned into shape
    return (sum(map(self, expr.children)) * numpy.product([i.extent for i in expr.multiindex], dtype=int))


def count_flops(expressions):
    """An approximation to flops required for each expression.

    :arg expressions: iterable of gem expression.
    :returns: list of flop counts for each expression
    """
    mapper = Memoizer(flops)
    return [mapper(expr) for expr in expressions]
