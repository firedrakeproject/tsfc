from functools import singledispatch
import numpy 
from numpy import asarray

@singledispatch
def flops(expr):
    raise NotImplementedError


class Int: #Integer value
    children = () 
    def __init__(self, value):
        self.value = value
@flops.register(Int)
def flops_int(expr):
    return 0


class Failure:
    children = () 
    def __init__(self, shape, exception):
        self.shape = shape
        self.exception = exception
@flops.register(Failure)
def flops_failure(expr):
    return 0


class Variable: 
    children = () 
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
@flops.register(Variable)
def flops_variable(expr):
    return 0


class Identity: #Identity matrix of size dim
    children = () 
    def __init__(self, dim):
        self.dim = dim
@flops.register(Identity)
def flops_identity(expr):
    return 0


class Delta: #Kronecker Delta
    children = () 
    def __init__(self, value):
        self.value = value
@flops.register(Delta)
def flops_delta(expr):
    return 0


class Zero: #Sets value to 0.0
    def __init__(self, shape):
        self.shape = shape
@flops.register(Zero)
def flops_zero(expr):
    return 0


class Literal: #Literal array of numbers
    def __init__(self, array):
        array = asarray(array)
        try:
            self.array = array.astype(float, casting="safe")
        except TypeError:
            self.array = array.astype(complex)
@flops.register(Literal)
def flops_literal(expr):
    return 0


class Plus:
    def __init__(self, left, right):
        self.children = (left, right)
@flops.register(Plus)
def flops_plus(expr):
    return 1 + sum(map(flops, expr.children))


class Minus: #Minus not listed in gem list
    def __init__(self, left, right):
        self.children = (left, right)
@flops.register(Minus)
def flops_minus(expr):
    return 1 + sum(map(flops, expr.children))


class Times:
    def __init__(self, left, right):
        self.children = (left, right)
@flops.register(Times)
def flops_times(expr):
    return 1 + sum(map(flops, expr.children))


class Divide:
    def __init__(self, left, right):
        self.children = (left, right)
@flops.register(Divide)
def flops_divide(expr):
    return 1 + sum(map(flops, expr.children))


class Power:
    def __init__(self, base, exponent):
        self.children = (base, exponent)
@flops.register(Power)
def flops_power(expr):
    exponent = expr.exponent
    if exponent == 0:
        return 0
    elif abs(exponent) < 1:
        exponent = 1/exponent
    if exponent < 0:
        exponent = -1*exponent
    if exponent < 5:
        return exponent + sum(map(flops, expr.condition))
    else:
        return 5 + sum(map(flops, expr.condition))


class MathFunction: #e.g. sin(some_expr)
    def __init__(self, name, child):
        self.name= name
        self.children = child
@flops.register(MathFunction)
def flops_mathfunction(expr):
    return 1 + sum(map(flops, expr.children)) 


class Conditional:
    def __init__(self, condition, left, right):
        self.condition = condition
        self.left = left
        self.right = right
@flops.register(Conditional)
def flops_conditional(expr):
    return sum(map(flops, expr.condition)) + max(sum(map(flops, expr.left)), sum(map(flops, expr.right)))


class MinValue:
    def __init__(self, left, right):
        self.children = (left, right)
@flops.register()
def flops_minvalue(expr):
    return 1 + sum(map(flops, expr.children))


class MaxValue:
    def __init__(self, left, right):
        self.children = (left, right)
@flops.register()
def flops_maxvalue(expr):
    return 1 + sum(map(flops, expr.children))


class LogicalNot:
    def __init__(self, child):
        self.children = child
@flops.register(LogicalNot)
def flops_logicalnot(expr):
    return 1 + sum(map(flops, expr.children)) #Should be 1+children or 0+children? 


class LogicalAnd:
    def __init__(self, left, right):
        self.children = (left, right)
@flops.register(LogicalAnd)
def flops_logicaland(expr):
    return 1 + sum(map(flops, expr.children))


class LogicalOr:
    def __init__(self, left, right):
        self.children = (left, right)
@flops.register(LogicalOr)
def flops_logicalor(expr):
    return 1 + sum(map(flops, expr.children))


class Index:
    children = () 
    def __init__(self, name=None, extent=None):
        self.name = name
        Index._count += 1
        self.count = Index._count
        self.extent = extent
@flops.register(Index)
def flops_index(expr):
    return 0


class Indexed:
    def __init__(self, child):
        self.children = child
@flops.register(Indexed)
def flops_indexed(expr):
    return sum(map(flops, expr.children)) / numpy.product(expr.child.shape) #Average flops per entry


class FlexiblyIndexed:
    def __init__(self, child):
        self.children = child
@flops.register(FlexiblyIndexed)
def flops_flexiblyindexed(expr):
    return sum(map(flops, expr.children)) / numpy.product(expr.child.shape) #Average flops per entry, same as Indexed


class VariableIndex: 
    children = () 
    def __init__(self, expression):
        self.expression = expression
@flops.register(VariableIndex)
def flops_variableindex(expr):
    return 0


class IndexSum:
    def __init__(self, child, index):
        self.children = child
        self.index = index
@flops.register(IndexSum)
def flops_indexsum(expr):
    return sum(map(flops, expr.children)) * max(expr.index) #Sum of the child flops multiplied by the extent of the indicies being summed over


class Comparison:
    def __init__(self, left, right):
        self.children = (left, right)
@flops.register(Comparison)
def flops_comparison(expr):
    return 1 + sum(map(flops, expr.children))


class Inverse:
    def __init__(self, child):
        self.children = child
@flops.register(Inverse)
def flops_inverse(expr):
    return 2*(expr.child.shape[0])**3 + sum(map(flops, expr.child)) #2n^3 + children flop count


class Solve:
    def __init__(self, A, B):
        self.children = (A, B)
        self.A = A
        self.B = B
@flops.register(Solve)
def flops_solve(expr):
    return 2*(expr.A.shape[0])*(expr.B.shape[0]) + 2*(expr.A.shape[0])**3 + sum(map(flops, expr.children)) #2mn + inversion cost of A + children flop count 


class ListTensor:
    def __init__(self, array):
        self.array = array
@flops.register(ListTensor)
def flops_listtensor(expr):
    return numpy.sum(expr.array) + sum(map(flops, expr.children)) #Sum of array elements + children flop count


class ComponentTensor:
    def __init__(self, child, index):
        self.children = child
        self.index = index
@flops.register(ComponentTensor)
def flops_componenttensor(expr):
    return sum(map(flops, expr.children)) * max(expr.index) #Sum of the child flops multiplied by the extent of the indicies being turned into shape

