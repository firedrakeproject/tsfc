from functools import singledispatch
import numpy as np

@singledispatch
def flops(expr):
    raise NotImplementedError


class Int:
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
    def __init__(self, value):
        self.value = value
@flops.register(Variable)
def flops_variable(expr):
    return 0


class Identity:
    children = () 
    def __init__(self, value):
        self.value = value
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
    return 1 + sum(map(flops, expr.children)) 


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
    return np.sum(expr.array) + sum(map(flops, expr.children)) #Sum of array elements + children flop count



#Index sum = flops in the inside term * size of sum in the outside
#Component tensor = flops in the scalar term * size of matrix


['Node', 'Index', 'Literal', 'Power', 'MathFunction', 'Conditional', 'Indexed', 'IndexSum', 'VariableIndex', 'FlexiblyIndexed', 'ComponentTensor']

