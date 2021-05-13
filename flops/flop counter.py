from functools import singledispatch

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
    def __init__(self, left, right):
        self.value = 0

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
    def __init__(self, expression):
        self.children = expression

@flops.register(LogicalNot)
def flops_logicalnot(expr):
    return 1 + sum(map(flops, expr.children)) #Should be 1+expression? 


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


class IndexSum:
    def __init__(self, left, right):
        self.children = (left, right)

@flops.register(IndexSum)
def flops_indexsum(expr):
    return 

#List tensor = sum of things in it 
#Index sum = flops in the inside term * size of sum in the outside
#Component tensor = flops in the scalar term * size of matrix


['Node', 
'Literal', #0
'Failure', #0


'Variable', #0
'Power', 
'MathFunction', 
'Conditional',
'Index', 'Indexed', 
'Inverse', 
'Solve',


'VariableIndex', 'FlexiblyIndexed', 'ListTensor', 'IndexSum', 'ComponentTensor']

