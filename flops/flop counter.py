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

class IndexSum:
    def __init__(self, left, right):
        self.children = (left, right)

@flops.register(IndexSum)
def flops_IndexSum(expr):
    return 

#Index sum = flops in the inside term * size of sum in the outside
#Component tensor = flops in the scalar term * size of matrix
#List tensor = sum of things in it 

['Node', 'Identity', 'Literal', 'Zero', 'Failure',
'Variable', 'Sum', 'Product', 'Division', 'Power',
'MathFunction', 'MinValue', 'MaxValue', 'Comparison',
'LogicalNot', 'LogicalAnd', 'LogicalOr', 'Conditional',
'Index', 'VariableIndex', 'Indexed', 'ComponentTensor',
'IndexSum', 'ListTensor', 'Delta', 'FlexiblyIndexed',
'Inverse', 'Solve']
