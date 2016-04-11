"""GEM is the intermediate language of TSFC for describing
tensor-valued mathematical expressions and tensor operations.
It is similar to Einstein's notation.

Its design was heavily inspired by UFL, with some major differences:
 - GEM has got nothing FEM-specific.
 - In UFL free indices are just unrolled shape, thus UFL is very
   restrictive about operations on expressions with different sets of
   free indices. GEM is much more relaxed about free indices.

Similarly to UFL, all GEM nodes have 'shape' and 'free_indices'
attributes / properties. Unlike UFL, however, index extents live on
the Index objects in GEM, not on all the nodes that have those free
indices.
"""

from __future__ import absolute_import

from abc import ABCMeta
from itertools import chain
from numpy import asarray, eye, unique

from gem.node import Node as NodeBase


__all__ = ['Node', 'Identity', 'Literal', 'Zero', 'Variable', 'Sum',
           'Product', 'Division', 'Power', 'MathFunction', 'MinValue',
           'MaxValue', 'Comparison', 'LogicalNot', 'LogicalAnd',
           'LogicalOr', 'Conditional', 'Index', 'VariableIndex',
           'Indexed', 'ComponentTensor', 'IndexSum', 'ListTensor',
           'Delta', 'partial_indexed']


class NodeMeta(type):
    """Metaclass of GEM nodes.

    When a GEM node is constructed, this metaclass automatically
    collects its free indices if 'free_indices' has not been set yet.
    """

    def __call__(self, *args, **kwargs):
        # Create and initialise object
        obj = super(NodeMeta, self).__call__(*args, **kwargs)

        # Set free_indices if not set already
        if not hasattr(obj, 'free_indices'):
            cfi = list(chain(*[c.free_indices for c in obj.children]))
            obj.free_indices = tuple(unique(cfi))

        return obj


class Node(NodeBase):
    """Abstract GEM node class."""

    __metaclass__ = NodeMeta

    __slots__ = ('free_indices')

    def is_equal(self, other):
        """Common subexpression eliminating equality predicate.

        When two (sub)expressions are equal, the children of one
        object are reassigned to the children of the other, so some
        duplicated subexpressions are eliminated.
        """
        result = NodeBase.is_equal(self, other)
        if result:
            self.children = other.children
        return result


class Terminal(Node):
    """Abstract class for terminal GEM nodes."""

    __slots__ = ()

    children = ()

    is_equal = NodeBase.is_equal


class Scalar(Node):
    """Abstract class for scalar-valued GEM nodes."""

    __slots__ = ()

    shape = ()


class Constant(Terminal):
    """Abstract base class for constant types.

    Convention:
     - array: numpy array of values
     - value: float value (scalars only)
    """
    __slots__ = ()


class Zero(Constant):
    """Symbolic zero tensor"""

    __slots__ = ('shape',)
    __front__ = ('shape',)

    def __init__(self, shape=()):
        self.shape = shape

    @property
    def value(self):
        assert not self.shape
        return 0.0


class Identity(Constant):
    """Identity matrix"""

    __slots__ = ('dim',)
    __front__ = ('dim',)

    def __init__(self, dim):
        self.dim = dim

    @property
    def shape(self):
        return (self.dim, self.dim)

    @property
    def array(self):
        return eye(self.dim)


class Literal(Constant):
    """Tensor-valued constant"""

    __slots__ = ('array',)
    __front__ = ('array',)

    def __new__(cls, array):
        array = asarray(array)
        if (array == 0).all():
            # All zeros, make symbolic zero
            return Zero(array.shape)
        else:
            return super(Literal, cls).__new__(cls)

    def __init__(self, array):
        self.array = asarray(array, dtype=float)

    def is_equal(self, other):
        if type(self) != type(other):
            return False
        if self.shape != other.shape:
            return False
        return tuple(self.array.flat) == tuple(other.array.flat)

    def get_hash(self):
        return hash((type(self), self.shape, tuple(self.array.flat)))

    @property
    def value(self):
        return float(self.array)

    @property
    def shape(self):
        return self.array.shape


class Variable(Terminal):
    """Symbolic variable tensor"""

    __slots__ = ('name', 'shape')
    __front__ = ('name', 'shape')

    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class Sum(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape

        # Zero folding
        if isinstance(a, Zero):
            return b
        elif isinstance(b, Zero):
            return a

        self = super(Sum, cls).__new__(cls)
        self.children = a, b
        return self


class Product(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape

        # Zero folding
        if isinstance(a, Zero) or isinstance(b, Zero):
            return Zero()

        self = super(Product, cls).__new__(cls)
        self.children = a, b
        return self


class Division(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape

        # Zero folding
        if isinstance(b, Zero):
            raise ValueError("division by zero")
        if isinstance(a, Zero):
            return Zero()

        self = super(Division, cls).__new__(cls)
        self.children = a, b
        return self


class Power(Scalar):
    __slots__ = ('children',)

    def __new__(cls, base, exponent):
        assert not base.shape
        assert not exponent.shape

        # Zero folding
        if isinstance(base, Zero):
            if isinstance(exponent, Zero):
                raise ValueError("cannot solve 0^0")
            return Zero()
        elif isinstance(exponent, Zero):
            return Literal(1)

        self = super(Power, cls).__new__(cls)
        self.children = base, exponent
        return self


class MathFunction(Scalar):
    __slots__ = ('name', 'children')
    __front__ = ('name',)

    def __init__(self, name, argument):
        assert isinstance(name, str)
        assert not argument.shape

        self.name = name
        self.children = argument,


class MinValue(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class MaxValue(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class Comparison(Scalar):
    __slots__ = ('operator', 'children')
    __front__ = ('operator',)

    def __init__(self, op, a, b):
        assert not a.shape
        assert not b.shape

        if op not in [">", ">=", "==", "!=", "<", "<="]:
            raise ValueError("invalid operator")

        self.operator = op
        self.children = a, b


class LogicalNot(Scalar):
    __slots__ = ('children',)

    def __init__(self, expression):
        assert not expression.shape

        self.children = expression,


class LogicalAnd(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class LogicalOr(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class Conditional(Node):
    __slots__ = ('children', 'shape')

    def __init__(self, condition, then, else_):
        assert not condition.shape
        assert then.shape == else_.shape

        self.children = condition, then, else_
        self.shape = then.shape


class IndexBase(object):
    """Abstract base class for indices."""

    __metaclass__ = ABCMeta

IndexBase.register(int)


class Index(IndexBase):
    """Free index"""

    # Not true object count, just for naming purposes
    _count = 0

    __slots__ = ('name', 'extent', 'count')

    def __init__(self, name=None):
        self.name = name
        Index._count += 1
        self.count = Index._count
        # Initialise with indefinite extent
        self.extent = None

    def set_extent(self, value):
        # Set extent, check for consistency
        if self.extent is None:
            self.extent = value
        elif self.extent != value:
            raise ValueError("Inconsistent index extents!")

    def __str__(self):
        if self.name is None:
            return "i_%d" % self.count
        return self.name

    def __repr__(self):
        if self.name is None:
            return "Index(%r)" % self.count
        return "Index(%r)" % self.name


class VariableIndex(IndexBase):
    """An index that is constant during a single execution of the
    kernel, but whose value is not known at compile time."""

    def __init__(self, expression):
        assert isinstance(expression, Node)
        assert not expression.free_indices
        assert not expression.shape
        self.expression = expression

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        return self.expression == other.expression

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((VariableIndex, self.expression))


class Indexed(Scalar):
    __slots__ = ('children', 'multiindex')
    __back__ = ('multiindex',)

    def __new__(cls, aggregate, multiindex):
        # Set index extents from shape
        assert len(aggregate.shape) == len(multiindex)
        for index, extent in zip(multiindex, aggregate.shape):
            assert isinstance(index, IndexBase)
            if isinstance(index, Index):
                index.set_extent(extent)

        # Zero folding
        if isinstance(aggregate, Zero):
            return Zero()

        # All indices fixed
        if all(isinstance(i, int) for i in multiindex):
            if isinstance(aggregate, Constant):
                return Literal(aggregate.array[multiindex])
            elif isinstance(aggregate, ListTensor):
                return aggregate.array[multiindex]

        self = super(Indexed, cls).__new__(cls)
        self.children = (aggregate,)
        self.multiindex = multiindex

        new_indices = tuple(i for i in multiindex if isinstance(i, Index))
        self.free_indices = tuple(unique(aggregate.free_indices + new_indices))

        return self


class ComponentTensor(Node):
    __slots__ = ('children', 'multiindex', 'shape')
    __back__ = ('multiindex',)

    def __new__(cls, expression, multiindex):
        assert not expression.shape

        # Collect shape
        shape = tuple(index.extent for index in multiindex)
        assert all(shape)

        # Zero folding
        if isinstance(expression, Zero):
            return Zero(shape)

        self = super(ComponentTensor, cls).__new__(cls)
        self.children = (expression,)
        self.multiindex = multiindex
        self.shape = shape

        # Collect free indices
        assert set(multiindex) <= set(expression.free_indices)
        self.free_indices = tuple(set(expression.free_indices) - set(multiindex))

        return self


class IndexSum(Scalar):
    __slots__ = ('children', 'index')
    __back__ = ('index',)

    def __new__(cls, summand, index):
        # Sum zeros
        assert not summand.shape
        if isinstance(summand, Zero):
            return summand

        # Sum a single expression
        if index.extent == 1:
            return Indexed(ComponentTensor(summand, (index,)), (0,))

        self = super(IndexSum, cls).__new__(cls)
        self.children = (summand,)
        self.index = index

        # Collect shape and free indices
        assert index in summand.free_indices
        self.free_indices = tuple(set(summand.free_indices) - {index})

        return self


class ListTensor(Node):
    __slots__ = ('array',)

    def __new__(cls, array):
        array = asarray(array)

        # Zero folding
        if all(isinstance(elem, Zero) for elem in array.flat):
            assert all(elem.shape == () for elem in array.flat)
            return Zero(array.shape)

        self = super(ListTensor, cls).__new__(cls)
        self.array = array
        return self

    @property
    def children(self):
        return tuple(self.array.flat)

    @property
    def shape(self):
        return self.array.shape

    def reconstruct(self, *args):
        return ListTensor(asarray(args).reshape(self.array.shape))

    def __repr__(self):
        return "ListTensor(%r)" % self.array.tolist()

    def is_equal(self, other):
        """Common subexpression eliminating equality predicate."""
        if type(self) != type(other):
            return False
        if (self.array == other.array).all():
            self.array = other.array
            return True
        return False

    def get_hash(self):
        return hash((type(self), self.shape, self.children))


class Delta(Scalar, Terminal):
    __slots__ = ('i', 'j')
    __front__ = ('i', 'j')

    def __new__(cls, i, j):
        assert isinstance(i, IndexBase)
        assert isinstance(j, IndexBase)

        # \delta_{i,i} = 1
        if i == j:
            return Literal(1)

        # Fixed indices
        if isinstance(i, int) and isinstance(j, int):
            return Literal(int(i == j))

        self = super(Delta, cls).__new__(cls)
        self.i = i
        self.j = j
        # Set up free indices
        free_indices = tuple(index for index in (i, j) if isinstance(index, Index))
        self.free_indices = tuple(unique(free_indices))
        return self


def partial_indexed(tensor, indices):
    """Generalised indexing into a tensor.  The number of indices may
    be less than or equal to the rank of the tensor, so the result may
    have a non-empty shape.

    :arg tensor: tensor-valued GEM expression
    :arg indices: indices, at most as many as the rank of the tensor
    :returns: a potentially tensor-valued expression
    """
    if len(indices) == 0:
        return tensor
    elif len(indices) < len(tensor.shape):
        rank = len(tensor.shape) - len(indices)
        shape_indices = tuple(Index() for i in range(rank))
        return ComponentTensor(
            Indexed(tensor, indices + shape_indices),
            shape_indices)
    elif len(indices) == len(tensor.shape):
        return Indexed(tensor, indices)
    else:
        raise ValueError("More indices than rank!")
