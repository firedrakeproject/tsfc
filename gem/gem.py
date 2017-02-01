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

from __future__ import absolute_import, print_function, division
from six import with_metaclass
from six.moves import range, zip

from abc import ABCMeta
from itertools import chain
from operator import attrgetter

import numpy
from numpy import asarray

from gem.node import Node as NodeBase


__all__ = ['Node', 'Identity', 'Literal', 'Zero', 'Failure',
           'Variable', 'Sum', 'Product', 'Division', 'Power',
           'MathFunction', 'MinValue', 'MaxValue', 'Comparison',
           'LogicalNot', 'LogicalAnd', 'LogicalOr', 'Conditional',
           'Index', 'VariableIndex', 'Indexed', 'ComponentTensor',
           'IndexSum', 'ListTensor', 'Delta', 'index_sum',
           'partial_indexed', 'reshape', 'view']


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
            obj.free_indices = unique(chain(*[c.free_indices
                                              for c in obj.children]))

        return obj


class Node(with_metaclass(NodeMeta, NodeBase)):
    """Abstract GEM node class."""

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


class Failure(Terminal):
    """Abstract class for failure GEM nodes."""

    __slots__ = ('shape', 'exception')
    __front__ = ('shape', 'exception')

    def __init__(self, shape, exception):
        self.shape = shape
        self.exception = exception


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
        return numpy.eye(self.dim)


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

        # Constant folding
        if isinstance(a, Zero):
            return b
        elif isinstance(b, Zero):
            return a

        if isinstance(a, Constant) and isinstance(b, Constant):
            return Literal(a.value + b.value)

        self = super(Sum, cls).__new__(cls)
        self.children = a, b
        return self


class Product(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape

        # Constant folding
        if isinstance(a, Zero) or isinstance(b, Zero):
            return Zero()

        if a == one:
            return b
        if b == one:
            return a

        if isinstance(a, Constant) and isinstance(b, Constant):
            return Literal(a.value * b.value)

        self = super(Product, cls).__new__(cls)
        self.children = a, b
        return self


class Division(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape

        # Constant folding
        if isinstance(b, Zero):
            raise ValueError("division by zero")
        if isinstance(a, Zero):
            return Zero()

        if b == one:
            return a

        if isinstance(a, Constant) and isinstance(b, Constant):
            return Literal(a.value / b.value)

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
            return one

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

    def __new__(cls, condition, then, else_):
        assert not condition.shape
        assert then.shape == else_.shape

        # If both branches are the same, just return one of them.  In
        # particular, this will help constant-fold zeros.
        if then == else_:
            return then

        self = super(Conditional, cls).__new__(cls)
        self.children = condition, then, else_
        self.shape = then.shape
        return self


class IndexBase(with_metaclass(ABCMeta)):
    """Abstract base class for indices."""
    pass


IndexBase.register(int)


class Index(IndexBase):
    """Free index"""

    # Not true object count, just for naming purposes
    _count = 0

    __slots__ = ('name', 'extent', 'count')

    def __init__(self, name=None, extent=None):
        self.name = name
        Index._count += 1
        self.count = Index._count
        self.extent = extent

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

    def __lt__(self, other):
        # Allow sorting of free indices in Python 3
        return id(self) < id(other)


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

        # Empty multiindex
        if not multiindex:
            return aggregate

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
        self.free_indices = unique(aggregate.free_indices + new_indices)

        return self


class FlexiblyIndexed(Scalar):
    """Flexible indexing of :py:class:`Variable`s to implement views and
    reshapes (splitting dimensions only)."""

    __slots__ = ('children', 'dim2idxs')
    __back__ = ('dim2idxs',)

    def __init__(self, variable, dim2idxs):
        """Construct a flexibly indexed node.

        :arg variable: a :py:class:`Variable`
        :arg dim2idxs: describes the mapping of indices

        For example, if ``variable`` is rank two, and ``dim2idxs`` is

            ((1, ((i, 12), (j, 4), (k, 1))), (0, ()))

        then this corresponds to the indexing:

            variable[1 + i*12 + j*4 + k][0]

        """
        assert isinstance(variable, Variable)
        assert len(variable.shape) == len(dim2idxs)

        dim2idxs_ = []
        free_indices = []
        for dim, (offset, idxs) in zip(variable.shape, dim2idxs):
            offset_ = offset
            idxs_ = []
            last = 0
            for idx in idxs:
                index, stride = idx
                if isinstance(index, Index):
                    assert index.extent is not None
                    free_indices.append(index)
                    idxs_.append((index, stride))
                    last += (index.extent - 1) * stride
                elif isinstance(index, int):
                    offset_ += index * stride
                else:
                    raise ValueError("Unexpected index type for flexible indexing")

            if dim is not None and offset_ + last >= dim:
                raise ValueError("Offset {0} and indices {1} exceed dimension {2}".format(offset, idxs, dim))

            dim2idxs_.append((offset_, tuple(idxs_)))

        self.children = (variable,)
        self.dim2idxs = tuple(dim2idxs_)
        self.free_indices = unique(free_indices)


class ComponentTensor(Node):
    __slots__ = ('children', 'multiindex', 'shape')
    __back__ = ('multiindex',)

    def __new__(cls, expression, multiindex):
        assert not expression.shape

        # Empty multiindex
        if not multiindex:
            return expression

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
        self.free_indices = unique(set(expression.free_indices) - set(multiindex))

        return self


class IndexSum(Scalar):
    __slots__ = ('children', 'multiindex')
    __back__ = ('multiindex',)

    def __new__(cls, summand, multiindex):
        # Sum zeros
        assert not summand.shape
        if isinstance(summand, Zero):
            return summand

        # Unroll singleton sums
        unroll = tuple(index for index in multiindex if index.extent <= 1)
        if unroll:
            assert numpy.prod([index.extent for index in unroll]) == 1
            summand = Indexed(ComponentTensor(summand, unroll),
                              (0,) * len(unroll))
            multiindex = tuple(index for index in multiindex
                               if index not in unroll)

        # No indices case
        multiindex = tuple(multiindex)
        if not multiindex:
            return summand

        self = super(IndexSum, cls).__new__(cls)
        self.children = (summand,)
        self.multiindex = multiindex

        # Collect shape and free indices
        assert set(multiindex) <= set(summand.free_indices)
        self.free_indices = unique(set(summand.free_indices) - set(multiindex))

        return self


class ListTensor(Node):
    __slots__ = ('array',)

    def __new__(cls, array):
        array = asarray(array)
        assert numpy.prod(array.shape)

        # Handle children with shape
        child_shape = array.flat[0].shape
        assert all(elem.shape == child_shape for elem in array.flat)

        if child_shape:
            # Destroy structure
            direct_array = numpy.empty(array.shape + child_shape, dtype=object)
            for alpha in numpy.ndindex(array.shape):
                for beta in numpy.ndindex(child_shape):
                    direct_array[alpha + beta] = Indexed(array[alpha], beta)
            array = direct_array

        # Constant folding
        if all(isinstance(elem, Constant) for elem in array.flat):
            return Literal(numpy.vectorize(attrgetter('value'))(array))

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
            return one

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


def unique(indices):
    """Sorts free indices and eliminates duplicates.

    :arg indices: iterable of indices
    :returns: sorted tuple of unique free indices
    """
    return tuple(sorted(set(indices), key=id))


def index_sum(expression, indices):
    """Eliminates indices from the free indices of an expression by
    summing over them.  Skips any index that is not a free index of
    the expression."""
    multiindex = tuple(index for index in indices
                       if index in expression.free_indices)
    return IndexSum(expression, multiindex)


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


def strides_of(shape):
    """Calculate cumulative strides from per-dimension capacities.

    For example:

        [2, 3, 4] ==> [12, 4, 1]

    """
    temp = numpy.flipud(numpy.cumprod(numpy.flipud(list(shape)[1:])))
    return list(temp) + [1]


def decompose_variable_view(expression):
    """Extract ComponentTensor + FlexiblyIndexed view onto a variable."""
    if isinstance(expression, Variable):
        variable = expression
        indexes = tuple(Index(extent=extent) for extent in expression.shape)
        dim2idxs = tuple((0, ((index, 1),)) for index in indexes)
    elif isinstance(expression, ComponentTensor) and isinstance(expression.children[0], FlexiblyIndexed):
        variable = expression.children[0].children[0]
        indexes = expression.multiindex
        dim2idxs = expression.children[0].dim2idxs
    else:
        raise ValueError("Cannot handle {} objects.".format(type(expression).__name__))

    return variable, dim2idxs, indexes


def reshape(expression, *shapes):
    """Reshape a variable (splitting indices only).

    :arg expression: view of a :py:class:`Variable`
    :arg shapes: one shape tuple for each dimension of the variable.
    """
    variable, dim2idxs, indexes = decompose_variable_view(expression)
    assert len(indexes) == len(shapes)
    shape_of = dict(zip(indexes, shapes))

    dim2idxs_ = []
    indices = []
    for offset, idxs in dim2idxs:
        idxs_ = []
        for idx in idxs:
            index, stride = idx
            assert isinstance(index, Index)
            dim = index.extent
            shape = shape_of[index]
            if dim is not None and numpy.prod(shape) != dim:
                raise ValueError("Shape {} does not match extent {}.".format(shape, dim))
            strides = strides_of(shape)
            for extent, stride_ in zip(shape, strides):
                index = Index(extent=extent)
                idxs_.append((index, stride_ * stride))
                indices.append(index)
        dim2idxs_.append((offset, tuple(idxs_)))

    expr = FlexiblyIndexed(variable, tuple(dim2idxs_))
    return ComponentTensor(expr, tuple(indices))


def view(expression, *slices):
    """View a part of a variable.

    :arg expression: view of a :py:class:`Variable`
    :arg slices: one slice object for each dimension of the variable.
    """
    variable, dim2idxs, indexes = decompose_variable_view(expression)
    assert len(indexes) == len(slices)
    slice_of = dict(zip(indexes, slices))

    dim2idxs_ = []
    indices = []
    for offset, idxs in dim2idxs:
        offset_ = offset
        idxs_ = []
        for idx in idxs:
            index, stride = idx
            assert isinstance(index, Index)
            dim = index.extent
            s = slice_of[index]
            start = s.start or 0
            stop = s.stop or dim
            if stop is None:
                raise ValueError("Unknown extent!")
            if dim is not None and stop > dim:
                raise ValueError("Slice exceeds dimension extent!")
            step = s.step or 1
            offset_ += start * stride
            extent = 1 + (stop - start - 1) // step
            index_ = Index(extent=extent)
            indices.append(index_)
            idxs_.append((index_, step * stride))
        dim2idxs_.append((offset_, tuple(idxs_)))

    expr = FlexiblyIndexed(variable, tuple(dim2idxs_))
    return ComponentTensor(expr, tuple(indices))


# Static one object for quicker constant folding
one = Literal(1)
