import pytest
import gem
import sparse
import numpy


def test_sparse_literal_creation():
    # Create from numpy array
    x_dense = numpy.array([[1, 2], [3, 4]])
    x_dense_gem = gem.Literal(x_dense)
    x_sparse = sparse.as_coo(x_dense)
    x_sparse_gem = gem.SparseLiteral(x_dense)
    assert x_sparse_gem.shape == (2, 2)
    assert x_sparse_gem == x_sparse_gem
    assert x_sparse_gem != x_dense_gem
    assert x_sparse_gem.array == x_sparse
    y_dense = numpy.array([[2, 0], [0, 0]])
    y_sparse_gem = gem.SparseLiteral(y_dense)
    assert y_sparse_gem != x_dense_gem
    # Create from sparse array
    assert gem.SparseLiteral(x_sparse) == x_sparse_gem
    x_sparse_dok = sparse.DOK.from_numpy(x_dense)
    assert gem.SparseLiteral(x_sparse_dok) == x_sparse_gem


def test_sparse_literal_value():
    pytest.importorskip("sparse", minversion='0.11.3', reason="bug when creating sparse COO from 0d numpy array: fixed on master")
    # Get value for scalar
    assert gem.SparseLiteral(numpy.array(1)).value == 1
    assert gem.SparseLiteral(numpy.array(1)).shape == ()


def test_sparse_literal_zero():
    # Get Zero for zero vector
    assert gem.SparseLiteral(numpy.zeros((2, 1, 1))) == gem.Zero((2, 1, 1))
    assert gem.SparseLiteral(sparse.COO([], [], (5, 6, 3))) == gem.Zero((5, 6, 3))


def test_sparse_literal_dtype():
    # Datatype is always float or complex float
    assert gem.SparseLiteral(sparse.COO([[2], [3], [2]], [10], (5, 6, 4))).array.dtype == float
    assert gem.SparseLiteral(sparse.DOK((5, 5, 6), {(0, 0, 0): 10.0})).array.dtype == float
    assert gem.SparseLiteral(sparse.COO([[2], [3], [2]], [10 + 10j], (5, 6, 4))).array.dtype == complex
