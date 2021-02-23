import sparse
import numpy

print("2 Dimensional Matrix Vector contraction")

# Create sparse tensor - will be a gem.SparseLiteral
ndim = 2
nnz = 4
shape = (6, 5)
assert len(shape) == ndim
coords = numpy.empty((ndim, nnz))
data = numpy.empty((nnz,))
# Set 6 on main diagonal, all set to 100
for n in range(nnz):
    coords[:, n] = numpy.array([n, n])
    data[n] = 100
A = sparse.COO(coords, data, shape=shape)

# Create vector to contract with - a gem.Indexed
x = numpy.ones((A.shape[1],))

# Do contraction, yielding a dense tensor - a
# gem.SparseContract(A[n, p], x[p], (p,)) operation which should return
# a scalar valued tensor with 1 free index n
dep_indices_for_contraction = [1]  # Contract over the 2nd of A's shape indices (p,)
result_shape = numpy.delete(A.shape, dep_indices_for_contraction)
result = numpy.zeros(result_shape)  # Start with zero vector for the result
assert A.nnz == nnz
# The index over the non-zeros is an independent index that yields dependent
# indices (p and n) into the sparse tensor which are used for the contraction
for indep_index in range(nnz):
    dep_indices = A.coords[:, indep_index]
    x_indices = tuple(dep_indices[dep_indices_for_contraction])
    result_indices = tuple(numpy.delete(dep_indices,dep_indices_for_contraction))
    result[result_indices] += A.data[indep_index] * x[x_indices]

print("Calculating Ax. A is:")
print(A.todense())
print("x is:")
print(x)
print("Ax using this calculation is:")
print(result)
print("Ax correct is:")
print(A.todense() @ x)


print("3 Dimensional Sparse Tensor / 2 Dimensional Dense Tensor contraction")

# Create sparse tensor - will be a gem.SparseLiteral
ndim = 3
nnz = 4
shape = (6, 5, 4)
assert len(shape) == ndim
coords = numpy.empty((ndim, nnz))
data = numpy.empty((nnz,))
# Set 6 on main diagonal, all set to 100
for n in range(nnz):
    coords[:, n] = numpy.array([n, n, n])
    data[n] = 100
M = sparse.COO(coords, data, shape=shape)

# Create vector to contract with - a gem.Indexed
A = numpy.ones((M.shape[1], M.shape[2]))

# Do contraction, yielding a dense tensor - a
# gem.SparseContract(M[n, p, l], A[p, l], (p, l)) operation which should return
# a scalar valued tensor with 1 free index n
dep_indices_for_contraction = [1, 2]  # Contract over the 2nd & 3rd of M's shape indices (p, l)
result_shape = numpy.delete(M.shape, dep_indices_for_contraction)
result = numpy.zeros(result_shape)  # Start with zero vector for the result
assert M.nnz == nnz
# The index over the non-zeros is an independent index that yields dependent
# indices (p and n) into the sparse tensor which are used for the contraction
for indep_index in range(nnz):
    dep_indices = M.coords[:, indep_index]
    A_indices = tuple(dep_indices[dep_indices_for_contraction])
    result_indices = tuple(numpy.delete(dep_indices,dep_indices_for_contraction))
    result[result_indices] += M.data[indep_index] * A[A_indices]

print("Calculating einsum('npl,pl->n', M, A). M is:")
print(M.todense())
print("A is:")
print(A)
print("einsum('npl,pl->n', M, A) using this calculation is:")
print(result)
print("einsum('npl,pl->n', M, A) correct is:")
print(numpy.einsum('npl,pl->n', M.todense(), A))