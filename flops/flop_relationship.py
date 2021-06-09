import numpy
from matplotlib import pyplot as plt
from coffee.visitors import EstimateFlops
from gem.flop_count import count_flops
from tsfc import compile_form
from ufl import (FiniteElement, FunctionSpace, Mesh, TestFunction,
                 TrialFunction, VectorElement, dx, grad, inner, quadrilateral)
mesh = Mesh(VectorElement("Q", quadrilateral, 1))
naive_parameters = {"mode": "vanilla"}
optimised_parameters = {"mode": "spectral"}
new_flops = []
old_flops = []
K = range(1, 33)

for k in K:
    V = FunctionSpace(mesh, FiniteElement("Q", quadrilateral, k))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v)*dx + inner(grad(u), grad(v))*dx
    kernel, = compile_form(a, prefix="form",
                           parameters=optimised_parameters,
                           coffee=True)
    # Record new flops here, and compare asymptotics and approximate
    # order of magnitude.
    
    newflops = count_flops(kernel.ast)
    new_flops.append(newflops)

    oldflops = EstimateFlops().visit(kernel.ast)
    old_flops.append(oldflops)
    
    print(f"New flops for degree {k}: {newflops}")
    print(f"Old flops for degree {k}: {oldflops}")

old_flops = numpy.asarray(old_flops)

# A_ij = \sum_{k} w_ijk phi_{ik} * psi_{jk}
# Naive implementation:
# for q in quad_points: <- O(k^d) where d is dimension (here 2)
#   for i in trial_function_points: <- O(k^d)
#     for j in test_function_points: <- O(k^d)
#        output[i, j] += some_expr(i, j, q) <- naively k^{3d} things = k^6
# Optimised version does k^5 (or k^{2d + 1})

plt.figure(figsize=(14,7))
plt.plot(K, new_flops, label='New')
plt.plot(K, old_flops, label='Old')
plt.xlabel("Degree", fontsize=16)
plt.ylabel("Flops", fontsize=16)
