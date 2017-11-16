from __future__ import division, absolute_import, print_function

from ufl import (
    triangle, Mesh, VectorElement, FiniteElement, Coefficient,
    TestFunction, FunctionSpace, dx)

from tsfc.tsfc_to_loopy import tsfc_to_loopy
from tsfc import compile_form

import six

# import pyopencl as cl
import loopy as lp

import numpy as np

cell = triangle
mesh = Mesh(VectorElement('P', cell, 1))
V = FunctionSpace(mesh, FiniteElement('P', cell, 2))
f = Coefficient(V)
v = TestFunction(V)

L = f*v*dx

kernel, = compile_form(L)

# print(kernel._ir)
print(kernel.ast)

knl = tsfc_to_loopy(kernel._ir, kernel._ir[0][0].free_indices)

# ctx = cl.create_some_context()

knl = lp.add_and_infer_dtypes(knl, {"coords,w_0,A_0": np.float64})

# knl = lp.to_batched(knl, "nelements", ("A_0", "coords",), batch_iname_prefix="iel")
# knl = lp.tag_inames(knl, "j:ilp.seq")

for rule in list(six.itervalues(knl.substitutions)):
    knl = lp.precompute(knl, rule.name, rule.arguments)

print(knl)
code = lp.generate_code_v2(knl)
print(code.device_code())
# lp.auto_test_vs_ref(knl, ctx, knl, parameters={"nelements": 200})

