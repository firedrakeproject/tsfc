from ufl import *
from tsfc import compile_form
import six

cell = triangle
mesh = Mesh(VectorElement('P', cell, 1))
V = FunctionSpace(mesh, FiniteElement('P', cell, 2))
f = Coefficient(V)
v = TestFunction(V)

L = f*v*dx

kernel, = compile_form(L)

print(kernel._ir)

print(kernel.ast)

from tsfc_to_loopy import tsfc_to_loopy
knl = tsfc_to_loopy(kernel._ir)

import pyopencl as cl
import loopy as lp
ctx = cl.create_some_context()

import numpy as np
knl = lp.add_and_infer_dtypes(knl, {"coords,w_0,A0": np.float64})

knl = lp.to_batched(knl, "nelements", ("A0", "coords",), batch_iname_prefix="iel")
#knl = lp.tag_inames(knl, "j:ilp.seq")

for rule in list(six.itervalues(knl.substitutions)):
    knl = lp.precompute(knl, rule.name, rule.arguments)

print(knl)

lp.auto_test_vs_ref(knl, ctx, knl, parameters={"nelements": 200})
