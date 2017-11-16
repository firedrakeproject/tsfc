from firedrake import *
from tsfc import compile_form
from tsfc.loopy import generate as generate_loopy
from tsfc import parameters
import numpy as np
import loopy as lp

mesh = UnitSquareMesh(2,2)
V = FunctionSpace(mesh, "CG", 1)
f = Function(V)
v = TestFunction(V)

L = f*v*dx

kernel, = compile_form(L)
impero_c = kernel._impero_c

knl = generate_loopy(impero_c, parameters.default_parameters()['precision'])

knl = lp.add_and_infer_dtypes(knl, {"coords, w_0, t1, t2, t3, t5": np.float64})

# knl = lp.to_batched(knl, "nelements", ("A_0", "coords",), batch_iname_prefix="iel")
# knl = lp.tag_inames(knl, "j:ilp.seq")

# for rule in list(six.itervalues(knl.substitutions)):
#     knl = lp.precompute(knl, rule.name, rule.arguments)

print(knl)
code = lp.generate_code_v2(knl)
print(code.device_code())
# lp.auto_test_vs_ref(knl, ctx, knl, parameters={"nelements": 200})
