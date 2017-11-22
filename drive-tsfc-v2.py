from firedrake import *
from tsfc import compile_form
import loopy as lp

mesh = UnitSquareMesh(2,2)
V = FunctionSpace(mesh, "CG", 1)
f = Function(V)
v = TestFunction(V)

L = f*v*dx

x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
# print(assemble(L).vector()[:])
print(f.vector()[:])
exit(0)

kernel, = compile_form(L)
knl = kernel.ast

# knl = lp.to_batched(knl, "nelements", ("A_0", "coords",), batch_iname_prefix="iel")
# knl = lp.tag_inames(knl, "j:ilp.seq")

# for rule in list(six.itervalues(knl.substitutions)):
#     knl = lp.precompute(knl, rule.name, rule.arguments)

print(knl)
code = lp.generate_code_v2(knl)
print(code.device_code())
# lp.auto_test_vs_ref(knl, ctx, knl, parameters={"nelements": 200})
