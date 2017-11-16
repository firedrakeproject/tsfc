from gem.gem import Index, Indexed, Product, Variable, Division, Literal, Sum, IndexSum
from gem.impero_utils import compile_gem
from tsfc.loopy import generate as generate_loopy
from tsfc import parameters
import loopy as lp
import numpy as np

I = 20
J = K = 10
i = Index('i', I)
j = Index('j', J)
k = Index('k', K)

A = Variable('a', (J,))
Aj = Indexed(A, (j,))

B = Variable('b', (J,))
C = Variable('c', (J,))
Bj = Indexed(B, (j,))
Cj = Indexed(C, (j,))

E = Variable('e', (K,))
F = Variable('f', (K,))
G = Variable('g', (K,))
Ek = Indexed(E, (k,))
Fk = Indexed(F, (k,))
Gk = Indexed(G, (k,))

H = Variable('h', (I, J))
Hij = Indexed(H, (i,j))

# Bj*Cj
lhs = Aj
# rhs = Product(Bj, Cj)
rhs = IndexSum(Hij, (i,))

assignments = [(lhs, rhs)]
index_ordering = (i,)
impero_c = compile_gem(assignments, index_ordering)

knl = generate_loopy(impero_c, parameters.default_parameters()['precision'])

# from IPython import embed; embed()
# knl = lp.add_and_infer_dtypes(knl, {"a, b, c": np.float64})
knl = lp.add_and_infer_dtypes(knl, {"a, h": np.float64})
print(knl)
code = lp.generate_code_v2(knl)
print(code.device_code())

