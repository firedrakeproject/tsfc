from __future__ import absolute_import, print_function, division

import pytest

from gem.gem import Index, Indexed, Product, Variable, Division, Literal, Sum, IndexSum
from gem.optimise import replace_division, reassociate_product, factorise, count_flop


def test_replace_div():
    i = Index()
    A = Variable('A', ())
    B = Variable('B', (6,))
    Bi = Indexed(B, (i,))
    d = Division(Bi, A)
    result = replace_division([d])[0]

    assert isinstance(result, Product)
    assert isinstance(result.children[1], Division)


def test_reassociate_product():
    """Test recursive reassociation of products according to the rank of the
    factors. For example: ::

        C[i,j]*(D[i,j,k] + Q[i,j]*t*P[i])*A*B[i]

    becomes: ::

        A*B[i]*C[i,j]*(D[i,j,k] + t*P[i]*Q[i,j])

    """
    i = Index()
    j = Index()
    k = Index()
    A = Variable('A', ())
    Bi = Indexed(Variable('B', (6,)), (i,))
    Cij = Indexed(Variable('C', (6, 6)), (i, j))
    Dijk = Indexed(Variable('D', (6, 6, 6)), (i, j, k))
    Pi = Indexed(Variable('P', (6,)), (i,))
    Qij = Indexed(Variable('Q', (6, 6)), (i, j))
    t = Literal(8)
    p = Product(Product(Product(Cij, Sum(Dijk, Product(Product(Qij, t), Pi))), A), Bi)
    result = reassociate_product([p])[0]
    # D[i,j,k] + t*P[i]*Q[i,j]
    assert isinstance(result.children[1], Sum)
    # t
    assert result.children[1].children[1].children[0].children[0] == t
    # Q[i,j]
    assert result.children[1].children[1].children[1] == Qij
    # A
    assert result.children[0].children[0].children[0] == A
    # B[i]
    assert result.children[0].children[0].children[1] == Bi
    # C[i,j]
    assert result.children[0].children[1] == Cij


def test_factorise_1():
    I = J = 10
    i = Index('i', I)
    j = Index('j', J)
    B = Variable('b', (J,))
    C = Variable('c', (I,))
    D = Variable('d', (I,))
    E = Variable('e', (I,))
    Bj = Indexed(B, (j,))
    Ci = Indexed(C, (i,))
    Di = Indexed(D, (i,))
    Ei = Indexed(E, (i,))
    S = Sum(Product(Bj, Ci), Product(Bj, Di))
    expr = IndexSum(S, (i,))
    result = factorise(expr)
    assert count_flop(result) == 110


# def test_factorise():
#     """Test factorising a summation. For example: ::
#
#         A[i] + A[i]*B[j] + A[i]*(C[j]+D[j]) + A[i]*E[i]*F[i] + G[i]
#
#     becomes: ::
#
#         A[i] * (1 + B[j] + C[j] + D[j] + E[i]*F[i]) + G[i]
#
#     """
#     i = Index()
#     j = Index()
#     A = Variable('A', (6,))
#     B = Variable('B', (6,))
#     C = Variable('C', (6,))
#     D = Variable('D', (6,))
#     E = Variable('E', (6,))
#     F = Variable('F', (6,))
#     G = Variable('G', (6,))
#     Ai = Indexed(A, (i,))
#     Bj = Indexed(B, (j,))
#     Cj = Indexed(C, (j,))
#     Dj = Indexed(D, (j,))
#     Ei = Indexed(E, (i,))
#     Fi = Indexed(F, (i,))
#     Gi = Indexed(G, (i,))
#     p1 = Product(Ai, Bj)
#     p2 = Product(Ai, Sum(Cj, Dj))
#     p3 = Product(Ei, Fi)
#     s = Sum(Sum(Sum(Sum(p1, p2), Product(Ai, p3)), Ai), Gi)
#     result = factorise(s)
#     # G[i]
#     assert result.children[1] == Gi
#     # A[i]
#     assert result.children[0].children[0] == Ai
#     # 1
#     assert result.children[0].children[1].children[1] == Literal(1)
#     # E[i]*F[i]
#     assert result.children[0].children[1].children[0].children[1] == p3
#     # B[j]+C[j]+D[j]
#     assert result.children[0].children[1].children[0].children[0] == Sum(Bj, Sum(Cj, Dj))
#
#
# def test_factorise_recursion():
#     """Test recursive factorisation. For example: ::
#
#         A[i]*C[i] + A[i]*D[i] + B[i]*C[i] + B[i]*D[i]
#
#     becomes: ::
#
#         (A[i]+B[i]) * (C[i]+D[i])
#
#     """
#     i = Index()
#     A = Variable('A', (6,))
#     B = Variable('B', (6,))
#     C = Variable('C', (6,))
#     D = Variable('D', (6,))
#     Ai = Indexed(A, (i,))
#     Bi = Indexed(B, (i,))
#     Ci = Indexed(C, (i,))
#     Di = Indexed(D, (i,))
#     p1 = Product(Ai, Ci)
#     p2 = Product(Ai, Di)
#     p3 = Product(Bi, Ci)
#     p4 = Product(Bi, Di)
#     s = reduce(Sum, [p1, p2, p3, p4])
#     result = factorise(s)
#     assert isinstance(result, Product)
#     assert result.children[0] == Sum(Ci, Di)
#     assert result.children[1] == Sum(Bi, Ai)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
