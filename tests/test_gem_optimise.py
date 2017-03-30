from __future__ import absolute_import, print_function, division

import pytest

from gem.gem import Index, Indexed, Product, Variable, Division, Literal, Sum
from gem.optimise import replace_division, reassociate_product, optimise, count_flop


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

        (D[i,j,k] + Q[i,j]*(t*P[i]))*(C[i,j]*(A*B[i]))

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
    assert isinstance(result.children[0], Sum)
    # C[i,j]
    assert result.children[1].children[0] == Cij
    # A * B[i]
    assert result.children[1].children[1] == Product(A, Bi)
    # t * P[i]
    assert result.children[0].children[1].children[1] == Product(t, Pi)


def test_loop_optimise():
    I = 20
    J = K = 10
    i = Index('i', I)
    j = Index('j', J)
    k = Index('k', K)

    A1 = Variable('a1', (I,))
    A2 = Variable('a2', (I,))
    A3 = Variable('a3', (I,))
    A1i = Indexed(A1, (i,))
    A2i = Indexed(A2, (i,))
    A3i = Indexed(A3, (i,))

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

    Z = Variable('z', ())

    # Test Bj*Ek + Bj*Fk => Bj*(Ek + Fk)
    expr = Sum(Product(Bj, Ek), Product(Bj, Fk))
    result = optimise(expr, ((i,),), ((j,), (k,)))
    assert count_flop(result) == 110

    # Test that all common factors from optimal factors are applied
    # Bj*Ek + Bj*Fk + Bj*Gk + Cj*Ek + Cj*Fk =>
    # Bj*(Ek + Fk + Gk) + Cj*(Ek+Fk)
    expr = Sum(Sum(Sum(Sum(Product(Bj, Ek), Product(Bj, Fk)), Product(Bj, Gk)),
                   Product(Cj, Ek)), Product(Cj, Fk))
    result = optimise(expr, ((i,),), ((j,), (k,)))
    assert count_flop(result) == 320

    # Test that consts are factorised
    # Z*A1i*Bj*Ek + Z*A2i*Bj*Ek + A3i*Bj*Ek + Z*A1i*Bj =>
    # Bj*(Ek*(Z*A1i + Z*A2i) + A3i) + Z*A1i)
    # Note, constant factorisation (Z in this case) not implemented yet

    expr = Sum(Sum(Sum(Product(Z, Product(A1i, Product(Bj, Ek))),
                       Product(Z, Product(A2i, Product(Bj, Ek)))),
                   Product(A3i, Product(Bj, Ek))), Product(Z, Product(A1i, Bj)))
    result = optimise(expr, ((i,),), ((j,), (k,)))
    assert count_flop(result) == 2480


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
