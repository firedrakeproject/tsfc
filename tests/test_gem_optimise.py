from __future__ import absolute_import, print_function, division

import pytest

from gem.gem import Index, Indexed, Product, Variable, Division, Literal, Sum
from gem.optimise import replace_division, reassociate_product


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


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
