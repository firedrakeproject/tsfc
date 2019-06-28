import pytest

import FIAT
from FIAT.discontinuous_lagrange import HigherOrderDiscontinuousLagrange as FIAT_DiscontinuousLagrange

import ufl
from tsfc.fiatinterface import create_element, supported_elements


@pytest.fixture(params=["BDM",
                        "BDFM",
                        "DRT",
                        "Lagrange",
                        "N1curl",
                        "N2curl",
                        "RT",
                        "Regge"])
def triangle_names(request):
    return request.param


@pytest.fixture
def ufl_element(triangle_names):
    return ufl.FiniteElement(triangle_names, ufl.triangle, 2)


def test_triangle_basic(ufl_element):
    element = create_element(ufl_element)
    assert isinstance(element, supported_elements[ufl_element.family()])


@pytest.fixture(params=["CG", "DG", "DG L2"], scope="module")
def tensor_name(request):
    return request.param


@pytest.fixture(params=[ufl.interval, ufl.triangle,
                        ufl.quadrilateral],
                ids=lambda x: x.cellname(),
                scope="module")
def ufl_A(request, tensor_name):
    return ufl.FiniteElement(tensor_name, request.param, 1)


@pytest.fixture
def ufl_B(tensor_name):
    return ufl.FiniteElement(tensor_name, ufl.interval, 1)


def test_tensor_prod_simple(ufl_A, ufl_B):
    tensor_ufl = ufl.TensorProductElement(ufl_A, ufl_B)

    tensor = create_element(tensor_ufl)
    A = create_element(ufl_A)
    B = create_element(ufl_B)

    assert isinstance(tensor, FIAT.TensorProductElement)

    assert tensor.A is A
    assert tensor.B is B


@pytest.mark.parametrize(('family', 'expected_cls'),
                         [('P', FIAT.Lagrange),
                          ('DP', FIAT_DiscontinuousLagrange),
                          ('DP L2', FIAT_DiscontinuousLagrange)])
def test_interval_variant_default(family, expected_cls):
    ufl_element = ufl.FiniteElement(family, ufl.interval, 3)
    assert isinstance(create_element(ufl_element), expected_cls)


@pytest.mark.parametrize(('family', 'variant', 'expected_cls'),
                         [('P', 'equispaced', FIAT.Lagrange),
                          ('P', 'spectral', FIAT.GaussLobattoLegendre),
                          ('P', 'mse', FIAT.GaussLobattoLegendre),
                          ('P', 'dualmse', FIAT.ExtendedGaussLegendre),
                          ('DP', 'equispaced', FIAT_DiscontinuousLagrange),
                          ('DP', 'spectral', FIAT.GaussLegendre),
                          ('DP', 'mse', FIAT.EdgeGaussLobattoLegendre),
                          ('DP', 'dualmse', FIAT.EdgeExtendedGaussLegendre),
                          ('DP L2', 'equispaced', FIAT_DiscontinuousLagrange),
                          ('DP L2', 'spectral', FIAT.GaussLegendre),
                          ('DP L2', 'mse', FIAT.EdgeGaussLobattoLegendre),
                          ('DP L2', 'dualmse', FIAT.EdgeExtendedGaussLegendre)])
def test_interval_variant(family, variant, expected_cls):
    ufl_element = ufl.FiniteElement(family, ufl.interval, 3, variant=variant)
    assert isinstance(create_element(ufl_element), expected_cls)


def test_triangle_variant_spectral_fail():
    ufl_element = ufl.FiniteElement('DP', ufl.triangle, 2, variant='spectral')
    with pytest.raises(ValueError):
        create_element(ufl_element)


def test_triangle_variant_spectral_fail_l2():
    ufl_element = ufl.FiniteElement('DP L2', ufl.triangle, 2, variant='spectral')
    with pytest.raises(ValueError):
        create_element(ufl_element)


def test_quadrilateral_variant_spectral_q():
    element = create_element(ufl.FiniteElement('Q', ufl.quadrilateral, 3, variant='spectral'))
    assert isinstance(element.element.A, FIAT.GaussLobattoLegendre)
    assert isinstance(element.element.B, FIAT.GaussLobattoLegendre)


def test_quadrilateral_variant_mse_q():
    element = create_element(ufl.FiniteElement('Q', ufl.quadrilateral, 3, variant='mse'))
    assert isinstance(element.element.A, FIAT.GaussLobattoLegendre)
    assert isinstance(element.element.B, FIAT.GaussLobattoLegendre)


def test_quadrilateral_variant_dualmse_q():
    element = create_element(ufl.FiniteElement('Q', ufl.quadrilateral, 3, variant='dualmse'))
    assert isinstance(element.element.A, FIAT.ExtendedGaussLegendre)
    assert isinstance(element.element.B, FIAT.ExtendedGaussLegendre)


def test_quadrilateral_variant_spectral_dq():
    element = create_element(ufl.FiniteElement('DQ', ufl.quadrilateral, 1, variant='spectral'))
    assert isinstance(element.element.A, FIAT.GaussLegendre)
    assert isinstance(element.element.B, FIAT.GaussLegendre)


def test_quadrilateral_variant_spectral_dq_l2():
    element = create_element(ufl.FiniteElement('DQ L2', ufl.quadrilateral, 1, variant='spectral'))
    assert isinstance(element.element.A, FIAT.GaussLegendre)
    assert isinstance(element.element.B, FIAT.GaussLegendre)


def test_quadrilateral_variant_spectral_rtcf():
    element = create_element(ufl.FiniteElement('RTCF', ufl.quadrilateral, 2, variant='spectral'))
    assert isinstance(element.element._elements[0].A, FIAT.GaussLobattoLegendre)
    assert isinstance(element.element._elements[0].B, FIAT.GaussLegendre)
    assert isinstance(element.element._elements[1].A, FIAT.GaussLegendre)
    assert isinstance(element.element._elements[1].B, FIAT.GaussLobattoLegendre)


def test_quadrilateral_variant_spectral_rtce():
    element = create_element(ufl.FiniteElement('RTCE', ufl.quadrilateral, 2, variant='spectral'))
    assert isinstance(element.element._elements[0].A, FIAT.GaussLobattoLegendre)
    assert isinstance(element.element._elements[0].B, FIAT.GaussLegendre)
    assert isinstance(element.element._elements[1].A, FIAT.GaussLegendre)
    assert isinstance(element.element._elements[1].B, FIAT.GaussLobattoLegendre)


def test_cache_hit(ufl_element):
    A = create_element(ufl_element)
    B = create_element(ufl_element)

    assert A is B


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
