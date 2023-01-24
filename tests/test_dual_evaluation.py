import pytest
import ufl
from tsfc.finatinterface import create_element
from tsfc import compile_expression_dual_evaluation


def test_ufl_only_simple():
    mesh = ufl.Mesh(ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, ufl.FiniteElement("P", ufl.triangle, 2))
    v = ufl.Coefficient(V)
    expr = ufl.inner(v, v)
    W = V
    to_element = create_element(W.ufl_element())
    kernel = compile_expression_dual_evaluation(expr, to_element, W.ufl_element())
    assert kernel.needs_external_coords is False


def test_ufl_only_spatialcoordinate():
    mesh = ufl.Mesh(ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, ufl.FiniteElement("P", ufl.triangle, 2))
    x, y = ufl.SpatialCoordinate(mesh)
    expr = x*y - y**2 + x
    W = V
    to_element = create_element(W.ufl_element())
    kernel = compile_expression_dual_evaluation(expr, to_element, W.ufl_element())
    assert kernel.needs_external_coords is True


def test_ufl_only_from_contravariant_piola():
    mesh = ufl.Mesh(ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, ufl.FiniteElement("RT", ufl.triangle, 1))
    v = ufl.Coefficient(V)
    expr = ufl.inner(v, v)
    W = ufl.FunctionSpace(mesh, ufl.FiniteElement("P", ufl.triangle, 2))
    to_element = create_element(W.ufl_element())
    kernel = compile_expression_dual_evaluation(expr, to_element, W.ufl_element())
    assert kernel.needs_external_coords is True


def test_ufl_only_to_contravariant_piola():
    mesh = ufl.Mesh(ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, ufl.FiniteElement("P", ufl.triangle, 2))
    v = ufl.Coefficient(V)
    expr = ufl.as_vector([v, v])
    W = ufl.FunctionSpace(mesh, ufl.FiniteElement("RT", ufl.triangle, 1))
    to_element = create_element(W.ufl_element())
    kernel = compile_expression_dual_evaluation(expr, to_element, W.ufl_element())
    assert kernel.needs_external_coords is True


def test_ufl_only_shape_mismatch():
    mesh = ufl.Mesh(ufl.VectorElement("P", ufl.triangle, 1))
    V = ufl.FunctionSpace(mesh, ufl.FiniteElement("RT", ufl.triangle, 1))
    v = ufl.Coefficient(V)
    expr = ufl.inner(v, v)
    assert expr.ufl_shape == ()
    W = V
    to_element = create_element(W.ufl_element())
    assert to_element.value_shape == (2,)
    with pytest.raises(ValueError):
        compile_expression_dual_evaluation(expr, to_element, W.ufl_element())


def test_interpolate_zany_argument():
    mesh = ufl.Mesh(ufl.VectorElement("P", ufl.triangle, 2))
    V = ufl.FunctionSpace(mesh, ufl.FiniteElement("P", ufl.triangle, 1))
    Q = ufl.FunctionSpace(mesh, ufl.FiniteElement("Argyris", ufl.triangle, 5))
    expr = ufl.TestFunction(Q)
    to_element = create_element(V.ufl_element())
    kernel = compile_expression_dual_evaluation(expr, to_element, V.ufl_element())
    assert kernel.needs_external_coords
