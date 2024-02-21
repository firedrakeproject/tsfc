import numpy
import pytest

from ufl import (Mesh, FunctionSpace, TestFunction, TrialFunction, TensorProductCell, dx,
                 action, interval, quadrilateral, dot, grad)
from finat.ufl import FiniteElement, VectorElement

from tsfc import compile_form


def mass_cg(cell, degree):
    m = Mesh(VectorElement('Q', cell, 1))
    V = FunctionSpace(m, FiniteElement('Q', cell, degree, variant='spectral'))
    u = TrialFunction(V)
    v = TestFunction(V)
    return u*v*dx(scheme="gll", degree=2*degree-1)


def mass_dg(cell, degree):
    m = Mesh(VectorElement('Q', cell, 1))
    V = FunctionSpace(m, FiniteElement('DQ', cell, degree, variant='spectral'))
    u = TrialFunction(V)
    v = TestFunction(V)
    return u*v*dx(scheme="gl", degree=2*degree)


def laplace(cell, degree):
    m = Mesh(VectorElement('Q', cell, 1))
    V = FunctionSpace(m, FiniteElement('Q', cell, degree, variant='spectral'))
    u = TrialFunction(V)
    v = TestFunction(V)
    return dot(grad(u), grad(v))*dx(scheme="gll", degree=2*degree-1)


def count_flops(form):
    kernel, = compile_form(form, parameters=dict(mode='spectral'))
    return kernel.flop_count


@pytest.mark.parametrize('form', [mass_cg, mass_dg])
@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 2),
                          (TensorProductCell(interval, interval), 2),
                          (TensorProductCell(quadrilateral, interval), 3)])
def test_mass(form, cell, order):
    degrees = numpy.arange(4, 10)
    flops = [count_flops(form(cell, int(degree))) for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees + 1))
    assert (rates < order).all()


@pytest.mark.parametrize('form', [mass_cg, mass_dg])
@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 2),
                          (TensorProductCell(interval, interval), 2),
                          (TensorProductCell(quadrilateral, interval), 3)])
def test_mass_action(form, cell, order):
    degrees = numpy.arange(4, 10)
    flops = [count_flops(action(form(cell, int(degree)))) for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees + 1))
    assert (rates < order).all()


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 4),
                          (TensorProductCell(interval, interval), 4),
                          (TensorProductCell(quadrilateral, interval), 5)])
def test_laplace(cell, order):
    degrees = numpy.arange(4, 10)
    flops = [count_flops(laplace(cell, int(degree))) for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees + 1))
    assert (rates < order).all()


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
