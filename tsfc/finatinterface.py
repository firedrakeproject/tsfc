# -*- coding: utf-8 -*-
#
# This file was modified from FFC
# (http://bitbucket.org/fenics-project/ffc), copyright notice
# reproduced below.
#
# Copyright (C) 2009-2013 Kristian B. Oelgaard and Anders Logg
#
# This file is part of FFC.
#
# FFC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FFC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FFC. If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import, print_function, division
from six import iteritems

from singledispatch import singledispatch
import weakref

from gem.utils import DynamicallyScoped

import finat
from finat.fiat_elements import FiatElementBase

import ufl

from tsfc.fiatinterface import as_fiat_cell
from tsfc.ufl_utils import spanning_degree


__all__ = ("create_element", "supported_elements", "as_fiat_cell")


supported_elements = {
    # These all map directly to FInAT elements
    "Brezzi-Douglas-Marini": finat.BrezziDouglasMarini,
    "Brezzi-Douglas-Fortin-Marini": finat.BrezziDouglasFortinMarini,
    "Discontinuous Lagrange": finat.DiscontinuousLagrange,
    "Discontinuous Raviart-Thomas": finat.DiscontinuousRaviartThomas,
    "Lagrange": finat.Lagrange,
    "Nedelec 1st kind H(curl)": finat.Nedelec,
    "Nedelec 2nd kind H(curl)": finat.NedelecSecondKind,
    "Raviart-Thomas": finat.RaviartThomas,
    "Regge": finat.Regge,
    # These require special treatment below
    "DQ": None,
    "Q": None,
    "RTCE": None,
    "RTCF": None,
}
"""A :class:`.dict` mapping UFL element family names to their
FInAT-equivalent constructors.  If the value is ``None``, the UFL
element is supported, but must be handled specially because it doesn't
have a direct FInAT equivalent."""


class FiatElementWrapper(FiatElementBase):
    def __init__(self, element, degree=None):
        super(FiatElementWrapper, self).__init__(element)
        self._degree = degree

    @property
    def degree(self):
        if self._degree is not None:
            return self._degree
        else:
            return super(FiatElementWrapper, self).degree


def fiat_compat(element):
    from tsfc.fiatinterface import create_element
    return FiatElementWrapper(create_element(element),
                              degree=spanning_degree(element))


@singledispatch
def convert(element):
    """Handler for converting UFL elements to FInAT elements.

    :arg element: The UFL element to convert.

    Do not use this function directly, instead call
    :func:`create_element`."""
    if element.family() in supported_elements:
        raise ValueError("Element %s supported, but no handler provided" % element)
    return fiat_compat(element)


# Base finite elements first
@convert.register(ufl.FiniteElement)
def convert_finiteelement(element):
    cell = as_fiat_cell(element.cell())
    if element.family() == "Quadrature":
        degree = element.degree()
        scheme = element.quadrature_scheme()
        if degree is None or scheme is None:
            raise ValueError("Quadrature scheme and degree must be specified!")

        return finat.QuadratureElement(cell, degree, scheme)
    if element.family() not in supported_elements:
        return fiat_compat(element)
    lmbda = supported_elements.get(element.family())
    if lmbda is None:
        if element.cell().cellname() != "quadrilateral":
            raise ValueError("%s is supported, but handled incorrectly" %
                             element.family())
        # Handle quadrilateral short names like RTCF and RTCE.
        element = element.reconstruct(cell=quad_tpc)
        return finat.QuadrilateralElement(create_element(element))

    kind = element.variant()
    if kind is None:
        kind = 'equispaced'  # default variant

    if element.family() == "Lagrange":
        if kind == 'equispaced':
            lmbda = finat.Lagrange
        elif kind == 'spectral' and element.cell().cellname() == 'interval':
            lmbda = finat.GaussLobattoLegendre
        else:
            raise ValueError("Variant %r not supported on %s" % (kind, element.cell()))
    elif element.family() == "Discontinuous Lagrange":
        kind = element.variant() or 'equispaced'
        if kind == 'equispaced':
            lmbda = finat.DiscontinuousLagrange
        elif kind == 'spectral' and element.cell().cellname() == 'interval':
            lmbda = finat.GaussLegendre
        else:
            raise ValueError("Variant %r not supported on %s" % (kind, element.cell()))
    return lmbda(cell, element.degree())


# EnrichedElement case
@convert.register(ufl.EnrichedElement)
def convert_enrichedelement(element):
    return finat.EnrichedElement([create_element(elem) for elem in element._elements])


# Generic MixedElement case
@convert.register(ufl.MixedElement)
def convert_mixedelement(element):
    return finat.MixedElement([create_element(elem) for elem in element.sub_elements()])


# VectorElement case
@convert.register(ufl.VectorElement)
def convert_vectorelement(element):
    scalar_element = create_element(element.sub_elements()[0])
    collecting_deps.value[shape_innermost] = shape_innermost.value
    return finat.TensorFiniteElement(scalar_element,
                                     (element.num_sub_elements(),),
                                     transpose=not shape_innermost.value)


# TensorElement case
@convert.register(ufl.TensorElement)
def convert_tensorelement(element):
    scalar_element = create_element(element.sub_elements()[0])
    collecting_deps.value[shape_innermost] = shape_innermost.value
    return finat.TensorFiniteElement(scalar_element,
                                     element.reference_value_shape(),
                                     transpose=not shape_innermost.value)


# TensorProductElement case
@convert.register(ufl.TensorProductElement)
def convert_tensorproductelement(element):
    cell = element.cell()
    if type(cell) is not ufl.TensorProductCell:
        raise ValueError("TensorProductElement not on TensorProductCell?")
    return finat.TensorProductElement([create_element(elem)
                                       for elem in element.sub_elements()])


# HDivElement case
@convert.register(ufl.HDivElement)
def convert_hdivelement(element):
    return finat.HDivElement(create_element(element._element))


# HDivElement case
@convert.register(ufl.HCurlElement)
def convert_hcurlelement(element):
    return finat.HCurlElement(create_element(element._element))


quad_tpc = ufl.TensorProductCell(ufl.interval, ufl.interval)
_cache = weakref.WeakKeyDictionary()

shape_innermost = DynamicallyScoped(True)
"""Relevant for vector/tensor elements: tensor shape indices come
after scalar basis function indices when True, i.e. use the
Firedrake-style XYZ XYZ XYZ XYZ DoF ordering instead of the
FEniCS-style XXXX YYYY ZZZZ.
"""

collecting_deps = DynamicallyScoped()
"""Runtime dependencies with keys that were employed during element
conversion, thus must be part of the cache key."""


def create_element(ufl_element):
    """Create a FInAT element (suitable for tabulating with) given a UFL element.

    :arg ufl_element: The UFL element to create a FInAT element from.
    """
    try:
        cache = _cache[ufl_element]
    except KeyError:
        _cache[ufl_element] = {}
        cache = _cache[ufl_element]

    config = frozenset((variable, variable.value)
                       for variable in [shape_innermost])
    for deps, finat_element in iteritems(cache):
        if deps <= config:
            return finat_element

    if ufl_element.cell() is None:
        raise ValueError("Don't know how to build element when cell is not given")

    with collecting_deps.let({}):
        finat_element = convert(ufl_element)
        current_deps = frozenset(iteritems(collecting_deps.value))
    cache[current_deps] = finat_element
    return finat_element
