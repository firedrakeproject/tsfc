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

from functools import singledispatch, partial
import weakref

import FIAT
import finat
import ufl


__all__ = ("as_fiat_cell", "create_base_element",
           "create_element", "supported_elements")


supported_elements = {
    # These all map directly to FInAT elements
    "Bernstein": finat.Bernstein,
    "Brezzi-Douglas-Marini": finat.BrezziDouglasMarini,
    "Brezzi-Douglas-Fortin-Marini": finat.BrezziDouglasFortinMarini,
    "Bubble": finat.Bubble,
    "FacetBubble": finat.FacetBubble,
    "Crouzeix-Raviart": finat.CrouzeixRaviart,
    "Discontinuous Lagrange": finat.DiscontinuousLagrange,
    "Discontinuous Raviart-Thomas": lambda c, d: finat.DiscontinuousElement(finat.RaviartThomas(c, d)),
    "Discontinuous Taylor": finat.DiscontinuousTaylor,
    "Gauss-Legendre": finat.GaussLegendre,
    "Gauss-Lobatto-Legendre": finat.GaussLobattoLegendre,
    "HDiv Trace": finat.HDivTrace,
    "Hellan-Herrmann-Johnson": finat.HellanHerrmannJohnson,
    "Nonconforming Arnold-Winther": finat.ArnoldWintherNC,
    "Conforming Arnold-Winther": finat.ArnoldWinther,
    "Hermite": finat.Hermite,
    "Kong-Mulder-Veldhuizen": finat.KongMulderVeldhuizen,
    "Argyris": finat.Argyris,
    "Mardal-Tai-Winther": finat.MardalTaiWinther,
    "Morley": finat.Morley,
    "Bell": finat.Bell,
    "Lagrange": finat.Lagrange,
    "Nedelec 1st kind H(curl)": finat.Nedelec,
    "Nedelec 2nd kind H(curl)": finat.NedelecSecondKind,
    "Raviart-Thomas": finat.RaviartThomas,
    "Regge": finat.Regge,
    "BDMCE": finat.BrezziDouglasMariniCubeEdge,
    "BDMCF": finat.BrezziDouglasMariniCubeFace,
    # These require special treatment below
    "DQ": None,
    "Q": None,
    "RTCE": None,
    "RTCF": None,
    "NCE": None,
    "NCF": None,
    "Real": finat.Real,
    "DPC": finat.DPC,
    "S": finat.Serendipity,
    "SminusF": finat.TrimmedSerendipityFace,
    "SminusDiv": finat.TrimmedSerendipityDiv,
    "SminusE": finat.TrimmedSerendipityEdge,
    "SminusCurl": finat.TrimmedSerendipityCurl,
    "DPC L2": finat.DPC,
    "Discontinuous Lagrange L2": finat.DiscontinuousLagrange,
    "Gauss-Legendre L2": finat.GaussLegendre,
    "DQ L2": None,
    "Direct Serendipity": finat.DirectSerendipity,
}
"""A :class:`.dict` mapping UFL element family names to their
FInAT-equivalent constructors.  If the value is ``None``, the UFL
element is supported, but must be handled specially because it doesn't
have a direct FInAT equivalent."""


def as_fiat_cell(cell):
    """Convert a ufl cell to a FIAT cell.

    :arg cell: the :class:`ufl.Cell` to convert."""
    if not isinstance(cell, ufl.AbstractCell):
        raise ValueError("Expecting a UFL Cell")
    return FIAT.ufc_cell(cell)


@singledispatch
def convert(element, **kwargs):
    """Handler for converting UFL elements to FInAT elements.

    :arg element: The UFL element to convert.

    Do not use this function directly, instead call
    :func:`create_element`."""
    if element.family() in supported_elements:
        raise ValueError("Element %s supported, but no handler provided" % element)
    raise ValueError("Unsupported element type %s" % type(element))


# Base finite elements first
@convert.register(ufl.FiniteElement)
def convert_finiteelement(element, **kwargs):
    cell = as_fiat_cell(element.cell())
    if element.family() == "Quadrature":
        degree = element.degree()
        scheme = element.quadrature_scheme()
        if degree is None or scheme is None:
            raise ValueError("Quadrature scheme and degree must be specified!")

        return finat.make_quadrature_element(cell, degree, scheme), set()
    lmbda = supported_elements[element.family()]
    if element.family() == "Real" and element.cell().cellname() in {"quadrilateral", "hexahedron"}:
        lmbda = None
        element = ufl.FiniteElement("DQ", element.cell(), 0)
    if lmbda is None:
        if element.cell().cellname() == "quadrilateral":
            # Handle quadrilateral short names like RTCF and RTCE.
            element = element.reconstruct(cell=quadrilateral_tpc)
        elif element.cell().cellname() == "hexahedron":
            # Handle hexahedron short names like NCF and NCE.
            element = element.reconstruct(cell=hexahedron_tpc)
        else:
            raise ValueError("%s is supported, but handled incorrectly" %
                             element.family())
        finat_elem, deps = _create_element(element, **kwargs)
        return finat.FlattenedDimensions(finat_elem), deps

    kind = element.variant()
    is_interval = element.cell().cellname() == 'interval'
    if kind is None:
        kind = 'spectral' if is_interval else 'equispaced'  # default variant

    if element.family() == "Lagrange":
        if kind == 'equispaced':
            lmbda = finat.Lagrange
        elif kind == 'spectral' and is_interval:
            lmbda = finat.GaussLobattoLegendre
        elif kind == 'hierarchical' and is_interval:
            lmbda = finat.IntegratedLegendre
        elif kind in ['fdm', 'fdm_ipdg'] and is_interval:
            lmbda = finat.FDMLagrange
        elif kind == 'fdm_quadrature' and is_interval:
            lmbda = finat.FDMQuadrature
        elif kind == 'fdm_broken' and is_interval:
            lmbda = finat.FDMBrokenH1
        elif kind == 'fdm_hermite' and is_interval:
            lmbda = finat.FDMHermite
        elif kind in ['mgd', 'feec', 'qb', 'mse']:
            degree = element.degree()
            shift_axes = kwargs["shift_axes"]
            restriction = kwargs["restriction"]
            deps = {"shift_axes", "restriction"}
            return finat.RuntimeTabulated(cell, degree, variant=kind, shift_axes=shift_axes, restriction=restriction), deps
        else:
            raise ValueError("Variant %r not supported on %s" % (kind, element.cell()))
    elif element.family() in {"Raviart-Thomas", "Nedelec 1st kind H(curl)",
                              "Brezzi-Douglas-Marini", "Nedelec 2nd kind H(curl)"}:
        lmbda = partial(lmbda, variant=element.variant())
    elif element.family() in ["Discontinuous Lagrange", "Discontinuous Lagrange L2"]:
        if kind == 'equispaced':
            lmbda = finat.DiscontinuousLagrange
        elif kind == 'spectral' and is_interval:
            lmbda = finat.GaussLegendre
        elif kind == 'hierarchical' and is_interval:
            lmbda = finat.Legendre
        elif kind in ['fdm', 'fdm_quadrature'] and is_interval:
            lmbda = finat.FDMDiscontinuousLagrange
        elif kind == 'fdm_ipdg' and is_interval:
            lmbda = lambda *args: finat.DiscontinuousElement(finat.FDMLagrange(*args))
        elif kind in 'fdm_broken' and is_interval:
            lmbda = finat.FDMBrokenL2
        elif kind in ['mgd', 'feec', 'qb', 'mse']:
            degree = element.degree()
            shift_axes = kwargs["shift_axes"]
            restriction = kwargs["restriction"]
            deps = {"shift_axes", "restriction"}
            return finat.RuntimeTabulated(cell, degree, variant=kind, shift_axes=shift_axes, restriction=restriction, continuous=False), deps
        else:
            raise ValueError("Variant %r not supported on %s" % (kind, element.cell()))
    elif element.family() == ["DPC", "DPC L2"]:
        if element.cell().geometric_dimension() == 2:
            element = element.reconstruct(cell=ufl.cell.hypercube(2))
        elif element.cell().geometric_dimension() == 3:
            element = element.reconstruct(cell=ufl.cell.hypercube(3))
    elif element.family() == "S":
        if element.cell().geometric_dimension() == 2:
            element = element.reconstruct(cell=ufl.cell.hypercube(2))
        elif element.cell().geometric_dimension() == 3:
            element = element.reconstruct(cell=ufl.cell.hypercube(3))

    return lmbda(cell, element.degree()), set()


# Element modifiers and compound element types
@convert.register(ufl.BrokenElement)
def convert_brokenelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.DiscontinuousElement(finat_elem), deps


@convert.register(ufl.EnrichedElement)
def convert_enrichedelement(element, **kwargs):
    elements, deps = zip(*[_create_element(elem, **kwargs)
                           for elem in element._elements])
    return finat.EnrichedElement(elements), set.union(*deps)


@convert.register(ufl.NodalEnrichedElement)
def convert_nodalenrichedelement(element, **kwargs):
    elements, deps = zip(*[_create_element(elem, **kwargs)
                           for elem in element._elements])
    return finat.NodalEnrichedElement(elements), set.union(*deps)


@convert.register(ufl.MixedElement)
def convert_mixedelement(element, **kwargs):
    elements, deps = zip(*[_create_element(elem, **kwargs)
                           for elem in element.sub_elements()])
    return finat.MixedElement(elements), set.union(*deps)


@convert.register(ufl.VectorElement)
@convert.register(ufl.TensorElement)
def convert_tensorelement(element, **kwargs):
    inner_elem, deps = _create_element(element.sub_elements()[0], **kwargs)
    shape = element.reference_value_shape()
    shape = shape[:len(shape) - len(inner_elem.value_shape)]
    shape_innermost = kwargs["shape_innermost"]
    return (finat.TensorFiniteElement(inner_elem, shape, not shape_innermost),
            deps | {"shape_innermost"})


@convert.register(ufl.TensorProductElement)
def convert_tensorproductelement(element, **kwargs):
    cell = element.cell()
    if type(cell) is not ufl.TensorProductCell:
        raise ValueError("TensorProductElement not on TensorProductCell?")
    shift_axes = kwargs["shift_axes"]
    dim_offset = 0
    elements = []
    deps = set()
    for elem in element.sub_elements():
        kwargs["shift_axes"] = shift_axes + dim_offset
        dim_offset += elem.cell().topological_dimension()
        finat_elem, ds = _create_element(elem, **kwargs)
        elements.append(finat_elem)
        deps.update(ds)
    return finat.TensorProductElement(elements), deps


@convert.register(ufl.HDivElement)
def convert_hdivelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.HDivElement(finat_elem), deps


@convert.register(ufl.HCurlElement)
def convert_hcurlelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.HCurlElement(finat_elem), deps


@convert.register(ufl.WithMapping)
def convert_withmapping(element, **kwargs):
    return _create_element(element.wrapee, **kwargs)


@convert.register(ufl.RestrictedElement)
def convert_restrictedelement(element, **kwargs):
    finat_elem, deps = _create_element(element._element, **kwargs)
    return finat.RestrictedElement(finat_elem, element.restriction_domain()), deps


hexahedron_tpc = ufl.TensorProductCell(ufl.interval, ufl.interval, ufl.interval)
quadrilateral_tpc = ufl.TensorProductCell(ufl.interval, ufl.interval)
_cache = weakref.WeakKeyDictionary()


def create_element(ufl_element, shape_innermost=True, shift_axes=0, restriction=None):
    """Create a FInAT element (suitable for tabulating with) given a UFL element.

    :arg ufl_element: The UFL element to create a FInAT element from.
    :arg shape_innermost: Vector/tensor indices come after basis function indices
    :arg restriction: cell restriction in interior facet integrals
                      (only for runtime tabulated elements)
    """
    finat_element, deps = _create_element(ufl_element,
                                          shape_innermost=shape_innermost,
                                          shift_axes=shift_axes,
                                          restriction=restriction)
    return finat_element


def _create_element(ufl_element, **kwargs):
    """A caching wrapper around :py:func:`convert`.

    Takes a UFL element and an unspecified set of parameter options,
    and returns the converted element with the set of keyword names
    that were relevant for conversion.
    """
    # Look up conversion in cache
    try:
        cache = _cache[ufl_element]
    except KeyError:
        _cache[ufl_element] = {}
        cache = _cache[ufl_element]

    for key, finat_element in cache.items():
        # Cache hit if all relevant parameter values match.
        if all(kwargs[param] == value for param, value in key):
            return finat_element, set(param for param, value in key)

    # Convert if cache miss
    if ufl_element.cell() is None:
        raise ValueError("Don't know how to build element when cell is not given")

    finat_element, deps = convert(ufl_element, **kwargs)

    # Store conversion in cache
    key = frozenset((param, kwargs[param]) for param in deps)
    cache[key] = finat_element

    # Forward result
    return finat_element, deps


def create_base_element(ufl_element, **kwargs):
    """Create a "scalar" base FInAT element given a UFL element.
    Takes a UFL element and an unspecified set of parameter options,
    and returns the converted element.
    """
    finat_element = create_element(ufl_element, **kwargs)
    if isinstance(finat_element, finat.TensorFiniteElement):
        finat_element = finat_element.base_element
    return finat_element
