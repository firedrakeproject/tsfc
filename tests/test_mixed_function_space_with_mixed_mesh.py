from tsfc import compile_form
from ufl import (triangle, Mesh, MixedMesh, FunctionSpace, TestFunction, TrialFunction, Coefficient,
                 Measure, SpatialCoordinate, inner, grad, curl, div, split, as_vector, )
from finat.ufl import FiniteElement, MixedElement, VectorElement
from tsfc.ufl_utils import compute_form_data
from tsfc import kernel_args


def test_mixed_function_space_with_mixed_mesh_restrictions_base():
    cell = triangle
    elem0 = FiniteElement("Discontinuous Lagrange", cell, 2)
    elem1 = FiniteElement("Discontinuous Lagrange", cell, 3)
    elem = MixedElement([elem0, elem1])
    mesh0 = Mesh(VectorElement("Lagrange", cell, 1), ufl_id=100)
    mesh1 = Mesh(VectorElement("Lagrange", cell, 1), ufl_id=101)
    domain = MixedMesh([mesh0, mesh1])
    V = FunctionSpace(domain, elem)
    V0 = FunctionSpace(mesh0, elem0)
    V1 = FunctionSpace(mesh1, elem1)
    f = Coefficient(V, count=1000)
    f0, f1 = split(f)
    u1 = TrialFunction(V1)
    v0 = TestFunction(V0)
    dx1 = Measure("dx", mesh1)
    ds1 = Measure("ds", mesh1)
    dS0 = Measure("dS", mesh0)
    f0_split = Coefficient(V0)
    f1_split = Coefficient(V1)
    # a
    form = inner(grad(f1('|')), as_vector([1, 0])) * ds1(777)
    form_data = compute_form_data(form, do_split_coefficients={f: [f0_split, f1_split]})
    integral_data, = form_data.integral_data
    assert len(integral_data.domain_integral_type_map) == 1
    assert integral_data.domain_integral_type_map[mesh1] == "exterior_facet"
    # b
    form = inner(grad(f1('|')), grad(f1('|'))) * dS0(777)
    form_data = compute_form_data(form, do_split_coefficients={f: [f0_split, f1_split]})
    integral_data, = form_data.integral_data
    assert len(integral_data.domain_integral_type_map) == 2
    assert integral_data.domain_integral_type_map[mesh0] == "interior_facet"
    assert integral_data.domain_integral_type_map[mesh1] == "exterior_facet"
    # c
    form = div(f) * inner(grad(f1), grad(f1)) * inner(grad(u1), grad(v0)) * dx1
    form_data = compute_form_data(form, do_split_coefficients={f: [f0_split, f1_split]})
    integral_data, = form_data.integral_data
    assert len(integral_data.domain_integral_type_map) == 2
    assert integral_data.domain_integral_type_map[mesh0] == "cell"
    assert integral_data.domain_integral_type_map[mesh1] == "cell"


def test_mixed_function_space_with_mixed_mesh_3_cg3_bdm3_dg2_dx1():
    cell = triangle
    gdim = 2
    elem0 = FiniteElement("Lagrange", cell, 3)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 3)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 2)
    elem = MixedElement([elem0, elem1, elem2])
    mesh0 = Mesh(VectorElement("Lagrange", cell, 1), ufl_id=100)
    mesh1 = Mesh(VectorElement("Lagrange", cell, 1), ufl_id=101)
    mesh2 = Mesh(VectorElement("Lagrange", cell, 1), ufl_id=102)
    domain = MixedMesh([mesh0, mesh1, mesh2])
    V = FunctionSpace(domain, elem)
    V0 = FunctionSpace(mesh0, elem0)
    V1 = FunctionSpace(mesh1, elem1)
    V2 = FunctionSpace(mesh2, elem2)
    f = Coefficient(V, count=1000)
    u0 = TrialFunction(V0)
    v1 = TestFunction(V1)
    f0, f1, f2 = split(f)
    f0_split = Coefficient(V0)
    f1_split = Coefficient(V1)
    f2_split = Coefficient(V2)
    x2 = SpatialCoordinate(mesh2)
    dx1 = Measure("dx", mesh1)
    form = inner(x2, x2) * f2 * inner(grad(u0), v1) * dx1(999)
    form_data = compute_form_data(form, do_split_coefficients={f: [f0_split, f1_split, f2_split]})
    integral_data, = form_data.integral_data
    assert len(integral_data.domain_integral_type_map) == 3
    assert integral_data.domain_integral_type_map[mesh0] == "cell"
    assert integral_data.domain_integral_type_map[mesh1] == "cell"
    assert integral_data.domain_integral_type_map[mesh2] == "cell"
    kernel, = compile_form(form)
    assert kernel.domain_number == 0
    assert kernel.integral_type == "cell"
    assert kernel.subdomain_id == (999, )
    assert kernel.active_domain_numbers.coordinates == (0, 1, 2)
    assert kernel.active_domain_numbers.cell_orientations == ()
    assert kernel.active_domain_numbers.cell_sizes == ()
    assert kernel.active_domain_numbers.exterior_facets == ()
    assert kernel.active_domain_numbers.interior_facets == ()
    assert kernel.coefficient_numbers == ((0, (2, )), )
    assert isinstance(kernel.arguments[0], kernel_args.OutputKernelArg)
    assert isinstance(kernel.arguments[1], kernel_args.CoordinatesKernelArg)
    assert isinstance(kernel.arguments[2], kernel_args.CoordinatesKernelArg)
    assert isinstance(kernel.arguments[3], kernel_args.CoordinatesKernelArg)
    assert isinstance(kernel.arguments[4], kernel_args.CoefficientKernelArg)
    assert kernel.arguments[0].loopy_arg.shape == (20, 10)
    assert kernel.arguments[1].loopy_arg.shape == (3 * gdim, )
    assert kernel.arguments[2].loopy_arg.shape == (3 * gdim, )
    assert kernel.arguments[3].loopy_arg.shape == (3 * gdim, )
    assert kernel.arguments[4].loopy_arg.shape == (6, )


def test_mixed_function_space_with_mixed_mesh_restrictions_bdm3_dg2_dS0():
    cell = triangle
    gdim = 2
    elem0 = FiniteElement("Brezzi-Douglas-Marini", cell, 3)
    elem1 = FiniteElement("Discontinuous Lagrange", cell, 2)
    elem = MixedElement([elem0, elem1])
    mesh0 = Mesh(VectorElement("Lagrange", cell, 1), ufl_id=100)
    mesh1 = Mesh(VectorElement("Lagrange", cell, 1), ufl_id=101)
    domain = MixedMesh([mesh0, mesh1])
    V = FunctionSpace(domain, elem)
    V0 = FunctionSpace(mesh0, elem0)
    V1 = FunctionSpace(mesh1, elem1)
    f = Coefficient(V, count=1000)
    f0, f1 = split(f)
    f0_split = Coefficient(V0)
    f1_split = Coefficient(V1)
    u1 = TrialFunction(V1)
    v0 = TestFunction(V0)
    dS0 = Measure("dS", mesh0)
    form = inner(curl(f1('|')), curl(f1('|'))) * inner(grad(u1('|')), v0('+')) * dS0(777)
    form_data = compute_form_data(form, do_split_coefficients={f: [f0_split, f1_split]})
    integral_data, = form_data.integral_data
    assert len(integral_data.domain_integral_type_map) == 2
    assert integral_data.domain_integral_type_map[mesh0] == "interior_facet"
    assert integral_data.domain_integral_type_map[mesh1] == "exterior_facet"
    kernel, = compile_form(form)
    assert kernel.domain_number == 0
    assert kernel.integral_type == "interior_facet"
    assert kernel.subdomain_id == (777, )
    assert kernel.active_domain_numbers.coordinates == (0, 1)
    assert kernel.active_domain_numbers.cell_orientations == ()
    assert kernel.active_domain_numbers.cell_sizes == ()
    assert kernel.active_domain_numbers.exterior_facets == (1, )
    assert kernel.active_domain_numbers.interior_facets == (0, )
    assert kernel.coefficient_numbers == ((0, (1, )), )
    assert isinstance(kernel.arguments[0], kernel_args.OutputKernelArg)
    assert isinstance(kernel.arguments[1], kernel_args.CoordinatesKernelArg)
    assert isinstance(kernel.arguments[2], kernel_args.CoordinatesKernelArg)
    assert isinstance(kernel.arguments[3], kernel_args.CoefficientKernelArg)
    assert isinstance(kernel.arguments[4], kernel_args.ExteriorFacetKernelArg)
    assert isinstance(kernel.arguments[5], kernel_args.InteriorFacetKernelArg)
    assert kernel.arguments[0].loopy_arg.shape == (2 * 20, 6)
    assert kernel.arguments[1].loopy_arg.shape == (2 * (3 * gdim), )
    assert kernel.arguments[2].loopy_arg.shape == (3 * gdim, )
    assert kernel.arguments[3].loopy_arg.shape == (6, )
    assert kernel.arguments[4].loopy_arg.shape == (1, )
    assert kernel.arguments[5].loopy_arg.shape == (2, )


def test_mixed_function_space_with_mixed_mesh_restrictions_dg2_dg3_ds1():
    cell = triangle
    gdim = 2
    elem0 = FiniteElement("Discontinuous Lagrange", cell, 2)
    elem1 = FiniteElement("Discontinuous Lagrange", cell, 3)
    elem = MixedElement([elem0, elem1])
    mesh0 = Mesh(VectorElement("Lagrange", cell, 1), ufl_id=100)
    mesh1 = Mesh(VectorElement("Lagrange", cell, 1), ufl_id=101)
    domain = MixedMesh([mesh0, mesh1])
    V = FunctionSpace(domain, elem)
    V0 = FunctionSpace(mesh0, elem0)
    V1 = FunctionSpace(mesh1, elem1)
    f = Coefficient(V, count=1000)
    f0_split = Coefficient(V0)
    f1_split = Coefficient(V1)
    f0, f1 = split(f)
    u0 = TrialFunction(V0)
    v1 = TestFunction(V1)
    ds1 = Measure("ds", mesh1)
    form = inner(grad(f1('|')), grad(f0('-'))) * inner(grad(u0('-')), grad(v1('|'))) * ds1(777)
    form_data = compute_form_data(form, do_split_coefficients={f: [f0_split, f1_split]})
    integral_data, = form_data.integral_data
    assert len(integral_data.domain_integral_type_map) == 2
    assert integral_data.domain_integral_type_map[mesh0] == "interior_facet"
    assert integral_data.domain_integral_type_map[mesh1] == "exterior_facet"
    kernel, = compile_form(form)
    assert kernel.domain_number == 0
    assert kernel.integral_type == "exterior_facet"
    assert kernel.subdomain_id == (777, )
    assert kernel.active_domain_numbers.coordinates == (0, 1)
    assert kernel.active_domain_numbers.cell_orientations == ()
    assert kernel.active_domain_numbers.cell_sizes == ()
    assert kernel.active_domain_numbers.exterior_facets == (0, )
    assert kernel.active_domain_numbers.interior_facets == (1, )
    assert kernel.coefficient_numbers == ((0, (0, 1)), )
    assert isinstance(kernel.arguments[0], kernel_args.OutputKernelArg)
    assert isinstance(kernel.arguments[1], kernel_args.CoordinatesKernelArg)
    assert isinstance(kernel.arguments[2], kernel_args.CoordinatesKernelArg)
    assert isinstance(kernel.arguments[3], kernel_args.CoefficientKernelArg)
    assert isinstance(kernel.arguments[4], kernel_args.CoefficientKernelArg)
    assert isinstance(kernel.arguments[5], kernel_args.ExteriorFacetKernelArg)
    assert isinstance(kernel.arguments[6], kernel_args.InteriorFacetKernelArg)
    assert kernel.arguments[0].loopy_arg.shape == (10, 2 * 6)
    assert kernel.arguments[1].loopy_arg.shape == (1 * (3 * gdim), )
    assert kernel.arguments[2].loopy_arg.shape == (2 * (3 * gdim), )
    assert kernel.arguments[3].loopy_arg.shape == (2 * 6, )
    assert kernel.arguments[4].loopy_arg.shape == (10, )
    assert kernel.arguments[5].loopy_arg.shape == (1, )
    assert kernel.arguments[6].loopy_arg.shape == (2, )
