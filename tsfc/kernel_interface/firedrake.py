import numpy
from collections import namedtuple
from itertools import chain, product
from functools import partial

from ufl import Coefficient, Subspace, MixedElement as ufl_MixedElement, FunctionSpace, FiniteElement

import coffee.base as coffee

import gem
from gem.node import traversal
from gem.optimise import remove_componenttensors as prune

import finat

from tsfc.finatinterface import create_element
from tsfc.kernel_interface.common import KernelBuilderBase as _KernelBuilderBase
from tsfc.coffee import generate as generate_coffee


# Expression kernel description type
ExpressionKernel = namedtuple('ExpressionKernel', ['ast', 'oriented', 'needs_cell_sizes', 'coefficients', 'tabulations'])


def make_builder(*args, **kwargs):
    return partial(KernelBuilder, *args, **kwargs)


class Kernel(object):
    __slots__ = ("ast", "integral_type", "oriented", "subdomain_id",
                 "domain_number", "needs_cell_sizes", "tabulations", "quadrature_rule",
                 "coefficient_numbers", "subspace_numbers", "subspace_parts", "__weakref__")
    """A compiled Kernel object.

    :kwarg ast: The COFFEE ast for the kernel.
    :kwarg integral_type: The type of integral.
    :kwarg oriented: Does the kernel require cell_orientations.
    :kwarg subdomain_id: What is the subdomain id for this kernel.
    :kwarg domain_number: Which domain number in the original form
        does this kernel correspond to (can be used to index into
        original_form.ufl_domains() to get the correct domain).
    :kwarg coefficient_numbers: A list of which coefficients from the
        form the kernel needs.
    :kwarg subspace_numbers: A list of which subspaces from the
        form the kernel needs.
    :kwarg subspace_parts: A list of lists of subspace components
        that are actually used in the given integral_data. If `None` is given
        instead of a list of components, it is assumed that all components of
        that subspace are to be used.
    :kwarg quadrature_rule: The finat quadrature rule used to generate this kernel
    :kwarg tabulations: The runtime tabulations this kernel requires
    :kwarg needs_cell_sizes: Does the kernel require cell sizes.
    """
    def __init__(self, ast=None, integral_type=None, oriented=False,
                 subdomain_id=None, domain_number=None, quadrature_rule=None,
                 coefficient_numbers=(),
                 subspace_numbers=(),
                 subspace_parts=None,
                 needs_cell_sizes=False):
        # Defaults
        self.ast = ast
        self.integral_type = integral_type
        self.oriented = oriented
        self.domain_number = domain_number
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        self.subspace_numbers = subspace_numbers
        self.subspace_parts = subspace_parts
        self.needs_cell_sizes = needs_cell_sizes
        super(Kernel, self).__init__()


class KernelBuilderBase(_KernelBuilderBase):

    def __init__(self, scalar_type, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        super(KernelBuilderBase, self).__init__(scalar_type=scalar_type,
                                                interior_facet=interior_facet)

        # Cell orientation
        if self.interior_facet:
            cell_orientations = gem.Variable("cell_orientations", (2,))
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),
                                       gem.Indexed(cell_orientations, (1,)))
        else:
            cell_orientations = gem.Variable("cell_orientations", (1,))
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),)

    def _coefficient(self, coefficient, name):
        """Prepare a coefficient. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coefficient name
        :returns: COFFEE function argument for the coefficient
        """
        funarg, expression = prepare_coefficient_filter(coefficient, name, self.scalar_type, interior_facet=self.interior_facet)
        self.coefficient_map[coefficient] = expression
        return funarg

    def _subspace(self, subspace, name, subspace_type):
        """Prepare a subspace. Adds glue code for the subspace
        and adds the subspace to the subspace map.

        :arg subspace: :class:`ufl.Subspace`
        :arg name: subspace name
        :arg subspace_type: 'ScalarSubspace' or `'RotatedSubspace'
        :returns: COFFEE function argument for the subspace
        """
        funarg, expression = prepare_coefficient_filter(subspace, name, self.scalar_type, interior_facet=self.interior_facet)
        shape = expression.shape
        ii = tuple(gem.Index(extent=extent) for extent in shape)
        jj = tuple(gem.Index(extent=extent) for extent in shape)
        if subspace_type.__name__ == 'ScalarSubspace':
            # diag(mat) = phi
            eye = gem.Literal(1)
            for i, j in zip(ii, jj):
                eye = gem.Product(eye, gem.Delta(i, j))
            mat = gem.ComponentTensor(gem.Product(eye, expression[ii]), ii + jj)
        elif subspace_type.__name__ == 'RotatedSubspace':
            # mat = phi * phi^T
            # Only implement vertex-wise rotations for now.
            finat_element = create_element(subspace.ufl_element())
            if type(finat_element) not in (finat.hermite.Hermite,
                                           finat.bell.Bell,
                                           finat.argyris.Argyris):
                raise NotImplementedError("`RotatedSubspace` interface not implemented for %s." % type(finat_element).__name__)
            entity_dofs = finat_element.entity_dofs()
            indicators = []
            for vert, dofs in entity_dofs[0].items():
                a = numpy.zeros(shape, dtype=self.scalar_type)
                for dof in dofs:
                    a[(dof, )] = 1.
                indicators.append(gem.Literal(a))
            comp = gem.Zero()
            for indicator in indicators:
                comp = gem.Sum(comp, gem.Product(gem.Product(expression[ii], indicator[ii]), gem.Product(expression[jj], indicator[jj])))
            mat = gem.ComponentTensor(comp, ii + jj)
        self.subspace_map[subspace] = mat
        return funarg

    def set_cell_sizes(self, domain):
        """Setup a fake coefficient for "cell sizes".

        :arg domain: The domain of the integral.

        This is required for scaling of derivative basis functions on
        physically mapped elements (Argyris, Bell, etc...).  We need a
        measure of the mesh size around each vertex (hence this lives
        in P1).

        Should the domain have topological dimension 0 this does
        nothing.
        """
        if domain.ufl_cell().topological_dimension() > 0:
            # Can't create P1 since only P0 is a valid finite element if
            # topological_dimension is 0 and the concept of "cell size"
            # is not useful for a vertex.
            f = Coefficient(FunctionSpace(domain, FiniteElement("P", domain.ufl_cell(), 1)))
            funarg, expression = prepare_coefficient_filter(f, "cell_sizes", self.scalar_type, interior_facet=self.interior_facet)
            self.cell_sizes_arg = funarg
            self._cell_sizes = expression

    def create_element(self, element, **kwargs):
        """Create a FInAT element (suitable for tabulating with) given
        a UFL element."""
        return create_element(element, **kwargs)


class ExpressionKernelBuilder(KernelBuilderBase):
    """Builds expression kernels for UFL interpolation in Firedrake."""

    def __init__(self, scalar_type):
        super(ExpressionKernelBuilder, self).__init__(scalar_type)
        self.oriented = False
        self.cell_sizes = False

    def set_coefficients(self, coefficients):
        """Prepare the coefficients of the expression.

        :arg coefficients: UFL coefficients from Firedrake
        """
        self.coefficients = []  # Firedrake coefficients for calling the kernel
        self.coefficient_split = {}
        self.kernel_args = []

        for i, coefficient in enumerate(coefficients):
            if type(coefficient.ufl_element()) == ufl_MixedElement:
                subcoeffs = coefficient.split()  # Firedrake-specific
                self.coefficients.extend(subcoeffs)
                self.coefficient_split[coefficient] = subcoeffs
                self.kernel_args += [self._coefficient(subcoeff, "w_%d_%d" % (i, j))
                                     for j, subcoeff in enumerate(subcoeffs)]
            else:
                self.coefficients.append(coefficient)
                self.kernel_args.append(self._coefficient(coefficient, "w_%d" % (i,)))

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        self.oriented, self.cell_sizes, self.tabulations = check_requirements(ir)

    def construct_kernel(self, return_arg, impero_c, index_names):
        """Constructs an :class:`ExpressionKernel`.

        :arg return_arg: COFFEE argument for the return value
        :arg body: function body (:class:`coffee.Block` node)
        :returns: :class:`ExpressionKernel` object
        """
        args = [return_arg]
        if self.oriented:
            args.append(cell_orientations_coffee_arg)
        if self.cell_sizes:
            args.append(self.cell_sizes_arg)
        args.extend(self.kernel_args)

        body = generate_coffee(impero_c, index_names, self.scalar_type)

        for name_, shape in self.tabulations:
            args.append(coffee.Decl(self.scalar_type, coffee.Symbol(
                name_, rank=shape), qualifiers=["const"]))

        kernel_code = super(ExpressionKernelBuilder, self).construct_kernel("expression_kernel", args, body)
        return ExpressionKernel(kernel_code, self.oriented, self.cell_sizes, self.coefficients, self.tabulations)


class KernelBuilder(KernelBuilderBase):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_type, subdomain_id, domain_number, scalar_type,
                 dont_split=(), diagonal=False):
        """Initialise a kernel builder."""
        super(KernelBuilder, self).__init__(scalar_type, integral_type.startswith("interior_facet"))

        self.kernel = Kernel(integral_type=integral_type, subdomain_id=subdomain_id,
                             domain_number=domain_number)
        self.diagonal = diagonal
        self.local_tensor = None
        self.coordinates_arg = None
        self.coefficient_args = []
        self.coefficient_split = {}
        self.subspace_args = []
        self.subspace_split = {}
        self.dont_split = frozenset(dont_split)

        # Facet number
        if integral_type in ['exterior_facet', 'exterior_facet_vert']:
            facet = gem.Variable('facet', (1,))
            self._entity_number = {None: gem.VariableIndex(gem.Indexed(facet, (0,)))}
        elif integral_type in ['interior_facet', 'interior_facet_vert']:
            facet = gem.Variable('facet', (2,))
            self._entity_number = {
                '+': gem.VariableIndex(gem.Indexed(facet, (0,))),
                '-': gem.VariableIndex(gem.Indexed(facet, (1,)))
            }
        elif integral_type == 'interior_facet_horiz':
            self._entity_number = {'+': 1, '-': 0}

    def set_arguments(self, arguments, multiindices):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg multiindices: GEM argument multiindices
        :returns: GEM expression representing the return variable
        """
        self.local_tensor, expressions = prepare_arguments(
            arguments, multiindices, self.scalar_type, interior_facet=self.interior_facet,
            diagonal=self.diagonal)
        return expressions

    def set_coordinates(self, domain):
        """Prepare the coordinate field.

        :arg domain: :class:`ufl.Domain`
        """
        # Create a fake coordinate coefficient for a domain.
        f = Coefficient(FunctionSpace(domain, domain.ufl_coordinate_element()))
        self.domain_coordinate[domain] = f
        self.coordinates_arg = self._coefficient(f, "coords")

    def set_coefficients(self, integral_data, form_data):
        """Prepare the coefficients of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        objects = []
        object_numbers = []
        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        enabled_objects = integral_data.enabled_coefficients
        reduced_objects = form_data.reduced_coefficients
        replace_map = form_data.function_replace_map
        for i in range(len(enabled_objects)):
            if enabled_objects[i]:
                original = reduced_objects[i]
                obj = replace_map[original]
                if type(obj.ufl_element()) == ufl_MixedElement:
                    if original in self.dont_split:
                        objects.append(obj)
                        self.coefficient_split[obj] = [obj]
                    else:
                        split = [Coefficient(FunctionSpace(obj.ufl_domain(), element))
                                 for element in obj.ufl_element().sub_elements()]
                        objects.extend(split)
                        self.coefficient_split[obj] = split
                else:
                    objects.append(obj)
                # This is which coefficient in the original form the
                # current coefficient is.
                # Consider f*v*dx + g*v*ds, the full form contains two
                # coefficients, but each integral only requires one.
                object_numbers.append(form_data.original_coefficient_positions[i])
        for i, obj in enumerate(objects):
            self.coefficient_args.append(self._coefficient(obj, "w_%d" % i))
        self.kernel.coefficient_numbers = tuple(object_numbers)

    def set_subspaces(self, integral_data, form_data):
        """Prepare the subspaces of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        objects = []
        object_numbers = []
        subspace_types = []
        # enabled_subspaces is a boolean array that indicates which
        # of reduced_subspaces the integral requires.
        enabled_objects = integral_data.enabled_subspaces
        reduced_objects = form_data.reduced_subspaces
        replace_map = form_data.subspace_replace_map
        for i in range(len(enabled_objects)):
            if enabled_objects[i]:
                original = reduced_objects[i]
                obj = replace_map[original]
                if type(obj.ufl_element()) == ufl_MixedElement:
                    if original in self.dont_split:
                        objects.append(obj)
                        self.subspace_split[obj] = [obj]
                        subspace_types.append(type(original))
                    else:
                        split = [Subspace(FunctionSpace(obj.ufl_domain(), element))
                                 for element in obj.ufl_element().sub_elements()]
                        objects.extend(split)
                        self.subspace_split[obj] = split
                        subspace_types.extend([type(original)] * len(split))
                else:
                    objects.append(obj)
                    subspace_types.append(type(original))
                # This is which subspace in the original form the
                # current subspace is.
                object_numbers.append(form_data.original_subspace_positions[i])
        for i, obj in enumerate(objects):
            self.subspace_args.append(self._subspace(obj, "r_%d" % i, subspace_types[i]))
        self.kernel.subspace_numbers = tuple(object_numbers)
        # TODO: Remove redundant components by appropriately setting this.
        # For now, this is only necessary for 'subspace' to deal with
        # splitting of arguments.
        self.kernel.subspace_parts = [None for _ in self.kernel.subspace_numbers]

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        knl = self.kernel
        knl.oriented, knl.needs_cell_sizes, knl.tabulations = check_requirements(ir)

    def construct_kernel(self, name, impero_c, index_names, quadrature_rule):
        """Construct a fully built :class:`Kernel`.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: function name
        :arg impero_c: ImperoC tuple with Impero AST and other data
        :arg index_names: pre-assigned index names
        :arg quadrature rule: quadrature rule

        :returns: :class:`Kernel` object
        """
        body = generate_coffee(impero_c, index_names, self.scalar_type)

        args = [self.local_tensor, self.coordinates_arg]
        if self.kernel.oriented:
            args.append(cell_orientations_coffee_arg)
        if self.kernel.needs_cell_sizes:
            args.append(self.cell_sizes_arg)
        args.extend(self.coefficient_args + self.subspace_args)
        if self.kernel.integral_type in ["exterior_facet", "exterior_facet_vert"]:
            args.append(coffee.Decl("unsigned int",
                                    coffee.Symbol("facet", rank=(1,)),
                                    qualifiers=["const"]))
        elif self.kernel.integral_type in ["interior_facet", "interior_facet_vert"]:
            args.append(coffee.Decl("unsigned int",
                                    coffee.Symbol("facet", rank=(2,)),
                                    qualifiers=["const"]))

        for name_, shape in self.kernel.tabulations:
            args.append(coffee.Decl(self.scalar_type, coffee.Symbol(
                name_, rank=shape), qualifiers=["const"]))

        self.kernel.quadrature_rule = quadrature_rule

        self.kernel.ast = KernelBuilderBase.construct_kernel(self, name, args, body)
        return self.kernel

    def construct_empty_kernel(self, name):
        """Return None, since Firedrake needs no empty kernels.

        :arg name: function name
        :returns: None
        """
        return None


def check_requirements(ir):
    """Look for cell orientations, cell sizes, and collect tabulations
    in one pass."""
    cell_orientations = False
    cell_sizes = False
    rt_tabs = {}
    for node in traversal(ir):
        if isinstance(node, gem.Variable):
            if node.name == "cell_orientations":
                cell_orientations = True
            elif node.name == "cell_sizes":
                cell_sizes = True
            elif node.name.startswith("rt_"):
                rt_tabs[node.name] = node.shape
    return cell_orientations, cell_sizes, tuple(sorted(rt_tabs.items()))


def prepare_coefficient_filter(obj, name, scalar_type, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients/Subspace.

    :arg obj: UFL Coefficient/Subspace
    :arg name: unique name to refer to the Coefficient/Subspace in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expression)
         funarg     - :class:`coffee.Decl` function argument
         expression - GEM expression referring to the Coefficient/Subspace
                      values
    """
    assert isinstance(interior_facet, bool)

    if obj.ufl_element().family() == 'Real':
        # Constant
        funarg = coffee.Decl(scalar_type, coffee.Symbol(name),
                             pointers=[("restrict",)],
                             qualifiers=["const"])

        expression = gem.reshape(gem.Variable(name, (None,)),
                                 obj.ufl_shape)

        return funarg, expression

    finat_element = create_element(obj.ufl_element())
    shape = finat_element.index_shape
    size = numpy.prod(shape, dtype=int)

    funarg = coffee.Decl(scalar_type, coffee.Symbol(name),
                         pointers=[("restrict",)],
                         qualifiers=["const"])

    if not interior_facet:
        expression = gem.reshape(gem.Variable(name, (size,)), shape)
    else:
        varexp = gem.Variable(name, (2 * size,))
        plus = gem.view(varexp, slice(size))
        minus = gem.view(varexp, slice(size, 2 * size))
        expression = (gem.reshape(plus, shape),
                      gem.reshape(minus, shape))
    return funarg, expression


def prepare_arguments(arguments, multiindices, scalar_type, interior_facet=False, diagonal=False):
    """Bridges the kernel interface and the GEM abstraction for
    Arguments.  Vector Arguments are rearranged here for interior
    facet integrals.

    :arg arguments: UFL Arguments
    :arg multiindices: Argument multiindices
    :arg interior_facet: interior facet integral?
    :arg diagonal: Are we assembling the diagonal of a rank-2 element tensor?
    :returns: (funarg, expression)
         funarg      - :class:`coffee.Decl` function argument
         expressions - GEM expressions referring to the argument
                       tensor
    """
    assert isinstance(interior_facet, bool)

    if len(arguments) == 0:
        # No arguments
        funarg = coffee.Decl(scalar_type, coffee.Symbol("A", rank=(1,)))
        expression = gem.Indexed(gem.Variable("A", (1,)), (0,))

        return funarg, [expression]

    elements = tuple(create_element(arg.ufl_element()) for arg in arguments)
    shapes = tuple(element.index_shape for element in elements)

    if diagonal:
        if len(arguments) != 2:
            raise ValueError("Diagonal only for 2-forms")
        try:
            element, = set(elements)
        except ValueError:
            raise ValueError("Diagonal only for diagonal blocks (test and trial spaces the same)")

        elements = (element, )
        shapes = tuple(element.index_shape for element in elements)
        multiindices = multiindices[:1]

    def expression(restricted):
        return gem.Indexed(gem.reshape(restricted, *shapes),
                           tuple(chain(*multiindices)))

    u_shape = numpy.array([numpy.prod(shape, dtype=int) for shape in shapes])
    if interior_facet:
        c_shape = tuple(2 * u_shape)
        slicez = [[slice(r * s, (r + 1) * s)
                   for r, s in zip(restrictions, u_shape)]
                  for restrictions in product((0, 1), repeat=len(arguments))]
    else:
        c_shape = tuple(u_shape)
        slicez = [[slice(s) for s in u_shape]]

    funarg = coffee.Decl(scalar_type, coffee.Symbol("A", rank=c_shape))
    varexp = gem.Variable("A", c_shape)
    expressions = [expression(gem.view(varexp, *slices)) for slices in slicez]
    return funarg, prune(expressions)


cell_orientations_coffee_arg = coffee.Decl("int", coffee.Symbol("cell_orientations"),
                                           pointers=[("restrict",)],
                                           qualifiers=["const"])
"""COFFEE function argument for cell orientations"""
