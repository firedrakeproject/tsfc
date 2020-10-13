import numpy
import collections
from collections import namedtuple
import operator
import string
from itertools import chain, product
from functools import reduce, partial

from ufl import Coefficient, MixedElement as ufl_MixedElement, FunctionSpace, FiniteElement

import gem
from gem.optimise import remove_componenttensors as prune

import loopy as lp

import finat

from tsfc.finatinterface import create_element
from tsfc.kernel_interface.common import KernelBuilderBase as _KernelBuilderBase
from tsfc.kernel_interface.common import KernelBuilderMixin
from tsfc.kernel_interface.firedrake import check_requirements
from tsfc.loopy import generate as generate_loopy


# Expression kernel description type
ExpressionKernel = namedtuple('ExpressionKernel', ['ast', 'oriented', 'needs_cell_sizes', 'coefficients', 'tabulations'])


def make_builder(*args, **kwargs):
    return partial(KernelBuilder, *args, **kwargs)


class Kernel(object):
    __slots__ = ("ast", "integral_type", "oriented", "subdomain_id",
                 "domain_number", "needs_cell_sizes", "tabulations",
                 "coefficient_numbers", "external_data_numbers", "external_data_parts", "__weakref__")
    """A compiled Kernel object.

    :kwarg ast: The loopy kernel object.
    :kwarg integral_type: The type of integral.
    :kwarg oriented: Does the kernel require cell_orientations.
    :kwarg subdomain_id: What is the subdomain id for this kernel.
    :kwarg domain_number: Which domain number in the original form
        does this kernel correspond to (can be used to index into
        original_form.ufl_domains() to get the correct domain).
    :kwarg coefficient_numbers: A list of which coefficients from the
        form the kernel needs.
    :kwarg tabulations: The runtime tabulations this kernel requires
    :kwarg needs_cell_sizes: Does the kernel require cell sizes.
    """
    def __init__(self, ast=None, integral_type=None, oriented=False,
                 subdomain_id=None, domain_number=None,
                 coefficient_numbers=(),
                 needs_cell_sizes=False):
        # Defaults
        self.ast = ast
        self.integral_type = integral_type
        self.oriented = oriented
        self.domain_number = domain_number
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        self.external_data_numbers = []
        self.external_data_parts = []
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
            shape = (2,)
            cell_orientations = gem.Variable("cell_orientations", shape)
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),
                                       gem.Indexed(cell_orientations, (1,)))
        else:
            shape = (1,)
            cell_orientations = gem.Variable("cell_orientations", shape)
            self._cell_orientations = (gem.Indexed(cell_orientations, (0,)),)
        self.cell_orientations_loopy_arg = lp.GlobalArg("cell_orientations", dtype=numpy.int32, shape=shape)

    def _coefficient(self, coefficient, name):
        """Prepare a coefficient. Adds glue code for the coefficient
        and adds the coefficient to the coefficient map.

        :arg coefficient: :class:`ufl.Coefficient`
        :arg name: coefficient name
        :returns: loopy argument for the coefficient
        """
        funarg, expression = prepare_coefficient(coefficient.ufl_element(), name, self.scalar_type, interior_facet=self.interior_facet)
        self.coefficient_map[coefficient] = expression
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
            funarg, expression = prepare_coefficient(f.ufl_element(), "cell_sizes", self.scalar_type, interior_facet=self.interior_facet)
            self.cell_sizes_arg = funarg
            self._cell_sizes = expression

    def create_element(self, element, **kwargs):
        """Create a FInAT element (suitable for tabulating with) given
        a UFL element."""
        return create_element(element, **kwargs)


class ExpressionKernelBuilder(KernelBuilderBase):
    """Builds expression kernels for UFL interpolation in Firedrake."""

    def __init__(self, scalar_type):
        super(ExpressionKernelBuilder, self).__init__(scalar_type=scalar_type)
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

        :arg return_arg: loopy.GlobalArg for the return value
        :arg impero_c: gem.ImperoC object that represents the kernel
        :arg index_names: pre-assigned index names
        :returns: :class:`ExpressionKernel` object
        """
        args = [return_arg]
        if self.oriented:
            args.append(self.cell_orientations_loopy_arg)
        if self.cell_sizes:
            args.append(self.cell_sizes_arg)
        args.extend(self.kernel_args)
        for name_, shape in self.tabulations:
            args.append(lp.GlobalArg(name_, dtype=self.scalar_type, shape=shape))

        loopy_kernel = generate_loopy(impero_c, args, self.scalar_type,
                                      "expression_kernel", index_names)
        return ExpressionKernel(loopy_kernel, self.oriented, self.cell_sizes,
                                self.coefficients, self.tabulations)


class KernelBuilder(KernelBuilderBase, KernelBuilderMixin):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_data, scalar_type, fem_scalar_type,
                 dont_split=(), function_replace_map={}, diagonal=False):
        """Initialise a kernel builder."""
        integral_type = integral_data.integral_type
        KernelBuilderBase.__init__(self, scalar_type, integral_type.startswith("interior_facet"))
        self.integral_data = integral_data

        self.diagonal = diagonal
        self.coordinates_arg = None
        self.coefficient_args = []
        self.coefficient_split = {}
        self.external_data_args = []
        # Map to raw ufl Coefficient.
        self.dont_split = frozenset(function_replace_map[f] for f in dont_split if f in function_replace_map)

        self.arguments = integral_data.arguments
        self.local_tensor, self.return_variables, self.argument_multiindices, self.argument_multiindices_dummy = self.set_arguments(self.arguments)
        self.mode_irs=collections.OrderedDict()

        self.quadrature_indices = []

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

        self.set_coordinates(integral_data.domain)
        self.set_cell_sizes(integral_data.domain)
        self.set_coefficients(integral_data.coefficients)

        self.fem_scalar_type = fem_scalar_type

    def set_arguments(self, arguments):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :arg multiindices: GEM argument multiindices
        :returns: GEM expression representing the return variable
        """
        argument_multiindices = tuple(create_element(arg.ufl_element()).get_indices()
                                      for arg in arguments)
        argument_multiindices_dummy = tuple(tuple(gem.Index(extent=a.extent) for a in arg) for arg in argument_multiindices)
        if self.diagonal:
            # Error checking occurs in the builder constructor.
            # Diagonal assembly is obtained by using the test indices for
            # the trial space as well.
            a, _ = argument_multiindices
            argument_multiindices = (a, a)
        local_tensor, return_variables = prepare_arguments(
                                             arguments,
                                             argument_multiindices,
                                             self.scalar_type,
                                             interior_facet=self.interior_facet,
                                             diagonal=self.diagonal)
        return local_tensor, return_variables, argument_multiindices, argument_multiindices_dummy

    def set_coordinates(self, domain):
        """Prepare the coordinate field.

        :arg domain: :class:`ufl.Domain`
        """
        # Create a fake coordinate coefficient for a domain.
        f = Coefficient(FunctionSpace(domain, domain.ufl_coordinate_element()))
        self.domain_coordinate[domain] = f
        self.coordinates_arg = self._coefficient(f, "coords")

    def set_coefficients(self, coefficients):
        """Prepare the coefficients of the form.

        :arg coefficients: a tuple of `ufl.Coefficient`s.
        """
        coeffs = []
        for c in coefficients:
            if type(c.ufl_element()) == ufl_MixedElement:
                if c in self.dont_split:
                    coeffs.append(c)
                    self.coefficient_split[c] = [c]
                else:
                    split = [Coefficient(FunctionSpace(c.ufl_domain(), element))
                             for element in c.ufl_element().sub_elements()]
                    coeffs.extend(split)
                    self.coefficient_split[c] = split
            else:
                coeffs.append(c)
        for i, c in enumerate(coeffs):
            self.coefficient_args.append(self._coefficient(c, "w_%d" % i))

    def set_external_data(self, elements):
        """Prepare external data.

        :arg elements: a tuple of `ufl.FiniteElement`s.
        """
        if any(type(element) == ufl_MixedElement for element in elements):
            raise ValueError("Unable to handle `MixedElement`s.")
        expressions = []
        for i, element in enumerate(elements):
            funarg, expression = prepare_coefficient(element, "e_%d" % i, self.scalar_type, interior_facet=self.interior_facet)
            self.external_data_args.append(funarg)
            expressions.append(expression)
        return tuple(expressions)

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        return check_requirements(ir)

    def construct_kernel(self, kernel_name):
        """Construct a fully built :class:`Kernel`.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: function name
        :arg impero_c: ImperoC tuple with Impero AST and other data
        :arg index_names: pre-assigned index names
        :returns: :class:`Kernel` object
        """
        integral_data = self.integral_data

        impero_c, oriented, needs_cell_sizes, tabulations = self.compile_gem()

        if impero_c is None:
            return self.construct_empty_kernel(kernel_name)

        kernel = Kernel(integral_type=integral_data.integral_type,
                        subdomain_id=integral_data.subdomain_id,
                        domain_number=integral_data.domain_number)

        index_cache = self.fem_config['index_cache']
        index_names = _get_index_names(self.quadrature_indices, self.argument_multiindices, index_cache)

        kernel.coefficient_numbers = integral_data.coefficient_numbers

        # requirements
        kernel.oriented = oriented
        kernel.needs_cell_sizes = needs_cell_sizes
        kernel.tabulations = tabulations

        args = [self.local_tensor, self.coordinates_arg]
        if kernel.oriented:
            args.append(self.cell_orientations_loopy_arg)
        if kernel.needs_cell_sizes:
            args.append(self.cell_sizes_arg)
        args.extend(self.coefficient_args + self.external_data_args)
        if kernel.integral_type in ["exterior_facet", "exterior_facet_vert"]:
            args.append(lp.GlobalArg("facet", dtype=numpy.uint32, shape=(1,)))
        elif kernel.integral_type in ["interior_facet", "interior_facet_vert"]:
            args.append(lp.GlobalArg("facet", dtype=numpy.uint32, shape=(2,)))

        for name, shape in kernel.tabulations:
            args.append(lp.GlobalArg(name, dtype=self.scalar_type, shape=shape))

        kernel.ast = generate_loopy(impero_c, args, self.scalar_type, kernel_name, index_names)
        return kernel

    def construct_empty_kernel(self, kernel_name):
        """Return None, since Firedrake needs no empty kernels.

        :arg name: function name
        :returns: None
        """
        return None


def prepare_coefficient(ufl_element, name, scalar_type, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg ufl_element: UFL element
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expression)
         funarg     - :class:`loopy.GlobalArg` function argument
         expression - GEM expression referring to the Coefficient
                      values
    """
    assert isinstance(interior_facet, bool)

    if ufl_element.family() == 'Real':
        # Constant
        funarg = lp.GlobalArg(name, dtype=scalar_type, shape=(ufl_element.value_size(),))
        expression = gem.reshape(gem.Variable(name, (None,)),
                                 ufl_element.value_shape())

        return funarg, expression

    finat_element = create_element(ufl_element)
    shape = finat_element.index_shape
    size = numpy.prod(shape, dtype=int)

    if not interior_facet:
        expression = gem.reshape(gem.Variable(name, (size,)), shape)
    else:
        varexp = gem.Variable(name, (2 * size,))
        plus = gem.view(varexp, slice(size))
        minus = gem.view(varexp, slice(size, 2 * size))
        expression = (gem.reshape(plus, shape), gem.reshape(minus, shape))
        size = size * 2
    funarg = lp.GlobalArg(name, dtype=scalar_type, shape=(size,))
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
         funarg      - :class:`loopy.GlobalArg` function argument
         expressions - GEM expressions referring to the argument
                       tensor
    """
    assert isinstance(interior_facet, bool)

    if len(arguments) == 0:
        # No arguments
        funarg = lp.GlobalArg("A", dtype=scalar_type, shape=(1,))
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

    funarg = lp.GlobalArg("A", dtype=scalar_type, shape=c_shape)
    varexp = gem.Variable("A", c_shape)
    expressions = [expression(gem.view(varexp, *slices)) for slices in slicez]
    return funarg, prune(expressions)


def _get_index_names(quadrature_indices, argument_multiindices, index_cache):
    index_names = []

    def name_index(index, name):
        index_names.append((index, name))
        if index in index_cache:
            for multiindex, suffix in zip(index_cache[index],
                                          string.ascii_lowercase):
                name_multiindex(multiindex, name + suffix)

    def name_multiindex(multiindex, name):
        if len(multiindex) == 1:
            name_index(multiindex[0], name)
        else:
            for i, index in enumerate(multiindex):
                name_index(index, name + str(i))

    name_multiindex(quadrature_indices, 'ip')
    for multiindex, name in zip(argument_multiindices, ['j', 'k']):
        name_multiindex(multiindex, name)
    return index_names
