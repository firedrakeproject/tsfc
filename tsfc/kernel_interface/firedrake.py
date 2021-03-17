import numpy
import collections
from collections import namedtuple
from itertools import chain, product
from functools import partial

from ufl import Coefficient, MixedElement as ufl_MixedElement, FunctionSpace, FiniteElement

import coffee.base as coffee

import gem
from gem.node import traversal
from gem.optimise import remove_componenttensors as prune

from tsfc.finatinterface import create_element
from tsfc.kernel_interface.common import KernelBuilderBase as _KernelBuilderBase, KernelBuilderMixin, get_index_names
from tsfc.coffee import generate as generate_coffee


# Expression kernel description type
ExpressionKernel = namedtuple('ExpressionKernel', ['ast', 'oriented', 'needs_cell_sizes', 'coefficients', 'tabulations'])


def make_builder(*args, **kwargs):
    return partial(KernelBuilder, *args, **kwargs)


class Kernel(object):
    __slots__ = ("ast", "integral_type", "oriented", "subdomain_id",
                 "domain_number", "needs_cell_sizes", "tabulations",
                 "coefficient_numbers", "external_data_numbers", "external_data_parts", "__weakref__")
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
    :kwarg external_data_numbers: A list of external data structures
        the kernel needs. These data structures do not originate in
        UFL forms, but in the operations potentially applied to gem
        expressions after compiling UFL and before compiling gem.
    :kwarg external_data_parts: A list of tuples of indices. Each
        tuple contains indices of the associated external data
        structure that the kernel needs.
    :kwarg tabulations: The runtime tabulations this kernel requires
    :kwarg needs_cell_sizes: Does the kernel require cell sizes.
    """
    def __init__(self, ast=None, integral_type=None, oriented=False,
                 subdomain_id=None, domain_number=None,
                 coefficient_numbers=(),
                 external_data_numbers=(), external_data_parts=(),
                 needs_cell_sizes=False,
                 tabulations=None):
        # Defaults
        self.ast = ast
        self.integral_type = integral_type
        self.oriented = oriented
        self.domain_number = domain_number
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        self.external_data_numbers = external_data_numbers
        self.external_data_parts = external_data_parts
        self.needs_cell_sizes = needs_cell_sizes
        self.tabulations = tabulations
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


class KernelBuilder(KernelBuilderBase, KernelBuilderMixin):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_data, scalar_type, fem_scalar_type,
                 dont_split=(), function_replace_map={}, diagonal=False):
        """Initialise a kernel builder."""
        KernelBuilderBase.__init__(self, scalar_type, integral_data.integral_type.startswith("interior_facet"))
        self.fem_scalar_type = fem_scalar_type

        self.diagonal = diagonal
        self.coordinates_arg = None
        self.coefficient_args = []
        self.coefficient_split = {}
        self.external_data_args = []
        # Map functions to raw coefficients here.
        self.dont_split = frozenset(function_replace_map[f] for f in dont_split if f in function_replace_map)

        # Facet number
        integral_type = integral_data.integral_type
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

        self.integral_data = integral_data
        self.arguments = integral_data.arguments
        self.local_tensor, self.return_variables, self.argument_multiindices = self.set_arguments(self.arguments)
        self.mode_irs = collections.OrderedDict()
        self.quadrature_indices = []

    def set_arguments(self, arguments):
        """Process arguments.

        :arg arguments: :class:`ufl.Argument`s
        :returns: :class:`loopy.GlobalArg` for the return variable,
            GEM expression representing the return variable,
            GEM argument multiindices.
        """
        argument_multiindices = tuple(create_element(arg.ufl_element()).get_indices()
                                      for arg in arguments)
        if self.diagonal:
            # Error checking occurs in the builder constructor.
            # Diagonal assembly is obtained by using the test indices for
            # the trial space as well.
            a, _ = argument_multiindices
            argument_multiindices = (a, a)
        local_tensor, return_variables = prepare_arguments(arguments,
                                                           argument_multiindices,
                                                           self.scalar_type,
                                                           interior_facet=self.interior_facet,
                                                           diagonal=self.diagonal)
        return local_tensor, return_variables, argument_multiindices

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
        """Prepare external data structures.

        :arg elements: a tuple of `ufl.FiniteElement`s.
        :returns: gem expressions for the data represented by elements.

        The retuned gem expressions are to be used in the operations
        applied to the gem expressions obtained by compiling UFL before
        compiling gem. The users are responsible for bridging these
        gem expressions and actual data by setting correct values in
        `external_data_numbers` and `external_data_parts` in the kernel.
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

    def construct_kernel(self, kernel_name, external_data_numbers=(), external_data_parts=()):
        """Construct a fully built :class:`Kernel`.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg kernel_name: function name
        :returns: :class:`Kernel` object
        """
        integral_data = self.integral_data

        impero_c, oriented, needs_cell_sizes, tabulations = self.compile_gem()

        if impero_c is None:
            return self.construct_empty_kernel(kernel_name)

        args = [self.local_tensor, self.coordinates_arg]
        if oriented:
            args.append(cell_orientations_coffee_arg)
        if needs_cell_sizes:
            args.append(self.cell_sizes_arg)
        args.extend(self.coefficient_args + self.external_data_args)
        if integral_data.integral_type in ["exterior_facet", "exterior_facet_vert"]:
            args.append(coffee.Decl("unsigned int",
                                    coffee.Symbol("facet", rank=(1,)),
                                    qualifiers=["const"]))
        elif integral_data.integral_type in ["interior_facet", "interior_facet_vert"]:
            args.append(coffee.Decl("unsigned int",
                                    coffee.Symbol("facet", rank=(2,)),
                                    qualifiers=["const"]))
        for name, shape in tabulations:
            args.append(coffee.Decl(self.scalar_type, coffee.Symbol(
                name, rank=shape), qualifiers=["const"]))
        index_cache = self.fem_config['index_cache']
        index_names = get_index_names(self.quadrature_indices, self.argument_multiindices, index_cache)
        body = generate_coffee(impero_c, index_names, self.scalar_type)
        ast = KernelBuilderBase.construct_kernel(self, kernel_name, args, body)

        return Kernel(ast=ast,
                      integral_type=integral_data.integral_type,
                      subdomain_id=integral_data.subdomain_id,
                      domain_number=integral_data.domain_number,
                      coefficient_numbers=integral_data.coefficient_numbers,
                      external_data_numbers=external_data_numbers,
                      external_data_parts=external_data_parts,
                      oriented=oriented,
                      needs_cell_sizes=needs_cell_sizes,
                      tabulations=tabulations)

    def construct_empty_kernel(self, kernel_name):
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


def prepare_coefficient(ufl_element, name, scalar_type, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg ufl_element: UFL element
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expression)
         funarg     - :class:`coffee.Decl` function argument
         expression - GEM expression referring to the Coefficient
                      values
    """
    assert isinstance(interior_facet, bool)

    if ufl_element.family() == 'Real':
        # Constant
        funarg = coffee.Decl(scalar_type, coffee.Symbol(name),
                             pointers=[("restrict",)],
                             qualifiers=["const"])

        expression = gem.reshape(gem.Variable(name, (None,)),
                                 ufl_element.value_shape())

        return funarg, expression

    finat_element = create_element(ufl_element)
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
