import numpy
from collections import namedtuple
from itertools import chain, product
from functools import partial

from ufl import Coefficient, MixedElement as ufl_MixedElement, FunctionSpace, FiniteElement

import coffee.base as coffee

import gem
from gem.node import traversal
from gem.optimise import remove_componenttensors as prune

from tsfc.finatinterface import create_element
from tsfc.kernel_interface.common import KernelBuilderBase as _KernelBuilderBase
from tsfc.coffee import generate as generate_coffee


# Expression kernel description type
ExpressionKernel = namedtuple('ExpressionKernel', ['ast', 'oriented', 'needs_cell_sizes', 'coefficients', 'tabulations'])


def make_builder(*args, **kwargs):
    return partial(KernelBuilder, *args, **kwargs)


class Kernel(object):
    __slots__ = ("ast", "integral_type", "oriented", "subdomain_id",
                 "domain_number", "domain_numbers", "needs_cell_sizes", "tabulations", "quadrature_rule",
                 "coefficient_numbers", "coefficient_parts", "__weakref__")
    """A compiled Kernel object.

    :kwarg ast: The COFFEE ast for the kernel.
    :kwarg integral_type: The type of integral.
    :kwarg oriented: Does the kernel require cell_orientations.
    :kwarg subdomain_id: What is the subdomain id for this kernel.
    :kwarg domain_number: Which domain number in the original form
        does this kernel correspond to (can be used to index into
        original_form.ufl_domains() to get the correct domain).
    :kwarg domain_numbers: List of domain numbers of the domains
        in the original form that are relevant in this kernel
        (can be used to index into original_form.ufl_domains() to get
        the correct domains).
    :kwarg coefficient_numbers: A list of which coefficients from the
        form the kernel needs.
    :kwarg coefficient_parts: A list of enabled parts corresponding to
        the coefficients represented by coefficient_numbers
        (Only significant when the coefficient is mixed).
    :kwarg quadrature_rule: The finat quadrature rule used to generate this kernel
    :kwarg tabulations: The runtime tabulations this kernel requires
    :kwarg needs_cell_sizes: Does the kernel require cell sizes.
    """
    def __init__(self, ast=None, integral_type=None, oriented=False,
                 subdomain_id=None, domain_number=None, domain_numbers=None, quadrature_rule=None,
                 coefficient_numbers=(), coefficient_parts=None,
                 needs_cell_sizes=False):
        # Defaults
        self.ast = ast
        self.integral_type = integral_type
        self.oriented = oriented
        self.domain_number = domain_number
        self.domain_numbers = domain_numbers
        self.subdomain_id = subdomain_id
        self.coefficient_numbers = coefficient_numbers
        self.coefficient_parts = coefficient_parts
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
        funarg, varexp, expression = prepare_coefficient(coefficient, name, self.scalar_type, interior_facet=self.interior_facet)
        self.coefficient_map[coefficient] = expression
        return funarg, varexp

    def set_cell_sizes(self, domain):
        """Setup a fake coefficient for "cell sizes".

        :arg domain: The domain of the integral.

        This is required for scaling of derivative basis functions on
        physically mapped elements (Argyris, Bell, etc...).  We need a
        measure of the mesh size around each vertex (hence this lives
        in P1).
        """
        f = Coefficient(FunctionSpace(domain, FiniteElement("P", domain.ufl_cell(), 1)))
        funarg, varexp, expression = prepare_coefficient(f, "cell_sizes", self.scalar_type, interior_facet=self.interior_facet)
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
                self.kernel_args += [self._coefficient(subcoeff, "w_%d_%d" % (i, j))[0]
                                     for j, subcoeff in enumerate(subcoeffs)]
            else:
                self.coefficients.append(coefficient)
                self.kernel_args.append(self._coefficient(coefficient, "w_%d" % (i,))[0])

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        self.oriented, self.cell_sizes, self.tabulations = check_requirements(ir)

    def construct_kernel(self, return_arg, impero_c, precision, index_names):
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

        body = generate_coffee(impero_c, index_names, precision, self.scalar_type)

        for name_, shape in self.tabulations:
            args.append(coffee.Decl(self.scalar_type, coffee.Symbol(
                name_, rank=shape), qualifiers=["const"]))

        kernel_code = super(ExpressionKernelBuilder, self).construct_kernel("expression_kernel", args, body)
        return ExpressionKernel(kernel_code, self.oriented, self.cell_sizes, self.coefficients, self.tabulations)


class KernelBuilder(KernelBuilderBase):
    """Helper class for building a :class:`Kernel` object."""

    def __init__(self, integral_type, subdomain_id, domain_number, domain_numbers, scalar_type, dont_split=(), diagonal=False):
        """Initialise a kernel builder."""
        super(KernelBuilder, self).__init__(scalar_type, integral_type.startswith("interior_facet"))

        self.kernel = Kernel(integral_type=integral_type, subdomain_id=subdomain_id,
                             domain_number=domain_number, domain_numbers=domain_numbers)
        self.diagonal = diagonal
        self.local_tensor = None
        self.coordinates_arg = []
        self.coefficient_args = []
        self.coefficient_split = {}
        self.coefficient_len = []
        self.dont_split = frozenset(dont_split)
        self.variable_to_coefficient_index_map = {}

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

    def set_coordinates(self, domains):
        """Prepare the coordinate field.

        :arg domains: a tuple of :class:`ufl.Domain`s
        """
        # Create a fake coordinate coefficient for each enabled domain.
        #f = Coefficient(FunctionSpace(domain, domain.ufl_coordinate_element()))
        #self.domain_coordinate[domain] = f
        #self.coordinates_arg = self._coefficient(f, "coords")
        for i, domain in enumerate(domains):
            f = Coefficient(FunctionSpace(domain, domain.ufl_coordinate_element()))
            self.domain_coordinate[domain] = f
            self.coordinates_arg.append(self._coefficient(f, "coords" + str(i))[0])

    def set_coefficients(self, integral_data, form_data):
        """Prepare the coefficients of the form.

        :arg integral_data: UFL integral data
        :arg form_data: UFL form data
        """
        coefficients = []
        coefficient_numbers = []
        # enabled_coefficients is a boolean array that indicates which
        # of reduced_coefficients the integral requires.
        for i in range(len(integral_data.enabled_coefficients)):
            if integral_data.enabled_coefficients[i]:
                original = form_data.reduced_coefficients[i]
                coefficient = form_data.function_replace_map[original]
                if type(coefficient.ufl_element()) == ufl_MixedElement:
                    if original in self.dont_split:
                        coefficients.append(coefficient)
                        self.coefficient_split[coefficient] = [coefficient]
                        self.variable_to_coefficient_index_map[coefficient] = None
                        self.coefficient_len.append(None)
                    else:
                        split = []
                        if coefficient.mixed():
                            assert isinstance(coefficient.ufl_domain(), tuple)
                            for idx, (d, e) in enumerate(zip(coefficient.ufl_domain(), coefficient.ufl_element().sub_elements())):
                                coeff = Coefficient(FunctionSpace(d, e))
                                split.append(coeff)
                                self.variable_to_coefficient_index_map[coeff] = (len(coefficient_numbers), idx)
                        else:
                            assert not isinstance(coefficient.ufl_domain(), tuple)
                            d = coefficient.ufl_domain()
                            for idx, e in enumerate(coefficient.ufl_element().sub_elements()):
                                coeff = Coefficient(FunctionSpace(d, e))
                                split.append(coeff)
                                self.variable_to_coefficient_index_map[coeff] = (len(coefficient_numbers), idx)
                        coefficients.extend(split)
                        self.coefficient_split[coefficient] = split
                        self.coefficient_len.append(len(split))
                else:
                    coefficients.append(coefficient)
                    self.variable_to_coefficient_index_map[coefficient] = None
                    self.coefficient_len.append(None)
                # This is which coefficient in the original form the
                # current coefficient is.
                # Consider f*v*dx + g*v*ds, the full form contains two
                # coefficients, but each integral only requires one.
                coefficient_numbers.append(form_data.original_coefficient_positions[i])
        for i, coefficient in enumerate(coefficients):
            funarg, varexp = self._coefficient(coefficient, "w_%d" % i)
            self.coefficient_args.append(funarg)
            # Update the key from ufl coeff to gen variable
            self.variable_to_coefficient_index_map[varexp] = self.variable_to_coefficient_index_map.pop(coefficient)
        self.kernel.coefficient_numbers = tuple(coefficient_numbers)

    def set_coefficient_parts(self, varset):
        """Set kernel.coefficient_parts.
        :arg varset: a tuple of `gem.Variable`s that are actually used in the form.
        Call this method only after self.kernel.coefficient_numbers is set.
        """
        # Map gem.Variable -> (mixed coefficient local number, idx)
        variable_to_coefficient_index_map = self.variable_to_coefficient_index_map
        # Initialise list: [] for mixed coefficients and None for the others
        coefficient_parts = [None] * len(self.kernel.coefficient_numbers)
        for _, val in variable_to_coefficient_index_map.items():
            if val is not None:
                coefficient_parts[val[0]] = []
        for var in varset:
            if var in variable_to_coefficient_index_map and variable_to_coefficient_index_map[var]:
                num = variable_to_coefficient_index_map[var][0]
                idx = variable_to_coefficient_index_map[var][1]
                coefficient_parts[num].append(idx)
        coefficient_parts = [sorted(parts) if parts is not None else None for parts in coefficient_parts]
        self.kernel.coefficient_parts = coefficient_parts

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface."""
        knl = self.kernel
        knl.oriented, knl.needs_cell_sizes, knl.tabulations = check_requirements(ir)

    def construct_kernel(self, name, impero_c, precision, index_names, quadrature_rule):
        """Construct a fully built :class:`Kernel`.

        This function contains the logic for building the argument
        list for assembly kernels.

        :arg name: function name
        :arg impero_c: ImperoC tuple with Impero AST and other data
        :arg precision: floating-point precision for printing
        :arg index_names: pre-assigned index names
        :arg quadrature rule: quadrature rule

        :returns: :class:`Kernel` object
        """
        body = generate_coffee(impero_c, index_names, precision, self.scalar_type)

        args = [self.local_tensor, ] + self.coordinates_arg
        if self.kernel.oriented:
            args.append(cell_orientations_coffee_arg)
        if self.kernel.needs_cell_sizes:
            args.append(self.cell_sizes_arg)
        if self.kernel.coefficient_parts is None:
            args.extend(self.coefficient_args)
        else:
            count = 0
            for i, coeff_parts in enumerate(self.kernel.coefficient_parts):
                if self.coefficient_len[i] is None:
                    args.append(self.coefficient_args[count])
                    count += 1
                else:
                    for j in range(self.coefficient_len[i]):
                        if j in coeff_parts:
                            args.append(self.coefficient_args[count])
                        count += 1
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


def prepare_coefficient(coefficient, name, scalar_type, interior_facet=False):
    """Bridges the kernel interface and the GEM abstraction for
    Coefficients.

    :arg coefficient: UFL Coefficient
    :arg name: unique name to refer to the Coefficient in the kernel
    :arg interior_facet: interior facet integral?
    :returns: (funarg, expression)
         funarg     - :class:`coffee.Decl` function argument
         expression - GEM expression referring to the Coefficient
                      values
    """
    assert isinstance(interior_facet, bool)

    if coefficient.ufl_element().family() == 'Real':
        # Constant
        funarg = coffee.Decl(scalar_type, coffee.Symbol(name),
                             pointers=[("restrict",)],
                             qualifiers=["const"])
        varexp = gem.Variable(name, (None,))
        expression = gem.reshape(varexp,
                                 coefficient.ufl_shape)

        return funarg, varexp, expression

    finat_element = create_element(coefficient.ufl_element())
    shape = finat_element.index_shape
    size = numpy.prod(shape, dtype=int)

    funarg = coffee.Decl(scalar_type, coffee.Symbol(name),
                         pointers=[("restrict",)],
                         qualifiers=["const"])

    if not interior_facet:
        varexp = gem.Variable(name, (size,))
        expression = gem.reshape(varexp, shape)
    else:
        varexp = gem.Variable(name, (2 * size,))
        plus = gem.view(varexp, slice(size))
        minus = gem.view(varexp, slice(size, 2 * size))
        expression = (gem.reshape(plus, shape),
                      gem.reshape(minus, shape))
    return funarg, varexp, expression


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
