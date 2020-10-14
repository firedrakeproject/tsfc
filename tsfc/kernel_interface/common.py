import collections
import operator
from functools import reduce
from itertools import chain

import numpy
from numpy import asarray

import coffee.base as coffee

from ufl.utils.sequences import max_degree

import gem

from gem.utils import cached_property
import gem.impero_utils as impero_utils

from tsfc import fem, ufl_utils
from tsfc.kernel_interface import KernelInterface
from tsfc.finatinterface import as_fiat_cell
from tsfc.logging import logger

from FIAT.reference_element import TensorProductCell

from finat.quadrature import AbstractQuadratureRule, make_quadrature


class KernelBuilderBase(KernelInterface):
    """Helper class for building local assembly kernels."""

    def __init__(self, scalar_type, interior_facet=False):
        """Initialise a kernel builder.

        :arg interior_facet: kernel accesses two cells
        """
        assert isinstance(interior_facet, bool)
        self.scalar_type = scalar_type
        self.interior_facet = interior_facet

        self.prepare = []
        self.finalise = []

        # Coordinates
        self.domain_coordinate = {}

        # Coefficients
        self.coefficient_map = {}

    @cached_property
    def unsummed_coefficient_indices(self):
        return frozenset()

    def coordinate(self, domain):
        return self.domain_coordinate[domain]

    def coefficient(self, ufl_coefficient, restriction):
        """A function that maps :class:`ufl.Coefficient`s to GEM
        expressions."""
        kernel_arg = self.coefficient_map[ufl_coefficient]
        if ufl_coefficient.ufl_element().family() == 'Real':
            return kernel_arg
        elif not self.interior_facet:
            return kernel_arg
        else:
            return kernel_arg[{'+': 0, '-': 1}[restriction]]

    def cell_orientation(self, restriction):
        """Cell orientation as a GEM expression."""
        f = {None: 0, '+': 0, '-': 1}[restriction]
        # Assume self._cell_orientations tuple is set up at this point.
        co_int = self._cell_orientations[f]
        return gem.Conditional(gem.Comparison("==", co_int, gem.Literal(1)),
                               gem.Literal(-1),
                               gem.Conditional(gem.Comparison("==", co_int, gem.Zero()),
                                               gem.Literal(1),
                                               gem.Literal(numpy.nan)))

    def cell_size(self, restriction):
        if not hasattr(self, "_cell_sizes"):
            raise RuntimeError("Haven't called set_cell_sizes")
        if self.interior_facet:
            return self._cell_sizes[{'+': 0, '-': 1}[restriction]]
        else:
            return self._cell_sizes

    def entity_number(self, restriction):
        """Facet or vertex number as a GEM index."""
        # Assume self._entity_number dict is set up at this point.
        return self._entity_number[restriction]

    def apply_glue(self, prepare=None, finalise=None):
        """Append glue code for operations that are not handled in the
        GEM abstraction.

        Current uses: mixed interior facet mess

        :arg prepare: code snippets to be prepended to the kernel
        :arg finalise: code snippets to be appended to the kernel
        """
        if prepare is not None:
            self.prepare.extend(prepare)
        if finalise is not None:
            self.finalise.extend(finalise)

    def construct_kernel(self, name, args, body):
        """Construct a COFFEE function declaration with the
        accumulated glue code.

        :arg name: function name
        :arg args: function argument list
        :arg body: function body (:class:`coffee.Block` node)
        :returns: :class:`coffee.FunDecl` object
        """
        assert isinstance(body, coffee.Block)
        body_ = coffee.Block(self.prepare + body.children + self.finalise)
        return coffee.FunDecl("void", name, args, body_, pred=["static", "inline"])

    def register_requirements(self, ir):
        """Inspect what is referenced by the IR that needs to be
        provided by the kernel interface.

        :arg ir: multi-root GEM expression DAG
        """
        # Nothing is required by default
        pass


class KernelBuilderMixin(object):

    def compile_ufl(self, integrand, params, argument_multiindices=None):
        # Split Coefficients
        if self.coefficient_split:
            integrand = ufl_utils.split_coefficients(integrand, self.coefficient_split)
        # Compile: ufl -> gem
        functions = list(self.arguments) + [self.coordinate(self.integral_data.domain)] + list(self.integral_data.coefficients)
        _set_quad_rule(params, self.integral_data.domain.ufl_cell(), self.integral_data.integral_type, functions)
        quad_rule = params["quadrature_rule"]
        config = self.fem_config.copy()
        config.update(quadrature_rule=quad_rule)
        config['argument_multiindices'] = argument_multiindices or self.argument_multiindices
        expressions = fem.compile_ufl(integrand,
                                      interior_facet=self.interior_facet,
                                      **config)
        self.quadrature_indices.extend(quad_rule.point_set.indices)
        return expressions

    def compile_gem(self):
        # Finalise mode representations into a set of assignments
        mode_irs = self.mode_irs
        index_cache = self.fem_config['index_cache']

        assignments = []
        for mode, var_reps in mode_irs.items():
            assignments.extend(mode.flatten(var_reps.items(), index_cache))

        if assignments:
            return_variables, expressions = zip(*assignments)
        else:
            return_variables = []
            expressions = []

        # Need optimised roots
        options = dict(reduce(operator.and_,
                              [mode.finalise_options.items()
                               for mode in mode_irs.keys()]))
        expressions = impero_utils.preprocess_gem(expressions, **options)

        # Let the kernel interface inspect the optimised IR to register
        # what kind of external data is required (e.g., cell orientations,
        # cell sizes, etc.).
        oriented, needs_cell_sizes, tabulations = self.register_requirements(expressions)

        # Construct ImperoC
        assignments = list(zip(return_variables, expressions))
        index_ordering = _get_index_ordering(self.quadrature_indices, return_variables)
        try:
            impero_c = impero_utils.compile_gem(assignments, index_ordering, remove_zeros=True)
        except impero_utils.NoopError:
            impero_c = None
        return impero_c, oriented, needs_cell_sizes, tabulations

    def construct_integrals(self, expressions, params):
        mode = pick_mode(params["mode"])
        return mode.Integrals(expressions,
                              params["quadrature_rule"].point_set.indices,
                              self.argument_multiindices,
                              params)

    def stash_integrals(self, reps, params):
        mode = pick_mode(params["mode"])
        mode_irs = self.mode_irs
        return_variables = self.return_variables
        mode_irs.setdefault(mode, collections.OrderedDict())
        for var, rep in zip(return_variables, reps):
            mode_irs[mode].setdefault(var, []).append(rep)

    @cached_property
    def fem_config(self):
        # Map from UFL FiniteElement objects to multiindices.  This is
        # so we reuse Index instances when evaluating the same coefficient
        # multiple times with the same table.
        #
        # We also use the same dict for the unconcatenate index cache,
        # which maps index objects to tuples of multiindices.  These two
        # caches shall never conflict as their keys have different types
        # (UFL finite elements vs. GEM index objects).
        #
        # -> fem_config['index_cache']
        integral_type = self.integral_data.integral_type
        cell = self.integral_data.domain.ufl_cell()
        fiat_cell = as_fiat_cell(cell)
        integration_dim, entity_ids = lower_integral_type(fiat_cell, integral_type)
        return dict(interface=self,
                    ufl_cell=cell,
                    integral_type=integral_type,
                    integration_dim=integration_dim,
                    entity_ids=entity_ids,
                    index_cache={},
                    scalar_type=self.fem_scalar_type)


def _get_index_ordering(quadrature_indices, return_variables):
    split_argument_indices = tuple(chain(*[var.index_ordering()
                                           for var in return_variables]))
    return tuple(quadrature_indices) + split_argument_indices


def _set_quad_rule(params, cell, integral_type, functions):
    # Check if the integral has a quad degree attached, otherwise use
    # the estimated polynomial degree attached by compute_form_data
    try:
        quadrature_degree = params["quadrature_degree"]
    except KeyError:
        quadrature_degree = params["estimated_polynomial_degree"]
        function_degrees = [f.ufl_function_space().ufl_element().degree() for f in functions]
        if all((asarray(quadrature_degree) > 10 * asarray(degree)).all()
               for degree in function_degrees):
            logger.warning("Estimated quadrature degree %s more "
                           "than tenfold greater than any "
                           "argument/coefficient degree (max %s)",
                           quadrature_degree, max_degree(function_degrees))
    if params.get("quadrature_rule") == "default":
        del params["quadrature_rule"]
    try:
        quad_rule = params["quadrature_rule"]
    except KeyError:
        fiat_cell = as_fiat_cell(cell)
        integration_dim, _ = lower_integral_type(fiat_cell, integral_type)
        integration_cell = fiat_cell.construct_subelement(integration_dim)
        quad_rule = make_quadrature(integration_cell, quadrature_degree)
        params["quadrature_rule"] = quad_rule

    if not isinstance(quad_rule, AbstractQuadratureRule):
        raise ValueError("Expected to find a QuadratureRule object, not a %s" %
                         type(quad_rule))


def lower_integral_type(fiat_cell, integral_type):
    """Lower integral type into the dimension of the integration
    subentity and a list of entity numbers for that dimension.

    :arg fiat_cell: FIAT reference cell
    :arg integral_type: integral type (string)
    """
    vert_facet_types = ['exterior_facet_vert', 'interior_facet_vert']
    horiz_facet_types = ['exterior_facet_bottom', 'exterior_facet_top', 'interior_facet_horiz']

    dim = fiat_cell.get_dimension()
    if integral_type == 'cell':
        integration_dim = dim
    elif integral_type in ['exterior_facet', 'interior_facet']:
        if isinstance(fiat_cell, TensorProductCell):
            raise ValueError("{} integral cannot be used with a TensorProductCell; need to distinguish between vertical and horizontal contributions.".format(integral_type))
        integration_dim = dim - 1
    elif integral_type == 'vertex':
        integration_dim = 0
    elif integral_type in vert_facet_types + horiz_facet_types:
        # Extrusion case
        if not isinstance(fiat_cell, TensorProductCell):
            raise ValueError("{} integral requires a TensorProductCell.".format(integral_type))
        basedim, extrdim = dim
        assert extrdim == 1

        if integral_type in vert_facet_types:
            integration_dim = (basedim - 1, 1)
        elif integral_type in horiz_facet_types:
            integration_dim = (basedim, 0)
    else:
        raise NotImplementedError("integral type %s not supported" % integral_type)

    if integral_type == 'exterior_facet_bottom':
        entity_ids = [0]
    elif integral_type == 'exterior_facet_top':
        entity_ids = [1]
    else:
        entity_ids = list(range(len(fiat_cell.get_topology()[integration_dim])))

    return integration_dim, entity_ids


def pick_mode(mode):
    "Return one of the specialized optimisation modules from a mode string."
    try:
        from firedrake_citations import Citations
        cites = {"vanilla": ("Homolya2017", ),
                 "coffee": ("Luporini2016", "Homolya2017", ),
                 "spectral": ("Luporini2016", "Homolya2017", "Homolya2017a"),
                 "tensor": ("Kirby2006", "Homolya2017", )}
        for c in cites[mode]:
            Citations().register(c)
    except ImportError:
        pass
    if mode == "vanilla":
        import tsfc.vanilla as m
    elif mode == "coffee":
        import tsfc.coffee_mode as m
    elif mode == "spectral":
        import tsfc.spectral as m
    elif mode == "tensor":
        import tsfc.tensor as m
    else:
        raise ValueError("Unknown mode: {}".format(mode))
    return m
