import collections
import operator
from functools import reduce
from itertools import chain

import numpy

import coffee.base as coffee

import gem

from gem.utils import cached_property
import gem.impero_utils as impero_utils

from tsfc import fem, ufl_utils
from tsfc.kernel_interface import KernelInterface


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

        # Subspaces
        self.subspace_map = {}

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

    def subspace(self, ufl_subspace, restriction):
        """A function that maps :class:`ufl.Subspace`s to GEM
        expressions."""
        kernel_arg = self.subspace_map[ufl_subspace]
        if ufl_subspace.ufl_element().family() == 'Real':
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

    def compile_ufl(self, integrand, params, kernel_config):
        # Split Coefficients
        if self.coefficient_split:
            integrand = ufl_utils.split_coefficients(integrand, self.coefficient_split, self.subspace_split)
        integrand = ufl_utils.split_subspaces(integrand, self.subspace_split)
        # Compile: ufl -> gem
        quad_rule = params["quadrature_rule"]
        config = kernel_config['fem_config'].copy()
        config.update(quadrature_rule=quad_rule)
        expressions = fem.compile_ufl(integrand,
                                      interior_facet=self.interior_facet,
                                      **config)
        self.quadrature_indices.extend(quad_rule.point_set.indices)
        return expressions

    def compile_gem(self, kernel_config):
        # Finalise mode representations into a set of assignments
        mode_irs = kernel_config["mode_irs"]
        index_cache = kernel_config['fem_config']['index_cache']

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
        kernel_config['oriented'] = oriented
        kernel_config['needs_cell_sizes'] = needs_cell_sizes
        kernel_config['tabulations'] =tabulations

        # Construct ImperoC
        assignments = list(zip(return_variables, expressions))
        index_ordering = _get_index_ordering(self.quadrature_indices, return_variables)
        try:
            impero_c = impero_utils.compile_gem(assignments, index_ordering, remove_zeros=True)
        except impero_utils.NoopError:
            impero_c = None
        return impero_c

    def construct_integrals(self, expressions, params, kernel_config):
        mode = pick_mode(params["mode"])
        return mode.Integrals(expressions,
                              params["quadrature_rule"].point_set.indices,
                              self.argument_multiindices,
                              params)

    def stash_integrals(self, reps, params, kernel_config):
        mode = pick_mode(params["mode"])
        mode_irs = kernel_config["mode_irs"]
        return_variables = self.return_variables
        mode_irs.setdefault(mode, collections.OrderedDict())
        for var, rep in zip(return_variables, reps):
            mode_irs[mode].setdefault(var, []).append(rep)


def _get_index_ordering(quadrature_indices, return_variables):
    split_argument_indices = tuple(chain(*[var.index_ordering()
                                           for var in return_variables]))
    return tuple(quadrature_indices) + split_argument_indices


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
