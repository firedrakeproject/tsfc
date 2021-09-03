import time
import sys
from functools import partial
from itertools import chain

import ufl
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.algorithms.analysis import has_type
from ufl.classes import Form, GeometricQuantity, Coefficient, FunctionSpace
from ufl.log import GREEN

import gem
from gem.utils import cached_property
import gem.impero_utils as impero_utils

import finat

from tsfc import fem, ufl_utils
from tsfc.logging import logger
from tsfc.parameters import default_parameters, is_complex
from tsfc.ufl_utils import apply_mapping

# To handle big forms. The various transformations might need a deeper stack
sys.setrecursionlimit(3000)


class TSFCFormData(object):
    r"""Mimic `ufl.FormData`.

    :arg form_data_tuple: A tuple of `ufl.FormData`s.
    :arg extraarg_tuple: A tuple of extra `ufl.Argument`s
        corresponding to form_data_tuple. These extra
        arguments are eventually replaced by the user with
        the associated functions in function_tuple after
        compiling UFL but before compiling gem. These
        arguments thus do not contribute to the rank of the form.
    :arg function_tuple: A tuple of functions corresponding
        to extraarg_tuple.
    :arg original_form: The form from which forms for
        `ufl.Formdata`s were extracted.
    :diagonal: A flag for diagonal matrix assembly.

    This class mimics `ufl.FormData`, but is to contain minimum
    information required by TSFC. This class can also combine
    multiple `ufl.FormData`s associated with forms that are
    extracted form a single form (`original_form`) in one way or
    another. This is useful when the user wants to insert some
    form specific operations after compiling ufl and before
    compiling gem:

                  split        compile     operations      compile
                                 UFL                         gem
                                             op_0
                     +--- form_0 ---- gem_0' ---- gem_0 ---+
                     |                       op_1          |
                     +--- form_1 ---- gem_1' ---- gem_1 ---+
    original_form ---|                                     |---> Kernel
                     |      :           :          :       |
                     |                       op_N          |
                     +--- form_N ---- gem_N' ---- gem_N ---+

    After preprocessing `ufl.FormData`s here:
        * Only essential information about the `ufl.FormData`s is retained.
        * TSFC can forget `ufl.FormData.original_form`,
        * `KernelBuilder`s only need to deal with raw `ufl.Coefficient`s.

    Illustration of the structures.
                                                            _____________TSFCFormData____________
                 ____________________     __________       | ________  ________         ________ |
                |Integral||Integral|       |Integral|      ||        ||        |       |        ||
    FormData 0  |  Data  ||  Data  |  ...  |  Data  |      ||        ||        |       |        ||
                |____0___||____1___|_     _|____M___|      ||  TSFC  ||  TSFC  |       |  TSFC  ||
                 ____________________     __________       ||Integral||Integral|       |Integral||
                |Integral||Integral|       |Integral|      ||  Data  ||  Data  |       |  Data  ||
    FormData 1  |  Data  ||  Data  |  ...  |  Data  |      ||    0   ||    1   |       |    M   ||
                |____0___||____1___|_     _|____M___| ---> ||        ||        |  ...  |        ||
                                                           |                                     |
            :                :                             |     :         :                :    |
                 ____________________     __________       |                                     |
                |Integral||Integral|       |Integral|      ||        ||        |       |        ||
    FormData N  |  Data  ||  Data  |  ...  |  Data  |      ||        ||        |       |        ||
                |____0___||____1___|_     _|____M___|      ||________||________|       |________||
                                                           |_____________________________________|
    """
    def __init__(self, form_data_tuple, extraarg_tuple, function_tuple, original_form, diagonal):
        arguments = set()
        for fd, extraarg in zip(form_data_tuple, extraarg_tuple):
            args = []
            for arg in fd.preprocessed_form.arguments():
                if arg not in extraarg:
                    args.append(arg)
            arguments.update((tuple(args), ))
        if len(arguments) != 1:
            raise ValueError("Found inconsistent sets of arguments in `FormData`s.")
        self.arguments, = tuple(arguments)
        # Gather all coefficients.
        # If a form contains extra arguments, those will be replaced by corresponding functions
        # after compiling UFL, so these functions must be included here, too.
        reduced_coefficients_set = set(c for fd in form_data_tuple for c in fd.reduced_coefficients)
        reduced_coefficients_set.update(chain(*function_tuple))
        reduced_coefficients = sorted(reduced_coefficients_set, key=lambda c: c.count())
        # Reconstruct `ufl.Coefficinet`s with count starting at 0.
        function_replace_map = {}
        for i, func in enumerate(reduced_coefficients):
            for fd in form_data_tuple:
                if func in fd.function_replace_map:
                    coeff = fd.function_replace_map[func]
                    new_coeff = Coefficient(coeff.ufl_function_space(), count=i)
                    function_replace_map[func] = new_coeff
                    break
            else:
                ufl_function_space = FunctionSpace(func.ufl_domain(), func.ufl_element())
                new_coeff = Coefficient(ufl_function_space, count=i)
                function_replace_map[func] = new_coeff
        self.reduced_coefficients = reduced_coefficients
        self.original_coefficient_positions = [i for i, f in enumerate(original_form.coefficients())
                                               if f in self.reduced_coefficients]
        self.function_replace_map = function_replace_map

        # Translate `ufl.IntegralData`s -> `TSFCIntegralData`.
        intg_data_info_dict = {}
        for form_data_index, form_data in enumerate(form_data_tuple):
            for intg_data in form_data.integral_data:
                domain = intg_data.domain
                integral_type = intg_data.integral_type
                subdomain_id = intg_data.subdomain_id
                key = (domain, integral_type, subdomain_id)
                # Add (intg_data, form_data, form_data_index).
                intg_data_info_dict.setdefault(key, []).append((intg_data, form_data, form_data_index))
        integral_data_list = []
        for key, intg_data_info in intg_data_info_dict.items():
            domain, _, _ = key
            domain_number = original_form.domain_numbering()[domain]
            integral_data_list.append(TSFCIntegralData(key, intg_data_info,
                                                       self, domain_number, function_tuple, len(form_data_tuple)))
        self.integral_data = tuple(integral_data_list)


class TSFCIntegralData(object):
    r"""Mimics `ufl.IntegralData`.

    :arg integral_data_key: (domain, integral_type, subdomain_id) tuple.
    :arg integral_data_info: A tuple of the lists of integral_data,
        form_data, and form_data_index.
    :arg tsfc_form_data: The `TSFCFormData` that is to contain this
        `TSFCIntegralData` object.
    :arg domain_number: The domain number associated with `domain`.
    :arg function_tuple: A tuple of functions.

    After preprocessing `ufl.IntegralData`s here:
        * Only essential information about the `ufl.IntegralData`s is retained.
        * TSFC can forget `ufl.IntegralData.enabled_coefficients`,
    """
    def __init__(self, integral_data_key, intg_data_info, tsfc_form_data, domain_number, function_tuple, n):
        # Gather/preprocess integrals.
        integrals_list = [[] for _ in range(n)]
        functions = set()
        for intg_data, form_data, form_data_index in intg_data_info:
            for integral in intg_data.integrals:
                integrand = integral.integrand()
                # Replace functions with Coefficients here.
                integrand = ufl.replace(integrand, tsfc_form_data.function_replace_map)
                new_integral = integral.reconstruct(integrand=integrand)
                integrals_list[form_data_index].append(new_integral)
            # Gather functions that are enabled in this `TSFCIntegralData`.
            functions.update(f for f, enabled in zip(form_data.reduced_coefficients, intg_data.enabled_coefficients) if enabled)
            functions.update(function_tuple[form_data_index])
        self.integrals_tuple = tuple(integrals_list)
        arguments = tsfc_form_data.arguments
        # This is which coefficient in the original form the
        # current coefficient is.
        # Ex:
        # original_form.coefficients()       : f0, f1, f2, f3, f4, f5
        # tsfc_form_data.reduced_coefficients: f1, f2, f3, f5
        # functions                          : f1, f5
        # self.coefficients                  : c1, c5
        # self.coefficent_numbers            :  1,  5
        functions = sorted(functions, key=lambda c: c.count())
        coefficients = tuple(tsfc_form_data.function_replace_map[f] for f in functions)
        coefficient_numbers = tuple(tsfc_form_data.original_coefficient_positions[tsfc_form_data.reduced_coefficients.index(f)] for f in functions)

        self.info = TSFCIntegralDataInfo(*integral_data_key, domain_number,
                                         arguments,
                                         coefficients, coefficient_numbers)

    @cached_property
    def integrals(self):
        return list(chain(*self.integrals_tuple))

    def __getattr__(self, attr):
        return getattr(self.info, attr)


class TSFCIntegralDataInfo(object):
    __slots__ = ("domain", "integral_type", "subdomain_id", "domain_number",
                 "arguments",
                 "coefficients", "coefficient_numbers")
    """A bag of high-level information of :class:`~.TSFCIntegralData`.

    :arg domain: The mesh.
    :arg integral_type: The type of integral.
    :arg subdomain_id: What is the subdomain id for this kernel.
    :arg domain_number: Which domain number in the original form
        does this kernel correspond to (can be used to index into
        original_form.ufl_domains() to get the correct domain).
    :arg coefficients: A list of coefficients.
    :arg coefficient_numbers: A list of which coefficients from the
        form the kernel needs.
    """
    def __init__(self, domain, integral_type, subdomain_id, domain_number,
                 arguments,
                 coefficients, coefficient_numbers):
        self.domain = domain
        self.integral_type = integral_type
        self.subdomain_id = subdomain_id
        self.arguments = arguments
        self.coefficients = coefficients
        self.domain_number = domain_number
        self.coefficient_numbers = coefficient_numbers
        super(TSFCIntegralDataInfo, self).__init__()


def compile_form(form, prefix="form", parameters=None, interface=None, coffee=True, diagonal=False):
    """Compiles a UFL form into a set of assembly kernels.

    :arg form: UFL form
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :arg coffee: compile coffee kernel instead of loopy kernel
    :arg diagonal: Are we building a kernel for the diagonal of a rank-2 element tensor?
    :returns: list of kernels
    """
    cpu_time = time.time()

    assert isinstance(form, Form)

    parameters = preprocess_parameters(parameters)

    # Determine whether in complex mode:
    complex_mode = parameters and is_complex(parameters.get("scalar_type"))

    form_data = ufl_utils.compute_form_data(form, complex_mode=complex_mode)
    if interface:
        interface = partial(interface, function_replace_map=form_data.function_replace_map)
    tsfc_form_data = TSFCFormData((form_data, ), ((), ), ((), ), form_data.original_form, diagonal)

    logger.info(GREEN % "compute_form_data finished in %g seconds.", time.time() - cpu_time)

    kernels = []
    for tsfc_integral_data in tsfc_form_data.integral_data:
        start = time.time()
        kernel = compile_integral(tsfc_integral_data, prefix, parameters, interface=interface, coffee=coffee, diagonal=diagonal)
        if kernel is not None:
            kernels.append(kernel)
        logger.info(GREEN % "compile_integral finished in %g seconds.", time.time() - start)

    logger.info(GREEN % "TSFC finished in %g seconds.", time.time() - cpu_time)
    return kernels


def compile_integral(tsfc_integral_data, prefix, parameters, interface, coffee, *, diagonal=False):
    """Compiles a UFL integral into an assembly kernel.

    :arg tsfc_integral_data: TSFCIntegralData
    :arg prefix: kernel name will start with this string
    :arg parameters: parameters object
    :arg interface: backend module for the kernel interface
    :arg diagonal: Are we building a kernel for the diagonal of a rank-2 element tensor?
    :returns: a kernel constructed by the kernel interface
    """
    if interface is None:
        if coffee:
            import tsfc.kernel_interface.firedrake as firedrake_interface_coffee
            interface = firedrake_interface_coffee.KernelBuilder
        else:
            # Delayed import, loopy is a runtime dependency
            import tsfc.kernel_interface.firedrake_loopy as firedrake_interface_loopy
            interface = firedrake_interface_loopy.KernelBuilder

    builder = interface(tsfc_integral_data.info,
                        parameters["scalar_type_c"] if coffee else parameters["scalar_type"],
                        parameters["scalar_type"],
                        diagonal=diagonal)
    ctx = builder.create_context()
    # Compile UFL -> gem
    for integral in tsfc_integral_data.integrals:
        params = parameters.copy()
        params.update(integral.metadata())  # integral metadata overrides
        integrand_exprs = builder.compile_ufl(integral.integrand(), params, ctx)
        integral_exprs = builder.construct_integrals(integrand_exprs, params)
        builder.stash_integrals(integral_exprs, params, ctx)
    # Compile gem -> kernel
    kernel_name = "%s_%s_integral_%s" % (prefix, tsfc_integral_data.integral_type, tsfc_integral_data.subdomain_id)
    kernel_name = kernel_name.replace("-", "_")  # Handle negative subdomain_id
    return builder.construct_kernel(kernel_name, ctx)


def preprocess_parameters(parameters):
    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _
    # Remove these here, they're handled later on.
    if parameters.get("quadrature_degree") in ["auto", "default", None, -1, "-1"]:
        del parameters["quadrature_degree"]
    if parameters.get("quadrature_rule") in ["auto", "default", None]:
        del parameters["quadrature_rule"]
    return parameters


def compile_expression_dual_evaluation(expression, to_element, *,
                                       domain=None, interface=None,
                                       parameters=None):
    """Compile a UFL expression to be evaluated against a compile-time known reference element's dual basis.

    Useful for interpolating UFL expressions into e.g. N1curl spaces.

    :arg expression: UFL expression
    :arg to_element: A FInAT element for the target space
    :arg domain: optional UFL domain the expression is defined on (required when expression contains no domain).
    :arg interface: backend module for the kernel interface
    :arg parameters: parameters object
    :returns: Loopy-based ExpressionKernel object.
    """
    if parameters is None:
        parameters = default_parameters()
    else:
        _ = default_parameters()
        _.update(parameters)
        parameters = _

    # Determine whether in complex mode
    complex_mode = is_complex(parameters["scalar_type"])

    # Find out which mapping to apply
    try:
        mapping, = set((to_element.mapping,))
    except ValueError:
        raise NotImplementedError("Don't know how to interpolate onto zany spaces, sorry")
    expression = apply_mapping(expression, mapping, domain)

    # Apply UFL preprocessing
    expression = ufl_utils.preprocess_expression(expression,
                                                 complex_mode=complex_mode)

    # Initialise kernel builder
    if interface is None:
        # Delayed import, loopy is a runtime dependency
        from tsfc.kernel_interface.firedrake_loopy import ExpressionKernelBuilder as interface

    builder = interface(parameters["scalar_type"])
    arguments = extract_arguments(expression)
    argument_multiindices = tuple(builder.create_element(arg.ufl_element()).get_indices()
                                  for arg in arguments)

    # Replace coordinates (if any) unless otherwise specified by kwarg
    if domain is None:
        domain = expression.ufl_domain()
    assert domain is not None

    # Collect required coefficients
    first_coefficient_fake_coords = False
    coefficients = extract_coefficients(expression)
    if has_type(expression, GeometricQuantity) or any(fem.needs_coordinate_mapping(c.ufl_element()) for c in coefficients):
        # Create a fake coordinate coefficient for a domain.
        coords_coefficient = ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))
        builder.domain_coordinate[domain] = coords_coefficient
        builder.set_cell_sizes(domain)
        coefficients = [coords_coefficient] + coefficients
        first_coefficient_fake_coords = True
    builder.set_coefficients(coefficients)

    # Split mixed coefficients
    expression = ufl_utils.split_coefficients(expression, builder.coefficient_split, )

    # Set up kernel config for translation of UFL expression to gem
    kernel_cfg = dict(interface=builder,
                      ufl_cell=domain.ufl_cell(),
                      # FIXME: change if we ever implement
                      # interpolation on facets.
                      integral_type="cell",
                      argument_multiindices=argument_multiindices,
                      index_cache={},
                      scalar_type=parameters["scalar_type"])

    # Allow interpolation onto QuadratureElements to refer to the quadrature
    # rule they represent
    if isinstance(to_element, finat.QuadratureElement):
        kernel_cfg["quadrature_rule"] = to_element._rule

    # Create callable for translation of UFL expression to gem
    fn = DualEvaluationCallable(expression, kernel_cfg)

    # Get the gem expression for dual evaluation and corresponding basis
    # indices needed for compilation of the expression
    evaluation, basis_indices = to_element.dual_evaluation(fn)

    # Build kernel body
    return_indices = basis_indices + tuple(chain(*argument_multiindices))
    return_shape = tuple(i.extent for i in return_indices)
    return_var = gem.Variable('A', return_shape)
    return_expr = gem.Indexed(return_var, return_indices)

    # TODO: one should apply some GEM optimisations as in assembly,
    # but we don't for now.
    evaluation, = impero_utils.preprocess_gem([evaluation])
    impero_c = impero_utils.compile_gem([(return_expr, evaluation)], return_indices)
    index_names = dict((idx, "p%d" % i) for (i, idx) in enumerate(basis_indices))
    # Handle kernel interface requirements
    builder.register_requirements([evaluation])
    builder.set_output(return_var)
    # Build kernel tuple
    return builder.construct_kernel(impero_c, index_names, first_coefficient_fake_coords)


class DualEvaluationCallable(object):
    """
    Callable representing a function to dual evaluate.

    When called, this takes in a
    :class:`finat.point_set.AbstractPointSet` and returns a GEM
    expression for evaluation of the function at those points.

    :param expression: UFL expression for the function to dual evaluate.
    :param kernel_cfg: A kernel configuration for creation of a
        :class:`GemPointContext` or a :class:`PointSetContext`

    Not intended for use outside of
    :func:`compile_expression_dual_evaluation`.
    """
    def __init__(self, expression, kernel_cfg):
        self.expression = expression
        self.kernel_cfg = kernel_cfg

    def __call__(self, ps):
        """The function to dual evaluate.

        :param ps: The :class:`finat.point_set.AbstractPointSet` for
            evaluating at
        :returns: a gem expression representing the evaluation of the
            input UFL expression at the given point set ``ps``.
            For point set points with some shape ``(*value_shape)``
            (i.e. ``()`` for scalar points ``(x)`` for vector points
            ``(x, y)`` for tensor points etc) then the gem expression
            has shape ``(*value_shape)`` and free indices corresponding
            to the input :class:`finat.point_set.AbstractPointSet`'s
            free indices alongside any input UFL expression free
            indices.
        """

        if not isinstance(ps, finat.point_set.AbstractPointSet):
            raise ValueError("Callable argument not a point set!")

        # Avoid modifying saved kernel config
        kernel_cfg = self.kernel_cfg.copy()

        if isinstance(ps, finat.point_set.UnknownPointSet):
            # Run time known points
            kernel_cfg.update(point_indices=ps.indices, point_expr=ps.expression)
            # GemPointContext's aren't allowed to have quadrature rules
            kernel_cfg.pop("quadrature_rule", None)
            translation_context = fem.GemPointContext(**kernel_cfg)
        else:
            # Compile time known points
            kernel_cfg.update(point_set=ps)
            translation_context = fem.PointSetContext(**kernel_cfg)

        gem_expr, = fem.compile_ufl(self.expression, translation_context, point_sum=False)
        # In some cases ps.indices may be dropped from expr, but nothing
        # new should now appear
        argument_multiindices = kernel_cfg["argument_multiindices"]
        assert set(gem_expr.free_indices) <= set(chain(ps.indices, *argument_multiindices))

        return gem_expr
