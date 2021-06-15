import collections
import time
import sys
from functools import partial
from itertools import chain

from numpy import asarray, allclose, isnan

import ufl
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.algorithms.analysis import has_type
from ufl.classes import Form, GeometricQuantity, Coefficient, FunctionSpace
from ufl.log import GREEN

import gem
import gem.impero_utils as impero_utils

import FIAT
from FIAT.functional import PointEvaluation

from finat.point_set import PointSet, UnknownPointSet
from finat.quadrature import AbstractQuadratureRule, make_quadrature, QuadratureRule
from finat.quadrature_element import QuadratureElement

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
                                                       self, domain_number, function_tuple))
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
    def __init__(self, integral_data_key, intg_data_info, tsfc_form_data, domain_number, function_tuple):
        self.domain, self.integral_type, self.subdomain_id = integral_data_key
        self.domain_number = domain_number
        # Gather/preprocess integrals.
        integrals = []
        _integral_index_to_form_data_index = []
        functions = set()
        for intg_data, form_data, form_data_index in intg_data_info:
            for integral in intg_data.integrals:
                integrand = integral.integrand()
                # Replace functions with Coefficients here.
                integrand = ufl.replace(integrand, tsfc_form_data.function_replace_map)
                new_integral = integral.reconstruct(integrand=integrand)
                integrals.append(new_integral)
                # Remember which form_data this integral is associated with.
                _integral_index_to_form_data_index.append(form_data_index)
            # Gather functions that are enabled in this `TSFCIntegralData`.
            functions.update(f for f, enabled in zip(form_data.reduced_coefficients, intg_data.enabled_coefficients) if enabled)
            functions.update(function_tuple[form_data_index])
        self.integrals = tuple(integrals)
        self._integral_index_to_form_data_index = _integral_index_to_form_data_index
        self.arguments = tsfc_form_data.arguments
        # This is which coefficient in the original form the
        # current coefficient is.
        # Ex:
        # original_form.coefficients()       : f0, f1, f2, f3, f4, f5
        # tsfc_form_data.reduced_coefficients: f1, f2, f3, f5
        # functions                          : f1, f5
        # self.coefficients                  : c1, c5
        # self.coefficent_numbers            :  1,  5
        functions = sorted(functions, key=lambda c: c.count())
        self.coefficients = tuple(tsfc_form_data.function_replace_map[f] for f in functions)
        self.coefficient_numbers = tuple(tsfc_form_data.original_coefficient_positions[tsfc_form_data.reduced_coefficients.index(f)] for f in functions)

    def integral_index_to_form_data_index(self, integral_index):
        r"""Return the form data index given an integral index."""
        return self._integral_index_to_form_data_index[integral_index]


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

    builder = interface(tsfc_integral_data,
                        parameters["scalar_type_c"] if coffee else parameters["scalar_type"],
                        parameters["scalar_type"],
                        diagonal=diagonal)
    # Compile UFL -> gem
    for integral in tsfc_integral_data.integrals:
        params = parameters.copy()
        params.update(integral.metadata())  # integral metadata overrides
        integrand_exprs = builder.compile_ufl(integral.integrand(), params)
        integral_exprs = builder.construct_integrals(integrand_exprs, params)
        builder.stash_integrals(integral_exprs, params)
    # Compile gem -> kernel
    kernel_name = "%s_%s_integral_%s" % (prefix, tsfc_integral_data.integral_type, tsfc_integral_data.subdomain_id)
    kernel_name = kernel_name.replace("-", "_")  # Handle negative subdomain_id
    return builder.construct_kernel(kernel_name)


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
                                       parameters=None, coffee=False):
    """Compile a UFL expression to be evaluated against a compile-time known reference element's dual basis.

    Useful for interpolating UFL expressions into e.g. N1curl spaces.

    :arg expression: UFL expression
    :arg to_element: A FInAT element for the target space
    :arg domain: optional UFL domain the expression is defined on (required when expression contains no domain).
    :arg interface: backend module for the kernel interface
    :arg parameters: parameters object
    :arg coffee: compile coffee kernel instead of loopy kernel
    """
    import coffee.base as ast
    import loopy as lp

    # Just convert FInAT element to FIAT for now.
    # Dual evaluation in FInAT will bring a thorough revision.
    finat_to_element = to_element
    to_element = finat_to_element.fiat_equivalent

    if any(len(dual.deriv_dict) != 0 for dual in to_element.dual_basis()):
        raise NotImplementedError("Can only interpolate onto dual basis functionals without derivative evaluation, sorry!")

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
        mapping, = set(to_element.mapping())
    except ValueError:
        raise NotImplementedError("Don't know how to interpolate onto zany spaces, sorry")
    expression = apply_mapping(expression, mapping, domain)

    # Apply UFL preprocessing
    expression = ufl_utils.preprocess_expression(expression,
                                                 complex_mode=complex_mode)

    # Initialise kernel builder
    if interface is None:
        if coffee:
            import tsfc.kernel_interface.firedrake as firedrake_interface_coffee
            interface = firedrake_interface_coffee.ExpressionKernelBuilder
        else:
            # Delayed import, loopy is a runtime dependency
            import tsfc.kernel_interface.firedrake_loopy as firedrake_interface_loopy
            interface = firedrake_interface_loopy.ExpressionKernelBuilder

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

    # Translate to GEM
    kernel_cfg = dict(interface=builder,
                      ufl_cell=domain.ufl_cell(),
                      # FIXME: change if we ever implement
                      # interpolation on facets.
                      integral_type="cell",
                      argument_multiindices=argument_multiindices,
                      index_cache={},
                      scalar_type=parameters["scalar_type"])

    # A FInAT QuadratureElement with a runtime tabulated UnknownPointSet
    # point set is the target element on the reference cell for dual evaluation
    # where the points are specified at runtime. This special casing will not
    # be necessary when FInAT dual evaluation is done - the dual evaluation
    # method of every FInAT element will create the necessary gem code.
    from finat.tensorfiniteelement import TensorFiniteElement
    runtime_quadrature_rule = (
        isinstance(finat_to_element, QuadratureElement) or
        (
            isinstance(finat_to_element, TensorFiniteElement) and
            isinstance(finat_to_element.base_element, QuadratureElement)
        ) and
        isinstance(finat_to_element._rule.point_set, UnknownPointSet)
    )

    if all(isinstance(dual, PointEvaluation) for dual in to_element.dual_basis()):
        # This is an optimisation for point-evaluation nodes which
        # should go away once FInAT offers the interface properly
        config = kernel_cfg.copy()
        if runtime_quadrature_rule:
            # Until FInAT dual evaluation is done, FIAT
            # QuadratureElements with UnknownPointSet point sets
            # advertise NaNs as their points for each node in the dual
            # basis. This has to be manually replaced with the real
            # UnknownPointSet point set used to create the
            # QuadratureElement rule.
            point_set = finat_to_element._rule.point_set
            config.update(point_indices=point_set.indices, point_expr=point_set.expression)
            context = fem.GemPointContext(**config)
        else:
            qpoints = []
            # Everything is just a point evaluation.
            for dual in to_element.dual_basis():
                ptdict = dual.get_point_dict()
                qpoint, = ptdict.keys()
                (qweight, component), = ptdict[qpoint]
                assert allclose(qweight, 1.0)
                assert component == ()
                qpoints.append(qpoint)
            point_set = PointSet(qpoints)
            config.update(point_set=point_set)

            # Allow interpolation onto QuadratureElements to refer to the quadrature
            # rule they represent
            if isinstance(to_element, FIAT.QuadratureElement):
                assert allclose(asarray(qpoints), asarray(to_element._points))
                quad_rule = QuadratureRule(point_set, to_element._weights)
                config["quadrature_rule"] = quad_rule

            context = fem.PointSetContext(**config)

        expr, = fem.compile_ufl(expression, context, point_sum=False)
        # In some cases point_set.indices may be dropped from expr, but nothing
        # new should now appear
        assert set(expr.free_indices) <= set(chain(point_set.indices, *argument_multiindices))
        shape_indices = tuple(gem.Index() for _ in expr.shape)
        basis_indices = point_set.indices
        ir = gem.Indexed(expr, shape_indices)
    else:
        # This is general code but is more unrolled than necssary.
        dual_expressions = []   # one for each functional
        broadcast_shape = len(expression.ufl_shape) - len(to_element.value_shape())
        shape_indices = tuple(gem.Index() for _ in expression.ufl_shape[:broadcast_shape])
        expr_cache = {}         # Sharing of evaluation of the expression at points
        for dual in to_element.dual_basis():
            pts = tuple(sorted(dual.get_point_dict().keys()))
            try:
                expr, point_set = expr_cache[pts]
            except KeyError:
                config = kernel_cfg.copy()
                if runtime_quadrature_rule:
                    # Until FInAT dual evaluation is done, FIAT
                    # QuadratureElements with UnknownPointSet point sets
                    # advertise NaNs as their points for each node in the dual
                    # basis. This has to be manually replaced with the real
                    # UnknownPointSet point set used to create the
                    # QuadratureElement rule.
                    assert isnan(pts).all()
                    point_set = finat_to_element._rule.point_set
                    config.update(point_indices=point_set.indices, point_expr=point_set.expression)
                    context = fem.GemPointContext(**config)
                else:
                    point_set = PointSet(pts)
                    config.update(point_set=point_set)
                    context = fem.PointSetContext(**config)
                expr, = fem.compile_ufl(expression, context, point_sum=False)
                # In some cases point_set.indices may be dropped from expr, but
                # nothing new should now appear
                assert set(expr.free_indices) <= set(chain(point_set.indices, *argument_multiindices))
                expr = gem.partial_indexed(expr, shape_indices)
                expr_cache[pts] = expr, point_set
            weights = collections.defaultdict(list)
            for p in pts:
                for (w, cmp) in dual.get_point_dict()[p]:
                    weights[cmp].append(w)
            qexprs = gem.Zero()
            for cmp in sorted(weights):
                qweights = gem.Literal(weights[cmp])
                qexpr = gem.Indexed(expr, cmp)
                qexpr = gem.index_sum(gem.Indexed(qweights, point_set.indices)*qexpr,
                                      point_set.indices)
                qexprs = gem.Sum(qexprs, qexpr)
            assert qexprs.shape == ()
            assert set(qexprs.free_indices) == set(chain(shape_indices, *argument_multiindices))
            dual_expressions.append(qexprs)
        basis_indices = (gem.Index(), )
        ir = gem.Indexed(gem.ListTensor(dual_expressions), basis_indices)

    # Build kernel body
    return_indices = basis_indices + shape_indices + tuple(chain(*argument_multiindices))
    return_shape = tuple(i.extent for i in return_indices)
    return_var = gem.Variable('A', return_shape)
    if coffee:
        return_arg = ast.Decl(parameters["scalar_type"], ast.Symbol('A', rank=return_shape))
    else:
        return_arg = lp.GlobalArg("A", dtype=parameters["scalar_type"], shape=return_shape)

    return_expr = gem.Indexed(return_var, return_indices)

    # TODO: one should apply some GEM optimisations as in assembly,
    # but we don't for now.
    ir, = impero_utils.preprocess_gem([ir])
    impero_c = impero_utils.compile_gem([(return_expr, ir)], return_indices)
    index_names = dict((idx, "p%d" % i) for (i, idx) in enumerate(basis_indices))
    # Handle kernel interface requirements
    builder.register_requirements([ir])
    # Build kernel tuple
    return builder.construct_kernel(return_arg, impero_c, index_names, first_coefficient_fake_coords)
