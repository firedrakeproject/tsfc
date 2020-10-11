import collections
import operator
import string
import time
import sys
from functools import reduce, singledispatch, partial
from itertools import chain

from numpy import asarray, allclose

import ufl
from ufl.algorithms import extract_arguments, extract_coefficients
from ufl.algorithms.analysis import has_type
from ufl.classes import Form, GeometricQuantity, Coefficient, FunctionSpace
from ufl.log import GREEN

import gem
import gem.impero_utils as impero_utils
from gem.node import MemoizerArg, reuse_if_untouched_arg
from gem.optimise import filtered_replace_indices

import FIAT
from FIAT.functional import PointEvaluation

from finat.point_set import PointSet
from finat.quadrature import QuadratureRule

from tsfc import fem, ufl_utils
from tsfc.finatinterface import as_fiat_cell, create_element
from tsfc.logging import logger
from tsfc.parameters import default_parameters, is_complex
from tsfc.ufl_utils import apply_mapping

# To handle big forms. The various transformations might need a deeper stack
sys.setrecursionlimit(3000)


class TSFCFormData(object):
    r"""Mimic `ufl.FormData`.

    :arg form_data_tuple: A tuple of `ufl.FormData`s.
    :arg original_form: The original form from which forms
        associated with `ufl.Formdata`s were extracted.
    :diagonal: A flag for diagonal matrix assembly.

    This class preprocesses data contained in potentially
    multiple `ufl.FormData`s and stores minimal data set,
    narrowing down the scope of `KernelBuilder`s.
    Specifically, after preprocessing data here,
    we can:
        * forget `form_data.original_form`,
        * forget `integral_data.enabled_coefficients`,
        * let `KernelBuilder` only deal with raw `ufl.Coefficient`s.
                                                             _____________TSFCFormData____________
                 ____________________     __________        | ________  ________         ________ |
                |Integral||Integral|       |Integral|       ||        ||        |       |        ||
    FormData 0  |  Data  ||  Data  |  ...  |  Data  |       ||        ||        |       |        ||
                |____0___||____1___|_     _|____M___|       ||  TSFC  ||  TSFC  |       |  TSFC  ||  
                 ____________________     __________        ||Integral||Integral|       |Integral||    
                |Integral||Integral|       |Integral|       ||  Data  ||  Data  |       |  Data  ||         
    FormData 1  |  Data  ||  Data  |  ...  |  Data  |       ||    0   ||    1   |       |    M   ||       
                |____0___||____1___|_     _|____M___|  ---> ||        ||        |  ...  |        ||
                                                            |                                     |
            :                :                              |     :         :                :    |
                 ____________________     __________        |                                     |          
                |Integral||Integral|       |Integral|       ||        ||        |       |        ||
    FormData N  |  Data  ||  Data  |  ...  |  Data  |       ||        ||        |       |        ||
                |____0___||____1___|_     _|____M___|       ||________||________|       |________||
                                                            |_____________________________________|
    """
    def __init__(self, form_data_tuple, original_form, diagonal, form_data_hidden_function_map={}):
        try:
            self.arguments, = set(tuple(fd.preprocessed_form.arguments()
                                        for fd in form_data_tuple))
        except ValueError:
            raise ValueError("All `FormData`s must share the same set of arguments.")
        reduced_coefficients_set = set(c for fd in form_data_tuple for c in fd.reduced_coefficients)
        for _, val in form_data_hidden_function_map.items():
            reduced_coefficients_set.update(val)
        reduced_coefficients = sorted(reduced_coefficients_set, key=lambda c: c.count())
        if False:#len(form_data_tuple) == 1:
            self.reduced_coefficients = form_data_tuple[0].reduced_coefficients
            self.original_coefficient_positions = form_data_tuple[0].original_coefficient_positions
            self.function_replace_map = form_data_tuple[0].function_replace_map
        else:
            # Having gathered coefficients from multiple forms,
            # reconstruct `ufl.Coefficinet`s with count starting at 0.
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
        # Subspace
        self.reduced_subspaces = form_data_tuple[0].reduced_subspaces
        self.original_subspace_positions = form_data_tuple[0].original_subspace_positions
        self.subspace_replace_map = form_data_tuple[0].subspace_replace_map

        # Translate `ufl.IntegralData`s -> `TSFCIntegralData`.
        intg_data_dict = {}
        form_data_dict = {}
        for form_data in form_data_tuple:
            for intg_data in form_data.integral_data:
                domain = intg_data.domain
                integral_type = intg_data.integral_type
                subdomain_id = intg_data.subdomain_id
                key = (domain, integral_type, subdomain_id)
                # Add intg_data.
                intg_data_dict.setdefault(key, []).append(intg_data)
                # Remember which form_data this intg_data came from.
                form_data_dict.setdefault(key, []).append(form_data)
        integral_data_list = []
        for key in intg_data_dict:
            intg_data_list = intg_data_dict[key]
            form_data_list = form_data_dict[key]
            domain, _, _ = key
            domain_number = original_form.domain_numbering()[domain]
            integral_data_list.append(TSFCIntegralData(key, intg_data_list, form_data_list,
                                                       self, domain_number,
                                                       form_data_hidden_function_map=form_data_hidden_function_map))
        self.integral_data = tuple(integral_data_list)


class TSFCIntegralData(object):
    r"""Mimics `ufl.IntegralData`.

    :arg integral_data: a `ufl.IntegralData`.
    :arg form_data: a `ufl.FormData`.
    :arg tsfc_form_data: a `TSFCFormData`.

    This class mimics `ufl.FormData`, but:
        * extracts information required by TSFC.
        * preprocesses integrals so that `KernelBuilder`s only
          need to deal with raw `ufl.Coefficient`s.
    """
    def __init__(self, integral_data_key, integral_data_list, form_data_list, tsfc_form_data, domain_number, form_data_hidden_function_map={}):
        self.domain, self.integral_type, self.subdomain_id = integral_data_key
        self.domain_number = domain_number

        integrals = []
        _integral_to_form_data_map = {}
        functions = set()
        for intg_data, form_data in zip(integral_data_list, form_data_list):
            for integral in intg_data.integrals:
                integrand = integral.integrand()
                integrand = ufl.replace(integrand, tsfc_form_data.function_replace_map)
                integrand = ufl.replace(integrand, form_data.subspace_replace_map)
                new_integral = integral.reconstruct(integrand=integrand)
                integrals.append(new_integral)
                _integral_to_form_data_map[new_integral] = form_data
            # Gather functions that are enabled in this `TSFCIntegralData`.
            functions.update(f for f, enabled in zip(form_data.reduced_coefficients, intg_data.enabled_coefficients) if enabled)
            if form_data in form_data_hidden_function_map:
                functions.update(form_data_hidden_function_map[form_data])
        self.integrals = tuple(integrals)
        self._integral_to_form_data_map = _integral_to_form_data_map

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

        form_data = form_data_list[0]
        intg_data = integral_data_list[0]
        subspaces = tuple(c for c, enabled in zip(form_data.reduced_subspaces, intg_data.enabled_subspaces) if enabled)
        self.original_subspaces = subspaces
        self.subspaces = tuple(form_data.subspace_replace_map[c] for c in subspaces)
        self.subspace_numbers = tuple(pos for pos, enabled in zip(form_data.original_subspace_positions, intg_data.enabled_subspaces) if enabled)

    def integral_to_form_data(self, integral):
        r"""Return `ufl.FormData` in which the given integral was found."""
        return self._integral_to_form_data_map[integral]


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
    tsfc_form_data = TSFCFormData((form_data, ), form_data.original_form, diagonal)

    logger.info(GREEN % "compute_form_data finished in %g seconds.", time.time() - cpu_time)

    kernels = []
    for integral_data in tsfc_form_data.integral_data:
        start = time.time()
        kernel = compile_integral(integral_data, tsfc_form_data, prefix, parameters, interface=interface, coffee=coffee, diagonal=diagonal)
        if kernel is not None:
            kernels.append(kernel)
        logger.info(GREEN % "compile_integral finished in %g seconds.", time.time() - start)

    logger.info(GREEN % "TSFC finished in %g seconds.", time.time() - cpu_time)
    return kernels


def compile_integral(integral_data, form_data, prefix, parameters, interface, coffee, *, diagonal=False):
    """Compiles a UFL integral into an assembly kernel.

    :arg integral_data: TSFCIntegralData
    :arg form_data: TSFCFormData
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

    # The same builder (in principle) can be used to compile different forms.
    builder = interface(integral_data.integral_type,
                        parameters["scalar_type_c"] if coffee else parameters["scalar_type"],
                        domain=integral_data.domain,
                        coefficients=integral_data.coefficients,
                        arguments=form_data.arguments,
                        diagonal=diagonal,
                        fem_scalar_type = parameters["scalar_type"],
                        integral_data=integral_data)#REMOVE this when we move subspace.

    # All form specific variables (such as arguments) are stored in kernel_config (not in KernelBuilder instance).
    # The followings are specific for the concrete form representation, so
    # not to be saved in KernelBuilders.
    kernel_name = "%s_%s_integral_%s" % (prefix, integral_data.integral_type, integral_data.subdomain_id)
    kernel_name = kernel_name.replace("-", "_")  # Handle negative subdomain_id
    kernel_config = create_kernel_config(kernel_name, integral_data, parameters, builder)

    for integral in integral_data.integrals:
        params = parameters.copy()
        params.update(integral.metadata())  # integral metadata overrides
        expressions = builder.compile_ufl(integral.integrand(), params, kernel_config)
        expressions = replace_argument_multiindices_dummy(expressions, kernel_config, chain(*builder.argument_multiindex), chain(*builder.argument_multiindex_dummy))
        reps = builder.construct_integrals(expressions, params, kernel_config)
        builder.stash_integrals(reps, params, kernel_config)
    return builder.construct_kernel(kernel_config)


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


def create_kernel_config(kernel_name, integral_data, parameters, builder):
    # Data required for kernel construction. 
    kernel_config = dict(name=kernel_name,
                         integral_type=integral_data.integral_type,
                         subdomain_id=integral_data.subdomain_id,
                         domain_number=integral_data.domain_number,
                         coefficient_numbers=integral_data.coefficient_numbers,
                         subspace_numbers=integral_data.subspace_numbers,
                         subspace_parts=[None for _ in integral_data.subspace_numbers],
                         mode_irs=collections.OrderedDict(),
                         oriented=None,
                         needs_cell_sizes=None,
                         tabulations=None)
    return kernel_config


def compile_expression_dual_evaluation(expression, to_element, coordinates, *,
                                       domain=None, interface=None,
                                       parameters=None, coffee=False):
    """Compile a UFL expression to be evaluated against a compile-time known reference element's dual basis.

    Useful for interpolating UFL expressions into e.g. N1curl spaces.

    :arg expression: UFL expression
    :arg to_element: A FInAT element for the target space
    :arg coordinates: the coordinate function
    :arg domain: optional UFL domain the expression is defined on (useful when expression contains no domain).
    :arg interface: backend module for the kernel interface
    :arg parameters: parameters object
    :arg coffee: compile coffee kernel instead of loopy kernel
    """
    import coffee.base as ast
    import loopy as lp

    # Just convert FInAT element to FIAT for now.
    # Dual evaluation in FInAT will bring a thorough revision.
    to_element = to_element.fiat_equivalent

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

    # Replace coordinates (if any)
    domain = expression.ufl_domain()
    if domain:
        assert coordinates.ufl_domain() == domain
        builder.domain_coordinate[domain] = coordinates
        builder.set_cell_sizes(domain)

    # Collect required coefficients
    coefficients = extract_coefficients(expression)
    if has_type(expression, GeometricQuantity) or any(fem.needs_coordinate_mapping(c.ufl_element()) for c in coefficients):
        coefficients = [coordinates] + coefficients
    builder.set_coefficients(coefficients)

    # Split mixed coefficients
    expression = ufl_utils.split_coefficients(expression, builder.coefficient_split, )

    # Translate to GEM
    kernel_cfg = dict(interface=builder,
                      ufl_cell=coordinates.ufl_domain().ufl_cell(),
                      argument_multiindices=argument_multiindices,
                      index_cache={},
                      scalar_type=parameters["scalar_type"])

    if all(isinstance(dual, PointEvaluation) for dual in to_element.dual_basis()):
        # This is an optimisation for point-evaluation nodes which
        # should go away once FInAT offers the interface properly
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
        config = kernel_cfg.copy()
        config.update(point_set=point_set)

        # Allow interpolation onto QuadratureElements to refer to the quadrature
        # rule they represent
        if isinstance(to_element, FIAT.QuadratureElement):
            assert allclose(asarray(qpoints), asarray(to_element._points))
            quad_rule = QuadratureRule(point_set, to_element._weights)
            config["quadrature_rule"] = quad_rule

        expr, = fem.compile_ufl(expression, **config, point_sum=False)
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
                point_set = PointSet(pts)
                config = kernel_cfg.copy()
                config.update(point_set=point_set)
                expr, = fem.compile_ufl(expression, **config, point_sum=False)
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
    return builder.construct_kernel(return_arg, impero_c, index_names)


def replace_argument_multiindices_dummy(expressions, kernel_config, argument_multiindex, argument_multiindex_dummy):
    r"""Replace dummy indices with true argument multiindices.
    
    :arg expressions: gem expressions written in terms of argument_multiindices_dummy.
    :arg kernel_config:

    Applying `Delta(i, i_dummy)` and then `IndexSum(..., i_dummy)` would result in
    too many `IndexSum`s and `gem.optimise.contraction` would complain.
    Here, instead, we use filtered_replace_indices to directly replace dummy argument
    multiindices with true ones.
    """
    # True/dummy argument multiindices.
    if argument_multiindex_dummy == argument_multiindex:
        return expressions
    substitution = tuple(zip(argument_multiindex_dummy, argument_multiindex))
    mapper = MemoizerArg(filtered_replace_indices)
    return tuple(mapper(expr, substitution) for expr in expressions)
