from __future__ import absolute_import, print_function, division

from gem.optimise import optimise_expressions
from gem.impero_utils import preprocess_gem
import tsfc.vanilla as vanilla


def Integrals(expressions, quadrature_multiindex, argument_multiindices, parameters):
    """Constructs an integral representation for each GEM integrand
    expression.

    :arg expressions: integrand multiplied with quadrature weight;
                      multi-root GEM expression DAG
    :arg quadrature_multiindex: quadrature multiindex (tuple)
    :arg argument_multiindices: tuple of argument multiindices,
                                one multiindex for each argument
    :arg parameters: parameters dictionary

    :returns: list of integral representations
    """
    # Need optimised roots for COFFEE
    expressions = vanilla.Integrals(expressions, quadrature_multiindex, argument_multiindices, parameters)
    expressions = preprocess_gem(expressions)
    return optimise_expressions(expressions, quadrature_multiindex, argument_multiindices)


flatten = vanilla.flatten

finalise_options = {'replace_delta': False, 'remove_componenttensors': False}
