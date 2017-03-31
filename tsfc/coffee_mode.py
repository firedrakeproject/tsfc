from __future__ import absolute_import, print_function, division

from functools import partial
from gem.optimise import replace_division
from gem.impero_utils import preprocess_gem
from gem.gem import (Conditional, Indexed, Failure)
from gem.node import traversal
import tsfc.vanilla as vanilla


flatten = vanilla.flatten

finalise_options = {'replace_delta': False, 'remove_componenttensors': False}


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
    expressions = replace_division(expressions)
    return optimise_expressions(expressions, quadrature_multiindex, argument_multiindices)


def optimise(node, quadrature_multiindex, argument_multiindices):
    from gem.refactorise import ATOMIC, COMPOUND, OTHER, collect_monomials
    argument_indices = tuple([i for indices in argument_multiindices for i in indices])

    def classify(argument_indices, expression):
        if isinstance(expression, Conditional): 
            return ATOMIC
        n = len(argument_indices.intersection(expression.free_indices))
        if n == 0:
            return OTHER
        elif n == 1:
            if isinstance(expression, Indexed):
                return ATOMIC
            else:
                return COMPOUND
        else:
            return COMPOUND
    classifier = partial(classify, set(argument_indices))

    monomial_sum, = collect_monomials([node], classifier)
    monomial_sum.argument_indices = argument_indices
    monomial_sum = monomial_sum.optimise()

    return monomial_sum.to_expression()


def optimise_expressions(expressions, quadrature_multiindices, argument_multiindices):
    """
    perform loop optimisations on gem DAGs
    :param expressions: list of gem DAGs
    :param quadrature_multiindices: quadrature multiindices, tuple of tuples
    :param argument_multiindices: argument multiindices, tuple of tuples
    :return: list of optimised gem DAGs
    """
    if propagate_failure(expressions):
        return expressions
    return [optimise(node, quadrature_multiindices, argument_multiindices) for node in expressions]


def propagate_failure(expressions):
    """
    Check if any gem nodes is Failure. In that case there is no need for subsequent optimisation.
    """
    for n in traversal(expressions):
        if isinstance(n, Failure):
            return True
    return False
