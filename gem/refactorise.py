"""Data structures and algorithms for generic expansion and
refactorisation."""

from __future__ import absolute_import, print_function, division
from six import iteritems
from six.moves import intern, map

from collections import Counter, OrderedDict, defaultdict, namedtuple
from itertools import product

from gem.node import Memoizer, traversal
from gem.gem import Node, Zero, Product, Sum, Indexed, ListTensor, one
from gem.optimise import (remove_componenttensors, sum_factorise,
                          traverse_product, traverse_sum, unroll_indexsum)


# Refactorisation labels

ATOMIC = intern('atomic')
"""Label: the expression need not be broken up into smaller parts"""

COMPOUND = intern('compound')
"""Label: the expression must be broken up into smaller parts"""

OTHER = intern('other')
"""Label: the expression is irrelevant with regards to refactorisation"""


Monomial = namedtuple('Monomial', ['sum_indices', 'atomics', 'rest'])
"""Monomial type, representation of a tensor product with some
distinguished factors (called atomics).

- sum_indices: indices to sum over
- atomics: tuple of expressions classified as ATOMIC
- rest: a single expression classified as OTHER

A :py:class:`Monomial` is a structured description of the expression:

.. code-block:: python

    IndexSum(reduce(Product, atomics, rest), sum_indices)

"""


class MonomialSum(object):
    """Represents a sum of :py:class:`Monomial`s.

    The set of :py:class:`Monomial` summands are represented as a
    mapping from a pair of unordered ``sum_indices`` and unordered
    ``atomics`` to a ``rest`` GEM expression.  This representation
    makes it easier to merge similar monomials.
    """
    def __init__(self):
        # (unordered sum_indices, unordered atomics) -> rest
        self.monomials = defaultdict(Zero)

        # We shall retain ordering for deterministic code generation:
        #
        # (unordered sum_indices, unordered atomics) ->
        #     (ordered sum_indices, ordered atomics)
        self.ordering = OrderedDict()

    def add(self, sum_indices, atomics, rest):
        """Updates the :py:class:`MonomialSum` adding a new monomial."""
        sum_indices = tuple(sum_indices)
        sum_indices_set = frozenset(sum_indices)
        # Sum indices cannot have duplicates
        assert len(sum_indices) == len(sum_indices_set)

        atomics = tuple(atomics)
        atomics_set = frozenset(iteritems(Counter(atomics)))

        assert isinstance(rest, Node)

        key = (sum_indices_set, atomics_set)
        self.monomials[key] = Sum(self.monomials[key], rest)
        self.ordering.setdefault(key, (sum_indices, atomics))

    def __iter__(self):
        """Iteration yields :py:class:`Monomial` objects"""
        for key, (sum_indices, atomics) in iteritems(self.ordering):
            rest = self.monomials[key]
            yield Monomial(sum_indices, atomics, rest)

    @staticmethod
    def sum(*args):
        """Sum of multiple :py:class:`MonomialSum`s"""
        result = MonomialSum()
        for arg in args:
            assert isinstance(arg, MonomialSum)
            # Optimised implementation: no need to decompose and
            # reconstruct key.
            for key, rest in iteritems(arg.monomials):
                result.monomials[key] = Sum(result.monomials[key], rest)
            for key, value in iteritems(arg.ordering):
                result.ordering.setdefault(key, value)
        return result

    @staticmethod
    def product(*args):
        """Product of multiple :py:class:`MonomialSum`s"""
        result = MonomialSum()
        for monomials in product(*args):
            sum_indices = []
            atomics = []
            rest = one
            for s, a, r in monomials:
                sum_indices.extend(s)
                atomics.extend(a)
                rest = Product(r, rest)
            result.add(sum_indices, atomics, rest)
        return result


class FactorisationError(Exception):
    """Raised when factorisation fails to achieve some desired form."""
    pass


def _collect_monomials(expression, self):
    """Refactorises an expression into a sum-of-products form, using
    distributivity rules (i.e. a*(b + c) -> a*b + a*c).  Expansion
    proceeds until all "compound" expressions are broken up.

    :arg expression: a GEM expression to refactorise
    :arg self: function for recursive calls

    :returns: :py:class:`MonomialSum`

    :raises FactorisationError: Failed to break up some "compound"
                                expressions with expansion.
    """
    # Phase 1: Collect and categorise product terms
    def stop_at(expr):
        # Break up compounds only
        return self.classifier(expr) != COMPOUND
    common_indices, terms = traverse_product(expression, stop_at=stop_at)
    common_indices = tuple(common_indices)

    common_atomics = []
    common_others = []
    compounds = []
    for term in terms:
        cls = self.classifier(term)
        if cls == ATOMIC:
            common_atomics.append(term)
        elif cls == COMPOUND:
            compounds.append(term)
        elif cls == OTHER:
            common_others.append(term)
        else:
            raise ValueError("Classifier returned illegal value.")
    common_atomics = tuple(common_atomics)

    # Phase 2: Attempt to break up compound terms into summands
    sums = []
    for expr in compounds:
        summands = traverse_sum(expr, stop_at=stop_at)
        if len(summands) <= 1:
            # Compound term is not an addition, avoid infinite
            # recursion and fail gracefully raising an exception.
            raise FactorisationError(expr)
        # Recurse into each summand, concatenate their results
        sums.append(MonomialSum.sum(*map(self, summands)))

    # Phase 3: Expansion
    #
    # Each element of ``sums`` is a MonomialSum.  Expansion produces a
    # series (representing a sum) of products of monomials.
    result = MonomialSum()
    for s, a, r in MonomialSum.product(*sums):
        all_indices = common_indices + s
        atomics = common_atomics + a

        # All free indices that appear in atomic terms
        atomic_indices = set().union(*[atomic.free_indices
                                       for atomic in atomics])

        # Sum indices that appear in atomic terms
        # (will go to the result :py:class:`Monomial`)
        sum_indices = tuple(index for index in all_indices
                            if index in atomic_indices)

        # Sum indices that do not appear in atomic terms
        # (can factorise them over atomic terms immediately)
        rest_indices = tuple(index for index in all_indices
                             if index not in atomic_indices)

        # Not really sum factorisation, but rather just an optimised
        # way of building a product.
        rest = sum_factorise(rest_indices, common_others + [r])

        result.add(sum_indices, atomics, rest)
    return result


def collect_monomials(expressions, classifier):
    """Refactorises expressions into a sum-of-products form, using
    distributivity rules (i.e. a*(b + c) -> a*b + a*c).  Expansion
    proceeds until all "compound" expressions are broken up.

    :arg expressions: GEM expressions to refactorise
    :arg classifier: a function that can classify any GEM expression
                     as ``ATOMIC``, ``COMPOUND``, or ``OTHER``.  This
                     classification drives the factorisation.

    :returns: list of :py:class:`MonomialSum`s

    :raises FactorisationError: Failed to break up some "compound"
                                expressions with expansion.
    """
    # Get ComponentTensors out of the way
    expressions = remove_componenttensors(expressions)

    # Get ListTensors out of the way
    must_unroll = []  # indices to unroll
    for node in traversal(expressions):
        if isinstance(node, Indexed):
            child, = node.children
            if isinstance(child, ListTensor) and classifier(node) == COMPOUND:
                must_unroll.extend(node.multiindex)
    if must_unroll:
        must_unroll = set(must_unroll)
        expressions = unroll_indexsum(expressions,
                                      predicate=lambda i: i in must_unroll)
        expressions = remove_componenttensors(expressions)

    # Finally, refactorise expressions
    mapper = Memoizer(_collect_monomials)
    mapper.classifier = classifier
    return list(map(mapper, expressions))
