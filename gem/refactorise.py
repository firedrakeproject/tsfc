"""Data structures and algorithms for generic expansion and
refactorisation."""

from __future__ import absolute_import, print_function, division
from six import iteritems, itervalues
from six.moves import intern, map

from collections import Counter, OrderedDict, defaultdict, namedtuple
from itertools import product, count

import numpy

from gem.node import Memoizer, traversal
from gem.gem import (Node, Zero, Product, Sum, Indexed, ListTensor, one,
                     IndexSum)
from gem.optimise import (remove_componenttensors, sum_factorise,
                          traverse_product, traverse_sum, unroll_indexsum,
                          fast_sum_factorise, associate_product, associate_sum)


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

    def argument_indices_extent(self, factor):
        if self.argument_indices is None:
            raise AssertionError("argument_indices property not initialised.")
        return numpy.product([i.extent for i in set(factor.free_indices).intersection(self.argument_indices)])

    def all_sum_indices(self):
        result = []
        collected = set()
        for (sum_indices_set, _), (sum_indices, _) in iteritems(self.ordering):
            if sum_indices_set not in collected:
                result.append((sum_indices_set, sum_indices))
                collected.add(sum_indices_set)
        return result

    def to_expression(self):
        # No argument factorisation here yet
        indexsums = []
        for sum_indices_set, sum_indices in self.all_sum_indices():
            all_atomics = []
            all_rest = []
            for key, (_, atomics) in iteritems(self.ordering):
                if sum_indices_set == key[0]:
                    rest = self.monomials[key]
                    if not sum_indices_set:
                        indexsums.append(fast_sum_factorise(sum_indices, atomics + (rest,)))
                        continue
                    all_atomics.append(atomics)
                    all_rest.append(rest)
            if not all_atomics:
                continue
            if len(all_atomics) == 1:
                indexsums.append(fast_sum_factorise(sum_indices, all_atomics[0] + (all_rest[0],)))
            else:
                products = [associate_product(atomics + (_rest,))[0] for atomics, _rest in zip(all_atomics, all_rest)]
                indexsums.append(IndexSum(associate_sum(products)[0], sum_indices))
                # indexsums.append(IndexSum(reduce(Sum, products, Zero()), sum_indices))
        return associate_sum(indexsums)[0]
        # return reduce(Sum, indexsums, Zero())

    def find_optimal_atomics(self, sum_indices):
        sum_indices_set, _ = sum_indices
        index = count()
        atomic_index = OrderedDict()  # Atomic gem node -> int
        connections = []
        # add connections (list of lists)
        for (_sum_indices, _), (_, atomics) in iteritems(self.ordering):
            if _sum_indices == sum_indices_set:
                connection = []
                for atomic in atomics:
                    if atomic not in atomic_index:
                        atomic_index[atomic] = next(index)
                    connection.append(atomic_index[atomic])
                connections.append(tuple(connection))

        if len(atomic_index) == 0:
            return ((), ())

        # set up the ILP
        import pulp as ilp
        ilp_prob = ilp.LpProblem('gem factorise', ilp.LpMinimize)
        ilp_var = ilp.LpVariable.dicts('node', range(len(atomic_index)), 0, 1, ilp.LpBinary)

        # Objective function
        # Minimise number of factors to pull. If same number, favour factor with larger extent
        big = 10000000  # some arbitrary big number
        ilp_prob += ilp.lpSum(ilp_var[index] * (big - self.argument_indices_extent(atomic)) for atomic, index in iteritems(atomic_index))

        # constraints
        for connection in connections:
            ilp_prob += ilp.lpSum(ilp_var[index] for index in connection) >= 1

        ilp_prob.solve()
        if ilp_prob.status != 1:
            raise AssertionError("Something bad happened during ILP")

        optimal_atomics = [atomic for atomic, _index in iteritems(atomic_index) if ilp_var[_index].value() == 1]
        other_atomics = [atomic for atomic, _index in iteritems(atomic_index) if ilp_var[_index].value() == 0]
        optimal_atomics = sorted(optimal_atomics, key=lambda x: self.argument_indices_extent(x), reverse=True)
        other_atomics = sorted(other_atomics, key=lambda x: self.argument_indices_extent(x), reverse=True)
        return (tuple(optimal_atomics), tuple(other_atomics))

    def factorise_atomics(self, optimal_atomics):
        if not optimal_atomics:
            return
        if len(self.ordering) < 2:
            return
        # pick the first atomic
        (sum_indices_set, sum_indices), atomic = optimal_atomics[0]
        factorised_out = []
        for key, (_, _atomics) in iteritems(self.ordering):
            if sum_indices_set == key[0] and atomic in _atomics:
                factorised_out.append(key)
        if len(factorised_out) <= 1:
            self.factorise_atomics(optimal_atomics[1:])
            return
        all_atomics = []
        all_rest = []
        for key in factorised_out:
            _atomics = list(self.ordering[key][1])
            _atomics.remove(atomic)
            all_atomics.append(_atomics)
            all_rest.append(self.monomials[key])
            del self.ordering[key]
            del self.monomials[key]
        new_monomial_sum = MonomialSum()
        new_monomial_sum.argument_indices = self.argument_indices
        for _atomics, _rest in zip(all_atomics, all_rest):
            new_monomial_sum.add((), _atomics, _rest)
        # new_optimal_atomics = [((frozenset(), ()), oa) for _, oa in optimal_atomics[1:]]
        new_monomial_sum.optimise()
        # new_monomial_sum.factorise_atomics(new_optimal_atomics)
        assert len(new_monomial_sum.ordering) != 0
        if len(new_monomial_sum.ordering) == 1:
            # result is a product
            new_atomics = list(itervalues(new_monomial_sum.ordering))[0][1] + (atomic,)
            new_rest = list(itervalues(new_monomial_sum.monomials))[0]
        else:
            # result is a sum
            new_node = new_monomial_sum.to_expression()
            new_atomics = [atomic]
            new_rest = one
            if set(self.argument_indices) & set(new_node.free_indices):
                new_atomics.append(new_node)
            else:
                new_rest = new_node
        self.add(sum_indices, new_atomics, new_rest)
        # factorise the next atomic
        self.factorise_atomics(optimal_atomics[1:])
        return

    def optimise(self):
        optimal_atomics = []  # [(sum_indices, optimal_atomics))]
        for sum_indices in self.all_sum_indices():
            atomics = self.find_optimal_atomics(sum_indices)
            optimal_atomics.extend([(sum_indices, _atomic) for _atomic in atomics[0]])
        # This algorithm is O(2^N), where N = len(optimal_atomics)
        # we could truncate the optimal_atomics list at say 10
        self.factorise_atomics(optimal_atomics)


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
        label = self.classifier(term)
        if label == ATOMIC:
            common_atomics.append(term)
        elif label == COMPOUND:
            compounds.append(term)
        elif label == OTHER:
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
