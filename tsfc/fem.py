from __future__ import absolute_import

import collections
import itertools

import numpy
from singledispatch import singledispatch

import ufl
from ufl.corealg.map_dag import map_expr_dag, map_expr_dags
from ufl.corealg.multifunction import MultiFunction
from ufl.classes import (Argument, Coefficient, FormArgument,
                         GeometricQuantity, QuadratureWeight,
                         ReferenceValue, Zero)
from ufl.classes import (Abs, CellOrientation, Expr, FloatValue,
                         Division, Product, ScalarValue, Sqrt)

from tsfc.constants import PRECISION
from tsfc.fiatinterface import create_element, as_fiat_cell
from tsfc.modified_terminals import is_modified_terminal, analyse_modified_terminal
from tsfc.node import MemoizerArg
from tsfc import compat
from tsfc import gem
from tsfc import ufl2gem
from tsfc import geometric


# FFC uses one less digits for rounding than for printing
epsilon = eval("1e-%d" % (PRECISION - 1))


class ReplaceSpatialCoordinates(MultiFunction):
    """Replace SpatialCoordinate nodes with the ReferenceValue of a
    Coefficient.  Assumes that the coordinate element only needs
    affine mapping.

    :arg coordinates: the coefficient to replace spatial coordinates with
    """
    def __init__(self, coordinates):
        self.coordinates = coordinates
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, t):
        return t

    def spatial_coordinate(self, o):
        assert o.ufl_domain().ufl_coordinate_element().mapping() == "identity"
        return ReferenceValue(self.coordinates)


class ModifiedTerminalMixin(object):
    """Mixin to use with MultiFunctions that operate on modified
    terminals."""

    def unexpected(self, o):
        assert False, "Not expected %r at this stage." % o

    # global derivates should have been pulled back
    grad = unexpected
    div = unexpected
    curl = unexpected

    # div and curl should have been algebraically lowered
    reference_div = unexpected
    reference_curl = unexpected

    def _modified_terminal(self, o):
        assert is_modified_terminal(o)
        return self.modified_terminal(o)

    # Unlike UFL, we do not regard Indexed as a terminal modifier.
    # indexed = _modified_terminal

    positive_restricted = _modified_terminal
    negative_restricted = _modified_terminal

    cell_avg = _modified_terminal
    facet_avg = _modified_terminal

    reference_grad = _modified_terminal
    reference_value = _modified_terminal

    terminal = _modified_terminal


class CollectModifiedTerminals(MultiFunction, ModifiedTerminalMixin):
    """Collect the modified terminals in a UFL expression.

    :arg return_list: modified terminals will be appended to this list
    """
    def __init__(self, return_list):
        MultiFunction.__init__(self)
        self.return_list = return_list

    def expr(self, o, *ops):
        pass  # operands visited

    def indexed(self, o, *ops):
        pass  # not a terminal modifier

    def multi_index(self, o):
        pass  # ignore

    def modified_terminal(self, o):
        self.return_list.append(o)


class PickRestriction(MultiFunction, ModifiedTerminalMixin):
    """Pick out parts of an expression with specified restrictions on
    the arguments.

    :arg test: The restriction on the test function.
    :arg trial:  The restriction on the trial function.

    Returns those parts of the expression that have the requested
    restrictions, or else :class:`ufl.classes.Zero` if no such part
    exists.
    """
    def __init__(self, test=None, trial=None):
        self.restrictions = {0: test, 1: trial}
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def multi_index(self, o):
        return o

    def modified_terminal(self, o):
        mt = analyse_modified_terminal(o)
        t = mt.terminal
        r = mt.restriction
        if isinstance(t, Argument) and r != self.restrictions[t.number()]:
            return Zero(o.ufl_shape, o.ufl_free_indices, o.ufl_index_dimensions)
        else:
            return o


def _spanning_degree(cell, degree):
    if cell is None:
        assert degree == 0
        return degree
    elif cell.cellname() in ["interval", "triangle", "tetrahedron"]:
        return degree
    elif cell.cellname() == "quadrilateral":
        # TODO: Tensor-product space assumed
        return 2 * degree
    elif isinstance(cell, ufl.TensorProductCell):
        try:
            # A component cell might be a quadrilateral, so recurse.
            return sum(_spanning_degree(sub_cell, d)
                       for sub_cell, d in zip(cell.sub_cells(), degree))
        except TypeError:
            assert degree == 0
            return 0
    else:
        raise ValueError("Unknown cell %s" % cell.cellname())


def spanning_degree(element):
    """Determine the degree of the polynomial space spanning an element.

    :arg element: The element to determine the degree of.

    .. warning::

       For non-simplex elements, this assumes a tensor-product
       space.
    """
    return _spanning_degree(element.cell(), element.degree())


def ufl_reuse_if_untouched(o, *ops):
    """Reuse object if operands are the same objects."""
    if all(a is b for a, b in zip(o.ufl_operands, ops)):
        return o
    else:
        return o._ufl_expr_reconstruct_(*ops)


@singledispatch
def _simplify_abs(o, self, in_abs):
    """Single-dispatch function to simplify absolute values.

    :arg o: UFL node
    :arg self: Callback handler for recursion
    :arg in_abs: Is ``o`` inside an absolute value?

    When ``in_abs`` we must return a non-negative value, potentially
    by wrapping the returned node with ``Abs``.
    """
    raise AssertionError("UFL node expected, not %s" % type(o))


@_simplify_abs.register(Expr)  # noqa
def _(o, self, in_abs):
    # General case, only wrap the outer expression (if necessary)
    operands = [self(op, False) for op in o.ufl_operands]
    result = ufl_reuse_if_untouched(o, *operands)
    if in_abs:
        result = Abs(result)
    return result


@_simplify_abs.register(Sqrt)  # noqa
def _(o, self, in_abs):
    # Square root is always non-negative
    return ufl_reuse_if_untouched(o, self(o.ufl_operands[0], False))


@_simplify_abs.register(ScalarValue)  # noqa
def _(o, self, in_abs):
    if not in_abs:
        return o
    # Inline abs(constant)
    return ufl.as_ufl(abs(o._value))


@_simplify_abs.register(CellOrientation)  # noqa
def _(o, self, in_abs):
    if not in_abs:
        return o
    # Cell orientation is +-1
    return FloatValue(1)


@_simplify_abs.register(Division)  # noqa
@_simplify_abs.register(Product)
def _(o, self, in_abs):
    if not in_abs:
        # Just reconstruct
        ops = [self(op, False) for op in o.ufl_operands]
        return ufl_reuse_if_untouched(o, *ops)

    # Visit children, distributing Abs
    ops = [self(op, True) for op in o.ufl_operands]

    # Strip Abs off again (we will put it outside now)
    stripped = False
    strip_ops = []
    for op in ops:
        if isinstance(op, Abs):
            stripped = True
            strip_ops.append(op.ufl_operands[0])
        else:
            strip_ops.append(op)

    # Rebuild, and wrap with Abs if necessary
    result = ufl_reuse_if_untouched(o, *strip_ops)
    if stripped:
        result = Abs(result)
    return result


@_simplify_abs.register(Abs)  # noqa
def _(o, self, in_abs):
    return self(o.ufl_operands[0], True)


def simplify_abs(expression):
    """Simplify absolute values in a UFL expression.  Its primary
    purpose is to "neutralise" CellOrientation nodes that are
    surrounded by absolute values and thus not at all necessary."""
    return MemoizerArg(_simplify_abs)(expression, False)


def _tabulate(ufl_element, order, points, entity):
    """Ask FIAT to tabulate ``points`` up to order ``order``, on entity
    ``entity`` with ``entity = (element dimension, id), then
    rearranges the result into a series of ``(c, D, table)`` tuples,
    where:

    c: component index (for vector-valued and tensor-valued elements)
    D: derivative tuple (e.g. (1, 2) means d/dx d^2/dy^2)
    table: tabulation matrix for the given component and derivative.
           shape: len(points) x space_dimension

    :arg ufl_element: element to tabulate
    :arg order: FIAT gives all derivatives up to this order
    :arg points: points to tabulate the element on
    :arg entity: specific entity that we are tabulating
    """
    element = create_element(ufl_element)
    phi = element.space_dimension()
    C = ufl_element.reference_value_size() - len(ufl_element.symmetry())
    q = len(points)
    for D, fiat_table in element.tabulate(order, points, entity).iteritems():
        reordered_table = fiat_table.reshape(phi, C, q).transpose(1, 2, 0)  # (C, q, phi)
        for c, table in enumerate(reordered_table):
            yield c, D, table


def tabulate(ufl_element, order, points, entity):
    """Same as the above, but also applies FFC rounding and recognises
    cellwise constantness.  Cellwise constantness is determined
    symbolically, but we also check the numerics to be safe."""
    for c, D, table in _tabulate(ufl_element, order, points, entity):
        # Copied from FFC (ffc/quadrature/quadratureutils.py)
        table[abs(table) < epsilon] = 0
        table[abs(table - 1.0) < epsilon] = 1.0
        table[abs(table + 1.0) < epsilon] = -1.0
        table[abs(table - 0.5) < epsilon] = 0.5
        table[abs(table + 0.5) < epsilon] = -0.5

        if spanning_degree(ufl_element) <= sum(D):
            assert compat.allclose(table, table.mean(axis=0, keepdims=True), equal_nan=True)
            table = table[0]

        yield c, D, table


def make_tabulator(points, entity):
    """Creates a tabulator for an array of points on a given entity."""
    return lambda elem, order: tabulate(elem, order, points, entity)


class TabulationManager(object):
    """Manages the generation of tabulation matrices for the different
    integral types."""

    def __init__(self, integral_type, cell, points):
        """Constructs a TabulationManager.

        :arg integral_type: integral type
        :arg cell: UFL cell
        :arg points: points on the integration entity (e.g. points on
                     an interval for facet integrals on a triangle)
        """
        self.integral_type = integral_type
        self.points = points

        self.tabulators = []
        self.tables = {}

        if integral_type == 'cell':
            self.tabulators.append(make_tabulator(points))

        elif integral_type in ['exterior_facet', 'interior_facet']:
            for entity in range(cell.num_facets()):

                # Facet transforms for non-tensor element is computed in FIAT
                # Simply pass entity id (facet dim, facet_id)
                entity_id = (cell.topological_dimension()-1, entity)
                self.tabulators.append(make_tabulator(points, entity_id))

        elif integral_type in ['exterior_facet_bottom', 'exterior_facet_top', 'interior_facet_horiz']:
            for entity in range(2):  # top and bottom

                entity_id = ((cell.topological_dimension()-1, 0), (0, entity))
                self.tabulators.append(make_tabulator(points, entity_id))

        elif integral_type in ['exterior_facet_vert', 'interior_facet_vert']:
            for entity in range(cell.sub_cells()[0].num_facets()):  # "base cell" facets
                entity_id = ((cell.topological_dimension()-2, 1), (entity, 0))
                self.tabulators.append(make_tabulator(points, entity_id))

        else:
            raise NotImplementedError("integral type %s not supported" % integral_type)

    def tabulate(self, ufl_element, max_deriv):
        """Prepare the tabulations of a finite element up to a given
        derivative order.

        :arg ufl_element: UFL element to tabulate
        :arg max_deriv: tabulate derivatives up this order
        """
        store = collections.defaultdict(list)
        for tabulator in self.tabulators:
            for c, D, table in tabulator(ufl_element, max_deriv):
                store[(ufl_element, c, D)].append(table)

        if self.integral_type == 'cell':
            for key, (table,) in store.iteritems():
                self.tables[key] = table
        else:
            for key, tables in store.iteritems():
                table = numpy.array(tables)
                if len(table.shape) == 2:
                    # Cellwise constant; must not depend on the facet
                    assert compat.allclose(table, table.mean(axis=0, keepdims=True), equal_nan=True)
                    table = table[0]
                self.tables[key] = table

    def __getitem__(self, key):
        return self.tables[key]


class Translator(MultiFunction, ModifiedTerminalMixin, ufl2gem.Mixin):
    """Contains all the context necessary to translate UFL into GEM."""

    def __init__(self, weights, quadrature_index, argument_indices, tabulation_manager,
                 coefficient_map, index_cache):
        MultiFunction.__init__(self)
        ufl2gem.Mixin.__init__(self)
        integral_type = tabulation_manager.integral_type
        self.weights = gem.Literal(weights)
        self.quadrature_index = quadrature_index
        self.argument_indices = argument_indices
        self.tabulation_manager = tabulation_manager
        self.integral_type = integral_type
        self.coefficient_map = coefficient_map
        self.index_cache = index_cache

        if integral_type in ['exterior_facet', 'exterior_facet_vert']:
            self.facet = {None: gem.VariableIndex(gem.Indexed(gem.Variable('facet', (1,)), (0,)))}
        elif integral_type in ['interior_facet', 'interior_facet_vert']:
            self.facet = {'+': gem.VariableIndex(gem.Indexed(gem.Variable('facet', (2,)), (0,))),
                          '-': gem.VariableIndex(gem.Indexed(gem.Variable('facet', (2,)), (1,)))}
        elif integral_type == 'exterior_facet_bottom':
            self.facet = {None: 0}
        elif integral_type == 'exterior_facet_top':
            self.facet = {None: 1}
        elif integral_type == 'interior_facet_horiz':
            self.facet = {'+': 1, '-': 0}
        else:
            self.facet = None

        if self.integral_type.startswith("interior_facet"):
            self.cell_orientations = gem.Variable("cell_orientations", (2, 1))
        else:
            self.cell_orientations = gem.Variable("cell_orientations", (1, 1))

    def select_facet(self, tensor, restriction):
        """Applies facet selection on a GEM tensor if necessary.

        :arg tensor: GEM tensor
        :arg restriction: restriction on the modified terminal
        :returns: another GEM tensor
        """
        if self.integral_type == 'cell':
            return tensor
        else:
            f = self.facet[restriction]
            return gem.partial_indexed(tensor, (f,))

    def modified_terminal(self, o):
        """Overrides the modified terminal handler from
        :class:`ModifiedTerminalMixin`."""
        mt = analyse_modified_terminal(o)
        return translate(mt.terminal, mt, self)


def iterate_shape(mt, callback):
    """Iterates through the components of a modified terminal, and
    calls ``callback`` with ``(ufl_element, c, D)`` keys which are
    used to look up tabulation matrix for that component.  Then
    assembles the result into a GEM tensor (if tensor-valued)
    corresponding to the modified terminal.

    :arg mt: analysed modified terminal
    :arg callback: callback to get the GEM translation of a component
    :returns: GEM translation of the modified terminal

    This is a helper for translating Arguments and Coefficients.
    """
    ufl_element = mt.terminal.ufl_element()
    dim = ufl_element.cell().topological_dimension()

    def flat_index(ordered_deriv):
        return tuple((numpy.asarray(ordered_deriv) == d).sum() for d in range(dim))

    ordered_derivs = itertools.product(range(dim), repeat=mt.local_derivatives)
    flat_derivs = map(flat_index, ordered_derivs)

    result = []
    for c in range(ufl_element.reference_value_size()):
        for flat_deriv in flat_derivs:
            result.append(callback((ufl_element, c, flat_deriv)))

    shape = mt.expr.ufl_shape
    assert len(result) == numpy.prod(shape)

    if shape:
        return gem.ListTensor(numpy.asarray(result).reshape(shape))
    else:
        return result[0]


@singledispatch
def translate(terminal, mt, params):
    """Translates modified terminals into GEM.

    :arg terminal: terminal, for dispatching
    :arg mt: analysed modified terminal
    :arg params: translator context
    :returns: GEM translation of the modified terminal
    """
    raise AssertionError("Cannot handle terminal type: %s" % type(terminal))


@translate.register(QuadratureWeight)  # noqa: Not actually redefinition
def _(terminal, mt, params):
    return gem.Indexed(params.weights, (params.quadrature_index,))


@translate.register(GeometricQuantity)  # noqa: Not actually redefinition
def _(terminal, mt, params):
    return geometric.translate(terminal, mt, params)


@translate.register(Argument)  # noqa: Not actually redefinition
def _(terminal, mt, params):
    argument_index = params.argument_indices[terminal.number()]

    def callback(key):
        table = params.tabulation_manager[key]
        if len(table.shape) == 1:
            # Cellwise constant
            row = gem.Literal(table)
        else:
            table = params.select_facet(gem.Literal(table), mt.restriction)
            row = gem.partial_indexed(table, (params.quadrature_index,))
        return gem.Indexed(row, (argument_index,))

    return iterate_shape(mt, callback)


@translate.register(Coefficient)  # noqa: Not actually redefinition
def _(terminal, mt, params):
    kernel_arg = params.coefficient_map[terminal]

    if terminal.ufl_element().family() == 'Real':
        assert mt.local_derivatives == 0
        return kernel_arg

    ka = gem.partial_indexed(kernel_arg, {None: (), '+': (0,), '-': (1,)}[mt.restriction])

    def callback(key):
        table = params.tabulation_manager[key]
        if len(table.shape) == 1:
            # Cellwise constant
            row = gem.Literal(table)
            if numpy.count_nonzero(table) <= 2:
                assert row.shape == ka.shape
                return reduce(gem.Sum,
                              [gem.Product(gem.Indexed(row, (i,)), gem.Indexed(ka, (i,)))
                               for i in range(row.shape[0])],
                              gem.Zero())
        else:
            table = params.select_facet(gem.Literal(table), mt.restriction)
            row = gem.partial_indexed(table, (params.quadrature_index,))

        r = params.index_cache[terminal.ufl_element()]
        return gem.IndexSum(gem.Product(gem.Indexed(row, (r,)),
                                        gem.Indexed(ka, (r,))), r)

    return iterate_shape(mt, callback)


def coordinate_coefficient(domain):
    """Create a fake coordinate coefficient for a domain."""
    return ufl.Coefficient(ufl.FunctionSpace(domain, domain.ufl_coordinate_element()))


def replace_coordinates(integrand, coordinate_coefficient):
    """Replace SpatialCoordinate nodes with Coefficients."""
    return map_expr_dag(ReplaceSpatialCoordinates(coordinate_coefficient), integrand)


def process(integral_type, integrand, tabulation_manager, quadrature_weights, quadrature_index,
            argument_indices, coefficient_map, index_cache):
    # Abs-simplification
    integrand = simplify_abs(integrand)

    # Collect modified terminals
    modified_terminals = []
    map_expr_dag(CollectModifiedTerminals(modified_terminals), integrand)

    # Collect maximal derivatives that needs tabulation
    max_derivs = collections.defaultdict(int)

    for mt in map(analyse_modified_terminal, modified_terminals):
        if isinstance(mt.terminal, FormArgument):
            ufl_element = mt.terminal.ufl_element()
            max_derivs[ufl_element] = max(mt.local_derivatives, max_derivs[ufl_element])

    # Collect tabulations for all components and derivatives
    for ufl_element, max_deriv in max_derivs.items():
        if ufl_element.family() != 'Real':
            tabulation_manager.tabulate(ufl_element, max_deriv)

    if integral_type.startswith("interior_facet"):
        expressions = []
        for rs in itertools.product(("+", "-"), repeat=len(argument_indices)):
            expressions.append(map_expr_dag(PickRestriction(*rs), integrand))
    else:
        expressions = [integrand]

    # Translate UFL to Einstein's notation,
    # lowering finite element specific nodes
    translator = Translator(quadrature_weights, quadrature_index,
                            argument_indices, tabulation_manager,
                            coefficient_map, index_cache)
    return map_expr_dags(translator, expressions)
