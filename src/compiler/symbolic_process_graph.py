"""Canonical SymPy projection for semantic ProcessGraphs.

AST and SymPy are source languages.  Neither source object's field layout is
the ProcessGraph schema: both are normalized to explicit value nodes and
canonical operations before another source language is rendered.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import sympy


@dataclass(frozen=True)
class SympyProcessGraphRule:
    """One public SymPy-node to canonical ProcessGraph translation rule.

    ``roles`` is empty for variadic positional operations. Special importers
    for control, indexing, inputs, and literals use the named roles to retain
    structure which cannot be inferred from a flat argument list later.
    """

    operation: str
    roles: tuple[str, ...] = ()
    node_type: str | None = None


# This is deliberately a data table rather than a chain of type-name tests.
# Backends and tests can inspect it, and adding a new exact SymPy primitive is
# a one-line change. Special shapes still use the same rule after normalizing
# their arguments (Piecewise is nested Select; Indexed has base/index roles).
SYMPY_PROCESS_GRAPH_TRANSLATIONS: Mapping[object, SympyProcessGraphRule] = (
    MappingProxyType({
        sympy.Symbol: SympyProcessGraphRule("input", node_type="Input"),
        sympy.IndexedBase: SympyProcessGraphRule(
            "input", node_type="Input"
        ),
        sympy.Integer: SympyProcessGraphRule("const", node_type="Constant"),
        sympy.Float: SympyProcessGraphRule("const", node_type="Constant"),
        sympy.Rational: SympyProcessGraphRule("const", node_type="Constant"),
        sympy.Add: SympyProcessGraphRule("Add"),
        sympy.Mul: SympyProcessGraphRule("Mul"),
        sympy.Pow: SympyProcessGraphRule("Pow"),
        sympy.Mod: SympyProcessGraphRule("Mod"),
        sympy.Abs: SympyProcessGraphRule("Abs"),
        sympy.sin: SympyProcessGraphRule("Sin"),
        sympy.cos: SympyProcessGraphRule("Cos"),
        sympy.tan: SympyProcessGraphRule("Tan"),
        sympy.tanh: SympyProcessGraphRule("Tanh"),
        sympy.exp: SympyProcessGraphRule("Exp"),
        sympy.log: SympyProcessGraphRule("Log"),
        sympy.floor: SympyProcessGraphRule("Floor"),
        sympy.ceiling: SympyProcessGraphRule("Ceiling"),
        sympy.Min: SympyProcessGraphRule("Min"),
        sympy.Max: SympyProcessGraphRule("Max"),
        sympy.Equality: SympyProcessGraphRule("Equality", ("left", "right")),
        sympy.Unequality: SympyProcessGraphRule(
            "Unequality", ("left", "right")
        ),
        sympy.StrictLessThan: SympyProcessGraphRule(
            "StrictLessThan", ("left", "right")
        ),
        sympy.LessThan: SympyProcessGraphRule(
            "LessThanOrEqual", ("left", "right")
        ),
        sympy.StrictGreaterThan: SympyProcessGraphRule(
            "StrictGreaterThan", ("left", "right")
        ),
        sympy.GreaterThan: SympyProcessGraphRule(
            "GreaterThanOrEqual", ("left", "right")
        ),
        sympy.And: SympyProcessGraphRule("LAnd"),
        sympy.Or: SympyProcessGraphRule("LOr"),
        sympy.Not: SympyProcessGraphRule("LNot", ("operand",)),
        sympy.Xor: SympyProcessGraphRule("LXor"),
        sympy.Indexed: SympyProcessGraphRule("Indexed", ("base", "index")),
        sympy.Piecewise: SympyProcessGraphRule(
            "Select", ("condition", "if_true", "if_false")
        ),
        sympy.Tuple: SympyProcessGraphRule("Tuple"),
        sympy.Function: SympyProcessGraphRule("Call"),
        "getitem": SympyProcessGraphRule("Indexed", ("base", "index")),
        "Bytes": SympyProcessGraphRule("const", node_type="Constant"),
        "String": SympyProcessGraphRule("const", node_type="Constant"),
    })
)


@dataclass(frozen=True)
class SymbolicReductionReport:
    source_nodes: int
    rebuilt_nodes: int
    original: tuple[sympy.Basic, ...]
    reduced: tuple[sympy.Basic, ...]


@dataclass(frozen=True)
class SymbolicProcessModel:
    """A solver-oriented relational view of one ProcessGraph.

    ``expressions`` maps every graph value ID to a SymPy value.  Inputs retain
    their source names; intermediate values use stable ``value_<id>`` names.
    Equations describe data flow without expanding away sharing, while
    ``constraints`` carry facts such as a branch selector being in ``{0, 1}``.
    Unsupported operations remain explicit uninterpreted SymPy functions and
    are listed in ``uninterpreted`` rather than silently receiving invented
    semantics.
    """

    expressions: Mapping[int, sympy.Basic]
    equations: tuple[sympy.Basic, ...]
    constraints: tuple[sympy.Basic, ...]
    inputs: Mapping[str, sympy.Basic]
    outputs: tuple[sympy.Basic, ...]
    uninterpreted: tuple[tuple[int, str], ...]
    node_specs: Mapping[int, "SymbolicProcessNode"]
    ordering_edges: tuple[tuple[int, int], ...]

    @property
    def relations(self) -> tuple[sympy.Basic, ...]:
        """All equations and domain constraints accepted by SymPy solvers."""

        return self.equations + self.constraints


@dataclass(frozen=True)
class SymbolicTransitionUnroll:
    """A bounded recurrence expressed as ordinary SymPy equations."""

    states: tuple[Mapping[str, sympy.Symbol], ...]
    equations: tuple[sympy.Basic, ...]


@dataclass(frozen=True)
class SymbolicProcessNode:
    """Process metadata carried beside a node's mathematical equation."""

    operation: str
    node_type: str
    label: str
    attributes: Mapping[str, Any]
    constant: Any
    tensor: Mapping[str, Any]
    bit_quanta: Mapping[str, Any]
    parents: tuple[tuple[int, str], ...]

_CANONICAL_FUNCTIONS = {
    "Sin": sympy.sin,
    "sin": sympy.sin,
    "Cos": sympy.cos,
    "cos": sympy.cos,
    "Tan": sympy.tan,
    "tan": sympy.tan,
    "Tanh": sympy.tanh,
    "tanh": sympy.tanh,
    "Exp": sympy.exp,
    "exp": sympy.exp,
    "Log": sympy.log,
    "log": sympy.log,
    "Abs": sympy.Abs,
    "abs": sympy.Abs,
    "Sqrt": sympy.sqrt,
    "sqrt": sympy.sqrt,
    "Floor": sympy.floor,
    "floor": sympy.floor,
    "Ceiling": sympy.ceiling,
    "ceiling": sympy.ceiling,
    "Min": sympy.Min,
    "min": sympy.Min,
    "minimum": sympy.Min,
    "Max": sympy.Max,
    "max": sympy.Max,
    "maximum": sympy.Max,
}

_LOGICAL_FUNCTIONS = {
    "LAnd": sympy.And,
    "LOr": sympy.Or,
    "LNot": sympy.Not,
    "LXor": sympy.Xor,
}

_BINARY = {
    "Add": lambda a, b: a + b,
    "add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "mul": lambda a, b: a * b,
    "Div": lambda a, b: a / b,
    "div": lambda a, b: a / b,
    "truediv": lambda a, b: a / b,
    "FloorDiv": lambda a, b: sympy.floor(a / b),
    "floordiv": lambda a, b: sympy.floor(a / b),
    "Mod": sympy.Mod,
    "mod": sympy.Mod,
    "Pow": lambda a, b: a**b,
    "pow": lambda a, b: a**b,
    "Equality": sympy.Eq,
    "equal": sympy.Eq,
    "Unequality": sympy.Ne,
    "not_equal": sympy.Ne,
    "StrictLessThan": sympy.Lt,
    "less": sympy.Lt,
    "LessThanOrEqual": sympy.Le,
    "less_equal": sympy.Le,
    "StrictGreaterThan": sympy.Gt,
    "greater": sympy.Gt,
    "GreaterThanOrEqual": sympy.Ge,
    "greater_equal": sympy.Ge,
    "eq": sympy.Eq,
    "ne": sympy.Ne,
    "lt": sympy.Lt,
    "le": sympy.Le,
    "gt": sympy.Gt,
    "ge": sympy.Ge,
}

_IDENTITY_OPERATIONS = {"return", "Return", "store", "Store", "output", "Output"}

_BOOLEAN_POLYNOMIAL_OPERATIONS = {
    "and": lambda a, b: a * b,
    "logical_and": lambda a, b: a * b,
    "bitand": lambda a, b: a * b,
    "nand": lambda a, b: 1 - a * b,
    "or": lambda a, b: a + b - a * b,
    "logical_or": lambda a, b: a + b - a * b,
    "bitor": lambda a, b: a + b - a * b,
    "xor": lambda a, b: a + b - 2 * a * b,
    "logical_xor": lambda a, b: a + b - 2 * a * b,
    "bitxor": lambda a, b: a + b - 2 * a * b,
    "LAnd": lambda a, b: a * b,
    "LOr": lambda a, b: a + b - a * b,
    "LXor": lambda a, b: a + b - 2 * a * b,
}

_BOOLEAN_NOT_OPERATIONS = {
    "not", "logical_not", "invert", "Invert", "LNot",
}


def _sympy_literal(value: Any) -> sympy.Basic:
    """Represent a Python literal as a genuine SymPy expression node."""

    if isinstance(value, bytes):
        return sympy.Function("Bytes")(*map(sympy.Integer, value))
    if isinstance(value, str):
        return sympy.Function("String")(*map(sympy.Integer, map(ord, value)))
    if isinstance(value, (tuple, list)):
        return sympy.Tuple(*(_sympy_literal(item) for item in value))
    if value is None:
        return sympy.Function("NoneValue")()
    if value is Ellipsis:
        return sympy.Function("EllipsisValue")()
    result = sympy.sympify(value)
    if isinstance(result, sympy.Basic):
        return result
    return sympy.Symbol(repr(value))


def boolean_domain_constraint(value: sympy.Basic) -> sympy.Basic:
    """Return the polynomial constraint which makes ``value`` Boolean.

    Over characteristic zero, ``x * (x - 1) = 0`` has exactly the roots zero
    and one.  This lets algebraic solvers reason about control predicates and
    bit operations without crossing into Python truth-value evaluation.
    """

    value = sympy.sympify(value)
    return sympy.Eq(value * (value - 1), 0)


def boolean_polynomial(operation: str, *operands: sympy.Basic) -> sympy.Basic:
    """Encode a Boolean primitive as a multilinear polynomial.

    The caller must include :func:`boolean_domain_constraint` for every free
    operand when the symbols do not already carry an equivalent assumption.
    """

    operation = str(operation)
    values = tuple(sympy.sympify(value) for value in operands)
    if operation in _BOOLEAN_NOT_OPERATIONS and len(values) == 1:
        return 1 - values[0]
    function = _BOOLEAN_POLYNOMIAL_OPERATIONS.get(operation)
    if function is None or len(values) < 2:
        raise ValueError(
            f"unsupported Boolean polynomial operation {operation!r} "
            f"with {len(values)} operands"
        )
    result = values[0]
    for value in values[1:]:
        result = function(result, value)
    return sympy.expand(result)


def polynomial_select(
    condition: sympy.Basic,
    if_true: sympy.Basic,
    if_false: sympy.Basic,
) -> sympy.Basic:
    """Return the exact 0/1 polynomial encoding of a control-flow merge."""

    condition, if_true, if_false = map(
        sympy.sympify, (condition, if_true, if_false)
    )
    return sympy.expand(if_false + condition * (if_true - if_false))


def _sympy_condition(value: sympy.Basic) -> sympy.Basic:
    value = sympy.sympify(value)
    if (
        value in (sympy.true, sympy.false)
        or isinstance(value, sympy.logic.boolalg.Boolean)
        or getattr(value, "is_Boolean", False)
    ):
        return value
    return sympy.Ne(value, 0)


def _selected_output_ids(
    graph: Any,
    output_ids: Iterable[int] | None,
) -> tuple[int, ...]:
    if output_ids is not None:
        return tuple(int(node_id) for node_id in output_ids)
    deployment_outputs = tuple(
        int(value)
        for value in graph.G.graph.get("deployment_outputs", ())
        if value in graph.G
    )
    identity_table = graph.G.graph.get("identity_table") or {}
    output_names = graph.G.graph.get("function_outputs") or ()
    selected = tuple(
        int(identity_table[name][-1])
        for name in output_names
        if name in identity_table and identity_table[name]
    )
    return deployment_outputs or selected or tuple(map(int, graph.roots))


def unsigned_bit_expression(bits: Sequence[sympy.Basic]) -> sympy.Basic:
    """Recombine little-endian Boolean bits into an unsigned integer."""

    return sympy.Add(*(
        (1 << index) * sympy.sympify(bit)
        for index, bit in enumerate(bits)
    ))


def unroll_symbolic_transition(
    transitions: Mapping[str, sympy.Basic],
    steps: int,
    *,
    initial: Mapping[str, sympy.Basic] | None = None,
    symbol_prefix: str = "state",
) -> SymbolicTransitionUnroll:
    """Turn a simultaneous state recurrence into a bounded equation system.

    ``transitions`` are written using symbols named after the state keys.  All
    right-hand sides are substituted from time ``t`` before any assignment is
    made, matching simultaneous/SSA state updates.  A branch can be expressed
    with :func:`polynomial_select` or ``sympy.Piecewise``.
    """

    if steps < 0:
        raise ValueError("steps must be non-negative")
    names = tuple(str(name) for name in transitions)
    if not names:
        raise ValueError("at least one state transition is required")
    if set(names) != set(map(str, transitions.keys())):
        raise ValueError("state names must have distinct string forms")
    base_symbols = {name: sympy.Symbol(name) for name in names}
    states = tuple(
        {
            name: sympy.Symbol(f"{symbol_prefix}_{name}_{step}")
            for name in names
        }
        for step in range(steps + 1)
    )
    equations: list[sympy.Basic] = []
    if initial is not None:
        unknown = set(map(str, initial)) - set(names)
        if unknown:
            raise ValueError(f"initial values name unknown states: {sorted(unknown)!r}")
        for name, value in initial.items():
            equations.append(sympy.Eq(states[0][str(name)], sympy.sympify(value)))
    normalized = {str(name): sympy.sympify(value) for name, value in transitions.items()}
    for step in range(steps):
        substitutions = {
            base_symbols[name]: states[step][name]
            for name in names
        }
        for name in names:
            equations.append(sympy.Eq(
                states[step + 1][name],
                normalized[name].subs(substitutions, simultaneous=True),
            ))
    return SymbolicTransitionUnroll(states, tuple(equations))


def aggressively_simplify_expression(
    expression: sympy.Basic,
    *,
    rounds: int = 2,
) -> sympy.Basic:
    """Try several SymPy normal forms and retain the least costly result.

    SymPy transformations are heuristic and no single normal form dominates
    for every expression.  This bounded search feeds each round's candidates
    through algebraic, rational, power, and trigonometric simplifiers, then
    chooses by operation count with deterministic ``srepr`` length/tie-breaks.
    A transformation which rejects the expression is simply not a candidate.
    """

    if rounds < 1:
        raise ValueError("rounds must be positive")
    original = sympy.sympify(expression)
    transformations = (
        sympy.simplify,
        sympy.cancel,
        sympy.factor,
        sympy.ratsimp,
        sympy.trigsimp,
        lambda value: sympy.powsimp(value, deep=True, force=True),
        lambda value: sympy.factor(sympy.expand(value)),
    )
    candidates = {sympy.srepr(original): original}
    frontier = (original,)
    for _round in range(rounds):
        next_frontier = []
        for candidate in frontier:
            for transform in transformations:
                try:
                    transformed = sympy.sympify(transform(candidate))
                except Exception:
                    # SymPy's transformation APIs do not share one rejection
                    # exception. For example, ratsimp(Tuple(...)) currently
                    # raises AttributeError. A failed heuristic is not a failed
                    # simplification search; retain the other candidates.
                    continue
                representation = sympy.srepr(transformed)
                if representation in candidates:
                    continue
                candidates[representation] = transformed
                next_frontier.append(transformed)
        if not next_frontier:
            break
        frontier = tuple(next_frontier)

    def cost(value: sympy.Basic) -> tuple[int, int, str]:
        representation = sympy.srepr(value)
        return int(sympy.count_ops(value)), len(representation), representation

    return min(candidates.values(), key=cost)


def _sympy_process_graph_rule(
    value: sympy.Basic,
) -> SympyProcessGraphRule | None:
    """Look up an exact, named-function, or inherited translation rule."""

    direct = SYMPY_PROCESS_GRAPH_TRANSLATIONS.get(type(value))
    if direct is not None:
        return direct
    function = getattr(value, "func", None)
    direct = SYMPY_PROCESS_GRAPH_TRANSLATIONS.get(function)
    if direct is not None:
        return direct
    function_name = getattr(function, "__name__", None)
    direct = SYMPY_PROCESS_GRAPH_TRANSLATIONS.get(function_name)
    if direct is not None:
        return direct
    for base in type(value).__mro__[1:]:
        inherited = SYMPY_PROCESS_GRAPH_TRANSLATIONS.get(base)
        if inherited is not None:
            return inherited
    return None


def ingest_sympy_expression(
    graph: Any,
    expression: sympy.Basic,
    *,
    strict: bool = False,
) -> int:
    """Translate a SymPy tree back into canonical ProcessGraph operations.

    Translation is driven by :data:`SYMPY_PROCESS_GRAPH_TRANSLATIONS`.
    Undefined applied functions are explicit ``Call`` nodes whose ``callee``
    attribute retains the SymPy function name. Unknown non-function node
    classes are recorded in ``graph.G.graph['sympy_translation_fallbacks']``;
    ``strict=True`` rejects them instead of emitting an uninterpreted node.
    Common subexpressions retain sharing through a node memo.
    """

    expression = sympy.sympify(expression)
    graph.domain_shape = (1,)
    graph.roots = []
    memo: dict[sympy.Basic, int] = {}
    fallbacks: list[str] = []
    next_id = max(
        (int(node_id) for node_id in graph.G if isinstance(node_id, int)),
        default=-1,
    ) + 1

    def make_node(
        value: sympy.Basic,
        rule: SympyProcessGraphRule,
        parent_ids: Sequence[int],
        roles: Sequence[str],
        attributes: Mapping[str, Any] | None = None,
    ) -> int:
        nonlocal next_id
        node_id = next_id
        next_id += 1
        attributes = dict(attributes or {})
        attributes.setdefault("source_type", type(value).__name__)
        node_type = rule.node_type or rule.operation
        operation = rule.operation
        parents = list(zip(parent_ids, roles))
        graph.G.add_node(
            node_id,
            type=node_type,
            op=operation,
            label=str(value),
            expr_obj=value,
            attributes=attributes,
            constant=attributes.get("value"),
            tensor={},
            bit_quanta=(
                {"quanta": 1}
                if operation in _LOGICAL_FUNCTIONS
                else {}
            ),
            parents=parents,
            children=[],
        )
        graph.node_map[node_id] = value
        for parent_id, role in parents:
            graph.G.add_edge(parent_id, node_id)
            graph.G.nodes[parent_id].setdefault("children", []).append(
                (node_id, role)
            )
        return node_id

    def add_node(value: sympy.Basic) -> int:
        value = sympy.sympify(value)
        if value in memo:
            return memo[value]

        no_literal = object()

        if isinstance(value, (sympy.Symbol, sympy.IndexedBase)):
            rule = SYMPY_PROCESS_GRAPH_TRANSLATIONS[type(value)]
            node_id = make_node(
                value,
                rule,
                (),
                (),
                {
                    "binding_name": str(value),
                    "binding_kind": "symbol",
                },
            )
            memo[value] = node_id
            return node_id

        function_name = getattr(getattr(value, "func", None), "__name__", "")
        # Classify numeric atoms by their SymPy domain before singleton-like
        # literals.  Python considers bool a subclass of int and symbolic
        # runtimes can normalize singleton identity tests; the mathematical
        # type is the durable distinction.  In particular ``One`` must remain
        # integer 1 while BooleanTrue remains a Boolean atom.
        if value == sympy.pi:
            # Pi remains a semantic operation until backend lowering.  This
            # lets a caller choose a native literal, a bounded construction,
            # or rejection without changing the authored SymPy expression.
            rule = SympyProcessGraphRule("Pi", node_type="Constant")
            node_id = make_node(
                value, rule, (), (), {"constant_identity": "pi"}
            )
            graph.G.nodes[node_id]["tensor"] = {
                "dtype": "float64", "shape": (),
            }
            memo[value] = node_id
            return node_id

        if value.is_Number:
            if value.is_Integer:
                literal: Any = int(value)
            elif isinstance(value, sympy.Float):
                literal = float(value)
            else:
                literal = value
        elif isinstance(value, sympy.NumberSymbol):
            # Exact named constants such as pi and E are symbolic atoms, not
            # calls or uninterpreted operators. Retain the exact SymPy value
            # as a canonical ProcessGraph constant.
            literal = value
        elif value is sympy.true or value is sympy.false:
            literal = bool(value)
        elif function_name == "Bytes":
            literal = bytes(int(item) for item in value.args)
        elif function_name == "String":
            literal = "".join(chr(int(item)) for item in value.args)
        elif function_name == "NoneValue" and not value.args:
            literal = None
        elif function_name == "EllipsisValue" and not value.args:
            literal = Ellipsis
        else:
            literal = no_literal
        if literal is not no_literal:
            rule = SympyProcessGraphRule("const", node_type="Constant")
            node_id = make_node(
                value, rule, (), (), {"value": literal}
            )
            memo[value] = node_id
            return node_id

        rule = _sympy_process_graph_rule(value)
        if rule is None:
            fallback_name = type(value).__name__
            if strict:
                raise TypeError(
                    "no SymPy to ProcessGraph translation rule for "
                    f"{fallback_name}: {value!r}"
                )
            fallbacks.append(fallback_name)
            rule = SympyProcessGraphRule(fallback_name)

        if isinstance(value, sympy.Piecewise):
            # Lower N arms into nested three-input Select nodes. The final
            # implicit value is NaN, matching SymPy Piecewise semantics when
            # no condition is true and no unconditional arm was supplied.
            selected_id: int | None = None
            for pair in reversed(value.args):
                arm, condition = pair.args
                arm_id = add_node(arm)
                if condition is sympy.true:
                    selected_id = arm_id
                    continue
                if selected_id is None:
                    selected_id = add_node(sympy.nan)
                condition_id = add_node(condition)
                selected_id = make_node(
                    value,
                    rule,
                    (condition_id, arm_id, selected_id),
                    rule.roles,
                    {"piecewise_arms": len(value.args)},
                )
            if selected_id is None:
                selected_id = add_node(sympy.nan)
            memo[value] = selected_id
            return selected_id

        arguments = tuple(value.args)
        # SymPy represents associative arithmetic as variadic nodes, while
        # repository SSA and every scalar backend give Add/Mul/Min/Max an
        # exact binary arity.  Preserve the authored expression as a stable
        # left-associated ProcessGraph chain instead of asking each backend to
        # invent its own n-ary convention.
        if rule.operation in {"Add", "Mul", "Min", "Max"} and len(arguments) > 2:
            left = add_node(arguments[0])
            accumulated = arguments[0]
            for index, argument in enumerate(arguments[1:], start=1):
                right = add_node(argument)
                accumulated = value if index == len(arguments) - 1 else value.func(
                    accumulated, argument, evaluate=False,
                )
                left = make_node(
                    accumulated,
                    rule,
                    (left, right),
                    ("arg:0", "arg:1"),
                )
            memo[value] = left
            return left
        parent_ids = tuple(add_node(argument) for argument in arguments)
        if isinstance(value, sympy.Indexed) or function_name == "getitem":
            roles = ("base", *("index" for _ in arguments[1:]))
        elif rule.roles:
            if len(rule.roles) != len(arguments):
                raise ValueError(
                    f"translation rule for {type(value).__name__} expects "
                    f"{len(rule.roles)} operands, got {len(arguments)}"
                )
            roles = rule.roles
        else:
            roles = tuple(f"arg:{index}" for index in range(len(arguments)))
        attributes: dict[str, Any] = {}
        if rule.operation == "Call":
            attributes["callee"] = function_name or type(value).__name__
        node_id = make_node(
            value,
            rule,
            parent_ids,
            roles,
            attributes,
        )
        memo[value] = node_id
        return node_id

    root = add_node(expression)
    graph.roots.append(root)
    graph.G.graph["function_outputs"] = ("result",)
    graph.G.graph["symbolic_source"] = "sympy"
    graph.G.graph["sympy_translation_table"] = (
        "SYMPY_PROCESS_GRAPH_TRANSLATIONS"
    )
    graph.G.graph["sympy_translation_fallbacks"] = tuple(fallbacks)
    return root


def ingest_sympy_expressions(
    graph: Any,
    expressions: Sequence[sympy.Basic],
    *,
    output_names: Sequence[str] | None = None,
    strict: bool = False,
) -> tuple[int, ...]:
    """Ingest a named expression set as one shared canonical ProcessGraph.

    A temporary SymPy ``Tuple`` gives :func:`ingest_sympy_expression` one tree
    in which its existing memo can retain common subexpressions across every
    result.  The tuple is only a construction envelope: it is removed again,
    and its operands become the graph's actual deployment roots.  Thus no
    invented tuple operation reaches repository SSA or a backend.
    """

    authored = tuple(sympy.sympify(expression) for expression in expressions)
    if not authored:
        raise ValueError("SymPy expression set must contain at least one output")
    names = (
        tuple(str(name) for name in output_names)
        if output_names is not None
        else tuple(f"result_{index}" for index in range(len(authored)))
    )
    if len(names) != len(authored):
        raise ValueError("SymPy output names must match the expression count")
    if len(names) != len(set(names)):
        raise ValueError("SymPy output names must be unique")

    tuple_root = ingest_sympy_expression(
        graph, sympy.Tuple(*authored), strict=strict,
    )
    tuple_data = graph.G.nodes[tuple_root]
    if tuple_data.get("op") != "Tuple":
        raise RuntimeError("SymPy expression-set envelope did not lower to Tuple")
    roots = tuple(int(parent) for parent, _role in tuple_data.get("parents", ()))
    if len(roots) != len(authored):
        raise RuntimeError("SymPy expression-set envelope lost an output")

    for root in roots:
        children = graph.G.nodes[root].get("children") or []
        graph.G.nodes[root]["children"] = [
            child for child in children if int(child[0]) != int(tuple_root)
        ]
    graph.G.remove_node(tuple_root)
    graph.node_map.pop(tuple_root, None)
    graph.roots = list(roots)
    graph.G.graph.update(
        function_outputs=names,
        deployment_outputs=roots,
        deployment_inputs=tuple(
            int(node_id)
            for node_id, data in graph.G.nodes(data=True)
            if data.get("op") in {"input", "Input", "Symbol"}
        ),
    )
    return roots


def process_graph_to_sympy_expressions(
    graph: Any,
    output_ids: Iterable[int] | None = None,
) -> tuple[sympy.Basic, ...]:
    """Render canonical ProcessGraph outputs as SymPy expressions."""

    output_ids = _selected_output_ids(graph, output_ids)

    cache: dict[int, sympy.Basic] = {}
    identity_names = {
        int(value_id): str(name)
        for name, value_ids in (
            graph.G.graph.get("identity_table") or {}
        ).items()
        for value_id in value_ids
    }
    indexed_bases = {
        int(parent_id)
        for _node_id, data in graph.G.nodes(data=True)
        if str(data.get("op") or data.get("type")) in {"Indexed", "indexed"}
        for parent_id, role in data.get("parents") or ()
        if str(role) == "base"
    }

    def emit(node_id: int) -> sympy.Basic:
        if node_id in cache:
            return cache[node_id]
        data = graph.G.nodes[node_id]
        operation = str(data.get("op") or data.get("type"))
        attributes = data.get("attributes") or {}
        parents_by_role: dict[str, list[tuple[int, sympy.Basic]]] = defaultdict(list)
        for parent_id, role in data.get("parents") or ():
            parents_by_role[str(role)].append(
                (int(parent_id), emit(int(parent_id)))
            )

        if operation in {"input", "Input", "Symbol"}:
            name = str(
                attributes.get("binding_name")
                or identity_names.get(node_id)
                or data.get("label")
                or f"value_{node_id}"
            )
            result = (
                sympy.IndexedBase(name)
                if node_id in indexed_bases
                else sympy.Symbol(name)
            )
        elif operation in {
            "const", "constant", "Constant", "Integer", "Float", "Rational"
        }:
            value = attributes.get("value", data.get("constant"))
            result = _sympy_literal(value)
        else:
            ordered = [
                value
                for values in parents_by_role.values()
                for _node, value in values
            ]
            if operation in {"Indexed", "indexed"}:
                base = parents_by_role.get("base", ())
                indices = parents_by_role.get("index", ())
                if len(base) != 1 or not indices:
                    raise ValueError(
                        f"Indexed node {node_id} lacks base/index roles"
                    )
                try:
                    result = sympy.Indexed(
                        base[0][1],
                        *(value for _node, value in indices),
                    )
                except TypeError:
                    # SymPy Indexed only accepts an addressable base such as
                    # Symbol/IndexedBase. ProcessGraph can index any computed
                    # tensor value, so retain that dependency as an explicit
                    # mathematical getitem until tensor-index semantics are
                    # scalarized by a later pass.
                    result = sympy.Function("getitem")(
                        base[0][1],
                        *(value for _node, value in indices),
                    )
            elif operation in {"select", "Select", "Phi", "phi", "mu"}:
                condition = (
                    parents_by_role.get("condition")
                    or parents_by_role.get("test")
                    or parents_by_role.get("selector")
                )
                if_true = (
                    parents_by_role.get("if_true")
                    or parents_by_role.get("body")
                )
                if_false = (
                    parents_by_role.get("if_false")
                    or parents_by_role.get("orelse")
                )
                if operation == "mu" and len(ordered) == 3 and not condition:
                    if_false = [(0, ordered[0])]
                    if_true = [(0, ordered[1])]
                    condition = [(0, ordered[2])]
                if not condition or not if_true or not if_false:
                    raise ValueError(
                        f"control merge node {node_id} lacks selector/arm roles"
                    )
                result = sympy.Piecewise(
                    (if_true[0][1], _sympy_condition(condition[0][1])),
                    (if_false[0][1], True),
                )
            elif operation in _IDENTITY_OPERATIONS and len(ordered) == 1:
                result = ordered[0]
            elif operation in _BINARY and len(ordered) >= 2:
                result = ordered[0]
                for value in ordered[1:]:
                    result = _BINARY[operation](result, value)
            elif operation in _CANONICAL_FUNCTIONS:
                result = _CANONICAL_FUNCTIONS[operation](*ordered)
            elif operation in _LOGICAL_FUNCTIONS:
                result = _LOGICAL_FUNCTIONS[operation](*ordered)
            elif operation == "Tuple":
                result = sympy.Tuple(*ordered)
            elif operation == "Call" and attributes.get("callee"):
                result = sympy.Function(str(attributes["callee"]))(*ordered)
            elif operation in {"Neg", "neg"} and len(ordered) == 1:
                result = -ordered[0]
            elif (
                operation in _BOOLEAN_NOT_OPERATIONS
                and len(ordered) == 1
                and int((data.get("bit_quanta") or {}).get("quanta", 0)) == 1
            ):
                result = boolean_polynomial(operation, ordered[0])
            elif (
                operation in _BOOLEAN_POLYNOMIAL_OPERATIONS
                and len(ordered) >= 2
                and int((data.get("bit_quanta") or {}).get("quanta", 0)) == 1
            ):
                result = boolean_polynomial(operation, *ordered)
            else:
                result = sympy.Function(operation)(*ordered)
        cache[node_id] = result
        return result

    return tuple(emit(int(node_id)) for node_id in output_ids)


def process_graph_to_sympy_relations(
    graph: Any,
    output_ids: Iterable[int] | None = None,
    *,
    live_only: bool = True,
) -> SymbolicProcessModel:
    """Encode a ProcessGraph as equations suitable for inverse solving.

    Unlike the expression projection, this form keeps an equation per graph
    node.  Branch merges use a polynomial mux and one-bit Turing primitives
    use their exact multilinear Boolean polynomials.  Wider BitOps graphs must
    first expose individual bit lanes; otherwise the operation is retained as
    an uninterpreted vector function and reported as such.
    """

    import networkx as nx

    selected_outputs = _selected_output_ids(graph, output_ids)
    if live_only:
        live_nodes = set(selected_outputs)
        pending = list(selected_outputs)
        while pending:
            child = pending.pop()
            semantic_parents = {
                *graph.G.predecessors(child),
                *(
                    int(parent)
                    for parent, _role in graph.G.nodes[child].get("parents") or ()
                    if parent in graph.G
                ),
            }
            for parent in semantic_parents - live_nodes:
                live_nodes.add(parent)
                pending.append(parent)
    else:
        live_nodes = set(map(int, graph.G.nodes))
    identity_names = {
        int(value_id): str(name)
        for name, value_ids in (
            graph.G.graph.get("identity_table") or {}
        ).items()
        for value_id in value_ids
    }
    indexed_bases = {
        int(parent_id)
        for node_id in live_nodes
        for parent_id, role in (
            graph.G.nodes[node_id].get("parents") or ()
        )
        if str(
            graph.G.nodes[node_id].get("op")
            or graph.G.nodes[node_id].get("type")
        ) in {"Indexed", "indexed"}
        and str(role) == "base"
    }
    expressions: dict[int, sympy.Basic] = {}
    inputs: dict[str, sympy.Basic] = {}
    equations: list[sympy.Basic] = []
    constraints: list[sympy.Basic] = []
    constrained: set[sympy.Basic] = set()
    uninterpreted: list[tuple[int, str]] = []
    node_specs: dict[int, SymbolicProcessNode] = {}

    def constrain_boolean(value: sympy.Basic) -> None:
        if value not in constrained:
            constraints.append(boolean_domain_constraint(value))
            constrained.add(value)

    for node_id in nx.topological_sort(graph.G):
        node_id = int(node_id)
        if node_id not in live_nodes:
            continue
        data = graph.G.nodes[node_id]
        operation = str(data.get("op") or data.get("type"))
        attributes = data.get("attributes") or {}
        accounting = data.get("bit_quanta") or {}
        if hasattr(accounting, "__dict__"):
            accounting = dict(accounting.__dict__)
        node_specs[node_id] = SymbolicProcessNode(
            operation=operation,
            node_type=str(data.get("type") or operation),
            label=str(data.get("label") or operation),
            attributes=copy.deepcopy(dict(attributes)),
            constant=copy.deepcopy(data.get("constant")),
            tensor=copy.deepcopy(dict(data.get("tensor") or {})),
            bit_quanta=copy.deepcopy(dict(accounting)),
            parents=tuple(
                (int(parent), str(role))
                for parent, role in data.get("parents") or ()
                if int(parent) in live_nodes
            ),
        )
        parents_by_role: dict[str, list[sympy.Basic]] = defaultdict(list)
        ordered: list[sympy.Basic] = []
        for parent_id, role in data.get("parents") or ():
            parent = expressions[int(parent_id)]
            parents_by_role[str(role)].append(parent)
            ordered.append(parent)

        if operation in {"input", "Input", "Symbol"}:
            provenance_metadata = getattr(
                data.get("expr_obj"), "metadata", {}
            ) or {}
            name = str(
                attributes.get("binding_name")
                or provenance_metadata.get("name")
                or identity_names.get(node_id)
                or data.get("label")
                or f"value_{node_id}"
            )
            value = inputs.setdefault(
                name,
                (
                    sympy.IndexedBase(name)
                    if node_id in indexed_bases
                    else sympy.Symbol(name)
                ),
            )
            expressions[node_id] = value
            continue

        value = sympy.Symbol(f"value_{node_id}")
        expressions[node_id] = value
        rhs: sympy.Basic
        if operation in {
            "const", "constant", "Constant", "Integer", "Float", "Rational"
        }:
            rhs = _sympy_literal(attributes.get("value", data.get("constant")))
        elif operation in {"Indexed", "indexed"}:
            base = parents_by_role.get("base") or ordered[:1]
            indices = parents_by_role.get("index") or ordered[1:]
            if len(base) != 1 or not indices:
                raise ValueError(
                    f"Indexed node {node_id} lacks base/index operands"
                )
            try:
                rhs = sympy.Indexed(base[0], *indices)
            except TypeError:
                rhs = sympy.Function("getitem")(base[0], *indices)
        elif operation in _IDENTITY_OPERATIONS and len(ordered) == 1:
            rhs = ordered[0]
        elif operation in {"select", "Select", "Phi", "phi", "mu"}:
            condition = (
                parents_by_role.get("condition")
                or parents_by_role.get("test")
                or parents_by_role.get("selector")
            )
            if_true = parents_by_role.get("if_true") or parents_by_role.get("body")
            if_false = parents_by_role.get("if_false") or parents_by_role.get("orelse")
            if operation == "mu" and len(ordered) == 3 and not condition:
                if_false = [ordered[0]]
                if_true = [ordered[1]]
                condition = [ordered[2]]
            if not condition or not if_true or not if_false:
                raise ValueError(
                    f"control merge node {node_id} lacks selector/arm roles"
                )
            constrain_boolean(condition[0])
            rhs = polynomial_select(condition[0], if_true[0], if_false[0])
        elif operation in _BOOLEAN_NOT_OPERATIONS and len(ordered) == 1:
            quanta = int((data.get("bit_quanta") or {}).get("quanta", 1))
            if quanta == 1:
                constrain_boolean(ordered[0])
                constrain_boolean(value)
                rhs = boolean_polynomial(operation, ordered[0])
            else:
                rhs = sympy.Function(operation)(*ordered)
                uninterpreted.append((node_id, operation))
        elif operation in _BOOLEAN_POLYNOMIAL_OPERATIONS and len(ordered) >= 2:
            quanta = int((data.get("bit_quanta") or {}).get("quanta", 1))
            if quanta == 1:
                for operand in ordered:
                    constrain_boolean(operand)
                constrain_boolean(value)
                rhs = boolean_polynomial(operation, *ordered)
            else:
                rhs = sympy.Function(operation)(*ordered)
                uninterpreted.append((node_id, operation))
        elif operation in _BINARY and len(ordered) >= 2:
            rhs = ordered[0]
            for operand in ordered[1:]:
                rhs = _BINARY[operation](rhs, operand)
            if operation in {
                "Equality", "equal", "eq", "Unequality", "not_equal", "ne",
                "StrictLessThan", "less", "lt", "LessThanOrEqual", "less_equal", "le",
                "StrictGreaterThan", "greater", "gt", "GreaterThanOrEqual", "greater_equal", "ge",
            }:
                rhs = sympy.Piecewise((1, rhs), (0, True))
                constrain_boolean(value)
        elif operation in {"Neg", "neg"} and len(ordered) == 1:
            rhs = -ordered[0]
        elif operation in _CANONICAL_FUNCTIONS:
            rhs = _CANONICAL_FUNCTIONS[operation](*ordered)
        elif operation in _LOGICAL_FUNCTIONS:
            polynomial_operation = {
                "LAnd": "and",
                "LOr": "or",
                "LNot": "not",
                "LXor": "xor",
            }[operation]
            for operand in ordered:
                constrain_boolean(operand)
            constrain_boolean(value)
            rhs = boolean_polynomial(polynomial_operation, *ordered)
        elif operation == "Tuple":
            rhs = sympy.Tuple(*ordered)
        elif operation == "Call" and attributes.get("callee"):
            rhs = sympy.Function(str(attributes["callee"]))(*ordered)
            uninterpreted.append((node_id, str(attributes["callee"])))
        else:
            rhs = sympy.Function(operation)(*ordered)
            uninterpreted.append((node_id, operation))
        equations.append(sympy.Eq(value, rhs))

    return SymbolicProcessModel(
        expressions=expressions,
        equations=tuple(equations),
        constraints=tuple(constraints),
        inputs=inputs,
        outputs=tuple(expressions[node_id] for node_id in selected_outputs),
        uninterpreted=tuple(uninterpreted),
        node_specs=node_specs,
        ordering_edges=tuple(
            (int(source), int(target))
            for source, target in graph.G.edges
            if int(source) in live_nodes and int(target) in live_nodes
        ),
    )


def aggressively_simplify_process_relations(
    model: SymbolicProcessModel,
    *,
    rounds: int = 1,
) -> tuple[sympy.Equality, ...]:
    """Simplify every program equation without collapsing program nodes.

    A compact output expression is useful for solving but cannot carry
    isolated effects. Per-node equations let SymPy transform every right-hand
    side while retaining one named result for every ProcessGraph operation.
    """

    reduced: list[sympy.Equality] = []
    for equation in model.equations:
        if not isinstance(equation, sympy.Equality):
            raise TypeError(f"expected a SymPy Equality, got {equation!r}")
        reduced.append(sympy.Eq(
            equation.lhs,
            aggressively_simplify_expression(equation.rhs, rounds=rounds),
            evaluate=False,
        ))
    return tuple(reduced)


def ingest_sympy_process_model(
    graph: Any,
    model: SymbolicProcessModel,
    *,
    equations: Sequence[sympy.Equality] | None = None,
) -> Mapping[int, int]:
    """Rebuild a complete ProcessGraph from its SymPy equation model.

    Every equation is ingested independently so identical effect expressions
    do not collapse through SymPy structural equality. Equation result symbols
    bind later equations to the rebuilt producer. Original non-data ordering
    edges and operation metadata accompany the algebra and are restored after
    translation. The returned mapping is ``original node id -> rebuilt id``.
    """

    import networkx as nx

    from ..transmogrifier.graph.graph_express2 import ProcessGraph

    if graph.G.number_of_nodes():
        raise ValueError("SymPy process-model ingestion requires an empty graph")
    selected_equations = tuple(model.equations if equations is None else equations)
    lhs_to_original = {
        expression: int(node_id)
        for node_id, expression in model.expressions.items()
    }
    graph.domain_shape = (1,)
    graph.roots = []
    graph.node_map = {}
    symbol_bindings: dict[sympy.Basic, int] = {}
    original_to_rebuilt: dict[int, int] = {}
    fallbacks: list[str] = []
    next_id = 0

    def append_node(
        *,
        expression: sympy.Basic,
        data: Mapping[str, Any],
        parents: Sequence[tuple[int, str]],
    ) -> int:
        nonlocal next_id
        node_id = next_id
        next_id += 1
        payload = copy.deepcopy(dict(data))
        payload["parents"] = list(parents)
        payload["children"] = []
        graph.G.add_node(node_id, **payload)
        graph.node_map[node_id] = expression
        for parent_id, role in parents:
            graph.G.add_edge(parent_id, node_id)
            graph.G.nodes[parent_id].setdefault("children", []).append(
                (node_id, role)
            )
        return node_id

    # Inputs have no equations, so materialize their carried specifications
    # before translating equation right-hand sides.
    for original_id, spec in model.node_specs.items():
        if spec.operation not in {"input", "Input", "Symbol"}:
            continue
        expression = model.expressions[original_id]
        node_id = append_node(
            expression=expression,
            data={
                "type": spec.node_type,
                "op": spec.operation,
                "label": spec.label,
                "expr_obj": expression,
                "attributes": copy.deepcopy(dict(spec.attributes)),
                "constant": copy.deepcopy(spec.constant),
                "tensor": copy.deepcopy(dict(spec.tensor)),
                "bit_quanta": copy.deepcopy(dict(spec.bit_quanta)),
            },
            parents=(),
        )
        symbol_bindings[expression] = node_id
        original_to_rebuilt[original_id] = node_id

    def merge_expression(expression: sympy.Basic) -> tuple[int, bool]:
        temporary = ProcessGraph(materialize_memory=False)
        ingest_sympy_expression(temporary, expression)
        fallbacks.extend(temporary.G.graph["sympy_translation_fallbacks"])
        local: dict[int, int] = {}
        created: set[int] = set()
        for temporary_id in nx.topological_sort(temporary.G):
            temporary_id = int(temporary_id)
            data = temporary.G.nodes[temporary_id]
            temporary_expression = temporary.node_map[temporary_id]
            if (
                data.get("op") in {"input", "Input", "Symbol"}
                and temporary_expression in symbol_bindings
            ):
                local[temporary_id] = symbol_bindings[temporary_expression]
                continue
            parents = tuple(
                (local[int(parent_id)], str(role))
                for parent_id, role in data.get("parents") or ()
            )
            node_id = append_node(
                expression=temporary_expression,
                data={
                    key: value
                    for key, value in data.items()
                    if key not in {"parents", "children"}
                },
                parents=parents,
            )
            local[temporary_id] = node_id
            created.add(node_id)
        root_id = local[int(temporary.roots[0])]
        return root_id, root_id in created

    for equation in selected_equations:
        if not isinstance(equation, sympy.Equality):
            raise TypeError(f"expected a SymPy Equality, got {equation!r}")
        original_id = lhs_to_original.get(equation.lhs)
        if original_id is None or original_id not in model.node_specs:
            raise KeyError(f"equation has no ProcessGraph node: {equation!r}")
        spec = model.node_specs[original_id]
        root_id, root_created = merge_expression(equation.rhs)
        if not root_created:
            bound_parents = tuple(
                (root_id, role)
                for _parent, role in spec.parents[:1]
            ) or ((root_id, "value"),)
            root_id = append_node(
                expression=equation.rhs,
                data={
                    "type": spec.node_type,
                    "op": spec.operation,
                    "label": spec.label,
                    "expr_obj": equation.rhs,
                    "attributes": copy.deepcopy(dict(spec.attributes)),
                    "constant": copy.deepcopy(spec.constant),
                    "tensor": copy.deepcopy(dict(spec.tensor)),
                    "bit_quanta": copy.deepcopy(dict(spec.bit_quanta)),
                },
                parents=bound_parents,
            )
        root_data = graph.G.nodes[root_id]
        function_name = getattr(
            getattr(equation.rhs, "func", None), "__name__", ""
        )
        if (
            function_name == spec.operation
            or root_data.get("op") == spec.operation
            or root_data.get("op") == "const"
        ):
            root_data.update(
                type=spec.node_type,
                op=spec.operation,
                label=spec.label,
                attributes=copy.deepcopy(dict(spec.attributes)),
                constant=copy.deepcopy(spec.constant),
            )
            if len(root_data.get("parents") or ()) == len(spec.parents):
                root_data["parents"] = [
                    (parent_id, role)
                    for (parent_id, _old_role), (_old_parent, role) in zip(
                        root_data["parents"], spec.parents
                    )
                ]
        root_data["tensor"] = copy.deepcopy(dict(spec.tensor))
        root_data["bit_quanta"] = copy.deepcopy(dict(spec.bit_quanta))
        root_data.setdefault("attributes", {})[
            "symbolic_original_node_id"
        ] = original_id
        original_to_rebuilt[original_id] = root_id
        symbol_bindings[equation.lhs] = root_id

    for original_source, original_target in model.ordering_edges:
        source = original_to_rebuilt.get(original_source)
        target = original_to_rebuilt.get(original_target)
        if source is None or target is None or graph.G.has_edge(source, target):
            continue
        graph.G.add_edge(source, target, ordering_only=True)

    graph.roots = [symbol_bindings[output] for output in model.outputs]
    graph.G.graph.update(
        symbolic_source="sympy_process_model",
        sympy_translation_table="SYMPY_PROCESS_GRAPH_TRANSLATIONS",
        sympy_translation_fallbacks=tuple(fallbacks),
        symbolic_constraints=model.constraints,
        deployment_inputs=tuple(
            original_to_rebuilt[original_id]
            for original_id, spec in model.node_specs.items()
            if spec.operation in {"input", "Input", "Symbol"}
        ),
        deployment_outputs=tuple(graph.roots),
        function_outputs=tuple(
            f"result_{index}" for index in range(len(graph.roots))
        ),
    )
    return original_to_rebuilt


def symbolically_reduce_process_graph(
    graph: Any,
    *,
    aggressive: bool = False,
    aggressive_rounds: int = 2,
):
    """Round-trip one planner-filtered graph through SymPy simplification."""

    from ..transmogrifier.graph.graph_express2 import ProcessGraph

    original = process_graph_to_sympy_expressions(graph)
    reduced = tuple(
        (
            aggressively_simplify_expression(
                expression,
                rounds=aggressive_rounds,
            )
            if aggressive
            else sympy.simplify(expression)
        )
        for expression in original
    )
    if len(reduced) != 1:
        raise NotImplementedError(
            "symbolic ProcessGraph reconstruction currently requires one "
            f"deployment output, got {len(reduced)}"
        )
    rebuilt = ProcessGraph(materialize_memory=False)
    ingest_sympy_expression(rebuilt, reduced[0])

    source_by_name = {}
    source_name_by_id = {}
    identity_table = graph.G.graph.get("identity_table") or {}
    for name, value_ids in identity_table.items():
        for value_id in value_ids:
            if value_id in graph.G:
                source_by_name[str(name)] = graph.G.nodes[value_id]
                source_name_by_id[int(value_id)] = str(name)
    for _node_id, data in rebuilt.G.nodes(data=True):
        if data.get("type") != "Input":
            continue
        attributes = data.setdefault("attributes", {})
        name = str(attributes.get("binding_name"))
        if name.startswith("value_") and name[6:].isdigit():
            source_name = source_name_by_id.get(int(name[6:]))
            if source_name is not None:
                attributes["binding_name"] = source_name
                data["label"] = source_name
                name = source_name
        source = source_by_name.get(name)
        if source is not None:
            data["tensor"] = dict(source.get("tensor") or {})

    output_id = rebuilt.roots[0]
    store_id = max(int(node_id) for node_id in rebuilt.G) + 1
    rebuilt.G.add_node(
        store_id,
        type="Store",
        op="store",
        label="symbolic_result",
        expr_obj=None,
        attributes={"symbolically_reduced": True},
        constant=None,
        tensor=dict(rebuilt.G.nodes[output_id].get("tensor") or {}),
        parents=[(output_id, "value")],
        children=[],
    )
    rebuilt.G.add_edge(output_id, store_id)
    rebuilt.G.nodes[output_id]["children"].append((store_id, "value"))
    rebuilt.roots = [store_id]
    rebuilt.G.graph.update(
        source_kind="sympy_reduction",
        deployment_inputs=tuple(
            node_id
            for node_id, data in rebuilt.G.nodes(data=True)
            if data.get("type") == "Input"
        ),
        deployment_outputs=(output_id,),
        symbolic_original=tuple(map(str, original)),
        symbolic_reduced=tuple(map(str, reduced)),
    )
    return rebuilt, SymbolicReductionReport(
        source_nodes=graph.G.number_of_nodes(),
        rebuilt_nodes=rebuilt.G.number_of_nodes(),
        original=original,
        reduced=reduced,
    )


def process_graph_to_sympy_package(graph: Any):
    """Compatibility package matching the historical ``to_sympy`` API."""

    from ..transmogrifier.graph.graph_express2 import ExpressionTensor

    expressions = process_graph_to_sympy_expressions(graph)
    registry = list(expressions)
    try:
        import torch
    except Exception:
        torch = None
    if torch is None:
        import numpy as np

        data = np.arange(len(expressions), dtype=int).reshape(1, 1, -1)
    else:
        data = torch.arange(
            len(expressions), dtype=torch.long
        ).reshape(1, 1, -1)
    return registry, ExpressionTensor(
        data,
        contexts=[0],
        sequence_length=1,
        domain_shape=(len(expressions),),
        function_index=None,
    )


__all__ = [
    "SYMPY_PROCESS_GRAPH_TRANSLATIONS",
    "SympyProcessGraphRule",
    "SymbolicProcessNode",
    "SymbolicProcessModel",
    "ingest_sympy_expression",
    "ingest_sympy_expressions",
    "process_graph_to_sympy_expressions",
    "process_graph_to_sympy_relations",
    "ingest_sympy_process_model",
    "process_graph_to_sympy_package",
    "symbolically_reduce_process_graph",
    "SymbolicReductionReport",
    "SymbolicTransitionUnroll",
    "aggressively_simplify_expression",
    "aggressively_simplify_process_relations",
    "boolean_domain_constraint",
    "boolean_polynomial",
    "polynomial_select",
    "unsigned_bit_expression",
    "unroll_symbolic_transition",
]
