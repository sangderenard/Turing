"""Reduce marked Python class dispatch into the existing control vocabulary.

An ``AbstractTensorStateMachine`` subclass supplies ``transition`` whose
outer statement is a ``match`` over one class-owned scalar state field. Each
literal case dispatches to one method on ``self``. Those methods remain normal
numeric regions; this module creates only the existing ``StateMachineTick``
control shell and never introduces an SSA operator.

Two layers, because a graph value id exists only after reduction:

* ``plan_marked_state_machines`` reads the module AST (before any graph node
  exists) and says WHICH ``match`` is a dispatch (``StateMachineASTPlan``:
  class, state field, case -> method).  Its ``control`` is the AST-level
  shell with a placeholder state uniform; it names no graph value.
* the reducer ingests a planned dispatch (``planned_dispatch`` /
  ``dispatch_arms_are_effects`` gate it), the state selector and each case
  literal becoming operands of the dispatch node, and
  ``install_state_machine_control`` builds the tick FROM THAT NODE -- its
  identity, its state value, its literals, the callsites in each arm -- into
  the function's ControlProgram.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

from .control_source import (
    ControlProgram,
    ControlUniform,
    StateMachineTick,
    StatementBlock,
)


def _qualified_name(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        owner = _qualified_name(expression.value)
        return f"{owner}.{expression.attr}" if owner else expression.attr
    return None


def _is_marked(definition: ast.ClassDef) -> bool:
    return any(
        (name := _qualified_name(base)) is not None
        and name.rsplit(".", 1)[-1] == "AbstractTensorStateMachine"
        for base in definition.bases
    )


@dataclass(frozen=True, slots=True)
class StateMachineASTShortfall:
    class_name: str
    location: str
    reason: str


@dataclass(frozen=True, slots=True)
class StateMachineASTPlan:
    class_name: str
    state_field: str
    case_methods: tuple[tuple[int, str], ...]
    control: ControlProgram


def _transition(definition: ast.ClassDef):
    return next((
        member for member in definition.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        and member.name == "transition"
    ), None)


def _outer_match(transition):
    executable = [
        statement for statement in transition.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]
    return (
        executable[0]
        if len(executable) == 1 and isinstance(executable[0], ast.Match)
        else None
    )


def _state_field(subject: ast.expr) -> str | None:
    if (
        isinstance(subject, ast.Attribute)
        and isinstance(subject.value, ast.Name)
        and subject.value.id in {"self", "state"}
    ):
        return subject.attr
    if isinstance(subject, ast.Name):
        return subject.id
    # Ordinary Python runtime spelling for a scalar tensor selector:
    # ``match int(state.phase.item())``.
    if (
        isinstance(subject, ast.Call)
        and isinstance(subject.func, ast.Name)
        and subject.func.id == "int"
        and len(subject.args) == 1
    ):
        inner = subject.args[0]
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "item"
            and not inner.args
        ):
            return _state_field(inner.func.value)
    return None


def _literal_case(pattern: ast.pattern) -> int | None:
    if isinstance(pattern, ast.MatchValue) and isinstance(pattern.value, ast.Constant):
        value = pattern.value.value
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    if isinstance(pattern, ast.MatchSingleton) and isinstance(pattern.value, bool):
        return int(pattern.value)
    return None


def _case_method(case: ast.match_case) -> str | None:
    if case.guard is not None or len(case.body) != 1:
        return None
    statement = case.body[0]
    expression = statement.value if isinstance(statement, (ast.Expr, ast.Return)) else None
    if not isinstance(expression, ast.Call):
        return None
    function = expression.func
    if not (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "self"
    ):
        return None
    return function.attr


def lower_marked_state_machine_class(
    definition: ast.ClassDef,
    *,
    state_value_id: int = 0,
) -> tuple[StateMachineASTPlan | None, tuple[StateMachineASTShortfall, ...]]:
    """Build an existing ``StateMachineTick`` for one marked class."""

    if not _is_marked(definition):
        return None, ()
    transition = _transition(definition)
    if transition is None:
        return None, (StateMachineASTShortfall(
            definition.name, "transition",
            "marked state machine has no transition method",
        ),)
    dispatch = _outer_match(transition)
    if dispatch is None:
        return None, (StateMachineASTShortfall(
            definition.name, "transition.body",
            "transition must contain one outer match statement",
        ),)
    field = _state_field(dispatch.subject)
    if field is None:
        return None, (StateMachineASTShortfall(
            definition.name, "transition.match",
            "state selector must be a scalar state field",
        ),)

    declared_methods = {
        member.name for member in definition.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    cases: list[tuple[int, str]] = []
    shortfalls: list[StateMachineASTShortfall] = []
    seen: set[int] = set()
    for index, case in enumerate(dispatch.cases):
        value = _literal_case(case.pattern)
        method = _case_method(case)
        location = f"transition.match.case[{index}]"
        if value is None:
            shortfalls.append(StateMachineASTShortfall(
                definition.name, location,
                "state case must use an integer or boolean literal",
            ))
        elif value in seen:
            shortfalls.append(StateMachineASTShortfall(
                definition.name, location, f"duplicate state case {value}",
            ))
        elif method is None or method not in declared_methods:
            shortfalls.append(StateMachineASTShortfall(
                definition.name, location,
                "state case must call one method defined on self",
            ))
        else:
            seen.add(value)
            cases.append((value, method))
    if shortfalls or not cases:
        if not shortfalls:
            shortfalls.append(StateMachineASTShortfall(
                definition.name, "transition.match", "state machine has no cases",
            ))
        return None, tuple(shortfalls)

    case_methods = tuple(cases)
    control = ControlProgram(
        root=StateMachineTick(
            field,
            tuple(
                (
                    str(case_value),
                    StatementBlock((f"__scheduled_region_{region_index}__",)),
                )
                for region_index, (case_value, _method) in enumerate(case_methods)
            ),
        ),
        region_indices=tuple(range(len(case_methods))),
        uniforms=(ControlUniform(field, int(state_value_id), "int"),),
    )
    return StateMachineASTPlan(
        definition.name, field, case_methods, control
    ), ()


def dispatch_arms_are_effects(statement: ast.Match) -> bool:
    """Whether every case arm is one bare method call (an expression
    statement): the shape whose arms rebind nothing and return nothing, so
    the tick's arms are exactly their callsites."""

    return all(
        len(case.body) == 1
        and isinstance(case.body[0], ast.Expr)
        and isinstance(case.body[0].value, ast.Call)
        for case in statement.cases
    )


def planned_dispatch(plans, scope, function_definition, statement):
    """The plan whose dispatch ``statement`` is, else None.

    ``plans`` is ``G.graph["state_machine_controls"]``; ``scope`` the
    class-qualified source function the reducer is reducing
    (``"Class.transition"``); the statement must be the one outer ``match``
    of that class's ``transition``.  Anything else keeps the reducer's
    ordinary handling.
    """

    transition = f"{{}}.transition"
    for plan in plans or ():
        qualified = transition.format(plan.class_name)
        if scope != qualified and not str(scope).endswith("." + qualified):
            continue
        if not isinstance(
            function_definition, (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        if _outer_match(function_definition) is statement:
            return plan
    return None


def install_state_machine_control(graph, control, hierarchy_plan=None):
    """Install the tick of every planned dispatch in ``graph`` into
    ``control``, the function's ControlProgram.

    The reducer (``reduce_state_dispatch``) left one dispatch node per
    planned ``match``: its ``state`` operand is the selector's value, its
    ``case:i`` operands the case literals' constants.  The tick is built
    from that node -- ``source_node_id`` its identity, ``state_value_id`` and
    ``case_value_ids`` its operands -- and each arm holds the planned
    callsites (``__plan_callsite_N__``) whose call lies in that case's body,
    so the callsite scheduler finds the markers already placed.  ``control``
    is returned unchanged for a function with no dispatch node.
    """

    from dataclasses import replace

    from .control_source import (
        ControlProgram, SequenceBlock, StateMachineTick, StatementBlock,
        _flatten_control_sequence,
    )
    from .hierarchical_plan import PlanCall

    G = getattr(graph, "G", graph)
    dispatches = tuple(
        (node_id, data) for node_id, data in G.nodes(data=True)
        if (data.get("attributes") or {}).get("state_machine_dispatch")
        and isinstance(data.get("expr_obj"), ast.Match)
    )
    if not dispatches:
        return control
    planned_calls = frozenset(
        int(item.callsite_id)
        for item in getattr(hierarchy_plan, "items", ())
        if isinstance(item, PlanCall)
    )
    ticks = []
    for node_id, data in sorted(dispatches, key=lambda item: int(item[0])):
        match = data["expr_obj"]
        roles = {role: parent for parent, role in data.get("parents") or ()}
        state_value_id = roles.get("state")
        field = _state_field(match.subject)
        if state_value_id is None or field is None:
            continue
        arms = []
        for index, case in enumerate(match.cases):
            value = _literal_case(case.pattern)
            if value is None:
                break
            inside = {
                id(member) for statement in case.body
                for member in ast.walk(statement)
            }
            callsites = sorted(
                int(candidate) for candidate, candidate_data in G.nodes(data=True)
                if int(candidate) in planned_calls
                and id(candidate_data.get("expr_obj")) in inside
            )
            arms.append((
                str(value),
                SequenceBlock(tuple(
                    StatementBlock((f"__plan_callsite_{callsite}__",))
                    for callsite in callsites
                )),
                roles.get(f"case:{index}"),
            ))
        else:
            ticks.append(StateMachineTick(
                field,
                tuple((label, body) for label, body, _ in arms),
                state_value_id=int(state_value_id),
                source_node_id=int(node_id),
                case_value_ids=tuple(
                    None if literal is None else int(literal)
                    for _, _, literal in arms
                ),
            ))
    if not ticks:
        return control
    if control is None:
        control = ControlProgram(SequenceBlock(()))
    root = SequenceBlock((*_flatten_control_sequence(control.root), *ticks))
    return replace(control, root=root)


def plan_marked_state_machines(tree: ast.AST):
    """Plan every marked class without importing or executing its module."""

    plans: list[StateMachineASTPlan] = []
    shortfalls: list[StateMachineASTShortfall] = []
    for definition in ast.walk(tree):
        if not isinstance(definition, ast.ClassDef) or not _is_marked(definition):
            continue
        plan, failures = lower_marked_state_machine_class(definition)
        if plan is not None:
            plans.append(plan)
        shortfalls.extend(failures)
    return tuple(plans), tuple(shortfalls)


__all__ = [
    "StateMachineASTPlan",
    "dispatch_arms_are_effects",
    "StateMachineASTShortfall",
    "lower_marked_state_machine_class",
    "install_state_machine_control",
    "plan_marked_state_machines",
    "planned_dispatch",
]

