"""Reduce marked Python class dispatch into the existing control vocabulary.

An ``AbstractTensorStateMachine`` subclass supplies ``transition`` whose
outer statement is a ``match`` over one class-owned scalar state field. Each
literal case dispatches to one method on ``self``. Those methods remain normal
numeric regions; this module creates only the existing ``StateMachineTick``
control shell and never introduces an SSA operator.
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
    "StateMachineASTShortfall",
    "lower_marked_state_machine_class",
    "plan_marked_state_machines",
]

