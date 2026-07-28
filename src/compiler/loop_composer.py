"""Planner-owned loop classification and backend-source composition.

The ProcessGraph remains the semantic authority.  This module decides how a
backend should realize each retained loop; it does not reinterpret tensor
operators and it does not execute a Python loop as a substitute for compiled
control flow.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class LoopStrategy(str, Enum):
    CONSTANT = "constant"
    UNROLL = "unroll"
    NATIVE_SOURCE = "native_source"
    DISPATCH = "dispatch"
    KPN = "kpn"


@dataclass(frozen=True)
class LoopDescriptor:
    node_id: int
    source_type: str
    target: str
    iterator_kind: str
    body_nodes: tuple[int, ...]
    condition_nodes: tuple[int, ...]
    target_bindings: tuple[tuple[str, int], ...] = ()
    carried_bindings: tuple[tuple[str, int, int], ...] = ()
    start: Any = None
    stop: Any = None
    step: Any = None
    iterable_node: int | None = None
    trip_count: int | None = None


@dataclass(frozen=True)
class LoopPlan:
    loop: LoopDescriptor
    strategy: LoopStrategy
    reason: str


@dataclass(frozen=True)
class LoopBackendCapabilities:
    backend: str
    native_for: bool = False
    native_while: bool = False
    dynamic_bounds: bool = False
    kpn: bool = False
    unroll_limit: int = 8


def _constant(graph: Any, node_id: int | None) -> tuple[bool, Any]:
    if node_id is None or node_id not in graph.G:
        return False, None
    data = graph.G.nodes[node_id]
    expression = data.get("expr_obj")
    if isinstance(expression, ast.Constant):
        return True, expression.value
    if "constant" in data:
        return True, data["constant"]
    attributes = data.get("attributes") or {}
    if "value" in attributes:
        return True, attributes["value"]
    return False, None


def _trip_count(start: Any, stop: Any, step: Any) -> int | None:
    if not all(isinstance(value, int) for value in (start, stop, step)):
        return None
    if step == 0:
        return None
    return len(range(start, stop, step))


class LoopComposer:
    """Classify ProcessGraph loops for one backend deployment planner."""

    def __init__(self, capabilities: LoopBackendCapabilities):
        self.capabilities = capabilities

    def describe(self, graph: Any, node_id: int) -> LoopDescriptor:
        data = graph.G.nodes[node_id]
        expression = data.get("expr_obj")
        if not isinstance(
            expression,
            (ast.For, ast.While, ast.comprehension),
        ):
            raise TypeError(f"node {node_id} is not a ProcessGraph loop")
        attributes = data.get("attributes") or {}
        parents = tuple(data.get("parents") or ())
        by_role: dict[str, list[int]] = {}
        for parent, role in parents:
            by_role.setdefault(str(role), []).append(parent)

        source_type = type(expression).__name__
        target = str(attributes.get("target", ""))
        if isinstance(expression, (ast.For, ast.comprehension)):
            target = target or ast.unparse(expression.target)
        iterator_kind = str(
            attributes.get(
                "iterator_kind",
                "while" if isinstance(expression, ast.While) else "iterable",
            )
        )
        body_nodes = tuple(
            by_role.get("body", ())
            or by_role.get("generators", ())
        )
        expression_nodes = {
            id(node_data.get("expr_obj")): candidate
            for candidate, node_data in graph.G.nodes(data=True)
            if node_data.get("expr_obj") is not None
        }
        if not body_nodes and isinstance(expression, (ast.For, ast.While)):
            body_nodes = tuple(
                expression_nodes[id(member)]
                for statement in expression.body
                for member in ast.walk(statement)
                if id(member) in expression_nodes
            )
        elif not body_nodes and isinstance(expression, ast.comprehension):
            body_nodes = tuple(
                parent
                for successor in graph.G.successors(node_id)
                if isinstance(
                    graph.G.nodes[successor].get("expr_obj"),
                    (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
                )
                for parent, role in (
                    graph.G.nodes[successor].get("parents") or ()
                )
                if str(role) in {"elt", "key", "value"}
            )
        condition_nodes = tuple(
            by_role.get("ifs", ())
            or by_role.get("test", ())
        )
        iterable_node = next(
            iter(by_role.get("iterable", ()) or by_role.get("iter", ())),
            None,
        )

        start = attributes.get("start")
        stop = attributes.get("stop")
        step = attributes.get("step")
        for role, default in (("start", 0), ("stop", None), ("step", 1)):
            nodes = by_role.get(role, ())
            if nodes:
                known, value = _constant(graph, nodes[0])
                if known:
                    if role == "start":
                        start = value
                    elif role == "stop":
                        stop = value
                    else:
                        step = value
            elif role == "start" and start is None:
                start = default
            elif role == "step" and step is None:
                step = default

        count = _trip_count(start, stop, step)
        if count is None and iterable_node is not None:
            known, iterable = _constant(graph, iterable_node)
            if known and isinstance(iterable, (tuple, list, range)):
                count = len(iterable)

        target_bindings = dict(
            attributes.get("loop_target_bindings") or {}
        )
        for name, target_id in tuple(target_bindings.items()):
            if target_id in graph.G and graph.G.out_degree(target_id):
                continue
            candidates = [
                candidate
                for candidate, candidate_data in graph.G.nodes(data=True)
                if candidate_data.get("type") == "Input"
                and (
                    candidate_data.get("attributes") or {}
                ).get("binding_kind") == "loop"
                and (
                    candidate_data.get("attributes") or {}
                ).get("binding_name") == name
                and graph.G.out_degree(candidate)
            ]
            if candidates:
                target_bindings[name] = min(candidates)

        return LoopDescriptor(
            node_id=int(node_id),
            source_type=source_type,
            target=target,
            iterator_kind=iterator_kind,
            body_nodes=body_nodes,
            condition_nodes=condition_nodes,
            target_bindings=tuple(
                sorted(
                    (
                        str(name),
                        int(value_id),
                    )
                    for name, value_id in target_bindings.items()
                )
            ),
            carried_bindings=tuple(
                sorted(
                    (
                        str(name),
                        int(binding[0]),
                        int(binding[1]),
                    )
                    for name, binding in (
                        attributes.get("loop_carried_bindings") or {}
                    ).items()
                )
            ),
            start=start,
            stop=stop,
            step=step,
            iterable_node=iterable_node,
            trip_count=count,
        )

    def plan(self, loop: LoopDescriptor) -> LoopPlan:
        if loop.trip_count == 0:
            return LoopPlan(
                loop,
                LoopStrategy.CONSTANT,
                "the loop has no iterations",
            )
        if (
            loop.trip_count is not None
            and loop.trip_count <= self.capabilities.unroll_limit
        ):
            return LoopPlan(
                loop,
                LoopStrategy.UNROLL,
                "the static trip count fits the backend unroll limit",
            )
        if (
            loop.iterator_kind == "while"
            and self.capabilities.native_while
        ):
            return LoopPlan(
                loop,
                LoopStrategy.NATIVE_SOURCE,
                "the backend can retain this while loop in compiled source",
            )
        if self.capabilities.native_for and (
            loop.trip_count is not None
            or self.capabilities.dynamic_bounds
        ):
            return LoopPlan(
                loop,
                LoopStrategy.NATIVE_SOURCE,
                "the backend can retain this iteration in compiled source",
            )
        if self.capabilities.kpn:
            return LoopPlan(
                loop,
                LoopStrategy.KPN,
                "the backend delegates unresolved loop control to a KPN",
            )
        return LoopPlan(
            loop,
            LoopStrategy.DISPATCH,
            "the loop requires planner-coordinated dispatches",
        )

    def compose(self, graph: Any) -> tuple[LoopPlan, ...]:
        plans = []
        for node_id, data in graph.G.nodes(data=True):
            if isinstance(
                data.get("expr_obj"),
                (ast.For, ast.While, ast.comprehension),
            ):
                plans.append(self.plan(self.describe(graph, node_id)))
        return tuple(plans)


def indent_source(lines: Iterable[str], spaces: int = 4) -> tuple[str, ...]:
    prefix = " " * int(spaces)
    return tuple(prefix + line if line else line for line in lines)


__all__ = [
    "LoopBackendCapabilities",
    "LoopComposer",
    "LoopDescriptor",
    "LoopPlan",
    "LoopStrategy",
    "indent_source",
]
