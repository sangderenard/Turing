"""Planner-owned loop classification and backend-source composition.

The ProcessGraph remains the semantic authority.  This module decides how a
backend should realize each retained loop; it does not reinterpret tensor
operators and it does not execute a Python loop as a substitute for compiled
control flow.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping

import networkx as nx

from .control_source import (
    ControlDeploymentLane,
    ControlDeploymentRegion,
    ControlProgram,
    ControlExpression,
    ControlUniform,
    LoopControlBlock,
    LoopBlock,
    RecursionRegion,
    SequenceBlock,
    StatementBlock,
    StreamPublishBlock,
    WhileBlock,
)
from .loop_ir import (
    ConditionDomain,
    IterableDomain,
    IterableAccess,
    LoopCarriedState,
    LoopDomainKind,
    LoopEffects,
    LoopPolicy,
    LoopRealization,
    LoopIterationOutput,
    LoopStateEffect,
    LoopStateEffectMode,
    LoopValue,
    RangeDomain,
    SemanticLoop,
)
from .hierarchical_plan import PlanClosure, PlanLine


class LoopStrategy(str, Enum):
    CONSTANT = "constant"
    UNROLL = "unroll"
    NATIVE_SOURCE = "native_source"
    DISPATCH = "dispatch"
    KPN = "kpn"


_CONTROL_IR_NODE_TYPES = frozenset({
    "LoopExit",
    "LoopStateTransition",
    "LoopResult",
    "LoopStatePort",
    "LoopAggregateResult",
})


def _is_control_ir_node(data: Mapping[str, Any] | dict[str, Any]) -> bool:
    return (
        str(data.get("type")) in _CONTROL_IR_NODE_TYPES
        or isinstance(
            data.get("expr_obj"),
            (ast.For, ast.While, ast.comprehension),
        )
    )


def _destructure_loop_target(
    target: ast.AST,
    value: object,
) -> tuple[tuple[str, object], ...]:
    """Apply Python loop-target destructuring without executing the loop."""

    if isinstance(target, ast.Name):
        return ((target.id, value),)
    if isinstance(target, ast.Starred):
        return _destructure_loop_target(target.value, value)
    if not isinstance(target, (ast.Tuple, ast.List)):
        raise ValueError(
            "unsupported loop target for static destructuring: "
            f"{ast.dump(target, include_attributes=False)}"
        )
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(
            f"cannot destructure non-iterable loop item {value!r}"
        ) from error
    starred = tuple(
        index
        for index, element in enumerate(target.elts)
        if isinstance(element, ast.Starred)
    )
    if len(starred) > 1:
        raise ValueError("loop target contains multiple starred bindings")
    if not starred:
        if len(items) != len(target.elts):
            raise ValueError(
                "loop item does not match target destructuring: "
                f"target={ast.unparse(target)!r}, item={value!r}"
            )
        assigned = tuple(zip(target.elts, items))
    else:
        star = starred[0]
        trailing = len(target.elts) - star - 1
        if len(items) < len(target.elts) - 1:
            raise ValueError(
                "loop item does not match starred target destructuring: "
                f"target={ast.unparse(target)!r}, item={value!r}"
            )
        assigned = (
            *tuple(zip(target.elts[:star], items[:star])),
            (target.elts[star], list(items[star:len(items) - trailing])),
            *tuple(zip(
                target.elts[star + 1:],
                items[len(items) - trailing:] if trailing else (),
            )),
        )
    return tuple(
        binding
        for element, item in assigned
        for binding in _destructure_loop_target(element, item)
    )


@dataclass(frozen=True)
class LoopDescriptor:
    node_id: int
    source_type: str
    target: str
    iterator_kind: str
    body_nodes: tuple[int, ...]
    condition_nodes: tuple[int, ...]
    break_nodes: tuple[tuple[int, int | None, bool], ...] = ()
    continue_nodes: tuple[tuple[int, int | None, bool], ...] = ()
    target_bindings: tuple[tuple[str, int], ...] = ()
    carried_bindings: tuple[tuple[str, int, int], ...] = ()
    start: Any = None
    stop: Any = None
    step: Any = None
    start_node: int | None = None
    stop_node: int | None = None
    step_node: int | None = None
    iterable_node: int | None = None
    iterable_constant: tuple[object, ...] | None = None
    trip_count: int | None = None
    yield_nodes: tuple[int, ...] = ()
    # (source statement node, published value node, optional count node).
    # These are stream effects such as yield, not loop reductions.
    publication_nodes: tuple[tuple[int, int, int | None], ...] = ()
    state_effects: tuple[LoopStateEffect, ...] = ()
    iteration_outputs: tuple[LoopIterationOutput, ...] = ()
    backpressured_output: bool = False


@dataclass(frozen=True)
class LoopPlan:
    loop: LoopDescriptor
    strategy: LoopStrategy
    reason: str
    semantic: SemanticLoop | None = None


@dataclass(frozen=True)
class LoopShaderReduction:
    """Planner verdict for replacing coordinator iterations with one shader."""

    loop_node_id: int
    region_indices: tuple[int, ...]
    carried_bindings: tuple[tuple[str, int, int], ...]
    collapsible: bool
    blockers: tuple[str, ...]
    estimated_dispatches_removed: int | None
    control_program: ControlProgram | None = None
    preferred_shell: str = "glsl"
    dispatch_closure_count: int = 0


def planned_collection_bindings(
    graph: Any,
    loop: LoopDescriptor,
    resident_value_ids: frozenset[int] | None = None,
) -> tuple[tuple[int, int, str, int], ...]:
    """Lower supported loop-state effects to resident indexed writes."""

    induction_name = f"iteration_{int(loop.node_id)}"
    source_start = int(0 if loop.start is None else loop.start)
    bindings = []
    for effect in loop.state_effects:
        if (
            effect.mode is not LoopStateEffectMode.INDEXED_PUBLICATION
            or not effect.argument_value_ids
            or effect.loop_result_id is None
        ):
            continue
        value_id = int(effect.argument_value_ids[0])
        value_attributes = graph.G.nodes[value_id].get("attributes") or {}
        if value_attributes.get("producer_kind") == "aggregate":
            continue
        # The latch commits a carried result into its state slot before loop
        # effects are published.  A body effect that consumes that result must
        # therefore read the post-commit state slot, not the transient Phi
        # endpoint that selected the result inside the body.
        source_id = next(
            (
                int(initial)
                for _name, initial, updated in loop.carried_bindings
                if int(updated) == value_id
            ),
            value_id,
        )
        bindings.append((
            source_id,
            int(effect.loop_result_id),
            induction_name,
            source_start,
        ))
    for output in loop.iteration_outputs:
        value_id = int(output.value_id)
        if (
            resident_value_ids is not None
            and value_id not in resident_value_ids
        ):
            continue
        value_attributes = graph.G.nodes[value_id].get("attributes") or {}
        if value_attributes.get("producer_kind") == "aggregate":
            continue
        bindings.append((
            value_id,
            int(output.result_value_id),
            induction_name,
            source_start,
        ))
    return tuple(dict.fromkeys(bindings))


@dataclass(frozen=True)
class LoopBackendCapabilities:
    backend: str
    native_for: bool = False
    native_while: bool = False
    dynamic_bounds: bool = False
    kpn: bool = False
    unroll_limit: int = 8


def _rebuild_graph_edges(graph: Any) -> None:
    """Make NetworkX edges and cached parent/child tables agree."""

    graph.G.remove_edges_from(tuple(graph.G.edges))
    for node_id in graph.G:
        graph.G.nodes[node_id]["children"] = []
    for node_id, data in graph.G.nodes(data=True):
        (data.get("attributes") or {}).pop("recursion_region_id", None)
        (data.get("attributes") or {}).pop("control_ir_owned", None)
        normalized = []
        for parent, role in data.get("parents") or ():
            parent = int(parent)
            if parent not in graph.G:
                continue
            normalized.append((parent, role))
            graph.G.add_edge(parent, node_id, role=role)
            graph.G.nodes[parent]["children"].append((node_id, role))
        data["parents"] = normalized
    try:
        generations = tuple(nx.topological_generations(graph.G))
    except nx.NetworkXUnfeasible:
        # A retained loop deliberately has feedback: its next-iteration value
        # depends on the body, while the body depends on the loop-carried
        # input port.  That is irreducible recursion, not an invalid attempt
        # to schedule the body as straight-line SSA.  Preserve the feedback
        # edges for SemanticLoop/LoopBlock lowering and topologically order
        # only the condensation graph, whose nodes are the strongly connected
        # components of this graph.
        components = tuple(nx.strongly_connected_components(graph.G))
        condensed = nx.condensation(graph.G, scc=components)
        recursive_components = []
        generations = []
        for generation in nx.topological_generations(condensed):
            members = set()
            for component_id in generation:
                component_members = frozenset(
                    int(node_id)
                    for node_id in condensed.nodes[component_id]["members"]
                )
                members.update(component_members)
                if (
                    len(component_members) > 1
                    or any(
                        graph.G.has_edge(node_id, node_id)
                        for node_id in component_members
                    )
                ):
                    recursive_components.append(component_members)
            generations.append(tuple(sorted(members)))
        recursion_table = {}
        for component_index, component in enumerate(recursive_components):
            incoming = []
            outgoing = []
            feedback = []
            for source, target, edge_data in graph.G.edges(data=True):
                edge = (
                    int(source),
                    int(target),
                    str(edge_data.get("role", "")),
                )
                if source in component and target in component:
                    feedback.append(edge)
                elif source not in component and target in component:
                    incoming.append(edge)
                elif source in component and target not in component:
                    outgoing.append(edge)
            recursion_table[component_index] = {
                "kind": "irreducible_recursion",
                "lower_as": "while",
                "control_ir": True,
                "members": tuple(sorted(component)),
                "control_members": tuple(sorted(
                    node_id
                    for node_id in component
                    if _is_control_ir_node(graph.G.nodes[node_id])
                )),
                "incoming": tuple(sorted(incoming)),
                "outgoing": tuple(sorted(outgoing)),
                "feedback": tuple(sorted(feedback)),
            }
            for node_id in component:
                attributes = graph.G.nodes[node_id].setdefault(
                    "attributes", {}
                )
                attributes["recursion_region_id"] = component_index
                if _is_control_ir_node(graph.G.nodes[node_id]):
                    attributes["control_ir_owned"] = True
        graph.G.graph["recursion_table"] = recursion_table
    else:
        graph.G.graph.pop("recursion_table", None)
    graph.levels = {
        int(node_id): int(level)
        for level, generation in enumerate(generations)
        for node_id in generation
    }


def _replace_parent_value(
    graph: Any,
    old_value_id: int,
    replacements: tuple[int, ...],
) -> None:
    """Replace one value use without identifying its distinct consumers."""

    old_value_id = int(old_value_id)
    for _node_id, data in graph.G.nodes(data=True):
        parents = tuple(data.get("parents") or ())
        if not any(int(parent) == old_value_id for parent, _role in parents):
            continue
        rewritten = []
        for parent, role in parents:
            if int(parent) != old_value_id:
                rewritten.append((int(parent), role))
                continue
            if len(replacements) == 1:
                rewritten.append((int(replacements[0]), role))
                continue
            role_text = str(role)
            prefix = (
                "arg"
                if role_text.startswith("arg")
                else "elts"
                if role_text in {"elt", "elts"}
                else role_text
            )
            rewritten.extend(
                (
                    int(replacement),
                    f"{prefix}{index}" if prefix == "arg" else prefix,
                )
                for index, replacement in enumerate(replacements)
            )
        data["parents"] = rewritten
    graph.roots = [
        (
            int(replacements[-1])
            if int(root) == old_value_id and replacements
            else int(root)
        )
        for root in graph.roots
    ]


def evaporate_unrolled_loops(
    graph: Any,
    plans: Iterable[LoopPlan],
) -> tuple[LoopPlan, ...]:
    """Rewrite planner-selected finite loops into straight-line value SSA.

    This runs only after graph value IDs are canonical.  It does not emit a
    control block and it does not preserve source container syntax.  Each
    iteration receives its own induction constants and numerical body values;
    carried values are threaded directly between those copies.  Collection
    effects and comprehension outputs become inputs of their existing,
    distinct materialization producer.
    """

    if not graph.G.graph.get("canonical_value_ids"):
        raise ValueError("loop evaporation requires canonical value IDs")

    plans = tuple(plans)
    plans_by_expression = {
        id(graph.G.nodes[int(plan.loop.node_id)].get("expr_obj")): plan
        for plan in plans
        if int(plan.loop.node_id) in graph.G
    }
    nested_groups: dict[int, tuple[LoopPlan, ...]] = {}
    for _node_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.GeneratorExp):
            continue
        group = tuple(
            plans_by_expression[id(generator)]
            for generator in expression.generators
            if id(generator) in plans_by_expression
        )
        if len(group) < 2:
            continue
        for member in group:
            nested_groups[int(member.loop.node_id)] = group
    evaporated: list[LoopPlan] = []
    deployment_regions = list(
        graph.G.graph.get("control_deployment_regions", ())
    )
    next_deployment_region_id = max(
        (
            int(region.region_id)
            for region in deployment_regions
        ),
        default=-1,
    ) + 1
    handled_loop_ids: set[int] = set()
    next_value_id = max(graph.G.nodes, default=-1) + 1

    def add_clone(source_id: int, parents: tuple[tuple[int, str], ...]) -> int:
        nonlocal next_value_id
        clone_id = next_value_id
        next_value_id += 1
        cloned = copy.deepcopy(dict(graph.G.nodes[int(source_id)]))
        cloned["parents"] = list(parents)
        cloned["children"] = []
        cloned["value_id"] = clone_id
        attributes = dict(cloned.get("attributes") or {})
        attributes["unrolled_from"] = int(source_id)
        cloned["attributes"] = attributes
        graph.G.add_node(clone_id, **cloned)
        return clone_id

    def add_constant(value: object, loop_id: int) -> int:
        nonlocal next_value_id
        constant_id = next_value_id
        next_value_id += 1
        expression = ast.Constant(value=value)
        graph.G.add_node(
            constant_id,
            type="Constant",
            label=repr(value),
            op="const",
            expr_obj=expression,
            constant=value,
            value_id=constant_id,
            parents=[],
            children=[],
            attributes={
                "value": value,
                "unrolled_induction_of": int(loop_id),
            },
        )
        return constant_id

    for plan in plans:
        loop_id = int(plan.loop.node_id)
        if loop_id in handled_loop_ids:
            continue
        group = nested_groups.get(loop_id)
        if group is not None and plan is not group[0]:
            continue
        selected_plans = group or (plan,)
        loop = selected_plans[0].loop
        iteration_assignments: tuple[dict[int, object], ...] | None = None
        if group is not None:
            generator_expression = next(
                data.get("expr_obj")
                for _node_id, data in graph.G.nodes(data=True)
                if isinstance(data.get("expr_obj"), ast.GeneratorExp)
                and tuple(
                    id(generator)
                    for generator in data["expr_obj"].generators
                )
                == tuple(
                    id(graph.G.nodes[int(member.loop.node_id)]["expr_obj"])
                    for member in group
                )
            )
            specializations = {
                **dict(graph.G.graph.get("parameter_defaults") or {}),
                **dict(graph.G.graph.get("planner_specializations") or {}),
            }
            assignments: list[dict[int, object]] = []

            def expand_generator(
                index: int,
                environment: dict[str, object],
                bound: dict[int, object],
            ) -> bool:
                if index == len(group):
                    assignments.append(dict(bound))
                    return True
                member = group[index]
                generator = generator_expression.generators[index]
                items = _static_iterable_expression(
                    generator.iter,
                    {**specializations, **environment},
                )
                if items is None:
                    return False
                target_bindings = member.loop.target_bindings
                for item in items:
                    destructured = dict(
                        _destructure_loop_target(generator.target, item)
                    )
                    if any(
                        str(name) not in destructured
                        for name, _target_id in target_bindings
                    ):
                        return False
                    nested_environment = dict(environment)
                    nested_bound = dict(bound)
                    for name, target_id in target_bindings:
                        value = destructured[str(name)]
                        nested_environment[str(name)] = value
                        nested_bound[int(target_id)] = value
                    try:
                        accepted = all(
                            _static_predicate_expression(
                                condition,
                                {**specializations, **nested_environment},
                            )
                            for condition in generator.ifs
                        )
                    except ValueError:
                        return False
                    if accepted and not expand_generator(
                        index + 1,
                        nested_environment,
                        nested_bound,
                    ):
                        return False
                return True

            if not expand_generator(0, {}, {}):
                continue
            iteration_assignments = tuple(assignments)
            iteration_values = tuple(range(len(iteration_assignments)))
        elif plan.strategy not in {
            LoopStrategy.UNROLL,
            LoopStrategy.CONSTANT,
        }:
            continue
        elif loop.iterator_kind == "arithmetic_sequence":
            if None in (loop.start, loop.stop, loop.step):
                continue
            iteration_values = tuple(
                range(int(loop.start), int(loop.stop), int(loop.step))
            )
        elif loop.iterable_constant is not None:
            iteration_values = tuple(loop.iterable_constant)
        else:
            continue

        empty_generator_materializers = set()
        if not iteration_values:
            for member in selected_plans:
                member_id = int(member.loop.node_id)
                if member_id not in graph.G or not isinstance(
                    graph.G.nodes[member_id].get("expr_obj"),
                    ast.comprehension,
                ):
                    continue
                for successor in tuple(graph.G.successors(member_id)):
                    if not isinstance(
                        graph.G.nodes[successor].get("expr_obj"),
                        ast.GeneratorExp,
                    ):
                        continue
                    empty_value = add_constant((), member_id)
                    _replace_parent_value(
                        graph,
                        int(successor),
                        (empty_value,),
                    )
                    empty_generator_materializers.add(int(successor))

        target_ids = tuple(dict.fromkeys(
            int(target_id)
            for member in selected_plans
            for _name, target_id in member.loop.target_bindings
        ))
        body_ids = frozenset(
            int(value_id)
            for member in selected_plans
            for value_id in member.loop.body_nodes
        )
        required_outputs = {
            int(updated)
            for member in selected_plans
            for _name, _initial, updated in member.loop.carried_bindings
        }
        required_outputs.update(
            int(value_id)
            for member in selected_plans
            for effect in member.loop.state_effects
            for value_id in effect.argument_value_ids
        )
        required_outputs.update(
            int(output.value_id)
            for member in selected_plans
            for output in member.loop.iteration_outputs
        )
        carried_bindings = tuple(dict.fromkeys(
            binding
            for member in selected_plans
            for binding in member.loop.carried_bindings
        ))
        state_effects = tuple(dict.fromkeys(
            effect
            for member in selected_plans
            for effect in member.loop.state_effects
        ))
        iteration_outputs = tuple(dict.fromkeys(
            output
            for member in selected_plans
            for output in member.loop.iteration_outputs
        ))
        iteration_dependent = {
            int(node_id)
            for target_id in target_ids
            for node_id in nx.descendants(graph.G, target_id)
        }
        iteration_dependent.discard(int(loop.node_id))
        needed = set(required_outputs)
        pending = list(required_outputs)
        while pending:
            value_id = pending.pop()
            if value_id not in graph.G:
                continue
            for parent, _role in graph.G.nodes[value_id].get("parents") or ():
                parent = int(parent)
                if (
                    parent in body_ids or parent in iteration_dependent
                ) and parent not in needed:
                    needed.add(parent)
                    pending.append(parent)
        body_order = tuple(
            value_id
            for value_id in nx.topological_sort(graph.G)
            if (
                int(value_id) in needed
                and int(value_id) not in target_ids
            )
        )

        carried_values = {
            int(initial): int(initial)
            for _name, initial, _updated in carried_bindings
        }
        publications: dict[int, list[int]] = {
            int(
                effect.state_input_id
                if effect.loop_result_id is None
                else effect.loop_result_id
            ): []
            for effect in state_effects
            if effect.mode is LoopStateEffectMode.INDEXED_PUBLICATION
        }
        materializations: dict[int, list[int]] = {
            int(output.materializer_node_id): []
            for output in iteration_outputs
        }
        last_mapping: dict[int, int] = {}
        iteration_lane_nodes: list[tuple[int, ...]] = []

        for iteration_index, iteration in enumerate(iteration_values):
            mapping = dict(carried_values)
            if iteration_assignments is not None:
                mapping.update(
                    (
                        target_id,
                        add_constant(value, loop.node_id),
                    )
                    for target_id, value in (
                        iteration_assignments[iteration_index].items()
                    )
                )
            else:
                source_expression = graph.G.nodes[
                    int(loop.node_id)
                ].get("expr_obj")
                if not isinstance(
                    source_expression, (ast.For, ast.comprehension)
                ):
                    raise ValueError(
                        "iterable loop has no destructurable source target: "
                        f"loop={loop.node_id}"
                    )
                values_by_name = dict(_destructure_loop_target(
                    source_expression.target, iteration
                ))
                mapping.update(
                    (
                        int(target_id),
                        add_constant(values_by_name[str(name)], loop.node_id),
                    )
                    for name, target_id in loop.target_bindings
                )
            for source_id in body_order:
                parents = tuple(
                    (int(mapping.get(int(parent), int(parent))), str(role))
                    for parent, role in (
                        graph.G.nodes[source_id].get("parents") or ()
                    )
                )
                mapping[source_id] = add_clone(source_id, parents)
            for _name, initial, updated in carried_bindings:
                carried_values[int(initial)] = int(mapping[int(updated)])
            for effect in state_effects:
                if (
                    effect.mode
                    is not LoopStateEffectMode.INDEXED_PUBLICATION
                    or not effect.argument_value_ids
                ):
                    continue
                collection_id = int(
                    effect.state_input_id
                    if effect.loop_result_id is None
                    else effect.loop_result_id
                )
                publications[collection_id].append(
                    int(mapping.get(
                        int(effect.argument_value_ids[0]),
                        int(effect.argument_value_ids[0]),
                    ))
                )
            for output in iteration_outputs:
                materializations[
                    int(output.materializer_node_id)
                ].append(int(mapping[int(output.value_id)]))
            last_mapping = mapping
            iteration_lane_nodes.append(tuple(dict.fromkeys(
                int(value_id)
                for value_id in mapping.values()
                if int(value_id) in graph.G
            )))

        for _name, _initial, updated in carried_bindings:
            # ``last_mapping`` only gets assigned inside the per-iteration
            # loop above (line ~765). A loop whose traced iterable is empty
            # never enters that loop, so ``last_mapping`` stays the {}
            # this block was seeded with and every carried binding's
            # "updated" value looks unavailable. That is not a genuine
            # unavailability: zero iterations means the carried value never
            # changed, so its correct final value is simply its pre-loop
            # ``_initial`` binding. Falling through to "skip the redirect"
            # instead left every downstream consumer (for example a value
            # only used inside a sibling ``if`` branch's separately
            # extracted region) pointed at a node this same function then
            # deletes as part of the evaporated loop body, surfacing much
            # later as "missing ProcessGraph input" with no loop plan left
            # to explain it.
            final_value = last_mapping.get(int(updated), int(_initial))
            _replace_parent_value(
                graph,
                next(
                    (
                        int(node_id)
                        for node_id, data in graph.G.nodes(data=True)
                        if data.get("type") == "LoopExit"
                        and (
                            data.get("attributes") or {}
                        ).get("binding_name") == _name
                        and any(
                            int(parent) == int(loop.node_id)
                            and str(role) == "control"
                            for parent, role in (
                                data.get("parents") or ()
                            )
                        )
                    ),
                    int(updated),
                ),
                (int(final_value),),
            )
        for collection_id, values in publications.items():
            if values:
                _replace_parent_value(
                    graph, int(collection_id), tuple(values)
                )
        for materializer_id, values in materializations.items():
            if materializer_id not in graph.G:
                continue
            materializer = graph.G.nodes[materializer_id]
            parents = tuple(materializer.get("parents") or ())
            structural = {
                int(parent)
                for parent, _role in parents
                if parent in graph.G
                and isinstance(
                    graph.G.nodes[int(parent)].get("expr_obj"),
                    (ast.GeneratorExp, ast.comprehension),
                )
            }
            materializer["parents"] = [
                (int(parent), role)
                for parent, role in parents
                if int(parent) not in structural
                and int(parent) != int(loop.node_id)
                and str(role) not in {"elt", "generators"}
            ]
            materializer["parents"].extend(
                (int(value_id), f"arg{index}")
                for index, value_id in enumerate(values)
            )
            attributes = dict(materializer.get("attributes") or {})
            attributes["materialization_kind"] = "unrolled_loop"
            attributes["materialized_value_ids"] = tuple(values)
            materializer["attributes"] = attributes

        removable = {
            *(
                int(member.loop.node_id)
                for member in selected_plans
            ),
            *target_ids,
            *body_ids,
            *empty_generator_materializers,
            *(
                int(effect.state_output_id)
                for effect in state_effects
                if effect.state_output_id is not None
            ),
            *(
                int(effect.loop_result_id)
                for effect in state_effects
                if effect.loop_result_id is not None
            ),
        }
        removable.update(
            int(node_id)
            for node_id, data in graph.G.nodes(data=True)
            if data.get("type") == "LoopExit"
            and any(
                int(parent) in {
                    int(member.loop.node_id)
                    for member in selected_plans
                }
                and str(role) == "control"
                for parent, role in data.get("parents") or ()
            )
        )
        _rebuild_graph_edges(graph)
        removable = {
            node_id
            for node_id in removable
            if node_id in graph.G and node_id not in graph.roots
        }
        externally_used = {
            node_id
            for node_id in removable
            if any(
                int(successor) not in removable
                for successor in graph.G.successors(node_id)
            )
        }
        graph.G.remove_nodes_from(removable - externally_used)
        _rebuild_graph_edges(graph)
        parallel_candidate = bool(
            not carried_bindings
            and not any(
                member.loop.backpressured_output
                for member in selected_plans
            )
            and (state_effects or iteration_outputs)
            and iteration_lane_nodes
        )
        if parallel_candidate:
            deployment_id = next_deployment_region_id
            next_deployment_region_id += 1
            live_lane_nodes = tuple(
                tuple(
                    value_id
                    for value_id in node_ids
                    if value_id in graph.G
                )
                for node_ids in iteration_lane_nodes
            )
            live_lane_nodes = tuple(
                node_ids for node_ids in live_lane_nodes if node_ids
            )
            lanes = tuple(
                ControlDeploymentLane(
                    index=index,
                    value_ids=node_ids,
                    source_node_ids=node_ids,
                )
                for index, node_ids in enumerate(live_lane_nodes)
            )
            if lanes:
                region = ControlDeploymentRegion(
                    region_id=deployment_id,
                    kind="parallel_candidate",
                    schedule="independent_lanes",
                    schedule_preference=str(
                        graph.G.graph.get(
                            "deployment_schedule_preference", "alap"
                        )
                    ),
                    lanes=lanes,
                    origin="unrolled_loop",
                    source_loop_node_id=int(loop.node_id),
                )
                deployment_regions.append(region)
                for lane in lanes:
                    for value_id in lane.source_node_ids:
                        attributes = dict(
                            graph.G.nodes[value_id].get("attributes") or {}
                        )
                        memberships = tuple(attributes.get(
                            "deployment_memberships", ()
                        ))
                        attributes["deployment_memberships"] = tuple(
                            dict.fromkeys((
                                *memberships,
                                (deployment_id, lane.index),
                            ))
                        )
                        graph.G.nodes[value_id]["attributes"] = attributes
        handled_loop_ids.update(
            int(member.loop.node_id) for member in selected_plans
        )
        evaporated.extend(selected_plans)

    if evaporated:
        live = {
            int(root)
            for root in graph.roots
            if int(root) in graph.G
        }
        # Dataflow ancestry alone is insufficient to retain source control:
        # loop nodes own their bodies semantically, and are not necessarily a
        # value predecessor of the function's return roots.  Evaporating an
        # unrelated finite comprehension must therefore keep every loop that
        # was *not* evaporated, together with the values in its iteration
        # body.  Otherwise the planner deletes the enclosing while/for while
        # leaving their body reads alive as impossible external inputs.
        for plan in plans:
            if int(plan.loop.node_id) in handled_loop_ids:
                continue
            live.add(int(plan.loop.node_id))
            live.update(map(int, plan.loop.body_nodes))
            live.update(map(int, plan.loop.condition_nodes))
            live.update(
                int(value_id)
                for _name, value_id in plan.loop.target_bindings
            )
            if plan.loop.iterable_node is not None:
                live.add(int(plan.loop.iterable_node))
            for output in plan.loop.iteration_outputs:
                live.update((
                    int(output.value_id),
                    int(output.result_value_id),
                    int(output.materializer_node_id),
                ))
            for effect in plan.loop.state_effects:
                live.add(int(effect.effect_node_id))
                live.add(int(effect.state_input_id))
                if effect.state_output_id is not None:
                    live.add(int(effect.state_output_id))
                if effect.loop_result_id is not None:
                    live.add(int(effect.loop_result_id))
                live.update(map(int, effect.argument_value_ids))
        # Loop plans are snapshots taken before evaporation rewrites the
        # graph.  A sibling/nested rewrite can legitimately remove IDs still
        # named by a retained plan; only live graph nodes may seed ancestry.
        live.intersection_update(int(node_id) for node_id in graph.G)
        for root in tuple(live):
            live.update(int(node_id) for node_id in nx.ancestors(graph.G, root))
        graph.G.remove_nodes_from(
            tuple(node_id for node_id in graph.G if node_id not in live)
        )
        _rebuild_graph_edges(graph)
        identities = graph.G.graph.get("identity_table") or {}
        graph.G.graph["identity_table"] = {
            str(name): tuple(
                int(value_id)
                for value_id in value_ids
                if int(value_id) in graph.G
            )
            for name, value_ids in identities.items()
            if any(int(value_id) in graph.G for value_id in value_ids)
        }
        graph.G.graph["evaporated_loop_plans"] = tuple(evaporated)
        live_deployment_regions = []
        for region in deployment_regions:
            live_lanes = []
            for lane in region.lanes:
                live_nodes = tuple(
                    value_id
                    for value_id in lane.source_node_ids
                    if int(value_id) in graph.G
                )
                if not live_nodes:
                    continue
                new_index = len(live_lanes)
                for value_id in live_nodes:
                    attributes = dict(
                        graph.G.nodes[value_id].get("attributes") or {}
                    )
                    memberships = tuple(
                        membership
                        for membership in attributes.get(
                            "deployment_memberships", ()
                        )
                        if tuple(map(int, membership))
                        != (int(region.region_id), int(lane.index))
                    )
                    attributes["deployment_memberships"] = tuple(
                        dict.fromkeys((
                            *memberships,
                            (int(region.region_id), new_index),
                        ))
                    )
                    graph.G.nodes[value_id]["attributes"] = attributes
                live_lanes.append(replace(
                    lane,
                    index=new_index,
                    value_ids=tuple(
                        value_id
                        for value_id in lane.value_ids
                        if int(value_id) in graph.G
                    ),
                    source_node_ids=live_nodes,
                ))
            if live_lanes:
                live_deployment_regions.append(replace(
                    region, lanes=tuple(live_lanes)
                ))
        graph.G.graph["control_deployment_regions"] = tuple(
            live_deployment_regions
        )
        graph.G.graph["canonical_value_ids"] = True
    return tuple(evaporated)


def bind_control_deployments_to_regions(
    deployments: Iterable[ControlDeploymentRegion],
    scheduled_regions: Iterable[Iterable[int]],
) -> tuple[ControlDeploymentRegion, ...]:
    """Bind source-node deployment lanes to scheduled numerical regions.

    Loop evaporation records exact cloned node membership before scheduling.
    Region reduction may fuse those nodes; this adapter translates stable
    node provenance into the scheduled-region identities consumed by Control
    IR and SSA without selecting a backend.
    """

    scheduled = tuple(
        frozenset(map(int, node_ids)) for node_ids in scheduled_regions
    )
    bound = []
    for deployment in deployments:
        lanes = []
        for lane in deployment.lanes:
            source_nodes = frozenset(map(int, lane.source_node_ids))
            region_indices = tuple(
                index
                for index, node_ids in enumerate(scheduled)
                if source_nodes.intersection(node_ids)
            )
            if not region_indices:
                continue
            lanes.append(replace(
                lane,
                index=len(lanes),
                region_indices=region_indices,
            ))
        if lanes:
            bound.append(replace(deployment, lanes=tuple(lanes)))
    return tuple(bound)


def materialize_retained_loop_ports(
    graph: Any,
    plans: Iterable[LoopPlan],
) -> tuple[LoopPlan, ...]:
    """Create result/state ports only for loops that survived evaporation."""

    if not graph.G.graph.get("canonical_value_ids"):
        raise ValueError("retained loop ports require canonical value IDs")
    next_value_id = max(graph.G.nodes, default=-1) + 1
    identities = {
        str(name): list(value_ids)
        for name, value_ids in (
            graph.G.graph.get("identity_table") or {}
        ).items()
    }
    materialized_plans = []

    def add_port(
        node_type: str,
        label: str,
        parents: tuple[tuple[int, str], ...],
        attributes: dict[str, object],
    ) -> int:
        nonlocal next_value_id
        node_id = next_value_id
        next_value_id += 1
        graph.G.add_node(
            node_id,
            type=node_type,
            label=label,
            op=node_type.lower(),
            expr_obj=None,
            value_id=node_id,
            parents=list(parents),
            children=[],
            attributes=attributes,
        )
        return node_id

    def rewire_continuation(
        old_value_id: int,
        new_value_id: int,
        owned_nodes: frozenset[int],
    ) -> None:
        for node_id, data in graph.G.nodes(data=True):
            if int(node_id) in owned_nodes or int(node_id) == new_value_id:
                continue
            data["parents"] = [
                (
                    new_value_id if int(parent) == old_value_id else int(parent),
                    role,
                )
                for parent, role in data.get("parents") or ()
            ]
        graph.roots = [
            new_value_id if int(root) == old_value_id else int(root)
            for root in graph.roots
        ]

    for plan in tuple(plans):
        if plan.strategy in {LoopStrategy.UNROLL, LoopStrategy.CONSTANT}:
            materialized_plans.append(plan)
            continue
        loop = plan.loop
        loop_data = graph.G.nodes[int(loop.node_id)]
        attributes = loop_data.setdefault("attributes", {})
        if attributes.get("loop_ports_materialized"):
            materialized_plans.append(plan)
            continue
        recursion_region_id = attributes.get("recursion_region_id")
        region_nodes = (
            tuple(
                int(node_id)
                for node_id, data in graph.G.nodes(data=True)
                if (
                    recursion_region_id is not None
                    and (data.get("attributes") or {}).get(
                        "recursion_region_id"
                    ) == recursion_region_id
                )
            )
            if recursion_region_id is not None
            else ()
        )
        carried_update_cone = {
            int(ancestor)
            for _name, _initial, updated in loop.carried_bindings
            if int(updated) in graph.G
            for ancestor in nx.ancestors(graph.G, int(updated))
        }
        owned_nodes = frozenset((
            int(loop.node_id),
            *map(int, loop.body_nodes),
            *region_nodes,
            *carried_update_cone,
            *(int(effect.effect_node_id) for effect in loop.state_effects),
        ))

        carried_results = {}
        for name, _initial, updated in loop.carried_bindings:
            result_id = add_port(
                "LoopResult",
                str(name),
                (
                    (int(updated), "value"),
                    (int(loop.node_id), "control"),
                ),
                {
                    "binding_name": str(name),
                    "loop_id": int(loop.node_id),
                    "result_kind": "carried",
                },
            )
            rewire_continuation(
                int(updated), result_id, owned_nodes
            )
            identities.setdefault(str(name), []).append(result_id)
            carried_results[str(name)] = result_id

        planned_iteration_outputs = []
        for output_index, output in enumerate(loop.iteration_outputs):
            collection_id = add_port(
                "LoopResult",
                f"iteration_output_{output_index}",
                (
                    (int(output.value_id), "value"),
                    (int(loop.node_id), "control"),
                ),
                {
                    "loop_id": int(loop.node_id),
                    "result_kind": "collection",
                    "materializer_node_id": int(
                        output.materializer_node_id
                    ),
                },
            )
            materializer = graph.G.nodes[
                int(output.materializer_node_id)
            ]
            materializer_attributes = dict(
                materializer.get("attributes") or {}
            )
            materializer_attributes.update({
                "producer_kind": "aggregate_materialization",
                "materialized_source_value_ids": (collection_id,),
                "collection_owner_id": collection_id,
            })
            materializer["attributes"] = materializer_attributes
            planned_iteration_outputs.append(LoopIterationOutput(
                value_id=int(output.value_id),
                result_value_id=collection_id,
                materializer_node_id=int(output.materializer_node_id),
            ))

        effects = []
        for effect in loop.state_effects:
            aggregate_leaves: tuple[int, ...] = ()
            if effect.argument_value_ids:
                argument_id = int(effect.argument_value_ids[0])
                if argument_id in graph.G:
                    argument_attributes = (
                        graph.G.nodes[argument_id].get("attributes") or {}
                    )
                    if (
                        argument_attributes.get("producer_kind")
                        == "aggregate"
                    ):
                        aggregate_leaves = tuple(map(
                            int,
                            argument_attributes.get(
                                "aggregate_leaf_value_ids",
                                (),
                            ),
                        ))
            if aggregate_leaves:
                aggregate_consumers = tuple(
                    int(node_id)
                    for node_id, data in graph.G.nodes(data=True)
                    if int(node_id) not in owned_nodes
                    and any(
                        int(parent) == int(effect.state_input_id)
                        for parent, _role in (
                            data.get("parents") or ()
                        )
                    )
                )
                result_ids = []
                for path_index, leaf_id in enumerate(aggregate_leaves):
                    state_port = add_port(
                        "LoopStatePort",
                        (
                            f"{effect.state_name}."
                            f"{path_index}.{effect.operator}"
                        ),
                        (
                            (int(effect.state_input_id), "state"),
                            (int(effect.effect_node_id), "effect"),
                        ),
                        {
                            "binding_name": str(effect.state_name),
                            "operator": str(effect.operator),
                            "loop_id": int(loop.node_id),
                            "aggregate_path": (int(path_index),),
                        },
                    )
                    result_id = add_port(
                        "LoopResult",
                        f"{effect.state_name}.{path_index}",
                        (
                            (state_port, "value"),
                            (int(loop.node_id), "control"),
                        ),
                        {
                            "binding_name": str(effect.state_name),
                            "loop_id": int(loop.node_id),
                            "result_kind": "state_leaf",
                            "aggregate_path": (int(path_index),),
                        },
                    )
                    result_ids.append(result_id)
                    effects.append({
                        "state_name": str(effect.state_name),
                        "operator": str(effect.operator),
                        "effect_mode": effect.mode.value,
                        "state_input_id": int(effect.state_input_id),
                        "effect_node_id": int(effect.effect_node_id),
                        "state_output_id": state_port,
                        "loop_result_id": result_id,
                        "argument_value_ids": (int(leaf_id),),
                    })
                for consumer_id in aggregate_consumers:
                    consumer = graph.G.nodes[consumer_id]
                    consumer["parents"] = [
                        (int(parent), role)
                        for parent, role in (
                            consumer.get("parents") or ()
                        )
                        if int(parent) != int(effect.state_input_id)
                    ]
                    consumer["parents"].extend(
                        (int(result_id), f"arg{index}")
                        for index, result_id in enumerate(result_ids)
                    )
                    consumer_attributes = dict(
                        consumer.get("attributes") or {}
                    )
                    consumer_attributes["materialization_kind"] = (
                        "retained_loop_aggregate"
                    )
                    consumer_attributes["materialized_value_ids"] = tuple(
                        result_ids
                    )
                    consumer_attributes["loop_aggregate_axis"] = 0
                    consumer["attributes"] = consumer_attributes
                if int(effect.state_input_id) in set(graph.roots):
                    aggregate_id = add_port(
                        "LoopAggregateResult",
                        str(effect.state_name),
                        tuple(
                            (int(result_id), f"arg{index}")
                            for index, result_id in enumerate(result_ids)
                        ),
                        {
                            "binding_name": str(effect.state_name),
                            "materialization_kind": (
                                "retained_loop_aggregate"
                            ),
                            "materialized_value_ids": tuple(result_ids),
                            "loop_aggregate_axis": 0,
                        },
                    )
                    graph.roots = [
                        aggregate_id
                        if int(root) == int(effect.state_input_id)
                        else int(root)
                        for root in graph.roots
                    ]
                    identities.setdefault(
                        str(effect.state_name), []
                    ).append(aggregate_id)
                continue
            state_port = add_port(
                "LoopStatePort",
                f"{effect.state_name}.{effect.operator}",
                (
                    (int(effect.state_input_id), "state"),
                    (int(effect.effect_node_id), "effect"),
                ),
                {
                    "binding_name": str(effect.state_name),
                    "operator": str(effect.operator),
                    "loop_id": int(loop.node_id),
                },
            )
            result_id = add_port(
                "LoopResult",
                str(effect.state_name),
                (
                    (state_port, "value"),
                    (int(loop.node_id), "control"),
                ),
                {
                    "binding_name": str(effect.state_name),
                    "loop_id": int(loop.node_id),
                    "result_kind": "state",
                },
            )
            rewire_continuation(
                int(effect.state_input_id),
                result_id,
                owned_nodes | {state_port},
            )
            identities.setdefault(
                str(effect.state_name), []
            ).append(result_id)
            effects.append({
                "state_name": str(effect.state_name),
                "operator": str(effect.operator),
                "effect_mode": effect.mode.value,
                "state_input_id": int(effect.state_input_id),
                "effect_node_id": int(effect.effect_node_id),
                "state_output_id": state_port,
                "loop_result_id": result_id,
                "argument_value_ids": tuple(
                    map(int, effect.argument_value_ids)
                ),
            })
        if effects:
            attributes["loop_state_effects"] = tuple(effects)
        attributes["loop_result_ports"] = {
            **carried_results,
            **{
                str(effect["state_name"]): int(effect["loop_result_id"])
                for effect in effects
            },
        }
        attributes["loop_ports_materialized"] = True
        planned_effects = tuple(
            LoopStateEffect(
                state_name=str(effect["state_name"]),
                operator=str(effect["operator"]),
                state_input_id=int(effect["state_input_id"]),
                effect_node_id=int(effect["effect_node_id"]),
                state_output_id=int(effect["state_output_id"]),
                loop_result_id=int(effect["loop_result_id"]),
                argument_value_ids=tuple(
                    map(int, effect["argument_value_ids"])
                ),
                mode=LoopStateEffectMode(
                    effect.get("effect_mode", "opaque")
                ),
            )
            for effect in effects
        )
        planned_loop = replace(
            loop,
            state_effects=planned_effects,
            iteration_outputs=tuple(planned_iteration_outputs),
        )
        planned_semantic = (
            None
            if plan.semantic is None
            else replace(
                plan.semantic,
                state_effects=planned_effects,
                iteration_outputs=tuple(planned_iteration_outputs),
            )
        )
        materialized_plans.append(replace(
            plan,
            loop=planned_loop,
            semantic=planned_semantic,
        ))

    graph.G.graph["identity_table"] = {
        name: tuple(dict.fromkeys(map(int, value_ids)))
        for name, value_ids in identities.items()
    }
    _rebuild_graph_edges(graph)
    return tuple(materialized_plans)


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


def _static_iterable_expression(
    expression: ast.AST | None,
    specializations: dict[str, object],
) -> tuple[object, ...] | None:
    """Evaluate structural iterable syntax from compiler-known literals only."""

    def resolve(node: ast.AST):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in specializations:
            return specializations[node.id]
        if isinstance(node, ast.Attribute) and not node.attr.startswith("_"):
            base = resolve(node.value)
            try:
                return getattr(base, node.attr)
            except AttributeError as error:
                # A statically-resolved base can be a real None (the base
                # expression's own constant-propagated value at this point
                # in the traced code, not a compiler artifact) -- that is
                # exactly "not a compiler-known iterable literal" here, not
                # a crash. Re-raising as ValueError keeps this function's
                # own contract: every non-resolvable expression fails the
                # same way, regardless of which branch decided it couldn't.
                raise ValueError(str(error)) from error
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [resolve(item) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"range", "enumerate", "zip"}
        ):
            args = [resolve(argument) for argument in node.args]
            keywords = {
                keyword.arg: resolve(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            constructor = {
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
            }[node.func.id]
            return constructor(*args, **keywords)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"items", "keys", "values"}
            and not node.args
            and not node.keywords
        ):
            owner = resolve(node.func.value)
            if isinstance(owner, Mapping):
                return getattr(owner, node.func.attr)()
        raise ValueError("expression is not a compiler-known iterable literal")

    if expression is None:
        return None
    try:
        value = resolve(expression)
        if isinstance(value, (str, bytes, bytearray)):
            return None
        return tuple(value)
    except (TypeError, ValueError):
        return None


def _static_predicate_expression(
    expression: ast.AST,
    environment: dict[str, object],
) -> bool:
    """Evaluate a comprehension predicate from source-known scalar values."""

    def resolve(node: ast.AST):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in environment:
            return environment[node.id]
        if isinstance(node, ast.UnaryOp):
            value = resolve(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if isinstance(node.op, ast.USub):
                return -value
        if isinstance(node, ast.BoolOp):
            values = [bool(resolve(value)) for value in node.values]
            return (
                all(values)
                if isinstance(node.op, ast.And)
                else any(values)
            )
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = resolve(node.left)
            right = resolve(node.comparators[0])
            operation = node.ops[0]
            if isinstance(operation, ast.Eq):
                return left == right
            if isinstance(operation, ast.NotEq):
                return left != right
            if isinstance(operation, ast.Lt):
                return left < right
            if isinstance(operation, ast.LtE):
                return left <= right
            if isinstance(operation, ast.Gt):
                return left > right
            if isinstance(operation, ast.GtE):
                return left >= right
        raise ValueError("predicate is not source-static")

    return bool(resolve(expression))


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
        if (
            isinstance(expression, (ast.For, ast.comprehension))
            and iterator_kind == "arithmetic_sequence"
        ):
            # Arithmetic iteration is a source fact established below from an
            # actual ``range(...)`` AST.  Discovery metadata is not allowed to
            # relabel an arbitrary iterable as a numeric range.
            iterator_kind = "iterable"
        body_nodes = tuple(
            by_role.get("body", ())
            or by_role.get("generators", ())
        )
        expression_nodes = {
            id(node_data.get("expr_obj")): candidate
            for candidate, node_data in graph.G.nodes(data=True)
            if node_data.get("expr_obj") is not None
        }
        signature_nodes: dict[tuple[Any, ...], int] | None = None

        def ast_signature(member: ast.AST) -> tuple[Any, ...]:
            return (
                type(member),
                getattr(member, "lineno", None),
                getattr(member, "col_offset", None),
                getattr(member, "end_lineno", None),
                getattr(member, "end_col_offset", None),
                ast.dump(member, include_attributes=False),
            )

        def graph_node_for_ast(member: ast.AST) -> int | None:
            nonlocal signature_nodes
            direct = expression_nodes.get(id(member))
            if direct is not None:
                return int(direct)
            if signature_nodes is None:
                signature_nodes = {}
                for candidate, node_data in graph.G.nodes(data=True):
                    candidate_expression = node_data.get("expr_obj")
                    if not isinstance(candidate_expression, ast.AST):
                        continue
                    signature = ast_signature(candidate_expression)
                    signature_nodes[signature] = min(
                        int(candidate),
                        signature_nodes.get(signature, int(candidate)),
                    )
            return signature_nodes.get(ast_signature(member))

        loop_controls: list[tuple[str, int, int | None, bool]] = []

        def collect_loop_controls(
            statements: Iterable[ast.stmt],
            guard: tuple[int, bool] | None = None,
        ) -> None:
            for statement in statements:
                if isinstance(statement, (ast.For, ast.While, ast.AsyncFor)):
                    # Its break/continue edges belong to the nested loop.
                    continue
                if isinstance(statement, (ast.Break, ast.Continue)):
                    statement_id = graph_node_for_ast(statement)
                    if statement_id is not None:
                        loop_controls.append((
                            "break" if isinstance(statement, ast.Break)
                            else "continue",
                            statement_id,
                            None if guard is None else guard[0],
                            True if guard is None else guard[1],
                        ))
                    continue
                if isinstance(statement, ast.If):
                    predicate_id = graph_node_for_ast(statement.test)
                    next_true = (
                        guard if predicate_id is None
                        else (int(predicate_id), True)
                    )
                    next_false = (
                        guard if predicate_id is None
                        else (int(predicate_id), False)
                    )
                    collect_loop_controls(statement.body, next_true)
                    collect_loop_controls(statement.orelse, next_false)
                    continue
                for field in ("body", "orelse", "finalbody"):
                    nested = getattr(statement, field, None)
                    if isinstance(nested, list):
                        collect_loop_controls(nested, guard)

        if isinstance(expression, (ast.For, ast.While)):
            collect_loop_controls(expression.body)
        # A range is control structure, not an opaque iterable value.  Some
        # ingested graphs retain only the ``iterable`` edge on the loop node;
        # recover the exact range-argument nodes from the already-ingested AST
        # identities so dynamic bounds remain graph dependencies rather than
        # becoming a fabricated ``u_control_None``.
        iterator_expression = (
            expression.iter
            if isinstance(expression, (ast.For, ast.comprehension))
            else None
        )
        if (
            isinstance(iterator_expression, ast.Call)
            and isinstance(iterator_expression.func, ast.Name)
            and iterator_expression.func.id == "range"
            and 1 <= len(iterator_expression.args) <= 3
        ):
            iterator_kind = "arithmetic_sequence"
            argument_ids = tuple(
                graph_node_for_ast(argument)
                for argument in iterator_expression.args
            )
            if all(node is not None for node in argument_ids):
                if len(argument_ids) == 1:
                    by_role["stop"] = [argument_ids[0]]
                else:
                    by_role["start"] = [argument_ids[0]]
                    by_role["stop"] = [argument_ids[1]]
                    if len(argument_ids) == 3:
                        by_role["step"] = [argument_ids[2]]
        if isinstance(expression, (ast.For, ast.While)):
            # Reduction may preserve structurally identical AST nodes without
            # preserving their Python object identity.  Use the same
            # identity-or-source-signature lookup used for iterator bounds so
            # an evaporated loop owns (and removes/clones) its complete body,
            # rather than leaving reads of its induction variable behind as
            # impossible external inputs.
            ast_body_nodes = tuple(
                member_node
                for statement in expression.body
                for member in ast.walk(statement)
                if (member_node := graph_node_for_ast(member)) is not None
            )
            body_nodes = tuple(dict.fromkeys((
                *body_nodes,
                *ast_body_nodes,
            )))
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
        if body_nodes:
            original_position = {
                int(member): position
                for position, member in enumerate(body_nodes)
            }

            def body_order(member: int):
                member_expression = graph.G.nodes[member].get("expr_obj")
                return (
                    getattr(member_expression, "lineno", float("inf")),
                    getattr(member_expression, "col_offset", float("inf")),
                    original_position.get(int(member), len(original_position)),
                    int(member),
                )

            body_graph = graph.G.subgraph(tuple(dict.fromkeys(body_nodes)))
            try:
                body_nodes = tuple(
                    int(member)
                    for member in nx.lexicographical_topological_sort(
                        body_graph,
                        key=body_order,
                    )
                )
            except nx.NetworkXUnfeasible as error:
                raise ValueError(
                    f"loop {node_id} has a cyclic single-iteration body; "
                    "loop-carried dependencies must cross explicit ports"
                ) from error
        condition_nodes = tuple(
            by_role.get("ifs", ())
            or by_role.get("test", ())
        )
        iterable_node = next(
            iter(by_role.get("iterable", ()) or by_role.get("iter", ())),
            None,
        )

        # Numeric bounds belong only to source arithmetic sequences.  Graph
        # ingestion may annotate an ordinary iterable loop with the iteration
        # observed while discovering the program (often ``stop=1`` for one
        # captured visit).  Treating that observation as a bound compiles an
        # execution instance, drops the iterable binding, and severs every
        # aggregate element/field identity.  A non-range loop is bounded by its
        # iterable identity or by a source-static iterable—never by discovery
        # values.
        if iterator_kind == "arithmetic_sequence":
            start = attributes.get("start")
            stop = attributes.get("stop")
            step = attributes.get("step")
        else:
            start = 0
            stop = None
            step = 1
        if iterator_kind == "arithmetic_sequence":
            for role, default in (
                ("start", 0),
                ("stop", None),
                ("step", 1),
            ):
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

        iterable_constant = None
        count = _trip_count(start, stop, step)
        if count is None and iterator_kind != "arithmetic_sequence":
            iterable_constant = _static_iterable_expression(
                iterator_expression,
                {
                    **dict(
                        graph.G.graph.get("parameter_defaults") or {}
                    ),
                    **dict(
                        graph.G.graph.get("planner_specializations") or {}
                    ),
                },
            )
            if iterable_constant is not None:
                count = len(iterable_constant)
        if (
            count is None
            and iterator_kind != "arithmetic_sequence"
            and isinstance(iterator_expression, (ast.Tuple, ast.List))
        ):
            try:
                literal_iterable = ast.literal_eval(iterator_expression)
            except (ValueError, TypeError):
                literal_iterable = None
            if isinstance(literal_iterable, (tuple, list)):
                iterable_constant = tuple(literal_iterable)
                count = len(iterable_constant)
        if count is None and iterable_node is not None:
            known, iterable = _constant(graph, iterable_node)
            if known and isinstance(iterable, (tuple, list, range)):
                count = len(iterable)
                iterable_constant = tuple(iterable)

        yield_nodes = tuple(
            sorted(
                expression_nodes[id(member)]
                for statement in (
                    expression.body
                    if isinstance(expression, (ast.For, ast.While))
                    else ()
                )
                for member in ast.walk(statement)
                if isinstance(member, (ast.Yield, ast.YieldFrom))
                and id(member) in expression_nodes
            )
        )
        state_effects = tuple(dict.fromkeys(
            LoopStateEffect(
                state_name=str(effect["state_name"]),
                operator=str(effect["operator"]),
                state_input_id=int(effect["state_input_id"]),
                effect_node_id=int(effect["effect_node_id"]),
                state_output_id=(
                    None
                    if effect.get("state_output_id") is None
                    else int(effect["state_output_id"])
                ),
                loop_result_id=(
                    None
                    if effect.get("loop_result_id") is None
                    else int(effect["loop_result_id"])
                ),
                argument_value_ids=tuple(
                    int(value_id)
                    for value_id in effect["argument_value_ids"]
                ),
                mode=LoopStateEffectMode(
                    effect.get("effect_mode", "opaque")
                ),
            )
            for effect in attributes.get("loop_state_effects", ())
        ))
        iteration_outputs = tuple(dict.fromkeys(
            LoopIterationOutput(
                value_id=int(output["value_id"]),
                result_value_id=int(output["result_value_id"]),
                materializer_node_id=int(
                    output["materializer_node_id"]
                ),
            )
            for output in attributes.get("loop_iteration_outputs", ())
        ))
        if not iteration_outputs and isinstance(expression, ast.comprehension):
            # A generator expression is itself the materializer when its
            # consumer is a reduction such as any/all/sum or receives it via
            # starred arguments.  Collection comprehensions and tuple/list
            # calls publish an explicit materializer in the reducer; this is
            # the general finite-generator fallback.
            generator_owner = next((
                int(successor)
                for successor in graph.G.successors(int(node_id))
                if isinstance(
                    graph.G.nodes[int(successor)].get("expr_obj"),
                    ast.GeneratorExp,
                )
                and any(
                    int(parent) == int(node_id)
                    and str(role) == "generators"
                    for parent, role in (
                        graph.G.nodes[int(successor)].get("parents") or ()
                    )
                )
            ), None)
            if generator_owner is not None:
                element_id = next((
                    int(parent)
                    for parent, role in (
                        graph.G.nodes[generator_owner].get("parents") or ()
                    )
                    if str(role) == "elt"
                ), None)
                if element_id is not None:
                    iteration_outputs = (LoopIterationOutput(
                        value_id=element_id,
                        result_value_id=generator_owner,
                        materializer_node_id=generator_owner,
                    ),)
        publication_nodes = tuple(
            (
                int(yield_id),
                next(
                    int(parent)
                    for parent, role in (
                        graph.G.nodes[yield_id].get("parents") or ()
                    )
                    if str(role) == "value"
                ),
                None,
            )
            for yield_id in yield_nodes
        )
        target_bindings = dict(
            attributes.get("loop_target_bindings") or {}
        )

        def lexical_target_names(target_node: ast.AST) -> tuple[str, ...]:
            if isinstance(target_node, ast.Name):
                return (target_node.id,)
            if isinstance(target_node, (ast.Tuple, ast.List)):
                return tuple(
                    name
                    for element in target_node.elts
                    for name in lexical_target_names(element)
                )
            return ()

        ordered_target_names = (
            lexical_target_names(expression.target)
            if isinstance(expression, (ast.For, ast.comprehension))
            else tuple(target_bindings)
        )
        if (
            iterable_constant is not None
            and isinstance(expression, ast.comprehension)
            and expression.ifs
            and ordered_target_names
        ):
            filtered = []
            specializations = dict(
                graph.G.graph.get("parameter_defaults") or {}
            )
            specializations.update(
                graph.G.graph.get("planner_specializations") or {}
            )
            for item in iterable_constant:
                values = dict(_destructure_loop_target(
                    expression.target, item
                ))
                environment = {
                    **specializations,
                    **{
                        name: values[name]
                        for name in ordered_target_names
                    },
                }
                try:
                    accepted = all(
                        _static_predicate_expression(
                            predicate,
                            environment,
                        )
                        for predicate in expression.ifs
                    )
                except ValueError:
                    accepted = True
                if accepted:
                    filtered.append(item)
            iterable_constant = tuple(filtered)
            count = len(iterable_constant)

        return LoopDescriptor(
            node_id=int(node_id),
            source_type=source_type,
            target=target,
            iterator_kind=iterator_kind,
            body_nodes=body_nodes,
            condition_nodes=condition_nodes,
            break_nodes=tuple(
                (node_id, predicate_id, expect_true)
                for kind, node_id, predicate_id, expect_true in loop_controls
                if kind == "break"
            ),
            continue_nodes=tuple(
                (node_id, predicate_id, expect_true)
                for kind, node_id, predicate_id, expect_true in loop_controls
                if kind == "continue"
            ),
            target_bindings=tuple(
                (
                    str(name),
                    int(target_bindings[name]),
                )
                for name in ordered_target_names
                if name in target_bindings
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
            start_node=(
                next(iter(by_role.get("start", ())), None)
                if iterator_kind == "arithmetic_sequence"
                else None
            ),
            stop_node=(
                next(iter(by_role.get("stop", ())), None)
                if iterator_kind == "arithmetic_sequence"
                else None
            ),
            step_node=(
                next(iter(by_role.get("step", ())), None)
                if iterator_kind == "arithmetic_sequence"
                else None
            ),
            iterable_node=iterable_node,
            iterable_constant=iterable_constant,
            trip_count=count,
            yield_nodes=yield_nodes,
            publication_nodes=publication_nodes,
            state_effects=state_effects,
            iteration_outputs=iteration_outputs,
            backpressured_output=bool(publication_nodes),
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
            and all(
                effect.mode
                is LoopStateEffectMode.INDEXED_PUBLICATION
                for effect in loop.state_effects
            )
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

    def discover(self, graph: Any) -> tuple[LoopPlan, ...]:
        """Discover and select loop realizations without constructing loop IR."""

        if not graph.G.graph.get("canonical_value_ids"):
            raise ValueError(
                "loop discovery requires canonical value IDs and specialization"
            )
        plans = []
        for node_id, data in graph.G.nodes(data=True):
            if isinstance(
                data.get("expr_obj"),
                (ast.For, ast.While, ast.comprehension),
            ):
                descriptor = self.describe(graph, node_id)
                plans.append(self.plan(descriptor))
        return tuple(plans)

    def materialize_semantic_ir(
        self,
        graph: Any,
        plans: Iterable[LoopPlan],
    ) -> tuple[LoopPlan, ...]:
        """Build exactly one semantic IR for each retained loop."""

        if not graph.G.graph.get("canonical_value_ids"):
            raise ValueError(
                "semantic loop IR requires canonical value IDs"
            )
        plans = tuple(plans)
        duplicate = tuple(
            int(plan.loop.node_id)
            for plan in plans
            if plan.semantic is not None
        )
        if duplicate:
            raise ValueError(
                "semantic loop IR already exists for retained loops "
                f"{duplicate!r}"
            )
        return tuple(
            replace(plan, semantic=self._semantic_loop(graph, plan))
            for plan in plans
        )

    def compose(self, graph: Any) -> tuple[LoopPlan, ...]:
        """Compatibility API for callers that need semantic retained-loop IR."""

        return self.materialize_semantic_ir(graph, self.discover(graph))

    def _semantic_loop(self, graph: Any, plan: LoopPlan) -> SemanticLoop:
        """Translate discovered structure into the shared configurable IR."""

        loop = plan.loop
        realization = {
            LoopStrategy.UNROLL: LoopRealization.UNROLL,
            LoopStrategy.NATIVE_SOURCE: LoopRealization.NATIVE,
            LoopStrategy.KPN: LoopRealization.KPN,
            LoopStrategy.DISPATCH: LoopRealization.DISPATCH,
            LoopStrategy.CONSTANT: LoopRealization.UNROLL,
        }[plan.strategy]
        if loop.iterator_kind == "arithmetic_sequence":
            kind = LoopDomainKind.RANGE
            domain = RangeDomain(
                LoopValue(loop.start_node, loop.start),
                LoopValue(loop.stop_node, loop.stop),
                LoopValue(loop.step_node, loop.step),
            )
        elif loop.source_type == "While":
            if len(loop.condition_nodes) != 1:
                raise ValueError(
                    f"while loop {loop.node_id} needs one condition value"
                )
            kind = LoopDomainKind.CONDITION
            domain = ConditionDomain(int(loop.condition_nodes[0]))
        else:
            if loop.iterable_node is None:
                raise ValueError(
                    f"iterable loop {loop.node_id} has no iterable value"
                )
            kind = LoopDomainKind.ITERABLE
            access = (
                IterableAccess.STATIC
                if loop.iterable_constant is not None
                else IterableAccess.RESIDENT
            )
            source_value_ids: tuple[int, ...] = ()
            iterable_data = graph.G.nodes[int(loop.iterable_node)]
            iterable_expression = iterable_data.get("expr_obj")
            if isinstance(iterable_expression, ast.GeneratorExp):
                access = IterableAccess.GENERATOR
            elif (
                (iterable_data.get("attributes") or {}).get("producer_kind")
                in {
                    "loop_materialization",
                    "aggregate_materialization",
                }
            ):
                argument_ids = tuple(map(
                    int,
                    (iterable_data.get("attributes") or {}).get(
                        "materialized_source_value_ids",
                        (),
                    ),
                ))
                if not argument_ids:
                    argument_ids = tuple(
                        int(parent)
                        for parent, role in (
                            iterable_data.get("parents") or ()
                        )
                        if str(role).startswith("arg:")
                    )
                access = IterableAccess.CLOSURE_AGGREGATE
                source_value_ids = argument_ids
            domain = IterableDomain(
                LoopValue(
                    int(loop.iterable_node),
                    loop.iterable_constant,
                ),
                loop.target_bindings,
                access,
                source_value_ids,
            )
        body = set(loop.body_nodes)
        captures = tuple(sorted({
            int(parent)
            for node_id in body
            if node_id in graph.G
            for parent, _role in (
                graph.G.nodes[node_id].get("parents") or ()
            )
            if parent not in body
        }))
        body_closure = PlanClosure(
            name=f"loop_{loop.node_id}_body",
            captures=captures,
            items=tuple(
                PlanLine.create(
                    str(
                        graph.G.nodes[node_id].get("op")
                        or graph.G.nodes[node_id].get("type")
                        or "node"
                    ),
                    inputs=tuple(
                        parent
                        for parent, _role in (
                            graph.G.nodes[node_id].get("parents") or ()
                        )
                    ),
                    outputs=(node_id,),
                )
                for node_id in loop.body_nodes
                if node_id in graph.G
            ),
        )
        return SemanticLoop(
            loop_id=loop.node_id,
            domain_kind=kind,
            domain=domain,
            body_node_ids=loop.body_nodes,
            body_closure=body_closure,
            carried=tuple(
                LoopCarriedState(name, initial, updated)
                for name, initial, updated in loop.carried_bindings
            ),
            state_effects=loop.state_effects,
            iteration_outputs=loop.iteration_outputs,
            effects=LoopEffects(
                break_value_ids=tuple(
                    predicate_id
                    for _node_id, predicate_id, _expect_true
                    in loop.break_nodes
                    if predicate_id is not None
                ),
                continue_value_ids=tuple(
                    predicate_id
                    for _node_id, predicate_id, _expect_true
                    in loop.continue_nodes
                    if predicate_id is not None
                ),
                yield_value_ids=tuple(
                    value_id
                    for _statement_id, value_id, _count_id
                    in loop.publication_nodes
                )
            ),
            policy=LoopPolicy(
                realization=realization,
                unroll_limit=self.capabilities.unroll_limit,
                backpressure=loop.backpressured_output,
                allow_parallel_iterations=(
                    not loop.carried_bindings
                    and not loop.backpressured_output
                    and bool(
                        loop.state_effects or loop.iteration_outputs
                    )
                    and all(
                        effect.mode
                        is LoopStateEffectMode.INDEXED_PUBLICATION
                        and effect.loop_result_id is not None
                        for effect in loop.state_effects
                    )
                ),
            ),
        )


def analyze_shader_loop_reductions(
    graph: Any,
    plans: Iterable[LoopPlan],
    region_nodes: Iterable[Iterable[int]],
) -> tuple[LoopShaderReduction, ...]:
    """Classify loops after shader compartmentalization.

    This is deliberately a second planner pass.  Loop discovery alone cannot
    decide whether retaining a loop in backend source is profitable or legal;
    that decision needs the actual deployment regions.  The pass records
    blockers instead of silently falling back to a CPU iteration loop.
    """

    regions = tuple(tuple(nodes) for nodes in region_nodes)
    recursion_regions = tuple(
        RecursionRegion(
            region_id=int(region_id),
            kind=str(record["kind"]),
            lower_as=str(record["lower_as"]),
            members=tuple(map(int, record["members"])),
            control_ir=bool(record.get("control_ir", True)),
            control_members=tuple(map(
                int, record.get("control_members", ())
            )),
            incoming=tuple(record.get("incoming", ())),
            outgoing=tuple(record.get("outgoing", ())),
            feedback=tuple(record.get("feedback", ())),
        )
        for region_id, record in sorted(
            (graph.G.graph.get("recursion_table") or {}).items()
        )
    )
    # ``ast.Try`` is deliberately absent here.  A Try node is ordinary,
    # already-evaluable dataflow -- ``topological_reducer.py``'s ``ast.Try``
    # reduction resolves a name whose branches disagree straight to the
    # Try node's own id (mirroring how an ``ast.If``'s differing branches
    # become a ``Phi``), and ``evaluate_node``'s own ``ast.Try`` handling
    # already knows how to run body/handlers and return whichever arm's
    # value applies.  What remains forbidden is control divergence with no
    # such resolution: ``raise`` has no continuation value to merge, and
    # ``with``/``async with``/``await`` have no reduction at all yet.
    forbidden = (
        ast.Raise,
        ast.With,
        ast.AsyncWith,
        ast.Await,
    )
    reductions = []
    identity_table = graph.G.graph.get("identity_table") or {}
    planner_constants = {
        **dict(graph.G.graph.get("parameter_defaults") or {}),
        **dict(graph.G.graph.get("planner_specializations") or {}),
    }

    def specialized_input_value(data: Mapping[str, Any]) -> object | None:
        attributes = data.get("attributes") or {}
        name = str(attributes.get("binding_name") or data.get("label") or "")
        value = planner_constants.get(name)
        return value if isinstance(value, (bool, int, float)) else None

    def control_scalar_expression(
        node_id: int | None,
        visiting: frozenset[int] = frozenset(),
    ) -> tuple[str, tuple[int, ...]]:
        """Lower a scalar bound expression to control source and root inputs."""

        if node_id is None:
            raise ValueError("control expression has no node")
        node_id = int(node_id)
        if node_id in visiting:
            return f"u_control_{node_id}", (node_id,)
        data = graph.G.nodes[node_id]
        value = data.get("constant")
        if value is None:
            value = (data.get("attributes") or {}).get("value")
        if isinstance(value, (bool, int, float)):
            return repr(value), ()
        if (
            data.get("type") == "Input"
            and (data.get("attributes") or {}).get("binding_kind")
            == "parameter"
        ):
            specialized = specialized_input_value(data)
            if specialized is not None:
                return repr(specialized), ()
            return f"u_control_{node_id}", (node_id,)

        parents = tuple(data.get("parents") or ())
        binary = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "mult": "*",
            "div": "/",
            "floordiv": "/",
            "mod": "%",
        }.get(str(data.get("op") or data.get("type")).lower())
        if binary is not None and len(parents) == 2:
            ordered = sorted(
                parents,
                key=lambda item: {
                    "lhs": 0,
                    "left": 0,
                    "rhs": 1,
                    "right": 1,
                }.get(str(item[1]), 2),
            )
            left, left_uniforms = control_scalar_expression(
                int(ordered[0][0]), visiting | {node_id}
            )
            right, right_uniforms = control_scalar_expression(
                int(ordered[1][0]), visiting | {node_id}
            )
            return (
                f"({left} {binary} {right})",
                tuple(dict.fromkeys((*left_uniforms, *right_uniforms))),
            )
        if (
            str(data.get("op") or data.get("type")).lower()
            in {"neg", "usub"}
            and len(parents) == 1
        ):
            operand, uniforms = control_scalar_expression(
                int(parents[0][0]), visiting | {node_id}
            )
            return f"(-{operand})", uniforms
        return f"u_control_{node_id}", (node_id,)

    def structured_control_expression(
        node_id: int,
        visiting: frozenset[int] = frozenset(),
    ) -> ControlExpression | None:
        node_id = int(node_id)
        if node_id in visiting:
            return ControlExpression("value", value_id=node_id)
        data = graph.G.nodes[node_id]
        known, literal = _constant(graph, node_id)
        if known and isinstance(literal, (bool, int, float)):
            return ControlExpression(
                "const", value_id=node_id, literal=literal
            )
        expression = data.get("expr_obj")
        op = str(data.get("op") or data.get("type") or "").lower()
        parents = tuple(data.get("parents") or ())
        if data.get("type") == "Input":
            specialized = specialized_input_value(data)
            if specialized is not None:
                return ControlExpression(
                    "const", value_id=node_id, literal=specialized
                )
            return ControlExpression("value", value_id=node_id)
        if isinstance(expression, ast.BoolOp):
            operator_name = "and" if isinstance(expression.op, ast.And) else "or"
            values = [
                structured_control_expression(parent, visiting | {node_id})
                for parent, role in sorted(
                    parents,
                    key=lambda item: int(str(item[1]).split(":")[-1])
                    if str(item[1]).startswith("value:") else 0,
                )
                if str(role).startswith("value")
            ]
            if not values or any(value is None for value in values):
                return None
            result = values[0]
            for value in values[1:]:
                result = ControlExpression(
                    operator_name, (result, value), value_id=node_id
                )
            return result
        operation = {
            "add": "add", "sub": "sub", "mul": "mul",
            "div": "div", "truediv": "div",
            "less": "lt", "lt": "lt",
            "lessequal": "le", "le": "le",
            "greater": "gt", "gt": "gt",
            "greaterequal": "ge", "ge": "ge",
            "equal": "eq", "eq": "eq",
            "notequal": "ne", "ne": "ne",
            "logical_and": "and", "land": "and",
            "logical_or": "or", "lor": "or",
            "logical_not": "not", "lnot": "not",
            "neg": "neg", "usub": "neg",
            "item": "item", "float": "float",
            "int": "int", "bool": "bool",
        }.get(op)
        if operation is None and isinstance(expression, ast.Call):
            if isinstance(expression.func, ast.Name):
                operation = {
                    "float": "float", "int": "int", "bool": "bool"
                }.get(expression.func.id)
        if operation is None:
            return ControlExpression("value", value_id=node_id)
        ignored_roles = {"callee", "ops", "operator"}
        operands = tuple(
            operand
            for parent, role in parents
            if str(role) not in ignored_roles
            for operand in (
                structured_control_expression(
                    int(parent), visiting | {node_id}
                ),
            )
            if operand is not None
        )
        arity = 1 if operation in {
            "item", "float", "int", "bool", "not", "neg"
        } else 2
        if len(operands) < arity:
            return None
        return ControlExpression(
            operation, operands[:arity], value_id=node_id
        )

    for plan in plans:
        loop = plan.loop
        loop_members = {int(loop.node_id), *map(int, loop.body_nodes)}
        recursion_region_id = next(
            (
                region.region_id
                for region in recursion_regions
                if loop_members.intersection(region.members)
            ),
            None,
        )
        # Source spellings are not identities.  Separate lexical loops
        # routinely reuse ``i``, ``item`` or ``packet``; using that spelling as
        # a control key lets projection retain or discard another loop's
        # bindings.  The planner-assigned loop node is stable and unique within
        # this graph, so every backend receives an identity-derived induction
        # symbol while the source name remains diagnostic metadata only.
        induction_name = f"iteration_{int(loop.node_id)}"
        projected_iterable_bindings = ()
        if (
            loop.stop is None
            and loop.stop_node is None
            and loop.iterable_node is not None
            and len(loop.target_bindings) > 1
            and loop.iterable_constant is None
        ):
            iterable_id = int(loop.iterable_node)
            iterable_data = graph.G.nodes[iterable_id]
            reference = (iterable_data.get("attributes") or {}).get(
                "static_python_reference"
            )
            if reference == "enumerate":
                source_id = next(
                    (
                        int(parent)
                        for parent, role in iterable_data.get("parents", ())
                        if str(role) in {"arg", "args", "arg:0", "arg0"}
                    ),
                    None,
                )
                if source_id is not None and len(loop.target_bindings) == 2:
                    projected_iterable_bindings = (
                        (
                            source_id,
                            int(loop.target_bindings[0][1]),
                            induction_name,
                            "induction",
                        ),
                        (
                            source_id,
                            int(loop.target_bindings[1][1]),
                            induction_name,
                            None,
                        ),
                    )
            else:
                projected_iterable_bindings = tuple(
                    (
                        iterable_id,
                        int(target_id),
                        induction_name,
                        int(position),
                    )
                    for position, (_name, target_id)
                    in enumerate(loop.target_bindings)
                )
        iterable_extent_id = (
            int(projected_iterable_bindings[0][0])
            if projected_iterable_bindings
            else loop.iterable_node
        )
        dynamic_bounds = tuple(
            (
                name,
                *control_scalar_expression(node_id),
            )
            for name, node_id, value in (
                ("start", loop.start_node, loop.start),
                ("stop", loop.stop_node, loop.stop),
                ("step", loop.step_node, loop.step),
            )
            if node_id is not None and value is None
        )
        bound_expressions = {
            name: expression
            for name, expression, _uniforms in dynamic_bounds
        }
        bound_uniform_ids = tuple(dict.fromkeys(
            value_id
            for _name, _expression, uniforms in dynamic_bounds
            for value_id in uniforms
        ))
        body = set(loop.body_nodes)
        lexical_position = {
            int(node_id): position
            for position, node_id in enumerate(loop.body_nodes)
        }
        body_region_indices = tuple(sorted(
            (
            index
            for index, nodes in enumerate(regions)
            if body.intersection(nodes)
            ),
            key=lambda index: min(
                lexical_position[node_id]
                for node_id in regions[index]
                if node_id in lexical_position
            ),
        ))
        condition = set(map(int, loop.condition_nodes))
        condition_region_indices = tuple(
            index
            for index, nodes in enumerate(regions)
            if condition.intersection(nodes)
            and index not in body_region_indices
        )
        region_indices = tuple(dict.fromkeys((
            *condition_region_indices,
            *body_region_indices,
        )))
        while_predicate_expression = (
            structured_control_expression(loop.condition_nodes[0])
            if loop.source_type == "While" and loop.condition_nodes
            else None
        )
        blockers = []
        if plan.strategy not in {
            LoopStrategy.NATIVE_SOURCE,
            LoopStrategy.UNROLL,
        }:
            blockers.append(f"strategy={plan.strategy.value}")
        if (
            plan.semantic is not None
            and isinstance(plan.semantic.domain, IterableDomain)
            and plan.semantic.domain.access
            in {
                IterableAccess.CLOSURE_AGGREGATE,
                IterableAccess.GENERATOR,
            }
        ):
            if not (
                plan.semantic.domain.access
                is IterableAccess.CLOSURE_AGGREGATE
                and plan.semantic.domain.source_value_ids
                and len(loop.target_bindings) == 1
            ):
                blockers.append(
                    f"iterable-access={plan.semantic.domain.access.value}"
                )
        if not region_indices:
            blockers.append("no-shader-region")
        if (
            loop.source_type == "While"
            and not condition_region_indices
            and while_predicate_expression is None
        ):
            blockers.append("no-condition-region")
        if any(
            effect.mode is LoopStateEffectMode.OPAQUE
            for effect in loop.state_effects
        ):
            blockers.append("opaque-state-effect")
        if (
            loop.source_type != "While"
            and loop.stop is None
            and loop.stop_node is None
            and not (
                loop.iterable_node is not None
                and bool(loop.target_bindings)
                and (
                    len(loop.target_bindings) == 1
                    or loop.iterable_constant is not None
                    or bool(projected_iterable_bindings)
                )
            )
        ):
            blockers.append("unresolved-loop-bound")
        for node_id in loop.body_nodes:
            if node_id not in graph.G:
                continue
            expression = graph.G.nodes[node_id].get("expr_obj")
            if isinstance(expression, forbidden):
                blockers.append(type(expression).__name__)
        blockers = list(dict.fromkeys(blockers))

        # ``identity_table`` groups every value ever correlated with a source
        # name.  That class is scope-free: for one carried name it holds the
        # caller result, a local copy, an IndexedStore, the loop's LoopResult
        # port and a nested callee's parameter alongside the real body update.
        # Offering all of them as backedge candidates emits several pairs per
        # binding, and only the one produced inside the loop can ever satisfy
        # the header Phi.  Restrict the candidates to the loop body's induced
        # subgraph, which is the graph-level statement of "this version was
        # written by this cycle"; the lexical update stays authoritative.
        body_scope = frozenset(map(int, loop.body_nodes))
        carried_aliases = tuple(dict.fromkeys(
            (
                int(alias),
                int(initial),
            )
            for name, initial, updated in loop.carried_bindings
            for alias in (*tuple(identity_table.get(name, ())), updated)
            if int(alias) != int(initial)
            and (int(alias) == int(updated) or int(alias) in body_scope)
        ))
        body_items: list[tuple[int, object]] = [
            (
                min(
                    lexical_position[node_id]
                    for node_id in regions[region_index]
                    if node_id in lexical_position
                ),
                StatementBlock((f"__scheduled_region_{region_index}__",)),
            )
            for region_index in body_region_indices
        ]
        body_items.extend(
            (
                lexical_position[node_id],
                LoopControlBlock(
                    action,
                    predicate_value_id,
                    expect_true,
                    (
                        None if predicate_value_id is None
                        else structured_control_expression(predicate_value_id)
                    ),
                ),
            )
            for action, controls in (
                ("break", loop.break_nodes),
                ("continue", loop.continue_nodes),
            )
            for node_id, predicate_value_id, expect_true in controls
            if node_id in lexical_position
        )
        for yield_id, publish_value_id, publish_count_id in loop.publication_nodes:
            if yield_id not in lexical_position:
                continue
            predicate = next((
                int(parent)
                for _owner_id, owner in graph.G.nodes(data=True)
                if any(
                    int(candidate) == int(yield_id) and str(role) == "body"
                    for candidate, role in (owner.get("parents") or ())
                )
                for parent, role in (owner.get("parents") or ())
                if str(role) == "test"
            ), None)
            body_items.append((
                lexical_position[yield_id],
                StreamPublishBlock(
                    stream_id=0,
                    value_id=publish_value_id,
                    count_value_id=publish_count_id,
                    predicate_value_id=predicate,
                ),
            ))
        scheduled_body = SequenceBlock(tuple(
            block for _position, block in sorted(
                body_items, key=lambda item: item[0]
            )
        ))

        def planned_root():
            if loop.source_type == "While":
                return WhileBlock(
                    predicate_value_id=int(loop.condition_nodes[0]),
                    condition=SequenceBlock(tuple(
                        StatementBlock((f"__scheduled_region_{index}__",))
                        for index in condition_region_indices
                    )),
                    body=scheduled_body,
                    carried_aliases=carried_aliases,
                    recursion_region_id=recursion_region_id,
                    predicate_expression=while_predicate_expression,
                )
            return LoopBlock(
                induction=induction_name,
                start=(
                    bound_expressions.get("start", "0")
                    if loop.start is None else str(loop.start)
                ),
                stop=str(loop.stop) if loop.stop is not None else (
                    str(len(plan.semantic.domain.source_value_ids))
                    if (
                        plan.semantic is not None
                        and isinstance(plan.semantic.domain, IterableDomain)
                        and plan.semantic.domain.access
                        is IterableAccess.CLOSURE_AGGREGATE
                        and plan.semantic.domain.source_value_ids
                    )
                    else f"__iterable_extent_{iterable_extent_id}__"
                    if loop.stop_node is None and loop.iterable_node is not None
                    and bool(loop.target_bindings)
                    and loop.iterable_constant is None
                    else str(len(loop.iterable_constant))
                    if loop.iterable_constant is not None
                    else bound_expressions.get(
                        "stop", f"u_control_{loop.stop_node}"
                    )
                ),
                step=(
                    bound_expressions.get("step", "1")
                    if loop.step is None else str(loop.step)
                ),
                body=scheduled_body,
                carried_aliases=carried_aliases,
                parallel_iterations=bool(
                    not prefer_c_dispatch
                    and plan.semantic is not None
                    and plan.semantic.policy.allow_parallel_iterations
                ),
                dispatch_shell="c" if prefer_c_dispatch else "glsl",
                recursion_region_id=recursion_region_id,
                schedule_preference=str(
                    graph.G.graph.get(
                        "deployment_schedule_preference", "alap"
                    )
                ),
            )
        trip_count = loop.trip_count
        removed = (
            None
            if trip_count is None
            else max(0, trip_count * len(region_indices) - len(region_indices))
        )
        # A retained extensive loop around a few coarse shader closures is a
        # dispatch program, not shader-internal control.  Dissolve it to a C
        # command shell by default.  Dense loops with many numerical regions
        # remain candidates for genuine same-ABI GLSL fusion/workgroup
        # mapping; they are not serialized into thousands of C commands.
        prefer_c_dispatch = bool(
            not blockers
            and loop.iterator_kind == "arithmetic_sequence"
            and not loop.carried_bindings
            and bool(loop.state_effects or loop.iteration_outputs)
            and 0 < len(region_indices) <= 4
            and (
                trip_count is None
                or trip_count > int(
                    (
                        plan.semantic.policy.unroll_limit
                        if plan.semantic is not None
                        and plan.semantic.policy.unroll_limit is not None
                        else 8
                    )
                )
            )
            and not loop.backpressured_output
        )
        reductions.append(LoopShaderReduction(
            loop_node_id=loop.node_id,
            region_indices=region_indices,
            carried_bindings=loop.carried_bindings,
            collapsible=not blockers,
            blockers=tuple(blockers),
            estimated_dispatches_removed=removed,
            control_program=(
                None
                if blockers
                else ControlProgram(
                    region_indices=region_indices,
                    uniforms=tuple(
                        ControlUniform(
                            f"u_control_{node_id}",
                            int(node_id),
                        )
                        for node_id in bound_uniform_ids
                    ),
                    value_aliases=tuple(
                        dict.fromkeys(
                            (
                                int(alias),
                                int(initial),
                            )
                            for name, initial, updated
                            in loop.carried_bindings
                            for alias in (
                                *tuple(identity_table.get(name, ())),
                                updated,
                            )
                            if int(alias) != int(initial)
                        )
                    ),
                    iterable_bindings=(
                        (
                            int(loop.iterable_node),
                            int(loop.target_bindings[0][1]),
                            induction_name,
                        ),
                    )
                    if (
                        loop.stop is None
                        and loop.stop_node is None
                        and loop.iterable_node is not None
                        and len(loop.target_bindings) == 1
                        and loop.iterable_constant is None
                    )
                    else (),
                    static_iterable_bindings=tuple(
                        (
                            int(loop.iterable_node),
                            int(target_id),
                            induction_name,
                            tuple(
                                (
                                    tuple(item)[position]
                                    if len(loop.target_bindings) > 1
                                    else item
                                )
                                for item in loop.iterable_constant
                            ),
                        )
                        for position, (_name, target_id)
                        in enumerate(loop.target_bindings)
                    )
                    if (
                        loop.stop is None
                        and loop.stop_node is None
                        and loop.iterable_node is not None
                        and bool(loop.target_bindings)
                        and loop.iterable_constant is not None
                    )
                    else (),
                    collection_bindings=planned_collection_bindings(
                        graph,
                        loop,
                        frozenset(
                            node_id
                            for region in regions
                            for node_id in region
                        ),
                    ),
                    closure_iterable_bindings=(
                        (
                            int(loop.iterable_node),
                            int(loop.target_bindings[0][1]),
                            induction_name,
                            tuple(
                                int(value_id)
                                for value_id in (
                                    plan.semantic.domain.source_value_ids
                                )
                            ),
                        ),
                    )
                    if (
                        plan.semantic is not None
                        and isinstance(
                            plan.semantic.domain, IterableDomain
                        )
                        and plan.semantic.domain.access
                        is IterableAccess.CLOSURE_AGGREGATE
                        and plan.semantic.domain.source_value_ids
                        and loop.iterable_node is not None
                        and len(loop.target_bindings) == 1
                    )
                    else (),
                    projected_iterable_bindings=(
                        projected_iterable_bindings
                    ),
                    recursion_regions=tuple(
                        region
                        for region in recursion_regions
                        if region.region_id == recursion_region_id
                    ),
                    root=planned_root(),
                )
            ),
            preferred_shell=("c" if prefer_c_dispatch else "glsl"),
            dispatch_closure_count=len(region_indices),
        ))
    return tuple(reductions)


def indent_source(lines: Iterable[str], spaces: int = 4) -> tuple[str, ...]:
    prefix = " " * int(spaces)
    return tuple(prefix + line if line else line for line in lines)


__all__ = [
    "LoopBackendCapabilities",
    "LoopComposer",
    "LoopDescriptor",
    "LoopPlan",
    "LoopShaderReduction",
    "LoopStrategy",
    "analyze_shader_loop_reductions",
    "bind_control_deployments_to_regions",
    "evaporate_unrolled_loops",
    "indent_source",
    "materialize_retained_loop_ports",
    "planned_collection_bindings",
]
