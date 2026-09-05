"""Planner-owned loop classification and backend-source composition.

The ProcessGraph remains the semantic authority.  This module decides how a
backend should realize each retained loop; it does not reinterpret tensor
operators and it does not execute a Python loop as a substitute for compiled
control flow.
"""

from __future__ import annotations

import ast
import os
import sys
import copy
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping

import networkx as nx

from .control_source import (
    ControlDeploymentLane,
    ControlDeploymentRegion,
    ControlProgram,
    ControlSequenceMutation,
    ControlExpression,
    ControlUniform,
    LoopControlBlock,
    LoopBlock,
    RecursionRegion,
    SequenceBlock,
    StatementBlock,
    StreamPublishBlock,
    ValidationBlock,
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
from .process_graph_value_ids import next_process_value_id


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
    return_nodes: tuple[tuple[int, int | None, bool], ...] = ()
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
    # Every source ``return`` with a value inside this loop's body:
    # (return-value node id, guard chain ((predicate id, expect_true), ...
    # outermost first), per-slot value ids in function-output order).
    # Unlike ``return_nodes`` (the sole-root "terminal loop exit" special
    # case), these carry the return's OWN values so the function exit can
    # merge them per slot (control-aware result merging).
    return_controls: tuple[
        tuple[int, tuple[tuple[int, bool], ...], tuple[int | None, ...]], ...
    ] = ()
    # Every source break/continue in this loop's body:
    # (statement node id, action, guard chain outermost-first,
    #  ((pre-loop identity, value at the site), ...), arm_owned).
    # The last element is the enclosing ``if`` arm's (first, last) source
    # line when the statement ends that arm.  Such a site is ARM-OWNED when
    # a value it carries is produced by a body region INSIDE the arm (the
    # value dominates the exit edge only there): the conditional program
    # places it inside that arm and the loop schedule must not place it a
    # second time.  Otherwise (a bare ``if c: break``, an arm that only
    # rebinds constants/parameters, or values computed before the ``if``)
    # the lexical placement under the full guard chain is exact and a
    # region-less conditional program would have nothing to nest it into
    # the loop.
    control_sites: tuple[
        tuple[
            int, str, tuple[tuple[int, bool], ...],
            tuple[tuple[int, int], ...], tuple[int, int] | None,
        ], ...
    ] = ()
    # Names bound ONLY on break paths (never on the fall-through path):
    # (name, pre-loop identity, continuation identity).  The continuation
    # is the last break site's value; port materialization rewires its
    # post-loop consumers onto a LoopResult port whose exit Phi merges the
    # zero-trip/normal value with every break edge's value.
    break_bindings: tuple[tuple[str, int, int], ...] = ()


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
    # Numerical regions whose complete semantics are emitted by the control
    # program itself (for example a filtered-comprehension predicate lowered
    # as a predicated resident append).  They remain compilation artifacts for
    # provenance, but the outer shell must not schedule them a second time.
    structurally_owned_region_indices: tuple[int, ...] = ()


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


_CACHED_VALUE_ID_LEDGERS = (
    "aggregate_leaf_value_ids",
    "materialized_source_value_ids",
    "materialized_value_ids",
)


def _retarget_cached_value_ids(
    data: dict[str, Any],
    old_value_id: int,
    replacements: tuple[int, ...],
) -> None:
    """Keep a node's cached id copies equal to its (about to be) rewritten edges.

    Several node attributes are verbatim copies of parent edges or of other
    nodes' ids -- the leaf ledgers (`aggregate_leaf_value_ids`,
    `materialized_source_value_ids`, `materialized_value_ids`), a port's
    `value_source_id`, and a loop node's own `loop_carried_bindings` /
    `loop_target_initials` / `loop_state_effects` /
    `loop_iteration_outputs`. Every consumer that prefers the cache over the
    edge (region capture expansion, iterable trip counts, the retained-loop
    port builder itself, `_tensor_descriptor`'s identity check) reads the
    PRE-rewrite id forever if only the edge moves. The canonical relabel in
    `topological_reducer._normalize_lexical_values` already remaps exactly
    this set; the two edge rewriters here must too. `replacements` follows
    `_replace_parent_value`: one id substitutes in place, several expand a
    ledger entry the way an `elts`/`argN` edge is expanded.
    """

    old_value_id = int(old_value_id)
    new_ids = tuple(int(value_id) for value_id in replacements)
    if not new_ids:
        return
    single = new_ids[0] if len(new_ids) == 1 else None

    def substitute(value_ids) -> tuple[int, ...]:
        rewritten: list[int] = []
        for value_id in value_ids:
            if int(value_id) == old_value_id:
                rewritten.extend(new_ids)
            else:
                rewritten.append(int(value_id))
        return tuple(rewritten)

    attributes = data.get("attributes")
    if not isinstance(attributes, dict):
        return
    for ledger in _CACHED_VALUE_ID_LEDGERS:
        entries = attributes.get(ledger)
        if entries and any(int(entry) == old_value_id for entry in entries):
            attributes[ledger] = substitute(entries)
    if single is not None:
        if attributes.get("value_source_id") == old_value_id:
            attributes["value_source_id"] = single
        carried = attributes.get("loop_carried_bindings")
        if isinstance(carried, dict) and any(
            int(initial) == old_value_id or int(updated) == old_value_id
            for initial, updated in carried.values()
        ):
            attributes["loop_carried_bindings"] = {
                name: (
                    single if int(initial) == old_value_id else int(initial),
                    single if int(updated) == old_value_id else int(updated),
                )
                for name, (initial, updated) in carried.items()
            }
        break_bindings = attributes.get("loop_break_bindings")
        if isinstance(break_bindings, dict) and any(
            int(initial) == old_value_id or int(continuation) == old_value_id
            for initial, continuation in break_bindings.values()
        ):
            attributes["loop_break_bindings"] = {
                name: (
                    single if int(initial) == old_value_id else int(initial),
                    single if int(continuation) == old_value_id
                    else int(continuation),
                )
                for name, (initial, continuation) in break_bindings.items()
            }
        break_sites = attributes.get("loop_break_sites")
        if isinstance(break_sites, dict) and any(
            int(initial) == old_value_id or int(value) == old_value_id
            for site_values in break_sites.values()
            for initial, value in site_values.items()
        ):
            attributes["loop_break_sites"] = {
                span: {
                    (single if int(initial) == old_value_id else int(initial)):
                    (single if int(value) == old_value_id else int(value))
                    for initial, value in site_values.items()
                }
                for span, site_values in break_sites.items()
            }
        initials = attributes.get("loop_target_initials")
        if isinstance(initials, dict) and any(
            int(value_id) == old_value_id for value_id in initials.values()
        ):
            attributes["loop_target_initials"] = {
                name: single if int(value_id) == old_value_id else int(value_id)
                for name, value_id in initials.items()
            }
        effects = attributes.get("loop_state_effects")
        if effects and any(
            int(effect.get(key, -1)) == old_value_id
            for effect in effects
            for key in ("state_input_id", "effect_node_id")
        ):
            attributes["loop_state_effects"] = tuple(
                {
                    **effect,
                    **{
                        key: single
                        for key in ("state_input_id", "effect_node_id")
                        if int(effect.get(key, -1)) == old_value_id
                    },
                }
                for effect in effects
            )
        outputs = attributes.get("loop_iteration_outputs")
        if outputs and any(
            int(output.get(key, -1)) == old_value_id
            for output in outputs
            for key in ("value_id", "result_value_id", "materializer_node_id")
        ):
            attributes["loop_iteration_outputs"] = tuple(
                {
                    **output,
                    **{
                        key: single
                        for key in (
                            "value_id", "result_value_id",
                            "materializer_node_id",
                        )
                        if int(output.get(key, -1)) == old_value_id
                    },
                }
                for output in outputs
            )


def _retarget_plan_value_ids(
    plan: LoopPlan, old_value_id: int, new_value_id: int
) -> LoopPlan:
    """Return ``plan`` with every cached ``old_value_id`` renamed.

    A plan is built from the loop node's attributes before any port exists
    (`_loop_plans`), so its `state_input_id`s, carried initials and
    iteration outputs are copies of graph values. Once an earlier loop's
    `rewire_continuation` redirects those values' consumer edges, the copies
    are stale exactly like a node attribute would be; `add_port` builds the
    later loop's state port from the copy, so it must be renamed too.
    """

    old_value_id = int(old_value_id)
    new_value_id = int(new_value_id)

    def rename(value_id: int) -> int:
        return new_value_id if int(value_id) == old_value_id else int(value_id)

    def rename_effects(
        effects: tuple[LoopStateEffect, ...],
    ) -> tuple[LoopStateEffect, ...]:
        return tuple(
            replace(
                effect,
                state_input_id=rename(effect.state_input_id),
                effect_node_id=rename(effect.effect_node_id),
                argument_value_ids=tuple(
                    rename(value_id) for value_id in effect.argument_value_ids
                ),
            )
            if old_value_id in (
                int(effect.state_input_id), int(effect.effect_node_id),
                *map(int, effect.argument_value_ids),
            )
            else effect
            for effect in effects
        )

    def rename_outputs(
        outputs: tuple[LoopIterationOutput, ...],
    ) -> tuple[LoopIterationOutput, ...]:
        return tuple(
            replace(
                output,
                value_id=rename(output.value_id),
                result_value_id=rename(output.result_value_id),
                materializer_node_id=rename(output.materializer_node_id),
            )
            if old_value_id in (
                int(output.value_id), int(output.result_value_id),
                int(output.materializer_node_id),
            )
            else output
            for output in outputs
        )

    loop = plan.loop
    carried = tuple(
        (name, rename(initial), rename(updated))
        for name, initial, updated in loop.carried_bindings
    )
    return_controls = tuple(
        (
            rename(node_id),
            tuple((rename(predicate_id), expect_true) for predicate_id, expect_true in chain),
            tuple(None if slot is None else rename(slot) for slot in slots),
        )
        for node_id, chain, slots in loop.return_controls
    )
    control_sites = tuple(
        (
            rename(site_id),
            action,
            tuple(
                (rename(predicate_id), expect_true)
                for predicate_id, expect_true in chain
            ),
            tuple(
                (rename(initial), rename(value)) for initial, value in site_values
            ),
            arm_span,
        )
        for site_id, action, chain, site_values, arm_span in loop.control_sites
    )
    break_bindings = tuple(
        (name, rename(initial), rename(continuation))
        for name, initial, continuation in loop.break_bindings
    )
    renamed_loop = (
        replace(
            loop,
            carried_bindings=carried,
            state_effects=rename_effects(loop.state_effects),
            iteration_outputs=rename_outputs(loop.iteration_outputs),
            return_controls=return_controls,
            control_sites=control_sites,
            break_bindings=break_bindings,
        )
        if (
            carried != tuple(loop.carried_bindings)
            or return_controls != tuple(loop.return_controls)
            or control_sites != tuple(loop.control_sites)
            or break_bindings != tuple(loop.break_bindings)
            or any(
                old_value_id in (
                    int(effect.state_input_id), int(effect.effect_node_id),
                    *map(int, effect.argument_value_ids),
                )
                for effect in loop.state_effects
            )
            or any(
                old_value_id in (
                    int(output.value_id), int(output.result_value_id),
                    int(output.materializer_node_id),
                )
                for output in loop.iteration_outputs
            )
        )
        else loop
    )
    semantic = plan.semantic
    renamed_semantic = semantic
    if semantic is not None:
        semantic_carried = tuple(
            replace(
                state,
                initial_value_id=rename(state.initial_value_id),
                next_value_id=rename(state.next_value_id),
            )
            if old_value_id in (
                int(state.initial_value_id), int(state.next_value_id),
            )
            else state
            for state in semantic.carried
        )
        renamed_semantic = replace(
            semantic,
            carried=semantic_carried,
            state_effects=rename_effects(semantic.state_effects),
            iteration_outputs=rename_outputs(semantic.iteration_outputs),
        )
    if renamed_loop is loop and renamed_semantic is semantic:
        return plan
    return replace(plan, loop=renamed_loop, semantic=renamed_semantic)


def _replace_parent_value(
    graph: Any,
    old_value_id: int,
    replacements: tuple[int, ...],
) -> None:
    """Replace one value use without identifying its distinct consumers."""

    old_value_id = int(old_value_id)
    for _node_id, data in graph.G.nodes(data=True):
        _retarget_cached_value_ids(data, old_value_id, replacements)
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

    def add_clone(source_id: int, parents: tuple[tuple[int, str], ...]) -> int:
        clone_id = next_process_value_id(graph)
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
        constant_id = next_process_value_id(graph)
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
        # Collection mutations are resident memory effects, not values that can
        # be replaced by cloning the loop body's numerical producer cone. The
        # evaporator expands indexed publications and comprehension
        # materializers explicitly below; until it likewise emits one
        # append/add/extend call per iteration, deleting the source loop would
        # delete the mutation. Preserve iterative control so ordinary
        # sequence-SSA lowering owns the complete effect.
        if any(
            effect.mode in {
                LoopStateEffectMode.SEQUENCE_MUTATION,
                LoopStateEffectMode.MAPPING_MUTATION,
            }
            for member in selected_plans
            for effect in member.loop.state_effects
        ):
            continue
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
            # A Python default describes how a missing call argument is
            # supplied; it does not freeze that parameter for every call.
            # Only an explicit planner specialization may make a parameter-
            # dependent iterable source-static and therefore evaporatable.
            specializations = dict(
                graph.G.graph.get("planner_specializations") or {}
            )
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
                if int(updated) in mapping:
                    carried_values[int(initial)] = int(mapping[int(updated)])
                elif int(updated) not in graph.G:
                    # Static branch specialization may remove the update Phi
                    # before this finite loop is evaporated.  That is a proven
                    # no-update iteration, so retain the incoming carried
                    # identity instead of manufacturing an ABI input.
                    carried_values[int(initial)] = int(
                        mapping.get(int(initial), int(initial))
                    )
                else:
                    raise ValueError(
                        "unrolled loop omitted a live carried update: "
                        f"loop={loop.node_id}, binding={_name!r}, "
                        f"initial={initial}, updated={updated}"
                    )
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
        node_id = next_process_value_id(graph)
        semantic_sources = tuple(
            int(parent)
            for parent, role in parents
            if str(role) in {"value", "state"} and int(parent) in graph.G
        )
        port_attributes = dict(attributes)
        port_tensor = {}
        if len(semantic_sources) == 1:
            source_id = semantic_sources[0]
            source_data = graph.G.nodes[source_id]
            port_attributes["value_source_id"] = source_id
            port_tensor = copy.deepcopy(dict(source_data.get("tensor") or {}))
            source_attributes = source_data.get("attributes") or {}
            for key in (
                "producer_kind", "aggregate_kind",
                "aggregate_leaf_value_ids", "tensor_output_descriptors",
            ):
                if key in source_attributes:
                    port_attributes[key] = copy.deepcopy(source_attributes[key])
        graph.G.add_node(
            node_id,
            type=node_type,
            label=label,
            op=node_type.lower(),
            expr_obj=None,
            value_id=node_id,
            parents=list(parents),
            children=[],
            attributes=port_attributes,
            tensor=port_tensor,
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
            # Every cached copy of an id this edge rewrite touches -- a
            # port's `value_source_id`, the leaf ledgers, and a not-yet
            # materialized loop node's own carried/initial/state-effect
            # records -- must move with the edge, or a later loop's rewire
            # leaves an earlier loop's port (or a later loop's plan) naming
            # the PRE-rewire value forever (diagnosed via
            # tools/repro_step_with_dt_control_used.py: "loopstateport
            # value-source identity conflicts with its semantic edge", and
            # the parents-graph cycle behind it: loop A retargeted loop B's
            # effect-node edge while B's pending plan still said
            # state_input_id=8).
            _retarget_cached_value_ids(data, old_value_id, (new_value_id,))
        graph.roots = [
            new_value_id if int(root) == old_value_id else int(root)
            for root in graph.roots
        ]
        # The plan records of loops not yet materialized are the same cache
        # one level up: `state_input_id`, a carried binding's initial, and an
        # iteration output all name graph values whose consumer edges were
        # just redirected. `add_port` for those loops reads the plan, not the
        # edge, so retarget the plan the same way (and its semantic mirror).
        for index in range(materialized_count[0], len(pending_plans)):
            pending_plans[index] = _retarget_plan_value_ids(
                pending_plans[index], old_value_id, new_value_id
            )

    # Plans still ahead in this list are retargeted by `rewire_continuation`
    # as earlier loops redirect the values they name (see the helper); the
    # plan actually processed is always read from the list at that moment.
    pending_plans = list(plans)
    materialized_count = [0]
    for index in range(len(pending_plans)):
        materialized_count[0] = index + 1
        plan = pending_plans[index]
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
        for name, initial, continuation in loop.break_bindings:
            if str(name) in carried_results:
                continue
            result_id = add_port(
                "LoopResult",
                str(name),
                (
                    (int(continuation), "value"),
                    (int(loop.node_id), "control"),
                ),
                {
                    "binding_name": str(name),
                    "loop_id": int(loop.node_id),
                    "result_kind": "break_bound",
                    "initial_value_id": int(initial),
                },
            )
            rewire_continuation(
                int(continuation), result_id, owned_nodes
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
                # A comprehension owns resident sequence storage.  Its
                # elements are collected by the retained producer loop; it
                # is not a frozen closure aggregate whose members can be
                # enumerated as loop-source identities.
                "producer_kind": "sequence_materialization",
                "materialized_source_value_ids": (collection_id,),
                "collection_owner_id": collection_id,
            })
            materializer["attributes"] = materializer_attributes
            # A carried binding rewires its continuation onto the LoopResult
            # port; a collection output did not, so a reduction such as
            # ``any(... for ...)`` kept consuming the comprehension node --
            # which the loop no longer produces.  The loop published every
            # element into the collection and the reduction read an unrelated
            # anonymous slot.  The collection port is the value that leaves
            # this loop, so the continuation must name it.
            rewire_continuation(
                int(output.materializer_node_id),
                collection_id,
                owned_nodes,
            )
            planned_iteration_outputs.append(LoopIterationOutput(
                value_id=int(output.value_id),
                result_value_id=collection_id,
                materializer_node_id=int(output.materializer_node_id),
            ))

        effects = []
        for effect in loop.state_effects:
            if effect.mode in {
                LoopStateEffectMode.SEQUENCE_MUTATION,
                LoopStateEffectMode.MAPPING_MUTATION,
            }:
                # The sequence descriptor points at caller-owned arena,
                # length and status cells.  append/add/extend mutate those
                # cells in place, so manufacturing a LoopStatePort here
                # falsely turns the mutation into a value-carried update
                # with no producer in the retained loop body.
                effects.append({
                    "state_name": str(effect.state_name),
                    "operator": str(effect.operator),
                    "effect_mode": effect.mode.value,
                    "sequence_policy": effect.sequence_policy,
                    "argument_kind": effect.argument_kind,
                    "state_input_id": int(effect.state_input_id),
                    "effect_node_id": int(effect.effect_node_id),
                    "state_output_id": None,
                    "loop_result_id": None,
                    "argument_value_ids": tuple(
                        map(int, effect.argument_value_ids)
                    ),
                })
                continue
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
                "sequence_policy": effect.sequence_policy,
                "argument_kind": effect.argument_kind,
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
        attributes["loop_carried_updated_ids"] = tuple(
            int(updated) for _name, _initial, updated in loop.carried_bindings
        )
        attributes["loop_result_ports"] = {
            **carried_results,
            **{
                str(effect["state_name"]): int(effect["loop_result_id"])
                for effect in effects
                if effect["loop_result_id"] is not None
            },
        }
        attributes["loop_ports_materialized"] = True
        planned_effects = tuple(
            LoopStateEffect(
                state_name=str(effect["state_name"]),
                operator=str(effect["operator"]),
                state_input_id=int(effect["state_input_id"]),
                effect_node_id=int(effect["effect_node_id"]),
                state_output_id=(
                    None
                    if effect["state_output_id"] is None
                    else int(effect["state_output_id"])
                ),
                loop_result_id=(
                    None
                    if effect["loop_result_id"] is None
                    else int(effect["loop_result_id"])
                ),
                argument_value_ids=tuple(
                    map(int, effect["argument_value_ids"])
                ),
                mode=LoopStateEffectMode(
                    effect.get("effect_mode", "opaque")
                ),
                sequence_policy=effect.get("sequence_policy"),
                argument_kind=str(effect.get("argument_kind", "value")),
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
            matched = signature_nodes.get(ast_signature(member))
            if matched is not None:
                return matched
            # Lexical normalization materializes Name/GetAttr values without
            # retaining a raw ``expr_obj`` on those synthetic nodes. Recover
            # them from their canonical identity and receiver edge rather than
            # requiring an AST object that intentionally no longer exists.
            if isinstance(member, ast.Name):
                history = tuple(
                    (graph.G.graph.get("identity_table") or {}).get(
                        member.id, ()
                    )
                )
                for candidate in reversed(history):
                    if int(candidate) in graph.G:
                        return int(candidate)
                candidates = [
                    int(candidate)
                    for candidate, node_data in graph.G.nodes(data=True)
                    if (
                        (node_data.get("attributes") or {}).get(
                            "binding_name"
                        ) == member.id
                        or node_data.get("label") == member.id
                    )
                ]
                return min(candidates) if candidates else None
            if isinstance(member, ast.Attribute):
                receiver = graph_node_for_ast(member.value)
                candidates = [
                    int(candidate)
                    for candidate, node_data in graph.G.nodes(data=True)
                    if node_data.get("type") == "GetAttr"
                    and (node_data.get("attributes") or {}).get(
                        "attribute"
                    ) == member.attr
                    and (
                        receiver is None
                        or any(
                            int(parent) == int(receiver)
                            for parent, role in (
                                node_data.get("parents") or ()
                            )
                            if str(role) in {"value", "object", "operand"}
                        )
                    )
                ]
                return min(candidates) if candidates else None
            return None

        loop_controls: list[tuple[str, int, int | None, bool]] = []
        control_sites: list[tuple[
            int, str, tuple[tuple[int, bool], ...],
            tuple[tuple[int, int], ...], bool,
        ]] = []
        loop_break_sites = dict(
            (graph.G.nodes[int(node_id)].get("attributes") or {})
            .get("loop_break_sites") or {}
        ) if int(node_id) in graph.G else {}
        # A later specialization prunes dead bindings.  A site VALUE that no
        # longer exists names a binding nothing reads, so its pair is
        # dropped.  A pre-loop identity may be pruned while still valid: a
        # parameter's Input node has no consumer once the loop reads only
        # the port, yet the identity table still declares it.
        declared_identity_ids = {
            int(value_id)
            for history in (graph.G.graph.get("identity_table") or {}).values()
            for value_id in history
        }

        def live_site_pair(initial: int, value: int) -> bool:
            return int(value) in graph.G and (
                int(initial) in graph.G or int(initial) in declared_identity_ids
            )
        # (return-value node id, guard chain, per-slot value ids): every
        # ``return <value>`` in this loop's body, whatever its nesting.
        return_controls: list[
            tuple[int, tuple[tuple[int, bool], ...], tuple[int | None, ...]]
        ] = []
        return_slot_values = dict(
            graph.G.graph.get("return_slot_values") or {}
        )

        def terminal_return_expressions(
            statements: Iterable[ast.stmt],
        ) -> set[int]:
            statements = tuple(statements)
            if not statements:
                return set()
            terminal = statements[-1]
            if isinstance(terminal, ast.Return):
                return {id(terminal)}
            if isinstance(terminal, ast.If):
                return {
                    *terminal_return_expressions(terminal.body),
                    *terminal_return_expressions(terminal.orelse),
                }
            return set()

        loop_statements = (
            tuple(expression.body)
            if isinstance(expression, (ast.For, ast.While, ast.AsyncFor))
            else ()
        )
        terminal_return_ids = terminal_return_expressions(loop_statements)
        for candidate in ast.walk(expression):
            if isinstance(candidate, ast.If):
                terminal_return_ids.update(
                    terminal_return_expressions(candidate.body)
                )
                terminal_return_ids.update(
                    terminal_return_expressions(candidate.orelse)
                )

        def collect_loop_controls(
            statements: Iterable[ast.stmt],
            guard: tuple[int, bool] | None = None,
            chain: tuple[tuple[int, bool], ...] = (),
            arm: bool = False,
        ) -> None:
            statements = list(statements)
            for statement in statements:
                if isinstance(statement, (ast.For, ast.While, ast.AsyncFor)):
                    # Its break/continue edges belong to the nested loop.
                    continue
                if isinstance(statement, (ast.Break, ast.Continue)):
                    statement_id = graph_node_for_ast(statement)
                    action = (
                        "break" if isinstance(statement, ast.Break)
                        else "continue"
                    )
                    if statement_id is not None:
                        loop_controls.append((
                            action,
                            statement_id,
                            None if guard is None else guard[0],
                            True if guard is None else guard[1],
                        ))
                        site_values = loop_break_sites.get((
                            int(getattr(statement, "lineno", -1)),
                            int(getattr(statement, "col_offset", -1)),
                            int(getattr(statement, "end_lineno", -1)),
                            int(getattr(statement, "end_col_offset", -1)),
                        )) or {}
                        control_sites.append((
                            int(statement_id),
                            action,
                            tuple(chain),
                            # A later specialization prunes dead bindings;
                            # an id that no longer exists names a value
                            # nothing reads, so its pair is dropped here
                            # (the builder applies the same filter).
                            tuple(sorted(
                                (int(initial), int(value))
                                for initial, value in site_values.items()
                                if live_site_pair(initial, value)
                            )),
                            # The enclosing ``if`` arm's source line span when
                            # this statement ends that arm; whether the arm
                            # OWNS the site is settled at body assembly,
                            # where the body regions are known.
                            (
                                (
                                    min(
                                        int(getattr(item, "lineno", -1))
                                        for item in statements
                                    ),
                                    max(
                                        int(getattr(
                                            item, "end_lineno",
                                            getattr(item, "lineno", -1),
                                        ))
                                        for item in statements
                                    ),
                                )
                                if arm and statement is statements[-1]
                                else None
                            ),
                        ))
                    continue
                if isinstance(statement, ast.Return) and statement.value is not None:
                    return_value_id = graph_node_for_ast(statement.value)
                    returned = statement.value
                    slot_values = return_slot_values.get((
                        int(getattr(returned, "lineno", -1)),
                        int(getattr(returned, "col_offset", -1)),
                        int(getattr(returned, "end_lineno", -1)),
                        int(getattr(returned, "end_col_offset", -1)),
                    ))
                    if return_value_id is not None and slot_values is not None:
                        return_controls.append((
                            int(return_value_id),
                            tuple(chain),
                            tuple(slot_values),
                        ))
                    condition_id = (
                        graph_node_for_ast(expression.test)
                        if isinstance(expression, ast.While) else None
                    )
                    condition_is_true = False
                    if condition_id is not None:
                        known, condition_value = _constant(graph, condition_id)
                        condition_is_true = bool(known and condition_value is True)
                    # A return from an unconditional loop whose value is the
                    # callable's sole public root is a terminal loop exit. It
                    # may branch to the function epilogue without executing a
                    # distinct post-loop authored path.
                    if (
                        condition_is_true
                        and id(statement) in terminal_return_ids
                        and return_value_id is not None
                        and tuple(map(int, graph.roots)) == (int(return_value_id),)
                    ):
                        loop_controls.append((
                            "loop-return",
                            int(return_value_id),
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
                    # ``guard`` keeps the innermost predicate only (the
                    # legacy break/continue contract); ``chain`` keeps every
                    # enclosing predicate outermost-first so a control two
                    # ``if``s deep is guarded by both.
                    true_chain = (
                        chain if predicate_id is None
                        else (*chain, (int(predicate_id), True))
                    )
                    false_chain = (
                        chain if predicate_id is None
                        else (*chain, (int(predicate_id), False))
                    )
                    collect_loop_controls(
                        statement.body, next_true, true_chain, arm=True,
                    )
                    collect_loop_controls(
                        statement.orelse, next_false, false_chain, arm=True,
                    )
                    continue
                for field in ("body", "orelse", "finalbody"):
                    nested = getattr(statement, field, None)
                    if isinstance(nested, list):
                        collect_loop_controls(nested, guard, chain)

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
            # ``graph_node_for_ast(Name)`` may resolve a load in this body to
            # the value defined before the loop.  That value is a captured
            # dependency, not loop-owned work.  Keeping it in ``body_nodes``
            # makes sequential loops which exchange a value appear to overlap
            # lexically and can assign both controls to one numerical region.
            # Retain only nodes whose authored source span is inside the loop
            # body, plus the loop's target bindings (which are synthetic and
            # intentionally have no source span).
            body_lines = tuple(
                (
                    int(getattr(statement, "lineno", -1)),
                    int(getattr(
                        statement,
                        "end_lineno",
                        getattr(statement, "lineno", -1),
                    )),
                )
                for statement in expression.body
                if getattr(statement, "lineno", None) is not None
            )
            target_value_ids = frozenset(map(
                int,
                (attributes.get("loop_target_bindings") or {}).values(),
            ))

            def is_lexical_body_node(candidate: int) -> bool:
                candidate = int(candidate)
                if candidate in target_value_ids:
                    return True
                if candidate not in graph.G:
                    return False
                source_span = graph.G.nodes[candidate].get("source_span") or {}
                line = source_span.get("line")
                if line is None:
                    candidate_expression = graph.G.nodes[candidate].get(
                        "expr_obj"
                    )
                    line = getattr(candidate_expression, "lineno", None)
                if line is None:
                    return False
                line = int(line)
                return any(start <= line <= end for start, end in body_lines)

            # A synthesized effect node (the ``SetAttr`` minted for
            # ``obj.field += v`` in the body) has no authored AST member, so
            # ``graph_node_for_ast`` never proposes it as a candidate -- yet
            # it IS the body's work and carries the writing statement's
            # span. Left unowned, port materialization rewires the loop's
            # own effect chain onto its own result port and closes a cycle
            # through the loop (the 13-node SCC measured by
            # tools/audit_ancestry_retained_loop_graph.py). Admit every
            # node the lexical test itself accepts, not only the ones an
            # AST walk happened to reach.
            body_nodes = tuple(dict.fromkeys((
                *body_nodes,
                *(
                    int(candidate)
                    for candidate, candidate_data in graph.G.nodes(data=True)
                    if candidate_data.get("expr_obj") is None
                    and (candidate_data.get("source_span") or {}).get("line")
                    is not None
                ),
            )))
            body_nodes = tuple(
                candidate
                for candidate in body_nodes
                if is_lexical_body_node(candidate)
            )
        elif not body_nodes and isinstance(expression, ast.comprehension):
            comprehension_owners = tuple(
                successor
                for successor in graph.G.successors(node_id)
                if isinstance(
                    graph.G.nodes[successor].get("expr_obj"),
                    (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
                )
            )
            element_roots = tuple(
                parent
                for successor in comprehension_owners
                for parent, role in (
                    graph.G.nodes[successor].get("parents") or ()
                )
                if str(role) in {"elt", "key", "value"}
            )
            # The element expression *is* the comprehension's body, and a
            # ``for`` statement claims its whole body subtree.  Claiming only
            # the element's root node left every operand below it -- a cast, a
            # mapping lookup, an arithmetic step -- outside the loop, where it
            # bound the target's pre-loop value and the loop's own projected
            # load was left with no consumer.  Walk the authored element and
            # filter conditions the same way ``ast.For`` walks its statements.
            authored_members = [
                member
                for successor in comprehension_owners
                for field in ("elt", "key", "value")
                if (
                    owner_expression := getattr(
                        graph.G.nodes[successor].get("expr_obj"), field, None
                    )
                ) is not None
                for member in ast.walk(owner_expression)
            ]
            authored_members.extend(
                member
                for condition in expression.ifs
                for member in ast.walk(condition)
            )
            element_nodes = tuple(dict.fromkeys(
                member_node
                for member in authored_members
                if (member_node := graph_node_for_ast(member)) is not None
            ))
            # A load resolved inside the element may name a value defined
            # before the comprehension -- the mapping being looked up, a bound
            # constant.  Those are captured dependencies the planner is right
            # to evaluate once.  Only work that reaches the loop's own target
            # is loop-owned, so keep exactly what descends from a target
            # binding, plus the element roots themselves.
            target_value_ids = frozenset(map(
                int,
                (attributes.get("loop_target_bindings") or {}).values(),
            ))
            target_dependents: set[int] = set()
            pending = [
                target_id
                for target_id in target_value_ids
                if target_id in graph.G
            ]
            while pending:
                current = pending.pop()
                for successor in graph.G.successors(current):
                    successor = int(successor)
                    if successor in target_dependents:
                        continue
                    target_dependents.add(successor)
                    pending.append(successor)
            body_nodes = tuple(dict.fromkeys((
                *element_roots,
                *(
                    member
                    for member in element_nodes
                    if int(member) in target_dependents
                ),
            )))
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
        if iterable_node is None and iterator_expression is not None:
            # Lexical reduction can retain the loop and the iterable
            # expression as separate nodes while dropping their redundant raw
            # AST parent edge. Recover the already-ingested value by the same
            # identity/signature rule used for range arguments and loop bodies;
            # this is the authored iterable, not a captured trip-count guess.
            recovered_iterable = graph_node_for_ast(iterator_expression)
            if recovered_iterable is not None and recovered_iterable in graph.G:
                iterable_node = int(recovered_iterable)

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
                dict(graph.G.graph.get("planner_specializations") or {}),
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
                sequence_policy=effect.get("sequence_policy"),
                argument_kind=str(effect.get("argument_kind", "value")),
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
            return_nodes=tuple(
                (node_id, predicate_id, expect_true)
                for kind, node_id, predicate_id, expect_true in loop_controls
                if kind == "loop-return"
            ),
            return_controls=tuple(return_controls),
            control_sites=tuple(control_sites),
            break_bindings=tuple(sorted(
                (str(name), int(binding[0]), int(binding[1]))
                for name, binding in (
                    attributes.get("loop_break_bindings") or {}
                ).items()
                if live_site_pair(binding[0], binding[1])
            )),
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

        # Realization decisions form a hierarchy.  A multi-carried loop is a
        # coordinated recurrence: its whole (initial, updated) vector must
        # advance on one backedge.  That semantic-preservation requirement
        # outranks the optional unroll identity, and it closes over lexical
        # owners because evaporating an owner would also erase its retained
        # child control.  Only loops outside this closure remain eligible for
        # low-priority unrolling.
        nested_loop_ids = {
            int(candidate.loop.node_id)
            for owner in plans
            for candidate in plans
            if (
                int(candidate.loop.node_id) != int(owner.loop.node_id)
                and int(candidate.loop.node_id) in set(map(
                    int, owner.loop.body_nodes,
                ))
            )
        }
        protected = {
            int(plan.loop.node_id)
            for plan in plans
            if (
                len(plan.loop.carried_bindings) > 1
                or (
                    plan.loop.carried_bindings
                    and int(plan.loop.node_id) in nested_loop_ids
                )
            )
        }
        changed = True
        while changed:
            changed = False
            for plan in plans:
                loop_id = int(plan.loop.node_id)
                if loop_id in protected:
                    continue
                if protected.intersection(map(int, plan.loop.body_nodes)):
                    protected.add(loop_id)
                    changed = True

        def preserve(plan: LoopPlan) -> LoopPlan:
            if int(plan.loop.node_id) not in protected:
                return plan
            if (
                plan.loop.iterator_kind == "while"
                and self.capabilities.native_while
            ) or (
                plan.loop.iterator_kind != "while"
                and self.capabilities.native_for
            ):
                strategy = LoopStrategy.NATIVE_SOURCE
            elif self.capabilities.kpn:
                strategy = LoopStrategy.KPN
            else:
                strategy = LoopStrategy.DISPATCH
            return replace(
                plan,
                strategy=strategy,
                reason=(
                    "coordinated multi-carried recurrence preservation "
                    "outranks loop unrolling"
                ),
            )

        return tuple(preserve(plan) for plan in plans)

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
                loop_data = graph.G.nodes[int(loop.node_id)]
                loop_expression = loop_data.get("expr_obj")
                try:
                    source = ast.unparse(loop_expression)
                except Exception:  # noqa: BLE001 -- diagnostic only
                    source = repr(loop_expression)
                raise ValueError(
                    f"iterable loop {loop.node_id} has no iterable value; "
                    f"source={source!r}; "
                    f"parents={tuple(loop_data.get('parents') or ())!r}; "
                    f"attributes={dict(loop_data.get('attributes') or {})!r}"
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
                ) + tuple(
                    predicate_id
                    for _node_id, predicate_id, _expect_true
                    in loop.return_nodes
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

    plans = tuple(plans)
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
    nodes_by_value = {
        int(data.get("value_id", node_id)): data
        for node_id, data in graph.G.nodes(data=True)
    }
    node_ids_by_value = {
        int(data.get("value_id", node_id)): int(node_id)
        for node_id, data in graph.G.nodes(data=True)
    }

    def expanded_row_arguments(value_ids: Iterable[int]) -> tuple[int, ...]:
        """Expose fixed aggregate leaves as sequence row columns."""

        expanded: list[int] = []
        for value_id in map(int, value_ids):
            data = nodes_by_value.get(value_id, {})
            attributes = data.get("attributes") or {}
            leaf_ids = tuple(map(
                int, attributes.get("aggregate_leaf_value_ids", ())
            ))
            if attributes.get("aggregate_kind") == "tuple" and leaf_ids:
                expanded.extend(leaf_ids)
            else:
                expanded.append(value_id)
        return tuple(expanded)
    planner_constants = dict(
        graph.G.graph.get("planner_specializations") or {}
    )

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

    carried_initial_value_ids = {
        int(initial)
        for planned_loop in plans
        for _name, initial, updated in planned_loop.loop.carried_bindings
        if int(initial) != int(updated)
    }

    def structured_control_expression(
        node_id: int,
        visiting: frozenset[int] = frozenset(),
    ) -> ControlExpression | None:
        node_id = int(node_id)
        if node_id in visiting:
            return ControlExpression("value", value_id=node_id)
        # State-effect and aggregate records speak in deterministic SSA value
        # identities, while loop/control membership speaks in graph node
        # identities.  They often coincide, but topology reduction may remove
        # the original graph node while retaining its canonical value on a
        # replacement node. Resolve through the graph's explicit ``value_id``
        # correlation; never treat the old integer as an alias or remint it.
        resolved_node_id = (
            node_id
            if node_id in graph.G
            else node_ids_by_value.get(node_id)
        )
        if resolved_node_id is None:
            return ControlExpression("value", value_id=node_id)
        data = graph.G.nodes[int(resolved_node_id)]
        value_id = int(data.get("value_id", node_id))
        if value_id in carried_initial_value_ids:
            return ControlExpression("value", value_id=value_id)
        known, literal = _constant(graph, int(resolved_node_id))
        if known and isinstance(literal, (bool, int, float)):
            return ControlExpression(
                "const", value_id=value_id, literal=literal
            )
        expression = data.get("expr_obj")
        op = str(data.get("op") or data.get("type") or "").lower()
        parents = tuple(data.get("parents") or ())
        attributes = data.get("attributes") or {}
        if (
            attributes.get("producer_kind")
            in {"aggregate", "aggregate_materialization", "loop_materialization"}
        ):
            # Python aggregate truth is a storage query, not the aggregate
            # pointer re-used as a boolean value.  Retain it as explicit
            # control IR so the SSA lowerer reloads the mutable length at the
            # latch and produces the next while predicate.
            return ControlExpression(
                "sequence_nonempty",
                (ControlExpression("value", value_id=value_id),),
                value_id=value_id,
                literal=attributes.get("aggregate_kind") in {"dict", "set"},
            )
        if data.get("type") == "Input":
            specialized = specialized_input_value(data)
            if specialized is not None:
                return ControlExpression(
                    "const", value_id=value_id, literal=specialized
                )
            return ControlExpression("value", value_id=value_id)
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
                    operator_name, (result, value), value_id=value_id
                )
            return result
        operation = {
            "add": "add", "sub": "sub", "mul": "mul",
            "div": "div", "truediv": "div",
            "less": "lt", "lt": "lt",
            "lessequal": "le", "less_equal": "le", "le": "le",
            "greater": "gt", "gt": "gt",
            "greaterequal": "ge", "greater_equal": "ge", "ge": "ge",
            "equal": "eq", "eq": "eq",
            "notequal": "ne", "not_equal": "ne", "ne": "ne",
            "logical_and": "and", "land": "and",
            "logical_or": "or", "lor": "or",
            "logical_not": "not", "lnot": "not",
            "neg": "neg", "usub": "neg",
            "bitand": "bitand", "bitor": "bitor",
            "bitxor": "bitxor", "shl": "shl", "shr": "shr",
            "item": "item", "float": "float",
            "int": "int", "bool": "bool",
        }.get(op)
        if operation is None and isinstance(expression, ast.Call):
            if isinstance(expression.func, ast.Name):
                operation = {
                    "float": "float", "int": "int", "bool": "bool"
                }.get(expression.func.id)
        if operation is None:
            return ControlExpression("value", value_id=value_id)
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
            operation, operands[:arity], value_id=value_id
        )

    plans_by_loop_id = {
        int(plan.loop.node_id): plan for plan in plans
    }

    def expanded_loop_body_nodes(
        loop: LoopDescriptor,
        visiting: frozenset[int] = frozenset(),
    ) -> tuple[int, ...]:
        """Include the authored work owned by structurally nested loops.

        Statement loops already receive their complete AST subtree from
        discovery.  A comprehension is different: each ``generator`` clause
        is a sibling field of the materializer AST, and a nested collection
        comprehension inside its element is represented by its own loop
        descriptor.  The outer descriptor therefore contains the child loop
        node but not necessarily the child's numerical body nodes.  Region
        compartmentalization then gave the outer control only its local
        markers while the hierarchy correctly declared the child nested,
        leaving no lexical marker at which overlay could insert it.

        Expand through the compiler's authored loop-containment relation,
        retaining encounter order.  This is identity correlation, not a
        region-number offset: the child loop's real graph node is the proof
        of containment and its own descriptor supplies the exact body IDs.
        """

        loop_id = int(loop.node_id)
        if loop_id in visiting:
            raise ValueError("cyclic authored loop containment")
        expanded: list[int] = []
        for member in map(int, loop.body_nodes):
            expanded.append(member)
            child = plans_by_loop_id.get(member)
            if child is not None:
                expanded.extend(expanded_loop_body_nodes(
                    child.loop, visiting | {loop_id},
                ))
        return tuple(dict.fromkeys(expanded))

    routed_generator_mutations: dict[
        int, list[ControlSequenceMutation]
    ] = {}
    routed_generator_outputs: dict[int, set[int]] = {}
    routed_effect_nodes: set[int] = set()
    for consumer_plan in plans:
        for effect in consumer_plan.loop.state_effects:
            if (
                effect.mode is not LoopStateEffectMode.SEQUENCE_MUTATION
                or effect.argument_kind not in {"generator", "filtered_sequence"}
                or len(effect.argument_value_ids) != 1
                or effect.sequence_policy is None
            ):
                continue
            materializer_id = int(effect.argument_value_ids[0])
            if materializer_id not in graph.G:
                continue
            materializer = graph.G.nodes[materializer_id]
            expression = materializer.get("expr_obj")
            if not isinstance(expression, (ast.GeneratorExp, ast.ListComp)):
                continue
            generators = tuple(expression.generators)
            # A nested generator/comprehension is several dependent retained loops.  Route
            # it only once hierarchy composition has an explicit multi-loop
            # yield edge; keeping the typed iterator shortfall is safer than
            # flattening or materializing it here.
            if len(generators) != 1:
                continue
            generator_loop_id = next((
                int(parent)
                for parent, role in (materializer.get("parents") or ())
                if str(role).startswith("generators")
                and int(parent) in plans_by_loop_id
            ), None)
            if generator_loop_id is None:
                continue
            producer_plan = plans_by_loop_id[generator_loop_id]
            yielded_value_id = next((
                int(parent)
                for parent, role in (materializer.get("parents") or ())
                if str(role) == "elt"
            ), None)
            if yielded_value_id is None:
                continue
            predicates = tuple(
                structured_control_expression(int(value_id))
                for value_id in producer_plan.loop.condition_nodes
            )
            if any(predicate is None for predicate in predicates):
                continue
            predicate_expression = None
            for predicate in predicates:
                predicate_expression = (
                    predicate
                    if predicate_expression is None
                    else ControlExpression(
                        "and", (predicate_expression, predicate)
                    )
                )
            yielded_arguments = expanded_row_arguments((yielded_value_id,))
            routed_generator_mutations.setdefault(
                generator_loop_id, []
            ).append(ControlSequenceMutation(
                sequence_value_id=int(effect.state_input_id),
                operator=(
                    "add"
                    if effect.sequence_policy == "unique" else "append"
                ),
                argument_value_ids=yielded_arguments,
                effect_node_id=int(effect.effect_node_id),
                policy=effect.sequence_policy,
                argument_kind="value",
                predicate_expression=predicate_expression,
            ))
            routed_generator_outputs.setdefault(
                generator_loop_id, set()
            ).add(materializer_id)
            routed_effect_nodes.add(int(effect.effect_node_id))

    for plan in plans:
        loop = plan.loop
        if int(loop.node_id) in routed_generator_outputs:
            routed_outputs = routed_generator_outputs[int(loop.node_id)]
            loop = replace(
                loop,
                iteration_outputs=tuple(
                    output
                    for output in loop.iteration_outputs
                    if (
                        int(output.result_value_id) not in routed_outputs
                        and int(output.materializer_node_id)
                        not in routed_outputs
                    )
                ),
            )
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
        # A range loop's lexical target is the induction value itself.  Record
        # that identity through the existing projected-iterable binding table
        # so region calls consuming the source target ID receive the loop
        # Phi, rather than accidentally promoting that ID to a function
        # argument.  ``iterable_id`` is provenance only for an ``induction``
        # projection; no collection load is emitted.
        projected_iterable_bindings = (
            (
                int(loop.node_id),
                int(loop.target_bindings[0][1]),
                induction_name,
                "induction",
            ),
        ) if (
            loop.iterator_kind == "arithmetic_sequence"
            and len(loop.target_bindings) == 1
        ) else ()
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
        nested_body_nodes = expanded_loop_body_nodes(loop)
        body = set(nested_body_nodes)
        expression_nodes = {
            id(data.get("expr_obj")): int(node_id)
            for node_id, data in graph.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.AST)
        }

        def expression_signature(expression: ast.AST) -> tuple[Any, ...]:
            return (
                type(expression),
                getattr(expression, "lineno", None),
                getattr(expression, "col_offset", None),
                getattr(expression, "end_lineno", None),
                getattr(expression, "end_col_offset", None),
                ast.dump(expression, include_attributes=False),
            )

        signature_nodes = {
            expression_signature(data["expr_obj"]): int(node_id)
            for node_id, data in graph.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.AST)
        }

        def node_for_expression(expression: ast.AST) -> int | None:
            return expression_nodes.get(id(expression), signature_nodes.get(
                expression_signature(expression)
            ))

        # ``expanded_loop_body_nodes`` is a graph expansion, not a lexical
        # traversal.  In particular, an expression nested below an assignment
        # can appear after the following ``continue`` node.  Sorting region
        # markers against loop-control edges with that order places the
        # calculation in an unreachable block after the terminator.  Derive
        # the primary rank from the authored AST: all descendants of one
        # statement execute before the next statement begins.  Keep graph
        # expansion only as a deterministic fallback for synthesized nodes
        # which have no correlated source expression.
        lexical_nodes: list[int] = []
        loop_expression = graph.G.nodes[int(loop.node_id)].get("expr_obj")
        def source_order_walk(node: ast.AST):
            """Pre-order: a statement's descendants before its successor.

            ``ast.walk`` is breadth-first, so an arm's ``Break`` node sorted
            BEFORE the arm's own assignment expressions and the exit edge
            was taken before the value it carries was computed.
            """

            yield node
            for child in ast.iter_child_nodes(node):
                yield from source_order_walk(child)

        if isinstance(loop_expression, (ast.For, ast.AsyncFor, ast.While)):
            for statement in loop_expression.body:
                for expression in source_order_walk(statement):
                    node_id = node_for_expression(expression)
                    if (
                        node_id is not None
                        and int(node_id) in body
                        and int(node_id) not in lexical_nodes
                    ):
                        lexical_nodes.append(int(node_id))
        lexical_nodes.extend(
            int(node_id)
            for node_id in nested_body_nodes
            if int(node_id) not in lexical_nodes
        )
        lexical_position = {
            node_id: position for position, node_id in enumerate(lexical_nodes)
        }

        # A guard whose only true-arm action is raise is compiled validation,
        # including when it is lexically inside a retained loop.  Record it at
        # its source position so it runs on every iteration, not once after the
        # loop.  Other Raise shapes remain blockers.
        #
        # The mirror shape -- `if cond: body else: raise` -- is the same
        # guard clause with the branches swapped: the loop's real work is
        # the non-raising arm, reached only once the raise is proven
        # unreachable this iteration.  The reducer no longer merges a Phi
        # across an if where exactly one arm is a dead end (see the
        # terminal-branch skip in topological_reducer.py's ast.If
        # handling), so validating the raise here is enough on its own to
        # make the whole statement compile.
        validations: list[tuple[int, int, bool]] = []
        validated_raise_signatures: set[tuple[Any, ...]] = set()
        for node_id in loop.body_nodes:
            if node_id not in graph.G:
                continue
            statement = graph.G.nodes[node_id].get("expr_obj")
            if not isinstance(statement, ast.If):
                continue
            # A validation arm may prepare diagnostics before its terminal
            # raise.  Requiring every statement in the arm to be ``Raise``
            # rejects ordinary authored guards such as ``if failed: log();
            # raise RuntimeError(...)`` once they live in a retained loop.
            # The preceding statements remain owned by the conditional; only
            # the terminal transfer becomes the validation edge.
            body_is_raise = bool(statement.body) and isinstance(
                statement.body[-1], ast.Raise
            )
            orelse_is_raise = bool(statement.orelse) and isinstance(
                statement.orelse[-1], ast.Raise
            )
            if body_is_raise and not statement.orelse:
                raise_items = (statement.body[-1],)
                raises_when_true = True
            elif orelse_is_raise and not body_is_raise:
                raise_items = (statement.orelse[-1],)
                raises_when_true = False
            else:
                continue
            test = statement.test
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                test = test.operand
                raises_when_true = not raises_when_true
            predicate_id = node_for_expression(test)
            if predicate_id is None:
                continue
            raise_node_id = node_for_expression(raise_items[0])
            if raise_node_id is None:
                continue
            validations.append((
                int(raise_node_id),
                int(predicate_id),
                not raises_when_true,
            ))
            validated_raise_signatures.update(
                expression_signature(item) for item in raise_items
            )
        def earliest_member(index: int) -> int:
            return min(
                lexical_position[node_id]
                for node_id in regions[index]
                if node_id in lexical_position
            )

        body_region_indices = tuple(sorted(
            (
            index
            for index, nodes in enumerate(regions)
            if body.intersection(nodes)
            ),
            key=earliest_member,
        ))
        # A region's earliest member says when it *may* start, not when it may
        # run: the same region also holds later members whose operands another
        # region produces.  Ordering by the earliest member alone put a fused
        # cast-and-compare ahead of the region computing the compare's other
        # operand, so the compare read that operand's pre-loop value while the
        # real one was versioned into a value nothing read -- a use before
        # definition that every backend emitted without complaint.  Order the
        # body over the regions' own dependency graph instead, and keep each
        # region as early as its dependencies allow.
        body_region_owner = {
            int(node_id): index
            for index in body_region_indices
            for node_id in regions[index]
        }
        # The same recursion-table control set the fusion reducer and
        # `_topological_region_order` discount: the loop's own While /
        # LoopStatePort / LoopResult / LoopExit nodes, through which raw
        # ancestry wraps around to the previous iteration.
        recursion_control_members = frozenset(
            int(member)
            for record in (graph.G.graph.get("recursion_table") or {}).values()
            if record.get("control_ir", True)
            for member in record.get("control_members", ())
        )
        region_dependencies = nx.DiGraph()
        region_dependencies.add_nodes_from(body_region_indices)
        for index in body_region_indices:
            for node_id in regions[index]:
                if int(node_id) not in graph.G:
                    continue
                # A dependency may pass THROUGH nodes no region owns -- a
                # source-linked call mediating region_1 -> stencil ->
                # region_3 is invisible to a direct parent check, which left
                # the accumulator free to schedule before the stencil whose
                # outputs it consumes.  Project each path onto its nearest
                # region-owned ancestors, exactly as the fusion reducer
                # projects execution edges through structural nodes.
                pending = [
                    int(parent)
                    for parent, _role in (
                        graph.G.nodes[int(node_id)].get("parents") or ()
                    )
                ]
                seen: set[int] = set()
                while pending:
                    parent = pending.pop()
                    if parent in seen or parent not in graph.G:
                        continue
                    seen.add(parent)
                    if parent in recursion_control_members:
                        # Walking through the loop's own control/port nodes
                        # (While, LoopStatePort, LoopResult, LoopExit) wraps
                        # around to the previous iteration and reports every
                        # region as depending on every other. The fusion
                        # reducer discounts exactly these nodes before its
                        # own acyclicity check; do the same here, so the
                        # condensation fallback below is reserved for
                        # genuine same-iteration recursion instead of being
                        # the normal path for any loop with a state effect.
                        continue
                    producer = body_region_owner.get(parent)
                    if producer is not None:
                        if producer != index:
                            region_dependencies.add_edge(producer, index)
                        continue
                    pending.extend(
                        int(grandparent)
                        for grandparent, _role in (
                            graph.G.nodes[parent].get("parents") or ()
                        )
                    )
        try:
            body_region_indices = tuple(nx.lexicographical_topological_sort(
                region_dependencies, key=earliest_member,
            ))
        except nx.NetworkXUnfeasible:
            # A retained loop deliberately has feedback: a region may consume
            # another's loop-carried value from the previous iteration.  That
            # is irreducible recursion, not an ordering error -- the same
            # verdict _rebuild_graph_edges already reaches for the node
            # graph.  Order the condensation; mutually-recursive regions
            # share a rank and fall to the earliest-member tie-break.
            condensed = nx.condensation(region_dependencies)
            body_region_indices = tuple(
                member
                for component in nx.lexicographical_topological_sort(
                    condensed,
                    key=lambda component: min(
                        earliest_member(member)
                        for member in condensed.nodes[component]["members"]
                    ),
                )
                for member in sorted(
                    condensed.nodes[component]["members"],
                    key=earliest_member,
                )
            )
        body_region_positions = {}
        for index in body_region_indices:
            body_region_positions[index] = max((
                earliest_member(index),
                *(
                    body_region_positions[producer]
                    for producer in region_dependencies.predecessors(index)
                    # A feedback predecessor (same strongly connected
                    # component, later in the condensed order) constrains the
                    # NEXT iteration, not this statement's position.
                    if producer in body_region_positions
                ),
            ))
        condition = set(map(int, loop.condition_nodes))
        structurally_owned_region_indices: tuple[int, ...] = ()
        if loop.iteration_outputs and condition:
            # A retained comprehension publishes through a predicated append
            # below. Its filter is evaluated inside the loop from the loaded
            # target row. Leaving the same predicate region in the flat
            # schedule executes it outside the loop and can give a later
            # reduction/validation false ownership of the loop's region.
            def expression_value_ids(expression: ControlExpression | None):
                if expression is None:
                    return frozenset()
                found = {
                    int(expression.value_id)
                    for _ in (0,)
                    if expression.value_id is not None
                }
                for operand in expression.operands:
                    found.update(expression_value_ids(operand))
                return frozenset(found)

            structurally_owned_values = frozenset().union(*(
                expression_value_ids(structured_control_expression(node_id))
                for node_id in sorted(condition)
            ))
            predicate_regions = {
                index
                for index, nodes in enumerate(regions)
                if condition.intersection(map(int, nodes))
                and frozenset(map(int, nodes)).issubset(
                    structurally_owned_values
                )
            }
            structurally_owned_region_indices = tuple(sorted(predicate_regions))
            body_region_indices = tuple(
                index for index in body_region_indices
                if index not in predicate_regions
            )
            body_region_positions = {
                index: position
                for index, position in body_region_positions.items()
                if index in body_region_indices
            }
        import os as _os, sys as _sys
        if _os.environ.get("TURING_DEBUG_LOOP_ORDER"):
            _fn = graph.G.graph.get("function_name")
            _loop_expr = graph.G.nodes[int(loop.node_id)].get("expr_obj")
            print(
                f"DEBUGLOOP fn={_fn} loop_node={int(loop.node_id)} "
                f"source={ast.unparse(_loop_expr) if isinstance(_loop_expr, ast.AST) else None!r} "
                f"body_region_indices={body_region_indices} "
                f"body_region_positions={body_region_positions} "
                f"region_dependencies_edges={list(region_dependencies.edges())} "
                f"region_members={ {i: tuple(regions[i]) for i in body_region_indices} }",
                file=_sys.stderr,
            )
        condition_region_indices = tuple(
            index
            for index, nodes in enumerate(regions)
            if condition.intersection(nodes)
            and index not in body_region_indices
            and not loop.iteration_outputs
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
        sequence_mutations = []
        expression_nodes = {
            id(data.get("expr_obj")): int(node_id)
            for node_id, data in graph.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.AST)
        }
        branch_memberships: dict[int, set[tuple[int, str]]] = {}
        def arm_has_terminal_return(statements: Iterable[ast.stmt]) -> bool:
            statements = tuple(statements)
            if not statements:
                return False
            terminal = statements[-1]
            if isinstance(terminal, ast.Return):
                return True
            if isinstance(terminal, ast.If):
                return (
                    arm_has_terminal_return(terminal.body)
                    or arm_has_terminal_return(terminal.orelse)
                )
            return False
        for owner_id, owner in graph.G.nodes(data=True):
            conditional = owner.get("expr_obj")
            if not isinstance(conditional, ast.If):
                continue
            for arm, statements in (
                ("body", conditional.body),
                ("orelse", conditional.orelse),
            ):
                for statement in statements:
                    for member in ast.walk(statement):
                        member_id = expression_nodes.get(id(member))
                        if member_id is not None and member_id != int(owner_id):
                            branch_memberships.setdefault(
                                member_id, set()
                            ).add((int(owner_id), arm))
        # A terminal arm guards the lexical fallthrough just as surely as an
        # explicit ``else``.  In ``if done: return; out.append(x)``, the append
        # is semantically in the false arm even though Python source does not
        # indent it there.  Record that control fact so terminal returns can be
        # moved after predicated resident effects without executing fallthrough
        # work on the returning path.
        if isinstance(graph.G.nodes[int(loop.node_id)].get("expr_obj"), ast.While):
            loop_expression = graph.G.nodes[int(loop.node_id)]["expr_obj"]
            for position, statement in enumerate(loop_expression.body):
                if not isinstance(statement, ast.If):
                    continue
                body_terminal = arm_has_terminal_return(statement.body)
                orelse_terminal = arm_has_terminal_return(statement.orelse)
                if body_terminal == orelse_terminal:
                    continue
                owner_id = node_for_expression(statement)
                if owner_id is None:
                    continue
                surviving_arm = "orelse" if body_terminal else "body"
                for later in loop_expression.body[position + 1:]:
                    for member in ast.walk(later):
                        member_id = expression_nodes.get(id(member))
                        if member_id is not None:
                            branch_memberships.setdefault(
                                int(member_id), set()
                            ).add((int(owner_id), surviving_arm))

        def mutation_predicate(effect_node_id: int):
            candidates = []
            for owner_id, arm in branch_memberships.get(
                int(effect_node_id), ()
            ):
                if int(owner_id) not in loop.body_nodes:
                    continue
                owner = graph.G.nodes.get(int(owner_id), {})
                if not isinstance(owner.get("expr_obj"), ast.If):
                    continue
                predicate_id = next((
                    int(parent)
                    for parent, role in owner.get("parents") or ()
                    if str(role) == "test"
                ), None)
                if predicate_id is not None:
                    candidates.append((int(owner_id), str(arm), predicate_id))
            if not candidates:
                return None
            predicates = []
            for _owner_id, arm, predicate_id in sorted(candidates):
                predicate = structured_control_expression(int(predicate_id))
                if predicate is None:
                    continue
                predicates.append(
                    predicate
                    if arm == "body"
                    else ControlExpression("not", (predicate,))
                )
            result = predicates[0]
            for predicate in predicates[1:]:
                result = ControlExpression("and", (result, predicate))
            return result

        for effect in loop.state_effects:
            if effect.mode not in {
                LoopStateEffectMode.SEQUENCE_MUTATION,
                LoopStateEffectMode.MAPPING_MUTATION,
            }:
                continue
            if int(effect.effect_node_id) in routed_effect_nodes:
                continue
            policy = effect.sequence_policy
            state_node = (
                graph.G.nodes[int(effect.state_input_id)]
                if int(effect.state_input_id) in graph.G else {}
            )
            state_expression = state_node.get("expr_obj")
            state_type = str(state_node.get("type") or "").lower()
            if policy is None and (
                isinstance(state_expression, ast.List)
                or state_type in {"list", "tuple"}
            ):
                policy = "duplicates"
            if policy is None and (
                isinstance(state_expression, ast.Set)
                or state_type == "set"
                or (
                    isinstance(state_expression, ast.Call)
                    and isinstance(state_expression.func, ast.Name)
                    and state_expression.func.id == "set"
                )
            ):
                policy = "unique"
            argument_kind = effect.argument_kind
            if (
                effect.operator == "extend"
                and effect.argument_value_ids
                and argument_kind == "value"
            ):
                argument_node = graph.G.nodes.get(
                    int(effect.argument_value_ids[0]), {}
                )
                argument_expression = argument_node.get("expr_obj")
                argument_kind = (
                    "generator"
                    if isinstance(argument_expression, ast.GeneratorExp)
                    else (
                        "filtered_sequence"
                        if isinstance(argument_expression, ast.ListComp)
                        and any(
                            generator.ifs
                            for generator in argument_expression.generators
                        )
                        else "sequence"
                    )
                )
            mutation_arguments = expanded_row_arguments(
                effect.argument_value_ids
            )
            if (
                effect.mode is LoopStateEffectMode.MAPPING_MUTATION
                and effect.operator == "update"
                and len(effect.argument_value_ids) == 1
            ):
                mapping_node = nodes_by_value.get(
                    int(effect.argument_value_ids[0]), {}
                )
                mapping_attributes = mapping_node.get("attributes") or {}
                leaves = tuple(map(
                    int,
                    mapping_attributes.get("aggregate_leaf_value_ids", ()),
                ))
                if mapping_attributes.get("aggregate_kind") == "dict":
                    midpoint = len(leaves) // 2
                    mutation_arguments = tuple(
                        value_id
                        for pair in zip(
                            leaves[:midpoint], leaves[midpoint:]
                        )
                        for value_id in pair
                    )
            mapping_argument_kind = None
            if effect.mode is LoopStateEffectMode.MAPPING_MUTATION:
                mapping_argument_kind = f"mapping_{effect.operator}"
                if effect.operator == "update":
                    mapping_argument_kind = "mapping_items"
                elif (
                    effect.operator == "pop"
                    and len(effect.argument_value_ids) == 2
                ):
                    default_node_id = node_ids_by_value.get(
                        int(effect.argument_value_ids[1])
                    )
                    default_known, default_literal = (
                        (False, None)
                        if default_node_id is None
                        else _constant(graph, int(default_node_id))
                    )
                    if default_known and default_literal is None:
                        mapping_argument_kind = "mapping_pop_default_none"
            sequence_mutations.append(ControlSequenceMutation(
                sequence_value_id=int(effect.state_input_id),
                operator=str(effect.operator),
                argument_value_ids=mutation_arguments,
                effect_node_id=int(effect.effect_node_id),
                policy=policy,
                argument_kind=(mapping_argument_kind or argument_kind),
                predicate_expression=mutation_predicate(
                    int(effect.effect_node_id)
                ),
                argument_expressions=tuple(
                    structured_control_expression(int(value_id))
                    for value_id in mutation_arguments
                ),
                extraction_identity=(
                    (graph.G.nodes[int(effect.effect_node_id)].get(
                        "attributes"
                    ) or {}).get("extraction_identity")
                    if int(effect.effect_node_id) in graph.G else None
                ),
            ))
        sequence_mutations.extend(
            routed_generator_mutations.get(int(loop.node_id), ())
        )
        # A retained comprehension is a compacting sequence producer, not an
        # induction-indexed array write.  In particular, a filtered generator
        # must advance its output length only for accepted elements.  Publish
        # every iteration output through the same append ABI used by authored
        # lists, with the comprehension predicates attached to the effect.
        output_predicate = None
        for condition_id in loop.condition_nodes:
            predicate = (
                structured_control_expression(int(condition_id))
                or ControlExpression("value", value_id=int(condition_id))
            )
            output_predicate = (
                predicate
                if output_predicate is None
                else ControlExpression("and", (output_predicate, predicate))
            )
        sequence_mutations.extend(
            ControlSequenceMutation(
                sequence_value_id=int(output.result_value_id),
                operator="append",
                argument_value_ids=expanded_row_arguments((output.value_id,)),
                effect_node_id=int(output.materializer_node_id),
                policy="duplicates",
                argument_kind="value",
                predicate_expression=output_predicate,
                argument_expressions=tuple(
                    structured_control_expression(int(value_id))
                    for value_id in expanded_row_arguments((output.value_id,))
                ),
            )
            for output in loop.iteration_outputs
        )
        sequence_mutations = tuple(sequence_mutations)
        represented_effect_nodes = {
            int(mutation.effect_node_id) for mutation in sequence_mutations
        }
        represented_effect_nodes.update(
            int(node_id)
            for controls in (
                loop.break_nodes, loop.continue_nodes, loop.return_nodes,
            )
            for node_id, _predicate_id, _expect_true in controls
        )

        def statement_is_specialized(statement: ast.stmt) -> bool:
            # A nested conditional requires its own composed predicate; its
            # descendants cannot make the outer branch complete by accident.
            if isinstance(statement, ast.If):
                return False
            descendant_ids = {
                int(node_id)
                for member in ast.walk(statement)
                for node_id in (node_for_expression(member),)
                if node_id is not None
            }
            return bool(descendant_ids.intersection(represented_effect_nodes))

        specialized_conditional_node_ids = tuple(
            int(node_id)
            for node_id in loop.body_nodes
            if node_id in graph.G
            and isinstance(graph.G.nodes[node_id].get("expr_obj"), ast.If)
            and all(
                statement_is_specialized(statement)
                for statement in (
                    *graph.G.nodes[node_id]["expr_obj"].body,
                    *graph.G.nodes[node_id]["expr_obj"].orelse,
                )
            )
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
        # A resident sequence mutation or an internally source-linked call is
        # itself compiled loop work.  The call body is linked after control SSA
        # owns the lexical insertion point, but the loop must survive now so
        # that later linking cannot hoist it to the function boundary.  Merely
        # spelling ``Call`` is insufficient: only a resolved function/method/
        # constructor reference proves an internal authored call.
        source_linked_calls = tuple(
            int(node_id)
            for node_id in loop.body_nodes
            if node_id in graph.G
            and any(
                (graph.G.nodes[node_id].get("attributes") or {}).get(key)
                is not None
                for key in (
                    "callee_ref", "method_ref", "constructor_ref", "class_ref"
                )
            )
        )
        if not region_indices and not sequence_mutations and not source_linked_calls:
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
            if (
                isinstance(expression, ast.Raise)
                and expression_signature(expression)
                in validated_raise_signatures
            ):
                continue
            if isinstance(expression, forbidden):
                blockers.append(type(expression).__name__)
        blockers = list(dict.fromkeys(blockers))

        # ``identity_table`` groups every value ever correlated with a source
        # name.  That class is scope-free: for one carried name it holds the
        # caller result, a local copy, an IndexedStore, the loop's LoopResult
        # port and a nested callee's parameter alongside the real body update.
        # Offering all of them as backedge candidates emits several pairs per
        # binding, and only the reducer's lexical update is the authoritative
        # version written by this cycle.  Use exactly that version here.
        carried_aliases = tuple(dict.fromkeys(
            (
                int(updated),
                int(initial),
            )
            for _name, initial, updated in loop.carried_bindings
            if int(updated) != int(initial)
            # Callsite structural specialization can prove the only branch
            # containing this update unreachable. The carried identity then
            # has no backedge value in the specialized graph.
            and int(updated) in graph.G
            # An IndexedStore is the write instruction for resident table
            # storage, not a newly produced table value.  The table pointer
            # remains the same across iterations; requiring this instruction
            # as a Phi backedge invents a value result that Store cannot and
            # should not produce.
            and graph.G.nodes[int(updated)].get("type") != "IndexedStore"
        ))
        def guarded_expression(
            chain: tuple[tuple[int, bool], ...],
        ) -> ControlExpression | None:
            """Conjoin every enclosing predicate, outermost first."""

            combined: ControlExpression | None = None
            for predicate_value_id, expect_true in chain:
                term = structured_control_expression(int(predicate_value_id))
                if term is None:
                    term = ControlExpression(
                        "value", value_id=int(predicate_value_id)
                    )
                if not expect_true:
                    term = ControlExpression("not", (term,))
                combined = (
                    term if combined is None
                    else ControlExpression("and", (combined, term))
                )
            return combined

        body_items: list[tuple[int, object]] = [
            (
                body_region_positions[region_index],
                StatementBlock((f"__scheduled_region_{region_index}__",)),
            )
            for region_index in body_region_indices
        ]
        # A break/continue ending an ``if`` arm is placed inside that arm by
        # the conditional program (the arm's values dominate the edge there);
        # every other site is placed here at its lexical position, guarded
        # by EVERY enclosing predicate (a site two ``if``s deep must not
        # fire when the outer predicate is false).
        body_region_nodes = {
            int(member)
            for index in body_region_indices
            for member in regions[index]
        }

        def node_line(value_id: int) -> int | None:
            data = graph.G.nodes.get(int(value_id)) or {}
            line = (data.get("source_span") or {}).get("line")
            if line is None:
                line = getattr(data.get("expr_obj"), "lineno", None)
            return None if line is None else int(line)

        def arm_owned_site(
            arm_span: tuple[int, int] | None,
            site_values: tuple[tuple[int, int], ...],
        ) -> bool:
            if arm_span is None:
                return False
            first, last = arm_span
            owned = False
            for _initial, value in site_values:
                if int(value) not in body_region_nodes:
                    continue
                line = node_line(value)
                if line is not None and first <= line <= last:
                    owned = True
                    break
            if os.environ.get("TURING_DEBUG_BREAK_EDGE"):
                print(
                    "DEBUG-ARM-OWNED composer "
                    f"fn={graph.G.graph.get('function_name')} "
                    f"loop={loop.node_id} arm_span={arm_span} "
                    f"site_values={site_values} owned={owned}",
                    file=sys.stderr, flush=True,
                )
            return owned

        body_items.extend(
            (
                lexical_position[site_id],
                LoopControlBlock(
                    action,
                    chain[-1][0] if chain else None,
                    True,
                    guarded_expression(chain),
                    source_action=action,
                    site_node_id=int(site_id),
                    site_values=tuple(site_values),
                ),
            )
            for site_id, action, chain, site_values, arm_span
            in loop.control_sites
            if site_id in lexical_position
            and not arm_owned_site(arm_span, site_values)
        )


        # A source ``return`` inside the body is a loop exit that leaves the
        # FUNCTION, carrying its own slot values: placed at its lexical
        # position (like break/continue) so the non-returning path's later
        # work never runs on the returning path, and guarded by every
        # enclosing predicate rather than the innermost only.
        body_items.extend(
            (
                lexical_position[node_id],
                LoopControlBlock(
                    "return",
                    chain[-1][0] if chain else None,
                    True,
                    guarded_expression(chain),
                    source_action="return",
                    return_value_ids=tuple(slot_values),
                ),
            )
            for node_id, chain, slot_values in loop.return_controls
            if node_id in lexical_position
        )
        body_items.extend(
            (
                lexical_position[node_id],
                ValidationBlock(
                    predicate_value_id,
                    error_code=node_id,
                    expect_true=expect_true,
                    predicate_expression=structured_control_expression(
                        predicate_value_id
                    ),
                    extraction_identity=(
                        f"builtins.{expression.exc.func.id}"
                        if isinstance((expression := graph.G.nodes[node_id].get(
                            "expr_obj"
                        )), ast.Raise)
                        and isinstance(expression.exc, ast.Call)
                        and isinstance(expression.exc.func, ast.Name)
                        else None
                    ),
                ),
            )
            for node_id, predicate_value_id, expect_true in validations
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
        terminal_controls = tuple(
            LoopControlBlock(
                "break",
                predicate_value_id,
                expect_true,
                (
                    None if predicate_value_id is None
                    else structured_control_expression(predicate_value_id)
                ),
                source_action="loop-return",
            )
            for _node_id, predicate_value_id, expect_true in loop.return_nodes
        )

        def planned_root():
            # The materialized ports (name -> LoopResult id) are the graph's
            # own statement of what each carried name means after the loop;
            # pair them with the discovery's (name, initial, updated) so the
            # SSA lowering can bind each port to its carried Phi's exit.
            loop_ports = dict(
                (
                    graph.G.nodes[int(loop.node_id)].get("attributes") or {}
                ).get("loop_result_ports") or {}
            ) if int(loop.node_id) in graph.G else {}
            carried_names = {str(name) for name, _i, _u in loop.carried_bindings}
            result_ports = (
                *(
                    (int(loop_ports[str(name)]), int(initial), int(updated))
                    for name, initial, updated in loop.carried_bindings
                    if str(name) in loop_ports
                ),
                # A break-bound port has no backedge version: its exit Phi
                # merges the pre-loop identity with the break edges.
                *(
                    (int(loop_ports[str(name)]), int(initial), int(initial))
                    for name, initial, _continuation in loop.break_bindings
                    if str(name) in loop_ports and str(name) not in carried_names
                ),
            )
            control_site_ids = tuple(
                int(site_id) for site_id, _a, _c, _v, _o in loop.control_sites
            )
            # A carried seed that is a literal in the graph (``peak = 0.0``)
            # may be folded away by region planning; carry the literal so the
            # SSA lowering emits a Const instead of a producerless argument.
            carried_seeds = []
            for _name, initial, _updated in loop.carried_bindings:
                initial = int(initial)
                if initial not in graph.G:
                    continue
                seed_data = graph.G.nodes[initial]
                if str(seed_data.get("type")) not in {
                    "Const", "const", "Constant",
                }:
                    continue
                literal = seed_data.get("constant")
                if literal is None:
                    literal = (seed_data.get("attributes") or {}).get("value")
                if isinstance(literal, (int, float)) and not isinstance(
                    literal, bool
                ):
                    carried_seeds.append((initial, float(literal)))
            carried_seeds = tuple(carried_seeds)
            if loop.source_type == "While":
                return WhileBlock(
                    predicate_value_id=int(loop.condition_nodes[0]),
                    condition=SequenceBlock(tuple(
                        StatementBlock((f"__scheduled_region_{index}__",))
                        for index in condition_region_indices
                    )),
                    body=scheduled_body,
                    carried_aliases=carried_aliases,
                    result_ports=result_ports,
                    control_site_ids=control_site_ids,
                    carried_seeds=carried_seeds,
                    recursion_region_id=recursion_region_id,
                    predicate_expression=while_predicate_expression,
                    sequence_mutations=sequence_mutations,
                    source_loop_node_id=int(loop.node_id),
                    terminal_controls=terminal_controls,
                )
            return LoopBlock(
                induction=induction_name,
                result_ports=result_ports,
                control_site_ids=control_site_ids,
                carried_seeds=carried_seeds,
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
                sequence_mutations=sequence_mutations,
                terminal_controls=terminal_controls,
                source_loop_node_id=int(loop.node_id),
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
                    collection_bindings=(
                        ()
                        if loop.iteration_outputs
                        else planned_collection_bindings(
                            graph,
                            loop,
                            frozenset(
                                node_id
                                for region in regions
                                for node_id in region
                            ),
                        )
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
                    specialized_conditional_node_ids=(
                        specialized_conditional_node_ids
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
            structurally_owned_region_indices=(
                structurally_owned_region_indices
            ),
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
