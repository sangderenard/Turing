"""Backend-neutral fusion planning over semantic :class:`ProcessGraph` dataflow.

Backends advertise a capability/cost profile.  The planner finds connected
regions that can cross one backend dispatch boundary, while leaving layout,
reduction, synchronization, and unsupported operations visible as boundaries.
It does not contain backend algorithms or application-specific rewrites.

The FusedProgram adapter is intentionally a bridge, not a replacement IR.  It
lets existing captured AbstractTensor programs enter ProcessGraph scheduling
today, and lets one selected region reuse the established backend lowerers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import networkx as nx

from ..common.tensors.fused_ir import (
    FusedProgram,
    Meta,
    OpStep,
    canonical_elementwise_op,
    ordered_feed_ids,
)
from ..transmogrifier.graph.graph_express2 import ProcessGraph


@dataclass(frozen=True)
class BackendFusionProfile:
    """Capabilities and coarse costs used to select dispatch regions."""

    name: str
    fusible_ops: frozenset[str]
    max_bindings: int = 16
    max_steps: int = 4096
    launch_cost: float = 32.0
    intermediate_traffic_cost: float = 2.0
    binding_cost: float = 0.25


@dataclass(frozen=True)
class DispatchRegion:
    """One connected subgraph proposed as a backend dispatch."""

    node_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    outputs: tuple[tuple[str, int], ...]
    score: float

    @property
    def operation_count(self) -> int:
        return len(self.node_ids)

    @property
    def binding_count(self) -> int:
        return len(self.input_ids) + len(self.outputs)


@dataclass(frozen=True)
class ProcessGraphDispatchPlan:
    """Inspectible fusion result; uncovered nodes remain explicit boundaries."""

    backend: str
    regions: tuple[DispatchRegion, ...]
    uncovered_nodes: tuple[int, ...]


@dataclass(frozen=True)
class ScheduledOperatorPattern:
    """One same-operator batch at one existing ProcessGraph schedule level."""

    level: int
    operator: str
    node_ids: tuple[int, ...]
    batch_index: int
    batch_count: int


@dataclass(frozen=True)
class FlatComputeDispatch:
    """One ordered backend dispatch described directly by scheduled nodes."""

    kind: str
    node_ids: tuple[int, ...]
    levels: tuple[int, ...]
    operator_pattern: tuple[str, ...]
    dependency_columns: tuple[tuple[int, ...], ...] = ()

    @property
    def operation_count(self) -> int:
        return len(self.node_ids)


@dataclass(frozen=True)
class ScheduledProcessGraphDispatchPlan:
    """Flat serialization of the ProcessGraph's existing execution schedule."""

    patterns: tuple[ScheduledOperatorPattern, ...]
    dispatches: tuple[FlatComputeDispatch, ...]
    dependency_columns: tuple[tuple[int, ...], ...]
    levels: tuple[tuple[int, tuple[int, ...]], ...]
    node_locations: Mapping[int, tuple[int, int]]


def _isolated_dependency_columns(
    graph: ProcessGraph,
    levels: Mapping[int, int],
    topological: tuple[int, ...],
    cap: int,
) -> tuple[tuple[int, ...], ...]:
    """Find maximal scheduled chains with no internal fan-in or fan-out."""

    visited: set[int] = set()
    columns: list[tuple[int, ...]] = []
    for start in topological:
        if start in visited:
            continue
        predecessors = tuple(graph.G.predecessors(start))
        is_continuation = (
            len(predecessors) == 1
            and graph.G.out_degree(predecessors[0]) == 1
            and int(levels[start]) == int(levels[predecessors[0]]) + 1
        )
        if is_continuation:
            continue

        chain = [start]
        current = start
        while True:
            successors = tuple(graph.G.successors(current))
            if len(successors) != 1:
                break
            successor = successors[0]
            if (
                graph.G.in_degree(successor) != 1
                or int(levels[successor]) != int(levels[current]) + 1
            ):
                break
            chain.append(successor)
            current = successor

        if len(chain) < 2:
            continue
        visited.update(chain)
        for offset in range(0, len(chain), cap):
            part = tuple(chain[offset:offset + cap])
            if len(part) > 1:
                columns.append(part)
            elif part:
                visited.discard(part[0])
    return tuple(columns)


def serialize_scheduled_operator_dispatches(
    graph: ProcessGraph,
    *,
    max_nodes_per_dispatch: int = 256,
    schedule: str = "asap",
) -> ScheduledProcessGraphDispatchPlan:
    """Serialize scheduled operator batches and linear forward-record runs.

    The ProcessGraph scheduler remains authoritative.  Within each level,
    nodes with the same operation form one pattern and are split only at the
    explicit dispatch cap.  Consecutive levels containing exactly one node are
    emitted as a single ``forward_record`` dispatch; all other patterns become
    ordinary same-operator batches.
    """

    cap = int(max_nodes_per_dispatch)
    if cap < 1:
        raise ValueError("max_nodes_per_dispatch must be positive")
    if not nx.is_directed_acyclic_graph(graph.G):
        raise ValueError(
            "scheduled dispatch serialization requires an acyclic graph"
        )

    computed = graph.compute_levels(method=schedule, order="dependency")
    levels = computed if computed is not None else graph.levels
    topological = tuple(nx.topological_sort(graph.G))
    order_index = {
        node_id: index for index, node_id in enumerate(topological)
    }

    level_nodes: dict[int, list[int]] = {}
    for node_id in topological:
        level_nodes.setdefault(int(levels[node_id]), []).append(node_id)

    patterns: list[ScheduledOperatorPattern] = []
    for level in sorted(level_nodes):
        by_operator: dict[str, list[int]] = {}
        for node_id in sorted(
            level_nodes[level],
            key=order_index.__getitem__,
        ):
            by_operator.setdefault(_operation(graph, node_id), []).append(
                node_id
            )
        for operator, node_ids in by_operator.items():
            batch_count = (len(node_ids) + cap - 1) // cap
            for batch_index, start in enumerate(range(0, len(node_ids), cap)):
                patterns.append(
                    ScheduledOperatorPattern(
                        level=level,
                        operator=operator,
                        node_ids=tuple(node_ids[start:start + cap]),
                        batch_index=batch_index,
                        batch_count=batch_count,
                    )
                )

    dependency_columns = _isolated_dependency_columns(
        graph,
        levels,
        topological,
        cap,
    )
    column_nodes = {
        node_id
        for column in dependency_columns
        for node_id in column
    }

    # Columns with the same level span and operation sequence are independent
    # instances of one forward record.  Group them into the outer batch
    # dimension without exceeding the same explicit node cap.
    column_groups: dict[
        tuple[tuple[int, ...], tuple[str, ...]],
        list[tuple[int, ...]],
    ] = {}
    for column in dependency_columns:
        key = (
            tuple(int(levels[node_id]) for node_id in column),
            tuple(_operation(graph, node_id) for node_id in column),
        )
        column_groups.setdefault(key, []).append(column)

    dispatches: list[FlatComputeDispatch] = []
    for (column_levels, operator_pattern), columns in column_groups.items():
        columns_per_dispatch = max(1, cap // len(operator_pattern))
        for start in range(0, len(columns), columns_per_dispatch):
            batch = tuple(columns[start:start + columns_per_dispatch])
            dispatches.append(
                FlatComputeDispatch(
                    kind="forward_record",
                    node_ids=tuple(
                        node_id for column in batch for node_id in column
                    ),
                    levels=column_levels,
                    operator_pattern=operator_pattern,
                    dependency_columns=batch,
                )
            )

    for pattern in patterns:
        remaining = tuple(
            node_id
            for node_id in pattern.node_ids
            if node_id not in column_nodes
        )
        if not remaining:
            continue
        dispatches.append(
            FlatComputeDispatch(
                kind="operator_batch",
                node_ids=remaining,
                levels=(pattern.level,),
                operator_pattern=(pattern.operator,),
            )
        )

    dispatches.sort(
        key=lambda dispatch: (
            min(dispatch.levels),
            min(order_index[node_id] for node_id in dispatch.node_ids),
        )
    )

    node_locations = {
        node_id: (dispatch_index, lane_index)
        for dispatch_index, dispatch in enumerate(dispatches)
        for lane_index, node_id in enumerate(dispatch.node_ids)
    }
    serialized_levels = tuple(
        (level, tuple(level_nodes[level]))
        for level in sorted(level_nodes)
    )
    return ScheduledProcessGraphDispatchPlan(
        patterns=tuple(patterns),
        dispatches=tuple(dispatches),
        dependency_columns=dependency_columns,
        levels=serialized_levels,
        node_locations=node_locations,
    )


def _node_payload(
    op: str,
    *,
    label: str | None = None,
    parents: Iterable[tuple[int, str]] = (),
    attributes: Mapping[str, Any] | None = None,
    constant: Any = None,
    meta: Meta | None = None,
) -> dict[str, Any]:
    parent_list = list(parents)
    tensor = {}
    if meta is not None:
        tensor = {
            "shape": tuple(meta.shape or ()),
            "dtype": meta.dtype,
            "device": meta.device,
        }
    attrs = dict(attributes or {})
    return {
        "label": label or op,
        "type": op,
        "op": op,
        "expr_obj": None,
        "extra_args": attrs,
        "attributes": attrs,
        "constant": constant,
        "tensor": tensor,
        "bit_quanta": None,
        "control": {},
        "source_span": None,
        "input_roles": tuple(role for _, role in parent_list),
        "output_roles": ("result",),
        "schema_version": 1,
        "domain_node": None,
        "store_id": None,
        "parents": parent_list,
        "children": [],
    }


def fused_program_to_process_graph(program: FusedProgram) -> ProcessGraph:
    """Project an established FusedProgram into semantic ProcessGraph form."""

    graph = ProcessGraph(materialize_memory=False)
    metadata = program.meta or {}
    defined: set[int] = set()

    def add_node(node_id: int, payload: dict[str, Any]) -> None:
        if node_id in graph.G:
            raise ValueError(f"duplicate ProcessGraph value id {node_id}")
        graph.G.add_node(node_id, **payload)
        for parent_id, role in payload["parents"]:
            if parent_id not in graph.G:
                raise ValueError(
                    f"ProcessGraph value {node_id} reads undefined {parent_id}"
                )
            graph.G.add_edge(parent_id, node_id, role=role)
            graph.G.nodes[parent_id]["children"].append((node_id, role))
        defined.add(node_id)

    for feed_id in ordered_feed_ids(program):
        add_node(
            feed_id,
            _node_payload(
                "input",
                label=f"feed_{feed_id}",
                attributes={"name": f"feed_{feed_id}"},
                meta=metadata.get(feed_id),
            ),
        )

    next_id = max(
        [
            *defined,
            *(step.result_id for step in program.steps),
            *program.outputs.values(),
        ],
        default=-1,
    ) + 1

    for step in program.steps:
        op, prefix_reverse = canonical_elementwise_op(step.op_name)
        attrs = dict(step.attrs)
        reverse = prefix_reverse ^ bool(attrs.pop("reverse", False))
        scalar_present = "right_scalar" in attrs
        scalar = attrs.pop("right_scalar", None)
        if attrs:
            raise ValueError(
                f"step {step.step_id} has unsupported ProcessGraph attrs: "
                f"{sorted(attrs)}"
            )
        if scalar_present:
            constant_id = next_id
            next_id += 1
            add_node(
                constant_id,
                _node_payload("const", label=repr(scalar), constant=scalar),
            )
            value_id = step.input_ids[0]
            parents = (
                ((constant_id, "lhs"), (value_id, "rhs"))
                if reverse
                else ((value_id, "lhs"), (constant_id, "rhs"))
            )
        else:
            input_ids = list(step.input_ids)
            if reverse and len(input_ids) == 2:
                input_ids.reverse()
            roles = ("operand",) if len(input_ids) == 1 else ("lhs", "rhs")
            parents = tuple(zip(input_ids, roles))
        add_node(
            step.result_id,
            _node_payload(
                op,
                parents=parents,
                meta=metadata.get(step.result_id),
            ),
        )

    for name, output_id in program.outputs.items():
        return_id = next_id
        next_id += 1
        payload = _node_payload(
            "return",
            label=f"return_{name}",
            parents=((output_id, "value"),),
            attributes={"name": str(name)},
        )
        payload["output_roles"] = ()
        add_node(return_id, payload)
        graph.roots.append(return_id)

    graph.domain_shape = (1,)
    graph.G.graph["feed_order"] = ordered_feed_ids(program)
    graph.G.graph["source_ir"] = "FusedProgram"
    return graph


def _operation(graph: ProcessGraph, node_id: int) -> str:
    data = graph.G.nodes[node_id]
    return str(data.get("op") or data.get("type") or data.get("label"))


def plan_process_graph_dispatches(
    graph: ProcessGraph,
    profile: BackendFusionProfile,
) -> ProcessGraphDispatchPlan:
    """Select maximal profitable connected regions for one backend.

    The current cost model is deliberately small and inspectible.  Maximal
    compatible components are accepted when their binding/step limits fit and
    fusion saves at least one launch or one materialized intermediate.  More
    sophisticated search can replace this policy without changing the graph or
    backend contracts.
    """

    if not nx.is_directed_acyclic_graph(graph.G):
        raise ValueError(
            "fusion planning requires loop structure to be normalized first"
        )
    fusible = {
        node_id
        for node_id in graph.G
        if _operation(graph, node_id) in profile.fusible_ops
    }
    induced = graph.G.subgraph(fusible)
    components = list(nx.weakly_connected_components(induced))
    topological = list(nx.topological_sort(graph.G))
    order_index = {node_id: index for index, node_id in enumerate(topological)}
    regions: list[DispatchRegion] = []
    covered: set[int] = set()

    for component in components:
        nodes = tuple(sorted(component, key=order_index.__getitem__))
        if not nodes or len(nodes) > profile.max_steps:
            continue
        node_set = set(nodes)
        input_ids = tuple(
            node_id
            for node_id in topological
            if node_id not in node_set
            and any(child in node_set for child in graph.G.successors(node_id))
            and _operation(graph, node_id) != "const"
        )
        output_names: dict[int, str] = {}
        for node_id in nodes:
            for child in graph.G.successors(node_id):
                if child in node_set:
                    continue
                child_data = graph.G.nodes[child]
                if _operation(graph, child) == "return":
                    name = str(
                        (child_data.get("attributes") or {}).get(
                            "name", f"result_{len(output_names)}"
                        )
                    )
                else:
                    name = f"value_{node_id}"
                output_names.setdefault(node_id, name)
        if not output_names:
            continue
        outputs = tuple(
            (name, node_id) for node_id, name in output_names.items()
        )
        binding_count = len(input_ids) + len(outputs)
        if binding_count > profile.max_bindings:
            continue
        internal_edges = sum(
            1
            for left, right in graph.G.edges
            if left in node_set and right in node_set
        )
        score = (
            max(0, len(nodes) - 1) * profile.launch_cost
            + internal_edges * profile.intermediate_traffic_cost
            - binding_count * profile.binding_cost
        )
        if score <= 0:
            continue
        region = DispatchRegion(nodes, input_ids, outputs, score)
        regions.append(region)
        covered.update(nodes)

    regions.sort(key=lambda region: order_index[region.node_ids[0]])
    uncovered = tuple(
        node_id for node_id in topological if node_id not in covered
    )
    return ProcessGraphDispatchPlan(profile.name, tuple(regions), uncovered)


def dispatch_region_to_fused_program(
    graph: ProcessGraph,
    region: DispatchRegion,
) -> FusedProgram:
    """Lower one selected elementwise ProcessGraph region to FusedProgram."""

    node_set = set(region.node_ids)
    steps: list[OpStep] = []
    metadata: dict[int, Meta] = {}
    for value_id in (*region.input_ids, *region.node_ids):
        tensor = graph.G.nodes[value_id].get("tensor") or {}
        metadata[value_id] = Meta(
            shape=tuple(tensor.get("shape") or ()),
            dtype=tensor.get("dtype"),
            device=tensor.get("device"),
        )

    for node_id in region.node_ids:
        data = graph.G.nodes[node_id]
        op, _ = canonical_elementwise_op(_operation(graph, node_id))
        parents = list(data.get("parents") or ())
        value_parents: list[int] = []
        scalar_parent: tuple[int, Any] | None = None
        for parent_id, _role in parents:
            parent_data = graph.G.nodes[parent_id]
            if _operation(graph, parent_id) == "const":
                scalar_parent = (parent_id, parent_data.get("constant"))
            else:
                value_parents.append(parent_id)
        attrs: dict[str, Any] = {}
        if scalar_parent is not None:
            if len(value_parents) != 1:
                raise ValueError(f"{op} has an invalid scalar operand layout")
            attrs["right_scalar"] = scalar_parent[1]
            if parents[0][0] == scalar_parent[0]:
                attrs["reverse"] = True
        steps.append(
            OpStep(
                step_id=len(steps),
                op_name=op,
                input_ids=value_parents,
                attrs=attrs,
                result_id=node_id,
            )
        )

    return FusedProgram(
        version=1,
        feeds=set(region.input_ids),
        steps=steps,
        outputs=dict(region.outputs),
        meta=metadata,
    )


__all__ = [
    "BackendFusionProfile",
    "DispatchRegion",
    "FlatComputeDispatch",
    "ProcessGraphDispatchPlan",
    "ScheduledOperatorPattern",
    "ScheduledProcessGraphDispatchPlan",
    "dispatch_region_to_fused_program",
    "fused_program_to_process_graph",
    "plan_process_graph_dispatches",
    "serialize_scheduled_operator_dispatches",
]
