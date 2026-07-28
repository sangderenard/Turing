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
from .process_graph_contract import NON_VALUE_EDGE_ROLES


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
    return str(data.get("op") or data.get("label"))


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
    induced = nx.DiGraph()
    induced.add_nodes_from(fusible)
    induced.add_edges_from(
        (left, right)
        for left, right, data in graph.G.edges(data=True)
        if left in fusible
        and right in fusible
        and data.get("role") not in NON_VALUE_EDGE_ROLES
    )
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
            and any(
                child in node_set
                and graph.G.edges[node_id, child].get("role")
                not in NON_VALUE_EDGE_ROLES
                for child in graph.G.successors(node_id)
            )
            and _operation(graph, node_id) != "const"
        )
        output_names: dict[int, str] = {}
        for node_id in nodes:
            for child in graph.G.successors(node_id):
                role = graph.G.edges[node_id, child].get("role")
                if child in node_set or role in NON_VALUE_EDGE_ROLES:
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
            and graph.G.edges[left, right].get("role")
            not in NON_VALUE_EDGE_ROLES
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
        parents = [
            (parent, role)
            for parent, role in (data.get("parents") or ())
            if role not in NON_VALUE_EDGE_ROLES
        ]
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
    "ProcessGraphDispatchPlan",
    "dispatch_region_to_fused_program",
    "fused_program_to_process_graph",
    "plan_process_graph_dispatches",
]
