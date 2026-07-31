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

import copy
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
    rewrite_history: tuple[str, ...] = ()

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


def extract_clean_process_subgraph(
    graph: ProcessGraph,
    node_ids: Iterable[int],
) -> ProcessGraph:
    """Copy an induced subgraph without obligations to excluded nodes."""

    included = set(node_ids)
    extracted = copy.copy(graph)
    extracted.G = graph.G.subgraph(included).copy()
    for node_id in extracted.G:
        data = extracted.G.nodes[node_id]
        data["parents"] = [
            (parent, role)
            for parent, role in data.get("parents", ())
            if parent in included
        ]
        data["children"] = [
            (child, role)
            for child, role in data.get("children", ())
            if child in included
        ]
    extracted.levels = {
        node_id: level
        for node_id, level in graph.levels.items()
        if node_id in included
    }
    extracted.roots = [
        node_id for node_id in graph.roots if node_id in included
    ]
    extracted.scheduler = copy.copy(graph.scheduler)
    extracted.scheduler.G = extracted.G
    return extracted


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


def _planner_levels(
    graph: ProcessGraph,
    *,
    fallback_schedule: str,
) -> Mapping[int, int]:
    """Consume the ProcessGraph planner's schedule without replacing it."""

    existing = dict(getattr(graph, "levels", {}) or {})
    if set(existing) == set(graph.G):
        return existing
    computed = graph.compute_levels(
        method=fallback_schedule,
        order="dependency",
    )
    levels = computed if computed is not None else graph.levels
    if set(levels) != set(graph.G):
        missing = set(graph.G) - set(levels)
        raise ValueError(
            "ProcessGraph planner did not schedule every graph node: "
            + ", ".join(map(str, sorted(missing)))
        )
    return dict(levels)


def serialize_scheduled_operator_dispatches(
    graph: ProcessGraph,
    *,
    max_nodes_per_dispatch: int = 256,
    schedule: str = "asap",
) -> ScheduledProcessGraphDispatchPlan:
    """Serialize scheduled operator batches and linear forward-record runs.

    CRITICAL SHADER-PLANNING INVARIANTS
    -----------------------------------
    A topological schedule states which operations must precede which other
    operations.  It does *not* state where shader or dispatch boundaries must
    be placed.  Any planner built on this serialization must preserve all
    three of these facts:

    1. Horizontal groups of independent work can be lanes of one batched
       dispatch.  They do not require one dispatch per graph operation.
    2. Vertical dependency chains can be flattened into one shader.  Executing
       the dependent expressions serially inside one shader invocation is
       massively cheaper than paying for an individual GPU dispatch and
       materialized intermediate at every step.
    3. Causally separated subgraphs can coexist in that same shader, provided
       the emitted shader obeys every dependency edge and execution-order
       constraint.  Absence of a causal edge permits co-scheduling; it is not
       a reason to manufacture sequential dispatches.

    Consequently, grouping by schedule level or by identical operator is only
    an intermediate description of available work.  It is not a valid final
    fusion boundary.  Final shader regions should be maximal compatible DAGs,
    split only by real backend, resource, synchronization, control-flow, or
    externally observable materialization constraints.

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

    levels = _planner_levels(graph, fallback_schedule=schedule)
    topological = tuple(
        nx.lexicographical_topological_sort(
            graph.G, key=lambda node_id: int(node_id)
        )
    )
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


def reduce_scheduled_shader_regions(
    graph: ProcessGraph,
    executable_node_ids: Iterable[int],
    *,
    max_nodes_per_region: int = 256,
    max_bindings_per_region: int | None = None,
    partition_keys: Mapping[int, Any] | None = None,
    extra_dependency_edges: Iterable[tuple[int, int]] = (),
    fusible_node_ids: Iterable[int] | None = None,
    schedule: str = "asap",
) -> ScheduledProcessGraphDispatchPlan:
    """Reduce executable nodes to maximal shader regions by fixed point.

    The semantic ProcessGraph is never rewritten.  This function constructs a
    quotient graph whose vertices are shader regions and monotonically merges
    them using three ordered identities: homogeneous horizontal batching,
    vertical dependency fusion, then heterogeneous same-level packing.
    Because every accepted rewrite reduces the number of quotient vertices,
    the process is guaranteed to terminate.

    ``extra_dependency_edges`` declares causal ordering the caller knows about
    but the graph does not spell as an edge — for example a value routed into
    a callee by name through an identity table rather than through a parent
    reference.  Those edges constrain legality and ordering exactly like
    graph-visible edges through structural nodes; they are never treated as
    direct numerical edges and therefore never justify vertical fusion.

    ``fusible_node_ids`` restricts merging to operations the backend can emit
    inside one shader body.  An operation outside that set is a real dispatch
    boundary rather than a planning preference, so it stays alone in its own
    region; the default admits every executable node.

    """

    cap = int(max_nodes_per_region)
    if cap < 1:
        raise ValueError("max_nodes_per_region must be positive")
    if not nx.is_directed_acyclic_graph(graph.G):
        raise ValueError("shader-region reduction requires an acyclic graph")

    levels = _planner_levels(graph, fallback_schedule=schedule)
    topological = tuple(
        nx.lexicographical_topological_sort(
            graph.G, key=lambda node_id: int(node_id)
        )
    )
    order_index = {
        node_id: index for index, node_id in enumerate(topological)
    }
    executable = {
        node_id for node_id in executable_node_ids if node_id in graph.G
    }
    keys = dict(partition_keys or {})
    fusible = (
        set(executable)
        if fusible_node_ids is None
        else {node_id for node_id in fusible_node_ids if node_id in executable}
    )
    graph_edges = set(graph.G.edges)
    semantic_edges = graph_edges | {
        (parent, node_id)
        for node_id, data in graph.G.nodes(data=True)
        for parent, _role in data.get("parents", ())
        if parent in graph.G
    } | {
        (left, right)
        for left, right in extra_dependency_edges
        if left in graph.G and right in graph.G
    }
    semantic_successors: dict[int, set[int]] = {
        node_id: set() for node_id in graph.G
    }
    for left, right in semantic_edges:
        semantic_successors[left].add(right)
    direct_execution_edges = {
        (left, right)
        for left, right in graph_edges
        if left in executable and right in executable
    }
    # Structural/coordinator nodes are not shader instructions, but paths
    # through them still impose causal ordering on numerical regions.  Project
    # each such path onto its nearest executable endpoints so horizontal
    # packing cannot accidentally contract a hidden A -> coordinator -> B path
    # into a cyclic region.  These projected edges participate in legality and
    # scheduling only; vertical fusion below still requires a direct numerical
    # edge and therefore never absorbs a coordinator boundary.
    projected_execution_edges = {
        (left, right)
        for left, right in semantic_edges
        if left in executable and right in executable
    }
    for source in executable:
        pending = [
            child
            for child in semantic_successors[source]
            if child not in executable
        ]
        visited_structural = set()
        while pending:
            current = pending.pop()
            if current in visited_structural:
                continue
            visited_structural.add(current)
            for child in semantic_successors[current]:
                if child in executable:
                    projected_execution_edges.add((source, child))
                else:
                    pending.append(child)

    level_nodes: dict[int, list[int]] = {}
    for node_id in topological:
        level_nodes.setdefault(int(levels[node_id]), []).append(node_id)
    patterns = []
    for level in sorted(level_nodes):
        by_operator: dict[str, list[int]] = {}
        for node_id in level_nodes[level]:
            if node_id in executable:
                by_operator.setdefault(
                    _operation(graph, node_id), []
                ).append(node_id)
        for operator, node_ids in by_operator.items():
            patterns.append(ScheduledOperatorPattern(
                level=level,
                operator=operator,
                node_ids=tuple(node_ids),
                batch_index=0,
                batch_count=1,
            ))

    regions: dict[int, set[int]] = {
        index: {node_id}
        for index, node_id in enumerate(
            node_id for node_id in topological if node_id in executable
        )
    }
    histories: dict[int, list[str]] = {
        region_id: [] for region_id in regions
    }
    next_region_id = len(regions)

    def region_key(members):
        member_keys = {keys.get(node_id) for node_id in members}
        return next(iter(member_keys)) if len(member_keys) == 1 else object()

    def quotient(candidate_regions=None):
        active = regions if candidate_regions is None else candidate_regions
        owner = {
            node_id: region_id
            for region_id, members in active.items()
            for node_id in members
        }
        quotient_graph = nx.DiGraph()
        quotient_graph.add_nodes_from(active)
        for left, right in projected_execution_edges:
            left_owner = owner.get(left)
            right_owner = owner.get(right)
            if (
                left_owner is not None
                and right_owner is not None
                and left_owner != right_owner
            ):
                quotient_graph.add_edge(left_owner, right_owner)
        return quotient_graph

    def boundary_outputs(members):
        return {
            node_id
            for node_id in members
            if (
                graph.G.out_degree(node_id) == 0
                or any(
                    child not in members
                    for child in graph.G.successors(node_id)
                )
            )
        }

    def binding_count(members):
        inputs = {
            parent
            for node_id in members
            for parent in graph.G.predecessors(node_id)
            if parent not in members
        }
        return len(inputs) + len(boundary_outputs(members))

    def can_merge(region_ids, quotient_graph=None):
        region_ids = tuple(dict.fromkeys(region_ids))
        if len(region_ids) < 2:
            return False
        members = set().union(*(regions[item] for item in region_ids))
        if len(members) > cap:
            return False
        if not members <= fusible:
            # An operation the backend cannot emit inside a shader body is a
            # dispatch of its own.  Absorbing it would produce a region no
            # lowerer can accept, which is worse than not fusing at all.
            return False
        if any(
            left in members
            and right in members
            and (left, right) not in direct_execution_edges
            for left, right in projected_execution_edges
        ):
            # The dependency between these numerical endpoints crosses at
            # least one structural/coordinator node.  Internalizing both ends
            # would leave that structural node as an external shader input
            # which itself depends on a shader-local result, creating an
            # impossible region-boundary cycle.
            return False
        member_keys = {keys.get(node_id) for node_id in members}
        if len(member_keys) > 1:
            return False
        if (
            max_bindings_per_region is not None
            and binding_count(members) > int(max_bindings_per_region)
        ):
            return False
        # Contracting vertices in a DAG creates a cycle exactly when a path
        # leaves the contracted set and later re-enters it.  Test that property
        # directly on the current quotient instead of cloning every region,
        # rebuilding the full quotient, and running a global DAG check for
        # every candidate.  The old formulation made reduction superlinear in
        # both allocations and graph traversals on large compiled shells.
        current_quotient = (
            quotient() if quotient_graph is None else quotient_graph
        )
        selected = set(region_ids)
        pending = [
            child
            for region_id in selected
            for child in current_quotient.successors(region_id)
            if child not in selected
        ]
        visited = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for child in current_quotient.successors(current):
                if child in selected:
                    return False
                pending.append(child)
        return True

    def merge(region_ids, identity):
        nonlocal next_region_id
        region_ids = tuple(dict.fromkeys(region_ids))
        members = set().union(*(regions.pop(item) for item in region_ids))
        history = [
            entry
            for item in region_ids
            for entry in histories.pop(item)
        ]
        history.append(identity)
        merged_id = next_region_id
        next_region_id += 1
        regions[merged_id] = members
        histories[merged_id] = history
        return merged_id

    changed = True
    while changed:
        changed = False

        # Identity 1: same-level calls of the same operator and execution
        # partition become lanes of one horizontal batch.
        quotient_graph = quotient()
        quotient_levels = {
            region_id: int(level)
            for region_id, level in nx.get_node_attributes(
                quotient_graph, "level"
            ).items()
        }
        if not quotient_levels:
            quotient_levels = {
                region_id: generation
                for generation, generation_nodes in enumerate(
                    nx.topological_generations(quotient_graph)
                )
                for region_id in generation_nodes
            }
        homogeneous: dict[tuple[Any, ...], list[int]] = {}
        for region_id, members in regions.items():
            operators = {_operation(graph, node_id) for node_id in members}
            if len(operators) != 1:
                continue
            homogeneous.setdefault((
                quotient_levels[region_id],
                next(iter(operators)),
                region_key(members),
            ), []).append(region_id)
        for candidates in homogeneous.values():
            while len(candidates) > 1:
                group = candidates[:cap]
                while len(group) > 1 and not can_merge(group):
                    group.pop()
                if len(group) < 2:
                    break
                merged = merge(group, "horizontal-batch")
                candidates[:len(group)] = [merged]
                changed = True

        # Identity 2: dependency-connected regions become one internally
        # topologically ordered shader.
        while True:
            quotient_graph = quotient()
            merged_vertical = False
            for left, right in sorted(
                quotient_graph.edges,
                key=lambda edge: (
                    min(order_index[node] for node in regions[edge[0]]),
                    min(order_index[node] for node in regions[edge[1]]),
                ),
            ):
                if (
                    any(
                        source in regions[left]
                        and target in regions[right]
                        for source, target in direct_execution_edges
                    )
                    and can_merge((left, right), quotient_graph)
                ):
                    merge((left, right), "vertical-fusion")
                    changed = True
                    merged_vertical = True
                    break
            if not merged_vertical:
                break

        # Identity 3: causally independent regions ready at the same quotient
        # level may share one shader, with their internal instructions emitted
        # in a dependency-respecting order.
        quotient_graph = quotient()
        same_level: dict[tuple[int, Any], list[int]] = {}
        for generation, generation_nodes in enumerate(
            nx.topological_generations(quotient_graph)
        ):
            for region_id in generation_nodes:
                same_level.setdefault((
                    generation,
                    region_key(regions[region_id]),
                ), []).append(region_id)
        for candidates in same_level.values():
            while len(candidates) > 1:
                group = candidates[:cap]
                while len(group) > 1 and not can_merge(group):
                    group.pop()
                if len(group) < 2:
                    break
                merged = merge(group, "horizontal-shader-pack")
                candidates[:len(group)] = [merged]
                changed = True

    dispatches = []
    for region_id, members in regions.items():
        ordered = tuple(sorted(members, key=order_index.__getitem__))
        dispatches.append(FlatComputeDispatch(
            kind="shader_region",
            node_ids=ordered,
            levels=tuple(sorted({int(levels[node]) for node in ordered})),
            operator_pattern=tuple(_operation(graph, node) for node in ordered),
            rewrite_history=tuple(histories[region_id]),
        ))
    dispatches.sort(
        key=lambda dispatch: min(
            order_index[node_id] for node_id in dispatch.node_ids
        )
    )
    node_locations = {
        node_id: (dispatch_index, lane_index)
        for dispatch_index, dispatch in enumerate(dispatches)
        for lane_index, node_id in enumerate(dispatch.node_ids)
    }
    return ScheduledProcessGraphDispatchPlan(
        patterns=tuple(patterns),
        dispatches=tuple(dispatches),
        dependency_columns=(),
        levels=tuple(
            (level, tuple(level_nodes[level]))
            for level in sorted(level_nodes)
        ),
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
    topological = list(
        nx.lexicographical_topological_sort(
            graph.G, key=lambda node_id: int(node_id)
        )
    )
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
    "reduce_scheduled_shader_regions",
    "serialize_scheduled_operator_dispatches",
]
