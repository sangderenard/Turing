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
import numpy as np

from ..common.tensors.fused_ir import (
    AXIS_REDUCTION_FOLDS,
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    FusedProgram,
    Meta,
    OpStep,
    canonical_elementwise_op,
    flatten_tensor_constant,
    ordered_feed_ids,
    uniform_tensor_constant,
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

    def isolate_metadata(value):
        """Copy mutable metadata containers without cloning semantic objects."""

        if isinstance(value, dict):
            return {
                isolate_metadata(key): isolate_metadata(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [isolate_metadata(item) for item in value]
        if isinstance(value, tuple):
            return tuple(isolate_metadata(item) for item in value)
        if isinstance(value, set):
            return {isolate_metadata(item) for item in value}
        if isinstance(value, frozenset):
            return frozenset(isolate_metadata(item) for item in value)
        return value

    included = {
        int(node_id) for node_id in node_ids if int(node_id) in graph.G
    }
    extracted = copy.copy(graph)
    extracted.G = graph.G.subgraph(included).copy()
    extracted.G.graph = isolate_metadata(dict(graph.G.graph))
    for node_id in extracted.G:
        data = extracted.G.nodes[node_id]
        for key, value in tuple(data.items()):
            if key != "expr_obj":
                data[key] = isolate_metadata(value)
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
    indivisible_node_groups: Iterable[Iterable[int]] = (),
    control_node_ids: Iterable[int] = (),
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

    ``indivisible_node_groups`` are compiler-owned regions whose membership
    was settled before ordinary fusion. They enter the quotient as one vertex
    and are never split or enlarged by the opportunistic shader rewrites
    below. This is the region reducer's reservation mechanism for a semantic
    section whose interior operations are meaningful only when lowered
    together.

    ``control_node_ids`` are cached recursion/loop-IR nodes.  Fusion removes
    only edges incident to those nodes from its scheduling projection.  The
    semantic graph remains unchanged, and numerical nodes inside the same SCC
    remain executable and reducible.

    """

    cap = int(max_nodes_per_region)
    if cap < 1:
        raise ValueError("max_nodes_per_region must be positive")
    control_nodes = {
        int(node_id)
        for node_id in control_node_ids
        if int(node_id) in graph.G
    }
    planning_graph = graph.G.copy()
    planning_graph.remove_edges_from(tuple(
        (left, right)
        for left, right in planning_graph.edges
        if left in control_nodes or right in control_nodes
    ))
    if not nx.is_directed_acyclic_graph(planning_graph):
        raise ValueError("shader-region reduction requires an acyclic graph")

    levels = _planner_levels(graph, fallback_schedule=schedule)
    topological = tuple(
        nx.lexicographical_topological_sort(
            planning_graph, key=lambda node_id: int(node_id)
        )
    )
    order_index = {
        node_id: index for index, node_id in enumerate(topological)
    }
    executable = {
        node_id
        for node_id in executable_node_ids
        if node_id in graph.G and node_id not in control_nodes
    }
    keys = dict(partition_keys or {})
    fusible = (
        set(executable)
        if fusible_node_ids is None
        else {node_id for node_id in fusible_node_ids if node_id in executable}
    )
    graph_edges = set(planning_graph.edges)
    semantic_edges = graph_edges | {
        (parent, node_id)
        for node_id, data in graph.G.nodes(data=True)
        for parent, _role in data.get("parents", ())
        if (
            parent in graph.G
            and parent not in control_nodes
            and node_id not in control_nodes
        )
    } | {
        (left, right)
        for left, right in extra_dependency_edges
        if (
            left in graph.G
            and right in graph.G
            and left not in control_nodes
            and right not in control_nodes
        )
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
    # Keep the path's boundary provenance separately from its endpoints.
    # A -> B and A -> coordinator -> B can coexist. The direct edge does
    # not license swallowing the second path into a region: the coordinator
    # still needs A's publication before B can execute.
    coordinator_execution_edges = projected_execution_edges - direct_execution_edges
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
                    coordinator_execution_edges.add((source, child))
                else:
                    pending.append(child)

    # Index the coordinator-crossing edges by source so a merge candidate
    # is checked in time proportional to its members' degrees, not to the
    # size of the whole projected edge set.
    coordinator_successors: dict[int, set[int]] = {}
    for left, right in coordinator_execution_edges:
        coordinator_successors.setdefault(left, set()).add(right)

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

    reserved_groups = []
    reserved_members: set[int] = set()
    for supplied_group in indivisible_node_groups:
        group = {
            int(node_id) for node_id in supplied_group
            if int(node_id) in executable
        }
        if len(group) < 2:
            continue
        overlap = group.intersection(reserved_members)
        if overlap:
            raise ValueError(
                "indivisible dispatch regions overlap: "
                f"members={tuple(sorted(overlap))!r}"
            )
        member_keys = {keys.get(node_id) for node_id in group}
        if len(member_keys) > 1:
            raise ValueError(
                "indivisible dispatch region crosses a control partition: "
                f"members={tuple(sorted(group))!r}"
            )
        reserved_groups.append(group)
        reserved_members.update(group)

    initial_groups = [
        *reserved_groups,
        *(
            {node_id} for node_id in topological
            if node_id in executable and node_id not in reserved_members
        ),
    ]
    regions: dict[int, set[int]] = {
        index: set(group) for index, group in enumerate(initial_groups)
    }
    reserved_region_ids = set(range(len(reserved_groups)))
    histories: dict[int, list[str]] = {
        region_id: (
            ["indivisible-reservation"]
            if region_id in reserved_region_ids else []
        )
        for region_id in regions
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
        if any(region_id in reserved_region_ids for region_id in region_ids):
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
            right in members
            for left in members
            for right in coordinator_successors.get(left, ())
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
            # One owner map per quotient: the direct-edge test and the
            # ordering key are then constant-time per quotient edge.  The
            # previous per-edge scans of every direct edge and every member
            # made this pass superlinear enough to stall a 64-step unrolled
            # recurrent training graph indefinitely.
            owner = {
                node_id: region_id
                for region_id, members in regions.items()
                for node_id in members
            }
            direct_region_pairs = {
                (owner[source], owner[target])
                for source, target in direct_execution_edges
                if owner[source] != owner[target]
            }
            region_start = {
                region_id: min(order_index[node] for node in members)
                for region_id, members in regions.items()
            }
            for left, right in sorted(
                quotient_graph.edges,
                key=lambda edge: (region_start[edge[0]], region_start[edge[1]]),
            ):
                if (
                    (left, right) in direct_region_pairs
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

    # Regions execute in the order they are listed, so that order must respect
    # the quotient graph, not the position of each region's earliest member.
    # Fusing a node with a consumer that depends on a later region moved the
    # whole region ahead of its own producer: the consumer then read the
    # producer's pre-loop value while the real result was versioned into a
    # value nothing read -- a use before definition that emitted cleanly.
    # Every accepted merge keeps the quotient acyclic, so a topological order
    # always exists; the earliest-member index remains the tie-break, which
    # leaves every already-legal listing exactly as it was.
    # Projected edges through structural nodes can put two regions in an
    # apparent mutual dependency the merge legality never examined, so the
    # quotient is not guaranteed acyclic.  Order through the condensation:
    # strongly-connected regions share a rank (their internal order falls to
    # the earliest-member tie-break) while every true dependency still holds.
    quotient_graph = quotient()
    condensed = nx.condensation(quotient_graph)
    component_rank = {
        member: position
        for position, component in enumerate(
            nx.lexicographical_topological_sort(
                condensed,
                key=lambda component: min(
                    order_index[node_id]
                    for member in condensed.nodes[component]["members"]
                    for node_id in regions[member]
                ),
            )
        )
        for member in condensed.nodes[component]["members"]
    }
    region_order = {
        region_id: (
            component_rank[region_id],
            min(order_index[node_id] for node_id in regions[region_id]),
        )
        for region_id in regions
    }
    dispatches = []
    for region_id, members in sorted(
        regions.items(), key=lambda item: region_order[item[0]]
    ):
        ordered = tuple(sorted(members, key=order_index.__getitem__))
        dispatches.append(FlatComputeDispatch(
            kind="shader_region",
            node_ids=ordered,
            levels=tuple(sorted({int(levels[node]) for node in ordered})),
            operator_pattern=tuple(_operation(graph, node) for node in ordered),
            rewrite_history=tuple(histories[region_id]),
        ))
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
            # ``None`` is unknown; ``()`` is a proven scalar.  Collapsing the
            # former into the latter makes every later backend broadcast a
            # genuinely tensor-valued feed from lane zero.
            "shape": tuple(meta.shape) if meta.shape is not None else None,
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
        if step.op_name == "tensor_from_list":
            # A constructor is semantic data, not a WASM policy decision.
            # Preserve its complete nested value and tensor metadata here;
            # a backend may later choose an immediate for a uniform value or
            # materialize a varying value in its native storage.
            if "values" not in step.attrs:
                raise ValueError(
                    f"step {step.step_id} tensor_from_list has no values"
                )
            constant = copy.deepcopy(step.attrs["values"])
            attributes = {
                key: copy.deepcopy(value)
                for key, value in step.attrs.items()
                if key != "values"
            }
            attributes["creation_op"] = "tensor_from_list"
            add_node(
                step.result_id,
                _node_payload(
                    "const",
                    label=f"tensor constant %{step.result_id}",
                    constant=constant,
                    attributes=attributes,
                    meta=metadata.get(step.result_id),
                ),
            )
            continue
        if step.op_name in AXIS_REDUCTION_FOLDS:
            add_node(
                step.result_id,
                _node_payload(
                    step.op_name,
                    parents=tuple(
                        (value_id, "operand") for value_id in step.input_ids
                    ),
                    attributes=copy.deepcopy(dict(step.attrs)),
                    meta=metadata.get(step.result_id),
                ),
            )
            continue
        if step.op_name in {"reshape", "broadcast_to"}:
            add_node(
                step.result_id,
                _node_payload(
                    step.op_name,
                    parents=tuple(
                        (value_id, "operand") for value_id in step.input_ids
                    ),
                    attributes=copy.deepcopy(dict(step.attrs)),
                    meta=metadata.get(step.result_id),
                ),
            )
            continue
        if step.op_name == "where":
            if len(step.input_ids) != 3:
                raise ValueError(
                    f"where step {step.step_id} needs condition, true, and false inputs"
                )
            add_node(
                step.result_id,
                _node_payload(
                    "where",
                    parents=tuple(zip(
                        step.input_ids, ("condition", "true", "false")
                    )),
                    attributes=copy.deepcopy(dict(step.attrs)),
                    meta=metadata.get(step.result_id),
                ),
            )
            continue
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
    attributes = data.get("attributes") or {}
    # Tensor resolution is a ProcessGraph provenance receipt.  The syntactic
    # node can still be a generic ``Call`` after resolution, while the receipt
    # states the exact numerical operator (for example builtins ``abs`` over a
    # tensor).  Transcribing the syntax instead discards that settled identity
    # and hands backends a fictitious opaque call.
    raw = str(
        attributes.get("tensor")
        or attributes.get("tensor_operation")
        or data.get("op")
        or data.get("type")
        or data.get("label")
    )
    # A ProcessGraph built from a SymPy expression carries SSA-Handler-style
    # capitalized spellings ("Add", "Mul", "Pow", ...; see
    # symbolic_process_graph.py's _SYMPY_TO_CANONICAL) rather than this
    # module's lowercase tape-op vocabulary. Canonicalize once, here, so both
    # the fusibility check below and dispatch_region_to_fused_program's
    # OpStep.op_name agree with profile.fusible_ops, which is stated in the
    # lowercase vocabulary. Non-elementwise labels ("const", "input",
    # "return") have no canonical form and pass through unchanged.
    try:
        canonical, _ = canonical_elementwise_op(raw)
    except KeyError:
        return raw
    return canonical


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
    def has_only_numeric_constant_operands(node_id: int) -> bool:
        for parent_id, _role in graph.G.nodes[node_id].get("parents") or ():
            if _operation(graph, int(parent_id)) != "const":
                continue
            try:
                flatten_tensor_constant(
                    graph.G.nodes[int(parent_id)].get("constant")
                )
            except (TypeError, ValueError):
                return False
        return True

    fusible = {
        node_id
        for node_id in graph.G
        if _operation(graph, node_id) in profile.fusible_ops
        and has_only_numeric_constant_operands(int(node_id))
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
        # A component member that is also a graph root has no consumer at
        # all -- inside the region or out -- which is exactly what "root"
        # means for this graph. That makes it a program output by
        # definition, not something to drop for lack of a successor edge.
        # graph_express2.ProcessGraph.build_from_expression records roots
        # this way without also inserting an explicit "return" node.
        for node_id in nodes:
            if node_id in output_names:
                continue
            if node_id in graph.roots:
                output_names[node_id] = f"value_{node_id}"
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
        shape = tensor.get("shape")
        metadata[value_id] = Meta(
            shape=tuple(shape) if shape is not None else None,
            dtype=tensor.get("dtype"),
            device=tensor.get("device"),
        )

    emitted_tensor_constants: set[int] = set()
    descriptor_derivations: dict[int, dict[str, Any]] = {}

    def append_tensor_constant(
        parent_id: int,
        parent_data: Mapping[str, Any],
    ) -> None:
        if parent_id in emitted_tensor_constants:
            return
        tensor = parent_data.get("tensor") or {}
        shape = tensor.get("shape")
        attrs = {
            key: copy.deepcopy(value)
            for key, value in (parent_data.get("attributes") or {}).items()
            if key != "creation_op"
        }
        constant = parent_data.get("constant")
        # Structural constants can retain their payload in the provenance
        # attributes while the syntax-level ``constant`` slot is merely the
        # placeholder ``None`` from the authored AST.  Do not erase a known
        # payload during ProcessGraph -> FusedProgram transcription.
        if constant is not None or "values" not in attrs:
            attrs["values"] = copy.deepcopy(constant)
        if shape is None and attrs.get("shape") is not None:
            shape = tuple(attrs["shape"])
        dtype = tensor.get("dtype")
        if shape is None and attrs.get("values") is not None:
            try:
                literal = np.asarray(attrs["values"])
            except (TypeError, ValueError):
                literal = None
            if literal is not None and literal.dtype.kind in "biufc":
                shape = tuple(map(int, literal.shape))
                dtype = dtype or str(literal.dtype)
        metadata[parent_id] = Meta(
            shape=tuple(shape) if shape is not None else None,
            dtype=dtype,
            device=tensor.get("device"),
        )
        steps.append(
            OpStep(
                step_id=len(steps),
                op_name="tensor_from_list",
                input_ids=[],
                attrs=attrs,
                result_id=parent_id,
            )
        )
        emitted_tensor_constants.add(parent_id)

    for node_id in region.node_ids:
        data = graph.G.nodes[node_id]
        raw_op = _operation(graph, node_id)
        parents = [
            (int(parent_id), role)
            for parent_id, role in (data.get("parents") or ())
            if str(role).casefold() not in {
                "callee", "func", "function", "definition", "operator",
                "operator_reference",
            }
        ]
        # ``max``/``min`` name both Python's binary scalar operations and
        # tensor axis reductions.  Arity disambiguates them at this semantic
        # boundary: a reduction consumes one tensor; a two-parent node is the
        # ordinary elementwise binary operation and may legitimately carry a
        # scalar constant operand (for example ``max(speed, 1e-30)`` in the
        # managed-dt controller).
        node_attributes = dict(data.get("attributes") or {})
        reduction = raw_op in AXIS_REDUCTION_FOLDS and (
            len(parents) == 1
            or "axis" in node_attributes
            or "dim" in node_attributes
        )
        if reduction and len(parents) > 1:
            tensor_parents = [
                parent
                for parent in parents
                if _operation(graph, parent[0]) != "const"
            ]
            if len(tensor_parents) == 1:
                # The dimension/keepdim literals remain in the ProcessGraph
                # provenance, while the numeric reduction consumes only its
                # tensor receiver.  Their values are already recorded in the
                # operation attributes by the concordance.
                parents = tensor_parents
        # The builder is a faithful transcriber, not a translator: an op that is
        # neither a fused-elementwise op nor an axis reduction (a reshape/view/
        # cast/native kernel) is emitted under its own name with its operands
        # and attributes intact, exactly as reductions are. Its *semantics* --
        # a reshape being a view, say -- are the SSA-stage translator's job, not
        # this adapter's. Only genuine elementwise ops are canonicalized (and
        # only they carry the right_scalar operand form).
        if reduction:
            elementwise = False
            op = raw_op
        elif raw_op in {"min", "max"} and len(parents) == 2:
            elementwise = True
            op = {"min": "minimum", "max": "maximum"}[raw_op]
        else:
            try:
                op = canonical_elementwise_op(raw_op)[0]
                elementwise = True
            except KeyError:
                op = raw_op
                elementwise = False
        if elementwise:
            expected_arity = 1 if op in ELEMENTWISE_UNARY else 2
            # A resolved Python call keeps its callable/name parent in the
            # ProcessGraph for provenance.  Some graph normalizations label
            # that edge ``operand`` rather than ``callee``; numeric IR arity is
            # the reliable semantic boundary.  Remove only surplus Load/Name
            # references, never an Input or computed value.
            while len(parents) > expected_arity:
                callable_position = next((
                    index
                    for index, (parent_id, _role) in enumerate(parents)
                    if str(_operation(graph, parent_id)).casefold()
                    in {"load", "name"}
                ), None)
                if callable_position is None:
                    break
                parents.pop(callable_position)
        structural = not reduction and not elementwise
        value_parents: list[int] = []
        scalar_parent: tuple[int, Any] | None = None
        for parent_id, _role in parents:
            parent_data = graph.G.nodes[parent_id]
            if _operation(graph, parent_id) == "const":
                constant = parent_data.get("constant")
                scalar = uniform_tensor_constant(constant)
                # A non-elementwise op has no right_scalar operand slot; keep
                # every constant parent as a plain constant input so the op's
                # arguments (a reshape's target extent, a pad's width) survive
                # verbatim for the translator to interpret.
                if scalar is not None and not structural:
                    if scalar_parent is not None:
                        # FusedProgram represents a binary scalar operand in
                        # right_scalar, so two constant operands cannot both
                        # occupy that slot. Keep the first as an ordinary
                        # tensor constructor and use the second as the scalar;
                        # this preserves operand order without folding graph
                        # semantics inside the ProcessGraph adapter.
                        previous_id, _previous_scalar = scalar_parent
                        append_tensor_constant(
                            previous_id,
                            graph.G.nodes[previous_id],
                        )
                        value_parents.append(previous_id)
                    scalar_parent = (parent_id, scalar)
                else:
                    append_tensor_constant(parent_id, parent_data)
                    value_parents.append(parent_id)
            else:
                value_parents.append(parent_id)
        attrs: dict[str, Any] = (
            copy.deepcopy(dict(data.get("attributes") or {}))
            if reduction or structural else {}
        )
        if reduction and "axis" not in attrs and "dim" in attrs:
            attrs["axis"] = attrs["dim"]
        if scalar_parent is not None:
            if reduction:
                raise ValueError(f"{op} cannot consume a scalar constant operand")
            if len(value_parents) == 0:
                # A unary op whose only operand is a constant (log(const)) has a
                # tensor value operand, not the right-hand scalar of a binary
                # a-OP-scalar form. Keep the constant as an ordinary tensor
                # constant input rather than forcing it into the scalar slot.
                append_tensor_constant(
                    scalar_parent[0], graph.G.nodes[scalar_parent[0]]
                )
                value_parents.append(scalar_parent[0])
            elif len(value_parents) != 1:
                raise ValueError(f"{op} has an invalid scalar operand layout")
            else:
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
        current = metadata.get(node_id)
        exact_shape = None
        exact_dtype = None
        exact_source = None
        if op in {"unsqueeze", "squeeze"} and value_parents:
            source_id = int(value_parents[0])
            source_meta = metadata.get(source_id)
            if source_meta is not None and source_meta.shape is not None:
                axis = None
                if len(value_parents) > 1:
                    axis_data = graph.G.nodes[value_parents[1]]
                    axis_value = axis_data.get("constant")
                    if axis_value is None:
                        axis_value = (
                            axis_data.get("attributes") or {}
                        ).get("values")
                    uniform_axis = uniform_tensor_constant(axis_value)
                    if uniform_axis is not None and float(uniform_axis).is_integer():
                        axis = int(uniform_axis)
                source_shape = list(map(int, source_meta.shape))
                if op == "unsqueeze" and axis is not None:
                    normalized = axis if axis >= 0 else axis + len(source_shape) + 1
                    if 0 <= normalized <= len(source_shape):
                        source_shape.insert(normalized, 1)
                        exact_shape = tuple(source_shape)
                elif op == "squeeze":
                    if axis is None:
                        exact_shape = tuple(dim for dim in source_shape if dim != 1)
                    else:
                        normalized = axis if axis >= 0 else axis + len(source_shape)
                        if (
                            0 <= normalized < len(source_shape)
                            and source_shape[normalized] == 1
                        ):
                            del source_shape[normalized]
                            exact_shape = tuple(source_shape)
                if exact_shape is not None:
                    exact_dtype = source_meta.dtype
                    exact_source = source_id
                    descriptor_derivations[int(node_id)] = {
                        "operation": op,
                        "source_id": source_id,
                        "shape": exact_shape,
                        "dtype": exact_dtype,
                    }
        if (
            exact_shape is not None
            or current is None
            or current.shape is None
            or current.dtype is None
        ):
            known = [
                metadata[parent_id]
                for parent_id in value_parents
                if parent_id in metadata
                and metadata[parent_id].shape is not None
            ]
            inferred_shape = None
            if known and elementwise:
                try:
                    inferred_shape = tuple(np.broadcast_shapes(*(
                        tuple(item.shape) for item in known
                    )))
                except ValueError:
                    inferred_shape = None
            inferred_dtype = next(
                (item.dtype for item in known if item.dtype is not None), None
            )
            if exact_shape is not None:
                inferred_shape = exact_shape
                inferred_dtype = exact_dtype
            if op in {
                "less", "less_equal", "greater", "greater_equal", "equal",
                "not_equal", "logical_and", "logical_or", "logical_not",
                "isfinite", "isinf", "isnan",
            }:
                inferred_dtype = "bool"
            if inferred_shape is not None or inferred_dtype is not None:
                metadata[node_id] = Meta(
                    shape=(
                        inferred_shape
                        if inferred_shape is not None
                        else (current.shape if current is not None else None)
                    ),
                    dtype=(
                        inferred_dtype
                        if inferred_dtype is not None
                        else (current.dtype if current is not None else None)
                    ),
                    device=(current.device if current is not None else None),
                    source_id=exact_source,
                )

    produced_ids = {int(step.result_id) for step in steps}
    consumed_ids = {
        int(value_id) for step in steps for value_id in step.input_ids
    }
    live_feeds = (
        consumed_ids | set(map(int, dict(region.outputs).values()))
    ) & set(map(int, region.input_ids)) - produced_ids
    return FusedProgram(
        version=1,
        # Callable/name references removed from numeric op arity must not
        # survive as phantom buffer parameters merely because the structural
        # deployment boundary listed them before transcription.
        feeds=live_feeds,
        steps=steps,
        outputs=dict(region.outputs),
        meta=metadata,
        extras=(
            {"descriptor_derivations": descriptor_derivations}
            if descriptor_derivations else {}
        ),
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
