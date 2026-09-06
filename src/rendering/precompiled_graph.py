"""Observe compiler IR topology with free-running CUDA spring physics.

The visualizer consumes a small immutable graph snapshot.  Front ends are not
special-cased by the renderer: AST, SymPy, GLSL ingestion, and other sources
first reach ProcessGraph/precompile as usual, while completed FusedProgram,
SSA, DualIR/AOT, Nodus GraphIR, and serialized graph packages are adapted to
the same :class:`VisualGraph` contract.

The active evolution-metagraph launcher uses resident Torch CUDA BoundSpring
state and append-only topology.  Compiler events cross a one-way unbounded
relay into a latest-value topology page; neither physics nor display can exert
backpressure on compilation. Physics publishes optional CUDA observation pages
and drops observations when display lags. OpenGL imports those pages directly
into a persistent position VBO; indexed edge shaders read the same allocation,
so neither display latency nor host geometry construction can pause evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import deque
import math
import queue
import re
import statistics
import threading
import time
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np

from .opengl_render.api import LineLayer, PointLayer


@dataclass(frozen=True, slots=True)
class VisualNode:
    key: str
    label: str
    kind: str = "operation"
    level: int | None = None
    group: str = ""
    network: str = ""
    source_scope: tuple[str, ...] = ()
    source_class: str | None = None
    schedule_group: int | None = None
    execution_groups: tuple[int, ...] = ()
    execution_kind: str | None = None
    control_path: tuple[str, ...] = ()
    loop_depth: int = 0
    branch: str | None = None
    deployment_region: int | None = None
    deployment_lanes: tuple[int, ...] = ()
    state: str = "live"


@dataclass(frozen=True, slots=True)
class VisualEdge:
    source: str
    target: str
    role: str = "data"


@dataclass(frozen=True, slots=True)
class VisualGraph:
    nodes: tuple[VisualNode, ...]
    edges: tuple[VisualEdge, ...]
    source_kind: str
    revision: int = 0

    def __post_init__(self) -> None:
        keys = {node.key for node in self.nodes}
        if len(keys) != len(self.nodes):
            raise ValueError("visual graph node keys must be unique")
        missing = {
            endpoint
            for edge in self.edges
            for endpoint in (edge.source, edge.target)
            if endpoint not in keys
        }
        if missing:
            raise ValueError(f"visual graph edges reference missing nodes: {missing}")


@dataclass(frozen=True, slots=True)
class VisualGraphDelta:
    """One append/update publication from the evolution projector."""

    revision: int
    source_kind: str
    nodes_added: tuple[VisualNode, ...] = ()
    nodes_updated: tuple[VisualNode, ...] = ()
    edges_added: tuple[VisualEdge, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.nodes_added or self.nodes_updated or self.edges_added)


class VisualGraphAccumulator:
    """Mutable append-only membership; materialize only for rare consumers."""

    def __init__(self, source_kind: str = "evolution-metagraph") -> None:
        self.source_kind = source_kind
        self.revision = 0
        self.nodes: list[VisualNode] = []
        self.edges: list[VisualEdge] = []
        self._node_index: dict[str, int] = {}
        self._edge_keys: set[tuple[str, str, str]] = set()

    def apply(self, delta: VisualGraphDelta) -> None:
        self.source_kind = delta.source_kind
        self.revision = max(self.revision, int(delta.revision))
        for node in delta.nodes_added:
            if node.key in self._node_index:
                continue
            self._node_index[node.key] = len(self.nodes)
            self.nodes.append(node)
        for node in delta.nodes_updated:
            ordinal = self._node_index.get(node.key)
            if ordinal is not None:
                self.nodes[ordinal] = node
        for edge in delta.edges_added:
            key = (edge.source, edge.target, edge.role)
            if key not in self._edge_keys:
                self._edge_keys.add(key)
                self.edges.append(edge)

    def graph(self) -> VisualGraph:
        return VisualGraph(
            tuple(self.nodes), tuple(self.edges), self.source_kind, self.revision
        )


def _coalesce_visual_deltas(deltas: Sequence[VisualGraphDelta]) -> VisualGraphDelta:
    """Merge ordered publications without losing add-then-update semantics."""

    if not deltas:
        return VisualGraphDelta(0, "evolution-metagraph")
    added: dict[str, VisualNode] = {}
    updated: dict[str, VisualNode] = {}
    edges: dict[tuple[str, str, str], VisualEdge] = {}
    for delta in deltas:
        for node in delta.nodes_added:
            added[node.key] = node
            updated.pop(node.key, None)
        for node in delta.nodes_updated:
            if node.key in added:
                added[node.key] = node
            else:
                updated[node.key] = node
        for edge in delta.edges_added:
            edges[(edge.source, edge.target, edge.role)] = edge
    latest = deltas[-1]
    return VisualGraphDelta(
        revision=max(int(delta.revision) for delta in deltas),
        source_kind=latest.source_kind,
        nodes_added=tuple(added.values()),
        nodes_updated=tuple(updated.values()),
        edges_added=tuple(edges.values()),
    )


@dataclass(frozen=True, slots=True)
class ExpansionHotspot:
    """One source-owned branch in the observational growth census."""

    owner: str
    node_count: int
    rate: float
    relative_size: float
    depth: int
    height: int
    stages: tuple[tuple[str, int], ...]
    boundaries: tuple[str, ...]
    boundary_hint: str


@dataclass(frozen=True, slots=True)
class ExpansionReport:
    node_count: int
    edge_count: int
    hotspots: tuple[ExpansionHotspot, ...]

    def summary(self) -> str:
        if not self.hotspots:
            return f"nodes={self.node_count} edges={self.edge_count} | no attributed branches"
        top = self.hotspots[0]
        sign = "+" if top.relative_size >= 0.0 else ""
        return (
            f"nodes={self.node_count} edges={self.edge_count} | "
            f"{top.owner}: {top.node_count} ({sign}{top.relative_size:.2f} rel) "
            f"{top.rate:.1f}/s d={top.depth} h={top.height} "
            f"boundary={','.join(top.boundaries) if top.boundaries else 'unmatched'}"
            f" hint={top.boundary_hint}"
        )


@dataclass
class ExpansionTelemetry:
    """Incremental, non-authoritative attribution beside compiler evolution.

    Ownership flows along observed links and handoffs, but never back into the
    graph or its spring system.  The resulting counts drive reports and a
    renderer-only haze channel.
    """

    top_k: int = 8
    eager_ownership: bool = False
    _graphs: dict[str, str] = field(default_factory=dict)
    _nodes: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    _attributes: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    _out: dict[str, set[str]] = field(default_factory=dict)
    _incoming: dict[str, set[str]] = field(default_factory=dict)
    _owners: dict[str, set[str]] = field(default_factory=dict)
    _owner_nodes: dict[str, set[str]] = field(default_factory=dict)
    _roots: dict[str, set[str]] = field(default_factory=dict)
    _last_counts: dict[str, int] = field(default_factory=dict)
    _rates: dict[str, float] = field(default_factory=dict)
    _last_sample: float = field(default_factory=time.perf_counter)
    _ownership_dirty: bool = False

    @staticmethod
    def _key(ref: Any) -> str:
        return f"{ref.graph_id}/{ref.local_id}"

    @staticmethod
    def _direct_owners(attributes: Mapping[str, Any]) -> tuple[str, ...]:
        owners = []
        source_class = attributes.get("source_class")
        if source_class:
            owners.append(f"class {source_class}")
        scope = tuple(attributes.get("source_scope") or ())
        if len(scope) > 1:
            owners.append("scope " + ".".join(map(str, scope)))
        elif scope and not source_class:
            owners.append("scope " + ".".join(map(str, scope)))
        return tuple(owners)

    def apply(self, event: Any) -> None:
        if event.kind == "graph-open" and event.graph is not None:
            self._graphs[event.graph.id] = str(event.graph.stage)
            return
        if event.component is None:
            return
        target = self._key(event.component)
        if event.kind in {"component-spawn", "component-update"}:
            attributes = dict(event.detail.get("attributes") or {})
            label = str(event.detail.get("label") or event.component.local_id)
            kind = str(event.detail.get("kind") or "component")
            self._nodes[target] = (event.component.graph_id, label, kind)
            self._attributes[target] = attributes
            for owner in self._direct_owners(attributes):
                self._roots.setdefault(owner, set()).add(target)
                if self.eager_ownership:
                    self._add_owners(target, {owner})
                else:
                    self._ownership_dirty = True
        if event.sources and event.kind in {
            "component-spawn", "component-update",
            "component-link", "component-handoff",
        }:
            for source_ref in event.sources:
                source = self._key(source_ref)
                self._out.setdefault(source, set()).add(target)
                self._incoming.setdefault(target, set()).add(source)
                if self.eager_ownership:
                    self._add_owners(target, self._owners.get(source, set()))
                else:
                    self._ownership_dirty = True

    def _ensure_ownership(self) -> None:
        """Propagate attribution once for a whole event batch, not per edge."""

        if not self._ownership_dirty:
            return
        self._owners.clear()
        self._owner_nodes.clear()
        for owner, roots in self._roots.items():
            for root in roots:
                self._add_owners(root, {owner})
        self._ownership_dirty = False

    def _add_owners(self, key: str, owners: set[str]) -> None:
        if not owners:
            return
        queue = deque(((key, set(owners)),))
        while queue:
            node, incoming = queue.popleft()
            current = self._owners.setdefault(node, set())
            added = incoming - current
            if not added:
                continue
            current.update(added)
            for owner in added:
                self._owner_nodes.setdefault(owner, set()).add(node)
            for child in self._out.get(node, ()):
                queue.append((child, added))

    def _counts(self) -> dict[str, int]:
        self._ensure_ownership()
        return {
            owner: len(self._owner_nodes.get(owner, ()))
            for owner in self._roots
        }

    def _sample_rates(self, counts: Mapping[str, int]) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_sample
        if elapsed < 0.2:
            return
        for owner, count in counts.items():
            instant = max(0, count - self._last_counts.get(owner, 0)) / elapsed
            previous = self._rates.get(owner, instant)
            self._rates[owner] = previous * 0.65 + instant * 0.35
        self._last_counts = dict(counts)
        self._last_sample = now

    def _extent(self, owned: set[str]) -> tuple[int, int]:
        """Return SCC-safe maximum forward depth and reverse height."""

        if not owned:
            return 0, 0
        try:
            import networkx as nx

            graph = nx.DiGraph()
            graph.add_nodes_from(owned)
            graph.add_edges_from(
                (source, target)
                for source in owned
                for target in self._out.get(source, ())
                if target in owned
            )
            condensed = nx.condensation(graph)
            order = list(nx.topological_sort(condensed))
            depth = {node: 0 for node in order}
            for node in order:
                for child in condensed.successors(node):
                    depth[child] = max(depth[child], depth[node] + 1)
            height = {node: 0 for node in order}
            for node in reversed(order):
                for parent in condensed.predecessors(node):
                    height[parent] = max(height[parent], height[node] + 1)
            return max(depth.values(), default=0), max(height.values(), default=0)
        except Exception:
            return 0, 0

    def report(self) -> ExpansionReport:
        counts = self._counts()
        self._sample_rates(counts)
        median = statistics.median(counts.values()) if counts else 0.0
        ranked = sorted(
            counts,
            key=lambda owner: (counts[owner], self._rates.get(owner, 0.0), owner),
            reverse=True,
        )[: max(1, int(self.top_k))]
        hotspots = []
        for owner in ranked:
            owned = {
                key for key, owners in self._owners.items() if owner in owners
            }
            stage_counts: dict[str, int] = {}
            for key in owned:
                graph_id = self._nodes.get(key, (key.split("/", 1)[0], "", ""))[0]
                stage = self._graphs.get(graph_id, graph_id)
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            depth, height = self._extent(owned)
            boundaries = tuple(sorted({
                str(rule)
                for key in owned
                if (rule := self._attributes.get(key, {}).get("boundary_rule"))
            }))
            relative = math.log2((counts[owner] + 1.0) / (median + 1.0))
            hotspots.append(ExpansionHotspot(
                owner,
                counts[owner],
                self._rates.get(owner, 0.0),
                relative,
                depth,
                height,
                tuple(sorted(stage_counts.items())),
                boundaries,
                self.boundary_hint(owner),
            ))
        return ExpansionReport(len(self._nodes), sum(map(len, self._out.values())), tuple(hotspots))

    def boundary_hint(self, owner: str) -> str:
        roots = self._roots.get(owner, ())
        candidates = []
        for key in roots:
            attributes = self._attributes.get(key, {})
            language = str(attributes.get("source_language") or "python")
            scope = tuple(map(str, attributes.get("source_scope") or ()))
            if scope:
                candidates.append("/".join((language, *scope)))
        if candidates:
            return max(candidates, key=lambda value: (value.count("/"), value))
        if owner.startswith("class "):
            return "python/" + owner.removeprefix("class ")
        if owner.startswith("scope "):
            return "python/" + owner.removeprefix("scope ").replace(".", "/")
        return "python"

    def haze_scores(self) -> dict[str, float]:
        report = self.report()
        scores: dict[str, float] = {}
        for hotspot in report.hotspots:
            relative = max(-1.0, min(1.0, hotspot.relative_size / 3.0))
            absolute = min(1.0, math.log2(hotspot.node_count + 1.0) / 12.0)
            score = max(relative, absolute) if relative >= 0.0 else relative
            roots = self._roots.get(hotspot.owner, ())
            definition_roots = {
                key for key in roots
                if self._nodes.get(key, ("", "", ""))[2] == "ClassDef"
            }
            for key in definition_roots or set(roots):
                scores[key] = score
        return scores


class ExpansionLimitExceeded(RuntimeError):
    """A live translation crossed its explicitly configured safety budget."""

    def __init__(
        self,
        message: str,
        *,
        owner: str,
        boundary_hint: str,
        node_count: int,
        depth: int | None,
        height: int | None,
        stages: Mapping[str, int],
    ) -> None:
        super().__init__(message)
        self.owner = owner
        self.boundary_hint = boundary_hint
        self.node_count = int(node_count)
        self.depth = depth
        self.height = height
        self.stages = dict(stages)


@dataclass
class ExpansionEmergencyClamp:
    """Hard compiler-event circuit breaker for runaway graph evolution.

    The clamp subscribes beside the one-way visual relay.  It can stop the
    isolated compiler attempt, but never edits topology, positions, forces,
    renderer cadence, or the spring scheduler.
    """

    max_depth: int = 512
    max_height: int = 512
    max_nodes_per_branch: int = 50_000
    check_interval: int = 64
    telemetry: ExpansionTelemetry = field(
        default_factory=lambda: ExpansionTelemetry(top_k=32, eager_ownership=True)
    )
    _edge_events: int = 0

    def __post_init__(self) -> None:
        for name in ("max_depth", "max_height", "max_nodes_per_branch"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.check_interval) <= 0:
            raise ValueError("check_interval must be positive")

    def __call__(self, event: Any) -> None:
        self.telemetry.apply(event)
        if event.kind not in {"component-link", "component-handoff"}:
            return
        self._edge_events += 1
        for owner, nodes in self.telemetry._owner_nodes.items():
            if len(nodes) > self.max_nodes_per_branch:
                depth, height = self.telemetry._extent(nodes)
                self._raise(owner, len(nodes), depth, height)
        # A branch cannot have topological extent greater than its node count,
        # so small branches need no SCC condensation at all.
        minimum_extent = min(self.max_depth, self.max_height) + 1
        for owner, nodes in self.telemetry._owner_nodes.items():
            if len(nodes) < minimum_extent:
                continue
            depth, height = self.telemetry._extent(nodes)
            if depth > self.max_depth or height > self.max_height:
                self._raise(owner, len(nodes), depth, height)

    def final_check(self) -> None:
        """Check the tail smaller than ``check_interval`` before success."""

        for owner, nodes in self.telemetry._owner_nodes.items():
            depth, height = self.telemetry._extent(nodes)
            if (
                len(nodes) > self.max_nodes_per_branch
                or depth > self.max_depth
                or height > self.max_height
            ):
                self._raise(owner, len(nodes), depth, height)

    def _raise(
        self,
        owner: str,
        node_count: int,
        depth: int | None,
        height: int | None,
    ) -> None:
        stage_counts: dict[str, int] = {}
        for key in self.telemetry._owner_nodes.get(owner, ()):
            graph_id = self.telemetry._nodes.get(
                key, (key.split("/", 1)[0], "", "")
            )[0]
            stage = self.telemetry._graphs.get(graph_id, graph_id)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        census = ", ".join(
            f"{stage}:{count}" for stage, count in sorted(stage_counts.items())
        )
        boundary_hint = self.telemetry.boundary_hint(owner)
        message = (
            "translation growth emergency clamp tripped for "
            f"{owner}: nodes={node_count}/{self.max_nodes_per_branch}, "
            f"depth={depth if depth is not None else 'pending'}/{self.max_depth}, "
            f"height={height if height is not None else 'pending'}/{self.max_height}; "
            f"stages=[{census}]. Raise the limits only with the explicit CLI "
            "growth-limit boost after inspecting this branch. Suggested "
            f"boundary directory: {boundary_hint}."
        )
        raise ExpansionLimitExceeded(
            message,
            owner=owner,
            boundary_hint=boundary_hint,
            node_count=node_count,
            depth=depth,
            height=height,
            stages=stage_counts,
        )


class LiveEvolutionEventBuffer:
    """Unbounded one-way compiler-event transport with no backpressure."""

    def __init__(self, max_backlog: int = 256) -> None:
        if int(max_backlog) <= 0:
            raise ValueError("max_backlog must be positive")
        # Retained only for API/telemetry compatibility. It is never enforced.
        self.max_backlog = int(max_backlog)
        self._events = deque()
        self._active = False
        self._closed = False

    def publish(self, event: Any) -> None:
        # deque append is atomic under CPython and never waits for the
        # consumer. There is intentionally no mutex or capacity condition.
        if not self._closed:
            self._events.append(event)

    def activate(self) -> None:
        # Normalize any replay batch once before consumption begins. Live
        # isolated-compiler delivery is already authoritative and ordered.
        if not self._active and len(self._events) > 1:
            self._events = deque(sorted(
                self._events,
                key=lambda item: int(getattr(item, "sequence", 0)),
            ))
        self._active = True

    def pop(self) -> Any | None:
        try:
            return self._events.popleft()
        except IndexError:
            return None

    def close(self) -> None:
        self._closed = True

    def __len__(self) -> int:
        return len(self._events)


def _evolution_event_summary(event: Any | None) -> str:
    if event is None:
        return "waiting for compiler"
    target = ""
    if event.component is not None:
        target = f" {event.component.graph_id}/{event.component.local_id}"
    elif event.graph is not None:
        target = f" {event.graph.id}"
    label = event.detail.get("label") if event.detail else None
    if label:
        target += f" {label}"
    if event.kind == "graph-schedule-finalized":
        target += (
            f" {event.detail.get('method', 'retained')}"
            f" groups={len(event.detail.get('groups', ()))}"
        )
    elif event.kind == "compiled-execution-plan-finalized":
        frames = tuple(event.detail.get("frames", ()))
        loops = sum(
            str(frame.get("kind", "")).startswith("loop-")
            for frame in frames
        )
        deployments = sum(
            str(frame.get("kind", "")).startswith("deployment")
            for frame in frames
        )
        target += f" frames={len(frames)} loops={loops} deployments={deployments}"
    return f"#{event.sequence} {event.kind}{target}"


def _key(value: Any, *, prefix: str = "") -> str:
    return prefix + str(value)


def _visual_from_networkx(graph: Any, *, revision: int = 0) -> VisualGraph:
    nx_graph = getattr(graph, "G", graph)
    authored_levels = tuple(sorted({
        int(data["level"])
        for _node_id, data in nx_graph.nodes(data=True)
        if data.get("level") is not None
    }))
    group_by_level = {
        level: ordinal for ordinal, level in enumerate(authored_levels)
    }
    nodes = tuple(
        VisualNode(
            key=_key(node_id),
            label=str(data.get("label") or data.get("op") or data.get("type") or node_id),
            kind=str(data.get("type") or data.get("op") or "operation"),
            level=(None if data.get("level") is None else int(data["level"])),
            source_scope=tuple(data.get("source_scope") or ()),
            source_class=data.get("source_class"),
            schedule_group=(
                None
                if data.get("level") is None
                else group_by_level[int(data["level"])]
            ),
        )
        for node_id, data in nx_graph.nodes(data=True)
    )
    edges = tuple(
        VisualEdge(
            _key(source),
            _key(target),
            str(data.get("role") or data.get("type") or "data"),
        )
        for source, target, data in nx_graph.edges(data=True)
    )
    return VisualGraph(nodes, edges, "process-graph", revision)


def _visual_from_fused(program: Any) -> VisualGraph:
    feed_ids = sorted(int(value) for value in program.feeds)
    nodes: dict[int, VisualNode] = {
        value_id: VisualNode(_key(value_id), f"feed {value_id}", "feed", 0)
        for value_id in feed_ids
    }
    edges: list[VisualEdge] = []
    for ordinal, step in enumerate(program.steps, start=1):
        result_id = int(step.result_id)
        nodes[result_id] = VisualNode(
            _key(result_id),
            str(step.op_name),
            "operation",
            int(step.level) if step.level is not None else ordinal,
        )
        for position, input_id in enumerate(step.input_ids):
            input_id = int(input_id)
            nodes.setdefault(
                input_id,
                VisualNode(_key(input_id), f"value {input_id}", "value"),
            )
            edges.append(VisualEdge(_key(input_id), _key(result_id), f"arg{position}"))
    for output_name, value_id in program.outputs.items():
        output_key = f"output:{output_name}"
        nodes_key = int(value_id)
        nodes.setdefault(
            nodes_key,
            VisualNode(_key(nodes_key), f"value {nodes_key}", "value"),
        )
        edges.append(VisualEdge(_key(nodes_key), output_key, "output"))
        nodes[output_key] = VisualNode(output_key, str(output_name), "output")  # type: ignore[index]
    return VisualGraph(tuple(nodes.values()), tuple(edges), "fused-program")


def _visual_from_ssa(module: Any) -> VisualGraph:
    nodes: dict[str, VisualNode] = {}
    edges: list[VisualEdge] = []
    for function_name, function in module.functions.items():
        prefix = f"{function_name}:"
        for argument in function.args:
            key = _key(argument.id, prefix=prefix)
            nodes[key] = VisualNode(key, argument.name(), "argument", 0)
        block_tail: dict[str, str] = {}
        for level, (block_name, block) in enumerate(function.blocks.items(), start=1):
            previous: str | None = None
            for instruction in block.instrs:
                result_key = _key(instruction.res.id, prefix=prefix)
                nodes[result_key] = VisualNode(
                    result_key, str(instruction.op), "operation", level
                )
                for position, argument in enumerate(instruction.args):
                    argument_key = _key(argument.id, prefix=prefix)
                    nodes.setdefault(
                        argument_key,
                        VisualNode(argument_key, argument.name(), "value"),
                    )
                    edges.append(VisualEdge(argument_key, result_key, f"arg{position}"))
                if previous is not None and not instruction.args:
                    edges.append(VisualEdge(previous, result_key, "order"))
                previous = result_key
            if previous is not None:
                block_tail[block_name] = previous
        for block_name, block in function.blocks.items():
            source = block_tail.get(block_name)
            if source is None:
                continue
            for successor in block.successors:
                target_block = function.blocks.get(successor)
                if target_block and target_block.instrs:
                    target = _key(target_block.instrs[0].res.id, prefix=prefix)
                    edges.append(VisualEdge(source, target, "control"))
    return VisualGraph(tuple(nodes.values()), tuple(edges), "ssa")


_NODUS_NODE = re.compile(r'^(n\d+)\s*=\s*tensor_node\("((?:[^"\\]|\\.)*)"\);', re.MULTILINE)
_NODUS_OUTPUT = re.compile(r'^(o\d+)\s*=\s*tensor_output\((n\d+),', re.MULTILINE)
_NODUS_INPUT = re.compile(r'^(i\d+_\d+)\s*=\s*tensor_input\((n\d+),\s*"([^"]*)"\);', re.MULTILINE)
_NODUS_CONNECT = re.compile(r'^connect\((o\d+),\s*(i\d+_\d+)\);', re.MULTILINE)


def _visual_from_nodus(source: str) -> VisualGraph:
    labels = dict(_NODUS_NODE.findall(source))
    output_owner = dict(_NODUS_OUTPUT.findall(source))
    inputs = {port: (owner, role) for port, owner, role in _NODUS_INPUT.findall(source)}
    nodes = tuple(
        VisualNode(key, bytes(label, "utf-8").decode("unicode_escape"), "operation")
        for key, label in labels.items()
    )
    edges = []
    for output, input_port in _NODUS_CONNECT.findall(source):
        if output in output_owner and input_port in inputs:
            target, role = inputs[input_port]
            edges.append(VisualEdge(output_owner[output], target, role))
    return VisualGraph(nodes, tuple(edges), "nodus-graph-ir")


def _visual_from_mapping(package: Mapping[str, Any]) -> VisualGraph:
    if "table" in package and isinstance(package["table"], Sequence):
        rows = list(package["table"])
        nodes = tuple(
            VisualNode(
                _key(row.get("id")),
                str(row.get("label") or row.get("type") or row.get("id")),
                str(row.get("type") or "operation"),
            )
            for row in rows
        )
        keys = {node.key for node in nodes}
        edges = tuple(
            VisualEdge(_key(parent), _key(row.get("id")), "parent")
            for row in rows
            for parent in row.get("parents", ())
            if _key(parent) in keys
        )
        return VisualGraph(nodes, edges, "process-graph-summary", int(package.get("revision", 0)))
    if "steps" in package and "feeds" in package and "outputs" in package:
        from ..common.tensors.fused_ir import FusedProgram, OpStep

        steps = [step if hasattr(step, "op_name") else OpStep(**step) for step in package["steps"]]
        return _visual_from_fused(FusedProgram(
            version=int(package.get("version", 1)),
            feeds=set(package["feeds"]),
            steps=steps,
            outputs=dict(package["outputs"]),
        ))
    if "nodes" in package and "edges" in package and isinstance(package["nodes"], Sequence):
        raw_nodes = list(package["nodes"])
        nodes = tuple(
            VisualNode(
                _key(item.get("id", index)),
                str(item.get("label") or item.get("op") or item.get("type") or index),
                str(item.get("kind") or item.get("type") or "operation"),
                item.get("level"),
            )
            for index, item in enumerate(raw_nodes)
        )
        edges = tuple(
            VisualEdge(
                _key(item.get("source", item.get("src"))),
                _key(item.get("target", item.get("dst"))),
                str(item.get("role") or "data"),
            )
            for item in package["edges"]
        )
        return VisualGraph(nodes, edges, str(package.get("kind", "serialized-ir")), int(package.get("revision", 0)))
    raise TypeError("mapping is not a recognized precompiled graph package")


def visual_graph_from_ir(package: Any) -> VisualGraph:
    """Normalize a live or completed compiler IR package for visualization."""

    if isinstance(package, VisualGraph):
        return package
    if isinstance(package, Mapping):
        return _visual_from_mapping(package)
    if isinstance(package, str) and "tensor_node(" in package:
        return _visual_from_nodus(package)
    if hasattr(package, "graph") and hasattr(package, "revision"):
        return _visual_from_networkx(
            package.graph,
            revision=int(package.revision),
        )
    if hasattr(package, "snapshot") and callable(package.snapshot):
        snapshot = package.snapshot()
        graph = getattr(snapshot, "graph", snapshot)
        return _visual_from_networkx(graph, revision=int(getattr(snapshot, "revision", 0)))
    if hasattr(package, "module") and hasattr(package.module, "functions"):
        return _visual_from_ssa(package.module)
    if hasattr(package, "functions") and isinstance(package.functions, Mapping):
        return _visual_from_ssa(package)
    if hasattr(package, "compiled_shell_program"):
        return visual_graph_from_ir(package.compiled_shell_program)
    if hasattr(package, "program") and hasattr(package.program, "steps"):
        return _visual_from_fused(package.program)
    if hasattr(package, "steps") and hasattr(package, "feeds") and hasattr(package, "outputs"):
        return _visual_from_fused(package)
    if hasattr(package, "G") or (
        hasattr(package, "nodes") and callable(package.nodes)
        and hasattr(package, "edges") and callable(package.edges)
    ):
        return _visual_from_networkx(package)
    raise TypeError(f"unsupported IR package for graph visualization: {type(package).__name__}")


class FluxSpringVisualSimulation:
    """Thin layer adapter over FluxSpring's multi-network evolution system."""

    def __init__(self, graph: VisualGraph, *, seed: int = 0) -> None:
        from ..common.tensors.autoautograd.fluxspring.evolution import (
            MultiNetworkFluxSpring,
        )

        self.graph = graph
        self.fluxspring = MultiNetworkFluxSpring(seed=seed)
        self._keys: tuple[str, ...] = ()
        self.replace_graph(graph)

    def set_haze(self, scores: Mapping[str, float]) -> None:
        """Publish renderer-only urgency without changing spring mechanics."""

        index = {key: ordinal for ordinal, key in enumerate(self._keys)}
        self.fluxspring.set_visual_haze({
            index[key]: float(value)
            for key, value in scores.items()
            if key in index
        })

    @property
    def positions(self) -> np.ndarray:
        return self.fluxspring.positions

    @property
    def velocities(self) -> np.ndarray:
        return self.fluxspring.velocities

    def replace_graph(self, graph: VisualGraph) -> None:
        from ..common.tensors.autoautograd.fluxspring.evolution import (
            EvolutionEdge,
            EvolutionNode,
        )

        self.graph = graph
        self._keys = tuple(node.key for node in graph.nodes)
        self.fluxspring.synchronize(
            (
                EvolutionNode(
                    node.key,
                    node.network or graph.source_kind or "graph",
                    node.kind,
                    node.group or node.kind,
                )
                for node in graph.nodes
            ),
            (EvolutionEdge(edge.source, edge.target, edge.role) for edge in graph.edges),
        )

    def step(self, dt: float = 1.0 / 60.0) -> None:
        self.fluxspring.step(dt)

    def layers(self, elapsed: float) -> Mapping[str, Any]:
        frame = self.fluxspring.frame()
        if len(frame.edge_indices):
            line_positions = frame.positions[frame.edge_indices.reshape(-1)]
            line_colors = np.repeat(frame.edge_colors, 2, axis=0)
        else:
            line_positions = np.zeros((0, 3), dtype=np.float32)
            line_colors = np.zeros((0, 4), dtype=np.float32)
        return {
            "lines": LineLayer(line_positions, colors=line_colors, width=1.5),
            "points": PointLayer(
                frame.positions,
                colors=frame.colors,
                sizes_px=frame.sizes,
                size_px_default=7.0,
            ),
        }


# Compatibility for callers importing the earlier presentation class name.
SpringVisualSimulation = FluxSpringVisualSimulation


class EvolutionVisualProjector:
    """Incrementally materialize metagraph events as concurrent geometries."""

    def __init__(self) -> None:
        self._graphs: dict[str, Any] = {}
        self._graph_order: dict[str, int] = {}
        self._nodes: dict[str, VisualNode] = {}
        self._edges: dict[tuple[str, str, str], VisualEdge] = {}
        self._schedule_by_graph: dict[str, dict[str, tuple[int, int]]] = {}
        self._execution_by_component: dict[str, list[Mapping[str, Any]]] = {}
        self._revision = 0
        self._pending_nodes: dict[str, VisualNode] = {}
        self._pending_node_updates: dict[str, VisualNode] = {}
        self._pending_edges: dict[tuple[str, str, str], VisualEdge] = {}
        self.telemetry = ExpansionTelemetry()

    @staticmethod
    def _component_key(ref: Any) -> str:
        return f"{ref.graph_id}/{ref.local_id}"

    def apply(
        self, event: Any, *, materialize: bool = True,
    ) -> VisualGraph | None:
        self.telemetry.apply(event)
        if event.kind == "graph-open" and event.graph is not None:
            self._graphs[event.graph.id] = event.graph
            self._graph_order.setdefault(event.graph.id, len(self._graph_order))
        elif event.kind == "graph-schedule-finalized" and event.graph is not None:
            group_by_level = {
                int(level): int(group)
                for level, group in event.detail.get("groups", ())
            }
            schedule = {
                str(local_id): (int(raw_level), group_by_level[int(raw_level)])
                for local_id, raw_level in event.detail.get("levels", ())
            }
            self._schedule_by_graph[event.graph.id] = schedule
            for local_id, (level, schedule_group) in schedule.items():
                key = f"{event.graph.id}/{local_id}"
                node = self._nodes.get(key)
                if node is None:
                    continue
                updated = replace(
                    node,
                    level=level,
                    schedule_group=schedule_group,
                    state=(
                        "finalized" if node.state == "finalized" else "scheduled"
                    ),
                )
                self._nodes[key] = updated
                if key in self._pending_nodes:
                    self._pending_nodes[key] = updated
                else:
                    self._pending_node_updates[key] = updated
        elif (
            event.kind == "compiled-execution-plan-finalized"
            and event.graph is not None
        ):
            touched: set[str] = set()
            for raw_frame in event.detail.get("frames", ()):
                frame = dict(raw_frame)
                group = int(frame["group"])
                for graph_id, local_id in frame.get("components", ()):
                    key = f"{graph_id}/{local_id}"
                    memberships = self._execution_by_component.setdefault(key, [])
                    if not any(int(item["group"]) == group for item in memberships):
                        memberships.append(frame)
                    touched.add(key)
            for key in touched:
                node = self._nodes.get(key)
                if node is None:
                    continue
                memberships = self._execution_by_component[key]
                primary = memberships[0]
                updated = replace(
                    node,
                    execution_groups=tuple(
                        int(item["group"]) for item in memberships
                    ),
                    execution_kind=str(primary.get("kind") or "execution"),
                    control_path=tuple(primary.get("control_path") or ()),
                    loop_depth=int(primary.get("loop_depth") or 0),
                    branch=primary.get("branch"),
                    deployment_region=primary.get("deployment_region"),
                    deployment_lanes=tuple(
                        map(int, primary.get("deployment_lanes") or ())
                    ),
                    state="scheduled",
                )
                self._nodes[key] = updated
                if key in self._pending_nodes:
                    self._pending_nodes[key] = updated
                else:
                    self._pending_node_updates[key] = updated
        elif event.kind == "graph-close" and event.graph is not None:
            for key, node in tuple(self._nodes.items()):
                if node.network != event.graph.id:
                    continue
                updated = replace(node, state="finalized")
                self._nodes[key] = updated
                if key in self._pending_nodes:
                    self._pending_nodes[key] = updated
                else:
                    self._pending_node_updates[key] = updated
        elif event.kind in {"component-spawn", "component-update"} and event.component is not None:
            graph = self._graphs.get(event.component.graph_id)
            stage = "graph" if graph is None else str(graph.stage)
            self._graph_order.setdefault(
                event.component.graph_id,
                len(self._graph_order),
            )
            key = self._component_key(event.component)
            attributes = dict(event.detail.get("attributes") or {})
            existed = key in self._nodes
            previous = self._nodes.get(key)
            authored_schedule = self._schedule_by_graph.get(
                event.component.graph_id, {}
            ).get(str(event.component.local_id))
            execution_memberships = self._execution_by_component.get(key, [])
            execution_primary = (
                execution_memberships[0] if execution_memberships else None
            )
            if authored_schedule is not None:
                schedule_level, schedule_group = authored_schedule
            else:
                raw_level = attributes.get("schedule_level")
                raw_group = attributes.get("schedule_group")
                schedule_level = (
                    previous.level
                    if raw_level is None and previous is not None
                    else None if raw_level is None else int(raw_level)
                )
                schedule_group = (
                    previous.schedule_group
                    if raw_group is None and previous is not None
                    else None if raw_group is None else int(raw_group)
                )
            state = str(
                attributes.get("compiler_state")
                or (
                    previous.state
                    if previous is not None
                    else "scheduled" if authored_schedule is not None else "live"
                )
            )
            node = VisualNode(
                key=key,
                label=str(event.detail.get("label") or event.component.local_id),
                kind=str(event.detail.get("kind") or "component"),
                level=schedule_level,
                group=stage,
                network=event.component.graph_id,
                source_scope=tuple(attributes.get("source_scope") or ()),
                source_class=attributes.get("source_class"),
                schedule_group=schedule_group,
                execution_groups=tuple(
                    int(item["group"]) for item in execution_memberships
                ),
                execution_kind=(
                    None if execution_primary is None
                    else str(execution_primary.get("kind") or "execution")
                ),
                control_path=(
                    tuple(attributes.get("control_path") or ())
                    if execution_primary is None
                    else tuple(execution_primary.get("control_path") or ())
                ),
                loop_depth=(
                    int(attributes.get("loop_depth") or 0)
                    if execution_primary is None
                    else int(execution_primary.get("loop_depth") or 0)
                ),
                branch=(
                    attributes.get("branch")
                    if execution_primary is None
                    else execution_primary.get("branch")
                ),
                deployment_region=(
                    attributes.get("deployment_region", attributes.get("region_id"))
                    if execution_primary is None
                    else execution_primary.get("deployment_region")
                ),
                deployment_lanes=(
                    tuple(
                        int(lane)
                        for _region, lane in attributes.get(
                            "deployment_memberships", ()
                        )
                    )
                    if execution_primary is None
                    else tuple(map(
                        int, execution_primary.get("deployment_lanes") or ()
                    ))
                ),
                state=state,
            )
            # Compiler-phase markers describe orchestration status, not a
            # physical graph component. They have no dependency edges and
            # otherwise appear as lone presentation-colored particles.
            if node.kind != "compiler-phase":
                self._nodes[key] = node
                if existed and key not in self._pending_nodes:
                    self._pending_node_updates[key] = node
                else:
                    self._pending_nodes[key] = node
        elif event.kind in {"component-link", "component-handoff"} and event.component is not None:
            target = self._component_key(event.component)
            role = (
                "handoff"
                if event.kind == "component-handoff"
                else str(event.detail.get("role") or "data")
            )
            for source_ref in event.sources:
                source = self._component_key(source_ref)
                if source in self._nodes and target in self._nodes:
                    edge_key = (source, target, role)
                    if edge_key not in self._edges:
                        edge = VisualEdge(source, target, role)
                        self._edges[edge_key] = edge
                        self._pending_edges[edge_key] = edge
        self._revision = max(self._revision + 1, int(event.sequence) + 1)
        return self.graph() if materialize else None

    def apply_batch(self, events: Sequence[Any]) -> VisualGraph:
        """Apply one queue snapshot and materialize its final membership once."""

        for event in events:
            self.apply(event, materialize=False)
        return self.graph()

    def report(self) -> ExpansionReport:
        return self.telemetry.report()

    def haze_scores(self) -> dict[str, float]:
        return self.telemetry.haze_scores()

    def graph(self) -> VisualGraph:
        return VisualGraph(
            tuple(self._nodes.values()),
            tuple(self._edges.values()),
            "evolution-metagraph",
            self._revision,
        )

    def take_delta(self) -> VisualGraphDelta:
        """Take all membership changes since the preceding publication."""

        delta = VisualGraphDelta(
            revision=self._revision,
            source_kind="evolution-metagraph",
            nodes_added=tuple(self._pending_nodes.values()),
            nodes_updated=tuple(self._pending_node_updates.values()),
            edges_added=tuple(self._pending_edges.values()),
        )
        self._pending_nodes.clear()
        self._pending_node_updates.clear()
        self._pending_edges.clear()
        return delta


def _projection_update_process(
    request_queue: Any,
    result_queue: Any,
    stop_event: Any,
    top_k: int,
    report_hz: float,
    publish_hz: float = 30.0,
) -> None:
    """Own event projection outside the physics and OpenGL process threads."""

    projector = EvolutionVisualProjector()
    projector.telemetry.top_k = max(1, int(top_k))
    publish_dt = 1.0 / max(1.0, float(publish_hz))
    report_dt = 1.0 / max(0.1, float(report_hz))
    next_publish = 0.0
    next_report = 0.0
    last_event = None
    report = projector.report()
    dirty = False

    def publish(now: float) -> None:
        nonlocal dirty, next_publish, next_report, report
        # A short compile can open, populate, and close all of its graphs
        # before the ordinary telemetry interval expires.  Recompute on the
        # terminal compiler-run close as well, otherwise the HUD can remain
        # forever stuck at its first sample (commonly nodes=1 edges=0) even
        # though all topology deltas reached resident physics.
        terminal_compile = bool(
            last_event is not None
            and last_event.kind == "graph-close"
            and last_event.graph is not None
            and str(last_event.graph.stage) == "compiler-run"
        )
        report_changed = now >= next_report or terminal_compile
        delta = projector.take_delta()
        if report_changed:
            report = projector.report()
            next_report = now + report_dt
        # Membership crosses first. Expensive attribution/reporting is never
        # allowed to stand between a completed compiler event and physics.
        result_queue.put_nowait((
            delta, last_event, report, report_changed,
        ))
        dirty = False
        next_publish = now + publish_dt

    while not stop_event.is_set():
        now = time.perf_counter()
        timeout = (
            max(0.0, min(0.05, next_publish - now))
            if dirty else 0.05
        )
        try:
            batch = request_queue.get(timeout=timeout)
        except queue.Empty:
            batch = ()
        if batch is None:
            if dirty:
                publish(time.perf_counter())
            break
        if batch:
            saw_stop = False
            for event in batch:
                projector.apply(event, materialize=False)
                last_event = event
            dirty = True
            # Collapse every batch already waiting into the same projection
            # revision before serializing one immutable topology snapshot.
            while True:
                try:
                    newer = request_queue.get_nowait()
                except queue.Empty:
                    break
                if newer is None:
                    saw_stop = True
                    break
                for event in newer:
                    projector.apply(event, materialize=False)
                    last_event = event
            if saw_stop:
                if dirty:
                    publish(time.perf_counter())
                break
        now = time.perf_counter()
        if dirty and now >= next_publish:
            publish(now)


def run_evolution_metagraph(
    metagraph: Any,
    *,
    duration: float = math.inf,
    size: tuple[int, int] = (1200, 800),
    fps: int = 60,
    release_hz: float = 6.0,
    physics_hz: float = 240.0,
    top_k: int = 8,
    report_hz: float = 1.0,
    max_event_backlog: int = 256,
    event_trace: bool = False,
    event_inbox: LiveEvolutionEventBuffer | None = None,
    spring_node_capacity: int = 1_048_576,
    spring_edge_capacity: int = 2_097_152,
    physics_device: str = "cuda",
    spring_damping: float = 0.902,
    spring_stiffness: float = 64.0,
    spring_repulsion_strength: float = 0.005,
    spring_edge_pulse_mode: str = "compiler-schedule",
    spring_schedule_hz: float = 3.0,
    spring_schedule_contraction: float = 0.50,
    spring_contraction_response_hz: float = 12.0,
    spring_schedule_yank_impulse: float = 0.0,
    spring_boundary_radius: float = 0.0,
    spring_target_max_velocity: float = 100.0,
    spring_min_dt_scale: float = 0.1,
    spring_max_dt_scale: float = 2.0,
    compiler_status: Mapping[str, str] | None = None,
) -> None:
    """Run compiler ingestion, CUDA BoundSpring, and display independently.

    The isolated-compiler path supplies ``event_inbox`` through an unbounded,
    one-way relay. Physics never subscribes to the compiler, acknowledges an
    event, or applies flow control to it. Projected membership crosses as
    append/update deltas and is installed directly into resident arrays.
    Schedule membership is accepted only from compiler-authored
    ``graph-schedule-finalized`` events. CUDA FluxSpring advances on its own
    clock and publishes leased CUDA observation pages. A slow display loses
    observations and never pauses physics.

    ``release_hz`` is retained only as an API compatibility argument; event
    release is deliberately no longer paced.
    """

    from .opengl_render.api import make_threaded_draw_hook
    from .opengl_render.renderer import GLRenderer
    from .gpu_resident_bound_spring import (
        GpuResidentBoundSpringSimulation,
    )
    import multiprocessing

    # Visualization is an observer, never a compiler admission gate.
    owns_inbox = event_inbox is None
    events = (
        LiveEvolutionEventBuffer(max_event_backlog)
        if event_inbox is None
        else event_inbox
    )
    if owns_inbox:
        # Compatibility path for in-process callers.  The isolated compiler
        # demo supplies its own relay-fed inbox and never subscribes physics to
        # the compiler-owned metagraph.
        unsubscribe = metagraph.subscribe(events.publish, replay=True)
        events.activate()
    else:
        unsubscribe = lambda: None
    projector = EvolutionVisualProjector()
    projector.telemetry.top_k = max(1, int(top_k))
    simulation = GpuResidentBoundSpringSimulation(
        projector.graph(),
        node_capacity=spring_node_capacity,
        edge_capacity=spring_edge_capacity,
        device=physics_device,
        observation_hz=max(1.0, float(fps)),
        damping=spring_damping,
        spring_stiffness=spring_stiffness,
        repulsion_strength=spring_repulsion_strength,
        edge_pulse_mode=spring_edge_pulse_mode,
        schedule_hz=spring_schedule_hz,
        schedule_contraction=spring_schedule_contraction,
        contraction_response_hz=spring_contraction_response_hz,
        schedule_yank_impulse=spring_schedule_yank_impulse,
        target_max_velocity=spring_target_max_velocity,
        min_dt_scale=spring_min_dt_scale,
        max_dt_scale=spring_max_dt_scale,
        boundary_radius=(
            None
            if float(spring_boundary_radius) <= 0.0
            else float(spring_boundary_radius)
        ),
    )
    stop_simulation = threading.Event()
    stop_ingestion = threading.Event()
    compiler_status = compiler_status or {
        "state": "observing",
        "message": "",
    }

    draw_hook, render_thread = make_threaded_draw_hook(
        lambda: GLRenderer(size=size),
        viewport=size,
        history=1,
        loop_mode="idle",
        ghost_trail=False,
    )
    started = time.perf_counter()
    projection_context = multiprocessing.get_context("spawn")
    projection_requests = projection_context.Queue()
    # Deltas cannot use a latest-value mailbox: dropping an intermediate delta
    # would drop membership. The consumer drains and coalesces this unbounded
    # one-way queue before one resident installation.
    projection_results = projection_context.Queue()
    projection_stop = projection_context.Event()
    projection_process = projection_context.Process(
        target=_projection_update_process,
        args=(
            projection_requests,
            projection_results,
            projection_stop,
            top_k,
            report_hz,
        ),
        name="turing-evolution-topology-projector",
        daemon=True,
    )
    projection_process.start()
    # Single-writer, latest-value topology mailbox. Assignment publishes a
    # complete immutable tuple atomically under CPython; consumers never lock,
    # acknowledge, or ask the producer to retain superseded visual pages.
    observer_status_page = (None, projector.report())

    def print_report(report: ExpansionReport) -> None:
        if not report.hotspots:
            return
        print("[translation-growth] " + report.summary(), flush=True)
        for hotspot in report.hotspots:
            stages = ", ".join(
                f"{stage}:{count}" for stage, count in hotspot.stages
            )
            print(
                "  "
                f"{hotspot.owner}: size={hotspot.node_count} "
                f"rate={hotspot.rate:.1f}/s "
                f"relative={hotspot.relative_size:+.2f} "
                f"depth={hotspot.depth} height={hotspot.height} "
                f"boundary={','.join(hotspot.boundaries) if hotspot.boundaries else 'unmatched'} "
                f"hint={hotspot.boundary_hint} "
                f"[{stages}]",
                flush=True,
            )

    def event_worker() -> None:
        """Relay complete queue snapshots without doing graph work locally."""

        while not stop_ingestion.is_set():
            event = events.pop()
            if event is None:
                stop_ingestion.wait(0.001)
                continue
            # Consume exactly the backlog that existed at this observation.
            # Events published concurrently form the next batch, so a hot
            # compiler cannot starve topology publication indefinitely.
            batch = [event]
            for _ in range(len(events)):
                pending = events.pop()
                if pending is None:
                    break
                batch.append(pending)
            if event_trace:
                print(
                    "[translation-event-batch] "
                    f"count={len(batch)} "
                    f"first={_evolution_event_summary(batch[0])} "
                    f"last={_evolution_event_summary(batch[-1])}",
                    flush=True,
                )
            # Unbounded multiprocessing queues hand the batch to their feeder
            # thread without waiting for projection or physics.
            projection_requests.put_nowait(tuple(batch))

    def simulation_worker() -> None:
        nonlocal observer_status_page
        # One thread owns every mutation of resident CUDA state. Projection
        # crosses the process boundary as immutable deltas; installation and
        # integration are then enqueued in-order on this worker's CUDA stream.
        simulation.bind_mutation_owner()
        physics_dt = 1.0 / max(1.0, float(physics_hz))
        next_physics = time.perf_counter()
        while not stop_simulation.is_set():
            now = time.perf_counter()
            deltas: list[VisualGraphDelta] = []
            report_to_print = None
            while True:
                try:
                    projected_delta, projected_event, projected_report, report_changed = (
                        projection_results.get_nowait()
                    )
                except queue.Empty:
                    break
                observer_status_page = (projected_event, projected_report)
                if not projected_delta.empty:
                    deltas.append(projected_delta)
                if report_changed:
                    report_to_print = projected_report
            if deltas:
                delta = _coalesce_visual_deltas(deltas)
                simulation.apply_delta(delta)
            if report_to_print is not None:
                print_report(report_to_print)
            steps = 0
            while now >= next_physics and steps < 16:
                simulation.step(physics_dt)
                next_physics += physics_dt
                steps += 1
            if steps == 16 and now >= next_physics:
                # Discard host-scheduler debt; do not enlarge physical dt.
                next_physics = now + physics_dt

            delay = next_physics - time.perf_counter()
            stop_simulation.wait(max(0.0001, min(0.002, delay)))

    def frame_worker() -> None:
        """Consume completed DMA pages; never touch live CUDA physics state."""

        frame_dt = 1.0 / max(1.0, float(fps))
        next_frame = time.perf_counter()
        while not stop_simulation.is_set():
            now = time.perf_counter()
            if now >= next_frame:
                frame = simulation.consume_layers(now - started)
                last_event, last_report = observer_status_page
                if frame is not None:
                    frame = dict(frame)
                    compiler_state = str(
                        compiler_status.get("state", "unknown")
                    )
                    compiler_message = str(
                        compiler_status.get("message", "")
                    )
                    frame["hud_text"] = (
                        f"compiler={compiler_state} | CUDA-resident observer",
                        _evolution_event_summary(last_event),
                        f"observer_pending={len(events)} | {last_report.summary()}",
                        (
                            f"GPU FluxSpring {simulation.completed_hz:.1f} completed steps/s | "
                            f"resident nodes={simulation.node_count} "
                            f"edges={simulation.edge_count}"
                        ),
                        (
                            f"OpenGL {render_thread.completed_hz:.1f} presented frames/s | "
                            f"superseded={render_thread.dropped_frames}"
                        ),
                        (
                            f"damping={simulation.damping:g} | "
                            f"compiler-authored "
                            f"{'execution' if simulation.execution_plan_active else 'local DAG'} "
                            f"replay frame "
                            f"{simulation.active_schedule_group + 1}/"
                            f"{simulation.schedule_group_count} | "
                            f"covered={simulation.execution_scheduled_node_count if simulation.execution_plan_active else simulation.local_scheduled_node_count}/"
                            f"{simulation.node_count} | clean edges"
                        ),
                        *(
                            ("compiler detail: " + compiler_message[:180],)
                            if compiler_message else ()
                        ),
                    )
                    draw_hook(frame)
                next_frame = now + frame_dt
            stop_simulation.wait(max(0.0005, min(0.005, next_frame - time.perf_counter())))

    ingestion_thread = threading.Thread(
        target=event_worker,
        name="turing-topology-ingestion",
        daemon=False,
    )
    simulation_thread = threading.Thread(
        target=simulation_worker,
        name="turing-cuda-bound-spring-physics",
        daemon=False,
    )
    video_thread = threading.Thread(
        target=frame_worker,
        name="turing-lock-free-video-observer",
        daemon=False,
    )
    ingestion_thread.start()
    simulation_thread.start()
    video_thread.start()
    try:
        while (
            time.perf_counter() - started < duration
            and render_thread.is_alive()
        ):
            time.sleep(0.05)
    finally:
        # Close the one-way pipeline in producer-to-consumer order. Every
        # stage finishes its already-published work before its consumer exits;
        # no queue is terminated while a feeder thread may still own it.
        unsubscribe()
        stop_ingestion.set()
        ingestion_thread.join()
        projection_requests.put_nowait(None)
        projection_process.join()
        stop_simulation.set()
        simulation_thread.join()
        video_thread.join()
        if owns_inbox:
            events.close()
        render_thread.stop(timeout=None)
        if simulation.device.type == "cuda":
            simulation.torch.cuda.synchronize(simulation.device)
        for process_queue in (
            projection_requests,
            projection_results,
        ):
            try:
                process_queue.cancel_join_thread()
                process_queue.close()
            except Exception:
                pass
    if render_thread.last_error is not None:
        raise RuntimeError("OpenGL render worker failed") from render_thread.last_error


def run_precompiled_graph(
    package: Any,
    *,
    duration: float = math.inf,
    size: tuple[int, int] = (1100, 760),
    fps: int = 60,
    release_hz: float = 6.0,
    shader_sources: tuple[str, str] | None = None,
) -> None:
    """Open the reusable OpenGL graph surface for any supported IR package."""

    import pygame
    from ..common.tensors.autoautograd.spring_async_toy import LiveVizGLPoints

    source = package
    graph = visual_graph_from_ir(source)
    simulation = FluxSpringVisualSimulation(graph)
    pending = deque(maxlen=2048)
    unsubscribe = None
    if hasattr(source, "subscribe") and callable(source.subscribe):
        unsubscribe = source.subscribe(
            lambda snapshot: pending.append(visual_graph_from_ir(snapshot)),
            replay=False,
        )
    visualizer = None
    clock = pygame.time.Clock()
    started = time.perf_counter()
    running = True
    next_release = started
    try:
        while running and time.perf_counter() - started < duration:
            now = time.perf_counter()
            if pending and now >= next_release:
                simulation.replace_graph(pending.popleft())
                next_release = now + 1.0 / max(0.1, float(release_hz))
            elif hasattr(source, "snapshot") and callable(source.snapshot):
                current = visual_graph_from_ir(source)
                if current.revision != simulation.graph.revision:
                    simulation.replace_graph(current)
            dt = min(0.05, max(1e-4, clock.get_time() / 1000.0))
            simulation.step(dt)
            system = simulation.fluxspring._system
            if system is not None:
                if visualizer is None:
                    visualizer = LiveVizGLPoints(
                        system,
                        node_cmap="coolwarm",
                        edge_cmap="coolwarm",
                        base_point_size=10.0,
                        shader_sources=shader_sources,
                    )
                    visualizer.launch(*size)
                else:
                    visualizer.sys = system
                if visualizer.step(dt) is False:
                    running = False
            clock.tick(max(1, int(fps)))
    finally:
        if unsubscribe is not None:
            unsubscribe()
        if visualizer is not None:
            visualizer.close()


__all__ = [
    "ExpansionHotspot",
    "ExpansionEmergencyClamp",
    "ExpansionLimitExceeded",
    "ExpansionReport",
    "ExpansionTelemetry",
    "SpringVisualSimulation",
    "FluxSpringVisualSimulation",
    "LiveEvolutionEventBuffer",
    "EvolutionVisualProjector",
    "VisualEdge",
    "VisualGraph",
    "VisualGraphAccumulator",
    "VisualGraphDelta",
    "VisualNode",
    "run_precompiled_graph",
    "run_evolution_metagraph",
    "visual_graph_from_ir",
]
