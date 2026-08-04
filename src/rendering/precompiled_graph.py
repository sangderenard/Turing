"""Animate compiler IR topology with the original FluxSpring graph shader.

The visualizer consumes a small immutable graph snapshot.  Front ends are not
special-cased by the renderer: AST, SymPy, GLSL ingestion, and other sources
first reach ProcessGraph/precompile as usual, while completed FusedProgram,
SSA, DualIR/AOT, Nodus GraphIR, and serialized graph packages are adapted to
the same :class:`VisualGraph` contract.

The compiler graph remains observational and never receives geometry back.
Its visual state is nevertheless a real multi-network FluxSpring simulation:
DEC geometry, control/transport activation, budding, growth, and the complete
LiveViz display loop live in FluxSpring rather than in this compiler adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
import re
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


def _key(value: Any, *, prefix: str = "") -> str:
    return prefix + str(value)


def _visual_from_networkx(graph: Any, *, revision: int = 0) -> VisualGraph:
    nx_graph = getattr(graph, "G", graph)
    nodes = tuple(
        VisualNode(
            key=_key(node_id),
            label=str(data.get("label") or data.get("op") or data.get("type") or node_id),
            kind=str(data.get("type") or data.get("op") or "operation"),
            level=(None if data.get("level") is None else int(data["level"])),
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
        self._revision = 0

    @staticmethod
    def _component_key(ref: Any) -> str:
        return f"{ref.graph_id}/{ref.local_id}"

    def apply(self, event: Any) -> VisualGraph:
        if event.kind == "graph-open" and event.graph is not None:
            self._graphs[event.graph.id] = event.graph
            self._graph_order.setdefault(event.graph.id, len(self._graph_order))
        elif event.kind in {"component-spawn", "component-update"} and event.component is not None:
            graph = self._graphs.get(event.component.graph_id)
            stage = "graph" if graph is None else str(graph.stage)
            order = self._graph_order.setdefault(
                event.component.graph_id,
                len(self._graph_order),
            )
            key = self._component_key(event.component)
            self._nodes[key] = VisualNode(
                key=key,
                label=str(event.detail.get("label") or event.component.local_id),
                kind=str(event.detail.get("kind") or "component"),
                level=order * 12,
                group=stage,
                network=event.component.graph_id,
            )
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
                    self._edges[(source, target, role)] = VisualEdge(source, target, role)
        self._revision = max(self._revision + 1, int(event.sequence) + 1)
        return self.graph()

    def graph(self) -> VisualGraph:
        return VisualGraph(
            tuple(self._nodes.values()),
            tuple(self._edges.values()),
            "evolution-metagraph",
            self._revision,
        )


def run_evolution_metagraph(
    metagraph: Any,
    *,
    duration: float = math.inf,
    size: tuple[int, int] = (1200, 800),
    fps: int = 60,
    release_hz: float = 6.0,
) -> None:
    """Replay captured provenance as side-by-side spawning graph stages."""

    import pygame
    from ..common.tensors.autoautograd.spring_async_toy import LiveVizGLPoints

    events = deque()
    unsubscribe = metagraph.subscribe(events.append, replay=True)
    projector = EvolutionVisualProjector()
    simulation = FluxSpringVisualSimulation(projector.graph())
    visualizer = None
    clock = pygame.time.Clock()
    started = time.perf_counter()
    next_release = started
    running = True
    try:
        while running and time.perf_counter() - started < duration:
            now = time.perf_counter()
            if events and now >= next_release:
                simulation.replace_graph(projector.apply(events.popleft()))
                next_release = now + 1.0 / max(0.1, float(release_hz))
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
                    )
                    visualizer.launch(*size)
                else:
                    visualizer.sys = system
                if visualizer.step(dt) is False:
                    running = False
            clock.tick(max(1, int(fps)))
    finally:
        unsubscribe()
        if visualizer is not None:
            visualizer.close()


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
    "SpringVisualSimulation",
    "FluxSpringVisualSimulation",
    "EvolutionVisualProjector",
    "VisualEdge",
    "VisualGraph",
    "VisualNode",
    "run_precompiled_graph",
    "run_evolution_metagraph",
    "visual_graph_from_ir",
]
