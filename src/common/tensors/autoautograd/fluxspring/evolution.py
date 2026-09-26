"""Live multi-network geometry and activation for evolving FluxSpring graphs.

This is deliberately compiler-agnostic.  Callers synchronize stable node and
edge identities; FluxSpring owns birth-from-source, growth, network placement,
DEC spring geometry, and edge-control activation during simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import colorsys
import hashlib
import math
from typing import Any, Iterable

import numpy as np

from ...abstraction import AbstractTensor as AT
from .fs_dec import pump_tick
from .fs_types import (
    DECSpec,
    EdgeCtrl,
    EdgeSpec,
    EdgeTransport,
    EdgeTransportLearn,
    FluxSpringSpec,
    LearnCtrl,
    NodeCtrl,
    NodeSpec,
)


def _phase(key: str) -> float:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / (2**64) * math.tau


def _numpy(value: Any) -> np.ndarray:
    tensor = AT.get_tensor(value)
    return np.asarray(tensor.data if hasattr(tensor, "data") else tensor)


@dataclass(frozen=True, slots=True)
class EvolutionNode:
    key: str
    network: str
    kind: str = "operation"
    color_group: str = ""


@dataclass(frozen=True, slots=True)
class EvolutionEdge:
    source: str
    target: str
    role: str = "data"


@dataclass(frozen=True, slots=True)
class FluxSpringFrame:
    positions: np.ndarray
    colors: np.ndarray
    sizes: np.ndarray
    edge_indices: np.ndarray
    edge_colors: np.ndarray
    edge_activation: np.ndarray


class MultiNetworkFluxSpring:
    """FluxSpring state supporting named networks and provenance budding."""

    def __init__(
        self,
        *,
        seed: int = 0,
        network_spacing: float = 4.5,
        growth_seconds: float = 0.65,
    ) -> None:
        self.seed = int(seed)
        self.network_spacing = float(network_spacing)
        self.growth_seconds = max(1e-3, float(growth_seconds))
        self.nodes: tuple[EvolutionNode, ...] = ()
        self.edges: tuple[EvolutionEdge, ...] = ()
        self.keys: tuple[str, ...] = ()
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.velocities = np.zeros((0, 3), dtype=np.float32)
        self.age = np.zeros(0, dtype=np.float32)
        self.psi = np.zeros(0, dtype=np.float32)
        self.edge_activation = np.zeros(0, dtype=np.float32)
        self.visual_haze: dict[int, float] = {}
        self._network_order: dict[str, int] = {}
        self._known_handoffs: set[tuple[str, str]] = set()
        self._spec: FluxSpringSpec | None = None
        self._system: Any = None
        self._engine: Any = None
        self._edge_indices = np.zeros((0, 2), dtype=np.int64)
        self.elapsed = 0.0
        try:
            from matplotlib import colormaps

            self._coolwarm = colormaps.get_cmap("coolwarm")
        except Exception:
            self._coolwarm = None

    def synchronize(
        self,
        nodes: Iterable[EvolutionNode],
        edges: Iterable[EvolutionEdge],
    ) -> None:
        """Synchronize topology while retaining state and budding new handoffs."""

        nodes = tuple(nodes)
        edges = tuple(edges)
        previous_index = {key: i for i, key in enumerate(self.keys)}
        for node in nodes:
            self._network_order.setdefault(node.network, len(self._network_order))
        keys = tuple(node.key for node in nodes)
        index = {key: i for i, key in enumerate(keys)}
        incoming = {
            edge.target: edge.source for edge in edges if edge.role == "handoff"
        }
        handoffs = {
            (edge.source, edge.target) for edge in edges if edge.role == "handoff"
        }
        rng = np.random.default_rng(self.seed)
        positions = np.zeros((len(nodes), 3), dtype=np.float32)
        velocities = np.zeros_like(positions)
        age = np.zeros(len(nodes), dtype=np.float32)
        psi = np.zeros(len(nodes), dtype=np.float32)
        for ordinal, node in enumerate(nodes):
            old = previous_index.get(node.key)
            source_key = incoming.get(node.key)
            source_old = previous_index.get(source_key) if source_key else None
            newly_budded = (
                source_old is not None
                and (source_key, node.key) not in self._known_handoffs
            )
            if newly_budded:
                positions[ordinal] = self.positions[source_old]
                velocities[ordinal] = self.velocities[source_old] * 0.35
                age[ordinal] = 0.0
                psi[ordinal] = self.psi[source_old]
            elif old is not None:
                positions[ordinal] = self.positions[old]
                velocities[ordinal] = self.velocities[old]
                age[ordinal] = self.age[old]
                psi[ordinal] = self.psi[old]
            elif source_old is not None:
                positions[ordinal] = self.positions[source_old]
                psi[ordinal] = self.psi[source_old]
            else:
                anchor = self._network_order[node.network] * self.network_spacing
                phase = _phase(node.key)
                positions[ordinal] = (
                    anchor + 0.2 * math.sin(phase),
                    math.sin(phase),
                    math.cos(phase),
                )
                positions[ordinal] += rng.normal(0.0, 0.035, 3)
                psi[ordinal] = math.sin(phase)

        self.nodes = nodes
        self.edges = edges
        self.keys = keys
        self.positions = positions
        self.velocities = velocities
        self.age = age
        self.psi = psi
        self._known_handoffs = handoffs
        self._edge_indices = np.asarray(
            [(index[e.source], index[e.target]) for e in edges], dtype=np.int64
        ).reshape(-1, 2)
        self.edge_activation = np.zeros(len(edges), dtype=np.float32)
        self._rebuild_spec()

    def set_visual_haze(self, scores: dict[int, float]) -> None:
        """Set an observer-owned ring value; never feed it into physics."""

        self.visual_haze = {
            int(index): max(-1.0, min(1.0, float(value)))
            for index, value in scores.items()
            if 0 <= int(index) < len(self.nodes)
        }
        if self._system is not None:
            self._system.visual_haze = dict(self.visual_haze)

    def _rebuild_spec(self) -> None:
        node_specs = [
            NodeSpec(
                id=i,
                p0=AT.get_tensor(self.positions[i]),
                v0=AT.get_tensor(self.velocities[i]),
                mass=AT.tensor(1.0),
                ctrl=NodeCtrl(learn=LearnCtrl(False, False, False)),
                scripted_axes=[],
            )
            for i in range(len(self.nodes))
        ]
        edge_specs: list[EdgeSpec] = []
        d0: list[list[float]] = []
        for ordinal, (edge, endpoints) in enumerate(
            zip(self.edges, self._edge_indices)
        ):
            source, target = map(int, endpoints)
            handoff = edge.role == "handoff"
            edge_specs.append(EdgeSpec(
                src=source,
                dst=target,
                transport=EdgeTransport(
                    kappa=AT.tensor(1.0),
                    k=AT.tensor(0.12 if handoff else 0.42),
                    l0=AT.tensor(1.6 if handoff else 1.05),
                    lambda_s=AT.tensor(0.12),
                    x=AT.tensor(0.0),
                    learn=EdgeTransportLearn(False, False, False, False, False),
                ),
                ctrl=EdgeCtrl(
                    alpha=AT.tensor(0.55),
                    w=AT.tensor(1.0),
                    b=AT.tensor(0.0),
                    learn=LearnCtrl(False, False, False),
                ),
            ))
            row = [0.0] * len(self.nodes)
            row[source] = -1.0
            row[target] = 1.0
            d0.append(row)
        self._spec = FluxSpringSpec(
            version="multi-network-evolution-1.0",
            D=3,
            nodes=node_specs,
            edges=edge_specs,
            faces=[],
            dec=DECSpec(D0=d0, D1=[]),
            gamma=AT.tensor(0.0),
        )
        self._system = None
        self._engine = None
        if not self.nodes:
            return
        # Drive geometry with the original FluxSpring demo's actual
        # SpringRepulsorSystem and dt integrator. This wrapper owns topology
        # evolution; it does not substitute a second spring implementation.
        from ..spring_async_toy import (
            BoundaryPort,
            Edge as SpringEdge,
            Node as SpringNode,
            SpringDtEngine,
            SpringRepulsorSystem,
        )

        spring_nodes = [
            SpringNode(
                i,
                AT.get_tensor(self.positions[i]),
                v=AT.get_tensor(self.velocities[i]),
                ctrl=AT.get_tensor([0.55, 1.0, 0.0]),
                M0=1.0,
            )
            for i in range(len(self.nodes))
        ]
        spring_edges = [
            SpringEdge(
                (int(source), int(target), edge.role),
                int(source),
                int(target),
                edge.role,
                ctrl=AT.get_tensor([0.55, 1.0, 0.0]),
                l0=AT.tensor(1.6 if edge.role == "handoff" else 1.05),
                k=AT.tensor(0.12 if edge.role == "handoff" else 0.42),
            )
            for edge, (source, target) in zip(self.edges, self._edge_indices)
        ]
        self._system = SpringRepulsorSystem(
            spring_nodes,
            spring_edges,
            eta=0.035,
            gamma=0.91,
            dt=1.0 / 60.0,
        )
        self._system.visual_haze = dict(self.visual_haze)
        for i, node in enumerate(self.nodes):
            anchor = AT.get_tensor([
                self._network_order[node.network] * self.network_spacing,
                0.0,
                0.0,
            ])
            self._system.add_boundary(BoundaryPort(
                nid=i,
                alpha=0.14,
                target_fn=lambda _time, anchor=anchor: anchor,
            ))
        self._engine = SpringDtEngine(self._system)

    def step(self, dt: float = 1.0 / 60.0) -> None:
        if not self.nodes:
            return
        dt = min(0.05, max(1e-4, float(dt)))
        self.elapsed += dt
        self.age += dt
        growth = np.clip(self.age / self.growth_seconds, 0.0, 1.0)
        growth = growth * growth * (3.0 - 2.0 * growth)

        if self.edges and self._spec is not None:
            external = {
                i: AT.tensor(math.sin(self.elapsed * 1.7 + _phase(node.key)))
                for i, node in enumerate(self.nodes)
                if node.kind in {"feed", "argument", "input"}
            }
            with AT.autograd.no_grad():
                psi, stats = pump_tick(
                    AT.get_tensor(self.psi),
                    self._spec,
                    eta=dt * 0.7,
                    external=external,
                    leak=dt * 0.08,
                    norm="node",
                )
            self.psi = _numpy(psi).astype(np.float32, copy=False)
            self.edge_activation = _numpy(stats["q"]).astype(np.float32, copy=False)

        if self._system is not None and self._engine is not None:
            for i, spring_node in self._system.nodes.items():
                spring_node.M0 = 1.0 / max(0.08, float(growth[i]))
                spring_node.visual_growth = float(growth[i])
                spring_node.ctrl = AT.get_tensor([
                    0.55, float(self.psi[i]), 0.0
                ])
            for ordinal, spring_edge in enumerate(self._system.edge_list):
                source, target = self._edge_indices[ordinal]
                participation = min(float(growth[source]), float(growth[target]))
                base = 0.12 if self.edges[ordinal].role == "handoff" else 0.42
                spring_edge.k = AT.tensor(base * max(0.02, participation))
                spring_edge.activation = (
                    float(self.edge_activation[ordinal])
                    if ordinal < len(self.edge_activation)
                    else 0.0
                )
            self._engine.current_dt = dt
            self._engine.step(dt)
            self.positions = np.asarray([
                _numpy(self._system.nodes[i].p) for i in range(len(self.nodes))
            ], dtype=np.float32)
            self.velocities = np.asarray([
                _numpy(self._system.nodes[i].v) for i in range(len(self.nodes))
            ], dtype=np.float32)

    def frame(self) -> FluxSpringFrame:
        growth = np.clip(self.age / self.growth_seconds, 0.0, 1.0)
        growth = growth * growth * (3.0 - 2.0 * growth)
        if self._coolwarm is not None and len(self.nodes):
            colors = np.asarray(
                self._coolwarm((np.tanh(self.psi) + 1.0) * 0.5),
                dtype=np.float32,
            )
        else:
            colors = np.zeros((len(self.nodes), 4), dtype=np.float32)
            for i, node in enumerate(self.nodes):
                hue = (_phase(node.color_group or node.network) / math.tau) % 1.0
                colors[i, :3] = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
                colors[i, 3] = 1.0
        role_palette = {
            "feed": (0.078, 0.722, 0.651),
            "argument": (0.078, 0.722, 0.651),
            "input": (0.078, 0.722, 0.651),
            "output": (0.851, 0.275, 0.937),
        }
        for i, node in enumerate(self.nodes):
            if node.kind in role_palette:
                colors[i, :3] = role_palette[node.kind]
            colors[i, 3] = 1.0
        sizes = (3.0 + 15.0 * growth).astype(np.float32)
        for i, node in enumerate(self.nodes):
            if node.kind in {"feed", "argument", "input", "output"}:
                sizes[i] *= 1.35
        if len(self.edges):
            active = np.tanh(self.edge_activation)
            magnitude = np.abs(active)
            edge_colors = np.zeros((len(self.edges), 4), dtype=np.float32)
            edge_colors[:, 0] = np.where(active >= 0.0, 1.0, 0.2)
            edge_colors[:, 1] = 0.28 + 0.55 * magnitude
            edge_colors[:, 2] = np.where(active < 0.0, 1.0, 0.42)
            edge_colors[:, 3] = 0.22 + 0.76 * magnitude
            for i, edge in enumerate(self.edges):
                if edge.role == "handoff":
                    edge_colors[i, :3] = (1.0, 0.42, 0.82)
        else:
            edge_colors = np.zeros((0, 4), dtype=np.float32)
        return FluxSpringFrame(
            self.positions.copy(),
            colors,
            sizes,
            self._edge_indices.copy(),
            edge_colors,
            self.edge_activation.copy(),
        )


__all__ = [
    "EvolutionEdge",
    "EvolutionNode",
    "FluxSpringFrame",
    "MultiNetworkFluxSpring",
]
