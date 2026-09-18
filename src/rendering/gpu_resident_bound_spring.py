"""CUDA-resident FluxSpring evolution with lock-free observation pages.

This is the active integration of the original Transmogrifier BoundSpring
architecture: vectorized Torch device physics, append-only resident topology,
and the repository DoubleBuffer cursor model. Rendering never reads live
physics tensors. CUDA-to-CUDA observations are opportunistic and skipped when
both video pages are occupied; physics never waits for video.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
import hashlib
import math
import threading
import time
from typing import TYPE_CHECKING, Mapping

import numpy as np

from ..common.double_buffer import DoubleBuffer
from .opengl_render.api import CudaGraphLayer, LineLayer, PointLayer

if TYPE_CHECKING:
    from .precompiled_graph import VisualGraph, VisualGraphDelta


def _phase(value: str) -> float:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(2**64)


@dataclass(slots=True)
class _VideoPage:
    positions: object
    camera_bounds: object
    velocities: object | None = None
    event: object | None = None
    pending: bool = False
    ready: bool = False
    in_use: bool = False
    sequence: int = -1
    node_count: int = 0
    graph: object | None = None
    haze: Mapping[str, float] | None = None
    source_state_page: int = -1
    active_schedule_group: int = -1
    schedule_pulse: float = 0.0
    schedule_groups: object | None = None
    presentation: object | None = None


@dataclass(frozen=True, slots=True)
class _PresentationState:
    """Immutable topology-derived render data shared by video observations."""

    revision: int
    node_colors: np.ndarray
    node_sizes: np.ndarray
    edge_indices: np.ndarray
    edge_colors: np.ndarray
    camera_center: np.ndarray
    camera_radius: float


class GpuResidentBoundSpringSimulation:
    """Free-running CUDA spring physics with lossy lock-free observation."""

    def __init__(
        self,
        graph: "VisualGraph",
        *,
        node_capacity: int = 1_048_576,
        edge_capacity: int = 2_097_152,
        state_pages: int = 4,
        device: str = "cuda",
        seed: int = 0,
        observation_hz: float = 120.0,
        damping: float = 0.902,
        spring_stiffness: float = 64.0,
        repulsion_strength: float = 0.005,
        edge_pulse_mode: str = "compiler-schedule",
        schedule_hz: float = 3.0,
        schedule_contraction: float = 0.50,
        contraction_response_hz: float = 12.0,
        schedule_yank_impulse: float = 0.0,
        boundary_radius: float | None = None,
        target_max_velocity: float = 100.0,
        min_dt_scale: float = 0.1,
        max_dt_scale: float = 2.0,
    ) -> None:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU-resident BoundSpring requires CUDA; no CPU fallback is used"
            )
        self.torch = torch
        self.device = torch.device(device)
        self.seed = int(seed)
        self.node_capacity = max(1, int(node_capacity))
        self.edge_capacity = max(1, int(edge_capacity))
        self.observation_dt = 1.0 / max(1.0, float(observation_hz))
        self.damping = max(0.0, float(damping))
        self.spring_stiffness = max(0.0, float(spring_stiffness))
        self.repulsion_strength = max(0.0, float(repulsion_strength))
        if edge_pulse_mode not in {"compiler-schedule", "off"}:
            raise ValueError(
                "edge_pulse_mode must be 'compiler-schedule' or 'off'"
            )
        self.edge_pulse_mode = edge_pulse_mode
        self.schedule_hz = max(0.01, float(schedule_hz))
        self.schedule_contraction = min(0.95, max(0.0, float(schedule_contraction)))
        self.contraction_response_hz = max(
            0.01, float(contraction_response_hz)
        )
        self.schedule_yank_impulse = max(0.0, float(schedule_yank_impulse))
        self.boundary_radius = (
            None
            if boundary_radius is None or float(boundary_radius) <= 0.0
            else float(boundary_radius)
        )
        self.target_max_velocity = max(0.0, float(target_max_velocity))
        self.min_dt_scale = max(1.0e-4, float(min_dt_scale))
        self.max_dt_scale = max(self.min_dt_scale, float(max_dt_scale))
        self._previous_max_velocity = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._dt_scale = torch.ones(
            (), dtype=torch.float32, device=self.device
        )
        self._effective_dt = torch.ones(
            (), dtype=torch.float32, device=self.device
        )
        self._next_observation = 0.0
        # Camera reductions are observational work on the copy stream.  They
        # run much less often than video publication and can never hold up the
        # physics stream.
        self._next_camera_observation = 0.0
        self._camera_observation_dt = 1.0 / 12.0
        self._camera_bounds_device = torch.tensor(
            (0.0, 0.0, 0.0, 8.0), dtype=torch.float32, device=self.device
        )
        self._position_pages = [
            torch.zeros((self.node_capacity, 3), dtype=torch.float32, device=self.device)
            for _ in range(max(3, int(state_pages)))
        ]
        self._velocity_pages = [torch.zeros_like(page) for page in self._position_pages]
        self._force = torch.zeros_like(self._position_pages[0])
        self._current_page = 0
        self._edge_index = torch.zeros(
            (self.edge_capacity, 2), dtype=torch.long, device=self.device
        )
        self._base_rest = torch.ones(
            self.edge_capacity, dtype=torch.float32, device=self.device
        )
        self._edge_phase = torch.zeros_like(self._base_rest)
        self._rest_state = torch.ones_like(self._base_rest)
        self._edge_schedule_group = torch.zeros(
            self.edge_capacity, dtype=torch.long, device=self.device
        )
        # Capacity and membership are deliberately separate. Compiler growth
        # claims already-resident slots by flipping these masks; it never
        # sizes, replaces, or clears the physics world for an ordinary graph
        # revision.
        self._node_occupancy = torch.zeros(
            self.node_capacity, dtype=torch.bool, device=self.device
        )
        self._edge_occupancy = torch.zeros(
            self.edge_capacity, dtype=torch.bool, device=self.device
        )
        self._node_weight = torch.zeros(
            self.node_capacity, dtype=torch.float32, device=self.device
        )
        self._edge_weight = torch.zeros(
            self.edge_capacity, dtype=torch.float32, device=self.device
        )
        # FluxSpring data-path state.  These are the vectorized D0/transpose-D0
        # mechanics from fs_dec.pump_tick, represented by endpoint gathers and
        # scatter-adds rather than a dense incidence matrix.
        self._psi = torch.zeros(
            self.node_capacity, dtype=torch.float32, device=self.device
        )
        self._node_flux = torch.zeros_like(self._psi)
        self._node_nonlinear = torch.zeros_like(self._psi)
        self._input_phase = torch.zeros_like(self._psi)
        self._input_mask = torch.zeros(
            self.node_capacity, dtype=torch.bool, device=self.device
        )
        self._input_indices = torch.zeros(
            self.node_capacity, dtype=torch.long, device=self.device
        )
        self._input_count = 0
        self._edge_flux_raw = torch.zeros(
            self.edge_capacity, dtype=torch.float32, device=self.device
        )
        self._edge_activation = torch.zeros_like(self._edge_flux_raw)
        self._edge_stiffness = torch.ones_like(self._edge_flux_raw)
        self._node_ordinals = torch.arange(
            self.node_capacity, dtype=torch.long, device=self.device
        )
        self._repulsion_offsets = torch.arange(
            1, 33, dtype=torch.long, device=self.device
        )
        self._edge_schedule_group_host = np.zeros(self.edge_capacity, dtype=np.int64)
        self.schedule_group_count = 0
        self.execution_plan_active = False
        self.execution_scheduled_node_count = 0
        self.local_scheduled_node_count = 0
        self.active_schedule_group = -1
        self.schedule_pulse = 0.0
        self._repulsion_neighbors = None
        self._repulsion_batch_width = 8
        self._keys: list[str] = []
        self._edge_keys: list[tuple[str, str, str]] = []
        self._edge_key_set: set[tuple[str, str, str]] = set()
        self._key_to_index: dict[str, int] = {}
        self._network_order_map: dict[str, int] = {}
        self._node_kinds: list[str] = []
        self._node_schedule_groups: list[int] = []
        self._node_execution_groups: list[tuple[int, ...]] = []
        self._node_states: list[str] = []
        self.graph = type(graph)((), (), graph.source_kind)
        self.node_count = 0
        self.edge_count = 0
        self.elapsed = 0.0
        self.steps = 0
        self.completed_hz = 0.0
        self._rate_windows: list[tuple[object, object, int]] = []
        if self.device.type == "cuda":
            self._rate_start = torch.cuda.Event(enable_timing=True)
            self._rate_start.record(torch.cuda.current_stream(self.device))
        else:
            self._rate_start = time.perf_counter()
        self._rate_start_step = 0
        self._haze: dict[str, float] = {}
        self._presentation = _PresentationState(
            revision=-1,
            node_colors=np.zeros((0, 4), dtype=np.float32),
            node_sizes=np.zeros(0, dtype=np.float32),
            edge_indices=np.zeros((0, 2), dtype=np.int64),
            edge_colors=np.zeros((0, 4), dtype=np.float32),
            camera_center=np.zeros(3, dtype=np.float32),
            camera_radius=8.0,
        )
        self._presentation_node_capacity = 0
        self._presentation_edge_capacity = 0
        self._node_colors_host = np.zeros((0, 4), dtype=np.float32)
        self._node_sizes_host = np.zeros(0, dtype=np.float32)
        self._edge_indices_host = np.zeros((0, 2), dtype=np.int64)
        self._edge_colors_host = np.zeros((0, 4), dtype=np.float32)
        self._render_activation_key: object | None = None
        self._render_active_nodes = np.zeros(0, dtype=np.float32)
        self._render_active_edges = np.zeros(0, dtype=np.float32)
        self._schedule_snapshot = np.zeros(0, dtype=np.int64)
        self._camera_fit_indices = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        self._topology_publication = (
            0, 0, 0, self._presentation, self.graph,
            self._schedule_snapshot, self._camera_fit_indices, None,
        )
        self._applied_topology_ready_event = None
        self._last_consumed_sequence = -1
        self._last_consumed_topology_revision = -1
        self._mutation_owner_thread: int | None = None
        self._retired_resident_storage: list[object] = []
        self._retired_video_buffers: list[DoubleBuffer] = []
        self.video = self._new_video_buffer(self.node_capacity)
        self._copy_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self.replace_graph(graph)

    def bind_mutation_owner(self) -> None:
        """Bind every resident-topology and physics mutation to this thread."""

        owner = threading.get_ident()
        if (
            self._mutation_owner_thread is not None
            and self._mutation_owner_thread != owner
        ):
            raise RuntimeError("GPU spring mutation ownership is already bound")
        self._mutation_owner_thread = owner

    def _assert_mutation_owner(self) -> None:
        owner = self._mutation_owner_thread
        if owner is not None and owner != threading.get_ident():
            raise RuntimeError(
                "GPU spring topology and physics must share one owning thread"
            )

    def _new_video_buffer(self, capacity: int) -> DoubleBuffer:
        pin = self.device.type == "cuda"
        pages = [
            _VideoPage(
                self.torch.empty(
                    (capacity, 3), dtype=self.torch.float32,
                    device=self.device if pin else "cpu",
                    pin_memory=False,
                ),
                # Only four floats cross to the host.  The page event makes
                # them readable without a synchronization or an ``item()`` on
                # a live CUDA tensor.
                camera_bounds=self.torch.empty(
                    4,
                    dtype=self.torch.float32,
                    device="cpu",
                    pin_memory=pin,
                ),
            )
            for _ in range(2)
        ]
        return DoubleBuffer(num_agents=2, frames=pages)

    def _ensure_resident_capacity(self, nodes: int, edges: int) -> None:
        """Grow device and observation pages geometrically without a reset."""

        if nodes <= self.node_capacity and edges <= self.edge_capacity:
            return
        torch = self.torch
        old_node_capacity = self.node_capacity
        old_edge_capacity = self.edge_capacity
        new_node_capacity = old_node_capacity
        new_edge_capacity = old_edge_capacity
        while nodes > new_node_capacity:
            new_node_capacity *= 2
        while edges > new_edge_capacity:
            new_edge_capacity *= 2

        old_storage = (
            self._position_pages,
            self._velocity_pages,
            self._edge_index,
            self._base_rest,
            self._edge_phase,
            self._rest_state,
            self._edge_schedule_group,
            self._edge_schedule_group_host,
            self._node_occupancy,
            self._edge_occupancy,
            self._node_weight,
            self._edge_weight,
            self._psi,
            self._node_flux,
            self._node_nonlinear,
            self._input_phase,
            self._input_mask,
            self._input_indices,
            self._edge_flux_raw,
            self._edge_activation,
            self._edge_stiffness,
            self._node_ordinals,
            self._force,
        )
        if new_node_capacity != old_node_capacity:
            new_positions = [
                torch.zeros(
                    (new_node_capacity, 3),
                    dtype=torch.float32,
                    device=self.device,
                )
                for _ in self._position_pages
            ]
            new_velocities = [torch.zeros_like(page) for page in new_positions]
            for target, source in zip(new_positions, self._position_pages):
                target[:self.node_count].copy_(source[:self.node_count])
            for target, source in zip(new_velocities, self._velocity_pages):
                target[:self.node_count].copy_(source[:self.node_count])
            self._position_pages = new_positions
            self._velocity_pages = new_velocities
            node_occupancy = torch.zeros(
                new_node_capacity, dtype=torch.bool, device=self.device
            )
            node_occupancy[:self.node_count].copy_(
                self._node_occupancy[:self.node_count]
            )
            self._node_occupancy = node_occupancy
            node_weight = torch.zeros(
                new_node_capacity, dtype=torch.float32, device=self.device
            )
            node_weight[:self.node_count].copy_(
                self._node_weight[:self.node_count]
            )
            self._node_weight = node_weight
            for name in (
                "_psi", "_node_flux", "_node_nonlinear", "_input_phase"
            ):
                source = getattr(self, name)
                target = torch.zeros(
                    new_node_capacity, dtype=source.dtype, device=source.device
                )
                target[:self.node_count].copy_(source[:self.node_count])
                setattr(self, name, target)
            input_mask = torch.zeros(
                new_node_capacity, dtype=torch.bool, device=self.device
            )
            input_mask[:self.node_count].copy_(self._input_mask[:self.node_count])
            self._input_mask = input_mask
            input_indices = torch.zeros(
                new_node_capacity, dtype=torch.long, device=self.device
            )
            input_indices[:self._input_count].copy_(
                self._input_indices[:self._input_count]
            )
            self._input_indices = input_indices
            self._node_ordinals = torch.arange(
                new_node_capacity, dtype=torch.long, device=self.device
            )
            force = torch.zeros(
                (new_node_capacity, 3),
                dtype=torch.float32,
                device=self.device,
            )
            force[:self.node_count].copy_(self._force[:self.node_count])
            self._force = force
            self.node_capacity = new_node_capacity
            old_video = self.video
            self.video = self._new_video_buffer(new_node_capacity)
            self._retired_video_buffers.append(old_video)

        if new_edge_capacity != old_edge_capacity:
            def grow(source, *, fill: float = 0.0):
                shape = (new_edge_capacity, *source.shape[1:])
                target = torch.full(
                    shape,
                    fill,
                    dtype=source.dtype,
                    device=source.device,
                )
                target[:self.edge_count].copy_(source[:self.edge_count])
                return target

            self._edge_index = grow(self._edge_index)
            self._base_rest = grow(self._base_rest, fill=1.0)
            self._edge_phase = grow(self._edge_phase)
            self._rest_state = grow(self._rest_state, fill=1.0)
            self._edge_schedule_group = grow(self._edge_schedule_group)
            self._edge_occupancy = grow(self._edge_occupancy)
            self._edge_weight = grow(self._edge_weight)
            self._edge_flux_raw = grow(self._edge_flux_raw)
            self._edge_activation = grow(self._edge_activation)
            self._edge_stiffness = grow(self._edge_stiffness, fill=1.0)
            schedule_host = np.zeros(new_edge_capacity, dtype=np.int64)
            schedule_host[:self.edge_count] = self._edge_schedule_group_host[
                :self.edge_count
            ]
            self._edge_schedule_group_host = schedule_host
            self.edge_capacity = new_edge_capacity

        # Copies and video observations already queued against the old pages
        # remain valid. Retaining the logarithmic sequence of old allocations
        # avoids a device-wide synchronization during growth.
        self._retired_resident_storage.append(old_storage)

    def set_haze(self, scores: Mapping[str, float]) -> None:
        self._haze = {str(key): float(value) for key, value in scores.items()}
        # Haze is presentation metadata. Rebuild it at the next topology
        # publication, never in the video-frame hot path.
        self._presentation = _PresentationState(
            revision=-1,
            node_colors=self._presentation.node_colors,
            node_sizes=self._presentation.node_sizes,
            edge_indices=self._presentation.edge_indices,
            edge_colors=self._presentation.edge_colors,
            camera_center=self._presentation.camera_center,
            camera_radius=self._presentation.camera_radius,
        )

    def _presentation_for(self, graph: "VisualGraph") -> _PresentationState:
        """Build immutable colors/endpoints once per topology revision."""

        if (
            self._presentation.revision == int(graph.revision)
            and len(self._presentation.node_colors) == len(graph.nodes)
            and len(self._presentation.edge_indices) == len(graph.edges)
        ):
            return self._presentation
        index = {node.key: ordinal for ordinal, node in enumerate(graph.nodes)}
        node_colors = np.zeros((len(graph.nodes), 4), dtype=np.float32)
        node_sizes = np.full(len(graph.nodes), 9.0, dtype=np.float32)
        for ordinal, node in enumerate(graph.nodes):
            hue = _phase(node.group or node.kind or node.network)
            node_colors[ordinal, :3] = colorsys.hsv_to_rgb(hue, 0.88, 1.0)
            node_colors[ordinal, 3] = min(
                1.0, 0.72 + abs(self._haze.get(node.key, 0.0)) * 0.28
            )
            if node.kind in {"feed", "argument", "input", "output"}:
                node_sizes[ordinal] = 12.0
        if graph.edges:
            edge_indices = np.asarray(
                [(index[edge.source], index[edge.target]) for edge in graph.edges],
                dtype=np.int64,
            )
            edge_colors = np.zeros((len(graph.edges), 4), dtype=np.float32)
            for ordinal, edge in enumerate(graph.edges):
                edge_colors[ordinal, :3] = colorsys.hsv_to_rgb(
                    _phase(edge.role), 0.78, 1.0
                )
                edge_colors[ordinal, 3] = 0.24
            edge_colors = np.repeat(edge_colors, 2, axis=0)
        else:
            edge_indices = np.zeros((0, 2), dtype=np.int64)
            edge_colors = np.zeros((0, 4), dtype=np.float32)
        for array in (node_colors, node_sizes, edge_indices, edge_colors):
            array.setflags(write=False)
        network_count = len({node.network for node in graph.nodes})
        camera_center = np.asarray(
            (max(0, network_count - 1) * 2.25, 0.0, 0.0), dtype=np.float32
        )
        # The physical world has no boundary. This is only a conservative
        # camera estimate and does not feed back into integration.
        camera_radius = max(
            8.0,
            network_count * 3.0,
            math.sqrt(max(1, len(graph.nodes))) * 0.32,
        )
        camera_center.setflags(write=False)
        self._presentation = _PresentationState(
            revision=int(graph.revision),
            node_colors=node_colors,
            node_sizes=node_sizes,
            edge_indices=edge_indices,
            edge_colors=edge_colors,
            camera_center=camera_center,
            camera_radius=camera_radius,
        )
        return self._presentation

    def _initial_position(self, key: str, network: str, network_order: int) -> np.ndarray:
        phase = _phase(f"{self.seed}:{key}") * math.tau
        return np.asarray(
            (
                network_order * 4.5 + 0.35 * math.sin(phase),
                1.2 * math.cos(phase),
                1.2 * math.sin(2.0 * phase),
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _spawn_jitter(key: str, radius: float = 0.16) -> np.ndarray:
        """Return a deterministic isotropic offset without a preferred axis."""

        direction = np.asarray(
            [2.0 * _phase(f"{key}:{axis}") - 1.0 for axis in "xyz"],
            dtype=np.float32,
        )
        length = float(np.linalg.norm(direction))
        if length < 1.0e-6:
            direction[:] = (1.0, 0.0, 0.0)
            length = 1.0
        return direction * (float(radius) / length)

    def _neighbor_grown_positions(
        self,
        nodes: tuple[object, ...],
        edges: tuple[object, ...],
        start_node: int,
    ):
        """Place a new suffix from its incident geometry and local stress."""

        torch = self.torch
        count = len(nodes)
        if not count:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)

        # New-new connectivity defines components that should emerge together.
        parent = list(range(count))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        indexed_edges: list[tuple[int, int, object]] = []
        for edge in edges:
            source = self._key_to_index[edge.source]
            target = self._key_to_index[edge.target]
            indexed_edges.append((source, target, edge))
            if source >= start_node and target >= start_node:
                union(source - start_node, target - start_node)
        roots: dict[int, int] = {}
        component = np.empty(count, dtype=np.int64)
        for ordinal in range(count):
            root = find(ordinal)
            component[ordinal] = roots.setdefault(root, len(roots))
        component_count = len(roots)

        # A component is seeded at the centroid of every live node it touches.
        anchor_components: list[int] = []
        anchor_nodes: list[int] = []
        for source, target, _edge in indexed_edges:
            if source >= start_node and target < start_node:
                anchor_components.append(int(component[source - start_node]))
                anchor_nodes.append(target)
            elif target >= start_node and source < start_node:
                anchor_components.append(int(component[target - start_node]))
                anchor_nodes.append(source)

        if start_node:
            sample_count = min(256, start_node)
            sample = torch.linspace(
                0, start_node - 1, sample_count,
                dtype=torch.long, device=self.device,
            )
            fallback = self._position_pages[self._current_page][sample].mean(dim=0)
        else:
            fallback = torch.zeros(3, dtype=torch.float32, device=self.device)
        seeds = fallback[None, :].repeat(component_count, 1)
        if anchor_nodes:
            anchor_component_tensor = torch.as_tensor(
                anchor_components, dtype=torch.long, device=self.device
            )
            anchor_node_tensor = torch.as_tensor(
                anchor_nodes, dtype=torch.long, device=self.device
            )
            sums = torch.zeros(
                (component_count, 3), dtype=torch.float32, device=self.device
            )
            counts = torch.zeros(
                component_count, dtype=torch.float32, device=self.device
            )
            sums.index_add_(
                0,
                anchor_component_tensor,
                self._position_pages[self._current_page][anchor_node_tensor],
            )
            counts.index_add_(
                0,
                anchor_component_tensor,
                torch.ones_like(anchor_component_tensor, dtype=torch.float32),
            )
            anchored = counts > 0
            seeds[anchored] = sums[anchored] / counts[anchored, None]
        component_tensor = torch.as_tensor(
            component, dtype=torch.long, device=self.device
        )
        jitter = torch.as_tensor(
            np.stack([self._spawn_jitter(node.key) for node in nodes]),
            dtype=torch.float32,
            device=self.device,
        )
        positions = seeds[component_tensor] + jitter

        # Minimize local edge-length stress. Only new endpoints move; existing
        # geometry is sampled once and remains an immutable boundary condition.
        incident = [
            item for item in indexed_edges
            if item[0] >= start_node or item[1] >= start_node
        ]
        if not incident:
            return positions
        source_host = np.asarray([item[0] for item in incident], dtype=np.int64)
        target_host = np.asarray([item[1] for item in incident], dtype=np.int64)
        rest_host = np.asarray([
            1.6 if item[2].role == "handoff" else 1.05 for item in incident
        ], dtype=np.float32)
        source_new_host = source_host >= start_node
        target_new_host = target_host >= start_node
        source_new_rows = np.flatnonzero(source_new_host)
        target_new_rows = np.flatnonzero(target_new_host)
        source_local = source_host[source_new_host] - start_node
        target_local = target_host[target_new_host] - start_node
        edge_count = len(incident)
        source_fixed = torch.zeros(
            (edge_count, 3), dtype=torch.float32, device=self.device
        )
        target_fixed = torch.zeros_like(source_fixed)
        position_page = self._position_pages[self._current_page]
        if (~source_new_host).any():
            rows = np.flatnonzero(~source_new_host)
            source_fixed[torch.as_tensor(rows, device=self.device)] = position_page[
                torch.as_tensor(source_host[rows], dtype=torch.long, device=self.device)
            ]
        if (~target_new_host).any():
            rows = np.flatnonzero(~target_new_host)
            target_fixed[torch.as_tensor(rows, device=self.device)] = position_page[
                torch.as_tensor(target_host[rows], dtype=torch.long, device=self.device)
            ]
        source_rows = torch.as_tensor(source_new_rows, dtype=torch.long, device=self.device)
        target_rows = torch.as_tensor(target_new_rows, dtype=torch.long, device=self.device)
        source_local_tensor = torch.as_tensor(
            source_local, dtype=torch.long, device=self.device
        )
        target_local_tensor = torch.as_tensor(
            target_local, dtype=torch.long, device=self.device
        )
        rest = torch.as_tensor(rest_host, dtype=torch.float32, device=self.device)
        degree = torch.zeros(count, dtype=torch.float32, device=self.device)
        if len(source_local):
            degree.index_add_(
                0, source_local_tensor, torch.ones_like(source_local_tensor, dtype=torch.float32)
            )
        if len(target_local):
            degree.index_add_(
                0, target_local_tensor, torch.ones_like(target_local_tensor, dtype=torch.float32)
            )
        degree.clamp_min_(1.0)
        for _ in range(8):
            source_position = source_fixed.clone()
            target_position = target_fixed.clone()
            if len(source_local):
                source_position[source_rows] = positions[source_local_tensor]
            if len(target_local):
                target_position[target_rows] = positions[target_local_tensor]
            displacement = source_position - target_position
            length = displacement.norm(dim=1).clamp_min(1.0e-5)
            correction = -(
                (length - rest)[:, None] * displacement / length[:, None]
            )
            update = torch.zeros_like(positions)
            if len(source_local):
                update.index_add_(0, source_local_tensor, correction[source_rows])
            if len(target_local):
                update.index_add_(0, target_local_tensor, -correction[target_rows])
            update.mul_(0.38).div_(degree[:, None])
            update_length = update.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
            update.mul_(torch.clamp(0.30 / update_length, max=1.0))
            positions.add_(update)
        return positions

    def _ensure_presentation_capacity(self, nodes: int, edges: int) -> None:
        """Grow host-side static attributes geometrically, never per frame."""

        if nodes > self._presentation_node_capacity:
            capacity = max(256, self._presentation_node_capacity)
            while capacity < nodes:
                capacity *= 2
            colors = np.zeros((capacity, 4), dtype=np.float32)
            sizes = np.zeros(capacity, dtype=np.float32)
            old = min(self.node_count, len(self._presentation.node_colors))
            if old:
                colors[:old] = self._presentation.node_colors[:old]
                sizes[:old] = self._presentation.node_sizes[:old]
            self._node_colors_host = colors
            self._node_sizes_host = sizes
            self._presentation_node_capacity = capacity
        if edges > self._presentation_edge_capacity:
            capacity = max(256, self._presentation_edge_capacity)
            while capacity < edges:
                capacity *= 2
            indices = np.zeros((capacity, 2), dtype=np.int64)
            colors = np.zeros((capacity * 2, 4), dtype=np.float32)
            old = min(self.edge_count, len(self._presentation.edge_indices))
            if old:
                indices[:old] = self._presentation.edge_indices[:old]
                colors[:old * 2] = self._presentation.edge_colors[:old * 2]
            self._edge_indices_host = indices
            self._edge_colors_host = colors
            self._presentation_edge_capacity = capacity

    def _write_node_style(self, ordinal: int, node: object) -> None:
        execution_kind = str(getattr(node, "execution_kind", "") or "")
        semantic_kind = execution_kind or str(node.kind)
        semantic_hues = {
            "branch": 0.13,
            "branch-merge": 0.16,
            "loop-header": 0.82,
            "loop-latch": 0.90,
            "loop-exit": 0.76,
            "deployment": 0.53,
            "deployment-lanes": 0.49,
            "deployment-join": 0.59,
            "dispatch-enter": 0.30,
            "dispatch-instruction": 0.36,
            "dispatch-exit": 0.42,
            "call": 0.05,
            "call-return": 0.10,
        }
        hue = semantic_hues.get(
            semantic_kind, _phase(node.group or node.kind or node.network)
        )
        saturation = 0.96 if execution_kind else 0.78
        rgb = np.asarray(colorsys.hsv_to_rgb(hue, saturation, 1.0), dtype=np.float32)
        finalized = str(getattr(node, "state", "live")) == "finalized"
        if finalized:
            # A closed compiler-owned IR/shader surface is visibly sealed,
            # analogous to DualIR rollup state. This is retained state, not a
            # renderer-side claim that compilation succeeded.
            rgb = rgb * 0.72 + 0.28
        self._node_colors_host[ordinal, :3] = rgb
        self._node_colors_host[ordinal, 3] = min(
            1.0,
            (0.9 if finalized else 0.72)
            + abs(self._haze.get(node.key, 0.0)) * 0.28,
        )
        size = (
            12.0 if node.kind in {"feed", "argument", "input", "output"} else 9.0
        )
        if str(getattr(node, "group", "")).startswith("backend:"):
            size += 3.0
        if execution_kind:
            size += 2.0 + min(3.0, float(getattr(node, "loop_depth", 0)))
        if finalized:
            size += 1.5
        self._node_sizes_host[ordinal] = size

    def _write_edge_style(self, ordinal: int, edge: object) -> None:
        self._edge_indices_host[ordinal] = (
            self._key_to_index[edge.source], self._key_to_index[edge.target]
        )
        semantic_hues = {
            "control-next": 0.12,
            "branch-true": 0.31,
            "branch-false": 0.01,
            "branch-merge": 0.16,
            "loop-body": 0.82,
            "loop-latch": 0.88,
            "loop-back": 0.94,
            "loop-exit": 0.73,
            "deployment-lane": 0.50,
            "deployment-join": 0.58,
            "dispatch-membership": 0.39,
            "dispatch-feed": 0.28,
            "dispatch-result": 0.43,
        }
        hue = semantic_hues.get(edge.role, _phase(edge.role))
        alpha = 0.42 if edge.role in semantic_hues else 0.24
        color = np.asarray(
            (*colorsys.hsv_to_rgb(hue, 0.88, 1.0), alpha),
            dtype=np.float32,
        )
        self._edge_colors_host[ordinal * 2:ordinal * 2 + 2] = color

    def _incremental_presentation(
        self, revision: int, node_count: int, edge_count: int
    ) -> _PresentationState:
        network_count = max(1, len(self._network_order_map))
        center = np.asarray(
            ((network_count - 1) * 2.25, 0.0, 0.0), dtype=np.float32
        )
        return _PresentationState(
            revision=int(revision),
            node_colors=self._node_colors_host[:node_count],
            node_sizes=self._node_sizes_host[:node_count],
            edge_indices=self._edge_indices_host[:edge_count],
            edge_colors=self._edge_colors_host[:edge_count * 2],
            camera_center=center,
            camera_radius=max(
                8.0,
                network_count * 3.0,
                math.sqrt(max(1, node_count)) * 0.32,
            ),
        )

    def _camera_indices_for(self, presentation: _PresentationState):
        """Return resident indices of nodes participating in visible edges."""

        if len(presentation.edge_indices):
            indices = np.unique(presentation.edge_indices.reshape(-1))
        else:
            indices = np.arange(len(presentation.node_colors), dtype=np.int64)
        return self.torch.as_tensor(
            indices, dtype=self.torch.long, device=self.device
        )

    def _record_topology_ready(self):
        """Record completion of resident topology writes on their CUDA stream."""

        if self.device.type != "cuda":
            return None
        event = self.torch.cuda.Event(blocking=False)
        event.record(self.torch.cuda.current_stream(self.device))
        return event

    def _install_compiler_schedule(self, edge_count: int) -> None:
        """Project compiler-authored node groups onto their incoming edges.

        No levels are calculated here. ``VisualNode.schedule_group`` is the
        ordinal published by the authoritative IR scheduler; an edge simply
        activates when its target component executes in that group.
        """

        groups = self._edge_schedule_group_host[:edge_count]
        groups.fill(-1)
        execution_mode = any(self._node_execution_groups)
        self.execution_plan_active = execution_mode
        self.execution_scheduled_node_count = sum(
            bool(groups) for groups in self._node_execution_groups
        )
        self.local_scheduled_node_count = sum(
            group >= 0 for group in self._node_schedule_groups
        )
        if edge_count:
            edge_indices = (
                self._edge_indices_host[:edge_count]
                if len(self._edge_indices_host) >= edge_count
                else self._presentation.edge_indices[:edge_count]
            )
            targets = edge_indices[:, 1]
            node_groups = np.asarray(
                [
                    memberships[0]
                    if execution_mode and memberships else
                    -1 if execution_mode else schedule_group
                    for memberships, schedule_group in zip(
                        self._node_execution_groups,
                        self._node_schedule_groups,
                    )
                ],
                dtype=np.int64,
            )
            groups[:] = node_groups[targets]
            self._edge_schedule_group[:edge_count].copy_(self.torch.as_tensor(
                groups, dtype=self.torch.long, device=self.device
            ))
        authored_groups = (
            [group for memberships in self._node_execution_groups for group in memberships]
            if execution_mode else groups[groups >= 0]
        )
        self.schedule_group_count = (
            int(max(authored_groups)) + 1 if len(authored_groups) else 0
        )
        self._schedule_snapshot = groups

    def apply_delta(self, delta: "VisualGraphDelta") -> None:
        """Install only newly published membership into resident storage."""

        self._assert_mutation_owner()
        torch = self.torch
        added_nodes = tuple(delta.nodes_added)
        added_edges = tuple(delta.edges_added)
        start_node = self.node_count
        start_edge = self.edge_count
        for node in added_nodes:
            if node.key in self._key_to_index:
                raise ValueError(f"duplicate visual node in delta: {node.key}")
        for edge in added_edges:
            key = (edge.source, edge.target, edge.role)
            if key in self._edge_key_set:
                raise ValueError(f"duplicate visual edge in delta: {key}")
        new_node_count = start_node + len(added_nodes)
        new_edge_count = start_edge + len(added_edges)
        committed_input_count = self._input_count
        self._ensure_resident_capacity(new_node_count, new_edge_count)
        self._ensure_presentation_capacity(new_node_count, new_edge_count)

        if added_nodes:
            for node in added_nodes:
                network = node.network or delta.source_kind
                self._network_order_map.setdefault(
                    network, len(self._network_order_map)
                )
                ordinal = len(self._keys)
                self._keys.append(node.key)
                self._node_kinds.append(node.kind)
                self._node_schedule_groups.append(
                    -1 if node.schedule_group is None else int(node.schedule_group)
                )
                self._node_execution_groups.append(tuple(map(
                    int, getattr(node, "execution_groups", ()) or ()
                )))
                self._node_states.append(str(node.state))
                self._key_to_index[node.key] = ordinal
                self._write_node_style(ordinal, node)
            initial = self._neighbor_grown_positions(
                added_nodes, added_edges, start_node
            )
            for positions, velocities in zip(self._position_pages, self._velocity_pages):
                positions[start_node:new_node_count].copy_(initial)
                velocities[start_node:new_node_count].zero_()
            self._node_occupancy[start_node:new_node_count].fill_(True)
            self._node_weight[start_node:new_node_count].fill_(1.0)
            phases = torch.as_tensor(
                [_phase(node.key) * math.tau for node in added_nodes],
                dtype=torch.float32,
                device=self.device,
            )
            self._input_phase[start_node:new_node_count].copy_(phases)
            self._psi[start_node:new_node_count].copy_(torch.sin(phases))
            for offset, node in enumerate(added_nodes):
                if node.kind in {"feed", "argument", "input"}:
                    ordinal = start_node + offset
                    self._input_mask[ordinal] = True
                    self._input_indices[committed_input_count] = ordinal
                    committed_input_count += 1

        input_kind_changed = False
        for node in delta.nodes_updated:
            ordinal = self._key_to_index.get(node.key)
            if ordinal is not None:
                input_kind_changed |= (
                    (self._node_kinds[ordinal] in {"feed", "argument", "input"})
                    != (node.kind in {"feed", "argument", "input"})
                )
                self._node_kinds[ordinal] = node.kind
                self._node_schedule_groups[ordinal] = (
                    -1 if node.schedule_group is None else int(node.schedule_group)
                )
                self._node_execution_groups[ordinal] = tuple(map(
                    int, getattr(node, "execution_groups", ()) or ()
                ))
                self._node_states[ordinal] = str(node.state)
                self._write_node_style(ordinal, node)
        if input_kind_changed:
            input_ordinals = [
                ordinal for ordinal, kind in enumerate(self._node_kinds)
                if kind in {"feed", "argument", "input"}
            ]
            self._input_mask[:new_node_count].zero_()
            committed_input_count = len(input_ordinals)
            if input_ordinals:
                indices = torch.as_tensor(
                    input_ordinals, dtype=torch.long, device=self.device
                )
                self._input_indices[:committed_input_count].copy_(indices)
                self._input_mask[indices] = True

        if added_edges:
            endpoints_host = np.asarray([
                (self._key_to_index[edge.source], self._key_to_index[edge.target])
                for edge in added_edges
            ], dtype=np.int64)
            endpoints = torch.as_tensor(
                endpoints_host, dtype=torch.long, device=self.device
            )
            self._edge_index[start_edge:new_edge_count].copy_(endpoints)
            positions = self._position_pages[self._current_page]
            semantic_rest = torch.as_tensor(
                [1.6 if edge.role == "handoff" else 1.05 for edge in added_edges],
                dtype=torch.float32,
                device=self.device,
            )
            self._base_rest[start_edge:new_edge_count].copy_(semantic_rest)
            self._rest_state[start_edge:new_edge_count].copy_(
                self._base_rest[start_edge:new_edge_count]
            )
            self._edge_occupancy[start_edge:new_edge_count].fill_(True)
            self._edge_weight[start_edge:new_edge_count].fill_(1.0)
            self._edge_phase[start_edge:new_edge_count].copy_(torch.as_tensor(
                [_phase(edge.role) * math.tau for edge in added_edges],
                dtype=torch.float32,
                device=self.device,
            ))
            self._edge_schedule_group[start_edge:new_edge_count].fill_(-1)
            self._edge_schedule_group_host[start_edge:new_edge_count] = -1
            for offset, edge in enumerate(added_edges):
                ordinal = start_edge + offset
                edge_key = (edge.source, edge.target, edge.role)
                self._edge_keys.append(edge_key)
                self._edge_key_set.add(edge_key)
                self._write_edge_style(ordinal, edge)

        self._install_compiler_schedule(new_edge_count)
        presentation = self._incremental_presentation(
            delta.revision, new_node_count, new_edge_count
        )
        graph = type(self.graph)((), (), delta.source_kind, int(delta.revision))
        schedule_snapshot = self._schedule_snapshot
        # This tuple assignment is the lock-free occupancy publication.
        # Physics and video readers see one complete old/new prefix.
        self._input_count = committed_input_count
        self.node_count = new_node_count
        self.edge_count = new_edge_count
        self.graph = graph
        self._presentation = presentation
        self._camera_fit_indices = self._camera_indices_for(presentation)
        self._schedule_snapshot = schedule_snapshot
        self._repulsion_neighbors = None
        sample_budget = 2_097_152
        maximum = max(1, sample_budget // max(1, self.node_count))
        self._repulsion_batch_width = max(
            width for width in (1, 2, 4, 8, 16, 32)
            if width <= min(32, maximum)
        )
        self._topology_publication = (
            new_node_count,
            new_edge_count,
            committed_input_count,
            presentation,
            graph,
            schedule_snapshot,
            self._camera_fit_indices,
            self._record_topology_ready(),
        )

    def replace_graph(
        self,
        graph: "VisualGraph",
    ) -> None:
        """Append topology into growable device storage without resetting state."""

        self._assert_mutation_owner()
        torch = self.torch
        previous_node_count = self.node_count
        node_keys = [node.key for node in graph.nodes]
        edge_keys = [(edge.source, edge.target, edge.role) for edge in graph.edges]
        if node_keys[: len(self._keys)] != self._keys:
            raise ValueError("GPU spring topology requires append-only node identities")
        if edge_keys[: len(self._edge_keys)] != self._edge_keys:
            raise ValueError("GPU spring topology requires append-only edge identities")
        self._ensure_resident_capacity(len(node_keys), len(edge_keys))

        network_order: dict[str, int] = {}
        for node in graph.nodes:
            network_order.setdefault(node.network or graph.source_kind, len(network_order))
        start_node = len(self._keys)
        if len(node_keys) > start_node:
            initial = torch.as_tensor(
                np.stack([
                    self._initial_position(
                        node.key,
                        node.network or graph.source_kind,
                        network_order[node.network or graph.source_kind],
                    )
                    for node in graph.nodes[start_node:]
                ]),
                dtype=torch.float32,
                device=self.device,
            )
            for positions, velocities in zip(self._position_pages, self._velocity_pages):
                positions[start_node:len(node_keys)].copy_(initial)
                velocities[start_node:len(node_keys)].zero_()
            self._node_occupancy[start_node:len(node_keys)].fill_(True)
            self._node_weight[start_node:len(node_keys)].fill_(1.0)
            phases = torch.as_tensor(
                [_phase(node.key) * math.tau for node in graph.nodes[start_node:]],
                dtype=torch.float32,
                device=self.device,
            )
            self._input_phase[start_node:len(node_keys)].copy_(phases)
            self._psi[start_node:len(node_keys)].copy_(torch.sin(phases))
            input_membership = torch.as_tensor(
                [
                    node.kind in {"feed", "argument", "input"}
                    for node in graph.nodes[start_node:]
                ],
                dtype=torch.bool,
                device=self.device,
            )
            self._input_mask[start_node:len(node_keys)].copy_(input_membership)

        index = {key: ordinal for ordinal, key in enumerate(node_keys)}
        start_edge = len(self._edge_keys)
        if len(edge_keys) > start_edge:
            endpoints = torch.as_tensor(
                [
                    (index[edge.source], index[edge.target])
                    for edge in graph.edges[start_edge:]
                ],
                dtype=torch.long,
                device=self.device,
            )
            stop = len(edge_keys)
            self._edge_index[start_edge:stop].copy_(endpoints)
            positions = self._position_pages[self._current_page]
            delta = positions[endpoints[:, 1]] - positions[endpoints[:, 0]]
            self._base_rest[start_edge:stop].copy_(delta.norm(dim=1).clamp_min(0.25))
            self._rest_state[start_edge:stop].copy_(self._base_rest[start_edge:stop])
            phases = [
                _phase(edge.role) * math.tau for edge in graph.edges[start_edge:]
            ]
            self._edge_phase[start_edge:stop].copy_(torch.as_tensor(
                phases, dtype=torch.float32, device=self.device
            ))
            self._edge_occupancy[start_edge:stop].fill_(True)
            self._edge_weight[start_edge:stop].fill_(1.0)

        self.graph = graph
        self._keys = node_keys
        self._edge_keys = edge_keys
        self._edge_key_set = set(edge_keys)
        self._key_to_index = {key: ordinal for ordinal, key in enumerate(node_keys)}
        self._network_order_map = dict(network_order)
        self._node_kinds = [node.kind for node in graph.nodes]
        authored_levels = tuple(sorted({
            int(node.level)
            for node in graph.nodes
            if node.level is not None
        }))
        group_by_level = {
            level: ordinal for ordinal, level in enumerate(authored_levels)
        }
        self._node_schedule_groups = [
            int(node.schedule_group)
            if node.schedule_group is not None
            else group_by_level[int(node.level)]
            if node.level is not None
            else -1
            for node in graph.nodes
        ]
        self._node_execution_groups = [
            tuple(map(int, getattr(node, "execution_groups", ()) or ()))
            for node in graph.nodes
        ]
        self._node_states = [str(node.state) for node in graph.nodes]
        self.node_count = len(node_keys)
        self.edge_count = len(edge_keys)
        input_ordinals = [
            ordinal
            for ordinal, node in enumerate(graph.nodes)
            if node.kind in {"feed", "argument", "input"}
        ]
        self._input_count = len(input_ordinals)
        if input_ordinals:
            self._input_indices[:self._input_count].copy_(torch.as_tensor(
                input_ordinals, dtype=torch.long, device=self.device
            ))
        self._presentation_for(graph)
        self._camera_fit_indices = self._camera_indices_for(self._presentation)
        self._install_compiler_schedule(self.edge_count)
        # Deterministic sampled neighbors depend only on the occupied node
        # prefix. Cache the modest table for ordinary graphs; at huge counts,
        # derive bounded chunks on device so the cache cannot consume hundreds
        # of megabytes. Schedule-only revisions do neither operation.
        if self.node_count != previous_node_count:
            if self.node_count <= 65_536:
                ordinals = self._node_ordinals[:self.node_count]
                self._repulsion_neighbors = (
                    ordinals[:, None]
                    + self._repulsion_offsets[None, :] * 2654435761
                ) % self.node_count
                self._repulsion_batch_width = 32
            else:
                self._repulsion_neighbors = None
                sample_budget = 2_097_152
                maximum = max(1, sample_budget // self.node_count)
                self._repulsion_batch_width = max(
                    width for width in (1, 2, 4, 8, 16, 32)
                    if width <= min(32, maximum)
                )
        self._topology_publication = (
            self.node_count,
            self.edge_count,
            self._input_count,
            self._presentation,
            self.graph,
            self._schedule_snapshot,
            self._camera_fit_indices,
            self._record_topology_ready(),
        )

    def _copying_state_pages(self) -> set[int]:
        video = self.video
        return {
            int(page.source_state_page)
            for page in video.frames
            if page.pending
        }

    def _poll_video_copies(self) -> None:
        video = self.video
        for page in video.frames:
            if page.pending and (page.event is None or page.event.query()):
                page.pending = False
                page.ready = True

    def _poll_completed_rate(self) -> None:
        if self.device.type != "cuda":
            now = time.perf_counter()
            elapsed = now - float(self._rate_start)
            if elapsed >= 0.5:
                self.completed_hz = (self.steps - self._rate_start_step) / elapsed
                self._rate_start = now
                self._rate_start_step = self.steps
            return
        remaining = []
        for start, end, count in self._rate_windows:
            if end.query():
                elapsed_ms = float(start.elapsed_time(end))
                if elapsed_ms > 0.0:
                    self.completed_hz = count * 1000.0 / elapsed_ms
            else:
                remaining.append((start, end, count))
        self._rate_windows = remaining[-4:]

    def _destination_page(self) -> int:
        blocked = self._copying_state_pages()
        for offset in range(1, len(self._position_pages) + 1):
            candidate = (self._current_page + offset) % len(self._position_pages)
            if candidate not in blocked:
                return candidate
        # A slow video path is never allowed to stop physics. Grow the device
        # state ring instead of waiting for an observation copy.
        self._position_pages.append(self.torch.empty_like(self._position_pages[0]))
        self._velocity_pages.append(self.torch.empty_like(self._velocity_pages[0]))
        return len(self._position_pages) - 1

    def step(self, dt: float = 1.0 / 240.0) -> None:
        """Enqueue one entirely device-resident spring integration step."""

        self._assert_mutation_owner()
        torch = self.torch
        self._poll_video_copies()
        self._poll_completed_rate()
        publication = self._topology_publication
        topology_ready_event = publication[7]
        if (
            topology_ready_event is not None
            and topology_ready_event is not self._applied_topology_ready_event
        ):
            # Topology installation may enqueue on a different thread's CUDA
            # stream. Establish device-side ownership before gathering new
            # endpoints or positions; the host and compiler never wait.
            torch.cuda.current_stream(self.device).wait_event(
                topology_ready_event
            )
            self._applied_topology_ready_event = topology_ready_event
        n, e, input_count = publication[:3]
        if not n:
            return
        source_page = self._current_page
        destination_page = self._destination_page()
        position = self._position_pages[source_page][:n]
        velocity = self._velocity_pages[source_page][:n]
        base_dt = float(dt)
        if self.target_max_velocity > 0.0:
            # ``velocity`` is the completed previous state.  The exact global
            # reduction and timestep normalization stay device-resident; no
            # host readback can serialize the physics loop.
            self._previous_max_velocity.copy_(
                velocity.square().sum(dim=1).amax().sqrt()
            )
            self._dt_scale.copy_(
                self.target_max_velocity
                / self._previous_max_velocity.clamp_min(1.0e-6)
            )
            self._dt_scale.clamp_(self.min_dt_scale, self.max_dt_scale)
        else:
            self._dt_scale.fill_(1.0)
        self._effective_dt.fill_(base_dt).mul_(self._dt_scale)
        force = self._force[:n]
        force.zero_()
        if e:
            endpoints = self._edge_index[:e]
            source, target = endpoints[:, 0], endpoints[:, 1]
            edge_membership = self._edge_weight[:e]
            # FluxSpring pump_tick on resident arrays:
            #   dpsi = D0 @ psi
            #   q = w_e * mix(raw, tanh(raw), alpha_e)
            #   s = D0.T @ q
            #   psi += eta * w_n * mix(s, tanh(s), alpha_n) / N
            psi = self._psi[:n]
            psi.mul_(1.0 - float(dt) * 0.08)
            if input_count:
                input_indices = self._input_indices[:input_count]
                psi[input_indices] = torch.sin(
                    self.elapsed * 1.7 + self._input_phase[input_indices]
                )
            edge_raw = self._edge_flux_raw[:e]
            torch.sub(psi[target], psi[source], out=edge_raw)
            displacement = position[source] - position[target]
            length = displacement.norm(dim=1).clamp_min(1.0e-6)
            active = None
            if (
                self.edge_pulse_mode == "compiler-schedule"
                and self.schedule_group_count
            ):
                schedule_position = self.elapsed * self.schedule_hz
                self.active_schedule_group = (
                    int(schedule_position) % self.schedule_group_count
                )
                schedule_phase = schedule_position - math.floor(schedule_position)
                self.schedule_pulse = math.sin(math.pi * schedule_phase) ** 2
                active = (
                    self._edge_schedule_group[:e] == self.active_schedule_group
                ).to(self._base_rest.dtype) * edge_membership
                # The execution schedule is also a FluxSpring input: current
                # dependency edges receive a signed potential impulse, which
                # propagates through D0.T and reinforces functional grouping.
                edge_raw.add_(active, alpha=0.35 * self.schedule_pulse)
                target_rest = self._base_rest[:e] * (
                    1.0 - self.schedule_contraction * self.schedule_pulse * active
                )
                # The scheduled contraction is a physical input, not merely a
                # shader highlight. Inactive springs relax to their resident L0.
                response = 1.0 - math.exp(
                    -self.contraction_response_hz * float(dt)
                )
                self._rest_state[:e].lerp_(target_rest, response)
                rest = self._rest_state[:e]
            else:
                self.active_schedule_group = -1
                self.schedule_pulse = 0.0
                rest = self._base_rest[:e]
            edge_activation = self._edge_activation[:e]
            torch.tanh(edge_raw, out=edge_activation)
            edge_activation.mul_(0.55).add_(edge_raw, alpha=0.45)
            edge_activation.mul_(edge_membership)
            node_flux = self._node_flux[:n]
            node_flux.zero_()
            node_flux.index_add_(0, source, -edge_activation)
            node_flux.index_add_(0, target, edge_activation)
            node_nonlinear = self._node_nonlinear[:n]
            torch.tanh(node_flux, out=node_nonlinear)
            node_nonlinear.mul_(0.55).add_(node_flux, alpha=0.45)
            psi.add_(node_nonlinear, alpha=float(dt) * 0.7 / max(1, n))
            flux_stiffness = self._edge_stiffness[:e]
            torch.abs(edge_activation, out=flux_stiffness)
            torch.tanh(flux_stiffness, out=flux_stiffness)
            flux_stiffness.mul_(0.25).add_(1.0)
            edge_force = (
                self.spring_stiffness
                * (length - rest)[:, None]
                * displacement
                / length[:, None]
            ) * (edge_membership * flux_stiffness)[:, None]
            if active is not None and self.schedule_yank_impulse > 0.0:
                edge_force.add_(
                    (
                        2.0
                        * self.schedule_hz
                        * self.schedule_yank_impulse
                        * self.schedule_pulse
                        * active
                    )[:, None]
                    * displacement
                    / length[:, None]
                )
            force.index_add_(0, source, -edge_force)
            force.index_add_(0, target, edge_force)

        # Exact all-pairs repulsion for ordinary graphs; deterministic sampled
        # repulsion keeps the kernel O(N) once graphs become genuinely huge.
        if n <= 128:
            chunk = 256
            for start in range(0, n, chunk):
                stop = min(n, start + chunk)
                displacement = position[start:stop, None, :] - position[None, :, :]
                distance2 = (displacement * displacement).sum(dim=2).clamp_min(1.0e-4)
                rows = torch.arange(start, stop, device=self.device)
                distance2[torch.arange(stop - start, device=self.device), rows] = float("inf")
                member = self._node_weight[:n]
                force[start:stop] += (
                    self.repulsion_strength
                    * displacement
                    / distance2[:, :, None]
                    * member[None, :, None]
                ).sum(dim=1)
        else:
            ordinals = self._node_ordinals[:n]
            # Ordinary graphs reuse one compact resident neighbor table and
            # issue a single parallel batch. Huge graphs derive bounded chunks
            # so temporary displacement storage stays under a fixed budget.
            batch_width = self._repulsion_batch_width
            for offset_start in range(0, 32, batch_width):
                if self._repulsion_neighbors is not None:
                    neighbors = self._repulsion_neighbors[
                        :, offset_start:offset_start + batch_width
                    ]
                else:
                    offsets = self._repulsion_offsets[
                        offset_start:offset_start + batch_width
                    ]
                    neighbors = (
                        ordinals[:, None] + offsets[None, :] * 2654435761
                    ) % n
                displacement = position[:, None, :] - position[neighbors]
                distance2 = (
                    displacement * displacement
                ).sum(dim=2).clamp_min(1.0e-4)
                neighbor_membership = self._node_weight[neighbors]
                force += (
                    self.repulsion_strength
                    * displacement
                    / distance2[:, :, None]
                    * neighbor_membership[:, :, None]
                ).sum(dim=1)

        node_membership = self._node_occupancy[:n, None]
        force.masked_fill_(~node_membership, 0.0)
        # The destination page is the integration output. Do not allocate
        # transient next-position/next-velocity tensors and copy them into a
        # resident arena that already exists.
        new_velocity = self._velocity_pages[destination_page][:n]
        new_position = self._position_pages[destination_page][:n]
        new_velocity.copy_(force).mul_(self._effective_dt).add_(velocity)
        new_velocity.mul_(torch.exp(-self.damping * self._effective_dt))
        new_position.copy_(new_velocity).mul_(self._effective_dt).add_(position)
        # Inactive capacity is not state. This branch is free for the common
        # append-only case because every slot below the high-water mark is
        # occupied; it becomes authoritative if membership later develops
        # holes without reallocating the arena.
        torch.where(node_membership, new_position, position, out=new_position)
        torch.where(node_membership, new_velocity, velocity, out=new_velocity)
        if self.boundary_radius is not None:
            radius = self.boundary_radius
            distance = new_position.norm(dim=1, keepdim=True)
            escaped = distance > radius
            normal = new_position / distance.clamp_min(1.0e-6)
            projected = normal * radius
            outward = (new_velocity * normal).sum(dim=1, keepdim=True)
            slipped = new_velocity - outward.clamp_min(0.0) * normal
            torch.where(escaped, projected, new_position, out=new_position)
            torch.where(escaped, slipped, new_velocity, out=new_velocity)
        self._current_page = destination_page
        # Scheduling uses the nominal simulation clock; adaptive mechanical
        # time remains a device scalar so it cannot create a synchronization.
        self.elapsed += base_dt
        self.steps += 1
        if self.device.type == "cuda" and self.steps - self._rate_start_step >= 120:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
            self._rate_windows.append(
                (self._rate_start, end, self.steps - self._rate_start_step)
            )
            self._rate_start = end
            self._rate_start_step = self.steps
        self._publish_video_if_due(publication)

    def _publish_video_if_due(self, publication=None) -> None:
        now = time.perf_counter()
        if now < self._next_observation:
            return
        self._next_observation = now + self.observation_dt
        video = self.video
        page_index = video.get_write_page(0)
        page = video.frames[page_index]
        if page.pending or page.ready or page.in_use:
            return
        if publication is None:
            publication = self._topology_publication
        n = publication[0]
        page.node_count = n
        page.graph = publication[4]
        page.haze = None
        page.presentation = publication[3]
        page.sequence = self.steps
        page.source_state_page = self._current_page
        page.active_schedule_group = self.active_schedule_group
        page.schedule_pulse = self.schedule_pulse
        page.schedule_groups = publication[5]
        if self.device.type == "cuda":
            with self.torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_stream(self.torch.cuda.current_stream(self.device))
                observed_positions = self._position_pages[self._current_page][:n]
                page.positions[:n].copy_(observed_positions, non_blocking=True)
                if now >= self._next_camera_observation:
                    self._next_camera_observation = (
                        now + self._camera_observation_dt
                    )
                    # Fit the actual occupied field, not a node-count proxy.
                    # AABB midpoint/half-diagonal avoids the apparent camera
                    # drift caused by a lopsided population mean.
                    camera_positions = observed_positions[publication[6]]
                    lower = camera_positions.amin(dim=0)
                    upper = camera_positions.amax(dim=0)
                    center = (lower + upper) * 0.5
                    radius = (upper - lower).norm() * 0.5
                    self._camera_bounds_device[:3].copy_(center)
                    self._camera_bounds_device[3].copy_(radius)
                page.camera_bounds.copy_(
                    self._camera_bounds_device, non_blocking=True
                )
                page.event = self.torch.cuda.Event(blocking=False)
                page.event.record(self._copy_stream)
            page.pending = True
        else:
            page.positions[:n].copy_(self._position_pages[self._current_page][:n])
            if now >= self._next_camera_observation:
                self._next_camera_observation = now + self._camera_observation_dt
                observed = page.positions[:n][publication[6]]
                lower = observed.amin(dim=0)
                upper = observed.amax(dim=0)
                self._camera_bounds_device[:3].copy_((lower + upper) * 0.5)
                self._camera_bounds_device[3].copy_(
                    (upper - lower).norm() * 0.5
                )
            page.camera_bounds.copy_(self._camera_bounds_device)
            page.ready = True
        video.advance(0)

    def consume_layers(self, elapsed: float = 0.0):
        """Return the newest completed video page, or ``None`` without waiting."""

        self._poll_video_copies()
        video = self.video
        page_index = video.get_read_page(1)
        page = video.frames[page_index]
        if not page.ready or page.graph is None:
            return None
        n = page.node_count
        presentation = page.presentation
        if presentation is None:
            return None
        topology_revision = int(presentation.revision)
        if (
            int(page.sequence) <= self._last_consumed_sequence
            or topology_revision < self._last_consumed_topology_revision
        ):
            # An append-only graph and monotonic physics clock can never
            # legitimately regress. Drop a stale observation rather than
            # showing one frame from an older state/topology generation.
            page.ready = False
            page.in_use = False
            page.source_state_page = -1
            video.advance(1)
            return None
        active_group = page.active_schedule_group
        schedule_pulse = page.schedule_pulse
        schedule_groups = page.schedule_groups
        camera_bounds = np.array(page.camera_bounds.numpy(), copy=True)
        camera_center = camera_bounds[:3]
        camera_radius = max(1.0, float(camera_bounds[3]))
        page.ready = False
        page.in_use = self.device.type == "cuda"
        if not page.in_use:
            page.source_state_page = -1
        video.advance(1)
        self._last_consumed_sequence = int(page.sequence)
        self._last_consumed_topology_revision = topology_revision

        edge_indices = presentation.edge_indices
        activation_key = (
            presentation.revision,
            id(schedule_groups),
            int(active_group),
        )
        if activation_key != self._render_activation_key:
            active_edges = (
                schedule_groups == active_group
                if active_group >= 0 and len(edge_indices)
                else np.zeros(len(edge_indices), dtype=bool)
            )
            active_nodes = np.zeros(n, dtype=np.float32)
            if active_edges.any():
                active_nodes[
                    edge_indices[active_edges].reshape(-1)
                ] = 1.0
            self._render_active_nodes = active_nodes
            self._render_active_edges = np.repeat(
                active_edges.astype(np.float32, copy=False), 2
            )
            self._render_activation_key = activation_key
        if self.device.type == "cuda":
            released = False

            def release_page() -> None:
                nonlocal released
                if released:
                    return
                released = True
                page.in_use = False
                page.source_state_page = -1

            return {
                "cuda_graph": CudaGraphLayer(
                    positions=page.positions[:n],
                    node_count=n,
                    edge_indices=edge_indices,
                    node_colors=presentation.node_colors,
                    node_sizes=presentation.node_sizes,
                    edge_colors=presentation.edge_colors,
                    node_active=self._render_active_nodes,
                    edge_active=self._render_active_edges,
                    pulse=schedule_pulse,
                    width=1.5 + 4.0 * schedule_pulse,
                    topology_revision=presentation.revision,
                    activation_revision=activation_key,
                    camera_center=camera_center,
                    camera_radius=camera_radius,
                    release=release_page,
                )
            }

        positions = np.array(page.positions[:n].numpy(), dtype=np.float32, copy=True)
        if len(edge_indices):
            line_positions = positions[edge_indices.reshape(-1)]
            line_colors = presentation.edge_colors
        else:
            line_positions = np.zeros((0, 3), dtype=np.float32)
            line_colors = np.zeros((0, 4), dtype=np.float32)
        return {
            "lines": LineLayer(
                line_positions,
                colors=line_colors,
                width=1.5 + 4.0 * schedule_pulse,
                active=self._render_active_edges,
                pulse=schedule_pulse,
                topology_revision=presentation.revision,
                activation_revision=activation_key,
            ),
            "points": PointLayer(
                positions,
                colors=presentation.node_colors,
                sizes_px=presentation.node_sizes,
                size_px_default=11.0,
                active=self._render_active_nodes,
                pulse=schedule_pulse,
                topology_revision=presentation.revision,
                activation_revision=activation_key,
            ),
        }


__all__ = ["GpuResidentBoundSpringSimulation"]
