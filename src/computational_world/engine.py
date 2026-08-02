"""Headless managed world state machine; contains no graphics surface."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Any

from ..common.abstract_tensor_state_machine import (
    AbstractTensorStateMachine,
    TensorStateField,
)
from ..common.dt_system.dt_controller import STController, Targets
from ..common.dt_system.dt_scaler import Metrics
from ..common.dt_system.state_table import StateTable
from ..common.dt_system.time_runtime import (
    ManagedCommitGate,
    ManagedTimeRuntime,
    TimeAdvanceReport,
    TimeWindowRequest,
)
from ..common.tensors.abstraction import AbstractTensor
from .state import ComputationalWorldState
from .spring import BoundSpringParameters, advance_bound_spring


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Stable compiler provenance reference, never an owned compiler object."""

    sequence: int
    graph_id: str
    component_id: str
    kind: str
    artifact_reference: str = ""
    captured_ns: int | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("provenance sequence must be non-negative")
        if not self.graph_id or not self.component_id or not self.kind:
            raise ValueError("provenance identity and kind are required")


@dataclass(frozen=True, slots=True)
class WorldBoundaryEvent:
    """Records admitted at one authored managed-time event boundary."""

    event_time: float
    provenance: tuple[ProvenanceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.event_time)):
            raise ValueError("world event time must be finite")


@dataclass(frozen=True, slots=True)
class WorldStatusBatch:
    """Immutable shell material captured only after game-mode admission."""

    player_intent: tuple[float, float, float] = (0.0, 0.0, 0.0)
    boundary_events: tuple[WorldBoundaryEvent, ...] = ()

    def __post_init__(self) -> None:
        if len(self.player_intent) != 3 or not all(
            math.isfinite(float(value)) for value in self.player_intent
        ):
            raise ValueError("player intent must be three finite values")
        times = tuple(float(event.event_time) for event in self.boundary_events)
        if times != tuple(sorted(set(times))):
            raise ValueError("world boundary events must be unique and ordered")
        sequences = tuple(
            record.sequence
            for event in self.boundary_events
            for record in event.provenance
        )
        if sequences != tuple(sorted(set(sequences))):
            raise ValueError("provenance records must be unique and ordered")


class ComputationalWorld(AbstractTensorStateMachine):
    """Sparse voxel/object/player/provenance mechanics under managed dt."""

    state_fields = (
        TensorStateField("entity_id", ("N",), "int64", scope="world"),
        TensorStateField("position", ("N", 3), "float32", scope="world"),
        TensorStateField("velocity", ("N", 3), "float32", scope="world"),
        TensorStateField("edge_index", (2, "E"), "int64", scope="world"),
        TensorStateField("occupied_block_coord", ("K", 3), "int64", scope="world"),
        TensorStateField("player_intent", (1, 3), "float32", scope="world"),
        TensorStateField("provenance_cursor", (1,), "int64", scope="world"),
        TensorStateField("managed_time", (1,), "float64", scope="world"),
        TensorStateField("phase", (1,), "int32", scope="world"),
    )

    def __init__(
        self,
        state: ComputationalWorldState,
        *,
        spring_parameters: BoundSpringParameters | None = None,
    ) -> None:
        self._state = state
        self.spring_parameters = spring_parameters or BoundSpringParameters()
        self.world_time = float(state.managed_time.item())
        self.observer_time = self.world_time

    def transition(self, state, dt, *, state_table):
        match int(state.phase.item()):
            case 0:
                return self.advance_world(state, dt, state_table=state_table)

    def advance_world(
        self,
        state: ComputationalWorldState,
        dt: float,
        *,
        state_table: StateTable,
    ) -> tuple[bool, Metrics, ComputationalWorldState]:
        """Apply one admitted slice; subdivision remains dt-system-owned."""

        state.validate_sparse_shapes()
        spring_ok, spring_metrics = advance_bound_spring(
            state, dt, self.spring_parameters
        )
        if not spring_ok:
            return False, spring_metrics, state
        if state.pending_status:
            if len(state.pending_status) != 1 or not isinstance(
                state.pending_status[0], WorldStatusBatch
            ):
                raise ValueError("world has an invalid pending shell batch")
            batch = state.pending_status[0]
            state.player_intent = AbstractTensor.tensor(
                [batch.player_intent], dtype="float32"
            )
        else:
            batch = None

        player_id = int(state.player_entity.item())
        ids = state.entity_id.tolist()
        if player_id >= 0 and player_id in ids:
            player_index = ids.index(player_id)
            state.velocity[player_index] = state.player_intent[0]
            state.position[player_index] = (
                state.position[player_index] + state.velocity[player_index] * dt
            )

        admitted_end = float(state.managed_time.item()) + float(dt)
        state.managed_time = AbstractTensor.tensor(
            [admitted_end], dtype="float64"
        )
        if batch is not None:
            consumed = tuple(
                event for event in batch.boundary_events
                if float(event.event_time) <= admitted_end + 1.0e-15
            )
            remaining = tuple(
                event for event in batch.boundary_events
                if float(event.event_time) > admitted_end + 1.0e-15
            )
            records = tuple(
                record for event in consumed for record in event.provenance
            )
            if records:
                cursor = int(state.provenance_cursor.item())
                if records[0].sequence <= cursor:
                    raise ValueError("provenance sequence was replayed or reordered")
                state.provenance_cursor = AbstractTensor.tensor(
                    [records[-1].sequence], dtype="int64"
                )
                state.provenance_records += records
                known = list(state.artifact_references)
                for record in records:
                    if record.artifact_reference and record.artifact_reference not in known:
                        known.append(record.artifact_reference)
                state.artifact_references = tuple(known)
            state.pending_status = (
                (replace(batch, boundary_events=remaining),) if remaining else ()
            )

        speed = 0.0
        if int(state.velocity.shape[0]):
            speed = float((state.velocity * state.velocity).sum(dim=1).sqrt().max().item())
        metrics = Metrics(
            max_vel=max(speed, float(spring_metrics.max_vel)),
            max_flux=0.0,
            div_inf=0.0,
            mass_err=0.0,
            dt_limit=spring_metrics.dt_limit,
            error_channels={
                "world_sparse_shape": 0.0,
                **dict(spring_metrics.error_channels),
            },
            advanced_dt=float(dt),
        )
        return True, metrics, state

    def publish_committed(
        self,
        state: ComputationalWorldState,
        table: StateTable,
        metrics: Metrics,
    ) -> None:
        """Publish only state from an accepted complete managed window."""

        table.set("world", "entities", "id", state.entity_id)
        table.set("world", "entities", "position", state.position)
        table.set("world", "entities", "velocity", state.velocity)
        table.set("world", "topology", "edge_index", state.edge_index)
        table.set("world", "voxels", "coordinate", state.occupied_block_coord)
        table.set("world", "player", "intent", state.player_intent)
        table.set("world", "provenance", "cursor", state.provenance_cursor)
        table.set("world", "managed", "time", state.managed_time)
        table.set("engine", "spring", "position", state.spring_position)
        table.set("engine", "spring", "velocity", state.spring_velocity)
        table.set("engine", "spring", "rest_length", state.spring_rest_length)
        table.set("engine", "spring", "group_index", state.spring_group_index)
        table.set("engine", "spring", "glow_alpha", state.spring_glow_alpha)
        table.set("engine", "spring", "glow_radius", state.spring_glow_radius)
        table.set("dt_tape", "computational_world", "metrics", metrics)

    def get_state(self, state=None):
        if state is not None:
            self._state = state
        return self._state

    def snapshot(self):
        return {
            "world_time": float(self.world_time),
            "observer_time": float(self.observer_time),
        }

    def restore(self, snapshot) -> None:
        self.world_time = float(snapshot["world_time"])
        self.observer_time = float(snapshot["observer_time"])


class WorldTickLease:
    """Shell admission gate around the existing ManagedTimeRuntime."""

    def __init__(
        self,
        world: ComputationalWorld,
        state: ComputationalWorldState,
        state_table: StateTable,
        *,
        dx: float = 1.0,
        targets: Targets | None = None,
        controller: STController | None = None,
        generation: int = 0,
    ) -> None:
        self.world = world
        self.state = state
        self.state_table = state_table
        self.active = False

        def advance(managed_state, dt):
            ok, metrics, returned = world.step(
                dt, managed_state, state_table
            )
            if returned is not managed_state:
                raise RuntimeError("world transition replaced managed state identity")
            return ok, metrics

        self.runtime = ManagedTimeRuntime(
            state,
            advance,
            dx=dx,
            targets=targets or Targets(
                cfl=1.0,
                div_max=1.0,
                mass_max=1.0,
                error_limits={
                    "world_sparse_shape": 0.0,
                    "spring_causal_dt_excess": 0.0,
                },
            ),
            controller=controller or STController(dt_min=1.0e-9),
            generation=generation,
            initial_time=float(state.managed_time.item()),
        )

    def set_active(self, active: bool) -> None:
        self.active = bool(active)

    def advance_from_shell(
        self,
        request: TimeWindowRequest,
        status_supplier: Callable[[], WorldStatusBatch],
        *,
        commit_gate: ManagedCommitGate | None = None,
    ) -> TimeAdvanceReport | None:
        """Harvest and advance only when the shell has enabled game mode."""

        if not self.active:
            return None
        request.validate()
        if self.state.pending_status:
            raise RuntimeError("world already has an unconsumed shell batch")
        batch = status_supplier()
        if not isinstance(batch, WorldStatusBatch):
            raise TypeError("world status supplier must return WorldStatusBatch")
        event_times = tuple(
            float(event.event_time) for event in batch.boundary_events
        )
        if event_times != tuple(request.event_times):
            raise ValueError(
                "world boundary events must equal managed-time request events"
            )
        if not math.isclose(
            float(self.state.managed_time.item()),
            float(request.t_start),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise RuntimeError("world state time is not the managed request start")
        self.state.pending_status = (batch,)
        report = self.runtime.advance(request, commit_gate=commit_gate)
        if self.state.pending_status:
            raise RuntimeError("accepted world window left shell material pending")
        self.world.publish_committed(
            self.state,
            self.state_table,
            report.result.metrics or Metrics(0.0, 0.0, 0.0, 0.0),
        )
        return report

    def advance_from_evolution(
        self,
        request: TimeWindowRequest,
        metagraph: Any,
        *,
        player_intent: tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_records_per_boundary: int = 1,
        commit_gate: ManagedCommitGate | None = None,
    ) -> TimeAdvanceReport | None:
        """Harvest compiler provenance only after active-mode admission.

        The metagraph remains authoritative. This method snapshots it only
        after ``active`` is true, selects records after the world's committed
        cursor, and admits at most the authored boundary capacity. A rejected
        window does not advance either cursor.
        """

        if not self.active:
            return None
        if max_records_per_boundary <= 0:
            raise ValueError("provenance release capacity must be positive")

        def supply() -> WorldStatusBatch:
            snapshot = metagraph.snapshot()
            cursor = int(self.state.provenance_cursor.item())
            available = [
                event for event in snapshot.events
                if int(event.sequence) > cursor
            ]
            capacity = len(request.event_times) * int(max_records_per_boundary)
            selected = available[:capacity]
            boundary_events = []
            for boundary_index, event_time in enumerate(request.event_times):
                start = boundary_index * int(max_records_per_boundary)
                chunk = selected[start:start + int(max_records_per_boundary)]
                records = tuple(
                    ProvenanceRecord(
                        sequence=int(event.sequence),
                        graph_id=(
                            event.graph.id if event.graph is not None
                            else event.component.graph_id
                            if event.component is not None
                            else "compiler"
                        ),
                        component_id=(
                            event.component.local_id
                            if event.component is not None
                            else f"event:{event.sequence}"
                        ),
                        kind=str(event.kind),
                        artifact_reference=str(
                            event.detail.get("artifact_reference", "")
                        ),
                        captured_ns=int(event.captured_ns),
                    )
                    for event in chunk
                )
                boundary_events.append(WorldBoundaryEvent(
                    float(event_time), records
                ))
            return WorldStatusBatch(
                player_intent=player_intent,
                boundary_events=tuple(boundary_events),
            )

        return self.advance_from_shell(
            request,
            supply,
            commit_gate=commit_gate,
        )

    def retry_pending(
        self,
        request: TimeWindowRequest,
        *,
        commit_gate: ManagedCommitGate | None = None,
    ) -> TimeAdvanceReport:
        """Retry a rejected window with the rollback-restored status batch."""

        if not self.active:
            raise RuntimeError("world tick lease is inactive")
        if len(self.state.pending_status) != 1:
            raise RuntimeError("world has no single pending shell batch to retry")
        report = self.runtime.advance(request, commit_gate=commit_gate)
        if self.state.pending_status:
            raise RuntimeError("accepted world retry left shell material pending")
        self.world.publish_committed(
            self.state,
            self.state_table,
            report.result.metrics or Metrics(0.0, 0.0, 0.0, 0.0),
        )
        return report


__all__ = [
    "ComputationalWorld",
    "ProvenanceRecord",
    "WorldBoundaryEvent",
    "WorldStatusBatch",
    "WorldTickLease",
]
