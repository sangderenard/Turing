"""Rollback-complete sparse tensor state for the headless world engine."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from ..common.tensors.abstraction import AbstractTensor


def _tensor(value, *, dtype: str) -> AbstractTensor:
    return AbstractTensor.tensor(value, dtype=dtype)


@dataclass(frozen=True, slots=True)
class WorldStateSnapshot:
    """Isolated checkpoint of every mutable authoritative world field."""

    tensors: tuple[tuple[str, AbstractTensor], ...]
    control: tuple[tuple[str, Any], ...]


@dataclass
class ComputationalWorldState:
    """Structure-of-arrays state with no renderer or Python graph shadow.

    Graph relationships use COO ``edge_index`` with shape ``(2, E)``. Voxel
    storage contains only occupied or changed cells.  Stable artifact strings
    and pending immutable shell records remain small control data; all numeric
    world mechanics live in AbstractTensor fields.
    """

    entity_id: AbstractTensor
    entity_kind: AbstractTensor
    entity_flags: AbstractTensor
    position: AbstractTensor
    velocity: AbstractTensor
    edge_index: AbstractTensor
    edge_kind: AbstractTensor
    edge_state: AbstractTensor
    occupied_block_coord: AbstractTensor
    occupied_block_kind: AbstractTensor
    occupied_block_state: AbstractTensor
    component_entity: AbstractTensor
    component_kind: AbstractTensor
    component_state: AbstractTensor
    player_entity: AbstractTensor
    player_intent: AbstractTensor
    provenance_cursor: AbstractTensor
    managed_time: AbstractTensor
    phase: AbstractTensor
    spring_position: AbstractTensor
    spring_velocity: AbstractTensor
    spring_edge_index: AbstractTensor
    spring_mass: AbstractTensor
    spring_rest_length: AbstractTensor
    spring_base_length: AbstractTensor
    spring_natural_rest_length: AbstractTensor
    spring_done_growing: AbstractTensor
    spring_edge_level_mask: AbstractTensor
    spring_edge_type_mask: AbstractTensor
    spring_edge_role_mask: AbstractTensor
    spring_node_level_mask: AbstractTensor
    spring_node_type_mask: AbstractTensor
    spring_node_role_mask: AbstractTensor
    spring_glow_alpha: AbstractTensor
    spring_glow_radius: AbstractTensor
    spring_group_index: AbstractTensor
    spring_cycle_time: AbstractTensor
    spring_boundary_center: AbstractTensor
    spring_boundary_radius: AbstractTensor
    spring_node_network: AbstractTensor
    spring_edge_network: AbstractTensor
    artifact_references: tuple[str, ...] = ()
    provenance_records: tuple[Any, ...] = ()
    pending_status: tuple[Any, ...] = ()

    @classmethod
    def empty(cls) -> "ComputationalWorldState":
        """Create the canonical zero-entity sparse world."""

        return cls(
            entity_id=_tensor([], dtype="int64"),
            entity_kind=_tensor([], dtype="int32"),
            entity_flags=_tensor([], dtype="int32"),
            position=_tensor([], dtype="float32").reshape((0, 3)),
            velocity=_tensor([], dtype="float32").reshape((0, 3)),
            edge_index=_tensor([[], []], dtype="int64"),
            edge_kind=_tensor([], dtype="int32"),
            edge_state=_tensor([], dtype="float32").reshape((0, 0)),
            occupied_block_coord=_tensor([], dtype="int64").reshape((0, 3)),
            occupied_block_kind=_tensor([], dtype="int32"),
            occupied_block_state=_tensor([], dtype="float32").reshape((0, 0)),
            component_entity=_tensor([], dtype="int64"),
            component_kind=_tensor([], dtype="int32"),
            component_state=_tensor([], dtype="float32").reshape((0, 0)),
            player_entity=_tensor([-1], dtype="int64"),
            player_intent=_tensor([[0.0, 0.0, 0.0]], dtype="float32"),
            provenance_cursor=_tensor([-1], dtype="int64"),
            managed_time=_tensor([0.0], dtype="float64"),
            phase=_tensor([0], dtype="int32"),
            spring_position=_tensor([], dtype="float32").reshape((0, 3)),
            spring_velocity=_tensor([], dtype="float32").reshape((0, 3)),
            spring_edge_index=_tensor([[], []], dtype="int64"),
            spring_mass=_tensor([], dtype="float32"),
            spring_rest_length=_tensor([], dtype="float32"),
            spring_base_length=_tensor([], dtype="float32"),
            spring_natural_rest_length=_tensor([], dtype="float32"),
            spring_done_growing=_tensor([], dtype="bool"),
            spring_edge_level_mask=_tensor([], dtype="bool").reshape((0, 0)),
            spring_edge_type_mask=_tensor([], dtype="bool").reshape((0, 0)),
            spring_edge_role_mask=_tensor([], dtype="bool").reshape((0, 0)),
            spring_node_level_mask=_tensor([], dtype="bool").reshape((0, 0)),
            spring_node_type_mask=_tensor([], dtype="bool").reshape((0, 0)),
            spring_node_role_mask=_tensor([], dtype="bool").reshape((0, 0)),
            spring_glow_alpha=_tensor([], dtype="float32").reshape((0, 1)),
            spring_glow_radius=_tensor([], dtype="float32").reshape((0, 1)),
            spring_group_index=_tensor([0], dtype="int32"),
            spring_cycle_time=_tensor([0.0], dtype="float64"),
            spring_boundary_center=_tensor([], dtype="float32").reshape((0, 3)),
            spring_boundary_radius=_tensor([], dtype="float32"),
            spring_node_network=_tensor([], dtype="int32"),
            spring_edge_network=_tensor([], dtype="int32"),
        )

    @classmethod
    def with_player(cls, *, entity_id: int = 0) -> "ComputationalWorldState":
        """Create a sparse world containing one headless player object."""

        state = cls.empty()
        state.entity_id = _tensor([int(entity_id)], dtype="int64")
        state.entity_kind = _tensor([1], dtype="int32")
        state.entity_flags = _tensor([0], dtype="int32")
        state.position = _tensor([[0.0, 0.0, 0.0]], dtype="float32")
        state.velocity = _tensor([[0.0, 0.0, 0.0]], dtype="float32")
        state.player_entity = _tensor([int(entity_id)], dtype="int64")
        return state

    @classmethod
    def _tensor_field_names(cls) -> tuple[str, ...]:
        return tuple(
            descriptor.name
            for descriptor in fields(cls)
            if descriptor.name not in {
                "artifact_references", "provenance_records", "pending_status"
            }
        )

    def copy_shallow(self) -> WorldStateSnapshot:
        """Clone all tensors for a dt-system scientific checkpoint."""

        return WorldStateSnapshot(
            tensors=tuple(
                (name, getattr(self, name).clone())
                for name in self._tensor_field_names()
            ),
            control=(
                ("artifact_references", tuple(self.artifact_references)),
                ("provenance_records", tuple(self.provenance_records)),
                ("pending_status", tuple(self.pending_status)),
            ),
        )

    def restore(self, snapshot: WorldStateSnapshot) -> None:
        """Restore in place so ManagedTimeRuntime retains state identity."""

        if not isinstance(snapshot, WorldStateSnapshot):
            raise TypeError(
                "ComputationalWorldState.restore requires WorldStateSnapshot"
            )
        expected = self._tensor_field_names()
        restored = dict(snapshot.tensors)
        if tuple(restored) != expected:
            raise ValueError("world state snapshot schema does not match")
        for name in expected:
            setattr(self, name, restored[name].clone())
        control = dict(snapshot.control)
        expected_control = {
            "artifact_references", "provenance_records", "pending_status"
        }
        if set(control) != expected_control:
            raise ValueError("world state control snapshot schema does not match")
        self.artifact_references = tuple(control["artifact_references"])
        self.provenance_records = tuple(control["provenance_records"])
        self.pending_status = tuple(control["pending_status"])

    def validate_sparse_shapes(self) -> None:
        """Reject table misalignment before an admitted transition mutates it."""

        entity_count = int(self.entity_id.shape[0])
        for name in ("entity_kind", "entity_flags", "position", "velocity"):
            if int(getattr(self, name).shape[0]) != entity_count:
                raise ValueError(f"world entity table is misaligned at {name}")
        if tuple(self.position.shape[1:]) != (3,):
            raise ValueError("world positions must have shape (N, 3)")
        if tuple(self.velocity.shape[1:]) != (3,):
            raise ValueError("world velocities must have shape (N, 3)")
        if tuple(self.edge_index.shape[:1]) != (2,):
            raise ValueError("world edge_index must have shape (2, E)")
        edge_count = int(self.edge_index.shape[1])
        for name in ("edge_kind", "edge_state"):
            if int(getattr(self, name).shape[0]) != edge_count:
                raise ValueError(f"world edge table is misaligned at {name}")
        if tuple(self.occupied_block_coord.shape[1:]) != (3,):
            raise ValueError("occupied voxel coordinates must have shape (K, 3)")
        spring_nodes = int(self.spring_position.shape[0])
        for name in (
            "spring_velocity", "spring_mass", "spring_glow_alpha",
            "spring_glow_radius", "spring_node_network",
        ):
            if int(getattr(self, name).shape[0]) != spring_nodes:
                raise ValueError(f"spring node table is misaligned at {name}")
        if tuple(self.spring_position.shape[1:]) != (3,):
            raise ValueError("spring positions must have shape (S, 3)")
        if tuple(self.spring_edge_index.shape[:1]) != (2,):
            raise ValueError("spring edge_index must have shape (2, SE)")
        spring_edges = int(self.spring_edge_index.shape[1])
        for name in (
            "spring_rest_length", "spring_base_length",
            "spring_natural_rest_length", "spring_done_growing",
            "spring_edge_network",
        ):
            if int(getattr(self, name).shape[0]) != spring_edges:
                raise ValueError(f"spring edge table is misaligned at {name}")
        edge_masks = (
            self.spring_edge_level_mask,
            self.spring_edge_type_mask,
            self.spring_edge_role_mask,
        )
        node_masks = (
            self.spring_node_level_mask,
            self.spring_node_type_mask,
            self.spring_node_role_mask,
        )
        groups = int(edge_masks[0].shape[0]) if edge_masks else 0
        if any(tuple(mask.shape) != (groups, spring_edges) for mask in edge_masks):
            raise ValueError("spring edge activation masks are misaligned")
        if any(tuple(mask.shape) != (groups, spring_nodes) for mask in node_masks):
            raise ValueError("spring node activation masks are misaligned")
        network_count = max(
            (int(value) for value in self.spring_node_network.tolist()),
            default=-1,
        ) + 1
        if tuple(self.spring_boundary_center.shape) != (network_count, 3):
            raise ValueError("spring network boundary centers are misaligned")
        if tuple(self.spring_boundary_radius.shape) != (network_count,):
            raise ValueError("spring network boundary radii are misaligned")

    @staticmethod
    def _append_rows(existing: AbstractTensor, rows, *, dtype: str) -> AbstractTensor:
        incoming = _tensor(rows, dtype=dtype)
        if len(existing.shape) == 1:
            incoming = incoming.reshape((-1,))
        return AbstractTensor.cat([existing, incoming], dim=0)

    @staticmethod
    def _feature_row(existing: AbstractTensor, values, *, table: str) -> AbstractTensor:
        row = _tensor([tuple(float(value) for value in values)], dtype="float32")
        current_width = int(existing.shape[1])
        row_width = int(row.shape[1])
        if int(existing.shape[0]) == 0 and current_width == 0:
            return row
        if row_width != current_width:
            raise ValueError(
                f"{table} feature width {row_width} does not match {current_width}"
            )
        return AbstractTensor.cat([existing, row], dim=0)

    def spawn_entity(
        self,
        kind: int,
        position,
        *,
        velocity=(0.0, 0.0, 0.0),
        flags: int = 0,
        entity_id: int | None = None,
    ) -> int:
        """Append one sparse object and return its stable entity ID."""

        ids = tuple(int(value) for value in self.entity_id.tolist())
        stable_id = (
            max(ids, default=-1) + 1 if entity_id is None else int(entity_id)
        )
        if stable_id < 0 or stable_id in ids:
            raise ValueError("entity ID must be unique and non-negative")
        if len(tuple(position)) != 3 or len(tuple(velocity)) != 3:
            raise ValueError("entity position and velocity must be 3D")
        self.entity_id = self._append_rows(
            self.entity_id, [stable_id], dtype="int64"
        )
        self.entity_kind = self._append_rows(
            self.entity_kind, [int(kind)], dtype="int32"
        )
        self.entity_flags = self._append_rows(
            self.entity_flags, [int(flags)], dtype="int32"
        )
        self.position = self._append_rows(
            self.position, [tuple(position)], dtype="float32"
        )
        self.velocity = self._append_rows(
            self.velocity, [tuple(velocity)], dtype="float32"
        )
        self.validate_sparse_shapes()
        return stable_id

    def connect_entities(
        self,
        source_entity: int,
        target_entity: int,
        kind: int,
        *,
        features=(),
    ) -> int:
        """Append a COO relationship between stable entity identities."""

        ids = tuple(int(value) for value in self.entity_id.tolist())
        try:
            source_index = ids.index(int(source_entity))
            target_index = ids.index(int(target_entity))
        except ValueError as exc:
            raise KeyError("world relationship endpoint is not registered") from exc
        column = _tensor([[source_index], [target_index]], dtype="int64")
        self.edge_index = AbstractTensor.cat([self.edge_index, column], dim=1)
        self.edge_kind = self._append_rows(
            self.edge_kind, [int(kind)], dtype="int32"
        )
        self.edge_state = self._feature_row(
            self.edge_state, features, table="edge"
        )
        self.validate_sparse_shapes()
        return int(self.edge_index.shape[1]) - 1

    def set_voxel(self, coordinate, kind: int, *, features=()) -> int:
        """Insert or update one occupied/changed voxel cell."""

        coord = tuple(int(value) for value in coordinate)
        if len(coord) != 3:
            raise ValueError("voxel coordinate must be 3D")
        coordinates = tuple(tuple(int(v) for v in row) for row in self.occupied_block_coord.tolist())
        if coord in coordinates:
            index = coordinates.index(coord)
            self.occupied_block_kind[index] = int(kind)
            feature_values = tuple(float(value) for value in features)
            if len(feature_values) != int(self.occupied_block_state.shape[1]):
                raise ValueError("voxel feature width does not match existing table")
            if feature_values:
                self.occupied_block_state[index] = _tensor(
                    feature_values, dtype="float32"
                )
            return index
        self.occupied_block_coord = self._append_rows(
            self.occupied_block_coord, [coord], dtype="int64"
        )
        self.occupied_block_kind = self._append_rows(
            self.occupied_block_kind, [int(kind)], dtype="int32"
        )
        self.occupied_block_state = self._feature_row(
            self.occupied_block_state, features, table="voxel"
        )
        self.validate_sparse_shapes()
        return int(self.occupied_block_coord.shape[0]) - 1

    def attach_component(self, entity_id: int, kind: int, *, features=()) -> int:
        """Append one sparse component row owned by an existing entity."""

        if int(entity_id) not in tuple(int(value) for value in self.entity_id.tolist()):
            raise KeyError("component entity is not registered")
        self.component_entity = self._append_rows(
            self.component_entity, [int(entity_id)], dtype="int64"
        )
        self.component_kind = self._append_rows(
            self.component_kind, [int(kind)], dtype="int32"
        )
        self.component_state = self._feature_row(
            self.component_state, features, table="component"
        )
        return int(self.component_entity.shape[0]) - 1


__all__ = ["ComputationalWorldState", "WorldStateSnapshot"]
