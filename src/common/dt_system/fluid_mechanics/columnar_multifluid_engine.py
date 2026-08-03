"""Managed 2D-column / continuous-z multifluid voxel state machine.

One occupied slot is one unit voxel and one unit of physics.  World tiles pack
uniform ``x/y`` columns; the voxel centroid is grid-locked in ``x/y`` during
columnar settling and continuous in ``z``.  A player voxel opens a local
physics domain in which captured centroids may move freely in all three axes.

Python owns static topology and state-machine control.  Per-tick voxel math is
expressed as bulk :class:`AbstractTensor` operations.  This class owns no clock
or substepper: ``dt_system`` supplies every admitted timestep.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Sequence

from ...abstract_tensor_state_machine import (
    AbstractTensorStateMachine,
    TensorStateField,
)
from ...tensors.abstraction import AbstractTensor
from ...tensors.youngman import extract_isosurface, tetrahedra_from_axes
from ..dt_scaler import Metrics
from .columnar_multifluid_kernels import advance_columnar_surface_spring_local


def _tensor(value, dtype: str) -> AbstractTensor:
    return AbstractTensor.tensor(value, dtype=dtype)


@dataclass(frozen=True, slots=True)
class ColumnarMultifluidConfig:
    """Immutable topology and physical constants for one state machine."""

    tile_shape: tuple[int, int] = (8, 8)
    slots_per_column: int = 12
    material_names: tuple[str, ...] = ("solid", "granular", "liquid", "gas")
    material_mobility: tuple[float, ...] = (0.0, 0.35, 1.0, 0.8)
    floor_z: float = 0.0
    gravity: float = -9.81
    collision_damping: float = 0.15
    player_capture_radius: float = 1.75
    transfer_height_threshold: float = 1.0
    surface_smoothing: float = 0.08
    surface_spring_stiffness: float = 20.0
    surface_spring_damping: float = 8.0
    surface_spring_coupling: float = 4.0
    surface_load_depth: float = 0.42
    surface_load_radius: float = 1.35
    entity_interior_half_extent: float = 0.55
    entity_rejection_stiffness: float = 34.0
    ink_band_names: tuple[str, ...] = (
        "red", "yellow", "green", "cyan", "blue", "magenta"
    )
    ink_band_diffusivity: tuple[float, ...] = (
        0.11, 0.15, 0.19, 0.23, 0.27, 0.31
    )
    ink_injection_rate: float = 2.8
    ink_decay_rate: float = 0.055
    ink_injection_radius: float = 0.62
    ink_hue_angular_velocity: float = 0.42
    player_path_amplitude: tuple[float, float, float] = (0.75, 0.5, 0.08)
    player_path_angular_frequency: tuple[float, float, float] = (0.7, 1.1, 1.7)

    def __post_init__(self) -> None:
        if len(self.tile_shape) != 2 or any(int(size) <= 0 for size in self.tile_shape):
            raise ValueError("columnar tile shape must contain two positive sizes")
        if int(self.slots_per_column) <= 0:
            raise ValueError("columnar slots_per_column must be positive")
        if not self.material_names or len(set(self.material_names)) != len(self.material_names):
            raise ValueError("columnar material names must be unique and non-empty")
        if len(self.material_mobility) != len(self.material_names):
            raise ValueError("one material mobility is required per material")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in self.material_mobility):
            raise ValueError("material mobilities must be finite and non-negative")
        for name in (
            "floor_z", "gravity", "collision_damping",
            "player_capture_radius", "transfer_height_threshold",
            "surface_smoothing", "surface_spring_stiffness",
            "surface_spring_damping", "surface_spring_coupling",
            "surface_load_depth", "surface_load_radius",
            "entity_interior_half_extent", "entity_rejection_stiffness",
            "ink_injection_rate", "ink_decay_rate", "ink_injection_radius",
            "ink_hue_angular_velocity",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"columnar {name} must be finite")
        if not 0.0 <= float(self.collision_damping) <= 1.0:
            raise ValueError("collision damping must be between zero and one")
        if float(self.player_capture_radius) <= 0.0:
            raise ValueError("player capture radius must be positive")
        if float(self.transfer_height_threshold) < 0.0:
            raise ValueError("transfer height threshold cannot be negative")
        if float(self.surface_smoothing) < 0.0:
            raise ValueError("surface smoothing cannot be negative")
        if float(self.surface_spring_stiffness) <= 0.0:
            raise ValueError("surface spring stiffness must be positive")
        if float(self.surface_spring_damping) < 0.0:
            raise ValueError("surface spring damping cannot be negative")
        if float(self.surface_spring_coupling) < 0.0:
            raise ValueError("surface spring coupling cannot be negative")
        if float(self.surface_load_depth) < 0.0:
            raise ValueError("surface load depth cannot be negative")
        if float(self.surface_load_radius) <= 0.0:
            raise ValueError("surface load radius must be positive")
        if float(self.entity_interior_half_extent) <= 0.0:
            raise ValueError("entity interior half extent must be positive")
        if float(self.entity_rejection_stiffness) < 0.0:
            raise ValueError("entity rejection stiffness cannot be negative")
        if not self.ink_band_names or len(set(self.ink_band_names)) != len(
            self.ink_band_names
        ):
            raise ValueError("ink band names must be unique and non-empty")
        if len(self.ink_band_diffusivity) != len(self.ink_band_names):
            raise ValueError("one diffusivity is required per ink liquid")
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.ink_band_diffusivity
        ):
            raise ValueError("ink diffusivities must be finite and non-negative")
        if float(self.ink_injection_rate) < 0.0:
            raise ValueError("ink injection rate cannot be negative")
        if float(self.ink_decay_rate) < 0.0:
            raise ValueError("ink decay rate cannot be negative")
        if float(self.ink_injection_radius) <= 0.0:
            raise ValueError("ink injection radius must be positive")
        for name in ("player_path_amplitude", "player_path_angular_frequency"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or any(not math.isfinite(value) for value in values):
                raise ValueError(f"columnar {name} must contain three finite values")

    @property
    def material_count(self) -> int:
        return len(self.material_names)

    @property
    def ink_band_count(self) -> int:
        return len(self.ink_band_names)


@dataclass(frozen=True, slots=True)
class ColumnarMultifluidSnapshot:
    """Rollback-complete clone of the mutable tensor state."""

    tensors: tuple[tuple[str, AbstractTensor], ...]


@dataclass
class ColumnarMultifluidState:
    """Fixed-capacity tile × column × voxel-slot storage.

    ``voxel_centroid`` is the public spatial reference.  Slot indices are only
    stable storage addresses used for vectorization; they are not world IDs.
    """

    tile_coord: AbstractTensor
    column_centroid: AbstractTensor
    column_neighbor_index: AbstractTensor
    voxel_occupied: AbstractTensor
    voxel_centroid: AbstractTensor
    voxel_velocity: AbstractTensor
    voxel_material_fraction: AbstractTensor
    voxel_density: AbstractTensor
    voxel_pressure: AbstractTensor
    voxel_temperature: AbstractTensor
    voxel_is_player: AbstractTensor
    voxel_physics_domain: AbstractTensor
    player_voxel_index: AbstractTensor
    player_path_origin: AbstractTensor
    column_rest_surface_z: AbstractTensor
    column_displacement: AbstractTensor
    column_displacement_velocity: AbstractTensor
    column_surface_z: AbstractTensor
    column_material_mass: AbstractTensor
    column_mean_velocity: AbstractTensor
    column_ink_fraction: AbstractTensor
    transfer_flux: AbstractTensor
    managed_time: AbstractTensor
    phase: AbstractTensor

    @classmethod
    def regular(
        cls,
        tile_coordinates: Sequence[tuple[int, int]],
        config: ColumnarMultifluidConfig,
    ) -> "ColumnarMultifluidState":
        """Allocate an empty regular tile set and its fixed column stencil."""

        coordinates = tuple((int(x), int(y)) for x, y in tile_coordinates)
        if not coordinates or len(set(coordinates)) != len(coordinates):
            raise ValueError("tile coordinates must be unique and non-empty")
        tile_count = len(coordinates)
        width, height = (int(value) for value in config.tile_shape)
        slots = int(config.slots_per_column)
        materials = int(config.material_count)
        tile_coord = _tensor(coordinates, "int64")

        tile_x = tile_coord[:, 0].reshape((tile_count, 1, 1)) * width
        tile_y = tile_coord[:, 1].reshape((tile_count, 1, 1)) * height
        local_x = AbstractTensor.arange(width, dtype="float32").reshape((1, width, 1))
        local_y = AbstractTensor.arange(height, dtype="float32").reshape((1, 1, height))
        column_plane = AbstractTensor.ones((tile_count, width, height), dtype="float32")
        column_x = column_plane * (tile_x + local_x + 0.5)
        column_y = column_plane * (tile_y + local_y + 0.5)
        column_centroid = AbstractTensor.stack((column_x, column_y), dim=-1)

        slot_axis = AbstractTensor.arange(slots, dtype="float32").reshape((1, 1, 1, slots))
        voxel_plane = AbstractTensor.ones(
            (tile_count, width, height, slots), dtype="float32"
        )
        voxel_x = voxel_plane * column_x.reshape((tile_count, width, height, 1))
        voxel_y = voxel_plane * column_y.reshape((tile_count, width, height, 1))
        voxel_z = voxel_plane * (float(config.floor_z) + 0.5 + slot_axis)
        voxel_centroid = AbstractTensor.stack((voxel_x, voxel_y, voxel_z), dim=-1)

        global_to_flat: dict[tuple[int, int], int] = {}
        flat = 0
        for tile_x_index, tile_y_index in coordinates:
            for local_x_index in range(width):
                for local_y_index in range(height):
                    global_to_flat[
                        (
                            tile_x_index * width + local_x_index,
                            tile_y_index * height + local_y_index,
                        )
                    ] = flat
                    flat += 1
        neighbors = []
        for coordinate, column_index in sorted(
            global_to_flat.items(), key=lambda item: item[1]
        ):
            x, y = coordinate
            neighbors.append(tuple(
                global_to_flat.get(neighbor, -1)
                for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            ))
        column_count = tile_count * width * height
        state = cls(
            tile_coord=tile_coord,
            column_centroid=column_centroid,
            column_neighbor_index=_tensor(neighbors, "int64"),
            voxel_occupied=AbstractTensor.zeros(
                (tile_count, width, height, slots), dtype="bool"
            ),
            voxel_centroid=voxel_centroid,
            voxel_velocity=AbstractTensor.zeros(
                (tile_count, width, height, slots, 3), dtype="float32"
            ),
            voxel_material_fraction=AbstractTensor.zeros(
                (tile_count, width, height, slots, materials), dtype="float32"
            ),
            voxel_density=AbstractTensor.zeros(
                (tile_count, width, height, slots), dtype="float32"
            ),
            voxel_pressure=AbstractTensor.zeros(
                (tile_count, width, height, slots), dtype="float32"
            ),
            voxel_temperature=AbstractTensor.zeros(
                (tile_count, width, height, slots), dtype="float32"
            ),
            voxel_is_player=AbstractTensor.zeros(
                (tile_count, width, height, slots), dtype="bool"
            ),
            voxel_physics_domain=AbstractTensor.zeros(
                (tile_count, width, height, slots), dtype="int32"
            ),
            player_voxel_index=_tensor([], "int64"),
            player_path_origin=AbstractTensor.zeros((0, 3), dtype="float32"),
            column_rest_surface_z=AbstractTensor.full(
                (tile_count, width, height), float(config.floor_z), dtype="float32"
            ),
            column_displacement=AbstractTensor.zeros(
                (tile_count, width, height), dtype="float32"
            ),
            column_displacement_velocity=AbstractTensor.zeros(
                (tile_count, width, height), dtype="float32"
            ),
            column_surface_z=AbstractTensor.full(
                (tile_count, width, height), float(config.floor_z), dtype="float32"
            ),
            column_material_mass=AbstractTensor.zeros(
                (tile_count, width, height, materials), dtype="float32"
            ),
            column_mean_velocity=AbstractTensor.zeros(
                (tile_count, width, height, 3), dtype="float32"
            ),
            column_ink_fraction=AbstractTensor.zeros(
                (tile_count, width, height, config.ink_band_count),
                dtype="float32",
            ),
            transfer_flux=AbstractTensor.zeros(
                (column_count, 4, materials), dtype="float32"
            ),
            managed_time=_tensor([0.0], "float64"),
            phase=_tensor([0], "int32"),
        )
        state.validate(config)
        return state

    @classmethod
    def _tensor_field_names(cls) -> tuple[str, ...]:
        return tuple(descriptor.name for descriptor in fields(cls))

    def copy_shallow(self) -> ColumnarMultifluidSnapshot:
        return ColumnarMultifluidSnapshot(tuple(
            (name, getattr(self, name).clone()) for name in self._tensor_field_names()
        ))

    def restore(self, snapshot: ColumnarMultifluidSnapshot) -> None:
        if not isinstance(snapshot, ColumnarMultifluidSnapshot):
            raise TypeError("columnar multifluid restore requires its own snapshot")
        values = dict(snapshot.tensors)
        if tuple(values) != self._tensor_field_names():
            raise ValueError("columnar multifluid snapshot schema does not match")
        for name in self._tensor_field_names():
            setattr(self, name, values[name].clone())

    def validate(self, config: ColumnarMultifluidConfig) -> None:
        tiles = int(self.tile_coord.shape[0])
        width, height = (int(value) for value in config.tile_shape)
        slots = int(config.slots_per_column)
        materials = int(config.material_count)
        columns = tiles * width * height
        shapes = {
            "tile_coord": (tiles, 2),
            "column_centroid": (tiles, width, height, 2),
            "column_neighbor_index": (columns, 4),
            "voxel_occupied": (tiles, width, height, slots),
            "voxel_centroid": (tiles, width, height, slots, 3),
            "voxel_velocity": (tiles, width, height, slots, 3),
            "voxel_material_fraction": (tiles, width, height, slots, materials),
            "voxel_density": (tiles, width, height, slots),
            "voxel_pressure": (tiles, width, height, slots),
            "voxel_temperature": (tiles, width, height, slots),
            "voxel_is_player": (tiles, width, height, slots),
            "voxel_physics_domain": (tiles, width, height, slots),
            "column_rest_surface_z": (tiles, width, height),
            "column_displacement": (tiles, width, height),
            "column_displacement_velocity": (tiles, width, height),
            "column_surface_z": (tiles, width, height),
            "column_material_mass": (tiles, width, height, materials),
            "column_mean_velocity": (tiles, width, height, 3),
            "column_ink_fraction": (
                tiles, width, height, config.ink_band_count
            ),
            "transfer_flux": (columns, 4, materials),
            "managed_time": (1,),
            "phase": (1,),
        }
        for name, expected in shapes.items():
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"columnar multifluid state is misaligned at {name}")
        capacity = tiles * width * height * slots
        player_indices = tuple(int(value) for value in self.player_voxel_index.tolist())
        if tuple(self.player_path_origin.shape) != (len(player_indices), 3):
            raise ValueError("each player voxel requires one sinusoidal path origin")
        if len(set(player_indices)) != len(player_indices):
            raise ValueError("player voxel storage references must be unique")
        if any(index < 0 or index >= capacity for index in player_indices):
            raise ValueError("player voxel storage reference is outside capacity")
        material_sum = self.voxel_material_fraction.sum(dim=-1)
        expected_mass = self.voxel_occupied.to_dtype("float32")
        error = (material_sum - expected_mass).abs().max().item()
        if float(error) > 2.0e-6:
            raise ValueError("occupied voxels must contain one unit of material")

    def fill_columns(
        self,
        fill_count,
        material_index,
        config: ColumnarMultifluidConfig,
        *,
        density: float = 1.0,
        temperature: float = 0.0,
    ) -> None:
        """Fill every column in one broadcasted AbstractTensor operation."""

        tiles = int(self.tile_coord.shape[0])
        width, height = config.tile_shape
        slots = int(config.slots_per_column)
        materials = int(config.material_count)
        counts = _tensor(fill_count, "int32").reshape((tiles, width, height))
        selected_material = _tensor(material_index, "int32").reshape(
            (tiles, width, height)
        )
        if bool(((counts < 0) | (counts > slots)).max().item()):
            raise ValueError("column fill counts must fit the allocated slots")
        if bool(((selected_material < 0) | (selected_material >= materials)).max().item()):
            raise ValueError("column material index is outside the material table")
        slot = AbstractTensor.arange(slots, dtype="int32").reshape(
            (1, 1, 1, slots)
        ).expand((tiles, width, height, slots))
        occupied = slot < counts.reshape(
            (tiles, width, height, 1)
        ).expand((tiles, width, height, slots))
        material_axis = AbstractTensor.arange(materials, dtype="int32").reshape(
            (1, 1, 1, 1, materials)
        )
        one_hot = material_axis.expand(
            (tiles, width, height, slots, materials)
        ) == selected_material.reshape(
            (tiles, width, height, 1, 1)
        ).expand((tiles, width, height, slots, materials))
        self.voxel_occupied = occupied
        self.voxel_material_fraction = (
            occupied.reshape((tiles, width, height, slots, 1)).to_dtype("float32")
            * one_hot.to_dtype("float32")
        )
        self.voxel_density = occupied.to_dtype("float32") * float(density)
        self.voxel_temperature = occupied.to_dtype("float32") * float(temperature)
        self.voxel_pressure = AbstractTensor.zeros_like(self.voxel_density)
        self.voxel_is_player = AbstractTensor.zeros_like(occupied, dtype="bool")
        self.voxel_physics_domain = AbstractTensor.zeros_like(occupied, dtype="int32")
        self.player_voxel_index = _tensor([], "int64")
        self.player_path_origin = AbstractTensor.zeros((0, 3), dtype="float32")
        self.column_rest_surface_z = (
            counts.to_dtype("float32") + float(config.floor_z)
        )
        self.column_displacement = AbstractTensor.zeros_like(
            self.column_rest_surface_z
        )
        self.column_displacement_velocity = AbstractTensor.zeros_like(
            self.column_rest_surface_z
        )
        self.column_surface_z = self.column_rest_surface_z.clone()
        self.column_ink_fraction = AbstractTensor.zeros_like(
            self.column_ink_fraction
        )
        self.validate(config)

    def place_player(
        self,
        tile: int,
        local_x: int,
        local_y: int,
        slot: int,
        material: int,
        config: ColumnarMultifluidConfig,
    ) -> int:
        """Mark one existing unit voxel as a player physics-domain source."""

        address = tuple(int(value) for value in (tile, local_x, local_y, slot))
        limits = (
            int(self.tile_coord.shape[0]),
            int(config.tile_shape[0]),
            int(config.tile_shape[1]),
            int(config.slots_per_column),
        )
        if any(index < 0 or index >= limit for index, limit in zip(address, limits)):
            raise IndexError("player voxel address is outside packed grid")
        if not bool(self.voxel_occupied[address].item()):
            raise ValueError("player must occupy an existing unit voxel")
        if int(material) < 0 or int(material) >= config.material_count:
            raise ValueError("player material is outside the material table")
        self.voxel_material_fraction[address] = _tensor(
            [1.0 if index == int(material) else 0.0 for index in range(config.material_count)],
            "float32",
        )
        self.voxel_is_player[address] = True
        flat_index = (
            ((address[0] * limits[1] + address[1]) * limits[2] + address[2])
            * limits[3] + address[3]
        )
        self.player_voxel_index = AbstractTensor.cat(
            [self.player_voxel_index, _tensor([flat_index], "int64")], dim=0
        )
        origin = self.voxel_centroid[address].reshape((1, 3)).clone()
        self.player_path_origin = AbstractTensor.cat(
            [self.player_path_origin, origin], dim=0
        )
        self.validate(config)
        return flat_index


@dataclass(frozen=True, slots=True)
class ColumnarSurfacePrimitives:
    """Bulk unit-box/column-box inputs to the implicit surface function."""

    center: AbstractTensor
    half_extent: AbstractTensor
    active: AbstractTensor


class ColumnarSurfaceField:
    """AbstractTensor SDF union consumed directly by YoungMan."""

    def __init__(
        self,
        primitives: ColumnarSurfacePrimitives,
        *,
        smoothing: float = 0.0,
    ) -> None:
        self.primitives = primitives
        self.smoothing = float(smoothing)
        if self.smoothing < 0.0 or not math.isfinite(self.smoothing):
            raise ValueError("surface smoothing must be finite and non-negative")

    def __call__(self, points: AbstractTensor) -> AbstractTensor:
        original_shape = tuple(points.shape[:-1])
        flat = points.reshape((-1, 3))
        center = self.primitives.center.reshape((1, -1, 3))
        half_extent = self.primitives.half_extent.reshape((1, -1, 3))
        q = (flat.reshape((-1, 1, 3)) - center).abs() - half_extent
        positive = AbstractTensor.where(q > 0.0, q, 0.0)
        outside = (positive * positive).sum(dim=-1).sqrt()
        largest_axis = q.max(dim=-1)
        inside = AbstractTensor.where(largest_axis < 0.0, largest_axis, 0.0)
        distance = outside + inside
        active = self.primitives.active.reshape((1, -1)).expand(distance.shape)
        distance = AbstractTensor.where(
            active, distance, 1.0e6
        )
        if self.smoothing > 0.0:
            minimum = distance.min(dim=-1, keepdim=True)
            shifted = (-(distance - minimum) / self.smoothing).exp()
            result = minimum.reshape((-1,)) - self.smoothing * shifted.sum(dim=-1).log()
        else:
            result = distance.min(dim=-1)
        return result.reshape(original_shape)


def advance_columnar_surface_spring(
    column_centroid,
    column_neighbor_index,
    displacement,
    displacement_velocity,
    player_centroid,
    spring_stiffness,
    spring_damping,
    spring_coupling,
    load_depth,
    load_radius,
    dt,
):
    """Advance the complete column sheet as parallel AbstractTensor math.

    This is ordinary Python used directly by the state machine.  Keeping the
    operation factored makes its tensor topology visible to the Python
    recompiler; it is not a second implementation or a manually authored IR.
    """

    flat_centroid = column_centroid.reshape((-1, 2))
    flat_displacement = displacement.reshape((-1,))
    flat_velocity = displacement_velocity.reshape((-1,))
    next_displacement, next_velocity, _load = (
        advance_columnar_surface_spring_local(
            flat_centroid,
            flat_displacement,
            flat_velocity,
            player_centroid,
            spring_stiffness,
            spring_damping,
            load_depth,
            load_radius,
            dt,
        )
    )

    neighbors = column_neighbor_index
    valid = neighbors >= 0
    safe = AbstractTensor.where(valid, neighbors, 0).reshape((-1,))
    neighbor_displacement = flat_displacement.index_select(0, safe).reshape(
        neighbors.shape
    )
    valid_weight = valid.to_dtype("float32")
    neighbor_count = valid_weight.sum(dim=-1)
    neighbor_mean = (
        (neighbor_displacement * valid_weight).sum(dim=-1)
        / AbstractTensor.where(neighbor_count > 0.0, neighbor_count, 1.0)
    )
    laplacian = AbstractTensor.where(
        neighbor_count > 0.0,
        neighbor_mean - flat_displacement,
        0.0,
    )

    next_velocity = next_velocity + spring_coupling * laplacian * dt
    next_displacement = next_displacement + spring_coupling * laplacian * dt * dt
    return (
        next_displacement.reshape(displacement.shape),
        next_velocity.reshape(displacement_velocity.shape),
    )


class ColumnarMultifluidEngine(AbstractTensorStateMachine):
    """First-class managed state machine for the packed column physics."""

    state_fields = (
        TensorStateField("tile_coord", ("T", 2), "int64", scope="columnar_world"),
        TensorStateField("column_centroid", ("T", "X", "Y", 2), "float32", scope="columnar_world"),
        TensorStateField("column_neighbor_index", ("C", 4), "int64", scope="columnar_world"),
        TensorStateField("voxel_occupied", ("T", "X", "Y", "S"), "bool", scope="columnar_world"),
        TensorStateField("voxel_centroid", ("T", "X", "Y", "S", 3), "float32", scope="columnar_world"),
        TensorStateField("voxel_velocity", ("T", "X", "Y", "S", 3), "float32", scope="columnar_world"),
        TensorStateField("voxel_material_fraction", ("T", "X", "Y", "S", "M"), "float32", scope="columnar_world"),
        TensorStateField("voxel_physics_domain", ("T", "X", "Y", "S"), "int32", scope="columnar_world"),
        TensorStateField("player_path_origin", ("P", 3), "float32", scope="columnar_world"),
        TensorStateField("column_rest_surface_z", ("T", "X", "Y"), "float32", scope="columnar_world"),
        TensorStateField("column_displacement", ("T", "X", "Y"), "float32", scope="columnar_world"),
        TensorStateField("column_displacement_velocity", ("T", "X", "Y"), "float32", scope="columnar_world"),
        TensorStateField("column_surface_z", ("T", "X", "Y"), "float32", scope="columnar_world"),
        TensorStateField("column_material_mass", ("T", "X", "Y", "M"), "float32", scope="columnar_world"),
        TensorStateField("column_ink_fraction", ("T", "X", "Y", "H"), "float32", scope="columnar_world"),
        TensorStateField("transfer_flux", ("C", 4, "M"), "float32", scope="columnar_world"),
        TensorStateField("managed_time", (1,), "float64", scope="columnar_world"),
        TensorStateField("phase", (1,), "int32", scope="columnar_world"),
    )

    def __init__(
        self,
        state: ColumnarMultifluidState,
        config: ColumnarMultifluidConfig | None = None,
    ) -> None:
        self.config = config or ColumnarMultifluidConfig()
        state.validate(self.config)
        self._state = state
        self.world_time = float(state.managed_time.item())
        self.observer_time = self.world_time

    def transition(self, state, dt, *, state_table):
        match int(state.phase.item()):
            case 0:
                return self.advance_columns(state, dt, state_table=state_table)

    def _player_domains(self, state: ColumnarMultifluidState) -> AbstractTensor:
        shape = tuple(state.voxel_occupied.shape)
        if int(state.player_voxel_index.shape[0]) == 0:
            return AbstractTensor.zeros(shape, dtype="int32")
        flat_centroid = state.voxel_centroid.reshape((-1, 3))
        players = flat_centroid.index_select(0, state.player_voxel_index)
        delta = flat_centroid.reshape((-1, 1, 3)) - players.reshape((1, -1, 3))
        distance_squared = (delta * delta).sum(dim=-1)
        nearest_distance = distance_squared.min(dim=-1)
        nearest_player = distance_squared.argmin(dim=-1).to_dtype("int32") + 1
        captured = (
            (nearest_distance <= float(self.config.player_capture_radius) ** 2)
            & state.voxel_occupied.reshape((-1,))
        )
        return AbstractTensor.where(captured, nearest_player, 0).reshape(shape)

    def _advance_prescribed_players(
        self, state: ColumnarMultifluidState, dt: float
    ) -> None:
        """Move every player from managed time, without owning a clock."""

        if int(state.player_voxel_index.shape[0]) == 0:
            return
        next_time = float(state.managed_time.item()) + float(dt)
        amplitude = _tensor(self.config.player_path_amplitude, "float32").reshape(
            (1, 3)
        )
        frequency = _tensor(
            self.config.player_path_angular_frequency, "float32"
        ).reshape((1, 3))
        phase = _tensor((0.0, math.pi * 0.5, 0.0), "float32").reshape((1, 3))
        target = state.player_path_origin + amplitude * (
            frequency * next_time + phase
        ).sin()

        columns = state.column_centroid.reshape((-1, 2))
        xy_delta = (
            target[:, :2].reshape((-1, 1, 2))
            - columns.reshape((1, -1, 2))
        )
        nearest_column = (xy_delta * xy_delta).sum(dim=-1).argmin(dim=-1)
        support = state.column_surface_z.reshape((-1,)).index_select(
            0, nearest_column
        )
        target = AbstractTensor.cat(
            [target[:, :2], (support + 0.5 + target[:, 2] - state.player_path_origin[:, 2]).reshape((-1, 1))],
            dim=-1,
        )

        flat_centroid = state.voxel_centroid.reshape((-1, 3))
        old_centroid = flat_centroid.index_select(0, state.player_voxel_index)
        state.voxel_centroid = AbstractTensor.scatter(
            flat_centroid,
            state.player_voxel_index,
            target - old_centroid,
            dim=0,
        ).reshape(state.voxel_centroid.shape)
        flat_velocity = state.voxel_velocity.reshape((-1, 3))
        old_velocity = flat_velocity.index_select(0, state.player_voxel_index)
        target_velocity = (target - old_centroid) / float(dt)
        state.voxel_velocity = AbstractTensor.scatter(
            flat_velocity,
            state.player_voxel_index,
            target_velocity - old_velocity,
            dim=0,
        ).reshape(state.voxel_velocity.shape)

    def _advance_surface_spring(
        self, state: ColumnarMultifluidState, dt: float
    ) -> None:
        if int(state.player_voxel_index.shape[0]) == 0:
            return
        players = state.voxel_centroid.reshape((-1, 3)).index_select(
            0, state.player_voxel_index
        )
        (
            state.column_displacement,
            state.column_displacement_velocity,
        ) = advance_columnar_surface_spring(
            state.column_centroid,
            state.column_neighbor_index,
            state.column_displacement,
            state.column_displacement_velocity,
            players,
            float(self.config.surface_spring_stiffness),
            float(self.config.surface_spring_damping),
            float(self.config.surface_spring_coupling),
            float(self.config.surface_load_depth),
            float(self.config.surface_load_radius),
            float(dt),
        )
        state.column_surface_z = (
            state.column_rest_surface_z + state.column_displacement
        )

    def _entity_rejection_acceleration(
        self, state: ColumnarMultifluidState
    ) -> AbstractTensor:
        """Push matter out of every entity's unit-voxel interior in bulk."""

        shape = tuple(state.voxel_velocity.shape)
        if int(state.player_voxel_index.shape[0]) == 0:
            return AbstractTensor.zeros(shape, dtype="float32")
        flat_centroid = state.voxel_centroid.reshape((-1, 3))
        players = flat_centroid.index_select(0, state.player_voxel_index)
        delta = flat_centroid.reshape((-1, 1, 3)) - players.reshape((1, -1, 3))
        cube_distance = delta.abs().max(dim=-1)
        penetration = (
            float(self.config.entity_interior_half_extent) - cube_distance
        ).maximum(0.0)
        distance = ((delta * delta).sum(dim=-1) + 1.0e-8).sqrt()
        strength = (
            penetration * float(self.config.entity_rejection_stiffness)
            / distance
        )
        acceleration = (
            delta * strength.reshape((*strength.shape, 1))
        ).sum(dim=1)
        dynamic = (
            state.voxel_occupied.reshape((-1,))
            & ~state.voxel_is_player.reshape((-1,))
        ).reshape((-1, 1)).expand(tuple(acceleration.shape))
        return AbstractTensor.where(dynamic, acceleration, 0.0).reshape(shape)

    def _advance_ink(
        self, state: ColumnarMultifluidState, dt: float
    ) -> None:
        """Inject and diffuse independent hue-band liquid channels."""

        config = self.config
        bands = int(config.ink_band_count)
        flat_ink = state.column_ink_fraction.reshape((-1, bands))
        neighbors = state.column_neighbor_index
        valid = neighbors >= 0
        safe = AbstractTensor.where(valid, neighbors, 0).reshape((-1,))
        gathered = flat_ink.index_select(0, safe).reshape((-1, 4, bands))
        valid_lanes = valid.reshape((-1, 4, 1)).expand(tuple(gathered.shape))
        neighbor_sum = AbstractTensor.where(valid_lanes, gathered, 0.0).sum(dim=1)
        neighbor_count = valid.to_dtype("float32").sum(
            dim=1, keepdim=True
        ).expand(tuple(flat_ink.shape))
        neighbor_mean = neighbor_sum / AbstractTensor.where(
            neighbor_count > 0.0, neighbor_count, 1.0
        )
        diffusivity = _tensor(config.ink_band_diffusivity, "float32").reshape(
            (1, bands)
        )
        diffusion = diffusivity * (neighbor_mean - flat_ink)

        source = AbstractTensor.zeros((int(flat_ink.shape[0]), 1), dtype="float32")
        if int(state.player_voxel_index.shape[0]):
            players = state.voxel_centroid.reshape((-1, 3)).index_select(
                0, state.player_voxel_index
            )
            delta = (
                state.column_centroid.reshape((-1, 1, 2))
                - players[:, :2].reshape((1, -1, 2))
            )
            distance_squared = (delta * delta).sum(dim=-1)
            radius = float(config.ink_injection_radius)
            source = (-distance_squared / (2.0 * radius * radius)).exp().max(
                dim=-1
            ).reshape((-1, 1))
        phase = (
            AbstractTensor.arange(bands, dtype="float32")
            * (2.0 * math.pi / bands)
        ).reshape((1, bands))
        hue = (
            (float(state.managed_time.item()) + float(dt))
            * float(config.ink_hue_angular_velocity)
        )
        band_weight = (hue - phase).cos().maximum(0.0)
        band_weight = band_weight * band_weight
        band_weight = band_weight / AbstractTensor.where(
            band_weight.sum(dim=-1, keepdim=True) > 0.0,
            band_weight.sum(dim=-1, keepdim=True),
            1.0,
        )
        injection = (
            source * band_weight * float(config.ink_injection_rate)
        )
        next_ink = flat_ink + float(dt) * (
            diffusion + injection - float(config.ink_decay_rate) * flat_ink
        )
        state.column_ink_fraction = next_ink.maximum(0.0).minimum(1.0).reshape(
            state.column_ink_fraction.shape
        )

    def _refresh_column_stencils(self, state: ColumnarMultifluidState) -> None:
        config = self.config
        grid = state.voxel_occupied & (state.voxel_physics_domain == 0)
        grid_mass = grid.to_dtype("float32")
        z = state.voxel_centroid[..., 2]
        state.column_surface_z = (
            state.column_rest_surface_z + state.column_displacement
        )
        state.column_material_mass = (
            state.voxel_material_fraction * grid_mass.reshape((*grid_mass.shape, 1))
        ).sum(dim=3)
        momentum = (
            state.voxel_velocity * grid_mass.reshape((*grid_mass.shape, 1))
        ).sum(dim=3)
        column_mass = grid_mass.sum(dim=3, keepdim=True)
        velocity_shape = tuple(momentum.shape)
        mass_for_velocity = column_mass.expand(velocity_shape)
        state.column_mean_velocity = AbstractTensor.where(
            mass_for_velocity > 0.0,
            momentum / AbstractTensor.where(
                mass_for_velocity > 0.0, mass_for_velocity, 1.0
            ),
            0.0,
        )

        surface = state.column_surface_z.reshape((-1,))
        neighbors = state.column_neighbor_index
        valid = neighbors >= 0
        safe_neighbors = AbstractTensor.where(valid, neighbors, 0).reshape((-1,))
        neighbor_surface = surface.index_select(0, safe_neighbors).reshape(neighbors.shape)
        source_surface = surface.reshape((-1, 1))
        height_excess = source_surface - neighbor_surface - float(
            config.transfer_height_threshold
        )
        height_excess = AbstractTensor.where(
            valid & (height_excess > 0.0), height_excess, 0.0
        )
        column_material = state.column_material_mass.reshape(
            (-1, config.material_count)
        )
        total_material = column_material.sum(dim=-1, keepdim=True)
        composition = column_material / AbstractTensor.where(
            total_material > 0.0, total_material, 1.0
        )
        mobility = _tensor(config.material_mobility, "float32").reshape(
            (1, 1, config.material_count)
        )
        state.transfer_flux = (
            height_excess.reshape((-1, 4, 1))
            * composition.reshape((-1, 1, config.material_count))
            * mobility
        )

    def advance_columns(
        self,
        state: ColumnarMultifluidState,
        dt: float,
        *,
        state_table,
    ) -> tuple[bool, Metrics, ColumnarMultifluidState]:
        """Advance one externally admitted slice with bulk tensor math."""

        state.validate(self.config)
        self._advance_prescribed_players(state, dt)
        self._advance_surface_spring(state, dt)
        self._advance_ink(state, dt)
        state.voxel_physics_domain = self._player_domains(state)
        occupied = state.voxel_occupied
        grid = occupied & (state.voxel_physics_domain == 0)
        occupancy_lane = occupied.to_dtype("float32").reshape((*occupied.shape, 1))
        player_lane = state.voxel_is_player.reshape((*occupied.shape, 1)).expand(
            (*occupied.shape, 3)
        )
        dynamic_lane = (
            occupied & ~state.voxel_is_player
        ).to_dtype("float32").reshape((*occupied.shape, 1))
        gravity = _tensor((0.0, 0.0, float(self.config.gravity)), "float32").reshape(
            (1, 1, 1, 1, 3)
        )
        rejection = self._entity_rejection_acceleration(state)
        velocity = state.voxel_velocity + dynamic_lane * (
            gravity + rejection
        ) * float(dt)
        predicted = AbstractTensor.where(
            player_lane,
            state.voxel_centroid,
            state.voxel_centroid + velocity * float(dt),
        )

        slots = int(self.config.slots_per_column)
        support_z = (
            float(self.config.floor_z)
            + 0.5
            + AbstractTensor.arange(slots, dtype="float32").reshape((1, 1, 1, slots))
            + state.column_displacement.reshape((*state.column_displacement.shape, 1))
        ).expand(occupied.shape)
        predicted_z = predicted[..., 2]
        hits_support = grid & (predicted_z < support_z)
        settled_z = AbstractTensor.where(hits_support, support_z, predicted_z)
        fixed_xy = state.column_centroid.reshape(
            (*state.column_centroid.shape[:3], 1, 2)
        ).expand((*grid.shape, 2))
        grid_xy = grid.reshape((*grid.shape, 1)).expand((*grid.shape, 2))
        settled_xy = AbstractTensor.where(
            grid_xy, fixed_xy, predicted[..., :2]
        )
        state.voxel_centroid = AbstractTensor.cat(
            [settled_xy, settled_z.reshape((*settled_z.shape, 1))], dim=-1
        )

        vertical_velocity = velocity[..., 2]
        bounced = -vertical_velocity * float(self.config.collision_damping)
        vertical_velocity = AbstractTensor.where(
            hits_support & (vertical_velocity < 0.0), bounced, vertical_velocity
        )
        grid_velocity_xy = grid.reshape((*grid.shape, 1)).expand((*grid.shape, 2))
        free_xy_velocity = AbstractTensor.where(
            grid_velocity_xy, 0.0, velocity[..., :2]
        )
        state.voxel_velocity = AbstractTensor.cat(
            [free_xy_velocity, vertical_velocity.reshape((*vertical_velocity.shape, 1))],
            dim=-1,
        ) * occupancy_lane

        self._refresh_column_stencils(state)
        state.managed_time = _tensor(
            [float(state.managed_time.item()) + float(dt)], "float64"
        )

        material_error = (
            state.voxel_material_fraction.sum(dim=-1)
            - occupied.to_dtype("float32")
        ).abs().max().item()
        speed = (state.voxel_velocity * state.voxel_velocity).sum(dim=-1).sqrt()
        max_speed = float(speed.max().item()) if int(speed.shape[0]) else 0.0
        max_flux = float(state.transfer_flux.max().item())
        dt_limit = None if max_speed <= 1.0e-12 else 0.5 / max_speed
        metrics = Metrics(
            max_vel=max_speed,
            max_flux=max_flux,
            div_inf=0.0,
            mass_err=float(material_error),
            dt_limit=dt_limit,
            error_channels={
                "columnar_material_unit_error": float(material_error),
                "columnar_nonfinite": 0.0,
            },
            advanced_dt=float(dt),
        )
        return True, metrics, state

    def surface_primitives(
        self, state: ColumnarMultifluidState | None = None
    ) -> ColumnarSurfacePrimitives:
        """Build all ground columns and player unit boxes in parallel arrays."""

        state = self._state if state is None else state
        surface = state.column_surface_z.reshape((-1,))
        column_xy = state.column_centroid.reshape((-1, 2))
        height = surface - float(self.config.floor_z)
        half_height = AbstractTensor.where(height > 0.0, height * 0.5, 0.5)
        column_center = AbstractTensor.cat(
            [
                column_xy,
                (float(self.config.floor_z) + half_height).reshape((-1, 1)),
            ],
            dim=-1,
        )
        column_half_extent = AbstractTensor.cat(
            [
                AbstractTensor.full((int(height.shape[0]), 2), 0.5, dtype="float32"),
                half_height.reshape((-1, 1)),
            ],
            dim=-1,
        )
        column_active = height > 0.0

        if int(state.player_voxel_index.shape[0]):
            player_center = state.voxel_centroid.reshape((-1, 3)).index_select(
                0, state.player_voxel_index
            )
            player_half_extent = AbstractTensor.full(
                tuple(player_center.shape), 0.5, dtype="float32"
            )
            player_active = AbstractTensor.ones(
                (int(player_center.shape[0]),), dtype="bool"
            )
            center = AbstractTensor.cat([column_center, player_center], dim=0)
            half_extent = AbstractTensor.cat(
                [column_half_extent, player_half_extent], dim=0
            )
            active = AbstractTensor.cat([column_active, player_active], dim=0)
        else:
            center, half_extent, active = (
                column_center, column_half_extent, column_active
            )
        return ColumnarSurfacePrimitives(center, half_extent, active)

    def surface_field(
        self,
        state: ColumnarMultifluidState | None = None,
        *,
        smoothing: float | None = None,
    ) -> ColumnarSurfaceField:
        return ColumnarSurfaceField(
            self.surface_primitives(state),
            smoothing=(
                float(self.config.surface_smoothing)
                if smoothing is None else float(smoothing)
            ),
        )

    def extract_surface(
        self,
        x_axis,
        y_axis,
        z_axis,
        state: ColumnarMultifluidState | None = None,
        *,
        smoothing: float | None = None,
    ):
        """Run the existing YoungMan bulk crossing over the current state."""

        tetrahedra = tetrahedra_from_axes(x_axis, y_axis, z_axis)
        return extract_isosurface(
            tetrahedra,
            self.surface_field(state, smoothing=smoothing),
        )

    def publish_committed(self, state, state_table, metrics: Metrics) -> None:
        """Expose only an accepted state through the shared table."""

        state_table.set("columnar_world", "tiles", "coordinate", state.tile_coord)
        state_table.set(
            "columnar_world", "voxels", "occupied", state.voxel_occupied
        )
        state_table.set(
            "columnar_world", "voxels", "centroid", state.voxel_centroid
        )
        state_table.set(
            "columnar_world", "voxels", "velocity", state.voxel_velocity
        )
        state_table.set(
            "columnar_world", "voxels", "material_fraction",
            state.voxel_material_fraction,
        )
        state_table.set(
            "columnar_world", "voxels", "physics_domain",
            state.voxel_physics_domain,
        )
        state_table.set(
            "columnar_world", "columns", "surface_z", state.column_surface_z
        )
        state_table.set(
            "columnar_world", "columns", "displacement",
            state.column_displacement,
        )
        state_table.set(
            "columnar_world", "columns", "transfer_flux", state.transfer_flux
        )
        state_table.set(
            "columnar_world", "columns", "ink_fraction",
            state.column_ink_fraction,
        )
        state_table.set(
            "columnar_world", "managed", "time", state.managed_time
        )
        state_table.set("dt_tape", "columnar_multifluid", "metrics", metrics)

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


__all__ = [
    "ColumnarMultifluidConfig",
    "ColumnarMultifluidEngine",
    "ColumnarMultifluidSnapshot",
    "ColumnarMultifluidState",
    "ColumnarSurfaceField",
    "ColumnarSurfacePrimitives",
    "advance_columnar_surface_spring",
]
