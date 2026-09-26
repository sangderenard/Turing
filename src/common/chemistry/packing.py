"""Heterogeneous owner and boundary packing for compiled chemistry.

The chemistry law sees one stable state axis.  Geometry only determines which
owners exist, which phases and reaction families are active on each owner, and
which oriented boundaries may exchange state.  Arenas are contiguous so a
compiled kernel can march a dense span without losing the identity of the
machine part, voxel, film, pool, or reservoir that owns each row.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Mapping, Sequence

from src.common.tensors.abstraction import AbstractTensor

from .compendium import ADSORBED, AQUEOUS, GAS, LIQUID, PLASMA, SOLID
from .state import CHEMISTRY_PRECISION_LIMBS


class ArenaKind(IntEnum):
    GAS_VOLUME = 1
    FLUID_VOLUME = 2
    REACTIVE_SURFACE = 3
    SOLID_VOLUME = 4
    METALLURGY_VOLUME = 5
    PORT_RESERVOIR = 6
    POPULATION = 7
    SCALAR_GAP = 8


class BoundaryKind(IntEnum):
    VOLUME_FACE = 1
    OPEN_PORT = 2
    PHASE_INTERFACE = 3
    SURFACE_CONTACT = 4
    SOLID_CONTACT = 5
    FLUID_LINE = 6
    THERMAL_INTERFACE = 7
    GRAIN_BOUNDARY = 8


@dataclass(frozen=True)
class ArenaSpec:
    identity: str
    kind: ArenaKind
    count: int
    phases: tuple[str, ...]
    reaction_families: tuple[str, ...] = ()
    geometry: str = "unspecified"
    carries_inventory: bool = True
    scalar_fields: tuple[str, ...] = ("temperature_k", "internal_energy_j", "pressure_pa")

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("arena identity must not be empty")
        if int(self.count) <= 0:
            raise ValueError(f"{self.identity}: arena count must be positive")
        if self.kind is ArenaKind.SCALAR_GAP and self.carries_inventory:
            raise ValueError(f"{self.identity}: scalar gaps cannot carry chemical inventory")


@dataclass(frozen=True)
class BoundarySpec:
    identity: str
    left_arena: str
    right_arena: str
    pairs: tuple[tuple[int, int], ...]
    kind: BoundaryKind
    transport_families: tuple[str, ...] = ()
    reaction_families: tuple[str, ...] = ()
    geometry: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("boundary identity must not be empty")
        if not self.pairs:
            raise ValueError(f"{self.identity}: boundary must contain at least one owner pair")

    @property
    def count(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True)
class ArenaSlice:
    arena: ArenaSpec
    owner_start: int | None
    owner_stop: int | None

    @property
    def carries_inventory(self) -> bool:
        return self.owner_start is not None

    def owner(self, local_index: int) -> int:
        if self.owner_start is None:
            raise ValueError(f"{self.arena.identity}: no chemical owner allocation")
        if not 0 <= int(local_index) < self.arena.count:
            raise IndexError(local_index)
        return self.owner_start + int(local_index)


@dataclass(frozen=True)
class BoundarySlice:
    boundary: BoundarySpec
    boundary_start: int
    boundary_stop: int


@dataclass(frozen=True)
class PackedChemistryMetadata:
    """Flat AbstractTensor tables passed beside the packed chemistry state."""

    owner_arena: AbstractTensor
    owner_kind: AbstractTensor
    owner_local_index: AbstractTensor
    owner_phase_mask: AbstractTensor
    state_phase: AbstractTensor
    owner_state_mask: AbstractTensor
    owner_reaction_mask: AbstractTensor
    boundary_left_owner: AbstractTensor
    boundary_right_owner: AbstractTensor
    boundary_kind: AbstractTensor
    boundary_transport_mask: AbstractTensor
    boundary_reaction_mask: AbstractTensor


class ChemistryPackingPlan:
    """A stable ABI for all chemical owners and interfaces in one deployment."""

    def __init__(self, *, states: Sequence[str], arenas: Iterable[ArenaSpec],
                 boundaries: Iterable[BoundarySpec],
                 limbs: int = CHEMISTRY_PRECISION_LIMBS,
                 abi_identity: str | None = None):
        self.states = tuple(map(str, states))
        self.arenas = tuple(arenas)
        self.boundaries = tuple(boundaries)
        self.limbs = int(limbs)
        self.abi_identity = None if abi_identity is None else str(abi_identity)
        if not self.states:
            raise ValueError("chemistry packing requires a nonempty state axis")
        if len(set(self.states)) != len(self.states):
            raise ValueError("chemistry state identities must be unique")
        if self.limbs <= 0:
            raise ValueError("precision limb count must be positive")

        arena_names = [arena.identity for arena in self.arenas]
        if len(set(arena_names)) != len(arena_names):
            raise ValueError("arena identities must be unique")
        boundary_names = [boundary.identity for boundary in self.boundaries]
        if len(set(boundary_names)) != len(boundary_names):
            raise ValueError("boundary identities must be unique")

        owner_cursor = 0
        arena_slices: dict[str, ArenaSlice] = {}
        for arena in self.arenas:
            if arena.carries_inventory:
                arena_slices[arena.identity] = ArenaSlice(
                    arena, owner_cursor, owner_cursor + arena.count)
                owner_cursor += arena.count
            else:
                arena_slices[arena.identity] = ArenaSlice(arena, None, None)
        self.arena_slices = arena_slices
        self.owner_count = owner_cursor

        boundary_cursor = 0
        boundary_slices: dict[str, BoundarySlice] = {}
        for boundary in self.boundaries:
            if boundary.left_arena not in arena_slices:
                raise KeyError(f"{boundary.identity}: unknown left arena {boundary.left_arena}")
            if boundary.right_arena not in arena_slices:
                raise KeyError(f"{boundary.identity}: unknown right arena {boundary.right_arena}")
            left = arena_slices[boundary.left_arena]
            right = arena_slices[boundary.right_arena]
            if not left.carries_inventory or not right.carries_inventory:
                raise ValueError(
                    f"{boundary.identity}: chemical boundaries require two inventory owners")
            for left_local, right_local in boundary.pairs:
                left.owner(left_local)
                right.owner(right_local)
            boundary_slices[boundary.identity] = BoundarySlice(
                boundary, boundary_cursor, boundary_cursor + boundary.count)
            boundary_cursor += boundary.count
        self.boundary_slices = boundary_slices
        self.boundary_count = boundary_cursor

        state_phases: list[str] = []
        for state in self.states:
            if "@" not in state:
                raise ValueError(f"chemical state lacks a phase suffix: {state}")
            state_phases.append(state.rsplit("@", 1)[1])
        self.state_phases = tuple(state_phases)
        self.phases = tuple(dict.fromkeys(
            (*self.state_phases,
             *(phase for arena in self.arenas for phase in arena.phases))))
        self.reaction_families = tuple(dict.fromkeys((
            *(family for arena in self.arenas for family in arena.reaction_families),
            *(family for boundary in self.boundaries for family in boundary.reaction_families),
        )))
        self.transport_families = tuple(dict.fromkeys(
            family
            for boundary in self.boundaries for family in boundary.transport_families))

    @property
    def logical_inventory_size(self) -> int:
        return self.owner_count * len(self.states)

    @property
    def physical_inventory_size(self) -> int:
        return self.logical_inventory_size * self.limbs

    def owner(self, arena: str, local_index: int) -> int:
        return self.arena_slices[str(arena)].owner(local_index)

    def logical_index(self, arena: str, local_index: int, state: str) -> int:
        try:
            state_index = self.states.index(str(state))
        except ValueError as error:
            raise KeyError(state) from error
        return self.owner(arena, local_index) * len(self.states) + state_index

    def physical_index(self, arena: str, local_index: int, state: str,
                       limb: int = 0) -> int:
        if not 0 <= int(limb) < self.limbs:
            raise IndexError(limb)
        return self.logical_index(arena, local_index, state) * self.limbs + int(limb)

    def empty_inventory(self) -> AbstractTensor:
        return AbstractTensor.get_tensor([0.0] * self.physical_inventory_size)

    def pack_inventory(self, values: Mapping[tuple[str, int, str], object]
                       ) -> AbstractTensor:
        """Pack scalar or explicit limb values in owner/state/limb order."""
        packed = [0.0] * self.physical_inventory_size
        for (arena, local_index, state), value in values.items():
            if isinstance(value, (tuple, list)):
                limbs = tuple(value)
                if len(limbs) != self.limbs:
                    raise ValueError(
                        f"{arena}[{local_index}] {state}: expected {self.limbs} limbs")
            else:
                limbs = (value,) + (0.0,) * (self.limbs - 1)
            for limb, term in enumerate(limbs):
                packed[self.physical_index(arena, local_index, state, limb)] = term
        return AbstractTensor.get_tensor(packed)

    def boundary_owner_rows(self) -> tuple[tuple[int, int], ...]:
        rows: list[tuple[int, int]] = []
        for boundary in self.boundaries:
            for left_local, right_local in boundary.pairs:
                rows.append((self.owner(boundary.left_arena, left_local),
                             self.owner(boundary.right_arena, right_local)))
        return tuple(rows)

    def scatter_boundary_flux(self, flux: AbstractTensor) -> AbstractTensor:
        """Scatter oriented boundary flux; positive values move left to right."""
        expected = self.boundary_count * len(self.states)
        if int(flux.numel()) != expected:
            raise ValueError(f"expected {expected} boundary/state fluxes, got {flux.numel()}")
        delta = AbstractTensor.get_tensor([0.0] * self.logical_inventory_size)
        for boundary_index, (left, right) in enumerate(self.boundary_owner_rows()):
            for state_index in range(len(self.states)):
                flux_index = boundary_index * len(self.states) + state_index
                left_index = left * len(self.states) + state_index
                right_index = right * len(self.states) + state_index
                value = flux[flux_index]
                delta[left_index] = delta[left_index] - value
                delta[right_index] = delta[right_index] + value
        return delta

    def materialize_metadata(self) -> PackedChemistryMetadata:
        arena_number = {arena.identity: index for index, arena in enumerate(self.arenas)}
        phase_number = {phase: index for index, phase in enumerate(self.phases)}
        reaction_number = {
            family: index for index, family in enumerate(self.reaction_families)}
        transport_number = {
            family: index for index, family in enumerate(self.transport_families)}

        owner_arena: list[float] = []
        owner_kind: list[float] = []
        owner_local: list[float] = []
        owner_phase = [0.0] * (self.owner_count * len(self.phases))
        owner_state = [0.0] * (self.owner_count * len(self.states))
        owner_reaction = [0.0] * (self.owner_count * len(self.reaction_families))
        for arena in self.arenas:
            arena_slice = self.arena_slices[arena.identity]
            if not arena_slice.carries_inventory:
                continue
            for local_index in range(arena.count):
                owner = arena_slice.owner(local_index)
                owner_arena.append(float(arena_number[arena.identity]))
                owner_kind.append(float(int(arena.kind)))
                owner_local.append(float(local_index))
                for phase in arena.phases:
                    owner_phase[owner * len(self.phases) + phase_number[phase]] = 1.0
                for state_index, phase in enumerate(self.state_phases):
                    if phase in arena.phases:
                        owner_state[owner * len(self.states) + state_index] = 1.0
                for family in arena.reaction_families:
                    owner_reaction[
                        owner * len(self.reaction_families) + reaction_number[family]
                    ] = 1.0

        rows = self.boundary_owner_rows()
        boundary_kind: list[float] = []
        boundary_transport = [0.0] * (
            self.boundary_count * len(self.transport_families))
        boundary_reaction = [0.0] * (
            self.boundary_count * len(self.reaction_families))
        cursor = 0
        for boundary in self.boundaries:
            for _pair in boundary.pairs:
                boundary_kind.append(float(int(boundary.kind)))
                for family in boundary.transport_families:
                    boundary_transport[
                        cursor * len(self.transport_families) + transport_number[family]
                    ] = 1.0
                for family in boundary.reaction_families:
                    boundary_reaction[
                        cursor * len(self.reaction_families) + reaction_number[family]
                    ] = 1.0
                cursor += 1

        tensor = AbstractTensor.get_tensor
        return PackedChemistryMetadata(
            owner_arena=tensor(owner_arena),
            owner_kind=tensor(owner_kind),
            owner_local_index=tensor(owner_local),
            owner_phase_mask=tensor(owner_phase),
            state_phase=tensor([float(phase_number[phase]) for phase in self.state_phases]),
            owner_state_mask=tensor(owner_state),
            owner_reaction_mask=tensor(owner_reaction),
            boundary_left_owner=tensor([float(left) for left, _right in rows]),
            boundary_right_owner=tensor([float(right) for _left, right in rows]),
            boundary_kind=tensor(boundary_kind),
            boundary_transport_mask=tensor(boundary_transport),
            boundary_reaction_mask=tensor(boundary_reaction),
        )


def _pairs(left: Iterable[int], right: int = 0) -> tuple[tuple[int, int], ...]:
    return tuple((int(index), int(right)) for index in left)


def rectilinear_neighbor_pairs(shape: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Positive-axis faces for a flattened x/y/z voxel lattice."""
    if len(shape) != 3 or any(int(size) <= 0 for size in shape):
        raise ValueError("rectilinear chemistry shape must contain three positive sizes")
    nx, ny, nz = map(int, shape)
    flat = lambda x, y, z: (x * ny + y) * nz + z
    pairs: list[tuple[int, int]] = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                here = flat(x, y, z)
                if x + 1 < nx:
                    pairs.append((here, flat(x + 1, y, z)))
                if y + 1 < ny:
                    pairs.append((here, flat(x, y + 1, z)))
                if z + 1 < nz:
                    pairs.append((here, flat(x, y, z + 1)))
    return tuple(pairs)


def dewar_chemistry_plan(states: Sequence[str], chamber_shape=(5, 5, 5), *,
                         limbs: int = CHEMISTRY_PRECISION_LIMBS,
                         abi_identity: str | None = None,
                         ) -> ChemistryPackingPlan:
    """Pack the visible chamber and its engine-owned material connections."""
    nx, ny, nz = map(int, chamber_shape)
    voxel_count = nx * ny * nz
    flat = lambda x, y, z: (x * ny + y) * nz + z
    top_center = flat(nx // 2, ny // 2, nz - 1)
    floor = tuple(flat(x, y, 0) for x in range(nx) for y in range(ny))
    wall_groups = (
        tuple(flat(x, 0, z) for x in range(nx) for z in range(nz)),
        tuple(flat(x, ny - 1, z) for x in range(nx) for z in range(nz)),
        tuple(flat(0, y, z) for y in range(ny) for z in range(nz)),
        tuple(flat(nx - 1, y, z) for y in range(ny) for z in range(nz)),
        floor,
    )

    arenas = (
        ArenaSpec("chamber.gas", ArenaKind.GAS_VOLUME, voxel_count, (GAS,),
                  ("gas-phase", "phase"), "rectilinear chamber voxels"),
        ArenaSpec("chamber.aerosol", ArenaKind.POPULATION, voxel_count,
                  (LIQUID, SOLID), ("nucleation", "droplet", "phase"),
                  "one composition owner per chamber voxel"),
        ArenaSpec("chamber.pool", ArenaKind.FLUID_VOLUME, 1,
                  (AQUEOUS, LIQUID, SOLID),
                  ("aqueous-acid-base", "aqueous-mineral", "phase"),
                  "bottom condensate pool"),
        ArenaSpec("chamber.wall_surface", ArenaKind.REACTIVE_SURFACE, 5,
                  (ADSORBED, SOLID),
                  ("adsorption", "phase", "corrosion", "deposition"),
                  "floor, left, right, front, and back inner surfaces"),
        ArenaSpec("chamber.wall_solid", ArenaKind.SOLID_VOLUME, 5, (SOLID,),
                  ("solid-diffusion",), "five vessel wall bodies"),
        ArenaSpec("cold_head.surface", ArenaKind.REACTIVE_SURFACE, 1,
                  (ADSORBED, SOLID, LIQUID),
                  ("adsorption", "phase", "deposition"),
                  "cold tip protruding into top-center voxel"),
        ArenaSpec("cold_head.solid", ArenaKind.METALLURGY_VOLUME, 1, (SOLID,),
                  ("solid-diffusion", "oxidation", "phase-transformation"),
                  "copper cold-tip body"),
        ArenaSpec("cold_head.working_gas", ArenaKind.FLUID_VOLUME, 2,
                  (GAS, LIQUID), ("gas-phase", "phase"),
                  "supply and return heat-exchange passages"),
        ArenaSpec("cold_head.supply", ArenaKind.PORT_RESERVOIR, 1, (GAS, LIQUID),
                  (), "regulated working-gas bottle"),
        ArenaSpec("drain.receiver", ArenaKind.PORT_RESERVOIR, 1,
                  (AQUEOUS, LIQUID, SOLID, GAS),
                  ("aqueous-acid-base", "aqueous-mineral", "phase"),
                  "drain tank"),
        ArenaSpec("outside.atmosphere", ArenaKind.PORT_RESERVOIR, 1, (GAS,),
                  ("gas-phase", "phase"), "ambient supply at the open top port"),
        ArenaSpec("jacket.vacuum", ArenaKind.FLUID_VOLUME, 1, (GAS,),
                  ("gas-phase", "phase"),
                  "non-voxel vacuum annulus; pressure is also consumed as a "
                  "thermal-efficiency shorthand", True,
                  ("temperature_k", "internal_energy_j", "pressure_pa",
                   "vacuum_integrity", "thermal_conductance_w_k")),
    )

    boundaries: list[BoundarySpec] = [
        BoundarySpec("chamber.aerosol_exchange", "chamber.gas", "chamber.aerosol",
                     tuple((index, index) for index in range(voxel_count)),
                     BoundaryKind.PHASE_INTERFACE,
                     ("species", "energy"), ("nucleation", "droplet", "phase"),
                     "voxel-local gas/droplet exchange"),
        BoundarySpec("chamber.open_port", "outside.atmosphere", "chamber.gas",
                     ((0, top_center),), BoundaryKind.OPEN_PORT,
                     ("pressure-flow", "species", "energy"), geometry="top-center port"),
        BoundarySpec("jacket.vacuum.top_port", "outside.atmosphere", "jacket.vacuum",
                     ((0, 0),), BoundaryKind.FLUID_LINE,
                     ("pressure-flow", "species", "energy"),
                     geometry="normally closed top vacuum-service valve"),
        BoundarySpec("jacket.vacuum.bottom_port", "outside.atmosphere", "jacket.vacuum",
                     ((0, 0),), BoundaryKind.FLUID_LINE,
                     ("pressure-flow", "species", "energy"),
                     geometry="normally closed bottom vacuum-service valve"),
        BoundarySpec("chamber.cold_tip", "chamber.gas", "cold_head.surface",
                     ((top_center, 0),), BoundaryKind.SURFACE_CONTACT,
                     ("species", "energy"),
                     ("adsorption", "phase", "deposition"),
                     "cold tip exposed in top-center voxel"),
        BoundarySpec("cold_head.surface_to_solid", "cold_head.surface", "cold_head.solid",
                     ((0, 0),), BoundaryKind.SOLID_CONTACT,
                     ("solid-diffusion", "energy"), ("oxidation", "deposition"),
                     "surface/bulk metal interface"),
        BoundarySpec("cold_head.supply_line", "cold_head.supply", "cold_head.working_gas",
                     ((0, 0),), BoundaryKind.FLUID_LINE,
                     ("pressure-flow", "species", "energy"), geometry="regulated supply"),
        BoundarySpec("cold_head.return_line", "cold_head.working_gas", "cold_head.supply",
                     ((1, 0),), BoundaryKind.FLUID_LINE,
                     ("pressure-flow", "species", "energy"), geometry="closed-loop return"),
        BoundarySpec("cold_head.heat_exchange", "cold_head.working_gas", "cold_head.solid",
                     ((0, 0), (1, 0)), BoundaryKind.THERMAL_INTERFACE,
                     ("energy",), geometry="recuperator/expander/cold-tip exchange"),
        BoundarySpec("chamber.pool_contact", "chamber.gas", "chamber.pool",
                     _pairs(floor), BoundaryKind.PHASE_INTERFACE,
                     ("species", "energy"), ("phase",),
                     "bottom gas/pool faces"),
        BoundarySpec("chamber.drain", "chamber.pool", "drain.receiver",
                     ((0, 0),), BoundaryKind.FLUID_LINE,
                     ("gravity-flow", "species", "energy"), geometry="maintainable drain"),
    ]
    gas_faces = rectilinear_neighbor_pairs(chamber_shape)
    if gas_faces:
        boundaries.insert(0, BoundarySpec(
            "chamber.gas_faces", "chamber.gas", "chamber.gas", gas_faces,
            BoundaryKind.VOLUME_FACE, ("advection", "diffusion", "energy"),
            geometry="voxel faces"))
    for panel, voxels in enumerate(wall_groups):
        boundaries.append(BoundarySpec(
            f"chamber.wall_contact.{panel}", "chamber.gas", "chamber.wall_surface",
            _pairs(voxels, panel), BoundaryKind.SURFACE_CONTACT,
            ("species", "energy"),
            ("adsorption", "phase", "corrosion", "deposition"),
            "gas/inner-wall faces"))
        boundaries.append(BoundarySpec(
            f"chamber.wall_bulk.{panel}", "chamber.wall_surface", "chamber.wall_solid",
            ((panel, panel),), BoundaryKind.SOLID_CONTACT,
            ("solid-diffusion", "energy"), ("corrosion",), "surface/wall bulk"))
    return ChemistryPackingPlan(
        states=states, arenas=arenas, boundaries=boundaries, limbs=limbs,
        abi_identity=abi_identity)


def solid_surface_reaction_plan(states: Sequence[str], *, surface_patches: int = 1,
                                limbs: int = CHEMISTRY_PRECISION_LIMBS,
                                abi_identity: str | None = None,
                                ) -> ChemistryPackingPlan:
    arenas = (
        ArenaSpec("environment", ArenaKind.FLUID_VOLUME, surface_patches,
                  (GAS, AQUEOUS, LIQUID), ("gas-phase", "aqueous-acid-base")),
        ArenaSpec("surface", ArenaKind.REACTIVE_SURFACE, surface_patches,
                  (ADSORBED, SOLID, LIQUID),
                  ("adsorption", "oxidation", "corrosion", "deposition", "dissolution")),
        ArenaSpec("substrate", ArenaKind.SOLID_VOLUME, surface_patches, (SOLID,),
                  ("solid-diffusion", "phase-transformation")),
    )
    pairs = tuple((index, index) for index in range(surface_patches))
    boundaries = (
        BoundarySpec("environment_to_surface", "environment", "surface", pairs,
                     BoundaryKind.SURFACE_CONTACT, ("species", "energy"),
                     ("adsorption", "oxidation", "corrosion", "deposition", "dissolution")),
        BoundarySpec("surface_to_substrate", "surface", "substrate", pairs,
                     BoundaryKind.SOLID_CONTACT, ("solid-diffusion", "energy"),
                     ("oxidation", "corrosion", "phase-transformation")),
    )
    return ChemistryPackingPlan(
        states=states, arenas=arenas, boundaries=boundaries, limbs=limbs,
        abi_identity=abi_identity)


def metallurgy_chemistry_plan(states: Sequence[str], *, cells: int = 1,
                              limbs: int = CHEMISTRY_PRECISION_LIMBS,
                              abi_identity: str | None = None,
                              ) -> ChemistryPackingPlan:
    arenas = (
        ArenaSpec("furnace.atmosphere", ArenaKind.GAS_VOLUME, cells, (GAS,),
                  ("gas-phase", "oxidation")),
        ArenaSpec("metal.surface", ArenaKind.REACTIVE_SURFACE, cells,
                  (ADSORBED, SOLID, LIQUID),
                  ("adsorption", "oxidation", "deposition", "dissolution")),
        ArenaSpec("metal.bulk", ArenaKind.METALLURGY_VOLUME, cells,
                  (SOLID, LIQUID),
                  ("solid-diffusion", "alloying", "precipitation", "phase-transformation")),
        ArenaSpec("metal.grain_boundary", ArenaKind.METALLURGY_VOLUME, cells,
                  (SOLID, LIQUID),
                  ("grain-boundary-diffusion", "precipitation", "phase-transformation")),
        ArenaSpec("slag", ArenaKind.FLUID_VOLUME, cells, (LIQUID, SOLID),
                  ("slag-reaction", "dissolution", "precipitation")),
    )
    pairs = tuple((index, index) for index in range(cells))
    boundaries = (
        BoundarySpec("atmosphere_to_metal", "furnace.atmosphere", "metal.surface", pairs,
                     BoundaryKind.SURFACE_CONTACT, ("species", "energy"),
                     ("adsorption", "oxidation", "deposition")),
        BoundarySpec("surface_to_bulk", "metal.surface", "metal.bulk", pairs,
                     BoundaryKind.SOLID_CONTACT, ("solid-diffusion", "energy"),
                     ("oxidation", "dissolution", "phase-transformation")),
        BoundarySpec("bulk_to_grain_boundary", "metal.bulk", "metal.grain_boundary", pairs,
                     BoundaryKind.GRAIN_BOUNDARY,
                     ("grain-boundary-diffusion", "energy"),
                     ("precipitation", "phase-transformation")),
        BoundarySpec("bulk_to_slag", "metal.bulk", "slag", pairs,
                     BoundaryKind.PHASE_INTERFACE, ("species", "energy"),
                     ("slag-reaction", "dissolution", "precipitation")),
    )
    return ChemistryPackingPlan(
        states=states, arenas=arenas, boundaries=boundaries, limbs=limbs,
        abi_identity=abi_identity)


__all__ = [
    "ArenaKind", "BoundaryKind", "ArenaSpec", "BoundarySpec", "ArenaSlice",
    "BoundarySlice", "PackedChemistryMetadata", "ChemistryPackingPlan",
    "rectilinear_neighbor_pairs", "dewar_chemistry_plan",
    "solid_surface_reaction_plan", "metallurgy_chemistry_plan",
]
