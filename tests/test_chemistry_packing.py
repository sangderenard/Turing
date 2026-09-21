from __future__ import annotations

from src.common.tensors.abstraction import AbstractTensor

from src.common.chemistry import (
    ArenaKind,
    CHEMISTRY_PRECISION_LIMBS,
    dewar_chemistry_plan,
    metallurgy_chemistry_plan,
    solid_surface_reaction_plan,
)


STATES = ("N2@gas", "H2O@gas", "H2O@liquid", "Fe@solid", "O@adsorbed")


def test_dewar_arenas_are_contiguous_and_vacuum_is_a_fluid_volume():
    plan = dewar_chemistry_plan(STATES, chamber_shape=(2, 2, 2))

    assert plan.owner_count == sum(
        arena.count for arena in plan.arenas if arena.carries_inventory)
    assert plan.arena_slices["chamber.gas"].owner_start == 0
    assert plan.arena_slices["chamber.gas"].owner_stop == 8
    assert plan.arena_slices["chamber.aerosol"].owner_start == 8
    assert plan.arena_slices["jacket.vacuum"].owner_start is not None
    assert plan.arena_slices["jacket.vacuum"].owner_stop == (
        plan.arena_slices["jacket.vacuum"].owner_start + 1)
    assert plan.arena_slices["jacket.vacuum"].arena.kind is ArenaKind.FLUID_VOLUME
    assert plan.arena_slices["jacket.vacuum"].arena.phases == ("gas",)
    assert plan.arena_slices["jacket.vacuum"].arena.scalar_fields == (
        "temperature_k", "internal_energy_j", "pressure_pa",
        "vacuum_integrity", "thermal_conductance_w_k")
    assert plan.boundary_slices["jacket.vacuum.top_port"].boundary.count == 1
    assert plan.boundary_slices["jacket.vacuum.bottom_port"].boundary.count == 1


def test_inventory_is_owner_state_limb_interleaved_on_abstract_tensor():
    plan = dewar_chemistry_plan(STATES, chamber_shape=(1, 1, 1))
    packed = plan.pack_inventory({
        ("chamber.gas", 0, "N2@gas"): (2.0, 0.25),
        ("outside.atmosphere", 0, "H2O@gas"): 3.0,
    })

    assert isinstance(packed, AbstractTensor)
    assert CHEMISTRY_PRECISION_LIMBS == plan.limbs == 2
    assert int(packed.numel()) == plan.physical_inventory_size
    values = packed.tolist()
    n2 = plan.physical_index("chamber.gas", 0, "N2@gas")
    water = plan.physical_index("outside.atmosphere", 0, "H2O@gas")
    assert values[n2:n2 + 2] == [2.0, 0.25]
    assert values[water:water + 2] == [3.0, 0.0]


def test_oriented_boundary_scatter_conserves_every_state():
    plan = solid_surface_reaction_plan(STATES, surface_patches=2)
    count = plan.boundary_count * len(STATES)
    flux = AbstractTensor.get_tensor([float(index + 1) for index in range(count)])

    delta = plan.scatter_boundary_flux(flux).tolist()
    for state_index in range(len(STATES)):
        assert sum(delta[state_index::len(STATES)]) == 0.0


def test_metadata_keeps_owner_identity_and_independent_activity_masks():
    plan = metallurgy_chemistry_plan(STATES, cells=2)
    metadata = plan.materialize_metadata()
    surface = plan.owner("metal.surface", 0)
    bulk = plan.owner("metal.bulk", 0)
    oxidation = plan.reaction_families.index("oxidation")
    alloying = plan.reaction_families.index("alloying")
    width = len(plan.reaction_families)
    mask = metadata.owner_reaction_mask.tolist()

    assert isinstance(metadata.owner_arena, AbstractTensor)
    assert int(metadata.owner_kind.tolist()[bulk]) == int(ArenaKind.METALLURGY_VOLUME)
    assert mask[surface * width + oxidation] == 1.0
    assert mask[surface * width + alloying] == 0.0
    assert mask[bulk * width + oxidation] == 0.0
    assert mask[bulk * width + alloying] == 1.0
    state_mask = metadata.owner_state_mask.tolist()
    state_width = len(STATES)
    assert state_mask[surface * state_width + STATES.index("O@adsorbed")] == 1.0
    assert state_mask[bulk * state_width + STATES.index("O@adsorbed")] == 0.0
    assert state_mask[bulk * state_width + STATES.index("Fe@solid")] == 1.0


def test_rectilinear_dewar_face_count_and_global_owner_references_are_stable():
    plan = dewar_chemistry_plan(STATES, chamber_shape=(2, 3, 4))
    gas_faces = plan.boundary_slices["chamber.gas_faces"]
    # x faces + y faces + z faces
    assert gas_faces.boundary.count == (1 * 3 * 4) + (2 * 2 * 4) + (2 * 3 * 3)
    rows = plan.boundary_owner_rows()
    assert all(0 <= left < plan.owner_count and 0 <= right < plan.owner_count
               for left, right in rows)
    assert plan.owner("outside.atmosphere", 0) != plan.owner("chamber.gas", 0)
