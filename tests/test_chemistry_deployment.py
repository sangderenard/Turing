from __future__ import annotations

import pytest

from src.common.tensors.abstraction import AbstractTensor

from src.common.chemistry import (
    GAS,
    ChemicalCompendium,
    ChemicalSpecies,
    ChemicalState,
    ComponentChemistry,
    Element,
    EngineSpeciesExchangeLayout,
    KineticLaw,
    Reaction,
    atmospheric_aqueous_nucleus,
    deployment_manifest,
    dewar_chemistry_plan,
    state_key,
)


def test_component_reachability_allocates_products_and_phase_transitions_only():
    master = atmospheric_aqueous_nucleus()
    manifest = deployment_manifest(master, (
        ComponentChemistry("ambient-water", (state_key("H2O", "aqueous"),)),
        ComponentChemistry("air-nitrogen", (state_key("N2", "gas"),)),
    ))

    assert state_key("H+", "aqueous") in manifest.states
    assert state_key("OH-", "aqueous") in manifest.states
    assert state_key("H2O", "gas") in manifest.states
    assert state_key("H2O", "solid") in manifest.states
    assert state_key("N2", "liquid") in manifest.states
    assert state_key("N2", "solid") in manifest.states
    assert state_key("Na+", "aqueous") not in manifest.states
    assert state_key("CO2", "gas") not in manifest.states


def _transition_chain(length: int) -> ChemicalCompendium:
    source = "arbitrary-capacity-test"
    element = Element("X", 999, 1.0, source)
    species = tuple(
        ChemicalSpecies.make(f"X{index}", {"X": 1}, 0, 1.0, (GAS,), source)
        for index in range(length)
    )
    states = tuple(ChemicalState(row.identity, GAS, source) for row in species)
    reactions = tuple(
        Reaction.make(
            f"transition-{index}",
            {states[index].identity: -1, states[index + 1].identity: 1},
            KineticLaw(1, 0), "configured-transition", source)
        for index in range(length - 1)
    )
    return ChemicalCompendium(
        elements=(element,), species=species, states=states, reactions=reactions)


def test_capacity_is_derived_from_arbitrarily_long_reachable_transition_chain():
    master = _transition_chain(19)
    first = state_key("X0", GAS)
    manifest = deployment_manifest(
        master, (ComponentChemistry("feed", (first,)),), families=("configured-transition",))
    exchange = manifest.engine_exchange_layout(("supply", "chamber"))
    plan = manifest.dewar_packing_plan(chamber_shape=(1, 1, 1))

    assert manifest.state_capacity == manifest.species_capacity == 19
    assert manifest.states == plan.states
    assert plan.abi_identity == manifest.abi_identity
    assert exchange.logical_size == 2 * manifest.state_capacity
    assert exchange.physical_size == exchange.logical_size * 2
    assert len(manifest.abi_identity) == 64


def test_engine_exchange_uses_canonical_state_identity_and_manifest_capacity():
    master = atmospheric_aqueous_nucleus()
    water = state_key("H2O", GAS)
    nitrogen = state_key("N2", GAS)
    manifest = deployment_manifest(master, (
        ComponentChemistry("atmosphere", (water, nitrogen)),
    ))
    exchange = EngineSpeciesExchangeLayout(manifest, ("outside", "port"))
    packed = exchange.pack_kilograms({
        "outside": {water: 0.01801528, nitrogen: 0.0280134},
    })
    unpacked = exchange.unpack_moles(packed)

    assert isinstance(packed, AbstractTensor)
    assert unpacked["outside"][water] == pytest.approx(1.0)
    assert unpacked["outside"][nitrogen] == pytest.approx(1.0)
    assert unpacked["port"][water] == 0.0
    with pytest.raises(KeyError):
        exchange.pack_kilograms({"outside": {state_key("NaCl", "solid"): 1.0}})


def test_manifest_rejects_a_different_state_axis_on_the_other_side():
    master = atmospheric_aqueous_nucleus()
    manifest = deployment_manifest(master, (
        ComponentChemistry("atmosphere", (state_key("N2", GAS),)),
    ))
    manifest.require_state_axis(manifest.states)
    with pytest.raises(ValueError, match="ABI mismatch"):
        manifest.require_state_axis(tuple(reversed(manifest.states)))
