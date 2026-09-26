"""The organic extension to the compendium, and the engine concordance."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sympy as sp

from src.common.chemistry import (
    GAS, LIQUID, SOLID,
    ChemicalSpecies,
    ComponentChemistry,
    atmospheric_aqueous_nucleus,
    deployment_manifest,
    engine_toy_identity_registry,
    state_key,
)
from src.common.chemistry.organics import (
    ENGINE_CONCORDANCE,
    UNMAPPED_ENGINE_SPECIES,
    combustion_identity_registry,
    organic_species,
    with_organic_combustion,
)


_ENGINE_TOY = Path(__file__).resolve().parents[2] / "engine_toy"


def _engine_rows():
    """engine_toy's species table, or skip.

    The extension stands on its own; only the CONCORDANCE needs the
    other side present to be checkable."""
    if str(_ENGINE_TOY) not in sys.path:
        sys.path.insert(0, str(_ENGINE_TOY))
    try:
        import organic_species
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"engine_toy not importable: {error}")
    return organic_species.MOLECULES


@pytest.fixture(scope="module")
def extended():
    return with_organic_combustion(atmospheric_aqueous_nucleus())


def test_the_extension_balances_every_reaction_old_and_new(extended):
    """Construction re-validates: an unbalanced row would have raised.

    `ChemicalCompendium._validate` re-runs `reaction_balance` over
    elements AND charge for every reaction in the merged table, so this
    passing is the balance check rather than a stand-in for one."""
    for reaction in extended.reactions.values():
        elements, charge = extended.reaction_balance(reaction)
        assert not {k: v for k, v in elements.items() if v}, reaction.identity
        assert charge == 0, reaction.identity


def test_it_adds_what_a_flame_needs_and_the_nucleus_lacked(extended):
    base = atmospheric_aqueous_nucleus()
    assert "S" not in base.elements and "S" in extended.elements
    for missing in ("CO", "NO2", "SO2", "H2SO4", "CH4", "soot", "diesel"):
        assert missing not in base.species
        assert missing in extended.species
    # and nothing inorganic was disturbed
    for kept in base.species:
        assert kept in extended.species


def test_fractional_blend_formulas_survive_exactly():
    """`ChemicalSpecies.make` is annotated `Mapping[str, int]`, which
    understates it: `_exact` routes every count through
    `sp.Rational(str(value))`, so nothing is coerced or lost.

    This matters because every blend in the table is fractional by
    construction, and a silent int() would turn diesel's CH1.80 into CH1
    and its fifteen-parts-per-million sulphur into none at all."""
    rows = {s.identity: s for s in organic_species()}
    diesel = rows["diesel"].atoms
    assert diesel["H"] == sp.Rational(9, 5)
    assert diesel["S"] == sp.Rational(1, 100000)
    assert rows["gasoline"].atoms["H"] == sp.Rational(187, 100)
    assert rows["soot"].atoms["H"] == sp.Rational(1, 5)
    # a float round trip would lose the ppm entirely
    assert float(diesel["S"]) == 1.0e-5

    # and the same holds going through make() directly with a float
    direct = ChemicalSpecies.make("probe", {"C": 1, "H": 1.87}, 0, 0.0139,
                                  (GAS,), "test")
    assert direct.atoms["H"] == sp.Rational(187, 100)


def test_a_blend_is_a_one_carbon_basis_pseudo_species():
    """Its molar mass is mass per mole of CARBON, not of a molecule.

    Stated as a test because the number looks wrong otherwise: nothing
    weighs 13.9 g/mol, and a reader who assumed a molecular weight would
    conclude the row was broken."""
    rows = {s.identity: s for s in organic_species()}
    for blend in ("gasoline", "diesel", "kerosene", "residual-fuel-oil",
                  "lube-oil"):
        assert rows[blend].atoms["C"] == 1
        assert 0.012 < rows[blend].molar_mass_kg_mol < 0.016
    # the pure compounds are real molecular weights, side by side
    assert rows["C7H8"].molar_mass_kg_mol == pytest.approx(0.0921, abs=1e-4)


def test_the_concordance_is_formula_identical_on_both_sides(extended):
    """The check that makes the mapping real rather than plausible.

    `verify_external_composition` compares the elemental formula of
    every aliased engine row against the compendium's own, as exact
    Rationals -- fractional blend formulas included."""
    registry = combustion_identity_registry(extended)
    registry.verify_external_composition(_engine_rows())


def test_every_engine_species_is_either_mapped_or_explicitly_unmapped():
    """No species may simply go missing between the two tables."""
    rows = _engine_rows()
    mapped = {alias for _, aliases in ENGINE_CONCORDANCE for alias in aliases}
    unaccounted = sorted(set(rows) - mapped - set(UNMAPPED_ENGINE_SPECIES))
    assert unaccounted == []
    # and nothing is claimed both ways
    assert not (mapped & set(UNMAPPED_ENGINE_SPECIES))


def test_it_supersedes_the_five_row_registry_without_contradicting_it(extended):
    """`engine_toy_identity_registry` carries the inorganic five. The
    combustion registry repeats them verbatim rather than forking."""
    small = engine_toy_identity_registry(extended)
    big = combustion_identity_registry(extended)
    for name in ("water", "carbon-dioxide", "nitrogen", "oxygen", "argon"):
        assert small.canonical_species(name) == big.canonical_species(name)


def test_identical_spellings_are_a_coincidence_not_a_rule(extended):
    """`soot` maps to `soot` and `water` maps to `H2O`. Nothing may
    infer either one from the other's spelling."""
    registry = combustion_identity_registry(extended)
    assert registry.canonical_species("soot") == "soot"
    assert registry.canonical_species("water") == "H2O"
    with pytest.raises(KeyError):
        registry.canonical_species("H2O_not_declared")
    # an engine species that exists but was never mapped is refused,
    # not guessed at
    with pytest.raises(KeyError):
        registry.canonical_species("n-heptane")


def test_phase_states_exist_for_what_an_engine_actually_emits(extended):
    registry = combustion_identity_registry(extended)
    assert registry.state("water", GAS) == state_key("H2O", GAS)
    assert registry.state("soot", SOLID) == state_key("soot", SOLID)
    assert registry.state("sulfuric-acid", LIQUID) == state_key("H2SO4", LIQUID)
    # lubricating oil has no gas state on purpose: what gets past a ring
    # leaves as a droplet, which is why blue smoke is not unburnt fuel
    assert registry.state("lubricating-oil", LIQUID)
    with pytest.raises(KeyError):
        registry.state("lubricating-oil", GAS)


def test_no_combustion_reaction_was_authored(extended):
    """The extension adds species and real phase transitions only.

    Authoring global oxidation rows with invented equilibrium constants
    would put a second reaction solver inside the compendium. If that
    ever becomes wanted it should be a deliberate change with real
    thermochemistry behind it, not a drift."""
    base = atmospheric_aqueous_nucleus()
    added = set(extended.reactions) - set(base.reactions)
    assert added
    for identity in added:
        assert extended.reactions[identity].family == "phase", identity


def test_the_engine_boundary_states_survive_a_deployment(extended):
    """The organic states reduce and pack like any others."""
    registry = combustion_identity_registry(extended)
    seeds = tuple(registry.state(name, phase) for name, phase in (
        ("carbon-dioxide", GAS), ("water", GAS), ("nitrogen", GAS),
        ("oxygen", GAS), ("soot", SOLID), ("sulfuric-acid", LIQUID),
        ("diesel-blend", GAS), ("lubricating-oil", LIQUID)))
    manifest = deployment_manifest(
        extended, [ComponentChemistry("engine-bay", seeds)])
    for state in seeds:
        assert state in manifest.states
        manifest.state_index(state)
    assert manifest.abi_identity
