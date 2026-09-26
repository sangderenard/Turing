"""Organic and combustion species, authored for the master compendium.

WHAT THIS ADDS AND WHY IT IS SEPARATE

`atmospheric_aqueous_nucleus()` is inorganic and aqueous: water, its
ions, the carbonate system, halite, and the three air gases. Everything
a dewar, a brine and an atmosphere need. What it cannot express is
anything a FLAME makes, because it has no element sulphur, no carbon
monoxide, no nitrogen oxide, no hydrocarbon of any kind, and no
condensed carbonaceous matter.

This module authors those, in the nucleus's own idiom -- `Element`,
`ChemicalSpecies.make`, `ChemicalState`, `Reaction.make` -- so they
enter the same compendium, are balanced by the same
`reaction_balance` over elements AND charge, and are reduced by the
same `reduce_reachable`. It is a separate module because it is a
separate provenance, not a separate mechanism: the inorganic rows cite
PHREEQC and the repository phase table, and these cite the combustion
literature and `engine_toy`'s own fuel catalogue.

IT AUTHORS NO COMBUSTION REACTION, ON PURPOSE

There are no oxidation rows here. A combustion mechanism is hundreds of
elementary reactions with real rate constants, and writing a handful of
global ones with invented equilibrium constants would put a second,
worse reaction solver inside the compendium that the real one would
then have to be reconciled against. The phase transitions below ARE
authored, because each is a real cited transition temperature and
latent heat already carried in this repository, and each balances
trivially and exactly.

What burns is delivered instead as a BOUNDARY SOURCE, in moles per
state, through `EngineSpeciesExchangeLayout` -- see
`engine_toy/chemistry_boundary.py`. An engine is a thing that injects
species amounts at a boundary. It is not a reaction the solver should
be integrating.

A BLEND IS A PSEUDO-SPECIES ON A CARBON BASIS, AND SAYS SO

Gasoline, diesel, kerosene, residual fuel oil, lubricating oil and soot
have no molecular formula, because they are distributions of hundreds
of compounds. The refinery characterises each by its ULTIMATE ANALYSIS
-- average atoms per carbon, written CH1.87 -- and that is what is
authored here, as a one-carbon-basis pseudo-species. Its molar mass is
therefore the mass per mole of CARBON, not the mass of a molecule, and
a mole of `gasoline` is a mole of fuel carbon. Every stoichiometric and
mass/mole conversion at the boundary is correct on that basis; anything
that wants a real molecular weight wants a real compound, and the pure
compounds are here too.

`ChemicalSpecies.make` preserves these counts exactly: `_exact` routes
every value through `sp.Rational(str(value))`, so 1.87 is stored as
187/100 and a fifteen-parts-per-million sulphur content as 1/100000.
The `Mapping[str, int]` annotation on `make` understates what the
implementation does; nothing is coerced or lost.
"""

from __future__ import annotations

import sympy as sp

from .compendium import (
    AQUEOUS, GAS, LIQUID, SOLID,
    ChemicalCompendium, ChemicalSpecies, ChemicalState, Reaction,
    TransitionEquilibrium, state_key,
)
from .identity import ChemicalIdentityRegistry, SpeciesIdentityRecord


# ---------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------
CATALOGUE = "engine_toy/working_fluids.py fuel catalogue"
CRYO = "engine_toy/cryogenics.py CRYOGENS"
SOOT_SOURCE = ("combustion literature: nascent soot H/C ~0.2, "
               "primary-particle density 1800 kg/m3")
NIST = "NIST standard reference molar masses"


def organic_elements():
    """Sulphur. Every other element these species need is already in the
    nucleus (H, C, N, O), and importing `Element` for the ones that are
    would create the duplicate `_unique` exists to refuse."""
    from .compendium import Element

    return (Element("S", 16, 32.06, NIST),)


# ---------------------------------------------------------------------
# species
# ---------------------------------------------------------------------
# Molar masses are the real ones for compounds. For the blends they are
# the mass per mole of carbon implied by the authored formula, which is
# what a one-carbon-basis pseudo-species means; they are written out
# rather than computed so the row states its own number the way every
# other row in the compendium does.

def organic_species():
    return (
        # -- combustion companions the nucleus lacks ------------------
        ChemicalSpecies.make("CO", {"C": 1, "O": 1}, 0, 0.0280101,
                             (GAS,), NIST),
        ChemicalSpecies.make("NO", {"N": 1, "O": 1}, 0, 0.0300061,
                             (GAS,), NIST),
        ChemicalSpecies.make("NO2", {"N": 1, "O": 2}, 0, 0.0460055,
                             (GAS,), NIST),
        ChemicalSpecies.make("SO2", {"S": 1, "O": 2}, 0, 0.0640638,
                             (GAS, AQUEOUS), NIST),
        ChemicalSpecies.make("SO3", {"S": 1, "O": 3}, 0, 0.0800632,
                             (GAS,), NIST),
        ChemicalSpecies.make("H2SO4", {"H": 2, "S": 1, "O": 4}, 0, 0.0980785,
                             (LIQUID, AQUEOUS), NIST),

        # -- pure hydrocarbons and oxygenates -------------------------
        ChemicalSpecies.make("CH4", {"C": 1, "H": 4}, 0, 0.0160425,
                             (GAS, LIQUID), CRYO),
        ChemicalSpecies.make("C3H8", {"C": 3, "H": 8}, 0, 0.0440956,
                             (GAS, LIQUID), CATALOGUE),
        ChemicalSpecies.make("H2", {"H": 2}, 0, 0.00201588,
                             (GAS, LIQUID), CRYO),
        ChemicalSpecies.make("C6H6", {"C": 6, "H": 6}, 0, 0.0781118,
                             (GAS, LIQUID), NIST),
        ChemicalSpecies.make("C7H8", {"C": 7, "H": 8}, 0, 0.0921384,
                             (GAS, LIQUID), NIST),
        ChemicalSpecies.make("C8H10", {"C": 8, "H": 10}, 0, 0.106165,
                             (GAS, LIQUID), NIST),
        ChemicalSpecies.make("C10H8", {"C": 10, "H": 8}, 0, 0.128174,
                             (GAS, LIQUID, SOLID), NIST),
        ChemicalSpecies.make("CH4O", {"C": 1, "H": 4, "O": 1}, 0, 0.0320419,
                             (GAS, LIQUID), CATALOGUE),
        ChemicalSpecies.make("C2H6O", {"C": 2, "H": 6, "O": 1}, 0, 0.0460684,
                             (GAS, LIQUID), CATALOGUE),
        ChemicalSpecies.make("CH3NO2", {"C": 1, "H": 3, "N": 1, "O": 2}, 0,
                             0.0610400, (GAS, LIQUID), CATALOGUE),

        # -- blends, one-carbon basis; see the module docstring -------
        ChemicalSpecies.make("gasoline", {"C": 1, "H": "1.87"}, 0,
                             0.0138958, (GAS, LIQUID), CATALOGUE),
        # The gas state on the heavy fuels is the unburnt vapour that
        # leaves a chamber, which is real even for a fuel that does not
        # distil cleanly -- a diesel's hydrocarbon emission is partly
        # vapour and partly condensate adsorbed onto soot. Lubricating
        # oil deliberately has no gas state: what gets past a ring
        # leaves as a droplet, and that is exactly the difference
        # between blue smoke and unburnt fuel.
        ChemicalSpecies.make("diesel", {"C": 1, "H": "1.80", "S": "1.0e-5"}, 0,
                             0.0138254, (GAS, LIQUID), CATALOGUE),
        ChemicalSpecies.make("kerosene", {"C": 1, "H": "1.92", "S": "2.1e-4"}, 0,
                             0.0139463, (GAS, LIQUID), CATALOGUE),
        ChemicalSpecies.make("residual-fuel-oil",
                             {"C": 1, "H": "1.55", "S": "1.085e-2"}, 0,
                             0.0139212, (GAS, LIQUID), CATALOGUE),
        ChemicalSpecies.make("crude-oil",
                             {"C": 1, "H": "1.75", "S": "1.05e-2"}, 0,
                             0.0141376, (GAS, LIQUID), CATALOGUE),
        ChemicalSpecies.make("lube-oil",
                             {"C": 1, "H": "1.87", "S": "2.3e-3"}, 0,
                             0.0139696, (LIQUID,), CATALOGUE),
        ChemicalSpecies.make("wood", {"C": 1, "H": "1.44", "O": "0.66"}, 0,
                             0.0240215, (SOLID,), "engine_toy/gas_works.py"),
        ChemicalSpecies.make("coal",
                             {"C": 1, "H": "0.80", "O": "0.13", "S": "0.005"}, 0,
                             0.0150585, (SOLID,), "engine_toy/gas_works.py"),

        # -- condensed carbonaceous matter ----------------------------
        # Soot is not elemental carbon and the difference is measurable:
        # nascent soot keeps about a fifth of a hydrogen per carbon and
        # sheds it as it ages, which is why old soot absorbs light
        # differently from new soot.
        ChemicalSpecies.make("soot", {"C": 1, "H": "0.20"}, 0, 0.0122126,
                             (SOLID,), SOOT_SOURCE),
        ChemicalSpecies.make("C", {"C": 1}, 0, 0.0120111, (SOLID,),
                             "graphitised elemental carbon, thermal-optical EC"),
    )


def organic_states(species):
    return tuple(
        ChemicalState(spec.identity, phase, spec.source)
        for spec in species for phase in spec.allowed_phases
    )


def organic_phase_reactions():
    """Real, cited transitions only -- no oxidation rows.

    Each latent heat below is the figure `engine_toy/cryogenics.py`
    already carries for the same species, converted from J/kg to J/mol
    by that species' own molar mass, exactly as the nucleus does for
    water, nitrogen and oxygen."""
    return (
        Reaction.make(
            "methane-vaporization",
            {state_key("CH4", LIQUID): -1, state_key("CH4", GAS): 1},
            TransitionEquilibrium("111.7", 511_000 * sp.Rational("0.0160425")),
            "phase", CRYO),
        Reaction.make(
            "hydrogen-vaporization",
            {state_key("H2", LIQUID): -1, state_key("H2", GAS): 1},
            TransitionEquilibrium("20.28", 449_000 * sp.Rational("0.00201588")),
            "phase", CRYO),
        Reaction.make(
            "propane-vaporization",
            {state_key("C3H8", LIQUID): -1, state_key("C3H8", GAS): 1},
            TransitionEquilibrium("231.0", 430_000 * sp.Rational("0.0440956")),
            "phase", CATALOGUE),
        Reaction.make(
            "methanol-vaporization",
            {state_key("CH4O", LIQUID): -1, state_key("CH4O", GAS): 1},
            TransitionEquilibrium("337.85", 1_100_000 * sp.Rational("0.0320419")),
            "phase", CATALOGUE),
        Reaction.make(
            "ethanol-vaporization",
            {state_key("C2H6O", LIQUID): -1, state_key("C2H6O", GAS): 1},
            TransitionEquilibrium("351.4", 841_000 * sp.Rational("0.0460684")),
            "phase", CATALOGUE),
        Reaction.make(
            "naphthalene-sublimation",
            {state_key("C10H8", SOLID): -1, state_key("C10H8", GAS): 1},
            TransitionEquilibrium("353.4", 72_600), "phase", NIST),
        Reaction.make(
            "sulfuric-acid-dissolution",
            {state_key("H2SO4", LIQUID): -1, state_key("H2SO4", AQUEOUS): 1},
            TransitionEquilibrium("298.15", 95_300), "phase", NIST),
        Reaction.make(
            "sulfur-dioxide-dissolution",
            {state_key("SO2", GAS): -1, state_key("SO2", AQUEOUS): 1},
            TransitionEquilibrium("263.0", -24_900), "phase", NIST),
    )


def with_organic_combustion(base: ChemicalCompendium) -> ChemicalCompendium:
    """`base` plus the organic and combustion rows, as one compendium.

    Construction re-validates everything: `_unique` refuses a duplicate
    identity, and `_validate` re-balances every reaction, old and new,
    over elements and charge. A collision with a nucleus species or an
    unbalanced organic transition fails here rather than downstream."""
    species = organic_species()
    return ChemicalCompendium(
        elements=(*base.elements.values(), *organic_elements()),
        species=(*base.species.values(), *species),
        states=(*base.states.values(), *organic_states(species)),
        reactions=(*base.reactions.values(), *organic_phase_reactions()),
    )


# ---------------------------------------------------------------------
# the concordance: engine identity <-> canonical identity, stated
# ---------------------------------------------------------------------
# Every pair is written out. Nothing is matched by name similarity, by
# prefix, by lowercasing or by any other inference: `water` and `H2O`
# look nothing alike and neither do `soot` and `soot`, and the fact that
# one of those pairs happens to be spelled identically is not a rule
# anything may rely on. `ChemicalIdentityRegistry` refuses a name that
# maps to two canonical species, and
# `ChemicalIdentityRegistry.verify_external_composition` proves each
# alias describes the same elemental formula on both sides -- so a
# concordance row that is merely plausible fails a test rather than
# silently mis-routing a species.
#
# left: the canonical compendium species. right: every engine-side
# identity that means it (`engine_toy/organic_species.py` keys).
ENGINE_CONCORDANCE: tuple[tuple[str, tuple[str, ...]], ...] = (
    # inorganic, already declared by engine_toy_identity_registry and
    # repeated here so ONE table is the whole concordance
    ("H2O", ("water",)),
    ("CO2", ("carbon-dioxide",)),
    ("N2", ("nitrogen",)),
    ("O2", ("oxygen",)),
    ("Ar", ("argon",)),
    # combustion companions
    ("CO", ("carbon-monoxide",)),
    ("NO", ("nitric-oxide",)),
    ("NO2", ("nitrogen-dioxide",)),
    ("SO2", ("sulfur-dioxide",)),
    ("H2SO4", ("sulfuric-acid",)),
    # pure compounds
    ("CH4", ("methane",)),
    ("C3H8", ("propane",)),
    ("H2", ("hydrogen-gas",)),
    ("C6H6", ("benzene",)),
    ("C7H8", ("toluene",)),
    ("C8H10", ("xylene",)),
    ("C10H8", ("naphthalene",)),
    ("CH4O", ("methanol",)),
    ("C2H6O", ("ethanol",)),
    ("CH3NO2", ("nitromethane",)),
    # blends
    ("gasoline", ("gasoline-blend",)),
    ("diesel", ("diesel-blend",)),
    ("kerosene", ("kerosene-blend",)),
    ("residual-fuel-oil", ("heavy-fuel-oil-blend",)),
    ("crude-oil", ("crude-blend",)),
    ("lube-oil", ("lubricating-oil",)),
    ("wood", ("wood-fuel",)),
    ("coal", ("coal-fuel",)),
    # condensed carbonaceous matter
    ("soot", ("soot",)),
    ("C", ("elemental-carbon",)),
)

#: Engine-side species with NO canonical counterpart, and the reason.
#: Listed rather than omitted, so a species cannot go missing at the
#: boundary without something saying which one and why.
UNMAPPED_ENGINE_SPECIES: dict[str, str] = {
    "ethane": "no compendium row authored; add C2H6 when something needs it",
    "n-butane": "no compendium row authored",
    "iso-octane": "no compendium row authored; the octane-scale reference",
    "n-heptane": "no compendium row authored; the octane-scale reference",
    "n-decane": "no compendium row authored; kerosene surrogate",
    "n-dodecane": "no compendium row authored; diesel surrogate",
    "n-hexadecane": "no compendium row authored; the cetane-scale reference",
    "cyclohexane": "no compendium row authored",
    "pyrene": "no compendium row authored; the soot nucleation step",
    "methyl-oleate": "no compendium row authored; biodiesel surrogate",
    "triolein": "no compendium row authored; vegetable oil",
    "coke-fuel": "no compendium row authored",
    "charcoal-fuel": "no compendium row authored",
}


def combustion_identity_registry(
        compendium: ChemicalCompendium) -> ChemicalIdentityRegistry:
    """The whole engine/chemistry concordance, inorganic and organic.

    Supersedes `engine_toy_identity_registry`, which carries the first
    five rows only; those five are repeated in `ENGINE_CONCORDANCE`
    verbatim so there is one table rather than two that can disagree."""
    return ChemicalIdentityRegistry(compendium, tuple(
        SpeciesIdentityRecord(canonical, aliases)
        for canonical, aliases in ENGINE_CONCORDANCE))


__all__ = [
    "CATALOGUE", "CRYO", "ENGINE_CONCORDANCE", "UNMAPPED_ENGINE_SPECIES",
    "combustion_identity_registry", "organic_elements",
    "organic_phase_reactions", "organic_species", "organic_states",
    "with_organic_combustion",
]
