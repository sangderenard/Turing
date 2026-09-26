"""One canonical chemical identity with explicit engine-facing names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import sympy as sp

from .compendium import ChemicalCompendium, state_key


@dataclass(frozen=True)
class SpeciesIdentityRecord:
    canonical: str
    external_identities: tuple[str, ...] = ()


class ChemicalIdentityRegistry:
    """Authoritative aliases as a derived view of compendium identities."""

    def __init__(self, compendium: ChemicalCompendium,
                 records: Iterable[SpeciesIdentityRecord]):
        self.compendium = compendium
        self.records = tuple(records)
        by_name: dict[str, str] = {}
        for record in self.records:
            if record.canonical not in compendium.species:
                raise KeyError(f"unknown canonical species {record.canonical}")
            for name in (record.canonical, *record.external_identities):
                previous = by_name.get(name)
                if previous is not None and previous != record.canonical:
                    raise ValueError(
                        f"chemical identity {name!r} maps to both {previous} "
                        f"and {record.canonical}")
                by_name[name] = record.canonical
        self._by_name = by_name

    def canonical_species(self, identity: str) -> str:
        try:
            return self._by_name[str(identity)]
        except KeyError as error:
            raise KeyError(f"unregistered chemical identity {identity!r}") from error

    def state(self, identity: str, phase: str) -> str:
        candidate = state_key(self.canonical_species(identity), phase)
        if candidate not in self.compendium.states:
            raise KeyError(f"chemical state is not declared: {candidate}")
        return candidate

    def verify_external_composition(self, rows: Mapping[str, object]) -> None:
        """Prove aliased external rows describe the same elemental formula."""
        fields = {
            "H": "hydrogen", "C": "carbon", "N": "nitrogen",
            "O": "oxygen", "S": "sulfur", "Ar": "argon",
        }
        for external, canonical in self._by_name.items():
            if external == canonical or external not in rows:
                continue
            row = rows[external]
            observed = {
                element: sp.Rational(str(getattr(row, attribute)))
                for element, attribute in fields.items()
                if getattr(row, attribute, 0.0) != 0.0
            }
            expected = dict(self.compendium.species[canonical].formula)
            if observed != expected:
                raise ValueError(
                    f"identity discordance for {external!r}/{canonical!r}: "
                    f"external={observed}, compendium={expected}")


def engine_toy_identity_registry(
        compendium: ChemicalCompendium) -> ChemicalIdentityRegistry:
    """The explicit overlap between engine_toy and the initial compendium."""
    return ChemicalIdentityRegistry(compendium, (
        SpeciesIdentityRecord("H2O", ("water",)),
        SpeciesIdentityRecord("CO2", ("carbon-dioxide",)),
        SpeciesIdentityRecord("N2", ("nitrogen",)),
        SpeciesIdentityRecord("O2", ("oxygen",)),
        SpeciesIdentityRecord("Ar", ("argon",)),
    ))


__all__ = [
    "SpeciesIdentityRecord", "ChemicalIdentityRegistry",
    "engine_toy_identity_registry",
]
