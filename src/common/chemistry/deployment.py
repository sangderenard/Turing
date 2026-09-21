"""One species-capacity contract for chemistry and engine exchange."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Mapping, Sequence

from src.common.tensors.abstraction import AbstractTensor

from .compendium import ChemicalCompendium, PHASES, ReducedCompendium
from .state import CHEMISTRY_PRECISION_LIMBS


@dataclass(frozen=True)
class ExternalStateBinding:
    endpoint: str
    external_identity: str
    chemical_state: str


@dataclass(frozen=True)
class ComponentChemistry:
    """Exact chemical identities initially present in one configured component."""

    identity: str
    initial_states: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("component identity must not be empty")
        if not self.initial_states:
            raise ValueError(f"{self.identity}: component needs at least one chemical state")


@dataclass(frozen=True)
class ChemistryDeploymentManifest:
    """Frozen state/reaction identity table shared by every deployment side."""

    components: tuple[ComponentChemistry, ...]
    reduced: ReducedCompendium
    allowed_phases: tuple[str, ...]
    enabled_families: tuple[str, ...]
    abi_identity: str

    @property
    def states(self) -> tuple[str, ...]:
        return self.reduced.states

    @property
    def species(self) -> tuple[str, ...]:
        master = self.reduced.master
        return tuple(dict.fromkeys(master.states[state].species for state in self.states))

    @property
    def state_capacity(self) -> int:
        return len(self.states)

    @property
    def species_capacity(self) -> int:
        return len(self.species)

    @property
    def reactions(self) -> tuple[str, ...]:
        return tuple(reaction.identity for reaction in self.reduced.reactions)

    def state_index(self, state: str) -> int:
        try:
            return self.states.index(str(state))
        except ValueError as error:
            raise KeyError(state) from error

    def require_state_axis(self, states: Sequence[str]) -> None:
        supplied = tuple(map(str, states))
        if supplied != self.states:
            raise ValueError(
                "chemistry state ABI mismatch: both sides must use manifest.states")

    def dewar_packing_plan(self, chamber_shape=(5, 5, 5), *,
                           limbs: int = CHEMISTRY_PRECISION_LIMBS):
        from .packing import dewar_chemistry_plan

        return dewar_chemistry_plan(
            self.states, chamber_shape, limbs=limbs, abi_identity=self.abi_identity)

    def engine_exchange_layout(self, endpoints: Sequence[str], *,
                               limbs: int = CHEMISTRY_PRECISION_LIMBS
                               ) -> "EngineSpeciesExchangeLayout":
        return EngineSpeciesExchangeLayout(self, tuple(map(str, endpoints)), limbs)


def deployment_manifest(compendium: ChemicalCompendium,
                        components: Iterable[ComponentChemistry], *,
                        allowed_phases: Iterable[str] = PHASES,
                        families: Iterable[str] | None = None
                        ) -> ChemistryDeploymentManifest:
    """Determine capacity from every transition reachable by the components."""
    components = tuple(components)
    if not components:
        raise ValueError("chemistry deployment requires at least one component")
    names = [component.identity for component in components]
    if len(set(names)) != len(names):
        raise ValueError("component chemistry identities must be unique")
    phases = tuple(dict.fromkeys(map(str, allowed_phases)))
    enabled_families = (
        tuple(dict.fromkeys(map(str, families))) if families is not None
        else tuple(dict.fromkeys(
            reaction.family for reaction in compendium.reactions.values()))
    )
    seeds = tuple(dict.fromkeys(
        state for component in components for state in component.initial_states))
    reduced = compendium.reduce_reachable(
        seeds, allowed_phases=phases, families=enabled_families)
    fingerprint_rows = (
        *(f"component:{component.identity}:{','.join(component.initial_states)}"
          for component in components),
        *(f"state:{state}" for state in reduced.states),
        *(f"reaction:{reaction.identity}" for reaction in reduced.reactions),
        f"phases:{','.join(phases)}",
        f"families:{','.join(enabled_families)}",
    )
    abi_identity = sha256("\n".join(fingerprint_rows).encode("utf-8")).hexdigest()
    return ChemistryDeploymentManifest(
        components, reduced, phases, enabled_families, abi_identity)


@dataclass(frozen=True)
class EngineSpeciesExchangeLayout:
    """Engine-facing endpoint spans using the manifest's exact state axis.

    The existing engine mixture containers are dictionaries and therefore have
    no hard species ceiling. This layout fixes their exchange representation at
    deployment: `[endpoint][state][precision limb]`, in moles. Keys crossing
    the boundary are canonical chemical-state identities, including phase.
    """

    manifest: ChemistryDeploymentManifest
    endpoints: tuple[str, ...]
    limbs: int = CHEMISTRY_PRECISION_LIMBS
    bindings: tuple[ExternalStateBinding, ...] = ()

    def __post_init__(self) -> None:
        if not self.endpoints:
            raise ValueError("species exchange requires at least one endpoint")
        if len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("species exchange endpoint identities must be unique")
        if int(self.limbs) <= 0:
            raise ValueError("precision limb count must be positive")
        seen: set[tuple[str, str]] = set()
        for binding in self.bindings:
            key = (binding.endpoint, binding.external_identity)
            if binding.endpoint not in self.endpoints:
                raise KeyError(f"binding uses unknown endpoint {binding.endpoint}")
            self.manifest.state_index(binding.chemical_state)
            if key in seen:
                raise ValueError(f"duplicate external chemical binding {key}")
            seen.add(key)

    @property
    def logical_size(self) -> int:
        return len(self.endpoints) * self.manifest.state_capacity

    @property
    def physical_size(self) -> int:
        return self.logical_size * int(self.limbs)

    def physical_index(self, endpoint: str, state: str, limb: int = 0) -> int:
        try:
            endpoint_index = self.endpoints.index(str(endpoint))
        except ValueError as error:
            raise KeyError(endpoint) from error
        if not 0 <= int(limb) < int(self.limbs):
            raise IndexError(limb)
        logical = endpoint_index * self.manifest.state_capacity
        logical += self.manifest.state_index(state)
        return logical * int(self.limbs) + int(limb)

    def pack_moles(self, mixtures: Mapping[str, Mapping[str, object]]) -> AbstractTensor:
        packed = [0.0] * self.physical_size
        for endpoint, mixture in mixtures.items():
            if endpoint not in self.endpoints:
                raise KeyError(endpoint)
            for state, value in mixture.items():
                if isinstance(value, (tuple, list)):
                    terms = tuple(value)
                    if len(terms) != self.limbs:
                        raise ValueError(
                            f"{endpoint} {state}: expected {self.limbs} precision limbs")
                else:
                    terms = (value,) + (0.0,) * (self.limbs - 1)
                for limb, term in enumerate(terms):
                    packed[self.physical_index(endpoint, state, limb)] = term
        return AbstractTensor.get_tensor(packed)

    def pack_kilograms(self, mixtures: Mapping[str, Mapping[str, float]]) -> AbstractTensor:
        master = self.manifest.reduced.master
        bindings = {
            (binding.endpoint, binding.external_identity): binding.chemical_state
            for binding in self.bindings
        }
        moles: dict[str, dict[str, float]] = {}
        for endpoint, mixture in mixtures.items():
            converted: dict[str, float] = {}
            for supplied_identity, mass_kg in mixture.items():
                state = bindings.get(
                    (endpoint, supplied_identity), supplied_identity)
                state_index = self.manifest.state_index(state)
                canonical = self.manifest.states[state_index]
                species = master.species[master.states[canonical].species]
                converted[canonical] = float(mass_kg) / species.molar_mass_kg_mol
            moles[endpoint] = converted
        return self.pack_moles(moles)

    def unpack_moles(self, packed: AbstractTensor) -> dict[str, dict[str, float]]:
        if int(packed.numel()) != self.physical_size:
            raise ValueError(
                f"expected {self.physical_size} exchange values, got {packed.numel()}")
        values = packed.tolist()
        return {
            endpoint: {
                state: sum(float(values[self.physical_index(endpoint, state, limb)])
                           for limb in range(self.limbs))
                for state in self.manifest.states
            }
            for endpoint in self.endpoints
        }


__all__ = [
    "ComponentChemistry", "ChemistryDeploymentManifest",
    "EngineSpeciesExchangeLayout", "ExternalStateBinding", "deployment_manifest",
]
