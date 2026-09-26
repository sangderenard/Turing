"""Chemical identity, reduction, and coupled SymPy residuals.

This module is the data-to-law boundary for the chamber chemistry.  A master
compendium may contain far more species and reactions than one simulation can
reach.  :meth:`ChemicalCompendium.reduce` selects the element/phase/reaction
closure for a configured chamber and removes linearly dependent equilibrium
constraints before any compiler sees the program.

The reduced object authors one implicit residual over reaction extents and
temperature.  Amounts are moles of explicit ``(species, phase)`` states.
Molarity, molality, ionic strength, Davies activities, gas fugacity, reaction
heat, and phase heat all derive from those same amounts.

This file does not run PHREEQC and does not replace its database.  The initial
rows below are a small, provenance-bearing atmospheric/aqueous nucleus copied
from the repository's existing phase table and the official USGS
``phreeqc.dat``.  The loader for the full upstream compendium is subsequent
work; callers can already construct a larger :class:`ChemicalCompendium`
without changing the symbolic law.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import sympy as sp

from src.compiler.abstract_ui_vehicles import extra_precision_closure

from .state import CHEMISTRY_PRECISION_LIMBS, ChemistryPieceLayout


R_GAS = sp.Rational("8.31446261815324")
P_STANDARD_PA = sp.Integer(101325)
T_REFERENCE_K = sp.Rational("298.15")
MOLAL_STANDARD = sp.Integer(1)
EPS = sp.Rational(1, 10 ** 30)
GAS = "gas"
AQUEOUS = "aqueous"
LIQUID = "liquid"
SOLID = "solid"
ADSORBED = "adsorbed"
PLASMA = "plasma"
PHASES = (GAS, AQUEOUS, LIQUID, SOLID, ADSORBED, PLASMA)


def state_key(species: str, phase: str) -> str:
    return f"{species}@{phase}"


@dataclass(frozen=True)
class Element:
    symbol: str
    atomic_number: int
    atomic_weight_g_mol: float
    source: str


@dataclass(frozen=True)
class ChemicalSpecies:
    identity: str
    formula: tuple[tuple[str, sp.Expr], ...]
    charge: int
    molar_mass_kg_mol: float
    allowed_phases: tuple[str, ...]
    source: str
    aqueous_role: str = "solute"  # solute | solvent

    @classmethod
    def make(cls, identity: str, formula: Mapping[str, int], charge: int,
             molar_mass_kg_mol: float, allowed_phases: Sequence[str],
             source: str, aqueous_role: str = "solute") -> "ChemicalSpecies":
        exact_formula = tuple(sorted(
            (str(element), _exact(count))
            for element, count in formula.items() if _exact(count) != 0
        ))
        return cls(identity, exact_formula,
                   int(charge), float(molar_mass_kg_mol), tuple(allowed_phases),
                   str(source), str(aqueous_role))

    @property
    def atoms(self) -> dict[str, sp.Expr]:
        return dict(self.formula)


@dataclass(frozen=True)
class ChemicalState:
    species: str
    phase: str
    source: str

    @property
    def identity(self) -> str:
        return state_key(self.species, self.phase)


@dataclass(frozen=True)
class EquilibriumLaw:
    """Van't Hoff log-K law around a cited reference temperature."""

    log10_k_ref: object
    delta_h_j_mol: object
    reference_temperature_k: object = "298.15"

    def __post_init__(self):
        object.__setattr__(self, "log10_k_ref", _exact(self.log10_k_ref))
        object.__setattr__(self, "delta_h_j_mol", _exact(self.delta_h_j_mol))
        object.__setattr__(self, "reference_temperature_k",
                           _exact(self.reference_temperature_k))

    def ln_k(self, temperature):
        t = sp.sympify(temperature)
        tr = self.reference_temperature_k
        dh = self.delta_h_j_mol
        return (sp.log(10) * self.log10_k_ref
                - dh / R_GAS * (1 / t - 1 / tr))


@dataclass(frozen=True)
class TransitionEquilibrium:
    """Constant-latent phase equilibrium anchored where K equals one."""

    transition_temperature_k: object
    delta_h_j_mol: object

    def __post_init__(self):
        object.__setattr__(self, "transition_temperature_k",
                           _exact(self.transition_temperature_k))
        object.__setattr__(self, "delta_h_j_mol", _exact(self.delta_h_j_mol))

    def ln_k(self, temperature):
        t = sp.sympify(temperature)
        tr = self.transition_temperature_k
        dh = self.delta_h_j_mol
        return -dh / R_GAS * (1 / t - 1 / tr)


@dataclass(frozen=True)
class KineticLaw:
    forward_prefactor: object
    reverse_prefactor: object
    forward_activation_j_mol: object = 0
    reverse_activation_j_mol: object = 0

    def __post_init__(self):
        for name in ("forward_prefactor", "reverse_prefactor",
                     "forward_activation_j_mol", "reverse_activation_j_mol"):
            object.__setattr__(self, name, _exact(getattr(self, name)))


@dataclass(frozen=True)
class Reaction:
    identity: str
    stoichiometry: tuple[tuple[str, sp.Expr], ...]
    law: EquilibriumLaw | TransitionEquilibrium | KineticLaw
    family: str
    source: str

    @classmethod
    def make(cls, identity: str, stoichiometry: Mapping[str, int], law,
             family: str, source: str) -> "Reaction":
        rows = tuple(sorted(
            (str(state), _exact(coefficient))
            for state, coefficient in stoichiometry.items()
            if _exact(coefficient) != 0
        ))
        return cls(str(identity), rows, law, str(family), str(source))

    @property
    def nu(self) -> dict[str, int]:
        return dict(self.stoichiometry)

    @property
    def equilibrium(self) -> bool:
        return not isinstance(self.law, KineticLaw)

    @property
    def delta_h_j_mol(self) -> sp.Expr:
        return self.law.delta_h_j_mol if hasattr(self.law, "delta_h_j_mol") else sp.Integer(0)


@dataclass(frozen=True)
class ReductionReceipt:
    seed_states: tuple[str, ...]
    active_elements: tuple[str, ...]
    retained_states: tuple[str, ...]
    retained_reactions: tuple[str, ...]
    dependent_equilibria: tuple[str, ...]


class ChemicalCompendium:
    def __init__(self, *, elements: Iterable[Element],
                 species: Iterable[ChemicalSpecies], states: Iterable[ChemicalState],
                 reactions: Iterable[Reaction]):
        self.elements = _unique(elements, lambda x: x.symbol, "element")
        self.species = _unique(species, lambda x: x.identity, "species")
        self.states = _unique(states, lambda x: x.identity, "chemical state")
        self.reactions = _unique(reactions, lambda x: x.identity, "reaction")
        self._validate()

    def _validate(self) -> None:
        for state in self.states.values():
            if state.species not in self.species:
                raise ValueError(f"{state.identity}: unknown species {state.species}")
            if state.phase not in PHASES:
                raise ValueError(f"{state.identity}: unknown phase {state.phase}")
            if state.phase not in self.species[state.species].allowed_phases:
                raise ValueError(f"{state.identity}: phase is not allowed by its species")
        for reaction in self.reactions.values():
            unknown = sorted(set(reaction.nu) - set(self.states))
            if unknown:
                raise ValueError(f"{reaction.identity}: unknown chemical states {unknown}")
            element_balance, charge_balance = self.reaction_balance(reaction)
            bad_elements = {k: v for k, v in element_balance.items() if v}
            if bad_elements or charge_balance:
                raise ValueError(
                    f"{reaction.identity}: unbalanced elements={bad_elements}, "
                    f"charge={charge_balance}")

    def reaction_balance(self, reaction: Reaction) -> tuple[dict[str, int], int]:
        element_balance = {symbol: 0 for symbol in self.elements}
        charge = 0
        for key, coefficient in reaction.stoichiometry:
            state = self.states[key]
            species = self.species[state.species]
            for element, count in species.formula:
                if element not in element_balance:
                    raise ValueError(f"{reaction.identity}: unknown element {element}")
                element_balance[element] += coefficient * count
            charge += coefficient * species.charge
        return element_balance, charge

    def reduce(self, seed_states: Iterable[str], *,
               allowed_phases: Iterable[str] = PHASES,
               families: Iterable[str] | None = None) -> "ReducedCompendium":
        seeds = tuple(dict.fromkeys(map(str, seed_states)))
        unknown = sorted(set(seeds) - set(self.states))
        if unknown:
            raise KeyError(f"unknown seed chemical states: {unknown}")
        phases = set(allowed_phases)
        family_set = None if families is None else set(families)
        active_elements = {
            element
            for key in seeds
            for element, _count in self.species[self.states[key].species].formula
        }
        selected_states = {
            key for key, state in self.states.items()
            if state.phase in phases
            and set(self.species[state.species].atoms).issubset(active_elements)
        }
        selected_states.update(seeds)
        reactions = [
            reaction for reaction in self.reactions.values()
            if set(reaction.nu).issubset(selected_states)
            and (family_set is None or reaction.family in family_set)
        ]
        state_order = tuple(key for key in self.states if key in selected_states)
        retained, dependent = _independent_reactions(reactions, state_order)
        receipt = ReductionReceipt(
            seeds, tuple(sorted(active_elements)), state_order,
            tuple(r.identity for r in retained), tuple(r.identity for r in dependent))
        return ReducedCompendium(self, state_order, tuple(retained), receipt)

    def reduce_reachable(self, seed_states: Iterable[str], *,
                         allowed_phases: Iterable[str] = PHASES,
                         families: Iterable[str] | None = None
                         ) -> "ReducedCompendium":
        """Close exact component states over possible reaction directions.

        A reaction becomes reachable only when every reactant on one enabled
        direction is already reachable. It then contributes every state in
        that reaction. This is the deployment allocation closure: it reserves
        every possible transformation without allocating unrelated compounds
        merely because they contain the same elements.
        """
        seeds = tuple(dict.fromkeys(map(str, seed_states)))
        unknown = sorted(set(seeds) - set(self.states))
        if unknown:
            raise KeyError(f"unknown seed chemical states: {unknown}")
        phases = set(map(str, allowed_phases))
        bad_seed_phases = [key for key in seeds if self.states[key].phase not in phases]
        if bad_seed_phases:
            raise ValueError(
                f"seed states excluded by allowed phases: {bad_seed_phases}")
        family_set = None if families is None else set(map(str, families))
        candidates = tuple(
            reaction for reaction in self.reactions.values()
            if (family_set is None or reaction.family in family_set)
            and all(self.states[key].phase in phases for key in reaction.nu)
        )
        reachable = set(seeds)
        activated: set[str] = set()
        changed = True
        while changed:
            changed = False
            for reaction in candidates:
                reactants = {key for key, coefficient in reaction.stoichiometry
                             if coefficient < 0}
                products = {key for key, coefficient in reaction.stoichiometry
                            if coefficient > 0}
                can_run = (
                    _direction_enabled(reaction, reverse=False)
                    and reactants.issubset(reachable)
                ) or (
                    _direction_enabled(reaction, reverse=True)
                    and products.issubset(reachable)
                )
                if not can_run:
                    continue
                activated.add(reaction.identity)
                before = len(reachable)
                reachable.update(reaction.nu)
                changed |= len(reachable) != before

        state_order = tuple(key for key in self.states if key in reachable)
        reactions = tuple(
            reaction for reaction in candidates
            if reaction.identity in activated and set(reaction.nu).issubset(reachable)
        )
        retained, dependent = _independent_reactions(reactions, state_order)
        active_elements = {
            element
            for key in state_order
            for element in self.species[self.states[key].species].atoms
        }
        receipt = ReductionReceipt(
            seeds, tuple(sorted(active_elements)), state_order,
            tuple(reaction.identity for reaction in retained),
            tuple(reaction.identity for reaction in dependent))
        return ReducedCompendium(self, state_order, tuple(retained), receipt)


@dataclass(frozen=True)
class SymbolicOwnerSystem:
    owner: str
    amount_now: Mapping[str, sp.Expr]
    amount_next: Mapping[str, sp.Expr]
    extents: tuple[sp.Symbol, ...]
    temperature_next: sp.Symbol
    molarity_mol_l: Mapping[str, sp.Expr]
    molality_mol_kg: Mapping[str, sp.Expr]
    ionic_strength_mol_kg: sp.Expr
    activities: Mapping[str, sp.Expr]
    residuals: tuple[sp.Expr, ...]
    jacobian: sp.Matrix
    element_totals_now: Mapping[str, sp.Expr]
    element_totals_next: Mapping[str, sp.Expr]
    charge_now_mol: sp.Expr
    charge_next_mol: sp.Expr


@dataclass(frozen=True)
class ReducedCompendium:
    master: ChemicalCompendium
    states: tuple[str, ...]
    reactions: tuple[Reaction, ...]
    receipt: ReductionReceipt

    def symbolic_owner_system(self, owner: str, *, phase_volumes_m3=None,
                              solvent_kg=None, include_energy: bool = True
                              ) -> SymbolicOwnerSystem:
        """Author one closed-owner implicit chemistry system.

        Transport is an input delta for every chemical state, so a later
        multi-owner assembly can compute paired edge fluxes once and substitute
        them here.  Reaction extents and (optionally) temperature are the
        nonlinear unknowns.  All returned expressions are ordinary SymPy and
        can enter the repository's source/SSA path after the active closure is
        fixed.
        """
        clean = _symbol_token(owner)
        t_now = sp.Symbol(f"{clean}__T_now", positive=True)
        t_next = sp.Symbol(f"{clean}__T_next", positive=True)
        dt = sp.Symbol("dt", positive=True)
        supplied_volumes = dict(phase_volumes_m3 or {})
        phase_volumes = {
            phase: (sp.sympify(supplied_volumes[phase]) if phase in supplied_volumes
                    else sp.Symbol(f"{clean}__volume_{phase}_m3", positive=True))
            for phase in PHASES
        }
        solvent = sp.Symbol(f"{clean}__solvent_kg", positive=True) if solvent_kg is None else sp.sympify(solvent_kg)
        amount_now = {key: sp.Symbol(f"{clean}__n__{_symbol_token(key)}", nonnegative=True)
                      for key in self.states}
        transport = {key: sp.Symbol(f"{clean}__dn_transport__{_symbol_token(key)}")
                     for key in self.states}
        extents = tuple(sp.Symbol(f"{clean}__xi__{_symbol_token(r.identity)}", real=True)
                        for r in self.reactions)
        amount_next = {}
        for key in self.states:
            reaction_delta = sum(
                sp.Integer(reaction.nu.get(key, 0)) * extent
                for reaction, extent in zip(self.reactions, extents)
            )
            amount_next[key] = amount_now[key] + transport[key] + reaction_delta

        aqueous = [key for key in self.states if self.master.states[key].phase == AQUEOUS]
        molality = {key: amount_next[key] / (solvent + EPS) for key in aqueous}
        molarity = {key: amount_next[key] / (
                        sp.Integer(1000) * phase_volumes[AQUEOUS] + EPS)
                    for key in aqueous}
        ionic_strength = sp.Rational(1, 2) * sum(
            molality[key] * self.master.species[self.master.states[key].species].charge ** 2
            for key in aqueous
        )
        sqrt_i = sp.sqrt(ionic_strength + EPS)
        davies_a = sp.Rational("0.509")
        activities: dict[str, sp.Expr] = {}
        liquid_totals = {
            phase: sum(amount_next[key] for key in self.states
                       if self.master.states[key].phase == phase)
            for phase in (AQUEOUS, LIQUID)
        }
        for key in self.states:
            state = self.master.states[key]
            species = self.master.species[state.species]
            if state.phase == GAS:
                activities[key] = (amount_next[key] * R_GAS * t_next
                                   / (phase_volumes[GAS] * P_STANDARD_PA + EPS))
            elif state.phase == AQUEOUS:
                if species.aqueous_role == "solvent":
                    activities[key] = amount_next[key] / (liquid_totals[AQUEOUS] + EPS)
                else:
                    z = sp.Integer(species.charge)
                    log10_gamma = -davies_a * z ** 2 * (
                        sqrt_i / (1 + sqrt_i) - sp.Rational("0.3") * ionic_strength)
                    activities[key] = sp.Pow(10, log10_gamma) * molality[key] / MOLAL_STANDARD
            elif state.phase == LIQUID:
                activities[key] = amount_next[key] / (liquid_totals[LIQUID] + EPS)
            else:
                # Unit activity is the pure condensed-phase convention.  Phase
                # appearance/disappearance is an active-set/complementarity
                # decision made by the combined owner assembler.
                activities[key] = sp.Integer(1)

        residuals: list[sp.Expr] = []
        for reaction, extent in zip(self.reactions, extents):
            if reaction.equilibrium:
                ln_q = sum(sp.Integer(coefficient) * sp.log(activities[key] + EPS)
                           for key, coefficient in reaction.stoichiometry)
                residuals.append(ln_q - reaction.law.ln_k(t_next))
            else:
                law = reaction.law
                kf = law.forward_prefactor * sp.exp(
                    -law.forward_activation_j_mol / (R_GAS * t_next))
                kr = law.reverse_prefactor * sp.exp(
                    -law.reverse_activation_j_mol / (R_GAS * t_next))
                forward = sp.Integer(1)
                reverse = sp.Integer(1)
                for key, coefficient in reaction.stoichiometry:
                    if coefficient < 0:
                        forward *= activities[key] ** (-coefficient)
                    elif coefficient > 0:
                        reverse *= activities[key] ** coefficient
                residuals.append(extent - dt * (kf * forward - kr * reverse))

        unknowns: list[sp.Expr] = list(extents)
        if include_energy:
            heat_capacity = sp.Symbol(f"{clean}__heat_capacity_j_k", positive=True)
            heat_external = sp.Symbol(f"{clean}__heat_external_j", real=True)
            reaction_heat = sum(r.delta_h_j_mol * x
                                for r, x in zip(self.reactions, extents))
            residuals.append(heat_capacity * (t_next - t_now) + reaction_heat - heat_external)
            unknowns.append(t_next)
        else:
            residuals = [expr.subs(t_next, t_now) for expr in residuals]

        element_now = self._element_totals(amount_now)
        element_next = self._element_totals(amount_next)
        charge_now = self._charge_total(amount_now)
        charge_next = self._charge_total(amount_next)
        matrix = sp.Matrix(residuals)
        jacobian = matrix.jacobian(unknowns) if unknowns else sp.zeros(0, 0)
        return SymbolicOwnerSystem(
            owner, amount_now, amount_next, extents, t_next,
            molarity, molality, ionic_strength, activities,
            tuple(residuals), jacobian, element_now, element_next,
            charge_now, charge_next)

    def _element_totals(self, amounts: Mapping[str, sp.Expr]) -> dict[str, sp.Expr]:
        return {
            element: sum(
                amounts[key] * self.master.species[self.master.states[key].species].atoms.get(element, 0)
                for key in self.states
            )
            for element in self.receipt.active_elements
        }

    def _charge_total(self, amounts: Mapping[str, sp.Expr]) -> sp.Expr:
        return sum(
            amounts[key] * self.master.species[self.master.states[key].species].charge
            for key in self.states
        )

    def first_order_transition_piece_spec(
        self, reaction: str, *, cells: int,
    ) -> tuple[object, "ChemistryPieceLayout"]:
        """Author the first compiled chemistry test case as one span kernel.

        This is intentionally a narrow member of the eventual nonlinear
        chemistry family: one reversible first-order ``A <-> B`` transition
        with constant rate coefficients.  Its implicit extent has a closed
        form, so the test exercises the final packed inventory ABI and the
        precision-region compiler without introducing a second numerical
        solver beside the future bounded Newton solve.

        Anything outside that exact case refuses.  In particular, zero
        activation energies are required because temperature-dependent
        Arrhenius and equilibrium rows belong to the coupled temperature/
        composition Newton system, not to this seam test.
        """

        from src.compiler.kernel_bank import KernelSpec

        if int(cells) <= 0:
            raise ValueError("chemistry piece requires at least one cell")
        selected = next(
            (row for row in self.reactions if row.identity == str(reaction)),
            None,
        )
        if selected is None:
            raise KeyError(f"reaction {reaction!r} is not in the reduced system")
        if not isinstance(selected.law, KineticLaw):
            raise ValueError(
                f"{selected.identity}: first-order piece requires a kinetic law"
            )
        negative = [(key, value) for key, value in selected.stoichiometry if value < 0]
        positive = [(key, value) for key, value in selected.stoichiometry if value > 0]
        if (
            len(negative) != 1 or len(positive) != 1
            or negative[0][1] != -1 or positive[0][1] != 1
        ):
            raise ValueError(
                f"{selected.identity}: first-order piece requires exactly "
                "one -1 reactant and one +1 product"
            )
        if (
            selected.law.forward_activation_j_mol != 0
            or selected.law.reverse_activation_j_mol != 0
        ):
            raise ValueError(
                f"{selected.identity}: temperature-dependent rates require "
                "the coupled Newton chemistry piece"
            )

        layout = ChemistryPieceLayout(
            states=self.states,
            cells=int(cells),
            limbs=CHEMISTRY_PRECISION_LIMBS,
        )
        reactant = self.states.index(negative[0][0])
        product = self.states.index(positive[0][0])
        state_count = len(self.states)
        function_name = "chemistry_first_order_transition"
        copy_lines = [
            f"        inventory_next[base + {index}] = inventory[base + {index}]"
            for index in range(state_count)
        ]
        source = "\n".join((
            "",
            f"def {function_name}(inventory: Precision[2], dt: Precision[2], "
            "k_forward: Precision[2], k_reverse: Precision[2], "
            "inventory_next: Precision[2], cells):",
            "    for cell in range(cells):",
            f"        base = cell * {state_count}",
            *copy_lines,
            f"        reactant = inventory[base + {reactant}]",
            f"        product = inventory[base + {product}]",
            "        forward = k_forward[cell]",
            "        reverse = k_reverse[cell]",
            "        extent = dt[cell] * (forward * reactant - reverse * product) / (1.0 + dt[cell] * (forward + reverse))",
            f"        inventory_next[base + {reactant}] = reactant - extent",
            f"        inventory_next[base + {product}] = product + extent",
            "    return inventory_next",
            "",
        ))

        def example_inputs(sizes, rng):
            count = int(sizes["cells"])
            logical = count * state_count
            inventory = _wide_zeros(logical, CHEMISTRY_PRECISION_LIMBS)
            for cell in range(count):
                inventory[(cell * state_count + reactant) * 2] = rng.uniform(0.5, 2.0)
                inventory[(cell * state_count + product) * 2] = rng.uniform(0.05, 0.4)
            return {
                "inventory": inventory,
                "dt": _wide_constant(rng.uniform(1.0e-4, 2.0e-2, count), 2),
                "k_forward": _wide_constant(rng.uniform(0.1, 0.8, count), 2),
                "k_reverse": _wide_constant(rng.uniform(0.05, 0.3, count), 2),
                "inventory_next": _wide_zeros(logical, 2),
                "cells": count,
            }

        def reference(inventory, dt, k_forward, k_reverse, inventory_next, cells):
            inventory_next[...] = inventory
            for cell in range(int(cells)):
                base = cell * state_count
                left = _wide_value(inventory, base + reactant, 2)
                right = _wide_value(inventory, base + product, 2)
                forward = _wide_value(k_forward, cell, 2)
                reverse = _wide_value(k_reverse, cell, 2)
                step_dt = _wide_value(dt, cell, 2)
                extent = step_dt * (forward * left - reverse * right)
                extent /= 1.0 + step_dt * (forward + reverse)
                _wide_put(inventory_next, base + reactant, left - extent, 2)
                _wide_put(inventory_next, base + product, right + extent, 2)
            return inventory_next

        spec = KernelSpec(
            name=f"{function_name}_{_symbol_token(selected.identity)}_{state_count}",
            source=source,
            function_name=function_name,
            reference=reference,
            parameter_order=(
                "inventory", "dt", "k_forward", "k_reverse",
                "inventory_next", "cells",
            ),
            size_parameters=("cells",),
            example_inputs=example_inputs,
            # Wide buffers carry an extra physical limb dimension which the
            # current extent vocabulary cannot express.  This is the same
            # explicit refusal used by the compiled signal kernels.
            extents=None,
            limb_width=CHEMISTRY_PRECISION_LIMBS,
        )
        return spec, layout

    def compile_first_order_transition_piece(
        self, reaction: str, *, cells: int, directory: str | Path,
    ):
        """Compile the supported transition into a real :class:`LLVMPiece`."""

        from src.compiler.kernel_bank import KernelBank
        from src.compiler.native_law_kernels import LLVMPiece

        spec, layout = self.first_order_transition_piece_spec(
            reaction, cells=cells,
        )
        variant = KernelBank(Path(directory), {spec.name: spec}).get(
            spec.name, specialized={"cells": int(cells)}, backend="llvm",
        )
        entry = next(
            name for name in variant.module.functions
            if name.endswith(f"__{spec.function_name}")
        )
        # Admission and profiling execute the artifact, which warms its
        # ctypes entry cache.  An LLVMPiece is a persistent object and the
        # cached foreign-function pointer is process-local and unpicklable;
        # the artifact will lazily reopen the same library on its next call.
        variant.native._entry = None
        arguments = tuple(
            name for name in spec.parameter_order if name != "cells"
        )
        output_id = int(variant.id_by_name["inventory_next"])
        piece = LLVMPiece(
            artifact=variant.native,
            argument_names=arguments,
            argument_ids=tuple(int(variant.id_by_name[name]) for name in arguments),
            output_names=("inventory_next",),
            output_ids={"inventory_next": output_id},
            batch=int(cells),
            module=variant.module,
            entry=entry,
            outputs=variant.outputs,
            source=spec.source,
        )
        return piece, layout


def _unique(rows, key, kind):
    out = {}
    for row in rows:
        identity = key(row)
        if identity in out:
            raise ValueError(f"duplicate {kind} identity {identity!r}")
        out[identity] = row
    return out


def _symbol_token(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(value))


def _exact(value) -> sp.Expr:
    """Keep authored decimal data exact until the numerical backend casts it."""
    return value if isinstance(value, sp.Basic) else sp.Rational(str(value))


def _wide_zeros(logical_size: int, limbs: int):
    import numpy as np

    return np.zeros(int(logical_size) * int(limbs), dtype=np.float64)


def _wide_constant(values, limbs: int):
    import numpy as np

    source = np.asarray(values, dtype=np.float64).reshape(-1)
    result = _wide_zeros(len(source), limbs)
    result[0::int(limbs)] = source
    return result


def _wide_value(values, logical_index: int, limbs: int) -> float:
    start = int(logical_index) * int(limbs)
    return sum(float(values[start + limb]) for limb in range(int(limbs)))


def _wide_put(values, logical_index: int, value: float, limbs: int) -> None:
    start = int(logical_index) * int(limbs)
    values[start] = float(value)
    for limb in range(1, int(limbs)):
        values[start + limb] = 0.0


def precision2_chemistry_law(function):
    """Run a reduced numerical chemistry law at AbstractTensor precision 2.

    ``function`` is the AbstractTensor-stage residual/update callable produced
    from the reduced SymPy law.  The repository's existing precision boundary
    promotes every numerical input with ``Precision.of(value, 2)`` and keeps
    the expansion through supported arithmetic and transcendental operations.
    It collapses once, at the returned publication boundary.
    """

    return extra_precision_closure(function, limbs=CHEMISTRY_PRECISION_LIMBS)


def _direction_enabled(reaction: Reaction, *, reverse: bool) -> bool:
    if not isinstance(reaction.law, KineticLaw):
        return True
    prefactor = (reaction.law.reverse_prefactor if reverse
                 else reaction.law.forward_prefactor)
    return sp.simplify(prefactor) != 0


def _independent_reactions(reactions: Sequence[Reaction], states: Sequence[str]):
    retained: list[Reaction] = []
    dependent: list[Reaction] = []
    equilibrium_rows: list[list[int]] = []
    equilibrium_reactions: list[Reaction] = []
    rank = 0
    for reaction in reactions:
        if not reaction.equilibrium:
            retained.append(reaction)
            continue
        row = [reaction.nu.get(key, 0) for key in states]
        candidate = equilibrium_rows + [row]
        candidate_rank = sp.Matrix(candidate).rank()
        if candidate_rank > rank:
            equilibrium_rows.append(row)
            equilibrium_reactions.append(reaction)
            retained.append(reaction)
            rank = candidate_rank
        else:
            # A dependent stoichiometric row is redundant only when its
            # thermodynamics obey the same linear relation.  Silently dropping
            # an inconsistent log-K row would turn contradictory source data
            # into an apparently valid faster system.
            basis = sp.Matrix(equilibrium_rows)
            target = sp.Matrix([row])
            coefficients = list(sp.linsolve(
                (basis.T, target.T), *sp.symbols(f"c0:{len(equilibrium_rows)}")))
            if not coefficients:
                raise ValueError(f"cannot resolve dependent reaction {reaction.identity}")
            coeff = coefficients[0]
            if any(value.free_symbols for value in coeff):
                # Choose the zero value for any free basis coefficient.  Every
                # valid choice represents the same stoichiometric row; the
                # thermodynamic consistency check below remains mandatory.
                free = set().union(*(value.free_symbols for value in coeff))
                coeff = tuple(value.subs({symbol: 0 for symbol in free}) for value in coeff)
            temperature = sp.Symbol("T_consistency", positive=True)
            expected = sum(value * prior.law.ln_k(temperature)
                           for value, prior in zip(coeff, equilibrium_reactions))
            mismatch = sp.simplify(reaction.law.ln_k(temperature) - expected)
            if mismatch != 0:
                raise ValueError(
                    f"dependent equilibrium {reaction.identity} has inconsistent log-K")
            dependent.append(reaction)
    return retained, dependent


def atmospheric_aqueous_nucleus() -> ChemicalCompendium:
    """Initial authoritative nucleus; intended to grow through database import."""
    usgs = ("USGS PHREEQC phreeqc.dat, usgs-coupled-subtrees/"
            "phreeqc3-database master, retrieved 2026-09-20")
    phase = "engine_toy/phase_table.py existing repository authority"
    elements = (
        Element("H", 1, 1.008, usgs), Element("C", 6, 12.0111, usgs),
        Element("N", 7, 14.0067, usgs), Element("O", 8, 16.0, usgs),
        Element("Na", 11, 22.9898, usgs), Element("Cl", 17, 35.453, usgs),
        Element("Ar", 18, 39.948, "engine_toy/air_separation.py"),
    )
    species = (
        ChemicalSpecies.make("H2O", {"H": 2, "O": 1}, 0, 0.01801528,
                             (GAS, AQUEOUS, SOLID), phase, "solvent"),
        ChemicalSpecies.make("H+", {"H": 1}, 1, 0.001008, (AQUEOUS,), usgs),
        ChemicalSpecies.make("OH-", {"O": 1, "H": 1}, -1, 0.017007,
                             (AQUEOUS,), usgs),
        ChemicalSpecies.make("Na+", {"Na": 1}, 1, 0.0229898, (AQUEOUS,), usgs),
        ChemicalSpecies.make("Cl-", {"Cl": 1}, -1, 0.035453, (AQUEOUS,), usgs),
        ChemicalSpecies.make("NaCl", {"Na": 1, "Cl": 1}, 0, 0.0584428,
                             (SOLID,), usgs),
        ChemicalSpecies.make("CO2", {"C": 1, "O": 2}, 0, 0.0440095,
                             (GAS, AQUEOUS, SOLID), phase),
        ChemicalSpecies.make("HCO3-", {"H": 1, "C": 1, "O": 3}, -1,
                             0.0610168, (AQUEOUS,), usgs),
        ChemicalSpecies.make("CO3-2", {"C": 1, "O": 3}, -2, 0.0600089,
                             (AQUEOUS,), usgs),
        ChemicalSpecies.make("N2", {"N": 2}, 0, 0.0280134,
                             (GAS, LIQUID, SOLID), phase),
        ChemicalSpecies.make("O2", {"O": 2}, 0, 0.0319988,
                             (GAS, LIQUID, SOLID), phase),
        ChemicalSpecies.make("Ar", {"Ar": 1}, 0, 0.039948,
                             (GAS, LIQUID), "engine_toy/air_separation.py"),
    )
    states = tuple(
        ChemicalState(spec.identity, phase_name, spec.source)
        for spec in species for phase_name in spec.allowed_phases
    )
    aq = lambda name: state_key(name, AQUEOUS)
    reactions = (
        Reaction.make("water-autoionization",
                      {aq("H2O"): -1, aq("H+"): 1, aq("OH-"): 1},
                      EquilibriumLaw("-14", "56400"), "aqueous-acid-base", usgs),
        Reaction.make("carbonate-to-bicarbonate",
                      {aq("CO3-2"): -1, aq("H+"): -1, aq("HCO3-"): 1},
                      EquilibriumLaw("10.329", sp.Rational("-3.561") * 4184),
                      "aqueous-acid-base", usgs),
        Reaction.make("carbonate-to-carbon-dioxide",
                      {aq("CO3-2"): -1, aq("H+"): -2,
                       aq("CO2"): 1, aq("H2O"): 1},
                      EquilibriumLaw("16.681", sp.Rational("-5.738") * 4184),
                      "aqueous-acid-base", usgs),
        Reaction.make("halite-dissolution",
                      {state_key("NaCl", SOLID): -1,
                       aq("Na+"): 1, aq("Cl-"): 1},
                      EquilibriumLaw("1.57", sp.Rational("1.37") * 4184),
                      "aqueous-mineral", usgs),
        Reaction.make("water-vaporization",
                      {aq("H2O"): -1, state_key("H2O", GAS): 1},
                      TransitionEquilibrium("373.15", 2_256_000 * sp.Rational("0.01801528")),
                      "phase", phase),
        Reaction.make("water-melting",
                      {state_key("H2O", SOLID): -1, aq("H2O"): 1},
                      TransitionEquilibrium("273.15", 333_550 * sp.Rational("0.01801528")),
                      "phase", phase),
        Reaction.make("carbon-dioxide-sublimation",
                      {state_key("CO2", SOLID): -1, state_key("CO2", GAS): 1},
                      TransitionEquilibrium("194.65", 571_000 * sp.Rational("0.0440095")),
                      "phase", phase),
        Reaction.make("nitrogen-vaporization",
                      {state_key("N2", LIQUID): -1, state_key("N2", GAS): 1},
                      TransitionEquilibrium("77.36", 199_000 * sp.Rational("0.0280134")),
                      "phase", phase),
        Reaction.make("nitrogen-melting",
                      {state_key("N2", SOLID): -1, state_key("N2", LIQUID): 1},
                      TransitionEquilibrium("63.15", 25_700 * sp.Rational("0.0280134")),
                      "phase", phase),
        Reaction.make("oxygen-vaporization",
                      {state_key("O2", LIQUID): -1, state_key("O2", GAS): 1},
                      TransitionEquilibrium("90.19", 213_000 * sp.Rational("0.0319988")),
                      "phase", phase),
        Reaction.make("oxygen-melting",
                      {state_key("O2", SOLID): -1, state_key("O2", LIQUID): 1},
                      TransitionEquilibrium("54.36", 13_900 * sp.Rational("0.0319988")),
                      "phase", phase),
    )
    return ChemicalCompendium(elements=elements, species=species,
                              states=states, reactions=reactions)


__all__ = [
    "ADSORBED", "AQUEOUS", "GAS", "LIQUID", "PHASES", "PLASMA", "SOLID",
    "CHEMISTRY_PRECISION_LIMBS", "ChemicalCompendium", "ChemicalSpecies", "ChemicalState",
    "ChemistryPieceLayout", "Element",
    "EquilibriumLaw", "KineticLaw", "Reaction", "ReducedCompendium",
    "ReductionReceipt", "SymbolicOwnerSystem", "TransitionEquilibrium",
    "atmospheric_aqueous_nucleus", "precision2_chemistry_law", "state_key",
]
