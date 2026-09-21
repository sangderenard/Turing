from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from src.common.tensors.abstraction import AbstractTensor

from src.common.chemistry import (
    AQUEOUS,
    GAS,
    SOLID,
    ChemicalCompendium,
    ChemistryScale,
    ChemicalSpecies,
    ChemicalState,
    CHEMISTRY_PRECISION_LIMBS,
    Element,
    EquilibriumLaw,
    KineticLaw,
    Reaction,
    atmospheric_aqueous_nucleus,
    precision2_chemistry_law,
    state_key,
)
from src.compiler.native_law_kernels import LLVMPiece


def test_numerical_chemistry_uses_repository_abstract_tensor_precision_two():
    seen = {}

    def cancellation_sensitive(value):
        seen["limbs"] = value.limbs
        return (value + 1.0e16) - 1.0e16

    numerical_law = precision2_chemistry_law(cancellation_sensitive)
    result = numerical_law(AbstractTensor.get_tensor([1.0]))

    assert CHEMISTRY_PRECISION_LIMBS == 2
    assert seen["limbs"] == 2
    assert result.tolist() == [1.0]


def test_chemistry_scaling_uses_dimensionless_wide_abstract_tensor_coordinates():
    scale = ChemistryScale.of(
        variable=[1.0e-20, 1.0e6],
        residual=[1.0e-30, 1.0e3],
        limbs=2,
    )

    variables = scale.normalize_variables([2.0e-20, 3.0e6])
    residuals = scale.normalize_residuals([5.0e-30, 7.0e3])
    restored = scale.physical_variables(variables)

    assert variables.limbs == residuals.limbs == restored.limbs == 2
    assert variables.collapse().tolist() == pytest.approx([2.0, 3.0])
    assert residuals.collapse().tolist() == pytest.approx([5.0, 7.0])
    assert restored.collapse().tolist() == pytest.approx([2.0e-20, 3.0e6])


def test_chemistry_scaling_rejects_nonpositive_characteristic_magnitudes():
    with pytest.raises(ValueError, match="positive"):
        ChemistryScale.of(variable=[1.0, 0.0], residual=[1.0, 1.0])


def test_master_nucleus_reactions_conserve_elements_and_charge():
    chemistry = atmospheric_aqueous_nucleus()
    assert len(chemistry.reactions) == 11
    for reaction in chemistry.reactions.values():
        elements, charge = chemistry.reaction_balance(reaction)
        assert all(value == 0 for value in elements.values()), reaction.identity
        assert charge == 0, reaction.identity


def test_unbalanced_reaction_is_rejected_before_symbolic_reduction():
    source = "test"
    h = Element("H", 1, 1.008, source)
    proton = ChemicalSpecies.make("H+", {"H": 1}, 1, 0.001008,
                                  (AQUEOUS,), source)
    state = ChemicalState("H+", AQUEOUS, source)
    bad = Reaction.make("charge-from-nothing", {state.identity: 1},
                        EquilibriumLaw(0.0, 0.0), "test", source)
    with pytest.raises(ValueError, match="unbalanced"):
        ChemicalCompendium(elements=(h,), species=(proton,), states=(state,),
                           reactions=(bad,))


def test_reduction_selects_element_phase_closure_in_stable_master_order():
    chemistry = atmospheric_aqueous_nucleus()
    reduced = chemistry.reduce(
        (state_key("H2O", AQUEOUS), state_key("NaCl", SOLID)),
        allowed_phases=(AQUEOUS, SOLID),
    )
    assert reduced.receipt.active_elements == ("Cl", "H", "Na", "O")
    assert state_key("Na+", AQUEOUS) in reduced.states
    assert state_key("Cl-", AQUEOUS) in reduced.states
    assert state_key("N2", AQUEOUS) not in reduced.states
    assert "water-autoionization" in reduced.receipt.retained_reactions
    assert "halite-dissolution" in reduced.receipt.retained_reactions


def test_symbolic_owner_uses_one_inventory_for_molarity_ions_and_energy():
    chemistry = atmospheric_aqueous_nucleus()
    reduced = chemistry.reduce(
        (state_key("H2O", AQUEOUS), state_key("NaCl", SOLID)),
        allowed_phases=(AQUEOUS, SOLID),
        families=("aqueous-acid-base", "aqueous-mineral"),
    )
    system = reduced.symbolic_owner_system("pool")
    na = state_key("Na+", AQUEOUS)
    cl = state_key("Cl-", AQUEOUS)
    assert na in system.molarity_mol_l
    assert cl in system.molality_mol_kg
    assert system.ionic_strength_mol_kg.has(system.amount_next[na])
    assert system.ionic_strength_mol_kg.has(system.amount_next[cl])
    assert len(system.residuals) == len(system.extents) + 1
    assert system.jacobian.shape == (len(system.residuals), len(system.extents) + 1)

    # Reaction extents cannot change elemental totals or net charge.  Only
    # the explicit transport inputs remain in next-minus-now closure.
    extent_symbols = set(system.extents)
    for element in reduced.receipt.active_elements:
        difference = sp.expand(
            system.element_totals_next[element] - system.element_totals_now[element])
        assert not (difference.free_symbols & extent_symbols)
    charge_difference = sp.expand(system.charge_next_mol - system.charge_now_mol)
    assert not (charge_difference.free_symbols & extent_symbols)


def test_reduction_only_drops_thermodynamically_consistent_dependent_rows():
    chemistry = atmospheric_aqueous_nucleus()
    base = chemistry.reactions["halite-dissolution"]
    doubled = Reaction.make(
        "two-halite-dissolution",
        {key: 2 * value for key, value in base.stoichiometry},
        EquilibriumLaw(2 * base.law.log10_k_ref,
                       2 * base.law.delta_h_j_mol),
        base.family, base.source)
    expanded = ChemicalCompendium(
        elements=chemistry.elements.values(),
        species=chemistry.species.values(),
        states=chemistry.states.values(),
        reactions=(*chemistry.reactions.values(), doubled))
    reduced = expanded.reduce(
        (state_key("H2O", AQUEOUS), state_key("NaCl", SOLID)),
        allowed_phases=(AQUEOUS, SOLID),
        families=("aqueous-mineral",),
    )
    assert reduced.receipt.dependent_equilibria == ("two-halite-dissolution",)

    inconsistent = Reaction.make(
        "bad-two-halite-dissolution",
        {key: 2 * value for key, value in base.stoichiometry},
        EquilibriumLaw(2 * base.law.log10_k_ref + 0.1,
                       2 * base.law.delta_h_j_mol),
        base.family, base.source)
    broken = ChemicalCompendium(
        elements=chemistry.elements.values(),
        species=chemistry.species.values(),
        states=chemistry.states.values(),
        reactions=(*chemistry.reactions.values(), inconsistent))
    with pytest.raises(ValueError, match="inconsistent log-K"):
        broken.reduce(
            (state_key("H2O", AQUEOUS), state_key("NaCl", SOLID)),
            allowed_phases=(AQUEOUS, SOLID),
            families=("aqueous-mineral",),
        )


def test_balanced_first_order_transition_is_one_precision_llvm_piece(tmp_path):
    source = "measured seam fixture"
    elements = (
        Element("H", 1, 1.008, source),
        Element("O", 8, 16.0, source),
    )
    water = ChemicalSpecies.make(
        "H2O", {"H": 2, "O": 1}, 0, 0.01801528,
        (AQUEOUS, GAS), source, "solvent",
    )
    liquid = ChemicalState("H2O", AQUEOUS, source)
    vapor = ChemicalState("H2O", GAS, source)
    transition = Reaction.make(
        "water-first-order-transition",
        {liquid.identity: -1, vapor.identity: 1},
        KineticLaw("0.75", "0.2"),
        "test-transition", source,
    )
    reduced = ChemicalCompendium(
        elements=elements, species=(water,), states=(liquid, vapor),
        reactions=(transition,),
    ).reduce((liquid.identity, vapor.identity))

    piece, layout = reduced.compile_first_order_transition_piece(
        transition.identity, cells=2, directory=tmp_path / "chemistry-bank",
    )

    assert isinstance(piece, LLVMPiece)
    assert piece.module.metadata["precision_pipeline"]["status"] == "lowered"
    assert "llvm.fma.f64" in piece.artifact.llvm_ir
    assert piece.output_names == ("inventory_next",)
    assert layout.states == (liquid.identity, vapor.identity)
    assert layout.logical_size == 4
    assert layout.physical_size == 8
    piece_path = tmp_path / "water_transition.llvm-piece.pkl"
    piece.save(piece_path)
    piece = LLVMPiece.load(piece_path)

    inventory = np.zeros(layout.physical_size, dtype=np.float64)
    initial = ((2.0, 0.25), (1.25, 0.5))
    for cell, (liquid_mol, vapor_mol) in enumerate(initial):
        inventory[layout.physical_index(cell, liquid.identity)] = liquid_mol
        inventory[layout.physical_index(cell, vapor.identity)] = vapor_mol
    dt = np.asarray([0.1, 0.0, 0.025, 0.0], dtype=np.float64)
    forward = np.asarray([0.75, 0.0, 0.4, 0.0], dtype=np.float64)
    reverse = np.asarray([0.2, 0.0, 0.1, 0.0], dtype=np.float64)
    destination = np.zeros_like(inventory)

    (result,) = piece(inventory, dt, forward, reverse, destination)

    assert result is destination
    assert np.any(result[1::layout.limbs] != 0.0)
    for cell, (liquid_mol, vapor_mol) in enumerate(initial):
        expected_extent = (
            dt[cell * 2] * (forward[cell * 2] * liquid_mol
                        - reverse[cell * 2] * vapor_mol)
            / (1.0 + dt[cell * 2] * (forward[cell * 2] + reverse[cell * 2]))
        )
        liquid_next = sum(
            result[layout.physical_index(cell, liquid.identity, limb)]
            for limb in range(layout.limbs)
        )
        vapor_next = sum(
            result[layout.physical_index(cell, vapor.identity, limb)]
            for limb in range(layout.limbs)
        )
        assert liquid_next == pytest.approx(liquid_mol - expected_extent)
        assert vapor_next == pytest.approx(vapor_mol + expected_extent)
        assert liquid_next + vapor_next == pytest.approx(liquid_mol + vapor_mol)
