"""Output-only native piece fields are managed dt-state columns."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from src.compiler.native_package import piece_from_law  # noqa: E402
from src.compiler.symbolic_equation_compiler import compile_sympy_equations  # noqa: E402
from src.common.dt_system.dt_controller import (  # noqa: E402
    STController,
    Targets,
    run_superstep,
)
from llvm_dt_system import (  # noqa: E402
    column_names_of,
    dt_system_contract,
    instantiate_state,
)

BATCH = 4


def test_column_names_keep_read_order_before_declared_write_only_columns():
    pieces = [
        SimpleNamespace(
            argument_names=("x", "dt"),
            output_names=("applied_force_next", "dt_next"),
        ),
        SimpleNamespace(
            argument_names=("torque", "x"),
            output_names=(
                "torque_next", "angular_momentum_next", "applied_force_next",
            ),
        ),
    ]

    assert column_names_of(pieces) == (
        "x", "torque", "applied_force", "angular_momentum",
    )


def test_output_only_native_readout_is_registered_persisted_and_restored(tmp_path):
    x, dt = sp.symbols("x dt")
    law = compile_sympy_equations([
        sp.Eq(sp.Symbol("x_next"), x + dt, evaluate=False),
        # No law reads applied_force; this is a declared endpoint readout.
        sp.Eq(sp.Symbol("applied_force_next"), 2 * dt, evaluate=False),
        sp.Eq(sp.Symbol("mass_err"), dt, evaluate=False),
    ], name="output_only_readout")
    piece_file = tmp_path / "output_only_readout.piece"
    piece_from_law(
        law, "output_only_readout", BATCH, directory=tmp_path,
    ).save(piece_file)

    # Read columns stay first; declared write-only state follows, while dt
    # remains the managed integrator's special column.
    from llvm_dt_system import LLVMPiece  # noqa: E402

    piece = LLVMPiece.load(piece_file)
    assert column_names_of([piece]) == ("x", "applied_force")
    abi = dt_system_contract(
        "dt_system_over", column_names_of([piece]), BATCH,
    ).program_abi.receipt()
    assert abi["records"]["PieceState"]["fields"]["applied_force"] == {
        "storage": "span", "dtype": "float64", "rank": 1,
        "shape": [BATCH], "mutable": True,
    }

    x0 = np.arange(BATCH, dtype=np.float64)
    initial_force = np.full(BATCH, -3.0)
    columns = {"x": x0.copy(), "applied_force": initial_force.copy()}
    targets = Targets(cfl=1.0, div_max=1.0, mass_max=0.05)
    controller = STController(dt_min=None)
    state = instantiate_state([piece], columns, targets=targets)
    owned_force = state.applied_force
    attempts = []
    readouts_before_attempt = []

    def advance(current, step_dt):
        readouts_before_attempt.append(current.applied_force.copy())
        return current.program["advance_pieces"](current, step_dt)

    advanced, _dt_next, _metrics = run_superstep(
        state,
        round_max=0.1,
        dt_init=0.1,
        dx=1.0,
        targets=targets,
        ctrl=controller,
        advance=advance,
        attempt_log=attempts,
        rollback=True,
    )

    assert attempts[0]["accepted"] is False
    assert attempts[1]["accepted"] is True
    assert attempts[1]["dt"] < attempts[0]["dt"]
    np.testing.assert_array_equal(readouts_before_attempt[0], initial_force)
    np.testing.assert_array_equal(readouts_before_attempt[1], initial_force)
    assert float(advanced) == 0.1
    np.testing.assert_allclose(state.x, x0 + 0.1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(state.applied_force, 0.1, rtol=0, atol=1e-12)
    assert state.applied_force is owned_force

    # The output-only column is part of the same registered state snapshot
    # that run_superstep uses for rollback, and restoration writes in place.
    snapshot = state.copy_shallow()
    state.applied_force[...] = -99.0
    state.restore(snapshot)
    assert state.applied_force is owned_force
    np.testing.assert_allclose(state.applied_force, 0.1, rtol=0, atol=1e-12)
