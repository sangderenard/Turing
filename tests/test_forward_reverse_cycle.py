import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.abstract_nn import (
    ClippedCorrection,
    FixedTargets,
    ForwardReverseSolver,
    GradientCorrection,
    capture_forward_reverse_cycle,
)
from src.compiler.ssa_fortran_backend import fortran_compiler


def _forward(values):
    prediction = values["parameter"] * 2.0 + 1.0
    _uncaptured = values["parameter"] * values["parameter"]
    return {"prediction": prediction}


def _capture(correction=GradientCorrection(0.1)):
    return capture_forward_reverse_cycle(
        _forward,
        {"parameter": AT.tensor((1.0, 2.0))},
        solve_for=("parameter",),
        targets=FixedTargets({"prediction": (5.0, 7.0)}),
        correction=correction,
    )


def test_cycle_fuses_unpruned_forward_and_live_targeted_reverse_graphs():
    capture = _capture()
    result = capture.run_python()

    assert set(capture.forward_program.outputs) == {
        "prediction", "uncaptured_1_mul",
    }
    assert set(capture.target_ids) == set(capture.forward_program.outputs)
    assert set(capture.target_ids.values()).issubset(capture.fused_program.feeds)
    assert set(capture.forward_program.outputs.values()).isdisjoint(
        capture.fused_program.feeds
    )
    np.testing.assert_allclose(result.forward_outputs["prediction"].tolist(), (3.0, 5.0))
    np.testing.assert_allclose(result.proposed_parameters["parameter"].tolist(), (1.4, 2.4))

    changed = capture.run_python({
        capture.target_ids["prediction"]: AT.tensor((3.0, 5.0)),
    })
    np.testing.assert_allclose(changed.proposed_parameters["parameter"].tolist(), (1.0, 2.0))


def test_solver_retargets_and_recaptures_until_parameters_converge():
    solver = ForwardReverseSolver(
        _forward,
        {"parameter": AT.tensor((1.0, 2.0))},
        solve_for=("parameter",),
        targets=FixedTargets({"prediction": (5.0, 7.0)}),
        correction=GradientCorrection(0.2),
    )

    steps = solver.solve(8)

    assert len(steps) == 8
    np.testing.assert_allclose(
        solver.feeds["parameter"].tolist(), (2.0, 3.0), atol=3e-6
    )


def test_host_correction_hook_can_clip_a_reverse_proposal():
    solver = ForwardReverseSolver(
        _forward,
        {"parameter": AT.tensor((1.0, 2.0))},
        solve_for=("parameter",),
        targets=FixedTargets({"prediction": (101.0, 101.0)}),
        correction=ClippedCorrection(schedule=1.0, maximum_change=0.25),
    )

    step = solver.step()

    np.testing.assert_allclose(step.parameters["parameter"].tolist(), (1.25, 2.25))
    with pytest.raises(ValueError, match="host correction"):
        step.capture.emit_fortran()


def test_fused_cycle_emits_complete_fortran():
    artifact = _capture().emit_fortran(name="test_forward_reverse_cycle")

    assert artifact.module.complete
    assert "subroutine test_forward_reverse_cycle" in artifact.module.source
    assert "proposed_parameter" in str(artifact.program.outputs)


def test_retained_nonlinear_output_blocks_false_matmul_replacement():
    analysis = _capture().analyze_matmul_replacement()

    assert not analysis.fully_replaceable
    assert any(piece.operation == "mul" for piece in analysis.local_blockers)


@pytest.mark.skipif(fortran_compiler() is None, reason="no Fortran compiler installed")
def test_compiled_fortran_cycle_matches_python(tmp_path):
    capture = _capture()
    expected = capture.run_python()
    executable = capture.emit_fortran(name="native_forward_reverse_cycle").compile(tmp_path)

    actual = executable()

    np.testing.assert_allclose(
        actual["forward_prediction"], expected.forward_outputs["prediction"].tolist()
    )
    np.testing.assert_allclose(
        actual["proposed_parameter"], expected.proposed_parameters["parameter"].tolist()
    )
    native_history = executable.cycle(2)
    np.testing.assert_allclose(
        native_history[1]["forward_prediction"],
        native_history[0]["proposed_parameter"] * 2.0 + 1.0,
    )
