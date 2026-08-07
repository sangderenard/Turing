import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.abstract_nn import (
    build_fused_program,
    capture_reverse_fused_program,
    retain_uncaptured_outputs,
)
from src.common.tensors.autograd import GradTape, autograd


def _unpruned_program():
    autograd.tape = GradTape()
    source = AT.tensor((1.0, 2.0))
    source.requires_grad_(True)
    visible = source * 2.0
    uncaptured = source + 3.0
    program = build_fused_program(
        autograd.tape.graph.copy(),
        outputs={"visible": id(visible)},
    )
    values = {
        id(source): source,
        id(visible): visible,
        id(uncaptured): uncaptured,
    }
    return source, visible, uncaptured, program, values


def test_retention_promotes_terminal_results_before_pruning():
    _, visible, uncaptured, program, _ = _unpruned_program()

    retained = retain_uncaptured_outputs(program)

    assert retained.outputs["visible"] == id(visible)
    added = {name: value_id for name, value_id in retained.outputs.items() if name != "visible"}
    assert added == {"uncaptured_1_add": id(uncaptured)}
    assert program.outputs == {"visible": id(visible)}


def test_reverse_capture_parameterizes_all_outputs_and_returns_incidentals():
    source, visible, uncaptured, program, values = _unpruned_program()
    retained = retain_uncaptured_outputs(program)
    targets = {
        "visible": AT.tensor((4.0, 6.0)),
        "uncaptured_1_add": AT.tensor((5.0, 6.0)),
    }

    captured = capture_reverse_fused_program(
        program,
        values,
        targets,
        step_size=0.1,
    )
    result = captured.run()

    assert set(captured.output_parameters) == set(retained.outputs)
    assert set(captured.output_parameters.values()).issubset(captured.program.feeds)
    assert set(captured.output_parameters.values()).isdisjoint(captured.incidental_feed_ids)
    assert id(source) in captured.incidental_feed_ids
    assert set(program.feeds).issubset(captured.incidental_feed_ids)
    assert captured.missing_backward == ()
    proposal = next(iter(result.proposed_inputs.values()))
    # d(1/2*((2x-target_visible)^2 + (x+3-target_dead)^2))/dx
    # is (-5, -5), so x - 0.1*gradient is (1.5, 2.5).
    np.testing.assert_allclose(proposal.tolist(), (1.5, 2.5))
    assert set(result.incidentals) == set(captured.incidental_feed_ids)

    changed = captured.run({
        captured.output_parameters["visible"]: AT.tensor((2.0, 4.0)),
    })
    changed_proposal = next(iter(changed.proposed_inputs.values()))
    np.testing.assert_allclose(changed_proposal.tolist(), (1.1, 2.1))


def test_reverse_capture_requires_a_parameter_for_every_retained_output():
    _, _, _, program, values = _unpruned_program()

    with pytest.raises(ValueError, match="must match every retained output"):
        capture_reverse_fused_program(
            program,
            values,
            {"visible": AT.tensor((4.0, 6.0))},
        )
