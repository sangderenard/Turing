import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.abstract_nn import (
    Identity,
    Linear,
    MSELoss,
    ProgramRunner,
    Sequential,
    Tanh,
    capture_backward_program,
)
from src.common.tensors.autograd import GradTape, autograd


def test_backward_can_be_obtained_and_replayed_like_a_forward_program():
    autograd.tape = GradTape()
    values = AT.tensor((1.0, 2.0, 3.0))
    values.requires_grad_(True)
    loss = (values * values).sum()

    captured = capture_backward_program(loss, (values,))
    replayed = ProgramRunner(captured.program)(captured.feed_values)

    assert captured.missing_backward == ()
    assert len(captured.program.steps) > 0
    np.testing.assert_allclose(replayed["grad_0"].tolist(), (2.0, 4.0, 6.0))


def test_backward_capture_reports_missing_rule_and_accepts_override():
    autograd.tape = GradTape()
    values = AT.tensor((1.0, 2.0, 3.0))
    values.requires_grad_(True)
    with autograd.no_grad():
        opaque = values * 1.0
    opaque.requires_grad_(True)
    autograd.capture_all = True
    try:
        autograd.record("test_opaque", (values,), opaque)
    finally:
        autograd.capture_all = False
    loss = opaque.sum()

    with pytest.raises(RuntimeError, match="test_opaque"):
        capture_backward_program(loss, (values,))

    captured = capture_backward_program(
        loss,
        (values,),
        backward_overrides={"test_opaque": lambda gradient, source: gradient},
    )
    replayed = ProgramRunner(captured.program)(captured.feed_values)
    assert captured.missing_backward == ()
    np.testing.assert_allclose(replayed["grad_0"].tolist(), (1.0, 1.0, 1.0))


def test_abstract_nn_backward_program_replays_parameter_gradients():
    autograd.tape = GradTape()
    inputs = AT.tensor(((0.1, 0.2), (0.3, 0.4)))
    targets = AT.tensor(((0.5,), (0.2,)))
    model = Sequential(
        [
            Linear(2, 3, like=inputs, init="xavier"),
            Linear(3, 1, like=inputs, init="xavier"),
        ],
        [Tanh(), Identity()],
    )
    parameters = tuple(model.parameters())
    loss = MSELoss()(model.forward(inputs), targets)
    captured = capture_backward_program(loss, parameters)
    expected = [np.asarray(parameter.grad.tolist()) for parameter in parameters]
    replayed = ProgramRunner(captured.program)(captured.feed_values)
    for index, gradient in enumerate(expected):
        np.testing.assert_allclose(
            replayed[f"grad_{index}"].tolist(), gradient, rtol=1e-7, atol=1e-7
        )
