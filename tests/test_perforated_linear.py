import numpy as np

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.abstract_nn.demo_perforated_regression import (
    _train_phase,
    run_comparison,
)
from src.common.tensors.abstract_nn.fused_program import capture_forward_program
from src.common.tensors.abstract_nn.perforated import PerforatedLinear
from src.common.tensors.autograd import GradTape, autograd


def test_perforated_linear_is_abstract_and_backwardable():
    previous_tape = autograd.tape
    autograd.tape = GradTape()
    try:
        with AT.use_backend("numpy", "cpu"):
            x = AT.tensor([[0.2, -0.4], [0.8, 0.1]], dtype="float64")
            model = PerforatedLinear(
                2, 1, like=x, dendrites_per_neuron=2, active=True
            )
            output = model.forward(x)
            loss = (output * output).mean()
            gradients = autograd.grad(loss, model.parameters(), allow_unused=False)

            assert type(output).__name__ == "NumPyTensorOperations"
            assert output.shape == (2, 1)
            assert all(gradient is not None for gradient in gradients)
            assert [gradient.shape for gradient in gradients] == [
                parameter.shape for parameter in model.parameters()
            ]

            program, input_id = capture_forward_program(model, x)
            assert input_id in program.feeds
            assert {"matmul", "tanh", "mul"} <= {
                step.op_name for step in program.steps
            }
    finally:
        autograd.tape = previous_tape


def test_dendrite_phase_does_not_mutate_base_parameters():
    previous_tape = autograd.tape
    autograd.tape = GradTape()
    try:
        with AT.use_backend("numpy", "cpu"):
            x = AT.tensor([[0.0, 0.0], [1.0, -1.0]], dtype="float64")
            y = AT.tensor([[0.5], [-0.25]], dtype="float64")
            model = PerforatedLinear(2, 1, like=x, active=True)
            before = [np.asarray(p.tolist()) for p in model.base_parameters()]
            _train_phase(
                model,
                model.dendrite_parameters(),
                x,
                y,
                steps=2,
                learning_rate=0.02,
            )
            after = [np.asarray(p.tolist()) for p in model.base_parameters()]
            assert all(np.array_equal(a, b) for a, b in zip(before, after))
    finally:
        autograd.tape = previous_tape


def test_runtime_branch_mask_can_disable_perforated_subgraphs():
    previous_tape = autograd.tape
    autograd.tape = GradTape()
    try:
        with AT.use_backend("numpy", "cpu"):
            x = AT.tensor([[0.3, -0.2]], dtype="float64")
            model = PerforatedLinear(2, 1, like=x, dendrites_per_neuron=2,
                                     active=True)
            off = AT.tensor([[0.0, 0.0]], dtype="float64")
            np.testing.assert_allclose(
                np.asarray(model.forward(x, dendrite_mask=off).tolist()),
                np.asarray(model.forward_base(x).tolist()),
            )
    finally:
        autograd.tape = previous_tape


def test_perforated_comparison_beats_the_linear_baseline_on_held_out_data():
    result = run_comparison(
        backend="numpy",
        samples=96,
        base_steps=35,
        dendrite_steps=90,
        consolidate_steps=25,
        learning_rate=0.03,
        seed=1729,
    )
    assert result.perforated_final_loss < result.perforated_initial_loss
    assert result.perforated_final_loss < result.ordinary_final_loss
