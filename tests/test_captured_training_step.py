"""A captured training step must contain its own backward, not a frozen one.

``IRGraphedModel.capture`` builds one ``FusedProgram`` covering forward, loss,
backward and the Adam update, with ``param{i}_new`` and optimizer state as
outputs. That shape is what makes a compiled training loop possible: feed the
outputs back as the next step's feeds and no tape is needed again.

The failure this guards against is quiet. If the backward runs under
``no_grad`` during capture, its gradients never become steps -- they enter the
program as constant feeds. The program still has a loss output, still has
``param*_new``, still tracks an eager optimizer for the first couple of steps
while Adam's bias correction dominates, and then applies the capture-time
gradient forever. Nothing raises; the loss simply climbs.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor as AT, tensor_identity
from src.common.tensors.abstract_nn.fused_program import IRGraphedModel
from src.common.tensors.autograd import GradTape, autograd


class _TinyNet:
    """Two dense layers on the AbstractTensor surface, nothing else."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights = []
        for shape, scale in (((2, 4), 0.6), ((4,), 0.0), ((4, 1), 0.6), ((1,), 0.0)):
            made = AT.tensor((rng.normal(size=shape) * scale).tolist())
            made.requires_grad_(True)
            self.weights.append(made)

    def parameters(self):
        return self.weights

    def forward(self, x):
        # ``capture`` resets the tape before calling this, and these were built
        # earlier, so they still point at the previous tape. Attach them before
        # any operator records against them, or the optimizer's ``p - update``
        # records on the stale tape and never enters the program.
        for each in self.weights:
            each._tape = autograd.tape
            autograd.tape.create_tensor_node(each)
        hidden = (x.matmul(self.weights[0]) + self.weights[1]).tanh()
        return hidden.matmul(self.weights[2]) + self.weights[3]


@pytest.fixture
def captured():
    rng = np.random.default_rng(1)
    inputs = rng.uniform(-1.0, 1.0, size=(16, 2))
    targets = ((inputs[:, 0] * inputs[:, 1]) < 0).astype(np.float64).reshape(16, 1)

    autograd.tape = GradTape()
    net = _TinyNet()
    graphed = IRGraphedModel(net).config(lr=5e-2)
    graphed.capture(AT.tensor(inputs.tolist()), AT.tensor(targets.tolist()))
    return net, graphed


def test_the_whole_training_step_is_one_program(captured):
    _net, graphed = captured
    program = graphed.program

    assert program is not None
    for name in ("loss", "pred", "opt_t_new"):
        assert name in program.outputs
    for index in range(4):
        assert f"param{index}_new" in program.outputs
        assert f"opt_m{index}_new" in program.outputs
        assert f"opt_v{index}_new" in program.outputs


def test_every_declared_output_is_produced_by_a_step(captured):
    """An output no step produces is a program that cannot be replayed.

    ``opt_t_new`` was the one that failed: an operator only records when an
    operand requires grad, so ``t + 1.0`` on a plain scalar recorded nothing
    and the step counter never entered the program.
    """

    _net, graphed = captured
    program = graphed.program
    produced = {step.result_id for step in program.steps}

    nowhere = {
        name: value
        for name, value in program.outputs.items()
        if value not in produced and value not in program.feeds
    }
    assert nowhere == {}


def test_the_gradients_are_computed_by_the_program_not_fed_into_it(captured):
    """The defect that makes a compiled loop diverge while looking healthy."""

    net, graphed = captured
    program = graphed.program
    produced = {step.result_id for step in program.steps}

    # Each parameter's update consumes that parameter and its gradient. The
    # gradient has to be a value the program computes; if it is a feed, the
    # loop replays one frozen gradient forever.
    parameter_ids = {tensor_identity(p) for p in net.parameters()}
    update_steps = [
        step
        for step in program.steps
        if step.op_name == "sub" and any(i in parameter_ids for i in step.input_ids)
    ]
    assert update_steps, "no parameter update step found"

    def reaches_a_feed_only(value_id, depth=0):
        """True when nothing upstream of this value is computed."""
        if value_id in produced:
            return False
        return True

    # The backward has to have left steps behind at all: a capture whose
    # backward ran under no_grad produces only the forward's operations.
    assert len(program.steps) > 40, (
        f"only {len(program.steps)} steps captured; the backward pass is missing"
    )
    assert not all(
        reaches_a_feed_only(i) for step in update_steps for i in step.input_ids
    )


def test_optimizer_state_is_structural_not_a_trainable_parameter(captured):
    """It must require grad to record, and must never be trained."""

    _net, graphed = captured
    tape = autograd.tape
    for state in graphed.opt_m + graphed.opt_v + [graphed.opt_t]:
        assert state.requires_grad, "state that does not require grad records nothing"
        assert tape.is_structural(state), "optimizer state must not be a parameter"
