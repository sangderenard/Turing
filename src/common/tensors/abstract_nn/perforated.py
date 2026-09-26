"""Dendritic ("perforated") layers on the portable AbstractTensor surface.

The implementation deliberately contains no NumPy or PyTorch arithmetic.  A
backend is selected only when tensors are created; the forward graph is made
from AbstractTensor matmul, tanh, multiply, and addition
operators, so the same graph remains available to autograd and translators.
"""

from __future__ import annotations

import math
import random
from typing import List

from ..abstraction import AbstractTensor
from ..autograd import autograd
from .core import Linear, _ensure_batch_dim, _shape_of
from .utils import from_list_like


def _random_matrix(
    rows: int,
    columns: int,
    *,
    like: AbstractTensor,
    scale: float,
    label: str,
) -> AbstractTensor:
    values = [
        [random.gauss(0.0, 1.0) * scale for _ in range(columns)]
        for _ in range(rows)
    ]
    tensor = from_list_like(
        values, like=like, requires_grad=True, tape=autograd.tape
    )
    tensor._label = label
    autograd.tape.create_tensor_node(tensor)
    autograd.tape.annotate(tensor, label=label)
    return tensor


class PerforatedLinear:
    """A linear layer augmented by neuron-specific nonlinear dendrites.

    For input ``x`` and output neuron ``j`` the layer computes

    ``x @ W + b + sum_k(gain[j, k] * tanh(x @ D[:, j, k] + db[j, k]))``.

    ``perforate(False)`` removes the dendritic term from the executed graph.
    This permits the three useful training phases without changing gradient
    machinery: train the base neuron, activate and train only dendrites, then
    freeze dendrites and consolidate the base neuron.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        like: AbstractTensor,
        dendrites_per_neuron: int = 2,
        bias: bool = True,
        init: str = "xavier",
        active: bool = False,
        label_prefix: str = "PerforatedLinear",
    ) -> None:
        if dendrites_per_neuron < 1:
            raise ValueError("dendrites_per_neuron must be at least one")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.dendrites_per_neuron = int(dendrites_per_neuron)
        self.active = bool(active)
        self.base = Linear(
            in_dim,
            out_dim,
            like=like,
            bias=bias,
            init=init,
            _label_prefix=f"{label_prefix}.base",
        )

        branch_width = self.out_dim * self.dendrites_per_neuron
        self.dendrite_weight = _random_matrix(
            self.in_dim,
            branch_width,
            like=like,
            scale=math.sqrt(1.0 / float(self.in_dim)),
            label=f"{label_prefix}.dendrite_weight",
        )
        self.dendrite_bias = _random_matrix(
            1,
            branch_width,
            like=like,
            scale=0.05,
            label=f"{label_prefix}.dendrite_bias",
        )
        # Nonzero gains let gradients reach both sides of every branch on the
        # first dendrite-training step.  Dendrites are absent, not merely
        # multiplied by zero, while ``active`` is false.
        gain_values = [
            [0.05 / self.dendrites_per_neuron] * branch_width
        ]
        self.dendrite_gain = from_list_like(
            gain_values, like=like, requires_grad=True, tape=autograd.tape
        )
        self.dendrite_gain._label = f"{label_prefix}.dendrite_gain"
        autograd.tape.create_tensor_node(self.dendrite_gain)
        autograd.tape.annotate(
            self.dendrite_gain, label=self.dendrite_gain._label
        )
        route_values = [
            [
                1.0 if output_index == branch_index // self.dendrites_per_neuron
                else 0.0
                for output_index in range(self.out_dim)
            ]
            for branch_index in range(branch_width)
        ]
        self.dendrite_route = from_list_like(route_values, like=like)

    def perforate(self, active: bool = True) -> "PerforatedLinear":
        self.active = bool(active)
        return self

    def base_parameters(self) -> List[AbstractTensor]:
        return list(self.base.parameters())

    def dendrite_parameters(self) -> List[AbstractTensor]:
        return [
            self.dendrite_weight,
            self.dendrite_bias,
            self.dendrite_gain,
        ]

    def parameters(self) -> List[AbstractTensor]:
        return self.base_parameters() + self.dendrite_parameters()

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def forward_base(self, x: AbstractTensor) -> AbstractTensor:
        return self.base.forward(x)

    def forward(self, x: AbstractTensor,
                dendrite_mask: AbstractTensor | None = None) -> AbstractTensor:
        base_output = self.forward_base(x)
        if not self.active:
            return base_output

        x, added = _ensure_batch_dim(x, target_ndim=2)
        tape = autograd.tape
        for parameter in self.dendrite_parameters():
            parameter._tape = tape
            tape.create_tensor_node(parameter)

        branches = ((x @ self.dendrite_weight) + self.dendrite_bias).tanh()
        gated = branches * self.dendrite_gain
        if dendrite_mask is not None:
            gated = gated * dendrite_mask
        dendritic_signal = gated @ self.dendrite_route
        output = base_output + dendritic_signal
        if added:
            output = output.reshape(*_shape_of(output)[1:])
        return output


__all__ = ["PerforatedLinear"]
