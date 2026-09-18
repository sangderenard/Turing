"""Portable recurrent dendritic transition cell.

The cell is deliberately expressed only through :class:`AbstractTensor`
operators.  Hidden state is an explicit input and output, so callers can keep
it alive for an entire machine trajectory and graph-generated VJPs can pass a
hidden-state cotangent backward through time without using a tape.
"""

from __future__ import annotations

from ..abstraction import AbstractTensor


class PerforatedRecurrentTransition:
    """One persistent machine-state transition with perforated dendrites.

    Parameters are supplied explicitly.  This makes the cell suitable for
    compiler construction, serialization, and non-Python runtimes instead of
    hiding authoritative state inside a Python object.
    """

    def __init__(self, *, retain: float = 0.35) -> None:
        if not 0.0 <= retain < 1.0:
            raise ValueError("retain must be in [0, 1)")
        self.retain = float(retain)

    def forward(
        self,
        x: AbstractTensor,
        hidden: AbstractTensor,
        *,
        input_weight: AbstractTensor,
        recurrent_weight: AbstractTensor,
        hidden_bias: AbstractTensor,
        dendrite_input_weight: AbstractTensor,
        dendrite_recurrent_weight: AbstractTensor,
        dendrite_bias: AbstractTensor,
        dendrite_gain: AbstractTensor,
        dendrite_route: AbstractTensor,
        dendrite_mask: AbstractTensor,
        output_weight: AbstractTensor,
        direct_weight: AbstractTensor,
        output_bias: AbstractTensor,
    ) -> tuple[AbstractTensor, AbstractTensor]:
        branches = (
            x @ dendrite_input_weight
            + hidden @ dendrite_recurrent_weight
            + dendrite_bias
        ).tanh()
        dendritic = (branches * dendrite_gain * dendrite_mask) @ dendrite_route
        candidate = (
            x @ input_weight
            + hidden @ recurrent_weight
            + hidden_bias
            + dendritic
        ).tanh()
        hidden_next = hidden * self.retain + candidate * (1.0 - self.retain)
        prediction = x @ direct_weight + hidden_next @ output_weight + output_bias
        return prediction, hidden_next


__all__ = ["PerforatedRecurrentTransition"]
