"""Use :class:`AbstractTensor` as the carrier for the Turing bit calculus.

The bit calculus deliberately knows nothing about storage.  It asks a carrier
for eight mechanics through :class:`~src.turing_machine.turing.Hooks`, then
defines Boolean logic and integer arithmetic from those mechanics.  This
module supplies those mechanics by composing ordinary AbstractTensor
operations; it does not add bit-specific methods to AbstractTensor or bypass
its selected backend.
"""

from __future__ import annotations

from typing import Any

from ..common.tensors.abstraction import AbstractTensor
from ..turing_machine.turing import Hooks, Turing
from ..turing_machine.turing_provenance import (
    ProvenanceGraph,
    instrument_hooks,
)
from .bitops_translator import BitOpsTranslator


def abstract_tensor_hooks(like: AbstractTensor) -> Hooks:
    """Return the eight Turing carrier primitives for ``like``'s backend.

    Bitstrings are rank-one tensors whose entries are numerically zero or one.
    NAND and ``mu`` are expressed arithmetically so they remain compositions of
    the AbstractTensor primitive vocabulary on every backend.  Shape-changing
    mechanics use the existing slice and concatenate surfaces.
    """

    if not isinstance(like, AbstractTensor):
        raise TypeError("like must be an AbstractTensor")

    def zeros(n: int) -> AbstractTensor:
        if n < 0:
            raise ValueError("bitstring length cannot be negative")
        return like.ensure_tensor([0] * n)

    def length(value: AbstractTensor) -> int:
        if not isinstance(value, AbstractTensor):
            raise TypeError("the AbstractTensor bit carrier received a foreign value")
        return int(value.numel())

    def concat(left: AbstractTensor, right: AbstractTensor) -> AbstractTensor:
        return AbstractTensor.cat((left, right), dim=0)

    def slice_bits(value: AbstractTensor, start: int, stop: int) -> AbstractTensor:
        return value[start:stop]

    def nand(left: AbstractTensor, right: AbstractTensor) -> AbstractTensor:
        return 1 - left * right

    def sigma_left(value: AbstractTensor, count: int) -> AbstractTensor:
        if count < 0:
            raise ValueError("shift count cannot be negative")
        return concat(value, zeros(count))

    def sigma_right(value: AbstractTensor, count: int) -> AbstractTensor:
        if count < 0:
            raise ValueError("shift count cannot be negative")
        return slice_bits(value, 0, max(length(value) - count, 0))

    def mu(
        if_false: AbstractTensor,
        if_true: AbstractTensor,
        selector: AbstractTensor,
    ) -> AbstractTensor:
        return if_false + selector * (if_true - if_false)

    return Hooks(
        nand=nand,
        sigma_L=sigma_left,
        sigma_R=sigma_right,
        concat=concat,
        slice=slice_bits,
        mu=mu,
        length=length,
        zeros=zeros,
    )


class AbstractTensorBitOpsTranslator(BitOpsTranslator):
    """Run the existing BitOps recipes on one AbstractTensor backend."""

    def __init__(self, bit_width: int, *, like: AbstractTensor):
        if bit_width <= 0:
            raise ValueError("bit_width must be positive")
        if not isinstance(like, AbstractTensor):
            raise TypeError("like must be an AbstractTensor")
        self.bit_width = bit_width
        self.like = like
        self.graph = ProvenanceGraph()
        self.tm = Turing(
            instrument_hooks(abstract_tensor_hooks(like), self.graph)
        )

    def bits_from_int(self, val: int) -> AbstractTensor:
        mask = (1 << self.bit_width) - 1
        value = int(val) & mask
        bits = [
            (value >> (self.bit_width - 1 - index)) & 1
            for index in range(self.bit_width)
        ]
        return self.like.ensure_tensor(bits)

    def int_from_bits(self, bits: AbstractTensor) -> int:
        """Cross the tensor boundary only when a caller requests a Python int."""

        value = 0
        for bit in bits.tolist():
            value = (value << 1) | (int(round(float(bit))) & 1)
        return value


def abstract_tensor_bitops_factory(like: AbstractTensor):
    """Build a ProcessGraph-compatible translator factory for ``like``."""

    def factory(bit_width: int) -> AbstractTensorBitOpsTranslator:
        return AbstractTensorBitOpsTranslator(bit_width, like=like)

    return factory


__all__ = [
    "AbstractTensorBitOpsTranslator",
    "abstract_tensor_bitops_factory",
    "abstract_tensor_hooks",
]
