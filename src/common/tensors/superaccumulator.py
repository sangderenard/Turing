"""A Kulisch-style superaccumulator built from ordinary tensor arithmetic.

An expansion (see ``extended_precision``) renormalises on every addition, and
that renormalisation is the expensive, sequential, order-sensitive part. A
superaccumulator removes it: a wide FIXED-POINT register absorbs any double
exactly, with no rounding at all, so accumulation becomes associative,
order-independent, and O(1) per term. Renormalisation happens once, at the end.

WHY THIS IS A MODULE AND NOT A DTYPE. A dtype is a backend-level concept --
introducing one means teaching numpy, torch, the C path and LLVM about it, and
that discards the property that makes the extended-precision shim work at all:
it needs no backend changes, because it is written in operations every backend
already has. Nothing here needs a new dtype either. A float64 IS an exact
integer container below 2**53, so a fixed-point register is just a tensor of
integer-valued floats, and the limb index is an ordinary tensor AXIS -- which
means it vectorises, batches, and compiles with no new machinery.

THE REPRESENTATION. The accumulator holds ``limbs`` slots, slot ``j`` carrying
an integer-valued float weighted by ``2**(low + j*width)``. The value is the
exact sum of ``slot[j] * 2**(low + j*width)``. Each slot has ``53 - width``
bits of headroom, so with the default width that is over a million additions
before a carry is even possible -- carries are DEFERRED, not propagated per
term, which is the other half of why absorption is cheap.

THE SPLIT. Placing a double into the grid usually wants exponent extraction.
It does not have to. For a shifter ``S = 1.5 * 2**(s + 52)``, the expression
``(x + S) - S`` rounds ``x`` to a multiple of ``2**s`` exactly and without a
branch, and ``x - that`` is the exact remainder. Cascading from the top slot
down splits any in-range double across the grid using nothing but ``+`` and
``-``. That matters twice: it needs no operation the vocabulary lacks, and it
gives the compiler nothing but adds to translate.

THE COST THIS REMOVES. Absorbing ``a*b`` exactly is ``two_product`` followed
by absorbing both halves -- so an exact dot product costs a constant per term
instead of an expansion renormalisation per term. That is the structure ExBLAS
uses, and it is what makes exact summation competitive rather than merely
correct.

Like the error-free transformations it builds on, every line here depends on
the arithmetic NOT being reassociated: ``(x + S) - S`` is algebraically ``x``,
and an optimiser permitted to say so deletes the entire mechanism silently.
"""

from __future__ import annotations

from typing import Any, Sequence

from .extended_precision import two_product, two_sum, renormalise

# Bits carried per slot. 24 leaves 29 bits of headroom in a float64's exact
# integer range, so ~5*10**8 absorptions can land before a carry is possible.
DEFAULT_WIDTH = 24


class SuperAccumulator:
    """An exact fixed-point accumulator over a bounded exponent range.

    The range is bounded on purpose. A register spanning the whole binary64
    exponent range needs ~2100 bits and around ninety slots, and every
    absorption would touch all of them. Real accumulations occupy a narrow
    band, so the range is a PARAMETER.

    That range is a PRECONDITION, not a checked invariant. Absorbing a value
    larger than the top slot can hold silently loses the excess, and anything
    below the bottom slot falls off as ``residual``. Verifying it per
    absorption would need a reduction over the tensor, which costs more than
    the absorption and would put a branch in the middle of the straight-line
    arithmetic this exists to keep. Call ``check_range`` explicitly when the
    magnitudes are not known in advance.
    """

    def __init__(self, slots: Any, low: int, width: int = DEFAULT_WIDTH):
        self.slots = list(slots)
        self.residual = None
        self.low = int(low)
        self.width = int(width)

    # -- construction ------------------------------------------------------

    @classmethod
    def zeros(cls, like: Any, magnitude_bits: int = 160,
              width: int = DEFAULT_WIDTH) -> "SuperAccumulator":
        """An empty accumulator covering ``+/-2**magnitude_bits``.

        ``like`` supplies shape and backend: the slots are zeros shaped like
        the values that will be absorbed, so a whole tensor accumulates
        elementwise in parallel.
        """

        count = 2 * (int(magnitude_bits) // int(width)) + 1
        low = -(count // 2) * int(width)
        zero = like * 0.0
        return cls([zero + 0.0 for _ in range(count)], low, width)

    def scale_of(self, index: int) -> int:
        return self.low + index * self.width

    # -- absorption --------------------------------------------------------

    def absorb(self, value: Any) -> "SuperAccumulator":
        """Add one tensor of doubles exactly. No rounding occurs anywhere.

        The cascade runs from the most significant slot down, so the remainder
        entering each step is already smaller than that slot's weight and the
        shifter is always in range.
        """

        rest = value + 0.0
        for index in range(len(self.slots) - 1, -1, -1):
            scale = self.scale_of(index)
            shifter = 1.5 * (2.0 ** (scale + 52))
            part = (rest + shifter) - shifter
            self.slots[index] = self.slots[index] + part * (2.0 ** -scale)
            rest = rest - part
        # What survives the cascade lies below the smallest slot. It is kept
        # rather than discarded so a caller can see that the grid was too
        # coarse instead of having to infer it from a wrong answer.
        self.residual = rest
        return self

    def check_range(self, value: Any) -> None:
        """Raise if ``value`` does not fit the grid. Costs a reduction."""

        top = 2.0 ** (self.scale_of(len(self.slots) - 1) + 52)
        largest = float(value.abs().max().tolist())
        if largest >= top:
            raise ValueError(
                f"magnitude {largest:.3e} exceeds this accumulator's top slot "
                f"({top:.3e}); widen magnitude_bits or the excess is lost"
            )

    def absorb_product(self, left: Any, right: Any) -> "SuperAccumulator":
        """Add ``left * right`` exactly.

        ``two_product`` splits the product into two doubles whose sum is the
        exact product, and both are absorbed -- so a dot product accumulated
        this way carries no rounding at all until it is finalised once.
        """

        high, low = two_product(left, right)
        return self.absorb(high).absorb(low)

    def merge(self, other: "SuperAccumulator") -> "SuperAccumulator":
        """Combine two accumulators. Exact, and therefore associative.

        This is what makes a parallel or chunked reduction give bit-identical
        results regardless of how the work was divided.
        """

        if self.low != other.low or self.width != other.width:
            raise ValueError("accumulators must share a grid to merge")
        self.slots = [a + b for a, b in zip(self.slots, other.slots)]
        return self

    # -- finalisation ------------------------------------------------------

    def terms(self) -> list:
        """Every slot restored to its true weight, most significant first.

        Scaling by a power of two is exact, so this loses nothing; it is a
        change of representation, not a rounding.
        """

        return [self.slots[index] * (2.0 ** self.scale_of(index))
                for index in range(len(self.slots) - 1, -1, -1)]

    def to_expansion(self, limbs: int = 2) -> list:
        """The exact total as an ``limbs``-limb expansion.

        Summing from the least significant end through the distillation in
        ``extended_precision`` is what turns an exact fixed-point total back
        into a correctly-rounded floating result.
        """

        return renormalise(self.terms(), limbs)

    def value(self) -> Any:
        """The correctly-rounded double nearest the exact total."""

        return self.to_expansion(1)[0]


def exact_sum(values: Sequence, magnitude_bits: int = 160) -> Any:
    """Sum a sequence of tensors with no rounding until the final result."""

    if not values:
        raise ValueError("nothing to sum")
    accumulator = SuperAccumulator.zeros(values[0], magnitude_bits)
    for value in values:
        accumulator.absorb(value)
    return accumulator.value()


def exact_dot(left: Sequence, right: Sequence,
              magnitude_bits: int = 160) -> Any:
    """Dot product with every product and every sum carried exactly."""

    if len(left) != len(right):
        raise ValueError("dot product needs matching lengths")
    accumulator = SuperAccumulator.zeros(left[0], magnitude_bits)
    for a, b in zip(left, right):
        accumulator.absorb_product(a, b)
    return accumulator.value()
