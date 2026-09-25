"""Insert the limb-correct basic-operator surface into Precision.

Idempotent: refuses if the surface is already present, so a re-run cannot
duplicate the class the way an earlier hand patch did.
"""
from pathlib import Path

ANCHOR = '    def __neg__(self): return Precision.dispatch("neg", self, None)'

ADDITION = '''

    # -- the rest of the basic surface, limb-correct ---------------------
    #
    # Everything below is built from the operators above and from ordinary
    # tensor methods applied to individual LIMBS, which are ordinary
    # tensors. What is never done is hand the interleaved payload to a
    # tensor operation that does not know about limbs: that reads the
    # channels as data and answers confidently from a fraction of the
    # value, which is the failure this whole type exists to prevent.

    def element_count(self) -> int:
        """How many VALUES this holds, never how many limbs."""

        from .abstraction import AbstractTensor

        return int(
            AbstractTensor.get_tensor(self.term(0)).reshape(-1).shape[0]
        )

    def _map_limbs(self, function) -> "Precision":
        """Apply an EXACT per-limb map: one that cannot round.

        Multiplying every limb by the same sign is exact and distributes
        over the expansion. Nothing that ROUNDS may use this -- rounding a
        limb alone loses its relationship to the ones below it, which is
        the entire content of the representation.
        """

        return Precision(
            interleave([function(term) for term in self.terms()]), self.limbs
        )

    def _slice_limbs(self, start: int, stop: int) -> "Precision":
        """Elements ``[start:stop]``, with each one's limbs kept together."""

        return Precision(
            interleave([
                term.reshape(-1)[start:stop] for term in self.terms()
            ]),
            self.limbs,
        )

    def sign(self):
        """The sign of the whole expansion, as an ordinary tensor.

        The leading limb decides: a distilled expansion keeps every tail
        below an ulp of the head, so no tail can outvote it.
        """

        return self.term(0).sign()

    def __abs__(self) -> "Precision":
        """Magnitude, by negating the WHOLE expansion where it is negative.

        Taking the absolute value of each limb separately is a different
        number: ``|a| + |b|`` is not ``|a + b|`` when the tail opposes the
        head, which for a distilled expansion is the usual case.
        """

        sign = self.sign()
        return self._map_limbs(lambda term: term * sign)

    def floor(self):
        """The integer part, as an ordinary tensor.

        Deliberately not a ``Precision``: an integer needs no limbs, and
        returning one would invite a caller to carry a tail that is
        exactly zero. The collapsed sum decides and the residual the
        collapse discarded corrects it -- a value whose head sits just
        above an integer while its tail pulls it below belongs to the
        lower one, and the head alone cannot see that.
        """

        candidate = self.collapse().floor()
        residual = (self - candidate).collapse()
        return candidate - (residual < 0.0) * 1.0

    def sqrt(self) -> "Precision":
        """Newton's iteration, which is a fixed point of the answer.

        Needs only multiply, add and divide, so it is exactly as wide as
        the expansion it runs on -- no core, no interval, no table. The
        double seed carries about sixteen digits and every step doubles
        them, so the count is read from the width rather than fixed, plus
        one for the seed's own last bit.
        """

        import math

        seed = Precision.of(self.collapse().sqrt(), self.limbs)
        steps = max(1, int(math.ceil(math.log2(max(self.limbs, 1))))) + 1
        root = seed
        for _step in range(steps):
            root = (root + self / root) * 0.5
        return root

    def __pow__(self, exponent) -> "Precision":
        """Integer powers by repeated squaring, and nothing else.

        A fractional power is a transcendental and belongs to the signal
        cores, which state the interval they are valid on and measure
        their own error. Answering one here would mean collapsing to a
        double and discarding every limb the caller asked for.
        """

        if isinstance(exponent, float) and exponent.is_integer():
            exponent = int(exponent)
        if not isinstance(exponent, int):
            raise TypeError(
                f"Precision supports integer powers; {exponent!r} is a "
                "transcendental one -- take it through the signal cores, "
                "which state the interval they are valid on"
            )
        one = Precision.of(self.collapse() * 0.0 + 1.0, self.limbs)
        if exponent == 0:
            return one
        if exponent < 0:
            return one / (self ** -exponent)
        result, base, remaining = None, self, exponent
        while remaining:
            if remaining & 1:
                result = base if result is None else result * base
            remaining >>= 1
            if remaining:
                base = base * base
        return result

    def sum(self) -> "Precision":
        """Add every element WITHOUT collapsing on the way.

        The reduction that made this type necessary. Summing a collapsed
        expansion rounds each element to one double before adding, so a
        two-limb field reports a one-limb total and the width bought
        nothing -- measured at eight parts in ten to the seventeenth on a
        three-element field, which is one whole double of the answer
        discarded.

        Done by PAIRWISE HALVING: fold the upper half onto the lower with
        one wide add and repeat. Every add is the exact expansion add, so
        the total is exact; there are log2(n) of them and each is
        vectorised, so it costs a handful of passes rather than one
        sequential step per element. An odd element is carried into the
        next round untouched rather than dropped or counted twice.
        """

        from .abstraction import AbstractTensor

        held = self
        count = held.element_count()
        while count > 1:
            half = count // 2
            folded = (
                held._slice_limbs(0, half)
                + held._slice_limbs(half, 2 * half)
            )
            if count % 2:
                tail = held._slice_limbs(count - 1, count)
                held = Precision(
                    interleave([
                        AbstractTensor.concat([left, right], dim=0)
                        for left, right in zip(folded.terms(), tail.terms())
                    ]),
                    held.limbs,
                )
                count = half + 1
            else:
                held = folded
                count = half
        return held

    def mean(self) -> "Precision":
        """The average, as wide as the values averaged.

        Divides by the ELEMENT count. Dividing by the widened count --
        elements times limbs -- is what returned half of a two-limb value
        in the measurement this class's docstring opens with, and it
        looked like an ordinary answer.
        """

        return self.sum() / float(max(self.element_count(), 1))
'''

path = Path(__file__).resolve().parents[2] / "src/common/tensors/extended_precision.py"
text = path.read_text(encoding="utf-8")
if "the rest of the basic surface" in text:
    raise SystemExit("surface already present; refusing to duplicate")
if text.count(ANCHOR) != 1:
    raise SystemExit(f"anchor appears {text.count(ANCHOR)} times; refusing")
path.write_text(text.replace(ANCHOR, ANCHOR + ADDITION, 1), encoding="utf-8")
print("patched once")
