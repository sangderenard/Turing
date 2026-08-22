"""N-double arithmetic as a shim in the operator dispatch.

An extended value is an ordinary tensor plus a tuple of correction tensors in
``_limbs``: the value it denotes is the exact sum of all of them, and every
limb is a plain tensor of the base dtype. Nothing here introduces a new dtype,
a new backend entry point, or a new opcode -- every step is built from ``+``,
``-`` and ``*`` on the base dtype, which is why the shim can sit above the
backend unwrap and serve every backend at once.

``n`` is a parameter, not a fixed choice of two. One limb is ordinary
arithmetic and the shim declines; two is double-double; four is quad-double;
larger is an expansion in Shewchuk's sense. Each limb buys roughly 53 more
bits, so ``n`` limbs is about ``16*n`` decimal digits.

Three properties make this practical:

* the limbs are DATA-PARALLEL. Each element's limb chain is dependent, but
  elements are independent, so it vectorises and the cost is flop count rather
  than latency;
* the expansion is ordinary tensor arithmetic, so it RECORDS on the tape and
  the gradient flows through the chain that actually computed the value. There
  is no second derivative to maintain;
* every step preserves the exact sum. ``two_sum`` and ``two_product`` are
  error-free transformations, and the renormalisation below only ever replaces
  a pair by another pair with the same total, so the represented value is
  never silently damaged -- only the final truncation to ``n`` limbs discards
  anything, and it discards exactly what falls below the requested precision.

WHERE THIS STOPS BEING THE RIGHT TOOL. Multiplication is O(n**2) limb
products, and the renormalisation is O(n**2) sequential dependent sums. Around
eight limbs (~128 digits) that is already worse than carrying an integer
significand, and for the millions-of-digits regime the answer is fixed-point
limbs with FFT multiplication, not floating expansions. This is the right tool
from 2 to roughly 8.

The error-free transformations depend on the arithmetic NOT being
reassociated: an optimiser that "simplifies" ``a - (s - a)`` to zero deletes
the correction and silently collapses everything here to plain double. The
compiler must treat these as strict-FP, and recognising ``two_product`` to
fold it onto a fused multiply-add is a compiler identity -- both belong there,
not in workarounds here.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Sequence

# Dekker's splitting constant for binary64: 2**27 + 1.
_SPLIT = 134217729.0

# Beyond this the expansion is the wrong representation; see the module note.
SENSIBLE_LIMIT = 8

_state = threading.local()


# --------------------------------------------------------------------------
# the precision parameter


def limbs_for_digits(digits: int) -> int:
    """Limbs needed to carry ``digits`` decimal digits."""

    import math

    needed = int(math.ceil(float(digits) / 15.95))
    return int(min(max(needed, 1), SENSIBLE_LIMIT))


def active_limbs() -> int:
    """The ambient precision: 1 unless a ``precision`` block is open."""

    return int(getattr(_state, "limbs", 1) or 1)


@contextmanager
def precision(limbs: int):
    """Ask every operator inside this block for ``limbs``-double arithmetic.

    This is the "supplied or nonzero" switch: with it open the dispatch shim
    promotes plain operands and realises the requested precision; with it shut
    the shim declines and the ordinary path runs untouched.
    """

    limbs = int(limbs)
    if limbs < 1:
        raise ValueError(f"precision needs at least one limb, got {limbs}")
    previous = getattr(_state, "limbs", 1)
    _state.limbs = limbs
    try:
        yield limbs
    finally:
        _state.limbs = previous


def _expanding() -> bool:
    return getattr(_state, "active", False)


class _Expansion:
    """While held, the shim declines so limb arithmetic runs as plain ops."""

    def __enter__(self):
        self.previous = getattr(_state, "active", False)
        _state.active = True
        return self

    def __exit__(self, *_):
        _state.active = self.previous
        return False


# --------------------------------------------------------------------------
# error-free transformations


def two_sum(a, b):
    """Knuth: ``a + b == s + e`` exactly, for any a and b."""

    s = a + b
    shifted = s - a
    return s, (a - (s - shifted)) + (b - shifted)


def _split(a):
    """Dekker: halve a significand into non-overlapping pieces."""

    c = a * _SPLIT
    high = c - (c - a)
    return high, a - high


def two_product(a, b):
    """Dekker: ``a * b == p + e`` exactly.

    The splitting form, because it needs no operation the base dtype lacks.
    A backend with a fused multiply-add can do this in two instructions; that
    substitution is the compiler's job.
    """

    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl


# --------------------------------------------------------------------------
# expansions


def renormalise(terms: Sequence, limbs: int) -> list:
    """Reduce a pile of terms to ``limbs`` non-overlapping ones.

    Sweeping ``two_sum`` along the sequence pushes each rounding error one
    place down while leaving the TOTAL untouched, so repeating the sweep sorts
    magnitude into place without a single branch -- which is what makes this
    usable on a tensor, where the classic zero-skipping compress cannot go.

    The tail below ``limbs`` is folded into the last kept limb rather than
    dropped, so the result is the closest representable value rather than a
    truncation.
    """

    working = list(terms)
    if not working:
        return []

    # Distillation, branch-free. A FORWARD sweep is the wrong primitive: it
    # only ever lets a term see its neighbour, so m terms need m sweeps and
    # the cost goes quadratic. A BACKWARD accumulation lets the head see every
    # term in one pass, and its tail holds the exact remainder -- so repeating
    # it peels off one correctly-rounded limb at a time, O(m) per limb.
    kept = []
    rest = working
    for _ in range(limbs):
        if not rest:
            break
        carry = rest[-1]
        tail = [None] * (len(rest) - 1)
        for index in range(len(rest) - 2, -1, -1):
            carry, tail[index] = two_sum(rest[index], carry)
        kept.append(carry)
        rest = tail
    while len(kept) < limbs:
        kept.append(kept[-1] * 0.0)
    # Whatever is left lies below the requested precision; folding it into the
    # last limb keeps the result the nearest representable value.
    for leftover in rest:
        kept[-1] = kept[-1] + leftover
    return kept


def add_expansions(left: Sequence, right: Sequence, limbs: int) -> list:
    """Exact sum of two expansions, renormalised to ``limbs``."""

    return renormalise(list(left) + list(right), limbs)


def negate(terms: Sequence) -> list:
    return [-term for term in terms]


def multiply_expansions(left: Sequence, right: Sequence, limbs: int) -> list:
    """Every limb against every limb, exactly, then renormalised.

    Each partial product contributes BOTH halves of its error-free form, so no
    information is lost before the renormalisation decides what fits.
    """

    pieces = []
    for a in left:
        for b in right:
            high, low = two_product(a, b)
            pieces.append(high)
            pieces.append(low)
    return renormalise(pieces, limbs)


def _lead(terms: Sequence):
    """An approximation of an expansion's value from its top limbs.

    Two are enough: the distillation guarantees the rest are below the second
    limb's rounding, so this is accurate to about a limb -- which is exactly
    what a long-division digit needs.
    """

    if len(terms) == 1:
        return terms[0]
    return terms[0] + terms[1]


def divide_expansions(left: Sequence, right: Sequence, limbs: int) -> list:
    """Long division: take a leading quotient digit, subtract, repeat.

    Each pass gains one limb of the quotient, so ``limbs + 1`` passes give the
    requested precision with one to spare for the final renormalisation.
    """

    remainder = list(left)
    quotient = []
    for _ in range(limbs + 1):
        # The estimate must read the top TWO limbs, not just the first. The
        # branch-free distillation cannot skip zeros the way the classic
        # compress does, so an exact cancellation leaves a zero in the leading
        # slot with the real value one place down. Reading only slot zero then
        # yields a zero quotient digit and the pass is wasted -- which is why
        # division gained a limb only every other width.
        digit = _lead(remainder) / _lead(right)
        quotient.append(digit)
        product = multiply_expansions([digit], right, limbs + 2)
        remainder = add_expansions(remainder, negate(product), limbs + 2)
    return renormalise(quotient, limbs)


# --------------------------------------------------------------------------
# carrying limbs on a tensor


def limb_count(value: Any) -> int:
    return 1 + len(getattr(value, "_limbs", ()))


def carries_limbs(value: Any) -> bool:
    return bool(getattr(value, "_limbs", ()))


def attach(leading: Any, corrections: Sequence) -> Any:
    leading._limbs = tuple(corrections)
    return leading


def limbs_of(value: Any, width: int, like: Any) -> list:
    """Every operand as a list of exactly ``width`` limbs."""

    if hasattr(value, "_limbs") or hasattr(value, "shape"):
        terms = [value] + list(getattr(value, "_limbs", ()))
    else:
        terms = [like * 0.0 + value]
    zero = like * 0.0
    while len(terms) < width:
        terms.append(zero)
    return terms[:width]


def extended(value: Any, limbs: int = 2, correction: Any = None) -> Any:
    """Promote a plain tensor to an extended one.

    Returns a fresh head so promoting an operand never mutates it -- reusing a
    promoted tensor as though it were still plain is otherwise an easy and
    very confusing mistake.
    """

    with _Expansion():
        head = value + 0.0
        corrections = [value * 0.0 for _ in range(max(limbs - 1, 0))]
        if correction is not None and corrections:
            corrections[0] = correction
    return attach(head, corrections)


def constant(like: Any, high: float, low: float = 0.0,
             limbs: int = 2) -> Any:
    """An extended constant from a value a double could not hold.

    Baked coefficients are the obvious case: the exact Taylor coefficient of a
    core is not representable, so the bake keeps the remainder and this puts
    both halves back together as one extended operand.
    """

    with _Expansion():
        head = like * 0.0 + high
        rest = [like * 0.0 + low]
        rest += [like * 0.0 for _ in range(max(limbs - 2, 0))]
    return attach(head, rest)


def constant_limbs(like: Any, parts: Sequence) -> Any:
    """An extended constant from a value decomposed into any number of limbs.

    Two limbs cap a coefficient at about 32 digits. A core asked to converge
    below that needs its coefficients carried to whatever width the target
    implies, so the count is a parameter here rather than a constant.
    """

    with _Expansion():
        head = like * 0.0 + float(parts[0])
        rest = [like * 0.0 + float(part) for part in parts[1:]]
    return attach(head, rest)


def pair(high: Any, low: Any) -> Any:
    """An extended value from two tensors already holding the halves."""

    with _Expansion():
        head = high + 0.0
        rest = [low + 0.0]
    return attach(head, rest)


def collapse(value: Any) -> Any:
    """Round an extended value back to a single limb."""

    with _Expansion():
        total = value + 0.0
        for limb in getattr(value, "_limbs", ()):
            total = total + limb
    return total


def to_float_list(value: Any) -> list:
    """Exact Python floats of every limb, for handing to arbitrary precision."""

    limbs = [value] + list(getattr(value, "_limbs", ()))
    return [limb.tolist() for limb in limbs]


# --------------------------------------------------------------------------
# the dispatch shim


_DIRECT = {"add", "iadd", "sub", "isub", "mul", "imul", "truediv", "itruediv"}
_REVERSED = {"radd", "rsub", "rmul", "rtruediv"}
_HANDLED = _DIRECT | _REVERSED | {"neg"}


def apply(op: str, left: Any, right: Any):
    """The dispatch shim. Returns the extended result, or None to decline.

    Declining is the common case and costs a couple of attribute lookups, so a
    build that never asks for extended precision pays essentially nothing.
    """

    if _expanding() or op not in _HANDLED:
        return None

    width = max(active_limbs(), limb_count(left), limb_count(right))
    if width <= 1:
        return None
    if width > SENSIBLE_LIMIT:
        raise ValueError(
            f"{width} limbs asked of an expansion; past about {SENSIBLE_LIMIT} "
            f"this representation is the wrong one -- carry an integer "
            f"significand in fixed point instead"
        )

    like = left if hasattr(left, "shape") else right
    if not hasattr(like, "shape"):
        return None

    with _Expansion():
        if op == "neg":
            return attach(*_head_and_tail(negate(limbs_of(left, width, like))))
        first = limbs_of(left, width, like)
        second = limbs_of(right, width, like)
        if op in _REVERSED:
            first, second = second, first
        base = op[1:] if op in _REVERSED else op.lstrip("i")
        if base == "add":
            result = add_expansions(first, second, width)
        elif base == "sub":
            result = add_expansions(first, negate(second), width)
        elif base == "mul":
            result = multiply_expansions(first, second, width)
        else:
            result = divide_expansions(first, second, width)
    return attach(*_head_and_tail(result))


def _head_and_tail(terms: Sequence):
    return terms[0], tuple(terms[1:])
