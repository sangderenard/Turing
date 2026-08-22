"""N-double arithmetic as a shim in the operator dispatch.

An extended value is an ORDINARY TENSOR whose last dimension carries its
limbs, interleaved at stride ``limbs``: the value it denotes is the exact sum
of them, and every limb is a plain slice of the base dtype. Nothing here
introduces a new dtype, a new wrapper, or a new opcode -- every step is built
from ``+``, ``-`` and ``*`` on the base dtype, which is why this can sit in
the operator dispatch and serve every backend at once.

Limbs as CHANNELS is what makes it fit. The return type never changes -- an
operator hands back a tensor, as it always did, merely wider -- so no caller
learns a new type and no dispatch grows a case for one. A four-channel pixel
at two limbs is eight in the last dimension read with stride two, so RGBA
keeps striding and the precision rides alongside it. Widening is a shape
change and collapsing is a strided sum, both ordinary tensor work.

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

from typing import Any, Sequence

# Dekker's splitting constant for binary64: 2**27 + 1.
_SPLIT = 134217729.0

# Beyond this the expansion is the wrong representation; see the module note.
SENSIBLE_LIMIT = 8

def limbs_for_digits(digits: int) -> int:
    """Limbs needed to carry ``digits`` decimal digits."""

    import math

    needed = int(math.ceil(float(digits) / 15.95))
    return int(min(max(needed, 1), SENSIBLE_LIMIT))


# --------------------------------------------------------------------------
# the representation: limbs are CHANNELS, in the last dimension


def limb(value: Any, index: int, limbs: int) -> Any:
    """Limb ``index`` of an interleaved value -- stride ``limbs``, last axis.

    Interleaved rather than planar: every step of an expansion uses one
    value's limbs together, so they belong adjacent. A four-channel pixel at
    two limbs is eight in the last dimension read with stride two, which is
    why RGBA keeps working -- the channels are still there, just striding.
    """

    if int(limbs) <= 1:
        return value
    return value[..., int(index)::int(limbs)]


def interleave(parts: Sequence) -> Any:
    """``k`` parts of shape ``(..., C)`` into one ``(..., C*k)``."""

    from .abstraction import AbstractTensor

    parts = list(parts)
    if len(parts) == 1:
        return parts[0]
    stacked = AbstractTensor.stack(parts, dim=-1)
    shape = list(stacked.shape)
    return stacked.reshape(*shape[:-2], int(shape[-2]) * int(shape[-1]))


def widen(value: Any, limbs: int) -> Any:
    """Promote to ``limbs`` limbs: the value, then zeros beside it."""

    limbs = int(limbs)
    if limbs <= 1:
        return value
    zero = plain(value, "mul", 0.0)
    wide = interleave([value] + [zero] * (limbs - 1))
    wide.limbs = limbs
    return wide


def narrow(value: Any, limbs: int) -> Any:
    """Collapse the limbs back to one channel per value."""

    limbs = int(limbs)
    if limbs <= 1:
        return value
    total = limb(value, 0, limbs)
    for index in range(1, limbs):
        total = plain(total, "add", limb(value, index, limbs))
    total.limbs = 1
    return total


# --------------------------------------------------------------------------
# error-free transformations


def plain(left: Any, op: str, right: Any) -> Any:
    """One operator at a SINGLE limb -- the calculator's primitive layer.

    The transformations below are the implementation of extended precision, so
    they cannot themselves be extended: asking the calculator for its default
    width here would call this code to implement it, without end. Spelling the
    width explicitly says that outright, and needs no ambient flag to say it
    -- which matters because an ambient flag is exactly what does not survive
    into a compiled program.
    """

    from .abstraction import AbstractTensor

    return AbstractTensor._apply_operator(left, op, left, right, limbs=1)


def two_sum(a, b):
    """Knuth: ``a + b == s + e`` exactly, for any a and b."""

    s = plain(a, "add", b)
    shifted = plain(s, "sub", a)
    return s, plain(plain(a, "sub", plain(s, "sub", shifted)), "add",
                    plain(b, "sub", shifted))


def _split(a):
    """Dekker: halve a significand into non-overlapping pieces."""

    c = plain(a, "mul", _SPLIT)
    high = plain(c, "sub", plain(c, "sub", a))
    return high, plain(a, "sub", high)


def two_product(a, b):
    """Dekker: ``a * b == p + e`` exactly.

    The splitting form, because it needs no operation the base dtype lacks.
    A backend with a fused multiply-add can do this in two instructions; that
    substitution is the compiler's job.
    """

    p = plain(a, "mul", b)
    ah, al = _split(a)
    bh, bl = _split(b)
    error = plain(plain(ah, "mul", bh), "sub", p)
    error = plain(error, "add", plain(ah, "mul", bl))
    error = plain(error, "add", plain(al, "mul", bh))
    return p, plain(error, "add", plain(al, "mul", bl))


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
        kept.append(plain(kept[-1], "mul", 0.0))
    # Whatever is left lies below the requested precision; folding it into the
    # last limb keeps the result the nearest representable value.
    for leftover in rest:
        kept[-1] = plain(kept[-1], "add", leftover)
    return kept


def add_expansions(left: Sequence, right: Sequence, limbs: int) -> list:
    """Exact sum of two expansions, renormalised to ``limbs``."""

    return renormalise(list(left) + list(right), limbs)


def negate(terms: Sequence) -> list:
    return [plain(term, "mul", -1.0) for term in terms]


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
    return plain(terms[0], "add", terms[1])


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
        digit = plain(_lead(remainder), "truediv", _lead(right))
        quotient.append(digit)
        product = multiply_expansions([digit], right, limbs + 2)
        remainder = add_expansions(remainder, negate(product), limbs + 2)
    return renormalise(quotient, limbs)


# --------------------------------------------------------------------------
# carrying limbs on a tensor


def limbs_of(value: Any, width: int, like: Any) -> list:
    """An operand as ``width`` ordinary tensors, one per limb.

    A tensor operand is assumed ALREADY widened -- the promotion happens once
    at the boundary and the width travels with the value after that, so an
    operator does not re-promote and cannot double-count. A scalar has no
    channels to stride, so it becomes its own leading limb and zeros.

    What comes back are ordinary tensors with nothing attached, which is why
    the arithmetic below cannot re-enter this: each one asks its operators for
    the default single limb.
    """

    if hasattr(value, "shape"):
        # A tensor says how wide it already is. Slicing a plain tensor at
        # stride ``width`` would hand back an EMPTY limb for every channel
        # past the first, which is the shape error that makes this the one
        # thing a tensor cannot be left to guess about itself.
        held = int(getattr(value, "limbs", 1) or 1)
        if held < width:
            value = widen(narrow(value, held) if held > 1 else value, width)
        return [limb(value, index, width) for index in range(width)]
    seed = limb(like, 0, width)
    zero = plain(seed, "mul", 0.0)
    return ([plain(zero, "add", value)]
            + [zero for _ in range(width - 1)])


def to_float_list(value: Any, limbs: int) -> list:
    """Every limb as plain Python floats, for handing to exact arithmetic."""

    return [limb(value, index, limbs).tolist() for index in range(int(limbs))]


# --------------------------------------------------------------------------
# the dispatch shim


_DIRECT = {"add", "iadd", "sub", "isub", "mul", "imul", "truediv", "itruediv"}
_REVERSED = {"radd", "rsub", "rmul", "rtruediv"}
_HANDLED = _DIRECT | _REVERSED | {"neg"}


def apply(op: str, left: Any, right: Any, *, limbs: int = 1,
          accumulator: Any = None, accumulate_output: bool = False):
    """The limb work, driven by ARGUMENTS, returning an ordinary tensor.

    ``limbs`` is what the caller will ACCEPT, which is not what the operands
    carry. Two 2-limb values have a 4-limb exact product, and down a chain
    that grows without bound, so renormalising to ``limbs`` IS the precision
    choice: the operands say what is available, this says what is kept.

    It is a parameter rather than ambient state because ambient state does not
    survive compilation -- whatever sets it is a store nothing in the program
    reads, so it is deleted as dead and every operator lowers single-limb while
    appearing to have honoured the request.

    The result is a TENSOR, not a pair or a wrapper. Limbs live in the last
    dimension at stride ``limbs``, so widening is a shape change and every
    return type stays what it was. That is what lets this sit in the operator
    dispatch at all.

    ``accumulator`` takes the exact intermediate instead of it being
    renormalised away, so a chain pays the truncation once at the end rather
    than at every step; ``accumulate_output`` hands the accumulator back so
    the caller can keep chaining exactly.
    """

    if op not in _HANDLED:
        return None
    width = int(limbs or 1)
    if width <= 1 and accumulator is None:
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

    if op == "neg":
        pieces = negate(limbs_of(left, width, like))
    else:
        first = limbs_of(left, width, like)
        second = limbs_of(right, width, like)
        if op in _REVERSED:
            first, second = second, first
        base = op[1:] if op in _REVERSED else op.lstrip("i")
        if base == "add":
            pieces = add_expansions(first, second, width)
        elif base == "sub":
            pieces = add_expansions(first, negate(second), width)
        elif base == "mul":
            pieces = multiply_expansions(first, second, width)
        else:
            pieces = divide_expansions(first, second, width)

    if accumulator is not None:
        for piece in pieces:
            accumulator.absorb(piece)
        if accumulate_output:
            return accumulator
        pieces = accumulator.to_expansion(width)
    result = interleave(pieces)
    result.limbs = width
    return result




def constant(like: Any, parts: Sequence, limbs: int) -> Any:
    """A constant whose limbs the caller derived, laid out as channels."""

    from .abstraction import AbstractTensor

    seed = plain(limb(like, 0, limbs), "mul", 0.0)
    return interleave([plain(seed, "add", float(part))
                       for part in parts[:limbs]])
