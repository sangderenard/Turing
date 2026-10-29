"""N-double arithmetic owned by a precision-aware tensor wrapper.

An extended value stores an ordinary tensor whose last dimension carries its
limbs, interleaved at stride ``limbs``: the value it denotes is the exact sum
of them, and every limb is a plain slice of the base dtype. ``Precision`` owns
that representation so an unaware tensor operation cannot accidentally treat
limbs as ordinary channels. Every numerical step is still built from the base
tensor's ``+``, ``-`` and ``*`` operations, so operator dispatch and recording
remain backend-neutral.

Limbs as CHANNELS is what makes the storage fit existing tensor backends. The
wrapper remains visible until an explicit collapse, while its payload is an
ordinary wider tensor. A four-channel pixel at two limbs is eight in the last
dimension read with stride two, so RGBA keeps striding and the precision rides
alongside it. Widening is a shape change and collapsing is a strided sum, both
ordinary tensor work.

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

import math
from typing import Any, Sequence

#: Everything about a limb that depends on which float carries it, stated
#: once. Two elements exist because two worlds do: the CPU lanes hold limbs
#: in binary64, while WGSL and GLSL have no float64 at all, so their limbs
#: are binary32 and there are roughly twice as many of them for the same
#: precision.
#:
#: * ``split`` is Veltkamp's constant ``2**ceil(t/2) + 1`` for a t-bit
#:   significand -- splitting with the wrong one does not degrade Dekker's
#:   product, it invalidates the theorem.
#: * ``digits_per_limb`` is ``t * log10(2)``, used to size orders and the
#:   exact-rational working precision.
#: * ``max_limbs`` is the conservative automatic-planning default retained
#:   for compatibility; it is not a repository-lowering limit. ``ladder`` is
#:   the widths a lane recommends. binary64 walks 2/3/4; binary32
#:   walks 2/4/6/8 -- even steps, because each rung is priced against the
#:   binary64 rung it replaces (two f32 limbs per f64 limb) and odd rungs
#:   would be tiers nothing on the other side corresponds to.
#: * ``sensible_limit`` is where the expansion stops being the right
#:   representation at all (see the module note); it scales with limb
#:   count, not with carried bits, because the O(n**2) cost does.
LIMB_ELEMENTS: dict[str, dict] = {
    "float64": {
        "split": 134217729.0,  # 2**27 + 1
        "digits_per_limb": 15.95,
        "max_limbs": 4,
        "ladder": (2, 3, 4),
        "sensible_limit": 8,
    },
    "float32": {
        "split": 4097.0,  # 2**12 + 1
        "digits_per_limb": 7.22,
        "max_limbs": 8,
        "ladder": (2, 4, 6, 8),
        "sensible_limit": 16,
    },
    # INTEGER limbs: a Python ``int`` carried as a bigint. The same type,
    # the same planar high-first stack, the same ``limbs`` width -- only the
    # carrier differs, so only the facts a carrier owns differ.
    #
    # * ``container`` is the integer dtype a limb is held in. ``int128``
    #   exists only where a lane spells it (``dtype_layout.DTYPES``); NumPy
    #   spells none, so the eager surface carries ``int64`` limbs.
    # * ``digit_bits`` is the integer counterpart of Veltkamp's half
    #   significand: one bit short of half the container, so a limb
    #   product (``integer_split``) is exact in the container with room for
    #   the column sums a multiply accumulates.
    # * ``digits_per_limb`` is ``digit_bits * log10(2)``, as for floats.
    #
    # ``split`` (a float splitting constant) and the error-tolerance
    # planning rows ``max_limbs``/``ladder`` have no integer meaning: an
    # integer limb never rounds, so there is no error to plan a width
    # against. ``sensible_limit`` is where a floating expansion should give
    # way to an integer significand (module note); this IS one, so None.
    "int64": {
        "container": "int64",
        "digit_bits": 31,
        "digits_per_limb": 9.33,
        "sensible_limit": None,
    },
    "int128": {
        "container": "int128",
        "digit_bits": 63,
        "digits_per_limb": 18.96,
        "sensible_limit": None,
    },
}

_ELEMENT_SPELLINGS = {
    "float64": "float64", "f64": "float64", "double": "float64",
    "float32": "float32", "f32": "float32", "single": "float32",
    "int64": "int64", "i64": "int64",
    "int128": "int128", "i128": "int128",
}

def integer_limb_element(element: Any) -> bool:
    """Whether ``element`` names an integer limb element: its
    ``LIMB_ELEMENTS`` row is the one that carries ``digit_bits``."""

    name = _ELEMENT_SPELLINGS.get(str(element).casefold())
    return name is not None and "digit_bits" in LIMB_ELEMENTS[name]


def limb_element_name(element: Any = None) -> str:
    """The canonical name of a limb element.

    ``None`` means the value never had a dtype stamped, which today is
    always the binary64 default the scalar path assumes. A PRESENT but
    unrecognised element is refused: guessing a splitting constant produces
    plausible numbers whose residuals are quietly wrong.
    """

    if element is None:
        return "float64"
    name = _ELEMENT_SPELLINGS.get(str(element).casefold())
    if name is None:
        raise ValueError(
            f"no limb arithmetic facts for element {element!r}; "
            "supported elements are float64, float32, int64 and int128"
        )
    return name


def limb_element_facts(element: Any = None) -> dict:
    """The facts row for a limb element (see ``limb_element_name``)."""

    return LIMB_ELEMENTS[limb_element_name(element)]


def _complex_dtype(value: Any) -> bool:
    """Whether ``value`` uses a native complex scalar dtype.

    A floating expansion's error-free transforms apply to one real IEEE
    operation. Native complex multiplication is already a compound of real
    operations, so it cannot be treated as one error-free limb operation.
    """

    dtype = getattr(value, "dtype", None)
    if dtype is None:
        try:
            dtype = value.get_dtype()
        except (AttributeError, NotImplementedError):
            dtype = getattr(getattr(value, "data", None), "dtype", None)
    return "complex" in str(dtype).casefold()


def _dtype_of(value: Any):
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        try:
            dtype = value.get_dtype()
        except (AttributeError, NotImplementedError):
            dtype = getattr(getattr(value, "data", None), "dtype", None)
    return dtype


def _integer_dtype(value: Any) -> bool:
    """Whether ``value`` is a signed-integer tensor (an integer limb carrier).

    A Python ``int`` scalar is not a tensor and is handled where it meets
    one (``_python_int_limbs``); ``bool`` is not an integer here.
    """

    from ...transmogrifier.dtype_layout import resolve

    if not hasattr(value, "shape"):
        return False
    layout = resolve(str(_dtype_of(value)))
    return layout is not None and layout.kind == "int"


def _as_int64(value: Any) -> Any:
    """An integer tensor in the eager integer container (``LIMB_ELEMENTS``
    ``int64`` row): NumPy spells no int128, so the eager lane is int64."""

    from ...transmogrifier.dtype_layout import resolve

    container = LIMB_ELEMENTS["int64"]["container"]
    layout = resolve(str(_dtype_of(value)))
    return value if layout is not None and layout.name == container else value.long()


def _any_true(predicate: Any) -> bool:
    held = predicate.tolist() if hasattr(predicate, "tolist") else predicate
    return any(bool(each) for each in _flatten(held))


def _raise_integer_overflow(predicate: Any, where: str) -> None:
    """``INT_OVERFLOW`` where an integer result left its limbs: Python's
    int never loses digits, so the value is refused, never truncated."""

    if _any_true(predicate):
        from ...compiler.native_faults import INT_OVERFLOW, NativeIntOverflow

        raise NativeIntOverflow(
            f"{where}: the integer result does not fit its Precision limbs "
            f"(fault {INT_OVERFLOW}); a wider Precision holds it exactly"
        )


def _raise_zero_division(predicate: Any, where: str) -> None:
    if _any_true(predicate):
        from ...compiler.native_faults import (
            INT_ZERO_DIVISION, NativeZeroDivisionError,
        )

        raise NativeZeroDivisionError(
            f"{where}: integer division or modulo by zero "
            f"(fault {INT_ZERO_DIVISION})"
        )


# Dekker's splitting constant for binary64: 2**27 + 1. The binary64 alias
# survives because the eager path below IS binary64; element-aware callers
# read ``limb_element_facts`` instead.
_SPLIT = LIMB_ELEMENTS["float64"]["split"]

# Beyond this the expansion is the wrong representation; see the module note.
SENSIBLE_LIMIT = LIMB_ELEMENTS["float64"]["sensible_limit"]

def limbs_for_digits(digits: int, element: Any = None) -> int:
    """Limbs needed to carry ``digits`` decimal digits."""

    import math

    facts = limb_element_facts(element)
    needed = int(math.ceil(float(digits) / facts["digits_per_limb"]))
    if facts["sensible_limit"] is None:
        return int(max(needed, 1))
    return int(min(max(needed, 1), facts["sensible_limit"]))


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

    The transformations below are the IMPLEMENTATION of extended precision, so
    they cannot themselves be extended. They operate on limb slices, which are
    ordinary tensors, and dispatch only diverts for a ``Precision`` -- so this
    is a plain operator call and cannot re-enter. Spelling it out says the
    arithmetic here is deliberately primitive rather than incidentally so.
    """

    from .abstraction import AbstractTensor

    return AbstractTensor._apply_operator(left, op, left, right)


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
# integer expansions: a Python ``int`` on integer limbs
#
# The same representation as above, with an integer carrier. A float limb
# carries its own weight in its exponent; an integer limb has none, so its
# weight is its place: high limb first, limb ``k`` of ``n`` weighs
# ``B**(n-1-k)`` with ``B = 2**digit_bits``, and the value is the exact sum
# of the weighted limbs. Non-overlapping is the integer reading of the
# distilled form: every limb below the head lies in ``[0, B)`` and the head,
# which carries the sign, in ``[-B, B)``. ``n`` limbs therefore hold
# exactly ``[-B**n, B**n)``.
#
# Every step keeps the exact total, as the float steps do. The error-free
# transformations become carries: ``integer_split`` writes a limb product
# as ``high * B + low`` exactly (``two_product``'s counterpart), and a carry
# sweep replaces a column and its neighbour by a pair with the same total
# (the distillation's). Nothing can fall BELOW an integer's precision, so
# the one place a float expansion discards -- the tail past ``limbs`` --
# has no counterpart; what can be lost is at the TOP, and that is reported
# as a predicate (``native_faults.INT_OVERFLOW``), never truncated.
#
# Each kernel returns its predicates beside its limbs, as the compiler twin
# (``ir_identities.WideExpansionEmitter``) does; ``Precision`` raises on
# them. Kernels are written in + - * & >> == < and ``where`` only, the
# operations every integer destination has, step for step as the twin
# spells them.


def _int_where(mask, when_true, when_false):
    from .abstraction import AbstractTensor

    return AbstractTensor.where(mask, when_true, when_false)


def _int_align(terms: Sequence, count: int) -> list:
    """``terms`` at ``count`` limbs: zeros above, so the total is unchanged."""

    terms = list(terms)
    zero = terms[0] * 0
    return [zero] * (count - len(terms)) + terms


def integer_split(product, digit_bits: int):
    """``product == high * B + low`` exactly, ``low`` in ``[0, B)``.

    The integer ``two_product`` residual: ``>>`` floors and ``&`` keeps the
    low ``digit_bits`` of the two's-complement value, so the pair is exact
    for either sign.
    """

    return product >> digit_bits, product & ((1 << digit_bits) - 1)


def integer_carry(columns: Sequence, digit_bits: int) -> list:
    """One carry sweep, low limb to high: the same total, every limb below
    the head in ``[0, B)``, the head signed. The distillation sweep's
    counterpart -- each step replaces a column and the carry into the next
    by a pair with the same total.
    """

    digits = list(columns)
    carry = None
    for index in range(len(digits) - 1, 0, -1):
        column = digits[index] if carry is None else digits[index] + carry
        carry, digits[index] = integer_split(column, digit_bits)
    if carry is not None:
        digits[0] = digits[0] + carry
    return digits


def integer_renormalise(columns: Sequence, limbs: int, digit_bits: int):
    """Reduce place-valued columns to ``limbs`` non-overlapping limbs.

    Returns ``(limbs, overflow)``. The columns are carried at one limb more
    than kept, so the part above the kept limbs is a signed excess; the
    total fits ``limbs`` exactly when that excess is 0 or -1 (the head then
    absorbs it), and ``overflow`` is true where it is anything else -- the
    value left ``[-B**limbs, B**limbs)`` and nothing was kept in its place.
    """

    working = list(columns)
    while len(working) < int(limbs) + 1:
        working.insert(0, working[0] * 0)
    digits = integer_carry(working, digit_bits)
    mask = (1 << digit_bits) - 1
    cut = len(digits) - int(limbs)
    excess, kept = digits[:cut], digits[cut:]
    held_zero = excess[0] == 0
    held_minus = excess[0] == -1
    for each in excess[1:]:
        held_zero = held_zero & (each == 0)
        held_minus = held_minus & (each == mask)
    head = _int_where(held_minus, kept[0] - mask - 1, kept[0])
    return [head, *kept[1:]], ~(held_zero | held_minus)


def integer_negate(terms: Sequence) -> list:
    """Each limb negated: the same columns with the total's sign flipped.
    Not yet non-overlapping; the consumer renormalises."""

    return [term * 0 - term for term in terms]


def integer_add_expansions(left: Sequence, right: Sequence, limbs: int,
                           digit_bits: int):
    """Exact sum, column by column, renormalised: ``(limbs, overflow)``."""

    count = max(len(left), len(right))
    columns = [
        a + b for a, b in zip(_int_align(left, count), _int_align(right, count))
    ]
    return integer_renormalise(columns, limbs, digit_bits)


def integer_multiply_expansions(left: Sequence, right: Sequence, limbs: int,
                                digit_bits: int):
    """Every limb against every limb, each product split exactly into the
    two columns it spans, then renormalised: ``(limbs, overflow)``."""

    count = len(left) + len(right)
    columns = [None] * count

    def deposit(position, value):
        columns[position] = (
            value if columns[position] is None else columns[position] + value
        )

    for i, a in enumerate(left):
        for j, b in enumerate(right):
            high, low = integer_split(a * b, digit_bits)
            deposit(i + j + 1, low)
            deposit(i + j, high)
    return integer_renormalise(columns, limbs, digit_bits)


def integer_compare(left: Sequence, right: Sequence, digit_bits: int):
    """``(negative, zero)`` of ``left - right``, exactly.

    The difference is carried at one limb more than either operand, where
    it cannot overflow; a non-overlapping value is negative exactly when its
    head is, and zero exactly when every limb is.
    """

    count = max(len(left), len(right)) + 1
    digits = integer_carry([
        a - b
        for a, b in zip(_int_align(left, count), _int_align(right, count))
    ], digit_bits)
    zero = digits[0] == 0
    for each in digits[1:]:
        zero = zero & (each == 0)
    return digits[0] < 0, zero


def integer_divmod_expansions(left: Sequence, right: Sequence, limbs: int,
                              digit_bits: int):
    """Python's floor ``//`` and ``%``: ``(quotient, remainder, zero_divisor,
    overflow)``.

    Restoring long division on the magnitudes, one bit of the dividend at a
    time, every step exact integer arithmetic; then Python's floor rule
    (the remainder takes the divisor's sign). The only quotient that can
    leave ``limbs`` is ``-B**limbs // -1``, reported as ``overflow``.
    ``zero_divisor`` is the ``ZeroDivisionError`` predicate; the arithmetic
    under it is never a native division, so it cannot trap.
    """

    mask = (1 << digit_bits) - 1
    count = int(limbs) + 1
    dividend = integer_carry(_int_align(left, count), digit_bits)
    divisor = integer_carry(_int_align(right, count), digit_bits)
    dividend_negative = dividend[0] < 0
    divisor_negative = divisor[0] < 0
    magnitude = integer_carry([
        _int_where(dividend_negative, digit * 0 - digit, digit)
        for digit in dividend
    ], digit_bits)
    divisor_magnitude = integer_carry([
        _int_where(divisor_negative, digit * 0 - digit, digit)
        for digit in divisor
    ], digit_bits)
    zero_divisor = divisor_magnitude[0] == 0
    for digit in divisor_magnitude[1:]:
        zero_divisor = zero_divisor & (digit == 0)

    zero = magnitude[0] * 0
    one = zero + 1
    remainder = [zero] * count
    quotient = [None] * count
    # A magnitude of ``limbs`` limbs is at most ``B**limbs``, so its top
    # limb is 0 or 1 and only that limb's lowest bit can be set.
    positions = [(0, 0)] + [
        (index, shift)
        for index in range(1, count)
        for shift in range(digit_bits - 1, -1, -1)
    ]
    for index, shift in positions:
        bit = (magnitude[index] >> shift) & 1
        shifted = []
        for place in range(count):
            incoming = (
                bit if place == count - 1
                else remainder[place + 1] >> (digit_bits - 1)
            )
            shifted.append(((remainder[place] * 2) & mask) + incoming)
        trial = integer_carry(
            [r - d for r, d in zip(shifted, divisor_magnitude)], digit_bits,
        )
        fits = trial[0] >= 0
        remainder = [_int_where(fits, t, r) for t, r in zip(trial, shifted)]
        taken = _int_where(fits, one, zero) * (1 << shift)
        quotient[index] = (
            taken if quotient[index] is None else quotient[index] + taken
        )

    opposite = dividend_negative != divisor_negative
    remainder_zero = remainder[0] == 0
    for digit in remainder[1:]:
        remainder_zero = remainder_zero & (digit == 0)
    adjust = opposite & ~remainder_zero
    quotient = [_int_where(opposite, d * 0 - d, d) for d in quotient]
    quotient[-1] = _int_where(adjust, quotient[-1] - 1, quotient[-1])
    remainder = [
        _int_where(dividend_negative, d * 0 - d, d) for d in remainder
    ]
    remainder = [
        _int_where(adjust, r + d, r) for r, d in zip(remainder, divisor)
    ]
    quotient, quotient_overflow = integer_renormalise(
        quotient, limbs, digit_bits,
    )
    remainder, remainder_overflow = integer_renormalise(
        remainder, limbs, digit_bits,
    )
    return (
        quotient, remainder, zero_divisor,
        quotient_overflow | remainder_overflow,
    )


def integer_int64_limbs(digit_bits: int) -> int:
    """Limbs that hold every int64: ``B**(n-1)`` times a head in ``[-B, B)``
    must reach ``2**63``."""

    return -(-63 // int(digit_bits))


def integer_from_int64(value, limbs: int, digit_bits: int):
    """An int64 as ``limbs`` integer limbs: ``(limbs, overflow)``, exactly.
    ``overflow`` where the value does not fit ``limbs``."""

    return integer_renormalise([value], limbs, digit_bits)


def integer_to_int64(terms: Sequence, digit_bits: int):
    """The value as one int64: ``(value, outside)``, exactly.

    ``outside`` where it is not in ``[-2**63, 2**63)`` -- the stage-1
    narrowing rule; there the value is 0 and must not be read. Built by
    Horner from limbs whose head is already bounded, so no step can wrap.
    """

    count = integer_int64_limbs(digit_bits)
    held, outside = integer_renormalise(terms, count, digit_bits)
    head = held[0]
    head_bits = 63 - (count - 1) * int(digit_bits)
    if head_bits < int(digit_bits):
        bound = 1 << head_bits
        outside = outside | (head < -bound) | (head >= bound)
    head = _int_where(outside, head * 0, head)
    value = head
    for digit in held[1:]:
        value = value * (1 << digit_bits) + _int_where(
            outside, digit * 0, digit,
        )
    return value, outside


def _python_int_limbs(number: int, digit_bits: int) -> list:
    """An exact Python ``int`` as the fewest non-overlapping limbs."""

    base = 1 << int(digit_bits)
    rest, low = int(number), []
    while not -base <= rest < base:
        rest, digit = divmod(rest, base)
        low.append(digit)
    return [rest, *reversed(low)]


# --------------------------------------------------------------------------
# carrying limbs on a tensor



def _flatten(nested):
    """Every leaf of a nested list, in order."""

    if isinstance(nested, list):
        for item in nested:
            yield from _flatten(item)
    else:
        yield nested


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

#: The operators integer limbs carry: Python's ``int`` arithmetic minus
#: true division (an ``int / int`` is a float in Python).
_INTEGER_ORDER = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})
_INTEGER_HANDLED = frozenset({
    "add", "sub", "mul", "floordiv", "mod", "neg",
}) | _INTEGER_ORDER
_REVERSED_INTEGER = frozenset({"radd", "rsub", "rmul", "rfloordiv", "rmod"})


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

    # ``like`` is ONE channel wide already -- a caller hands a limb slice, not
    # an interleaved value -- so slicing here would halve it again.
    seed = plain(like, "mul", 0.0)
    return interleave([plain(seed, "add", part) for part in parts[:limbs]])


# --------------------------------------------------------------------------
# The type that owns the width


class Precision:
    """An AbstractTensor plus the limb channels that give it its width.

    WHY THIS IS A TYPE and not a flag on every tensor. The limbs live in extra
    channels, and a general tensor operation cannot be taught to see past
    channels it does not know about. Measured on a two-limb value: ``mean``
    returned half of it, because it divided by the widened element count.
    ``sum``, ``max``, ``abs`` and ``reshape`` happened to be right, which is
    worse -- it means the failures are scattered and silent rather than
    systematic and loud.

    So the width is owned by a type that also owns the operators that
    understand it. A `Precision` supports the basic unary and binary
    arithmetic and nothing else; anything outside that either collapses at the
    boundary or raises. Being unsupported is a fine answer, being quietly
    halved is not.

    Anything that wants enduring precision therefore does two things: make the
    operands `Precision`, and stay within the basic operators. That is the
    whole contract.
    """

    #: How two widths combine. KEEP-WIDEST: an operation touching a 3-limb
    #: operand yields 3 limbs, so precision is never silently discarded by
    #: meeting a narrower value. The alternative rules -- take the narrower,
    #: or track significant digits -- are policies this deliberately does not
    #: choose for the caller, because dropping precision is the kind of thing
    #: that should be asked for rather than inherited.
    COMBINE = "widest"

    __slots__ = ("_stack", "limbs")

    @classmethod
    def __class_getitem__(cls, width):
        """Make authored ``Precision[n]`` annotations valid Python objects."""

        import types

        return types.GenericAlias(cls, width)

    def __init__(self, value: Any, limbs: int):
        # PLANAR STORAGE. The limbs are held as one contiguous tensor per
        # limb, not as channels strided through a single payload.
        #
        # Interleaving was never chosen on its merits: the compiler's own
        # arithmetic has always been one SSA value per limb, and the only
        # thing that needed a single object was an ARRAY, which a planar
        # stack provides just as well. What interleaving cost was paid on
        # every access -- a limb was a strided view, so every operation on
        # it read memory with a gap -- and measured on the eager path,
        # planar limbs run about 1.35x faster at every size from a
        # thousand elements to a million.
        #
        # It also stops the shape from lying. An interleaved payload is a
        # perfectly ordinary tensor of n*limbs elements, so an operation
        # that does not know about limbs computes confidently on the wrong
        # count -- which is exactly how ``mean`` came to return half of a
        # two-limb value. A planar stack keeps every limb the shape the
        # caller declared.
        #
        # A sequence is taken as the terms themselves; a tensor is taken
        # as an interleaved payload and split once, so every existing
        # caller and every stored artifact keeps working unchanged.
        from .abstraction import AbstractTensor

        self.limbs = int(limbs)
        if isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            parts = [
                limb(value, index, self.limbs) for index in range(self.limbs)
            ]
        if any(
            _complex_dtype(part) or isinstance(part, complex)
            for part in parts
        ):
            raise TypeError(
                "Precision limbs must be real IEEE tensors; native complex "
                "multiplication is compound and cannot be used as an "
                "error-free limb operation. Use ComplexPrecision so real "
                "and imaginary coefficients are widened independently."
            )
        # The limbs are a LEADING AXIS, not a tuple and not channels.
        #
        # A tuple is planar but it is not a tensor, so the value has no
        # shape of its own and nothing can reason about it without
        # unpacking first. Channels were worse: they made shape[-1] a lie,
        # so every operation that did not know about limbs computed
        # confidently on limbs*n elements. A leading axis is the only
        # arrangement where the width is an ordinary dimension -- shape[0]
        # is the limb count, shape[1:] is exactly what the caller declared
        # -- so widening a program does not disturb a single shape
        # assumption below it, and the trailing axes keep meaning what
        # they always meant.
        self._stack = AbstractTensor.stack(
            [AbstractTensor.get_tensor(part) for part in parts], dim=0,
        )

    # -- crossing the boundary --------------------------------------------

    @classmethod
    def constant(cls, like: "Precision", parts: Sequence) -> "Precision":
        """A constant the caller derived to more digits than a double holds.

        The whole point of a wide chain is defeated if the values entering it
        are single doubles. An exact coefficient is not representable -- the
        sine core's second term is -1/6, whose remainder is -9.25e-18 -- so
        adding ``float(-1/6)`` at every Horner step reintroduces exactly the
        error the width was bought to avoid.
        """

        seed = like.term(0)
        return cls(
            [plain(plain(seed, "mul", 0.0), "add", part)
             for part in list(parts)[:like.limbs]],
            like.limbs,
        )

    @classmethod
    def of(cls, value: Any, limbs: int = 2) -> "Precision":
        """Promote an ordinary tensor. This is where width is decided."""

        width = max(int(limbs), 1)
        if _complex_dtype(value) or isinstance(value, complex):
            raise TypeError(
                "Precision.of() accepts real coefficients only. Use "
                "ComplexPrecision.of() to widen real and imaginary "
                "coefficients independently."
            )
        if _integer_dtype(value):
            # An int tensor is an integer expansion: the value is split into
            # place-valued limbs exactly, or the promotion is refused where
            # it does not fit ``limbs`` (``INT_OVERFLOW``).
            digit_bits = limb_element_facts("int64")["digit_bits"]
            terms, overflow = integer_from_int64(
                _as_int64(value), width, digit_bits,
            )
            _raise_integer_overflow(overflow, f"Precision.of(..., {width})")
            return cls(terms, width)
        zero = plain(value, "mul", 0.0)
        return cls([value] + [zero] * (width - 1), width)

    @property
    def integer(self) -> bool:
        """Whether the limbs are an integer element (a Python ``int``)."""

        return _integer_dtype(self.term(0))

    def _integer_only(self, name: str) -> None:
        if not self.integer:
            raise TypeError(
                f"Precision.{name} is defined on integer limbs; a floating "
                "expansion has no exact floor division, modulus or order"
            )

    def _float_only(self, name: str) -> None:
        if self.integer:
            raise TypeError(
                f"Precision does not endorse {name!r} on integer limbs. "
                "Integer Precision supports + - * // % neg, comparisons "
                "and collapse() to an exact int64."
            )

    def collapse(self) -> Any:
        """Back to an ordinary tensor, paying the rounding once.

        Integer limbs have no rounding to pay: the value is recombined
        exactly into one int64, and a value outside int64 is refused
        (``INT_OVERFLOW``, the stage-1 narrowing rule) rather than wrapped.
        """

        if self.integer:
            digit_bits = limb_element_facts("int64")["digit_bits"]
            value, outside = integer_to_int64(self.terms(), digit_bits)
            _raise_integer_overflow(outside, "Precision.collapse() to int64")
            return value
        total = self.term(0)
        for index in range(1, self.limbs):
            total = plain(total, "add", self.term(index))
        return total

    @property
    def value(self) -> Any:
        """The ordinary tensor this denotes -- COLLAPSED, never the wide one.

        Reaching for ``.value`` is what a caller naturally does, so it gives
        back something that is safe to hand anywhere: a plain tensor of the
        expected shape, rounded once. The interleaved form stays internal,
        because passing THAT to a general operation is the failure being
        contained -- it sees channels it cannot know are limbs, and divides by
        the wrong count.

        Losing the extra digits here is the point. It happens at a named
        boundary, in one place, rather than silently in whichever operation
        happened not to understand the layout.
        """

        return self.collapse()

    # -- layout: a boundary fact, declared per destination ---------------
    #
    # There is no universally best arrangement of limbs in a buffer, and
    # the measurements say so plainly. A compiled per-element kernel wants
    # ELEMENT-MAJOR ("interleaved"): element i's limbs land on one cache
    # line, and on the CPU lanes that measured 1.2x to 1.7x faster than
    # the alternative. A GPU wants LIMB-MAJOR ("blocked"): adjacent
    # invocations then read adjacent addresses instead of addresses a
    # stride apart, which is what coalescing rewards. The eager surface
    # wants neither, because it holds its limbs planar and never packs
    # them at all.
    #
    # So the layout is not a property of the value. It is a property of
    # the DESTINATION, chosen where the value crosses into one, and the
    # only thing this type owes is an exact conversion in both directions.

    def interleaved(self):
        """Element-major: element i, limb k at flat index ``i * limbs + k``.

        What every compiled kernel in this tree addresses today, and what
        the artifacts and feeds already on disk contain.
        """

        # The limb axis moved to the END, then flattened. Both layouts
        # are the SAME STACK in two axis orders, which is the whole
        # argument for holding the limbs as a dimension rather than as a
        # tuple of arrays: a destination is handed the memory it wants by
        # permuting an axis and flattening, never by packing element by
        # element.
        stack = self._stack
        order = tuple(range(1, len(tuple(stack.shape)))) + (0,)
        return stack.transpose(*order).reshape(-1)

    def blocked(self):
        """Limb-major: element i, limb k at flat index ``k * count + i``.

        Each limb contiguous. Offered because a dispatch-parallel
        destination reads it better, not because anything here prefers it.
        """

        # Already the stack's own axis order: limb zero's elements, then
        # limb one's, and so on. Flattening is the whole conversion.
        return self._stack.reshape(-1)

    @classmethod
    def from_interleaved(cls, value, limbs: int) -> "Precision":
        """Adopt an element-major buffer without copying its meaning."""

        return cls(value, int(limbs))

    @classmethod
    def from_blocked(cls, value, limbs: int) -> "Precision":
        """Adopt a limb-major buffer, splitting it back into terms."""

        from .abstraction import AbstractTensor

        flat = AbstractTensor.get_tensor(value).reshape(-1)
        width = max(int(limbs), 1)
        count = int(flat.shape[0]) // width
        return cls(
            [flat[position * count:(position + 1) * count]
             for position in range(width)],
            width,
        )

    @property
    def shape(self):
        """The shape the CALLER declared: the stack without its limb axis.

        This is the whole point of holding the limbs as a leading axis. A
        two-limb field of a hundred elements reports a hundred, not two
        hundred and not a tuple of two arrays, so every shape assumption
        written against the narrow program still holds when it is widened.
        """

        return tuple(self._stack.shape)[1:]

    @property
    def stacked(self):
        """The payload: limbs on axis zero, the declared shape after it."""

        return self._stack

    @property
    def _value(self):
        """The interleaved payload, built on demand.

        Kept because the compiled ABI is interleaved -- a per-element
        kernel wants one element's limbs adjacent -- and because artifacts
        and feeds already on disk are written that way. It is a boundary
        format now, not the storage: nothing inside this type reads it.
        """

        return interleave(self.terms())

    def to_float_lists(self) -> list:
        return [self._stack[index].tolist() for index in range(self.limbs)]

    # -- the representation ------------------------------------------------

    def term(self, index: int) -> Any:
        from .abstraction import AbstractTensor

        # Indexing a zero-dimensional eager backend stack may unwrap the limb
        # to a NumPy scalar.  Re-enter through the tensor boundary so scalar
        # equation parameters retain operator dispatch just like vector limbs.
        return AbstractTensor.get_tensor(self._stack[int(index)])

    def __getitem__(self, key) -> "Precision":
        """Select elements, every limb with the same key.

        The limbs are ordinary tensors of the declared shape, so indexing
        each one alike selects whole expansions: no limb of an element is
        separated from the others.
        """

        return Precision([term[key] for term in self.terms()], self.limbs)

    def terms(self, width: int | None = None) -> list:
        width = self.limbs if width is None else int(width)
        if width == self.limbs:
            return [self.term(index) for index in range(self.limbs)]
        if self.integer:
            # Place-valued limbs widen ABOVE (zeros of higher weight leave
            # the total unchanged) and narrow only by renormalising, which
            # refuses a value the narrower width cannot hold.
            held = [self.term(index) for index in range(self.limbs)]
            if width > self.limbs:
                return _int_align(held, width)
            digit_bits = limb_element_facts("int64")["digit_bits"]
            narrowed, overflow = integer_renormalise(held, width, digit_bits)
            _raise_integer_overflow(overflow, f"Precision.terms({width})")
            return narrowed
        if width < self.limbs:
            return [self.term(index) for index in range(width)]
        held = [self.term(index) for index in range(self.limbs)]
        zero = plain(held[0], "mul", 0.0)
        return held + [zero] * (width - self.limbs)

    def __repr__(self) -> str:
        return f"Precision(limbs={self.limbs})"

    def __getattr__(self, name):
        """Refuse every tensor method this type has not endorsed.

        The interleaved value is deliberately not reachable. Handing it to a
        general tensor operation is the whole failure being contained -- that
        operation sees channels it does not know are limbs, and ``mean``
        divides by the wrong count while ``sum`` happens to survive. A caller
        that wants ordinary tensor work should ``collapse()`` first and mean
        it.
        """

        raise AttributeError(
            f"Precision does not endorse {name!r}. Use the basic arithmetic "
            f"operators, or collapse() for an ordinary tensor -- reaching "
            f"past this hands limb channels to something that cannot see them."
        )

    # -- arithmetic --------------------------------------------------------

    @staticmethod
    def width_of(operand: Any) -> int:
        return operand.limbs if isinstance(operand, Precision) else 1

    @classmethod
    def dispatch(cls, op: str, left: Any, right: Any, *, limbs: int = 1,
                 accumulator: Any = None,
                 accumulate_output: bool = False) -> "Precision":
        """One operator at a stated width, on operands of any width."""

        if any(
            isinstance(each, Precision) and each.integer
            for each in (left, right)
        ):
            return cls._integer_dispatch(op, left, right, limbs=limbs)
        if op not in _HANDLED:
            raise TypeError(
                f"Precision supports the basic arithmetic operators; {op!r} "
                f"is not one of them. Collapse first if that is what you mean "
                f"-- a wide value put through an operation that cannot see "
                f"its limbs comes back wrong rather than refused."
            )
        width = max(int(limbs or 1), cls.width_of(left), cls.width_of(right))
        # ``term(0)`` is ONE channel -- element-shaped, not interleaved --
        # so it must never be handed to ``limbs_of``, whose scalar branch
        # slices its template again and halves it (measured: a three-element
        # value against a scalar broadcast (3,) into (2,); a two-element one
        # broadcast by coincidence, which is why the defect survived its own
        # test). The scalar expansion is built here from the element-shaped
        # template directly: the leading limb carries the value, the rest
        # are exact zeros. ``limbs_of`` keeps its interleaved contract for
        # the callers that honour it.
        template = (left.term(0) if isinstance(left, Precision)
                    else right.term(0) if isinstance(right, Precision)
                    else (left if hasattr(left, "shape") else right))
        template_zero = plain(template, "mul", 0.0)

        def operand(value):
            if isinstance(value, Precision):
                return value.terms(width)
            if hasattr(value, "shape"):
                # A plain tensor meeting a Precision value is one leading
                # limb plus exact zero tails.  It is not an already-packed
                # interleaved buffer; treating it as one broke scalar factors
                # such as the binade scale used by real powers.
                value_zero = plain(value, "mul", 0.0)
                return [value] + [value_zero for _index in range(width - 1)]
            return ([plain(template_zero, "add", value)]
                    + [template_zero for _index in range(width - 1)])

        if op == "neg":
            pieces = negate(operand(left))
        else:
            first, second = operand(left), operand(right)
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
        # The class owns a planar leading limb axis.  Passing the terms
        # directly also preserves zero-dimensional scalar tensors; routing a
        # scalar through the legacy channel interleaver has no channel axis to
        # flatten and used to make scalar AOT parameters fail here.
        return cls(pieces, width)

    @classmethod
    def _integer_dispatch(cls, op: str, left: Any, right: Any, *,
                          limbs: int = 1):
        """One operator on integer limbs: Python ``int`` semantics, exactly.

        The same width rule as the floating arm -- the widest of the
        caller's ``limbs`` and the operands' -- where a plain int64 tensor
        is as wide as it takes to hold any int64 and a Python ``int`` as
        wide as its own limbs. Results are renormalised to that width; one
        that does not fit raises ``INT_OVERFLOW`` instead of losing its top.
        """

        base = op[1:] if op in _REVERSED_INTEGER else op.lstrip("i")
        if base not in _INTEGER_HANDLED:
            raise TypeError(
                f"integer Precision supports + - * // % neg and comparisons; "
                f"{op!r} is not one of them (Python's int / int is a float)"
            )
        digit_bits = limb_element_facts("int64")["digit_bits"]
        native_width = integer_int64_limbs(digit_bits)
        template = (left.term(0) if isinstance(left, Precision)
                    and left.integer else right.term(0))

        def width_of(value):
            if isinstance(value, Precision):
                if not value.integer:
                    raise TypeError(
                        "an integer Precision cannot meet a floating one; "
                        "collapse one side first and mean it"
                    )
                return value.limbs
            if value is None:
                return 1
            if isinstance(value, bool) or not isinstance(value, int):
                if not _integer_dtype(value):
                    raise TypeError(
                        "an integer Precision meets ints only; "
                        f"got {type(value).__name__}"
                    )
                return native_width
            return len(_python_int_limbs(value, digit_bits))

        width = max(int(limbs or 1), width_of(left), width_of(right))

        def operand(value):
            if isinstance(value, Precision):
                return value.terms(width)
            if hasattr(value, "shape"):
                terms, _overflow = integer_from_int64(
                    _as_int64(value), width, digit_bits,
                )
                return terms
            zero = template * 0
            return _int_align(
                [zero + digit for digit in _python_int_limbs(value, digit_bits)],
                width,
            )

        if base == "neg":
            pieces, overflow = integer_renormalise(
                integer_negate(operand(left)), width, digit_bits,
            )
            _raise_integer_overflow(overflow, "-Precision")
            return cls(pieces, width)
        first, second = operand(left), operand(right)
        if op in _REVERSED_INTEGER:
            first, second = second, first
        if base in _INTEGER_ORDER:
            negative, zero = integer_compare(first, second, digit_bits)
            return {
                "lt": negative, "le": negative | zero, "eq": zero,
                "ne": ~zero, "gt": ~(negative | zero), "ge": ~negative,
            }[base]
        if base in {"floordiv", "mod"}:
            quotient, remainder, zero_divisor, overflow = (
                integer_divmod_expansions(first, second, width, digit_bits)
            )
            _raise_zero_division(zero_divisor, f"Precision {base}")
            pieces = quotient if base == "floordiv" else remainder
        elif base == "mul":
            pieces, overflow = integer_multiply_expansions(
                first, second, width, digit_bits,
            )
        else:
            pieces, overflow = integer_add_expansions(
                first,
                second if base == "add" else integer_negate(second),
                width, digit_bits,
            )
        _raise_integer_overflow(overflow, f"Precision {base}")
        return cls(pieces, width)

    def __add__(self, other): return _composed_binary("add", self, other)
    def __radd__(self, other): return _composed_binary("add", other, self)
    def __sub__(self, other): return _composed_binary("sub", self, other)
    def __rsub__(self, other): return _composed_binary("sub", other, self)
    def __mul__(self, other): return _composed_binary("mul", self, other)
    def __rmul__(self, other): return _composed_binary("mul", other, self)
    def __truediv__(self, other): return _composed_binary("truediv", self, other)
    def __rtruediv__(self, other): return _composed_binary("truediv", other, self)
    def __neg__(self) -> "Precision": return Precision.dispatch("neg", self, None)

    # Floor division, modulus and order: Python's ``int`` operations on
    # integer limbs. A floating expansion refuses them (``dispatch``).
    def __floordiv__(self, other): return _composed_binary("floordiv", self, other)
    def __rfloordiv__(self, other): return _composed_binary("floordiv", other, self)
    def __mod__(self, other): return _composed_binary("mod", self, other)
    def __rmod__(self, other): return _composed_binary("mod", other, self)
    def __lt__(self, other): return _composed_binary("lt", self, other)
    def __le__(self, other): return _composed_binary("le", self, other)
    def __gt__(self, other): return _composed_binary("gt", self, other)
    def __ge__(self, other): return _composed_binary("ge", self, other)

    def __eq__(self, other):
        # A floating Precision keeps identity equality, as before.
        if not self.integer:
            return NotImplemented
        return _composed_binary("eq", self, other)

    def __ne__(self, other):
        if not self.integer:
            return NotImplemented
        return _composed_binary("ne", self, other)

    __hash__ = object.__hash__

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
            [function(term) for term in self.terms()], self.limbs
        )

    def _slice_limbs(self, start: int, stop: int) -> "Precision":
        """Elements ``[start:stop]``, with each one's limbs kept together."""

        return Precision(
            [term.reshape(-1)[start:stop] for term in self.terms()],
            self.limbs,
        )

    def sign(self):
        """The sign of the whole expansion, as an ordinary tensor.

        The leading limb decides: a distilled expansion keeps every tail
        below an ulp of the head, so no tail can outvote it. Integer limbs
        are read the same way from the head down: a non-overlapping head is
        negative exactly when the value is, and a zero head defers to the
        next limb (an integer's limbs have fixed places, so a small value's
        head is zero rather than its leading digit).
        """

        if self.integer:
            digit_bits = limb_element_facts("int64")["digit_bits"]
            terms = self.terms()
            negative, zero = integer_compare(terms, [terms[0] * 0], digit_bits)
            one = terms[0] * 0 + 1
            return _int_where(negative, one * -1, _int_where(zero, one * 0, one))
        return self.term(0).sign()

    def __abs__(self) -> "Precision":
        """Magnitude, by negating the WHOLE expansion where it is negative.

        Taking the absolute value of each limb separately is a different
        number: ``|a| + |b|`` is not ``|a + b|`` when the tail opposes the
        head, which for a distilled expansion is the usual case.
        """

        self._float_only("abs")

        sign = self.sign()
        return self._map_limbs(lambda term: term * sign)

    def _select(self, other: Any, take_self: Any) -> "Precision":
        """Elementwise choice between two values, each kept WHOLE.

        ``take_self`` is a 0/1 mask.  A limb times 0 or 1 is exact, and the
        sum of two limbs of which exactly one is zero is exact, so the chosen
        expansion arrives untouched -- no limb is ever rounded on its own.
        """

        self._float_only("where")

        width = max(self.limbs, Precision.width_of(other))
        mine = self.terms(width)
        if isinstance(other, Precision):
            theirs = other.terms(width)
        else:
            head = self.term(0) * 0.0 + other
            theirs = [head] + [head * 0.0 for _index in range(width - 1)]
        keep = take_self * 1.0
        drop = 1.0 - keep
        return Precision(
            [a * keep + b * drop for a, b in zip(mine, theirs)], width,
        )

    def maximum(self, other: Any) -> "Precision":
        """Elementwise larger value, selecting the WHOLE expansion.

        The comparison is made on the difference's leading limb, which for
        a distilled expansion carries the sign of the whole difference.
        """

        return self._select(other, (self - other).term(0) >= 0.0)

    def minimum(self, other: Any) -> "Precision":
        """Elementwise smaller value, selecting the WHOLE expansion."""

        return self._select(other, (self - other).term(0) <= 0.0)

    def floor(self):
        """The integer part, as an ordinary tensor.

        Deliberately not a ``Precision``: an integer needs no limbs, and
        returning one would invite a caller to carry a tail that is
        exactly zero. The collapsed sum decides and the residual the
        collapse discarded corrects it -- a value whose head sits just
        above an integer while its tail pulls it below belongs to the
        lower one, and the head alone cannot see that.
        """

        self._float_only("floor")

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

        self._float_only("sqrt")

        import math

        seed = Precision.of(self.collapse().sqrt(), self.limbs)
        steps = max(1, int(math.ceil(math.log2(max(self.limbs, 1))))) + 1
        root = seed
        for _step in range(steps):
            root = (root + self / root) * 0.5
        return root

    def __pow__(self, exponent) -> "Precision":
        """Integer powers exactly; real powers through wide log and exp.

        The real route is ``exp(exponent * log(self))``.  Both transcendental
        operations range-reduce into their proved limb-aware cores, so this is
        not a binary64 answer widened after the fact.  A real exponent requires
        a strictly positive base; negative-base integer powers keep using the
        exact repeated-squaring path.
        """

        self._float_only("pow")

        if isinstance(exponent, float) and exponent.is_integer():
            exponent = int(exponent)
        if not isinstance(exponent, int):
            return (exponent * self.log()).exp()
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

    def __rpow__(self, base) -> "Precision":
        """A scalar/tensor base raised to this wide real exponent."""

        return Precision.of(base, self.limbs) ** self

    # -- transcendentals: the cores, at this width -----------------------
    #
    # These are not arithmetic and cannot be built from it, so they are
    # not implemented here: they are ROUTED to the materialised proof
    # cores, which are the same programs the compiler lowers and which
    # measure their own error against an exact oracle. The width travels
    # with the argument, so a wide value gets a wide Horner chain rather
    # than a double's worth of answer widened afterwards.
    #
    # A core is valid on ITS OWN INTERVAL and carries no range reduction,
    # so an argument outside that interval is refused by name rather than
    # pushed through a polynomial that approximates nothing there. That is
    # the same rule the compiled kernels follow -- their router falls back
    # to the eager surface beyond the radius -- and it is deliberate: a
    # silently extrapolated core returns a plausible number, which is the
    # one outcome this type exists to prevent. Reduction belongs to the
    # caller, or to the signal surface that owns turns and binades.

    def _core(self, name: str) -> "Precision":
        self._float_only(name)
        from .signal_symbolic import CORE_RADII, evaluate_proof

        radius = CORE_RADII.get(name)
        if radius is None:
            raise AttributeError(
                f"no proof core is registered for {name!r}"
            )
        magnitude = abs(self).collapse()
        try:
            worst = max(
                abs(float(each))
                for each in _flatten(magnitude.tolist())
            )
        except (TypeError, ValueError):
            worst = float("inf")
        if worst > float(radius):
            raise ValueError(
                f"{name}: argument reaches {worst!r}, outside the core's "
                f"proven interval +-{radius}. Reduce the range first -- a "
                "core evaluated beyond its interval approximates nothing "
                "and would return a plausible wrong answer at every limb"
            )
        return evaluate_proof(name, self, self.limbs)

    def sin(self) -> "Precision":
        return self._core("sin")

    def cos(self) -> "Precision":
        return self._core("cos")

    def tan(self) -> "Precision":
        return self._core("tan")

    def exp(self) -> "Precision":
        """Wide exponential with ln(2) range reduction.

        The collapsed value selects an integer binade only; it never supplies
        significant digits to the result.  The reduced remainder and final
        product remain at this value's limb width.
        """

        self._float_only("exp")

        import math

        from .signal_symbolic import constant_limbs

        ln2 = Precision.constant(self, constant_limbs("ln2", self.limbs))
        binade = (self.collapse() / float(math.log(2.0))).round()
        remainder = self - ln2 * binade
        scale = 2.0 ** binade
        return remainder._core("exp") * scale

    def log(self) -> "Precision":
        """Natural logarithm for positive values, at the owned limb width.

        A binary64 logarithm chooses the binade but contributes no digits to
        the answer.  Two square roots move the mantissa into the proved
        ``log1p`` interval; the wide ln(2) reconstruction then restores scale.
        """

        self._float_only("log")

        import math

        from .signal_symbolic import constant_limbs

        collapsed = self.collapse()
        try:
            minimum = min(float(each) for each in _flatten(collapsed.tolist()))
        except (TypeError, ValueError):
            minimum = float("-inf")
        if minimum <= 0.0:
            raise ValueError("real Precision.log and real powers require a positive base")
        binade = (collapsed.log() / float(math.log(2.0))).floor()
        mantissa = self * (2.0 ** (-binade))
        reduced = mantissa.sqrt().sqrt()
        ln2 = Precision.constant(self, constant_limbs("ln2", self.limbs))
        return (reduced - 1.0).log1p() * 4.0 + ln2 * binade

    def expm1(self) -> "Precision":
        return self._core("expm1")

    def log1p(self) -> "Precision":
        return self._core("log1p")

    def sinh(self) -> "Precision":
        return self._core("sinh")

    def cosh(self) -> "Precision":
        return self._core("cosh")

    def tanh(self) -> "Precision":
        return self._core("tanh")

    def atan(self) -> "Precision":
        return self._core("atan")

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

        self._float_only("mean")

        return self.sum() / float(max(self.element_count(), 1))


class ComplexPrecision:
    """A complex value with independent real precision expansions.

    This composes the complex algebra with the existing ``Precision``
    coefficient type. It deliberately does not store native complex limbs:
    Dekker/Knuth transforms prove one rounded real operation, while a native
    complex multiply has already rounded several real operations.
    """

    __slots__ = ("real", "imag", "limbs")

    @classmethod
    def __class_getitem__(cls, width):
        import types

        return types.GenericAlias(cls, width)

    def __init__(self, real: Any, imag: Any, limbs: int = 2):
        width = max(
            int(limbs), Precision.width_of(real), Precision.width_of(imag), 1
        )
        self.real = self._coefficient(real, width)
        self.imag = self._coefficient(imag, width, like=self.real)
        self.limbs = width

    @staticmethod
    def _coefficient(
        value: Any, width: int, like: Precision | None = None,
    ) -> Precision:
        if isinstance(value, Precision):
            return Precision(value.terms(width), width)
        if _complex_dtype(value) or isinstance(value, complex):
            raise TypeError(
                "ComplexPrecision components must be real; split a native "
                "complex tensor with ComplexPrecision.of()."
            )
        if like is not None and not hasattr(value, "shape"):
            value = like.term(0) * 0.0 + value
        return Precision.of(value, width)

    @classmethod
    def of(cls, value: Any, limbs: int = 2) -> "ComplexPrecision":
        """Promote a native complex tensor, component pair, or real value."""

        from .abstraction import AbstractTensor

        if isinstance(value, cls):
            width = max(int(limbs), value.limbs)
            return cls(value.real, value.imag, width)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return cls(value[0], value[1], limbs)
        if isinstance(value, complex):
            return cls(
                AbstractTensor.get_tensor(value.real),
                AbstractTensor.get_tensor(value.imag),
                limbs,
            )
        if _complex_dtype(value):
            tensor = (
                value if isinstance(value, AbstractTensor)
                else AbstractTensor.get_tensor(value)
            )
            return cls(
                AbstractTensor.real(tensor), AbstractTensor.imag(tensor), limbs,
            )
        tensor = (
            value if hasattr(value, "shape")
            else AbstractTensor.get_tensor(value)
        )
        return cls(tensor, tensor * 0.0, limbs)

    @property
    def shape(self):
        return self.real.shape

    def components(self) -> tuple[Precision, Precision]:
        """Compiler-facing planar pair: real expansion, imaginary expansion."""

        return self.real, self.imag

    def collapse_components(self):
        return self.real.collapse(), self.imag.collapse()

    def collapse(self):
        from .abstraction import AbstractTensor

        real, imag = self.collapse_components()
        return AbstractTensor.complex(real, imag)

    def conjugate(self) -> "ComplexPrecision":
        return type(self)(self.real, -self.imag, self.limbs)

    conj = conjugate

    def norm2(self) -> Precision:
        return self.real * self.real + self.imag * self.imag

    def __abs__(self) -> Precision:
        return self.norm2().sqrt()

    def _pair(self, other: Any) -> "ComplexPrecision":
        return type(self).of(other, self.limbs)

    def __add__(self, other):
        return _composed_binary("add", self, other)

    def __radd__(self, other) -> "ComplexPrecision":
        return _composed_binary("add", other, self)

    def __sub__(self, other):
        return _composed_binary("sub", self, other)

    def __rsub__(self, other) -> "ComplexPrecision":
        return _composed_binary("sub", other, self)

    def __neg__(self) -> "ComplexPrecision":
        return type(self)(-self.real, -self.imag, self.limbs)

    def __mul__(self, other):
        return _composed_binary("mul", self, other)

    def __rmul__(self, other) -> "ComplexPrecision":
        return _composed_binary("mul", other, self)

    def __truediv__(self, other):
        return _composed_binary("truediv", self, other)

    def __rtruediv__(self, other) -> "ComplexPrecision":
        return _composed_binary("truediv", other, self)

    def sum(self) -> "ComplexPrecision":
        return type(self)(self.real.sum(), self.imag.sum(), self.limbs)

    def mean(self) -> "ComplexPrecision":
        return type(self)(self.real.mean(), self.imag.mean(), self.limbs)

    def __repr__(self) -> str:
        return f"ComplexPrecision(limbs={self.limbs}, shape={self.shape!r})"


# --------------------------------------------------------------------------
# Structural rationals and canonical numerical composition


def _component_zero(like: Any):
    return like * 0


def _component_one(like: Any):
    return _component_zero(like) + 1


def _broadcast_shape(left: Any, right: Any) -> tuple[int, ...]:
    from .abstraction_methods.elementwise import _broadcast_result_shape

    return tuple(_broadcast_result_shape("rational", left.shape, right.shape))


#: A rational result whose denominator's leading limb leaves [2**-400, 2**400]
#: is rebalanced at once.  Its low limbs sit about 53 * (limbs - 1) bits below
#: the leading one, so inside this band every limb stays a normal float64;
#: outside it, products overflow or tails fall into subnormals and lose bits.
#: Measured: a 21-knot spline solve drove the rhs components to 2**-857 with
#: tails at 2**-1043, and the solved values lost everything past 1e-16.
_REBALANCE_BAND = 2.0 ** 400


def _rebalanced(numerator: Any, denominator: Any, *, band: float | None = None):
    """Scale numerator and denominator together by an exact power of two,
    element by element, so each denominator's leading limb sits in [1, 2).

    The quotient is unchanged exactly: both components are multiplied by the
    same power of two, and a power-of-two multiply is exact while normal.
    With ``band``, only elements whose denominator has left the band move.
    """

    from .abstraction import AbstractTensor
    from .power_scale import _log2_floor, _shift_raw

    head = denominator.term(0) if isinstance(denominator, Precision) else denominator
    if "float" not in str(getattr(head, "dtype", "") or "").lower():
        return numerator, denominator            # integer quotients stay integers
    magnitude = abs(head)
    shift = _log2_floor(magnitude)
    if band is not None:
        drifted = ((magnitude > band) + (magnitude < 1.0 / band)) > 0
        shift = AbstractTensor.where(drifted, shift, magnitude * 0.0)   # 2**0: exact no-op
    return _shift_raw(numerator, -shift), _shift_raw(denominator, -shift)


class _RationalBase:
    __slots__ = ("numerator", "denominator")

    @property
    def shape(self) -> tuple[int, ...]:
        return _broadcast_shape(self.numerator, self.denominator)

    def components(self):
        return self.numerator, self.denominator

    def _binary_same(self, other: "_RationalBase", operation: str):
        result = self._binary_components(other, operation)
        numerator, denominator = result.components()
        numerator, denominator = _rebalanced(
            numerator, denominator, band=_REBALANCE_BAND,
        )
        return type(result)(numerator, denominator)

    def _binary_components(self, other: "_RationalBase", operation: str):
        a, b = self.components()
        c, d = other.components()
        if operation in {"add", "sub"}:
            left = a * d
            right = b * c
            numerator = left - right if operation == "sub" else left + right
            denominator = b * d
        elif operation == "mul":
            numerator, denominator = a * c, b * d
        elif operation == "truediv":
            numerator, denominator = a * d, b * c
        else:
            raise TypeError(f"unsupported rational operation {operation!r}")
        return type(self)(numerator, denominator)

    def reciprocal(self):
        return type(self)(self.denominator, self.numerator)

    def quotient(self):
        return self.numerator / self.denominator

    def collapse(self):
        quotient = self.quotient()
        return quotient.collapse() if isinstance(quotient, Precision) else quotient

    def __add__(self, other): return _composed_binary("add", self, other)
    def __radd__(self, other): return _composed_binary("add", other, self)
    def __sub__(self, other): return _composed_binary("sub", self, other)
    def __rsub__(self, other): return _composed_binary("sub", other, self)
    def __mul__(self, other): return _composed_binary("mul", self, other)
    def __rmul__(self, other): return _composed_binary("mul", other, self)
    def __truediv__(self, other): return _composed_binary("truediv", self, other)
    def __rtruediv__(self, other): return _composed_binary("truediv", other, self)

    def __neg__(self):
        return type(self)(-self.numerator, self.denominator)

    # -- element structure -------------------------------------------------

    def __getitem__(self, key):
        """Select elements: the numerator and denominator with the same key,
        so each selected element keeps its whole quotient."""

        return type(self)(self.numerator[key], self.denominator[key])

    def __len__(self) -> int:
        return int(self.shape[0])

    # -- order ---------------------------------------------------------------

    def sign(self):
        """Elementwise sign as an ordinary tensor.

        Construction normalizes every denominator positive, so the quotient's
        sign is its numerator's -- exact, no division.
        """

        return self.numerator.sign()

    def __abs__(self):
        return type(self)(abs(self.numerator), self.denominator)

    def _order(self, other):
        return (self - other).sign()

    def __lt__(self, other): return self._order(other) < 0
    def __le__(self, other): return self._order(other) <= 0
    def __gt__(self, other): return self._order(other) > 0
    def __ge__(self, other): return self._order(other) >= 0

    # -- accumulation ----------------------------------------------------------

    def cumsum(self, dim: int = 0):
        """Running sum along the leading axis, each partial sum a quotient."""

        if int(dim) != 0:
            raise ValueError("rational cumsum runs along the leading axis only")
        count = len(self)
        running = self[0:1]
        pieces = [running]
        for index in range(1, count):
            running = running + self[index:index + 1]
            pieces.append(running)
        return concat(pieces, dim=0)

    # -- readout ---------------------------------------------------------------

    def limb_terms(self, limbs: int) -> list:
        """The quotient as ``limbs`` ordinary tensors whose sum is it.

        Division happens here, once, at the components' own width; the
        result is then renormalised to exactly ``limbs`` limbs (the tail is
        folded into the last limb, so the readout is the nearest
        ``limbs``-limb value, not a truncation).
        """

        numerator, denominator = _rebalanced(self.numerator, self.denominator)
        quotient = numerator / denominator
        if isinstance(quotient, Precision):
            terms = quotient.terms()
        else:
            terms = [quotient]
        return renormalise(terms, int(limbs))

    def __pow__(self, exponent: int):
        if not isinstance(exponent, int):
            raise TypeError("rational powers require an integer exponent")
        if exponent < 0:
            return self.reciprocal() ** -exponent
        result = type(self)(_component_one(self.numerator), _component_one(self.denominator))
        base, remaining = self, exponent
        while remaining:
            if remaining & 1:
                result = result * base
            remaining >>= 1
            if remaining:
                base = base * base
        return result


class Rational(_RationalBase):
    """A lazy real AbstractTensor quotient."""

    def __init__(self, numerator: Any, denominator: Any):
        from .abstraction import AbstractTensor

        if isinstance(numerator, Precision) or isinstance(denominator, Precision):
            raise TypeError("use RationalPrecision for Precision coefficients")
        numerator = (
            numerator if hasattr(numerator, "shape")
            else AbstractTensor.get_tensor(numerator)
        )
        denominator = (
            denominator if hasattr(denominator, "shape")
            else numerator * 0.0 + denominator
        )
        self.numerator = numerator
        self.denominator = denominator

    @classmethod
    def of(cls, value: Any, *, like: Any = None) -> "Rational":
        if isinstance(value, cls):
            return value
        value = _real_coefficient(value, like=like)
        return cls(value, _component_one(value))

    @classmethod
    def ratio(cls, numerator: Any, denominator: Any) -> "Rational":
        return cls(numerator, denominator)

    def __repr__(self) -> str:
        return f"Rational(shape={self.shape!r})"


class RationalPrecision(_RationalBase):
    """A lazy quotient whose two real coefficients are Precision values."""

    __slots__ = ("limbs",)

    def __init__(self, numerator: Any, denominator: Any, limbs: int = 2):
        width = max(
            int(limbs), Precision.width_of(numerator),
            Precision.width_of(denominator), 1,
        )
        numerator = _precision_coefficient(numerator, width)
        denominator = _precision_coefficient(denominator, width, like=numerator)
        self.numerator = numerator
        self.denominator = denominator
        self.limbs = width

    @classmethod
    def of(
        cls, value: Any, limbs: int = 2, *, like: Any = None,
    ) -> "RationalPrecision":
        if isinstance(value, cls):
            return cls(value.numerator, value.denominator, max(limbs, value.limbs))
        coefficient = _precision_coefficient(value, limbs, like=like)
        return cls(coefficient, _component_one(coefficient), limbs)

    @classmethod
    def ratio(
        cls, numerator: Any, denominator: Any, limbs: int = 2,
    ) -> "RationalPrecision":
        return cls(numerator, denominator, limbs)

    def __repr__(self) -> str:
        return f"RationalPrecision(limbs={self.limbs}, shape={self.shape!r})"


class _ComplexRationalBase:
    __slots__ = ("real", "imag")

    @property
    def shape(self):
        return _broadcast_shape(self.real, self.imag)

    def components(self):
        return self.real, self.imag

    def collapse_components(self):
        return self.real.collapse(), self.imag.collapse()

    def collapse(self):
        from .abstraction import AbstractTensor

        real, imag = self.collapse_components()
        return AbstractTensor.complex(real, imag)

    def conjugate(self):
        return type(self)(self.real, -self.imag)

    conj = conjugate

    def reciprocal(self):
        one = type(self).of(_component_one(self.real.numerator), like=self)
        return one / self

    def _binary_same(self, other: "_ComplexRationalBase", operation: str):
        a, b = self.components()
        c, d = other.components()
        if operation == "add":
            return type(self)(a + c, b + d)
        if operation == "sub":
            return type(self)(a - c, b - d)
        if operation == "mul":
            return type(self)(a * c - b * d, a * d + b * c)
        if operation == "truediv":
            denominator = c * c + d * d
            return type(self)(
                (a * c + b * d) / denominator,
                (b * c - a * d) / denominator,
            )
        raise TypeError(f"unsupported complex-rational operation {operation!r}")

    def __add__(self, other): return _composed_binary("add", self, other)
    def __radd__(self, other): return _composed_binary("add", other, self)
    def __sub__(self, other): return _composed_binary("sub", self, other)
    def __rsub__(self, other): return _composed_binary("sub", other, self)
    def __mul__(self, other): return _composed_binary("mul", self, other)
    def __rmul__(self, other): return _composed_binary("mul", other, self)
    def __truediv__(self, other): return _composed_binary("truediv", self, other)
    def __rtruediv__(self, other): return _composed_binary("truediv", other, self)

    def __neg__(self):
        return type(self)(-self.real, -self.imag)


class ComplexRational(_ComplexRationalBase):
    """A complex value with two real Rational coefficients."""

    def __init__(self, real: Any, imag: Any):
        self.real = _promote_numeric(real, frozenset({"rational"}), 1)
        self.imag = _promote_numeric(
            imag, frozenset({"rational"}), 1, like=self.real,
        )

    @classmethod
    def of(cls, value: Any, *, like: Any = None) -> "ComplexRational":
        return _complex_rational_of(cls, value, 1, like=like)

    def __repr__(self) -> str:
        return f"ComplexRational(shape={self.shape!r})"


class ComplexRationalPrecision(_ComplexRationalBase):
    """A complex value with two RationalPrecision coefficients."""

    __slots__ = ("limbs",)

    def __init__(self, real: Any, imag: Any, limbs: int = 2):
        width = max(
            int(limbs), _numeric_width(real), _numeric_width(imag), 1,
        )
        self.real = _promote_numeric(
            real, frozenset({"rational", "precision"}), width,
        )
        self.imag = _promote_numeric(
            imag, frozenset({"rational", "precision"}), width,
            like=self.real,
        )
        self.limbs = width

    @classmethod
    def of(
        cls, value: Any, limbs: int = 2, *, like: Any = None,
    ) -> "ComplexRationalPrecision":
        return _complex_rational_of(cls, value, limbs, like=like)

    def __repr__(self) -> str:
        return (
            f"ComplexRationalPrecision(limbs={self.limbs}, "
            f"shape={self.shape!r})"
        )


def _numeric_features(value: Any) -> frozenset[str]:
    if isinstance(value, ComplexRationalPrecision):
        return frozenset({"complex", "rational", "precision"})
    if isinstance(value, ComplexRational):
        return frozenset({"complex", "rational"})
    if isinstance(value, RationalPrecision):
        return frozenset({"rational", "precision"})
    if isinstance(value, Rational):
        return frozenset({"rational"})
    if isinstance(value, ComplexPrecision):
        return frozenset({"complex", "precision"})
    if isinstance(value, Precision):
        return frozenset({"precision"})
    if _complex_dtype(value) or isinstance(value, complex):
        return frozenset({"complex"})
    return frozenset()


def _numeric_width(value: Any) -> int:
    return max(int(getattr(value, "limbs", 1) or 1), 1)


def _coefficient_template(value: Any):
    if isinstance(value, Precision):
        return value.term(0)
    if isinstance(value, ComplexPrecision):
        return value.real.term(0)
    if isinstance(value, _RationalBase):
        return _coefficient_template(value.numerator)
    if isinstance(value, _ComplexRationalBase):
        return _coefficient_template(value.real)
    if hasattr(value, "shape"):
        return value
    return None


def _real_coefficient(value: Any, *, like: Any = None):
    from .abstraction import AbstractTensor

    if _complex_dtype(value) or isinstance(value, complex):
        raise TypeError("a real coefficient cannot absorb a complex value")
    if hasattr(value, "shape"):
        return value
    template = _coefficient_template(like)
    if template is not None:
        return template * 0.0 + value
    return AbstractTensor.get_tensor(value)


def _precision_coefficient(
    value: Any, width: int, *, like: Any = None,
) -> Precision:
    width = max(int(width), 1)
    if isinstance(value, Precision):
        if value.limbs == width:
            return value
        return Precision(value.terms(width), width)
    template = _coefficient_template(like)
    if template is not None and _integer_dtype(template):
        # Meeting integer limbs: an int stays an int, at the width it takes.
        digit_bits = limb_element_facts("int64")["digit_bits"]
        if isinstance(value, int) and not isinstance(value, bool):
            digits = _python_int_limbs(value, digit_bits)
            width = max(width, len(digits))
            zero = template * 0
            return Precision(
                _int_align([zero + digit for digit in digits], width), width,
            )
        if _integer_dtype(value):
            return Precision.of(
                value, max(width, integer_int64_limbs(digit_bits)),
            )
        raise TypeError(
            "an integer Precision meets ints only; "
            f"got {type(value).__name__}"
        )
    return Precision.of(_real_coefficient(value, like=like), width)


def _complex_rational_of(cls, value: Any, limbs: int, *, like: Any = None):
    from .abstraction import AbstractTensor

    if isinstance(value, cls):
        if cls is ComplexRationalPrecision:
            return cls(value.real, value.imag, max(limbs, value.limbs))
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return cls(value[0], value[1], limbs) if cls is ComplexRationalPrecision else cls(*value)
    if isinstance(value, _ComplexRationalBase):
        return cls(value.real, value.imag, limbs) if cls is ComplexRationalPrecision else cls(value.real, value.imag)
    if isinstance(value, ComplexPrecision):
        return (
            cls(value.real, value.imag, limbs)
            if cls is ComplexRationalPrecision
            else cls(value.real, value.imag)
        )
    if isinstance(value, complex):
        template = _coefficient_template(like)
        if template is None:
            real = AbstractTensor.get_tensor(value.real)
            imag = AbstractTensor.get_tensor(value.imag)
        else:
            real = template * 0.0 + value.real
            imag = template * 0.0 + value.imag
        return cls(real, imag, limbs) if cls is ComplexRationalPrecision else cls(real, imag)
    if _complex_dtype(value):
        tensor = value if isinstance(value, AbstractTensor) else AbstractTensor.get_tensor(value)
        real, imag = AbstractTensor.real(tensor), AbstractTensor.imag(tensor)
        return cls(real, imag, limbs) if cls is ComplexRationalPrecision else cls(real, imag)
    real = value
    template = _coefficient_template(real)
    if template is None:
        template = _coefficient_template(like)
    imag = template * 0.0 if template is not None else 0.0
    return cls(real, imag, limbs) if cls is ComplexRationalPrecision else cls(real, imag)


_FEATURE_TYPES = {
    frozenset({"precision"}): Precision,
    frozenset({"complex", "precision"}): ComplexPrecision,
    frozenset({"rational"}): Rational,
    frozenset({"rational", "precision"}): RationalPrecision,
    frozenset({"complex", "rational"}): ComplexRational,
    frozenset({"complex", "rational", "precision"}): ComplexRationalPrecision,
}


def numeric_feature_type_registry() -> dict[str, type]:
    """Return the eager classes that implement each numeric feature set.

    The source compiler uses these exact class objects when an annotation
    admits wrapper source into the dependency graph. Keeping the registry
    beside eager promotion prevents annotation dispatch and runtime promotion
    from acquiring separate notions of the canonical class.
    """

    return {
        numeric_type.__name__: numeric_type
        for numeric_type in _FEATURE_TYPES.values()
    }


def _promote_numeric(
    value: Any, features: frozenset[str], limbs: int, *, like: Any = None,
):
    held = _numeric_features(value)
    width = max(int(limbs), _numeric_width(value), 1)
    if held == features and ("precision" not in features or _numeric_width(value) == width):
        return value
    if features == frozenset({"precision"}):
        return _precision_coefficient(value, width, like=like)
    if features == frozenset({"complex", "precision"}):
        return ComplexPrecision.of(value, width)
    if features == frozenset({"rational"}):
        return Rational.of(value, like=like)
    if features == frozenset({"rational", "precision"}):
        if isinstance(value, Rational):
            return RationalPrecision(value.numerator, value.denominator, width)
        return RationalPrecision.of(value, width, like=like)
    if features == frozenset({"complex", "rational"}):
        return ComplexRational.of(value, like=like)
    if features == frozenset({"complex", "rational", "precision"}):
        return ComplexRationalPrecision.of(value, width, like=like)
    raise TypeError(
        f"no canonical numerical type for feature set {sorted(features)!r}"
    )


def _same_binary(operation: str, left: Any, right: Any):
    if isinstance(left, Precision):
        return Precision.dispatch(operation, left, right)
    if isinstance(left, ComplexPrecision):
        width = max(left.limbs, right.limbs)
        if operation == "add":
            return ComplexPrecision(left.real + right.real, left.imag + right.imag, width)
        if operation == "sub":
            return ComplexPrecision(left.real - right.real, left.imag - right.imag, width)
        if operation == "mul":
            return ComplexPrecision(
                left.real * right.real - left.imag * right.imag,
                left.real * right.imag + left.imag * right.real,
                width,
            )
        if operation == "truediv":
            denominator = right.real * right.real + right.imag * right.imag
            return ComplexPrecision(
                (left.real * right.real + left.imag * right.imag) / denominator,
                (left.imag * right.real - left.real * right.imag) / denominator,
                width,
            )
    if isinstance(left, _RationalBase):
        return left._binary_same(right, operation)
    if isinstance(left, _ComplexRationalBase):
        return left._binary_same(right, operation)
    raise TypeError(f"unsupported composed numerical type {type(left).__name__}")


def _composed_binary(operation: str, left: Any, right: Any):
    features = _numeric_features(left) | _numeric_features(right)
    width = max(_numeric_width(left), _numeric_width(right), 1)
    if not features:
        raise TypeError("composed dispatch requires a numerical wrapper")
    left_value = _promote_numeric(left, features, width, like=right)
    right_value = _promote_numeric(right, features, width, like=left_value)
    if type(left_value) is not type(right_value):
        target = _FEATURE_TYPES.get(features)
        raise TypeError(
            f"promotion to {getattr(target, '__name__', features)!r} disagreed: "
            f"{type(left_value).__name__}, {type(right_value).__name__}"
        )
    return _same_binary(operation, left_value, right_value)


def concat(values: Sequence[Any], dim: int = 0):
    """Concatenate numerical values of one feature set along ``dim``.

    Plain tensors concatenate directly.  A ``Precision`` concatenates limb
    by limb at the widest width present.  A rational concatenates its
    numerators and denominators separately -- elements are independent
    quotients, so no common denominator is formed.
    """

    from .abstraction import AbstractTensor

    values = list(values)
    if not values:
        raise ValueError("concat needs at least one value")
    features = frozenset().union(*(_numeric_features(value) for value in values))
    if not features:
        return AbstractTensor.concat(values, dim=dim)
    width = max(_numeric_width(value) for value in values)
    promoted = [_promote_numeric(value, features, width, like=values[0]) for value in values]
    first = promoted[0]
    if isinstance(first, Precision):
        return Precision(
            [
                AbstractTensor.concat([value.terms(width)[limb] for value in promoted], dim=dim)
                for limb in range(width)
            ],
            width,
        )
    if isinstance(first, _RationalBase):
        numerator = concat([value.numerator for value in promoted], dim=dim)
        denominator = concat([value.denominator for value in promoted], dim=dim)
        return type(first)(numerator, denominator)
    raise TypeError(f"concat does not support {type(first).__name__}")


def where(mask: Any, when_true: Any, when_false: Any):
    """Elementwise choice between two numerical values, each kept WHOLE.

    A ``Precision`` chooses whole expansions (``Precision._select``: a limb
    times 0 or 1 is exact).  A rational chooses its numerator and
    denominator with the same mask, so each element's quotient arrives
    intact.  Plain tensors use ``AbstractTensor.where``.
    """

    from .abstraction import AbstractTensor

    features = _numeric_features(when_true) | _numeric_features(when_false)
    if not features:
        return AbstractTensor.where(mask, when_true, when_false)
    width = max(_numeric_width(when_true), _numeric_width(when_false), 1)
    true_value = _promote_numeric(when_true, features, width, like=when_false)
    false_value = _promote_numeric(when_false, features, width, like=true_value)
    if isinstance(true_value, Precision):
        return true_value._select(false_value, mask)
    if isinstance(true_value, _RationalBase):
        return type(true_value)(
            where(mask, true_value.numerator, false_value.numerator),
            where(mask, true_value.denominator, false_value.denominator),
        )
    raise TypeError(f"where does not support {type(true_value).__name__}")


def solve_tridiagonal(lower: Any, diagonal: Any, upper: Any, rhs: Any):
    """Solve a tridiagonal system by forward elimination and back substitution.

    ``lower[i]`` multiplies ``x[i-1]`` and ``upper[i]`` multiplies ``x[i+1]``
    in row ``i`` (``lower[0]`` and ``upper[-1]`` are ignored).  Written only
    in the operations every numerical type here supports -- element
    slicing, ``+ - * /`` and ``concat`` -- so the same elimination runs on
    plain tensors, ``Precision`` and rationals.  On a rational every division
    is deferred as a quotient of component products.
    """

    count = len(diagonal) if hasattr(diagonal, "__len__") else int(diagonal.shape[0])
    diag = [diagonal[i:i + 1] for i in range(count)]
    right = [rhs[i:i + 1] for i in range(count)]
    for i in range(1, count):
        factor = lower[i:i + 1] / diag[i - 1]
        diag[i] = diag[i] - factor * upper[i - 1:i]
        right[i] = right[i] - factor * right[i - 1]
    solution = [None] * count
    solution[-1] = right[-1] / diag[-1]
    for i in range(count - 2, -1, -1):
        solution[i] = (right[i] - upper[i:i + 1] * solution[i + 1]) / diag[i]
    return concat(solution, dim=0)

