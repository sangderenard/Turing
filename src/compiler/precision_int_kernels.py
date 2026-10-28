"""Integer limb kernels for the Precision int arm, written as plain Python.

These functions are the arithmetic a Python ``int`` continues with once it
leaves int64.  They are ordinary Python over lists of ints so that Turing
compiles them to native (C and LLVM) itself; no C or LLVM routine is
hand-written for them.  Run under CPython they agree with ``int`` exactly,
and ``tests/test_precision_int_kernels.py`` holds both lanes to that.

Representation
--------------
One value is one ``list[int]`` ``v``:

* ``v[0]`` is the sign: ``-1``, ``0`` or ``1``;
* ``v[1:]`` is the magnitude, little-endian, in unsigned limbs of
  ``LIMB_BITS = 30`` bits (each limb in ``[0, 2**30)``).

A normalised value has no zero top limb, and zero is exactly ``[0]`` (sign
0, no limbs).  Every kernel produces a normalised value.

Why 30-bit limbs and not 64-bit
-------------------------------
A 64-bit limb product needs a 128-bit intermediate (``u64 * u64``) and a
128-bit carry chain, while Turing's Python ints are int64 lanes.  A 32-bit
limb product fits uint64 but not int64 (``(2**32-1)**2 > 2**63``), so the
signed lanes would still need overflow checks.  With 30-bit limbs (CPython's
own digit width, for the same reason) every intermediate is bounded well
inside int64: a limb product is below ``2**60``, and a product plus a limb
plus a carry stays below ``2**61``.  Each limb read is masked with
``LIMB_MASK``, so an intermediate's range follows from the masks rather
than from trust in the list contents.

Calling convention
------------------
A kernel that produces a value writes it into its last argument ``out``
(cleared first) and returns ``None``; ``out`` must not be one of the
inputs.  Kernels producing a scalar (``compare``, ``less``, ``equal``,
``to_int64_checked``) return it.  Writing into the caller's list keeps each
result one storage the caller owns: a function returning one of several
different lists is not one storage, and Turing refuses that by name
(``returns different collections in return slot 0``).

Relation to ``Precision``
-------------------------
``Precision[n]`` (``src/common/tensors/extended_precision.py``) already
carries an exact integer as integer-valued double limbs: ``two_sum`` and
``two_product`` are error-free, so ``+ - *`` are exact while the result fits
``n`` limbs and stays below the binary64 range (Dekker's split overflows
past about ``2**996``).  The tests check that in CPython and natively.  It
has no exact form of what follows, which is why these kernels exist: floor
division and modulo (its ``divide_expansions`` is a float quotient, its
``floor`` collapses to one double), right shift (a floor), two's complement
``& | ^``, exact int64/int128 entry and checked exit, and values past the
binary64 exponent.  Its ``renormalise`` also folds what does not fit into
the last limb, which is rounding: an integer result too wide for ``n``
limbs comes back silently wrong unless the width is proven or checked.

Native status
-------------
Every kernel matches ``int`` in CPython.  Compiled through Turing, the
list-mutating loops are refused by the compiler today:
``_trim``'s ``while len(out) > 1 and out[len(out) - 1] == 0: out.pop()``
on a list formal raises ``control effect order conflicts with value
dependencies`` (the popping loop's signature publishes the formal's arena
and the later-producer rule orders it before the earlier read), and a
function returning one of two lists raises ``returns different collections
in return slot 0``.  The out-argument convention below avoids the second.

Python semantics
----------------
``floordiv``/``mod`` floor toward negative infinity and the remainder takes
the divisor's sign; ``shift_right`` of a negative value floors; ``and_``,
``or_`` and ``xor`` act on the infinite two's complement form.  Division by
zero raises ``ZeroDivisionError``; a negative shift count raises
``ValueError``; ``to_int64_checked`` raises ``OverflowError`` when the
value lies outside int64.
"""

from __future__ import annotations

LIMB_BITS = 30
LIMB_MASK = 1073741823  # 2**30 - 1
LIMB_BASE = 1073741824  # 2**30
LIMB_HALF = 536870912  # 2**29, the top bit of one limb


# --- normalisation and construction -----------------------------------------


def _trim(out: list[int]) -> None:
    """Strip ``out``'s zero top limbs in place; no limbs left means zero."""

    while len(out) > 1 and out[len(out) - 1] == 0:
        out.pop()
    if len(out) == 1:
        out[0] = 0


def normalise(v: list[int], out: list[int]) -> None:
    """``v`` with its zero top limbs stripped; zero becomes ``[0]``."""

    out.clear()
    out.append(v[0])
    index = 1
    while index < len(v):
        out.append(v[index] & LIMB_MASK)
        index += 1
    _trim(out)


def _copy_signed(v: list[int], sign: int, out: list[int]) -> None:
    """``|v|`` with sign ``sign`` (``v`` normalised); zero stays ``[0]``."""

    out.clear()
    out.append(sign)
    index = 1
    while index < len(v):
        out.append(v[index] & LIMB_MASK)
        index += 1
    _trim(out)


def _from_twos(t: list[int], out: list[int]) -> None:
    """The value of the two's complement limbs ``t``.

    ``t`` is little-endian in 30-bit limbs; bit 29 of its last limb is the
    sign bit.
    """

    n = len(t)
    out.clear()
    if ((t[n - 1] & LIMB_MASK) >> (LIMB_BITS - 1)) == 1:
        # magnitude = ~t + 1, limb by limb
        out.append(-1)
        carry = 1
        index = 0
        while index < n:
            limb = ((t[index] & LIMB_MASK) ^ LIMB_MASK) + carry
            out.append(limb & LIMB_MASK)
            carry = limb >> LIMB_BITS
            index += 1
    else:
        out.append(1)
        index = 0
        while index < n:
            out.append(t[index] & LIMB_MASK)
            index += 1
    _trim(out)


def _to_twos(v: list[int], n: int, out: list[int]) -> None:
    """``n`` two's complement limbs of ``v`` (``n`` must exceed its limb count)."""

    out.clear()
    count = len(v) - 1
    flip = 0
    carry = 0
    if v[0] < 0:
        flip = LIMB_MASK
        carry = 1
    index = 0
    while index < n:
        limb = 0
        if index < count:
            limb = v[index + 1] & LIMB_MASK
        limb = (limb ^ flip) + carry
        out.append(limb & LIMB_MASK)
        carry = limb >> LIMB_BITS
        index += 1


def from_int128(hi: int, lo: int, out: list[int]) -> None:
    """The value ``hi * 2**64 + (lo mod 2**64)`` of an int128 word pair.

    ``hi`` is the signed high word and ``lo`` the low word's 64-bit pattern
    read as an int64.
    """

    t = [lo & LIMB_MASK]
    t.append((lo >> 30) & LIMB_MASK)
    t.append(((lo >> 60) & 15) | ((hi & 67108863) << 4))
    t.append((hi >> 26) & LIMB_MASK)
    t.append((hi >> 56) & LIMB_MASK)
    _from_twos(t, out)


def from_int64(x: int, out: list[int]) -> None:
    """The value of the int64 ``x``."""

    from_int128(x >> 63, x, out)


def to_int64_checked(v: list[int]) -> int:
    """``v`` as an int64; ``OverflowError`` when it lies outside int64."""

    count = len(v) - 1
    if count > 3:
        raise OverflowError("Precision int does not fit int64")
    l0 = 0
    l1 = 0
    l2 = 0
    if count >= 1:
        l0 = v[1] & LIMB_MASK
    if count >= 2:
        l1 = v[2] & LIMB_MASK
    if count >= 3:
        l2 = v[3] & 15
        if v[3] > 15:
            raise OverflowError("Precision int does not fit int64")
    if l2 >= 8:
        if v[0] < 0 and l2 == 8 and l1 == 0 and l0 == 0:
            return -9223372036854775807 - 1
        raise OverflowError("Precision int does not fit int64")
    magnitude = l0 | (l1 << 30) | (l2 << 60)
    if v[0] < 0:
        return -magnitude
    return magnitude


# --- magnitude helpers ------------------------------------------------------


def _compare_magnitude(a: list[int], b: list[int]) -> int:
    """-1, 0 or 1 as ``|a|`` is below, equal to or above ``|b|``."""

    na = len(a)
    nb = len(b)
    if na < nb:
        return -1
    if na > nb:
        return 1
    index = na - 1
    while index >= 1:
        x = a[index] & LIMB_MASK
        y = b[index] & LIMB_MASK
        if x < y:
            return -1
        if x > y:
            return 1
        index -= 1
    return 0


def _add_magnitude(a: list[int], b: list[int], sign: int, out: list[int]) -> None:
    """``sign * (|a| + |b|)``."""

    na = len(a) - 1
    nb = len(b) - 1
    n = na
    if nb > n:
        n = nb
    out.clear()
    out.append(sign)
    carry = 0
    index = 0
    while index < n:
        x = 0
        y = 0
        if index < na:
            x = a[index + 1] & LIMB_MASK
        if index < nb:
            y = b[index + 1] & LIMB_MASK
        total = x + y + carry
        out.append(total & LIMB_MASK)
        carry = total >> LIMB_BITS
        index += 1
    out.append(carry)
    _trim(out)


def _sub_magnitude(a: list[int], b: list[int], sign: int, out: list[int]) -> None:
    """``sign * (|a| - |b|)``; requires ``|a| >= |b|``."""

    na = len(a) - 1
    nb = len(b) - 1
    out.clear()
    out.append(sign)
    borrow = 0
    index = 0
    while index < na:
        x = a[index + 1] & LIMB_MASK
        y = 0
        if index < nb:
            y = b[index + 1] & LIMB_MASK
        difference = x - y - borrow
        out.append(difference & LIMB_MASK)
        borrow = (difference >> LIMB_BITS) & 1
        index += 1
    _trim(out)


def _shift_limbs_left(a: list[int], shift: int, out: list[int]) -> None:
    """``|a| << shift`` (``0 <= shift < 30``) as raw limbs ``[1, ...]``.

    Not trimmed: ``out`` has exactly one more limb than ``a``.
    """

    n = len(a) - 1
    out.clear()
    out.append(1)
    carry = 0
    index = 0
    while index < n:
        limb = a[index + 1] & LIMB_MASK
        out.append(((limb << shift) & LIMB_MASK) | carry)
        carry = limb >> (LIMB_BITS - shift)
        index += 1
    out.append(carry)


def _shift_magnitude_right(a: list[int], count: int, sign: int, out: list[int]) -> None:
    """``sign * (|a| >> count)`` for ``count >= 0``."""

    whole = count // LIMB_BITS
    part = count - whole * LIMB_BITS
    n = len(a) - 1
    out.clear()
    out.append(sign)
    index = whole
    while index < n:
        limb = (a[index + 1] & LIMB_MASK) >> part
        if index + 1 < n:
            limb = limb | (((a[index + 2] & LIMB_MASK) << (LIMB_BITS - part)) & LIMB_MASK)
        out.append(limb)
        index += 1
    _trim(out)


def _divmod_magnitude(a: list[int], b: list[int], q: list[int], r: list[int]) -> None:
    """``q = |a| // |b|`` and ``r = |a| % |b|``, each with sign 1 or 0.

    ``|b|`` must be nonzero.  Knuth's algorithm D on 30-bit limbs.
    """

    nb = len(b) - 1
    na = len(a) - 1
    if _compare_magnitude(a, b) < 0:
        q.clear()
        q.append(0)
        _copy_signed(a, 1, r)
        return
    if nb == 1:
        d = b[1] & LIMB_MASK
        digits = [0]
        index = 1
        while index < na:
            digits.append(0)
            index += 1
        remainder = 0
        index = na - 1
        while index >= 0:
            current = (remainder << LIMB_BITS) | (a[index + 1] & LIMB_MASK)
            digit = current // d
            digits[index] = digit & LIMB_MASK
            remainder = current - digit * d
            index -= 1
        q.clear()
        q.append(1)
        index = 0
        while index < na:
            q.append(digits[index])
            index += 1
        _trim(q)
        r.clear()
        r.append(1)
        r.append(remainder & LIMB_MASK)
        _trim(r)
        return
    # scale so the divisor's top limb has bit 29 set
    shift = 0
    top = b[nb] & LIMB_MASK
    while top < LIMB_HALF:
        top = top << 1
        shift += 1
    u = [0]
    _shift_limbs_left(a, shift, u)
    v = [0]
    _shift_limbs_left(b, shift, v)
    m = na - nb
    vtop = v[nb] & LIMB_MASK
    vnext = v[nb - 1] & LIMB_MASK
    digits = [0]
    index = 0
    while index < m:
        digits.append(0)
        index += 1
    j = m
    while j >= 0:
        numerator = ((u[j + nb + 1] & LIMB_MASK) << LIMB_BITS) | (u[j + nb] & LIMB_MASK)
        qhat = numerator // vtop
        if qhat > LIMB_MASK:
            qhat = LIMB_MASK
        rhat = numerator - qhat * vtop
        while rhat < LIMB_BASE and qhat * vnext > ((rhat << LIMB_BITS) | (u[j + nb - 1] & LIMB_MASK)):
            qhat -= 1
            rhat += vtop
        # u[j .. j+nb] -= qhat * v
        borrow = 0
        carry = 0
        index = 0
        while index < nb:
            product = qhat * (v[index + 1] & LIMB_MASK) + carry
            carry = product >> LIMB_BITS
            difference = (u[j + index + 1] & LIMB_MASK) - (product & LIMB_MASK) - borrow
            u[j + index + 1] = difference & LIMB_MASK
            borrow = (difference >> LIMB_BITS) & 1
            index += 1
        difference = (u[j + nb + 1] & LIMB_MASK) - carry - borrow
        u[j + nb + 1] = difference & LIMB_MASK
        if difference < 0:
            # qhat was one too large: add v back
            qhat -= 1
            carry = 0
            index = 0
            while index < nb:
                total = (u[j + index + 1] & LIMB_MASK) + (v[index + 1] & LIMB_MASK) + carry
                u[j + index + 1] = total & LIMB_MASK
                carry = total >> LIMB_BITS
                index += 1
            u[j + nb + 1] = ((u[j + nb + 1] & LIMB_MASK) + carry) & LIMB_MASK
        digits[j] = qhat & LIMB_MASK
        j -= 1
    q.clear()
    q.append(1)
    index = 0
    while index <= m:
        q.append(digits[index])
        index += 1
    _trim(q)
    scaled = [1]
    index = 0
    while index < nb:
        scaled.append(u[index + 1] & LIMB_MASK)
        index += 1
    _shift_magnitude_right(scaled, shift, 1, r)


# --- signed kernels ---------------------------------------------------------


def neg(a: list[int], out: list[int]) -> None:
    """``-a``."""

    _copy_signed(a, -a[0], out)


def add(a: list[int], b: list[int], out: list[int]) -> None:
    """``a + b``."""

    sa = a[0]
    sb = b[0]
    if sa == 0:
        _copy_signed(b, sb, out)
    elif sb == 0:
        _copy_signed(a, sa, out)
    elif sa == sb:
        _add_magnitude(a, b, sa, out)
    elif _compare_magnitude(a, b) >= 0:
        _sub_magnitude(a, b, sa, out)
    else:
        _sub_magnitude(b, a, sb, out)


def sub(a: list[int], b: list[int], out: list[int]) -> None:
    """``a - b``."""

    negated = [0]
    neg(b, negated)
    add(a, negated, out)


def mul(a: list[int], b: list[int], out: list[int]) -> None:
    """``a * b`` (schoolbook)."""

    na = len(a) - 1
    nb = len(b) - 1
    out.clear()
    out.append(a[0] * b[0])
    index = 0
    while index < na + nb:
        out.append(0)
        index += 1
    i = 0
    while i < na:
        x = a[i + 1] & LIMB_MASK
        carry = 0
        j = 0
        while j < nb:
            total = (out[i + j + 1] & LIMB_MASK) + x * (b[j + 1] & LIMB_MASK) + carry
            out[i + j + 1] = total & LIMB_MASK
            carry = total >> LIMB_BITS
            j += 1
        out[i + nb + 1] = carry & LIMB_MASK
        i += 1
    _trim(out)


def compare(a: list[int], b: list[int]) -> int:
    """-1, 0 or 1 as ``a`` is below, equal to or above ``b``."""

    sa = a[0]
    sb = b[0]
    if sa < sb:
        return -1
    if sa > sb:
        return 1
    return sa * _compare_magnitude(a, b)


def less(a: list[int], b: list[int]) -> bool:
    """``a < b``."""

    return compare(a, b) < 0


def equal(a: list[int], b: list[int]) -> bool:
    """``a == b``."""

    return compare(a, b) == 0


def floordiv(a: list[int], b: list[int], out: list[int]) -> None:
    """``a // b``, flooring toward negative infinity."""

    if b[0] == 0:
        raise ZeroDivisionError("integer division or modulo by zero")
    q = [0]
    r = [0]
    _divmod_magnitude(a, b, q, r)
    if a[0] == b[0] or q[0] == 0 and r[0] == 0:
        _copy_signed(q, q[0], out)
    elif r[0] == 0:
        _copy_signed(q, -1, out)
    else:
        one = [1, 1]
        _add_magnitude(q, one, -1, out)


def mod(a: list[int], b: list[int], out: list[int]) -> None:
    """``a % b``; a nonzero remainder takes the divisor's sign."""

    if b[0] == 0:
        raise ZeroDivisionError("integer division or modulo by zero")
    q = [0]
    r = [0]
    _divmod_magnitude(a, b, q, r)
    if r[0] == 0:
        out.clear()
        out.append(0)
    elif a[0] == b[0]:
        _copy_signed(r, b[0], out)
    else:
        _sub_magnitude(b, r, b[0], out)


def shift_left(a: list[int], count: int, out: list[int]) -> None:
    """``a << count``."""

    if count < 0:
        raise ValueError("negative shift count")
    whole = count // LIMB_BITS
    part = count - whole * LIMB_BITS
    shifted = [0]
    _shift_limbs_left(a, part, shifted)
    out.clear()
    out.append(a[0])
    if a[0] != 0:
        index = 0
        while index < whole:
            out.append(0)
            index += 1
        index = 1
        while index < len(shifted):
            out.append(shifted[index] & LIMB_MASK)
            index += 1
    _trim(out)


def shift_right(a: list[int], count: int, out: list[int]) -> None:
    """``a >> count``, flooring toward negative infinity."""

    if count < 0:
        raise ValueError("negative shift count")
    if a[0] < 0:
        # a >> n == -(((|a| - 1) >> n) + 1)
        one = [1, 1]
        smaller = [0]
        _sub_magnitude(a, one, 1, smaller)
        shifted = [0]
        _shift_magnitude_right(smaller, count, 1, shifted)
        _add_magnitude(shifted, one, -1, out)
    else:
        _shift_magnitude_right(a, count, a[0], out)


def _bitwise(a: list[int], b: list[int], op: int, out: list[int]) -> None:
    """``a & b`` (op 0), ``a | b`` (op 1) or ``a ^ b`` (op 2), two's complement."""

    # one limb beyond the longer magnitude holds the sign
    n = len(a)
    if len(b) > n:
        n = len(b)
    x = [0]
    _to_twos(a, n, x)
    y = [0]
    _to_twos(b, n, y)
    t = [0]
    t.clear()
    index = 0
    while index < n:
        left = x[index] & LIMB_MASK
        right = y[index] & LIMB_MASK
        if op == 0:
            t.append(left & right)
        elif op == 1:
            t.append(left | right)
        else:
            t.append(left ^ right)
        index += 1
    _from_twos(t, out)


def and_(a: list[int], b: list[int], out: list[int]) -> None:
    """``a & b``."""

    _bitwise(a, b, 0, out)


def or_(a: list[int], b: list[int], out: list[int]) -> None:
    """``a | b``."""

    _bitwise(a, b, 1, out)


def xor(a: list[int], b: list[int], out: list[int]) -> None:
    """``a ^ b``."""

    _bitwise(a, b, 2, out)


# --- host-side bridge (CPython only; tests and boundaries) -------------------


def from_python(value: int) -> list[int]:
    """The limb value of a Python ``int`` (host side)."""

    if value == 0:
        return [0]
    out = [1 if value > 0 else -1]
    magnitude = -value if value < 0 else value
    while magnitude:
        out.append(magnitude & LIMB_MASK)
        magnitude >>= LIMB_BITS
    return out


def to_python(v: list[int]) -> int:
    """The Python ``int`` of a limb value (host side)."""

    magnitude = 0
    for limb in reversed(v[1:]):
        magnitude = (magnitude << LIMB_BITS) | limb
    return -magnitude if v[0] < 0 else magnitude
