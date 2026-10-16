"""Authored scalar formatting for resident glyph-token sequences.

These are ordinary Python helpers on ``list[int]``.  Source pursuit lowers
their arithmetic, loops, branches, and ``append`` calls through the normal
ProcessGraph and resident-sequence path; no backend formatting opcode or host
object is involved.  Each list element is the incumbent reversible
``StringTable`` token for one character.
"""
from __future__ import annotations

from .string_table import string_token


_GLYPH_MINUS = string_token("-")
_GLYPH_LOWER_X = string_token("x")
_GLYPH_DIGITS = (
    string_token("0"),
    string_token("1"),
    string_token("2"),
    string_token("3"),
    string_token("4"),
    string_token("5"),
    string_token("6"),
    string_token("7"),
    string_token("8"),
    string_token("9"),
)
_GLYPH_LOWER_HEX_DIGITS = (
    string_token("0"),
    string_token("1"),
    string_token("2"),
    string_token("3"),
    string_token("4"),
    string_token("5"),
    string_token("6"),
    string_token("7"),
    string_token("8"),
    string_token("9"),
    string_token("a"),
    string_token("b"),
    string_token("c"),
    string_token("d"),
    string_token("e"),
    string_token("f"),
)
_INT64_MIN = -(1 << 63)
_UINT32_BASE = 4294967296
_UINT32_SIGN = 2147483648
_FNV_OFFSET_LOW = 2216829733
_FNV_OFFSET_HIGH = 3421674724
_FNV_PRIME_LOW = 435
_FNV_PRIME_HIGH = 256


def runtime_repr_utf8(value: object) -> str:
    """Return the exact Python repr consumed by the sanctioned host boundary.

    This function is never source-pursued.  The extraction contract binds it
    as one exact CPython external call whose result ABI is a resident UTF-8
    byte sequence.  Keeping the boundary on a repository-owned symbol makes
    the permission specific to unresolved ``FormattedValue(..., !r)`` rather
    than changing the policy for authored calls to ``builtins.repr``.
    """

    return repr(value)


def utf8_repr_bytes_to_glyph_tokens(value: list[int]) -> list[int]:
    """Decode one valid UTF-8 byte sequence into reversible glyph identities.

    A glyph identity is the FNV-1a hash of one Unicode scalar's exact UTF-8
    spelling.  The arithmetic is expressed with 32-bit limbs so authored
    Python and fixed-width native lowering agree without relying on signed
    overflow or constants outside int64.  The CPython adapter is the authority
    that the input is valid UTF-8; this helper only segments its scalar spans.
    """

    result: list[int] = []
    index = 0
    size = len(value)
    while index < size:
        first = value[index]
        width = 1
        if first >= 240:
            width = 4
        elif first >= 224:
            width = 3
        elif first >= 192:
            width = 2
        end = index + width
        low = _FNV_OFFSET_LOW
        high = _FNV_OFFSET_HIGH
        while index < end:
            low ^= value[index]
            low_product = low * _FNV_PRIME_LOW
            carry = low_product // _UINT32_BASE
            new_low = low_product % _UINT32_BASE
            high = (
                high * _FNV_PRIME_LOW
                + low * _FNV_PRIME_HIGH
                + carry
            ) % _UINT32_BASE
            low = new_low
            index += 1
        if high >= _UINT32_SIGN:
            token = (high - _UINT32_BASE) * _UINT32_BASE + low
        else:
            token = high * _UINT32_BASE + low
        result.append(token)
    return result


def decimal_digit_glyph_token(digit: int) -> int:
    """Return the single-glyph token for a proven decimal digit."""

    if digit == 0:
        return _GLYPH_DIGITS[0]
    if digit == 1:
        return _GLYPH_DIGITS[1]
    if digit == 2:
        return _GLYPH_DIGITS[2]
    if digit == 3:
        return _GLYPH_DIGITS[3]
    if digit == 4:
        return _GLYPH_DIGITS[4]
    if digit == 5:
        return _GLYPH_DIGITS[5]
    if digit == 6:
        return _GLYPH_DIGITS[6]
    if digit == 7:
        return _GLYPH_DIGITS[7]
    if digit == 8:
        return _GLYPH_DIGITS[8]
    return _GLYPH_DIGITS[9]


def append_decimal_integer_glyphs(destination: list[int], value: int) -> None:
    """Append Python's base-ten spelling of one signed int64 value.

    The minimum int64 value is written literally so negation never overflows
    in native fixed-width arithmetic.  Every other value is emitted from the
    most-significant digit, avoiding a second sequence or a reverse mutation.
    """

    if value == _INT64_MIN:
        destination.append(_GLYPH_MINUS)
        destination.append(_GLYPH_DIGITS[9])
        destination.append(_GLYPH_DIGITS[2])
        destination.append(_GLYPH_DIGITS[2])
        destination.append(_GLYPH_DIGITS[3])
        destination.append(_GLYPH_DIGITS[3])
        destination.append(_GLYPH_DIGITS[7])
        destination.append(_GLYPH_DIGITS[2])
        destination.append(_GLYPH_DIGITS[0])
        destination.append(_GLYPH_DIGITS[3])
        destination.append(_GLYPH_DIGITS[6])
        destination.append(_GLYPH_DIGITS[8])
        destination.append(_GLYPH_DIGITS[5])
        destination.append(_GLYPH_DIGITS[4])
        destination.append(_GLYPH_DIGITS[7])
        destination.append(_GLYPH_DIGITS[7])
        destination.append(_GLYPH_DIGITS[5])
        destination.append(_GLYPH_DIGITS[8])
        destination.append(_GLYPH_DIGITS[0])
        destination.append(_GLYPH_DIGITS[8])
        return
    if value < 0:
        destination.append(_GLYPH_MINUS)
        value = -value
    if value == 0:
        destination.append(_GLYPH_DIGITS[0])
        return
    divisor = 1
    while value // divisor >= 10:
        divisor *= 10
    while divisor > 0:
        digit = value // divisor
        destination.append(decimal_digit_glyph_token(digit))
        value %= divisor
        divisor //= 10


def decimal_integer_glyph_tokens(value: int) -> list[int]:
    """Construct one resident-sequence-shaped integer spelling."""

    result: list[int] = []
    append_decimal_integer_glyphs(result, value)
    return result


def lower_hex_digit_glyph_token(digit: int) -> int:
    """Return the single-glyph token for a proven hexadecimal digit."""

    if digit < 10:
        return decimal_digit_glyph_token(digit)
    if digit == 10:
        return _GLYPH_LOWER_HEX_DIGITS[10]
    if digit == 11:
        return _GLYPH_LOWER_HEX_DIGITS[11]
    if digit == 12:
        return _GLYPH_LOWER_HEX_DIGITS[12]
    if digit == 13:
        return _GLYPH_LOWER_HEX_DIGITS[13]
    if digit == 14:
        return _GLYPH_LOWER_HEX_DIGITS[14]
    return _GLYPH_LOWER_HEX_DIGITS[15]


def prefixed_lower_hex_integer_glyph_tokens(value: int) -> list[int]:
    """Construct Python's ``#x`` spelling for one signed int64 value."""

    result: list[int] = []
    if value == _INT64_MIN:
        result.append(_GLYPH_MINUS)
        result.append(_GLYPH_DIGITS[0])
        result.append(_GLYPH_LOWER_X)
        result.append(_GLYPH_LOWER_HEX_DIGITS[8])
        index = 0
        while index < 15:
            result.append(_GLYPH_DIGITS[0])
            index += 1
        return result
    if value < 0:
        result.append(_GLYPH_MINUS)
        value = -value
    result.append(_GLYPH_DIGITS[0])
    result.append(_GLYPH_LOWER_X)
    if value == 0:
        result.append(_GLYPH_DIGITS[0])
        return result
    divisor = 1
    while value // divisor >= 16:
        divisor *= 16
    while divisor > 0:
        digit = value // divisor
        result.append(lower_hex_digit_glyph_token(digit))
        value %= divisor
        divisor //= 16
    return result
