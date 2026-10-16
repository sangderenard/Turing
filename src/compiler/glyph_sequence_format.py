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


def quoted_string_glyph_tokens(value: str) -> list[int]:
    """Return a lossless, Python-readable quoted spelling of ``value``.

    This is the compiler's authored string representation.  It deliberately
    does not call CPython's ``repr``: every delimiter and escape is selected
    here and converted through the incumbent reversible ``StringTable``.
    Printable Unicode remains itself; control and non-printing code points
    use fixed-width hexadecimal escapes.
    """

    quote = '"' if "'" in value and '"' not in value else "'"
    spelling: list[str] = [quote]
    short_escapes = {
        "\\": "\\\\",
        "\t": "\\t",
        "\n": "\\n",
        "\r": "\\r",
    }
    for character in value:
        if character == quote:
            spelling.extend(("\\", quote))
            continue
        escaped = short_escapes.get(character)
        if escaped is not None:
            spelling.extend(escaped)
            continue
        codepoint = ord(character)
        if character.isprintable():
            spelling.append(character)
        elif codepoint <= 0xFF:
            spelling.extend((
                "\\", "x",
                "0123456789abcdef"[(codepoint >> 4) & 0xF],
                "0123456789abcdef"[codepoint & 0xF],
            ))
        elif codepoint <= 0xFFFF:
            spelling.extend(("\\", "u"))
            for shift in (12, 8, 4, 0):
                spelling.append(
                    "0123456789abcdef"[(codepoint >> shift) & 0xF]
                )
        else:
            spelling.extend(("\\", "U"))
            for shift in (28, 24, 20, 16, 12, 8, 4, 0):
                spelling.append(
                    "0123456789abcdef"[(codepoint >> shift) & 0xF]
                )
    spelling.append(quote)
    return [string_token(character) for character in spelling]


def static_literal_repr_glyph_tokens(value: object) -> list[int] | None:
    """Spell the closed literal subset with authored glyph operations.

    ``None`` means that the literal is outside this formatter's declared
    scalar faculty; callers must retain the ordinary resident-value path
    rather than asking the host interpreter for a representation.
    """

    if value is None:
        return [string_token(character) for character in "None"]
    if value is True:
        return [string_token(character) for character in "True"]
    if value is False:
        return [string_token(character) for character in "False"]
    if isinstance(value, int):
        return decimal_integer_glyph_tokens(value)
    if isinstance(value, str):
        return quoted_string_glyph_tokens(value)
    return None


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
