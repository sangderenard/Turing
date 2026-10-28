"""A callsite marker's result retention reads the caller's sequence table for any call.

``return low ^ byte, high`` (a callee that forwards a formal) reached the marker
retention without a method receiver, where ``caller_sequence_table`` had never
been assigned: a NameError out of ``replace_at_callsite_marker``.  The table is
read for every call now, so the program reaches the concordance and is refused
(or lowered) by name, never by a crash.
"""

from __future__ import annotations

from pathlib import Path

CONTRACT = str(
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)

SOURCE = (
    "def step(high: int, low: int, byte: int):\n"
    "    return low ^ byte, high\n"
    "\n"
    "def inner(high: int, low: int, atom: int):\n"
    "    index = 0\n"
    "    while index < 4:\n"
    "        high, low = step(high, low, index + atom)\n"
    "        index += 1\n"
    "    return high, low\n"
    "\n"
    "def root(domain: int, n: int):\n"
    "    h, l = inner(3, 5, domain)\n"
    "    return h, l\n"
)


def test_forwarding_callee_is_not_a_name_error():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    try:
        lower_ast_source_to_ssa(
            SOURCE, "root", name="x", extraction_contract=CONTRACT,
            progress=lambda _: None,
        )
    except NameError as error:  # the crash this guards against
        raise AssertionError(f"callsite marker NameError: {error}") from error
    except Exception:
        pass  # a named refusal is acceptable; a crash is not
