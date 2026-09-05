"""JavaScript-family spellings for the repository's existing SSA vocabulary.

Mirrors :mod:`glsl_source_tables`: these tables are lexical only. They select
an existing :class:`Handler` for a JS surface spelling; they do not add SSA
operations and JS syntax does not survive past the lookup. Operators whose
JS semantics have no exact existing Handler (type-coercing equality, unsigned
shift, unary `+`/`typeof`/`void`/`delete`) are listed as unsupported rather
than mapped to a Handler that would misrepresent their behavior.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..transmogrifier.ssa_registry import Handler


JAVASCRIPT_BINARY_TO_SSA: Mapping[str, Handler] = MappingProxyType({
    "+": Handler.Add,
    "-": Handler.Sub,
    "*": Handler.Mul,
    "/": Handler.Div,
    "%": Handler.Mod,
    "**": Handler.Pow,
    "<<": Handler.Shl,
    ">>": Handler.Shr,
    "&": Handler.And,
    "|": Handler.Or,
    "^": Handler.Xor,
    "&&": Handler.LAnd,
    "||": Handler.LOr,
    "===": Handler.Eq,
    "!==": Handler.Ne,
    "<": Handler.Lt,
    "<=": Handler.Le,
    ">": Handler.Gt,
    ">=": Handler.Ge,
})

JAVASCRIPT_UNARY_TO_SSA: Mapping[str, Handler] = MappingProxyType({
    "-": Handler.Neg,
    "!": Handler.LNot,
    "~": Handler.Not,
})

# One deterministic surface spelling per existing Handler. '==' and '!=' are
# deliberately excluded from the forward table (see JAVASCRIPT_UNSUPPORTED_*
# below), so '===' / '!==' are the sole, bijective spellings for Eq / Ne.
SSA_TO_JAVASCRIPT_BINARY: Mapping[Handler, str] = MappingProxyType({
    handler: spelling for spelling, handler in JAVASCRIPT_BINARY_TO_SSA.items()
})
SSA_TO_JAVASCRIPT_UNARY: Mapping[Handler, str] = MappingProxyType({
    handler: spelling for spelling, handler in JAVASCRIPT_UNARY_TO_SSA.items()
})

# Listed for precise diagnostics. Loose '==' / '!=' involve type coercion,
# '>>>' is unsigned shift, and unary '+'/'typeof'/'void'/'delete' have no
# arithmetic Handler equivalent. Naming them here is not a lowering table and
# does not invent an opcode -- the same discipline glsl_source_tables.py uses
# for WEBGL_UNLOWERED_CALLS.
JAVASCRIPT_UNSUPPORTED_BINARY = frozenset({"==", "!=", ">>>"})
JAVASCRIPT_UNSUPPORTED_UNARY = frozenset({"+", "typeof", "void", "delete"})

# Calls listed here already exist on the ProcessGraph/AbstractTensor surface,
# the same shared vocabulary GLSL_DIRECT_CALLS selects into via Handler.Call.
JAVASCRIPT_DIRECT_CALLS: Mapping[str, str] = MappingProxyType({
    spelling: name
    for spelling, name in (
        ("Math.abs", "abs"), ("Math.acos", "acos"), ("Math.acosh", "acosh"),
        ("Math.asin", "asin"), ("Math.asinh", "asinh"), ("Math.atan", "atan"),
        ("Math.atanh", "atanh"), ("Math.ceil", "ceil"), ("Math.cos", "cos"),
        ("Math.cosh", "cosh"), ("Math.exp", "exp"), ("Math.floor", "floor"),
        ("Math.log", "log"), ("Math.max", "maximum"), ("Math.min", "minimum"),
        ("Math.pow", "pow"), ("Math.round", "round"), ("Math.sign", "sign"),
        ("Math.sin", "sin"), ("Math.sinh", "sinh"), ("Math.sqrt", "sqrt"),
        ("Math.tan", "tan"), ("Math.tanh", "tanh"), ("Math.trunc", "trunc"),
        ("Number.isNaN", "isnan"),
    )
})


def validate_invertible_tables() -> None:
    """Fail if a lexical table maps two spellings to one emitted spelling."""

    for name, forward, reverse in (
        ("binary", JAVASCRIPT_BINARY_TO_SSA, SSA_TO_JAVASCRIPT_BINARY),
        ("unary", JAVASCRIPT_UNARY_TO_SSA, SSA_TO_JAVASCRIPT_UNARY),
    ):
        for spelling, handler in forward.items():
            if reverse.get(handler) != spelling:
                raise ValueError(
                    f"javascript {name} table is not invertible at {spelling!r}"
                )
    overlap = set(JAVASCRIPT_BINARY_TO_SSA) & JAVASCRIPT_UNSUPPORTED_BINARY
    if overlap:
        raise ValueError(
            f"javascript binary spellings both lowered and unsupported: {overlap!r}"
        )
    overlap = set(JAVASCRIPT_UNARY_TO_SSA) & JAVASCRIPT_UNSUPPORTED_UNARY
    if overlap:
        raise ValueError(
            f"javascript unary spellings both lowered and unsupported: {overlap!r}"
        )


validate_invertible_tables()


__all__ = [
    "JAVASCRIPT_BINARY_TO_SSA",
    "JAVASCRIPT_DIRECT_CALLS",
    "JAVASCRIPT_UNARY_TO_SSA",
    "JAVASCRIPT_UNSUPPORTED_BINARY",
    "JAVASCRIPT_UNSUPPORTED_UNARY",
    "SSA_TO_JAVASCRIPT_BINARY",
    "SSA_TO_JAVASCRIPT_UNARY",
    "validate_invertible_tables",
]
