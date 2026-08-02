"""GLSL-family spellings for the repository's existing SSA vocabulary.

These tables are intentionally lexical.  They do not add operations to SSA and
they are not an IR in which GLSL constructs survive.  The source reader uses a
spelling to select an existing :class:`Handler`; emission uses the preferred
spelling for that same Handler.  Built-ins that need more than one Handler are
lowered by ``glsl_source_ingestion`` and therefore do not appear here as SSA
operations.

WebGL 2 borrows the ordinary GLSL expression spellings.  Its small overlay is
kept separate so target-only syntax can be rejected or specialized without
copying the common table.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..transmogrifier.ssa_registry import Handler


GLSL_BINARY_TO_SSA: Mapping[str, Handler] = MappingProxyType({
    "+": Handler.Add,
    "-": Handler.Sub,
    "*": Handler.Mul,
    "/": Handler.Div,
    "%": Handler.Mod,
    "<<": Handler.Shl,
    ">>": Handler.Shr,
    "&": Handler.And,
    "|": Handler.Or,
    "^": Handler.Xor,
    "&&": Handler.LAnd,
    "||": Handler.LOr,
    "==": Handler.Eq,
    "!=": Handler.Ne,
    "<": Handler.Lt,
    "<=": Handler.Le,
    ">": Handler.Gt,
    ">=": Handler.Ge,
})

GLSL_UNARY_TO_SSA: Mapping[str, Handler] = MappingProxyType({
    "-": Handler.Neg,
    "!": Handler.LNot,
    "~": Handler.Not,
})

# One deterministic surface spelling per existing Handler.  Unary and binary
# '-' are disambiguated by parser position; every other spelling is bijective.
SSA_TO_GLSL_BINARY: Mapping[Handler, str] = MappingProxyType({
    handler: spelling for spelling, handler in GLSL_BINARY_TO_SSA.items()
})
SSA_TO_GLSL_UNARY: Mapping[Handler, str] = MappingProxyType({
    handler: spelling for spelling, handler in GLSL_UNARY_TO_SSA.items()
})

# Calls listed here already exist on the ProcessGraph/AbstractTensor surface.
# They lower through the existing Handler.Call with the canonical callee name;
# the GLSL spelling itself is not retained.
GLSL_DIRECT_CALLS: Mapping[str, str] = MappingProxyType({
    name: name
    for name in (
        "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh",
        "ceil", "cos", "cosh", "cross", "dot", "exp", "floor",
        "isinf", "isnan", "log", "maximum", "minimum", "pow", "round",
        "sign", "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
    )
})

# Constructors select an existing conversion Handler.  Vector/matrix
# constructors are deliberately absent: scalar SSA cannot pretend a structured
# constructor is scalar, and the normal shortfall path reports them.
GLSL_CASTS: Mapping[str, Handler] = MappingProxyType({
    "bool": Handler.Cast,
    "float": Handler.Cast,
    "double": Handler.Cast,
    "int": Handler.Cast,
    "uint": Handler.Cast,
})

# WebGL expressions inherit all common spellings.  These are target-only
# source calls for which the current SSA/ProcessGraph surface has no exact
# operation.  Listing them is useful for precise diagnostics; it is not a
# lowering table and does not invent an opcode.
WEBGL_UNLOWERED_CALLS = frozenset({
    "dFdx", "dFdy", "fwidth", "texture", "textureLod", "texelFetch",
})


def validate_invertible_tables() -> None:
    """Fail if a lexical table maps two spellings to one emitted spelling."""

    for name, forward, reverse in (
        ("binary", GLSL_BINARY_TO_SSA, SSA_TO_GLSL_BINARY),
        ("unary", GLSL_UNARY_TO_SSA, SSA_TO_GLSL_UNARY),
    ):
        for spelling, handler in forward.items():
            if reverse.get(handler) != spelling:
                raise ValueError(
                    f"GLSL {name} table is not invertible at {spelling!r}"
                )


validate_invertible_tables()


__all__ = [
    "GLSL_BINARY_TO_SSA",
    "GLSL_CASTS",
    "GLSL_DIRECT_CALLS",
    "GLSL_UNARY_TO_SSA",
    "SSA_TO_GLSL_BINARY",
    "SSA_TO_GLSL_UNARY",
    "WEBGL_UNLOWERED_CALLS",
    "validate_invertible_tables",
]
