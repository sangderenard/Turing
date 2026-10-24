"""JavaScript-family spellings for the repository's existing SSA vocabulary.

Mirrors :mod:`glsl_source_tables`: these tables are lexical only. They select
an existing :class:`Handler` for a JS surface spelling; they do not add SSA
operations and JS syntax does not survive past the lookup. Operators whose
JS semantics have no exact existing Handler (type-coercing equality,
ECMAScript remainder, unsigned shift, unary `+`/`typeof`/`void`/`delete`) are
listed as unsupported rather than mapped to a Handler that would misrepresent
their behavior. The semantic interpreter separately requires proved numeric,
coercion, realm and effect facets before using any lexical selection here.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..transmogrifier.ssa_registry import Handler
from ..transmogrifier.graph.javascript_identity_programs import (
    javascript_surface_identities,
    resolve_javascript_identity,
)


JAVASCRIPT_BINARY_TO_SSA: Mapping[str, Handler] = MappingProxyType({
    "+": Handler.Add,
    "-": Handler.Sub,
    "*": Handler.Mul,
    "/": Handler.Div,
    "**": Handler.Pow,
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

# Listed for precise diagnostics. Loose '==' / '!=' involve type coercion;
# shifts and bitwise operators need ECMAScript's ToInt32/ToUint32 programs.  ``Handler.And``/``Or``/``Xor`` and ``Handler.Not`` are logical in
# the repository evaluator, so selecting them here would be a silent semantic
# substitution.  They remain named shortfalls until those programs exist.
JAVASCRIPT_UNSUPPORTED_BINARY = frozenset({
    "==", "!=", "<<", ">>", ">>>", "&", "|", "^",
})
# ``%`` has no single Handler (the repository Mod is floored); the
# interpreter lowers it through its ECMAScript remainder program.
JAVASCRIPT_UNSUPPORTED_UNARY = frozenset({
    "+", "~", "typeof", "void", "delete",
})

# Calls listed here already exist on the ProcessGraph/AbstractTensor surface,
# the same shared vocabulary GLSL_DIRECT_CALLS selects into via Handler.Call.
JAVASCRIPT_SURFACE_IDENTITIES: Mapping[str, str] = javascript_surface_identities()
JAVASCRIPT_DIRECT_CALLS: Mapping[str, str] = MappingProxyType({
    spelling: resolve_javascript_identity(identity).operator
    for spelling, identity in JAVASCRIPT_SURFACE_IDENTITIES.items()
})

# These familiar spellings are intentionally not replacement identities.
# ECMAScript Math.round is ties-toward-+infinity, whereas the repository's
# existing round operation is ties-to-even.  A later exact JS program may
# decompose that behavior using existing operations; mapping it to ``round``
# here would silently change results.
JAVASCRIPT_UNSUPPORTED_CALLS = MappingProxyType({
    "Math.round": "ECMAScript ties-toward-+infinity has no exact direct operator",
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
    "JAVASCRIPT_SURFACE_IDENTITIES",
    "JAVASCRIPT_UNSUPPORTED_CALLS",
    "JAVASCRIPT_UNARY_TO_SSA",
    "JAVASCRIPT_UNSUPPORTED_BINARY",
    "JAVASCRIPT_UNSUPPORTED_UNARY",
    "SSA_TO_JAVASCRIPT_BINARY",
    "SSA_TO_JAVASCRIPT_UNARY",
    "validate_invertible_tables",
]
