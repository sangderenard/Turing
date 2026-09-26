"""Shared edge-role contract for ProcessGraph compiler consumers."""

from __future__ import annotations

# These edges preserve program structure and whole-entrypoint reachability.
# They must never be interpreted as numerical operands by SSA or backend
# region compilers.
NON_VALUE_EDGE_ROLES = frozenset(
    {
        "body",
        "compiles",
        "contains",
        "else",
        "environment",
        "finally",
        "handler",
        "invokes",
        "orelse",
        "parameter",
        "then",
    }
)


__all__ = ["NON_VALUE_EDGE_ROLES"]
