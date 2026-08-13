"""Canonical whole-source compiler entry-point policy.

Only :func:`src.compiler.fortran_c_shell.lower_ast_source_to_ssa` is the
public whole-program source compiler.  Backend emitters and IR-to-IR lowerers
remain supported implementation stages; older functions that independently
ingest source are compatibility adapters and must announce that fact.
"""

from __future__ import annotations

import warnings


CANONICAL_SOURCE_COMPILER = (
    "src.compiler.fortran_c_shell.lower_ast_source_to_ssa"
)


def warn_legacy_source_compiler(name: str) -> None:
    """Mark a compatibility source compiler without disabling old callers."""

    warnings.warn(
        f"{name} is a deprecated source-compilation entry point; use "
        f"{CANONICAL_SOURCE_COMPILER}. It ingests the complete authored "
        "program and lowers control, arithmetic, tensors, calls, and memory "
        "directly to repository SSA without numerical projection.",
        DeprecationWarning,
        stacklevel=2,
    )


__all__ = ["CANONICAL_SOURCE_COMPILER", "warn_legacy_source_compiler"]
