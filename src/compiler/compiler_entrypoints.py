"""Canonical whole-source compiler entry-point policy.

Only :func:`src.compiler.fortran_c_shell.lower_ast_source_to_ssa` is the
public whole-program source compiler.  This policy deprecates *independent
source ingestion and product selection* by older entry points; it does not
declare their downstream compiler machinery obsolete.

In particular, ``compile_ast_aot`` remains a compatibility adapter used by
many internal callers, and its ProcessGraph planning stages include proven
contracts that the canonical entry must share: bound-feed specialization,
linked-call propagation, hierarchy planning, and backend-neutral control
capture.  Migrating a caller to the canonical entry means producing repository
SSA directly instead of exposing an intermediate ``FusedProgram`` as the
application-facing product.  It does **not** mean copying, bypassing, or
silently reimplementing those established planning stages.

Backend emitters and IR-to-IR lowerers likewise remain supported implementation
stages.  Compatibility source adapters announce their status so new external
callers converge on one source API while existing internal compiler paths can
be migrated deliberately and with parity tests.
"""

from __future__ import annotations

import warnings


CANONICAL_SOURCE_COMPILER = (
    "src.compiler.fortran_c_shell.lower_ast_source_to_ssa"
)


def warn_legacy_source_compiler(name: str) -> None:
    """Mark an old public source entry without disowning its internal stages.

    The warning directs application callers to the canonical repository-SSA
    product.  It is not permission to remove the adapter, skip its proven
    ProcessGraph contracts, or treat every function it calls as deprecated.
    """

    warnings.warn(
        f"{name} is a deprecated source-compilation entry point; use "
        f"{CANONICAL_SOURCE_COMPILER}. It ingests the complete authored "
        "program and lowers control, arithmetic, tensors, calls, and memory "
        "directly to repository SSA without numerical projection.",
        DeprecationWarning,
        stacklevel=2,
    )


__all__ = ["CANONICAL_SOURCE_COMPILER", "warn_legacy_source_compiler"]
