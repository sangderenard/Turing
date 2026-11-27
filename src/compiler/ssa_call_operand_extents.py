"""A call operand must hold at least what its callee formal reads.

Program: declared ``Sum(k * x, (k, 1, n))`` over a tensor ``x`` (LLVM).
Wrong record: region 6's formal for the mask shape is a static two-element
``int32`` span, and the caller bound it to the one-element rank cell.  No
check compared the two, the region read ``shape[1]`` past the cell, and at
O3 that undefined read sized loops that never ended.  Rule: at every
internal call, an operand whose posted row states a static element count
(a rank-0 scalar holds one, a static-shape span holds its count) must hold
at least the static count its callee formal's posted row states.  A runtime
span states no static count and is not judged here: its extents are runtime
facts, guarded where they are computed.

Backends read this one rule (LLVM and C call emission), so a refusal is the
same refusal in every lane.
"""

from __future__ import annotations

from math import prod
from typing import Any


def _static_count(fact: Any, value: Any) -> int | None:
    """The element count a posted row states statically, or None."""

    if fact is None:
        return None
    storage = str(getattr(fact, "storage", "") or "")
    shape = tuple(getattr(fact, "shape", None) or getattr(value, "shape", None) or ())
    if storage == "scalar" and int(getattr(fact, "rank", 0) or 0) == 0 and not shape:
        return 1
    if storage == "span" and shape and all(
        isinstance(extent, int) and extent > 0 for extent in shape
    ):
        return int(prod(shape))
    return None


def undersized_call_operands(
    module: Any, caller_name: str, callee_name: str, arguments: Any,
) -> list[tuple[int, int, int, int]]:
    """``(position, actual id, holds, formal reads)`` for each operand that
    holds fewer static elements than its callee formal reads."""

    from .identity_concordance import identity_book
    from .ssa_record_return_state import posted_ssa_value_fact

    functions = getattr(module, "functions", {}) or {}
    caller = functions.get(str(caller_name))
    callee = functions.get(str(callee_name))
    if caller is None or callee is None:
        return []
    book = identity_book(module)
    found: list[tuple[int, int, int, int]] = []
    for position, (actual, formal) in enumerate(zip(arguments, callee.args)):
        reads = _static_count(
            posted_ssa_value_fact(callee, int(formal.id), book=book), formal,
        )
        if reads is None or reads <= 1:
            continue
        holds = _static_count(
            posted_ssa_value_fact(caller, int(actual.id), book=book), actual,
        )
        if holds is not None and holds < reads:
            found.append((position, int(actual.id), holds, reads))
    return found


def undersized_call_message(
    callee_name: str, findings: list[tuple[int, int, int, int]],
) -> str:
    return (
        f"call to {callee_name} binds operand(s) smaller than the formal "
        "reads: "
        + "; ".join(
            f"position {position} (%t{actual}) holds {holds} element(s), "
            f"the formal reads {reads}"
            for position, actual, holds, reads in findings
        )
        + ". Refusing: the callee would read past the operand's storage."
    )
