"""Declare the physical cell of structural ``None``.

``None`` is a semantic dtype (``"none"``) with no scalar layout of its own: the
dtype catalogue deliberately has no row for it, so a backend that must allocate
or return the value asks the value's recorded PHYSICAL dtype instead.  Native
``None`` is the record/token zero sentinel, carried in the repository's inert
scalar cell (the C lane's declared carrier for conceptual ``None``).  Nothing
used to record that physical fact, so each resident ``none`` value reached the
emitters with only its semantic spelling and the layout catalogue refused it.

This pass is the writer of that fact.  Rule: the ``ssa_value`` row is posted
first and every resident occurrence's accounting mirrors the row.
"""

from __future__ import annotations

from typing import Any

#: Physical carrier of the zero sentinel (see ``ssa_c_backend._scalar_c_type``).
NONE_SENTINEL_PHYSICAL_DTYPE = "float64"


def declare_none_sentinel_carriers(functions: Any) -> int:
    """Post the sentinel cell for every scalar ``none`` value; return the count."""

    from .concordance_declarations import SCALAR_DTYPE_SETTLEMENT
    from .frame_identity_book import (
        _frame_accounting_from_posted, _frame_physical_fields,
    )
    from .identity_concordance import current_identity_book
    from .ssa_record_return_state import (
        accepted_ssa_component, revise_ssa_value,
    )

    book = current_identity_book()
    declared = 0
    for function in dict(functions or {}).values():
        if accepted_ssa_component(function):
            continue  # immutable source; its own unit declared its Nones
        occurrences: dict[int, list[Any]] = {}
        for value in (
            *function.args,
            *(
                item
                for block in function.blocks.values()
                for instruction in block.instrs
                for item in (*instruction.args, instruction.res)
                if item is not None
            ),
        ):
            if str(value.dtype or "").casefold() != "none" or value.shape:
                continue
            bucket = occurrences.setdefault(int(value.id), [])
            if not any(value is incumbent for incumbent in bucket):
                bucket.append(value)
        for value_id, values in occurrences.items():
            cell = revise_ssa_value(
                function, value_id, dtype=values[0].dtype, shape=(),
                stage=SCALAR_DTYPE_SETTLEMENT, book=book,
                physical={
                    "physical_dtype": NONE_SENTINEL_PHYSICAL_DTYPE,
                    "storage": "scalar", "rank": 0, "arena": False,
                },
            )
            if cell is None:
                continue
            posted = _frame_physical_fields(function, value_id)
            for value in values:
                value.accounting = _frame_accounting_from_posted(
                    value.accounting, posted,
                )
            declared += 1
    return declared


__all__ = ["declare_none_sentinel_carriers", "NONE_SENTINEL_PHYSICAL_DTYPE"]
