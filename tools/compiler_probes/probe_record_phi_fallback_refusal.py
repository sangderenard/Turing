"""A record-field Phi read where it does not dominate, whose recorded initial
has no definition, is refused -- loudly, on the occurrence's row.

``repair_non_dominating_record_phi_uses`` substitutes a record-field Phi's
recorded ``initial_value_id`` for a use the Phi's block does not dominate.  A
Phi whose initial names an id the function does not define used to be skipped
with no word (``if fallback is None: continue``), leaving a read of a value
that is not there on that path (N=2 orbital dt system, before the pass-through
fold carried ``initial_value_id``: ``pub_limits`` of
``step_with_dt_control_used``).

    python -u tools/compiler_probes/probe_record_phi_fallback_refusal.py
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.concordance_declarations import (  # noqa: E402
    CONTROL_SSA_REGION, GRAPH_ID_WITHOUT_CANONICAL_CELL,
    RECORD_PHI_FALLBACK_NOT_DEFINED, RECORD_PHI_TEMPORAL_FALLBACK, SSA_VALUE,
    SSAValueFact, SSAValueOrigin,
)
from src.compiler.identity_concordance import (  # noqa: E402
    Mode, Unresolved, Unsourced, begin_identity_book, end_identity_book,
)
from src.compiler.ssa_record_return_state import (  # noqa: E402
    repair_non_dominating_record_phi_uses,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


def post_values(book, function, *values):
    for value in values:
        book.post(
            SSA_VALUE, (function.name, int(value.id)),
            SSAValueFact("float64", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
            stage=CONTROL_SSA_REGION,
            provenance=Unsourced(GRAPH_ID_WITHOUT_CANONICAL_CELL),
            mode=Mode.REVISE,
        )


def build(initial_id, *, use_after_merge=False):
    """``step``: formal ``f``; ``body`` reads the Phi's result before the merge
    (``function_exit``); the Phi's recorded initial is ``initial_id``."""
    formal = SSAValue(10, dtype="float64")
    result = SSAValue(20, dtype="float64")
    use = Instr("Call", [result], None, attributes={"callee": "consume"})
    phi = Instr("Phi", [formal], result, attributes={
        "binding": "return_merge", "record_field_phi": True,
        "initial_value_id": initial_id, "incoming_blocks": ("body",),
    })
    exit_use = Instr("Call", [result], None, attributes={"callee": "consume"})
    return formal, result, use, exit_use, Function(
        "step", [formal],
        {
            "entry": BasicBlock(
                "entry", [Instr("Br", [], None)], successors=["body"]),
            "body": BasicBlock(
                "body", [] if use_after_merge else [use, Instr("Br", [], None)],
                successors=["function_exit"]),
            "function_exit": BasicBlock("function_exit", [
                phi, *([exit_use] if use_after_merge else []),
                Instr("Ret", [result], None),
            ]),
        },
    )


def scenario(initial_id, **kwargs):
    """A fresh book and a fresh function per scenario (the page is CONCORD:
    one answer per occurrence)."""
    book, token = begin_identity_book()
    formal, result, use, exit_use, function = build(initial_id, **kwargs)
    post_values(book, function, formal, result)
    return book, token, formal, result, use, exit_use, function


def main() -> int:
    book, token, formal, result, use, _exit_use, function = scenario(10)
    try:
        receipts = repair_non_dominating_record_phi_uses(function)
        check("a defined initial still repairs the use",
              use.args == [formal] and len(receipts) >= 1)
    finally:
        end_identity_book(token)

    book, token, formal, result, use, _exit_use, function = scenario(99)
    try:
        try:
            repair_non_dominating_record_phi_uses(function)
        except ValueError as error:
            print(f"  refused: {error}")
            check("an undefined initial refuses the module", True)
            check("the refusal names the Phi, the use and the initial",
                  "%20" in str(error) and "%99" in str(error)
                  and "body[0]" in str(error))
        else:
            check("an undefined initial refuses the module", False)
        row = ("step", 20, "body", 0, 0)
        fact = book.page(RECORD_PHI_TEMPORAL_FALLBACK).latest(row)
        print(f"  row {row} -> {fact}")
        check("the occurrence's row is Unresolved(record_phi_fallback_not_defined)",
              isinstance(fact, Unresolved)
              and fact.reason == RECORD_PHI_FALLBACK_NOT_DEFINED)
        check("the use was not touched", use.args == [result])
    finally:
        end_identity_book(token)

    book, token, formal, result, use, exit_use, function = scenario(
        99, use_after_merge=True)
    try:
        try:
            receipts = repair_non_dominating_record_phi_uses(function)
            check("a use the Phi dominates needs no initial: no refusal, "
                  "no substitution", receipts == () and exit_use.args == [result])
        except ValueError as error:
            check(f"a use the Phi dominates needs no initial: {error}", False)
    finally:
        end_identity_book(token)
    print("FAILED: " + "; ".join(failures) if failures else "all green")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
