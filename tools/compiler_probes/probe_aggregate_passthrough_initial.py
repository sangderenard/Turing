"""A pass-through fold rebinds a Phi's ``initial_value_id`` through its row.

``legalize_aggregate_output_views`` / ``legalize_aggregate_adapters`` fold a
projected output (``Load(GEP(aggregate, k))``) into the call's own actual and
rebind every use of the projection's value (``_replace_exact_uses``), then
remove the projection's ``Load``.  The operands followed; the return-merge
field Phis' ``initial_value_id`` (the version that stood before the merge) did
not, and kept naming the removed ``Load`` -- an id with no definition in the
function (N=2 orbital dt system: ``pub_limits`` / ``pub_limits_present`` of
``step_with_dt_control_used``).

    python -u tools/compiler_probes/probe_aggregate_passthrough_initial.py

Each fold is now one ``aggregate_passthrough_rebinding`` row, DERIVED from the
original's and the replacement's ``ssa_value`` cells, and the Phi's
``initial_value_id`` is read back from THAT row.
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.concordance_declarations import (  # noqa: E402
    AGGREGATE_PASSTHROUGH_REBINDING, CONTROL_SSA_REGION,
    GRAPH_ID_WITHOUT_CANONICAL_CELL, SSA_VALUE, SSAValueFact, SSAValueOrigin,
)
from src.compiler.identity_concordance import (  # noqa: E402
    Mode, Unsourced, begin_identity_book, end_identity_book,
)
from src.compiler.ssa_aggregate_abi import _replace_exact_uses  # noqa: E402
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


def defined_ids(function):
    return {int(a.id) for a in function.args} | {
        int(i.res.id) for b in function.blocks.values() for i in b.instrs
        if i.res is not None
    }


def stale_initials(function):
    defined = defined_ids(function)
    return [
        (int(i.res.id), int(i.attributes["initial_value_id"]))
        for b in function.blocks.values() for i in b.instrs
        if i.res is not None and i.attributes
        and i.attributes.get("initial_value_id") is not None
        and int(i.attributes["initial_value_id"]) not in defined
    ]


def build():
    """``step``: formal ``actual`` (the call's own operand); the projection
    ``Load`` (the aggregate's member, to be folded into ``actual``); a
    return-merge field Phi whose initial and incomings are that projection;
    a second Phi whose initial is an unrelated value."""
    actual = SSAValue(10, dtype="float64", shape=(44,))
    other = SSAValue(11, dtype="float64", shape=(44,))
    aggregate = SSAValue(12, dtype="float64")
    projected = SSAValue(20, dtype="float64", shape=(44,))
    load = Instr("Load", [aggregate], projected)
    merged = SSAValue(30, dtype="float64", shape=(44,))
    unrelated = SSAValue(31, dtype="float64", shape=(44,))
    phi = Instr("Phi", [projected, projected], merged, attributes={
        "binding": "return_merge", "record_field_phi": True,
        "initial_value_id": 20, "incoming_blocks": ("left", "right"),
    })
    phi_other = Instr("Phi", [other, other], unrelated, attributes={
        "binding": "return_merge", "record_field_phi": True,
        "initial_value_id": 11, "incoming_blocks": ("left", "right"),
    })
    function = Function(
        "step", [actual, other, aggregate],
        {
            "entry": BasicBlock("entry", [
                load, Instr("CondBr", [aggregate], None),
            ], successors=["left", "right"]),
            "left": BasicBlock("left", [Instr("Br", [], None)],
                               successors=["function_exit"]),
            "right": BasicBlock("right", [Instr("Br", [], None)],
                                successors=["function_exit"]),
            "function_exit": BasicBlock("function_exit", [
                phi, phi_other, Instr("Ret", [merged, unrelated], None),
            ]),
        },
    )
    return actual, other, projected, merged, load, phi, phi_other, function


def main() -> int:
    book, token = begin_identity_book()
    try:
        actual, other, projected, merged, load, phi, phi_other, function = build()
        post_values(book, function, actual, other, projected, merged)
        check("before the fold the Phi's initial is defined (the Load)",
              stale_initials(function) == [])
        replacement = SSAValue(10, dtype="float64", shape=(44,), accounting={
            "ssa_storage_alias": 10, "ssa_aggregate_passthrough": ("callee", 7),
        })
        _replace_exact_uses(function, projected, replacement, via=("callee", 7))
        # The pass removes the projection's Load after rebinding.
        function.blocks["entry"].instrs.remove(load)
        check("the Phi's operands follow the fold",
              [int(a.id) for a in phi.args] == [10, 10])
        check("the Phi's initial_value_id follows the fold",
              phi.attributes["initial_value_id"] == 10)
        check("an unrelated Phi's initial is untouched",
              phi_other.attributes["initial_value_id"] == 11)
        check("no initial_value_id names an undefined id after the fold",
              stale_initials(function) == [])
        row = ("step", 20)
        page = book.page(AGGREGATE_PASSTHROUGH_REBINDING)
        print(f"  aggregate_passthrough_rebinding {row} -> {page.latest(row)}")
        check("the fold is a row: (replacement id, callee, output id)",
              page.latest(row) == (10, "callee", 7))
        ref = book.latest_ref(AGGREGATE_PASSTHROUGH_REBINDING, row)
        sources = tuple(s for s, _stage in book.edges_into(ref)) if ref else ()
        print(f"  <- {sources}")
        check("the row derives from the original's and the replacement's "
              "ssa_value cells",
              {s.row[1] for s in sources if s.page.name == "ssa_value"}
              == {10, 20})
    finally:
        end_identity_book(token)
    print("FAILED: " + "; ".join(failures) if failures else "all green")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
