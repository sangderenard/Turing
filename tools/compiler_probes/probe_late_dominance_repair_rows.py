"""The late dominance repairs run before the full-native gate and post rows.

``_canonicalize_non_dominating_loop_result_uses`` and
``repair_non_dominating_record_phi_uses`` substitute an operand only when CFG
dominance proves the replacement available.  They used to run in the public
wrapper, AFTER the implementation's first ``concord_compiler_frame_formals``
had CONCORDed the unrepaired actual (a return-merge field Phi in
``function_exit`` that does not dominate the use); the wrapper's second
concordance pass then proposed the repaired actual for the same immutable
``formal_actual_occurrence_concordance`` row (N=2 orbital dt system,
``step_1__coerce_metrics`` called in ``if_merge`` of
``step_with_dt_control_used``).  Each substitution was recorded only in
function metadata.

    python -u tools/compiler_probes/probe_late_dominance_repair_rows.py

Part A (hand-built IR, a real book): each repair posts one row per
substituted occurrence whose edges name the original's and the replacement's
``ssa_value`` cells and the ``ssa_block`` cell(s) of the dominance proof.
(``ssa_value`` rows are posted ``Unsourced(GRAPH_ID_WITHOUT_CANONICAL_CELL)``
as for any value adopted without a control lowering; the repair rows must
derive from them.)

Part B (a real program through ``lower_ast_source_to_ssa`` under the
full-native contract): the pre-gate repair ran and left nothing for the
wrapper (``_settle_late_dominance_repairs`` finds 0 afterwards).  This
program does not itself need a repair; the N=2 orbital dt system is the
program that does, and it is the check for the ordering.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.compiler.concordance_declarations import (  # noqa: E402
    CONTROL_SSA_REGION, GRAPH_ID_WITHOUT_CANONICAL_CELL,
    LOOP_RESULT_USE_REBINDING, RECORD_PHI_TEMPORAL_FALLBACK, SSA_VALUE,
    SSAValueFact, SSAValueOrigin,
)
from src.compiler.identity_concordance import (  # noqa: E402
    Mode, Unsourced, begin_identity_book, end_identity_book,
)
from src.compiler.precompile_to_ssa import (  # noqa: E402
    _canonicalize_non_dominating_loop_result_uses,
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


def sources_of(book, page, row):
    ref = book.latest_ref(page, row)
    if ref is None:
        return None
    return tuple(source for source, _stage in book.edges_into(ref))


def value_ids(sources):
    return {
        s.row[1] for s in sources or () if s.page.name == "ssa_value"
    }


def record_phi_function():
    initial = SSAValue(10, dtype="float64")
    result = SSAValue(20, dtype="float64")
    use = Instr("Call", [result], None, attributes={"callee": "consume"})
    phi = Instr("Phi", [result], result, attributes={
        "record_field_phi": True, "initial_value_id": 10,
        "incoming_blocks": ("body",),
    })
    return initial, result, use, Function(
        "step", [initial],
        {
            "entry": BasicBlock(
                "entry", [Instr("Br", [], None, attributes={"target": "body"})],
                successors=["body"]),
            "body": BasicBlock(
                "body", [use, Instr("Br", [], None, attributes={
                    "target": "function_exit"})],
                successors=["function_exit"]),
            "function_exit": BasicBlock(
                "function_exit", [phi, Instr("Ret", [result], None)]),
        },
    )


def loop_result_function():
    seed = SSAValue(30, dtype="float64")
    port = SSAValue(31, dtype="float64")
    use = Instr("Call", [port], None, attributes={"callee": "consume"})
    phi = Instr("Phi", [seed], port, attributes={"incoming_blocks": ("loop_body",)})
    return seed, port, use, Function(
        "loopy", [seed],
        {
            "entry": BasicBlock(
                "entry", [Instr("Br", [], None, attributes={"target": "loop_body"})],
                successors=["loop_body"]),
            "loop_body": BasicBlock(
                "loop_body", [use, Instr("Br", [], None, attributes={
                    "target": "loop_exit"})],
                successors=["loop_exit"]),
            "loop_exit": BasicBlock(
                "loop_exit", [phi, Instr("Ret", [port], None)]),
        },
    )


def part_a() -> None:
    book, token = begin_identity_book()
    try:
        initial, result, use, function = record_phi_function()
        post_values(book, function, initial, result)
        receipts = repair_non_dominating_record_phi_uses(function)
        check("record-Phi repair substituted the use and the self edge",
              use.args == [initial] and len(receipts) == 2)
        row = ("step", 20, "body", 0, 0)
        srcs = sources_of(book, RECORD_PHI_TEMPORAL_FALLBACK, row)
        print(f"  record_phi_temporal_fallback row {row} <- {srcs}")
        check("record-Phi row derives from the original's and the "
              "replacement's ssa_value cells", value_ids(srcs) == {10, 20})
        print(f"  fact: {book.page(RECORD_PHI_TEMPORAL_FALLBACK).latest(row)}")

        seed, port, use, function = loop_result_function()
        post_values(book, function, seed, port)
        receipts = _canonicalize_non_dominating_loop_result_uses(function)
        check("loop-result repair substituted the use",
              use.args == [seed] and len(receipts) == 1)
        row = ("loopy", "loop_body", 0, 0)
        srcs = sources_of(book, LOOP_RESULT_USE_REBINDING, row)
        print(f"  loop_result_use_rebinding row {row} <- {srcs}")
        check("loop-result row derives from the original's and the "
              "replacement's ssa_value cells", value_ids(srcs) == {30, 31})
        print(f"  fact: {book.page(LOOP_RESULT_USE_REBINDING).latest(row)}")
    finally:
        end_identity_book(token)


SOURCE = '''
from dataclasses import dataclass


@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0
    other: float = 0.0


def coerce(m: Metrics) -> Metrics:
    return m


def step(m: Metrics, rejected: bool) -> Metrics:
    if rejected:
        m.hard_failure = True
        m.value = 1.0
        return m
    if m.value > 1.0:
        m.value = m.value * 0.25
    m = coerce(m)
    if m.value > 0.5:
        m.other = 1.0
        return m
    return m
'''
NAME = "late_dominance_repair_rows"


def part_b() -> None:
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import (
        _settle_late_dominance_repairs, lower_ast_source_to_ssa,
    )

    contracts = REPO / "extraction_contracts"
    contract = (
        ExtractionContract(contracts / "program_extraction.yaml")
        .with_program_abi({
            "records": {"Metrics": {
                "identity": f"{NAME}.Metrics",
                "fields": {
                    "hard_failure": {"storage": "scalar", "dtype": "bool",
                                     "mutable": True},
                    "value": {"storage": "scalar", "dtype": "float64",
                              "mutable": True},
                    "other": {"storage": "scalar", "dtype": "float64",
                              "mutable": True},
                },
            }},
            "bindings": [
                {"function": "step", "parameter": "m", "record": "Metrics"},
            ],
            "values": [
                {"function": "step", "parameter": "rejected",
                 "storage": "scalar", "dtype": "bool", "rank": 0,
                 "python_type": "builtins.bool"},
            ],
        })
        .with_execution_file(contracts / "vehicle_full_native_execution.yaml")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            module, _o, _e = lower_ast_source_to_ssa(
                SOURCE, "step", name=NAME, python_bindings={},
                extraction_contract=contract, runtime_closure_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"lowering raised {type(exc).__name__}: {str(exc)[:1500]}")
            check("the program lowers through the full-native gate", False)
            return
    check("the program lowers through the full-native gate", True)
    repairs = module.metadata.get("pre_native_gate_dominance_repairs")
    print(f"  pre_native_gate_dominance_repairs = {repairs}")
    check("the repairs ran before the gate", repairs is not None)
    left = _settle_late_dominance_repairs(module)
    print(f"  left for the wrapper: {left}")
    check("nothing is left for the wrapper to repair", left == (0, 0))


def main() -> int:
    part_a()
    part_b()
    print("FAILED: " + "; ".join(failures) if failures else "all green")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
