"""Current call occurrences follow only their exact recorded control transition."""
import pytest
from src.compiler.identity_concordance import begin_identity_book, end_identity_book, Novel, Derived, Mode, concord_compiler_frame_formals
from src.compiler.concordance_declarations import (
    SSA_VALUE, SSAValueFact, SSAValueOrigin, INGESTION, INGEST_SOURCE,
    LOOP_RESULT_USE_REBINDING, CONTROL_SSA_FINISH, FORMAL_ACTUAL_OCCURRENCE,
)
from src.transmogrifier.ssa import SSAValue, Function, BasicBlock, Instr, IRModule


def _occurrence(book):
    scope = "caller@control:0"
    cells = {value: book.post(
        SSA_VALUE, (scope, value), SSAValueFact("int64", (), SSAValueOrigin.ADOPTED_GRAPH_ID,
                                               physical_dtype="int64", storage="span", rank=1),
        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
    ) for value in (17, 18)}
    old, carried = SSAValue(17, "int64"), SSAValue(18, "int64")
    call = Instr("Call", [old], None, attributes={"callee": "callee"})
    caller = Function("caller", [old, carried], {"entry": BasicBlock("entry", [call, Instr("Ret", [], None)])},
                      metadata={"tensor_shape_concordance_scope": scope, "parameter_names": (("old", 17), ("carried", 18))})
    callee = Function("callee", [SSAValue(17, "int64")], {"entry": BasicBlock("entry", [Instr("Ret", [], None)])},
                      metadata={"parameter_names": (("value", 17),)})
    module = IRModule({"caller": caller, "callee": callee}, metadata={"identity_book": book})
    row = ("callee", 17, "caller", "entry", 0, 0, scope)
    concord_compiler_frame_formals(module)
    previous = book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row)
    call.args = [carried]
    return module, cells, row, previous


def test_current_occurrence_revision_retains_exact_control_causes():
    book, token = begin_identity_book()
    try:
        module, cells, row, previous = _occurrence(book)
        transition = book.post(
            LOOP_RESULT_USE_REBINDING, (row[-1], "entry", 0, 0),
            (17, 18, ("declared_loop_scope", 323, "header", "latch")),
            stage=CONTROL_SSA_FINISH, provenance=Derived(tuple(cells.values())), mode=Mode.REVISE,
        )
        concord_compiler_frame_formals(module)
        current = book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row)
        assert current.column > previous.column
        assert book.page(FORMAL_ACTUAL_OCCURRENCE).latest(row) == 18
        assert {previous, transition, cells[18]}.issubset({ref for ref, _ in book.edges_into(current)})
        concord_compiler_frame_formals(module)
        assert book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row) == current
        from src.compiler.identity_concordance import CorrelationTable
        assert CorrelationTable._formal_actual_occurrence_findings(module) == []
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("poison", ["absent", "scope", "block", "instruction", "position", "old", "new"])
def test_occurrence_change_without_matching_control_receipt_is_refused(poison):
    book, token = begin_identity_book()
    try:
        module, cells, row, previous = _occurrence(book)
        key = [row[-1], "entry", 0, 0]
        old, new = 17, 18
        if poison == "scope": key[0] = "another@control:0"
        elif poison == "block": key[1] = "another"
        elif poison == "instruction": key[2] = 1
        elif poison == "position": key[3] = 1
        elif poison == "old": old = 999
        elif poison == "new": new = 999
        if poison != "absent":
            book.post(LOOP_RESULT_USE_REBINDING, tuple(key),
                      (old, new, ("declared_loop_scope", 323, "header", "latch")),
                      stage=CONTROL_SSA_FINISH, provenance=Derived(tuple(cells.values())), mode=Mode.REVISE)
        with pytest.raises(ValueError, match="formal_actual_occurrence_concordance disagreement"):
            concord_compiler_frame_formals(module)
        assert book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row) == previous
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("poison", ["block", "new", "previous", "actual"])
def test_audit_detects_occurrence_revision_that_bypasses_exact_transition(poison):
    from src.compiler.identity_concordance import CorrelationTable
    book, token = begin_identity_book()
    try:
        module, cells, row, previous = _occurrence(book)
        transition = book.post(
            LOOP_RESULT_USE_REBINDING, (row[-1], "wrong" if poison == "block" else "entry", 0, 0),
            (17, 999 if poison == "new" else 18, ("declared_loop_scope", 323, "header", "latch")),
            stage=CONTROL_SSA_FINISH, provenance=Derived(tuple(cells.values())), mode=Mode.REVISE,
        )
        causes = [transition, cells[17] if poison == "actual" else cells[18]]
        if poison != "previous": causes.append(previous)
        book.post(FORMAL_ACTUAL_OCCURRENCE, row, 18, stage=CONTROL_SSA_FINISH,
                  provenance=Derived(tuple(causes)), mode=Mode.REVISE)
        findings = CorrelationTable._formal_actual_occurrence_findings(module)
        assert len(findings) == 1 and findings[0].kind == "formal-actual-transition-disagreement"
    finally:
        end_identity_book(token)
