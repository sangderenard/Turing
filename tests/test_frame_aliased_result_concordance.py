import pytest

from src.compiler.concordance_declarations import (
    SSA_VALUE, SSAValueFact, SSAValueOrigin, CALL_RECORD, INGESTION, INGEST_SOURCE,
)
from src.compiler.identity_concordance import (
    begin_identity_book, end_identity_book, Novel, Derived, Mode, CorrelationTable,
)
from src.compiler.ssa_record_return_state import refresh_frame_aliased_result
from src.transmogrifier.ssa import SSAValue, Function, BasicBlock, Instr, IRModule, SSACallRecord, SSACallTable


def _fixture(book):
    data, length, dummy = SSAValue(20, "int64", (2,)), SSAValue(21, "int64", (1,)), SSAValue(30, "int64")
    formals = [SSAValue(9, "int64"), SSAValue(7, "int64", (2,)), SSAValue(8, "int64", (1,))]
    callee = Function("callee", formals, {"entry": BasicBlock("entry", [Instr("Ret", [formals[1]], None)])})
    call = Instr("Call", [dummy, data, length], data, attributes={
        "callee": "callee", "source_linked": True, "plan_callsite_id": 90,
        "callee_input_ids": (9, 7, 8), "result_aliases_frame": True,
        "semantic_result_id": 20, "ssa_output_formal_id": 7, "ssa_output_argument": 1,
    })
    caller = Function("caller", [dummy, data, length], {"entry": BasicBlock("entry", [call, Instr("Ret", [data], None)])})
    root = None
    for f in (caller, callee):
        for value in f.args:
            cell = book.post(SSA_VALUE, (f.name, value.id),
                             SSAValueFact(value.dtype, value.shape, SSAValueOrigin.ADOPTED_GRAPH_ID),
                             stage=INGESTION, mode=Mode.CONCORD,
                             provenance=Novel(INGEST_SOURCE, ()) if root is None else Derived((root,)))
            root = cell if root is None else root
    table = SSACallTable(owner="calls", book=book)
    record = SSACallRecord("caller", 90, None, "callee", "callee",
                           result_bindings=((7, 20),),
                           frame_bindings=((9, "caller_value", 30), (7, "caller_storage", 20), (8, "caller_storage", 21)),
                           resolution="native_call")
    book.post(CALL_RECORD, (table.owner, "caller"), (record,),
              stage=INGESTION, mode=Mode.CONCORD, provenance=Derived((root,)))
    record = table["caller"][0]
    refresh_frame_aliased_result(call, callee, book=book, record=record)
    module = IRModule({"caller": caller, "callee": callee}, call_table=table, metadata={"identity_book": book})
    return module, call, callee


@pytest.mark.parametrize("field,value", [("ssa_output_argument", 2), ("ssa_output_formal_id", 8)])
def test_audit_rejects_stale_result_position_even_when_argument_types_match(field, value):
    book, token = begin_identity_book()
    try:
        module, call, _callee = _fixture(book)
        assert not CorrelationTable._frame_aliased_result_findings(module)
        call.attributes[field] = value
        findings = CorrelationTable._frame_aliased_result_findings(module)
        assert len(findings) == 1 and findings[0].kind == "frame-aliased-result-disagreement"
    finally:
        end_identity_book(token)


def test_operand_pruning_reprojects_the_same_exact_record_result():
    from src.compiler.fortran_c_shell import _drop_formals_and_call_operands
    book, token = begin_identity_book()
    try:
        module, call, callee = _fixture(book)
        _drop_formals_and_call_operands(module.functions, callee, (0,))
        assert call.attributes["ssa_output_argument"] == 0
        assert call.attributes["ssa_output_formal_id"] == 7
        assert call.args[0].id == 20
        assert not CorrelationTable._frame_aliased_result_findings(module)
    finally:
        end_identity_book(token)


def test_result_proof_from_a_different_caller_is_not_identity():
    from dataclasses import replace
    book, token = begin_identity_book()
    try:
        module, call, _callee = _fixture(book)
        source = module.call_table["caller"][0]
        proof = book.post(CALL_RECORD, (module.call_table.owner, "other"),
                          (replace(source, caller="other"),), stage=INGESTION,
                          mode=Mode.CONCORD, provenance=Derived((source.cell,)))
        call.attributes["ssa_output_call_record"] = proof
        assert CorrelationTable._frame_aliased_result_findings(module)
    finally:
        end_identity_book(token)
