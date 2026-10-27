"""Returned spans are views of the exact callsite's existing actual storage."""
import pytest
import numpy as np

from src.compiler.concordance_declarations import (
    SSA_VALUE, SSAValueFact, SSAValueOrigin, CALL_RECORD, INGESTION, INGEST_SOURCE,
)
from src.compiler.identity_concordance import begin_identity_book, end_identity_book, Novel, Derived, Mode
from src.compiler.ssa_aggregate_abi import is_storage_view, legalize_aggregate_output_views
from src.transmogrifier.ssa import SSAValue, Instr, Function, BasicBlock, IRModule, SSACallRecord


def _sequence_fixture(book):
    from dataclasses import replace
    from src.compiler.precompile_to_ssa import _post_built_function_rows
    from src.transmogrifier.ssa import SSASequenceTable, SSASequenceDescriptor
    module, call, use, projected, occurrence = _fixture(book)
    caller, callee = module.functions["caller"], module.functions["callee"]
    actuals = [SSAValue(6, "int64", (1,)), SSAValue(7, "int64"), SSAValue(8, "int32", (1,))]
    formals = [SSAValue(11, "int64", (1,)), SSAValue(12, "int64"), SSAValue(13, "int32", (1,))]
    old = [SSAValue(31, "int64", (1,)), SSAValue(32, "int64"), SSAValue(33, "int32", (1,))]
    for value in old:
        value.accounting = {"compiler_frame_storage": "caller"}
    caller.args.extend((*actuals, *old))
    callee.args.extend(formals)
    call.args.extend(actuals)
    call.attributes["callee_input_ids"] = (10, 11, 12, 13)
    # A real consumer keeps each placeholder member live before the fold.
    index_value = next(instruction.res for instruction in caller.blocks["entry"].instrs
                       if instruction.res is not None and instruction.res.id == 200)
    member_uses, member_instructions = [], []
    for index, value in enumerate(old):
        if index == 1:
            operation = Instr("Cast", [value], SSAValue(300 + index, value.dtype))
            member_uses.append(operation)
            member_instructions.append(operation)
        else:
            address = SSAValue(310 + index, "ptr")
            operation = Instr("GetElementPtr", [value, index_value], address)
            member_uses.append(operation)
            member_instructions.extend((operation, Instr("Load", [address], SSAValue(300 + index, value.dtype))))
    caller.blocks["entry"].instrs[-1:-1] = member_instructions
    _post_built_function_rows(callee)
    _post_built_function_rows(caller)
    row = (("calls", 0), "caller")
    prior = book.latest_ref(CALL_RECORD, row)
    record = book.page(CALL_RECORD).latest(row)[0]
    book.post(CALL_RECORD, row, (replace(record, frame_bindings=(
        (10, "caller_value", 5), (11, "caller_value", 6),
        (12, "caller_value", 7), (13, "caller_value", 8),
    )),), stage=INGESTION, provenance=Derived((prior,)), mode=Mode.REVISE)
    source = SSASequenceTable(owner="callee", book=book)
    source.register(SSASequenceDescriptor(10, (10,), 11, 12, 13, column_dtypes=("int64",)),
                    sources=(book.latest_ref(SSA_VALUE, ("callee", 10)),), stage=INGESTION)
    target = SSASequenceTable(owner="caller", book=book)
    target.register(SSASequenceDescriptor(30, (30,), 31, 32, 33, column_dtypes=("unknown",)),
                    sources=(book.latest_ref(SSA_VALUE, ("caller@control:0", 30)),), stage=INGESTION)
    module.sequence_tables.update(caller=target, callee=source)
    return module, call, use, member_uses


def _fixture(book, *, view_shape=(2,), recorded_actual=5):
    formal, actual = SSAValue(10, "int64", (2,)), SSAValue(5, "int64", (2,))
    aggregate, projected = SSAValue(100, "ssa.aggregate"), SSAValue(30, "int64", (2,))
    occurrence = SSAValue(30, "int64", view_shape)
    index, address, result = SSAValue(200, "int64"), SSAValue(201, "ptr"), SSAValue(40, "int64")
    projection_address = SSAValue(202, "ptr")
    call = Instr("Call", [actual], aggregate, attributes={
        "callee": "callee", "plan_callsite_id": 90, "result_convention": "ssa.aggregate",
        "output_ids": (30,), "callee_output_ids": (10,), "output_positions": (0,),
    })
    use = Instr("GetElementPtr", [occurrence, index], address)
    caller = Function("caller", [actual], {"entry": BasicBlock("entry", [
        call, Instr("Const", [], index, attributes={"value": 0}),
        Instr("GetElementPtr", [aggregate, index], projection_address, attributes={"aggregate_index": 0}),
        Instr("Load", [projection_address], projected, attributes={"aggregate_index": 0}),
        use, Instr("Load", [address], result), Instr("Ret", [result], None),
    ])}, metadata={"tensor_shape_concordance_scope": "caller@control:0"})
    callee = Function("callee", [formal], {"entry": BasicBlock("entry", [Instr("Ret", [formal], None)])})
    root = None
    for value in (actual, aggregate, projected, index, address, projection_address, result):
        cell = book.post(SSA_VALUE, ("caller@control:0", value.id),
                         SSAValueFact(value.dtype, value.shape, SSAValueOrigin.ADOPTED_GRAPH_ID,
                                      physical_dtype=value.dtype, storage="span" if value.shape else "scalar",
                                      rank=len(value.shape)),
                         stage=INGESTION, mode=Mode.CONCORD,
                         provenance=Novel(INGEST_SOURCE, ()) if root is None else Derived((root,)))
        if root is None:
            root = cell
    record = SSACallRecord("caller", 90, None, "callee", "callee",
                           frame_bindings=((10, "caller_value", recorded_actual),),
                           result_bindings=((10, 30),), resolution="native_call")
    book.post(CALL_RECORD, (("calls", 0), "caller"), (record,),
              stage=INGESTION, provenance=Derived((root,)), mode=Mode.CONCORD)
    return IRModule({"caller": caller, "callee": callee}), call, use, projected, occurrence


def test_returned_formal_distinct_occurrences_share_posted_identity():
    book, token = begin_identity_book()
    try:
        module, call, use, projected, occurrence = _fixture(book)
        assert occurrence is not projected
        assert is_storage_view(occurrence, projected, function=module.functions["caller"])
        assert legalize_aggregate_output_views(module)
        assert call.attributes["output_ids"] == ()
        assert use.args[0].id == 5
        assert book.page("dead_structural_retirement").latest(("caller@control:0", 100)) == ("Call", "None")
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("difference", ["shape", "binding", "physical-view"])
def test_returned_formal_requires_same_view_and_exact_call_binding(difference):
    book, token = begin_identity_book()
    try:
        module, call, use, _projected, occurrence = _fixture(
            book, view_shape=(1, 2) if difference == "shape" else (2,),
            recorded_actual=6 if difference == "binding" else 5,
        )
        if difference == "physical-view":
            occurrence.accounting = {"physical_dtype": "float64"}
        assert not legalize_aggregate_output_views(module)
        assert call.attributes["output_ids"] == (30,)
        assert use.args[0] is occurrence
    finally:
        end_identity_book(token)


def test_raw_equal_numbers_without_posted_identity_are_not_alias_proof():
    first, second = SSAValue(30, "int64", (2,)), SSAValue(30, "int64", (2,))
    function = Function("raw", [], {})
    assert not is_storage_view(first, second, function=function)


def test_returned_sequence_transports_all_members_on_one_call_frame():
    book, token = begin_identity_book()
    try:
        module, call, use, member_uses = _sequence_fixture(book)
        assert legalize_aggregate_output_views(module)
        descriptor = module.sequence_tables["caller"].by_id(30)
        assert (descriptor.sequence_id, descriptor.column_value_ids,
                descriptor.length_address_id, descriptor.capacity_value_id,
                descriptor.status_address_id, descriptor.column_dtypes) == (30, (5,), 6, 7, 8, ("int64",))
        assert [instruction.args[0].id for instruction in member_uses] == [6, 7, 8]
        assert use.args[0].id == 5
        assert call.res is None
    finally:
        end_identity_book(token)


def test_returned_sequence_missing_capacity_binding_keeps_entire_view_unresolved():
    from dataclasses import replace
    book, token = begin_identity_book()
    try:
        module, call, use, member_uses = _sequence_fixture(book)
        row = (("calls", 0), "caller")
        prior = book.latest_ref(CALL_RECORD, row)
        record = book.page(CALL_RECORD).latest(row)[0]
        book.post(CALL_RECORD, row, (replace(record, frame_bindings=tuple(
            binding for binding in record.frame_bindings if binding[0] != 12
        )),), stage=INGESTION, provenance=Derived((prior,)), mode=Mode.REVISE)
        assert not legalize_aggregate_output_views(module)
        descriptor = module.sequence_tables["caller"].by_id(30)
        assert (descriptor.column_value_ids, descriptor.capacity_value_id) == ((30,), 32)
        assert call.attributes["output_ids"] == (30,)
        assert use.args[0].id == 30
        assert [instruction.args[0].id for instruction in member_uses] == [31, 32, 33]
    finally:
        end_identity_book(token)


def test_returned_sequence_audit_rejects_capacity_from_another_activation():
    from dataclasses import replace
    from src.compiler.concordance_declarations import SEQUENCE_DESCRIPTOR
    from src.compiler.identity_concordance import CorrelationTable
    book, token = begin_identity_book()
    try:
        module, _call, _use, _members = _sequence_fixture(book)
        module.metadata["identity_book"] = book
        assert legalize_aggregate_output_views(module)
        assert CorrelationTable._sequence_descriptor_findings(module) == []
        table = module.sequence_tables["caller"]
        descriptor = table.by_id(30)
        prior = book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner, 30))
        wrong = book.post(SSA_VALUE, ("caller@control:0", 99),
                          SSAValueFact("int64", (), SSAValueOrigin.ADOPTED_GRAPH_ID,
                                       physical_dtype="int64", storage="scalar", rank=0),
                          stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        table.sequences.assign(30, replace(descriptor, capacity_value_id=99),
                               sources=(prior, wrong), stage=INGESTION)
        # The descriptor/member index agrees internally. The wrong claim is
        # its relationship to the exact call activation that returned it.
        assert CorrelationTable._table_member_findings(module) == []
        findings = CorrelationTable._sequence_descriptor_findings(module)
        assert len(findings) == 1
        assert findings[0].kind == "returned-sequence-frame-disagreement"
        assert "capacity_value_id=99" in findings[0].detail
        assert "capacity_value_id=7" in findings[0].detail
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("lane", ["C", "LLVM", "Fortran"])
@pytest.mark.parametrize("whole_frame", [False, True], ids=["column", "sequence-frame"])
def test_returned_formal_posted_view_executes_native(tmp_path, lane, whole_frame):
    from src.compiler.precompile_to_ssa import _post_built_function_rows
    book, token = begin_identity_book()
    try:
        if whole_frame:
            module, _call, _use, _members = _sequence_fixture(book)
        else:
            module, _call, _use, _projected, _occurrence = _fixture(book)
        # Admit the standalone helper through the incumbent SSA unit producer.
        _post_built_function_rows(module.functions["callee"])
        assert legalize_aggregate_output_views(module)
        if lane == "C":
            from src.compiler.ssa_c_backend import emit_ssa_to_c
            artifact = emit_ssa_to_c(module, "caller")
            assert artifact.complete, artifact.shortfalls
            artifact.compile(tmp_path)
            prepare = artifact.prepare_execution
        elif lane == "LLVM":
            from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution
            artifact = emit_ssa_function_to_llvm(module, "caller")
            assert artifact.complete, artifact.shortfalls
            native = compile_artifact(artifact, directory=tmp_path)
            prepare = lambda feeds: prepare_artifact_execution(native, feeds)
        else:
            from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
            from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
            if fortran_compiler() is None:
                pytest.skip("native Fortran compiler unavailable")
            result = next(x.args[0] for x in module.functions["caller"].blocks["entry"].instrs if x.op == "Ret")
            artifact = emit_module(module, outputs={"caller": [result]}, extra_roots=("caller",), progress=lambda _: None)
            assert artifact.complete, artifact.shortfalls
            native = compile_fortran_module_c_shell(artifact, {}, tmp_path, entrypoint="caller", library=True)
            prepare = FortranCoreNative(native.executable_path, artifact.api.entry_point("caller"), artifact.api).prepare_execution
        feeds = {5: np.array([2**53 + 1, 9], dtype=np.int64)}
        if whole_frame:
            feeds.update({6: np.array([2], dtype=np.int64),
                          7: np.array([2], dtype=np.int64),
                          8: np.array([0], dtype=np.int32)})
        assert prepare(feeds).run().buffers[40].item() == 2**53 + 1
    finally:
        end_identity_book(token)
