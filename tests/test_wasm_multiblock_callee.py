"""The direct core executes a booked callee CFG through its Call edge."""

import shutil

import pytest

from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
)
from src.compiler.ssa_wasm_backend import (
    emit_ssa_module_to_wasm_core,
    prepare_wasm_core_execution,
)
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
from tests.compiler_provenance_inputs import publish_authored_ssa


def _counted_module():
    argument = SSAValue(0, "int64")
    formal = SSAValue(10, "int64")
    zero = SSAValue(11, "int64")
    one = SSAValue(12, "int64")
    current = SSAValue(13, "int64")
    advanced = SSAValue(14, "int64")
    condition = SSAValue(15, "bool")
    callee = Function("counted", [formal], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], zero, attributes={"value": 0}),
            Instr("Const", [], one, attributes={"value": 1}),
            Instr("Br", [], None, attributes={"target": "header"}),
        ]),
        "header": BasicBlock("header", [
            Instr("Phi", [zero, advanced], current,
                  attributes={"incoming_blocks": ("entry", "body")}),
            Instr("Lt", [current, formal], condition),
            Instr("CondBr", [condition], None, attributes={
                "true_target": "body", "false_target": "exit",
            }),
        ]),
        "body": BasicBlock("body", [
            Instr("Add", [current, one], advanced),
            Instr("Br", [], None, attributes={"target": "header"}),
        ]),
        "exit": BasicBlock("exit", [Instr("Ret", [current], None)]),
    })
    aggregate = SSAValue(1, "ssa.aggregate")
    index = SSAValue(2, "int32")
    projection = SSAValue(3, "ptr")
    output = SSAValue(4, "int64")
    root = Function("root", [argument], {"entry": BasicBlock("entry", [
        Instr("Call", [argument], aggregate, attributes={
            "callee": "counted", "output_ids": (current.id,),
        }),
        Instr("Const", [], index, attributes={"value": 0}),
        Instr("GetElementPtr", [aggregate, index], projection,
              attributes={"source_output_id": current.id}),
        Instr("Load", [projection], output),
        Instr("Ret", [output], None),
    ])})
    return IRModule({"root": root, "counted": callee}), output.id


@pytest.mark.skipif(shutil.which("node") is None, reason="Node executes Wasm")
def test_four_block_callee_phi_loop_executes_in_direct_core():
    module, output_id = _counted_module()
    book, token = begin_identity_book()
    try:
        publish_authored_ssa(module.functions)
        module.metadata["identity_book"] = book
        assert not CorrelationTable.build(module).findings(module)
        artifact = emit_ssa_module_to_wasm_core(module, "root")
        assert artifact.complete, artifact.shortfalls
        for count in (0, 1, 3, 7):
            execution = prepare_wasm_core_execution(artifact, {0: count})
            try:
                result = execution.run().buffers[output_id]
                assert int(result[0]) == count
            finally:
                execution.close()
    finally:
        end_identity_book(token)


def test_callee_exit_ids_must_match_declared_aggregate_output():
    module, _output_id = _counted_module()
    callee = module.functions["counted"]
    extra = SSAValue(16, "int64")
    callee.blocks["alternate"] = BasicBlock("alternate", [
        Instr("Const", [], extra, attributes={"value": 9}),
        Instr("Ret", [extra], None),
    ])
    book, token = begin_identity_book()
    try:
        publish_authored_ssa(module.functions)
        module.metadata["identity_book"] = book
        artifact = emit_ssa_module_to_wasm_core(module, "root")
        assert not artifact.complete
        assert any("aggregate result disagrees" in item.reason
                   for item in artifact.shortfalls)
    finally:
        end_identity_book(token)


def test_pointer_formal_identity_is_scoped_across_callee_frames():
    scalar = SSAValue(0, "int64")
    buffer = SSAValue(1, "int64", (4,), accounting={
        "physical_dtype": "int64", "program_abi_storage": "span",
        "program_abi_rank": 1,
    })
    formal = SSAValue(0, "ptr")  # same number as root scalar, different frame
    index = SSAValue(2, "int32")
    address = SSAValue(3, "ptr")
    helper = Function("helper", [formal], {"entry": BasicBlock("entry", [
        Instr("Const", [], index, attributes={"value": 0}),
        Instr("GetElementPtr", [formal, index], address),
        Instr("Ret", [], None),
    ])})
    root = Function("root", [scalar, buffer], {"entry": BasicBlock("entry", [
        Instr("Call", [buffer], None, attributes={"callee": "helper"}),
        Instr("Ret", [scalar], None),
    ])})
    module = IRModule({"root": root, "helper": helper})
    book, token = begin_identity_book()
    try:
        publish_authored_ssa(module.functions)
        module.metadata["identity_book"] = book
        artifact = emit_ssa_module_to_wasm_core(module, "root")
        assert artifact.parameters[:2] == (("i64", 0), ("buffer", 1))
    finally:
        end_identity_book(token)


def test_inlined_call_requires_exact_actual_formal_arity():
    module, _output_id = _counted_module()
    call = module.functions["root"].blocks["entry"].instrs[0]
    call.args.clear()
    book, token = begin_identity_book()
    try:
        publish_authored_ssa(module.functions)
        module.metadata["identity_book"] = book
        artifact = emit_ssa_module_to_wasm_core(module, "root")
        assert not artifact.complete
        assert any("actual/formal arity disagrees" in item.reason
                   for item in artifact.shortfalls)
    finally:
        end_identity_book(token)
