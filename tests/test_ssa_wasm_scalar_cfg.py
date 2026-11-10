"""Direct scalar slots use the core dispatcher's branches and Phi edge copies."""

import base64
import json
import shutil
import subprocess

import pytest

from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.compiler.ssa_wasm_backend import emit_ssa_function_to_wasm
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
from tests.compiler_provenance_inputs import publish_authored_ssa


def _emit(function):
    module = IRModule({function.name: function})
    book, token = begin_identity_book()
    try:
        publish_authored_ssa(module.functions)
        module.metadata["identity_book"] = book
        return emit_ssa_function_to_wasm(module, function.name)
    finally:
        end_identity_book(token)


def _run(artifact, rows):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node executes real Wasm")
    script = """
const binary = Buffer.from(process.argv[1], 'base64');
const spec = JSON.parse(process.argv[2]);
const calls = [];
const module = new WebAssembly.Module(binary);
const instance = new WebAssembly.Instance(module, {
  turing: { turing_validation_error(code) { calls.push(code); } }
});
const memory = new Float64Array(instance.exports.memory.buffer);
const io = 128;
const rows = [];
for (const values of spec.rows) {
  spec.inputs.forEach((offset, i) => memory[(io + offset) / 8] = values[i]);
  spec.outputs.forEach(offset => memory[(io + offset) / 8] = -12345);
  calls.length = 0;
  instance.exports[spec.entry](io);
  rows.push({values: spec.outputs.map(offset => memory[(io + offset) / 8]),
             faults: calls.slice()});
}
console.log(JSON.stringify({rows, imports: WebAssembly.Module.imports(module)}));
"""
    completed = subprocess.run(
        [node, "-e", script, base64.b64encode(artifact.binary).decode(), json.dumps({
            "inputs": artifact.input_offsets, "outputs": artifact.output_offsets,
            "entry": artifact.name, "rows": rows,
        })], capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _guarded():
    x, zero = SSAValue(0, "float64"), SSAValue(1, "float64")
    negative = SSAValue(2, "bool")
    return Function("guarded", [x], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], zero, attributes={"value": 0.0}),
            Instr("Lt", [x, zero], negative),
            Instr("CondBr", [negative], None, attributes={
                "true_target": "fault", "false_target": "passing",
            }),
        ]),
        "fault": BasicBlock("fault", [
            Instr("Call", [], None, attributes={
                "callee": "turing_validation_error", "error_code": 93,
            }),
            Instr("Br", [], None, attributes={"target": "passing"}),
        ]),
        "passing": BasicBlock("passing", [Instr("Ret", [x], None)]),
    }, metadata={"argument_names": ("x",), "output_names": ("result",)})


def test_fault_callback_returns_without_publishing_even_if_host_does_not_throw():
    artifact = _emit(_guarded())
    assert artifact.complete, artifact.shortfalls
    assert [(entry.module, entry.field, entry.parameter_types, entry.result_types)
            for entry in artifact.imports] == [
                ("turing", "turing_validation_error", ("i32",), ()),
            ]
    result = _run(artifact, [[3], [-2], [7]])
    assert result["imports"] == [
        {"module": "turing", "name": "turing_validation_error", "kind": "function"},
    ]
    assert result["rows"] == [
        {"values": [3], "faults": []},
        {"values": [-12345], "faults": [93]},
        {"values": [7], "faults": []},
    ]
    assert 'call $turing_validation_error return' in artifact.wat


def test_direct_loop_phi_copy_is_parallel_and_resets_each_invocation():
    values = [SSAValue(i, "float64") for i in range(9)]
    count, zero, one, left, right, index, first, second, advanced = values
    condition = SSAValue(9, "bool")
    function = Function("swap", [count], {
        "entry": BasicBlock("entry", [
            Instr("Const", [], zero, attributes={"value": 0.0}),
            Instr("Const", [], one, attributes={"value": 1.0}),
            Instr("Const", [], left, attributes={"value": 4.0}),
            Instr("Const", [], right, attributes={"value": 9.0}),
            Instr("Br", [], None, attributes={"target": "header"}),
        ]),
        "header": BasicBlock("header", [
            Instr("Phi", [zero, advanced], index,
                  attributes={"incoming_blocks": ("entry", "body")}),
            Instr("Phi", [left, second], first,
                  attributes={"incoming_blocks": ("entry", "body")}),
            Instr("Phi", [right, first], second,
                  attributes={"incoming_blocks": ("entry", "body")}),
            Instr("Lt", [index, count], condition),
            Instr("CondBr", [condition], None, attributes={
                "true_target": "body", "false_target": "exit",
            }),
        ]),
        "body": BasicBlock("body", [
            Instr("Add", [index, one], advanced),
            Instr("Br", [], None, attributes={"target": "header"}),
        ]),
        "exit": BasicBlock("exit", [Instr("Ret", [first, second], None)]),
    })
    artifact = _emit(function)
    assert artifact.complete, artifact.shortfalls
    result = _run(artifact, [[0], [1], [2], [7], [0]])
    assert [row["values"] for row in result["rows"]] == [
        [4, 9], [9, 4], [4, 9], [9, 4], [4, 9],
    ]
    assert result["imports"] == []


def test_each_return_publishes_its_own_values_into_shared_output_slots():
    x, y, condition = SSAValue(0, "float64"), SSAValue(1, "float64"), SSAValue(2, "bool")
    function = Function("choose", [x, y, condition], {
        "entry": BasicBlock("entry", [Instr("CondBr", [condition], None, attributes={
            "true_target": "first", "false_target": "second",
        })]),
        "first": BasicBlock("first", [Instr("Ret", [x], None)]),
        "second": BasicBlock("second", [Instr("Ret", [y], None)]),
    })
    artifact = _emit(function)
    assert artifact.complete, artifact.shortfalls
    result = _run(artifact, [[3, 7, 0], [3, 7, 1]])
    assert [row["values"] for row in result["rows"]] == [[7], [3]]


def test_unknown_cfg_target_is_a_shortfall_not_a_binary():
    function = _guarded()
    function.blocks["entry"].instrs[-1].attributes["true_target"] = "missing"
    artifact = _emit(function)
    assert not artifact.complete
    assert artifact.binary == b""
    assert any(item.operation == "CondBr" and "unknown target" in item.reason
               for item in artifact.shortfalls)


def test_inconsistent_return_arities_are_refused():
    function = _guarded()
    function.blocks["extra"] = BasicBlock("extra", [Instr("Ret", [], None)])
    artifact = _emit(function)
    assert not artifact.complete
    assert any(item.reason == "return arities disagree" for item in artifact.shortfalls)
