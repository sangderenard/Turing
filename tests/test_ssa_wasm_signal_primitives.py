"""Scalar Wasm primitives used by the existing expanded signal cores."""

import base64
import json
import math
import operator
import shutil
import subprocess

import pytest

from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
)
from src.compiler.ssa_wasm_backend import emit_ssa_function_to_wasm
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue
from tests.compiler_provenance_inputs import publish_authored_ssa


def _emit(arguments, instructions, outputs):
    function = Function("primitive_probe", arguments, {
        "entry": BasicBlock("entry", [*instructions, Instr("Ret", outputs, None)]),
    }, metadata={
        "argument_names": tuple(f"arg{value.id}" for value in arguments),
        "output_names": tuple(f"out{value.id}" for value in outputs),
    })
    module = IRModule({function.name: function})
    book, token = begin_identity_book()
    try:
        publish_authored_ssa(module.functions)
        module.metadata["identity_book"] = book
        assert not CorrelationTable.build(module).findings(module)
        return emit_ssa_function_to_wasm(module, function.name)
    finally:
        end_identity_book(token)


def _execute(artifact, rows):
    assert artifact.complete, artifact.shortfalls
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node executes the emitted Wasm binary")
    script = """
const payload = JSON.parse(process.argv[1]);
const module = new WebAssembly.Module(Buffer.from(payload.binary, 'base64'));
if (WebAssembly.Module.imports(module).length) throw Error('unexpected import');
const instance = new WebAssembly.Instance(module);
const slots = new Float64Array(instance.exports.memory.buffer);
const result = payload.rows.map(row => {
  row.forEach((value, index) => slots[payload.inputs[index] / 8] = Number(value));
  instance.exports.primitive_probe(0);
  return payload.outputs.map(offset => {
    const value = slots[offset / 8];
    return Object.is(value, -0) ? '-0' : String(value);
  });
});
console.log(JSON.stringify(result));
"""
    completed = subprocess.run([
        node, "-e", script, json.dumps({
            "binary": base64.b64encode(artifact.binary).decode(),
            "inputs": artifact.input_offsets,
            "outputs": artifact.output_offsets,
            # Strings preserve NaN, infinities and negative zero over JSON.
            "rows": [[str(value).replace("inf", "Infinity") for value in row]
                     for row in rows],
        }),
    ], capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    return [[float(value) for value in row] for row in json.loads(completed.stdout)]


@pytest.mark.parametrize("operation, compare", [
    ("Lt", operator.lt), ("Le", operator.le), ("Gt", operator.gt),
    ("Ge", operator.ge), ("Eq", operator.eq), ("Ne", operator.ne),
])
def test_scalar_comparisons_use_names_not_module_opcode_numbers(operation, compare):
    left, right = SSAValue(0, "float64"), SSAValue(1, "float64")
    result = SSAValue(2, "bool")
    artifact = _emit([left, right], [Instr(operation, [left, right], result)], [result])
    rows = [
        (-2.0, 1.0), (1.0, -2.0), (1.0, 1.0), (-0.0, 0.0),
        (math.nan, 1.0), (1.0, math.nan), (math.nan, math.nan),
        (math.inf, math.inf), (-math.inf, math.inf),
    ]
    assert f"f64.{operation.lower()} f64.convert_i32_u" in artifact.wat
    assert _execute(artifact, rows) == [[float(compare(*row))] for row in rows]


@pytest.mark.parametrize("operation, opcode", [("Floor", "floor"), ("Round", "nearest")])
def test_scalar_integral_rounding_matches_canonical_ieee_semantics(operation, opcode):
    value, result = SSAValue(0, "float64"), SSAValue(1, "float64")
    artifact = _emit([value], [Instr(operation, [value], result)], [result])
    rows = [(value,) for value in (
        -3.5, -2.5, -1.5, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5,
        1.5, 2.5, 3.5, math.nextafter(0.5, 0.0),
        math.nextafter(0.5, 1.0), 2.0 ** 52, math.inf, -math.inf, math.nan,
    )]
    assert f"f64.{opcode}" in artifact.wat
    for (value,), (actual,) in zip(rows, _execute(artifact, rows)):
        if math.isnan(value):
            assert math.isnan(actual)
            continue
        if math.isinf(value):
            expected = value
        else:
            expected = float(math.floor(value) if operation == "Floor" else round(value))
            if expected == 0.0:
                expected = math.copysign(0.0, value)
        assert actual == expected
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


def test_scalar_boolean_primitives_keep_f64_cast_and_select_convention():
    left, right = SSAValue(0, "float64"), SSAValue(1, "float64")
    left_bool, right_bool, both, either, inverse = (
        SSAValue(value_id, "bool") for value_id in range(2, 7)
    )
    selected = SSAValue(7, "float64")
    outputs = [both, either, inverse, selected]
    artifact = _emit([left, right], [
        Instr("Cast", [left], left_bool, attributes={"target_dtype": "bool"}),
        Instr("Cast", [right], right_bool, attributes={"target_dtype": "bool"}),
        Instr("And", [left_bool, right_bool], both),
        Instr("Or", [left_bool, right_bool], either),
        Instr("Not", [left_bool], inverse),
        Instr("Select", [both, left, right], selected),
    ], outputs)
    rows = [
        (0.0, 0.0), (-0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
        (-2.0, 0.0), (0.25, -0.5), (math.nan, 1.0), (math.inf, -math.inf),
    ]
    for (left_value, right_value), actual in zip(rows, _execute(artifact, rows)):
        both_true = bool(left_value) and bool(right_value)
        assert actual[:3] == [
            float(both_true), float(bool(left_value) or bool(right_value)),
            float(not left_value),
        ]
        expected = left_value if both_true else right_value
        if math.isnan(expected):
            assert math.isnan(actual[3])
        else:
            assert actual[3] == expected


@pytest.mark.parametrize("operation", ["And", "Or"])
def test_scalar_boolean_spelling_does_not_miscompile_integer_bitwise(operation):
    left, right, result = (SSAValue(index, "int32") for index in range(3))
    artifact = _emit([left, right], [Instr(operation, [left, right], result)], [result])
    assert not artifact.complete
    assert any(item.operation == operation for item in artifact.shortfalls)
