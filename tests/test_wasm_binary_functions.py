"""``build_module`` assembles several functions and value-returning imports.

The repository-SSA Wasm core needs what the LLVM and C module lanes have:
one function per SSA function, calls between them, and host math by import.
These run in node and read the values back exactly.
"""

import base64
import json
import shutil
import subprocess

import pytest

from src.compiler.wasm_binary import (
    CodeBuilder, WasmFunction, WasmImport, build_module,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node runs the module",
)


def _run(binary: bytes, script: str) -> dict:
    node = shutil.which("node")
    program = (
        "const bytes = Buffer.from(%r, 'base64');\n"
        "const imports = {math: {pow: Math.pow}};\n"
        "WebAssembly.instantiate(bytes, imports).then(({instance}) => {\n"
        "  const e = instance.exports;\n%s\n});\n"
    ) % (base64.b64encode(binary).decode(), script)
    completed = subprocess.run(
        [node, "--input-type=module", "--eval", program],
        capture_output=True, text=True, check=True,
    )
    return json.loads(completed.stdout)


def test_a_second_function_is_called_and_returns_two_values():
    # twice_and_square(x: f64) -> (f64, f64): returns 2x and x*x.
    helper = CodeBuilder("f64", parameter_count=1)
    helper.local_get(0).value_const(2.0).op("mul")
    helper.local_get(0).local_get(0).op("mul")
    # entry(p: i32): stores both results of helper(3.0) at p and p+8.
    entry = CodeBuilder("f64", parameter_count=1)
    a = entry.declare_local("f64")
    b = entry.declare_local("f64")
    entry.value_const(3.0).call(1)  # imports: none; primary 0; helper 1
    entry.local_set(b).local_set(a)
    entry.local_get(0).local_get(a).store()
    entry.local_get(0).i32_const(8).raw(0x6A).local_get(b).store()
    binary = build_module(
        function_name="entry", parameter_types=["i32"], body=entry,
        functions=[WasmFunction(
            "twice_and_square", ("f64",), helper, result_types=("f64", "f64"),
            export=True,
        )],
    )
    result = _run(binary, (
        "e.entry(16);\n"
        "const view = new Float64Array(e.memory.buffer, 16, 2);\n"
        "const direct = e.twice_and_square(5.0);\n"
        "console.log(JSON.stringify({a: view[0], b: view[1], direct}));"
    ))
    assert result == {"a": 6.0, "b": 9.0, "direct": [10.0, 25.0]}


def test_a_value_returning_import_is_callable():
    # entry(p: i32): memory[p] = pow(memory[p], memory[p+8]).
    entry = CodeBuilder("f64", parameter_count=1)
    entry.local_get(0)
    entry.local_get(0).load()
    entry.local_get(0).load(offset=8)
    entry.call(0)  # the import occupies index 0
    entry.store()
    binary = build_module(
        function_name="entry", parameter_types=["i32"], body=entry,
        imports=[WasmImport(
            module="math", field="pow", kind="func",
            parameter_types=("f64", "f64"), result_types=("f64",),
        )],
    )
    result = _run(binary, (
        "const view = new Float64Array(e.memory.buffer, 16, 2);\n"
        "view[0] = 2.0; view[1] = 10.0;\n"
        "e.entry(16);\n"
        "console.log(JSON.stringify({value: view[0]}));"
    ))
    assert result == {"value": 1024.0}
