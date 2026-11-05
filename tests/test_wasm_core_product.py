"""The Wasm core's host product: contract offsets agree with the node
harness, the page-shell binding runs the module from bytes, and the one-file
page carries module, contract and binding."""

import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.common.tensors import AbstractTensor
from src.compiler.native_law_kernels import batch_contract
from src.compiler.ssa_wasm_backend import (
    emit_ssa_module_to_wasm_core, prepare_wasm_core_execution,
)
from src.compiler.wasm_core_product import (
    core_contract, emit_core_single_file_page, write_core_web_product,
)

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "examples/mobile_reference_app/web/turing-wasm-shell.js"
BATCH = 4
LAW = '''
def drift(x, v, dt):
    x_next = x + dt * v
    dt_limit = x * 0.0 + 0.01
    max_vel = (v * v).sqrt()
    return x_next, dt_limit, max_vel
'''
pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node runs the Wasm core",
)


@pytest.fixture(scope="module")
def core():
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        LAW, "drift", python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name="drift_batched", runtime_closure_only=True,
        extraction_contract=batch_contract("drift", ("x", "v", "dt"), BATCH),
        progress_bars=False,
    )
    entry = exports[0]
    artifact = emit_ssa_module_to_wasm_core(module, entry, entry_name=entry)
    assert artifact.complete, artifact.shortfalls
    return artifact, module, entry


FEEDS = {
    "x": np.array([0.0, 1.0, -2.0, 10.0]),
    "v": np.array([1.0, 2.0, 3.0, -4.0]),
    "dt": np.full(BATCH, 0.05),
}


def test_contract_offsets_are_the_node_harness_layout(core):
    artifact, module, entry = core
    contract = core_contract(artifact, module, entry)
    assert contract["inputs"] == ["x", "v", "dt"]
    assert contract["outputs"] == ["x_next", "dt_limit", "max_vel"]
    by_name = {p["name"]: p for p in contract["parameters"]}
    feeds = {by_name[name]["value_id"]: value for name, value in FEEDS.items()}
    execution = prepare_wasm_core_execution(artifact, feeds)
    try:
        for parameter in contract["parameters"]:
            offset, length, dtype = execution.buffers._layout[parameter["value_id"]]
            assert (offset, dtype) == (parameter["offset"], parameter["dtype"]), parameter
            assert length == parameter["count"] * 8
        execution.run()
        expected = FEEDS["x"] + 0.05 * FEEDS["v"]
        np.testing.assert_array_equal(execution.buffers[by_name["x_next"]["value_id"]], expected)
    finally:
        execution.close()


def test_the_page_shell_binding_runs_the_module_from_bytes(core, tmp_path):
    artifact, module, entry = core
    contract = core_contract(artifact, module, entry)
    product = write_core_web_product(tmp_path, artifact, contract, identity="drift")
    assert (product / "program.wasm").read_bytes() == artifact.binary
    script = (
        "import {TuringWasmProgram} from %r;\n"
        "import {readFileSync} from 'node:fs';\n"
        "const contract = JSON.parse(readFileSync(%r, 'utf8'));\n"
        "const bytes = readFileSync(%r);\n"
        "const program = await TuringWasmProgram.fromBytes(bytes, contract);\n"
        "const result = program.invoke(%s);\n"
        "console.log(JSON.stringify(result));\n"
    ) % (
        (product / "turing-wasm-shell.js").resolve().as_uri(),
        str(product / "program.abi.json"), str(product / "program.wasm"),
        json.dumps({name: value.tolist() for name, value in FEEDS.items()}),
    )
    completed = subprocess.run(
        [shutil.which("node"), "--input-type=module", "--eval", script],
        capture_output=True, text=True, check=True,
    )
    result = json.loads(completed.stdout)
    np.testing.assert_array_equal(result["x_next"], FEEDS["x"] + 0.05 * FEEDS["v"])
    np.testing.assert_array_equal(result["max_vel"], np.abs(FEEDS["v"]))
    np.testing.assert_array_equal(result["dt_limit"], np.full(BATCH, 0.01))


def test_the_single_file_page_carries_module_contract_and_binding(core):
    artifact, module, entry = core
    contract = core_contract(artifact, module, entry)
    page = emit_core_single_file_page(artifact, contract, title="Drift")
    assert base64.b64encode(artifact.binary).decode("ascii") in page
    assert json.dumps(contract) in page
    assert "class TuringWasmProgram" in page and "export " not in page.split("<script type=\"module\">")[1]
    assert 'data-feed="x"' in page


def test_controls_are_published_per_parameter_and_bound_generically(core, tmp_path):
    """A parameter declared as a control carries its semantic action in the
    contract, and the shell binding fills it from a sampler by that action;
    a control the sampler cannot answer is refused by name."""
    artifact, module, entry = core
    contract = core_contract(artifact, module, entry, controls={"v": "move-forward"})
    by_name = {p["name"]: p for p in contract["parameters"]}
    assert by_name["v"]["control"] == "move-forward" and "control" not in by_name["x"]
    assert contract["controls"] == {"v": "move-forward"}
    with pytest.raises(ValueError, match="name no input parameter"):
        core_contract(artifact, module, entry, controls={"nothing": "jump"})
    product = write_core_web_product(tmp_path, artifact, contract, identity="drift-controls")
    script = (
        "import {TuringWasmProgram, TuringControlSampler} from %r;\n"
        "import {readFileSync} from 'node:fs';\n"
        "const contract = JSON.parse(readFileSync(%r, 'utf8'));\n"
        "const program = await TuringWasmProgram.fromBytes(readFileSync(%r), contract);\n"
        "const sampler = new TuringControlSampler([{action: 'move-forward', inputs: ['keyboard:KeyW']}]);\n"
        "sampler.keys.add('KeyW');\n"
        "const feeds = program.controlFeeds((action, p) => sampler.sample(action, p), %s);\n"
        "const result = program.invokeWithControls((action, p) => sampler.sample(action, p), %s);\n"
        "let refused = null;\n"
        "try { program.controlFeeds(() => undefined, %s); } catch (error) { refused = String(error.message); }\n"
        "console.log(JSON.stringify({v: feeds.v, x_next: result.x_next, refused}));\n"
    ) % (
        (product / "turing-wasm-shell.js").resolve().as_uri(),
        str(product / "program.abi.json"), str(product / "program.wasm"),
        json.dumps({"x": FEEDS["x"].tolist(), "dt": FEEDS["dt"].tolist()}),
        json.dumps({"x": FEEDS["x"].tolist(), "dt": FEEDS["dt"].tolist()}),
        json.dumps({"x": FEEDS["x"].tolist(), "dt": FEEDS["dt"].tolist()}),
    )
    completed = subprocess.run(
        [shutil.which("node"), "--input-type=module", "--eval", script],
        capture_output=True, text=True, check=True,
    )
    result = json.loads(completed.stdout)
    assert result["v"] == 1                                   # KeyW held: move-forward = 1
    np.testing.assert_array_equal(result["x_next"], FEEDS["x"] + 0.05 * 1.0)   # v broadcast to the span
    assert "move-forward" in result["refused"] and "not supplied" in result["refused"]
