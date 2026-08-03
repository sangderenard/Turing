import json
import shutil
import subprocess

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)
from src.compiler.dom_shader_demo import SOURCE, build_demo
from src.compiler.fused_program_wasm_backend import emit_wasm_module
from src.compiler.site_bundle import discover_source_contract


def _feeds(**overrides):
    contract = discover_source_contract(SOURCE)
    values = {
        name: (
            np.asarray(specification["values"], dtype=np.float64)
            if isinstance(specification, dict)
            else np.full(4, specification, dtype=np.float64)
        )
        for name, specification in contract.feeds.items()
    }
    values.update({
        name: np.asarray(value, dtype=np.float64)
        for name, value in overrides.items()
    })
    return values


def _compiled_module():
    contract = discover_source_contract(SOURCE)
    aot = compile_ast_aot(
        SOURCE,
        contract.entrypoint,
        _feeds(),
        backend="c",
        remove_loops=True,
        precompile_only=True,
    )
    program = project_public_numerical_program(aot)
    return emit_wasm_module(program, name=contract.entrypoint, dtype="float64")


def _run_node(module, tmp_path, feeds):
    wasm = tmp_path / "elastic_dom_page.wasm"
    wasm.write_bytes(module.binary)
    descriptor = module.api.to_mapping()
    entry = descriptor["entry_points"][0]
    parameters = [
        item for item in entry["parameters"]
        if item["role"] in {"input", "output"}
    ]
    count = len(next(iter(feeds.values())))
    script = tmp_path / "run.mjs"
    script.write_text(
        '''
import {readFileSync} from "node:fs";
const [wasmPath, payload] = process.argv.slice(2);
const descriptor = JSON.parse(payload);
const {instance} = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const count = descriptor.count;
const parameters = descriptor.parameters;
const offsets = parameters.map((_, index) => descriptor.reserved + index * count * 8);
parameters.forEach((parameter, index) => {
  const values = descriptor.feeds[parameter.name];
  if (values) new Float64Array(instance.exports.memory.buffer, offsets[index], count).set(values);
});
instance.exports.run(count, ...offsets);
const outputs = {};
parameters.forEach((parameter, index) => {
  if (parameter.role === "output") outputs[parameter.name] =
    Array.from(new Float64Array(instance.exports.memory.buffer, offsets[index], count));
});
console.log(JSON.stringify(outputs));
''',
        encoding="utf-8",
    )
    payload = {
        "count": count,
        "parameters": parameters,
        "reserved": descriptor["metadata"]["reserved_bytes"],
        "feeds": {name: value.tolist() for name, value in feeds.items()},
    }
    completed = subprocess.run(
        ["node", str(script), str(wasm), json.dumps(payload)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_single_authored_function_lowers_to_complete_wasm():
    module = _compiled_module()

    assert module.complete, module.shortfall_report()
    assert module.binary[:4] == b"\x00asm"
    parameters = module.api.to_mapping()["entry_points"][0]["parameters"]
    assert [item["name"] for item in parameters if item["role"] == "output"] == [
        "next_position_x", "next_position_y",
        "next_velocity_x", "next_velocity_y",
        "next_button_latch", "next_score", "activity",
        "rejected_steps", "advanced_time", "remaining_time",
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_wasm_click_is_an_impulse_not_a_position_retarget(tmp_path):
    module = _compiled_module()
    feeds = _feeds(
        pointer_x=np.full(4, 300.0),
        pointer_y=np.full(4, 150.0),
        pointer_buttons=np.ones(4),
    )
    outputs = _run_node(module, tmp_path, feeds)

    assert outputs["next_position_x"][0] == pytest.approx(90.0)
    assert outputs["next_position_x"][2] == pytest.approx(430.0)
    assert 260.0 < outputs["next_position_x"][1] < 300.0
    assert outputs["next_velocity_x"][1] > 50.0
    assert outputs["next_score"][1] == pytest.approx(1.0)
    assert outputs["next_score"][0] == pytest.approx(0.0)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_wasm_managed_time_rejects_rolls_back_and_lands_exactly(tmp_path):
    module = _compiled_module()
    feeds = _feeds(
        position_x=np.asarray([0.0, 260.0, 430.0, 600.0]),
        anchor_x=np.asarray([90.0, 260.0, 430.0, 600.0]),
        window_dt=np.full(4, 0.2),
    )
    outputs = _run_node(module, tmp_path, feeds)

    assert outputs["rejected_steps"][0] >= 1.0
    assert outputs["advanced_time"][0] == pytest.approx(0.2, abs=1.0e-12)
    assert outputs["remaining_time"][0] == pytest.approx(0.0, abs=1.0e-12)
    # A rejected provisional full-window step would overshoot the anchor.
    # The committed result is assembled only from admitted subdivisions.
    assert 0.0 < outputs["next_position_x"][0] < 90.0


def test_demo_embeds_shader_document_and_wasm_from_the_python_contract(tmp_path):
    bundle = build_demo(tmp_path)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))

    assert manifest["page"]["mode"] == "shader-execution"
    shader = manifest["page"]["shader"]
    assert shader["role"] == "shader-surface"
    assert shader["configuration"]["dom_surface"] is True
    assert shader["configuration"]["dom_io"]["inputs"]["window_dt"] == "window_dt"
    version = bundle.page_path.parent
    assert (version / shader["url"]).exists()
    assert (version / shader["configuration"]["document_url"]).exists()
    shader_source = (version / shader["url"]).read_text(encoding="utf-8")
    assert "bool rayBox" in shader_source
    assert "pointerGlow" not in shader_source
    html = bundle.page_path.read_text(encoding="utf-8")
    assert "window.TuringShaderLiaison" in html
    assert "authoredInputs[domIO.inputs.window_dt]" in html
    assert "elastic_dom_page" in html
