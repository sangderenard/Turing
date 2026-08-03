from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from src.common.dt_system.fluid_mechanics.columnar_multifluid_web_demo import (
    SOURCE,
    build_demo,
    build_pages,
)
from src.common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)
from src.compiler.fused_program_wasm_backend import emit_wasm_module
from src.compiler.site_bundle import discover_source_contract


def _feeds(count=8):
    return {
        "column_x": np.linspace(0.5, 9.5, count),
        "column_y": np.linspace(0.5, 6.5, count),
        "rest_surface": np.full(count, 1.5),
        "displacement": np.zeros(count),
        "displacement_velocity": np.zeros(count),
        "managed_time": np.zeros(count),
        "dt": np.full(count, 0.025),
        "ink_red": np.zeros(count),
        "ink_yellow": np.zeros(count),
        "ink_green": np.zeros(count),
        "ink_cyan": np.zeros(count),
        "ink_blue": np.zeros(count),
        "ink_magenta": np.zeros(count),
    }


def test_python_page_contract_declares_compiled_state_feedback():
    contract = discover_source_contract(SOURCE)

    assert contract.entrypoint == "columnar_multifluid_rgb_step"
    assert contract.state_feedback == {
        "displacement": "next_displacement",
        "displacement_velocity": "next_velocity",
        "managed_time": "next_time",
        "ink_red": "next_ink_red",
        "ink_yellow": "next_ink_yellow",
        "ink_green": "next_ink_green",
        "ink_cyan": "next_ink_cyan",
        "ink_blue": "next_ink_blue",
        "ink_magenta": "next_ink_magenta",
    }
    assert contract.autostart
    assert contract.render_fps == 30.0


def test_python_rgb_tick_recompiles_to_six_output_wasm():
    aot = compile_ast_aot(
        SOURCE,
        "columnar_multifluid_rgb_step",
        _feeds(),
        backend="c",
        remove_loops=True,
        precompile_only=True,
    )
    program = project_public_numerical_program(aot)
    module = emit_wasm_module(
        program, name="columnar_multifluid_rgb_step", dtype="float64"
    )

    assert module.complete, module.shortfall_report()
    outputs = [
        item.name
        for item in module.api.entry_points[0].parameters
        if item.role == "output"
    ]
    assert outputs == [
        "red",
        "green",
        "blue",
        "next_displacement",
        "next_velocity",
        "next_time",
        "next_ink_red",
        "next_ink_yellow",
        "next_ink_green",
        "next_ink_cyan",
        "next_ink_blue",
        "next_ink_magenta",
    ]
    assert module.binary[:4] == b"\x00asm"


def test_bundle_graduates_rgb_preview_to_full_viewport_shader(tmp_path):
    bundle = build_demo(tmp_path)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    html = bundle.page_path.read_text(encoding="utf-8")

    assert manifest["page"]["mode"] == "shader-execution"
    shader = manifest["page"]["shader"]
    assert shader["role"] == "shader-surface"
    assert shader["configuration"] == {
        "output_texture": {"channels": ["red", "green", "blue"]},
    }
    assert any(
        "shader-surface" in artifact["path"]
        for artifact in manifest["artifacts"]
    )
    assert '"state_feedback"' in html
    assert "acceptCompiledState(activeFeeds, result)" in html
    assert "createImageData(w, h)" in html
    assert "ctx.putImageData(image, 0, 0)" in html
    assert "next_displacement" in html
    assert "requestAnimationFrame" in html
    assert '<canvas id="shader-surface" tabindex="0"' in html
    assert "turing_output_texture" in html
    assert "outputFrame()" in html
    assert "uploadOutputTexture()" in html


def test_pages_build_has_stable_root_entrypoint(tmp_path):
    bundle = build_pages(tmp_path)
    deployment = json.loads(
        (tmp_path / "deployment.json").read_text(encoding="utf-8")
    )

    assert (tmp_path / "index.html").exists()
    assert (tmp_path / ".nojekyll").exists()
    assert deployment["entrypoint"] == bundle.page_path.relative_to(
        tmp_path
    ).as_posix()
    assert deployment["version"] == bundle.manifest["version"]["id"]
    assert deployment["entrypoint"] in (
        tmp_path / "index.html"
    ).read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_real_wasm_feedback_advances_managed_time_and_spring_state(tmp_path):
    aot = compile_ast_aot(
        SOURCE,
        "columnar_multifluid_rgb_step",
        _feeds(4),
        backend="c",
        remove_loops=True,
        precompile_only=True,
    )
    module = emit_wasm_module(
        project_public_numerical_program(aot),
        name="columnar_multifluid_rgb_step",
        dtype="float64",
    )
    wasm = tmp_path / "columnar_multifluid_rgb_step.wasm"
    wasm.write_bytes(module.binary)
    descriptor = module.api.to_mapping()
    parameters = descriptor["entry_points"][0]["parameters"]
    input_names = [item["name"] for item in parameters if item["role"] == "input"]
    output_names = [item["name"] for item in parameters if item["role"] == "output"]
    script = tmp_path / "two_ticks.mjs"
    script.write_text(
        '''
import {readFileSync} from "node:fs";
const [wasmPath] = process.argv.slice(2);
const {instance} = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const count = 4;
const inputNames = ''' + json.dumps(input_names) + ''';
const outputNames = ''' + json.dumps(output_names) + ''';
const reserved = ''' + str(descriptor["metadata"]["reserved_bytes"]) + ''';
const names = [...inputNames, ...outputNames];
const offsets = names.map((_, index) => reserved + index * count * 8);
const memory = instance.exports.memory;
const state = {
  column_x: [3.5, 4.5, 5.5, 6.5], column_y: [3.5, 3.5, 3.5, 3.5],
  rest_surface: [1.5, 1.5, 1.5, 1.5], displacement: [0, 0, 0, 0],
  displacement_velocity: [0, 0, 0, 0], managed_time: [0, 0, 0, 0],
  dt: [0.025, 0.025, 0.025, 0.025],
  ink_red: [0, 0, 0, 0], ink_yellow: [0, 0, 0, 0],
  ink_green: [0, 0, 0, 0], ink_cyan: [0, 0, 0, 0],
  ink_blue: [0, 0, 0, 0], ink_magenta: [0, 0, 0, 0],
};
const feedback = {
  displacement: "next_displacement",
  displacement_velocity: "next_velocity",
  managed_time: "next_time",
  ink_red: "next_ink_red", ink_yellow: "next_ink_yellow",
  ink_green: "next_ink_green", ink_cyan: "next_ink_cyan",
  ink_blue: "next_ink_blue", ink_magenta: "next_ink_magenta",
};
const reports = [];
for (let tick = 0; tick < 2; tick += 1) {
  inputNames.forEach((name, index) =>
    new Float64Array(memory.buffer, offsets[index], count).set(state[name]));
  instance.exports.run(count, ...offsets);
  const outputs = Object.fromEntries(outputNames.map((name, index) => [
    name,
    Array.from(new Float64Array(memory.buffer, offsets[inputNames.length + index], count)),
  ]));
  reports.push(outputs);
  for (const [inputName, outputName] of Object.entries(feedback)) {
    state[inputName] = outputs[outputName].slice();
  }
}
console.log(JSON.stringify(reports));
''',
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(wasm)],
        check=True,
        capture_output=True,
        text=True,
    )
    first, second = json.loads(completed.stdout)

    assert first["next_time"] == pytest.approx([0.025] * 4)
    assert second["next_time"] == pytest.approx([0.05] * 4)
    assert min(first["next_displacement"]) < 0.0
    assert second["next_displacement"] != pytest.approx(
        first["next_displacement"]
    )
    for channel in ("red", "green", "blue"):
        assert min(second[channel]) >= 0.0
        assert max(second[channel]) <= 255.0
    assert max(second["next_ink_red"]) > 0.0
