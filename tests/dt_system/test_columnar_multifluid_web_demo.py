from __future__ import annotations

import json
import functools
import http.server
from pathlib import Path
import shutil
import socketserver
import subprocess
import threading

import numpy as np
import pytest

from src.common.dt_system.fluid_mechanics.columnar_multifluid_web_demo import (
    FORTRAN_SOURCE,
    SOURCE,
)
from src.common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)
from src.compiler.fused_program_wasm_backend import emit_wasm_module
from src.compiler.site_bundle import build_program_bundle, discover_source_contract
from src.compiler.ssa_fortran_backend import fortran_compiler


def _chrome_executable():
    candidates = (
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    )
    return next((item for item in candidates if item and Path(item).is_file()), None)


def _feeds(count=8):
    return {
        "column_x": np.linspace(0.5, 9.5, count),
        "column_y": np.linspace(0.5, 6.5, count),
        "rest_surface": np.full(count, 1.5),
        "displacement": np.zeros(count),
        "displacement_velocity": np.zeros(count),
        "entity_x": np.full(count, 5.0),
        "entity_y": np.full(count, 3.5),
        "entity_velocity_x": np.full(count, 0.45),
        "entity_velocity_y": np.full(count, 0.12),
        "managed_time": np.zeros(count),
        "dt": np.full(count, 0.025),
        "audio_low": np.zeros(count),
        "audio_mid": np.zeros(count),
        "audio_high": np.zeros(count),
        "audio_level": np.zeros(count),
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
        "entity_x": "next_entity_x",
        "entity_y": "next_entity_y",
        "entity_velocity_x": "next_entity_velocity_x",
        "entity_velocity_y": "next_entity_velocity_y",
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


def test_native_fortran_contract_is_an_inspection_bundle_without_a_shader():
    contract = discover_source_contract(FORTRAN_SOURCE)

    assert contract.slug == "managed-columnar-multifluid-world-fortran"
    assert contract.presentation_entrypoint is None
    assert contract.entrypoint == "columnar_multifluid_rgb_step"


def test_python_rgb_tick_recompiles_with_reductions_to_wasm():
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
        "next_entity_x",
        "next_entity_y",
        "next_entity_velocity_x",
        "next_entity_velocity_y",
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
    bundle = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="columnar_multifluid_web_demo.py",
        include_backends=False,
        include_mathematics=False,
    )
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    html = bundle.page_path.read_text(encoding="utf-8")

    assert manifest["page"]["mode"] == "shader-execution"
    shader = manifest["page"]["shader"]
    assert shader["role"] == "shader-surface"
    assert shader["configuration"] == {
        "output_feed_bindings": {
            "turing_feed_0": "red",
            "turing_feed_1": "green",
            "turing_feed_2": "blue",
        },
    }
    assert any(
        "shader-surface" in artifact["path"]
        for artifact in manifest["artifacts"]
    )
    shader_source = (bundle.directory / shader["url"]).read_text(encoding="utf-8")
    assert "texture(turing_feed_0, turing_uv).r" in shader_source
    assert "turing_output_0 = vec4(v_4, v_6, v_8, 1.0);" in shader_source
    audio = manifest["page"]["audio"]
    assert audio["managed_time_output"] == "next_time"
    assert audio["pan_output"] == "next_entity_x"
    assert (bundle.directory / audio["audio_url"]).read_bytes()[:4] == b"RIFF"
    features = json.loads(
        (bundle.directory / audio["features_url"]).read_text(encoding="utf-8")
    )
    assert set(features["feeds"]) == {
        "audio_low", "audio_mid", "audio_high", "audio_level",
    }
    assert '"state_feedback"' in html
    assert "acceptCompiledState(activeFeeds, result)" in html
    assert "createImageData(w, h)" in html
    assert "ctx.putImageData(image, 0, 0)" in html
    assert "next_displacement" in html
    assert "requestAnimationFrame" in html
    assert '<canvas id="shader-surface" tabindex="0"' in html
    assert "output_feed_bindings" in html
    assert "outputFrame()" in html
    assert "uploadOutputTexture()" in html
    assert "managedDelta / wallDelta" in html
    assert "createStereoPanner" in html


@pytest.mark.skipif(
    fortran_compiler() is None, reason="no Fortran compiler installed"
)
def test_native_fortran_bundle_has_all_tick_outputs_and_no_shader(tmp_path):
    bundle = build_program_bundle(
        FORTRAN_SOURCE,
        tmp_path,
        source_filename="columnar_multifluid_fortran_demo.py",
        include_backends=True,
        include_mathematics=False,
    )
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    proof = json.loads(
        (bundle.directory / "verification/fortran-fidelity.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["page"]["mode"] == "inspection"
    assert manifest["page"]["shader"] is None
    assert proof["passed"] is True
    assert [item["name"] for item in proof["cases"][0]["outputs"]] == [
        "red",
        "green",
        "blue",
        "next_displacement",
        "next_velocity",
        "next_entity_x",
        "next_entity_y",
        "next_entity_velocity_x",
        "next_entity_velocity_y",
        "next_time",
        "next_ink_red",
        "next_ink_yellow",
        "next_ink_green",
        "next_ink_cyan",
        "next_ink_blue",
        "next_ink_magenta",
    ]


@pytest.mark.skipif(
    shutil.which("node") is None or _chrome_executable() is None,
    reason="Node.js and a Chromium browser are required",
)
def test_compiler_generated_webgl_presents_non_black_wasm_output(tmp_path):
    bundle = build_program_bundle(
        SOURCE,
        tmp_path,
        source_filename="columnar_multifluid_web_demo.py",
        include_backends=False,
        include_mathematics=False,
    )
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(tmp_path),
    )
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        relative = bundle.page_path.relative_to(tmp_path).as_posix()
        url = f"http://127.0.0.1:{server.server_address[1]}/{relative}"
        probe = Path(__file__).parents[1] / "browser_webgl_probe.mjs"
        completed = subprocess.run(
            ["node", str(probe), str(_chrome_executable()), url],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        server.shutdown()

    result = json.loads(completed.stdout)
    assert result["error"] is None
    assert result["revision"] >= 1
    assert result["glError"] == 0
    assert result["center"][3] == 255
    assert sum(result["center"][:3]) > 0


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
  entity_x: [5, 5, 5, 5], entity_y: [3.5, 3.5, 3.5, 3.5],
  entity_velocity_x: [0.45, 0.45, 0.45, 0.45],
  entity_velocity_y: [0.12, 0.12, 0.12, 0.12],
  dt: [0.025, 0.025, 0.025, 0.025],
  audio_low: [0, 0, 0, 0], audio_mid: [0, 0, 0, 0],
  audio_high: [0, 0, 0, 0], audio_level: [0, 0, 0, 0],
  ink_red: [0, 0, 0, 0], ink_yellow: [0, 0, 0, 0],
  ink_green: [0, 0, 0, 0], ink_cyan: [0, 0, 0, 0],
  ink_blue: [0, 0, 0, 0], ink_magenta: [0, 0, 0, 0],
};
const feedback = {
  displacement: "next_displacement",
  displacement_velocity: "next_velocity",
  entity_x: "next_entity_x", entity_y: "next_entity_y",
  entity_velocity_x: "next_entity_velocity_x",
  entity_velocity_y: "next_entity_velocity_y",
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
