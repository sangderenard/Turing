from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from src.compiler.engine_pcm_wasm import (
    BLOCK_SIZE,
    INPUT_NAMES,
    compile_engine_pcm_kernel,
    engine_pcm_kernel_bank_model,
)
from src.compiler.abstract_ui_div_map import DIV_MAP_JAVASCRIPT


def test_engine_pcm_bank_has_vehicle_power_unit_profiles():
    model = engine_pcm_kernel_bank_model()
    identities = {kernel["identity"] for kernel in model["kernels"]}
    assert {"flat-four", "inline-four", "flat-six", "monster-v8",
            "electric-drive", "servo-drive"} <= identities
    assert model["preset_profiles"]["monster-632-twin-turbo"] == "monster-v8"
    assert model["preset_profiles"]["servo-direct-drive-400"] == "servo-drive"
    assert model["runtime"].startswith("AudioWorklet-thread")
    assert all(kernel["binary_bytes"] < 4096 for kernel in model["kernels"])


def test_engine_pcm_bank_is_connected_to_audio_worklet_and_vehicle_telemetry():
    assert "class TuringEnginePCMProcessor extends AudioWorkletProcessor" in DIV_MAP_JAVASCRIPT
    assert "new AudioWorkletNode(context,\"turing-engine-pcm\"" in DIV_MAP_JAVASCRIPT
    assert "updateEngineSoundTelemetry(throttle)" in DIV_MAP_JAVASCRIPT
    assert "rpm,load:Math.min" in DIV_MAP_JAVASCRIPT
    assert "power:Math.min" in DIV_MAP_JAVASCRIPT
    assert "transient:Math.min" in DIV_MAP_JAVASCRIPT
    assert "damage:failed/memberCount" in DIV_MAP_JAVASCRIPT
    assert "armEngineSoundOnFirstGesture();" in DIV_MAP_JAVASCRIPT


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_inline_four_wasm_renders_a_finite_nonconstant_pcm_quantum(tmp_path):
    artifact = compile_engine_pcm_kernel("inline-four")
    wasm_path = tmp_path / "engine.wasm"
    wasm_path.write_bytes(artifact.binary)
    script = tmp_path / "render.mjs"
    script.write_text(
        """
import {readFileSync} from "node:fs";
const {instance}=await WebAssembly.instantiate(readFileSync(process.argv[2]),{});
const count=Number(process.argv[3]),inputCount=Number(process.argv[4]);
const base=64,offsets=Array.from({length:inputCount},(_,index)=>base+index*count*4),outputOffset=base+inputCount*count*4;
const memory=instance.exports.memory,views=offsets.map(offset=>new Float32Array(memory.buffer,offset,count));
for(let index=0;index<count;index++)views[0][index]=index;
views[1].fill(48000);views[2].fill(.17);views[3].fill(3600);views[4].fill(.82);
views[5].fill(.61);views[6].fill(.74);views[7].fill(.12);views[8].fill(.03);views[9].fill(0);
instance.exports.render_engine_pcm(count,...offsets,outputOffset);
console.log(JSON.stringify(Array.from(new Float32Array(memory.buffer,outputOffset,count))));
""".strip(), encoding="utf-8")
    completed = subprocess.run(
        ["node", str(script), str(wasm_path), str(BLOCK_SIZE), str(len(INPUT_NAMES))],
        check=True, capture_output=True, text=True,
    )
    samples = json.loads(completed.stdout)
    assert len(samples) == BLOCK_SIZE
    assert all(abs(sample) < 1 for sample in samples)
    assert max(samples) - min(samples) > .01
