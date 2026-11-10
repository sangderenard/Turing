from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import numpy as np

from src.compiler.engine_pcm_wasm import (
    BLOCK_SIZE,
    INPUT_NAMES,
    compile_engine_pcm_kernel,
    engine_pcm_kernel_bank_model,
    _pristine_lowpass_kernel,
)
from src.compiler.abstract_ui_div_map import DIV_MAP_JAVASCRIPT


def test_engine_pcm_bank_has_vehicle_power_unit_profiles():
    model = engine_pcm_kernel_bank_model()
    identities = {kernel["identity"] for kernel in model["kernels"]}
    assert {"flat-four", "inline-four", "inline-six", "flat-six", "aircraft-v12",
            "industrial-diesel-i6", "monster-v8", "electric-drive", "servo-drive"} <= identities
    assert model["preset_profiles"]["monster-632-twin-turbo"] == "monster-v8"
    assert model["preset_profiles"]["servo-direct-drive-400"] == "servo-drive"
    assert model["preset_profiles"]["amc-258-jeep-i6"] == "inline-six"
    assert model["preset_profiles"]["packard-merlin-v1650"] == "aircraft-v12"
    assert model["preset_profiles"]["cat-c18-industrial-diesel"] == "industrial-diesel-i6"
    assert model["runtime"].startswith("AudioWorklet-thread")
    assert all(kernel["binary_bytes"] < 4096 for kernel in model["kernels"])
    assert model["dsp_chain"][-1] == "pristine-mid-high-low-pass"
    assert len(model["mid_high_rolloff"]["coefficients"]) > 64
    assert "ignition_timing_cycles" in model["telemetry"]
    emitters = model["spatial_emitters"]
    assert emitters["engine_header"]["directivity"].startswith("per-header-vectored")
    assert emitters["chassis_structure"]["directivity"] == "omnidirectional-broadcast"
    assert emitters["pitch_authority"].startswith("crank-cylinder")


def test_engine_pcm_bank_is_connected_to_audio_worklet_and_vehicle_telemetry():
    assert "class TuringEnginePCMProcessor extends AudioWorkletProcessor" in DIV_MAP_JAVASCRIPT
    assert "new AudioWorkletNode(context,\"turing-engine-pcm\"" in DIV_MAP_JAVASCRIPT
    assert "updateEngineSoundTelemetry(throttle)" in DIV_MAP_JAVASCRIPT
    assert "rpm,load:Math.min" in DIV_MAP_JAVASCRIPT
    assert "power:Math.min" in DIV_MAP_JAVASCRIPT
    assert "transient:Math.min" in DIV_MAP_JAVASCRIPT
    assert "damage:Math.max(failed/memberCount,combustionDamage)" in DIV_MAP_JAVASCRIPT
    assert "ignition_timing_cycles:Number(" in DIV_MAP_JAVASCRIPT
    assert "this.filterHistory=new Float32Array" in DIV_MAP_JAVASCRIPT
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
views[10].fill(14/720);views[11].fill(1);
const order=JSON.parse(process.argv[5]);
instance.exports.render_engine_pcm(count,...order.map(i=>offsets[i]),outputOffset);
console.log(JSON.stringify(Array.from(new Float32Array(memory.buffer,outputOffset,count))));
""".strip(), encoding="utf-8")
    completed = subprocess.run(
        ["node", str(script), str(wasm_path), str(BLOCK_SIZE), str(len(INPUT_NAMES)),
         json.dumps([INPUT_NAMES.index(name) for name in artifact.input_names])],
        check=True, capture_output=True, text=True,
    )
    samples = json.loads(completed.stdout)
    assert len(samples) == BLOCK_SIZE
    assert all(abs(sample) < 1 for sample in samples)
    assert max(samples) - min(samples) > .01


def test_engine_lowpass_keeps_centered_impulse_and_attenuates_highs():
    coefficients = np.asarray(_pristine_lowpass_kernel(2850))
    assert np.allclose(coefficients, coefficients[::-1], atol=1e-14)
    assert coefficients.sum() == pytest.approx(1)
    response = np.abs(np.fft.rfft(coefficients, 32768))
    frequencies = np.fft.rfftfreq(32768, 1 / 48000)
    assert response.max() <= 1.001
    assert response[np.argmin(abs(frequencies - 100))] > .99
    assert .4 < response[np.argmin(abs(frequencies - 2850))] < .6
    assert response[frequencies >= 6000].max() < .01


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
@pytest.mark.parametrize("artifact", ["current", "preserved-repair"])
def test_actual_engine_worklet_binds_emitted_inputs_and_renders_idle(tmp_path, artifact):
    model = engine_pcm_kernel_bank_model()
    javascript = DIV_MAP_JAVASCRIPT
    if artifact == "preserved-repair":
        from tools.repair_mechanical_creature_engine_audio import repair
        original = Path(__file__).resolve().parents[1] / "build/dot-newer-tire-dynamics-20261110/index.html"
        if not original.exists():
            pytest.skip("preserved game artifact not installed")
        page, report = repair(original.read_bytes())
        javascript = page.decode("utf-8")
        model = json.JSONDecoder().raw_decode(javascript.split('"engine_audio":', 1)[1])[0]
        assert report["reverse_byte_restore"] and report["engine_binaries_unchanged"]
        assert report["replacements"] == 3
    (tmp_path / "model.json").write_text(json.dumps(model), encoding="utf-8")
    prefix = "function engineSoundWorkletSource(){return `"
    source = javascript.split(prefix, 1)[1].split("`}", 1)[0]
    (tmp_path / "worklet.js").write_text(source, encoding="utf-8")
    script = tmp_path / "verify.cjs"
    script.write_text(r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const d=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));let Processor;
vm.runInNewContext(fs.readFileSync(process.argv[3],'utf8'),{
  AudioWorkletProcessor:class{constructor(){this.port={};}},sampleRate:48000,
  WebAssembly,Float32Array,Math,Number,Object,Map,Set,Array,Error,
  registerProcessor:(_,p)=>Processor=p});
const options={blockSize:d.block_size,inputs:d.inputs,initialProfile:'inline-six',
  rolloffCoefficients:d.mid_high_rolloff.coefficients,kernels:d.kernels.map(k=>({
    identity:k.identity,entrypoint:k.entrypoint,reservedBytes:k.reserved_bytes,
    inputNames:k.input_names,binary:Buffer.from(k.binary_base64,'base64')}))};
assert.throws(()=>new Processor({processorOptions:{...options,kernels:options.kernels.map(k=>({...k,inputNames:undefined}))}}),/input contract/);
for(const descriptor of options.kernels){
  const p=new Processor({processorOptions:{...options,initialProfile:descriptor.identity}});
  p.port.onmessage({data:{type:'telemetry',rpm:650,load:.03,power:.004,throttle:0,
    transient:0,damage:0,stall:0}});
  let energy=0,peak=0;for(let block=0;block<375;block++){
    const ch=new Float32Array(128);p.process([],[[ch]]);
    for(const sample of ch){assert(Number.isFinite(sample));energy+=sample*sample;peak=Math.max(peak,Math.abs(sample));}
    if(block===0&&descriptor.identity==='inline-six'){
      // Independent authored first-sample expression: six Lorentzian firing pulses,
      // triangle resonance and both octave shadows, followed by rational saturation.
      const cycle=.5*650/(48000*120),tri=x=>Math.abs((x-Math.floor(x))-.5)*4-1;
      const weights=[1,.93,1.04,.91,1.02,.94];
      let sum=0;weights.forEach((w,i)=>{const x=cycle+.5-i/6,dist=Math.abs(x-Math.floor(x)-.5);sum+=w/(1+(dist*(38+.03*42))**2);});
      const dry=(sum/6-.055+.14*tri(cycle*6))*(.025+.03*.34+.004*.22);
      const shadow=dry*.52*tri(cycle*3),mixed=dry+shadow+shadow*.38*tri(cycle*1.5);
      const expected=mixed/(1+Math.abs(mixed));
      assert(Math.abs(p.bank.get(descriptor.identity).output[0]-expected)<1e-7);
    }
  }
  const rms=Math.sqrt(energy/48000);assert(rms>.0001&&rms<.02);assert(peak<.05);
  console.log(descriptor.identity,JSON.stringify({rms,peak}));
}
''', encoding="utf-8")
    result = subprocess.run(["node", str(script), str(tmp_path / "model.json"),
                             str(tmp_path / "worklet.js")], check=True, capture_output=True, text=True)
    assert len(result.stdout.splitlines()) == len(model["kernels"])


def test_engine_audio_repair_rejects_unverified_base():
    from tools.repair_mechanical_creature_engine_audio import repair
    with pytest.raises(ValueError, match="exact preserved"):
        repair(b"unverified page")
