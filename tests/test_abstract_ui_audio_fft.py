"""The music-room FFT is a real C-source → IR → WebAssembly artifact."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from src.compiler.abstract_ui_audio_fft import (
    FFT_BINS,
    FFT_SIZE,
    FFTFREE_RADIX2_C_SOURCE,
    compile_audio_fft_wasm,
)


def test_fftfree_butterfly_source_is_parsed_specialized_and_embedded():
    artifact = compile_audio_fft_wasm()
    model = artifact.to_model()
    assert artifact.binary.startswith(b"\0asm")
    assert model["source_language"] == "c"
    assert model["algorithm"] == "fftfree-radix2-dit-specialized"
    assert model["lowering"] == [
        "c-source", "pycparser-ast", "fixed-size-specialization",
        "fused-program", "webassembly",
    ]
    assert "float tr = wr * real[odd] - wi * imag[odd];" in FFTFREE_RADIX2_C_SOURCE
    assert model["fft_size"] == FFT_SIZE
    assert model["output_bins"] == FFT_BINS
    assert artifact.ast_node_count > 100
    assert artifact.operation_count > 2_000
    assert model["track"]["license"] == "original generated demo audio"
    assert len(model["track"]["wav_base64"]) > 100_000


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_embedded_wasm_fft_maps_an_impulse_to_unit_magnitude_bins(tmp_path):
    artifact = compile_audio_fft_wasm()
    wasm_path = tmp_path / "music_fft.wasm"
    wasm_path.write_bytes(artifact.binary)
    script_path = tmp_path / "run.mjs"
    script_path.write_text(
        """
import {readFileSync} from "node:fs";
const {instance}=await WebAssembly.instantiate(readFileSync(process.argv[2]),{});
const base=Number(process.argv[3]),bins=Number(process.argv[4]);
const inputOffset=Math.ceil(base/4)*4;
const outputs=Array.from({length:bins},(_,i)=>inputOffset+64*4+i*4);
const memory=instance.exports.memory;
new Float32Array(memory.buffer,inputOffset,64)[0]=1;
instance.exports.analyze_music_fft(1,inputOffset,...outputs);
console.log(JSON.stringify(outputs.map(offset=>new Float32Array(memory.buffer,offset,1)[0])));
""".strip(),
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script_path), str(wasm_path), str(artifact.reserved_bytes), str(FFT_BINS)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout) == pytest.approx([1.0] * FFT_BINS, abs=1e-6)
