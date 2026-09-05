"""C-source FFT kernel and original music loop for the AbstractUI demo.

The browser artifact is deliberately small and auditable: a fixed 64-point
radix-2 DIT kernel is parsed as C, specialized into the common numeric IR, and
then emitted by Turing's WebAssembly backend.  The butterfly is the same
``t=w*b; y0=a+t; y1=a-t`` operation used by fftfree's radix-2 kernel; importing
only that bounded algorithm avoids shipping Eigen and a C++ runtime to a
static page.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import io
import math
import struct
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

from pycparser import c_ast, c_parser

from ..common.tensors.fused_ir import FusedProgram, Meta, OpStep
from .fused_program_wasm_backend import emit_wasm_module


FFT_SIZE = 64
FFT_BINS = 24
TRACK_SAMPLE_RATE = 16_000
TRACK_SECONDS = 8

FFTFREE_RADIX2_C_SOURCE = r"""
void fftfree_radix2_64(float real[64], float imag[64]) {
  int span, base, lane;
  for (span = 2; span <= 64; span = span * 2) {
    for (base = 0; base < 64; base = base + span) {
      for (lane = 0; lane < span / 2; lane = lane + 1) {
        float angle = -6.283185307179586f * lane / span;
        float wr = cosf(angle);
        float wi = sinf(angle);
        int even = base + lane;
        int odd = even + span / 2;
        float tr = wr * real[odd] - wi * imag[odd];
        float ti = wr * imag[odd] + wi * real[odd];
        float ar = real[even];
        float ai = imag[even];
        real[even] = ar + tr;
        imag[even] = ai + ti;
        real[odd] = ar - tr;
        imag[odd] = ai - ti;
      }
    }
  }
}
""".strip()


@dataclass(frozen=True, slots=True)
class AudioFFTArtifact:
    binary: bytes
    entrypoint: str
    parameters: tuple[dict[str, str], ...]
    operation_count: int
    reserved_bytes: int
    ast_node_count: int
    source_sha256: str
    track_wav_base64: str
    track_sample_rate: int
    track_samples: int

    def to_model(self) -> dict[str, Any]:
        return {
            "schema": "abstract-ui-music-room-v0",
            "algorithm": "fftfree-radix2-dit-specialized",
            "source_language": "c",
            "source": FFTFREE_RADIX2_C_SOURCE,
            "source_sha256": self.source_sha256,
            "source_origin": {
                "repository": "https://github.com/sangderenard/fftfree",
                "path": "include/fftfree/detail/butterfly_kernel.hpp",
                "relationship": "bounded radix-2 DIT butterfly import",
            },
            "lowering": ["c-source", "pycparser-ast", "fixed-size-specialization",
                         "fused-program", "webassembly"],
            "ast_node_count": self.ast_node_count,
            "fft_size": FFT_SIZE,
            "output_bins": FFT_BINS,
            "value_type": "float32",
            "entrypoint": self.entrypoint,
            "parameters": [dict(item) for item in self.parameters],
            "operation_count": self.operation_count,
            "reserved_bytes": self.reserved_bytes,
            "binary_bytes": len(self.binary),
            "binary_base64": base64.b64encode(self.binary).decode("ascii"),
            "track": {
                "title": "Chromatic Impact Loop",
                "license": "original generated demo audio",
                "format": "audio/wav; pcm=s16le; channels=1",
                "sample_rate": self.track_sample_rate,
                "samples": self.track_samples,
                "wav_base64": self.track_wav_base64,
            },
            "analysis_contract": {
                "input": "64 contiguous mono PCM float32 samples",
                "outputs": "magnitude-squared bins 0 through 23",
                "synchronization": "AudioContext.currentTime playback cursor",
            },
        }


def _parse_and_validate_source() -> tuple[c_ast.FileAST, int]:
    source = FFTFREE_RADIX2_C_SOURCE.replace("6.283185307179586f", "6.283185307179586")
    tree = c_parser.CParser().parse(source)
    counts = {"for": 0, "array": 0, "call": 0}

    class Visitor(c_ast.NodeVisitor):
        def generic_visit(self, node):  # type: ignore[no-untyped-def]
            if isinstance(node, c_ast.For):
                counts["for"] += 1
            elif isinstance(node, c_ast.ArrayRef):
                counts["array"] += 1
            elif isinstance(node, c_ast.FuncCall):
                counts["call"] += 1
            super().generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    if counts["for"] != 3 or counts["array"] < 10 or counts["call"] != 2:
        raise ValueError(f"unexpected fftfree radix-2 C structure: {counts}")
    return tree, sum(1 for _ in _walk_ast(tree))


def _walk_ast(node: c_ast.Node):
    yield node
    for _name, child in node.children():
        yield from _walk_ast(child)


def _bit_reverse(value: int, bits: int) -> int:
    result = 0
    for _ in range(bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _fft_program() -> FusedProgram:
    base = 1
    next_id = FFT_SIZE + 2
    steps: list[OpStep] = []
    meta: dict[int, Meta] = {base: Meta(shape=(1, FFT_SIZE), dtype="float32")}
    real: list[int] = []
    imag: list[int] = []
    bits = int(math.log2(FFT_SIZE))
    for lane in range(FFT_SIZE):
        view_id = lane + 2
        meta[view_id] = Meta(shape=(1,), dtype="float32", source_id=base,
                             offset=_bit_reverse(lane, bits), stride=1)
        real.append(view_id)
        zero_id = next_id
        next_id += 1
        steps.append(OpStep(len(steps), "mul", [view_id], {"right_scalar": 0.0}, zero_id))
        imag.append(zero_id)

    def unary(op: str, value: int, scalar: float) -> int:
        nonlocal next_id
        result = next_id
        next_id += 1
        steps.append(OpStep(len(steps), op, [value], {"right_scalar": scalar}, result))
        return result

    def binary(op: str, left: int, right: int) -> int:
        nonlocal next_id
        result = next_id
        next_id += 1
        steps.append(OpStep(len(steps), op, [left, right], {}, result))
        return result

    span = 2
    while span <= FFT_SIZE:
        for start in range(0, FFT_SIZE, span):
            for lane in range(span // 2):
                even, odd = start + lane, start + lane + span // 2
                angle = -2.0 * math.pi * lane / span
                wr, wi = math.cos(angle), math.sin(angle)
                tr = binary("sub", unary("mul", real[odd], wr), unary("mul", imag[odd], wi))
                ti = binary("add", unary("mul", imag[odd], wr), unary("mul", real[odd], wi))
                ar, ai = real[even], imag[even]
                real[even], real[odd] = binary("add", ar, tr), binary("sub", ar, tr)
                imag[even], imag[odd] = binary("add", ai, ti), binary("sub", ai, ti)
        span *= 2

    outputs: dict[str, int] = {}
    for lane in range(FFT_BINS):
        rr = binary("mul", real[lane], real[lane])
        ii = binary("mul", imag[lane], imag[lane])
        outputs[f"bin_{lane}"] = binary("add", rr, ii)
    return FusedProgram(version=1, feeds=set(range(2, FFT_SIZE + 2)), steps=steps,
                        outputs=outputs, meta=meta,
                        extras={"fft_size": FFT_SIZE, "output_bins": FFT_BINS})


def _music_wav() -> tuple[str, int]:
    """Synthesize a compact, loopable original track with strong transients."""

    count = TRACK_SAMPLE_RATE * TRACK_SECONDS
    pcm = bytearray()
    chord = (110.0, 138.59, 164.81, 220.0)
    for index in range(count):
        t = index / TRACK_SAMPLE_RATE
        beat = (t * 2.0) % 1.0
        half = (t * 4.0) % 1.0
        kick_env = math.exp(-beat * 13.0)
        kick = math.sin(2 * math.pi * (48.0 + 38.0 * kick_env) * t) * kick_env
        click_env = math.exp(-half * 34.0)
        click = math.sin(2 * math.pi * 1320.0 * t) * click_env
        note = chord[int(t * 2) % len(chord)]
        bass = math.sin(2 * math.pi * note * t) * (0.38 + 0.12 * math.sin(2 * math.pi * .25 * t))
        shimmer = sum(math.sin(2 * math.pi * note * ratio * t + ratio)
                      for ratio in (2.0, 3.0, 4.02)) / 3.0
        sample = math.tanh(1.15 * (0.52 * kick + 0.12 * click + 0.34 * bass + 0.14 * shimmer))
        pcm.extend(struct.pack("<h", max(-32767, min(32767, round(sample * 32767)))))
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TRACK_SAMPLE_RATE)
        wav.writeframes(bytes(pcm))
    return base64.b64encode(stream.getvalue()).decode("ascii"), count


@lru_cache(maxsize=1)
def compile_audio_fft_wasm() -> AudioFFTArtifact:
    _tree, ast_nodes = _parse_and_validate_source()
    program = _fft_program()
    module = emit_wasm_module(program, name="abstract_ui_music_fft",
                              function_name="analyze_music_fft", dtype="float32")
    if not module.complete or module.binary is None:
        raise RuntimeError(module.shortfall_report())
    entry = module.api.entry_points[0]
    wav, samples = _music_wav()
    return AudioFFTArtifact(
        binary=module.binary,
        entrypoint=entry.symbol,
        parameters=tuple({"name": p.name, "role": p.role, "dtype": p.dtype}
                         for p in entry.parameters),
        operation_count=len(program.steps),
        reserved_bytes=int(module.api.metadata.get("reserved_bytes", 0)),
        ast_node_count=ast_nodes,
        source_sha256=hashlib.sha256(FFTFREE_RADIX2_C_SOURCE.encode()).hexdigest(),
        track_wav_base64=wav,
        track_sample_rate=TRACK_SAMPLE_RATE,
        track_samples=samples,
    )


__all__ = ["AudioFFTArtifact", "FFTFREE_RADIX2_C_SOURCE", "FFT_BINS", "FFT_SIZE",
           "compile_audio_fft_wasm"]
