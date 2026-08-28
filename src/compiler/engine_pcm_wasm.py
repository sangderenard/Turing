"""Compact streaming engine-PCM WebAssembly kernel bank.

Each kernel renders one Web Audio quantum from a baked 720-degree firing
pattern. Runtime telemetry is block-rate input; crank phase and all per-sample
detonation pulses are resolved inside WebAssembly.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..common.tensors.fused_ir import FusedProgram, Meta, OpStep
from .fused_program_wasm_backend import emit_wasm_module


BLOCK_SIZE = 128
SAMPLE_RATE = 48_000
INPUT_NAMES = (
    "sample_index", "sample_rate", "phase_start", "rpm", "load", "power", "throttle",
    "transient", "damage", "stall",
)


@dataclass(frozen=True, slots=True)
class EngineSoundProfile:
    identity: str
    label: str
    firing_weights: tuple[float, ...]
    pulse_sharpness: float
    resonance_gain: float
    mechanical_gain: float


PROFILES = (
    EngineSoundProfile("flat-four", "air-cooled flat four", (1.0, .84, 1.04, .80), 34.0, .18, .10),
    EngineSoundProfile("inline-four", "inline four", (1.0, .94, 1.02, .92), 39.0, .15, .08),
    EngineSoundProfile("flat-six", "flat six", (1.0, .93, 1.02, .95, 1.01, .92), 43.0, .13, .07),
    EngineSoundProfile("crossplane-v8", "cross-plane V8", (1.0, .78, 1.08, .72, .96, .84, 1.12, .76), 31.0, .23, .12),
    EngineSoundProfile("monster-v8", "blown monster V8", (1.12, .72, 1.18, .65, 1.06, .78, 1.22, .69), 25.0, .29, .18),
    EngineSoundProfile("electric-drive", "electric drive", tuple(1.0 for _ in range(12)), 58.0, .08, .20),
    EngineSoundProfile("servo-drive", "direct-drive servo", (1.0, .72, 1.0, .72, 1.0, .72), 72.0, .04, .27),
)


@dataclass(frozen=True, slots=True)
class EnginePCMKernel:
    profile: EngineSoundProfile
    binary: bytes
    entrypoint: str
    reserved_bytes: int
    operation_count: int

    def to_model(self) -> dict[str, Any]:
        return {
            "identity": self.profile.identity,
            "label": self.profile.label,
            "entrypoint": self.entrypoint,
            "binary_base64": base64.b64encode(self.binary).decode("ascii"),
            "binary_bytes": len(self.binary),
            "reserved_bytes": self.reserved_bytes,
            "operation_count": self.operation_count,
            "firing_weights": list(self.profile.firing_weights),
        }


class _ProgramBuilder:
    def __init__(self) -> None:
        self.steps: list[OpStep] = []
        self.next_id = len(INPUT_NAMES) + 1

    def unary(self, op: str, value: int) -> int:
        result = self.next_id
        self.next_id += 1
        self.steps.append(OpStep(len(self.steps), op, [value], {}, result))
        return result

    def scalar(self, op: str, value: int, scalar: float) -> int:
        result = self.next_id
        self.next_id += 1
        self.steps.append(OpStep(len(self.steps), op, [value], {"right_scalar": float(scalar)}, result))
        return result

    def binary(self, op: str, left: int, right: int) -> int:
        result = self.next_id
        self.next_id += 1
        self.steps.append(OpStep(len(self.steps), op, [left, right], {}, result))
        return result

    def ternary(self, op: str, first: int, second: int, third: int) -> int:
        result = self.next_id
        self.next_id += 1
        self.steps.append(OpStep(len(self.steps), op, [first, second, third], {}, result))
        return result

    def fract(self, value: int) -> int:
        return self.binary("sub", value, self.unary("floor", value))

    def triangle(self, value: int) -> int:
        centered = self.scalar("sub", self.fract(value), .5)
        return self.scalar("sub", self.scalar("mul", self.unary("abs", centered), 4.0), 1.0)


def _profile_program(profile: EngineSoundProfile) -> FusedProgram:
    builder = _ProgramBuilder()
    feed_ids = tuple(range(1, len(INPUT_NAMES) + 1))
    sample_index, sample_rate, phase_start, rpm, load, power, throttle, transient, damage, stall = feed_ids
    cycles_per_sample = builder.binary("div", rpm, builder.scalar("mul", sample_rate, 120.0))
    # Evaluate at sample centers; this also keeps pulse discontinuities off an
    # exact quantum boundary when phase_start is precisely zero.
    cycle = builder.binary("add", phase_start, builder.binary(
        "mul", builder.scalar("add", sample_index, .5), cycles_per_sample))
    zero = builder.scalar("mul", phase_start, 0.0)
    one = builder.scalar("add", zero, 1.0)
    sharpness = builder.binary("add", builder.scalar("mul", load, 42.0),
                               builder.scalar("mul", phase_start, 0.0))
    sharpness = builder.scalar("add", sharpness, profile.pulse_sharpness)

    pulse_sum: int | None = None
    event_count = len(profile.firing_weights)
    for index, weight in enumerate(profile.firing_weights):
        event_phase = index / event_count
        wrapped = builder.fract(builder.scalar("add", cycle, .5 - event_phase))
        distance = builder.unary("abs", builder.scalar("sub", wrapped, .5))
        scaled = builder.binary("mul", distance, sharpness)
        denominator = builder.scalar("add", builder.binary("mul", scaled, scaled), 1.0)
        inverse = builder.binary("div", one, denominator)
        pulse = builder.scalar("mul", inverse, weight)
        pulse_sum = pulse if pulse_sum is None else builder.binary("add", pulse_sum, pulse)
    assert pulse_sum is not None
    combustion = builder.scalar("mul", pulse_sum, 1.0 / event_count)
    combustion = builder.scalar("sub", combustion, .055)

    firing_phase = builder.scalar("mul", cycle, event_count)
    resonance = builder.scalar("mul", builder.triangle(firing_phase), profile.resonance_gain)
    mechanical_phase = builder.scalar("mul", cycle, event_count * 3.173)
    mechanical = builder.scalar("mul", builder.triangle(mechanical_phase), profile.mechanical_gain)
    amplitude = builder.scalar("add", builder.scalar("mul", load, .34), .025)
    for value, gain in ((power, .22), (throttle, .09), (transient, .16)):
        amplitude = builder.binary("add", amplitude, builder.scalar("mul", value, gain))
    core = builder.binary("mul", builder.binary("add", combustion, resonance), amplitude)
    roughness = builder.binary("add", builder.scalar("mul", damage, .13),
                               builder.scalar("mul", transient, .055))
    rough = builder.binary("mul", mechanical, roughness)
    stall_lug = builder.binary("mul", builder.scalar("mul", builder.triangle(
        builder.scalar("mul", cycle, .5)), .24), stall)
    mixed = builder.binary("add", builder.binary("add", core, rough), stall_lug)
    saturated = builder.binary("div", mixed, builder.scalar("add", builder.unary("abs", mixed), 1.0))
    saturated = builder.ternary("where", builder.unary("isfinite", saturated), saturated, zero)
    meta = {value: Meta(shape=(1, BLOCK_SIZE), dtype="float32") for value in feed_ids}
    program = FusedProgram(version=1, feeds=set(feed_ids), steps=builder.steps,
                           outputs={"pcm": saturated}, meta=meta,
                           extras={"block_size": BLOCK_SIZE, "sample_rate": SAMPLE_RATE})
    program.feed_order = feed_ids
    return program


@lru_cache(maxsize=None)
def compile_engine_pcm_kernel(profile_identity: str) -> EnginePCMKernel:
    profile = next((item for item in PROFILES if item.identity == profile_identity), None)
    if profile is None:
        raise KeyError(f"unknown engine sound profile: {profile_identity}")
    program = _profile_program(profile)
    module = emit_wasm_module(program, name=f"engine_pcm_{profile.identity.replace('-', '_')}",
                              function_name="render_engine_pcm", dtype="float32")
    if not module.complete or module.binary is None:
        raise RuntimeError(module.shortfall_report())
    entry = module.api.entry_points[0]
    return EnginePCMKernel(profile, module.binary, entry.symbol,
                           int(module.api.metadata.get("reserved_bytes", 0)), len(program.steps))


@lru_cache(maxsize=1)
def engine_pcm_kernel_bank_model() -> dict[str, Any]:
    return {
        "schema": "abstract-ui-engine-pcm-kernel-bank-v0",
        "block_size": BLOCK_SIZE,
        "sample_rate": SAMPLE_RATE,
        "inputs": list(INPUT_NAMES),
        "phase": "continuous-720-degree-four-stroke-cycle-owned-by-audio-worklet",
        "runtime": "AudioWorklet-thread-calls-selected-webassembly-kernel-per-quantum",
        "kernels": [compile_engine_pcm_kernel(profile.identity).to_model() for profile in PROFILES],
        "preset_profiles": {
            "aircooled-flat-four-1584": "flat-four",
            "springtail-i4-1600": "inline-four",
            "superbike-i4-1340": "inline-four",
            "gt-flat-six-4000": "flat-six",
            "supercharged-drag-v8-8200": "monster-v8",
            "monster-540-blown-methanol": "monster-v8",
            "monster-632-twin-turbo": "monster-v8",
            "dual-motor-ev-reference": "electric-drive",
            "servo-direct-drive-400": "servo-drive",
        },
        "telemetry": ["rpm", "load", "power", "throttle", "transient", "damage", "stall"],
    }


__all__ = ["BLOCK_SIZE", "INPUT_NAMES", "PROFILES", "SAMPLE_RATE", "EnginePCMKernel",
           "compile_engine_pcm_kernel", "engine_pcm_kernel_bank_model"]
