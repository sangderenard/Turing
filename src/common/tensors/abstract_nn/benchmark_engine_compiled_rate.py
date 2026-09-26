"""Compare real EngineCycleSim throughput with compiled learned inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .demo_engine_dendrite_live import _drive_engine, _render_audio_frame
from .demo_perforated_multifuel_engine import ENGINE_IDENTITY, _load_engine_toy
from ....compiler.compiled_perforated_adam import CompiledPerforatedAdam


def _load_weights(learner, path: Path | None) -> None:
    if path is None:
        return
    with np.load(path) as values:
        for name in learner.parameter_names:
            learner.parameters[name][...] = values[name]


def _compiled_rate(learner, repeats: int) -> dict:
    batch, in_dim = learner.shapes["x"]
    values = np.zeros((batch, in_dim), dtype=np.float64)
    for _ in range(5):
        learner.forward(values)
    start = time.perf_counter()
    for _ in range(repeats):
        learner.forward(values)
    elapsed = time.perf_counter() - start
    return {
        "batch": batch,
        "calls": repeats,
        "elapsed_seconds": elapsed,
        "milliseconds_per_call": elapsed * 1000.0 / repeats,
        "calls_per_second": repeats / elapsed,
        "samples_per_second": repeats * batch / elapsed,
    }


def _engine_rate(repeats: int, dt: float, *, audio: bool,
                 engine_toy_path=None) -> dict:
    EngineCycleSim, get_engine = _load_engine_toy(engine_toy_path)
    sim = EngineCycleSim(get_engine(ENGINE_IDENTITY))
    sim.start()
    _drive_engine(sim, 1, 5)
    audio_state = synth = None
    frames = 0
    if audio:
        from audio_stream import LiveAudioState
        from engine_sound import SAMPLE_RATE, EngineSoundSynth
        audio_state = LiveAudioState(sim.engine)
        synth = EngineSoundSynth(SAMPLE_RATE)
        frames = max(1, round(SAMPLE_RATE * dt))
        audio_state.set_from_sim(sim)
        _render_audio_frame(audio_state, synth, frames)
    for _ in range(5):
        sim.step(dt)
    start = time.perf_counter()
    for _ in range(repeats):
        sim.step(dt)
        if audio:
            audio_state.set_from_sim(sim)
            _render_audio_frame(audio_state, synth, frames)
        if sim.state.stalled:
            sim.start()
    elapsed = time.perf_counter() - start
    return {
        "calls": repeats,
        "simulated_dt_seconds": dt,
        "elapsed_seconds": elapsed,
        "milliseconds_per_call": elapsed * 1000.0 / repeats,
        "calls_per_second": repeats / elapsed,
        "simulated_seconds_per_wall_second": repeats * dt / elapsed,
        "audio": audio,
    }


def benchmark_engine_compiled_rate(
    directory: str | Path, *, in_dim: int = 416, out_dim: int = 212,
    branches: int = 2, inference_repeats: int = 100,
    engine_repeats: int = 100, engine_dt: float = 0.005,
    weights: str | Path | None = None, engine_toy_path=None,
):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    weight_path = Path(weights).resolve() if weights else None
    compiled = []
    for batch in (1, 16):
        start = time.perf_counter()
        learner = CompiledPerforatedAdam.compile(
            directory / f"batch-{batch}", batch=batch, in_dim=in_dim,
            out_dim=out_dim, dendrites_per_neuron=branches, seed=1729)
        compile_seconds = time.perf_counter() - start
        _load_weights(learner, weight_path)
        result = _compiled_rate(learner, inference_repeats)
        result["compile_seconds"] = compile_seconds
        compiled.append(result)
    engine = _engine_rate(
        engine_repeats, engine_dt, audio=False,
        engine_toy_path=engine_toy_path)
    engine_audio = _engine_rate(
        engine_repeats, engine_dt, audio=True,
        engine_toy_path=engine_toy_path)
    for result in compiled:
        result["speedup_over_engine_per_sample"] = (
            result["samples_per_second"] / engine["calls_per_second"])
        result["speedup_over_engine_audio_per_sample"] = (
            result["samples_per_second"] / engine_audio["calls_per_second"])
    payload = {
        "schema": "turing.engine-vs-compiled-rate",
        "engine": engine,
        "engine_with_audio": engine_audio,
        "compiled_inference": compiled,
        "comparison_unit": (
            "one 5ms EngineCycleSim call versus one learned next-state sample"
        ),
        "weights": str(weight_path) if weight_path else None,
    }
    report = directory / "engine-vs-compiled-rate.json"
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".turing-cache/engine-vs-compiled-rate")
    parser.add_argument("--in-dim", type=int, default=416)
    parser.add_argument("--out-dim", type=int, default=212)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--inference-repeats", type=int, default=100)
    parser.add_argument("--engine-repeats", type=int, default=100)
    parser.add_argument("--engine-dt", type=float, default=0.005)
    parser.add_argument("--weights")
    parser.add_argument("--engine-toy-path")
    args = parser.parse_args()
    payload, report = benchmark_engine_compiled_rate(
        args.output_dir, in_dim=args.in_dim, out_dim=args.out_dim,
        branches=args.branches, inference_repeats=args.inference_repeats,
        engine_repeats=args.engine_repeats, engine_dt=args.engine_dt,
        weights=args.weights, engine_toy_path=args.engine_toy_path)
    for label in ("engine", "engine_with_audio"):
        value = payload[label]
        print(f"{label}: {value['milliseconds_per_call']:.3f} ms/call, "
              f"{value['calls_per_second']:.2f} calls/s, "
              f"{value['simulated_seconds_per_wall_second']:.4f}x realtime")
    for value in payload["compiled_inference"]:
        print(f"compiled batch {value['batch']}: "
              f"{value['milliseconds_per_call']:.3f} ms/call, "
              f"{value['samples_per_second']:.1f} samples/s, "
              f"{value['speedup_over_engine_per_sample']:.1f}x engine")
    print(report)


if __name__ == "__main__":
    main()
