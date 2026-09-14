"""Benchmark LLVM perforated-network compile and execution batch sizes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from .compiled_perforated_adam import CompiledPerforatedAdam


@dataclass(frozen=True)
class BatchResult:
    batch: int
    compile_seconds: float | None
    forward_ms: float | None
    forward_samples_per_second: float | None
    train_ms: float | None
    train_samples_per_second: float | None
    library_bytes: int
    error: str | None = None


def benchmark_batches(directory: str | Path, *, batches=(1, 2, 4, 8, 16, 32),
                      in_dim: int = 416, out_dim: int = 212, branches: int = 2,
                      warmup: int = 2, repeats: int = 12, seed: int = 1729):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    results = []
    for batch in batches:
        start = time.perf_counter()
        try:
            learner = CompiledPerforatedAdam.compile(
                directory / f"batch-{batch}", batch=batch, in_dim=in_dim,
                out_dim=out_dim, dendrites_per_neuron=branches, seed=seed)
            compile_seconds = time.perf_counter() - start
            x = rng.normal(size=(batch, in_dim))
            y = rng.normal(size=(batch, out_dim))
            for _ in range(warmup):
                learner.forward(x)
            start = time.perf_counter()
            for _ in range(repeats):
                learner.forward(x)
            forward_s = (time.perf_counter() - start) / repeats
            for _ in range(warmup):
                learner.step(x, y)
            start = time.perf_counter()
            for _ in range(repeats):
                learner.step(x, y)
            train_s = (time.perf_counter() - start) / repeats
            libraries = tuple((directory / f"batch-{batch}").rglob("*.dll"))
            results.append(BatchResult(
                batch, compile_seconds, forward_s * 1000.0, batch / forward_s,
                train_s * 1000.0, batch / train_s,
                sum(path.stat().st_size for path in libraries),
            ))
        except Exception as exc:
            libraries = tuple((directory / f"batch-{batch}").rglob("*.dll"))
            results.append(BatchResult(
                batch, time.perf_counter() - start, None, None, None, None,
                sum(path.stat().st_size for path in libraries),
                f"{type(exc).__name__}: {exc}",
            ))
    report = directory / "batch-benchmark.json"
    report.write_text(json.dumps([asdict(result) for result in results], indent=2)
                      + "\n", encoding="utf-8")
    return tuple(results), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".turing-cache/perforated-batch-benchmark")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--in-dim", type=int, default=416)
    parser.add_argument("--out-dim", type=int, default=212)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    results, report = benchmark_batches(
        args.output_dir, batches=args.batches, in_dim=args.in_dim,
        out_dim=args.out_dim, branches=args.branches, repeats=args.repeats)
    for result in results:
        if result.error:
            print(f"batch={result.batch:3d} FAILED: {result.error}")
            continue
        print(
            f"batch={result.batch:3d} compile={result.compile_seconds:7.2f}s "
            f"forward={result.forward_ms:8.3f}ms "
            f"({result.forward_samples_per_second:10.1f} sample/s) "
            f"train={result.train_ms:8.3f}ms "
            f"({result.train_samples_per_second:10.1f} sample/s)"
        )
    print(report)


if __name__ == "__main__":
    main()
