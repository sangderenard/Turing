"""Monte Carlo benchmark for ASCII diff and NN classification pipelines."""
from __future__ import annotations

import cProfile
import os
import statistics
import time
from typing import Any

import numpy as np
import pstats

from ..ascii_render import AsciiRenderer
from .draw import _classifier_cache


def _summary_stats(data: list[float]) -> dict[str, float]:
    """Compute basic summary statistics for a list of times in milliseconds."""

    if not data:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "stdev": 0.0, "p95": 0.0}
    stats: dict[str, float] = {
        "min": min(data),
        "max": max(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "stdev": statistics.stdev(data) if len(data) > 1 else 0.0,
    }
    # statistics.quantiles uses 0-based indexing for n=100 percentiles
    try:
        stats["p95"] = statistics.quantiles(data, n=100)[94]
    except Exception:
        stats["p95"] = stats["max"]
    return stats


def run(
    iterations: int = 500,
    width: int = 64,
    height: int = 32,
    *,
    enable_cprofile: bool = True,
) -> None:
    """Execute the benchmark and print timing statistics."""

    os.environ["TURING_PROFILE"] = "1"
    renderer = AsciiRenderer(width, height)
    rng = np.random.default_rng(0)

    ascii_times: list[float] = []
    classify_times: list[float] = []
    patch_meta: list[dict[str, Any]] = []
    prev_classify_count = 0

    profiler = cProfile.Profile() if enable_cprofile else None
    if profiler:
        profiler.enable()

    for _ in range(iterations):
        if rng.random() < 0.5:
            # modify a small random patch
            w = int(rng.integers(1, max(2, width // 4)))
            h = int(rng.integers(1, max(2, height // 4)))
            x = int(rng.integers(0, width - w + 1))
            y = int(rng.integers(0, height - h + 1))
            patch = rng.integers(0, 256, size=(h, w, 1), dtype=np.uint8)
            renderer.canvas[y : y + h, x : x + w, 0] = patch[..., 0]
            patch_meta.append({"patch_type": "patch", "patch_size": (w, h)})
        else:
            # replace the entire canvas
            renderer.canvas[..., 0] = rng.integers(
                0, 256, size=renderer.canvas[..., 0].shape, dtype=np.uint8
            )
            patch_meta.append({"patch_type": "full", "patch_size": (width, height)})

        start = time.perf_counter()
        ascii_img = renderer.to_ascii_diff()
        print(ascii_img)
        ascii_times.append((time.perf_counter() - start) * 1000.0)

        classifier = _classifier_cache.get("classifier")
        if classifier:
            new_durations = classifier.classify_durations[prev_classify_count:]
            classify_times.append(sum(new_durations))
            prev_classify_count = len(classifier.classify_durations)
        else:
            classify_times.append(0.0)

    if profiler:
        profiler.disable()

    ascii_ms = renderer.profile_stats.get("to_ascii_diff_ms", 0.0)
    classifier = _classifier_cache.get("classifier")
    nn_ms = classifier.profile_stats.get("classify_ms", 0.0) if classifier else 0.0
    train_ms = classifier.profile_stats.get("train_ms", 0.0) if classifier else 0.0

    print(f"ASCII diff total time: {ascii_ms:.2f} ms")
    print(f"ASCII diff per-iteration stats (ms): {_summary_stats(ascii_times)}")
    print(f"NN classify total time: {nn_ms:.2f} ms")
    print(f"NN classify per-iteration stats (ms): {_summary_stats(classify_times)}")
    if classifier:
        print(f"NN training time: {train_ms:.2f} ms")
    full_count = sum(1 for m in patch_meta if m["patch_type"] == "full")
    print(
        f"Iterations: {len(patch_meta)}, full replacements: {full_count}, patches: {len(patch_meta) - full_count}"
    )

    if profiler:
        print("\nTop functions by cumulative time:")
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(40)


if __name__ == "__main__":
    run()

