"""Monte Carlo benchmark for ASCII diff and NN classification pipelines."""
from __future__ import annotations

import os
import numpy as np

from ..ascii_render import AsciiRenderer
from .draw import _classifier_cache


def run(iterations: int = 100, width: int = 64, height: int = 32) -> None:
    """Execute the benchmark and print timing statistics."""
    os.environ["TURING_PROFILE"] = "1"
    renderer = AsciiRenderer(width, height)
    rng = np.random.default_rng(0)

    for _ in range(iterations):
        if rng.random() < 0.5:
            # modify a small random patch
            w = int(rng.integers(1, max(2, width // 4)))
            h = int(rng.integers(1, max(2, height // 4)))
            x = int(rng.integers(0, width - w + 1))
            y = int(rng.integers(0, height - h + 1))
            patch = rng.integers(0, 256, size=(h, w, 1), dtype=np.uint8)
            renderer.canvas[y : y + h, x : x + w, 0] = patch[..., 0]
        else:
            # replace the entire canvas
            renderer.canvas[..., 0] = rng.integers(
                0, 256, size=renderer.canvas[..., 0].shape, dtype=np.uint8
            )
        renderer.to_ascii_diff()

    ascii_ms = renderer.profile_stats.get("to_ascii_diff_ms", 0.0)
    classifier = _classifier_cache.get("classifier")
    nn_ms = classifier.profile_stats.get("classify_ms", 0.0) if classifier else 0.0
    train_ms = classifier.profile_stats.get("train_ms", 0.0) if classifier else 0.0

    print(f"ASCII diff total time: {ascii_ms:.2f} ms")
    print(f"NN classify total time: {nn_ms:.2f} ms")
    if classifier:
        print(f"NN training time: {train_ms:.2f} ms")


if __name__ == "__main__":
    run()

