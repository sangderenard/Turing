"""Exact eight-value sorting oracle for the native affine learner."""

from __future__ import annotations

import numpy as np


def build_benchmark(*, seed: int, train_samples: int, validation_samples: int):
    rng = np.random.default_rng(seed)
    train = rng.uniform(-1.0, 1.0, size=(train_samples, 8))
    validation = rng.uniform(-1.0, 1.0, size=(validation_samples, 8))
    return {
        "name": "sort_eight_values",
        "train_inputs": train,
        "train_targets": np.sort(train, axis=1),
        "validation_inputs": validation,
        "validation_targets": np.sort(validation, axis=1),
        # A comparison sort needs roughly n*log2(n) comparisons.  This is a
        # transparent cost reference, not a wall-clock performance claim.
        "reference_operations": 8 * 3,
    }
