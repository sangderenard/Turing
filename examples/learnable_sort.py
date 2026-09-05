"""Exact eight-value sorting oracle for the native affine learner."""

from __future__ import annotations

import numpy as np


# Batcher sorting network for eight wires. ``process_forward`` is deliberately
# written against the AbstractTensor-compatible method surface so the compiler
# captures these compare/exchange segments as the real forward program.
COMPARATORS = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (1, 2), (5, 6),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (2, 4), (3, 5),
    (1, 2), (3, 4), (5, 6),
)


def process_forward(values):
    """Trainable sorting process captured into forward and reverse Graph IR."""

    wires = [values[f"x{index}"] for index in range(8)]
    for index, (left_index, right_index) in enumerate(COMPARATORS):
        left = wires[left_index]
        right = wires[right_index]
        exact_low = left.minimum(right)
        exact_high = left.maximum(right)
        gate = values[f"gate{index}"]
        wires[left_index] = left + gate * (exact_low - left)
        wires[right_index] = right + gate * (exact_high - right)
    return {f"sorted_{index}": wire for index, wire in enumerate(wires)}


def build_process_problem():
    return {
        "name": "sort_eight_values",
        "input_names": tuple(f"x{index}" for index in range(8)),
        "parameter_names": tuple(f"gate{index}" for index in range(len(COMPARATORS))),
        "forward": process_forward,
        "reference": lambda rows: np.sort(rows, axis=1),
        "comparators": COMPARATORS,
    }


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
