"""Native training of the learned-design palette scorer.

Follows ``examples/pattern_project/native_image_training.py``: the forward and
loss are ordinary AbstractTensor arithmetic recorded on an
``SSATensorProgram``; ``compile_native_training_schedule`` derives the
backward from ``BACKWARD_RULES`` and compiles one native LLVM library with a
stepping entry per parameter group.  Python owns only the buffers.

Model: a design's feature vector (``palette_designs.features``) is scored by a
shared two-layer network plus a personal linear layer weighted ``ALPHA``.
Feedback arrives as ranked pairs (the preferred design first) and slider
ratings in [0, 1]; empty slots carry mask 0.  The personal layer is pulled
toward zero so it only nudges the shared default.

    python -m examples.meowdoku.palette_native_training --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from .palette_designs import FEATURES


# The compiler runtime is imported where it is used: the game imports this
# module for its shapes and loss on every server start, and compiles or loads
# the schedule once, in the model's own thread.
def load_training_schedule(path):
    from src.compiler.llvm_training_runtime import load_training_schedule as load
    return load(path)


def run_parameter_group(schedule, group_name, buffers, *, steps=1, learning_rate=None):
    from src.compiler.llvm_training_runtime import run_parameter_group as run
    return run(schedule, group_name, buffers, steps=steps, learning_rate=learning_rate)

HIDDEN = 64         # the design is a palette and a token per page element (palette_designs.FEATURES)
PAIRS = 64          # ranked-pair slots per step
RATINGS = 16        # slider-rating slots per step
ALPHA = 0.25        # weight of the personal layer in a score
PULL = 0.3          # how strongly the personal layer is held near zero

# Binding order is the value-id order the recorded program assigns.
BINDINGS = (
    ("pair_preferred", (PAIRS, FEATURES)),
    ("pair_other", (PAIRS, FEATURES)),
    ("pair_mask", (PAIRS, 1)),
    ("rated", (RATINGS, FEATURES)),
    ("rating", (RATINGS, 1)),
    ("rating_mask", (RATINGS, 1)),
    ("shared.w1", (FEATURES, HIDDEN)),
    ("shared.b1", (1, HIDDEN)),
    ("shared.w2", (HIDDEN, 1)),
    ("shared.b2", (1, 1)),
    ("personal.u", (FEATURES, 1)),
)
SHARED = ("shared.w1", "shared.b1", "shared.w2", "shared.b2")
PERSONAL = ("personal.u",)


def _sigmoid(x):
    return ((x * 0.5).tanh() + 1.0) * 0.5


def _score(values, x):
    hidden = (x.matmul(values["shared.w1"]) + values["shared.b1"]).tanh()
    return (
        hidden.matmul(values["shared.w2"]) + values["shared.b2"]
        + x.matmul(values["personal.u"]) * ALPHA
    )


def palette_loss(values):
    """Ranked pairs want sigmoid(score difference) -> 1; ratings want
    sigmoid(score) -> slider; the personal layer stays light.  Returns the
    loss and the scores of the ``rated`` rows, which the game also reads to
    score designs (the observed output must be a value of the loss graph)."""

    pair = (_sigmoid(_score(values, values["pair_preferred"])
                     - _score(values, values["pair_other"])) - 1.0) * values["pair_mask"]
    rated_scores = _score(values, values["rated"])
    rate = (_sigmoid(rated_scores) - values["rating"]) * values["rating_mask"]
    personal = values["personal.u"]
    loss = (pair * pair).mean() + (rate * rate).mean() + (personal * personal).mean() * PULL
    return loss, rated_scores


def reference_loss(buffers: dict[int, np.ndarray], ids: dict[str, int]) -> float:
    """The same loss in NumPy, for checking the native result."""

    v = {name: np.asarray(buffers[ids[name]], dtype=np.float64) for name in ids}
    sig = lambda x: (np.tanh(x * 0.5) + 1.0) * 0.5
    score = lambda x: (np.tanh(x @ v["shared.w1"] + v["shared.b1"]) @ v["shared.w2"]
                       + v["shared.b2"] + x @ v["personal.u"] * ALPHA)
    pair = (sig(score(v["pair_preferred"]) - score(v["pair_other"])) - 1.0) * v["pair_mask"]
    rate = (sig(score(v["rated"])) - v["rating"]) * v["rating_mask"]
    return float((pair * pair).mean() + (rate * rate).mean()
                 + (v["personal.u"] ** 2).mean() * PULL)


def compile_palette_trainer(directory: Path, *, learning_rate: float = 0.5):
    from src.common.tensors.accelerator_backends.ssa_backend import (
        SSATensorOperations, SSATensorProgram,
    )
    from src.compiler.llvm_training_runtime import (
        NativeParameterGroup, compile_native_training_schedule,
        save_training_schedule,
    )

    program = SSATensorProgram("palette_scorer")
    values = {
        name: SSATensorOperations.input(program, shape)
        for name, shape in BINDINGS
    }
    ids = {name: int(value.data.value.id) for name, value in values.items()}
    loss, candidates = palette_loss(values)
    schedule = compile_native_training_schedule(
        loss,
        bindings=values,
        parameter_groups=(
            NativeParameterGroup("shared", tuple(ids[n] for n in SHARED), learning_rate),
            NativeParameterGroup("personal", tuple(ids[n] for n in PERSONAL), learning_rate),
        ),
        observed_outputs={"scores": int(candidates.data.value.id)},
        name="palette_scorer",
        directory=directory / "artifacts",
    )
    save_training_schedule(schedule, directory / "schedule.json")
    return schedule, ids


def initial_parameters(ids: dict[str, int], seed: int = 0) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    shapes = dict(BINDINGS)
    buffers = {ids[name]: np.zeros(shapes[name]) for name, _shape in BINDINGS}
    # Scaled by each layer's fan-in, so a wide design (FEATURES inputs) does not saturate the tanh layer at the start:
    # with 11 inputs and 16 hidden units these are the 0.3 the network began with before palettes.
    buffers[ids["shared.w1"]] = rng.normal(0.0, 1.0 / np.sqrt(FEATURES), shapes["shared.w1"])
    buffers[ids["shared.w2"]] = rng.normal(0.0, 1.2 / np.sqrt(HIDDEN), shapes["shared.w2"])
    return buffers


def feedback_batch(buffers, ids, *, pairs=(), ratings=()):
    """Write ranked pairs ``(preferred, other)`` and ``(features, slider)``
    ratings into the fixed slots; unused slots are masked out."""

    if len(pairs) > PAIRS or len(ratings) > RATINGS:
        raise ValueError(f"at most {PAIRS} pairs and {RATINGS} ratings per step")
    for name, rows in (("pair_preferred", [p for p, _ in pairs]),
                       ("pair_other", [o for _, o in pairs]),
                       ("rated", [f for f, _ in ratings])):
        block = np.zeros(dict(BINDINGS)[name])
        if rows:
            block[:len(rows)] = np.asarray(rows, dtype=np.float64)
        buffers[ids[name]] = block
    for name, count in (("pair_mask", len(pairs)), ("rating_mask", len(ratings))):
        mask = np.zeros(dict(BINDINGS)[name])
        mask[:count] = 1.0
        buffers[ids[name]] = mask
    rating = np.zeros((RATINGS, 1))
    rating[:len(ratings), 0] = [r for _, r in ratings]
    buffers[ids["rating"]] = rating


def demo(directory: Path, steps: int = 200) -> None:
    """Planted preference: feature 0 is liked.  Train shared then personal and
    report the native loss against NumPy and held-out pair agreement."""

    schedule, ids = compile_palette_trainer(directory)
    buffers = initial_parameters(ids)
    rng = np.random.default_rng(1)
    a, b = rng.random((PAIRS, FEATURES)), rng.random((PAIRS, FEATURES))
    swap = a[:, 0] < b[:, 0]
    a[swap], b[swap] = b[swap].copy(), a[swap].copy()
    rated = rng.random((RATINGS, FEATURES))
    feedback_batch(buffers, ids, pairs=list(zip(a, b)),
                   ratings=list(zip(rated, rated[:, 0])))
    before = reference_loss(buffers, ids)
    execution = run_parameter_group(schedule, "shared", buffers, steps=1, learning_rate=0.0)
    native = float(execution.buffers[schedule.outputs["loss_0"]])
    print(f"step 0 loss native={native:.15g} numpy={before:.15g}")
    run_parameter_group(schedule, "shared", buffers, steps=steps)
    run_parameter_group(schedule, "personal", buffers, steps=steps // 4)
    after = reference_loss(buffers, ids)
    test_a, test_b = rng.random((256, FEATURES)), rng.random((256, FEATURES))
    v = {name: buffers[ids[name]] for name in ids}
    score = lambda x: (np.tanh(x @ v["shared.w1"] + v["shared.b1"]) @ v["shared.w2"]
                       + v["shared.b2"] + x @ v["personal.u"] * ALPHA)
    agreement = float(np.mean((score(test_a) > score(test_b)).ravel()
                              == (test_a[:, 0] > test_b[:, 0])))
    print(f"loss {before:.6f} -> {after:.6f}; held-out agreement {agreement:.3f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--directory", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=200)
    arguments = parser.parse_args(argv)
    directory = arguments.directory or Path(tempfile.mkdtemp(prefix="palette_native_"))
    if arguments.demo:
        demo(directory, arguments.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
