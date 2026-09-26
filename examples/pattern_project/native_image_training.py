"""Continuous stateful LLVM training against a complex image signal.

The model is ordinary ``abstract_nn``: two ``RectConv2d`` layers with ReLU
and sigmoid, trained by MSE.  Its complete forward/loss/ProcessGraph-derived
backward motion is lowered to repository SSA and compiled to a native DLL.
Python owns only persistent buffers, reporting, and PNG snapshots.

Build once and run until interrupted::

    python -m examples.pattern_project.native_image_training --steps 0

Resume later without rebuilding::

    python -m examples.pattern_project.native_image_training --stage run --steps 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.compiler.llvm_training_runtime import (
    NativeParameterGroup,
    compile_native_training_schedule,
    load_training_schedule,
    run_parameter_group,
    save_training_schedule,
)


IMAGE_SIZE = 16
FEATURE_CHANNELS = 4
HIDDEN_CHANNELS = 6
PARAMETER_IDS = (1, 2, 3, 4)
GROUP_NAME = "generator"


def _conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    batch, _channels, height, width = x.shape
    out_channels, _in_channels, kh, kw = weight.shape
    padded = np.pad(x, ((0, 0), (0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2)))
    output = np.empty((batch, out_channels, height, width), dtype=np.float64)
    for n in range(batch):
        for channel in range(out_channels):
            for row in range(height):
                for column in range(width):
                    output[n, channel, row, column] = (
                        bias[channel]
                        + np.sum(
                            padded[n, :, row:row + kh, column:column + kw]
                            * weight[channel]
                        )
                    )
    return output


def image_problem() -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Deterministic textured input and a realizable complex teacher image."""

    axis = np.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    features = np.stack((
        np.sin(3.0 * np.pi * xx + 1.5 * np.pi * yy),
        np.cos(4.0 * np.pi * yy - np.pi * xx),
        np.sin(7.0 * np.pi * radius + 2.0 * xx),
        np.cos(5.0 * np.pi * (xx * yy + 0.25 * xx)),
    ), axis=0)[None, ...]

    teacher = np.random.default_rng(20260815)
    teacher_w1 = teacher.normal(0.0, 0.28, (HIDDEN_CHANNELS, FEATURE_CHANNELS, 3, 3))
    teacher_b1 = teacher.normal(0.0, 0.08, (HIDDEN_CHANNELS,))
    teacher_w2 = teacher.normal(0.0, 0.32, (1, HIDDEN_CHANNELS, 3, 3))
    teacher_b2 = teacher.normal(0.0, 0.05, (1,))
    hidden = np.maximum(_conv2d(features, teacher_w1, teacher_b1), 0.0)
    target = 1.0 / (1.0 + np.exp(-_conv2d(hidden, teacher_w2, teacher_b2)))

    student = np.random.default_rng(1701)
    values = {
        0: features,
        1: student.normal(0.0, 0.12, teacher_w1.shape),
        2: np.zeros_like(teacher_b1),
        3: student.normal(0.0, 0.12, teacher_w2.shape),
        4: np.zeros_like(teacher_b2),
        5: target,
    }
    return values, target


def compile_demo(directory: Path, learning_rate: float) -> None:
    from src.common.tensors.abstract_nn import MSELoss, ReLU, RectConv2d, Sigmoid
    from src.common.tensors.accelerator_backends.ssa_backend import (
        SSATensorOperations,
        SSATensorProgram,
    )

    program = SSATensorProgram("native_complex_image")
    shapes = (
        (1, FEATURE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
        (HIDDEN_CHANNELS, FEATURE_CHANNELS, 3, 3),
        (HIDDEN_CHANNELS,),
        (1, HIDDEN_CHANNELS, 3, 3),
        (1,),
        (1, 1, IMAGE_SIZE, IMAGE_SIZE),
    )
    features, w1, b1, w2, b2, target = [
        SSATensorOperations.input(program, shape) for shape in shapes
    ]
    first = RectConv2d(
        FEATURE_CHANNELS, HIDDEN_CHANNELS, 3, padding=1, like=features,
    )
    second = RectConv2d(
        HIDDEN_CHANNELS, 1, 3, padding=1, like=features,
    )
    first.W, first.b = w1, b1
    second.W, second.b = w2, b2
    prediction = Sigmoid().forward(
        second.forward(ReLU().forward(first.forward(features)))
    )
    loss = MSELoss()(prediction, target)
    prediction_id = int(prediction.data.value.id)
    schedule = compile_native_training_schedule(
        loss,
        bindings={
            "features": features,
            "conv1.weight": w1,
            "conv1.bias": b1,
            "conv2.weight": w2,
            "conv2.bias": b2,
            "target": target,
        },
        parameter_groups=(NativeParameterGroup(
            GROUP_NAME, PARAMETER_IDS, float(learning_rate),
        ),),
        observed_outputs={"prediction": prediction_id},
        name="native_complex_image",
        directory=directory / "artifacts",
    )
    save_training_schedule(schedule, directory / "schedule.json")
    print(
        f"compiled {schedule.name}: groups="
        f"{tuple(group.name for group in schedule.groups)!r}, "
        f"saved_forward_values={schedule.saved_binding_count}"
    )


def _save_state(path: Path, buffers: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **{
        f"value_{parameter_id}": buffers[parameter_id]
        for parameter_id in PARAMETER_IDS
    })
    temporary.replace(path)


def _load_state(path: Path, buffers: dict[int, np.ndarray]) -> bool:
    if not path.is_file():
        return False
    with np.load(path) as stored:
        for parameter_id in PARAMETER_IDS:
            key = f"value_{parameter_id}"
            value = np.asarray(stored[key], dtype=np.float64)
            if value.shape != buffers[parameter_id].shape:
                raise ValueError(
                    f"saved {key} shape {value.shape!r} does not match "
                    f"{buffers[parameter_id].shape!r}"
                )
            buffers[parameter_id] = value.copy()
    return True


def _save_snapshot(
    path: Path, prediction: np.ndarray, target: np.ndarray,
) -> None:
    from PIL import Image

    prediction = np.clip(prediction[0, 0], 0.0, 1.0)
    target = np.clip(target[0, 0], 0.0, 1.0)
    error = np.clip(np.abs(prediction - target) * 2.0, 0.0, 1.0)
    separator = np.full((IMAGE_SIZE, 2), 0.15, dtype=np.float64)
    composite = np.concatenate((
        target, separator, prediction, separator, error,
    ), axis=1)
    pixels = np.asarray(np.rint(composite * 255.0), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="L").resize(
        (pixels.shape[1] * 8, pixels.shape[0] * 8),
        resample=Image.Resampling.NEAREST,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run_demo(
    directory: Path,
    *,
    steps: int,
    native_steps_per_report: int,
    learning_rate: float | None,
    snapshot_every: int,
    reset: bool,
) -> None:
    if steps < 0:
        raise ValueError("steps must be non-negative (zero means indefinite)")
    if native_steps_per_report <= 0 or snapshot_every <= 0:
        raise ValueError("report and snapshot intervals must be positive")
    schedule = load_training_schedule(directory / "schedule.json")
    buffers, target = image_problem()
    state_path = directory / "state.npz"
    resumed = False if reset else _load_state(state_path, buffers)
    print(
        f"native image training: group={GROUP_NAME!r}, resumed={resumed}, "
        f"steps={'indefinite' if steps == 0 else steps}"
    )
    completed = 0
    report = 0
    started = time.perf_counter()
    native_seconds = 0.0
    last_loss = float("nan")
    try:
        while steps == 0 or completed < steps:
            batch = native_steps_per_report
            if steps:
                batch = min(batch, steps - completed)
            native_started = time.perf_counter()
            execution = run_parameter_group(
                schedule,
                GROUP_NAME,
                buffers,
                steps=batch,
                learning_rate=learning_rate,
            )
            native_seconds += time.perf_counter() - native_started
            completed += batch
            report += 1
            last_loss = float(execution.buffers[schedule.outputs["loss_0"]])
            prediction = execution.buffers[schedule.outputs["prediction"]].copy()
            elapsed = max(time.perf_counter() - started, 1e-12)
            print(
                f"step={completed:8d}  loss={last_loss:.12f}  "
                f"native_steps/s={completed / max(native_seconds, 1e-12):10.2f}  "
                f"wall_steps/s={completed / elapsed:10.2f}",
                flush=True,
            )
            _save_state(state_path, buffers)
            _save_snapshot(directory / "snapshots" / "latest.png", prediction, target)
            np.save(directory / "snapshots" / "latest_prediction.npy", prediction)
            if report == 1 or report % snapshot_every == 0:
                _save_snapshot(
                    directory / "snapshots" / f"step_{completed:08d}.png",
                    prediction,
                    target,
                )
    except KeyboardInterrupt:
        print(f"stopped after {completed} native steps")
    finally:
        _save_state(state_path, buffers)
    print(f"final loss={last_loss:.12f}; state={state_path.resolve()}")
    print(
        "snapshot panels are target | prediction | doubled absolute error: "
        f"{(directory / 'snapshots' / 'latest.png').resolve()}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-directory", type=Path,
        default=Path(".turing-cache") / "native-complex-image",
    )
    parser.add_argument(
        "--stage", choices=("all", "compile", "run"), default="all",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="native optimizer steps; zero runs until Ctrl+C",
    )
    parser.add_argument("--native-steps-per-report", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args(argv)
    directory = args.build_directory.resolve()
    if args.stage == "compile":
        compile_demo(directory, args.learning_rate)
        os._exit(0)
    if args.stage == "all":
        subprocess.run([
            sys.executable,
            "-m", "examples.pattern_project.native_image_training",
            "--stage", "compile",
            "--build-directory", str(directory),
            "--learning-rate", str(args.learning_rate),
        ], check=True)
    run_demo(
        directory,
        steps=args.steps,
        native_steps_per_report=args.native_steps_per_report,
        learning_rate=args.learning_rate,
        snapshot_every=args.snapshot_every,
        reset=args.reset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
