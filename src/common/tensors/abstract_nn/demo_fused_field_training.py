"""Live proof that an AbstractNN FusedProgram learns a moving field.

Run:
    python -m src.common.tensors.abstract_nn.demo_fused_field_training --live
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from ..abstraction import AbstractTensor as AT
from ..autograd import GradTape, autograd
from ..autograd_process import AutogradProcess
from . import (
    Identity,
    Linear,
    MSELoss,
    ProgramRunner,
    Sequential,
    Tanh,
    capture_backward_program,
    capture_forward_program,
)
from .utils import set_seed


def field_features(xy: np.ndarray, phase: np.ndarray | float) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    phase = np.broadcast_to(np.asarray(phase, dtype=np.float64), (len(xy),))
    x, y = xy.T
    wave = np.sin(2.4 * x + phase) * np.cos(2.1 * y - 0.7 * phase)
    rings = np.sin(3.6 * (x * x + y * y) - 1.3 * phase)
    shear = np.cos(3.2 * x - 2.7 * y + 0.4 * phase)
    return np.column_stack(
        (x, y, x * y, np.sin(phase), np.cos(phase), wave, rings, shear)
    )


def target_field(xy: np.ndarray, phase: np.ndarray | float) -> np.ndarray:
    """Animated interference/ring field used as independently computed truth."""
    x, y = np.asarray(xy, dtype=np.float64).T
    phase = np.broadcast_to(np.asarray(phase, dtype=np.float64), x.shape)
    wave = np.sin(2.4 * x + phase) * np.cos(2.1 * y - 0.7 * phase)
    rings = np.sin(3.6 * (x * x + y * y) - 1.3 * phase)
    shear = np.cos(3.2 * x - 2.7 * y + 0.4 * phase)
    value = np.tanh(0.9 * wave + 0.5 * rings + 0.35 * shear)
    value += 0.08 * wave * shear
    return value[:, None]


def make_training_set(resolution: int, phases: int):
    axis = np.linspace(-1.0, 1.0, resolution)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    inputs, targets = [], []
    for phase in np.linspace(0.0, 2.0 * np.pi, phases, endpoint=False):
        inputs.append(field_features(xy, phase))
        targets.append(target_field(xy, phase))
    return np.concatenate(inputs), np.concatenate(targets)


def _load_pluck_viewer():
    root = Path(__file__).resolve().parents[5]
    pluck = root / "spectral-analyzer"
    if str(pluck) not in sys.path:
        sys.path.insert(0, str(pluck))
    import ordinary_gl_mesh_viewer

    return ordinary_gl_mesh_viewer


def run_demo(
    *,
    epochs: int = 400,
    learning_rate: float = 0.03,
    hidden: int = 32,
    training_resolution: int = 18,
    training_phases: int = 8,
    display_resolution: int = 96,
    backend: str = "numpy",
    live: bool = False,
    render_every: int = 20,
    output: str | Path | None = "abstract_nn_fused_field.png",
    capture_backward: bool = False,
):
    selected_backend = "torch" if backend == "torch-cuda" else backend
    selected_device = "cuda" if backend == "torch-cuda" else None
    previous_tape = autograd.tape
    autograd.tape = GradTape()
    output_path = None if output is None else Path(output)
    dashboard = None
    try:
        if live or output_path is not None:
            dashboard = _load_pluck_viewer().HeatmapDashboard(
                title="AbstractTensor / AbstractNN / FusedProgram — Pluck OpenGL"
            )
        with AT.use_backend(selected_backend, selected_device):
            x_values, y_values = make_training_set(
                training_resolution, training_phases
            )
            x_train = AT.tensor(x_values, dtype="float64")
            y_train = AT.tensor(y_values, dtype="float64")
            set_seed(1729)
            model = Sequential(
                [
                    Linear(x_values.shape[1], hidden, like=x_train, init="xavier"),
                    Linear(hidden, hidden, like=x_train, init="xavier"),
                    Linear(hidden, 1, like=x_train, init="xavier"),
                ],
                [Tanh(), Tanh(), Identity()],
            )
            params = tuple(model.parameters())
            program, input_id = capture_forward_program(model, x_train)
            parameter_feeds = {id(parameter): parameter for parameter in params}
            unexplained = set(program.feeds) - {input_id} - set(parameter_feeds)
            if unexplained:
                raise RuntimeError(f"unexplained FusedProgram feeds: {unexplained}")
            runner = ProgramRunner(program)
            loss_module = MSELoss()
            backward_capture = None

            def predict(values):
                return runner(
                    {**parameter_feeds, input_id: values}, training=False
                )["prediction"]

            def loss_fn():
                loss = loss_module(predict(x_train), y_train)
                return loss, float(loss.item())

            if capture_backward:
                capture_loss = loss_module(predict(x_train), y_train)
                backward_capture = capture_backward_program(
                    capture_loss, params
                )
            process = AutogradProcess(autograd.tape)
            axis = np.linspace(-1.0, 1.0, display_resolution)
            xx, yy = np.meshgrid(axis, axis, indexing="xy")
            display_xy = np.column_stack((xx.ravel(), yy.ravel()))
            completed = 0
            display_open = True
            while completed < epochs:
                chunk = min(render_every if live else epochs, epochs - completed)
                process.training_loop(
                    loss_fn, params, steps=chunk, lr=learning_rate
                )
                completed += chunk
                phase = 2.0 * np.pi * completed / max(epochs, 1)
                display_input = AT.tensor(
                    field_features(display_xy, phase), dtype="float64"
                )
                with autograd.no_grad():
                    prediction = np.asarray(
                        predict(display_input).tolist(), dtype=np.float64
                    )[:, 0]
                target = target_field(display_xy, phase)[:, 0]
                losses = [row["loss"] for row in process.training_log]
                error = np.abs(prediction - target)
                if dashboard is not None:
                    dashboard.update(
                        target.reshape(display_resolution, display_resolution),
                        prediction.reshape(
                            display_resolution, display_resolution
                        ),
                        error.reshape(display_resolution, display_resolution),
                        losses,
                        status_lines=(
                            f"epoch {completed}/{epochs}  backend {backend}",
                            f"max error {error.max():.4g}",
                        ),
                    )
                    display_open = dashboard.pump()
                print(
                    f"\repoch {completed:5d}/{epochs}  "
                    f"loss {losses[-1]:.6g}  "
                    f"program {len(program.steps)} steps  "
                    f"max error {np.max(np.abs(prediction-target)):.4g}",
                    end="",
                    flush=True,
                )
                if not display_open:
                    break
            print()
            if dashboard is not None and output_path is not None and display_open:
                dashboard.save(output_path)
            if live and dashboard is not None and display_open:
                dashboard.wait()
            return {
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "program_steps": len(program.steps),
                "backward_program_steps": (
                    len(backward_capture.program.steps)
                    if backward_capture is not None
                    else 0
                ),
                "missing_backward": (
                    backward_capture.missing_backward
                    if backward_capture is not None
                    else ()
                ),
                "forward_nodes": len(process.forward_graph),
                "backward_nodes": len(process.backward_graph),
                "output": output_path,
            }
    finally:
        if dashboard is not None:
            dashboard.close()
        autograd.tape = previous_tape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--training-resolution", type=int, default=18)
    parser.add_argument("--training-phases", type=int, default=8)
    parser.add_argument("--display-resolution", type=int, default=96)
    parser.add_argument(
        "--backend",
        choices=("numpy", "c", "torch", "torch-cuda", "glsl", "nodus"),
        default="numpy",
    )
    parser.add_argument("--render-every", type=int, default=20)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", default="abstract_nn_fused_field.png")
    parser.add_argument("--capture-backward", action="store_true")
    args = parser.parse_args()
    result = run_demo(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden=args.hidden,
        training_resolution=args.training_resolution,
        training_phases=args.training_phases,
        display_resolution=args.display_resolution,
        backend=args.backend,
        live=args.live,
        render_every=args.render_every,
        output=args.output,
        capture_backward=args.capture_backward,
    )
    print(result)


if __name__ == "__main__":
    main()
