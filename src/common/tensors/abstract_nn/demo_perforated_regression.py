"""Pit an Adam-trained linear model against an AbstractTensor dendritic model.

Examples::

    python -m src.common.tensors.abstract_nn.demo_perforated_regression
    python -m src.common.tensors.abstract_nn.demo_perforated_regression --backend torch-cuda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
from typing import Sequence

import numpy as np

from ..abstraction import AbstractTensor as AT
from ..autograd import GradTape, autograd
from .core import Linear
from .optimizer import Adam
from .perforated import PerforatedLinear


@dataclass(frozen=True)
class ComparisonResult:
    backend: str
    ordinary_initial_loss: float
    ordinary_final_loss: float
    perforated_initial_loss: float
    perforated_final_loss: float
    ordinary_parameters: int
    perforated_parameters: int
    phase_losses: tuple[float, float, float]

    @property
    def improvement_ratio(self) -> float:
        return self.ordinary_final_loss / max(self.perforated_final_loss, 1e-30)


def make_regression_data(samples: int, in_dim: int, out_dim: int, seed: int):
    """Make independent train/test data from a dendrite-shaped ground truth."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, in_dim))
    linear_w = rng.normal(0.0, 0.7, size=(in_dim, out_dim))
    linear_b = rng.normal(0.0, 0.1, size=(1, out_dim))
    branch_w = rng.normal(0.0, 1.2, size=(in_dim, out_dim * 2))
    branch_b = rng.normal(0.0, 0.3, size=(1, out_dim * 2))
    branch_gain = rng.normal(0.0, 0.9, size=(1, out_dim, 2))
    branches = np.tanh(x @ branch_w + branch_b).reshape(samples, out_dim, 2)
    y = x @ linear_w + linear_b + np.sum(branches * branch_gain, axis=2)
    return x.astype(np.float64), y.astype(np.float64)


def _loss(model, x: AT, y: AT) -> AT:
    error = model.forward(x) - y
    return (error * error).mean()


def _train_phase(
    model,
    active_parameters: Sequence[AT],
    x: AT,
    y: AT,
    *,
    steps: int,
    learning_rate: float,
) -> float:
    parameters = list(active_parameters)
    optimizer = Adam(parameters, lr=learning_rate)
    loss_value = float(_loss(model, x, y).item())
    for _ in range(steps):
        autograd.tape = GradTape()
        autograd.tape.create_tensor_node(x)
        autograd.tape.create_tensor_node(y)
        for parameter in model.parameters():
            parameter.zero_grad()
        loss = _loss(model, x, y)
        gradients = list(
            autograd.grad(loss, parameters, retain_graph=False, allow_unused=False)
        )
        updated = optimizer.step(parameters, gradients)
        for parameter, new_parameter in zip(parameters, updated):
            AT.copyto(parameter, new_parameter)
        loss_value = float(loss.item())
    return loss_value


def _evaluate(model, x: AT, y: AT) -> float:
    with autograd.no_grad():
        return float(_loss(model, x, y).item())


def _parameter_count(parameters: Sequence[AT]) -> int:
    return sum(int(np.prod(parameter.shape)) for parameter in parameters)


def _automatic_backend() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "torch-cuda"
    except Exception:
        pass
    return "numpy"


def run_comparison(
    *,
    backend: str = "auto",
    samples: int = 256,
    in_dim: int = 4,
    out_dim: int = 3,
    base_steps: int = 120,
    dendrite_steps: int = 240,
    consolidate_steps: int = 80,
    learning_rate: float = 0.025,
    seed: int = 1729,
) -> ComparisonResult:
    """Run equal-step ordinary/perforated fits and report held-out MSE."""
    backend = _automatic_backend() if backend == "auto" else backend
    selected_backend = "torch" if backend == "torch-cuda" else backend
    device = "cuda" if backend == "torch-cuda" else "cpu"
    previous_tape = autograd.tape
    autograd.tape = GradTape()
    try:
        with AT.use_backend(selected_backend, device):
            x_values, y_values = make_regression_data(
                samples + samples // 2, in_dim, out_dim, seed
            )
            x_train = AT.tensor(x_values[:samples], dtype="float64")
            y_train = AT.tensor(y_values[:samples], dtype="float64")
            x_test = AT.tensor(x_values[samples:], dtype="float64")
            y_test = AT.tensor(y_values[samples:], dtype="float64")

            random.seed(seed)
            ordinary = Linear(in_dim, out_dim, like=x_train, init="xavier")
            random.seed(seed)
            perforated = PerforatedLinear(
                in_dim,
                out_dim,
                like=x_train,
                dendrites_per_neuron=2,
                init="xavier",
                active=False,
            )
            # Resetting Python's seed before each constructor gives both
            # contestants the same point-neuron initialization while keeping
            # their storage independent.  ``copyto`` is intentionally avoided
            # here because some eager backends may preserve source storage.

            ordinary_initial = _evaluate(ordinary, x_test, y_test)
            perforated_initial = _evaluate(perforated, x_test, y_test)
            total_steps = base_steps + dendrite_steps + consolidate_steps
            _train_phase(
                ordinary,
                ordinary.parameters(),
                x_train,
                y_train,
                steps=total_steps,
                learning_rate=learning_rate,
            )

            base_loss = _train_phase(
                perforated,
                perforated.base_parameters(),
                x_train,
                y_train,
                steps=base_steps,
                learning_rate=learning_rate,
            )
            perforated.perforate(True)
            dendrite_loss = _train_phase(
                perforated,
                perforated.dendrite_parameters(),
                x_train,
                y_train,
                steps=dendrite_steps,
                learning_rate=learning_rate,
            )
            consolidate_loss = _train_phase(
                perforated,
                perforated.base_parameters(),
                x_train,
                y_train,
                steps=consolidate_steps,
                learning_rate=learning_rate * 0.5,
            )
            return ComparisonResult(
                backend=backend,
                ordinary_initial_loss=ordinary_initial,
                ordinary_final_loss=_evaluate(ordinary, x_test, y_test),
                perforated_initial_loss=perforated_initial,
                perforated_final_loss=_evaluate(perforated, x_test, y_test),
                ordinary_parameters=_parameter_count(ordinary.parameters()),
                perforated_parameters=_parameter_count(perforated.parameters()),
                phase_losses=(base_loss, dendrite_loss, consolidate_loss),
            )
    finally:
        autograd.tape = previous_tape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("auto", "numpy", "torch", "torch-cuda"), default="auto"
    )
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--base-steps", type=int, default=120)
    parser.add_argument("--dendrite-steps", type=int, default=240)
    parser.add_argument("--consolidate-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    result = run_comparison(
        backend=args.backend,
        samples=args.samples,
        base_steps=args.base_steps,
        dendrite_steps=args.dendrite_steps,
        consolidate_steps=args.consolidate_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(f"backend: {result.backend}")
    print(
        f"ordinary:  {result.ordinary_initial_loss:.6g} -> "
        f"{result.ordinary_final_loss:.6g} ({result.ordinary_parameters} parameters)"
    )
    print(
        f"perforated: {result.perforated_initial_loss:.6g} -> "
        f"{result.perforated_final_loss:.6g} ({result.perforated_parameters} parameters)"
    )
    print(
        "perforated phase losses (base, dendrite, consolidate): "
        + ", ".join(f"{value:.6g}" for value in result.phase_losses)
    )
    print(f"held-out improvement: {result.improvement_ratio:.2f}x")


if __name__ == "__main__":
    main()
