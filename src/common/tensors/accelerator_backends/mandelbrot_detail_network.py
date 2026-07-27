"""AbstractNN controller that learns which complex-dream states have detail.

The fractal remains the source of truth. A small AbstractNN program is trained
on measurements from many low-resolution AbstractTensor solves, then predicts
detail cheaply enough to steer a live, high-resolution GLSL tour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..abstraction import AbstractTensor as AT
from ..abstract_nn import (
    Adam,
    Identity,
    Linear,
    MSELoss,
    Sequential,
    Tanh,
)
from ..abstract_nn.utils import set_seed
from ..autograd import GradTape, autograd


def detail_scores(fields: np.ndarray, iterations: int) -> np.ndarray:
    """Measure useful spatial structure without preferring a specific image."""
    values = np.asarray(fields, dtype=np.float64) / max(float(iterations), 1.0)
    if values.ndim != 3:
        raise ValueError("detail fields must have shape (samples, height, width)")

    dx = np.abs(np.diff(values, axis=2)).mean(axis=(1, 2))
    dy = np.abs(np.diff(values, axis=1)).mean(axis=(1, 2))
    edge = 1.0 - np.exp(-10.0 * (dx + dy))
    contrast = 1.0 - np.exp(-8.0 * values.std(axis=(1, 2)))

    interior = (values >= 1.0).mean(axis=(1, 2))
    balance = 1.0 - np.abs(2.0 * interior - 1.0)

    entropies = []
    for field in values:
        histogram = np.histogram(field, bins=16, range=(0.0, 1.0))[0]
        probabilities = histogram[histogram > 0] / histogram.sum()
        entropy = -np.sum(probabilities * np.log(probabilities))
        entropies.append(entropy / np.log(16.0))
    entropy = np.asarray(entropies)
    return np.clip(
        0.35 * edge + 0.25 * contrast + 0.25 * entropy + 0.15 * balance,
        0.0,
        1.0,
    )


def dream_features(
    travel: np.ndarray,
    bass: np.ndarray,
    low_mid: np.ndarray,
    high_mid: np.ndarray,
    span: np.ndarray,
    family_mix: np.ndarray,
) -> np.ndarray:
    """State features available before committing a full-resolution solve."""
    travel = np.asarray(travel, dtype=np.float64)
    return np.column_stack((
        np.sin(0.24 * travel),
        np.cos(0.24 * travel),
        np.sin(0.71 * travel),
        np.cos(0.71 * travel),
        np.sin(1.93 * travel),
        np.cos(1.93 * travel),
        np.asarray(bass, dtype=np.float64),
        np.asarray(low_mid, dtype=np.float64),
        np.asarray(high_mid, dtype=np.float64),
        np.log(np.maximum(np.asarray(span, dtype=np.float64), 1e-15)),
        np.asarray(family_mix, dtype=np.float64),
    ))


@dataclass(frozen=True)
class DetailController:
    """Trained AbstractNN predictor plus training evidence."""

    model: object
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    initial_loss: float
    final_loss: float
    validation_correlation: float
    samples: int
    epochs: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (
            np.asarray(features, dtype=np.float64) - self.feature_mean
        ) / self.feature_scale
        with AT.use_backend("numpy"):
            with autograd.no_grad():
                output = self.model.forward(
                    AT.tensor(normalized, dtype="float64")
                )
        return np.clip(
            np.asarray(output.tolist(), dtype=np.float64)[:, 0], 0.0, 1.0
        )


def train_detail_controller(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    hidden: int = 16,
    epochs: int = 120,
    learning_rate: float = 0.025,
    seed: int = 1947,
) -> DetailController:
    """Train and freeze a tiny AbstractNN/FusedProgram detail predictor."""
    features = np.asarray(features, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    if features.ndim != 2 or len(features) != len(scores) or len(features) < 10:
        raise ValueError("detail training needs at least ten feature/score rows")

    validation = np.arange(len(features)) % 5 == 0
    training = ~validation
    mean = features[training].mean(axis=0)
    scale = np.maximum(features[training].std(axis=0), 1e-8)
    normalized = (features - mean) / scale

    previous_tape = autograd.tape
    autograd.tape = GradTape()
    try:
        with AT.use_backend("numpy"):
            x_train = AT.tensor(normalized[training], dtype="float64")
            y_train = AT.tensor(scores[training], dtype="float64")
            set_seed(seed)
            model = Sequential(
                [
                    Linear(
                        features.shape[1], hidden, like=x_train,
                        init="xavier", _label_prefix="detail",
                    ),
                    Linear(
                        hidden, 1, like=x_train,
                        init="xavier", _label_prefix="detail",
                    ),
                ],
                [Tanh(), Identity()],
            )
            params = list(model.parameters())
            optimizer = Adam(params, lr=learning_rate)
            loss_module = MSELoss()
            losses = []
            for _ in range(epochs):
                autograd.tape._nodes.clear()
                autograd.tape.graph.clear()
                for parameter in params:
                    autograd.tape.create_tensor_node(parameter)
                prediction = model.forward(x_train)
                loss = loss_module.forward(prediction, y_train)
                losses.append(float(loss.item()))
                gradients = autograd.grad(loss, params, retain_graph=False)
                replacements = optimizer.step(params, gradients)
                with autograd.no_grad():
                    for parameter, replacement in zip(params, replacements):
                        AT.copyto(parameter, replacement)

        controller = DetailController(
            model=model,
            feature_mean=mean,
            feature_scale=scale,
            initial_loss=losses[0],
            final_loss=losses[-1],
            validation_correlation=0.0,
            samples=len(features),
            epochs=epochs,
        )
        prediction = controller.predict(features[validation])
        target_validation = scores[validation, 0]
        if (
            np.std(prediction) > 1e-12
            and np.std(target_validation) > 1e-12
        ):
            correlation = float(np.corrcoef(
                prediction, target_validation
            )[0, 1])
        else:
            correlation = 0.0
        return DetailController(
            **{
                **controller.__dict__,
                "validation_correlation": correlation,
            }
        )
    finally:
        autograd.tape = previous_tape


__all__ = [
    "DetailController",
    "detail_scores",
    "dream_features",
    "train_detail_controller",
]
