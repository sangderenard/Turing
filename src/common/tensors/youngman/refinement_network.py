"""AbstractTensor training over adaptive-mesh certificate data."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..abstraction import AbstractTensor as AT
from ..autograd import GradTape, autograd
from ..autograd_process import AutogradProcess


def triangle_spring_edges(triangles: np.ndarray) -> np.ndarray:
    """Return pairs of triangles sharing an edge for membrane regularization."""
    triangles = np.asarray(triangles, dtype=np.int64)
    owners: dict[tuple[int, int], list[int]] = {}
    for row, triangle in enumerate(triangles):
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            key = (int(min(a, b)), int(max(a, b)))
            owners.setdefault(key, []).append(row)
    pairs = [
        (rows[0], rows[1])
        for rows in owners.values()
        if len(rows) == 2
    ]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def triangle_hinge_angles(
    embedded: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Maximum principal tangent-plane angle at each triangle's hinges."""
    embedded = np.asarray(embedded, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    bases = []
    owners: dict[tuple[int, int], list[int]] = {}
    for row, triangle in enumerate(triangles):
        edges = np.stack((
            embedded[triangle[1]] - embedded[triangle[0]],
            embedded[triangle[2]] - embedded[triangle[0]],
        ), axis=1)
        bases.append(np.linalg.qr(edges)[0][:, :2])
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            owners.setdefault((int(min(a, b)), int(max(a, b))), []).append(row)
    result = np.zeros(len(triangles), dtype=np.float64)
    for rows in owners.values():
        if len(rows) != 2:
            continue
        left, right = rows
        singular = np.linalg.svd(
            bases[left].T @ bases[right], compute_uv=False
        )
        angle = float(np.arccos(np.clip(singular.min(), -1.0, 1.0)))
        result[left] = max(result[left], angle)
        result[right] = max(result[right], angle)
    return result


@dataclass(frozen=True)
class RefinementTrainingResult:
    initial_loss: float
    final_loss: float
    epochs: int
    forward_nodes: int
    backward_nodes: int
    concurrent_forward_width: int
    predictions: np.ndarray
    validation_loss: float
    validation_correlation: float
    baseline_validation_loss: float
    accepted: bool
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weight_hidden: np.ndarray
    bias_hidden: np.ndarray
    weight_output: np.ndarray
    bias_output: np.ndarray
    tensor_dtype: str
    tensor_backend: str | None
    tensor_device: str | None
    tensor_setup_sec: float
    optimization_sec: float
    inference_sec: float
    seed: int

    def predict_alpha(self, features: np.ndarray) -> np.ndarray:
        """Evaluate the learned normalized-error pressure through AbstractTensor."""
        values = (
            np.asarray(features, dtype=np.float64) - self.feature_mean
        ) / self.feature_scale
        backend = self.tensor_backend or AT._preferred_backend or "numpy"
        with AT.use_backend(backend, self.tensor_device):
            x = AT.tensor(values, dtype=self.tensor_dtype)
            hidden = (
                x @ AT.tensor(self.weight_hidden, dtype=self.tensor_dtype)
                + AT.tensor(self.bias_hidden, dtype=self.tensor_dtype)
            ).tanh()
            log_pressure = (
                hidden @ AT.tensor(self.weight_output, dtype=self.tensor_dtype)
                + AT.tensor(self.bias_output, dtype=self.tensor_dtype)
            )
            prediction = np.asarray(
                log_pressure.tolist(), dtype=np.float64
            )[:, 0]
        return np.maximum(np.expm1(prediction), 0.0)


def triangle_refinement_features(
    parameters: np.ndarray,
    triangles: np.ndarray,
    embedded: np.ndarray | None = None,
) -> np.ndarray:
    """Build topology-local features without using source/reference values."""
    vertices = np.asarray(parameters, dtype=np.float64)[
        np.asarray(triangles, dtype=np.int64)
    ]
    centroid = vertices.mean(axis=1)
    edges = np.stack((
        vertices[:, 1] - vertices[:, 0],
        vertices[:, 2] - vertices[:, 1],
        vertices[:, 0] - vertices[:, 2],
    ), axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    area = 0.5 * np.abs(
        edges[:, 0, 0] * (-edges[:, 2, 1])
        - edges[:, 0, 1] * (-edges[:, 2, 0])
    )
    columns = [
        centroid,
        lengths,
        area,
        centroid[:, 0] ** 2,
        centroid[:, 1] ** 2,
        centroid[:, 0] * centroid[:, 1],
        centroid[:, 0] ** 3,
        centroid[:, 1] ** 3,
        centroid[:, 0] ** 2 * centroid[:, 1],
        centroid[:, 0] * centroid[:, 1] ** 2,
        np.ones(len(vertices)),
    ]
    if embedded is not None:
        embedded_vertices = np.asarray(embedded, dtype=np.float64)[
            np.asarray(triangles, dtype=np.int64)
        ]
        embedded_edges = np.stack((
            embedded_vertices[:, 1] - embedded_vertices[:, 0],
            embedded_vertices[:, 2] - embedded_vertices[:, 1],
            embedded_vertices[:, 0] - embedded_vertices[:, 2],
        ), axis=1)
        embedded_lengths = np.linalg.norm(embedded_edges, axis=2)
        gram_00 = np.sum(embedded_edges[:, 0] ** 2, axis=1)
        gram_11 = np.sum(embedded_edges[:, 2] ** 2, axis=1)
        gram_01 = -np.sum(
            embedded_edges[:, 0] * embedded_edges[:, 2], axis=1
        )
        embedded_area = 0.5 * np.sqrt(np.maximum(
            gram_00 * gram_11 - gram_01 ** 2, 0.0
        ))
        quality = (
            4.0 * np.sqrt(3.0) * embedded_area
            / np.maximum(np.sum(embedded_lengths ** 2, axis=1), 1e-15)
        )
        columns.extend((
            embedded_vertices.mean(axis=1),
            embedded_lengths,
            embedded_area,
            quality,
            embedded_lengths.max(axis=1)
            / np.maximum(embedded_lengths.min(axis=1), 1e-15),
            triangle_hinge_angles(embedded, triangles),
        ))
    return np.column_stack(columns)


def train_refinement_predictor(
    features: np.ndarray,
    certificate_error: np.ndarray,
    *,
    epsilon: float,
    epochs: int = 80,
    learning_rate: float = 0.03,
    validation_fraction: float = 0.2,
    required_correlation: float = 0.5,
    hidden_dimension: int = 24,
    group_ids: np.ndarray | None = None,
    spring_edges: np.ndarray | None = None,
    spring_strength: float = 0.0,
    tensor_dtype: str = "float64",
    seed: int = 1729,
) -> RefinementTrainingResult:
    """Train an AbstractTensor linear neuron to predict log error pressure.

    The training uses Turing's tape reverse mode and ``AutogradProcess``. Its
    graph/schedule export is part of the acceptance result, not a silent
    best-effort side path.
    """
    features = np.asarray(features, dtype=np.float64)
    errors = np.asarray(certificate_error, dtype=np.float64)
    if features.ndim != 2 or errors.shape != (len(features),):
        raise ValueError("features/errors batch shapes do not agree")
    if epsilon <= 0.0 or epochs < 1:
        raise ValueError("epsilon and epochs must be positive")
    target = np.log1p(np.maximum(errors, 0.0) / epsilon)[:, None]
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and 0.5")
    if group_ids is None:
        validation_stride = max(2, int(round(1.0 / validation_fraction)))
        validation_mask = np.arange(len(features)) % validation_stride == 0
    else:
        group_ids = np.asarray(group_ids)
        if group_ids.shape != (len(features),):
            raise ValueError("group_ids must contain one group per row")
        groups = np.unique(group_ids)
        if len(groups) < 2:
            raise ValueError("grouped validation needs at least two groups")
        holdout_count = max(1, int(round(len(groups) * validation_fraction)))
        validation_mask = np.isin(group_ids, groups[-holdout_count:])
    training_mask = ~validation_mask
    if training_mask.sum() < 2 or validation_mask.sum() < 2:
        raise ValueError("need enough rows for training and validation")

    feature_mean = features[training_mask].mean(axis=0)
    feature_scale = np.maximum(features[training_mask].std(axis=0), 1e-8)
    feature_mean[-1] = 0.0
    feature_scale[-1] = 1.0
    normalized = (features - feature_mean) / feature_scale
    fit_target = target.copy()
    if spring_edges is not None and spring_strength > 0.0:
        edges = np.asarray(spring_edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("spring_edges must have shape (E, 2)")
        # Relax only supervised labels over their triangle adjacency graph.
        # Held-out labels are untouched and never enter this membrane solve.
        usable = training_mask[edges[:, 0]] & training_mask[edges[:, 1]]
        edges = edges[usable]
        for _ in range(4):
            neighbor_sum = np.zeros(len(features), dtype=np.float64)
            degree = np.zeros(len(features), dtype=np.float64)
            np.add.at(neighbor_sum, edges[:, 0], fit_target[edges[:, 1], 0])
            np.add.at(neighbor_sum, edges[:, 1], fit_target[edges[:, 0], 0])
            np.add.at(degree, edges[:, 0], 1.0)
            np.add.at(degree, edges[:, 1], 1.0)
            active = training_mask & (degree > 0.0)
            neighbor_mean = np.zeros(len(features), dtype=np.float64)
            neighbor_mean[active] = neighbor_sum[active] / degree[active]
            fit_target[active, 0] = (
                fit_target[active, 0]
                + spring_strength * neighbor_mean[active]
            ) / (1.0 + spring_strength)
    previous_tape = autograd.tape
    autograd.tape = GradTape()
    try:
        setup_started = perf_counter()
        x = AT.tensor(normalized, dtype=tensor_dtype)
        x_train = AT.tensor(normalized[training_mask], dtype=tensor_dtype)
        y_train = AT.tensor(fit_target[training_mask], dtype=tensor_dtype)
        rng = np.random.default_rng(seed)
        weight_hidden = AT.tensor(
            rng.normal(
                0.0, 0.15, size=(features.shape[1], hidden_dimension)
            ),
            dtype=tensor_dtype,
        )
        bias_hidden = AT.tensor(
            np.zeros((hidden_dimension,), dtype=np.float64),
            dtype=tensor_dtype,
        )
        weight_output = AT.tensor(
            rng.normal(0.0, 0.1, size=(hidden_dimension, 1)),
            dtype=tensor_dtype,
        )
        bias_output = AT.tensor(
            np.zeros((1,), dtype=np.float64), dtype=tensor_dtype
        )
        params = (weight_hidden, bias_hidden, weight_output, bias_output)
        for parameter in params:
            parameter.requires_grad_(True)
        tensor_setup_sec = perf_counter() - setup_started

        def predict(values):
            hidden = (values @ weight_hidden + bias_hidden).tanh()
            return hidden @ weight_output + bias_output

        def loss_fn():
            prediction = predict(x_train)
            difference = prediction - y_train
            loss = (difference * difference).mean()
            return loss, float(loss.item())

        process = AutogradProcess(autograd.tape)
        optimization_started = perf_counter()
        process.training_loop(
            loss_fn, params, steps=epochs, lr=learning_rate
        )
        optimization_sec = perf_counter() - optimization_started
        inference_started = perf_counter()
        prediction = np.asarray(predict(x).tolist(), dtype=np.float64)
        inference_sec = perf_counter() - inference_started
        validation_target = target[validation_mask, 0]
        validation_prediction = prediction[validation_mask, 0]
        validation_loss = float(np.mean(
            (validation_prediction - validation_target) ** 2
        ))
        baseline = float(np.mean(
            (validation_target - target[training_mask, 0].mean()) ** 2
        ))
        correlation = float(np.corrcoef(
            validation_prediction, validation_target
        )[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
        accepted = (
            validation_loss < baseline
            and correlation >= required_correlation
            and process.training_log[-1]["loss"] < process.training_log[0]["loss"]
        )
        levels = {}
        for _, data in process.forward_graph.nodes(data=True):
            levels[data.get("level", -1)] = levels.get(data.get("level", -1), 0) + 1
        return RefinementTrainingResult(
            initial_loss=process.training_log[0]["loss"],
            final_loss=process.training_log[-1]["loss"],
            epochs=epochs,
            forward_nodes=len(process.forward_graph),
            backward_nodes=len(process.backward_graph),
            concurrent_forward_width=max(levels.values(), default=0),
            predictions=prediction[:, 0],
            validation_loss=validation_loss,
            validation_correlation=correlation,
            baseline_validation_loss=baseline,
            accepted=accepted,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            weight_hidden=np.asarray(weight_hidden.tolist()),
            bias_hidden=np.asarray(bias_hidden.tolist()),
            weight_output=np.asarray(weight_output.tolist()),
            bias_output=np.asarray(bias_output.tolist()),
            tensor_dtype=tensor_dtype,
            tensor_backend=AT._preferred_backend,
            tensor_device=(
                None
                if AT._preferred_device is None
                else str(AT._preferred_device)
            ),
            tensor_setup_sec=tensor_setup_sec,
            optimization_sec=optimization_sec,
            inference_sec=inference_sec,
            seed=seed,
        )
    finally:
        autograd.tape = previous_tape
