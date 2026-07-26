"""Demonstrate a streaming piecewise 3D-to-N spline and induced metric."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .piecewise import StreamingPiecewiseSplineEngine


_CUBE_VERTICES = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ),
    dtype=np.float64,
)
_CUBE_TETRAHEDRA = np.asarray(
    (
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    ),
    dtype=np.int64,
)


def expanded_embedding(parameters: np.ndarray) -> np.ndarray:
    """A cubic 3D-to-5D map: three visible and two retained hidden channels."""
    u, v, w = np.asarray(parameters, dtype=np.float64).T
    return np.stack(
        (
            u + 0.10 * v * w,
            v + 0.05 * u * w,
            w + 0.08 * u * v,
            0.20 * u * u + 0.10 * v * w,
            0.15 * u * v * w + 0.05 * v * v,
        ),
        axis=1,
    )


def expanded_jacobian(parameters: np.ndarray) -> np.ndarray:
    u, v, w = np.asarray(parameters, dtype=np.float64).T
    jacobian = np.empty((len(parameters), 5, 3), dtype=np.float64)
    jacobian[:, 0, :] = np.stack((np.ones_like(u), 0.10 * w, 0.10 * v), axis=1)
    jacobian[:, 1, :] = np.stack((0.05 * w, np.ones_like(u), 0.05 * u), axis=1)
    jacobian[:, 2, :] = np.stack((0.08 * v, 0.08 * u, np.ones_like(u)), axis=1)
    jacobian[:, 3, :] = np.stack((0.40 * u, 0.10 * w, 0.10 * v), axis=1)
    jacobian[:, 4, :] = np.stack(
        (0.15 * v * w, 0.15 * u * w + 0.10 * v, 0.15 * u * v),
        axis=1,
    )
    return jacobian


def _sample_simplex(
    vertices: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    barycentric = rng.dirichlet(np.ones(len(vertices)), size=count)
    return barycentric @ vertices


def build_piecewise_report(
    samples_per_patch: int = 48,
    validation_per_patch: int = 16,
    fifo_batches: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    simplices = {
        patch_id: _CUBE_VERTICES[indices]
        for patch_id, indices in enumerate(_CUBE_TETRAHEDRA)
    }
    rng = np.random.default_rng(20260725)
    train_parameters = []
    train_patch_ids = []
    validation_parameters = []
    validation_patch_ids = []
    for patch_id, vertices in simplices.items():
        train_parameters.append(_sample_simplex(vertices, samples_per_patch, rng))
        train_patch_ids.append(np.full(samples_per_patch, patch_id, dtype=np.int64))
        validation_parameters.append(
            _sample_simplex(vertices, validation_per_patch, rng)
        )
        validation_patch_ids.append(
            np.full(validation_per_patch, patch_id, dtype=np.int64)
        )

    parameters = np.concatenate(train_parameters)
    patch_ids = np.concatenate(train_patch_ids)
    values = expanded_embedding(parameters)
    jacobians = expanded_jacobian(parameters)
    engine = StreamingPiecewiseSplineEngine(
        simplices,
        degree=3,
        derivative_weight=0.5,
        ridge=0.0,
    )
    for rows in np.array_split(np.arange(len(parameters)), fifo_batches):
        engine.submit(
            patch_ids[rows],
            parameters[rows],
            values[rows],
            jacobians[rows],
        )
    generation = engine.update()
    assert generation is not None

    query = np.concatenate(validation_parameters)
    expected_owner = np.concatenate(validation_patch_ids)
    predicted_owner = generation.locate(query)
    prediction = generation.evaluate(query)
    prediction_jacobian = generation.jacobian(query)
    target = expanded_embedding(query)
    target_jacobian = expanded_jacobian(query)
    target_metric = np.einsum(
        "nmi,nmj->nij", target_jacobian, target_jacobian
    )
    predicted_metric = generation.metric_tensor(query)

    expanded_error = np.linalg.norm(prediction - target, axis=1)
    spatial_error = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=1)
    jacobian_error = np.linalg.norm(
        prediction_jacobian - target_jacobian, axis=(1, 2)
    )
    metric_error = np.linalg.norm(
        predicted_metric - target_metric, axis=(1, 2)
    )

    patch_rows = []
    hidden_metric_norms = []
    for patch_id, patch in sorted(generation.patches.items()):
        mask = expected_owner == patch_id
        full, spatial, hidden = patch.collapsed_metric_components(query[mask])
        hidden_metric_norms.extend(np.linalg.norm(hidden, axis=(1, 2)))
        patch_rows.append(
            {
                "patch_id": patch_id,
                "intrinsic_dimension": patch.intrinsic_dimension,
                "embedding_dimension": patch.embedding_dimension,
                "degree": patch.degree,
                "control_points": patch.control_point_count,
                "validation_points": int(mask.sum()),
                "max_expanded_error": float(expanded_error[mask].max()),
                "max_jacobian_error": float(jacobian_error[mask].max()),
                "max_metric_error": float(metric_error[mask].max()),
                "mean_full_metric_norm": float(
                    np.linalg.norm(full, axis=(1, 2)).mean()
                ),
                "mean_spatial_metric_norm": float(
                    np.linalg.norm(spatial, axis=(1, 2)).mean()
                ),
            }
        )

    summary = pd.DataFrame(
        [
            {
                "generation": generation.generation,
                "patches": len(generation.patches),
                "fifo_batches_consumed": fifo_batches,
                "fifo_batches_pending": engine.pending_batches,
                "source_samples_retained": len(parameters),
                "control_points": generation.control_point_count,
                "intrinsic_dimension": 3,
                "spatial_dimensions": 3,
                "embedding_dimension": target.shape[1],
                "owner_mismatches": int(np.count_nonzero(
                    predicted_owner != expected_owner
                )),
                "mean_expanded_error": float(expanded_error.mean()),
                "max_expanded_error": float(expanded_error.max()),
                "mean_spatial_error": float(spatial_error.mean()),
                "max_spatial_error": float(spatial_error.max()),
                "mean_jacobian_error": float(jacobian_error.mean()),
                "max_jacobian_error": float(jacobian_error.max()),
                "mean_metric_error": float(metric_error.mean()),
                "max_metric_error": float(metric_error.max()),
                "mean_hidden_metric_contribution": float(
                    np.mean(hidden_metric_norms)
                ),
            }
        ]
    )
    return summary, pd.DataFrame(patch_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-patch", type=int, default=48)
    parser.add_argument("--validation-per-patch", type=int, default=16)
    parser.add_argument("--fifo-batches", type=int, default=12)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary, patches = build_piecewise_report(
        samples_per_patch=args.samples_per_patch,
        validation_per_patch=args.validation_per_patch,
        fifo_batches=args.fifo_batches,
    )
    print("\nPIECEWISE STREAM SUMMARY\n", summary.to_string(index=False))
    print("\nPATCH CERTIFICATES\n", patches.to_string(index=False))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_dir / "piecewise_summary.csv", index=False)
        patches.to_csv(args.output_dir / "piecewise_patches.csv", index=False)


if __name__ == "__main__":
    main()
