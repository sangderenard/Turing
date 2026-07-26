"""Reproducible cross-manifold benchmark for learned mesh refinement."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..abstraction import AbstractTensor
from .blackbox_roundtrip_demo import build_blackbox_roundtrip


def run_benchmark_matrix(
    *,
    manifolds: tuple[str, ...],
    phases: tuple[float, ...],
    seeds: tuple[int, ...],
    build_kwargs: dict,
) -> pd.DataFrame:
    """Run every requested held-out case and retain failures in the table."""
    rows = []
    for manifold in manifolds:
        for phase in phases:
            for seed in seeds:
                started = perf_counter()
                result = build_blackbox_roundtrip(
                    manifold=manifold,
                    time_value=phase,
                    training_seed=seed,
                    **build_kwargs,
                )
                summary = result.summary.iloc[0].to_dict()
                training = (
                    result.training.iloc[0].to_dict()
                    if len(result.training) else {}
                )
                row = {
                    "manifold": manifold,
                    "phase": phase,
                    "seed": seed,
                    "wall_sec": perf_counter() - started,
                    **summary,
                    **{
                        f"training_{key}": value
                        for key, value in training.items()
                    },
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _csv_tuple(value: str, cast):
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifolds",
        default="ripple,banana,saddle,twisted_ribbon",
    )
    parser.add_argument("--phases", type=int, default=4)
    parser.add_argument("--seeds", default="1729,2718,31415")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--youngman-resolution", type=int, default=3)
    parser.add_argument("--target-epsilon", type=float, default=0.02)
    parser.add_argument("--tangent-tolerance", type=float, default=1.0)
    parser.add_argument("--max-hinge-angle", type=float, default=0.5)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-triangles", type=int, default=20_000)
    parser.add_argument("--training-examples", type=int, default=7)
    parser.add_argument("--training-epochs", type=int, default=3000)
    parser.add_argument(
        "--training-target",
        choices=("laplace", "discretization", "reconstruction"),
        default="discretization",
    )
    parser.add_argument("--training-backend", default="torch")
    parser.add_argument("--training-device", default="cuda")
    parser.add_argument("--training-dtype", default="float32")
    args = parser.parse_args()

    if args.phases < 1:
        raise ValueError("--phases must be positive")
    manifolds = _csv_tuple(args.manifolds, str)
    seeds = _csv_tuple(args.seeds, int)
    phases = tuple(
        float(value)
        for value in np.linspace(0.0, 1.0, args.phases, endpoint=False)
    )
    AbstractTensor.set_default_backend("numpy", None)
    table = run_benchmark_matrix(
        manifolds=manifolds,
        phases=phases,
        seeds=seeds,
        build_kwargs={
            "youngman_resolution": args.youngman_resolution,
            "position_tolerance": args.target_epsilon,
            "tangent_tolerance": args.tangent_tolerance,
            "max_hinge_angle": args.max_hinge_angle,
            "max_rounds": args.max_rounds,
            "max_triangles": args.max_triangles,
            "training_examples": args.training_examples,
            "training_epochs": args.training_epochs,
            "training_target": args.training_target,
            "training_backend": args.training_backend,
            "training_device": args.training_device,
            "training_dtype": args.training_dtype,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    deployed = int(table["training_alpha_applied"].sum())
    print(table.to_string(index=False))
    print(f"\nDeployed {deployed}/{len(table)} candidates")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
