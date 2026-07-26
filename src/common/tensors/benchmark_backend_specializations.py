"""Compare specialization-level AbstractTensor workloads across backends."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .abstraction import AbstractTensor as AT
from .riemann import abstract_mesh_laplace


@dataclass
class PreparedWorkload:
    name: str
    execute: object
    setup_sec: float


def _neural_projection(backend: str, device, size: int) -> PreparedWorkload:
    rng = np.random.default_rng(1729)
    features = max(8, size // 2)
    outputs = max(4, features // 2)
    x_host = rng.normal(size=(size, features))
    w_host = rng.normal(scale=features ** -0.5, size=(features, outputs))
    bias_host = rng.normal(scale=0.05, size=(1, outputs))
    target_host = rng.uniform(size=(size, outputs))
    started = perf_counter()
    with AT.use_backend(backend, device):
        x = AT.tensor(x_host, dtype="float64", device=device)
        weight = AT.tensor(w_host, dtype="float64", device=device)
        bias = AT.tensor(bias_host, dtype="float64", device=device)
        target = AT.tensor(target_host, dtype="float64", device=device)
    setup_sec = perf_counter() - started

    def execute():
        logits = x @ weight + bias
        prediction = 1.0 / (1.0 + (-logits).exp())
        loss = ((prediction - target) ** 2).mean()
        return AT.cat((prediction.flatten(), loss.reshape(1)), dim=0)

    return PreparedWorkload("neural_projection_loss", execute, setup_sec)


def _pairwise_metric(backend: str, device, size: int) -> PreparedWorkload:
    rng = np.random.default_rng(2718)
    points_host = rng.normal(size=(size, 5))
    started = perf_counter()
    with AT.use_backend(backend, device):
        points = AT.tensor(points_host, dtype="float64", device=device)
    setup_sec = perf_counter() - started

    def execute():
        differences = points.unsqueeze(1) - points.unsqueeze(0)
        squared_distance = (differences * differences).sum(dim=2)
        affinity = (-0.25 * squared_distance).exp()
        normalized = affinity / affinity.sum(dim=1, keepdim=True)
        entropy = -(normalized * (normalized + 1e-12).log()).sum(dim=1)
        return AT.cat(
            (normalized.sum(dim=1), entropy, squared_distance.mean(dim=1)),
            dim=0,
        )

    return PreparedWorkload("pairwise_metric_field", execute, setup_sec)


def _mesh_laplace(backend: str, device, size: int) -> PreparedWorkload:
    side = max(3, int(round(np.sqrt(size))))
    vertices = np.asarray([
        (x, y, 0.15 * np.sin(np.pi * x) * np.sin(np.pi * y))
        for x in np.linspace(0.0, 1.0, side)
        for y in np.linspace(0.0, 1.0, side)
    ])
    triangles = []
    for i in range(side - 1):
        for j in range(side - 1):
            a = i * side + j
            triangles.extend(((a, a + side, a + side + 1),
                              (a, a + side + 1, a + 1)))
    triangle_array = np.asarray(triangles, dtype=np.int64)
    scalar = (
        vertices[:, 0] ** 2
        + vertices[:, 1] ** 2
        + 0.25 * vertices[:, 2]
    )
    started = perf_counter()
    with AT.use_backend(backend, device):
        vertex_tensor = AT.tensor(vertices, dtype="float64", device=device)
        scalar_tensor = AT.tensor(scalar, dtype="float64", device=device)
    setup_sec = perf_counter() - started

    def execute():
        return abstract_mesh_laplace(
            vertex_tensor, triangle_array, scalar_tensor
        )

    return PreparedWorkload("riemann_mesh_laplace", execute, setup_sec)


WORKLOADS = {
    "neural": _neural_projection,
    "metric": _pairwise_metric,
    "laplace": _mesh_laplace,
}


def run_comparison(
    *,
    backends=("numpy", "torch", "c"),
    tasks=("neural", "metric", "laplace"),
    size=32,
    warmup=2,
    repeats=7,
    torch_device="cpu",
) -> pd.DataFrame:
    """Run parity-certified workloads and return one timing row per case."""
    rows = []
    references = {}
    for backend in backends:
        device = torch_device if backend == "torch" else None
        for task in tasks:
            with AT.use_backend(backend, device):
                prepared = WORKLOADS[task](backend, device, size)
                for _ in range(warmup):
                    prepared.execute()
                timings = []
                result = None
                for _ in range(repeats):
                    started = perf_counter()
                    result = prepared.execute()
                    timings.append(perf_counter() - started)
                values = np.asarray(result.tolist(), dtype=np.float64)
            reference = references.setdefault(task, values)
            difference = np.abs(values - reference)
            rows.append({
                "task": prepared.name,
                "task_key": task,
                "backend": backend,
                "device": device or "cpu",
                "size": size,
                "output_values": values.size,
                "setup_sec": prepared.setup_sec,
                "warm_median_sec": float(np.median(timings)),
                "warm_min_sec": float(np.min(timings)),
                "warm_max_sec": float(np.max(timings)),
                "parity_max_abs": float(difference.max(initial=0.0)),
                "parity_rms": float(np.sqrt(np.mean(difference ** 2))),
                "parity_ok": bool(np.allclose(
                    values, reference, rtol=2e-5, atol=2e-7
                )),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="numpy,torch,c")
    parser.add_argument("--tasks", default="neural,metric,laplace")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    table = run_comparison(
        backends=tuple(args.backends.split(",")),
        tasks=tuple(args.tasks.split(",")),
        size=args.size,
        warmup=args.warmup,
        repeats=args.repeats,
        torch_device=args.torch_device,
    )
    print(table.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(args.output.resolve())
    if not table["parity_ok"].all():
        raise RuntimeError("one or more backend parity checks failed")


if __name__ == "__main__":
    main()
