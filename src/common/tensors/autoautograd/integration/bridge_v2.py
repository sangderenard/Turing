from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Any, Callable, List, Tuple, Dict

import numpy as np

from ..whiteboard_runtime import run_batched_vjp
from ..whiteboard_cache import WhiteboardCache
from ...abstraction import AbstractTensor
from ...linalg import norm as at_norm


@dataclass(frozen=True)
class _Job:
    job_id: str
    op: str
    src_ids: tuple[int, ...]
    residual: float | None
    scale: float
    weight: str | None
    backend_tag: Any = None


def _op_apply_factory(op_name: str) -> Callable[[Any], Any]:
    name = str(op_name).lower()
    def _sum_k(x):
        AT = type(x)
        try:
            return AT.sum(x, dim=0)
        except TypeError:
            return AT.sum(x, axis=0)
    def _mul2(x):
        return x[0] * x[1]
    if name in ("gather", "sum", "add", "sum_k"):
        return _sum_k
    if name in ("mul", "prod", "mul2", "prod_k"):
        return _mul2
    return _sum_k


def _inv_length_scale(sys, out_id: int, src_ids: Sequence[int]) -> float:
    po = sys.nodes[out_id].p
    ws: List[float] = []
    for i in src_ids:
        pi = sys.nodes[i].p
        d = AbstractTensor.tensor(po) - AbstractTensor.tensor(pi)
        n = at_norm(d, dim=-1)
        item = getattr(n, "item_", None)
        n_val = float(item()) if callable(item) else float(n)
        ws.append(1.0 / max(n_val, 1e-8))
    return float(np.mean(ws)) if ws else 1.0


def push_impulses_from_op_v2(
    sys,
    op_name: str,
    src_ids: Sequence[int],
    out_id: int,
    *,
    residual: float | None = None,
    scale: float = 1.0,
    weight: str | None = None,
    cache: WhiteboardCache | None = None,
) -> float:
    """Single op call via batched VJP; preserved for compatibility."""
    if weight == "inv_length":
        scale *= _inv_length_scale(sys, out_id, src_ids)

    job = _Job(
        job_id=f"{op_name}:{tuple(src_ids)}->{out_id}",
        op=str(op_name),
        src_ids=tuple(int(i) for i in src_ids),
        residual=None if residual is None else float(residual),
        scale=float(scale),
        weight=weight,
    )

    def get_attr(i: int):
        return sys.nodes[i].theta

    op_apply = _op_apply_factory(op_name)

    batch = run_batched_vjp(sys=sys, jobs=(job,), op_apply=op_apply, get_attr=get_attr, backend=None)
    y = batch.ys[0]
    grads = batch.grads_per_source[0]
    y_host = float(getattr(y, "item_", lambda: y)()) if hasattr(y, "item_") else float(y)
    if residual is not None:
        for i, g in zip(src_ids, grads):
            g_host = float(getattr(g, "item_", lambda: g)()) if hasattr(g, "item_") else float(g)
            sys.impulse(int(i), int(out_id), op_name, float(scale * g_host * (-float(residual))))
    return y_host


def batched_forward_v2(
    sys,
    specs: Sequence[Tuple[str, Sequence[int], int]],
    *,
    weight: str | None = None,
    scale: float = 1.0,
) -> List[Any]:
    """Forward-only for a list of (op_name, src_ids, out_id) specs, grouped by op."""
    ys_out: List[Any] = []
    by_op: Dict[str, List[Tuple[int, Tuple[int, ...], int]]] = {}
    for idx, (op_name, src_ids, out_id) in enumerate(specs):
        by_op.setdefault(op_name, []).append((idx, tuple(int(i) for i in src_ids), int(out_id)))

    def get_attr(i: int):
        return sys.nodes[i].theta

    ys_buffer: Dict[int, Any] = {}
    for op_name, items in by_op.items():
        op_apply = _op_apply_factory(op_name)
        jobs: List[_Job] = []
        for idx, src_ids, out_id in items:
            sc = scale * (_inv_length_scale(sys, out_id, src_ids) if weight == "inv_length" else 1.0)
            jobs.append(_Job(job_id=f"{op_name}:{src_ids}->{out_id}", op=op_name, src_ids=src_ids, residual=None, scale=sc, weight=weight))
        batch = run_batched_vjp(sys=sys, jobs=jobs, op_apply=op_apply, get_attr=get_attr, backend=None)
        for (idx, _src, _out), y in zip(items, batch.ys):
            ys_buffer[idx] = y
    for i in range(len(specs)):
        ys_out.append(ys_buffer[i])
    return ys_out


def push_impulses_from_ops_batched(
    sys,
    specs: Sequence[Tuple[str, Sequence[int], int]],
    residuals: Sequence[float],
    *,
    weight: str | None = None,
    scale: float = 1.0,
) -> List[Any]:
    """Batched impulse push for a set of homogeneous op groups (grouped by op)."""
    ys_out: List[Any] = [None] * len(specs)
    by_op: Dict[str, List[Tuple[int, Tuple[int, ...], int, float]]] = {}
    for idx, (spec, r) in enumerate(zip(specs, residuals)):
        op_name, src_ids, out_id = spec
        by_op.setdefault(op_name, []).append((idx, tuple(int(i) for i in src_ids), int(out_id), float(r)))

    def get_attr(i: int):
        return sys.nodes[i].theta

    for op_name, items in by_op.items():
        op_apply = _op_apply_factory(op_name)
        jobs: List[_Job] = []
        scales: List[float] = []
        for idx, src_ids, out_id, r in items:
            sc = scale * (_inv_length_scale(sys, out_id, src_ids) if weight == "inv_length" else 1.0)
            scales.append(sc)
            jobs.append(_Job(job_id=f"{op_name}:{src_ids}->{out_id}", op=op_name, src_ids=src_ids, residual=r, scale=sc, weight=weight))
        batch = run_batched_vjp(sys=sys, jobs=jobs, op_apply=op_apply, get_attr=get_attr, backend=None)
        for (idx, src_ids, out_id, r), y, grads, sc in zip(items, batch.ys, batch.grads_per_source, scales):
            ys_out[idx] = y
            for i, g in zip(src_ids, grads):
                g_host = float(getattr(g, "item_", lambda: g)()) if hasattr(g, "item_") else float(g)
                sys.impulse(int(i), int(out_id), op_name, float(sc * g_host * (-float(r))))
    return ys_out
