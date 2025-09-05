from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Any, Callable, List, Tuple, Dict, Optional


from ..whiteboard_runtime import run_batched_vjp
from ..whiteboard_cache import WhiteboardCache
from ...abstraction import AbstractTensor
from ...linalg import norm as at_norm


@dataclass(frozen=True)
class _Job:
    job_id: str
    op: str
    src_ids: tuple[int, ...]
    op_args: Optional[Tuple[Any, ...]]
    op_kwargs: Optional[Dict[str, Any]]
    residual: float | None
    scale: float | None
    weight: str | None
    backend_tag: Any = None


from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

def _normalize_chain(ops: Sequence[str]) -> Tuple[Callable[[Any], Any], ...]:
    """One-time normalization → list[callable]."""
    
    fns = [getattr(AbstractTensor, op, None) for op in ops]
    return tuple(f for f in fns if callable(f))

def _op_apply_factory(
    ops: Sequence[str], args: Optional[Sequence[Dict]] = None
) -> Callable[[Any], Any]:
    """
    Build a tiny, ultra-hot f(x)->y that applies a precompiled chain.
    No getattr/validation in hot path.
    """
    chain = _normalize_chain(ops)
    if not chain:
        def _apply_identity(x): return x
        return _apply_identity

    chain_local = chain  # closure binding
    def _apply(x, _chain=chain_local):
        y = x
        for i, f in enumerate(_chain):
            y = f(y, **(args[i] if args and i < len(args) and isinstance(args[i], dict) else {}))
        return y
    return _apply



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
    return float(AbstractTensor.mean(ws)) if ws else 1.0


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
    op_args: Optional[Tuple[Any, ...]] = None,
    op_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """Single op call via batched VJP; preserved for compatibility."""
    if weight == "inv_length":
        scale *= _inv_length_scale(sys, out_id, src_ids)

    

    job = _Job(
        job_id=f"{op_name}:{tuple(src_ids)}->{out_id}",
        op=str(op_name),
        op_args=tuple(op_args) if op_args is not None else None,
        op_kwargs=dict(op_kwargs) if op_kwargs is not None else None,
        src_ids=tuple(int(i) for i in src_ids),
        residual=None if residual is None else float(residual),
        scale=float(scale),
        weight=weight,
    )

    def get_attr(i: int):
        return sys.nodes[i].theta

    batch = run_batched_vjp(
        sys=sys,
        jobs=(job,),
        op_args=job.op_args or (),
        op_kwargs=job.op_kwargs,
        get_attr=get_attr,
        backend=None,
    )
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
    specs: Sequence[Tuple],
    *,
    weight: str | None = None,
    scale: float = 1.0,
) -> List[Any]:
    """Forward-only for specs of form `(op_name, src_ids, out_id, op_args, op_kwargs)`."""
    ys_out: List[Any] = []
    by_op: Dict[
        Tuple[str, Optional[Tuple[Any, ...]], Optional[Tuple[Tuple[str, Any], ...]]],
        List[Tuple[int, Tuple[int, ...], int, Optional[Tuple[Any, ...]], Optional[Dict[str, Any]]]],
    ] = {}
    for idx, spec in enumerate(specs):
        op_name, src_ids, out_id, op_args, op_kwargs = (*spec, None, None)[:5]
        op_args = tuple(op_args) if isinstance(op_args, (list, tuple)) else None
        key_kwargs: Optional[Tuple[Tuple[str, Any], ...]] = None
        if isinstance(op_kwargs, dict):
            key_kwargs = tuple(sorted((str(k), v) for k, v in op_kwargs.items()))
        key = (str(op_name), op_args, key_kwargs)
        by_op.setdefault(key, []).append(
            (idx, tuple(int(i) for i in src_ids), int(out_id), op_args, op_kwargs if isinstance(op_kwargs, dict) else None)
        )

    def get_attr(i: int):
        return sys.nodes[i].theta

    ys_buffer: Dict[int, Any] = {}
    for (op_name, key_args, key_kwargs), items in by_op.items():
        op_args = key_args or ()
        op_kwargs = {k: v for (k, v) in (key_kwargs or ())} if key_kwargs is not None else None
        jobs: List[_Job] = []
        for idx, src_ids, out_id, _args, _kwargs in items:
            sc = scale * (_inv_length_scale(sys, out_id, src_ids) if weight == "inv_length" else 1.0)
            jobs.append(
                _Job(
                    job_id=f"{op_name}:{src_ids}->{out_id}",
                    op=op_name,
                    src_ids=src_ids,
                    op_args=op_args,
                    op_kwargs=op_kwargs,
                    residual=None,
                    scale=sc,
                    weight=weight,
                )
            )
        batch = run_batched_vjp(
            sys=sys,
            jobs=jobs,
            op_args=op_args,
            op_kwargs=op_kwargs,
            get_attr=get_attr,
            backend=None,
        )
        for (idx, _src, _out, _args, _kwargs), y in zip(items, batch.ys):
            ys_buffer[idx] = y
    for i in range(len(specs)):
        ys_out.append(ys_buffer[i])
    return ys_out


def push_impulses_from_ops_batched(
    sys,
    specs: Sequence[Tuple],
    residuals: Sequence[float],
    *,
    weight: str | None = None,
    scale: float = 1.0,
) -> List[Any]:
    """Batched impulse push for specs `(op_name, src_ids, out_id, op_args, op_kwargs)`."""
    ys_out: List[Any] = [None] * len(specs)
    by_op: Dict[
        Tuple[str, Optional[Tuple[Any, ...]], Optional[Tuple[Tuple[str, Any], ...]]],
        List[Tuple[int, Tuple[int, ...], int, float, Optional[Tuple[Any, ...]], Optional[Dict[str, Any]]]],
    ] = {}
    for idx, (spec, r) in enumerate(zip(specs, residuals)):
        op_name, src_ids, out_id, op_args, op_kwargs = (*spec, None, None)[:5]
        op_args = tuple(op_args) if isinstance(op_args, (list, tuple)) else None
        key_kwargs: Optional[Tuple[Tuple[str, Any], ...]] = None
        if isinstance(op_kwargs, dict):
            key_kwargs = tuple(sorted((str(k), v) for k, v in op_kwargs.items()))
        key = (str(op_name), op_args, key_kwargs)
        by_op.setdefault(key, []).append(
            (idx, tuple(int(i) for i in src_ids), int(out_id), float(r), op_args, op_kwargs if isinstance(op_kwargs, dict) else None)
        )

    def get_attr(i: int):
        return sys.nodes[i].theta

    for (op_name, key_args, key_kwargs), items in by_op.items():
        op_args = key_args or ()
        op_kwargs = {k: v for (k, v) in (key_kwargs or ())} if key_kwargs is not None else None
        jobs: List[_Job] = []
        scales: List[float] = []
        for idx, src_ids, out_id, r, _args, _kwargs in items:
            sc = scale * (_inv_length_scale(sys, out_id, src_ids) if weight == "inv_length" else 1.0)
            scales.append(sc)
            jobs.append(
                _Job(
                    job_id=f"{op_name}:{src_ids}->{out_id}",
                    op=op_name,
                    src_ids=src_ids,
                    op_args=op_args,
                    op_kwargs=op_kwargs,
                    residual=r,
                    scale=sc,
                    weight=weight,
                )
            )
        batch = run_batched_vjp(
            sys=sys,
            jobs=jobs,
            op_args=op_args,
            op_kwargs=op_kwargs,
            get_attr=get_attr,
            backend=None,
        )
        for (idx, src_ids, out_id, r, _args, _kwargs), y, grads, sc in zip(
            items, batch.ys, batch.grads_per_source, scales
        ):
            ys_out[idx] = y
            for i, g in zip(src_ids, grads):
                g_host = float(getattr(g, "item_", lambda: g)()) if hasattr(g, "item_") else float(g)
                sys.impulse(int(i), int(out_id), op_name, float(sc * g_host * (-float(r))))
    return ys_out
