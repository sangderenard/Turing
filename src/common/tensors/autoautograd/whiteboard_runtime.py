from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .whiteboard_cache import WhiteboardCache

from ..autograd import autograd, GradTape
from .node_tensor import NodeAttrView

@contextmanager
def _tape():
    old = autograd.tape
    autograd.tape = GradTape()
    try:
        yield
    finally:
        autograd.tape = old

@dataclass(frozen=True)
class BatchSlices:
    index_of: Dict[str, int]
    job_ids: Tuple[str, ...]

@dataclass(frozen=True)
class BatchVJPResult:
    slices: BatchSlices
    ys: Tuple[Any, ...]
    grads_full: Tuple[Any, ...]
    grads_per_source: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class _WBJob:
    job_id: str
    op: str
    src_ids: Tuple[int, ...]
    residual: Optional[float]

def _residual_like(y: Any, residual: Optional[Any], backend: Any | None) -> Optional[Any]:
    """Ensure residual is backend-typed/broadcastable to y."""
    if residual is None:
        return None
    if backend is not None and hasattr(backend, "asarray"):
        try:
            return backend.asarray(residual)
        except Exception:
            pass
    return residual

def _reduce_per_source(g: Any) -> Tuple[float, ...]:
    """Sum grad over feature axes; keep source axis (k,)."""
    ndim = getattr(g, "ndim", 1)
    if ndim <= 1:
        return tuple(float(gi) for gi in g)
    axes = tuple(range(1, ndim))
    gk = g.sum(dim=axes)
    return tuple(float(gi) for gi in gk)


def run_op_and_grads_cached(
    sys: Any,
    op_name: str,
    src_ids: Sequence[int],
    *,
    scale: float = 1.0,
    residual: Optional[float] = None,
    weight: Optional[str] = None,
    cache: Optional[WhiteboardCache] = None,
    backend: Any | None = None,
    backend_tag: Any | None = None,
    op_args: Tuple[Any, ...] = (),
    op_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Tuple[float, ...]]:
    """Convenience wrapper: run single op with caching."""
    cache = cache or WhiteboardCache()
    versions = [int(getattr(sys.nodes[i], "version", 0)) for i in src_ids]
    sample = sys.nodes[src_ids[0]].sphere
    feat_shape = getattr(sample, "shape", ())  # full vector shape drives cache binning
    if backend_tag is None and backend is not None:
        backend_tag = getattr(backend, "name", None) or getattr(backend, "__name__", None) or id(backend)
    key = cache.make_key(
        op_name=op_name,
        src_ids=src_ids,
        versions=versions,
        feat_shape=feat_shape if isinstance(feat_shape, tuple) else (),
        weight=weight,
        scale=scale,
        residual=residual,
        backend_tag=backend_tag,
    )
    hit = cache.get(key)
    if hit is not None:
        return hit

    vals = [sys.nodes[i].sphere for i in src_ids]
    if op_name == "add" and len(vals) == 2 and not op_args and not op_kwargs:
        y = vals[0] + vals[1]
        grads = (1.0, 1.0)
    elif op_name == "mul" and len(vals) == 2 and not op_args and not op_kwargs:
        y = vals[0] * vals[1]
        grads = (float(vals[1]), float(vals[0]))
    else:
        job = _WBJob(
            job_id=f"{op_name}:{tuple(src_ids)}",
            op=op_name,
            src_ids=tuple(int(i) for i in src_ids),
            residual=residual,
        )

        batch = run_batched_vjp(
            sys=sys,
            jobs=(job,),
            op_args=op_args,
            op_kwargs=op_kwargs,
            backend=backend,
        )
        y = batch.ys[0]
        grads = batch.grads_per_source[0]
    cache.put(key, (y, grads))
    return y, grads

def run_batched_vjp(
    *,
    sys: Any,
    jobs: Sequence[Any],                 # expects: job_id, src_ids, residual                        # tensor method/property to call
    op_args: Tuple[Any, ...] = (),
    op_kwargs: Optional[Dict[str, Any]] = None,
    backend: Any | None = None,
) -> BatchVJPResult:
    """
    One tape, one VJP over the whole bin.

      x_all = NodeAttrView(sys.nodes, "sphere", indices=union(src_ids)).build().tensor
      x_j = x_all[slice_for_job]
      y_j = getattr(x_j, op_name)(*op_args, **op_kwargs)  # or property value if not callable

    Then L = sum_j <residual_j, y_j> and grads = dL/dx_all with slices mapped
    back to each job's original node ordering.
    """
    op_kwargs = op_kwargs or {}
    op_name = jobs[0].op if jobs else ""
    if not jobs:
        return BatchVJPResult(
            slices=BatchSlices(index_of={}, job_ids=()),
            ys=(),
            grads_full=(),
            grads_per_source=(),
        )

    idx_of: Dict[str, int] = {j.job_id: i for i, j in enumerate(jobs)}
    inv_ids: Tuple[str, ...] = tuple(j.job_id for j in jobs)

    ys: List[Any] = []
    residuals: List[Optional[Any]] = []
    # Prepare a single stacked view over all unique source ids
    union_ids: List[int] = []
    for j in jobs:
        for sid in j.src_ids:
            sid = int(sid)
            if sid not in union_ids:
                union_ids.append(sid)
    pos_of = {sid: i for i, sid in enumerate(union_ids)}
    slices_for_job: List[List[int]] = []

    scope_cm = getattr(backend, "scope", None)
    scope = scope_cm() if callable(scope_cm) else nullcontext()

    with scope, _tape():
        view = NodeAttrView(sys.nodes, "sphere", indices=union_ids).build()
        x_all = view.tensor
        if hasattr(x_all, "requires_grad_"):
            x_all = x_all.requires_grad_()
        for j in jobs:
            idxs = [pos_of[int(s)] for s in j.src_ids]
            slices_for_job.append(idxs)
            x_j = x_all[idxs]
            op = getattr(x_j, op_name)
            y_j = op(*op_args, **op_kwargs) if callable(op) else op
            ys.append(y_j)
            residuals.append(_residual_like(y_j, j.residual, backend))

        # L = sum_j <residual_j, y_j>
        L = None
        for y_j, r_j in zip(ys, residuals):
            if r_j is None:
                continue
            term = (y_j * r_j).sum() if getattr(y_j, "ndim", 0) > 0 else (y_j * r_j)
            L = term if L is None else (L + term)

        if L is None:
            g_all = x_all.zeros_like() if hasattr(x_all, "zeros_like") else (x_all * 0)
        else:
            g_all = autograd.grad(L, (x_all,), retain_graph=False, allow_unused=True)[0]

    if g_all is None:
        grads_list = [None for _ in jobs]
    else:
        grads_list = [g_all[idxs] for idxs in slices_for_job]

    grads_full = tuple(grads_list)
    grads_per_source = tuple(
        _reduce_per_source(g) if g is not None else tuple() for g in grads_full
    )

    return BatchVJPResult(
        slices=BatchSlices(index_of=idx_of, job_ids=inv_ids),
        ys=tuple(ys),
        grads_full=grads_full,
        grads_per_source=grads_per_source,
    )
