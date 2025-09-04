from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Contracts we rely on (already in your codebase):
# - autograd: has GradTape and grad(out, inputs, grad_outputs=None)
# - NodeAttrView: builds a backend tensor for the selected nodes/attr
from ..autograd import autograd, GradTape
from ..abstraction import AbstractTensor
from .node_tensor import NodeAttrView


@contextmanager
def whiteboard_tape():
    """
    Install a fresh autograd tape for a single batched VJP.
    This guarantees single-backward discipline and no persistent .grad accumulation.
    """
    old = autograd.tape
    autograd.tape = GradTape()
    try:
        yield
    finally:
        autograd.tape = old


@dataclass(frozen=True)
class BatchSlices:
    """
    Mapping of job_ids to their position in the batched run.

    We use an integer index (0..B-1) rather than Python slices, because each job's
    inputs are a separate AbstractTensor (shape (k, ...) per job). This keeps us
    free from any backend-specific stacking semantics and still enables a single
    tape/backward.
    """
    index_of: Mapping[str, int]           # job_id -> batch index
    job_ids: Tuple[str, ...]              # batch index -> job_id (inverse map)


@dataclass(frozen=True)
class BatchVJPResult:
    """
    Result of a full-batch whiteboard evaluation.
    - slices: mapping from job_id to its batch index (and inverse)
    - ys: list of per-job forward results (backend tensors or scalars)
    - grads_full: list of dL/dx_j tensors, shape like each x_j  (k, ...) per job
    - grads_per_source: list of tuples of per-source scalars, shape (k,) per job
    """
    slices: BatchSlices
    ys: Tuple[Any, ...]
    grads_full: Tuple[Any, ...]
    grads_per_source: Tuple[Tuple[Any, ...], ...]


def _as_seed_like(y: Any, residual: Optional[float | Any], *, backend: Any | None) -> Optional[Any]:
    """
    Build a VJP seed with the same shape as y (or broadcastable to it).
    - If residual is None: return None (no impulses).
    - If residual is scalar: prefer broadcasting via backend if possible.
    - If residual is a backend tensor already: pass through.
    """
    if residual is None:
        return None
    if backend is not None and hasattr(backend, "asarray"):
        try:
            # Let backend handle scalar→tensor
            return backend.asarray(residual)
        except Exception:
            pass
    # Fallback: rely on backend broadcasting via arithmetic when used
    return residual


def _reduce_to_per_source(g: Any) -> Tuple[Any, ...]:
    """
    Reduce a gradient tensor g (shape (k,) or (k, F, ...)) to one scalar per source.
    This sums over axes 1..N, leaving (k,) and returns Python floats.
    """
    ndim = getattr(g, "ndim", 1)
    if ndim <= 1:
        # Already (k,) — return backend scalars as-is (AbstractTensor or native)
        return tuple(gi for gi in g if gi is not None) if g is not None else ()
    axes = tuple(range(1, ndim))
    # Use PyTorch-style API: sum over trailing dims via dim=axes
    gk = g.sum(dim=axes)
    # Return per-source scalars as backend values (no Python float coercion)
    return tuple(gi for gi in gk)


def run_batched_vjp(
    *,
    sys: Any,
    jobs: Sequence[Any],                        # OpJob-like: .job_id .op .src_ids .residual .scale .weight
    op_apply: Callable[[Any], Any],             # callable that applies the op to a single x_j (AbstractTensor)
    get_attr: Callable[[int], Any],             # e.g., lambda i: sys.nodes[i].theta
    backend: Any | None = None,                 # optional backend scope/adapter; passed through only
) -> BatchVJPResult:
    """
    One tape; one backward:
      - For each job, build x_j = NodeAttrView(...).tensor (shape (k,) or (k, F, ...))
      - y_j = op_apply(x_j)           (no scalarization)
      - seed_j = seed(residual_j)     (shape-compatible with y_j; residual can be scalar)
      - L = sum_j <seed_j, y_j>       (inner product via elementwise * and sum)
      - g_j = dL/dx_j                 (list returned by autograd.grad)
      - grads_per_source[j] = sum over input feature axes only (one scalar per source)
    Returns a mapping job_id↔batch index, forward y list, full grads list, and per-source scalars.
    """
    B = len(jobs)
    if B == 0:
        return BatchVJPResult(
            slices=BatchSlices(index_of={}, job_ids=()),
            ys=(),
            grads_full=(),
            grads_per_source=(),
        )

    # Build batch index mapping
    idx_of: Dict[str, int] = {j.job_id: t for t, j in enumerate(jobs)}
    inv_ids: Tuple[str, ...] = tuple(j.job_id for j in jobs)
    slices = BatchSlices(index_of=idx_of, job_ids=inv_ids)

    # Build per-job inputs as separate backend tensors (avoid forcing a global stack),
    # so we can take a single grad over a list of inputs.
    xs: List[Any] = []
    ys: List[Any] = []
    seeds: List[Optional[Any]] = []

    # Optional backend scope: if backend exposes a scope() context manager, enter it.
    scope_cm = getattr(backend, "scope", None)
    scope = scope_cm() if callable(scope_cm) else nullcontext()

    with scope, whiteboard_tape():
        for j in jobs:
            # Construct x_j from provided getter when available to ensure AbstractTensor tensors.
            x_j = None
            if callable(get_attr):
                vals = [get_attr(i) for i in j.src_ids]
                # Wrap non-AT values
                at_vals = [v if isinstance(v, AbstractTensor) else AbstractTensor.tensor(v) for v in vals]
                # Prefer class stack with dim=0; fall back to axis=0
                try:
                    x_j = AbstractTensor.stack(at_vals, dim=0)
                except TypeError:
                    x_j = AbstractTensor.stack(at_vals, axis=0)
            else:
                # Fallback: policy-driven view
                x_j = NodeAttrView(sys.nodes, "theta", indices=j.src_ids).build().tensor
            xs.append(x_j)

            # Apply op via provided callable (scheduler-owned, no math invented here).
            y_j = op_apply(x_j)
            ys.append(y_j)

            # Build seed for VJP (can be scalar or tensor; None means skip impulses)
            seeds.append(_as_seed_like(y_j, j.residual, backend=backend))

        # Accumulate a single scalar objective L to drive a single backward pass:
        # L = sum_j <seed_j, y_j> ; if seed_j is None, treat as 0 (no contribution).
        L = None
        for y_j, s_j in zip(ys, seeds):
            if s_j is None:
                continue
            # Inner product: elementwise mul + sum; for scalar y_j this is y_j * s_j
            term = (y_j * s_j).sum() if getattr(y_j, "ndim", 0) > 0 else (y_j * s_j)
            L = term if L is None else (L + term)

        if L is None:
            # No residual anywhere; grads are zeros with the right shapes
            grads = [getattr(x_j, "zeros_like", lambda: x_j * 0)() for x_j in xs]  # backend zeros_like or x*0
        else:
            grads = autograd.grad(L, xs, retain_graph=False, allow_unused=True)

    grads_full = tuple(grads)
    grads_per_source = tuple(_reduce_to_per_source(g) for g in grads_full)
    return BatchVJPResult(
        slices=slices,
        ys=tuple(ys),
        grads_full=grads_full,
        grads_per_source=grads_per_source,
    )
