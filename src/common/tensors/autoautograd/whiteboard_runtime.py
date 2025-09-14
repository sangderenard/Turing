from __future__ import annotations
from dataclasses import dataclass, field
import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
from .whiteboard_cache import WhiteboardCache, CacheEntry

from ..autograd import autograd, GradTape
from .node_tensor import NodeAttrView
from ..abstraction import AbstractTensor as AT
from .job_batcher import JobBatcher
from ..autograd_probes import (
    annotate_params,
    probe_losses,
    set_strict_mode,
)

logger = logging.getLogger(__name__)

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
    grads_per_source_tensor: Any | None = None
    param_grads_full: Tuple[Any, ...] | None = None
    param_grads_tensor: Any | None = None


@dataclass(frozen=True)
class SubFnMeta:
    """Description of a post-op function in a composite kernel.

    ``param_refs`` lists indices or slices into the invoking node's ``param``
    vector that should be passed as positional operands.  Scalars broadcast
    according to the backend's normal rules so no extra annotation is required
    for them.
    """

    name: str
    param_refs: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class OpBatchMeta:
    """Metadata describing how an op may be batched.

    Attributes
    ----------
    shape:
        Optional textual description of shape constraints.  Examples include
        ``"all_equal"`` for elementwise ops or ``"matmul"`` for matrix
        multiplication style contracts.  The value is advisory only and consumed
        by higher level schedulers.
    dim_params:
        Names of keyword arguments that represent dimension selections (``dim``
        or ``axis``) which must be offset when tensors are stacked for batching.
    sub_fns:
        Composite kernels may apply a tiny chain of backend operations after the
        main op.  Each entry describes a method name and which parameters from
        the destination node's ``param`` vector should be supplied.  For example
        the current demo wires ``gather_and`` to ``__mul__`` then ``__add__``
        with ``param[1]`` and ``param[2]`` respectively.  Scalars are assumed to
        broadcast exactly as in the unbatched call; metadata merely records the
        existence of the chain.
    """

    shape: Optional[str] = None
    dim_params: Tuple[str, ...] = ()
    sub_fns: Tuple[SubFnMeta, ...] = ()


# Predefined metadata for the minimal set of elementwise ops currently used in
# tests.  Additional kernels can be registered at runtime via
# :func:`register_batchable_op`.
BATCHABLE_OPS: Dict[str, OpBatchMeta] = {
    "add": OpBatchMeta(shape="all_equal"),
    "mul": OpBatchMeta(shape="all_equal"),
    # `sum` accepts a `dim` argument so note it for callers that may need to
    # adjust that parameter when stacking tensors for batching.
    "sum": OpBatchMeta(shape="any", dim_params=("dim",)),
    # ``matmul`` is included as an example of a non element-wise operator.
    "matmul": OpBatchMeta(shape="matmul"),
    # ``gather_and`` gathers then applies a tiny op chain such as
    # ("__mul__", "__add__").  Parameter slices mirror how the autoautograd
    # demo wires weights then biases via ``slice(1, None, 3)`` and
    # ``slice(2, None, 3)``.
    "gather_and": OpBatchMeta(
        shape="gather",
        dim_params=("dim",),
        sub_fns=(
            SubFnMeta("__mul__", (slice(1, None, 3),)),
            SubFnMeta("__add__", (slice(2, None, 3),)),
        ),
    ),
    # fluxspring routing step and FFT analysis are batchable via custom helpers
    "route_batch": OpBatchMeta(shape="any"),
    "fft_analysis": OpBatchMeta(shape="any"),
}


def register_batchable_op(name: str, meta: OpBatchMeta) -> None:
    """Register ``name`` as batchable with the provided ``meta`` description."""

    BATCHABLE_OPS[name] = meta


@dataclass(frozen=True)
class _WBJob:
    job_id: str
    op: str | None
    src_ids: Tuple[int, ...]
    residual: Optional[float]
    param_lens: Tuple[int, ...] = ()
    fn: Callable[[Any], Any] | None = None
    param_schema: Tuple[str, ...] = ("p",)
    fn_args: Tuple[Any, ...] = ()
    fn_kwargs: dict[str, Any] = field(default_factory=dict)

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
    grad_mode: str = "scalar",
    param_lens: Sequence[int] | None = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
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
        grad_mode=grad_mode,
    )
    hit = cache.get(key)
    if hit is not None:
        return hit.y, hit.grads, hit.meta

    vals = [sys.nodes[i].sphere for i in src_ids]
    param_grads_full = None
    if op_name == "add" and len(vals) == 2 and not op_args and not op_kwargs:
        y = vals[0] + vals[1]
        ones0 = vals[0] * 0 + 1
        ones1 = vals[1] * 0 + 1
        grads_full = AT.stack([ones0, ones1], dim=0)
    elif op_name == "mul" and len(vals) == 2 and not op_args and not op_kwargs:
        y = vals[0] * vals[1]
        g0 = vals[1] * (vals[0] * 0 + 1)
        g1 = vals[0] * (vals[1] * 0 + 1)
        grads_full = AT.stack([g0, g1], dim=0)
    elif op_name == "route_batch":
        from .fluxspring import fs_dec

        kw = dict(op_kwargs or {})
        state = kw.pop("state")
        spec = kw.pop("spec")

        def _route_fn(_p):
            psi_next, _ = fs_dec.pump_tick(state, spec, **kw)
            return psi_next

        job = _WBJob(
            job_id=f"{op_name}:{tuple(src_ids)}",
            op=None,
            src_ids=tuple(int(i) for i in src_ids),
            residual=(1.0 if residual is None else residual),
            param_lens=tuple(int(l) for l in (param_lens or [])),
            fn=_route_fn,
        )

        batch = run_batched_vjp(
            sys=sys,
            jobs=(job,),
            backend=backend,
        )
        y = batch.ys[0]
        grads_full = batch.grads_full[0]
        if batch.param_grads_full:
            param_grads_full = batch.param_grads_full[0]
    elif op_name == "fft_analysis":
        from .fluxspring.spectral_readout import (
            gather_recent_windows,
            batched_bandpower_from_windows,
        )

        kw = dict(op_kwargs or {})
        cfg = kw.pop("cfg")
        harness = kw.pop("harness")
        node_ids = kw.pop("node_ids", list(src_ids))
        W, _ = gather_recent_windows(node_ids, cfg, harness)

        def _fft_fn(_p: Any) -> Any:
            return batched_bandpower_from_windows(W, cfg)

        win_len = int(getattr(W, "shape", (0, 0))[1]) if getattr(W, "shape", None) else 0
        job = _WBJob(
            job_id=f"{op_name}:{tuple(src_ids)}",
            op=None,
            src_ids=tuple(int(i) for i in src_ids),
            residual=(1.0 if residual is None else residual),
            param_lens=tuple(win_len for _ in src_ids),
            fn=_fft_fn,
        )

        batch = run_batched_vjp(
            sys=sys,
            jobs=(job,),
            backend=backend,
            op_kwargs={"param_tensor": W},
        )
        y = batch.ys[0]
        grads_full = batch.grads_full[0]
        if batch.param_grads_full:
            param_grads_full = batch.param_grads_full[0]
    else:
        job = _WBJob(
            job_id=f"{op_name}:{tuple(src_ids)}",
            op=op_name,
            src_ids=tuple(int(i) for i in src_ids),
            residual=1.0,
            param_lens=tuple(int(l) for l in (param_lens or [])),
            param_schema=("sphere",),
        )

        batch = run_batched_vjp(
            sys=sys,
            jobs=(job,),
            op_args=op_args,
            op_kwargs=op_kwargs,
            backend=backend,
        )
        y = batch.ys[0]
        grads_full = batch.grads_full[0]
        if batch.param_grads_full:
            param_grads_full = batch.param_grads_full[0]

    node0 = sys.nodes[src_ids[0]]
    sphere = node0.sphere
    sphere_shape = getattr(sphere, "shape", ())
    if sphere_shape:
        sphere_len = int(sphere_shape[0])
    else:
        try:
            sphere_len = len(sphere)
        except Exception:
            sphere_len = 0
    p_len = 0
    if hasattr(node0, "p"):
        p_shape = getattr(node0.p, "shape", ())
        if p_shape:
            p_len = int(p_shape[0])
        else:
            try:
                p_len = len(node0.p)
            except Exception:
                p_len = 0
    D = p_len
    P = max(sphere_len - D, 0)
    meta = {
        "y_shape": tuple(getattr(y, "shape", ()) or ()),
        "sphere_len": sphere_len,
        "D": D,
        "P": P,
    }

    if grad_mode == "scalar":
        if grads_full is None:
            grads = tuple(0.0 for _ in src_ids)
        else:
            grads = _reduce_per_source(grads_full)
    elif grad_mode == "param":
        if param_grads_full is not None:
            grads = param_grads_full
        elif grads_full is None:
            grads = AT.zeros((len(src_ids), P), float)
        else:
            grads = grads_full[:, D:] if P > 0 else grads_full[:, 0:0]
    else:  # "full"
        grads = grads_full if grads_full is not None else AT.zeros(
            (len(src_ids), sphere_len), float
        )

    entry = CacheEntry(y=y, grads=grads, meta=meta)
    cache.put(key, entry)
    return entry.y, entry.grads, entry.meta

def run_batched_vjp(
    *,
    sys: Any,
    jobs: Sequence[Any],                 # expects: job_id, src_ids, residual                        # tensor method/property to call
    op_args: Tuple[Any, ...] = (),
    op_kwargs: Optional[Dict[str, Any]] = None,
    backend: Any | None = None,
    force_probe: bool = True,
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
    fn_jobs = bool(jobs) and callable(getattr(jobs[0], "fn", None))
    op_name = jobs[0].op if (jobs and not fn_jobs) else ""

    param_tensor = op_kwargs.get("param_tensor")
    if param_tensor is None and op_name == "gather_and":
        if len(op_args) >= 4 and isinstance(op_args[0], int):
            # dim is the first positional argument
            param_tensor = op_args[3]
        elif len(op_args) >= 3:
            # dim is provided via keyword or omitted from position 0
            param_tensor = op_args[2]

    logger.debug(
        "run_batched_vjp: param_tensor_exists=%s requires_grad=%s",
        param_tensor is not None,
        getattr(param_tensor, "requires_grad", None) if param_tensor is not None else None,
    )

    probe_enabled = bool(int(os.environ.get("WHITEBOARD_PROBES", "0"))) or force_probe
    if probe_enabled:
        set_strict_mode(True)

    if not jobs:
        logger.debug("run_batched_vjp: empty jobs; returning empty result")
        return BatchVJPResult(
            slices=BatchSlices(index_of={}, job_ids=()),
            ys=(),
            grads_full=(),
            grads_per_source=(),
            grads_per_source_tensor=None,
            param_grads_full=(),
            param_grads_tensor=None,
        )

    idx_of: Dict[str, int] = {j.job_id: i for i, j in enumerate(jobs)}
    inv_ids: Tuple[str, ...] = tuple(j.job_id for j in jobs)

    # Prepare a single stacked view over all unique source ids and attributes
    union_ids: List[int] = []
    union_attrs: List[str] = []
    for j in jobs:
        for sid in j.src_ids:
            sid = int(sid)
            if sid not in union_ids:
                union_ids.append(sid)
        for a in getattr(j, "param_schema", ("p",)):
            if a not in union_attrs:
                union_attrs.append(a)
    pos_of = {sid: i for i, sid in enumerate(union_ids)}
    slices_for_job: List[List[int]] = [[pos_of[int(s)] for s in j.src_ids] for j in jobs]

    logger.debug(
        "run_batched_vjp: op=%s jobs=%d union_ids=%d slices_first=%s",
        op_name,
        len(jobs),
        len(union_ids),
        slices_for_job[0] if slices_for_job else (),
    )

    param_len_map: Dict[int, int] = {}
    for j in jobs:
        for sid, plen in zip(j.src_ids, getattr(j, "param_lens", ())):
            param_len_map[int(sid)] = int(plen)
    param_ids: List[int] = [sid for sid in union_ids if param_len_map.get(sid, 0) > 0]
    param_pos = {sid: i for i, sid in enumerate(param_ids)}
    param_offsets: Dict[int, int] = {}
    offset = 0
    for sid in param_ids:
        param_offsets[sid] = offset
        offset += param_len_map[sid]

    scope_cm = getattr(backend, "scope", None)
    scope = scope_cm() if callable(scope_cm) else nullcontext()

    with scope, _tape():
        view = NodeAttrView(sys.nodes, union_attrs, indices=union_ids).build()
        x_all = view.tensor
        if hasattr(x_all, "requires_grad_"):
            x_all = x_all.requires_grad_()
        logger.debug("run_batched_vjp: x_all=%s", AT.get_tensor(x_all))
        if param_tensor is not None and hasattr(param_tensor, "requires_grad_"):
            param_tensor = param_tensor.requires_grad_()
        probe_params: List[Any] = []
        if probe_enabled:
            probe_params = [x_all]
            if param_tensor is not None:
                probe_params.append(param_tensor)
            try:
                annotate_params(probe_params, ag=autograd)
            except Exception as e:  # pragma: no cover - best effort
                logger.debug("run_batched_vjp: annotate_params failed: %s", e)

        # Slice per-job tensors referencing the union view
        x_list = [x_all[idxs] for idxs in slices_for_job]
        params_per_job: List[List[Any]] = []
        for x_j, j in zip(x_list, jobs):
            params: List[Any] = []
            for a in getattr(j, "param_schema", ("p",)):
                sl = view._attr_slices[a]  # type: ignore[index]
                part = x_j[..., sl]
                shape = view._attr_shapes[a]  # type: ignore[index]
                if hasattr(part, "view") and shape is not None:
                    part = part.view((part.shape[0],) + shape)
                params.append(part)
            params_per_job.append(params)

        if fn_jobs:
            ys = [
                j.fn(*params, *j.fn_args, **j.fn_kwargs) if callable(j.fn) else params[0]
                for j, params in zip(jobs, params_per_job)
            ]
            logger.debug(
                "run_batched_vjp: callable jobs=%d y0_shape=%s",
                len(ys),
                tuple(getattr(ys[0], "shape", ())) if ys else (),
            )
        elif op_name in BATCHABLE_OPS:
            x_list = [p[0] for p in params_per_job]
            meta = BATCHABLE_OPS[op_name]

            # Offset dimension parameters for stacked call
            vec_kwargs = dict(op_kwargs)
            for dp in meta.dim_params:
                if dp in vec_kwargs and isinstance(vec_kwargs[dp], int):
                    vec_kwargs[dp] += 1

            class _JBJob:
                def __init__(self, x: Any) -> None:
                    self.args = (x,)
                    self.kwargs: Dict[str, Any] = {}
                    self.residual = None

            jb_jobs = [_JBJob(x) for x in x_list]

            def _fn(x, *, residual=None, **kw):
                op = getattr(x, op_name)
                if not callable(op):
                    return op
                # Support gather_and with dim passed via kwargs (no positional dim)
                if op_name == "gather_and" and len(op_args) >= 3 and not isinstance(op_args[0], int):
                    # Interpret op_args as (indices, fn_specs, param_tensor)
                    _kw = dict(op_kwargs)
                    _kw.setdefault("indices", op_args[0])
                    _kw.setdefault("fn_specs", op_args[1])
                    _kw.setdefault("param_tensor", op_args[2])
                    return op(**_kw)
                return op(*op_args, **op_kwargs)

            def _vec_fn(x, *, residual=None, **kw):
                op = getattr(x, op_name)
                if not callable(op):
                    return op
                # Support gather_and with dim passed via kwargs (no positional dim)
                if op_name == "gather_and" and len(op_args) >= 3 and not isinstance(op_args[0], int):
                    _kw = dict(vec_kwargs)
                    _kw.setdefault("indices", op_args[0])
                    _kw.setdefault("fn_specs", op_args[1])
                    _kw.setdefault("param_tensor", op_args[2])
                    return op(**_kw)
                # Otherwise, handle positional dim by offsetting when present
                vec_args = op_args
                if meta.dim_params and len(vec_args) > 0 and isinstance(vec_args[0], int):
                    vec_args = (vec_args[0] + 1,) + vec_args[1:]
                return op(*vec_args, **vec_kwargs)

            ys = JobBatcher.run_vectorized(jb_jobs, {"fn": _fn, "vectorized_fn": _vec_fn})
            logger.debug(
                "run_batched_vjp: vectorized op=%s outputs=%d y0_shape=%s",
                op_name,
                len(ys),
                tuple(getattr(ys[0], "shape", ())) if ys else (),
            )
        else:
            x_list = [p[0] for p in params_per_job]
            ys = []
            for x_j in x_list:
                op = getattr(x_j, op_name)
                y_j = op(*op_args, **op_kwargs) if callable(op) else op
                ys.append(y_j)
            logger.debug(
                "run_batched_vjp: scalar loop op=%s outputs=%d y0_shape=%s",
                op_name,
                len(ys),
                tuple(getattr(ys[0], "shape", ())) if ys else (),
            )

        # Convert residuals to backend-friendly tensors
        residuals = [_residual_like(y_j, j.residual, backend) for y_j, j in zip(ys, jobs)]
        any_res = any(r is not None for r in residuals)
        logger.debug(
            "run_batched_vjp: residuals provided=%s first_residual_type=%s",
            any_res,
            type(residuals[0]).__name__ if residuals and residuals[0] is not None else None,
        )

        # L = sum_j <residual_j, y_j>
        L = None
        if any(r is not None for r in residuals):
            try:
                y_stack = JobBatcher._stack(ys)
                res_stack = JobBatcher._stack([r if r is not None else (y_stack[0] * 0) for r in residuals])
                L = (y_stack * res_stack).sum()
            except Exception:
                for y_j, r_j in zip(ys, residuals):
                    if r_j is None:
                        continue
                    term = (y_j * r_j).sum() if getattr(y_j, "ndim", 0) > 0 else (y_j * r_j)
                    L = term if L is None else (L + term)

        if L is None:
            g_all = x_all.zeros_like() if hasattr(x_all, "zeros_like") else (x_all * 0)
            if param_tensor is not None:
                if hasattr(param_tensor, "zeros_like"):
                    g_param = param_tensor.zeros_like()
                else:
                    g_param = param_tensor * 0
            else:
                g_param = None
            logger.debug("run_batched_vjp: no residuals → zero grads")
        else:
            targets = (x_all, param_tensor) if param_tensor is not None else (x_all,)
            logger.debug(
                "run_batched_vjp: pre grad param_tensor_id=%s requires_grad=%s grad_fn=%s",
                id(param_tensor) if param_tensor is not None else None,
                getattr(param_tensor, "requires_grad", None) if param_tensor is not None else None,
                getattr(param_tensor, "grad_fn", None) if param_tensor is not None else None,
            )
            grads = autograd.grad(
                L, targets, retain_graph=probe_enabled, allow_unused=True
            )
            ops = getattr(autograd.tape, "operations", None)
            logger.debug(
                "run_batched_vjp: tape_ops=%s",
                len(ops) if ops is not None else None,
            )
            g_all = grads[0]
            g_param = grads[1] if param_tensor is not None else None
            logger.debug(
                "run_batched_vjp: computed grads g_all_shape=%s g_param_shape=%s",
                tuple(getattr(g_all, "shape", ())) if g_all is not None else None,
                tuple(getattr(g_param, "shape", ())) if g_param is not None else None,
            )
            if probe_enabled:
                logger.info("run_batched_vjp: running autograd probes")
                try:
                    probe_losses({"L": L}, probe_params, ag=autograd)
                except Exception as e:  # pragma: no cover - best effort
                    logger.warning(
                        "run_batched_vjp: autograd probes failed: %s", e
                    )
        def _has_any(t: Any) -> bool:
            try:
                return AT.get_tensor(t).any()
            except Exception:
                return False

        zero_all = g_all is None or not _has_any(g_all)
        zero_param = param_tensor is not None and (g_param is None or not _has_any(g_param))
        if zero_all or zero_param:
            job_ids = tuple(j.job_id for j in jobs)
            logger.debug(
                "run_batched_vjp: WARNING zero grads job_ids=%s union_ids_first=%s union_ids_len=%d",
                job_ids,
                tuple(union_ids[:5]),
                len(union_ids),
            )

    g_stack = g_all
    if g_stack is None:
        grads_list = [None for _ in jobs]
    else:
        grads_list = [g_stack[idxs] for idxs in slices_for_job]

    if g_param is None:
        param_grads_list = [None for _ in jobs]
    else:
        union_slices = [
            g_param[param_offsets[sid] : param_offsets[sid] + param_len_map[sid]]
            for sid in param_ids
        ]
        union_stack = JobBatcher._stack(union_slices)
        zero_full = g_param * 0
        param_grads_list = []
        for j in jobs:
            rows = []
            for sid, plen in zip(j.src_ids, getattr(j, "param_lens", ())):
                row = zero_full.clone() if hasattr(zero_full, "clone") else zero_full
                if plen > 0 and sid in param_pos:
                    off = param_offsets[sid]
                    row[off : off + plen] = union_stack[param_pos[sid]]
                rows.append(row)
            param_grads_list.append(JobBatcher._stack(rows))

    grads_full = tuple(grads_list)
    grads_per_source = tuple(
        _reduce_per_source(g) if g is not None else tuple(0.0 for _ in idxs)
        for g, idxs in zip(grads_full, slices_for_job)
    )

    logger.debug(
        "run_batched_vjp: done grads_full=%d per_source=%d",
        len(grads_full),
        len(grads_per_source),
    )
    for j, y_j, r_j, g_j, p_j in zip(jobs, ys, residuals, grads_full, param_grads_list):
        logger.debug(
            "run_batched_vjp: job=%s src=%s residual=%s y=%s grad=%s param_grad=%s",
            j.job_id,
            j.src_ids,
            r_j,
            AT.get_tensor(y_j),
            AT.get_tensor(g_j) if g_j is not None else None,
            AT.get_tensor(p_j) if p_j is not None else None,
        )
    return BatchVJPResult(
        slices=BatchSlices(index_of=idx_of, job_ids=inv_ids),
        ys=tuple(ys),
        grads_full=grads_full,
        grads_per_source=grads_per_source,
        grads_per_source_tensor=g_stack,
        param_grads_full=tuple(param_grads_list),
        param_grads_tensor=g_param,
    )
