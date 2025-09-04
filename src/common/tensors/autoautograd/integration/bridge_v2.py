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
    residual: float | None
    scale: float
    weight: str | None
    backend_tag: Any = None


from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

# tokens we accept in op_args
OpToken = Union[
    str,                             # name of an AT tensor instance method: "sum", "mean", ...
    Callable[[Any], Any],            # direct callable: lambda x: ...
    Tuple[str, Tuple, Dict],         # ("sum", (args,), {"axis": 0})
    Tuple[str, Tuple],               # ("sum", (args,))
    Tuple[str, Dict],                # ("sum", {"axis": 0})
    Tuple[str, Any],                 # ('add', c) | ('mul', c) sugar
]

def _compile_token(token: OpToken, *, AT: Any | None) -> Callable[[Any], Any]:
    """Compile one token into f(x)->y. All validation happens HERE (not hot)."""
    if callable(token):
        def _f(x, _f=token): return _f(x)
        return _f

    if isinstance(token, str):
        name = token
        def _f(x, _name=name):
            return getattr(x, _name)()
        return _f

    if isinstance(token, tuple) and token:
        name = token[0]

        # sugar for ('add', c) and ('mul', c)
        if name in ("add", "mul") and len(token) == 2:
            c = token[1]
            k = AT.asarray(c) if (AT is not None and hasattr(AT, "asarray")) else c
            if name == "add":
                def _f(x, _k=k): return x + _k
            else:
                def _f(x, _k=k): return x * _k
            return _f

        # structured call with args/kwargs
        args: Tuple = ()
        kwargs: Dict[str, Any] = {}
        if len(token) == 3 and isinstance(token[1], tuple) and isinstance(token[2], dict):
            args, kwargs = token[1], token[2]
        elif len(token) == 2 and isinstance(token[1], tuple):
            args = token[1]
        elif len(token) == 2 and isinstance(token[1], dict):
            kwargs = token[1]
        else:
            raise ValueError(f"Unsupported op tuple form: {token!r}")

        def _f(x, _name=name, _args=args, _kwargs=kwargs):
            return getattr(x, _name)(* _args, ** _kwargs)
        return _f

    raise ValueError(f"Unsupported op token: {token!r}")

def _normalize_chain(op_name: str, op_args: Optional[Union[OpToken, Sequence[OpToken]]], *, AT: Any | None):
    """One-time normalization → list[callable]."""
    name = str(op_name).lower()
    defaults = {
        "sum_k": ["sum"],    # reduce fully (or let backend default axes)
        "prod_k": ["prod"],
        "identity": [],
    }
    if op_args is None:
        tokens: Sequence[OpToken] = defaults.get(name, [])
    elif isinstance(op_args, (list, tuple)):
        tokens = list(op_args)
    else:
        tokens = [op_args]
    return tuple(_compile_token(tok, AT=AT) for tok in tokens)

def _op_apply_factory(
    op_name: str,
    op_args: Optional[Union[OpToken, Sequence[OpToken]]] = None,
    *,
    AT: Any | None = None,     # pass your AbstractTensor module/class if you want constant lifting
) -> Callable[[Any], Any]:
    """
    Build a tiny, ultra-hot f(x)->y that applies a precompiled chain.
    No getattr/validation in hot path.
    """
    chain = _normalize_chain(op_name, op_args, AT=AT)
    if not chain:
        def _apply_identity(x): return x
        return _apply_identity

    chain_local = chain  # closure binding
    def _apply(x, _chain=chain_local):
        y = x
        for f in _chain:
            y = f(y)
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
    op_args: Optional[Dict[str, Any]] = None,
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

    op_apply = _op_apply_factory(op_name, op_args)

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
    specs: Sequence[Tuple],
    *,
    weight: str | None = None,
    scale: float = 1.0,
) -> List[Any]:
    """Forward-only for specs of form (op_name, src_ids, out_id[, op_args]), grouped by (op,args)."""
    ys_out: List[Any] = []
    by_op: Dict[Tuple[str, Tuple[Tuple[str, Any], ...] | None], List[Tuple[int, Tuple[int, ...], int, Optional[Dict[str, Any]]]]] = {}
    for idx, spec in enumerate(specs):
        if len(spec) >= 4:
            op_name, src_ids, out_id, op_args = spec[0], spec[1], spec[2], spec[3]
        else:
            op_name, src_ids, out_id = spec
            op_args = None
        key_args: Optional[Tuple[Tuple[str, Any], ...]] = None
        if isinstance(op_args, dict):
            key_args = tuple(sorted((str(k), tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in op_args.items()))
        key = (str(op_name), key_args)
        by_op.setdefault(key, []).append((idx, tuple(int(i) for i in src_ids), int(out_id), op_args))

    def get_attr(i: int):
        return sys.nodes[i].theta

    ys_buffer: Dict[int, Any] = {}
    for (op_name, key_args), items in by_op.items():
        op_args = {k: v for (k, v) in (key_args or ())} if key_args is not None else None
        op_apply = _op_apply_factory(op_name, op_args)
        jobs: List[_Job] = []
        for idx, src_ids, out_id, _args in items:
            sc = scale * (_inv_length_scale(sys, out_id, src_ids) if weight == "inv_length" else 1.0)
            jobs.append(_Job(job_id=f"{op_name}:{src_ids}->{out_id}", op=op_name, src_ids=src_ids, residual=None, scale=sc, weight=weight))
        batch = run_batched_vjp(sys=sys, jobs=jobs, op_apply=op_apply, get_attr=get_attr, backend=None)
        for (idx, _src, _out, _args), y in zip(items, batch.ys):
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
    """Batched impulse push for specs (op_name, src_ids, out_id[, op_args])."""
    ys_out: List[Any] = [None] * len(specs)
    by_op: Dict[Tuple[str, Tuple[Tuple[str, Any], ...] | None], List[Tuple[int, Tuple[int, ...], int, float, Optional[Dict[str, Any]]]]] = {}
    for idx, (spec, r) in enumerate(zip(specs, residuals)):
        if len(spec) >= 4:
            op_name, src_ids, out_id, op_args = spec[0], spec[1], spec[2], spec[3]
        else:
            op_name, src_ids, out_id = spec
            op_args = None
        key_args: Optional[Tuple[Tuple[str, Any], ...]] = None
        if isinstance(op_args, dict):
            key_args = tuple(sorted((str(k), tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in op_args.items()))
        key = (str(op_name), key_args)
        by_op.setdefault(key, []).append((idx, tuple(int(i) for i in src_ids), int(out_id), float(r), op_args))

    def get_attr(i: int):
        return sys.nodes[i].theta

    for (op_name, key_args), items in by_op.items():
        op_args = {k: v for (k, v) in (key_args or ())} if key_args is not None else None
        op_apply = _op_apply_factory(op_name, op_args)
        jobs: List[_Job] = []
        scales: List[float] = []
        for idx, src_ids, out_id, r, _args in items:
            sc = scale * (_inv_length_scale(sys, out_id, src_ids) if weight == "inv_length" else 1.0)
            scales.append(sc)
            jobs.append(_Job(job_id=f"{op_name}:{src_ids}->{out_id}", op=op_name, src_ids=src_ids, residual=r, scale=sc, weight=weight))
        batch = run_batched_vjp(sys=sys, jobs=jobs, op_apply=op_apply, get_attr=get_attr, backend=None)
        for (idx, src_ids, out_id, r, _args), y, grads, sc in zip(items, batch.ys, batch.grads_per_source, scales):
            ys_out[idx] = y
            for i, g in zip(src_ids, grads):
                g_host = float(getattr(g, "item_", lambda: g)()) if hasattr(g, "item_") else float(g)
                sys.impulse(int(i), int(out_id), op_name, float(sc * g_host * (-float(r))))
    return ys_out
