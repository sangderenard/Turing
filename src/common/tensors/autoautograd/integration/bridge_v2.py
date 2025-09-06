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
    ops: Sequence[str], args: Optional[Sequence[Any]] = None
) -> Callable[[Any], Any]:
    """Compile a tiny f(x)->y chain with optional per-op arguments.

    Each entry in ``args`` may provide positional and/or keyword arguments for
    the corresponding operation:

    * ``(arg1, arg2, ...)`` → positional args
    * ``{"kw": val}``       → keyword args
    * ``((arg1, arg2), {"kw": val})`` → both positional and keyword args
    """

    chain = _normalize_chain(ops)
    if not chain:
        def _apply_identity(x):
            return x

        return _apply_identity

    chain_local = chain  # closure binding
    args_local = args or ()

    def _apply(x, _chain=chain_local, _args=args_local):
        y = x
        for i, f in enumerate(_chain):
            pos = ()
            kw = {}
            if i < len(_args):
                spec = _args[i]
                if (
                    isinstance(spec, tuple)
                    and len(spec) == 2
                    and isinstance(spec[0], (list, tuple))
                    and isinstance(spec[1], dict)
                ):
                    pos = tuple(spec[0])
                    kw = spec[1]
                elif isinstance(spec, dict):
                    kw = spec
                elif isinstance(spec, (list, tuple)):
                    pos = tuple(spec)
            y = f(y, *pos, **kw)
        return y

    return _apply



def _freeze_for_key(obj: Any) -> Any:
    """Recursively convert lists/dicts to tuples for hashing."""
    if isinstance(obj, dict):
        return tuple(sorted((str(k), _freeze_for_key(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze_for_key(x) for x in obj)
    return obj


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
    residual: AbstractTensor | None = None,
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
        residual=None if residual is None else residual,
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
        Tuple[str, Any, Any],
        List[Tuple[int, Tuple[int, ...], int, Optional[Tuple[Any, ...]], Optional[Dict[str, Any]]]],
    ] = {}
    for idx, spec in enumerate(specs):
        op_name, src_ids, out_id, op_args, op_kwargs = (*spec, None, None)[:5]
        op_args_tuple = tuple(op_args) if isinstance(op_args, (list, tuple)) else op_args
        op_kwargs_dict = dict(op_kwargs) if isinstance(op_kwargs, dict) else None
        key = (
            str(op_name),
            _freeze_for_key(op_args_tuple) if op_args_tuple is not None else None,
            _freeze_for_key(op_kwargs_dict) if op_kwargs_dict is not None else None,
        )
        by_op.setdefault(key, []).append(
            (idx, tuple(int(i) for i in src_ids), int(out_id), op_args_tuple, op_kwargs_dict)
        )

    def get_attr(i: int):
        return sys.nodes[i].theta

    ys_buffer: Dict[int, Any] = {}
    for (op_name, _key_args, _key_kwargs), items in by_op.items():
        op_args = items[0][3] or ()
        op_kwargs = items[0][4]
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
    *,
    weight: str | None = None,
    scale: float = 1.0,
) -> Tuple[List[Any], List[Tuple[Any, ...]]]:
    """Batched forward pass returning predictions and per-source gradients.

    Previously this helper also pushed impulses and required residuals to be
    supplied.  To avoid a separate forward pass, it now performs a single
    batched VJP with unit residuals and returns the raw gradients for each op.
    Callers can compute residuals from the predictions and apply impulses as
    needed.
    """
    ys_out: List[Any] = [None] * len(specs)
    grads_out: List[Tuple[Any, ...]] = [tuple() for _ in range(len(specs))]
    by_op: Dict[
        Tuple[str, Any, Any],
        List[Tuple[int, Tuple[int, ...], int, Optional[Tuple[Any, ...]], Optional[Dict[str, Any]]]],
    ] = {}
    for idx, spec in enumerate(specs):
        op_name, src_ids, out_id, op_args, op_kwargs = (*spec, None, None)[:5]
        op_args_tuple = tuple(op_args) if isinstance(op_args, (list, tuple)) else op_args
        op_kwargs_dict = dict(op_kwargs) if isinstance(op_kwargs, dict) else None
        key = (
            str(op_name),
            _freeze_for_key(op_args_tuple) if op_args_tuple is not None else None,
            _freeze_for_key(op_kwargs_dict) if op_kwargs_dict is not None else None,
        )
        by_op.setdefault(key, []).append(
            (
                idx,
                tuple(int(i) for i in src_ids),
                int(out_id),
                op_args_tuple,
                op_kwargs_dict,
            )
        )

    def get_attr(i: int):
        return sys.nodes[i].theta

    for (op_name, _key_args, _key_kwargs), items in by_op.items():
        op_args = items[0][3] or ()
        op_kwargs = items[0][4]
        jobs: List[_Job] = []
        for idx, src_ids, out_id, _args, _kwargs in items:
            jobs.append(
                _Job(
                    job_id=f"{op_name}:{src_ids}->{out_id}",
                    op=op_name,
                    src_ids=src_ids,
                    op_args=op_args,
                    op_kwargs=op_kwargs,
                    residual=1.0,
                    scale=None,
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
        for (idx, src_ids, out_id, _args, _kwargs), y, grads in zip(
            items, batch.ys, batch.grads_per_source
        ):
            ys_out[idx] = y
            grads_out[idx] = grads
    return ys_out, grads_out
