from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


from ..whiteboard_runtime import run_op_and_grads_cached
from ..whiteboard_cache import WhiteboardCache
from ...abstraction import AbstractTensor
from ...linalg import norm as at_norm
from .preop import preactivate_src



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

    def _resolve(item: Any, params: Optional[Any]):
        if isinstance(item, str) and item.startswith("@param[") and item.endswith("]"):
            idx = int(item[7:-1])
            if params is None:
                raise ValueError("param placeholder used but no params provided")
            return params[idx]
        if isinstance(item, (list, tuple)):
            return type(item)(_resolve(x, params) for x in item)
        if isinstance(item, dict):
            return {k: _resolve(v, params) for k, v in item.items()}
        return item

    def _apply(x, params=None, _chain=chain_local, _args=args_local):
        y = x
        for i, f in enumerate(_chain):
            pos = ()
            kw = {}
            if i < len(_args):
                spec = _resolve(_args[i], params)
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
                else:
                    pos = (spec,)
            y = f(y, *pos, **kw)
        return y

    return _apply



def _freeze_for_key(obj: Any) -> Any:
    """Recursively convert lists/dicts to tuples for hashing."""
    if isinstance(obj, slice):
        return (
            _freeze_for_key(obj.start),
            _freeze_for_key(obj.stop),
            _freeze_for_key(obj.step),
        )
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


def _preactivate_nodes(sys, node_ids: Sequence[int]) -> Dict[int, Tuple[Any, dict, Any]]:
    """Build cache of preactivated nodes keyed by id.

    Each entry stores ``(version, y, meta)`` so lookups can be refreshed when a
    node's ``version`` changes.
    """
    cache: Dict[int, Tuple[Any, Any, dict]] = {}
    for i in node_ids:
        node = sys.nodes.get(int(i)) if isinstance(sys.nodes, dict) else sys.nodes[int(i)]
        if hasattr(node, "p") and hasattr(node, "ctrl"):
            y, meta = preactivate_src(sys, int(i))
            cache[int(i)] = (getattr(node, "version", None), y, meta)
    return cache


def _get_preactivation(sys, nid: int, cache: Dict[int, Tuple[Any, Any, dict]]) -> Tuple[Any, dict]:
    """Fetch preactivation from cache, refreshing if the node changed."""
    node = sys.nodes.get(int(nid)) if isinstance(sys.nodes, dict) else sys.nodes[int(nid)]
    version = getattr(node, "version", None)
    entry = cache.get(int(nid))
    if entry is None or entry[0] != version:
        y, meta = preactivate_src(sys, int(nid))
        cache[int(nid)] = (version, y, meta)
        return y, meta
    return entry[1], entry[2]


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
) -> Tuple[Any, Tuple[dict, ...]]:
    """Single op call with caching; returns output and per-source metadata."""
    if weight == "inv_length":
        scale *= _inv_length_scale(sys, out_id, src_ids)

    pre_cache = _preactivate_nodes(sys, src_ids)
    metas: List[dict] = []
    for i in src_ids:
        node = sys.nodes.get(int(i)) if isinstance(sys.nodes, dict) else sys.nodes[int(i)]
        if hasattr(node, "p") and hasattr(node, "ctrl"):
            _y, meta = _get_preactivation(sys, int(i), pre_cache)
        else:
            meta = {}
        metas.append(meta)

    wb_cache = cache or WhiteboardCache()
    params: List[Any] = []
    param_lens: List[int] = []
    for i in src_ids:
        node = sys.nodes.get(int(i)) if isinstance(sys.nodes, dict) else sys.nodes[int(i)]
        if hasattr(node, "ctrl"):
            flat = AbstractTensor.get_tensor(node.ctrl).flatten()
            params.append(flat)
            param_lens.append(len(flat))
        else:
            param_lens.append(0)
    if params:
        params_vec = AbstractTensor.concat(params, dim=0)
    else:
        params_vec = AbstractTensor.get_tensor([])
    base_args = tuple(op_args) if op_args is not None else ()
    full_args = base_args + (params_vec,)
    y, g_param, _ = run_op_and_grads_cached(
        sys,
        op_name,
        tuple(int(i) for i in src_ids),
        scale=scale,
        residual=None if residual is None else float(residual),
        weight=weight,
        cache=wb_cache,
        backend=None,
        backend_tag=None,
        op_args=full_args,
        op_kwargs=dict(op_kwargs) if op_kwargs is not None else None,
        grad_mode="param",
        param_lens=tuple(param_lens),
    )
    if residual is not None:
        r_tensor = AbstractTensor.get_tensor(residual)
        prod = g_param * r_tensor
        extra_dims = tuple(range(1, getattr(prod, "ndim", 1)))
        g_scalar = prod.sum(dim=extra_dims) if extra_dims else prod
        for idx, i in enumerate(src_ids):
            g_val = float(g_scalar[idx])
            sys.impulse(int(i), int(out_id), op_name, float(-scale * g_val))
        param_nodes = []
        param_idx = []
        for idx, i in enumerate(src_ids):
            node = sys.nodes.get(int(i)) if isinstance(sys.nodes, dict) else sys.nodes[int(i)]
            if hasattr(node, "ctrl"):
                param_nodes.append(node)
                param_idx.append(idx)
        if param_nodes:
            gk = g_param[param_idx]
            prod = gk * r_tensor
            extra_dims = tuple(range(2, getattr(prod, "ndim", 2)))
            delta = prod.sum(dim=extra_dims) if extra_dims else prod
            params = AbstractTensor.stack([n.ctrl for n in param_nodes], dim=0)
            params = params + delta
            for node, new_param in zip(param_nodes, params):
                node.ctrl = new_param
    return y, tuple(metas)


def batched_forward_v2(
    sys,
    specs: Sequence[Tuple],
    *,
    weight: str | None = None,
    scale: float = 1.0,
    cache: WhiteboardCache | None = None,
) -> List[Any]:
    """Forward-only for specs of form `(op_name, src_ids, out_id, op_args, op_kwargs)`."""
    ys_out: List[Any] = []
    all_ids = {int(i) for _spec in specs for i in _spec[1]}
    pre_cache = _preactivate_nodes(sys, all_ids)
    wb_cache = cache or WhiteboardCache()
    by_op: Dict[
        Tuple[str, Any, Any],
        List[Tuple[int, Tuple[int, ...], int, Optional[Tuple[Any, ...]], Optional[Dict[str, Any]]]],
    ] = {}
    for idx, spec in enumerate(specs):
        op_name, src_ids, out_id, op_args, op_kwargs = (*spec, None, None)[:5]
        for i in src_ids:
            node = sys.nodes.get(int(i)) if isinstance(sys.nodes, dict) else sys.nodes[int(i)]
            if hasattr(node, "p") and hasattr(node, "ctrl"):
                _get_preactivation(sys, int(i), pre_cache)
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

    ys_buffer: Dict[int, Any] = {}
    for (op_name, _key_args, _key_kwargs), items in by_op.items():
        idx0, src_ids0, out_id0, op_args0, op_kwargs0 = items[0]
        sc = scale * (_inv_length_scale(sys, out_id0, src_ids0) if weight == "inv_length" else 1.0)
        params: List[Any] = []
        param_lens: List[int] = []
        for i in src_ids0:
            node = sys.nodes.get(int(i)) if isinstance(sys.nodes, dict) else sys.nodes[int(i)]
            if hasattr(node, "ctrl"):
                flat = AbstractTensor.get_tensor(node.ctrl).flatten()
                params.append(flat)
                param_lens.append(len(flat))
            else:
                param_lens.append(0)
        if params:
            params_vec = AbstractTensor.concat(params, dim=0)
        else:
            params_vec = AbstractTensor.get_tensor([])
        base_args = op_args0 or ()
        full_args = base_args + (params_vec,)
        y, _, _ = run_op_and_grads_cached(
            sys,
            op_name,
            src_ids0,
            scale=sc,
            residual=None,
            weight=weight,
            cache=wb_cache,
            backend=None,
            backend_tag=None,
            op_args=full_args,
            op_kwargs=op_kwargs0,
            grad_mode="scalar",
            param_lens=tuple(param_lens),
        )
        for idx, *_ in items:
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
    cache: WhiteboardCache | None = None,
) -> Tuple[List[Any], List[Tuple[Any, ...]], List[Tuple[dict, ...]]]:
    """Batched forward pass returning predictions, gradients and metadata.

    Previously this helper also pushed impulses and required residuals to be
    supplied.  To avoid a separate forward pass, it now performs a single
    batched VJP with unit residuals and returns the raw gradients for each op.
    Callers can compute residuals from the predictions and apply impulses as
    needed.
    """
    ys_out: List[Any] = [None] * len(specs)
    grads_out: List[Tuple[Any, ...]] = [tuple() for _ in range(len(specs))]
    metas_out: List[Tuple[dict, ...]] = [tuple() for _ in range(len(specs))]
    all_ids = {int(i) for _spec in specs for i in _spec[1]}
    pre_cache = _preactivate_nodes(sys, all_ids)
    wb_cache = cache or WhiteboardCache()
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

    for (op_name, _key_args, _key_kwargs), items in by_op.items():
        for idx, src_ids, out_id, op_args, op_kwargs in items:
            metas = []
            for i in src_ids:
                node = sys.nodes.get(int(i)) if isinstance(sys.nodes, dict) else sys.nodes[int(i)]
                if hasattr(node, "p") and hasattr(node, "ctrl"):
                    _y, meta = _get_preactivation(sys, int(i), pre_cache)
                else:
                    meta = {}
                metas.append(meta)
            metas_out[idx] = tuple(metas)
            sc = scale * (_inv_length_scale(sys, out_id, src_ids) if weight == "inv_length" else 1.0)
            params: List[Any] = []
            param_lens: List[int] = []
            for i in src_ids:
                node = sys.nodes.get(int(i)) if isinstance(sys.nodes, dict) else sys.nodes[int(i)]
                if hasattr(node, "ctrl"):
                    flat = AbstractTensor.get_tensor(node.ctrl).flatten()
                    params.append(flat)
                    param_lens.append(len(flat))
                else:
                    param_lens.append(0)
            if params:
                params_vec = AbstractTensor.concat(params, dim=0)
            else:
                params_vec = AbstractTensor.get_tensor([])
            base_args = op_args or ()
            full_args = base_args + (params_vec,)
            y, g_param, _ = run_op_and_grads_cached(
                sys,
                op_name,
                src_ids,
                scale=sc,
                residual=None,
                weight=weight,
                cache=wb_cache,
                backend=None,
                backend_tag=None,
                op_args=full_args,
                op_kwargs=op_kwargs,
                grad_mode="param",
                param_lens=tuple(param_lens),
            )
            ys_out[idx] = y
            grads_out[idx] = g_param
    return ys_out, grads_out, metas_out
