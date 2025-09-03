from __future__ import annotations
from contextlib import contextmanager
from typing import Sequence, Tuple, Optional
import numpy as np

from ..autograd import autograd, GradTape
from .whiteboard_cache import WhiteboardCache, ParamSig
from .node_tensor import NodeAttrView
from .ops.registry import registry

@contextmanager
def WhiteboardMode():
    """Isolate a fresh autograd tape for a single forward/backward pair."""
    old = autograd.tape
    autograd.tape = GradTape()
    try:
        yield
    finally:
        autograd.tape = old

def run_op_and_grads_cached(
    sys,
    op_name: str,
    src_ids: Sequence[int],
    *,
    scale: float = 1.0,
    residual: float | None = None,
    weight: str | None = None,
    cache: WhiteboardCache | None = None,
) -> Tuple[float, Tuple[float, ...]]:
    cache = cache or WhiteboardCache()
    sigs = [ParamSig(i, getattr(sys.nodes[i], "version", 0)) for i in src_ids]
    view = NodeAttrView(sys.nodes, "theta", indices=src_ids).build()
    tensor = view.tensor
    key = cache.make_key(
        op_name,
        sigs,
        fan_in=len(src_ids),
        feat_shape=tensor.shape[1:],
        scale=scale,
        residual=residual,
        weight=weight,
    )
    pkg = cache.get(key)
    if pkg is not None:
        return pkg
    with WhiteboardMode():
        fn = registry.get(op_name)
        y_val = fn(tensor)
        if op_name == "add" or op_name == "sum":
            grads = tuple(1.0 for _ in src_ids)
        elif op_name == "mul":
            grads = tuple(float(y_val / v) if v != 0 else 0.0 for v in tensor)
        else:
            grads = tuple(0.0 for _ in src_ids)
        pkg = (float(y_val), grads)
    cache.put(key, pkg)
    return pkg
