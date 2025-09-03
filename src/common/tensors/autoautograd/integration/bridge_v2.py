from __future__ import annotations
from typing import Sequence

import numpy as np

from ..whiteboard_runtime import run_op_and_grads_cached
from ..whiteboard_cache import WhiteboardCache


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
    """Vectorised op call that pushes impulses using NodeAttrView and caching."""
    if weight == "inv_length":
        po = sys.nodes[out_id].p
        ws = [1.0 / max(np.linalg.norm(po - sys.nodes[i].p), 1e-8) for i in src_ids]
        scale *= float(np.mean(ws)) if ws else 1.0
    y, grads = run_op_and_grads_cached(
        sys,
        op_name,
        src_ids,
        scale=scale,
        residual=residual,
        weight=weight,
        cache=cache,
    )
    if residual is not None:
        for i, g in zip(src_ids, grads):
            sys.impulse(i, out_id, op_name, scale * g * float(-residual))
    return y
