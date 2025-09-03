from __future__ import annotations

from typing import Sequence, Optional

from ..autoautograd.whiteboard_runtime import run_op_and_grads_cached
from ..scheduling.results import OpResult, ResultSink


def push_impulses_from_op_v2(
    sys,
    op: str,
    src_ids: Sequence[int],
    out_id: int,
    *,
    residual: Optional[float],
    scale: float,
    weight: str,
    result_sink: Optional[ResultSink] = None,
) -> float:
    """Cached whiteboard path for operator execution.

    This bridge should gather node attributes, compute the operator's forward
    value and local gradients (using caching when possible), push impulses into
    ``sys`` and optionally publish an ``OpResult``.
    """
    # TODO: Implement the bridge using ``run_op_and_grads_cached``.
    raise NotImplementedError

