from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

# ---- quantization for real-valued knobs (prevents micro-key explosions) ----
_DEF_Q = 1e-6

def _q(x: Optional[float], q: float = _DEF_Q) -> float:
    if x is None:
        # Treat absence of residual/scale as exact zero for key stability.
        # Using NaN would make keys non-equal across identical requests.
        return 0.0
    # avoid rounding artifacts by working in float
    return float(round(x / q) * q)


@dataclass(frozen=True)
class ParamSig:
    """Node-local causality token for caching."""
    node_id: int
    version: int  # incremented on commit


@dataclass(frozen=True)
class OpKey:
    """
    Stable per-job cache key:
      - op_name
      - fan_in k
      - input feature shape (tuple; () for scalar)
      - weight policy tag (string or None)
      - quantized scale & residual
      - ordered param signatures aligned to src_ids
      - optional backend tag (to avoid cross-backend reuse)
    """
    op: str
    k: int
    feat_shape: Tuple[int, ...]
    weight: Optional[str]
    scale_q: float
    residual_q: float
    params: Tuple[ParamSig, ...]
    backend_tag: Optional[Any] = None


class WhiteboardCache:
    """
    Naive in-memory cache for (forward, per-source grads) packages.
    Store types are backend-agnostic as long as they are picklable or small.
    """

    def __init__(self) -> None:
        self._store: Dict[OpKey, Tuple[Any, Tuple[float, ...]]] = {}
        self.hits = 0
        self.misses = 0

    def make_key(
        self,
        *,
        op_name: str,
        src_ids: Sequence[int],
        versions: Sequence[int],
        feat_shape: Sequence[int] | Tuple[int, ...],
        weight: Optional[str],
        scale: float,
        residual: Optional[float],
        backend_tag: Optional[Any] = None,
    ) -> OpKey:
        if len(src_ids) != len(versions):
            raise ValueError("src_ids and versions length mismatch")
        k = len(src_ids)
        fs = tuple(int(d) for d in tuple(feat_shape))
        params = tuple(ParamSig(int(i), int(v)) for i, v in zip(src_ids, versions))
        return OpKey(
            op=op_name,
            k=k,
            feat_shape=fs,
            weight=weight,
            scale_q=_q(scale),
            residual_q=_q(residual),
            params=params,
            backend_tag=backend_tag,
        )

    def get(self, key: OpKey) -> Optional[Tuple[Any, Tuple[float, ...]]]:
        pkg = self._store.get(key)
        if pkg is not None:
            self.hits += 1
        else:
            self.misses += 1
        return pkg

    def put(self, key: OpKey, pkg: Tuple[Any, Tuple[float, ...]]) -> None:
        self._store[key] = pkg
