from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Any, Optional, Iterable
import math

_DEF_Q = 1e-6

def _quantise(x: Optional[float], q: float = _DEF_Q) -> float:
    if x is None:
        return math.nan
    return float(round(x / q) * q)

@dataclass(frozen=True)
class ParamSig:
    node_id: int
    version: int

@dataclass(frozen=True)
class OpKey:
    op_name: str
    fan_in: int
    feat_shape: Tuple[int, ...]
    weight: str | None
    scale: float
    residual: float
    params: Tuple[ParamSig, ...]

class WhiteboardCache:
    """Naive in-memory cache for (forward, grads) packages."""

    def __init__(self, store: Optional[Dict[OpKey, Tuple[Any, Tuple[Any, ...]]]] = None):
        self._store = store if store is not None else {}
        self.hits = 0
        self.misses = 0

    def make_key(
        self,
        op_name: str,
        param_sigs: Sequence[ParamSig],
        *,
        fan_in: int,
        feat_shape: Iterable[int] = (),
        weight: str | None = None,
        scale: float = 1.0,
        residual: Optional[float] = None,
    ) -> OpKey:
        s = _quantise(scale)
        r = _quantise(residual)
        return OpKey(op_name, fan_in, tuple(feat_shape), weight, s, r, tuple(param_sigs))

    def get(self, key: OpKey):
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, key: OpKey, value):
        self._store[key] = value
