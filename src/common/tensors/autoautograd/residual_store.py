from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Iterable, Optional

from ..abstraction import AbstractTensor


class Space(Enum):
    """Discrete residual spaces used in the whiteboard pipeline."""

    F = auto()
    G = auto()
    TH = auto()


@dataclass
class ResidualItem:
    """Single residual entry."""

    value: AbstractTensor
    width: int
    axis: Optional[int] = None


class ResidualStore:
    """Container grouping residuals by ``Space`` and node id/axis."""

    def __init__(self) -> None:
        self._data: Dict[Space, Dict[int, Dict[Optional[int], ResidualItem]]] = {
            space: {} for space in Space
        }

    # ------------------------------------------------------------------
    def add(
        self,
        nid: int,
        value: AbstractTensor,
        *,
        space: Space,
        width: int,
        axis: Optional[int] = None,
    ) -> None:
        """Accumulate a residual item for ``nid`` in ``space``."""

        bucket = self._data[space].setdefault(nid, {})
        if axis in bucket:
            prev = bucket[axis]
            if getattr(prev.value, "shape", None) == getattr(value, "shape", None):
                new_val = prev.value + value
            else:
                extra = tuple(range(value.ndim - getattr(prev.value, "ndim", 0)))
                new_val = prev.value + value.sum(dim=extra).reshape(prev.value.shape)
            bucket[axis] = ResidualItem(new_val, width, axis)
        else:
            bucket[axis] = ResidualItem(value, width, axis)

    # ------------------------------------------------------------------
    def put(
        self,
        nid: int,
        value: AbstractTensor,
        *,
        space: Space,
        width: int,
        axis: Optional[int] = None,
    ) -> None:
        """Store residual for ``nid`` overwriting existing entry on ``axis``."""

        bucket = self._data[space].setdefault(nid, {})
        bucket[axis] = ResidualItem(value, width, axis)

    # ------------------------------------------------------------------
    def get_bucket(self, space: Space) -> Dict[int, Dict[Optional[int], ResidualItem]]:
        """Return mapping of node id to axis -> residuals for ``space``."""

        return self._data[space]

    # ------------------------------------------------------------------
    def get(self, nid: int, space: Optional[Space] = None):
        """Fetch aggregated residual for ``nid`` optionally scoped to ``space``."""

        def _combine(items):
            val = None
            for item in items.values():
                if val is None:
                    val = item.value
                else:
                    if getattr(val, "shape", None) == getattr(item.value, "shape", None):
                        val = val + item.value
                    else:
                        extra = tuple(range(item.value.ndim - getattr(val, "ndim", 0)))
                        val = val + item.value.sum(dim=extra).reshape(val.shape)
            return val

        if space is not None:
            items = self._data[space].get(nid)
            return _combine(items) if items else None

        val = None
        for bucket in self._data.values():
            items = bucket.get(nid)
            if not items:
                continue
            v = _combine(items)
            if val is None:
                val = v
            else:
                if getattr(val, "shape", None) == getattr(v, "shape", None):
                    val = val + v
                else:
                    extra = tuple(range(v.ndim - getattr(val, "ndim", 0)))
                    val = val + v.sum(dim=extra).reshape(val.shape)
        return val

    # ------------------------------------------------------------------
    def iter_values(self) -> Iterable[AbstractTensor]:
        for bucket in self._data.values():
            for items in bucket.values():
                for item in items.values():
                    yield item.value

    # ------------------------------------------------------------------
    def any(self) -> bool:
        return any(bool(bucket) for bucket in self._data.values())
