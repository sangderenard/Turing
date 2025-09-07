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
    """Container grouping residuals by ``Space`` and node id."""

    def __init__(self) -> None:
        self._data: Dict[Space, Dict[int, ResidualItem]] = {space: {} for space in Space}

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

        bucket = self._data[space]
        if nid in bucket:
            prev = bucket[nid]
            if getattr(prev.value, "shape", None) == getattr(value, "shape", None):
                new_val = prev.value + value
            else:
                extra = tuple(range(value.ndim - getattr(prev.value, "ndim", 0)))
                new_val = prev.value + value.sum(dim=extra).reshape(prev.value.shape)
            bucket[nid] = ResidualItem(new_val, width, axis if axis is not None else prev.axis)
        else:
            bucket[nid] = ResidualItem(value, width, axis)

    # ------------------------------------------------------------------
    def get_bucket(self, space: Space) -> Dict[int, ResidualItem]:
        """Return mapping of node id to residuals for ``space``."""

        return self._data[space]

    # ------------------------------------------------------------------
    def get(self, nid: int, space: Optional[Space] = None):
        """Fetch residual for ``nid`` optionally scoped to ``space``."""

        if space is not None:
            item = self._data[space].get(nid)
            return item.value if item else None

        val = None
        for bucket in self._data.values():
            item = bucket.get(nid)
            if not item:
                continue
            if val is None:
                val = item.value
            else:
                if getattr(val, "shape", None) == getattr(item.value, "shape", None):
                    val = val + item.value
                else:
                    extra = tuple(range(item.value.ndim - getattr(val, "ndim", 0)))
                    val = val + item.value.sum(dim=extra).reshape(val.shape)
        return val

    # ------------------------------------------------------------------
    def iter_values(self) -> Iterable[AbstractTensor]:
        for bucket in self._data.values():
            for item in bucket.values():
                yield item.value

    # ------------------------------------------------------------------
    def any(self) -> bool:
        return any(bool(bucket) for bucket in self._data.values())
