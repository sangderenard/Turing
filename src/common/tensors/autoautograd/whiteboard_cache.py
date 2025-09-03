from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Hashable, Protocol


@dataclass(frozen=True)
class ParamSig:
    """Node-local causality token for caching.

    Tracks the node identifier and either a version number or a value digest.
    """

    node_id: int
    attr: str
    version: Optional[int] = None
    value_digest8: Optional[bytes] = None


@dataclass(frozen=True)
class OpKey:
    """Stable cache key for a local op.

    Composes the operator identity with shape information, weighting policy,
    quantized scale and residual values, and an ordered tuple of ``ParamSig``
    instances aligned with the source identifiers.
    """

    op: str
    k: int
    F: int
    weight: str
    scale_q: int
    residual_q: int
    params: Tuple[ParamSig, ...]


class CacheBackend(Protocol):
    """Key→package storage with bounded footprint."""

    def get(self, key: OpKey) -> Optional[Tuple[Any, Any]]:  # pragma: no cover - protocol stub
        """Retrieve a cached package if available."""

    def put(self, key: OpKey, y: Any, g: Any) -> None:  # pragma: no cover - protocol stub
        """Store a cache package."""


class WhiteboardCache:
    """Hashed and versioned package cache.

    This class is a light wrapper that constructs cache keys and delegates
    storage duties to a provided backend implementation.
    """

    def __init__(self, store: CacheBackend) -> None:
        """Initialize the cache with a backend store."""
        # TODO: Wire up the backend store and any bookkeeping structures.
        raise NotImplementedError

    def make_key(
        self,
        *,
        op: str,
        src_ids: Sequence[int],
        F: int,
        weight: str,
        scale: float,
        residual: float,
        get_attr: callable,
        get_attr_version: Optional[callable] = None,
        attr_name: str = "theta",
    ) -> OpKey:
        """Build an ``OpKey`` for a job.

        The key incorporates node-local versions or value digests to maintain
        cache correctness without reference to global epochs.
        """
        # TODO: Implement key construction with quantization and version lookups.
        raise NotImplementedError

    def get(self, key: OpKey) -> Optional[Tuple[Any, Any]]:
        """Return a cached package if present."""
        # TODO: Delegate to the backend store.
        raise NotImplementedError

    def put(self, key: OpKey, y: Any, g: Any) -> None:
        """Insert a package into the cache."""
        # TODO: Delegate to the backend store.
        raise NotImplementedError

