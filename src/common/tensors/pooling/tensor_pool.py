from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict, deque
from threading import Lock
from typing import Any, Callable, Deque, Dict, Optional, Tuple

# ---------- Backend contract ----------
# Provide an object with these callables:
#   empty(shape: Tuple[int, ...], dtype: Any, device: Any) -> Any
#   fill0_(buf: Any) -> None
#   nbytes(buf: Any) -> int
#   detach(buf: Any) -> Any        # must break autograd/tape ties (identity if none)
#
# No NumPy here; the backend is your AbstractTensor backend (or similar).

@dataclass
class PoolPolicy:
    round_shape: Callable[[Tuple[int, ...]], Tuple[int, ...]] = lambda s: s
    clear_on_acquire: bool = False
    clear_on_release: bool = False
    hard_cap_bytes: Optional[int] = None
    ewma_alpha: float = 0.2
    min_ready: int = 1
    max_ready: int = 64


class TensorPool:
    """Preallocator keyed by ``(shape, dtype, device)``.

    - Backend-agnostic: all allocations go through the provided backend.
    - Thread-safe per-bucket and globally.
    - EWMA-based prewarm targets via `observe(...)` + `maintenance_tick()`.
    - Optional hard byte cap with simple eviction.
    """

    def __init__(self, *, backend: Any, policy: PoolPolicy | None = None) -> None:
        if backend is None:
            raise ValueError("TensorPool requires a backend with empty/fill0_/nbytes/detach.")
        self.B = backend
        self.P = policy or PoolPolicy()

        # key -> deque[buf]
        self._buckets: Dict[Tuple[Tuple[int, ...], Any, Any], Deque[Any]] = defaultdict(deque)
        self._locks: Dict[Tuple[Tuple[int, ...], Any, Any], Lock] = defaultdict(Lock)

        # target ready count per key (EWMA of recent demand)
        self._ready_target: Dict[Tuple[Tuple[int, ...], Any, Any], float] = defaultdict(float)

        # bookkeeping
        self._global_lock = Lock()
        self._bytes_in_pool: int = 0

        # registry of live buffers issued by acquire(): id(buf) -> (key, nbytes)
        self._live: Dict[int, Tuple[Tuple[Tuple[int, ...], Any, Any], int]] = {}

    # ----- public API -----

    def acquire(self, shape: Tuple[int, ...], *, dtype=None, device=None, clear: Optional[bool] = None) -> Any:
        """Return a tensor buffer with the requested specification."""
        key = self._key(shape, dtype, device)
        dq = self._buckets[key]
        self._observe_key(key, k=1)

        with self._locks[key]:
            if dq:
                buf = dq.popleft()
                # bytes_in_pool decreases when we pop from pool into live set
                with self._global_lock:
                    self._bytes_in_pool -= self.B.nbytes(buf)
            else:
                buf = self.B.empty(key[0], key[1], key[2])

        # ensure detached from any prior tape graph
        buf = self.B.detach(buf)

        do_clear = self.P.clear_on_acquire if clear is None else clear
        if do_clear:
            self.B.fill0_(buf)

        # register as live
        nb = self.B.nbytes(buf)
        self._live[id(buf)] = (key, nb)
        return buf

    def release(self, buf: Any) -> None:
        """Return a buffer to the pool."""
        meta = self._live.pop(id(buf), None)
        if meta is None:
            # Not from this pool; do not accept to avoid key mismatch.
            raise ValueError("Attempted to release a buffer not acquired from this TensorPool.")
        key, nb = meta
        dq = self._buckets[key]

        # detach again for safety before pooling
        buf = self.B.detach(buf)

        do_clear = self.P.clear_on_release
        if do_clear:
            self.B.fill0_(buf)

        with self._locks[key]:
            dq.append(buf)
            with self._global_lock:
                self._bytes_in_pool += nb
                self._maybe_evict_over_cap()

    def observe(self, shape: Tuple[int, ...], *, dtype=None, device=None) -> None:
        """Record allocation statistics for pre-warming."""
        key = self._key(shape, dtype, device)
        self._observe_key(key, k=1)

    # ----- optional helpers (call from your tick or scheduler) -----

    def maintenance_tick(self) -> None:
        """Prewarm buckets toward EWMA targets and enforce the byte cap."""
        # prewarm per key
        for key, tgt in list(self._ready_target.items()):
            want = int(max(self.P.min_ready, min(self.P.max_ready, round(tgt))))
            dq = self._buckets[key]
            with self._locks[key]:
                have = len(dq)
                need = max(0, want - have)
                for _ in range(need):
                    buf = self.B.empty(key[0], key[1], key[2])
                    dq.append(buf)
                    with self._global_lock:
                        self._bytes_in_pool += self.B.nbytes(buf)
        with self._global_lock:
            self._maybe_evict_over_cap()

    # ----- internals -----

    def _key(self, shape: Tuple[int, ...], dtype: Any, device: Any) -> Tuple[Tuple[int, ...], Any, Any]:
        shp = tuple(int(x) for x in shape)
        return (self.P.round_shape(shp), dtype, device)

    def _observe_key(self, key: Tuple[Tuple[int, ...], Any, Any], *, k: int) -> None:
        # smooth demand: ready_target = (1-a)*old + a*k
        a = self.P.ewma_alpha
        self._ready_target[key] = max(
            self.P.min_ready,
            min(self.P.max_ready, (1.0 - a) * self._ready_target[key] + a * k),
        )

    def _maybe_evict_over_cap(self) -> None:
        cap = self.P.hard_cap_bytes
        if cap is None:
            return
        if self._bytes_in_pool <= cap:
            return
        # naive eviction loop: walk buckets and pop until under cap
        for key, dq in list(self._buckets.items()):
            with self._locks[key]:
                while dq and self._bytes_in_pool > cap:
                    buf = dq.pop()
                    self._bytes_in_pool -= self.B.nbytes(buf)
            if self._bytes_in_pool <= cap:
                break
