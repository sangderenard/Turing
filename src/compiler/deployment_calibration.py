"""Measured deployment calibration: test the water, then earn the default.

``deployment_lowering`` decides what is *legal* per backend; this module
decides what is *fast* per machine, by measurement rather than belief.  The
premise comes from the repository's own performance handoff: whether a pool
helps depends on the workload's work/byte ratio and the host's memory
bandwidth -- an elementwise chain can saturate DRAM at two threads while a
compute-bound chain scales to the core count.  No static table can know
that; a two-minute probe on the actual machine can.

The water-testing protocol:

1. ``probe_span``/``probe_lanes`` run the same workload serially and
   against a ladder of pool sizes (best-of-k on a monotonic clock, warmup
   discarded), and return a ``CalibrationVerdict`` naming the winner.
2. Promotion needs a real margin: the pool must beat serial by
   ``PROMOTE_THRESHOLD`` (default 1.15x) before a verdict says ``pool`` --
   measurement noise and marginal wins stay serial, so flapping cannot
   start.
3. Verdicts persist through ``RepositoryArtifactCache`` (atomic writes,
   identity-validated), keyed by the workload signature and stamped with a
   machine fingerprint.  A verdict from a different machine is not
   evidence: lookups ignore it and the workload re-probes.
4. ``select_deployment_strategy(..., calibration=verdict)`` folds the
   measurement into the decision: a measured-slower pool degrades to
   serial with the measured ratio in the reason trail, and a measured
   winner carries its best worker count with it.

The commitment path is then gradual and reversible by construction:
*shadow* (probe and record, deploy serially), *measured* (deploy what the
verdict says -- the recommended default), *committed* (skip probing where
verdicts are stable).  At every stage an absent, stale, or foreign verdict
means the serial baseline -- the total fallback the whole layer is built
on -- so the system can only ever be wrong in the safe direction.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..common.tensors.accelerator_backends.artifact_cache import (
    RepositoryArtifactCache,
)
from .deployment_host_pool import HostDeploymentPool

PROMOTE_THRESHOLD = 1.15
_CACHE_TARGET = "deployment-calibration"


def machine_fingerprint() -> str:
    """Identity of the measuring machine; foreign verdicts are not evidence."""

    return "|".join((
        platform.system(),
        platform.machine(),
        platform.processor() or "unknown-cpu",
        f"cores={os.cpu_count() or 0}",
    ))


@dataclass(frozen=True)
class WorkloadSignature:
    """What was measured: enough identity to recognize the workload again.

    ``work`` is the scale axis (elements for a span, lanes for a lane
    frame); ``identity`` is an optional digest of the kernel/region so two
    different programs of the same size do not share a verdict.
    """

    backend: str
    kind: str  # "span" | "lanes"
    work: int
    identity: str = ""

    def record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "kind": self.kind,
            "work": int(self.work),
            "identity": self.identity,
        }


@dataclass(frozen=True)
class CalibrationVerdict:
    """One measured comparison, with everything needed to audit it later."""

    signature: WorkloadSignature
    machine: str
    best_strategy: str  # "serial" | "pool"
    best_workers: int
    speedup: float
    serial_seconds: float
    best_seconds: float
    samples: int

    def as_record(self) -> dict[str, Any]:
        return {
            "signature": self.signature.record(),
            "machine": self.machine,
            "best_strategy": self.best_strategy,
            "best_workers": int(self.best_workers),
            "speedup": float(self.speedup),
            "serial_seconds": float(self.serial_seconds),
            "best_seconds": float(self.best_seconds),
            "samples": int(self.samples),
        }

    @staticmethod
    def from_record(record: Mapping[str, Any]) -> "CalibrationVerdict":
        signature = record["signature"]
        return CalibrationVerdict(
            signature=WorkloadSignature(
                backend=str(signature["backend"]),
                kind=str(signature["kind"]),
                work=int(signature["work"]),
                identity=str(signature.get("identity", "")),
            ),
            machine=str(record["machine"]),
            best_strategy=str(record["best_strategy"]),
            best_workers=int(record["best_workers"]),
            speedup=float(record["speedup"]),
            serial_seconds=float(record["serial_seconds"]),
            best_seconds=float(record["best_seconds"]),
            samples=int(record["samples"]),
        )


class CalibrationStore:
    """Persisted verdicts on the repository artifact cache's terms."""

    def __init__(self, cache: RepositoryArtifactCache | None = None):
        self._cache = cache or RepositoryArtifactCache()

    def lookup(
        self,
        signature: WorkloadSignature,
        *,
        machine: str | None = None,
    ) -> CalibrationVerdict | None:
        import json

        cached = self._cache.load(
            _CACHE_TARGET, signature.record(), suffix=".verdict.json",
        )
        if cached is None:
            return None
        try:
            verdict = CalibrationVerdict.from_record(json.loads(cached.source))
        except (KeyError, TypeError, ValueError):
            return None
        expected = machine or machine_fingerprint()
        if verdict.machine != expected:
            return None  # foreign machine: not evidence, re-probe
        return verdict

    def store(self, verdict: CalibrationVerdict) -> None:
        import json

        self._cache.store(
            _CACHE_TARGET,
            verdict.signature.record(),
            json.dumps(verdict.as_record(), sort_keys=True, indent=2),
            suffix=".verdict.json",
        )


def _best_of(runs: int, thunk: Callable[[], None]) -> float:
    best = float("inf")
    for _ in range(runs):
        started = time.perf_counter()
        thunk()
        best = min(best, time.perf_counter() - started)
    return best


def _default_ladder() -> tuple[int, ...]:
    ceiling = max(1, (os.cpu_count() or 2) - 1)
    ladder = []
    workers = 1
    while workers <= ceiling:
        ladder.append(workers)
        workers *= 2
    if ladder and ladder[-1] != ceiling:
        ladder.append(ceiling)
    return tuple(ladder)


def _probe(
    signature: WorkloadSignature,
    serial_run: Callable[[], None],
    parallel_run: Callable[[HostDeploymentPool], None],
    *,
    worker_ladder: Sequence[int] | None = None,
    repeats: int = 3,
    promote_threshold: float = PROMOTE_THRESHOLD,
) -> CalibrationVerdict:
    repeats = max(1, int(repeats))
    serial_run()  # warmup
    serial_seconds = _best_of(repeats, serial_run)

    best_seconds = float("inf")
    best_workers = 0
    for workers in worker_ladder or _default_ladder():
        pool = HostDeploymentPool(workers=int(workers))
        try:
            parallel_run(pool)  # warmup
            seconds = _best_of(repeats, lambda: parallel_run(pool))
        finally:
            pool.close()
        if seconds < best_seconds:
            best_seconds = seconds
            best_workers = int(workers)

    speedup = (
        serial_seconds / best_seconds if best_seconds > 0 else float("inf")
    )
    promoted = speedup >= promote_threshold
    return CalibrationVerdict(
        signature=signature,
        machine=machine_fingerprint(),
        best_strategy="pool" if promoted else "serial",
        best_workers=best_workers if promoted else 0,
        speedup=speedup,
        serial_seconds=serial_seconds,
        best_seconds=best_seconds,
        samples=repeats,
    )


def probe_span(
    kernel: Callable[[int, int], None],
    total: int,
    *,
    backend: str = "python",
    identity: str = "",
    chunk: int | None = None,
    worker_ladder: Sequence[int] | None = None,
    repeats: int = 3,
    promote_threshold: float = PROMOTE_THRESHOLD,
) -> CalibrationVerdict:
    """Measure element-range splitting of ``kernel`` over ``total`` items."""

    signature = WorkloadSignature(
        backend=backend, kind="span", work=int(total), identity=identity,
    )
    return _probe(
        signature,
        lambda: kernel(0, int(total)),
        lambda pool: pool.deploy_span(kernel, int(total), chunk=chunk),
        worker_ladder=worker_ladder,
        repeats=repeats,
        promote_threshold=promote_threshold,
    )


def probe_lanes(
    lanes: Sequence[Callable[[], Any]],
    *,
    backend: str = "python",
    identity: str = "",
    worker_ladder: Sequence[int] | None = None,
    repeats: int = 3,
    promote_threshold: float = PROMOTE_THRESHOLD,
) -> CalibrationVerdict:
    """Measure lane-parallel execution of independent thunks."""

    frozen = list(lanes)
    signature = WorkloadSignature(
        backend=backend, kind="lanes", work=len(frozen), identity=identity,
    )

    def serial_run() -> None:
        for lane in frozen:
            lane()

    return _probe(
        signature,
        serial_run,
        lambda pool: pool.deploy(frozen),
        worker_ladder=worker_ladder,
        repeats=repeats,
        promote_threshold=promote_threshold,
    )


def calibrated_verdict(
    signature: WorkloadSignature,
    store: CalibrationStore,
    prober: Callable[[], CalibrationVerdict] | None = None,
) -> CalibrationVerdict | None:
    """Look up a verdict; probe-and-store when absent and a prober is given.

    The one-call path a shell uses for the *measured* policy: the first
    encounter with a workload pays the probe, every later encounter reads
    the cache, and a machine change invalidates automatically.
    """

    found = store.lookup(signature)
    if found is not None:
        return found
    if prober is None:
        return None
    verdict = prober()
    store.store(verdict)
    return verdict


__all__ = [
    "PROMOTE_THRESHOLD",
    "CalibrationStore",
    "CalibrationVerdict",
    "WorkloadSignature",
    "calibrated_verdict",
    "machine_fingerprint",
    "probe_lanes",
    "probe_span",
]
