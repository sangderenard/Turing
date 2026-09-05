"""Calibration: probing, persistence, and measured strategy selection.

Timing-dependent assertions are structural (fields consistent, winner
recorded, cache round-trips), never about absolute speed -- CI machines
may not show speedups and the layer must stay deterministic there.
"""

from __future__ import annotations

import pytest

from src.common.tensors.accelerator_backends.artifact_cache import (
    RepositoryArtifactCache,
)
from src.compiler.deployment_calibration import (
    CalibrationStore,
    CalibrationVerdict,
    WorkloadSignature,
    calibrated_verdict,
    machine_fingerprint,
    probe_lanes,
    probe_span,
)
from src.compiler.deployment_lowering import (
    POOL,
    SERIAL,
    select_deployment_strategy,
)


def _verdict(
    *,
    best_strategy: str,
    speedup: float,
    workers: int = 4,
    machine: str | None = None,
) -> CalibrationVerdict:
    return CalibrationVerdict(
        signature=WorkloadSignature(
            backend="c", kind="span", work=100_000, identity="k1",
        ),
        machine=machine or machine_fingerprint(),
        best_strategy=best_strategy,
        best_workers=workers if best_strategy == "pool" else 0,
        speedup=speedup,
        serial_seconds=1.0,
        best_seconds=1.0 / max(speedup, 1e-9),
        samples=3,
    )


def test_probe_span_produces_a_consistent_verdict():
    touched = []

    def kernel(start: int, stop: int) -> None:
        touched.append((start, stop))

    verdict = probe_span(
        kernel, 256, backend="python", identity="probe-test",
        worker_ladder=(1, 2), repeats=2,
    )
    assert verdict.signature.kind == "span"
    assert verdict.signature.work == 256
    assert verdict.machine == machine_fingerprint()
    assert verdict.best_strategy in ("serial", "pool")
    assert verdict.serial_seconds >= 0.0
    assert verdict.best_seconds >= 0.0
    assert verdict.speedup == pytest.approx(
        verdict.serial_seconds / verdict.best_seconds, rel=1e-6,
    )
    assert touched  # the kernel really ran


def test_probe_lanes_runs_every_lane():
    counts = [0, 0, 0]

    def lane(index: int):
        def run():
            counts[index] += 1
        return run

    verdict = probe_lanes(
        [lane(0), lane(1), lane(2)],
        worker_ladder=(2,), repeats=1,
    )
    assert verdict.signature.kind == "lanes"
    assert verdict.signature.work == 3
    assert all(count >= 2 for count in counts)  # serial + parallel at least


def test_marginal_speedup_stays_serial_by_hysteresis():
    # A fabricated 1.05x "win" is below the promotion threshold; verify via
    # selection, which is where the verdict's judgment lands.
    verdict = _verdict(best_strategy="serial", speedup=1.05, workers=0)
    choice = select_deployment_strategy(
        backend="c", execution_class="thread-workers",
        calibration=verdict,
    )
    assert choice.strategy == SERIAL
    assert any("demoted by calibration" in reason for reason in choice.reasons)


def test_store_round_trips_and_rejects_foreign_machines(tmp_path):
    cache = RepositoryArtifactCache(root=tmp_path)
    store = CalibrationStore(cache)
    mine = _verdict(best_strategy="pool", speedup=2.0)
    store.store(mine)
    assert store.lookup(mine.signature) == mine

    foreign = _verdict(
        best_strategy="pool", speedup=3.0, machine="someone-else|arm64",
    )
    store.store(foreign)  # overwrites the slot with a foreign measurement
    assert store.lookup(foreign.signature) is None  # not evidence here


def test_calibrated_verdict_probes_once_then_reads_the_cache(tmp_path):
    cache = RepositoryArtifactCache(root=tmp_path)
    store = CalibrationStore(cache)
    signature = WorkloadSignature(
        backend="python", kind="span", work=64, identity="cv",
    )
    probes = []

    def prober() -> CalibrationVerdict:
        probes.append(1)
        return probe_span(
            lambda start, stop: None, 64,
            backend="python", identity="cv",
            worker_ladder=(1,), repeats=1,
        )

    first = calibrated_verdict(signature, store, prober)
    second = calibrated_verdict(signature, store, prober)
    assert first is not None and second is not None
    assert len(probes) == 1  # second call was a cache read
    assert calibrated_verdict(signature, CalibrationStore(
        RepositoryArtifactCache(root=tmp_path / "empty")
    )) is None  # no prober, no verdict: shadow mode stays serial


def test_measured_winner_carries_its_worker_count_into_the_choice():
    verdict = _verdict(best_strategy="pool", speedup=2.4, workers=6)
    choice = select_deployment_strategy(
        backend="c", execution_class="thread-workers",
        calibration=verdict,
    )
    assert choice.strategy == POOL
    assert choice.workers == 6
    assert any("2.40x" in reason for reason in choice.reasons)


def test_calibration_never_overrides_a_legality_veto():
    # A glowing measurement cannot make a reduce join pooled on the C
    # backend, whose parallel set is barrier-only.
    verdict = _verdict(best_strategy="pool", speedup=8.0)
    choice = select_deployment_strategy(
        backend="c", execution_class="thread-workers", join_mode="reduce",
        calibration=verdict,
    )
    assert choice.strategy == SERIAL
    assert any("join mode 'reduce'" in reason for reason in choice.reasons)
