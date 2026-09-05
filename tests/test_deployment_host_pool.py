"""Host deployment pool: frame semantics, joins, failure containment."""

from __future__ import annotations

import threading

import pytest

from src.compiler.deployment_frame import DeploymentJoin
from src.compiler.deployment_host_pool import (
    HostDeploymentPool,
    LaneExecutionError,
)


def test_barrier_join_returns_results_indexed_by_lane():
    with HostDeploymentPool(workers=3) as pool:
        results = pool.deploy([
            (lambda lane=lane: lane * lane) for lane in range(8)
        ])
    assert results == (0, 1, 4, 9, 16, 25, 36, 49)


def test_zero_workers_is_the_serial_fallback_through_the_same_path():
    with HostDeploymentPool(workers=0) as pool:
        assert pool.worker_count == 0
        results = pool.deploy([lambda: "a", lambda: "b"])
    assert results == ("a", "b")


def test_pool_is_persistent_across_frames():
    with HostDeploymentPool(workers=2) as pool:
        first = pool.deploy([lambda: 1, lambda: 2])
        second = pool.deploy([lambda: 3, lambda: 4, lambda: 5])
    assert first == (1, 2)
    assert second == (3, 4, 5)


def test_reduce_join_folds_with_the_canonical_operator():
    join = DeploymentJoin(
        mode="reduce", reduction_operator="Add", allow_reassociation=True,
    )
    with HostDeploymentPool(workers=2) as pool:
        total = pool.deploy(
            [(lambda lane=lane: lane) for lane in range(10)], join=join,
        )
    assert total == sum(range(10))


def test_unlicensed_reduce_still_folds_in_lane_order():
    join = DeploymentJoin(mode="reduce", reduction_operator="Add")
    with HostDeploymentPool(workers=2) as pool:
        total = pool.deploy([lambda: 1.5, lambda: 2.5, lambda: 3.0], join=join)
    assert total == 7.0


def test_product_join_is_refused_visibly():
    with HostDeploymentPool(workers=1) as pool:
        with pytest.raises(ValueError, match="PRODUCT"):
            pool.deploy([lambda: 1], join=DeploymentJoin(mode="product"))


def test_lane_failure_names_the_lane_and_settles_the_frame():
    def boom():
        raise RuntimeError("lane exploded")

    with HostDeploymentPool(workers=2) as pool:
        with pytest.raises(LaneExecutionError) as caught:
            pool.deploy([lambda: 0, boom, lambda: 2])
        assert caught.value.lane_index == 1
        # The frame settled; the pool still serves the next deployment.
        assert pool.deploy([lambda: "recovered"]) == ("recovered",)


def test_deploy_span_covers_every_element_exactly_once():
    total = 1003
    counts = [0] * total
    lock = threading.Lock()

    def kernel(start: int, stop: int) -> None:
        with lock:
            for index in range(start, stop):
                counts[index] += 1

    with HostDeploymentPool(workers=4) as pool:
        chunks = pool.deploy_span(kernel, total, chunk=64)
    assert chunks == -(-total // 64)
    assert all(count == 1 for count in counts)


def test_real_thread_overlap_when_the_gil_is_released():
    # time.sleep releases the GIL like a ctypes call over native code does;
    # four sleeping lanes on four workers must overlap, not serialize.
    import time

    with HostDeploymentPool(workers=4) as pool:
        started = time.monotonic()
        pool.deploy([(lambda: time.sleep(0.2)) for _ in range(4)])
        elapsed = time.monotonic() - started
    assert elapsed < 0.6, f"lanes serialized: {elapsed:.2f}s for 4 x 0.2s"
