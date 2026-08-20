"""Tiled composition of prebaked kernels, and the routing that owns it.

The aim (owner's): the compiler takes a CUSTOM size and exploits a tiling
algorithm that uses only the PREBAKED operators at peak efficiency. The
labor division under test here (``docs/KERNEL_BANK_DESIGN.md``):

* the BANK owns variants and verified admission;
* the LAUNCH COORDINATOR owns per-call routing, where tiling is one route
  in one ladder -- exact-size specialized > tiled > parametric >
  reference -- not a rival orchestrator;
* the DEPLOYMENT machinery owns cross-call scheduling, and will one day
  lower the tile plan natively, replacing the composer's host loop but
  not the coordinator.

Also pinned here: the buffer-capture defect this work found in
``CompiledVariant.run`` -- the execution-creation path bound caller arrays
by REFERENCE while the reuse path copied, so a caller passing a numpy VIEW
(what a tiling composer naturally passes) had its memory silently
overwritten by the next same-signature call's inputs. Every tile call was
individually exact while the assembled result was wrong -- a
composition-only corruption no single-call test could see.

Compile hygiene: tile=16 keeps the specialized core's build cheap while
staying ABOVE the loop unroll limit (8), below which specialization is
separately defective (``test_compiled_linalg.py``). The bank lives in a
tmp directory so test builds never mix with the working bank.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.compiler.kernel_bank import (
    KernelBank,
    LaunchCoordinator,
    blas_kernel_specs,
)
from src.compiler.tiled_launch import TiledGemm, plan_gemm_tiling

TILE = 16


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KernelBank(
            tmp_path_factory.mktemp("kernel_bank"), blas_kernel_specs()
        )


@pytest.fixture(scope="module")
def tiled(bank):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TiledGemm(bank, tile=TILE)


def _oracle(a, b, c, alpha, beta, m, n, k):
    return alpha * (a.reshape(m, k) @ b.reshape(k, n)).reshape(-1) + beta * c


def test_the_plan_counts_full_and_edge_tiles():
    plan = plan_gemm_tiling(48, 48, 48, TILE)
    assert plan.specialized_calls == 27 and plan.parametric_calls == 0
    plan = plan_gemm_tiling(20, 35, 17, TILE)
    assert plan.full_tiles == (1, 2, 1)
    assert plan.remainders == (4, 3, 1)
    assert plan.specialized_calls == 2
    assert plan.parametric_calls == 2 * 3 * 2 - 2


def test_exact_multiples_use_only_the_prebaked_core(tiled):
    m = n = k = 3 * TILE
    rng = np.random.default_rng(1)
    a = rng.standard_normal(m * k)
    b = rng.standard_normal(k * n)
    c = rng.standard_normal(m * n)
    produced = tiled(a, b, c, 1.3, 0.7, m, n, k)
    assert np.allclose(produced, _oracle(a, b, c, 1.3, 0.7, m, n, k))


def test_awkward_sizes_compose_core_plus_parametric_edges(tiled):
    rng = np.random.default_rng(2)
    for m, n, k in [(20, 35, 17), (TILE + 1, 2 * TILE + 2, TILE),
                    (TILE, TILE, TILE + 3)]:
        a = rng.standard_normal(m * k)
        b = rng.standard_normal(k * n)
        c = rng.standard_normal(m * n)
        produced = tiled(a, b, c, 0.9, -0.4, m, n, k)
        assert np.allclose(
            produced, _oracle(a, b, c, 0.9, -0.4, m, n, k)
        ), (m, n, k)


def test_the_coordinator_routes_large_calls_tiled_and_logs_it(bank):
    coordinator = LaunchCoordinator(bank, tile=TILE)
    m, n, k = 2 * TILE + 5, 3 * TILE, TILE + 1
    rng = np.random.default_rng(3)
    a = rng.standard_normal(m * k)
    b = rng.standard_normal(k * n)
    c = rng.standard_normal(m * n)
    produced = coordinator.launch(
        "gemm", A=a, B=b, C=c.copy(), alpha=1.0, beta=0.5, m=m, n=n, k=k
    )
    assert np.allclose(
        np.asarray(produced).reshape(-1), _oracle(a, b, c, 1.0, 0.5, m, n, k)
    )
    import json

    last = json.loads(
        (bank.root / "routing_log.jsonl").read_text().splitlines()[-1]
    )
    assert last["route"] == "tiled" and last["tile"] == TILE


def test_a_small_call_does_not_route_tiled(bank):
    coordinator = LaunchCoordinator(bank, tile=TILE)
    n = TILE - 2
    rng = np.random.default_rng(4)
    a = rng.standard_normal(n * n)
    b = rng.standard_normal(n * n)
    c = rng.standard_normal(n * n)
    produced = coordinator.launch(
        "gemm", A=a, B=b, C=c.copy(), alpha=1.0, beta=0.0, m=n, n=n, k=n
    )
    assert np.allclose(
        np.asarray(produced).reshape(-1), _oracle(a, b, c, 1.0, 0.0, n, n, n)
    )
    import json

    last = json.loads(
        (bank.root / "routing_log.jsonl").read_text().splitlines()[-1]
    )
    assert last["route"] in {"parametric", "specialized"}


def test_run_does_not_capture_a_view_into_the_callers_memory(bank):
    """The regression this work found: creation-path buffer capture.

    A first call passing a VIEW into a larger array must not hand that
    memory to the cached execution -- otherwise the next same-signature
    call writes its own inputs through the view, corrupting the caller's
    array at a distance. Asserted directly on the caller's memory, not on
    any kernel result."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        variant = bank.get("scal")

    backing = np.arange(24.0)
    view = backing[8:12]          # a contiguous view -- no copy made
    variant.run({"x": view, "y": np.zeros(4), "alpha": 2.0, "n": 4})
    before = backing.copy()
    # Same signature, different values: reuses the cached execution.
    variant.run({
        "x": np.full(4, 77.0), "y": np.zeros(4), "alpha": 2.0, "n": 4,
    })
    assert np.array_equal(backing, before), (
        "the second call's inputs were written through the first call's "
        "view into the caller's array"
    )


# ---------------------------------------------------------------------------
# The deployment layer's strategic recognition (tiling_strategy).
# ---------------------------------------------------------------------------

from src.compiler.tiling_strategy import (  # noqa: E402
    build_gemm_tile_plan,
    decide_tiling,
)


def test_recognition_declines_without_an_admitted_core(tmp_path):
    empty = KernelBank(tmp_path / "empty_bank", blas_kernel_specs())
    decision = decide_tiling(
        empty, "gemm", {"m": 100, "n": 100, "k": 100}
    )
    assert not decision.tiled
    assert any("no admitted" in reason for reason in decision.reasons)


def test_recognition_chooses_a_core_and_explains_it(bank, tiled):
    # ``tiled`` fixture ensured the TILE-sized core is admitted.
    decision = decide_tiling(
        bank, "gemm", {"m": 5 * TILE, "n": 5 * TILE, "k": 5 * TILE}
    )
    assert decision.tiled and decision.tile == TILE
    assert decision.candidates  # the evidence set survives the decision


def test_recognition_declines_the_cores_own_exact_size(bank, tiled):
    decision = decide_tiling(
        bank, "gemm", {"m": TILE, "n": TILE, "k": TILE}
    )
    assert not decision.tiled
    assert any("exact" in reason for reason in decision.reasons)


def test_nested_parallelism_tempers_the_worker_budget(bank, tiled):
    flat = decide_tiling(
        bank, "gemm", {"m": 4 * TILE, "n": 4 * TILE, "k": 4 * TILE},
        cores=8, nested_parallelism=0,
    )
    nested = decide_tiling(
        bank, "gemm", {"m": 4 * TILE, "n": 4 * TILE, "k": 4 * TILE},
        cores=8, nested_parallelism=3,
    )
    assert flat.worker_budget == 8
    assert nested.worker_budget == 2
    assert any("tempered" in reason for reason in nested.reasons)
    # Tempering changes the budget, never correctness of the decision.
    assert flat.tiled and nested.tiled and flat.tile == nested.tile


def test_the_plan_is_lanes_of_disjoint_c_blocks_with_ordered_k_steps():
    plan = build_gemm_tile_plan(2 * TILE + 3, TILE, 3 * TILE + 1, TILE)
    blocks = {(lane.i0, lane.j0) for lane in plan.lanes}
    assert len(blocks) == len(plan.lanes)  # disjoint outputs = independence
    for lane in plan.lanes:
        assert [step.p0 for step in lane.steps] == sorted(
            step.p0 for step in lane.steps
        )
        # Only the first k-step carries the caller's beta (spelled nan);
        # every later step accumulates with 1.0.
        first, *rest = lane.steps
        assert np.isnan(first.beta)
        assert all(step.beta == 1.0 for step in rest)
    assert plan.join_mode == "barrier"
    assert plan.schedule == "independent_lanes"
