from __future__ import annotations

from types import SimpleNamespace

from src.compiler.kernel_bank import BankRefusal
from src.compiler.tiling_strategy import (
    build_gemm_tile_plan,
    decide_tiling,
    interpret_gemm_compute_matrix,
    prebake_gemm_launch_matrix,
)


class _MeasuredBank:
    specs = {
        "gemm": SimpleNamespace(size_parameters=("m", "n", "k")),
    }

    def inventory(self):
        return [
            self._row(32, 0.001),
            self._row(64, 0.002),
            self._row(128, 0.0001, admitted=False),
            self._row(256, 0.0001),  # Marked admitted but stale at get().
        ]

    @staticmethod
    def _row(size, seconds, *, admitted=True):
        return {
            "kernel": "gemm",
            "specialized": {"m": size, "n": size, "k": size},
            "verification": {
                "admitted": admitted,
                "probe_call_seconds": seconds * 2,
            },
            "profile": {"compute_avg_seconds": seconds},
        }

    def get(self, _name, *, specialized, **_kwargs):
        if specialized["m"] == 256:
            raise BankRefusal("stale compiler fingerprint")
        return object()


def test_chooser_uses_best_live_profile_and_is_deterministic():
    bank = _MeasuredBank()
    arguments = dict(
        bank=bank, name="gemm", sizes={"m": 192, "n": 192, "k": 192},
        contract="fast", cores=8, must_divide=True,
    )

    first = decide_tiling(**arguments)
    second = decide_tiling(**arguments)

    assert first == second
    assert first.tiled and first.tile == 64
    assert first.candidates == ((32, 0.001), (64, 0.002))


class _CompositionBank(_MeasuredBank):
    def inventory(self):
        return [
            self._row(32, 0.000101),
            self._row(64, 0.000459),
            self._row(128, 0.003361),
        ]

    def get(self, _name, **_kwargs):
        return object()


def test_chooser_ranks_composed_critical_path_not_isolated_core_gflops():
    decision = decide_tiling(
        _CompositionBank(), "gemm", {"m": 256, "n": 256, "k": 256},
        contract="fast", cores=8,
    )

    assert decision.tiled and decision.tile == 64
    reason = " ".join(decision.reasons)
    assert "projected composed critical path" in reason
    assert "7 background worker(s) plus caller" in reason
    assert "64:3.672ms" in reason
    assert "128:6.722ms" in reason


def test_must_divide_refuses_fitting_cores_that_leave_edges():
    decision = decide_tiling(
        _MeasuredBank(), "gemm", {"m": 100, "n": 100, "k": 100},
        contract="fast", must_divide=True,
    )

    assert not decision.tiled
    assert decision.tile is None
    assert "none divide every axis" in " ".join(decision.reasons)


def test_edge_capable_choice_can_select_a_nondividing_core():
    decision = decide_tiling(
        _MeasuredBank(), "gemm", {"m": 100, "n": 100, "k": 100},
        contract="fast", must_divide=False,
    )

    assert decision.tiled and decision.tile == 64


def test_one_worker_refuses_unsubstantiated_serial_composition():
    decision = decide_tiling(
        _MeasuredBank(), "gemm", {"m": 192, "n": 192, "k": 192},
        contract="fast", cores=1, must_divide=True,
    )
    assert not decision.tiled
    assert "no background workers" in " ".join(decision.reasons)


def test_prebaked_matrix_contains_source_strides_and_launch_spans():
    plan = build_gemm_tile_plan(256, 256, 256, 128, worker_budget=4)
    matrix = prebake_gemm_launch_matrix(
        plan,
        variant_key="gemm-fast-128",
        parameter_ids={"A": 0, "B": 1, "C": 2, "alpha": 3, "beta": 4},
        total_layout={"parameter_order": ["A", "B", "C", "alpha", "beta"]},
        core_layout={"tile": 128},
        chunk_size=2,
    )

    assert matrix["launch"] == {
        "join": "barrier", "workers": 4, "chunk_size": 2,
        "lane_count": 4, "spans": [[0, 2], [2, 4]],
    }
    assert matrix["logical_launch"] == {
        "kind": "tile_composition",
        "schedule": "independent_lanes",
        "join": "barrier",
        "lane_count": 4,
        "call_count": 8,
        "calls_per_lane": [2, 2, 2, 2],
        "call_order": "sequential_within_lane",
    }
    first = matrix["lanes"][0]["calls"][0]["parameters_by_name"]
    assert first["A"]["source_strides"] == [256, 1]
    assert first["A"]["packed_strides"] == [128, 1]
    assert first["B"]["source_strides"] == [256, 1]
    assert first["C"]["source_offset"] == 0
    assert matrix["lanes"][1]["calls"][0]["parameters_by_name"]["C"][
        "source_offset"
    ] == 128


def test_prebaked_matrix_zero_pads_arbitrary_edges_on_the_square_core():
    plan = build_gemm_tile_plan(130, 131, 129, 128, worker_budget=2)
    matrix = prebake_gemm_launch_matrix(
        plan, variant_key="gemm-fast-128", parameter_ids={},
        total_layout={}, core_layout={}, chunk_size=1,
    )
    edge = matrix["lanes"][-1]["calls"][-1]["parameters_by_name"]
    assert edge["A"]["source_shape"] == [2, 1]
    assert edge["B"]["source_shape"] == [1, 3]
    assert edge["C"]["source_shape"] == [2, 3]
    assert all(
        edge[name]["zero_fill_packed_margin"] for name in ("A", "B", "C")
    )


def test_glsl_and_webgpu_interpret_the_exact_same_logical_matrix():
    from src.compiler.deployment_lowering import ComputeDispatchLimits

    plan = build_gemm_tile_plan(256, 192, 128, 64, worker_budget=7)
    matrix = prebake_gemm_launch_matrix(
        plan, variant_key="universal-gemm-fast-64", parameter_ids={},
        total_layout={}, core_layout={}, chunk_size=2,
    )
    limits = ComputeDispatchLimits(
        max_group_count=(65535, 65535, 65535),
        max_group_size=(256, 256, 64),
        max_invocations=256,
    )
    glsl = interpret_gemm_compute_matrix(
        matrix, backend="glsl", limits=limits,
    )
    webgpu = interpret_gemm_compute_matrix(
        matrix, backend="webgpu", limits=limits,
    )

    assert glsl.matrix_sha256 == webgpu.matrix_sha256
    assert glsl.module_key == webgpu.module_key == "universal-gemm-fast-64"
    assert glsl.calls_per_lane == webgpu.calls_per_lane == (2,) * 12
    assert glsl.choice.compute == webgpu.choice.compute
    assert glsl.as_record()["logical"] == webgpu.as_record()["logical"]


def test_compute_interpreter_refuses_a_backend_rewritten_topology():
    from src.compiler.deployment_lowering import ComputeDispatchLimits

    plan = build_gemm_tile_plan(128, 128, 128, 64, worker_budget=3)
    matrix = prebake_gemm_launch_matrix(
        plan, variant_key="universal", parameter_ids={}, total_layout={},
        core_layout={}, chunk_size=1,
    )
    matrix["logical_launch"]["call_order"] = "backend_may_reorder"
    limits = ComputeDispatchLimits(
        max_group_count=(65535, 65535, 65535),
        max_group_size=(256, 256, 64), max_invocations=256,
    )
    import pytest
    with pytest.raises(ValueError, match="call_order"):
        interpret_gemm_compute_matrix(
            matrix, backend="glsl", limits=limits,
        )
