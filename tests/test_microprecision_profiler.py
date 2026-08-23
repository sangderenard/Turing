import json

from PIL import Image

from tools.benchmark_microprecision_matrix import (
    _prepare_core,
    _selection,
    profile_matrix,
    render_mandelbrot,
)


def test_backend_selection_does_not_redirect_or_fall_back():
    assert _selection(["glsl"], ("llvm", "glsl"), "backends") == ("glsl",)


def test_nonllvm_profile_stops_at_its_own_backend_frontier(tmp_path):
    rows, builds = profile_matrix(
        operators=("sin",),
        backends=("c", "fortran", "webgpu"),
        widths=(2,),
        sizes=(1, 8),
        repeats=1,
        warmups=0,
        accuracy_samples=2,
        oracle_dps=60,
        scratch=tmp_path,
    )

    assert len(rows) == 6
    assert {row.backend for row in rows} == {"c", "fortran", "webgpu"}
    assert not any(row.backend == "llvm" for row in rows)
    assert {row.status for row in rows if row.backend == "c"} == {"unavailable"}
    assert {row.status for row in rows if row.backend != "c"} == {"unsupported"}
    assert builds["sin w2"]["emit_ns"] == 0
    assert builds["sin w2"]["native_compile_ns"] == 0


def test_shared_preparation_retains_backend_neutral_module():
    prepared = _prepare_core("sin", 3)

    assert prepared.native is None
    assert prepared.build.emit_ns == 0
    assert prepared.build.native_compile_ns == 0
    assert prepared.build.static_fma_instructions > 0


def test_llvm_profile_separates_launch_throughput_and_oracles(tmp_path):
    rows, _builds = profile_matrix(
        operators=("sin",),
        backends=("llvm",),
        widths=(3,),
        sizes=(1, 16),
        repeats=2,
        warmups=1,
        accuracy_samples=6,
        oracle_dps=80,
        scratch=tmp_path,
    )

    assert len(rows) == 2
    assert all(row.status == "passed" for row in rows)
    assert all(row.prepare_ns > 0 for row in rows)
    assert all(row.first_launch_ns > 0 for row in rows)
    assert all(row.warm_ns_per_element > 0 for row in rows)
    assert all(row.static_fma_instructions > 0 for row in rows)
    assert all(row.polynomial_ulp_p99 is not None for row in rows)
    assert all(row.true_ulp_p99 is not None for row in rows)


def test_four_limb_mandelbrot_writes_image_and_receipt(tmp_path):
    destination = tmp_path / "four-limb.png"

    receipt = render_mandelbrot(
        destination,
        width=8,
        height=6,
        iterations=1,
        oracle_dps=80,
        span="1e-16",
    )

    assert receipt["precision_limbs"] == 4
    assert receipt["panel_width"] == 8
    assert receipt["panel_height"] == 6
    assert receipt["output_width"] == 8
    assert receipt["output_height"] == 6
    assert receipt["comparison_panels"] is False
    assert receipt["same_recurrence_source"] is True
    assert len(receipt["recurrence_source_sha256"]) == 64
    assert receipt["four_limb_unique_x_coordinates"] == 8
    assert receipt["binary64_unique_x_coordinates"] < 8
    assert destination.exists()
    assert Image.open(destination).size == (8, 6)
    persisted = json.loads(destination.with_suffix(".json").read_text())
    assert persisted["schema"] == "turing-four-limb-mandelbrot-v2"
