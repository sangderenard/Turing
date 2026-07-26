import pytest

from src.common.tensors.benchmark_nodus_calculator import (
    plot_comparison,
    run_comparison,
)


def test_common_chain_has_numpy_c_parity_without_nodus_process():
    table = run_comparison(
        backends=("numpy", "c"),
        elements=32,
        warmup=0,
        repeats=1,
        include_nodus=False,
    )

    assert list(table["backend"]) == ["numpy", "c"]
    assert table["parity_ok"].all()
    assert (table["median_sec"] >= 0.0).all()


def test_glsl_fused_row_reports_synchronized_gpu_timing_when_available():
    from src.common.tensors.accelerator_backends.glsl_backend import (
        GLContextUnavailable,
    )

    try:
        table = run_comparison(
            backends=("numpy",),
            elements=257,
            warmup=1,
            repeats=2,
            include_glsl=True,
            include_nodus=False,
        )
    except GLContextUnavailable as exc:
        pytest.skip(f"no OpenGL compute context: {exc}")

    glsl = table.loc[table["backend"] == "glsl"].iloc[0]
    assert glsl["execution"] == "fused_shader_resident_io"
    assert glsl["dtype"] == "float32"
    assert glsl["gpu_median_sec"] > 0.0
    assert glsl["readback_sec"] >= 0.0
    assert glsl["parity_ok"]


def test_backends_report_serial_start_and_completion_events():
    events = []
    run_comparison(
        backends=("numpy", "c"),
        elements=32,
        warmup=0,
        repeats=1,
        include_glsl=False,
        include_nodus=False,
        progress_callback=lambda event, label, row: events.append((event, label)),
    )

    assert events == [
        ("start", "numpy"),
        ("complete", "numpy"),
        ("start", "c"),
        ("complete", "c"),
    ]


def test_timing_graph_renders_without_a_gpu_row(tmp_path):
    table = run_comparison(
        backends=("numpy", "c"),
        elements=32,
        warmup=0,
        repeats=1,
        include_glsl=False,
        include_nodus=False,
    )

    output = plot_comparison(table, tmp_path / "timing.png")

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
