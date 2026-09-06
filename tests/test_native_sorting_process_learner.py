from pathlib import Path
from types import SimpleNamespace

from src.compiler.native_sorting_process_learner import (
    _bounded_cycle,
    _render_region,
    _value,
    capture_sorting_process,
)
from src.compiler.precompile_to_ssa import lower_fused_program_to_ssa
from src.compiler.ssa_fortran_backend import emit_function


def test_legacy_native_affine_cli_routes_process_sources_through_compiler(
    monkeypatch, tmp_path, capsys,
):
    from src.compiler import native_affine_learner
    from src.compiler import native_sorting_process_learner

    calls = []
    executable = tmp_path / "sorting_process_learning_window.exe"
    monkeypatch.setattr(
        native_sorting_process_learner,
        "compile_sorting_process_window",
        lambda source, output, **options: (
            calls.append((source, output, options))
            or SimpleNamespace(executable_path=executable)
        ),
    )

    result = native_affine_learner.main([
        "examples/learnable_sort.py", "--output", str(tmp_path),
        "--train-samples", "17", "--compile-only",
    ])

    assert result == 0
    assert calls[0][2]["batch_size"] == 17
    assert str(executable) in capsys.readouterr().out


def test_sorting_renderer_uses_compiler_spans_and_indexed_writes():
    captured = capture_sorting_process(
        Path("examples/learnable_sort.py"), batch_size=2, seed=3,
    )
    cycle = _bounded_cycle(captured)
    renderer = _render_region(captured, cycle, width=160, height=120)

    function, shortfalls = lower_fused_program_to_ssa(
        renderer.program, function_name="sorting_graph_renderer",
    )
    assert not shortfalls
    outputs = [
        _value(function, value_id)
        for value_id in renderer.program.outputs.values()
    ]
    source = emit_function(function, outputs=outputs).source

    # Fill is a scalar whole-span assignment. Static and live graph pixels are
    # both generic vector-subscript stores into those preallocated spans.
    assert " = 0.025_c_double" in source
    assert " = 0.035_c_double" in source
    assert " = 0.06_c_double" in source
    assert source.count(" + 1) = ") >= 6
    assert "reshape([" not in source
    assert max(value.size for value in renderer.feeds.values()) < 160 * 120


def test_captured_sorting_cycle_contains_forward_and_derived_reverse_graphs():
    captured = capture_sorting_process(
        Path("examples/learnable_sort.py"), batch_size=2, seed=5,
    )

    assert captured.cycle.forward_program.steps
    assert captured.cycle.reverse_capture.program.steps
    assert set(captured.cycle.reverse_capture.proposed_inputs) == {
        f"proposed_{name}" for name in captured.problem.parameter_names
    }
