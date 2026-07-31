from __future__ import annotations

import inspect

from src.common.tensors.accelerator_backends import demo_mandelbrot_ssa


def test_copied_demo_stops_at_precompile_before_glsl_realization():
    source = inspect.getsource(demo_mandelbrot_ssa.animate_glsl)

    assert "deployment.compile_process_graph()" in source
    assert "precompile_only=True" in source
    assert "lower_precompile_and_control_to_ssa(" in source
    assert "deployment.execute_named(" not in source
    assert "DoubleBufferedAVISink" not in source
    assert "require_gl_context()" not in source


def test_copied_demo_preserves_translation_and_precompile_entrypoint():
    source = inspect.getsource(demo_mandelbrot_ssa.animate_glsl)

    assert "build_parametric_mandelbrot_glsl_deployment(" in source
    assert "mandelbrot_recording_program" in source
    assert "deployment.capture_fused_programs(" in source
    assert "ssa_module_dictionary(" in source
