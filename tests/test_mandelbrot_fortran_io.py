import numpy as np
import pytest

from src.common.tensors.accelerator_backends.demo_mandelbrot_fortran_io import (
    IO, SOURCE,
)
from src.compiler.python_native_shell import (
    NativeShellRuntime, compile_ast_fortran_io_shell,
)
from src.compiler.ssa_fortran_backend import fortran_compiler


def test_mandelbrot_demo_declares_only_program_io_not_a_shell():
    mapping = IO.to_mapping()

    assert [item["resource"] for item in mapping["bindings"]] == [
        "display.unit_x", "display.unit_y", "option.iterations", "display.back",
    ]
    assert [item["name"] for item in mapping["options"]] == [
        "width", "height", "iterations", "fps",
    ]
    assert "files" not in {
        request["capability"] for request in mapping["requests"]
    }


@pytest.mark.skipif(
    fortran_compiler() is None, reason="no Fortran compiler installed"
)
def test_generated_fortran_shell_pixels_match_reference(tmp_path):
    options = {"width": 5, "height": 4, "iterations": 6, "fps": 30.0}
    compiled = compile_ast_fortran_io_shell(
        SOURCE, "mandelbrot_frame", IO, options, directory=tmp_path
    )
    runtime = NativeShellRuntime(compiled.module.api, compiled.library_path, options)
    try:
        actual = runtime.frame(0.0).copy()
    finally:
        runtime.close()

    unit_x = np.tile(
        np.linspace(-1.0, 1.0, options["width"], dtype=np.float32),
        options["height"],
    )
    unit_y = np.repeat(
        np.linspace(-1.0, 1.0, options["height"], dtype=np.float32),
        options["width"],
    )
    cx = unit_x * np.float32(1.5) - np.float32(0.75)
    cy = unit_y
    zx = np.zeros_like(cx)
    zy = np.zeros_like(cx)
    expected = np.zeros_like(cx)
    for _ in range(options["iterations"]):
        zx2 = zx * zx
        zy2 = zy * zy
        expected += zx2 + zy2 <= np.float32(4.0)
        zx, zy = zx2 - zy2 + cx, np.float32(2.0) * zx * zy + cy

    np.testing.assert_array_equal(actual, expected)
    launcher = compiled.launcher_path.read_text(encoding="utf-8")
    assert '"default": 5' in launcher
    assert '"default": 4' in launcher
