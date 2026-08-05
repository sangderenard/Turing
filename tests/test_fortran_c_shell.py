from __future__ import annotations

import json

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.compiler.compiled_program_api import (
    CompiledProgramAPI,
    EntryPoint,
    Parameter,
)
from src.compiler.fortran_c_shell import (
    compile_ast_fortran_c_shell,
    compile_fortran_module_c_shell,
)
from src.compiler.fortran_c_shell import emit_fortran_c_shell_source
from src.compiler.ssa_fortran_backend import FortranModule, fortran_compiler
from src.compiler.shell_io import (
    ShellIOManifest, ShellIORequest, SystemPort, attach_shell_io,
)


class _ArenaState:
    def __init__(self):
        self.values = np.arange(4.0)


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_ast_c_shell_resolves_and_rotates_dotted_object_arenas(tmp_path):
    artifact = compile_ast_fortran_c_shell(
        "def frame(state):\n"
        "    state.values += 1.0\n"
        "    return state.values\n",
        "frame",
        {"state": _ArenaState()},
        tmp_path,
        output_names=("values_out",),
        state_feedback={"state.values": "values_out"},
    )

    payload = json.loads(artifact.run(frames=2).stdout)

    assert payload["outputs"]["values_out"] == {
        "first": 2.0,
        "sum": 14.0,
    }


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_ast_c_shell_uses_captured_early_return_identity(tmp_path):
    artifact = compile_ast_fortran_c_shell(
        "def restore(value, ref):\n"
        "    if isinstance(ref, AbstractTensor):\n"
        "        return value\n"
        "    return float(value.item())\n"
        "\n"
        "def frame(dt):\n"
        "    return restore(dt * 2.0, dt)\n",
        "frame",
        {"dt": np.asarray([0.05], dtype=np.float64)},
        tmp_path,
        python_bindings={"AbstractTensor": AbstractTensor},
        output_names=("value_out",),
    )

    payload = json.loads(artifact.run().stdout)

    assert payload["outputs"]["value_out"] == {
        "first": pytest.approx(0.1),
        "sum": pytest.approx(0.1),
    }


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_generated_c_shell_launches_fortran_and_applies_feedback(tmp_path):
    source = """
module affine_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine affine(extent_4, x, y) bind(C, name="affine")
    integer(c_int), value :: extent_4
    real(c_double), intent(in) :: x(extent_4)
    real(c_double), intent(out) :: y(extent_4)
    y = x * 2.0_c_double + 1.0_c_double
  end subroutine affine
end module affine_fortran
"""
    api = CompiledProgramAPI(
        module="affine_fortran",
        language="fortran",
        entry="affine",
        entry_points=(EntryPoint(
            name="affine",
            symbol="affine",
            kind="numerical",
            parameters=(
                Parameter(
                    "extent_4", "extent", "int32", "int32_t",
                    "c_int32", "value",
                ),
                Parameter(
                    "x", "input", "float64", "double", "c_double",
                    "reference", (4,), "extent_4", "x",
                ),
                Parameter(
                    "y", "output", "float64", "double", "c_double",
                    "reference", (4,), "extent_4", "y",
                ),
            ),
        ),),
    )
    module = FortranModule("affine_fortran", source, api=api)

    artifact = compile_fortran_module_c_shell(
        module,
        {"x": np.arange(1.0, 9.0)},
        tmp_path,
        state_feedback={"x": "y"},
        extent_overrides={"extent_4": 8},
        name="affine_native",
    )
    payload = json.loads(artifact.run(frames=2).stdout)

    assert artifact.executable_path.is_file()
    assert artifact.final_outputs_path.is_file()
    assert payload["status"] == 1
    assert payload["frames"] == 2
    assert payload["outputs"]["y"] == {"first": 7.0, "sum": 168.0}
    assert payload["shell_ns_total"] > 0
    c_source = artifact.c_source_path.read_text(encoding="utf-8")
    assert "slots[0] = slots[1]" in c_source
    assert "memcpy(slots[0], slots[1]" not in c_source


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_generated_c_shell_preserves_all_extent_arguments_in_abi_order(tmp_path):
    source = """
module two_extent_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine copy_with_scalar_extent(extent_1, extent_4, x, y) &
      bind(C, name="copy_with_scalar_extent")
    integer(c_int), value :: extent_1
    integer(c_int), value :: extent_4
    real(c_double), intent(in) :: x(extent_4)
    real(c_double), intent(out) :: y(extent_4)
    y = x + real(extent_1, c_double)
  end subroutine copy_with_scalar_extent
end module two_extent_fortran
"""
    api = CompiledProgramAPI(
        module="two_extent_fortran",
        language="fortran",
        entry="copy_with_scalar_extent",
        entry_points=(EntryPoint(
            name="copy_with_scalar_extent",
            symbol="copy_with_scalar_extent",
            kind="control",
            parameters=(
                Parameter(
                    "extent_1", "extent", "int32", "int32_t",
                    "c_int32", "value",
                ),
                Parameter(
                    "extent_4", "extent", "int32", "int32_t",
                    "c_int32", "value",
                ),
                Parameter(
                    "x", "input", "float64", "double", "c_double",
                    "reference", (4,), "extent_4", "x",
                ),
                Parameter(
                    "y", "output", "float64", "double", "c_double",
                    "reference", (4,), "extent_4", "y",
                ),
            ),
        ),),
    )
    module = FortranModule("two_extent_fortran", source, api=api)

    artifact = compile_fortran_module_c_shell(
        module,
        {"x": np.arange(8.0)},
        tmp_path,
        extent_overrides={"extent_4": 8},
        name="two_extent_native",
    )
    payload = json.loads(artifact.run().stdout)

    assert "int32_t, int32_t, double *" in artifact.c_source_path.read_text()
    assert payload["outputs"]["y"] == {"first": 1.0, "sum": 36.0}


def test_generated_c_shell_uses_win32_rgb_blit_from_shared_io_contract():
    parameters = (
        Parameter(
            "extent_4", "extent", "int32", "int32_t", "c_int32", "value",
        ),
        *(
            Parameter(
                channel, "output", "float64", "double", "c_double",
                "reference", (4,), "extent_4", channel,
            )
            for channel in ("red", "green", "blue")
        ),
    )
    shell_io = {
        "requirements": {
            "requests": [{
                "capability": "display_double_buffer",
                "optional": False,
                "attributes": {
                    "pixel_format": "rgb_f64_planar",
                    "width": 2,
                    "height": 2,
                    "title": "Generated RGB",
                },
            }],
            "bindings": [
                {
                    "resource": f"display.{channel}",
                    "entry_point": "rgb",
                    "parameter": channel,
                }
                for channel in ("red", "green", "blue")
            ],
            "options": [],
        },
        "abi": {},
    }
    api = CompiledProgramAPI(
        module="rgb_fortran",
        language="fortran",
        entry="rgb",
        entry_points=(EntryPoint(
            name="rgb", symbol="rgb", kind="numerical",
            parameters=parameters,
        ),),
        metadata={"shell_io": shell_io},
    )
    module = FortranModule("rgb_fortran", "", api=api)

    source = emit_fortran_c_shell_source(
        module, extent_overrides={"extent_4": 4}
    )

    assert "StretchDIBits(" in source
    assert "CreateWindowExA(" in source
    assert "PeekMessageA(" in source
    assert "SDL" not in source
    assert "pygame" not in source
    assert "turing_display_present(" in source
    assert "#include <stdbool.h>" in source


def test_generated_c_shell_reads_declared_file_port_into_bound_abi_parameters():
    parameters = (
        Parameter(
            "subject_bytes", "input", "u8", "uint8_t", "c_uint8",
            "reference", (16,), None, "binary_bytes",
        ),
        Parameter(
            "subject_length", "input", "i64", "int64_t", "c_int64",
            "value", source_name="binary_length",
        ),
    )
    api = attach_shell_io(CompiledProgramAPI(
        "machine", "fortran", "load_subject",
        (EntryPoint("load_subject", "load_subject", "control", parameters),),
    ), ShellIOManifest(
        (ShellIORequest.create("files"),),
        system_ports=(SystemPort.create(
            "subject-binary", "file", "input", entry_point="load_subject",
            fields={"data": "binary_bytes", "length": "binary_length"},
            attributes={"maximum_bytes": 16},
        ),),
    ))
    module = FortranModule("machine", "", api=api)

    source = emit_fortran_c_shell_source(module)

    assert 'turing_argument_value(argc, argv, "--file-subject-binary")' in source
    assert "turing_read_file(" in source
    assert "(uint8_t *)slots[0], 16" in source
    assert "*((int64_t *)slots[1]) = (int64_t)loaded_bytes" in source
    assert "short initial state at binary_bytes" not in source


@pytest.mark.skipif(
    fortran_compiler() is None,
    reason="no Fortran compiler installed",
)
def test_native_c_file_handler_runs_fortran_with_exact_bytes_and_length(tmp_path):
    source = """
module file_subject_fortran
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine inspect_subject(subject_bytes, subject_length, result) &
      bind(C, name="inspect_subject")
    integer(c_int8_t), intent(in) :: subject_bytes(16)
    integer(c_int64_t), value :: subject_length
    integer(c_int64_t), intent(out) :: result
    result = int(subject_bytes(1), c_int64_t) + subject_length
  end subroutine inspect_subject
end module file_subject_fortran
"""
    parameters = (
        Parameter(
            "subject_bytes", "input", "u8", "uint8_t", "c_uint8",
            "reference", (16,), None, "binary_bytes",
        ),
        Parameter(
            "subject_length", "input", "i64", "int64_t", "c_int64",
            "value", source_name="binary_length",
        ),
        Parameter(
            "result", "output", "i64", "int64_t", "c_int64",
            "reference", source_name="result",
        ),
    )
    api = attach_shell_io(CompiledProgramAPI(
        "file_subject_fortran", "fortran", "inspect_subject",
        (EntryPoint("inspect_subject", "inspect_subject", "control", parameters),),
    ), ShellIOManifest(
        (ShellIORequest.create("files"),),
        system_ports=(SystemPort.create(
            "subject-binary", "file", "input", entry_point="inspect_subject",
            fields={"data": "binary_bytes", "length": "binary_length"},
            attributes={"maximum_bytes": 16},
        ),),
    ))
    module = FortranModule("file_subject_fortran", source, api=api)
    subject = tmp_path / "subject.exe"
    subject.write_bytes(b"MZ\x00\xff")

    artifact = compile_fortran_module_c_shell(
        module, {}, tmp_path / "build", name="file_subject_native",
    )
    payload = json.loads(artifact.run(files={"subject-binary": subject}).stdout)

    assert payload["outputs"]["result"] == {"first": 81.0, "sum": 81.0}
