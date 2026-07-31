import pytest

from src.common.tensors.accelerator_backends.control_ir_backends import (
    AcceleratedControlTarget,
    reduce_control_ir,
    reduce_control_ir_all_targets,
)
from src.compiler.control_source import (
    ControlProgram,
    LoopBlock,
    SequenceBlock,
    StatementBlock,
    StateMachineTick,
)
from src.compiler.ssa_fortran_backend import fortran_compiler


def _program():
    return ControlProgram(
        root=SequenceBlock(
            (
                LoopBlock(
                    "i", "0", "8", "1",
                    StatementBlock(("__scheduled_region_0__",)),
                ),
                StateMachineTick(
                    "mode",
                    ((0, StatementBlock(("__scheduled_region_1__",))),),
                ),
            )
        ),
        region_indices=(0, 1),
        uniforms=(),
        value_aliases={},
        iterable_bindings={},
        static_iterable_bindings={},
        collection_bindings={},
        closure_iterable_bindings={},
    )


def test_fortran_is_an_accelerated_control_target():
    assert AcceleratedControlTarget.FORTRAN.value == "fortran"
    rendered = reduce_control_ir_all_targets(_program())
    assert AcceleratedControlTarget.FORTRAN in rendered


def test_control_structure_renders_as_fortran():
    source = reduce_control_ir(
        _program(), AcceleratedControlTarget.FORTRAN
    ).source

    assert 'bind(C, name="turing_control")' in source
    # Fortran do-bounds are inclusive, so the exclusive stop loses one.
    assert "do i = 0, (8) - 1, 1" in source
    assert "end do" in source
    assert "select case (mode)" in source
    assert "end select" in source
    assert "end subroutine turing_control" in source
    # No brace-delimited syntax may leak through from the C renderer.
    assert "{" not in source and "}" not in source


def test_loop_inductions_are_declared():
    """Fortran has no in-statement declaration, so `do` variables need one."""

    source = reduce_control_ir(
        _program(), AcceleratedControlTarget.FORTRAN
    ).source
    assert "integer :: i" in source


def test_region_bodies_default_to_fortran_call_syntax():
    source = reduce_control_ir(
        _program(), AcceleratedControlTarget.FORTRAN
    ).source
    assert "call turing_region_0()" in source
    assert "call turing_region_1()" in source


def test_every_target_still_renders_the_same_regions():
    rendered = reduce_control_ir_all_targets(_program())
    for target, source in rendered.items():
        assert source.region_indices == (0, 1), target


@pytest.mark.skipif(
    fortran_compiler() is None, reason="no Fortran compiler installed"
)
def test_rendered_control_environment_compiles(tmp_path):
    import subprocess
    from pathlib import Path

    body = reduce_control_ir(
        _program(), AcceleratedControlTarget.FORTRAN
    ).source
    module = (
        "module turing_ctl\n"
        "  use, intrinsic :: iso_c_binding\n"
        "  implicit none\n"
        "  integer :: mode = 0\n"
        "contains\n"
        "  subroutine turing_region_0()\n  end subroutine\n"
        "  subroutine turing_region_1()\n  end subroutine\n"
        f"{body}"
        "end module turing_ctl\n"
    )
    source = Path(tmp_path) / "turing_ctl.f90"
    source.write_text(module, encoding="utf-8")

    import os

    compiler = fortran_compiler()
    # gfortran spawns f951, which loads its support DLLs from the toolchain's
    # own bin directory.  Invoked by absolute path without that on PATH it
    # exits non-zero with no diagnostic at all.
    environment = dict(os.environ)
    environment["PATH"] = (
        str(Path(compiler).parent) + os.pathsep + environment.get("PATH", "")
    )
    completed = subprocess.run(
        [compiler, "-std=f2008", "-Wall", "-c", str(source), "-o",
         str(Path(tmp_path) / "turing_ctl.o")],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
