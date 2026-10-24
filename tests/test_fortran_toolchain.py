from __future__ import annotations

import os

import pytest

from src.compiler.fortran_toolchain import (
    aggressive_c_flags,
    aggressive_fortran_flags,
    stage_fortran_runtime_dependencies,
    standalone_fortran_link_flags,
)


def test_gnu_policy_is_lto_native_and_standalone_without_fast_math():
    compiler = r"C:\toolchain\bin\gfortran.exe"

    fortran = aggressive_fortran_flags(compiler)
    c = aggressive_c_flags(compiler)
    link = standalone_fortran_link_flags(compiler)

    lto = f"-flto={max(1, os.cpu_count() or 1)}"
    assert {"-O3", "-march=native", lto, "-funroll-loops"} <= set(fortran)
    assert {"-O3", "-march=native", lto, "-funroll-loops"} <= set(c)
    assert {lto, "-static-libgfortran", "-static-libgcc"} <= set(link)
    assert "-ffast-math" not in fortran
    # Ordinary modules keep contraction; a module carrying a precision
    # section withdraws it, which is what lets the fortran lane declare
    # SECTION_ISOLATION. Both halves of the policy are pinned here.
    assert "-ffp-contract=off" not in fortran
    precision = aggressive_fortran_flags(compiler, precision_sections=True)
    assert "-ffp-contract=off" in precision
    assert "-ffast-math" not in precision
    if os.name == "nt":
        assert "-static" in link


def test_unknown_compiler_does_not_claim_unverified_standalone_linkage():
    with pytest.raises(ValueError, match="GNU Fortran"):
        standalone_fortran_link_flags("mystery-fortran")


def test_gnu_runtime_dll_is_staged_beside_dynamic_artifact(tmp_path):
    if os.name != "nt":
        pytest.skip("GNU runtime DLL staging is Windows-specific")
    compiler_bin = tmp_path / "toolchain" / "bin"
    compiler_bin.mkdir(parents=True)
    compiler = compiler_bin / "gfortran.exe"
    compiler.write_bytes(b"")
    (compiler_bin / "libgfortran-5.dll").write_bytes(b"runtime")
    destination = tmp_path / "artifact"

    copied = stage_fortran_runtime_dependencies(compiler, destination)

    assert copied == (destination / "libgfortran-5.dll",)
    assert copied[0].read_bytes() == b"runtime"


def test_linux_standalone_shared_object_stages_its_gnu_runtime(tmp_path):
    import shutil
    import subprocess
    import sys

    from src.compiler.fortran_toolchain import (
        stage_standalone_fortran_runtime,
        stages_shared_runtime,
    )

    compiler = shutil.which("gfortran")
    if os.name == "nt" or sys.platform == "darwin" or compiler is None:
        pytest.skip("Linux gfortran policy")
    assert stages_shared_runtime(compiler, shared=True)
    assert not stages_shared_runtime(compiler, shared=False)
    link = standalone_fortran_link_flags(compiler, shared=True)
    assert "-static-libgfortran" not in link
    assert "-Wl,-rpath,$ORIGIN" in link
    # ``error stop`` pulls the runtime; the non-PIC archive would not link.
    source = tmp_path / "boom.f90"
    source.write_text(
        'subroutine boom(n) bind(C, name="boom")\n'
        "  use iso_c_binding\n"
        "  integer(c_int64_t), value :: n\n"
        "  if (n > 3) error stop 75\n"
        "end subroutine\n",
        encoding="utf-8",
    )
    library = tmp_path / "libboom.so"
    completed = subprocess.run(
        [compiler, "-shared", "-fPIC", str(source), "-o", str(library), *link],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    staged = stage_standalone_fortran_runtime(compiler, tmp_path, shared=True)
    assert any(path.name.startswith("libgfortran.so.") for path in staged)
    assert all(path.parent == tmp_path and path.is_file() for path in staged)
