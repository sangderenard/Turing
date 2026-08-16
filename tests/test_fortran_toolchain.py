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

    assert {"-O3", "-march=native", "-flto", "-funroll-loops"} <= set(fortran)
    assert {"-O3", "-march=native", "-flto", "-funroll-loops"} <= set(c)
    assert {"-flto", "-static-libgfortran", "-static-libgcc"} <= set(link)
    assert "-ffast-math" not in fortran
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
