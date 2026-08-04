from __future__ import annotations

import os

import pytest

from src.compiler.fortran_toolchain import (
    aggressive_c_flags,
    aggressive_fortran_flags,
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
