"""Central optimization and standalone-link policy for generated Fortran."""

from __future__ import annotations

import os
from pathlib import Path


def is_gnu_fortran(compiler: str) -> bool:
    return "gfortran" in Path(compiler).name.casefold()


def aggressive_fortran_flags(compiler: str) -> tuple[str, ...]:
    """High optimization without relaxing IEEE/NaN program semantics."""

    name = Path(compiler).name.casefold()
    if "gfortran" in name:
        return (
            "-O3", "-march=native", "-flto", "-funroll-loops",
            "-fomit-frame-pointer",
        )
    if name.startswith(("ifx", "ifort")):
        return ("-O3", "-xHost", "-ipo")
    if "flang" in name:
        return ("-O3", "-march=native", "-flto")
    return ("-O3",)


def aggressive_c_flags(fortran_compiler: str) -> tuple[str, ...]:
    """C-host flags compatible with the selected Fortran link driver."""

    if is_gnu_fortran(fortran_compiler):
        return (
            "-O3", "-march=native", "-flto", "-funroll-loops",
            "-fomit-frame-pointer",
        )
    return ("-O3",)


def standalone_fortran_link_flags(compiler: str) -> tuple[str, ...]:
    """Link compiler runtimes into the artifact instead of requiring DLLs."""

    if not is_gnu_fortran(compiler):
        raise ValueError(
            "standalone runtime linkage is currently defined only for GNU "
            "Fortran; set standalone=False or add a verified compiler policy"
        )
    flags = ["-flto", "-static-libgfortran", "-static-libgcc"]
    if os.name == "nt":
        # This also selects archive forms of libquadmath and transitive GNU
        # support libraries. Windows system DLLs remain normal OS ABI imports.
        flags.append("-static")
    return tuple(flags)


__all__ = [
    "aggressive_c_flags",
    "aggressive_fortran_flags",
    "is_gnu_fortran",
    "standalone_fortran_link_flags",
]
