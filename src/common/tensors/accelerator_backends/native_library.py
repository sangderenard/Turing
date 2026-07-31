"""Compile C directly to a shared library and load it, without cffi.

``cffi.verify`` builds a full CPython extension module: it generates wrapper
glue, compiles it against the Python headers, links against the interpreter,
imports the result, and routes every call through the Python C API.  All of that
exists to make C callable *as Python*.  When the goal is a raw function pointer
handed to a launch shell, none of it is needed, and the extension machinery is
the most fragile part of the build -- it is what forces MSVC, what breaks on a
long temporary path, and what makes reimporting a rebuilt module unsafe.

This module takes the direct route: invoke a real compiler, produce a plain
shared library, ``dlopen`` it, and read the symbol address. No Python headers,
no extension module, no interpreter linkage. cffi remains available and is
tried **last**, only when no toolchain can be found.

Toolchain preference is deliberate.  A self-contained MinGW/Clang install can
build a shared library from a bare environment; MSVC cannot without the
developer environment variables already exported, so it is tried only when the
environment already looks initialised.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence


class ToolchainError(RuntimeError):
    """No usable native toolchain, or a compilation failed."""


@dataclass(frozen=True)
class Toolchain:
    """One compiler able to emit a loadable shared library."""

    name: str
    executable: str
    kind: str  # "gnu" or "msvc"

    def shared_flags(self, output: Path, sources: Sequence[Path]) -> list[str]:
        if self.kind == "msvc":
            return [
                self.executable,
                "/nologo",
                "/O2",
                "/LD",
                *[str(s) for s in sources],
                f"/Fe:{output}",
            ]
        return [
            self.executable,
            "-shared",
            "-O3",
            "-fPIC",
            *[str(s) for s in sources],
            "-o",
            str(output),
        ]


# Ordered by preference.  GNU toolchains first because they build a shared
# library from a bare environment; MSVC needs its developer environment.
_CANDIDATES = (
    ("gcc", "gnu", (r"C:\msys64\mingw64\bin\gcc.exe", r"C:\msys64\ucrt64\bin\gcc.exe")),
    ("clang", "gnu", (r"C:\msys64\clang64\bin\clang.exe",)),
    ("cc", "gnu", ()),
)


def _msvc_available() -> str | None:
    """MSVC only counts when its environment is already initialised."""

    if not os.environ.get("INCLUDE") or not os.environ.get("LIB"):
        return None
    return shutil.which("cl")


@lru_cache(maxsize=1)
def detect_toolchains() -> tuple[Toolchain, ...]:
    """Every usable toolchain, best first."""

    found: list[Toolchain] = []
    for name, kind, extra_paths in _CANDIDATES:
        executable = shutil.which(name)
        if executable is None:
            for candidate in extra_paths:
                if Path(candidate).exists():
                    executable = candidate
                    break
        if executable:
            found.append(Toolchain(name=name, executable=executable, kind=kind))
    msvc = _msvc_available()
    if msvc:
        found.append(Toolchain(name="cl", executable=msvc, kind="msvc"))
    return tuple(found)


def preferred_toolchain() -> Toolchain | None:
    toolchains = detect_toolchains()
    return toolchains[0] if toolchains else None


def _shared_suffix() -> str:
    if sys.platform == "win32":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


@dataclass
class NativeLibrary:
    """A plain shared library loaded by the OS, not a CPython extension."""

    path: Path
    toolchain: str
    handle: ctypes.CDLL

    def address(self, symbol: str) -> int:
        """Raw code address of ``symbol``, suitable for a launch shell."""

        try:
            function = getattr(self.handle, symbol)
        except AttributeError as error:
            raise ToolchainError(
                f"{self.path.name} does not export {symbol!r}"
            ) from error
        return ctypes.cast(function, ctypes.c_void_p).value or 0

    def function(
        self,
        symbol: str,
        *,
        restype=None,
        argtypes: Sequence = (),
    ):
        function = getattr(self.handle, symbol)
        function.restype = restype
        function.argtypes = list(argtypes)
        return function


def compile_shared_library(
    source: str,
    *,
    name: str,
    directory: str | Path | None = None,
    extra_flags: Iterable[str] = (),
    language: str = "c",
    toolchain: Toolchain | None = None,
) -> tuple[Path, Toolchain]:
    """Compile ``source`` to a shared library and return its path."""

    toolchain = toolchain or preferred_toolchain()
    if toolchain is None:
        raise ToolchainError(
            "no native toolchain found; install gcc/clang or initialise MSVC"
        )
    workdir = Path(directory or tempfile.mkdtemp(prefix="turing_native_"))
    workdir.mkdir(parents=True, exist_ok=True)
    suffix = ".cpp" if language in ("c++", "cpp") else ".c"
    source_path = workdir / f"{name}{suffix}"
    source_path.write_text(source, encoding="utf-8")
    output = workdir / f"{name}{_shared_suffix()}"

    command = toolchain.shared_flags(output, [source_path])
    command.extend(extra_flags)

    # A GNU toolchain invoked by absolute path needs its own bin directory on
    # PATH so the compiler's backend can load its support DLLs; without it the
    # driver exits non-zero with no diagnostic at all.
    environment = dict(os.environ)
    environment["PATH"] = (
        str(Path(toolchain.executable).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(workdir),
        env=environment,
    )
    if completed.returncode != 0 or not output.exists():
        raise ToolchainError(
            f"{toolchain.name} failed to build {name}:\n"
            f"{completed.stderr or completed.stdout}"
        )
    return output, toolchain


def compile_and_load(
    source: str,
    *,
    name: str,
    directory: str | Path | None = None,
    extra_flags: Iterable[str] = (),
    language: str = "c",
) -> NativeLibrary:
    """Compile and load ``source``, with no CPython extension in between."""

    path, toolchain = compile_shared_library(
        source,
        name=name,
        directory=directory,
        extra_flags=extra_flags,
        language=language,
    )
    return NativeLibrary(
        path=path,
        toolchain=toolchain.name,
        handle=ctypes.CDLL(str(path)),
    )


def compile_and_load_or_cffi(
    source: str,
    *,
    name: str,
    declarations: str,
    directory: str | Path | None = None,
    extra_flags: Iterable[str] = (),
):
    """Native path first; cffi only when no toolchain exists.

    Returns ``(library, backend)`` where ``backend`` is ``"native"`` or
    ``"cffi"``, so a caller can report -- or reject -- the fallback rather than
    silently paying for it.
    """

    if detect_toolchains():
        return (
            compile_and_load(
                source,
                name=name,
                directory=directory,
                extra_flags=extra_flags,
            ),
            "native",
        )

    from cffi import FFI

    ffi = FFI()
    ffi.cdef(declarations)
    return ffi.verify(source), "cffi"


__all__ = [
    "NativeLibrary",
    "Toolchain",
    "ToolchainError",
    "compile_and_load",
    "compile_and_load_or_cffi",
    "compile_shared_library",
    "detect_toolchains",
    "preferred_toolchain",
]
