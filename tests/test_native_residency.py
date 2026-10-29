"""Compiled native libraries stay resident: loaded once, bound once."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys

import pytest

from src.compiler import native_residency


def _shared_library(tmp_path, name="hot"):
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None or sys.platform == "win32":
        pytest.skip("needs a POSIX C compiler to build a shared library")
    source = tmp_path / f"{name}.c"
    source.write_text(
        "long long twice(long long x) { return 2 * x; }\n", encoding="utf-8",
    )
    library = tmp_path / f"lib{name}.so"
    subprocess.run(
        [compiler, "-shared", "-fPIC", "-O2", "-o", str(library), str(source)],
        check=True, capture_output=True,
    )
    return library


def test_one_image_per_path_for_the_life_of_the_process(tmp_path, monkeypatch):
    library_path = tmp_path / "artifact.so"
    library_path.write_bytes(b"loaded by monkeypatch")
    loads = []
    monkeypatch.setattr(
        ctypes, "CDLL", lambda path: loads.append(path) or object(),
    )
    first = native_residency.resident_library(library_path)
    # A relative spelling of the same file resolves to the same image.
    monkeypatch.chdir(tmp_path)
    second = native_residency.resident_library("artifact.so")
    assert first is second
    assert loads == [str(library_path.resolve())]
    record = next(
        item for item in native_residency.native_residency_state()["libraries"]
        if item["path"] == str(library_path.resolve())
    )
    assert record["reuses"] == 1
    assert record["load_seconds"] is not None


def test_entry_abi_is_declared_once_and_kept_per_signature(tmp_path):
    library_path = _shared_library(tmp_path)
    entry = native_residency.resident_entry(
        library_path, "twice", (ctypes.c_longlong,), ctypes.c_longlong,
    )
    assert entry(21) == 42
    assert native_residency.resident_entry(
        library_path, "twice", (ctypes.c_longlong,), ctypes.c_longlong,
    ) is entry
    # A second consumer binding the same symbol with another ABI gets its
    # own function object and does not rewrite the first one's argtypes.
    other = native_residency.resident_entry(
        library_path, "twice", (ctypes.c_void_p,), ctypes.c_void_p,
    )
    assert other is not entry
    assert tuple(entry.argtypes) == (ctypes.c_longlong,)
    assert entry.restype is ctypes.c_longlong
    assert entry(-4) == -8


def test_backends_share_the_resident_table(tmp_path):
    from src.compiler.ssa_fortran_backend import FortranCoreNative

    library_path = _shared_library(tmp_path, "shared")
    loaded = native_residency.resident_library(library_path)
    assert FortranCoreNative._LIBRARIES[str(library_path.resolve())] is loaded
    assert native_residency.preload_libraries((library_path,)) == (loaded,)
