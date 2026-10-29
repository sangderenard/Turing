"""Process-lifetime residency of compiled native libraries.

A Python library with native code loads its binary once, when the code
loads, and every later call goes straight into that resident image.  The
compiler's artifacts follow the same rule: one loaded image per published
library path, held for the life of the process and never unloaded, shared
by every artifact, verifier and installed callable that names that path.

The rule is the Fortran lane's, now kept in one place for every backend.  A
standalone-linked gfortran artifact carries its own runtime, and unloading
that runtime at DLL_PROCESS_DETACH takes the process down (observed as an
access violation inside garbage collection once the last reference to a
compiled core was dropped).  Holding the handle here means a library is
loaded once, never unloaded, and re-used by every later consumer that
resolves to the same path.

``os.add_dll_directory`` search handles are active only while the handle
lives, so the runtime directories a library's published contract names are
registered once and retained beside the library table.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping


#: Loaded libraries by resolved path, kept for the life of the process.
_RESIDENT: dict[str, Any] = {}
#: Retained ``os.add_dll_directory`` handles by resolved directory.
_DLL_DIRECTORIES: dict[str, Any] = {}
#: Bound entry points by (resolved path, symbol, argtypes, restype).
_ENTRIES: dict[tuple[str, str, tuple[Any, ...], Any], Any] = {}
#: Residency evidence by resolved path: load time and how often it was reused.
_RECORDS: dict[str, dict[str, Any]] = {}
_LOCK = threading.RLock()


def _resolved(path: str | os.PathLike[str]) -> str:
    return str(Path(path).resolve())


def retain_dll_directories(
    runtime_dependencies: Iterable[Mapping[str, Any]] = (),
) -> tuple[Any, ...]:
    """Register each published runtime directory once and keep it registered.

    ``runtime_dependencies`` is the ``metadata.runtime_dependencies`` list a
    compiled API publishes.  Only Windows has a per-process DLL search list;
    elsewhere the loader resolves dependencies itself and nothing is held.
    """

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return ()
    handles = []
    with _LOCK:
        for dependency in runtime_dependencies:
            dependency_path = Path(str(dependency.get("path") or ""))
            if not dependency_path.is_file():
                continue
            directory = str(dependency_path.parent.resolve())
            handle = _DLL_DIRECTORIES.get(directory)
            if handle is None:
                handle = os.add_dll_directory(directory)
                _DLL_DIRECTORIES[directory] = handle
            handles.append(handle)
    return tuple(handles)


def resident_library(
    path: str | os.PathLike[str],
    *,
    runtime_dependencies: Iterable[Mapping[str, Any]] = (),
) -> Any:
    """The process's one loaded image of ``path``, loading it on first use."""

    resolved = _resolved(path)
    with _LOCK:
        library = _RESIDENT.get(resolved)
        if library is not None:
            record = _RECORDS.setdefault(
                resolved, {"path": resolved, "load_seconds": None, "reuses": 0},
            )
            record["reuses"] += 1
            return library
        retain_dll_directories(runtime_dependencies)
        started = time.perf_counter()
        library = ctypes.CDLL(resolved)
        _RESIDENT[resolved] = library
        _RECORDS[resolved] = {
            "path": resolved,
            "load_seconds": time.perf_counter() - started,
            "reuses": 0,
        }
        return library


def resident_entry(
    path: str | os.PathLike[str],
    symbol: str,
    argtypes: Iterable[Any],
    restype: Any = None,
    *,
    runtime_dependencies: Iterable[Mapping[str, Any]] = (),
) -> Any:
    """A bound entry point whose ABI is declared once, not on every call.

    Each distinct declared ABI gets its own function object (``library[name]``
    returns a fresh one), so two consumers that bind one symbol differently
    never overwrite each other's ``argtypes``.
    """

    resolved = _resolved(path)
    argument_types = tuple(argtypes)
    key = (resolved, str(symbol), argument_types, restype)
    with _LOCK:
        entry = _ENTRIES.get(key)
        if entry is not None:
            return entry
        library = resident_library(
            resolved, runtime_dependencies=runtime_dependencies,
        )
        entry = library[str(symbol)]
        entry.argtypes = list(argument_types)
        entry.restype = restype
        _ENTRIES[key] = entry
        return entry


def preload_libraries(
    paths: Iterable[str | os.PathLike[str]],
    *,
    runtime_dependencies: Iterable[Mapping[str, Any]] = (),
) -> tuple[Any, ...]:
    """Load every named library now, so the first call is already hot."""

    dependencies = tuple(runtime_dependencies)
    return tuple(
        resident_library(path, runtime_dependencies=dependencies)
        for path in paths
    )


def native_residency_state() -> dict[str, Any]:
    """Which images are resident, what each load cost, and how often reused."""

    with _LOCK:
        return {
            "schema": "turing.native-residency-state.v1",
            "libraries": [dict(_RECORDS[path]) for path in sorted(_RECORDS)],
            "bound_entries": len(_ENTRIES),
            "dll_directories": sorted(_DLL_DIRECTORIES),
        }
