"""Reach nodus tensors from Python over the C ABI, using only the standard
library.

``nodus_tensor_core.dll`` owns the tensor arena: a reserved virtual range
committed incrementally, with coalescing span free-lists, clean/dirty leases
zeroed by a background thread, and migration when a range cannot grow in
place.  That is a real allocator, already written and already tested, and
reimplementing it in Python would be duplicating it badly.

The point of going through ``tensor_abi.h`` rather than any array-interchange
library is that tensor transport must not itself depend on numpy, torch,
pybind11, or DLPack.  Everything here is ``ctypes`` and ``struct``.  A payload
moves by mapping arena memory and writing into it, so nothing is copied
through an intermediate array type on the way.

Follows the discovery-and-Structure convention nodus's own
``nodus_headless_bridge.py`` already uses for ``canvas_tables.dll``.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

MAX_RANK = 8


class NodusArenaUnavailable(RuntimeError):
    """The tensor core library could not be found or loaded."""


class NodusArenaError(RuntimeError):
    """The library was reached but refused the request."""


# Mirrors NodusTensorDType in tensor_abi.h.
UNKNOWN = 0
F32 = 1
F64 = 2
I8 = 3
I16 = 4
I32 = 5
I64 = 6
U8 = 7
BYTES = 8
BYTES2 = 9
BYTES4 = 10
BYTES8 = 11
U16 = 12
U32 = 13
U64 = 14
BOOL = 15
PTR = 16

VIRTUAL_ONLY = 0
ALLOW_NON_VIRTUAL = 1

_STATUS = {
    0: "ok",
    -1: "invalid argument",
    -2: "unknown handle",
    -3: "rank above NODUS_TENSOR_MAX_RANK",
    -4: "allocation failed",
    -5: "unsupported",
}

# struct format characters per dtype, for callers that want plain Python
# numbers without bringing in an array library.
_STRUCT_CODE = {
    F32: "f", F64: "d",
    I8: "b", I16: "h", I32: "i", I64: "q",
    U8: "B", U16: "H", U32: "I", U64: "Q",
    BOOL: "?",
}


class NodusTensorDesc(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_uint8),
        ("layout", ctypes.c_uint8),
        ("is_readonly", ctypes.c_uint8),
        ("is_buffer", ctypes.c_uint8),
        ("rank", ctypes.c_uint32),
        ("shape", ctypes.c_uint64 * MAX_RANK),
        ("strides", ctypes.c_uint64 * MAX_RANK),
        ("element_count", ctypes.c_uint64),
        ("element_bytes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("total_bytes", ctypes.c_uint64),
    ]


class NodusArenaStats(ctypes.Structure):
    _fields_ = [
        ("reserve_bytes", ctypes.c_uint64),
        ("active_leases", ctypes.c_uint64),
        ("active_leased_bytes", ctypes.c_uint64),
        ("peak_active_leased_bytes", ctypes.c_uint64),
        ("alloc_calls", ctypes.c_uint64),
        ("free_calls", ctypes.c_uint64),
        ("alloc_fail_oom", ctypes.c_uint64),
        ("free_drop_no_nodes", ctypes.c_uint64),
        ("span_nodes_capacity", ctypes.c_uint32),
        ("span_nodes_free", ctypes.c_uint32),
        ("free_spans", ctypes.c_uint64),
        ("free_bytes", ctypes.c_uint64),
        ("largest_free_span", ctypes.c_uint64),
    ]


def find_tensor_core_library(explicit: str | os.PathLike | None = None) -> Path:
    """Locate ``nodus_tensor_core``.

    Mirrors how ``benchmark_nodus_calculator`` finds its executable: the
    sibling ``nodus`` checkout beside this repository, then the environment.
    """

    if explicit is not None:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate
        raise NodusArenaUnavailable(f"no tensor core library at {candidate}")

    names = (
        ["nodus_tensor_core.dll"]
        if sys.platform == "win32"
        else ["libnodus_tensor_core.so", "libnodus_tensor_core.dylib"]
    )
    roots: list[Path] = []
    environment = os.environ.get("NODUS_TENSOR_CORE")
    if environment:
        candidate = Path(environment)
        if candidate.is_file():
            return candidate
        roots.append(candidate)
    workspace = Path(__file__).resolve().parents[5]
    nodus = workspace / "nodus"
    roots.extend(
        (nodus / "build" / "Release", nodus / "build" / "Debug", nodus / "build")
    )
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
    raise NodusArenaUnavailable(
        "nodus_tensor_core not found; build it or set NODUS_TENSOR_CORE. "
        f"looked in: {', '.join(str(r) for r in roots)}"
    )


def _check(status: int, what: str) -> None:
    if status != 0:
        raise NodusArenaError(
            f"{what} failed: {_STATUS.get(status, f'status {status}')}"
        )


@dataclass(frozen=True)
class TensorView:
    """A borrowed, writable window onto one tensor's arena storage.

    Valid until ``release``; the arena will not move underneath it before
    then, which is exactly what unmapping gives back.
    """

    address: int
    bytes: int
    desc: NodusTensorDesc
    _arena: "NodusArena"
    _handle: int

    def as_ctypes(self, code: type | None = None):
        """The window as a ctypes array, without copying."""

        if code is None:
            code = {
                F32: ctypes.c_float, F64: ctypes.c_double,
                I8: ctypes.c_int8, I16: ctypes.c_int16,
                I32: ctypes.c_int32, I64: ctypes.c_int64,
                U8: ctypes.c_uint8, U16: ctypes.c_uint16,
                U32: ctypes.c_uint32, U64: ctypes.c_uint64,
                BOOL: ctypes.c_bool,
            }.get(int(self.desc.dtype), ctypes.c_uint8)
        count = self.bytes // ctypes.sizeof(code)
        return (code * count).from_address(self.address)

    def release(self) -> None:
        self._arena._library.nodus_tensor_unmap(ctypes.c_uint64(self._handle))

    def __enter__(self) -> "TensorView":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class NodusArena:
    """The nodus tensor arena, as seen from Python."""

    def __init__(self, library_path: str | os.PathLike | None = None):
        path = find_tensor_core_library(library_path)
        if sys.platform == "win32":
            # The tensor core links against torch/CUDA runtimes that sit
            # beside it; PATH alone has not satisfied Windows' DLL search
            # since Python 3.8.
            os.add_dll_directory(str(path.parent))
        try:
            self._library = ctypes.CDLL(str(path))
        except OSError as error:
            raise NodusArenaUnavailable(
                f"could not load {path}: {error}"
            ) from error
        self.path = path
        self._bind()

    def _bind(self) -> None:
        library = self._library
        library.nodus_tensor_arena_configure.restype = ctypes.c_int32
        library.nodus_tensor_arena_configure.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32
        ]
        library.nodus_tensor_arena_stats.restype = ctypes.c_int32
        library.nodus_tensor_arena_stats.argtypes = [
            ctypes.POINTER(NodusArenaStats)
        ]
        library.nodus_tensor_arena_system_available_bytes.restype = ctypes.c_uint64
        library.nodus_tensor_arena_reset.restype = None
        library.nodus_tensor_create.restype = ctypes.c_uint64
        library.nodus_tensor_create.argtypes = [
            ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)
        ]
        library.nodus_tensor_destroy.restype = None
        library.nodus_tensor_destroy.argtypes = [ctypes.c_uint64]
        library.nodus_tensor_describe.restype = ctypes.c_int32
        library.nodus_tensor_describe.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(NodusTensorDesc)
        ]
        library.nodus_tensor_map.restype = ctypes.c_int32
        library.nodus_tensor_map.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        library.nodus_tensor_unmap.restype = None
        library.nodus_tensor_unmap.argtypes = [ctypes.c_uint64]
        library.nodus_tensor_zero.restype = ctypes.c_int32
        library.nodus_tensor_zero.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64
        ]
        library.nodus_tensor_write.restype = ctypes.c_int64
        library.nodus_tensor_write.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64
        ]
        library.nodus_tensor_read.restype = ctypes.c_int64
        library.nodus_tensor_read.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64
        ]
        library.nodus_tensor_allocation.restype = ctypes.c_int32
        library.nodus_tensor_allocation.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint16),
        ]
        library.nodus_tensor_dtype_size.restype = ctypes.c_uint32
        library.nodus_tensor_dtype_size.argtypes = [ctypes.c_int32]

    # -- arena ------------------------------------------------------------

    def configure(
        self,
        *,
        reserve_bytes: int = 0,
        min_commit_bytes: int = 0,
        span_nodes: int = 0,
        backing_policy: int = VIRTUAL_ONLY,
    ) -> None:
        """Set arena policy. Must precede the first allocation: the backend
        fixes its backing at first use. Zero leaves a knob unchanged."""

        _check(
            self._library.nodus_tensor_arena_configure(
                reserve_bytes, min_commit_bytes, span_nodes, backing_policy
            ),
            "arena configure",
        )

    def stats(self) -> NodusArenaStats:
        stats = NodusArenaStats()
        _check(
            self._library.nodus_tensor_arena_stats(ctypes.byref(stats)),
            "arena stats",
        )
        return stats

    def system_available_bytes(self) -> int:
        return int(self._library.nodus_tensor_arena_system_available_bytes())

    def dtype_size(self, dtype: int) -> int:
        return int(self._library.nodus_tensor_dtype_size(dtype))

    # -- tensors ----------------------------------------------------------

    def create(self, dtype: int, shape: Sequence[int]) -> int:
        extents = tuple(int(size) for size in shape)
        if len(extents) > MAX_RANK:
            raise NodusArenaError(
                f"rank {len(extents)} exceeds NODUS_TENSOR_MAX_RANK={MAX_RANK}"
            )
        buffer = (ctypes.c_uint64 * max(len(extents), 1))(*extents)
        handle = int(
            self._library.nodus_tensor_create(dtype, len(extents), buffer)
        )
        if handle == 0:
            raise NodusArenaError(
                f"could not create tensor dtype={dtype} shape={extents}"
            )
        return handle

    def destroy(self, handle: int) -> None:
        self._library.nodus_tensor_destroy(ctypes.c_uint64(handle))

    def describe(self, handle: int) -> NodusTensorDesc:
        desc = NodusTensorDesc()
        _check(
            self._library.nodus_tensor_describe(
                ctypes.c_uint64(handle), ctypes.byref(desc)
            ),
            "describe",
        )
        return desc

    def map(self, handle: int) -> TensorView:
        address = ctypes.c_void_p()
        size = ctypes.c_uint64()
        _check(
            self._library.nodus_tensor_map(
                ctypes.c_uint64(handle), ctypes.byref(address), ctypes.byref(size)
            ),
            "map",
        )
        return TensorView(
            address=int(address.value or 0),
            bytes=int(size.value),
            desc=self.describe(handle),
            _arena=self,
            _handle=handle,
        )

    def zero(self, handle: int, *, byte_offset: int = 0, byte_count: int = 0) -> None:
        _check(
            self._library.nodus_tensor_zero(
                ctypes.c_uint64(handle), byte_offset, byte_count
            ),
            "zero",
        )

    def write_bytes(self, handle: int, payload: bytes, *, byte_offset: int = 0) -> int:
        written = int(
            self._library.nodus_tensor_write(
                ctypes.c_uint64(handle), byte_offset, payload, len(payload)
            )
        )
        if written < 0:
            _check(written, "write")
        return written

    def read_bytes(self, handle: int, count: int, *, byte_offset: int = 0) -> bytes:
        buffer = ctypes.create_string_buffer(count)
        got = int(
            self._library.nodus_tensor_read(
                ctypes.c_uint64(handle), byte_offset, buffer, count
            )
        )
        if got < 0:
            _check(got, "read")
        return buffer.raw[:got]

    def allocation(self, handle: int) -> tuple[int, int, int]:
        """Where this tensor sits in the arena, for a compiled kernel that
        addresses arena memory directly instead of through a mapping."""

        address = ctypes.c_void_p()
        size = ctypes.c_uint64()
        bucket = ctypes.c_uint16()
        _check(
            self._library.nodus_tensor_allocation(
                ctypes.c_uint64(handle),
                ctypes.byref(address),
                ctypes.byref(size),
                ctypes.byref(bucket),
            ),
            "allocation",
        )
        return int(address.value or 0), int(size.value), int(bucket.value)

    # -- convenience ------------------------------------------------------

    def from_values(
        self, dtype: int, shape: Sequence[int], values: Iterable[float]
    ) -> int:
        """Create a tensor and fill it from ordinary Python numbers.

        Uses ``struct`` rather than an array library so the transport keeps
        no dependency of its own.
        """

        import struct

        code = _STRUCT_CODE.get(dtype)
        if code is None:
            raise NodusArenaError(f"no struct code for dtype {dtype}")
        data = tuple(values)
        handle = self.create(dtype, shape)
        payload = struct.pack(f"<{len(data)}{code}", *data)
        self.write_bytes(handle, payload)
        return handle

    def to_values(self, handle: int) -> tuple:
        import struct

        desc = self.describe(handle)
        code = _STRUCT_CODE.get(int(desc.dtype))
        if code is None:
            raise NodusArenaError(f"no struct code for dtype {desc.dtype}")
        count = int(desc.element_count)
        raw = self.read_bytes(handle, count * int(desc.element_bytes))
        return struct.unpack(f"<{count}{code}", raw)


_ARENA: NodusArena | None = None


def arena() -> NodusArena:
    """The process-wide arena. One instance: the backend's own singleton is
    process-wide, so a second wrapper would be a second view of one thing."""

    global _ARENA
    if _ARENA is None:
        _ARENA = NodusArena()
    return _ARENA


__all__ = [
    "MAX_RANK",
    "NodusArena",
    "NodusArenaError",
    "NodusArenaStats",
    "NodusArenaUnavailable",
    "NodusTensorDesc",
    "TensorView",
    "arena",
    "find_tensor_core_library",
    "F32", "F64", "I8", "I16", "I32", "I64",
    "U8", "U16", "U32", "U64", "BOOL",
    "VIRTUAL_ONLY", "ALLOW_NON_VIRTUAL",
]
