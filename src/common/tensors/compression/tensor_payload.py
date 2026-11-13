"""A tensor as a compressed payload, and back: what ``pickle`` stored, without pickle.

A tensor is a handle and a data region, and the payload saves both:

- the handle: its dtype and shape, a ``tile_row_codec`` record (small, uncompressed);
- the region: its elements' bytes in C order, little-endian, compressed by ``octet_codec`` (AbstractTensor LZ77,
  then canonical Huffman).

Read back, the handle is restored first and the region is laid under it: an AbstractTensor of the same dtype and
shape whose elements are the same bits. A compiled program's ``pickle.dump(tensor, stream)`` and
``pickle.load(stream)`` are lowered by the ingester's special cases (``transmogrifier/graph/python_special_cases.py``)
to :func:`save` and :func:`load`, served by the hosting Python shell until this codec itself lowers to native code.
Anything that is not a tensor of one of :data:`DTYPES` is refused with ``TypeError`` before anything is written.

The host answers ``save`` once the payload is held: it keeps every saved payload in memory for
:data:`RETAIN_SECONDS` (a minute), and a ``load`` of that path inside the minute is served from memory without
touching storage. Durable storage is the ``store`` given to :func:`save` (by default the file at ``path``); a Nodus
file FIFO (a FIFO over ``gp_mem_backend_create_filesystem``) is meant to take that place.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

import numpy as np

from ....compiler import tile_row_codec

MAGIC = b"TURTP2"

# How long the host keeps a saved payload in memory, in seconds.
RETAIN_SECONDS = 60.0

_retained: dict[str, tuple[float, bytes]] = {}
_retained_lock = threading.Lock()

# The dtypes a payload carries, by the name its handle records.
DTYPES = ("bool", "uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64")


def dtype_name(value: Any) -> str:
    """The recorded name of a tensor's dtype (AbstractTensor, torch or NumPy)."""

    getter = getattr(value, "get_dtype", None)
    dtype = getter() if callable(getter) else getattr(value, "dtype", None)
    name = str(dtype).rsplit(".", 1)[-1].casefold()
    if name not in DTYPES:
        raise TypeError(f"a tensor payload cannot carry dtype {dtype!r}")
    return name


def _region(value: Any) -> np.ndarray:
    """The tensor's data region as a C-ordered NumPy view (copied only when it is not one already)."""

    if isinstance(value, np.ndarray):
        array = value
    elif callable(getattr(value, "numpy", None)):
        array = value.numpy()                         # AbstractTensor and torch: the backing data, on the host
        array = getattr(array, "data", array) if not isinstance(array, np.ndarray) else array
    else:
        raise TypeError(f"a tensor payload holds tensors, not {type(value).__name__}")
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{type(value).__name__} gave no host data region")
    # ``ascontiguousarray`` returns at least one axis: a rank-0 tensor (YoungMan's ``vertex_count``) was saved with
    # the handle ``(1,)``.  The handle is the tensor's own shape, so the copy keeps it.
    return np.ascontiguousarray(array).reshape(array.shape)


def dumps(value: Any) -> bytes:
    """The payload of one tensor: its handle, then its compressed region."""

    name = dtype_name(value)
    region = _region(value)
    if region.dtype != np.dtype(name):
        raise TypeError(f"the tensor's region is {region.dtype}, its handle says {name}")
    raw = region.astype(region.dtype.newbyteorder("<"), copy=False).tobytes(order="C")
    handle = (name, tuple(int(extent) for extent in region.shape))
    return MAGIC + tile_row_codec.encode_rows((handle, tile_row_codec.compress_octets(raw)))


def loads(data: bytes) -> Any:
    """The AbstractTensor a payload holds: its handle restored, its region laid under it."""

    from ..abstraction import AbstractTensor

    return AbstractTensor.tensor(_decoded_region(data))


def _decoded_region(data: bytes) -> np.ndarray:
    """A payload's region as a C-ordered NumPy array of its handle's dtype and shape."""

    data = bytes(data)
    if not data.startswith(MAGIC):
        raise ValueError("not a tensor payload")
    (name, shape), packed = tile_row_codec.decode_rows(data[len(MAGIC):])
    if name not in DTYPES:
        raise ValueError(f"tensor payload records unknown dtype {name!r}")
    dtype = np.dtype(name).newbyteorder("<")
    raw = tile_row_codec.decompress_octets(packed)
    count = 1
    for extent in shape:
        count *= int(extent)
    if len(raw) != count * dtype.itemsize:
        raise ValueError("tensor payload region disagrees with its handle")
    return np.frombuffer(raw, dtype=dtype).astype(np.dtype(name), copy=True).reshape(tuple(shape))


def _key(path: Any) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def write_file(path: Any, payload: bytes) -> None:
    """The default durable store: the file at ``path``, replaced whole (never seen half written)."""

    target = os.fspath(path)
    temporary = f"{target}.partial"
    with open(temporary, "wb") as stream:
        stream.write(payload)
    os.replace(temporary, target)


def retained(path: Any, now: float | None = None) -> bytes | None:
    """The payload saved to ``path`` within the last :data:`RETAIN_SECONDS`, or None."""

    now = time.monotonic() if now is None else now
    with _retained_lock:
        for key in [key for key, (deadline, _payload) in _retained.items() if deadline <= now]:
            del _retained[key]
        held = _retained.get(_key(path))
    return None if held is None else held[1]


def save(path: Any, value: Any, *, store: Callable[[Any, bytes], None] | None = write_file) -> int:
    """Save one tensor's payload under ``path``; the number of payload bytes.

    The payload is held in memory for :data:`RETAIN_SECONDS` and handed to ``store`` (``None``: memory only)."""

    payload = dumps(value)
    with _retained_lock:
        _retained[_key(path)] = (time.monotonic() + RETAIN_SECONDS, payload)
    if store is not None:
        store(path, payload)
    return len(payload)


def _payload(path: Any) -> bytes:
    """The payload saved under ``path``: from memory inside the minute, else from the file."""

    payload = retained(path)
    if payload is None:
        with open(os.fspath(path), "rb") as stream:
            payload = stream.read()
    return payload


def load(path: Any) -> Any:
    """The tensor saved under ``path``: from memory inside the minute, else from the file."""

    return loads(_payload(path))


# Caller-owned results.  A compiled caller provides all the memory a load needs (the SSA and backend ethos), so the
# region is sized before the load from a layout: the handle saved in the payload, or a stride tuple the caller states
# in its place.  The status is the shell's external-result convention (``utf8_sequence``): 1 complete, 2 over the
# caller's capacity, 3 failed.
STATUS_COMPLETE = 1
STATUS_OVER_CAPACITY = 2
STATUS_FAILED = 3


def read_handle(source: Any) -> tuple[str, tuple[int, ...]]:
    """A payload's handle, ``(dtype name, shape)``, read without decompressing its region.

    ``source`` is a payload (``bytes``) or the path it was saved under.
    """

    data = bytes(source) if isinstance(source, (bytes, bytearray, memoryview)) else _payload(source)
    if not data.startswith(MAGIC):
        raise ValueError("not a tensor payload")
    (name, shape), _packed = tile_row_codec.decode_rows(data[len(MAGIC):])
    if name not in DTYPES:
        raise ValueError(f"tensor payload records unknown dtype {name!r}")
    return str(name), tuple(int(extent) for extent in shape)


def c_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The element strides of a C-ordered region of ``shape`` (the layout a payload is saved in)."""

    strides, step = [], 1
    for extent in reversed(tuple(shape)):
        strides.append(step)
        step *= max(int(extent), 1)
    return tuple(reversed(strides))


def region_length(shape: tuple[int, ...], strides: tuple[int, ...] | None = None) -> int:
    """The elements a region of ``shape`` laid out by ``strides`` spans: what the caller provides.

    ``strides`` (in elements, one per axis) overrides the saved C order; the length is the last element's offset plus
    one, ``1 + sum((extent - 1) * stride)``, or 0 for an empty tensor.
    """

    shape = tuple(int(extent) for extent in shape)
    strides = c_strides(shape) if strides is None else tuple(int(stride) for stride in strides)
    if len(strides) != len(shape):
        raise ValueError(f"a stride tuple of {len(strides)} axes cannot lay out a rank-{len(shape)} tensor")
    if any(stride < 0 for stride in strides):
        raise ValueError("a region's strides are not negative")
    if any(extent == 0 for extent in shape):
        return 0
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))


def load_into(
    path: Any, region: Any, handle: Any,
    extents: tuple[int, ...] | None = None, strides: tuple[int, ...] | None = None,
) -> int:
    """Fill the caller's ``region`` with the tensor saved under ``path``; the status.

    ``region`` is the caller's writable one-dimensional buffer of the tensor's dtype, its length the capacity.
    ``extents`` reshapes on load: the saved elements, in their C order, are the caller's ``extents`` (same element
    count as the saved handle); without it the saved shape is kept.  ``strides`` (in elements, one per axis of those
    extents) lays them out in the region; without it, C order.  The caller sizes the region with
    :func:`region_length` of the same extents and strides.  ``handle`` is the caller's writable int64 buffer; it
    receives ``rank, *extents, *strides`` (the layout the region now holds), so its capacity is ``1 + 2 * rank``.
    Nothing is allocated for the caller.  A region or handle too small is :data:`STATUS_OVER_CAPACITY` with nothing
    written; a payload that is unreadable or of another dtype than the region, extents of another element count than
    the saved handle (or negative), or a stride tuple of another rank than the extents is :data:`STATUS_FAILED`.
    """

    try:
        name, saved = read_handle(path)
        shape = saved if extents is None else tuple(int(extent) for extent in extents)
        if any(extent < 0 for extent in shape) or int(np.prod(shape, dtype=np.int64)) != int(
            np.prod(saved, dtype=np.int64)
        ):
            return STATUS_FAILED
        layout = c_strides(shape) if strides is None else tuple(int(stride) for stride in strides)
        length = region_length(shape, layout)
        region_array = np.asarray(region)
        handle_array = np.asarray(handle)
        if (
            region_array.ndim != 1 or region_array.dtype != np.dtype(name)
            or handle_array.ndim != 1 or handle_array.dtype != np.dtype("int64")
        ):
            return STATUS_FAILED
        if length > region_array.shape[0] or 1 + 2 * len(shape) > handle_array.shape[0]:
            return STATUS_OVER_CAPACITY
        values = _decoded_region(_payload(path))
        if length:
            itemsize = region_array.dtype.itemsize
            view = np.lib.stride_tricks.as_strided(
                region_array, shape=shape, strides=tuple(stride * itemsize for stride in layout),
            )
            view[...] = values.reshape(shape)
        handle_array[0] = len(shape)
        handle_array[1:1 + len(shape)] = shape
        handle_array[1 + len(shape):1 + 2 * len(shape)] = layout
        return STATUS_COMPLETE
    except (OSError, ValueError, TypeError):
        return STATUS_FAILED


__all__ = [
    "DTYPES", "MAGIC", "RETAIN_SECONDS", "STATUS_COMPLETE", "STATUS_FAILED", "STATUS_OVER_CAPACITY", "c_strides",
    "dtype_name", "dumps", "load", "load_into", "loads", "read_handle", "region_length", "retained", "save",
    "write_file",
]
