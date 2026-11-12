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
    return np.ascontiguousarray(array)


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
    region = np.frombuffer(raw, dtype=dtype).astype(np.dtype(name), copy=True).reshape(tuple(shape))
    from ..abstraction import AbstractTensor

    return AbstractTensor.tensor(region)


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


def load(path: Any) -> Any:
    """The tensor saved under ``path``: from memory inside the minute, else from the file."""

    payload = retained(path)
    if payload is None:
        with open(os.fspath(path), "rb") as stream:
            payload = stream.read()
    return loads(payload)


__all__ = ["DTYPES", "MAGIC", "RETAIN_SECONDS", "dtype_name", "dumps", "load", "loads", "retained", "save", "write_file"]
