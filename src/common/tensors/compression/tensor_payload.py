"""A tensor as a compressed octet payload, and back: what ``pickle`` stored, without pickle.

A compiled program's ``pickle.dump(tensor, stream)`` and ``pickle.load(stream)`` are lowered by the ingester's
special cases (``transmogrifier/graph/python_special_cases.py``) to calls of :func:`save` and :func:`load`, served by
the hosting Python shell until this codec itself lowers to native code.

The payload is ``tile_row_codec`` rows ``(dtype, shape, values)`` -- floats as their exact binary64 bits, integers
exactly, booleans as booleans -- compressed by ``octet_codec`` (AbstractTensor LZ77, then canonical Huffman). It is
lossless for every dtype the rows can carry, and refuses anything else (a tensor of objects, an unknown dtype) with
``TypeError`` before anything is written. Read back, it is an AbstractTensor of the same dtype and shape.
"""

from __future__ import annotations

import os
from typing import Any

from ....compiler import tile_row_codec

MAGIC = b"TURTP1"

# The dtypes a payload carries, by the name it records. Each is read back as that dtype.
DTYPES = (
    "bool", "uint8", "int8", "int16", "int32", "int64",
    "float16", "bfloat16", "float32", "float64",
)


def dtype_name(value: Any) -> str:
    """The recorded name of a tensor's dtype (AbstractTensor, torch or NumPy)."""

    getter = getattr(value, "get_dtype", None)
    dtype = getter() if callable(getter) else getattr(value, "dtype", None)
    name = str(dtype).rsplit(".", 1)[-1].casefold()
    if name not in DTYPES:
        raise TypeError(f"a tensor payload cannot carry dtype {dtype!r}")
    return name


def _shape(value: Any) -> tuple[int, ...]:
    getter = getattr(value, "get_shape", None)
    shape = getter() if callable(getter) else getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{type(value).__name__} has no shape; a tensor payload holds tensors")
    return tuple(int(extent) for extent in shape)


def dumps(value: Any) -> bytes:
    """The compressed payload of one tensor."""

    if not hasattr(value, "tolist"):
        raise TypeError(f"a tensor payload holds tensors, not {type(value).__name__}")
    rows = (dtype_name(value), _shape(value), value.tolist())
    return MAGIC + tile_row_codec.compress_octets(tile_row_codec.encode_rows(rows))


def loads(data: bytes) -> Any:
    """The AbstractTensor a payload holds."""

    data = bytes(data)
    if not data.startswith(MAGIC):
        raise ValueError("not a tensor payload")
    name, shape, values = tile_row_codec.decode_rows(tile_row_codec.decompress_octets(data[len(MAGIC):]))
    if name not in DTYPES:
        raise ValueError(f"tensor payload records unknown dtype {name!r}")
    from ..abstraction import AbstractTensor

    tensor = AbstractTensor.tensor(values, dtype=name)
    if tuple(int(extent) for extent in tensor.get_shape()) != tuple(shape):
        tensor = tensor.reshape(tuple(shape))
    return tensor


def save(path: Any, value: Any) -> int:
    """Write one tensor's payload to ``path`` (replacing it); the number of bytes written."""

    payload = dumps(value)
    target = os.fspath(path)
    temporary = f"{target}.partial"
    with open(temporary, "wb") as stream:
        stream.write(payload)
    os.replace(temporary, target)
    return len(payload)


def load(path: Any) -> Any:
    """The tensor whose payload ``path`` holds."""

    with open(os.fspath(path), "rb") as stream:
        return loads(stream.read())


__all__ = ["DTYPES", "MAGIC", "dtype_name", "dumps", "load", "loads", "save"]
