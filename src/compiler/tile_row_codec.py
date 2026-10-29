"""The rows of one compressed tile as an octet string, and back.

This replaces pickle for ``CompressedTable``.  A tile's rows are written as a
tagged octet stream; the stream's compression is
``common.tensors.compression.octet_codec`` (AbstractTensor LZ77 matches,
then canonical Huffman).  Every value is one of a closed set of shapes, each with an exact
inverse:

- ``None``, ``bool``, ``int`` (unbounded: zigzag varint), ``float`` (IEEE
  binary64 bits), ``str`` and ``bytes``;
- ``tuple``, ``list``, ``dict``, ``set`` and ``frozenset`` of values;
- a class or a module-level function, by ``module`` and ``qualname``;
- an ``Enum`` member, by its class and member name;
- a named tuple, by its class and items;
- a record: a dataclass instance, or an instance of a class that keeps the
  default object state protocol, by its class and its attribute state;
- a reduced value: ``reducers[type](value)`` returned ``(function, args)``
  for a module-level function (the identity registry's pages, stages,
  transforms and reasons are written by name this way, and read back
  through the same function).

Strings are interned per tile: the first occurrence carries its text, later
ones its index, so repeated row text costs one varint.  Anything else --
a lambda, a cycle, a class with its own reduction -- is refused with
``TypeError`` before the tile is written, never written lossily.
"""

from __future__ import annotations

import dataclasses
import struct
import sys
from enum import Enum
from typing import Any, Callable, Mapping



def compress_octets(data: bytes) -> bytes:
    """``octet_codec.compress_octets``; imported on use, since the tensor
    package imports the tile store that imports this module."""

    from ..common.tensors.compression.octet_codec import compress_octets

    return compress_octets(data)


def decompress_octets(packet: bytes) -> bytes:
    from ..common.tensors.compression.octet_codec import decompress_octets

    return decompress_octets(packet)

_NONE, _FALSE, _TRUE, _INT, _FLOAT, _STR, _STR_REF, _BYTES = range(8)
_TUPLE, _LIST, _DICT, _SET, _FROZENSET = range(8, 13)
_GLOBAL, _ENUM, _NAMED_TUPLE, _RECORD, _REDUCED = range(13, 18)
_FLOAT_BITS = struct.Struct("<d")


def _varint(out: bytearray, value: int) -> None:
    while True:
        low = value & 0x7F
        value >>= 7
        if value:
            out.append(low | 0x80)
        else:
            out.append(low)
            return


def _global_name(value: Any) -> tuple[str, str]:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str) or (
        "<" in qualname
    ):
        raise TypeError(f"{value!r} has no importable module-level name")
    if _resolve(module, qualname) is not value:
        raise TypeError(f"{module}.{qualname} does not name {value!r}")
    return module, qualname


def _resolve(module: str, qualname: str) -> Any:
    owner = sys.modules.get(module)
    if owner is None:
        raise TypeError(f"module {module} is not loaded")
    value = owner
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def _default_state_protocol(cls: type) -> bool:
    return (
        cls.__reduce_ex__ is object.__reduce_ex__
        and cls.__reduce__ is object.__reduce__
        and getattr(cls, "__getstate__", object.__getstate__)
        is object.__getstate__
        and not hasattr(cls, "__setstate__")
        and not hasattr(cls, "__getnewargs_ex__")
        and not hasattr(cls, "__getnewargs__")
    )


def _slot_names(cls: type) -> tuple[str, ...]:
    names = []
    for base in cls.__mro__:
        slots = base.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name not in {"__dict__", "__weakref__"} and name not in names:
                names.append(name)
    return tuple(names)


def _record_state(value: Any) -> tuple[tuple[str, Any], ...]:
    state = list(getattr(value, "__dict__", {}).items())
    for name in _slot_names(type(value)):
        try:
            state.append((name, object.__getattribute__(value, name)))
        except AttributeError:
            continue
    return tuple(state)


class _Writer:
    def __init__(self, reducers: Mapping[type, Callable[[Any], Any]] | None):
        self.out = bytearray()
        self.strings: dict[str, int] = {}
        self.reducers = dict(reducers or {})
        self.path: set[int] = set()

    def text(self, value: str) -> None:
        index = self.strings.get(value)
        if index is not None:
            self.out.append(_STR_REF)
            _varint(self.out, index)
            return
        self.strings[value] = len(self.strings)
        encoded = value.encode("utf-8", "surrogatepass")
        self.out.append(_STR)
        _varint(self.out, len(encoded))
        self.out += encoded

    def global_ref(self, value: Any) -> None:
        module, qualname = _global_name(value)
        self.out.append(_GLOBAL)
        self.text(module)
        self.text(qualname)

    def items(self, tag: int, values) -> None:
        values = tuple(values)
        self.out.append(tag)
        _varint(self.out, len(values))
        for item in values:
            self.value(item)

    def value(self, value: Any) -> None:
        kind = type(value)
        if value is None:
            self.out.append(_NONE)
        elif kind is bool:
            self.out.append(_TRUE if value else _FALSE)
        elif kind is int:
            self.out.append(_INT)
            _varint(self.out, (value << 1) if value >= 0 else ((-value << 1) - 1))
        elif kind is float:
            self.out.append(_FLOAT)
            self.out += _FLOAT_BITS.pack(value)
        elif kind is str:
            self.text(value)
        elif kind is bytes:
            self.out.append(_BYTES)
            _varint(self.out, len(value))
            self.out += value
        elif isinstance(value, type) or kind.__name__ in {
            "function", "builtin_function_or_method",
        }:
            self.global_ref(value)
        else:
            self.container(value, kind)

    def container(self, value: Any, kind: type) -> None:
        if id(value) in self.path:
            raise TypeError(f"{kind.__qualname__} value refers to itself")
        self.path.add(id(value))
        try:
            reducer = self.reducers.get(kind)
            reduced = None if reducer is None else reducer(value)
            if (
                isinstance(reduced, tuple) and len(reduced) == 2
                and isinstance(reduced[1], tuple)
                and type(reduced[0]).__name__ == "function"
                and reduced[0].__module__ != "copyreg"
            ):
                self.out.append(_REDUCED)
                self.global_ref(reduced[0])
                self.items(_TUPLE, reduced[1])
            elif kind is tuple:
                self.items(_TUPLE, value)
            elif kind is list:
                self.items(_LIST, value)
            elif kind is dict:
                self.out.append(_DICT)
                _varint(self.out, len(value))
                for key, item in value.items():
                    self.value(key)
                    self.value(item)
            elif kind is set:
                self.items(_SET, value)
            elif kind is frozenset:
                self.items(_FROZENSET, value)
            elif isinstance(value, Enum):
                self.out.append(_ENUM)
                self.global_ref(kind)
                self.text(value.name)
            elif isinstance(value, tuple) and hasattr(kind, "_fields"):
                self.out.append(_NAMED_TUPLE)
                self.global_ref(kind)
                self.items(_TUPLE, value)
            elif dataclasses.is_dataclass(value) or _default_state_protocol(kind):
                if not dataclasses.is_dataclass(value) and kind.__module__ == "builtins":
                    raise TypeError(f"{kind.__qualname__} has no row encoding")
                state = _record_state(value)
                self.out.append(_RECORD)
                self.global_ref(kind)
                _varint(self.out, len(state))
                for name, item in state:
                    self.text(name)
                    self.value(item)
            else:
                raise TypeError(f"{kind.__qualname__} has no row encoding")
        finally:
            self.path.discard(id(value))


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.at = 0
        self.strings: list[str] = []

    def byte(self) -> int:
        value = self.data[self.at]
        self.at += 1
        return value

    def varint(self) -> int:
        result = shift = 0
        while True:
            byte = self.byte()
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7

    def take(self, count: int) -> bytes:
        if self.at + count > len(self.data):
            raise ValueError("tile row stream is truncated")
        value = self.data[self.at:self.at + count]
        self.at += count
        return value

    def text(self) -> str:
        value = self.value()
        if type(value) is not str:
            raise ValueError("tile row stream names a non-string")
        return value

    def values(self) -> list:
        return [self.value() for _ in range(self.varint())]

    def global_ref(self) -> Any:
        if self.byte() != _GLOBAL:
            raise ValueError("tile row stream expected a global name")
        module = self.text()
        return _resolve(module, self.text())

    def value(self) -> Any:
        tag = self.byte()
        if tag == _NONE:
            return None
        if tag in (_FALSE, _TRUE):
            return tag == _TRUE
        if tag == _INT:
            raw = self.varint()
            return -((raw + 1) >> 1) if raw & 1 else raw >> 1
        if tag == _FLOAT:
            return _FLOAT_BITS.unpack(self.take(8))[0]
        if tag == _STR:
            value = self.take(self.varint()).decode("utf-8", "surrogatepass")
            self.strings.append(value)
            return value
        if tag == _STR_REF:
            return self.strings[self.varint()]
        if tag == _BYTES:
            return bytes(self.take(self.varint()))
        if tag == _TUPLE:
            return tuple(self.values())
        if tag == _LIST:
            return self.values()
        if tag == _DICT:
            return {self.value(): self.value() for _ in range(self.varint())}
        if tag == _SET:
            return set(self.values())
        if tag == _FROZENSET:
            return frozenset(self.values())
        if tag == _GLOBAL:
            self.at -= 1
            return self.global_ref()
        if tag == _ENUM:
            cls = self.global_ref()
            return cls[self.text()]
        if tag == _NAMED_TUPLE:
            cls = self.global_ref()
            items = self.value()
            return cls(*items)
        if tag == _RECORD:
            cls = self.global_ref()
            record = cls.__new__(cls)
            for _ in range(self.varint()):
                name = self.text()
                object.__setattr__(record, name, self.value())
            return record
        if tag == _REDUCED:
            function = self.global_ref()
            arguments = self.value()
            return function(*arguments)
        raise ValueError(f"tile row stream has unknown tag {tag}")


def encode_rows(
    rows, reducers: Mapping[type, Callable[[Any], Any]] | None = None,
) -> bytes:
    """The tagged octet stream of ``rows`` (a sequence of values)."""

    writer = _Writer(reducers)
    writer.items(_TUPLE, rows)
    return bytes(writer.out)


def decode_rows(data: bytes) -> tuple:
    """Inverse of :func:`encode_rows`."""

    reader = _Reader(bytes(data))
    rows = reader.value()
    if reader.at != len(reader.data) or type(rows) is not tuple:
        raise ValueError("tile row stream has trailing or malformed data")
    return rows


__all__ = ["compress_octets", "decode_rows", "decompress_octets", "encode_rows"]
