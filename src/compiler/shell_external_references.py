"""Host implementation of the shell external-reference record ABI.

The compiler-facing ABI is deliberately smaller than Python's object model:
requests and completions are the eight-i32 records declared by :mod:`shell_io`,
while argument/result spans contain tagged scalar values or shell-owned opaque
handles.  No ``PyObject *`` and no guest pointer crosses this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import struct
from typing import Any, Mapping, Sequence


_RECORD = struct.Struct("<8i")
_U32 = struct.Struct("<I")
_I64 = struct.Struct("<q")
_F64 = struct.Struct("<d")
_ARGUMENT_HEADER = struct.Struct("<4sII")
_ARGUMENT_MAGIC = b"TXA1"

EXTERNAL_RESOLVE = 1
EXTERNAL_CALL = 2
EXTERNAL_RELEASE = 3

EXTERNAL_OK = 0
EXTERNAL_UNKNOWN_OPERATION = 1
EXTERNAL_UNKNOWN_REFERENCE = 2
EXTERNAL_INVOCATION_FAILED = 3
EXTERNAL_RESULT_CAPACITY = 4
EXTERNAL_INVALID_PAYLOAD = 5
EXTERNAL_CALLBACK_REJECTED = 6

_NONE = 0
_BOOL = 1
_INT64 = 2
_FLOAT64 = 3
_BYTES = 4
_TEXT = 5
_HANDLE = 6
_TUPLE = 7


@dataclass(frozen=True)
class ExternalReferenceRequestRecord:
    operation: int
    request_id: int
    reference_id: int = 0
    arguments_offset: int = 0
    arguments_length: int = 0
    result_offset: int = 0
    result_capacity: int = 0
    flags: int = 0

    def pack(self) -> bytes:
        return _RECORD.pack(*self.__dict__.values())

    @classmethod
    def unpack(cls, payload: bytes) -> "ExternalReferenceRequestRecord":
        if len(payload) != _RECORD.size:
            raise ValueError("external-reference request record must be 32 bytes")
        return cls(*_RECORD.unpack(payload))


@dataclass(frozen=True)
class ExternalReferenceCompletionRecord:
    operation: int
    request_id: int
    reference_id: int
    status: int
    result_length: int = 0
    effects_offset: int = 0
    effects_length: int = 0
    generation: int = 0

    def pack(self) -> bytes:
        return _RECORD.pack(*self.__dict__.values())

    @classmethod
    def unpack(cls, payload: bytes) -> "ExternalReferenceCompletionRecord":
        if len(payload) != _RECORD.size:
            raise ValueError("external-reference completion record must be 32 bytes")
        return cls(*_RECORD.unpack(payload))


class ExternalReferenceValueCodec:
    """Tagged values for external argument/result spans.

    Scalars and immutable byte/text values travel by value. Everything else is
    a broker-owned handle, which is the only representation used for Python
    objects such as file objects and unpickled graphs.
    """

    def __init__(self) -> None:
        self._objects: dict[int, Any] = {}
        self._next_handle = 1

    def retain(self, value: Any) -> int:
        handle = self._next_handle
        self._next_handle += 1
        self._objects[handle] = value
        return handle

    def object(self, handle: int) -> Any:
        try:
            return self._objects[int(handle)]
        except KeyError as error:
            raise ValueError(f"unknown external object handle {handle}") from error

    def release(self, handle: int) -> None:
        if self._objects.pop(int(handle), None) is None:
            raise ValueError(f"unknown external object handle {handle}")

    def encode(self, value: Any) -> bytes:
        if value is None:
            return bytes((_NONE,))
        if isinstance(value, bool):
            return bytes((_BOOL, int(value)))
        if isinstance(value, int) and -(1 << 63) <= value < (1 << 63):
            return bytes((_INT64,)) + _I64.pack(value)
        if isinstance(value, float):
            return bytes((_FLOAT64,)) + _F64.pack(value)
        if isinstance(value, bytes):
            return bytes((_BYTES,)) + _U32.pack(len(value)) + value
        if isinstance(value, str):
            payload = value.encode("utf-8")
            return bytes((_TEXT,)) + _U32.pack(len(payload)) + payload
        if isinstance(value, tuple):
            encoded = tuple(self.encode(item) for item in value)
            return bytes((_TUPLE,)) + _U32.pack(len(encoded)) + b"".join(
                _U32.pack(len(item)) + item for item in encoded
            )
        return bytes((_HANDLE,)) + _I64.pack(self.retain(value))

    def decode(self, payload: bytes) -> Any:
        if not payload:
            raise ValueError("empty external-reference value")
        tag = payload[0]
        body = payload[1:]
        if tag == _NONE and not body:
            return None
        if tag == _BOOL and len(body) == 1:
            return bool(body[0])
        if tag == _INT64 and len(body) == _I64.size:
            return _I64.unpack(body)[0]
        if tag == _FLOAT64 and len(body) == _F64.size:
            return _F64.unpack(body)[0]
        if tag in {_BYTES, _TEXT} and len(body) >= _U32.size:
            length = _U32.unpack_from(body)[0]
            value = body[_U32.size:]
            if len(value) != length:
                raise ValueError("external-reference scalar span length disagrees")
            return value if tag == _BYTES else value.decode("utf-8")
        if tag == _HANDLE and len(body) == _I64.size:
            return self.object(_I64.unpack(body)[0])
        if tag == _TUPLE and len(body) >= _U32.size:
            count = _U32.unpack_from(body)[0]
            cursor = _U32.size
            values = []
            for _ in range(count):
                if cursor + _U32.size > len(body):
                    raise ValueError("truncated external-reference tuple")
                length = _U32.unpack_from(body, cursor)[0]
                cursor += _U32.size
                end = cursor + length
                if end > len(body):
                    raise ValueError("truncated external-reference tuple value")
                values.append(self.decode(body[cursor:end]))
                cursor = end
            if cursor != len(body):
                raise ValueError("external-reference tuple has trailing bytes")
            return tuple(values)
        raise ValueError(f"unknown or malformed external-reference value tag {tag}")

    def encode_arguments(
        self, args: Sequence[Any], kwargs: Mapping[str, Any] | None = None,
    ) -> bytes:
        keywords = dict(kwargs or {})
        encoded_args = tuple(self.encode(value) for value in args)
        encoded_keywords = tuple(
            (str(name).encode("utf-8"), self.encode(value))
            for name, value in keywords.items()
        )
        return b"".join((
            _ARGUMENT_HEADER.pack(
                _ARGUMENT_MAGIC, len(encoded_args), len(encoded_keywords)
            ),
            *( _U32.pack(len(value)) + value for value in encoded_args ),
            *(
                _U32.pack(len(name)) + name + _U32.pack(len(value)) + value
                for name, value in encoded_keywords
            ),
        ))

    def decode_arguments(self, payload: bytes) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if len(payload) < _ARGUMENT_HEADER.size:
            raise ValueError("truncated external-reference argument frame")
        magic, argument_count, keyword_count = _ARGUMENT_HEADER.unpack_from(payload)
        if magic != _ARGUMENT_MAGIC:
            raise ValueError("unknown external-reference argument frame")
        cursor = _ARGUMENT_HEADER.size

        def take() -> bytes:
            nonlocal cursor
            if cursor + _U32.size > len(payload):
                raise ValueError("truncated external-reference argument length")
            length = _U32.unpack_from(payload, cursor)[0]
            cursor += _U32.size
            end = cursor + length
            if end > len(payload):
                raise ValueError("truncated external-reference argument")
            value = payload[cursor:end]
            cursor = end
            return value

        args = tuple(self.decode(take()) for _ in range(argument_count))
        kwargs = {}
        for _ in range(keyword_count):
            name = take().decode("utf-8")
            kwargs[name] = self.decode(take())
        if cursor != len(payload):
            raise ValueError("external-reference argument frame has trailing bytes")
        return args, kwargs


class ExistingModuleExternalReferenceHost:
    """Resolve and invoke exact identities in already importable modules."""

    def __init__(self) -> None:
        self.values = ExternalReferenceValueCodec()
        self._references: dict[int, Any] = {}
        self._next_reference = 1
        self._generation = 0

    @staticmethod
    def _span(memory: bytearray, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > len(memory):
            raise ValueError("external-reference span is outside artifact memory")
        return bytes(memory[offset:offset + length])

    def _resolve(self, identity: str) -> tuple[int, Any]:
        module_name, separator, qualname = identity.partition(".")
        if not separator or not module_name or not qualname:
            raise ValueError("external identity must be module-qualified")
        value: Any = importlib.import_module(module_name)
        for component in qualname.split("."):
            value = getattr(value, component)
        reference_id = self._next_reference
        self._next_reference += 1
        self._references[reference_id] = value
        return reference_id, value

    def service(
        self, request: ExternalReferenceRequestRecord, memory: bytearray,
    ) -> ExternalReferenceCompletionRecord:
        self._generation += 1
        reference_id = int(request.reference_id)
        result = b""
        status = EXTERNAL_OK
        try:
            payload = self._span(
                memory, request.arguments_offset, request.arguments_length
            )
            if request.operation == EXTERNAL_RESOLVE:
                identity = self.values.decode(payload)
                if not isinstance(identity, str):
                    raise ValueError("resolve payload must contain an identity string")
                reference_id, _value = self._resolve(identity)
                result = self.values.encode(reference_id)
            elif request.operation == EXTERNAL_CALL:
                try:
                    function = self._references[reference_id]
                except KeyError as error:
                    raise LookupError(reference_id) from error
                args, kwargs = self.values.decode_arguments(payload)
                if any(callable(value) for value in (*args, *kwargs.values())):
                    status = EXTERNAL_CALLBACK_REJECTED
                else:
                    result = self.values.encode(function(*args, **kwargs))
            elif request.operation == EXTERNAL_RELEASE:
                if reference_id not in self._references:
                    raise LookupError(reference_id)
                del self._references[reference_id]
            else:
                status = EXTERNAL_UNKNOWN_OPERATION
        except LookupError:
            status = EXTERNAL_UNKNOWN_REFERENCE
        except (ValueError, UnicodeError, struct.error):
            status = EXTERNAL_INVALID_PAYLOAD
        except Exception:
            # The physical completion reports failure without smuggling a
            # Python exception object into artifact memory. A future effects
            # journal can carry a durable, policy-scrubbed diagnostic.
            status = EXTERNAL_INVOCATION_FAILED
        if status == EXTERNAL_OK and len(result) > request.result_capacity:
            status = EXTERNAL_RESULT_CAPACITY
            result = b""
        if status == EXTERNAL_OK and result:
            start = int(request.result_offset)
            end = start + len(result)
            if start < 0 or end > len(memory):
                status = EXTERNAL_INVALID_PAYLOAD
                result = b""
            else:
                memory[start:end] = result
        return ExternalReferenceCompletionRecord(
            operation=request.operation,
            request_id=request.request_id,
            reference_id=reference_id,
            status=status,
            result_length=len(result),
            generation=self._generation,
        )


__all__ = [
    "EXTERNAL_CALL", "EXTERNAL_OK", "EXTERNAL_RELEASE", "EXTERNAL_RESOLVE",
    "ExistingModuleExternalReferenceHost", "ExternalReferenceCompletionRecord",
    "ExternalReferenceRequestRecord", "ExternalReferenceValueCodec",
]
