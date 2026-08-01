"""Assemble a WebAssembly binary module directly, with no external tool.

WAT is the human-readable form of WebAssembly; a browser only ever executes
the binary. So emitting WAT alone leaves a gap that has to be closed by
``wat2wasm``, and if that is not installed the emitted program cannot
actually run -- which made the HTML shell look broken while being perfectly
correct about its situation.

The binary format is closing that gap here instead. This is a real
assembler, but a small one, because the instruction set a fused elementwise
program needs is small: load, store, arithmetic on one float type, integer
index arithmetic, and one counted loop. Nothing in the format's harder
regions -- tables, globals, imports, multiple memories, indirect calls,
custom sections -- is used, so this stays under the size where writing it is
worse than depending on a toolchain.

Reference: the WebAssembly core specification's binary format. Sections are
emitted in the order the spec requires (type, function, memory, export,
code); integers are LEB128; floats are little-endian IEEE-754.

Emitting the binary alongside the WAT means the two must agree, so both come
from the same lowering in ``fused_program_wasm_backend`` -- this module is
handed an already-planned instruction list, not a second lowering that could
disagree with the first.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --- primitive encodings ---------------------------------------------------


def uleb(value: int) -> bytes:
    """Unsigned LEB128, the format's integer encoding."""

    if value < 0:
        raise ValueError(f"uleb is unsigned; got {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def sleb(value: int) -> bytes:
    """Signed LEB128, used for value-type constants and block types."""

    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        done = (value == 0 and not (byte & 0x40)) or (
            value == -1 and (byte & 0x40)
        )
        if done:
            out.append(byte)
            return bytes(out)
        out.append(byte | 0x80)


def _vector(items: Iterable[bytes]) -> bytes:
    payload = list(items)
    return uleb(len(payload)) + b"".join(payload)


def _section(section_id: int, payload: bytes) -> bytes:
    return bytes([section_id]) + uleb(len(payload)) + payload


def _name(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return uleb(len(encoded)) + encoded


# --- value types and opcodes ----------------------------------------------

I32 = 0x7F
F32 = 0x7D
F64 = 0x7C

_VALUE_TYPE = {"i32": I32, "f32": F32, "f64": F64}

# Opcodes, per value type. Only the ones a fused elementwise program reaches.
_OPCODES: dict[str, dict[str, int]] = {
    "f64": {
        "load": 0x2B, "store": 0x39, "const": 0x44,
        "abs": 0x99, "neg": 0x9A, "ceil": 0x9B, "floor": 0x9C,
        "trunc": 0x9D, "nearest": 0x9E, "sqrt": 0x9F,
        "add": 0xA0, "sub": 0xA1, "mul": 0xA2, "div": 0xA3,
        "min": 0xA4, "max": 0xA5,
        "eq": 0x61, "ne": 0x62, "lt": 0x63, "gt": 0x64, "le": 0x65, "ge": 0x66,
        "convert_i32_u": 0xB8,
    },
    "f32": {
        "load": 0x2A, "store": 0x38, "const": 0x43,
        "abs": 0x8B, "neg": 0x8C, "ceil": 0x8D, "floor": 0x8E,
        "trunc": 0x8F, "nearest": 0x90, "sqrt": 0x91,
        "add": 0x92, "sub": 0x93, "mul": 0x94, "div": 0x95,
        "min": 0x96, "max": 0x97,
        "eq": 0x5B, "ne": 0x5C, "lt": 0x5D, "gt": 0x5E, "le": 0x5F, "ge": 0x60,
        "convert_i32_u": 0xB3,
    },
}

# Structural and integer opcodes, which do not vary with the value type.
OP_BLOCK = 0x02
OP_LOOP = 0x03
OP_END = 0x0B
OP_BR = 0x0C
OP_BR_IF = 0x0D
OP_LOCAL_GET = 0x20
OP_LOCAL_SET = 0x21
OP_I32_CONST = 0x41
OP_I32_ADD = 0x6A
OP_I32_MUL = 0x6C
OP_I32_GE_S = 0x4E
EMPTY_BLOCK = 0x40


@dataclass
class CodeBuilder:
    """Instruction stream for one function body.

    Locals are declared by index: the function's parameters occupy the first
    indices, and anything appended here follows them.
    """

    value_type: str
    parameter_count: int
    locals: list[int] = field(default_factory=list)
    code: bytearray = field(default_factory=bytearray)

    @property
    def opcodes(self) -> dict[str, int]:
        return _OPCODES[self.value_type]

    def declare_local(self, type_name: str) -> int:
        """Add a local and return its index."""

        self.locals.append(_VALUE_TYPE[type_name])
        return self.parameter_count + len(self.locals) - 1

    # -- emitters ----------------------------------------------------------

    def local_get(self, index: int) -> "CodeBuilder":
        self.code += bytes([OP_LOCAL_GET]) + uleb(index)
        return self

    def local_set(self, index: int) -> "CodeBuilder":
        self.code += bytes([OP_LOCAL_SET]) + uleb(index)
        return self

    def i32_const(self, value: int) -> "CodeBuilder":
        self.code += bytes([OP_I32_CONST]) + sleb(value)
        return self

    def value_const(self, value: float) -> "CodeBuilder":
        opcode = self.opcodes["const"]
        packed = (
            struct.pack("<f", value)
            if self.value_type == "f32"
            else struct.pack("<d", value)
        )
        self.code += bytes([opcode]) + packed
        return self

    def op(self, name: str) -> "CodeBuilder":
        opcode = self.opcodes.get(name)
        if opcode is None:
            raise KeyError(f"no {self.value_type} opcode for {name!r}")
        self.code += bytes([opcode])
        return self

    def raw(self, *opcodes: int) -> "CodeBuilder":
        self.code += bytes(opcodes)
        return self

    def load(self, *, align: int | None = None, offset: int = 0) -> "CodeBuilder":
        # Alignment is expressed as a power of two; a natural alignment of 8
        # bytes is 3, of 4 bytes is 2.
        natural = 3 if self.value_type == "f64" else 2
        self.code += (
            bytes([self.opcodes["load"]])
            + uleb(natural if align is None else align)
            + uleb(offset)
        )
        return self

    def store(self, *, align: int | None = None, offset: int = 0) -> "CodeBuilder":
        natural = 3 if self.value_type == "f64" else 2
        self.code += (
            bytes([self.opcodes["store"]])
            + uleb(natural if align is None else align)
            + uleb(offset)
        )
        return self

    def block(self, block_type: int = EMPTY_BLOCK) -> "CodeBuilder":
        self.code += bytes([OP_BLOCK, block_type])
        return self

    def loop(self, block_type: int = EMPTY_BLOCK) -> "CodeBuilder":
        self.code += bytes([OP_LOOP, block_type])
        return self

    def br(self, depth: int) -> "CodeBuilder":
        self.code += bytes([OP_BR]) + uleb(depth)
        return self

    def br_if(self, depth: int) -> "CodeBuilder":
        self.code += bytes([OP_BR_IF]) + uleb(depth)
        return self

    def end(self) -> "CodeBuilder":
        self.code += bytes([OP_END])
        return self

    def to_body(self) -> bytes:
        """The function body: local declarations, code, terminating end."""

        # Locals are run-length encoded as (count, type) pairs.
        runs: list[tuple[int, int]] = []
        for local_type in self.locals:
            if runs and runs[-1][1] == local_type:
                runs[-1] = (runs[-1][0] + 1, local_type)
            else:
                runs.append((1, local_type))
        declarations = _vector(
            uleb(count) + bytes([local_type]) for count, local_type in runs
        )
        body = declarations + bytes(self.code) + bytes([OP_END])
        return uleb(len(body)) + body


def build_module(
    *,
    function_name: str,
    parameter_types: Sequence[str],
    body: CodeBuilder,
    memory_pages: int = 1,
    memory_name: str = "memory",
) -> bytes:
    """Assemble one exported function plus one exported memory."""

    type_section = _section(
        1,
        _vector([
            bytes([0x60])
            + _vector([bytes([_VALUE_TYPE[t]]) for t in parameter_types])
            + _vector([])  # no results; outputs are written through memory
        ]),
    )
    function_section = _section(3, _vector([uleb(0)]))
    memory_section = _section(5, _vector([bytes([0x00]) + uleb(memory_pages)]))
    export_section = _section(
        7,
        _vector([
            _name(memory_name) + bytes([0x02]) + uleb(0),
            _name(function_name) + bytes([0x00]) + uleb(0),
        ]),
    )
    code_section = _section(10, _vector([body.to_body()]))
    return (
        b"\x00asm"
        + struct.pack("<I", 1)
        + type_section
        + function_section
        + memory_section
        + export_section
        + code_section
    )


__all__ = [
    "CodeBuilder",
    "EMPTY_BLOCK",
    "F32",
    "F64",
    "I32",
    "build_module",
    "sleb",
    "uleb",
]
