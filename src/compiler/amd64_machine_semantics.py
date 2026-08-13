"""Deterministic AMD64 state semantics for the reversible machine program.

The decoder owns instruction boundaries and operand shapes.  This module owns
the corresponding architectural state transformations: little-endian memory,
sub-register writes, integer flags, stack effects, and condition evaluation.
Every operation returns a new immutable machine state so the execution journal
can restore an exact predecessor without running an inverse instruction.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
import re
import struct

from .machine_execution import (
    MACHINE_LOADER_CALLBACK_RETURN,
    MACHINE_TERMINATION_RETURN,
    MachineExecutionState,
    MachineExternalCallCompletion,
    MachineExternalReference,
)
from .machine_reference_vocabulary import (
    EffectiveAddressOperand,
    HighByteRegisterOperand,
    ImmediateOperand,
    MachineSemanticToken,
    RegisterOperand,
    RelativeAddressOperand,
    VectorRegisterOperand,
    X86HighByteRegister,
)
from .virtual_registry import VirtualRegistryState
from .virtual_memory import VirtualMemoryState


MASK64 = (1 << 64) - 1
EXTERNAL_TARGET_BASE = 0xFFFF800000000000
CF, PF, AF, ZF, SF, OF = 0, 2, 4, 6, 7, 11
ARITHMETIC_FLAGS = sum(1 << bit for bit in (CF, PF, AF, ZF, SF, OF))
LOGICAL_FLAGS = sum(1 << bit for bit in (CF, PF, ZF, SF, OF))


@dataclass(frozen=True, slots=True)
class PagedByteMemory(Mapping[int, int]):
    """Immutable sparse byte memory with copy-on-write pages."""

    pages: Mapping[int, bytes]
    page_size: int = 4096
    # Runtime-only provenance for the immediately preceding copy-on-write
    # update.  It lets the executor identify touched pages in O(writes)
    # instead of comparing every mapped page after every instruction.  The
    # identity check deliberately fails after serialization or reconstruction,
    # where callers fall back to the exact structural comparison.
    _parent_pages_identity: int = field(default=0, repr=False, compare=False)
    _changed_pages: tuple[int, ...] = field(
        default=(), repr=False, compare=False,
    )

    @classmethod
    def empty(cls, *, page_size: int = 4096) -> "PagedByteMemory":
        return cls(MappingProxyType({}), page_size)

    def __getitem__(self, address: int) -> int:
        page = int(address) // self.page_size
        offset = int(address) % self.page_size
        try:
            return self.pages[page][offset]
        except KeyError as error:
            raise KeyError(f"unmapped guest address {int(address):#x}") from error

    def __iter__(self) -> Iterator[int]:
        for page in self.pages:
            base = page * self.page_size
            yield from range(base, base + self.page_size)

    def __len__(self) -> int:
        return len(self.pages) * self.page_size

    def map_bytes(self, address: int, data: bytes | bytearray | memoryview) -> "PagedByteMemory":
        raw = bytes(data)
        if not raw:
            return self
        pages = dict(self.pages)
        changed_pages: list[int] = []
        cursor = 0
        while cursor < len(raw):
            absolute = int(address) + cursor
            page_index, page_offset = divmod(absolute, self.page_size)
            count = min(self.page_size - page_offset, len(raw) - cursor)
            page_was_mapped = page_index in pages
            previous = pages.get(page_index, bytes(self.page_size))
            page = bytearray(previous)
            page[page_offset:page_offset + count] = raw[cursor:cursor + count]
            updated = bytes(page)
            if not page_was_mapped or updated != previous:
                pages[page_index] = updated
                changed_pages.append(page_index)
            cursor += count
        if not changed_pages:
            return self
        return PagedByteMemory(
            MappingProxyType(pages), self.page_size,
            id(self.pages), tuple(changed_pages),
        )

    def map_zeroes(self, address: int, size: int) -> "PagedByteMemory":
        if size < 0:
            raise ValueError("mapped region size cannot be negative")
        return self.map_bytes(address, bytes(size))

    def unmap(self, address: int, size: int) -> "PagedByteMemory":
        if address % self.page_size or size < 0 or size % self.page_size:
            raise ValueError("unmapped region must be page aligned")
        pages = dict(self.pages)
        changed_pages = tuple(
            page
            for page in range(
                address // self.page_size, (address + size) // self.page_size,
            )
            if page in pages
        )
        if not changed_pages:
            return self
        for page in changed_pages:
            pages.pop(page)
        return PagedByteMemory(
            MappingProxyType(pages), self.page_size,
            id(self.pages), changed_pages,
        )

    def read(self, address: int, size: int) -> bytes:
        """Return one exact mapped byte range, including page crossings."""

        if size < 0:
            raise ValueError("memory read size cannot be negative")
        return bytes(self[int(address) + index] for index in range(int(size)))

    def read_unsigned(self, address: int, width: int) -> int:
        if width not in (8, 16, 32, 64, 128):
            raise ValueError(f"unsupported memory width {width}")
        return int.from_bytes(
            bytes(self[int(address) + index] for index in range(width // 8)),
            "little",
        )

    def write_unsigned(self, address: int, width: int, value: int) -> "PagedByteMemory":
        mask = (1 << width) - 1
        return self.map_bytes(
            int(address), int(value & mask).to_bytes(width // 8, "little"),
        )


def _as_memory(memory: Mapping[int, int]) -> PagedByteMemory:
    if isinstance(memory, PagedByteMemory):
        return memory
    result = PagedByteMemory.empty()
    for address, value in memory.items():
        result = result.map_bytes(int(address), bytes((int(value) & 0xFF,)))
    return result


def _mask(width: int) -> int:
    return (1 << width) - 1


def _sign_extend(value: int, source_width: int, target_width: int) -> int:
    value &= _mask(source_width)
    if value & (1 << (source_width - 1)):
        value -= 1 << source_width
    return value & _mask(target_width)


def effective_address(state: MachineExecutionState, instruction, operand: EffectiveAddressOperand) -> int:
    base = instruction.address + len(instruction.encoded) if operand.rip_relative else 0
    # Architectural segment override bytes are authoritative even where a
    # historical decoder token name calls 0x65 "FS"; AMD64 defines 0x64 as FS
    # and 0x65 as GS.
    if 0x64 in instruction.legacy_prefixes:
        base += state.fs_base
    if 0x65 in instruction.legacy_prefixes:
        base += state.gs_base
    if operand.base is not None:
        base += state.registers[int(operand.base)]
    if operand.index is not None:
        base += state.registers[int(operand.index)] * operand.scale
    return (base + operand.displacement) & MASK64


def _data_width(instruction, operand_index: int) -> int:
    operand = instruction.operands[operand_index]
    if isinstance(operand, (RegisterOperand, HighByteRegisterOperand)):
        return operand.width
    name = instruction.token.name
    if isinstance(operand, VectorRegisterOperand):
        match = re.search(
            r"(?:^|_)(?:XMMM|RM|M)(128|64|32)(?:_|$)", name,
        )
        return int(match.group(1)) if match else operand.width
    if isinstance(operand, EffectiveAddressOperand):
        memory_widths = re.findall(
            r"(?:^|_)(?:XMMM|RM|M)(128|64|32|16|8)(?:_|$)", name,
        )
        if memory_widths:
            return int(memory_widths[0])
    match = re.search(
        r"(?:^|_)(?:XMMM|RM|R|M)(128|64|32|16|8)(?:_|$)", name,
    )
    if match:
        return int(match.group(1))
    if isinstance(operand, ImmediateOperand):
        return operand.width
    raise ValueError(f"cannot infer data width for {name} operand {operand_index}")


def read_operand(state: MachineExecutionState, instruction, operand_index: int, *, width: int | None = None) -> int:
    operand = instruction.operands[operand_index]
    target_width = width or _data_width(instruction, operand_index)
    if isinstance(operand, RegisterOperand):
        return state.registers[int(operand.register)] & _mask(operand.width)
    if isinstance(operand, HighByteRegisterOperand):
        return (state.registers[int(operand.register)] >> 8) & 0xFF
    if isinstance(operand, VectorRegisterOperand):
        return state.vector_registers[int(operand.register)] & _mask(target_width)
    if isinstance(operand, ImmediateOperand):
        if operand.signed:
            return _sign_extend(operand.value, operand.width, target_width)
        return operand.value & _mask(target_width)
    if isinstance(operand, EffectiveAddressOperand):
        return _as_memory(state.memory).read_unsigned(
            effective_address(state, instruction, operand), target_width,
        )
    if isinstance(operand, RelativeAddressOperand):
        return operand.target_address & MASK64
    raise ValueError(f"unsupported AMD64 operand {operand!r}")


def write_operand(state: MachineExecutionState, instruction, operand_index: int, value: int, *, width: int | None = None) -> MachineExecutionState:
    operand = instruction.operands[operand_index]
    target_width = width or _data_width(instruction, operand_index)
    value &= _mask(target_width)
    if isinstance(operand, RegisterOperand):
        registers = list(state.registers)
        old = registers[int(operand.register)]
        if target_width == 64:
            updated = value
        elif target_width == 32:
            updated = value  # AMD64 32-bit GPR writes clear the upper half.
        else:
            updated = (old & ~_mask(target_width)) | value
        registers[int(operand.register)] = updated & MASK64
        return replace(state, registers=tuple(registers))
    if isinstance(operand, HighByteRegisterOperand):
        registers = list(state.registers)
        index = int(operand.register)
        registers[index] = (registers[index] & ~(0xFF << 8)) | ((value & 0xFF) << 8)
        return replace(state, registers=tuple(registers))
    if isinstance(operand, VectorRegisterOperand):
        vectors = list(state.vector_registers)
        old = vectors[int(operand.register)]
        vectors[int(operand.register)] = (
            value if target_width == 256
            else (old & ~_mask(target_width)) | value
        )
        return replace(state, vector_registers=tuple(vectors))
    if isinstance(operand, EffectiveAddressOperand):
        memory = _as_memory(state.memory).write_unsigned(
            effective_address(state, instruction, operand), target_width, value,
        )
        return replace(state, memory=memory)
    raise ValueError(f"operand is not writable: {operand!r}")


def _parity(value: int) -> bool:
    return (value & 0xFF).bit_count() % 2 == 0


def _set_flag(flags: int, bit: int, enabled: bool) -> int:
    return (flags | (1 << bit)) if enabled else (flags & ~(1 << bit))


def _arithmetic_flags(flags: int, left: int, right: int, result: int, width: int, *, subtract: bool) -> int:
    mask = _mask(width)
    sign = 1 << (width - 1)
    truncated = result & mask
    flags &= ~ARITHMETIC_FLAGS
    if subtract:
        carry = (left & mask) < (right & mask)
        overflow = bool(((left ^ right) & (left ^ truncated) & sign))
    else:
        carry = result > mask
        overflow = bool((~(left ^ right) & (left ^ truncated) & sign))
    for bit, enabled in (
        (CF, carry), (PF, _parity(truncated)),
        (AF, bool((left ^ right ^ truncated) & 0x10)),
        (ZF, truncated == 0), (SF, bool(truncated & sign)), (OF, overflow),
    ):
        flags = _set_flag(flags, bit, enabled)
    return flags


def _logical_flags(flags: int, result: int, width: int) -> int:
    truncated = result & _mask(width)
    flags &= ~LOGICAL_FLAGS
    for bit, enabled in ((PF, _parity(truncated)), (ZF, truncated == 0), (SF, bool(truncated & (1 << (width - 1))))):
        flags = _set_flag(flags, bit, enabled)
    return flags


def _binary_handler(kind: str, *, write: bool = True):
    def handler(state: MachineExecutionState, instruction) -> MachineExecutionState:
        width = _data_width(instruction, 0)
        left = read_operand(state, instruction, 0, width=width)
        right = read_operand(state, instruction, 1, width=width)
        if kind == "add":
            raw = left + right
            flags = _arithmetic_flags(state.flags, left, right, raw, width, subtract=False)
        elif kind in {"sub", "cmp"}:
            raw = left - right
            flags = _arithmetic_flags(state.flags, left, right, raw, width, subtract=True)
        else:
            raw = {"and": left & right, "or": left | right, "xor": left ^ right, "test": left & right}[kind]
            flags = _logical_flags(state.flags, raw, width)
        result = replace(state, flags=flags)
        return write_operand(result, instruction, 0, raw, width=width) if write else result
    return handler


def _multiply_signed(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    """Execute two/three-operand IMUL and define its architectural flags.

    CF and OF are clear exactly when the full signed product is the sign
    extension of the retained low-width result.  AMD64 leaves the remaining
    arithmetic flags undefined; the deterministic VM preserves their incoming
    values instead of inventing meanings for them.
    """

    accumulator_form = len(instruction.operands) == 1
    if accumulator_form:
        width = _data_width(instruction, 0)
        left = state.registers[0] & _mask(width)
        right = read_operand(state, instruction, 0, width=width)
    elif len(instruction.operands) == 2:
        width = _data_width(instruction, 0)
        left = read_operand(state, instruction, 0, width=width)
        right = read_operand(state, instruction, 1, width=width)
    elif len(instruction.operands) == 3:
        width = _data_width(instruction, 0)
        left = read_operand(state, instruction, 1, width=width)
        right_width = _data_width(instruction, 2)
        right = read_operand(state, instruction, 2, width=right_width)
        right = _sign_extend(right, right_width, width)
    else:
        raise ValueError("IMUL requires one, two, or three operands")

    sign = 1 << (width - 1)
    mask = _mask(width)
    signed_left = (left & mask) - (1 << width) if left & sign else left & mask
    signed_right = (right & mask) - (1 << width) if right & sign else right & mask
    full_product = signed_left * signed_right
    retained = full_product & mask
    signed_retained = retained - (1 << width) if retained & sign else retained
    overflow = full_product != signed_retained
    flags = _set_flag(state.flags, CF, overflow)
    flags = _set_flag(flags, OF, overflow)
    result = replace(state, flags=flags)
    if accumulator_form:
        registers = list(result.registers)
        registers[0] = retained
        registers[2] = (full_product >> width) & mask
        return replace(result, registers=tuple(registers))
    return write_operand(result, instruction, 0, retained, width=width)


def _subtract_with_borrow(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    left = read_operand(state, instruction, 0, width=width)
    right = read_operand(state, instruction, 1, width=width)
    borrowed = int(bool(state.flags & (1 << CF)))
    effective_right = right + borrowed
    raw = left - effective_right
    flags = _arithmetic_flags(
        state.flags, left, effective_right, raw, width, subtract=True,
    )
    flags = _set_flag(flags, CF, left < effective_right)
    return write_operand(
        replace(state, flags=flags), instruction, 0, raw, width=width,
    )


def _move(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    return write_operand(state, instruction, 0, read_operand(state, instruction, 1, width=width), width=width)


def _lea(state: MachineExecutionState, instruction) -> MachineExecutionState:
    source = instruction.operands[1]
    if not isinstance(source, EffectiveAddressOperand):
        raise ValueError("LEA source must be an effective address")
    return write_operand(state, instruction, 0, effective_address(state, instruction, source))


def _noop(state: MachineExecutionState, instruction) -> MachineExecutionState:
    return state


def _push(state: MachineExecutionState, instruction) -> MachineExecutionState:
    value = read_operand(state, instruction, 0, width=64)
    rsp = (state.registers[4] - 8) & MASK64
    registers = list(state.registers)
    registers[4] = rsp
    memory = _as_memory(state.memory).write_unsigned(rsp, 64, value)
    return replace(state, registers=tuple(registers), memory=memory)


def _pop(state: MachineExecutionState, instruction) -> MachineExecutionState:
    rsp = state.registers[4]
    value = _as_memory(state.memory).read_unsigned(rsp, 64)
    result = write_operand(state, instruction, 0, value, width=64)
    registers = list(result.registers)
    registers[4] = (rsp + 8) & MASK64
    return replace(result, registers=tuple(registers))


def _incdec(delta: int):
    def handler(state: MachineExecutionState, instruction) -> MachineExecutionState:
        width = _data_width(instruction, 0)
        left = read_operand(state, instruction, 0, width=width)
        raw = left + delta
        flags = _arithmetic_flags(state.flags, left, 1, raw, width, subtract=delta < 0)
        flags = _set_flag(flags, CF, bool(state.flags & (1 << CF)))
        return write_operand(replace(state, flags=flags), instruction, 0, raw, width=width)
    return handler


def _shift(kind: str):
    def handler(state: MachineExecutionState, instruction) -> MachineExecutionState:
        width = _data_width(instruction, 0)
        value = read_operand(state, instruction, 0, width=width)
        count = read_operand(state, instruction, 1, width=8)
        count &= 0x3F if width == 64 else 0x1F
        if count == 0:
            return state
        mask = _mask(width)
        if kind == "left":
            result = (value << count) & mask
            carry = bool((value >> (width - count)) & 1) if count <= width else False
        elif kind == "right":
            result = value >> count if count < width else 0
            carry = bool((value >> (count - 1)) & 1) if count <= width else False
        else:
            signed_value = value - (1 << width) if value & (1 << (width - 1)) else value
            result = (signed_value >> count) & mask
            carry = bool((value >> (count - 1)) & 1) if count <= width else bool(value >> (width - 1))
        flags = _logical_flags(state.flags, result, width)
        flags = _set_flag(flags, CF, carry)
        if count == 1:
            if kind == "left":
                overflow = bool(result & (1 << (width - 1))) != carry
            elif kind == "right":
                overflow = bool(value & (1 << (width - 1)))
            else:
                overflow = False
            flags = _set_flag(flags, OF, overflow)
        else:
            flags = _set_flag(flags, OF, bool(state.flags & (1 << OF)))
        return write_operand(replace(state, flags=flags), instruction, 0, result, width=width)
    return handler


def _rotate(kind: str):
    def handler(state: MachineExecutionState, instruction) -> MachineExecutionState:
        width = _data_width(instruction, 0)
        value = read_operand(state, instruction, 0, width=width)
        count = read_operand(state, instruction, 1, width=8) % width
        if count == 0:
            return state
        mask = _mask(width)
        if kind == "left":
            result = ((value << count) | (value >> (width - count))) & mask
            carry = bool(result & 1)
            overflow = bool(result & (1 << (width - 1))) != carry
        else:
            result = ((value >> count) | (value << (width - count))) & mask
            carry = bool(result & (1 << (width - 1)))
            overflow = bool((result ^ (result << 1)) & (1 << (width - 1)))
        flags = _set_flag(state.flags, CF, carry)
        if count == 1:
            flags = _set_flag(flags, OF, overflow)
        return write_operand(replace(state, flags=flags), instruction, 0, result, width=width)
    return handler


def _negate(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    value = read_operand(state, instruction, 0, width=width)
    raw = -value
    flags = _arithmetic_flags(state.flags, 0, value, raw, width, subtract=True)
    flags = _set_flag(flags, CF, value != 0)
    return write_operand(replace(state, flags=flags), instruction, 0, raw, width=width)


def _not(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    return write_operand(
        state, instruction, 0,
        ~read_operand(state, instruction, 0, width=width), width=width,
    )


def _compare_exchange(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    destination = read_operand(state, instruction, 0, width=width)
    source = read_operand(state, instruction, 1, width=width)
    accumulator = state.registers[0] & _mask(width)
    flags = _arithmetic_flags(
        state.flags, accumulator, destination,
        accumulator - destination, width, subtract=True,
    )
    flagged = replace(state, flags=flags)
    if accumulator == destination:
        return write_operand(flagged, instruction, 0, source, width=width)
    registers = list(flagged.registers)
    if width == 64:
        registers[0] = destination
    elif width == 32:
        registers[0] = destination  # zero extends
    else:
        registers[0] = (registers[0] & ~_mask(width)) | destination
    return replace(flagged, registers=tuple(registers))


def _exchange(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    left = read_operand(state, instruction, 0, width=width)
    right = read_operand(state, instruction, 1, width=width)
    result = write_operand(state, instruction, 0, right, width=width)
    return write_operand(result, instruction, 1, left, width=width)


def _exchange_add(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    left = read_operand(state, instruction, 0, width=width)
    right = read_operand(state, instruction, 1, width=width)
    raw = left + right
    flags = _arithmetic_flags(state.flags, left, right, raw, width, subtract=False)
    result = write_operand(replace(state, flags=flags), instruction, 0, raw, width=width)
    return write_operand(result, instruction, 1, left, width=width)


def _vector_xor(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    return write_operand(
        state, instruction, 0,
        read_operand(state, instruction, 0, width=width)
        ^ read_operand(state, instruction, 1, width=width),
        width=width,
    )


def _vector_and(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    return write_operand(
        state, instruction, 0,
        read_operand(state, instruction, 0, width=width)
        & read_operand(state, instruction, 1, width=width),
        width=width,
    )


def _vector_shift_right_logical(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    value = read_operand(state, instruction, 0, width=width)
    byte_count = read_operand(state, instruction, 1, width=8)
    result = value >> min(byte_count * 8, width)
    return write_operand(state, instruction, 0, result, width=width)


def _multiply_unsigned(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    source = read_operand(state, instruction, 0, width=width)
    accumulator = state.registers[0] & _mask(width)
    product = accumulator * source
    low = product & _mask(width)
    high = (product >> width) & _mask(width)
    registers = list(state.registers)
    if width == 64:
        registers[0], registers[2] = low, high
    elif width == 32:
        registers[0], registers[2] = low, high
    else:
        registers[0] = (registers[0] & ~_mask(width)) | low
        registers[2] = (registers[2] & ~_mask(width)) | high
    flags = _set_flag(state.flags, CF, high != 0)
    flags = _set_flag(flags, OF, high != 0)
    return replace(state, registers=tuple(registers), flags=flags)


def _sign_extend_accumulator(state: MachineExecutionState, instruction) -> MachineExecutionState:
    """Implement the implicit accumulator forms CDQ, CDQE, and CQO."""

    registers = list(state.registers)
    if instruction.token.name == "CDQ":
        registers[2] = 0xFFFFFFFF if registers[0] & (1 << 31) else 0
    elif instruction.token.name == "CDQE":
        registers[0] = _sign_extend(registers[0] & 0xFFFFFFFF, 32, 64)
    elif instruction.token.name == "CQO":
        registers[2] = MASK64 if registers[0] & (1 << 63) else 0
    else:
        raise ValueError(
            f"unsupported accumulator sign-extension form {instruction.token.name}"
        )
    return replace(state, registers=tuple(registers))


def _divide(signed: bool):
    """Divide the implicit RDX:RAX dividend by the decoded operand."""

    def handler(state: MachineExecutionState, instruction) -> MachineExecutionState:
        width = _data_width(instruction, 0)
        if width not in (32, 64):
            raise ValueError(f"unsupported divide width {width}")
        mask = _mask(width)
        divisor_bits = read_operand(state, instruction, 0, width=width)
        if divisor_bits == 0:
            raise ZeroDivisionError("AMD64 divide error: zero divisor")
        low = state.registers[0] & mask
        high = state.registers[2] & mask
        dividend_bits = (high << width) | low
        if signed:
            dividend_width = width * 2
            dividend = dividend_bits - (1 << dividend_width) \
                if dividend_bits & (1 << (dividend_width - 1)) else dividend_bits
            divisor = divisor_bits - (1 << width) \
                if divisor_bits & (1 << (width - 1)) else divisor_bits
            quotient = abs(dividend) // abs(divisor)
            if (dividend < 0) != (divisor < 0):
                quotient = -quotient
            remainder = dividend - quotient * divisor
            if not -(1 << (width - 1)) <= quotient < (1 << (width - 1)):
                raise OverflowError("AMD64 divide error: signed quotient overflow")
        else:
            quotient, remainder = divmod(dividend_bits, divisor_bits)
            if quotient > mask:
                raise OverflowError("AMD64 divide error: unsigned quotient overflow")
        registers = list(state.registers)
        # Both 32-bit destinations zero-extend in long mode.
        registers[0] = quotient & mask
        registers[2] = remainder & mask
        return replace(state, registers=tuple(registers))

    return handler


def _extend(signed: bool):
    def handler(state: MachineExecutionState, instruction) -> MachineExecutionState:
        source_width = _data_width(instruction, 1)
        target_width = _data_width(instruction, 0)
        value = read_operand(state, instruction, 1, width=source_width)
        if signed:
            value = _sign_extend(value, source_width, target_width)
        return write_operand(state, instruction, 0, value, width=target_width)
    return handler


def condition_holds(state: MachineExecutionState, instruction) -> bool:
    name = instruction.token.name
    condition = name.split("_", 1)[0]
    for prefix in ("CMOV", "SET", "J"):
        if condition.startswith(prefix):
            condition = condition[len(prefix):]
            break
    cf, zf, sf, of = (bool(state.flags & (1 << bit)) for bit in (CF, ZF, SF, OF))
    predicates = {
        "E": zf, "Z": zf, "NE": not zf, "NZ": not zf,
        "B": cf, "C": cf, "NAE": cf, "AE": not cf, "NB": not cf, "NC": not cf,
        "BE": cf or zf, "NA": cf or zf, "A": not cf and not zf, "NBE": not cf and not zf,
        "S": sf, "NS": not sf, "O": of, "NO": not of,
        "L": sf != of, "NGE": sf != of, "GE": sf == of, "NL": sf == of,
        "LE": zf or sf != of, "NG": zf or sf != of,
        "G": not zf and sf == of, "NLE": not zf and sf == of,
    }
    if condition not in predicates:
        raise ValueError(f"unsupported AMD64 condition in {name}")
    return predicates[condition]


def indirect_target(state: MachineExecutionState, instruction) -> int:
    return read_operand(state, instruction, 0, width=64)


def _conditional_move(state: MachineExecutionState, instruction) -> MachineExecutionState:
    return _move(state, instruction) if condition_holds(state, instruction) else state


def _conditional_set(state: MachineExecutionState, instruction) -> MachineExecutionState:
    return write_operand(state, instruction, 0, int(condition_holds(state, instruction)), width=8)


def _call_stack_effect(state: MachineExecutionState, instruction) -> MachineExecutionState:
    rsp = (state.registers[4] - 8) & MASK64
    registers = list(state.registers)
    registers[4] = rsp
    return replace(
        state,
        registers=tuple(registers),
        memory=_as_memory(state.memory).write_unsigned(rsp, 64, state.pc),
    )


def _return_stack_effect(state: MachineExecutionState, instruction) -> MachineExecutionState:
    registers = list(state.registers)
    registers[4] = (registers[4] + 8) & MASK64
    return replace(state, registers=tuple(registers))


def _string_store(state: MachineExecutionState, instruction) -> MachineExecutionState:
    if instruction.token.name == "STOSB":
        destination = state.registers[7]
        memory = _as_memory(state.memory).write_unsigned(
            destination, 8, state.registers[0],
        )
        registers = list(state.registers)
        registers[7] = (
            destination + (-1 if state.flags & (1 << 10) else 1)
        ) & MASK64
        return replace(state, registers=tuple(registers), memory=memory)
    if instruction.token.name == "REP_STOSB":
        count = state.registers[1]
        destination = state.registers[7]
        value = state.registers[0] & 0xFF
        direction = -1 if state.flags & (1 << 10) else 1
        memory = _as_memory(state.memory)
        if count > len(memory):
            raise ValueError("REP STOSB count exceeds mapped guest memory")
        for index in range(count):
            memory = memory.write_unsigned(
                (destination + index * direction) & MASK64, 8, value,
            )
        registers = list(state.registers)
        registers[1] = 0
        registers[7] = (destination + count * direction) & MASK64
        return replace(state, registers=tuple(registers), memory=memory)
    count = state.registers[1]
    destination = state.registers[7]
    value = state.registers[0] & 0xFFFF
    direction = -2 if state.flags & (1 << 10) else 2
    memory = _as_memory(state.memory)
    if count > len(memory) // 2:
        raise ValueError("REP STOSW count exceeds mapped guest memory")
    for index in range(count):
        address = (destination + index * direction) & MASK64
        memory.read_unsigned(address, 16)
        memory = memory.write_unsigned(address, 16, value)
    registers = list(state.registers)
    registers[1] = 0
    registers[7] = (destination + count * direction) & MASK64
    return replace(state, registers=tuple(registers), memory=memory)


def _string_move(state: MachineExecutionState, instruction) -> MachineExecutionState:
    count = state.registers[1]
    destination, source = state.registers[7], state.registers[6]
    direction = -8 if state.flags & (1 << 10) else 8
    memory = _as_memory(state.memory)
    if count > len(memory) // 8:
        raise ValueError("REP MOVSQ count exceeds mapped guest memory")
    for index in range(count):
        source_address = (source + index * direction) & MASK64
        destination_address = (destination + index * direction) & MASK64
        value = memory.read_unsigned(source_address, 64)
        memory.read_unsigned(destination_address, 64)
        memory = memory.write_unsigned(destination_address, 64, value)
    registers = list(state.registers)
    registers[1] = 0
    registers[6] = (source + count * direction) & MASK64
    registers[7] = (destination + count * direction) & MASK64
    return replace(state, registers=tuple(registers), memory=memory)


def _string_compare(state: MachineExecutionState, instruction) -> MachineExecutionState:
    destination = state.registers[7]
    memory_value = _as_memory(state.memory).read_unsigned(destination, 8)
    accumulator = state.registers[0] & 0xFF
    result = accumulator - memory_value
    flags = _arithmetic_flags(
        state.flags, accumulator, memory_value, result, 8, subtract=True,
    )
    direction = -1 if state.flags & (1 << 10) else 1
    registers = list(state.registers)
    registers[7] = (destination + direction) & MASK64
    return replace(state, registers=tuple(registers), flags=flags)


def _bit_test(state: MachineExecutionState, instruction) -> MachineExecutionState:
    destination, source = instruction.operands[:2]
    width = _data_width(instruction, 0)
    raw_index = read_operand(state, instruction, 1)
    memory = _as_memory(state.memory)
    address = None
    if isinstance(destination, EffectiveAddressOperand):
        address = effective_address(state, instruction, destination)
        if isinstance(source, RegisterOperand):
            source_width = source.width
            signed_index = raw_index
            if signed_index & (1 << (source_width - 1)):
                signed_index -= 1 << source_width
            address = (address + (signed_index // width) * (width // 8)) & MASK64
            bit = signed_index % width
        else:
            bit = raw_index % width
        value = memory.read_unsigned(address, width)
    else:
        bit = raw_index % width
        value = read_operand(state, instruction, 0, width=width)
    flags = _set_flag(state.flags, CF, bool(value & (1 << bit)))
    name = instruction.token.name
    if name.startswith("BTS_"):
        updated = value | (1 << bit)
    elif name.startswith("BTR_"):
        updated = value & ~(1 << bit)
    elif name.startswith("BTC_"):
        updated = value ^ (1 << bit)
    else:
        return replace(state, flags=flags)
    if address is not None:
        memory = memory.write_unsigned(address, width, updated)
        return replace(state, memory=memory, flags=flags)
    return replace(
        write_operand(state, instruction, 0, updated, width=width), flags=flags,
    )


def _bit_scan_reverse(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    source = read_operand(state, instruction, 1, width=width)
    flags = _set_flag(state.flags, ZF, source == 0)
    if source == 0:
        # AMD64 leaves the destination undefined. Preserving it gives the
        # reversible VM a deterministic representative without asserting an
        # architectural value that downstream SSA could depend upon.
        return replace(state, flags=flags)
    return write_operand(
        replace(state, flags=flags), instruction, 0,
        source.bit_length() - 1, width=width,
    )


def _signed_int64_to_float64_bits(value: int, mxcsr: int) -> tuple[int, bool]:
    """Encode signed int64 as binary64 using MXCSR rounding, without host FP."""

    source = int(value) & MASK64
    signed = source - (1 << 64) if source & (1 << 63) else source
    if signed == 0:
        return 0, False
    sign = signed < 0
    magnitude = -signed if sign else signed
    exponent = magnitude.bit_length() - 1
    if exponent <= 52:
        significand = magnitude << (52 - exponent)
        remainder = 0
        shift = 0
    else:
        shift = exponent - 52
        significand = magnitude >> shift
        remainder = magnitude & ((1 << shift) - 1)
    inexact = remainder != 0
    rounding = (int(mxcsr) >> 13) & 0x3
    increment = False
    if remainder:
        if rounding == 0:
            half = 1 << (shift - 1)
            increment = remainder > half or (
                remainder == half and bool(significand & 1)
            )
        elif rounding == 1:  # toward -infinity
            increment = sign
        elif rounding == 2:  # toward +infinity
            increment = not sign
        # rounding == 3 truncates toward zero.
    if increment:
        significand += 1
        if significand == 1 << 53:
            significand >>= 1
            exponent += 1
    encoded = (
        (int(sign) << 63)
        | ((exponent + 1023) << 52)
        | (significand & ((1 << 52) - 1))
    )
    return encoded, inexact


def _signed_int64_to_float32_bits(value: int, mxcsr: int) -> tuple[int, bool]:
    """Encode signed int64 as binary32 using MXCSR, without host FP."""

    source = int(value) & MASK64
    signed = source - (1 << 64) if source & (1 << 63) else source
    if signed == 0:
        return 0, False
    sign = signed < 0
    magnitude = -signed if sign else signed
    exponent = magnitude.bit_length() - 1
    if exponent <= 23:
        significand = magnitude << (23 - exponent)
        remainder = 0
        shift = 0
    else:
        shift = exponent - 23
        significand = magnitude >> shift
        remainder = magnitude & ((1 << shift) - 1)
    inexact = remainder != 0
    rounding = (int(mxcsr) >> 13) & 0x3
    increment = False
    if remainder:
        if rounding == 0:
            half = 1 << (shift - 1)
            increment = remainder > half or (
                remainder == half and bool(significand & 1)
            )
        elif rounding == 1:
            increment = sign
        elif rounding == 2:
            increment = not sign
    if increment:
        significand += 1
        if significand == 1 << 24:
            significand >>= 1
            exponent += 1
    return (
        (int(sign) << 31)
        | ((exponent + 127) << 23)
        | (significand & ((1 << 23) - 1)),
        inexact,
    )


def _signed_integer_to_scalar_float64(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    source = read_operand(state, instruction, 1, width=64)
    mxcsr = int(state.system_state.get("amd64.mxcsr", 0x1F80))
    encoded, inexact = _signed_int64_to_float64_bits(source, mxcsr)
    if inexact:
        if not (mxcsr & (1 << 12)):
            raise FloatingPointError("AMD64 SIMD precision exception")
        mxcsr |= 1 << 5
        system_state = dict(state.system_state)
        system_state["amd64.mxcsr"] = mxcsr
        state = replace(state, system_state=MappingProxyType(system_state))
    # Legacy CVTSI2SD replaces only the low scalar lane.
    return write_operand(state, instruction, 0, encoded, width=64)


def _signed_integer_to_scalar_float32(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    source = read_operand(state, instruction, 1, width=64)
    mxcsr = int(state.system_state.get("amd64.mxcsr", 0x1F80))
    encoded, inexact = _signed_int64_to_float32_bits(source, mxcsr)
    if inexact:
        if not (mxcsr & (1 << 12)):
            raise FloatingPointError("AMD64 SIMD precision exception")
        mxcsr |= 1 << 5
        system_state = dict(state.system_state)
        system_state["amd64.mxcsr"] = mxcsr
        state = replace(state, system_state=MappingProxyType(system_state))
    return write_operand(state, instruction, 0, encoded, width=32)


def _vector_move_low_zero_upper(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    width = 32 if instruction.token.name == "MOVD_XMM_RM32" else 64
    value = read_operand(state, instruction, 1, width=width)
    destination = instruction.operands[0]
    if not isinstance(destination, VectorRegisterOperand):
        raise ValueError("MOVQ zero-upper form requires an XMM destination")
    vectors = list(state.vector_registers)
    vectors[int(destination.register)] = value & _mask(width)
    return replace(state, vector_registers=tuple(vectors))


def _vector_insert_128_lane(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    if len(instruction.operands) != 4:
        raise ValueError("VINSERTF128 requires four explicit operands")
    destination, first_source, _second_source, selector = instruction.operands
    if not isinstance(destination, VectorRegisterOperand) or destination.width != 256:
        raise ValueError("VINSERTF128 destination must be a YMM register")
    if not isinstance(first_source, VectorRegisterOperand) or first_source.width != 256:
        raise ValueError("VINSERTF128 first source must be a YMM register")
    if not isinstance(selector, ImmediateOperand):
        raise ValueError("VINSERTF128 lane selector must be immediate")
    base = read_operand(state, instruction, 1, width=256)
    inserted = read_operand(state, instruction, 2, width=128)
    shift = 128 if selector.value & 1 else 0
    lane_mask = _mask(128) << shift
    result = (base & ~lane_mask) | ((inserted & _mask(128)) << shift)
    return write_operand(state, instruction, 0, result, width=256)


def _float64_nan_bits(value: int) -> bool:
    return (value & 0x7FF0000000000000) == 0x7FF0000000000000 \
        and (value & 0x000FFFFFFFFFFFFF) != 0


def _float64_signaling_nan_bits(value: int) -> bool:
    return _float64_nan_bits(value) and not bool(value & (1 << 51))


def _float32_nan_bits(value: int) -> bool:
    return (value & 0x7F800000) == 0x7F800000 and (value & 0x007FFFFF) != 0


def _float32_signaling_nan_bits(value: int) -> bool:
    return _float32_nan_bits(value) and not bool(value & (1 << 22))


def _mxcsr_invalid(state: MachineExecutionState) -> MachineExecutionState:
    system_state = dict(state.system_state)
    mxcsr = int(system_state.get("amd64.mxcsr", 0x1F80))
    if not (mxcsr & (1 << 7)):
        raise FloatingPointError("AMD64 SIMD invalid-operation exception")
    system_state["amd64.mxcsr"] = mxcsr | 1
    return replace(state, system_state=MappingProxyType(system_state))


def _scalar_float64_compare(
    state: MachineExecutionState, instruction, *, ordered: bool,
) -> MachineExecutionState:
    left_bits = read_operand(state, instruction, 0, width=64)
    right_bits = read_operand(state, instruction, 1, width=64)
    left = struct.unpack("<d", int(left_bits).to_bytes(8, "little"))[0]
    right = struct.unpack("<d", int(right_bits).to_bytes(8, "little"))[0]
    unordered = _float64_nan_bits(left_bits) or _float64_nan_bits(right_bits)
    invalid = (
        unordered if ordered else (
            _float64_signaling_nan_bits(left_bits)
            or _float64_signaling_nan_bits(right_bits)
        )
    )
    if invalid:
        state = _mxcsr_invalid(state)
    flags = state.flags
    for bit in (OF, SF, AF):
        flags = _set_flag(flags, bit, False)
    if unordered:
        cf = pf = zf = True
    elif left > right:
        cf = pf = zf = False
    elif left < right:
        cf, pf, zf = True, False, False
    else:
        cf, pf, zf = False, False, True
    flags = _set_flag(flags, CF, cf)
    flags = _set_flag(flags, PF, pf)
    flags = _set_flag(flags, ZF, zf)
    return replace(state, flags=flags)


def _scalar_float64_compare_unordered(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    return _scalar_float64_compare(state, instruction, ordered=False)


def _scalar_float64_compare_ordered(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    return _scalar_float64_compare(state, instruction, ordered=True)


def _scalar_float32_compare_ordered(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    left_bits = read_operand(state, instruction, 0, width=32)
    right_bits = read_operand(state, instruction, 1, width=32)
    left = struct.unpack("<f", int(left_bits).to_bytes(4, "little"))[0]
    right = struct.unpack("<f", int(right_bits).to_bytes(4, "little"))[0]
    unordered = _float32_nan_bits(left_bits) or _float32_nan_bits(right_bits)
    if unordered:
        state = _mxcsr_invalid(state)
    flags = state.flags
    for bit in (OF, SF, AF):
        flags = _set_flag(flags, bit, False)
    if unordered:
        cf = pf = zf = True
    elif left > right:
        cf = pf = zf = False
    elif left < right:
        cf, pf, zf = True, False, False
    else:
        cf, pf, zf = False, False, True
    flags = _set_flag(flags, CF, cf)
    flags = _set_flag(flags, PF, pf)
    flags = _set_flag(flags, ZF, zf)
    return replace(state, flags=flags)


def _scalar_float64_add(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    mxcsr = int(state.system_state.get("amd64.mxcsr", 0x1F80))
    rounding = (mxcsr >> 13) & 0x3
    if rounding != 0:
        raise ValueError(
            "ADDSD directed rounding awaits exact MXCSR rounding semantics"
        )
    left_bits = read_operand(state, instruction, 0, width=64)
    right_bits = read_operand(state, instruction, 1, width=64)
    left = struct.unpack("<d", int(left_bits).to_bytes(8, "little"))[0]
    right = struct.unpack("<d", int(right_bits).to_bytes(8, "little"))[0]
    encoded = int.from_bytes(struct.pack("<d", left + right), "little")
    # Legacy ADDSD replaces the low lane and preserves the destination high lane.
    return write_operand(state, instruction, 0, encoded, width=64)


def _scalar_float32_add(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    mxcsr = int(state.system_state.get("amd64.mxcsr", 0x1F80))
    if ((mxcsr >> 13) & 0x3) != 0:
        raise ValueError(
            "ADDSS directed rounding awaits exact MXCSR rounding semantics"
        )
    left_bits = read_operand(state, instruction, 0, width=32)
    right_bits = read_operand(state, instruction, 1, width=32)
    left = struct.unpack("<f", int(left_bits).to_bytes(4, "little"))[0]
    right = struct.unpack("<f", int(right_bits).to_bytes(4, "little"))[0]
    encoded = int.from_bytes(struct.pack("<f", left + right), "little")
    # Legacy ADDSS replaces only the low lane and preserves upper 96 bits.
    return write_operand(state, instruction, 0, encoded, width=32)


def _scalar_float32_divide(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    mxcsr = int(state.system_state.get("amd64.mxcsr", 0x1F80))
    if ((mxcsr >> 13) & 0x3) != 0:
        raise ValueError(
            "DIVSS directed rounding awaits exact MXCSR rounding semantics"
        )
    left_bits = read_operand(state, instruction, 0, width=32)
    right_bits = read_operand(state, instruction, 1, width=32)
    left = struct.unpack("<f", int(left_bits).to_bytes(4, "little"))[0]
    right = struct.unpack("<f", int(right_bits).to_bytes(4, "little"))[0]
    # The concrete VM is a diagnostic oracle for the default masked state;
    # repository SSA above retains the complete encoded/MXCSR operation.
    if right == 0.0:
        if left == 0.0:
            state = _mxcsr_invalid(state)
            encoded = 0x7FC00000
        else:
            system_state = dict(state.system_state)
            current = int(system_state.get("amd64.mxcsr", 0x1F80))
            if not (current & (1 << 9)):
                raise FloatingPointError("AMD64 SIMD divide-by-zero exception")
            system_state["amd64.mxcsr"] = current | (1 << 2)
            state = replace(state, system_state=MappingProxyType(system_state))
            sign = ((left_bits ^ right_bits) >> 31) & 1
            encoded = (sign << 31) | 0x7F800000
    else:
        encoded = int.from_bytes(struct.pack("<f", left / right), "little")
    return write_operand(state, instruction, 0, encoded, width=32)


def _scalar_float64_divide(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    mxcsr = int(state.system_state.get("amd64.mxcsr", 0x1F80))
    if ((mxcsr >> 13) & 0x3) != 0:
        raise ValueError(
            "DIVSD directed rounding awaits exact MXCSR rounding semantics"
        )
    left_bits = read_operand(state, instruction, 0, width=64)
    right_bits = read_operand(state, instruction, 1, width=64)
    left = struct.unpack("<d", int(left_bits).to_bytes(8, "little"))[0]
    right = struct.unpack("<d", int(right_bits).to_bytes(8, "little"))[0]
    if right == 0.0:
        if left == 0.0:
            state = _mxcsr_invalid(state)
            encoded = 0x7FF8000000000000
        else:
            system_state = dict(state.system_state)
            current = int(system_state.get("amd64.mxcsr", 0x1F80))
            if not (current & (1 << 9)):
                raise FloatingPointError("AMD64 SIMD divide-by-zero exception")
            system_state["amd64.mxcsr"] = current | (1 << 2)
            state = replace(state, system_state=MappingProxyType(system_state))
            sign = ((left_bits ^ right_bits) >> 63) & 1
            encoded = (sign << 63) | 0x7FF0000000000000
    else:
        encoded = int.from_bytes(struct.pack("<d", left / right), "little")
    return write_operand(state, instruction, 0, encoded, width=64)


def _scalar_float64_subtract(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    mxcsr = int(state.system_state.get("amd64.mxcsr", 0x1F80))
    if ((mxcsr >> 13) & 0x3) != 0:
        raise ValueError(
            "SUBSD directed rounding awaits exact MXCSR rounding semantics"
        )
    left_bits = read_operand(state, instruction, 0, width=64)
    right_bits = read_operand(state, instruction, 1, width=64)
    left = struct.unpack("<d", int(left_bits).to_bytes(8, "little"))[0]
    right = struct.unpack("<d", int(right_bits).to_bytes(8, "little"))[0]
    encoded = int.from_bytes(struct.pack("<d", left - right), "little")
    return write_operand(state, instruction, 0, encoded, width=64)


def _scalar_float64_to_signed_int64_truncate(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    bits = read_operand(state, instruction, 1, width=64)
    sign = (bits >> 63) & 1
    exponent = (bits >> 52) & 0x7FF
    fraction = bits & ((1 << 52) - 1)
    invalid = exponent == 0x7FF
    if not invalid:
        if exponent == 0:
            magnitude = 0
        else:
            unbiased = exponent - 1023
            significand = (1 << 52) | fraction
            if unbiased < 0:
                magnitude = 0
            elif unbiased >= 52:
                magnitude = significand << (unbiased - 52)
            else:
                magnitude = significand >> (52 - unbiased)
        invalid = magnitude > ((1 << 63) if sign else (1 << 63) - 1)
    if invalid:
        state = _mxcsr_invalid(state)
        result = 1 << 63
    else:
        result = (-magnitude if sign else magnitude) & MASK64
    return write_operand(state, instruction, 0, result, width=64)


def _scalar_float64_to_signed_int32_truncate(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    bits = read_operand(state, instruction, 1, width=64)
    sign = (bits >> 63) & 1
    exponent = (bits >> 52) & 0x7FF
    fraction = bits & ((1 << 52) - 1)
    invalid = exponent == 0x7FF
    if not invalid:
        if exponent == 0:
            magnitude = 0
        else:
            unbiased = exponent - 1023
            significand = (1 << 52) | fraction
            if unbiased < 0:
                magnitude = 0
            elif unbiased >= 52:
                magnitude = significand << (unbiased - 52)
            else:
                magnitude = significand >> (52 - unbiased)
        invalid = magnitude > ((1 << 31) if sign else (1 << 31) - 1)
    if invalid:
        state = _mxcsr_invalid(state)
        result = 1 << 31
    else:
        result = (-magnitude if sign else magnitude) & 0xFFFFFFFF
    return write_operand(state, instruction, 0, result, width=32)


def _byte_swap(state: MachineExecutionState, instruction) -> MachineExecutionState:
    width = _data_width(instruction, 0)
    value = read_operand(state, instruction, 0, width=width)
    swapped = int.from_bytes(
        int(value).to_bytes(width // 8, "little"), "big",
    )
    return write_operand(state, instruction, 0, swapped, width=width)


def _vector_unpack_low_qwords(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    left = read_operand(state, instruction, 0, width=128)
    right = read_operand(state, instruction, 1, width=128)
    result = (left & MASK64) | ((right & MASK64) << 64)
    return write_operand(state, instruction, 0, result, width=128)


def _vector_unpack_low_bytes(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    left = read_operand(state, instruction, 0, width=128)
    right = read_operand(state, instruction, 1, width=128)
    result = 0
    for index in range(8):
        result |= ((left >> (index * 8)) & 0xFF) << (index * 16)
        result |= ((right >> (index * 8)) & 0xFF) << (index * 16 + 8)
    return write_operand(state, instruction, 0, result, width=128)


def _vector_unpack_low_words(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    left = read_operand(state, instruction, 0, width=128)
    right = read_operand(state, instruction, 1, width=128)
    result = 0
    for index in range(4):
        result |= ((left >> (index * 16)) & 0xFFFF) << (index * 32)
        result |= ((right >> (index * 16)) & 0xFFFF) << (index * 32 + 16)
    return write_operand(state, instruction, 0, result, width=128)


def _scalar_float64_multiply(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    mxcsr = int(state.system_state.get("amd64.mxcsr", 0x1F80))
    if ((mxcsr >> 13) & 0x3) != 0:
        raise ValueError(
            "MULSD directed rounding awaits exact MXCSR rounding semantics"
        )
    left_bits = read_operand(state, instruction, 0, width=64)
    right_bits = read_operand(state, instruction, 1, width=64)
    left = struct.unpack("<d", int(left_bits).to_bytes(8, "little"))[0]
    right = struct.unpack("<d", int(right_bits).to_bytes(8, "little"))[0]
    encoded = int.from_bytes(struct.pack("<d", left * right), "little")
    return write_operand(state, instruction, 0, encoded, width=64)


def _vector_add_qwords(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    left = read_operand(state, instruction, 0, width=128)
    right = read_operand(state, instruction, 1, width=128)
    result = (
        ((left & MASK64) + (right & MASK64)) & MASK64
        | (
            ((((left >> 64) & MASK64) + ((right >> 64) & MASK64)) & MASK64)
            << 64
        )
    )
    return write_operand(state, instruction, 0, result, width=128)


def _vector_compare_equal_qwords(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    left = read_operand(state, instruction, 0, width=128)
    right = read_operand(state, instruction, 1, width=128)
    low = MASK64 if (left & MASK64) == (right & MASK64) else 0
    high = MASK64 if ((left >> 64) & MASK64) == ((right >> 64) & MASK64) else 0
    return write_operand(state, instruction, 0, low | (high << 64), width=128)


def _vector_subtract_qwords(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    left = read_operand(state, instruction, 0, width=128)
    right = read_operand(state, instruction, 1, width=128)
    result = (
        ((left & MASK64) - (right & MASK64)) & MASK64
        | (
            ((((left >> 64) & MASK64) - ((right >> 64) & MASK64)) & MASK64)
            << 64
        )
    )
    return write_operand(state, instruction, 0, result, width=128)


def _vector_shuffle_dwords(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    source = read_operand(state, instruction, 1, width=128)
    control = read_operand(state, instruction, 2, width=8)
    result = 0
    for destination_lane in range(4):
        source_lane = (control >> (2 * destination_lane)) & 0x3
        lane = (source >> (32 * source_lane)) & 0xFFFFFFFF
        result |= lane << (32 * destination_lane)
    return write_operand(state, instruction, 0, result, width=128)


def _vector_signed_int32_to_float64(
    state: MachineExecutionState, instruction,
) -> MachineExecutionState:
    packed = read_operand(state, instruction, 1, width=64)
    result = 0
    mxcsr = int(state.system_state.get("amd64.mxcsr", 0x1F80))
    for lane in range(2):
        raw = (packed >> (32 * lane)) & 0xFFFFFFFF
        signed = raw - (1 << 32) if raw & (1 << 31) else raw
        encoded, inexact = _signed_int64_to_float64_bits(signed, mxcsr)
        assert not inexact
        result |= encoded << (64 * lane)
    return write_operand(state, instruction, 0, result, width=128)


def default_effect_handlers() -> Mapping[int, object]:
    handlers = {
        MachineSemanticToken.INTEGER_MULTIPLY: _multiply_signed,
        MachineSemanticToken.EFFECTIVE_ADDRESS: _lea,
        MachineSemanticToken.INTEGER_SUBTRACT: _binary_handler("sub"),
        MachineSemanticToken.INTEGER_ADD: _binary_handler("add"),
        MachineSemanticToken.REGISTER_OR_MEMORY_WRITE: _move,
        MachineSemanticToken.REGISTER_OR_MEMORY_READ: _move,
        MachineSemanticToken.REGISTER_WRITE_IMMEDIATE: _move,
        MachineSemanticToken.INTEGER_COMPARE: _binary_handler("cmp", write=False),
        MachineSemanticToken.INTEGER_TEST: _binary_handler("test", write=False),
        MachineSemanticToken.BITWISE_AND: _binary_handler("and"),
        MachineSemanticToken.BITWISE_OR: _binary_handler("or"),
        MachineSemanticToken.BITWISE_XOR: _binary_handler("xor"),
        MachineSemanticToken.STACK_PUSH: _push,
        MachineSemanticToken.STACK_POP: _pop,
        MachineSemanticToken.INTEGER_INCREMENT: _incdec(1),
        MachineSemanticToken.INTEGER_DECREMENT: _incdec(-1),
        MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW: _subtract_with_borrow,
        MachineSemanticToken.INTEGER_NEGATE: _negate,
        MachineSemanticToken.BITWISE_NOT: _not,
        MachineSemanticToken.SHIFT_LEFT: _shift("left"),
        MachineSemanticToken.SHIFT_RIGHT_LOGICAL: _shift("right"),
        MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC: _shift("arithmetic"),
        MachineSemanticToken.ROTATE_LEFT: _rotate("left"),
        MachineSemanticToken.ROTATE_RIGHT: _rotate("right"),
        MachineSemanticToken.ATOMIC_COMPARE_EXCHANGE: _compare_exchange,
        MachineSemanticToken.ATOMIC_EXCHANGE_ADD: _exchange_add,
        MachineSemanticToken.ATOMIC_ADD: _binary_handler("add"),
        MachineSemanticToken.EXCHANGE: _exchange,
        MachineSemanticToken.VECTOR_XOR: _vector_xor,
        MachineSemanticToken.VECTOR_AND: _vector_and,
        MachineSemanticToken.VECTOR_MOVE: _move,
        MachineSemanticToken.VECTOR_SHIFT_RIGHT_LOGICAL: _vector_shift_right_logical,
        MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED: _multiply_unsigned,
        MachineSemanticToken.INTEGER_DIVIDE: _divide(False),
        MachineSemanticToken.INTEGER_DIVIDE_SIGNED: _divide(True),
        MachineSemanticToken.SIGN_EXTEND: _extend(True),
        MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR: _sign_extend_accumulator,
        MachineSemanticToken.ZERO_EXTEND: _extend(False),
        MachineSemanticToken.NO_OPERATION: _noop,
        MachineSemanticToken.CONDITIONAL_MOVE: _conditional_move,
        MachineSemanticToken.CONDITIONAL_SET: _conditional_set,
        MachineSemanticToken.DIRECT_RELATIVE_CALL: _call_stack_effect,
        MachineSemanticToken.INDIRECT_CALL: _call_stack_effect,
        MachineSemanticToken.RETURN: _return_stack_effect,
        MachineSemanticToken.STRING_STORE: _string_store,
        MachineSemanticToken.STRING_MOVE: _string_move,
        MachineSemanticToken.STRING_COMPARE: _string_compare,
        MachineSemanticToken.BIT_TEST: _bit_test,
        MachineSemanticToken.BIT_TEST_RESET: _bit_test,
        MachineSemanticToken.BIT_TEST_COMPLEMENT: _bit_test,
        MachineSemanticToken.BIT_SCAN_REVERSE: _bit_scan_reverse,
        MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT64: _signed_integer_to_scalar_float64,
        MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT32: _signed_integer_to_scalar_float32,
        MachineSemanticToken.VECTOR_MOVE_LOW_ZERO_UPPER: _vector_move_low_zero_upper,
        MachineSemanticToken.SCALAR_FLOAT64_COMPARE_UNORDERED: _scalar_float64_compare_unordered,
        MachineSemanticToken.SCALAR_FLOAT64_ADD: _scalar_float64_add,
        MachineSemanticToken.SCALAR_FLOAT32_ADD: _scalar_float32_add,
        MachineSemanticToken.SCALAR_FLOAT32_DIVIDE: _scalar_float32_divide,
        MachineSemanticToken.SCALAR_FLOAT64_DIVIDE: _scalar_float64_divide,
        MachineSemanticToken.SCALAR_FLOAT64_SUBTRACT: _scalar_float64_subtract,
        MachineSemanticToken.SCALAR_FLOAT64_TO_SIGNED_INT64_TRUNCATE: _scalar_float64_to_signed_int64_truncate,
        MachineSemanticToken.SCALAR_FLOAT64_TO_SIGNED_INT32_TRUNCATE: _scalar_float64_to_signed_int32_truncate,
        MachineSemanticToken.BYTE_SWAP: _byte_swap,
        MachineSemanticToken.VECTOR_UNPACK_LOW_QWORDS: _vector_unpack_low_qwords,
        MachineSemanticToken.VECTOR_UNPACK_LOW_BYTES: _vector_unpack_low_bytes,
        MachineSemanticToken.SCALAR_FLOAT64_MULTIPLY: _scalar_float64_multiply,
        MachineSemanticToken.VECTOR_ADD_QWORDS: _vector_add_qwords,
        MachineSemanticToken.VECTOR_COMPARE_EQUAL_QWORDS: _vector_compare_equal_qwords,
        MachineSemanticToken.VECTOR_SUBTRACT_QWORDS: _vector_subtract_qwords,
        MachineSemanticToken.VECTOR_SHUFFLE_DWORDS: _vector_shuffle_dwords,
        MachineSemanticToken.VECTOR_SIGNED_INT32_TO_FLOAT64: _vector_signed_int32_to_float64,
        MachineSemanticToken.SCALAR_FLOAT64_COMPARE_ORDERED: _scalar_float64_compare_ordered,
        MachineSemanticToken.SCALAR_FLOAT32_COMPARE_ORDERED: _scalar_float32_compare_ordered,
        MachineSemanticToken.VECTOR_UNPACK_LOW_WORDS: _vector_unpack_low_words,
        MachineSemanticToken.VECTOR_INSERT_128_LANE: _vector_insert_128_lane,
    }
    return MappingProxyType({int(token): handler for token, handler in handlers.items()})


def build_external_references(program) -> tuple[MachineExternalReference, ...]:
    return tuple(
        MachineExternalReference(
            reference_id=index + 1,
            target_address=EXTERNAL_TARGET_BASE + index * 16,
            domain="guest-binary",
            library=symbol.library,
            symbol=symbol.name if symbol.name is not None else f"ordinal:{symbol.ordinal}",
        )
        for index, symbol in enumerate((
            *getattr(program.image, "imports", ()),
            *getattr(program.image, "delay_imports", ()),
        ))
    )


def map_pe_image(
    memory: PagedByteMemory,
    image,
    *,
    load_address: int | None = None,
) -> tuple[PagedByteMemory, int, int]:
    """Map one bounded PE image and apply its loader relocations."""

    preferred_base = int(image.image_base)
    runtime_base = preferred_base if load_address is None else int(load_address)
    if runtime_base < 0 or runtime_base >= 1 << 64:
        raise ValueError("PE load address must fit an unsigned 64-bit address")
    load_bias = runtime_base - preferred_base
    sections = tuple(getattr(image, "sections", ()))
    encoded = bytes(getattr(image, "encoded", b""))
    if encoded:
        header_size = min((section.raw_offset for section in sections), default=len(encoded))
        memory = memory.map_bytes(runtime_base, encoded[:header_size])
        for section in sections:
            raw = encoded[section.raw_offset:section.raw_offset + section.raw_size]
            span = max(section.virtual_size, section.raw_size)
            memory = memory.map_zeroes(runtime_base + section.virtual_address, span)
            memory = memory.map_bytes(runtime_base + section.virtual_address, raw)
    relocations = tuple(getattr(image, "base_relocations", ()))
    if load_bias and not relocations:
        raise ValueError(
            f"PE image cannot move from {preferred_base:#x} to {runtime_base:#x}: "
            "no base relocation records"
        )
    for relocation in relocations:
        if not load_bias:
            break
        relocation_type = int(relocation.type)
        if relocation_type == 10 and bool(getattr(image, "pe32_plus", True)):
            width = 64  # IMAGE_REL_BASED_DIR64
        elif relocation_type == 3 and not bool(getattr(image, "pe32_plus", True)):
            width = 32  # IMAGE_REL_BASED_HIGHLOW
        else:
            raise ValueError(
                f"unsupported PE base relocation type {relocation_type} "
                f"at RVA {int(relocation.rva):#x}"
            )
        address = runtime_base + int(relocation.rva)
        original = memory.read_unsigned(address, width)
        memory = memory.write_unsigned(address, width, original + load_bias)
    return memory, runtime_base, len(relocations) if load_bias else 0


def _pe_reserved_span(image, runtime_base: int) -> tuple[int, int]:
    header_size = min(
        (int(section.raw_offset) for section in getattr(image, "sections", ())),
        default=len(getattr(image, "encoded", b"")),
    )
    image_size = max(
        (int(section.virtual_address) + max(
            int(section.virtual_size), int(section.raw_size),
        ) for section in getattr(image, "sections", ())),
        default=header_size,
    )
    return int(runtime_base), int(runtime_base) + max(header_size, image_size)


def build_initial_machine_state(program, *, load_address: int | None = None, additional_images=(), import_targets=None, module_handle_targets=None, stack_top: int = 0x00007FFF00000000, stack_size: int = 1024 * 1024, teb_base: int = 0x00007FFE00000000, peb_base: int = 0x00007FFD00000000, system_arena_base: int = 0x00007FFC00000000, system_arena_size: int = 2 * 1024 * 1024, external_references=(), virtual_filesystem=None, environment_state=None) -> MachineExecutionState:
    """Map a linked PE image set plus a zeroed, ABI-aligned guest stack."""

    image = program.image
    memory, runtime_base, applied_relocations = map_pe_image(
        PagedByteMemory.empty(), image, load_address=load_address,
    )
    preferred_base = int(image.image_base)
    load_bias = runtime_base - preferred_base
    relocations = tuple(getattr(image, "base_relocations", ()))
    mapped_images = [(image, runtime_base)]
    for linked_image, linked_base in additional_images:
        memory, actual_base, _count = map_pe_image(
            memory, linked_image, load_address=int(linked_base),
        )
        mapped_images.append((linked_image, actual_base))
    reserved_ranges = [
        (*_pe_reserved_span(mapped_image, base), f"PE image at {base:#x}")
        for mapped_image, base in mapped_images
    ]
    reserved_ranges.extend((
        (stack_top - stack_size, stack_top, "guest stack"),
        (teb_base, teb_base + 4096, "guest TEB"),
        (peb_base, peb_base + 4096, "guest PEB"),
        (
            system_arena_base,
            system_arena_base + system_arena_size,
            "guest system arena",
        ),
    ))
    for left, right in zip(sorted(reserved_ranges), sorted(reserved_ranges)[1:]):
        if left[1] > right[0]:
            raise ValueError(
                f"mapped address ranges overlap: {left[2]} and {right[2]}"
            )
    imports = (
        *getattr(image, "imports", ()),
        *getattr(image, "delay_imports", ()),
    )
    references = tuple(external_references)
    if import_targets is None:
        if len(imports) != len(references):
            if imports or references:
                raise ValueError("each PE import requires exactly one external reference")
        active_targets = {
            (runtime_base, int(symbol.iat_rva)): int(reference.target_address)
            for symbol, reference in zip(imports, references)
        }
    else:
        active_targets = {
            (int(base), int(rva)): int(target)
            for (base, rva), target in import_targets.items()
        }
    expected_slots = {
        (base, int(symbol.iat_rva))
        for mapped_image, base in mapped_images
        for symbol in (
            *getattr(mapped_image, "imports", ()),
            *getattr(mapped_image, "delay_imports", ()),
        )
    }
    if set(active_targets) != expected_slots:
        missing = expected_slots - set(active_targets)
        unexpected = set(active_targets) - expected_slots
        raise ValueError(
            f"linked PE import target plan mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    for mapped_image, base in mapped_images:
        for symbol in (
            *getattr(mapped_image, "imports", ()),
            *getattr(mapped_image, "delay_imports", ()),
        ):
            memory = memory.write_unsigned(
                base + symbol.iat_rva,
                64 if mapped_image.pe32_plus else 32,
                active_targets[(base, int(symbol.iat_rva))],
            )
    for (base, rva), target in (module_handle_targets or {}).items():
        owner_image = next(
            mapped_image for mapped_image, mapped_base in mapped_images
            if int(mapped_base) == int(base)
        )
        memory = memory.write_unsigned(
            int(base) + int(rva),
            64 if owner_image.pe32_plus else 32,
            int(target),
        )
    memory = memory.map_zeroes(stack_top - stack_size, stack_size)
    memory = memory.map_zeroes(teb_base, 4096)
    memory = memory.map_zeroes(peb_base, 4096)
    memory = memory.map_zeroes(system_arena_base, system_arena_size)
    # Minimal deterministic x64 Windows process/thread environment.  Fields
    # outside this declared page stay unmapped and therefore trap visibly.
    memory = memory.write_unsigned(teb_base + 0x30, 64, teb_base)
    memory = memory.write_unsigned(teb_base + 0x60, 64, peb_base)
    loader_order = (*mapped_images[1:], mapped_images[0])
    tls_images = tuple(
        (mapped_image, base, mapped_image.tls_directory)
        for mapped_image, base in loader_order
        if getattr(mapped_image, "tls_directory", None) is not None
    )
    arena_cursor = int(system_arena_base)
    tls_vector = 0
    tls_records: list[tuple[int, int, object]] = []
    if tls_images:
        arena_cursor = (arena_cursor + 15) & ~15
        tls_vector = arena_cursor
        arena_cursor += len(tls_images) * 8
        memory = memory.write_unsigned(teb_base + 0x58, 64, tls_vector)
        for tls_index, (mapped_image, base, tls) in enumerate(tls_images):
            arena_cursor = (arena_cursor + 15) & ~15
            tls_base = arena_cursor
            allocation_size = max(1, len(tls.template) + int(tls.zero_fill_size))
            arena_cursor += allocation_size
            if arena_cursor > system_arena_base + system_arena_size:
                raise ValueError("PE TLS allocations exceed the guest system arena")
            memory = memory.map_zeroes(tls_base, allocation_size)
            memory = memory.map_bytes(tls_base, tls.template)
            memory = memory.write_unsigned(tls_vector + tls_index * 8, 64, tls_base)
            # AddressOfIndex names a DWORD even in PE32+ images.
            memory = memory.write_unsigned(base + int(tls.index_rva), 32, tls_index)
            tls_records.append((tls_index, tls_base, tls))

    rsp = stack_top - 8
    memory = memory.write_unsigned(rsp, 64, 0)
    registers = [0] * 16
    registers[4] = rsp
    system_state = {
        "machine.memory.page_size": memory.page_size,
        "windows.system_arena.page_base": system_arena_base // memory.page_size,
        "windows.system_arena.page_count": system_arena_size // memory.page_size,
        "windows.system_arena_base": system_arena_base,
        "windows.system_arena_limit": system_arena_base + system_arena_size,
        "windows.system_arena_cursor": arena_cursor,
        "windows.loader.preferred_image_base": preferred_base,
        "windows.loader.image_base": runtime_base,
        "windows.loader.load_bias": load_bias,
        "windows.loader.base_relocation_catalog_count": len(relocations),
        "windows.loader.base_relocation_count": applied_relocations,
        "windows.loader.module_count": len(mapped_images),
        "windows.loader.tls_module_count": len(tls_records),
        "windows.loader.tls_vector": tls_vector,
    }
    for tls_index, tls_base, _tls in tls_records:
        system_state[f"windows.loader.tls.{tls_index}.base"] = tls_base
        system_state[f"windows.loader.tls.{tls_index}.size"] = max(
            1, len(_tls.template) + int(_tls.zero_fill_size),
        )
    startup_calls: list[tuple[int, int, int, int]] = []
    tls_callback_count = 0
    for mapped_image, base in loader_order:
        tls = getattr(mapped_image, "tls_directory", None)
        for callback_rva in (() if tls is None else tls.callbacks):
            # address, module base, kind (1=TLS), requires-success
            startup_calls.append((base + int(callback_rva), base, 1, 0))
            tls_callback_count += 1
        if mapped_image is not image and bool(getattr(mapped_image, "is_dll", False)):
            entry_rva = int(getattr(mapped_image, "entrypoint_rva", 0))
            if entry_rva:
                # DllMain returns BOOL and false aborts process attach.
                startup_calls.append((base + entry_rva, base, 2, 1))
    entrypoint = runtime_base + image.entrypoint_rva
    system_state["windows.loader.entrypoint"] = entrypoint
    system_state["windows.loader.tls_callback_count"] = tls_callback_count
    system_state["windows.loader.tls_callback_index"] = 0
    system_state["windows.loader.startup_call_count"] = len(startup_calls)
    system_state["windows.loader.startup_call_index"] = 0
    system_state["windows.loader.startup_direction"] = 1
    system_state["windows.loader.completion_action"] = 0
    system_state["windows.loader.startup_reason"] = 1  # DLL_PROCESS_ATTACH
    for call_index, (target, module_base, kind, requires_success) in enumerate(startup_calls):
        prefix = f"windows.loader.startup_call.{call_index}"
        system_state[f"{prefix}.address"] = target
        system_state[f"{prefix}.module_base"] = module_base
        system_state[f"{prefix}.kind"] = kind
        system_state[f"{prefix}.requires_success"] = requires_success
    call_stack = ()
    pc = entrypoint
    if startup_calls:
        pc, module_base, _kind, _requires_success = startup_calls[0]
        registers[1] = module_base
        registers[2] = 1  # DLL_PROCESS_ATTACH
        registers[8] = 0
        registers[4] -= 8
        memory = memory.write_unsigned(
            registers[4], 64, MACHINE_LOADER_CALLBACK_RETURN,
        )
        call_stack = (MACHINE_LOADER_CALLBACK_RETURN,)
    else:
        system_state["windows.loader.tls_callbacks_complete"] = 1
        system_state["windows.loader.startup_calls_complete"] = 1
    executable_pages: set[int] = set()
    image_pages: set[int] = set()
    for mapped_image, base in mapped_images:
        begin, end = _pe_reserved_span(mapped_image, base)
        image_pages.update(range(begin // 4096, (end + 4095) // 4096))
        for section in getattr(mapped_image, "sections", ()):
            if not bool(getattr(section, "executable", False)):
                continue
            section_begin = base + int(section.virtual_address)
            section_size = max(int(section.virtual_size), int(section.raw_size))
            if section_size:
                executable_pages.update(range(
                    section_begin // 4096,
                    (section_begin + section_size + 4095) // 4096,
                ))
    virtual_memory = VirtualMemoryState.from_mapped_pages(
        memory.pages, executable_pages=executable_pages,
        image_pages=image_pages, page_size=memory.page_size,
    )
    return MachineExecutionState(
        pc=pc,
        registers=tuple(registers),
        memory=memory,
        system_state=MappingProxyType(system_state),
        virtual_filesystem=virtual_filesystem,
        virtual_registry=VirtualRegistryState.create(),
        virtual_memory=virtual_memory,
        environment_state=MappingProxyType(dict(environment_state or {})),
        gs_base=teb_base,
        call_stack=call_stack,
    )


def complete_external_call_state(
    state: MachineExecutionState,
    completion: MachineExternalCallCompletion,
) -> MachineExecutionState:
    """Return from one captured external call without bypassing guest state."""

    matches = tuple(
        request for request in state.external_requests
        if request.request_id == int(completion.request_id)
    )
    if len(matches) != 1:
        raise KeyError(f"external request {completion.request_id} is not pending")
    request = matches[0]
    if not state.call_stack or state.call_stack[-1] != request.return_address:
        raise RuntimeError("external completion does not match the reversible call stack")
    registers = list(state.registers)
    registers[0] = int(completion.result) & MASK64
    written_registers: set[int] = set()
    for effect in completion.register_writes:
        if effect.register in written_registers:
            raise ValueError("external completion writes one register more than once")
        written_registers.add(effect.register)
        registers[effect.register] = int(effect.value) & MASK64
    memory = _as_memory(state.memory)
    virtual_memory = state.virtual_memory
    if completion.virtual_memory_effects:
        if virtual_memory is None:
            raise RuntimeError("external completion requires installed virtual-memory metadata")
        for effect in completion.virtual_memory_effects:
            virtual_memory = virtual_memory.apply(effect)
            if effect.operation == "allocate":
                memory = memory.map_zeroes(effect.base, effect.size)
            else:
                memory = memory.unmap(effect.base, effect.size)
    for effect in completion.memory_writes:
        # ``map_bytes`` intentionally requires the destination page to exist
        # conceptually; verify every byte first so a host cannot manufacture a
        # new guest mapping by returning an effect.
        for index in range(len(effect.data)):
            memory[effect.address + index]
        memory = memory.map_bytes(effect.address, effect.data)
    system_state = dict(state.system_state)
    for effect in completion.system_writes:
        system_state[effect.key] = int(effect.value) & MASK64
    virtual_filesystem = state.virtual_filesystem
    if completion.filesystem_effects:
        if virtual_filesystem is None:
            raise RuntimeError("external completion requires an installed virtual filesystem")
        for effect in completion.filesystem_effects:
            virtual_filesystem = virtual_filesystem.apply(effect)
    virtual_registry = state.virtual_registry
    if completion.registry_effects:
        if virtual_registry is None:
            raise RuntimeError("external completion requires an installed virtual registry")
        for effect in completion.registry_effects:
            virtual_registry = virtual_registry.apply(effect)
    environment_state = dict(state.environment_state)
    for effect in completion.environment_writes:
        existing = next((key for key in environment_state if key.casefold() == effect.key.casefold()), None)
        if existing is not None:
            del environment_state[existing]
        if effect.value is not None:
            environment_state[effect.key] = effect.value
    text_state = dict(state.text_state)
    for effect in completion.text_writes:
        text_state[effect.key] = effect.value
    device_state = dict(state.device_state)
    device_generations = dict(state.device_generations)
    for effect in completion.device_writes:
        previous = device_state.get(effect.device, b"") if effect.append else b""
        device_state[effect.device] = previous + effect.data
        device_generations[effect.device] = device_generations.get(effect.device, 0) + 1
    pending = tuple(
        item for item in state.external_requests
        if item.request_id != completion.request_id
    )
    guest_calls = tuple(int(address) & MASK64 for address in completion.guest_calls)
    transfer = completion.control_transfer
    if transfer is not None:
        if guest_calls or completion.terminate:
            raise ValueError("nonlocal external transfer cannot also call or terminate")
        return replace(
            state,
            pc=int(transfer.address) & MASK64,
            registers=tuple(registers),
            vector_registers=(
                state.vector_registers
                if transfer.vector_registers is None
                else tuple(transfer.vector_registers)
            ),
            memory=memory,
            system_state=MappingProxyType(system_state),
            virtual_filesystem=virtual_filesystem,
            virtual_registry=virtual_registry,
            virtual_memory=virtual_memory,
            environment_state=MappingProxyType(environment_state),
            text_state=MappingProxyType(text_state),
            device_state=MappingProxyType(device_state),
            device_generations=MappingProxyType(device_generations),
            call_stack=tuple(transfer.call_stack),
            external_requests=pending,
        )
    if guest_calls:
        # The original caller return is already at [RSP]. Push the remaining
        # callbacks in reverse order so ordinary RET semantics visit each and
        # finally consume that original return.
        memory = _as_memory(memory)
        base_stack = state.call_stack
        if completion.terminate:
            memory = memory.write_unsigned(
                registers[4], 64, MACHINE_TERMINATION_RETURN,
            )
            base_stack = (*state.call_stack[:-1], MACHINE_TERMINATION_RETURN)
        for address in reversed(guest_calls[1:]):
            registers[4] = (registers[4] - 8) & MASK64
            memory = memory.write_unsigned(registers[4], 64, address)
        return replace(
            state,
            pc=guest_calls[0],
            registers=tuple(registers),
            memory=memory,
            system_state=MappingProxyType(system_state),
            virtual_filesystem=virtual_filesystem,
            virtual_registry=virtual_registry,
            virtual_memory=virtual_memory,
            environment_state=MappingProxyType(environment_state),
            text_state=MappingProxyType(text_state),
            device_state=MappingProxyType(device_state),
            device_generations=MappingProxyType(device_generations),
            call_stack=(*base_stack, *reversed(guest_calls[1:])),
            external_requests=pending,
            termination_requested=completion.terminate or state.termination_requested,
            exit_code=(
                int(completion.exit_code) & 0xFFFFFFFF
                if completion.terminate else state.exit_code
            ),
        )
    registers[4] = (registers[4] + 8) & MASK64
    return replace(
        state,
        pc=request.return_address,
        registers=tuple(registers),
        memory=memory,
        system_state=MappingProxyType(system_state),
        virtual_filesystem=virtual_filesystem,
        virtual_registry=virtual_registry,
        virtual_memory=virtual_memory,
        environment_state=MappingProxyType(environment_state),
        text_state=MappingProxyType(text_state),
        device_state=MappingProxyType(device_state),
        device_generations=MappingProxyType(device_generations),
        call_stack=state.call_stack[:-1],
        external_requests=pending,
        termination_requested=False if completion.terminate else state.termination_requested,
        halted=completion.terminate or state.halted,
        exit_code=(
            int(completion.exit_code) & 0xFFFFFFFF
            if completion.terminate else state.exit_code
        ),
    )


__all__ = [
    "PagedByteMemory",
    "build_external_references",
    "build_initial_machine_state",
    "condition_holds",
    "complete_external_call_state",
    "default_effect_handlers",
    "effective_address",
    "indirect_target",
    "read_operand",
    "write_operand",
]
