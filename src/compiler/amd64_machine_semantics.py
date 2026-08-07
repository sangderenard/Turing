"""Deterministic AMD64 state semantics for the reversible machine program.

The decoder owns instruction boundaries and operand shapes.  This module owns
the corresponding architectural state transformations: little-endian memory,
sub-register writes, integer flags, stack effects, and condition evaluation.
Every operation returns a new immutable machine state so the execution journal
can restore an exact predecessor without running an inverse instruction.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
import re

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
        cursor = 0
        while cursor < len(raw):
            absolute = int(address) + cursor
            page_index, page_offset = divmod(absolute, self.page_size)
            count = min(self.page_size - page_offset, len(raw) - cursor)
            page = bytearray(pages.get(page_index, bytes(self.page_size)))
            page[page_offset:page_offset + count] = raw[cursor:cursor + count]
            pages[page_index] = bytes(page)
            cursor += count
        return PagedByteMemory(MappingProxyType(pages), self.page_size)

    def map_zeroes(self, address: int, size: int) -> "PagedByteMemory":
        if size < 0:
            raise ValueError("mapped region size cannot be negative")
        return self.map_bytes(address, bytes(size))

    def unmap(self, address: int, size: int) -> "PagedByteMemory":
        if address % self.page_size or size < 0 or size % self.page_size:
            raise ValueError("unmapped region must be page aligned")
        pages = dict(self.pages)
        for page in range(address // self.page_size, (address + size) // self.page_size):
            pages.pop(page, None)
        return PagedByteMemory(MappingProxyType(pages), self.page_size)

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
            value if target_width == 128
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
    """Implement the implicit accumulator forms CDQE and CQO."""

    registers = list(state.registers)
    if instruction.token.name == "CDQE":
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


def default_effect_handlers() -> Mapping[int, object]:
    handlers = {
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
