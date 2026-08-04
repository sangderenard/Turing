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
    X86HighByteRegister,
)


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
    match = re.search(r"(?:RM|R|M)(128|64|32|16|8)(?:_|$)", name)
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
        MachineSemanticToken.SIGN_EXTEND: _extend(True),
        MachineSemanticToken.ZERO_EXTEND: _extend(False),
        MachineSemanticToken.NO_OPERATION: _noop,
        MachineSemanticToken.CONDITIONAL_MOVE: _conditional_move,
        MachineSemanticToken.CONDITIONAL_SET: _conditional_set,
        MachineSemanticToken.DIRECT_RELATIVE_CALL: _call_stack_effect,
        MachineSemanticToken.INDIRECT_CALL: _call_stack_effect,
        MachineSemanticToken.RETURN: _return_stack_effect,
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
        for index, symbol in enumerate(getattr(program.image, "imports", ()))
    )


def build_initial_machine_state(program, *, stack_top: int = 0x00007FFF00000000, stack_size: int = 1024 * 1024, external_references=()) -> MachineExecutionState:
    """Map a preferred-base PE image and a zeroed, ABI-aligned guest stack."""

    image = program.image
    memory = PagedByteMemory.empty()
    sections = tuple(getattr(image, "sections", ()))
    encoded = bytes(getattr(image, "encoded", b""))
    if encoded:
        header_size = min((section.raw_offset for section in sections), default=len(encoded))
        memory = memory.map_bytes(image.image_base, encoded[:header_size])
        for section in sections:
            raw = encoded[section.raw_offset:section.raw_offset + section.raw_size]
            span = max(section.virtual_size, section.raw_size)
            memory = memory.map_zeroes(image.image_base + section.virtual_address, span)
            memory = memory.map_bytes(image.image_base + section.virtual_address, raw)
    imports = tuple(getattr(image, "imports", ()))
    references = tuple(external_references)
    if len(imports) != len(references):
        if imports or references:
            raise ValueError("each PE import requires exactly one external reference")
    for symbol, reference in zip(imports, references):
        memory = memory.write_unsigned(
            image.image_base + symbol.iat_rva,
            64 if image.pe32_plus else 32,
            reference.target_address,
        )
    memory = memory.map_zeroes(stack_top - stack_size, stack_size)
    rsp = stack_top - 8
    memory = memory.write_unsigned(rsp, 64, 0)
    registers = [0] * 16
    registers[4] = rsp
    return MachineExecutionState(
        pc=image.image_base + image.entrypoint_rva,
        registers=tuple(registers),
        memory=memory,
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
    registers[4] = (registers[4] + 8) & MASK64
    memory = _as_memory(state.memory)
    for effect in completion.memory_writes:
        # ``map_bytes`` intentionally requires the destination page to exist
        # conceptually; verify every byte first so a host cannot manufacture a
        # new guest mapping by returning an effect.
        for index in range(len(effect.data)):
            memory[effect.address + index]
        memory = memory.map_bytes(effect.address, effect.data)
    return replace(
        state,
        pc=request.return_address,
        registers=tuple(registers),
        memory=memory,
        call_stack=state.call_stack[:-1],
        external_requests=tuple(
            item for item in state.external_requests
                if item.request_id != completion.request_id
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
