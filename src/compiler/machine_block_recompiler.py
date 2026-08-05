"""Executable WebAssembly lowering for reversible AMD64 translated blocks.

The admitted tier covers scalar register work, bounded static memory, direct
control, and exact-state-specialized call/return boundaries. The emitted Wasm mutates the same contiguous
16-register/PC/flags/step ABI and writes a complete architectural checkpoint
plus an instruction provenance witness after every guest instruction.  A host
can therefore commit one ordinary reversible edge per checkpoint instead of
turning a compiled block into one opaque, irreversible transition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import math
import re
import struct
from types import MappingProxyType
from typing import Any, Mapping

from .machine_reference_vocabulary import (
    EffectiveAddressOperand,
    ImmediateOperand,
    MachineSemanticToken,
    RegisterOperand,
    RelativeAddressOperand,
    VectorRegisterOperand,
)
from .wasm_binary import (
    CodeBuilder,
    OP_I32_ADD,
    OP_I32_EQZ,
    OP_I32_AND,
    OP_I32_OR,
    OP_I32_XOR,
    OP_I64_ADD,
    OP_I64_AND,
    OP_I64_EQZ,
    OP_I64_EXTEND_I32_U,
    OP_I64_LT_U,
    OP_I64_MUL,
    OP_I64_OR,
    OP_I64_POPCNT,
    OP_I64_ROTL,
    OP_I64_SHL,
    OP_I64_SHR_S,
    OP_I64_SHR_U,
    OP_I64_SUB,
    OP_I64_XOR,
    OP_SELECT,
    build_module,
)


MASK64 = (1 << 64) - 1
MACHINE_BLOCK_STATE_SCHEMA = "turing.machine-block-state.v2"
MACHINE_BLOCK_JOURNAL_SCHEMA = "turing.machine-block-journal.v2"
REGISTER_COUNT = 16
VECTOR_REGISTER_COUNT = 16
PC_OFFSET = REGISTER_COUNT * 8
FLAGS_OFFSET = PC_OFFSET + 8
STEPS_OFFSET = FLAGS_OFFSET + 8
VECTOR_OFFSET = STEPS_OFFSET + 8
STATE_QWORD_COUNT = REGISTER_COUNT + 3 + VECTOR_REGISTER_COUNT * 2
STATE_SIZE = VECTOR_OFFSET + VECTOR_REGISTER_COUNT * 16
JOURNAL_STATE_OFFSET = 24
JOURNAL_EFFECT_OFFSET = JOURNAL_STATE_OFFSET + STATE_SIZE
JOURNAL_STACK_OFFSET = JOURNAL_EFFECT_OFFSET + 56
JOURNAL_STRIDE = JOURNAL_STACK_OFFSET + 24
MAXIMUM_GUEST_WINDOW_BYTES = 64 * 1024
CF, PF, AF, ZF, SF, OF = 0, 2, 4, 6, 7, 11
ARITHMETIC_FLAGS = sum(1 << bit for bit in (CF, PF, AF, ZF, SF, OF))
LOGICAL_FLAGS = sum(1 << bit for bit in (CF, PF, ZF, SF, OF))


class MachineBlockLoweringError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MachineBlockLoweringShortfall:
    operation_index: int
    address: int
    semantic: str
    reason: str


@dataclass(frozen=True, slots=True)
class MachineBlockInstructionWitness:
    operation_index: int
    address: int
    semantic: str
    semantic_id: int
    encoded: bytes
    encoded_digest: str
    journal_offset: int
    possible_next_addresses: tuple[int, ...]
    expected_stack_effect: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MachineBlockWasmArtifact:
    entry_address: int
    block_digest: str
    binary: bytes
    wat: str
    witnesses: tuple[MachineBlockInstructionWitness, ...]
    shortfalls: tuple[MachineBlockLoweringShortfall, ...]
    continuation_address: int
    possible_continuations: tuple[int, ...]
    guest_memory_base: int
    guest_memory_size: int
    state_abi: Mapping[str, Any]
    specialization_guard: Mapping[str, Any]

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    @property
    def covered_operation_count(self) -> int:
        return len(self.witnesses)

    def _validate_specialization(self, state) -> None:
        guard = self.specialization_guard
        for register, expected in guard.get("registers", ()):
            if int(state.registers[int(register)]) != int(expected):
                raise ValueError("machine Wasm specialization register mismatch")
        if "fs_base" in guard and int(state.fs_base) != int(guard["fs_base"]):
            raise ValueError("machine Wasm specialization FS base mismatch")
        if "gs_base" in guard and int(state.gs_base) != int(guard["gs_base"]):
            raise ValueError("machine Wasm specialization GS base mismatch")
        if "call_stack" in guard and tuple(state.call_stack) != tuple(guard["call_stack"]):
            raise ValueError("machine Wasm specialization call stack mismatch")
        if "termination_requested" in guard and bool(state.termination_requested) != bool(
            guard["termination_requested"]
        ):
            raise ValueError("machine Wasm specialization termination mismatch")

    def pack_state(self, state) -> bytes:
        self._validate_specialization(state)
        values = (
            *tuple(int(value) & MASK64 for value in state.registers),
            int(state.pc) & MASK64,
            int(state.flags) & MASK64,
            int(state.steps) & MASK64,
            *tuple(
                half
                for value in state.vector_registers
                for half in (
                    int(value) & MASK64,
                    (int(value) >> 64) & MASK64,
                )
            ),
        )
        return struct.pack(f"<{STATE_QWORD_COUNT}Q", *values)

    def pack_guest_memory(self, state) -> bytes:
        self._validate_specialization(state)
        if not self.guest_memory_size:
            return b""
        return bytes(
            state.memory[self.guest_memory_base + offset]
            for offset in range(self.guest_memory_size)
        )

    def states_from_journal(self, encoded: bytes, source_state) -> tuple[Any, ...]:
        self._validate_specialization(source_state)
        required = len(self.witnesses) * JOURNAL_STRIDE
        if len(encoded) < required:
            raise ValueError("compiled machine journal is truncated")
        states = []
        active_state = source_state
        for witness in self.witnesses:
            base = witness.operation_index * JOURNAL_STRIDE
            address, semantic_id, digest_prefix = struct.unpack_from("<QQQ", encoded, base)
            if address != witness.address or semantic_id != witness.semantic_id:
                raise ValueError("compiled machine journal provenance mismatch")
            if digest_prefix != int(witness.encoded_digest[:16], 16):
                raise ValueError("compiled machine journal instruction digest mismatch")
            values = struct.unpack_from(
                f"<{STATE_QWORD_COUNT}Q", encoded, base + JOURNAL_STATE_OFFSET,
            )
            if values[REGISTER_COUNT] not in witness.possible_next_addresses:
                raise ValueError("compiled machine journal selected an impossible successor")
            (
                effect_kind, effect_address, effect_width,
                before_low, before_high, after_low, after_high,
            ) = struct.unpack_from(
                "<7Q", encoded, base + JOURNAL_EFFECT_OFFSET,
            )
            before = before_low | (before_high << 64)
            after = after_low | (after_high << 64)
            stack_kind, stack_value, expected_depth = struct.unpack_from(
                "<3Q", encoded, base + JOURNAL_STACK_OFFSET,
            )
            if (stack_kind, stack_value, expected_depth) != witness.expected_stack_effect:
                raise ValueError("compiled machine journal call-stack witness mismatch")
            memory = active_state.memory
            if effect_kind:
                if effect_kind not in (1, 2) or effect_width not in (8, 16, 32, 64, 128):
                    raise ValueError("compiled machine journal has invalid memory effect")
                if effect_width < 128 and (before_high or after_high):
                    raise ValueError("compiled scalar memory effect has nonzero high halves")
                observed = memory.read_unsigned(effect_address, effect_width)
                if observed != before:
                    raise ValueError("compiled machine journal memory-read witness mismatch")
                if effect_kind == 1 and after != before:
                    raise ValueError("compiled machine journal read changed guest memory")
                if effect_kind == 2:
                    memory = memory.write_unsigned(effect_address, effect_width, after)
            call_stack = active_state.call_stack
            if stack_kind:
                if len(call_stack) != expected_depth:
                    raise ValueError("compiled machine journal call-stack depth mismatch")
                if stack_kind == 1:
                    call_stack = (*call_stack, stack_value)
                elif stack_kind == 2:
                    if not call_stack or int(call_stack[-1]) != stack_value:
                        raise ValueError("compiled machine journal return-stack mismatch")
                    call_stack = call_stack[:-1]
                else:
                    raise ValueError("compiled machine journal has invalid call-stack effect")
            active_state = replace(
                active_state,
                registers=tuple(values[:REGISTER_COUNT]),
                pc=values[REGISTER_COUNT],
                flags=values[REGISTER_COUNT + 1],
                steps=values[REGISTER_COUNT + 2],
                vector_registers=tuple(
                    values[REGISTER_COUNT + 3 + index * 2]
                    | (values[REGISTER_COUNT + 4 + index * 2] << 64)
                    for index in range(VECTOR_REGISTER_COUNT)
                ),
                memory=memory,
                call_stack=call_stack,
            )
            states.append(active_state)
        return tuple(states)


_MOV_SEMANTICS = {
    MachineSemanticToken.REGISTER_WRITE_IMMEDIATE,
    MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
    MachineSemanticToken.REGISTER_OR_MEMORY_READ,
}
_ARITHMETIC_SEMANTICS = {
    MachineSemanticToken.INTEGER_ADD: "add",
    MachineSemanticToken.INTEGER_SUBTRACT: "sub",
    MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW: "sbb",
    MachineSemanticToken.BITWISE_AND: "and",
    MachineSemanticToken.BITWISE_OR: "or",
    MachineSemanticToken.BITWISE_XOR: "xor",
    MachineSemanticToken.INTEGER_COMPARE: "sub",
    MachineSemanticToken.INTEGER_TEST: "and",
}
_FLAG_ONLY_ARITHMETIC = {
    MachineSemanticToken.INTEGER_COMPARE,
    MachineSemanticToken.INTEGER_TEST,
}
_EXTEND_SEMANTICS = {
    MachineSemanticToken.SIGN_EXTEND,
    MachineSemanticToken.ZERO_EXTEND,
}
_VECTOR_SEMANTICS = {
    MachineSemanticToken.VECTOR_MOVE,
    MachineSemanticToken.VECTOR_XOR,
}
_UNARY_SEMANTICS = {
    MachineSemanticToken.INTEGER_NEGATE,
    MachineSemanticToken.BITWISE_NOT,
}
_SHIFT_SEMANTICS = {
    MachineSemanticToken.SHIFT_LEFT,
    MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC,
    MachineSemanticToken.ROTATE_LEFT,
}
_INCDEC_SEMANTICS = {
    MachineSemanticToken.INTEGER_INCREMENT,
    MachineSemanticToken.INTEGER_DECREMENT,
}
_SCALAR_MISC_SEMANTICS = {
    *_UNARY_SEMANTICS,
    *_SHIFT_SEMANTICS,
    *_INCDEC_SEMANTICS,
    MachineSemanticToken.CONDITIONAL_SET,
    MachineSemanticToken.CONDITIONAL_MOVE,
    MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED,
    MachineSemanticToken.EXCHANGE,
}


def _condition_name(instruction: Any) -> str:
    name = str(instruction.token.name).split("_", 1)[0]
    for prefix in ("CMOV", "SET", "J"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _relative_target(instruction: Any) -> int | None:
    for operand in getattr(instruction, "operands", ()):
        if isinstance(operand, RelativeAddressOperand):
            return int(operand.target_address)
    return None


def _emit_flag_truth(builder: CodeBuilder, bit: int, *, inverted: bool = False) -> None:
    _address(builder, 0, FLAGS_OFFSET)
    builder.i64_load().i64_const(1 << bit).raw(OP_I64_AND, OP_I64_EQZ)
    if not inverted:
        builder.raw(OP_I32_EQZ)


def _emit_condition(builder: CodeBuilder, condition: str) -> str | None:
    aliases = {
        "Z": "E", "NZ": "NE", "C": "B", "NAE": "B",
        "NB": "AE", "NC": "AE", "NA": "BE", "NBE": "A",
        "NGE": "L", "NL": "GE", "NG": "LE", "NLE": "G",
    }
    condition = aliases.get(condition, condition)
    if condition == "E":
        _emit_flag_truth(builder, ZF)
    elif condition == "NE":
        _emit_flag_truth(builder, ZF, inverted=True)
    elif condition == "B":
        _emit_flag_truth(builder, CF)
    elif condition == "AE":
        _emit_flag_truth(builder, CF, inverted=True)
    elif condition == "BE":
        _emit_flag_truth(builder, CF)
        _emit_flag_truth(builder, ZF)
        builder.raw(OP_I32_OR)
    elif condition == "A":
        _emit_flag_truth(builder, CF, inverted=True)
        _emit_flag_truth(builder, ZF, inverted=True)
        builder.raw(OP_I32_AND)
    elif condition == "S":
        _emit_flag_truth(builder, SF)
    elif condition == "NS":
        _emit_flag_truth(builder, SF, inverted=True)
    elif condition == "O":
        _emit_flag_truth(builder, OF)
    elif condition == "NO":
        _emit_flag_truth(builder, OF, inverted=True)
    elif condition in {"L", "GE", "LE", "G"}:
        _emit_flag_truth(builder, SF)
        _emit_flag_truth(builder, OF)
        builder.raw(OP_I32_XOR)
        if condition in {"GE", "G"}:
            builder.raw(OP_I32_EQZ)
        if condition in {"LE", "G"}:
            _emit_flag_truth(builder, ZF, inverted=(condition == "G"))
            builder.raw(OP_I32_OR if condition == "LE" else OP_I32_AND)
    else:
        return f"unsupported AMD64 conditional jump {condition}"
    return None


def _control_targets(instruction: Any) -> tuple[int, ...] | None:
    target = _relative_target(instruction)
    if target is None:
        return None
    if instruction.semantic is MachineSemanticToken.DIRECT_RELATIVE_JUMP:
        return (target,)
    if instruction.semantic is MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP:
        fallthrough = int(instruction.address) + len(instruction.encoded)
        return (target, fallthrough) if target != fallthrough else (target,)
    return None


def _emit_control(builder: CodeBuilder, instruction: Any) -> str | None:
    targets = _control_targets(instruction)
    if targets is None:
        return "relative control instruction has no decoded target"
    _address(builder, 0, PC_OFFSET)
    if instruction.semantic is MachineSemanticToken.DIRECT_RELATIVE_JUMP:
        builder.i64_const(_signed_i64(targets[0]))
    else:
        builder.i64_const(_signed_i64(targets[0])).i64_const(
            _signed_i64(targets[-1]),
        )
        reason = _emit_condition(builder, _condition_name(instruction))
        if reason is not None:
            return reason
        builder.raw(OP_SELECT)
    builder.i64_store()
    return None


def _specialized_call_return(
    builder: CodeBuilder,
    instruction: Any,
    state: Any | None,
    *,
    guest_base: int,
    guest_size: int,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
    stack_kind_local: int,
    stack_value_local: int,
    stack_depth_local: int,
    resolved_indirect_target: int | None = None,
    indirect_external: bool = False,
) -> tuple[str | None, tuple[int, ...] | None]:
    if state is None:
        return "call/return Wasm lowering requires an exact specialization state", None
    depth = len(state.call_stack)
    if instruction.semantic in {
        MachineSemanticToken.DIRECT_RELATIVE_CALL,
        MachineSemanticToken.INDIRECT_CALL,
    }:
        if instruction.semantic is MachineSemanticToken.INDIRECT_CALL:
            if indirect_external:
                return "external indirect call remains at the capability boundary", None
            operands = tuple(getattr(instruction, "operands", ()))
            if not operands or not isinstance(operands[0], RegisterOperand):
                return "only register-resolved internal indirect calls are specialized", None
            target = resolved_indirect_target
        else:
            target = _relative_target(instruction)
        if target is None:
            return "call has no validated specialization target", None
        return_address = int(instruction.address) + len(instruction.encoded)
        stack_address = (int(state.registers[4]) - 8) & MASK64
        stack_kind, stack_value = 1, return_address
    elif instruction.semantic is MachineSemanticToken.RETURN:
        if not state.call_stack:
            return "outermost return remains in the interpreter lifecycle tier", None
        stack_value = int(state.call_stack[-1])
        if stack_value in {MASK64 - 1, MASK64 - 2} or state.termination_requested:
            return "loader/termination sentinel return remains in the interpreter", None
        target = stack_value
        stack_address = int(state.registers[4]) & MASK64
        stack_kind = 2
    else:
        return "instruction is not a specialized call/return", None
    if not guest_base <= stack_address or stack_address + 8 > guest_base + guest_size:
        return "specialized stack slot exceeds the compiled mirror window", None

    _guest_address(builder, stack_address, guest_base)
    builder.i64_load_width(64).local_set(effect_before_local)
    builder.local_get(effect_before_local).local_set(effect_after_local)
    builder.i64_const(2 if stack_kind == 1 else 1).local_set(effect_kind_local)
    builder.i64_const(_signed_i64(stack_address)).local_set(effect_address_local)
    builder.i64_const(64).local_set(effect_width_local)
    if stack_kind == 1:
        builder.i64_const(_signed_i64(return_address)).local_set(effect_after_local)
        _guest_address(builder, stack_address, guest_base)
        builder.local_get(effect_after_local).i64_store_width(64)

    _emit_scalar_store(
        builder, 0, 4 * 8,
        stack_address + (8 if stack_kind == 2 else 0),
    )
    _emit_scalar_store(builder, 0, PC_OFFSET, target)
    builder.i64_const(stack_kind).local_set(stack_kind_local)
    builder.i64_const(_signed_i64(stack_value)).local_set(stack_value_local)
    builder.i64_const(depth).local_set(stack_depth_local)
    return None, (int(target),)


def _specialized_indirect_jump(
    builder: CodeBuilder,
    instruction: Any,
    state: Any | None,
    resolved_target: int | None,
    *,
    indirect_external: bool,
) -> tuple[str | None, tuple[int, ...] | None]:
    if state is None or resolved_target is None:
        return "indirect jump requires an exact validated target state", None
    if indirect_external:
        return "external indirect jump remains at the capability boundary", None
    operands = tuple(getattr(instruction, "operands", ()))
    if not operands or not isinstance(operands[0], RegisterOperand):
        return "only register-resolved internal indirect jumps are specialized", None
    _emit_scalar_store(builder, 0, PC_OFFSET, int(resolved_target))
    return None, (int(resolved_target),)


def _specialized_stack_data(
    builder: CodeBuilder,
    instruction: Any,
    state: Any | None,
    *,
    guest_base: int,
    guest_size: int,
    result_local: int,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
) -> str | None:
    if state is None:
        return "stack operation requires an exact block-entry state"
    operands = tuple(getattr(instruction, "operands", ()))
    if not operands:
        return "stack operation has no decoded operand"
    rsp = int(state.registers[4]) & MASK64
    if instruction.semantic is MachineSemanticToken.STACK_PUSH:
        stack_address = (rsp - 8) & MASK64
        source = operands[0]
        if isinstance(source, RegisterOperand):
            _load_register(builder, 0, int(source.register))
        elif isinstance(source, ImmediateOperand):
            value = int(source.value) & ((1 << int(source.width)) - 1)
            if source.signed and value & (1 << (int(source.width) - 1)):
                value -= 1 << int(source.width)
            builder.i64_const(_signed_i64(value))
        else:
            return "compiled PUSH admits only register or immediate sources"
        builder.local_set(result_local)
        effect_kind = 2
    else:
        stack_address = rsp
        if not isinstance(operands[0], RegisterOperand):
            return "compiled POP admits only a register destination"
        effect_kind = 1
    if not guest_base <= stack_address or stack_address + 8 > guest_base + guest_size:
        return "specialized stack slot exceeds the compiled mirror window"
    _guest_address(builder, stack_address, guest_base)
    builder.i64_load_width(64).local_set(effect_before_local)
    builder.local_get(effect_before_local).local_set(effect_after_local)
    builder.i64_const(effect_kind).local_set(effect_kind_local)
    builder.i64_const(_signed_i64(stack_address)).local_set(effect_address_local)
    builder.i64_const(64).local_set(effect_width_local)
    if instruction.semantic is MachineSemanticToken.STACK_PUSH:
        builder.local_get(result_local).local_set(effect_after_local)
        _guest_address(builder, stack_address, guest_base)
        builder.local_get(result_local).i64_store_width(64)
        _emit_scalar_store(builder, 0, 4 * 8, stack_address)
    else:
        _address(builder, 0, int(operands[0].register) * 8)
        builder.local_get(effect_before_local).i64_store()
        _emit_scalar_store(builder, 0, 4 * 8, (stack_address + 8) & MASK64)
    return None


def _signed_i64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def _address(builder: CodeBuilder, parameter: int, offset: int) -> None:
    builder.local_get(parameter).i32_const(int(offset)).raw(OP_I32_ADD)


def _load_register(builder: CodeBuilder, state_parameter: int, register: int) -> None:
    _address(builder, state_parameter, int(register) * 8)
    builder.i64_load()


def _static_memory_address(instruction: Any, operand: Any) -> int | None:
    if not isinstance(operand, EffectiveAddressOperand):
        return None
    if operand.base is not None or operand.index is not None:
        return None
    return int(operand.displacement) + (
        int(instruction.address) + len(instruction.encoded)
        if operand.rip_relative else 0
    )


def _operand_data_width(instruction: Any, operand_index: int) -> int:
    operand = instruction.operands[operand_index]
    if isinstance(operand, (RegisterOperand, VectorRegisterOperand)):
        return int(operand.width)
    name = str(getattr(getattr(instruction, "token", None), "name", ""))
    if isinstance(operand, EffectiveAddressOperand):
        widths = re.findall(r"(?:XMMM|RM|M)(128|64|32|16|8)(?:_|$)", name)
        if widths:
            return int(widths[-1])
    if isinstance(operand, ImmediateOperand):
        return int(operand.width)
    widths = [
        int(getattr(item, "width", 0))
        for item in getattr(instruction, "operands", ())
        if int(getattr(item, "width", 0)) in (8, 16, 32, 64, 128)
    ]
    return max(widths, default=0)


def _specialized_memory_address(
    instruction: Any, operand: Any, state: Any | None,
) -> int | None:
    static = _static_memory_address(instruction, operand)
    if static is not None or not isinstance(operand, EffectiveAddressOperand):
        return static
    if state is None:
        return None
    base = int(instruction.address) + len(instruction.encoded) if operand.rip_relative else 0
    prefixes = tuple(getattr(instruction, "legacy_prefixes", ()))
    if 0x64 in prefixes:
        base += int(state.fs_base)
    if 0x65 in prefixes:
        base += int(state.gs_base)
    if operand.base is not None:
        base += int(state.registers[int(operand.base)])
    if operand.index is not None:
        base += int(state.registers[int(operand.index)]) * int(operand.scale)
    return (base + int(operand.displacement)) & MASK64


def _dynamic_memory_guard(instruction: Any, state: Any) -> dict[str, Any]:
    registers = {
        int(register): int(state.registers[int(register)])
        for operand in getattr(instruction, "operands", ())
        if isinstance(operand, EffectiveAddressOperand)
        for register in (operand.base, operand.index)
        if register is not None
    }
    guard: dict[str, Any] = {"registers": tuple(sorted(registers.items()))}
    prefixes = tuple(getattr(instruction, "legacy_prefixes", ()))
    if 0x64 in prefixes:
        guard["fs_base"] = int(state.fs_base)
    if 0x65 in prefixes:
        guard["gs_base"] = int(state.gs_base)
    return guard


def _guest_address(builder: CodeBuilder, address: int, guest_base: int) -> None:
    _address(builder, 2, int(address) - int(guest_base))


def _set_effect_metadata(
    builder: CodeBuilder,
    *,
    kind: int,
    address: int,
    width: int,
    before_local: int,
    after_local: int,
    kind_local: int,
    address_local: int,
    width_local: int,
) -> None:
    builder.i64_const(kind).local_set(kind_local)
    builder.i64_const(_signed_i64(address)).local_set(address_local)
    builder.i64_const(width).local_set(width_local)
    if kind == 1:
        builder.local_get(before_local).local_set(after_local)


def _emit_move(
    builder: CodeBuilder,
    instruction: Any,
    *,
    guest_base: int,
    guest_size: int,
    executable_pages: frozenset[int],
    result_local: int,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
    specialization_state: Any | None,
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if len(operands) < 2 or not isinstance(
        operands[0], (RegisterOperand, EffectiveAddressOperand),
    ):
        return "MOV destination is not an admitted register or static guest address"
    destination, source = operands[:2]
    width = int(getattr(destination, "width", getattr(source, "width", 0)))
    if width not in (8, 16, 32, 64):
        return f"unsupported register write width {width}"
    if not isinstance(source, (RegisterOperand, ImmediateOperand, EffectiveAddressOperand)):
        return "MOV source is not an admitted register, immediate, or static guest address"
    if isinstance(destination, EffectiveAddressOperand) and isinstance(source, EffectiveAddressOperand):
        return "memory-to-memory MOV is not an AMD64 operation"

    source_address = _specialized_memory_address(
        instruction, source, specialization_state,
    )
    if isinstance(source, EffectiveAddressOperand):
        if source_address is None:
            return "dynamic guest-memory source requires a block-entry specialization state"
        if not guest_base <= source_address or source_address + width // 8 > guest_base + guest_size:
            return "guest-memory source exceeds the compiled mirror window"
        _guest_address(builder, source_address, guest_base)
        builder.i64_load_width(width).local_set(effect_before_local)
        _set_effect_metadata(
            builder, kind=1, address=source_address, width=width,
            before_local=effect_before_local, after_local=effect_after_local,
            kind_local=effect_kind_local, address_local=effect_address_local,
            width_local=effect_width_local,
        )
        builder.local_get(effect_before_local).local_set(result_local)
    elif isinstance(source, RegisterOperand):
        _load_register(builder, 0, int(source.register))
        if width < 64:
            builder.i64_const((1 << width) - 1).raw(OP_I64_AND)
        builder.local_set(result_local)
    else:
        builder.i64_const(
            _signed_i64(int(source.value) & ((1 << width) - 1)),
        ).local_set(result_local)

    destination_address = _specialized_memory_address(
        instruction, destination, specialization_state,
    )
    if isinstance(destination, EffectiveAddressOperand):
        if destination_address is None:
            return "dynamic guest-memory destination requires a block-entry specialization state"
        if not guest_base <= destination_address or destination_address + width // 8 > guest_base + guest_size:
            return "guest-memory destination exceeds the compiled mirror window"
        if destination_address // 4096 in executable_pages:
            return "guest-memory destination targets a translated executable page"
        _guest_address(builder, destination_address, guest_base)
        builder.i64_load_width(width).local_set(effect_before_local)
        builder.local_get(result_local).local_set(effect_after_local)
        _set_effect_metadata(
            builder, kind=2, address=destination_address, width=width,
            before_local=effect_before_local, after_local=effect_after_local,
            kind_local=effect_kind_local, address_local=effect_address_local,
            width_local=effect_width_local,
        )
        _guest_address(builder, destination_address, guest_base)
        builder.local_get(result_local).i64_store_width(width)
        return None

    destination_offset = int(destination.register) * 8
    _address(builder, 0, destination_offset)
    if width < 32:
        _load_register(builder, 0, int(destination.register))
        builder.i64_const(_signed_i64(MASK64 ^ ((1 << width) - 1))).raw(OP_I64_AND)
    builder.local_get(result_local)
    if width < 32:
        builder.raw(OP_I64_OR)
    builder.i64_store()
    return None


def _emit_operand(
    builder: CodeBuilder, operand: Any, width: int, *, state_parameter: int = 0,
) -> str | None:
    if isinstance(operand, RegisterOperand):
        _load_register(builder, state_parameter, int(operand.register))
    elif isinstance(operand, ImmediateOperand):
        value = int(operand.value) & ((1 << int(operand.width)) - 1)
        if operand.signed and value & (1 << (int(operand.width) - 1)):
            value -= 1 << int(operand.width)
        builder.i64_const(_signed_i64(value & ((1 << width) - 1)))
    else:
        return "only register or immediate arithmetic sources are admitted"
    if width < 64:
        builder.i64_const((1 << width) - 1).raw(OP_I64_AND)
    return None


def _emit_arithmetic_operand(
    builder: CodeBuilder,
    instruction: Any,
    operand_index: int,
    width: int,
    *,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
) -> str | None:
    operand = instruction.operands[operand_index]
    if not isinstance(operand, EffectiveAddressOperand):
        return _emit_operand(builder, operand, width)
    address = _specialized_memory_address(
        instruction, operand, specialization_state,
    )
    if address is None:
        return "dynamic arithmetic memory operand requires block-entry specialization"
    if not guest_base <= address or address + width // 8 > guest_base + guest_size:
        return "arithmetic memory operand exceeds the compiled mirror window"
    _guest_address(builder, address, guest_base)
    builder.i64_load_width(width).local_set(effect_before_local)
    _set_effect_metadata(
        builder, kind=1, address=address, width=width,
        before_local=effect_before_local, after_local=effect_after_local,
        kind_local=effect_kind_local, address_local=effect_address_local,
        width_local=effect_width_local,
    )
    builder.local_get(effect_before_local)
    return None


def _emit_effective_address(
    builder: CodeBuilder,
    instruction: Any,
    *,
    specialization_state: Any | None,
    result_local: int,
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if (
        len(operands) < 2
        or not isinstance(operands[0], RegisterOperand)
        or not isinstance(operands[1], EffectiveAddressOperand)
    ):
        return "LEA requires a register destination and effective-address source"
    destination, source = operands[:2]
    width = int(destination.width)
    if width not in (16, 32, 64):
        return f"unsupported LEA destination width {width}"
    base = int(instruction.address) + len(instruction.encoded) if source.rip_relative else 0
    prefixes = tuple(getattr(instruction, "legacy_prefixes", ()))
    if 0x64 in prefixes or 0x65 in prefixes:
        if specialization_state is None:
            return "segmented LEA requires an exact specialization state"
        if 0x64 in prefixes:
            base += int(specialization_state.fs_base)
        if 0x65 in prefixes:
            base += int(specialization_state.gs_base)
    builder.i64_const(_signed_i64(base))
    if source.base is not None:
        _load_register(builder, 0, int(source.base))
        builder.raw(OP_I64_ADD)
    if source.index is not None:
        _load_register(builder, 0, int(source.index))
        builder.i64_const(int(source.scale)).raw(OP_I64_MUL, OP_I64_ADD)
    builder.i64_const(_signed_i64(int(source.displacement))).raw(OP_I64_ADD)
    if width < 64:
        builder.i64_const((1 << width) - 1).raw(OP_I64_AND)
    destination_offset = int(destination.register) * 8
    if width == 16:
        builder.local_set(result_local)
        _address(builder, 0, destination_offset)
        _load_register(builder, 0, int(destination.register))
        builder.i64_const(_signed_i64(MASK64 ^ 0xFFFF)).raw(OP_I64_AND)
        builder.local_get(result_local).raw(OP_I64_OR).i64_store()
    else:
        builder.local_set(result_local)
        _address(builder, 0, destination_offset)
        builder.local_get(result_local).i64_store()
    return None


def _emit_extend(
    builder: CodeBuilder,
    instruction: Any,
    *,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    result_local: int,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if len(operands) < 2 or not isinstance(operands[0], RegisterOperand):
        return "extension requires a register destination"
    destination, source = operands[:2]
    if not isinstance(source, (RegisterOperand, EffectiveAddressOperand)):
        return "extension source must be a register or guest-memory operand"
    source_width = _operand_data_width(instruction, 1)
    target_width = int(destination.width)
    if source_width not in (8, 16, 32) or target_width not in (16, 32, 64):
        return "extension widths are outside the admitted scalar tier"
    reason = _emit_arithmetic_operand(
        builder, instruction, 1, source_width,
        guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_before_local=effect_before_local,
        effect_after_local=effect_after_local,
    )
    if reason is not None:
        return reason
    if instruction.semantic is MachineSemanticToken.SIGN_EXTEND:
        sign = 1 << (source_width - 1)
        builder.i64_const(sign).raw(OP_I64_XOR).i64_const(sign).raw(OP_I64_SUB)
    if target_width < 64:
        builder.i64_const((1 << target_width) - 1).raw(OP_I64_AND)
    builder.local_set(result_local)
    _address(builder, 0, int(destination.register) * 8)
    if target_width == 16:
        _load_register(builder, 0, int(destination.register))
        builder.i64_const(_signed_i64(MASK64 ^ 0xFFFF)).raw(OP_I64_AND)
        builder.local_get(result_local).raw(OP_I64_OR)
    else:
        builder.local_get(result_local)
    builder.i64_store()
    return None


def _emit_bool_flag(
    builder: CodeBuilder, flags_local: int, bit: int, emit_condition,
) -> None:
    builder.local_get(flags_local)
    emit_condition()
    builder.raw(OP_I64_EXTEND_I32_U).i64_const(1 << bit).raw(
        OP_I64_MUL, OP_I64_OR,
    ).local_set(flags_local)


def _emit_arithmetic(
    builder: CodeBuilder,
    instruction: Any,
    *,
    left_local: int,
    right_local: int,
    result_local: int,
    flags_local: int,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
    executable_pages: frozenset[int],
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    flag_only = instruction.semantic in _FLAG_ONLY_ARITHMETIC
    if len(operands) < 2 or not isinstance(
        operands[0], (RegisterOperand, EffectiveAddressOperand),
    ):
        return "arithmetic destination is outside the admitted register/memory tier"
    if sum(isinstance(item, EffectiveAddressOperand) for item in operands[:2]) > 1:
        return "arithmetic cannot consume two guest-memory operands"
    destination, source = operands[:2]
    width = _operand_data_width(instruction, 0)
    if width not in (8, 16, 32, 64):
        return f"unsupported arithmetic width {width}"
    reason = _emit_arithmetic_operand(
        builder, instruction, 0, width,
        guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_before_local=effect_before_local,
        effect_after_local=effect_after_local,
    )
    if reason is not None:
        return reason
    builder.local_set(left_local)
    reason = _emit_arithmetic_operand(
        builder, instruction, 1, width,
        guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_before_local=effect_before_local,
        effect_after_local=effect_after_local,
    )
    if reason is not None:
        return reason
    builder.local_set(right_local)
    kind = _ARITHMETIC_SEMANTICS[instruction.semantic]
    opcode = {
        "add": OP_I64_ADD, "sub": OP_I64_SUB,
        "sbb": OP_I64_SUB,
        "and": OP_I64_AND, "or": OP_I64_OR, "xor": OP_I64_XOR,
    }[kind]
    if kind == "sbb":
        builder.local_get(right_local)
        _emit_flag_truth(builder, CF)
        builder.raw(OP_I64_EXTEND_I32_U, OP_I64_ADD)
        if width < 64:
            builder.i64_const((1 << width) - 1).raw(OP_I64_AND)
        builder.local_set(right_local)
    builder.local_get(left_local).local_get(right_local).raw(opcode)
    if width < 64:
        builder.i64_const((1 << width) - 1).raw(OP_I64_AND)
    builder.local_set(result_local)

    if not flag_only and isinstance(destination, EffectiveAddressOperand):
        address = _specialized_memory_address(
            instruction, destination, specialization_state,
        )
        if address is None:
            return "dynamic arithmetic destination requires block-entry specialization"
        if address // 4096 in executable_pages:
            return "arithmetic destination targets a translated executable page"
        builder.i64_const(2).local_set(effect_kind_local)
        builder.local_get(result_local).local_set(effect_after_local)
        _guest_address(builder, address, guest_base)
        builder.local_get(result_local).i64_store_width(width)
    elif not flag_only:
        # Write the result with AMD64's 32-bit zero-extension and 8/16-bit
        # upper-register preservation rules.
        _address(builder, 0, int(destination.register) * 8)
        if width < 32:
            _load_register(builder, 0, int(destination.register))
            builder.i64_const(_signed_i64(MASK64 ^ ((1 << width) - 1))).raw(OP_I64_AND)
            builder.local_get(result_local).raw(OP_I64_OR)
        else:
            builder.local_get(result_local)
        builder.i64_store()

    logical = kind in {"and", "or", "xor"}
    cleared = LOGICAL_FLAGS if logical else ARITHMETIC_FLAGS
    _address(builder, 0, FLAGS_OFFSET)
    builder.i64_load().i64_const(_signed_i64(MASK64 ^ cleared)).raw(
        OP_I64_AND,
    ).local_set(flags_local)

    if not logical:
        def carry_condition():
            if kind == "add":
                builder.local_get(result_local).local_get(left_local).raw(OP_I64_LT_U)
            elif kind == "sbb":
                builder.local_get(left_local).local_get(right_local).raw(OP_I64_LT_U)
                builder.local_get(right_local).raw(OP_I64_EQZ)
                _emit_flag_truth(builder, CF)
                builder.raw(OP_I32_AND, OP_I32_OR)
            else:
                builder.local_get(left_local).local_get(right_local).raw(OP_I64_LT_U)

        _emit_bool_flag(builder, flags_local, CF, carry_condition)

        def auxiliary_condition():
            builder.local_get(left_local).local_get(right_local).raw(OP_I64_XOR)
            builder.local_get(result_local).raw(OP_I64_XOR)
            builder.i64_const(0x10).raw(OP_I64_AND, OP_I64_EQZ, OP_I32_EQZ)

        _emit_bool_flag(builder, flags_local, AF, auxiliary_condition)

        def overflow_condition():
            builder.local_get(left_local).local_get(right_local).raw(OP_I64_XOR)
            if kind == "add":
                builder.i64_const(-1).raw(OP_I64_XOR)
            builder.local_get(left_local).local_get(result_local).raw(OP_I64_XOR)
            builder.raw(OP_I64_AND).i64_const(_signed_i64(1 << (width - 1))).raw(
                OP_I64_AND, OP_I64_EQZ, OP_I32_EQZ,
            )

        _emit_bool_flag(builder, flags_local, OF, overflow_condition)

    def parity_condition():
        builder.local_get(result_local).i64_const(0xFF).raw(
            OP_I64_AND, OP_I64_POPCNT,
        ).i64_const(1).raw(OP_I64_AND, OP_I64_EQZ)

    _emit_bool_flag(builder, flags_local, PF, parity_condition)

    def zero_condition():
        builder.local_get(result_local).raw(OP_I64_EQZ)

    _emit_bool_flag(builder, flags_local, ZF, zero_condition)

    def sign_condition():
        builder.local_get(result_local).i64_const(_signed_i64(1 << (width - 1))).raw(
            OP_I64_AND, OP_I64_EQZ, OP_I32_EQZ,
        )

    _emit_bool_flag(builder, flags_local, SF, sign_condition)
    _address(builder, 0, FLAGS_OFFSET)
    builder.local_get(flags_local).i64_store()
    return None


def _emit_result_destination(
    builder: CodeBuilder,
    instruction: Any,
    destination: Any,
    width: int,
    *,
    result_local: int,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_after_local: int,
    executable_pages: frozenset[int],
) -> str | None:
    """Commit one scalar result while preserving the exact memory witness."""

    if isinstance(destination, EffectiveAddressOperand):
        address = _specialized_memory_address(
            instruction, destination, specialization_state,
        )
        if address is None:
            return "dynamic scalar destination requires block-entry specialization"
        if not guest_base <= address or address + width // 8 > guest_base + guest_size:
            return "scalar destination exceeds the compiled mirror window"
        if address // 4096 in executable_pages:
            return "scalar destination targets a translated executable page"
        builder.i64_const(2).local_set(effect_kind_local)
        builder.i64_const(_signed_i64(address)).local_set(effect_address_local)
        builder.i64_const(width).local_set(effect_width_local)
        builder.local_get(result_local).local_set(effect_after_local)
        _guest_address(builder, address, guest_base)
        builder.local_get(result_local).i64_store_width(width)
        return None
    if not isinstance(destination, RegisterOperand):
        return "scalar destination must be a register or guest-memory operand"
    _address(builder, 0, int(destination.register) * 8)
    if width < 32:
        _load_register(builder, 0, int(destination.register))
        builder.i64_const(_signed_i64(MASK64 ^ ((1 << width) - 1))).raw(OP_I64_AND)
        builder.local_get(result_local).raw(OP_I64_OR)
    else:
        builder.local_get(result_local)
    builder.i64_store()
    return None


def _emit_result_szp_flags(
    builder: CodeBuilder, *, result_local: int, flags_local: int, width: int,
) -> None:
    def parity_condition():
        builder.local_get(result_local).i64_const(0xFF).raw(
            OP_I64_AND, OP_I64_POPCNT,
        ).i64_const(1).raw(OP_I64_AND, OP_I64_EQZ)

    _emit_bool_flag(builder, flags_local, PF, parity_condition)

    def zero_condition():
        builder.local_get(result_local).raw(OP_I64_EQZ)

    _emit_bool_flag(builder, flags_local, ZF, zero_condition)

    def sign_condition():
        builder.local_get(result_local).i64_const(_signed_i64(1 << (width - 1))).raw(
            OP_I64_AND, OP_I64_EQZ, OP_I32_EQZ,
        )

    _emit_bool_flag(builder, flags_local, SF, sign_condition)


def _emit_unary(
    builder: CodeBuilder,
    instruction: Any,
    *,
    left_local: int,
    result_local: int,
    flags_local: int,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
    executable_pages: frozenset[int],
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if not operands:
        return "unary scalar instruction has no destination"
    destination = operands[0]
    width = _operand_data_width(instruction, 0)
    if width not in (8, 16, 32, 64):
        return f"unsupported unary width {width}"
    reason = _emit_arithmetic_operand(
        builder, instruction, 0, width,
        guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_before_local=effect_before_local,
        effect_after_local=effect_after_local,
    )
    if reason is not None:
        return reason
    builder.local_set(left_local)
    if instruction.semantic is MachineSemanticToken.BITWISE_NOT:
        builder.local_get(left_local).i64_const(-1).raw(OP_I64_XOR)
    else:
        builder.i64_const(0).local_get(left_local).raw(OP_I64_SUB)
    if width < 64:
        builder.i64_const((1 << width) - 1).raw(OP_I64_AND)
    builder.local_set(result_local)
    reason = _emit_result_destination(
        builder, instruction, destination, width,
        result_local=result_local, guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_after_local=effect_after_local,
        executable_pages=executable_pages,
    )
    if reason is not None or instruction.semantic is MachineSemanticToken.BITWISE_NOT:
        return reason

    _address(builder, 0, FLAGS_OFFSET)
    builder.i64_load().i64_const(_signed_i64(MASK64 ^ ARITHMETIC_FLAGS)).raw(
        OP_I64_AND,
    ).local_set(flags_local)

    def carry_condition():
        builder.local_get(left_local).raw(OP_I64_EQZ, OP_I32_EQZ)

    _emit_bool_flag(builder, flags_local, CF, carry_condition)

    def auxiliary_condition():
        builder.local_get(left_local).local_get(result_local).raw(
            OP_I64_XOR,
        ).i64_const(0x10).raw(OP_I64_AND, OP_I64_EQZ, OP_I32_EQZ)

    _emit_bool_flag(builder, flags_local, AF, auxiliary_condition)

    def overflow_condition():
        builder.local_get(left_local).local_get(result_local).raw(
            OP_I64_AND,
        ).i64_const(_signed_i64(1 << (width - 1))).raw(
            OP_I64_AND, OP_I64_EQZ, OP_I32_EQZ,
        )

    _emit_bool_flag(builder, flags_local, OF, overflow_condition)
    _emit_result_szp_flags(
        builder, result_local=result_local, flags_local=flags_local, width=width,
    )
    _address(builder, 0, FLAGS_OFFSET)
    builder.local_get(flags_local).i64_store()
    return None


def _emit_shift_rotate(
    builder: CodeBuilder,
    instruction: Any,
    *,
    left_local: int,
    result_local: int,
    flags_local: int,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
    executable_pages: frozenset[int],
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if len(operands) < 2 or not isinstance(operands[1], ImmediateOperand):
        return "shift/rotate tier currently requires an immediate count"
    destination = operands[0]
    width = _operand_data_width(instruction, 0)
    if width not in (8, 16, 32, 64):
        return f"unsupported shift/rotate width {width}"
    count = int(operands[1].value) & (0x3F if width == 64 else 0x1F)
    if count == 0:
        return None
    reason = _emit_arithmetic_operand(
        builder, instruction, 0, width,
        guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_before_local=effect_before_local,
        effect_after_local=effect_after_local,
    )
    if reason is not None:
        return reason
    builder.local_set(left_local)
    opcode = {
        MachineSemanticToken.SHIFT_LEFT: OP_I64_SHL,
        MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC: OP_I64_SHR_S,
        MachineSemanticToken.ROTATE_LEFT: OP_I64_ROTL,
    }[instruction.semantic]
    if instruction.semantic is MachineSemanticToken.ROTATE_LEFT and width != 64:
        return "compiled rotate currently requires a 64-bit destination"
    if instruction.semantic is MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC and width != 64:
        return "compiled arithmetic right shift currently requires a 64-bit destination"
    builder.local_get(left_local).i64_const(count).raw(opcode)
    if width < 64:
        builder.i64_const((1 << width) - 1).raw(OP_I64_AND)
    builder.local_set(result_local)
    reason = _emit_result_destination(
        builder, instruction, destination, width,
        result_local=result_local, guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_after_local=effect_after_local,
        executable_pages=executable_pages,
    )
    if reason is not None:
        return reason
    rotate = instruction.semantic is MachineSemanticToken.ROTATE_LEFT
    cleared = (1 << CF) | ((1 << OF) if rotate and count == 1 else 0)
    if not rotate:
        cleared = LOGICAL_FLAGS
    _address(builder, 0, FLAGS_OFFSET)
    builder.i64_load().i64_const(_signed_i64(MASK64 ^ cleared)).raw(
        OP_I64_AND,
    ).local_set(flags_local)

    def carry_condition():
        if rotate:
            builder.local_get(result_local)
            bit = 1
        elif instruction.semantic is MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC:
            builder.local_get(left_local)
            bit = 1 << (count - 1) if count <= width else 1 << (width - 1)
        else:
            builder.local_get(left_local)
            bit = 1 << (width - count) if count <= width else 0
        builder.i64_const(_signed_i64(bit)).raw(
            OP_I64_AND, OP_I64_EQZ, OP_I32_EQZ,
        )

    _emit_bool_flag(builder, flags_local, CF, carry_condition)
    if count == 1:
        def overflow_condition():
            if instruction.semantic is MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC:
                builder.i32_const(0)
            else:
                builder.local_get(result_local).i64_const(
                    _signed_i64(1 << (width - 1)),
                ).raw(OP_I64_AND, OP_I64_EQZ, OP_I32_EQZ)
                carry_condition()
                builder.raw(OP_I32_XOR)

        _emit_bool_flag(builder, flags_local, OF, overflow_condition)
    else:
        def old_overflow_condition():
            _emit_flag_truth(builder, OF)

        _emit_bool_flag(builder, flags_local, OF, old_overflow_condition)
    if not rotate:
        _emit_result_szp_flags(
            builder, result_local=result_local, flags_local=flags_local, width=width,
        )
    _address(builder, 0, FLAGS_OFFSET)
    builder.local_get(flags_local).i64_store()
    return None


def _emit_increment_decrement(
    builder: CodeBuilder,
    instruction: Any,
    *,
    left_local: int,
    result_local: int,
    flags_local: int,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
    executable_pages: frozenset[int],
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if not operands:
        return "increment/decrement has no destination"
    destination = operands[0]
    width = _operand_data_width(instruction, 0)
    if width not in (8, 16, 32, 64):
        return f"unsupported increment/decrement width {width}"
    reason = _emit_arithmetic_operand(
        builder, instruction, 0, width,
        guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_before_local=effect_before_local,
        effect_after_local=effect_after_local,
    )
    if reason is not None:
        return reason
    builder.local_set(left_local)
    builder.local_get(left_local).i64_const(1).raw(
        OP_I64_ADD
        if instruction.semantic is MachineSemanticToken.INTEGER_INCREMENT
        else OP_I64_SUB,
    )
    if width < 64:
        builder.i64_const((1 << width) - 1).raw(OP_I64_AND)
    builder.local_set(result_local)
    reason = _emit_result_destination(
        builder, instruction, destination, width,
        result_local=result_local, guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_after_local=effect_after_local,
        executable_pages=executable_pages,
    )
    if reason is not None:
        return reason
    cleared = ARITHMETIC_FLAGS ^ (1 << CF)
    _address(builder, 0, FLAGS_OFFSET)
    builder.i64_load().i64_const(_signed_i64(MASK64 ^ cleared)).raw(
        OP_I64_AND,
    ).local_set(flags_local)

    def auxiliary_condition():
        builder.local_get(left_local).i64_const(1).raw(OP_I64_XOR)
        builder.local_get(result_local).raw(OP_I64_XOR)
        builder.i64_const(0x10).raw(OP_I64_AND, OP_I64_EQZ, OP_I32_EQZ)

    _emit_bool_flag(builder, flags_local, AF, auxiliary_condition)

    def overflow_condition():
        builder.local_get(left_local).i64_const(1).raw(OP_I64_XOR)
        if instruction.semantic is MachineSemanticToken.INTEGER_INCREMENT:
            builder.i64_const(-1).raw(OP_I64_XOR)
        builder.local_get(left_local).local_get(result_local).raw(OP_I64_XOR)
        builder.raw(OP_I64_AND).i64_const(_signed_i64(1 << (width - 1))).raw(
            OP_I64_AND, OP_I64_EQZ, OP_I32_EQZ,
        )

    _emit_bool_flag(builder, flags_local, OF, overflow_condition)
    _emit_result_szp_flags(
        builder, result_local=result_local, flags_local=flags_local, width=width,
    )
    _address(builder, 0, FLAGS_OFFSET)
    builder.local_get(flags_local).i64_store()
    return None


def _emit_conditional_set(
    builder: CodeBuilder,
    instruction: Any,
    *,
    result_local: int,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
    executable_pages: frozenset[int],
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if not operands:
        return "conditional set has no destination"
    destination = operands[0]
    if isinstance(destination, EffectiveAddressOperand):
        reason = _emit_arithmetic_operand(
            builder, instruction, 0, 8,
            guest_base=guest_base, guest_size=guest_size,
            specialization_state=specialization_state,
            effect_kind_local=effect_kind_local,
            effect_address_local=effect_address_local,
            effect_width_local=effect_width_local,
            effect_before_local=effect_before_local,
            effect_after_local=effect_after_local,
        )
        if reason is not None:
            return reason
        builder.local_set(result_local)  # discard the witnessed prior byte
    reason = _emit_condition(builder, _condition_name(instruction))
    if reason is not None:
        return reason
    builder.raw(OP_I64_EXTEND_I32_U).local_set(result_local)
    return _emit_result_destination(
        builder, instruction, destination, 8,
        result_local=result_local, guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_after_local=effect_after_local,
        executable_pages=executable_pages,
    )


def _emit_conditional_move(
    builder: CodeBuilder,
    instruction: Any,
    *,
    left_local: int,
    right_local: int,
    result_local: int,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if (
        len(operands) < 2
        or not isinstance(operands[0], RegisterOperand)
        or not isinstance(operands[1], (RegisterOperand, EffectiveAddressOperand))
    ):
        return "conditional move requires a register destination and scalar source"
    destination = operands[0]
    width = _operand_data_width(instruction, 0)
    if width not in (16, 32, 64):
        return f"unsupported conditional-move width {width}"
    _load_register(builder, 0, int(destination.register))
    if width == 16:
        builder.i64_const((1 << width) - 1).raw(OP_I64_AND)
    builder.local_set(left_local)
    reason = _emit_arithmetic_operand(
        builder, instruction, 1, width,
        guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_before_local=effect_before_local,
        effect_after_local=effect_after_local,
    )
    if reason is not None:
        return reason
    builder.local_set(right_local)
    builder.local_get(right_local).local_get(left_local)
    reason = _emit_condition(builder, _condition_name(instruction))
    if reason is not None:
        return reason
    builder.raw(OP_SELECT).local_set(result_local)
    _address(builder, 0, int(destination.register) * 8)
    if width == 16:
        _load_register(builder, 0, int(destination.register))
        builder.i64_const(_signed_i64(MASK64 ^ 0xFFFF)).raw(OP_I64_AND)
        builder.local_get(result_local).raw(OP_I64_OR)
    else:
        # A 32-bit CMOV destination follows the normal long-mode zero-extension
        # rule when written; the false path writes its already-truncated value.
        builder.local_get(result_local)
    builder.i64_store()
    return None


def _emit_exchange(
    builder: CodeBuilder,
    instruction: Any,
    *,
    left_local: int,
    right_local: int,
    result_local: int,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
    executable_pages: frozenset[int],
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if len(operands) < 2:
        return "exchange requires two operands"
    left, right = operands[:2]
    width = _operand_data_width(instruction, 0)
    if width != 64 or not isinstance(right, RegisterOperand):
        return "compiled exchange admits 64-bit register or memory with a register source"
    if isinstance(left, RegisterOperand):
        _load_register(builder, 0, int(left.register))
        builder.local_set(left_local)
        _load_register(builder, 0, int(right.register))
        builder.local_set(right_local)
        _address(builder, 0, int(left.register) * 8)
        builder.local_get(right_local).i64_store()
        _address(builder, 0, int(right.register) * 8)
        builder.local_get(left_local).i64_store()
        return None
    if not isinstance(left, EffectiveAddressOperand):
        return "compiled exchange destination is outside the scalar tier"
    reason = _emit_arithmetic_operand(
        builder, instruction, 0, 64,
        guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_before_local=effect_before_local,
        effect_after_local=effect_after_local,
    )
    if reason is not None:
        return reason
    builder.local_set(left_local)
    _load_register(builder, 0, int(right.register))
    builder.local_set(result_local)
    reason = _emit_result_destination(
        builder, instruction, left, 64,
        result_local=result_local, guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_after_local=effect_after_local,
        executable_pages=executable_pages,
    )
    if reason is not None:
        return reason
    _address(builder, 0, int(right.register) * 8)
    builder.local_get(left_local).i64_store()
    return None


def _emit_multiply_unsigned(
    builder: CodeBuilder,
    instruction: Any,
    *,
    left_local: int,
    right_local: int,
    result_local: int,
    high_local: int,
    work_local: int,
    flags_local: int,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_after_local: int,
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if len(operands) != 1:
        return "unsigned multiply requires one explicit source operand"
    if _operand_data_width(instruction, 0) != 64:
        return "compiled unsigned multiply currently requires MUL r/m64"
    _load_register(builder, 0, 0)
    builder.local_set(left_local)
    reason = _emit_arithmetic_operand(
        builder, instruction, 0, 64,
        guest_base=guest_base, guest_size=guest_size,
        specialization_state=specialization_state,
        effect_kind_local=effect_kind_local,
        effect_address_local=effect_address_local,
        effect_width_local=effect_width_local,
        effect_before_local=effect_before_local,
        effect_after_local=effect_after_local,
    )
    if reason is not None:
        return reason
    builder.local_set(right_local)
    builder.local_get(left_local).local_get(right_local).raw(OP_I64_MUL).local_set(
        result_local,
    )

    # Unsigned 64x64 -> high 64 using four 32-bit limbs. All intermediates
    # remain exact in i64; the low half is the ordinary wrapping multiply.
    builder.local_get(left_local).i64_const(32).raw(OP_I64_SHR_U)
    builder.local_get(right_local).i64_const(0xFFFFFFFF).raw(OP_I64_AND, OP_I64_MUL)
    builder.local_get(left_local).i64_const(0xFFFFFFFF).raw(OP_I64_AND)
    builder.local_get(right_local).i64_const(0xFFFFFFFF).raw(OP_I64_AND, OP_I64_MUL)
    builder.i64_const(32).raw(OP_I64_SHR_U, OP_I64_ADD).local_set(work_local)

    builder.local_get(left_local).i64_const(32).raw(OP_I64_SHR_U)
    builder.local_get(right_local).i64_const(32).raw(OP_I64_SHR_U, OP_I64_MUL)
    builder.local_get(work_local).i64_const(32).raw(OP_I64_SHR_U, OP_I64_ADD)
    builder.local_get(work_local).i64_const(0xFFFFFFFF).raw(OP_I64_AND)
    builder.local_get(left_local).i64_const(0xFFFFFFFF).raw(OP_I64_AND)
    builder.local_get(right_local).i64_const(32).raw(OP_I64_SHR_U, OP_I64_MUL)
    builder.raw(OP_I64_ADD).i64_const(32).raw(OP_I64_SHR_U, OP_I64_ADD).local_set(
        high_local,
    )

    _address(builder, 0, 0)
    builder.local_get(result_local).i64_store()
    _address(builder, 0, 2 * 8)
    builder.local_get(high_local).i64_store()
    _address(builder, 0, FLAGS_OFFSET)
    builder.i64_load().i64_const(
        _signed_i64(MASK64 ^ ((1 << CF) | (1 << OF))),
    ).raw(OP_I64_AND).local_set(flags_local)

    def high_nonzero():
        builder.local_get(high_local).raw(OP_I64_EQZ, OP_I32_EQZ)

    _emit_bool_flag(builder, flags_local, CF, high_nonzero)
    _emit_bool_flag(builder, flags_local, OF, high_nonzero)
    _address(builder, 0, FLAGS_OFFSET)
    builder.local_get(flags_local).i64_store()
    return None


def _emit_vector_operation(
    builder: CodeBuilder,
    instruction: Any,
    *,
    left_local: int,
    right_local: int,
    result_local: int,
    high_local: int,
    work_local: int,
    guest_base: int,
    guest_size: int,
    specialization_state: Any | None,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_before_high_local: int,
    effect_after_local: int,
    effect_after_high_local: int,
    executable_pages: frozenset[int],
) -> str | None:
    operands = tuple(getattr(instruction, "operands", ()))
    if len(operands) < 2:
        return "compiled vector tier requires two operands"
    destination, source = operands[:2]
    if isinstance(destination, EffectiveAddressOperand):
        if (
            instruction.semantic is not MachineSemanticToken.VECTOR_MOVE
            or not isinstance(source, VectorRegisterOperand)
        ):
            return "vector memory destination requires an XMM move source"
        address = _specialized_memory_address(
            instruction, destination, specialization_state,
        )
        if address is None:
            return "dynamic vector memory destination requires block-entry specialization"
        if not guest_base <= address or address + 16 > guest_base + guest_size:
            return "vector memory destination exceeds the compiled mirror window"
        if any(page in executable_pages for page in range(address // 4096, (address + 15) // 4096 + 1)):
            return "vector memory destination targets a translated executable page"
        if str(instruction.token.name).startswith(("MOVDQA", "MOVAPS")) and address % 16:
            return "aligned vector store destination is not 16-byte aligned"
        _guest_address(builder, address, guest_base)
        builder.i64_load().local_set(effect_before_local)
        _guest_address(builder, address + 8, guest_base)
        builder.i64_load().local_set(effect_before_high_local)
        source_offset = VECTOR_OFFSET + int(source.register) * 16
        _address(builder, 0, source_offset)
        builder.i64_load().local_set(result_local)
        _address(builder, 0, source_offset + 8)
        builder.i64_load().local_set(work_local)
        builder.i64_const(2).local_set(effect_kind_local)
        builder.i64_const(_signed_i64(address)).local_set(effect_address_local)
        builder.i64_const(128).local_set(effect_width_local)
        builder.local_get(result_local).local_set(effect_after_local)
        builder.local_get(work_local).local_set(effect_after_high_local)
        _guest_address(builder, address, guest_base)
        builder.local_get(result_local).i64_store()
        _guest_address(builder, address + 8, guest_base)
        builder.local_get(work_local).i64_store()
        return None
    if (
        not isinstance(destination, VectorRegisterOperand)
        or not isinstance(source, (VectorRegisterOperand, EffectiveAddressOperand))
    ):
        return "compiled vector tier requires an XMM destination and XMM/m128 source"
    if int(destination.width) != 128:
        return "compiled vector destination must be 128 bits"
    destination_offset = VECTOR_OFFSET + int(destination.register) * 16
    _address(builder, 0, destination_offset)
    builder.i64_load().local_set(left_local)
    _address(builder, 0, destination_offset + 8)
    builder.i64_load().local_set(high_local)

    if isinstance(source, VectorRegisterOperand):
        source_offset = VECTOR_OFFSET + int(source.register) * 16
        _address(builder, 0, source_offset)
        builder.i64_load().local_set(right_local)
        _address(builder, 0, source_offset + 8)
        builder.i64_load().local_set(work_local)
    else:
        address = _specialized_memory_address(
            instruction, source, specialization_state,
        )
        if address is None:
            return "dynamic vector memory source requires block-entry specialization"
        if not guest_base <= address or address + 16 > guest_base + guest_size:
            return "vector memory source exceeds the compiled mirror window"
        if str(instruction.token.name).startswith("MOVDQA") and address % 16:
            return "aligned vector load source is not 16-byte aligned"
        _guest_address(builder, address, guest_base)
        builder.i64_load().local_set(right_local)
        _guest_address(builder, address + 8, guest_base)
        builder.i64_load().local_set(work_local)
        builder.i64_const(1).local_set(effect_kind_local)
        builder.i64_const(_signed_i64(address)).local_set(effect_address_local)
        builder.i64_const(128).local_set(effect_width_local)
        builder.local_get(right_local).local_set(effect_before_local)
        builder.local_get(work_local).local_set(effect_before_high_local)
        builder.local_get(right_local).local_set(effect_after_local)
        builder.local_get(work_local).local_set(effect_after_high_local)

    if instruction.semantic is MachineSemanticToken.VECTOR_XOR:
        builder.local_get(left_local).local_get(right_local).raw(OP_I64_XOR)
        builder.local_set(result_local)
        builder.local_get(high_local).local_get(work_local).raw(OP_I64_XOR)
        builder.local_set(work_local)
    else:
        builder.local_get(right_local).local_set(result_local)
    _address(builder, 0, destination_offset)
    builder.local_get(result_local).i64_store()
    _address(builder, 0, destination_offset + 8)
    builder.local_get(work_local).i64_store()
    return None


def _emit_scalar_store(
    builder: CodeBuilder, parameter: int, offset: int, value: int,
) -> None:
    _address(builder, parameter, offset)
    builder.i64_const(_signed_i64(value)).i64_store()


def _emit_checkpoint(
    builder: CodeBuilder,
    witness: MachineBlockInstructionWitness,
    *,
    effect_kind_local: int,
    effect_address_local: int,
    effect_width_local: int,
    effect_before_local: int,
    effect_before_high_local: int,
    effect_after_local: int,
    effect_after_high_local: int,
    stack_kind_local: int,
    stack_value_local: int,
    stack_depth_local: int,
) -> None:
    record = witness.operation_index * JOURNAL_STRIDE
    _emit_scalar_store(builder, 1, record, witness.address)
    _emit_scalar_store(builder, 1, record + 8, witness.semantic_id)
    _emit_scalar_store(
        builder, 1, record + 16, int(witness.encoded_digest[:16], 16),
    )
    for offset in range(0, STATE_SIZE, 8):
        _address(builder, 1, record + JOURNAL_STATE_OFFSET + offset)
        _address(builder, 0, offset)
        builder.i64_load().i64_store()
    for offset, local in enumerate((
        effect_kind_local, effect_address_local, effect_width_local,
        effect_before_local, effect_before_high_local,
        effect_after_local, effect_after_high_local,
    )):
        _address(builder, 1, record + JOURNAL_EFFECT_OFFSET + offset * 8)
        builder.local_get(local).i64_store()
    for offset, local in enumerate((
        stack_kind_local, stack_value_local, stack_depth_local,
    )):
        _address(builder, 1, record + JOURNAL_STACK_OFFSET + offset * 8)
        builder.local_get(local).i64_store()


def _wat_address(parameter: str, offset: int) -> str:
    return f"(i32.add (local.get ${parameter}) (i32.const {int(offset)}))"


def _wat_guest_address(address: int, guest_base: int) -> str:
    return _wat_address("guest", int(address) - int(guest_base))


def _wat_load_width(width: int, address: str) -> str:
    operation = {8: "i64.load8_u", 16: "i64.load16_u", 32: "i64.load32_u", 64: "i64.load"}[width]
    return f"({operation} {address})"


def _wat_store_width(width: int, address: str, value: str) -> str:
    operation = {8: "i64.store8", 16: "i64.store16", 32: "i64.store32", 64: "i64.store"}[width]
    return f"    ({operation} {address} {value})"


def _wat_move(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    destination, source = tuple(instruction.operands)[:2]
    width = int(getattr(destination, "width", getattr(source, "width", 0)))
    lines = []
    source_address = _specialized_memory_address(
        instruction, source, specialization_state,
    )
    if isinstance(source, EffectiveAddressOperand):
        guest_address = _wat_guest_address(source_address, guest_base)
        lines.extend([
            f"    (local.set $effect_before {_wat_load_width(width, guest_address)})",
            "    (local.set $effect_after (local.get $effect_before))",
            "    (local.set $effect_kind (i64.const 1))",
            f"    (local.set $effect_address (i64.const {source_address}))",
            f"    (local.set $effect_width (i64.const {width}))",
            "    (local.set $result (local.get $effect_before))",
        ])
    elif isinstance(source, RegisterOperand):
        value = f"(i64.load {_wat_address('state', int(source.register) * 8)})"
        if width < 64:
            value = f"(i64.and {value} (i64.const {(1 << width) - 1}))"
        lines.append(f"    (local.set $result {value})")
    else:
        value = f"(i64.const {_signed_i64(int(source.value) & ((1 << width) - 1))})"
        lines.append(f"    (local.set $result {value})")
    static_destination = _specialized_memory_address(
        instruction, destination, specialization_state,
    )
    if isinstance(destination, EffectiveAddressOperand):
        guest_address = _wat_guest_address(static_destination, guest_base)
        lines.extend([
            f"    (local.set $effect_before {_wat_load_width(width, guest_address)})",
            "    (local.set $effect_after (local.get $result))",
            "    (local.set $effect_kind (i64.const 2))",
            f"    (local.set $effect_address (i64.const {static_destination}))",
            f"    (local.set $effect_width (i64.const {width}))",
            _wat_store_width(width, guest_address, "(local.get $result)"),
        ])
        return lines
    destination_address = _wat_address("state", int(destination.register) * 8)
    value = "(local.get $result)"
    if width < 32:
        old = f"(i64.load {destination_address})"
        keep = _signed_i64(MASK64 ^ ((1 << width) - 1))
        value = f"(i64.or (i64.and {old} (i64.const {keep})) {value})"
    lines.append(f"    (i64.store {destination_address} {value})")
    return lines


def _wat_operand(operand: Any, width: int) -> str:
    if isinstance(operand, RegisterOperand):
        value = f"(i64.load {_wat_address('state', int(operand.register) * 8)})"
    else:
        value_int = int(operand.value) & ((1 << int(operand.width)) - 1)
        if operand.signed and value_int & (1 << (int(operand.width) - 1)):
            value_int -= 1 << int(operand.width)
        value = f"(i64.const {_signed_i64(value_int & ((1 << width) - 1))})"
    return (
        f"(i64.and {value} (i64.const {(1 << width) - 1}))"
        if width < 64 else value
    )


def _wat_arithmetic_operand(
    instruction: Any,
    operand_index: int,
    width: int,
    guest_base: int,
    specialization_state: Any | None,
) -> tuple[list[str], str]:
    operand = instruction.operands[operand_index]
    if not isinstance(operand, EffectiveAddressOperand):
        return [], _wat_operand(operand, width)
    address = _specialized_memory_address(
        instruction, operand, specialization_state,
    )
    guest_address = _wat_guest_address(address, guest_base)
    return ([
        f"    (local.set $effect_before {_wat_load_width(width, guest_address)})",
        "    (local.set $effect_after (local.get $effect_before))",
        "    (local.set $effect_kind (i64.const 1))",
        f"    (local.set $effect_address (i64.const {_signed_i64(address)}))",
        f"    (local.set $effect_width (i64.const {width}))",
    ], "(local.get $effect_before)")


def _wat_effective_address(
    instruction: Any, specialization_state: Any | None,
) -> list[str]:
    destination, source = tuple(instruction.operands)[:2]
    width = int(destination.width)
    base = int(instruction.address) + len(instruction.encoded) if source.rip_relative else 0
    prefixes = tuple(getattr(instruction, "legacy_prefixes", ()))
    if 0x64 in prefixes:
        base += int(specialization_state.fs_base)
    if 0x65 in prefixes:
        base += int(specialization_state.gs_base)
    value = f"(i64.const {_signed_i64(base)})"
    if source.base is not None:
        register = f"(i64.load {_wat_address('state', int(source.base) * 8)})"
        value = f"(i64.add {value} {register})"
    if source.index is not None:
        register = f"(i64.load {_wat_address('state', int(source.index) * 8)})"
        value = f"(i64.add {value} (i64.mul {register} (i64.const {int(source.scale)})))"
    value = f"(i64.add {value} (i64.const {_signed_i64(int(source.displacement))}))"
    if width < 64:
        value = f"(i64.and {value} (i64.const {(1 << width) - 1}))"
    destination_address = _wat_address("state", int(destination.register) * 8)
    if width == 16:
        keep = _signed_i64(MASK64 ^ 0xFFFF)
        value = (
            f"(i64.or (i64.and (i64.load {destination_address}) "
            f"(i64.const {keep})) {value})"
        )
    return [_wat_store("state", int(destination.register) * 8, value)]


def _wat_extend(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    destination = instruction.operands[0]
    source_width = _operand_data_width(instruction, 1)
    target_width = int(destination.width)
    source_lines, value = _wat_arithmetic_operand(
        instruction, 1, source_width, guest_base, specialization_state,
    )
    if instruction.semantic is MachineSemanticToken.SIGN_EXTEND:
        sign = 1 << (source_width - 1)
        value = f"(i64.sub (i64.xor {value} (i64.const {sign})) (i64.const {sign}))"
    if target_width < 64:
        value = f"(i64.and {value} (i64.const {(1 << target_width) - 1}))"
    destination_address = _wat_address("state", int(destination.register) * 8)
    if target_width == 16:
        value = (
            f"(i64.or (i64.and (i64.load {destination_address}) "
            f"(i64.const {_signed_i64(MASK64 ^ 0xFFFF)})) {value})"
        )
    return [*source_lines, _wat_store(
        "state", int(destination.register) * 8, value,
    )]


def _wat_flag(flags: str, condition: str, bit: int) -> str:
    return (
        f"(i64.or {flags} (i64.mul (i64.extend_i32_u {condition}) "
        f"(i64.const {1 << bit})))"
    )


def _wat_arithmetic(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    destination, source = tuple(instruction.operands)[:2]
    width = _operand_data_width(instruction, 0)
    kind = _ARITHMETIC_SEMANTICS[instruction.semantic]
    operation = {
        "add": "i64.add", "sub": "i64.sub", "and": "i64.and",
        "sbb": "i64.sub",
        "or": "i64.or", "xor": "i64.xor",
    }[kind]
    raw_result = f"({operation} (local.get $left) (local.get $right))"
    result = (
        f"(i64.and {raw_result} (i64.const {(1 << width) - 1}))"
        if width < 64 else raw_result
    )
    destination_address = None
    write_value = "(local.get $result)"
    if (
        instruction.semantic not in _FLAG_ONLY_ARITHMETIC
        and isinstance(destination, RegisterOperand)
    ):
        destination_address = _wat_address("state", int(destination.register) * 8)
        if width < 32:
            keep = _signed_i64(MASK64 ^ ((1 << width) - 1))
            write_value = (
                f"(i64.or (i64.and (i64.load {destination_address}) "
                f"(i64.const {keep})) (local.get $result))"
            )
    logical = kind in {"and", "or", "xor"}
    cleared = LOGICAL_FLAGS if logical else ARITHMETIC_FLAGS
    flags = (
        f"(i64.and (i64.load {_wat_address('state', FLAGS_OFFSET)}) "
        f"(i64.const {_signed_i64(MASK64 ^ cleared)}))"
    )
    if not logical:
        if kind == "add":
            carry = "(i64.lt_u (local.get $result) (local.get $left))"
        elif kind == "sbb":
            carry = (
                f"(i32.or (i64.lt_u (local.get $left) (local.get $right)) "
                f"(i32.and (i64.eqz (local.get $right)) {_wat_flag_truth(CF)}))"
            )
        else:
            carry = "(i64.lt_u (local.get $left) (local.get $right))"
        flags = _wat_flag(flags, carry, CF)
        auxiliary_expr = (
            "(i64.and (i64.xor (i64.xor (local.get $left) "
            "(local.get $right)) (local.get $result)) (i64.const 16))"
        )
        flags = _wat_flag(flags, f"(i32.eqz (i64.eqz {auxiliary_expr}))", AF)
        pair = "(i64.xor (local.get $left) (local.get $right))"
        if kind == "add":
            pair = f"(i64.xor {pair} (i64.const -1))"
        overflow_expr = (
            f"(i64.and (i64.and {pair} (i64.xor (local.get $left) "
            f"(local.get $result))) (i64.const {_signed_i64(1 << (width - 1))}))"
        )
        flags = _wat_flag(flags, f"(i32.eqz (i64.eqz {overflow_expr}))", OF)
    parity = (
        "(i64.eqz (i64.and (i64.popcnt (i64.and (local.get $result) "
        "(i64.const 255))) (i64.const 1)))"
    )
    flags = _wat_flag(flags, parity, PF)
    flags = _wat_flag(flags, "(i64.eqz (local.get $result))", ZF)
    sign_expr = (
        f"(i64.and (local.get $result) (i64.const {_signed_i64(1 << (width - 1))}))"
    )
    flags = _wat_flag(flags, f"(i32.eqz (i64.eqz {sign_expr}))", SF)
    left_lines, left_value = _wat_arithmetic_operand(
        instruction, 0, width, guest_base, specialization_state,
    )
    right_lines, right_value = _wat_arithmetic_operand(
        instruction, 1, width, guest_base, specialization_state,
    )
    if kind == "sbb":
        right_value = (
            f"(i64.add {right_value} "
            f"(i64.extend_i32_u {_wat_flag_truth(CF)}))"
        )
        if width < 64:
            right_value = f"(i64.and {right_value} (i64.const {(1 << width) - 1}))"
    lines = [
        *left_lines,
        f"    (local.set $left {left_value})",
        *right_lines,
        f"    (local.set $right {right_value})",
        f"    (local.set $result {result})",
    ]
    if instruction.semantic not in _FLAG_ONLY_ARITHMETIC:
        if isinstance(destination, EffectiveAddressOperand):
            address = _specialized_memory_address(
                instruction, destination, specialization_state,
            )
            lines.extend([
                "    (local.set $effect_kind (i64.const 2))",
                "    (local.set $effect_after (local.get $result))",
                _wat_store_width(
                    width, _wat_guest_address(address, guest_base),
                    "(local.get $result)",
                ),
            ])
        else:
            assert destination_address is not None
            lines.append(f"    (i64.store {destination_address} {write_value})")
    lines.extend([
        f"    (local.set $flags {flags})",
        _wat_store("state", FLAGS_OFFSET, "(local.get $flags)"),
    ])
    return lines


def _wat_result_destination(
    instruction: Any,
    destination: Any,
    width: int,
    value: str,
    guest_base: int,
    specialization_state: Any | None,
) -> list[str]:
    if isinstance(destination, EffectiveAddressOperand):
        address = _specialized_memory_address(
            instruction, destination, specialization_state,
        )
        return [
            "    (local.set $effect_kind (i64.const 2))",
            f"    (local.set $effect_address (i64.const {_signed_i64(address)}))",
            f"    (local.set $effect_width (i64.const {width}))",
            f"    (local.set $effect_after {value})",
            _wat_store_width(
                width, _wat_guest_address(address, guest_base), value,
            ),
        ]
    address = _wat_address("state", int(destination.register) * 8)
    if width < 32:
        keep = _signed_i64(MASK64 ^ ((1 << width) - 1))
        value = (
            f"(i64.or (i64.and (i64.load {address}) (i64.const {keep})) {value})"
        )
    return [f"    (i64.store {address} {value})"]


def _wat_szp_flags(flags: str, result: str, width: int) -> str:
    parity = (
        f"(i64.eqz (i64.and (i64.popcnt (i64.and {result} "
        "(i64.const 255))) (i64.const 1)))"
    )
    flags = _wat_flag(flags, parity, PF)
    flags = _wat_flag(flags, f"(i64.eqz {result})", ZF)
    sign = f"(i64.and {result} (i64.const {_signed_i64(1 << (width - 1))}))"
    return _wat_flag(flags, f"(i32.eqz (i64.eqz {sign}))", SF)


def _wat_unary(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    destination = instruction.operands[0]
    width = _operand_data_width(instruction, 0)
    source_lines, value = _wat_arithmetic_operand(
        instruction, 0, width, guest_base, specialization_state,
    )
    raw = (
        f"(i64.xor {value} (i64.const -1))"
        if instruction.semantic is MachineSemanticToken.BITWISE_NOT
        else f"(i64.sub (i64.const 0) {value})"
    )
    result = (
        f"(i64.and {raw} (i64.const {(1 << width) - 1}))"
        if width < 64 else raw
    )
    lines = [
        *source_lines,
        f"    (local.set $left {value})",
        f"    (local.set $result {result})",
        *_wat_result_destination(
            instruction, destination, width, "(local.get $result)",
            guest_base, specialization_state,
        ),
    ]
    if instruction.semantic is MachineSemanticToken.BITWISE_NOT:
        return lines
    flags = (
        f"(i64.and (i64.load {_wat_address('state', FLAGS_OFFSET)}) "
        f"(i64.const {_signed_i64(MASK64 ^ ARITHMETIC_FLAGS)}))"
    )
    flags = _wat_flag(flags, "(i32.eqz (i64.eqz (local.get $left)))", CF)
    auxiliary = (
        "(i64.and (i64.xor (local.get $left) (local.get $result)) "
        "(i64.const 16))"
    )
    flags = _wat_flag(flags, f"(i32.eqz (i64.eqz {auxiliary}))", AF)
    overflow = (
        f"(i64.and (i64.and (local.get $left) (local.get $result)) "
        f"(i64.const {_signed_i64(1 << (width - 1))}))"
    )
    flags = _wat_flag(flags, f"(i32.eqz (i64.eqz {overflow}))", OF)
    flags = _wat_szp_flags(flags, "(local.get $result)", width)
    lines.extend([
        f"    (local.set $flags {flags})",
        _wat_store("state", FLAGS_OFFSET, "(local.get $flags)"),
    ])
    return lines


def _wat_shift_rotate(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    destination, count_operand = tuple(instruction.operands)[:2]
    width = _operand_data_width(instruction, 0)
    count = int(count_operand.value) & (0x3F if width == 64 else 0x1F)
    if count == 0:
        return []
    source_lines, value = _wat_arithmetic_operand(
        instruction, 0, width, guest_base, specialization_state,
    )
    operation = {
        MachineSemanticToken.SHIFT_LEFT: "i64.shl",
        MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC: "i64.shr_s",
        MachineSemanticToken.ROTATE_LEFT: "i64.rotl",
    }[instruction.semantic]
    raw = f"({operation} (local.get $left) (i64.const {count}))"
    result = (
        f"(i64.and {raw} (i64.const {(1 << width) - 1}))"
        if width < 64 else raw
    )
    lines = [
        *source_lines,
        f"    (local.set $left {value})",
        f"    (local.set $result {result})",
        *_wat_result_destination(
            instruction, destination, width, "(local.get $result)",
            guest_base, specialization_state,
        ),
    ]
    rotate = instruction.semantic is MachineSemanticToken.ROTATE_LEFT
    cleared = (1 << CF) | ((1 << OF) if rotate and count == 1 else 0)
    if not rotate:
        cleared = LOGICAL_FLAGS
    flags = (
        f"(i64.and (i64.load {_wat_address('state', FLAGS_OFFSET)}) "
        f"(i64.const {_signed_i64(MASK64 ^ cleared)}))"
    )
    if rotate:
        carry_value, carry_bit = "(local.get $result)", 1
    elif instruction.semantic is MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC:
        carry_value = "(local.get $left)"
        carry_bit = 1 << (count - 1) if count <= width else 1 << (width - 1)
    else:
        carry_value = "(local.get $left)"
        carry_bit = 1 << (width - count) if count <= width else 0
    carry = (
        f"(i32.eqz (i64.eqz (i64.and {carry_value} "
        f"(i64.const {_signed_i64(carry_bit)}))))"
    )
    flags = _wat_flag(flags, carry, CF)
    if count == 1:
        if instruction.semantic is MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC:
            overflow = "(i32.const 0)"
        else:
            sign = (
                f"(i32.eqz (i64.eqz (i64.and (local.get $result) "
                f"(i64.const {_signed_i64(1 << (width - 1))}))))"
            )
            overflow = f"(i32.xor {sign} {carry})"
    else:
        overflow = _wat_flag_truth(OF)
    flags = _wat_flag(flags, overflow, OF)
    if not rotate:
        flags = _wat_szp_flags(flags, "(local.get $result)", width)
    lines.extend([
        f"    (local.set $flags {flags})",
        _wat_store("state", FLAGS_OFFSET, "(local.get $flags)"),
    ])
    return lines


def _wat_conditional_set(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    destination = instruction.operands[0]
    lines: list[str] = []
    if isinstance(destination, EffectiveAddressOperand):
        source_lines, _value = _wat_arithmetic_operand(
            instruction, 0, 8, guest_base, specialization_state,
        )
        lines.extend(source_lines)
    condition = _wat_condition(_condition_name(instruction))
    result = f"(i64.extend_i32_u {condition})"
    lines.append(f"    (local.set $result {result})")
    lines.extend(_wat_result_destination(
        instruction, destination, 8, "(local.get $result)",
        guest_base, specialization_state,
    ))
    return lines


def _wat_conditional_move(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    destination = instruction.operands[0]
    width = _operand_data_width(instruction, 0)
    source_lines, source = _wat_arithmetic_operand(
        instruction, 1, width, guest_base, specialization_state,
    )
    address = _wat_address("state", int(destination.register) * 8)
    old = f"(i64.load {address})"
    false_value = (
        f"(i64.and {old} (i64.const 65535))" if width == 16 else old
    )
    condition = _wat_condition(_condition_name(instruction))
    selected = f"(select {source} {false_value} {condition})"
    if width == 16:
        selected = (
            f"(i64.or (i64.and {old} "
            f"(i64.const {_signed_i64(MASK64 ^ 0xFFFF)})) {selected})"
        )
    return [*source_lines, f"    (i64.store {address} {selected})"]


def _wat_increment_decrement(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    destination = instruction.operands[0]
    width = _operand_data_width(instruction, 0)
    source_lines, value = _wat_arithmetic_operand(
        instruction, 0, width, guest_base, specialization_state,
    )
    operation = (
        "i64.add"
        if instruction.semantic is MachineSemanticToken.INTEGER_INCREMENT
        else "i64.sub"
    )
    raw = f"({operation} (local.get $left) (i64.const 1))"
    result = (
        f"(i64.and {raw} (i64.const {(1 << width) - 1}))"
        if width < 64 else raw
    )
    lines = [
        *source_lines,
        f"    (local.set $left {value})",
        f"    (local.set $result {result})",
        *_wat_result_destination(
            instruction, destination, width, "(local.get $result)",
            guest_base, specialization_state,
        ),
    ]
    cleared = ARITHMETIC_FLAGS ^ (1 << CF)
    flags = (
        f"(i64.and (i64.load {_wat_address('state', FLAGS_OFFSET)}) "
        f"(i64.const {_signed_i64(MASK64 ^ cleared)}))"
    )
    auxiliary = (
        "(i64.and (i64.xor (i64.xor (local.get $left) (i64.const 1)) "
        "(local.get $result)) (i64.const 16))"
    )
    flags = _wat_flag(flags, f"(i32.eqz (i64.eqz {auxiliary}))", AF)
    pair = "(i64.xor (local.get $left) (i64.const 1))"
    if instruction.semantic is MachineSemanticToken.INTEGER_INCREMENT:
        pair = f"(i64.xor {pair} (i64.const -1))"
    overflow = (
        f"(i64.and (i64.and {pair} (i64.xor (local.get $left) "
        f"(local.get $result))) (i64.const {_signed_i64(1 << (width - 1))}))"
    )
    flags = _wat_flag(flags, f"(i32.eqz (i64.eqz {overflow}))", OF)
    flags = _wat_szp_flags(flags, "(local.get $result)", width)
    lines.extend([
        f"    (local.set $flags {flags})",
        _wat_store("state", FLAGS_OFFSET, "(local.get $flags)"),
    ])
    return lines


def _wat_exchange(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    left, right = tuple(instruction.operands)[:2]
    right_address = _wat_address("state", int(right.register) * 8)
    if isinstance(left, EffectiveAddressOperand):
        source_lines, value = _wat_arithmetic_operand(
            instruction, 0, 64, guest_base, specialization_state,
        )
        return [
            *source_lines,
            f"    (local.set $left {value})",
            f"    (local.set $result (i64.load {right_address}))",
            *_wat_result_destination(
                instruction, left, 64, "(local.get $result)",
                guest_base, specialization_state,
            ),
            f"    (i64.store {right_address} (local.get $left))",
        ]
    left_address = _wat_address("state", int(left.register) * 8)
    return [
        f"    (local.set $left (i64.load {left_address}))",
        f"    (local.set $right (i64.load {right_address}))",
        f"    (i64.store {left_address} (local.get $right))",
        f"    (i64.store {right_address} (local.get $left))",
    ]


def _wat_multiply_unsigned(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    source_lines, source = _wat_arithmetic_operand(
        instruction, 0, 64, guest_base, specialization_state,
    )
    a0 = "(i64.and (local.get $left) (i64.const 4294967295))"
    a1 = "(i64.shr_u (local.get $left) (i64.const 32))"
    b0 = "(i64.and (local.get $right) (i64.const 4294967295))"
    b1 = "(i64.shr_u (local.get $right) (i64.const 32))"
    work = (
        f"(i64.add (i64.mul {a1} {b0}) "
        f"(i64.shr_u (i64.mul {a0} {b0}) (i64.const 32)))"
    )
    high = (
        f"(i64.add (i64.add (i64.mul {a1} {b1}) "
        f"(i64.shr_u (local.get $work) (i64.const 32))) "
        f"(i64.shr_u (i64.add (i64.and (local.get $work) "
        f"(i64.const 4294967295)) (i64.mul {a0} {b1})) (i64.const 32)))"
    )
    flags = (
        f"(i64.and (i64.load {_wat_address('state', FLAGS_OFFSET)}) "
        f"(i64.const {_signed_i64(MASK64 ^ ((1 << CF) | (1 << OF)))}))"
    )
    high_nonzero = "(i32.eqz (i64.eqz (local.get $high)))"
    flags = _wat_flag(flags, high_nonzero, CF)
    flags = _wat_flag(flags, high_nonzero, OF)
    return [
        *source_lines,
        f"    (local.set $left (i64.load {_wat_address('state', 0)}))",
        f"    (local.set $right {source})",
        "    (local.set $result (i64.mul (local.get $left) (local.get $right)))",
        f"    (local.set $work {work})",
        f"    (local.set $high {high})",
        _wat_store("state", 0, "(local.get $result)"),
        _wat_store("state", 2 * 8, "(local.get $high)"),
        f"    (local.set $flags {flags})",
        _wat_store("state", FLAGS_OFFSET, "(local.get $flags)"),
    ]


def _wat_vector_operation(
    instruction: Any, guest_base: int, specialization_state: Any | None,
) -> list[str]:
    destination, source = tuple(instruction.operands)[:2]
    if isinstance(destination, EffectiveAddressOperand):
        address = _specialized_memory_address(
            instruction, destination, specialization_state,
        )
        source_offset = VECTOR_OFFSET + int(source.register) * 16
        low_address = _wat_guest_address(address, guest_base)
        high_address = _wat_guest_address(address + 8, guest_base)
        return [
            f"    (local.set $effect_before (i64.load {low_address}))",
            f"    (local.set $effect_before_high (i64.load {high_address}))",
            f"    (local.set $result (i64.load {_wat_address('state', source_offset)}))",
            f"    (local.set $work (i64.load {_wat_address('state', source_offset + 8)}))",
            "    (local.set $effect_kind (i64.const 2))",
            f"    (local.set $effect_address (i64.const {_signed_i64(address)}))",
            "    (local.set $effect_width (i64.const 128))",
            "    (local.set $effect_after (local.get $result))",
            "    (local.set $effect_after_high (local.get $work))",
            f"    (i64.store {low_address} (local.get $result))",
            f"    (i64.store {high_address} (local.get $work))",
        ]
    destination_offset = VECTOR_OFFSET + int(destination.register) * 16
    lines = [
        f"    (local.set $left (i64.load {_wat_address('state', destination_offset)}))",
        f"    (local.set $high (i64.load {_wat_address('state', destination_offset + 8)}))",
    ]
    if isinstance(source, VectorRegisterOperand):
        source_offset = VECTOR_OFFSET + int(source.register) * 16
        lines.extend([
            f"    (local.set $right (i64.load {_wat_address('state', source_offset)}))",
            f"    (local.set $work (i64.load {_wat_address('state', source_offset + 8)}))",
        ])
    else:
        address = _specialized_memory_address(
            instruction, source, specialization_state,
        )
        low = f"(i64.load {_wat_guest_address(address, guest_base)})"
        high = f"(i64.load {_wat_guest_address(address + 8, guest_base)})"
        lines.extend([
            f"    (local.set $right {low})",
            f"    (local.set $work {high})",
            "    (local.set $effect_kind (i64.const 1))",
            f"    (local.set $effect_address (i64.const {_signed_i64(address)}))",
            "    (local.set $effect_width (i64.const 128))",
            "    (local.set $effect_before (local.get $right))",
            "    (local.set $effect_before_high (local.get $work))",
            "    (local.set $effect_after (local.get $right))",
            "    (local.set $effect_after_high (local.get $work))",
        ])
    if instruction.semantic is MachineSemanticToken.VECTOR_XOR:
        lines.extend([
            "    (local.set $result (i64.xor (local.get $left) (local.get $right)))",
            "    (local.set $work (i64.xor (local.get $high) (local.get $work)))",
        ])
    else:
        lines.append("    (local.set $result (local.get $right))")
    lines.extend([
        _wat_store("state", destination_offset, "(local.get $result)"),
        _wat_store("state", destination_offset + 8, "(local.get $work)"),
    ])
    return lines


def _wat_store(parameter: str, offset: int, value: str) -> str:
    return f"    (i64.store {_wat_address(parameter, offset)} {value})"


def _wat_flag_truth(bit: int, *, inverted: bool = False) -> str:
    zero = (
        f"(i64.eqz (i64.and (i64.load {_wat_address('state', FLAGS_OFFSET)}) "
        f"(i64.const {1 << bit})))"
    )
    return zero if inverted else f"(i32.eqz {zero})"


def _wat_condition(condition: str) -> str | None:
    aliases = {
        "Z": "E", "NZ": "NE", "C": "B", "NAE": "B",
        "NB": "AE", "NC": "AE", "NA": "BE", "NBE": "A",
        "NGE": "L", "NL": "GE", "NG": "LE", "NLE": "G",
    }
    condition = aliases.get(condition, condition)
    simple = {
        "E": _wat_flag_truth(ZF), "NE": _wat_flag_truth(ZF, inverted=True),
        "B": _wat_flag_truth(CF), "AE": _wat_flag_truth(CF, inverted=True),
        "S": _wat_flag_truth(SF), "NS": _wat_flag_truth(SF, inverted=True),
        "O": _wat_flag_truth(OF), "NO": _wat_flag_truth(OF, inverted=True),
    }
    if condition in simple:
        return simple[condition]
    if condition == "BE":
        return f"(i32.or {_wat_flag_truth(CF)} {_wat_flag_truth(ZF)})"
    if condition == "A":
        return f"(i32.and {_wat_flag_truth(CF, inverted=True)} {_wat_flag_truth(ZF, inverted=True)})"
    relation = f"(i32.xor {_wat_flag_truth(SF)} {_wat_flag_truth(OF)})"
    if condition == "L":
        return relation
    if condition == "GE":
        return f"(i32.eqz {relation})"
    if condition == "LE":
        return f"(i32.or {_wat_flag_truth(ZF)} {relation})"
    if condition == "G":
        return f"(i32.and {_wat_flag_truth(ZF, inverted=True)} (i32.eqz {relation}))"
    return None


def _wat_control(instruction: Any) -> str:
    targets = _control_targets(instruction)
    if instruction.semantic is MachineSemanticToken.DIRECT_RELATIVE_JUMP:
        value = f"(i64.const {targets[0]})"
    else:
        condition = _wat_condition(_condition_name(instruction))
        value = (
            f"(select (i64.const {targets[0]}) (i64.const {targets[-1]}) "
            f"{condition})"
        )
    return _wat_store("state", PC_OFFSET, value)


def _wat_call_return(
    instruction: Any,
    state: Any,
    guest_base: int,
    resolved_indirect_target: int | None,
) -> list[str]:
    depth = len(state.call_stack)
    if instruction.semantic in {
        MachineSemanticToken.DIRECT_RELATIVE_CALL,
        MachineSemanticToken.INDIRECT_CALL,
    }:
        target = int(
            resolved_indirect_target
            if instruction.semantic is MachineSemanticToken.INDIRECT_CALL
            else _relative_target(instruction)
        )
        return_address = int(instruction.address) + len(instruction.encoded)
        stack_address = (int(state.registers[4]) - 8) & MASK64
        stack_kind, stack_value, effect_kind = 1, return_address, 2
    else:
        target = int(state.call_stack[-1])
        stack_address = int(state.registers[4]) & MASK64
        stack_kind, stack_value, effect_kind = 2, target, 1
    guest_address = _wat_guest_address(stack_address, guest_base)
    lines = [
        f"    (local.set $effect_before {_wat_load_width(64, guest_address)})",
        "    (local.set $effect_after (local.get $effect_before))",
        f"    (local.set $effect_kind (i64.const {effect_kind}))",
        f"    (local.set $effect_address (i64.const {_signed_i64(stack_address)}))",
        "    (local.set $effect_width (i64.const 64))",
    ]
    if stack_kind == 1:
        lines.extend([
            f"    (local.set $effect_after (i64.const {_signed_i64(stack_value)}))",
            _wat_store_width(64, guest_address, "(local.get $effect_after)"),
        ])
    lines.extend([
        _wat_store(
            "state", 4 * 8,
            f"(i64.const {_signed_i64(stack_address + (8 if stack_kind == 2 else 0))})",
        ),
        _wat_store("state", PC_OFFSET, f"(i64.const {_signed_i64(target)})"),
        f"    (local.set $stack_kind (i64.const {stack_kind}))",
        f"    (local.set $stack_value (i64.const {_signed_i64(stack_value)}))",
        f"    (local.set $stack_depth (i64.const {depth}))",
    ])
    return lines


def _wat_stack_data(
    instruction: Any, state: Any, guest_base: int,
) -> list[str]:
    operand = tuple(instruction.operands)[0]
    rsp = int(state.registers[4]) & MASK64
    push = instruction.semantic is MachineSemanticToken.STACK_PUSH
    stack_address = (rsp - 8) & MASK64 if push else rsp
    guest_address = _wat_guest_address(stack_address, guest_base)
    lines = [
        f"    (local.set $effect_before {_wat_load_width(64, guest_address)})",
        "    (local.set $effect_after (local.get $effect_before))",
        f"    (local.set $effect_kind (i64.const {2 if push else 1}))",
        f"    (local.set $effect_address (i64.const {_signed_i64(stack_address)}))",
        "    (local.set $effect_width (i64.const 64))",
    ]
    if push:
        if isinstance(operand, RegisterOperand):
            value = f"(i64.load {_wat_address('state', int(operand.register) * 8)})"
        else:
            value_int = int(operand.value) & ((1 << int(operand.width)) - 1)
            if operand.signed and value_int & (1 << (int(operand.width) - 1)):
                value_int -= 1 << int(operand.width)
            value = f"(i64.const {_signed_i64(value_int)})"
        lines.extend([
            f"    (local.set $result {value})",
            "    (local.set $effect_after (local.get $result))",
            _wat_store_width(64, guest_address, "(local.get $result)"),
            _wat_store("state", 4 * 8, f"(i64.const {_signed_i64(stack_address)})"),
        ])
    else:
        lines.extend([
            _wat_store(
                "state", int(operand.register) * 8,
                "(local.get $effect_before)",
            ),
            _wat_store(
                "state", 4 * 8,
                f"(i64.const {_signed_i64((stack_address + 8) & MASK64)})",
            ),
        ])
    return lines


def _wat_for(
    compiled_instructions: tuple[Any, ...],
    witnesses: tuple[MachineBlockInstructionWitness, ...],
    continuation_address: int,
    guest_base: int,
    specialization_state: Any | None,
    memory_pages: int,
    resolved_indirect_target: int | None,
) -> str:
    lines = [
        ";; turing.machine-block-wasm.v1",
        f'(module (memory (export "memory") {memory_pages})',
        '  (func (export "run") (param $state i32) (param $journal i32) (param $guest i32)',
        '    (local $left i64) (local $right i64)',
        '    (local $result i64) (local $flags i64)',
        '    (local $high i64) (local $work i64)',
        '    (local $effect_kind i64) (local $effect_address i64)',
        '    (local $effect_width i64) (local $effect_before i64) (local $effect_before_high i64)',
        '    (local $effect_after i64) (local $effect_after_high i64)',
        '    (local $stack_kind i64) (local $stack_value i64) (local $stack_depth i64)',
    ]
    for instruction, witness in zip(compiled_instructions, witnesses):
        for local in (
            "effect_kind", "effect_address", "effect_width",
            "effect_before", "effect_before_high", "effect_after",
            "effect_after_high", "stack_kind", "stack_value",
            "stack_depth",
        ):
            lines.append(f"    (local.set ${local} (i64.const 0))")
        lines.append(
            f"    ;; guest {witness.address:#x} {witness.semantic} "
            f"sha256:{witness.encoded_digest} -> journal+{witness.journal_offset} "
            f"successors={','.join(hex(item) for item in witness.possible_next_addresses)}"
        )
        if instruction.semantic in _MOV_SEMANTICS:
            lines.extend(_wat_move(instruction, guest_base, specialization_state))
        elif instruction.semantic in _ARITHMETIC_SEMANTICS:
            lines.extend(_wat_arithmetic(
                instruction, guest_base, specialization_state,
            ))
        elif instruction.semantic is MachineSemanticToken.EFFECTIVE_ADDRESS:
            lines.extend(_wat_effective_address(instruction, specialization_state))
        elif instruction.semantic in _EXTEND_SEMANTICS:
            lines.extend(_wat_extend(
                instruction, guest_base, specialization_state,
            ))
        elif instruction.semantic in _UNARY_SEMANTICS:
            lines.extend(_wat_unary(
                instruction, guest_base, specialization_state,
            ))
        elif instruction.semantic in _SHIFT_SEMANTICS:
            lines.extend(_wat_shift_rotate(
                instruction, guest_base, specialization_state,
            ))
        elif instruction.semantic in _INCDEC_SEMANTICS:
            lines.extend(_wat_increment_decrement(
                instruction, guest_base, specialization_state,
            ))
        elif instruction.semantic is MachineSemanticToken.CONDITIONAL_SET:
            lines.extend(_wat_conditional_set(
                instruction, guest_base, specialization_state,
            ))
        elif instruction.semantic is MachineSemanticToken.CONDITIONAL_MOVE:
            lines.extend(_wat_conditional_move(
                instruction, guest_base, specialization_state,
            ))
        elif instruction.semantic is MachineSemanticToken.EXCHANGE:
            lines.extend(_wat_exchange(
                instruction, guest_base, specialization_state,
            ))
        elif instruction.semantic is MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED:
            lines.extend(_wat_multiply_unsigned(
                instruction, guest_base, specialization_state,
            ))
        elif instruction.semantic in _VECTOR_SEMANTICS:
            lines.extend(_wat_vector_operation(
                instruction, guest_base, specialization_state,
            ))
        if instruction.semantic in {
            MachineSemanticToken.DIRECT_RELATIVE_JUMP,
            MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        }:
            lines.append(_wat_control(instruction))
        elif instruction.semantic in {
            MachineSemanticToken.DIRECT_RELATIVE_CALL,
            MachineSemanticToken.INDIRECT_CALL,
            MachineSemanticToken.RETURN,
        }:
            lines.extend(_wat_call_return(
                instruction, specialization_state, guest_base,
                resolved_indirect_target,
            ))
        elif instruction.semantic is MachineSemanticToken.INDIRECT_JUMP:
            lines.append(_wat_store(
                "state", PC_OFFSET,
                f"(i64.const {_signed_i64(int(resolved_indirect_target))})",
            ))
        elif instruction.semantic in {
            MachineSemanticToken.STACK_PUSH,
            MachineSemanticToken.STACK_POP,
        }:
            lines.extend(_wat_stack_data(
                instruction, specialization_state, guest_base,
            ))
        else:
            next_pc = int(instruction.address) + len(instruction.encoded)
            lines.append(_wat_store("state", PC_OFFSET, f"(i64.const {next_pc})"))
        steps_address = _wat_address("state", STEPS_OFFSET)
        lines.append(_wat_store(
            "state", STEPS_OFFSET,
            f"(i64.add (i64.load {steps_address}) (i64.const 1))",
        ))
        record = witness.journal_offset
        lines.append(_wat_store("journal", record, f"(i64.const {witness.address})"))
        lines.append(_wat_store("journal", record + 8, f"(i64.const {witness.semantic_id})"))
        lines.append(_wat_store(
            "journal", record + 16,
            f"(i64.const {_signed_i64(int(witness.encoded_digest[:16], 16))})",
        ))
        for offset in range(0, STATE_SIZE, 8):
            lines.append(_wat_store(
                "journal", record + JOURNAL_STATE_OFFSET + offset,
                f"(i64.load {_wat_address('state', offset)})",
            ))
        for offset, local in enumerate((
            "effect_kind", "effect_address", "effect_width",
            "effect_before", "effect_before_high",
            "effect_after", "effect_after_high",
        )):
            lines.append(_wat_store(
                "journal", record + JOURNAL_EFFECT_OFFSET + offset * 8,
                f"(local.get ${local})",
            ))
        for offset, local in enumerate((
            "stack_kind", "stack_value", "stack_depth",
        )):
            lines.append(_wat_store(
                "journal", record + JOURNAL_STACK_OFFSET + offset * 8,
                f"(local.get ${local})",
            ))
    lines.extend([
        f"    ;; continuation {continuation_address:#x}",
        "  )",
        ")",
    ])
    return "\n".join(lines) + "\n"


def lower_machine_block_to_wasm(
    block,
    *,
    strict: bool = False,
    executable_pages: frozenset[int] | None = None,
    specialization_state: Any | None = None,
    maximum_instructions: int | None = None,
    resolved_indirect_target: int | None = None,
    indirect_external: bool = False,
) -> MachineBlockWasmArtifact:
    """Lower the longest safe prefix of a translated AMD64 block to Wasm.

    Unsupported effects stop lowering at the exact instruction. ``strict``
    rejects a partial artifact; non-strict mode exposes its continuation RIP so
    the ordinary interpreter can resume without pretending the block was fully
    compiled.
    """

    if maximum_instructions is not None and int(maximum_instructions) <= 0:
        raise ValueError("machine block instruction limit must be positive")
    operations = tuple(block.operations)[
        :None if maximum_instructions is None else int(maximum_instructions)
    ]
    memory_ranges = []
    specialization_guard: dict[str, Any] = {}
    planning_shortfall: MachineBlockLoweringShortfall | None = None
    for operation_index, operation in enumerate(operations):
        instruction = operation.instruction
        prior_ranges = tuple(memory_ranges)
        prior_guard = dict(specialization_guard)
        if (
            instruction.semantic in _MOV_SEMANTICS
            or instruction.semantic in _ARITHMETIC_SEMANTICS
            or instruction.semantic in _EXTEND_SEMANTICS
            or instruction.semantic in _SCALAR_MISC_SEMANTICS
            or instruction.semantic in _VECTOR_SEMANTICS
        ):
            for operand_index, operand in enumerate(getattr(instruction, "operands", ())):
                address = _specialized_memory_address(
                    instruction, operand,
                    specialization_state if operation_index == 0 else None,
                )
                if address is not None:
                    width = _operand_data_width(instruction, operand_index)
                    if width in (8, 16, 32, 64, 128):
                        memory_ranges.append((address, address + width // 8))
                    if _static_memory_address(instruction, operand) is None:
                        specialization_guard.update(_dynamic_memory_guard(
                            instruction, specialization_state,
                        ))
        if operation_index == 0 and specialization_state is not None and instruction.semantic in {
            MachineSemanticToken.DIRECT_RELATIVE_CALL,
            MachineSemanticToken.INDIRECT_CALL,
            MachineSemanticToken.RETURN,
        }:
            stack_address = int(specialization_state.registers[4]) + (
                -8 if instruction.semantic in {
                    MachineSemanticToken.DIRECT_RELATIVE_CALL,
                    MachineSemanticToken.INDIRECT_CALL,
                } else 0
            )
            memory_ranges.append((stack_address & MASK64, (stack_address & MASK64) + 8))
            guarded_registers = dict(specialization_guard.get("registers", ()))
            guarded_registers[4] = int(specialization_state.registers[4])
            specialization_guard.update({
                "registers": tuple(sorted(guarded_registers.items())),
                "call_stack": tuple(specialization_state.call_stack),
                "termination_requested": bool(specialization_state.termination_requested),
            })
        if operation_index == 0 and specialization_state is not None and instruction.semantic in {
            MachineSemanticToken.STACK_PUSH,
            MachineSemanticToken.STACK_POP,
        }:
            stack_address = int(specialization_state.registers[4]) + (
                -8 if instruction.semantic is MachineSemanticToken.STACK_PUSH else 0
            )
            memory_ranges.append((stack_address & MASK64, (stack_address & MASK64) + 8))
            guarded_registers = dict(specialization_guard.get("registers", ()))
            guarded_registers[4] = int(specialization_state.registers[4])
            specialization_guard["registers"] = tuple(sorted(guarded_registers.items()))
        if (
            operation_index == 0
            and specialization_state is not None
            and instruction.semantic in {
                MachineSemanticToken.INDIRECT_CALL,
                MachineSemanticToken.INDIRECT_JUMP,
            }
            and getattr(instruction, "operands", ())
            and isinstance(instruction.operands[0], RegisterOperand)
        ):
            register = int(instruction.operands[0].register)
            guarded_registers = dict(specialization_guard.get("registers", ()))
            guarded_registers[register] = int(specialization_state.registers[register])
            specialization_guard["registers"] = tuple(sorted(guarded_registers.items()))
        if instruction.semantic is MachineSemanticToken.EFFECTIVE_ADDRESS:
            prefixes = tuple(getattr(instruction, "legacy_prefixes", ()))
            if specialization_state is not None and 0x64 in prefixes:
                specialization_guard["fs_base"] = int(specialization_state.fs_base)
            if specialization_state is not None and 0x65 in prefixes:
                specialization_guard["gs_base"] = int(specialization_state.gs_base)
        if memory_ranges:
            planned_base = min(item[0] for item in memory_ranges)
            planned_limit = max(item[1] for item in memory_ranges)
            if planned_limit - planned_base > MAXIMUM_GUEST_WINDOW_BYTES and operation_index:
                memory_ranges = list(prior_ranges)
                specialization_guard = prior_guard
                planning_shortfall = MachineBlockLoweringShortfall(
                    operation_index, int(instruction.address),
                    instruction.semantic.name,
                    "guest-memory mirror would exceed its bounded capacity",
                )
                operations = operations[:operation_index]
                break
    guest_base = min((item[0] for item in memory_ranges), default=0)
    guest_limit = max((item[1] for item in memory_ranges), default=guest_base)
    guest_size = guest_limit - guest_base
    executable_pages = (
        frozenset(int(value) for value in executable_pages)
        if executable_pages is not None else frozenset(
            int(operation.address) // 4096 for operation in block.operations
        )
    )

    builder = CodeBuilder("f64", parameter_count=3)
    left_local = builder.declare_local("i64")
    right_local = builder.declare_local("i64")
    result_local = builder.declare_local("i64")
    flags_local = builder.declare_local("i64")
    high_local = builder.declare_local("i64")
    work_local = builder.declare_local("i64")
    effect_kind_local = builder.declare_local("i64")
    effect_address_local = builder.declare_local("i64")
    effect_width_local = builder.declare_local("i64")
    effect_before_local = builder.declare_local("i64")
    effect_before_high_local = builder.declare_local("i64")
    effect_after_local = builder.declare_local("i64")
    effect_after_high_local = builder.declare_local("i64")
    stack_kind_local = builder.declare_local("i64")
    stack_value_local = builder.declare_local("i64")
    stack_depth_local = builder.declare_local("i64")
    witnesses: list[MachineBlockInstructionWitness] = []
    compiled_instructions: list[Any] = []
    shortfalls: list[MachineBlockLoweringShortfall] = (
        [] if planning_shortfall is None else [planning_shortfall]
    )
    continuation = int(block.entry_address)
    for index, operation in enumerate(operations):
        instruction = operation.instruction
        semantic = instruction.semantic
        reason = None
        control_targets = None
        expected_stack_effect = (0, 0, 0)
        for local in (
            effect_kind_local, effect_address_local, effect_width_local,
            effect_before_local, effect_before_high_local,
            effect_after_local, effect_after_high_local,
            stack_kind_local, stack_value_local, stack_depth_local,
        ):
            builder.i64_const(0).local_set(local)
        if semantic in _MOV_SEMANTICS:
            if guest_size > MAXIMUM_GUEST_WINDOW_BYTES:
                reason = "static guest-memory mirror exceeds its bounded capacity"
            else:
                reason = _emit_move(
                    builder, instruction,
                    guest_base=guest_base, guest_size=guest_size,
                    executable_pages=executable_pages,
                    result_local=result_local,
                    effect_kind_local=effect_kind_local,
                    effect_address_local=effect_address_local,
                    effect_width_local=effect_width_local,
                    effect_before_local=effect_before_local,
                    effect_after_local=effect_after_local,
                    specialization_state=(specialization_state if index == 0 else None),
                )
        elif semantic in _ARITHMETIC_SEMANTICS:
            reason = _emit_arithmetic(
                builder, instruction,
                left_local=left_local, right_local=right_local,
                result_local=result_local, flags_local=flags_local,
                guest_base=guest_base, guest_size=guest_size,
                specialization_state=(specialization_state if index == 0 else None),
                effect_kind_local=effect_kind_local,
                effect_address_local=effect_address_local,
                effect_width_local=effect_width_local,
                effect_before_local=effect_before_local,
                effect_after_local=effect_after_local,
                executable_pages=executable_pages,
            )
        elif semantic is MachineSemanticToken.EFFECTIVE_ADDRESS:
            reason = _emit_effective_address(
                builder, instruction,
                specialization_state=specialization_state,
                result_local=result_local,
            )
        elif semantic in _EXTEND_SEMANTICS:
            reason = _emit_extend(
                builder, instruction,
                guest_base=guest_base, guest_size=guest_size,
                specialization_state=(specialization_state if index == 0 else None),
                result_local=result_local,
                effect_kind_local=effect_kind_local,
                effect_address_local=effect_address_local,
                effect_width_local=effect_width_local,
                effect_before_local=effect_before_local,
                effect_after_local=effect_after_local,
            )
        elif semantic in _UNARY_SEMANTICS:
            reason = _emit_unary(
                builder, instruction,
                left_local=left_local, result_local=result_local,
                flags_local=flags_local,
                guest_base=guest_base, guest_size=guest_size,
                specialization_state=(specialization_state if index == 0 else None),
                effect_kind_local=effect_kind_local,
                effect_address_local=effect_address_local,
                effect_width_local=effect_width_local,
                effect_before_local=effect_before_local,
                effect_after_local=effect_after_local,
                executable_pages=executable_pages,
            )
        elif semantic in _SHIFT_SEMANTICS:
            reason = _emit_shift_rotate(
                builder, instruction,
                left_local=left_local, result_local=result_local,
                flags_local=flags_local,
                guest_base=guest_base, guest_size=guest_size,
                specialization_state=(specialization_state if index == 0 else None),
                effect_kind_local=effect_kind_local,
                effect_address_local=effect_address_local,
                effect_width_local=effect_width_local,
                effect_before_local=effect_before_local,
                effect_after_local=effect_after_local,
                executable_pages=executable_pages,
            )
        elif semantic in _INCDEC_SEMANTICS:
            reason = _emit_increment_decrement(
                builder, instruction,
                left_local=left_local, result_local=result_local,
                flags_local=flags_local,
                guest_base=guest_base, guest_size=guest_size,
                specialization_state=(specialization_state if index == 0 else None),
                effect_kind_local=effect_kind_local,
                effect_address_local=effect_address_local,
                effect_width_local=effect_width_local,
                effect_before_local=effect_before_local,
                effect_after_local=effect_after_local,
                executable_pages=executable_pages,
            )
        elif semantic is MachineSemanticToken.CONDITIONAL_SET:
            reason = _emit_conditional_set(
                builder, instruction, result_local=result_local,
                guest_base=guest_base, guest_size=guest_size,
                specialization_state=(specialization_state if index == 0 else None),
                effect_kind_local=effect_kind_local,
                effect_address_local=effect_address_local,
                effect_width_local=effect_width_local,
                effect_before_local=effect_before_local,
                effect_after_local=effect_after_local,
                executable_pages=executable_pages,
            )
        elif semantic is MachineSemanticToken.CONDITIONAL_MOVE:
            reason = _emit_conditional_move(
                builder, instruction,
                left_local=left_local, right_local=right_local,
                result_local=result_local,
                guest_base=guest_base, guest_size=guest_size,
                specialization_state=(specialization_state if index == 0 else None),
                effect_kind_local=effect_kind_local,
                effect_address_local=effect_address_local,
                effect_width_local=effect_width_local,
                effect_before_local=effect_before_local,
                effect_after_local=effect_after_local,
            )
        elif semantic is MachineSemanticToken.EXCHANGE:
            reason = _emit_exchange(
                builder, instruction,
                left_local=left_local, right_local=right_local,
                result_local=result_local,
                guest_base=guest_base, guest_size=guest_size,
                specialization_state=(specialization_state if index == 0 else None),
                effect_kind_local=effect_kind_local,
                effect_address_local=effect_address_local,
                effect_width_local=effect_width_local,
                effect_before_local=effect_before_local,
                effect_after_local=effect_after_local,
                executable_pages=executable_pages,
            )
        elif semantic is MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED:
            reason = _emit_multiply_unsigned(
                builder, instruction,
                left_local=left_local, right_local=right_local,
                result_local=result_local, high_local=high_local,
                work_local=work_local, flags_local=flags_local,
                guest_base=guest_base, guest_size=guest_size,
                specialization_state=(specialization_state if index == 0 else None),
                effect_kind_local=effect_kind_local,
                effect_address_local=effect_address_local,
                effect_width_local=effect_width_local,
                effect_before_local=effect_before_local,
                effect_after_local=effect_after_local,
            )
        elif semantic in _VECTOR_SEMANTICS:
            reason = _emit_vector_operation(
                builder, instruction,
                left_local=left_local, right_local=right_local,
                result_local=result_local, high_local=high_local,
                work_local=work_local,
                guest_base=guest_base, guest_size=guest_size,
                specialization_state=(specialization_state if index == 0 else None),
                effect_kind_local=effect_kind_local,
                effect_address_local=effect_address_local,
                effect_width_local=effect_width_local,
                effect_before_local=effect_before_local,
                effect_before_high_local=effect_before_high_local,
                effect_after_local=effect_after_local,
                effect_after_high_local=effect_after_high_local,
                executable_pages=executable_pages,
            )
        elif semantic in {
            MachineSemanticToken.DIRECT_RELATIVE_JUMP,
            MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        }:
            control_targets = _control_targets(instruction)
            reason = _emit_control(builder, instruction)
        elif semantic in {
            MachineSemanticToken.DIRECT_RELATIVE_CALL,
            MachineSemanticToken.INDIRECT_CALL,
            MachineSemanticToken.RETURN,
        }:
            if index:
                reason = "specialized call/return must begin its compiled block"
            else:
                reason, control_targets = _specialized_call_return(
                    builder, instruction, specialization_state,
                    guest_base=guest_base, guest_size=guest_size,
                    effect_kind_local=effect_kind_local,
                    effect_address_local=effect_address_local,
                    effect_width_local=effect_width_local,
                    effect_before_local=effect_before_local,
                    effect_after_local=effect_after_local,
                    stack_kind_local=stack_kind_local,
                    stack_value_local=stack_value_local,
                    stack_depth_local=stack_depth_local,
                    resolved_indirect_target=resolved_indirect_target,
                    indirect_external=indirect_external,
                )
                if reason is None:
                    if semantic in {
                        MachineSemanticToken.DIRECT_RELATIVE_CALL,
                        MachineSemanticToken.INDIRECT_CALL,
                    }:
                        stack_value = int(instruction.address) + len(instruction.encoded)
                        stack_kind = 1
                    else:
                        stack_value = int(specialization_state.call_stack[-1])
                        stack_kind = 2
                    expected_stack_effect = (
                        stack_kind, stack_value & MASK64,
                        len(specialization_state.call_stack),
                    )
        elif semantic is MachineSemanticToken.INDIRECT_JUMP:
            if index:
                reason = "specialized indirect jump must begin its compiled block"
            else:
                reason, control_targets = _specialized_indirect_jump(
                    builder, instruction, specialization_state,
                    resolved_indirect_target,
                    indirect_external=indirect_external,
                )
        elif semantic in {
            MachineSemanticToken.STACK_PUSH,
            MachineSemanticToken.STACK_POP,
        }:
            if index:
                reason = "specialized stack operation must begin its compiled block"
            else:
                reason = _specialized_stack_data(
                    builder, instruction, specialization_state,
                    guest_base=guest_base, guest_size=guest_size,
                    result_local=result_local,
                    effect_kind_local=effect_kind_local,
                    effect_address_local=effect_address_local,
                    effect_width_local=effect_width_local,
                    effect_before_local=effect_before_local,
                    effect_after_local=effect_after_local,
                )
        elif semantic is not MachineSemanticToken.NO_OPERATION:
            reason = "semantic is not in the reversible register-only Wasm tier"
        if reason is not None:
            shortfalls.append(MachineBlockLoweringShortfall(
                index, int(instruction.address), semantic.name, reason,
            ))
            break
        fallthrough = int(instruction.address) + len(instruction.encoded)
        next_addresses = control_targets or (fallthrough,)
        continuation = (
            next_addresses[0] if len(next_addresses) == 1 else -1
        )
        if control_targets is None:
            _emit_scalar_store(builder, 0, PC_OFFSET, continuation)
        _address(builder, 0, STEPS_OFFSET)
        _address(builder, 0, STEPS_OFFSET)
        builder.i64_load().i64_const(1).raw(OP_I64_ADD).i64_store()
        encoded_digest = sha256(bytes(instruction.encoded)).hexdigest()
        witness = MachineBlockInstructionWitness(
            index, int(instruction.address), semantic.name, int(semantic),
            bytes(instruction.encoded), encoded_digest, index * JOURNAL_STRIDE,
            tuple(next_addresses), expected_stack_effect,
        )
        witnesses.append(witness)
        compiled_instructions.append(instruction)
        _emit_checkpoint(
            builder, witness,
            effect_kind_local=effect_kind_local,
            effect_address_local=effect_address_local,
            effect_width_local=effect_width_local,
            effect_before_local=effect_before_local,
            effect_before_high_local=effect_before_high_local,
            effect_after_local=effect_after_local,
            effect_after_high_local=effect_after_high_local,
            stack_kind_local=stack_kind_local,
            stack_value_local=stack_value_local,
            stack_depth_local=stack_depth_local,
        )
    shortfalls.sort(key=lambda item: item.operation_index)
    if not witnesses:
        detail = shortfalls[0].reason if shortfalls else "block contains no operations"
        raise MachineBlockLoweringError(f"machine block has no Wasm-safe prefix: {detail}")
    if strict and shortfalls:
        item = shortfalls[0]
        raise MachineBlockLoweringError(
            f"machine block lowering stopped at {item.address:#x}: {item.reason}"
        )
    journal_bytes = len(witnesses) * JOURNAL_STRIDE
    guest_buffer_offset = (
        (1024 + journal_bytes + 4095) // 4096
    ) * 4096
    memory_limit = max(
        STATE_SIZE,
        1024 + journal_bytes,
        guest_buffer_offset + guest_size,
    )
    memory_pages = max(1, math.ceil(memory_limit / 65536))
    binary = build_module(
        function_name="run", parameter_types=("i32", "i32", "i32"), body=builder,
        memory_pages=memory_pages,
    )
    abi = MappingProxyType({
        "schema": MACHINE_BLOCK_STATE_SCHEMA,
        "journal_schema": MACHINE_BLOCK_JOURNAL_SCHEMA,
        "register_count": REGISTER_COUNT,
        "vector_register_count": VECTOR_REGISTER_COUNT,
        "pc_offset": PC_OFFSET,
        "flags_offset": FLAGS_OFFSET,
        "steps_offset": STEPS_OFFSET,
        "vector_offset": VECTOR_OFFSET,
        "state_size": STATE_SIZE,
        "journal_state_offset": JOURNAL_STATE_OFFSET,
        "journal_effect_offset": JOURNAL_EFFECT_OFFSET,
        "journal_stack_offset": JOURNAL_STACK_OFFSET,
        "journal_stride": JOURNAL_STRIDE,
        "guest_memory_base": guest_base,
        "guest_memory_size": guest_size,
        "guest_buffer_offset": guest_buffer_offset,
        "memory_pages": memory_pages,
    })
    possible_continuations = witnesses[-1].possible_next_addresses
    return MachineBlockWasmArtifact(
        int(block.entry_address), str(block.code_digest), binary,
        _wat_for(
            tuple(compiled_instructions), tuple(witnesses), continuation,
            guest_base, specialization_state, memory_pages,
            resolved_indirect_target,
        ),
        tuple(witnesses),
        tuple(shortfalls), continuation, possible_continuations,
        guest_base, guest_size, abi,
        MappingProxyType(dict(specialization_guard)),
    )


__all__ = [
    "FLAGS_OFFSET", "JOURNAL_STACK_OFFSET", "JOURNAL_STATE_OFFSET", "JOURNAL_STRIDE",
    "MachineBlockInstructionWitness", "MachineBlockLoweringError",
    "MachineBlockLoweringShortfall", "MachineBlockWasmArtifact",
    "PC_OFFSET", "REGISTER_COUNT", "STATE_SIZE", "STEPS_OFFSET",
    "lower_machine_block_to_wasm",
]
