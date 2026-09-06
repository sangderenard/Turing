"""Symbolic architectural effects for the internal AMD64 translated program.

The machine token graph remains the owning program representation and
``MachineTranslatedOperation`` remains the executable VM operation.  This
module adds the static resource contract needed to convert those operations
to SSA without executing a concrete path.  It deliberately describes machine
resources rather than guessing source-language values.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .machine_reference_vocabulary import (
    EffectiveAddressOperand,
    HighByteRegisterOperand,
    ImmediateOperand,
    MachineSemanticToken,
    RegisterOperand,
    RelativeAddressOperand,
    VectorRegisterOperand,
)


@dataclass(frozen=True, slots=True)
class MachineSymbolicEffect:
    """One instruction's complete static architectural resource contract."""

    semantic: MachineSemanticToken
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    effect_domains: tuple[str, ...]
    may_trap: bool = False
    conditional: bool = False


@dataclass(frozen=True, slots=True)
class MachineSymbolicSSAValue:
    """One version of an architectural resource on a symbolic path."""

    resource: str
    version: int

    @property
    def identity(self) -> str:
        return f"{self.resource}@{self.version}"


@dataclass(frozen=True, slots=True)
class MachineSymbolicSSAOperation:
    """One internal VM operation with explicit resource-version edges."""

    address: int
    semantic: MachineSemanticToken
    instruction: object
    inputs: tuple[MachineSymbolicSSAValue, ...]
    outputs: tuple[MachineSymbolicSSAValue, ...]
    effect: MachineSymbolicEffect


@dataclass(frozen=True, slots=True)
class MachineSymbolicSSABlock:
    """A translated basic block expressed without choosing concrete values."""

    entry_address: int
    operations: tuple[MachineSymbolicSSAOperation, ...]
    initial_values: Mapping[str, MachineSymbolicSSAValue]
    final_values: Mapping[str, MachineSymbolicSSAValue]


_BINARY_WRITE = frozenset({
    MachineSemanticToken.INTEGER_ADD,
    MachineSemanticToken.INTEGER_SUBTRACT,
    MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW,
    MachineSemanticToken.BITWISE_AND,
    MachineSemanticToken.BITWISE_OR,
    MachineSemanticToken.BITWISE_XOR,
    MachineSemanticToken.SHIFT_LEFT,
    MachineSemanticToken.SHIFT_RIGHT_LOGICAL,
    MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC,
    MachineSemanticToken.ROTATE_LEFT,
    MachineSemanticToken.ROTATE_RIGHT,
    MachineSemanticToken.ATOMIC_ADD,
})
_BINARY_FLAGS_ONLY = frozenset({
    MachineSemanticToken.INTEGER_COMPARE,
    MachineSemanticToken.INTEGER_TEST,
    MachineSemanticToken.BIT_TEST,
})
_UNARY_WRITE = frozenset({
    MachineSemanticToken.INTEGER_NEGATE,
    MachineSemanticToken.BITWISE_NOT,
    MachineSemanticToken.INTEGER_INCREMENT,
    MachineSemanticToken.INTEGER_DECREMENT,
    MachineSemanticToken.ATOMIC_INCREMENT,
})
_MOVE = frozenset({
    MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
    MachineSemanticToken.REGISTER_OR_MEMORY_READ,
    MachineSemanticToken.REGISTER_WRITE_IMMEDIATE,
    MachineSemanticToken.SIGN_EXTEND,
    MachineSemanticToken.ZERO_EXTEND,
    MachineSemanticToken.VECTOR_MOVE,
})
_CONDITIONAL = frozenset({
    MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
    MachineSemanticToken.CONDITIONAL_MOVE,
    MachineSemanticToken.CONDITIONAL_SET,
})
_CONTROL = frozenset({
    MachineSemanticToken.RETURN,
    MachineSemanticToken.DIRECT_RELATIVE_CALL,
    MachineSemanticToken.DIRECT_RELATIVE_JUMP,
    MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
    MachineSemanticToken.INDIRECT_CALL,
    MachineSemanticToken.INDIRECT_JUMP,
    MachineSemanticToken.BREAKPOINT_TRAP,
    MachineSemanticToken.SOFTWARE_INTERRUPT,
})
_FLAGS_WRITERS = frozenset({
    *_BINARY_WRITE,
    *_BINARY_FLAGS_ONLY,
    MachineSemanticToken.INTEGER_NEGATE,
    MachineSemanticToken.INTEGER_INCREMENT,
    MachineSemanticToken.INTEGER_DECREMENT,
    MachineSemanticToken.INTEGER_MULTIPLY,
    MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED,
    MachineSemanticToken.ATOMIC_COMPARE_EXCHANGE,
    MachineSemanticToken.ATOMIC_EXCHANGE_ADD,
    MachineSemanticToken.BIT_TEST_RESET,
    MachineSemanticToken.BIT_TEST_COMPLEMENT,
    MachineSemanticToken.ATOMIC_INCREMENT,
})


def _unique(items) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in items))


def _address_reads(instruction, operand: EffectiveAddressOperand) -> tuple[str, ...]:
    reads = []
    if operand.rip_relative:
        reads.append("control.rip")
    if operand.base is not None:
        reads.append(f"register.{operand.base.name.lower()}")
    if operand.index is not None:
        reads.append(f"register.{operand.index.name.lower()}")
    if 0x64 in instruction.legacy_prefixes:
        reads.append("register.fs_base")
    if 0x65 in instruction.legacy_prefixes:
        reads.append("register.gs_base")
    return tuple(reads)


def _operand_reads(instruction, index: int) -> tuple[str, ...]:
    operand = instruction.operands[index]
    if isinstance(operand, RegisterOperand):
        return (f"register.{operand.register.name.lower()}",)
    if isinstance(operand, HighByteRegisterOperand):
        owners = ("rax", "rcx", "rdx", "rbx")
        return (f"register.{owners[int(operand.register)]}",)
    if isinstance(operand, VectorRegisterOperand):
        return (f"vector.{operand.register.name.lower()}",)
    if isinstance(operand, EffectiveAddressOperand):
        return (*_address_reads(instruction, operand), "memory")
    if isinstance(operand, (ImmediateOperand, RelativeAddressOperand)):
        return ()
    raise TypeError(f"unsupported symbolic machine operand {operand!r}")


def _operand_writes(instruction, index: int) -> tuple[str, ...]:
    operand = instruction.operands[index]
    if isinstance(operand, RegisterOperand):
        return (f"register.{operand.register.name.lower()}",)
    if isinstance(operand, HighByteRegisterOperand):
        owners = ("rax", "rcx", "rdx", "rbx")
        return (f"register.{owners[int(operand.register)]}",)
    if isinstance(operand, VectorRegisterOperand):
        return (f"vector.{operand.register.name.lower()}",)
    if isinstance(operand, EffectiveAddressOperand):
        return ("memory",)
    raise TypeError(f"machine destination is not writable: {operand!r}")


def symbolic_effect_for_instruction(instruction) -> MachineSymbolicEffect:
    """Derive a non-executing effect contract from semantic and operand roles."""

    semantic = MachineSemanticToken(instruction.semantic)
    reads: list[str] = []
    writes: list[str] = []
    may_trap = semantic in {
        MachineSemanticToken.INTEGER_DIVIDE,
        MachineSemanticToken.INTEGER_DIVIDE_SIGNED,
        MachineSemanticToken.BREAKPOINT_TRAP,
        MachineSemanticToken.SOFTWARE_INTERRUPT,
    }

    def read(index: int) -> None:
        reads.extend(_operand_reads(instruction, index))

    def write(index: int) -> None:
        operand = instruction.operands[index]
        if isinstance(operand, EffectiveAddressOperand):
            reads.extend(_address_reads(instruction, operand))
        writes.extend(_operand_writes(instruction, index))

    if semantic in _BINARY_WRITE:
        read(0)
        read(1)
        write(0)
        if semantic is MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW:
            reads.append("flags")
    elif semantic in _BINARY_FLAGS_ONLY:
        read(0)
        read(1)
    elif semantic in _UNARY_WRITE:
        read(0)
        write(0)
        if semantic in {
            MachineSemanticToken.INTEGER_INCREMENT,
            MachineSemanticToken.INTEGER_DECREMENT,
            MachineSemanticToken.ATOMIC_INCREMENT,
        }:
            reads.append("flags")  # CF is preserved.
    elif semantic in _MOVE:
        if len(instruction.operands) > 1:
            read(1)
        write(0)
    elif semantic is MachineSemanticToken.EFFECTIVE_ADDRESS:
        source = instruction.operands[1]
        if not isinstance(source, EffectiveAddressOperand):
            raise TypeError("effective-address semantic requires an address operand")
        reads.extend(_address_reads(instruction, source))
        write(0)
    elif semantic is MachineSemanticToken.NO_OPERATION:
        pass
    elif semantic is MachineSemanticToken.STACK_PUSH:
        read(0)
        reads.extend(("register.rsp", "memory"))
        writes.extend(("register.rsp", "memory"))
    elif semantic is MachineSemanticToken.STACK_POP:
        reads.extend(("register.rsp", "memory"))
        write(0)
        writes.append("register.rsp")
    elif semantic is MachineSemanticToken.INTEGER_MULTIPLY:
        if len(instruction.operands) == 1:
            read(0)
            reads.append("register.rax")
            writes.extend(("register.rax", "register.rdx"))
        elif len(instruction.operands) == 2:
            # Two-operand IMUL reads and overwrites its destination.
            read(0)
            read(1)
        elif len(instruction.operands) == 3:
            # Three-operand IMUL's destination is not an input.
            read(1)
            read(2)
        else:
            raise ValueError("IMUL symbolic contract requires one, two or three operands")
        if len(instruction.operands) != 1:
            write(0)
    elif semantic is MachineSemanticToken.VECTOR_XOR:
        read(0)
        read(1)
        write(0)
    elif semantic is MachineSemanticToken.VECTOR_AND:
        read(0)
        read(1)
        write(0)
    elif semantic is MachineSemanticToken.VECTOR_INSERT_128_LANE:
        read(1)
        read(2)
        read(3)
        write(0)
    elif semantic is MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED:
        read(0)
        reads.append("register.rax")
        writes.extend(("register.rax", "register.rdx"))
    elif semantic in {MachineSemanticToken.INTEGER_DIVIDE, MachineSemanticToken.INTEGER_DIVIDE_SIGNED}:
        read(0)
        reads.extend(("register.rax", "register.rdx"))
        writes.extend(("register.rax", "register.rdx"))
    elif semantic is MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR:
        reads.append("register.rax")
        writes.extend(("register.rax", "register.rdx"))
    elif semantic in {MachineSemanticToken.CONDITIONAL_MOVE, MachineSemanticToken.CONDITIONAL_SET}:
        reads.append("flags")
        if semantic is MachineSemanticToken.CONDITIONAL_MOVE:
            read(0)
            read(1)
        write(0)
    elif semantic in {MachineSemanticToken.BIT_TEST_RESET, MachineSemanticToken.BIT_TEST_COMPLEMENT}:
        read(0)
        read(1)
        write(0)
    elif semantic is MachineSemanticToken.BIT_SCAN_REVERSE:
        # Destination preservation is the VM's deterministic representative
        # of the architecture's undefined zero-source result.
        read(0)
        read(1)
        write(0)
    elif semantic in {
        MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT64,
        MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT32,
        MachineSemanticToken.VECTOR_MOVE_LOW_ZERO_UPPER,
    }:
        read(1)
        if semantic in {
            MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT64,
            MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT32,
        }:
            # The legacy scalar conversion preserves the destination's upper
            # lane, so it is an input as well as an output.
            read(0)
            reads.append("system.amd64.mxcsr")
            writes.append("system.amd64.mxcsr")
        write(0)
    elif semantic in {
        MachineSemanticToken.SCALAR_FLOAT64_COMPARE_UNORDERED,
        MachineSemanticToken.SCALAR_FLOAT64_COMPARE_ORDERED,
        MachineSemanticToken.SCALAR_FLOAT32_COMPARE_ORDERED,
    }:
        read(0)
        read(1)
        reads.append("system.amd64.mxcsr")
        writes.append("system.amd64.mxcsr")
        writes.append("flags")
    elif semantic in {
        MachineSemanticToken.SCALAR_FLOAT64_ADD,
        MachineSemanticToken.SCALAR_FLOAT32_ADD,
        MachineSemanticToken.SCALAR_FLOAT32_DIVIDE,
        MachineSemanticToken.SCALAR_FLOAT64_DIVIDE,
        MachineSemanticToken.SCALAR_FLOAT64_SUBTRACT,
        MachineSemanticToken.SCALAR_FLOAT64_MULTIPLY,
        MachineSemanticToken.VECTOR_UNPACK_LOW_QWORDS,
        MachineSemanticToken.VECTOR_UNPACK_LOW_BYTES,
        MachineSemanticToken.VECTOR_UNPACK_LOW_WORDS,
        MachineSemanticToken.VECTOR_ADD_QWORDS,
        MachineSemanticToken.VECTOR_COMPARE_EQUAL_QWORDS,
        MachineSemanticToken.VECTOR_SUBTRACT_QWORDS,
    }:
        read(0)
        read(1)
        if semantic in {
            MachineSemanticToken.SCALAR_FLOAT64_ADD,
            MachineSemanticToken.SCALAR_FLOAT32_ADD,
            MachineSemanticToken.SCALAR_FLOAT32_DIVIDE,
            MachineSemanticToken.SCALAR_FLOAT64_DIVIDE,
            MachineSemanticToken.SCALAR_FLOAT64_SUBTRACT,
            MachineSemanticToken.SCALAR_FLOAT64_MULTIPLY,
        }:
            reads.append("system.amd64.mxcsr")
            writes.append("system.amd64.mxcsr")
        write(0)
    elif semantic is MachineSemanticToken.BYTE_SWAP:
        read(0)
        write(0)
    elif semantic is MachineSemanticToken.SCALAR_FLOAT64_TO_SIGNED_INT64_TRUNCATE:
        read(1)
        reads.append("system.amd64.mxcsr")
        writes.append("system.amd64.mxcsr")
        write(0)
    elif semantic is MachineSemanticToken.SCALAR_FLOAT64_TO_SIGNED_INT32_TRUNCATE:
        read(1)
        reads.append("system.amd64.mxcsr")
        writes.append("system.amd64.mxcsr")
        write(0)
    elif semantic is MachineSemanticToken.VECTOR_SHUFFLE_DWORDS:
        read(1)
        write(0)
    elif semantic is MachineSemanticToken.VECTOR_SIGNED_INT32_TO_FLOAT64:
        read(1)
        reads.append("system.amd64.mxcsr")
        writes.append("system.amd64.mxcsr")
        write(0)
    elif semantic is MachineSemanticToken.ATOMIC_COMPARE_EXCHANGE:
        read(0)
        read(1)
        reads.append("register.rax")
        write(0)
        writes.append("register.rax")
    elif semantic in {MachineSemanticToken.ATOMIC_EXCHANGE_ADD, MachineSemanticToken.EXCHANGE}:
        read(0)
        read(1)
        write(0)
        write(1)
    elif semantic is MachineSemanticToken.VECTOR_SHIFT_RIGHT_LOGICAL:
        read(0)
        read(1)
        write(0)
    elif semantic is MachineSemanticToken.STRING_STORE:
        reads.extend(("register.rdi", "register.rax", "flags", "memory"))
        writes.extend(("register.rdi", "memory"))
        if instruction.token.name in {"REP_STOSW", "REP_STOSB"}:
            reads.append("register.rcx")
            writes.append("register.rcx")
    elif semantic is MachineSemanticToken.STRING_MOVE:
        reads.extend(("register.rdi", "register.rsi", "register.rcx", "flags", "memory"))
        writes.extend(("register.rdi", "register.rsi", "register.rcx", "memory"))
    elif semantic is MachineSemanticToken.STRING_COMPARE:
        reads.extend(("register.rdi", "register.rax", "flags", "memory"))
        writes.extend(("register.rdi", "flags"))
    elif semantic in _CONTROL:
        reads.append("control.rip")
        if semantic in _CONDITIONAL:
            reads.append("flags")
        if semantic in {MachineSemanticToken.INDIRECT_CALL, MachineSemanticToken.INDIRECT_JUMP}:
            read(0)
        if semantic in {MachineSemanticToken.DIRECT_RELATIVE_CALL, MachineSemanticToken.INDIRECT_CALL}:
            reads.extend(("register.rsp", "memory"))
            writes.extend(("register.rsp", "memory", "control.call_stack"))
        elif semantic is MachineSemanticToken.RETURN:
            reads.extend(("register.rsp", "memory", "control.call_stack"))
            writes.extend(("register.rsp", "control.call_stack"))
    else:
        raise ValueError(f"no symbolic machine effect rule for {semantic.name}")

    if semantic in _FLAGS_WRITERS:
        writes.append("flags")
    writes.append("control.rip")
    domains = {
        resource.split(".", 1)[0] if "." in resource else resource
        for resource in (*reads, *writes)
    }
    return MachineSymbolicEffect(
        semantic,
        _unique(reads),
        _unique(writes),
        tuple(sorted(domains)),
        may_trap=may_trap,
        conditional=semantic in _CONDITIONAL,
    )


def translated_symbolic_effect(instruction) -> MachineSymbolicEffect | None:
    """Return a contract when an operation carries structurally valid operands.

    Embedders may install synthetic operations with custom effect handlers and
    intentionally omit decoded operands.  Such operations remain executable,
    but cannot honestly claim a statically derived architectural contract.
    Decoder-produced instructions always have the required operand structure.
    """

    try:
        return symbolic_effect_for_instruction(instruction)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def translated_block_to_symbolic_ssa(block) -> MachineSymbolicSSABlock:
    """Version all resources read and written by one translated VM block."""

    versions: dict[str, int] = {}
    current: dict[str, MachineSymbolicSSAValue] = {}
    initial: dict[str, MachineSymbolicSSAValue] = {}
    operations = []

    def value(resource: str) -> MachineSymbolicSSAValue:
        active = current.get(resource)
        if active is None:
            active = MachineSymbolicSSAValue(resource, 0)
            current[resource] = active
            initial[resource] = active
            versions[resource] = 0
        return active

    for translated in block.operations:
        effect = translated.symbolic_effect
        if effect is None:
            raise ValueError(
                f"translated operation at {int(translated.address):#x} has no "
                "derivable symbolic effect contract"
            )
        inputs = tuple(value(resource) for resource in effect.reads)
        outputs = []
        for resource in effect.writes:
            # A write is also ordered after the preceding resource version,
            # even for a destination whose data value is wholly overwritten.
            prior = value(resource)
            if prior not in inputs:
                inputs = (*inputs, prior)
            version = versions[resource] + 1
            versions[resource] = version
            active = MachineSymbolicSSAValue(resource, version)
            current[resource] = active
            outputs.append(active)
        operations.append(MachineSymbolicSSAOperation(
            int(translated.address), effect.semantic, translated.instruction,
            inputs, tuple(outputs), effect,
        ))
    return MachineSymbolicSSABlock(
        int(block.entry_address), tuple(operations),
        MappingProxyType(dict(initial)), MappingProxyType(dict(current)),
    )


def changed_architectural_resources(before, after) -> tuple[str, ...]:
    """Name exact architectural state changes visible to symbolic machine SSA."""

    changed = []
    register_names = getattr(before, "REGISTER_NAMES", ())
    for name, left, right in zip(register_names, before.registers, after.registers):
        if left != right:
            changed.append(f"register.{name}")
    for index, (left, right) in enumerate(zip(before.vector_registers, after.vector_registers)):
        if left != right:
            changed.append(f"vector.xmm{index}")
    for name in ("fs_base", "gs_base"):
        if getattr(before, name) != getattr(after, name):
            changed.append(f"register.{name}")
    if before.flags != after.flags:
        changed.append("flags")
    if before.memory != after.memory:
        changed.append("memory")
    if before.pc != after.pc:
        changed.append("control.rip")
    if before.call_stack != after.call_stack:
        changed.append("control.call_stack")
    return tuple(changed)


def validate_symbolic_transition(effect: MachineSymbolicEffect, before, after) -> tuple[str, ...]:
    """Reject a VM transition that changes undeclared architectural resources."""

    changed = changed_architectural_resources(before, after)
    undeclared = tuple(resource for resource in changed if resource not in effect.writes)
    if undeclared:
        raise ValueError(
            f"{effect.semantic.name} changed undeclared machine resources: "
            + ", ".join(undeclared)
        )
    return changed


__all__ = [
    "MachineSymbolicEffect", "MachineSymbolicSSABlock",
    "MachineSymbolicSSAOperation", "MachineSymbolicSSAValue",
    "changed_architectural_resources", "symbolic_effect_for_instruction",
    "translated_block_to_symbolic_ssa", "translated_symbolic_effect",
    "validate_symbolic_transition",
]
