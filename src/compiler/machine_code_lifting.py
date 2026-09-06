"""Small, evidence-preserving machine-code lifting primitives.

This is deliberately a prototype, not a general x86 decompiler. The primary
entry point consumes bytes through the repository's executable reference
vocabulary; the older objdump parser remains only as a compatibility fixture.
Unknown instructions fail closed instead of being copied through as guessed
semantics.

The useful architectural seam is already general: decoded instructions become
the repository's ordinary ``Function``/``BasicBlock``/``Instr`` SSA objects,
and a data-flow multigraph retains repeated operand edges.  More instruction
vocabularies can be added without changing the graph or similarity contracts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import operator
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable, Mapping, Sequence

import networkx as nx

from ..common.tensors import AbstractTensor
from ..transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
from ..transmogrifier.ssa_registry import Handler
from .machine_reference_vocabulary import (
    DecodeReport,
    DecodedInstruction,
    EffectiveAddressOperand,
    ImmediateOperand,
    MachineSemanticToken,
    RegisterOperand,
    RelativeAddressOperand,
    VectorRegisterOperand,
    VocabularyAuditReport,
    VocabularyDecodeError,
    VocabularyFailure,
    X86InstructionToken,
    X86ReferenceDecoder,
    X86Register,
    X86VectorRegister,
)


def _machine_group_fingerprints(function: Function) -> tuple[tuple[int, str], ...]:
    """Hash the complete ordered SSA expansion authored by each instruction."""

    grouped: dict[int, list[tuple[object, ...]]] = {}
    for block_name, block in function.blocks.items():
        for instruction in block.instrs:
            address = instruction.attributes.get("machine_address")
            if address is None:
                continue
            grouped.setdefault(int(address), []).append((
                str(block_name), str(instruction.op),
                tuple(int(arg.id) for arg in instruction.args),
                None if instruction.res is None else int(instruction.res.id),
                str(instruction.res.dtype) if instruction.res is not None else None,
                tuple(sorted(
                    (str(key), repr(value))
                    for key, value in instruction.attributes.items()
                )),
            ))
    return tuple(
        (
            address,
            sha256(repr(tuple(grouped[address])).encode("utf-8")).hexdigest(),
        )
        for address in sorted(grouped)
    )


def _stamp_machine_group_fingerprints(function: Function) -> Function:
    function.metadata["machine_group_fingerprints"] = (
        _machine_group_fingerprints(function)
    )
    function.metadata["machine_block_addresses"] = tuple(
        (
            str(block_name),
            tuple(dict.fromkeys(
                int(instruction.attributes["machine_address"])
                for instruction in block.instrs
                if instruction.attributes.get("machine_address") is not None
            )),
        )
        for block_name, block in function.blocks.items()
    )
    return function
from .machine_dialect_ssa import decoded_function_to_machine_ssa
from .evolution_metagraph import TokenPathAtlas


class MachineLiftError(ValueError):
    """The decoded program is outside the honest prototype vocabulary."""


@dataclass(frozen=True, slots=True)
class VocabularyStatistics:
    """Coverage measurements for one bounded binary-region ingestion."""

    region_capacity: int
    accepted_size: int
    decoded_bytes: int
    instruction_count: int
    failed_vocabulary_count: int
    byte_coverage: float
    byte_sum: int
    valid_byte_count: int
    instruction_token_counts: tuple[tuple[int, int], ...]
    semantic_token_counts: tuple[tuple[int, int], ...]
    stopped_at_return: bool
    tensor_math_used: bool
    tensor_math_error: str | None
    diagnostic_known_bytes: int = 0
    diagnostic_missing_bytes: int = 0
    diagnostic_gap_count: int = 0
    diagnostic_candidate_instruction_count: int = 0


@dataclass(frozen=True, slots=True)
class BinaryToSSAResult:
    """One-shot result from bounded machine bytes to repository SSA."""

    function: Function | None
    decoded: tuple[DecodedInstruction, ...]
    failed_vocabulary: tuple[VocabularyFailure, ...]
    statistics: VocabularyStatistics
    vocabulary_audit: VocabularyAuditReport | None = None

    @property
    def complete(self) -> bool:
        return self.function is not None and not self.failed_vocabulary


@dataclass(frozen=True)
class MachineInstruction:
    address: int
    encoded: bytes
    mnemonic: str
    operands: str


@dataclass(frozen=True)
class MachineFunction:
    name: str
    instructions: tuple[MachineInstruction, ...]


@dataclass(frozen=True)
class TopologyProfile:
    nodes: int
    edges: int
    components: int
    cycle_rank: int
    sources: int
    sinks: int
    branches: int
    merges: int
    degrees: tuple[tuple[int, int], ...]


_FUNCTION = re.compile(r"^([0-9a-fA-F]+) <([^>]+)>:$")
_INSTRUCTION = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+"
    r"((?:[0-9a-fA-F]{2}\s+)+)"
    r"([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*$"
)
_REGISTER_ALIASES = {
    "eax": "rax", "ax": "rax", "al": "rax",
    "ecx": "rcx", "cx": "rcx", "cl": "rcx",
    "edx": "rdx", "dx": "rdx", "dl": "rdx",
    "r8d": "r8", "r9d": "r9",
}


def parse_objdump_function(text: str, name: str) -> MachineFunction:
    """Read one function from GNU objdump's Intel-syntax disassembly."""

    active = False
    instructions: list[MachineInstruction] = []
    for raw_line in str(text).splitlines():
        header = _FUNCTION.match(raw_line.strip())
        if header:
            if active:
                break
            active = header.group(2) == name
            continue
        if not active:
            continue
        match = _INSTRUCTION.match(raw_line)
        if match is None:
            continue
        mnemonic = match.group(3).lower()
        if mnemonic == "nop":
            continue
        instructions.append(MachineInstruction(
            address=int(match.group(1), 16),
            encoded=bytes.fromhex(match.group(2)),
            mnemonic=mnemonic,
            operands=match.group(4).strip(),
        ))
    if not instructions:
        raise MachineLiftError(f"objdump contains no decoded function {name!r}")
    return MachineFunction(name, tuple(instructions))


def disassemble_gnu_object(path: str | Path, *, objdump: str | None = None) -> str:
    """Disassemble an object with the GNU decoder in Intel syntax."""

    object_path = Path(path)
    executable = objdump or shutil.which("objdump")
    if executable is None:
        adjacent = Path(r"C:\msys64\mingw64\bin\objdump.exe")
        executable = str(adjacent) if adjacent.exists() else None
    if executable is None:
        raise MachineLiftError("GNU objdump is required for machine decoding")
    completed = subprocess.run(
        [executable, "-d", "-M", "intel", str(object_path)],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise MachineLiftError(completed.stderr or completed.stdout)
    return completed.stdout


def _register(name: str) -> str:
    lowered = name.strip().lower()
    return _REGISTER_ALIASES.get(lowered, lowered)


class _AffineX86Lifter:
    def __init__(
        self,
        machine: MachineFunction,
        argument_registers: Sequence[str],
        argument_names: Sequence[str],
    ):
        if len(argument_registers) != len(argument_names):
            raise ValueError("argument registers and names must have equal length")
        self.machine = machine
        self.next_id = 0
        self.args: list[SSAValue] = []
        self.registers: dict[str, SSAValue] = {}
        self.instructions: list[Instr] = []
        for register, name in zip(argument_registers, argument_names):
            value = self.fresh()
            self.args.append(value)
            self.registers[_register(register)] = value

    def fresh(self) -> SSAValue:
        value = SSAValue(self.next_id, dtype="int32")
        self.next_id += 1
        return value

    def emit(
        self,
        handler: Handler,
        args: Iterable[SSAValue],
        source: MachineInstruction,
        **attributes,
    ) -> SSAValue:
        result = self.fresh()
        self.instructions.append(Instr(
            handler.value,
            list(args),
            result,
            attributes={
                "machine_address": source.address,
                "machine_mnemonic": source.mnemonic,
                "machine_bytes": source.encoded.hex(),
                **attributes,
            },
        ))
        return result

    def value(self, register: str, source: MachineInstruction) -> SSAValue:
        canonical = _register(register)
        try:
            return self.registers[canonical]
        except KeyError as error:
            raise MachineLiftError(
                f"{source.address:#x}: read of unknown register {register!r}"
            ) from error

    def lift_imul(self, source: MachineInstruction) -> None:
        operands = [item.strip() for item in source.operands.split(",")]
        if len(operands) != 2:
            raise MachineLiftError(
                f"{source.address:#x}: only two-operand imul is supported"
            )
        destination, right = operands
        left_value = self.value(destination, source)
        right_value = self.value(right, source)
        self.registers[_register(destination)] = self.emit(
            Handler.Mul, (left_value, right_value), source,
        )

    def lift_lea(self, source: MachineInstruction) -> None:
        match = re.fullmatch(r"([^,]+),\[([^]]+)]", source.operands.replace(" ", ""))
        if match is None:
            raise MachineLiftError(
                f"{source.address:#x}: unsupported lea form {source.operands!r}"
            )
        destination, expression = match.groups()
        terms: list[SSAValue] = []
        for token in expression.replace("-", "+-").split("+"):
            if not token:
                continue
            part = token.split("*")
            if len(part) > 2 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", part[0]):
                raise MachineLiftError(
                    f"{source.address:#x}: unsupported lea term {token!r}"
                )
            value = self.value(part[0], source)
            scale = int(part[1], 0) if len(part) == 2 else 1
            if scale != 1:
                literal = self.fresh()
                self.instructions.append(Instr(
                    Handler.Const.value, [], literal,
                    attributes={"value": scale, "machine_address": source.address},
                ))
                value = self.emit(Handler.Mul, (value, literal), source)
            terms.append(value)
        if not terms:
            raise MachineLiftError(f"{source.address:#x}: lea has no value terms")
        result = terms[0]
        for term in terms[1:]:
            result = self.emit(Handler.Add, (result, term), source)
        self.registers[_register(destination)] = result

    def finish(self) -> Function:
        for source in self.machine.instructions:
            if source.mnemonic == "imul":
                self.lift_imul(source)
            elif source.mnemonic == "lea":
                self.lift_lea(source)
            elif source.mnemonic == "ret":
                result = self.value("rax", source)
                self.instructions.append(Instr(
                    Handler.Ret.value,
                    [result],
                    None,
                    attributes={
                        "machine_address": source.address,
                        "machine_mnemonic": source.mnemonic,
                        "machine_bytes": source.encoded.hex(),
                    },
                ))
            else:
                raise MachineLiftError(
                    f"{source.address:#x}: unsupported machine instruction "
                    f"{source.mnemonic} {source.operands}".rstrip()
                )
        if not self.instructions or self.instructions[-1].op != Handler.Ret.value:
            raise MachineLiftError("prototype requires an explicit return")
        return Function(
            self.machine.name,
            self.args,
            {"entry": BasicBlock("entry", self.instructions)},
            metadata={
                "lifted_from": "x86_64-machine-code",
            },
        )


def lift_x86_64_affine_function(
    machine: MachineFunction,
    *,
    argument_registers: Sequence[str] = ("ecx", "edx", "r8d", "r9d"),
    argument_names: Sequence[str] = ("arg0", "arg1", "arg2", "arg3"),
) -> Function:
    """Lift the explicitly supported straight-line Win64 integer subset."""

    lifter = _AffineX86Lifter(machine, argument_registers, argument_names)
    function = lifter.finish()
    function.metadata["argument_names"] = tuple(argument_names)
    function.metadata["argument_registers"] = tuple(argument_registers)
    return function


def _as_x86_register(value: X86Register | str) -> X86Register:
    if isinstance(value, X86Register):
        return value
    name = _register(str(value)).upper()
    try:
        return X86Register[name]
    except KeyError as error:
        raise ValueError(f"unknown x86-64 argument register {value!r}") from error


class _StructuredX86Lifter:
    """Lower numeric machine tokens and structured operands into SSA."""

    def __init__(
        self,
        name: str,
        decoded: Sequence[DecodedInstruction],
        argument_registers: Sequence[X86Register | str],
        argument_names: Sequence[str],
    ) -> None:
        if len(argument_registers) != len(argument_names):
            raise ValueError("argument registers and names must have equal length")
        registers = tuple(_as_x86_register(item) for item in argument_registers)
        if len(set(registers)) != len(registers):
            raise ValueError("argument registers must be unique")
        if len(set(argument_names)) != len(argument_names):
            raise ValueError("argument names must be unique")
        self.name = str(name)
        self.decoded = tuple(decoded)
        self.vector_state_width = (
            256
            if any(
                isinstance(operand, VectorRegisterOperand)
                and int(operand.width) == 256
                for instruction in self.decoded
                for operand in instruction.operands
            )
            else 128
        )
        self.vector_state_dtype = f"int{self.vector_state_width}"
        self.next_id = 0
        self.args: list[SSAValue] = []
        self.registers: dict[X86Register, SSAValue] = {}
        self.instructions: list[Instr] = []
        self.argument_names = list(str(item) for item in argument_names)
        for register in registers:
            value = self.fresh()
            self.args.append(value)
            self.registers[register] = value
        self.argument_registers = list(registers)
        if X86Register.RSP not in self.registers:
            stack_pointer = self.fresh(dtype="int64")
            self.args.append(stack_pointer)
            self.registers[X86Register.RSP] = stack_pointer
            self.argument_registers.append(X86Register.RSP)
            self.argument_names.append("__machine_rsp")
        self.memory = self.fresh(dtype="memory")
        self.args.append(self.memory)
        self.argument_names.append("__machine_memory")
        self.pending_condition: SSAValue | None = None

    def fresh(self, *, dtype: str = "int32") -> SSAValue:
        value = SSAValue(self.next_id, dtype=dtype)
        self.next_id += 1
        return value

    @staticmethod
    def provenance(source: DecodedInstruction) -> dict[str, object]:
        return {
            "machine_address": source.address,
            "machine_token": int(source.token),
            "machine_semantic_token": int(source.semantic),
            "machine_bytes": source.encoded.hex(),
        }

    def emit(
        self,
        handler: Handler,
        args: Iterable[SSAValue],
        source: DecodedInstruction,
        dtype: str = "int32",
        **attributes: object,
    ) -> SSAValue:
        result = self.fresh(dtype=dtype)
        self.instructions.append(Instr(
            handler.value,
            list(args),
            result,
            attributes={**self.provenance(source), **attributes},
        ))
        return result

    def constant(
        self,
        value: int,
        source: DecodedInstruction,
        *,
        dtype: str = "int32",
        machine_operand_role: str | None = None,
    ) -> SSAValue:
        result = self.fresh(dtype=dtype)
        attributes = {**self.provenance(source), "value": int(value)}
        if machine_operand_role is not None:
            attributes["machine_operand_role"] = str(machine_operand_role)
        self.instructions.append(Instr(
            Handler.Const.value,
            [],
            result,
            attributes=attributes,
        ))
        return result

    def value(self, register: X86Register, source: DecodedInstruction) -> SSAValue:
        value = self.registers.get(register)
        if value is not None:
            return value
        value = self.fresh(dtype="int64")
        self.registers[register] = value
        self.args.append(value)
        self.argument_registers.append(register)
        self.argument_names.append(f"__machine_{register.name.lower()}")
        return value

    def effective_address(
        self,
        operand: EffectiveAddressOperand,
        source: DecodedInstruction,
        *,
        dtype: str = "int32",
    ) -> SSAValue:
        terms: list[SSAValue] = []
        if operand.rip_relative:
            terms.append(self.constant(
                source.address + len(source.encoded), source, dtype=dtype,
            ))
        elif operand.base is not None:
            terms.append(self.value(operand.base, source))
        if operand.index is not None:
            index = self.value(operand.index, source)
            if operand.scale != 1:
                index = self.emit(
                    Handler.Mul,
                    (index, self.constant(operand.scale, source)),
                    source,
                    dtype=dtype,
                    address_component="scaled_index",
                )
            terms.append(index)
        if operand.displacement or not terms:
            terms.append(self.constant(operand.displacement, source, dtype=dtype))
        result = terms[0]
        for term in terms[1:]:
            result = self.emit(
                Handler.Add,
                (result, term),
                source,
                dtype=dtype,
                address_component="sum",
            )
        return result

    def load_memory(
        self,
        address: EffectiveAddressOperand,
        source: DecodedInstruction,
    ) -> SSAValue:
        pointer = self.effective_address(address, source, dtype="int64")
        return self.emit(
            Handler.Load,
            (self.memory, pointer),
            source,
            dtype="int64",
            machine_state="memory",
            width=64,
        )

    def store_memory(
        self,
        address: EffectiveAddressOperand,
        value: SSAValue,
        source: DecodedInstruction,
    ) -> None:
        pointer = self.effective_address(address, source, dtype="int64")
        self.memory = self.emit(
            Handler.Store,
            (self.memory, pointer, value),
            source,
            dtype="memory",
            machine_state="memory",
            width=64,
        )

    def copy_register(
        self,
        destination: RegisterOperand,
        value: SSAValue,
        source: DecodedInstruction,
    ) -> None:
        self.registers[destination.register] = self.emit(
            Handler.Cast,
            (value,),
            source,
            dtype="int64",
            machine_copy=True,
            machine_register=destination.register.name,
        )

    def finish(self) -> Function:
        for source in self.decoded:
            if source.token is X86InstructionToken.IMUL_R32_RM32:
                if (
                    len(source.operands) != 2
                    or not all(isinstance(item, RegisterOperand) for item in source.operands)
                ):
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed IMUL vocabulary operands"
                    )
                destination, right = source.operands
                assert isinstance(destination, RegisterOperand)
                assert isinstance(right, RegisterOperand)
                self.registers[destination.register] = self.emit(
                    Handler.Mul,
                    (self.value(destination.register, source), self.value(right.register, source)),
                    source,
                )
            elif source.token is X86InstructionToken.LEA_R32_M:
                if (
                    len(source.operands) != 2
                    or not isinstance(source.operands[0], RegisterOperand)
                    or not isinstance(source.operands[1], EffectiveAddressOperand)
                ):
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed LEA vocabulary operands"
                    )
                destination = source.operands[0]
                address = source.operands[1]
                self.registers[destination.register] = self.effective_address(address, source)
            elif source.token is X86InstructionToken.MOV_RM64_R64:
                if (
                    len(source.operands) != 2
                    or not isinstance(source.operands[1], RegisterOperand)
                ):
                    raise MachineLiftError(f"{source.address:#x}: malformed MOV store operands")
                destination, source_register = source.operands
                moved = self.value(source_register.register, source)
                if isinstance(destination, RegisterOperand):
                    self.copy_register(destination, moved, source)
                elif isinstance(destination, EffectiveAddressOperand):
                    self.store_memory(destination, moved, source)
                else:
                    raise MachineLiftError(f"{source.address:#x}: invalid MOV destination")
            elif source.token is X86InstructionToken.MOV_R64_RM64:
                if (
                    len(source.operands) != 2
                    or not isinstance(source.operands[0], RegisterOperand)
                ):
                    raise MachineLiftError(f"{source.address:#x}: malformed MOV load operands")
                destination, source_operand = source.operands
                if isinstance(source_operand, RegisterOperand):
                    moved = self.value(source_operand.register, source)
                elif isinstance(source_operand, EffectiveAddressOperand):
                    moved = self.load_memory(source_operand, source)
                else:
                    raise MachineLiftError(f"{source.address:#x}: invalid MOV source")
                self.copy_register(destination, moved, source)
            elif source.token is X86InstructionToken.MOV_R64_IMM64:
                if (
                    len(source.operands) != 2
                    or not isinstance(source.operands[0], RegisterOperand)
                    or not isinstance(source.operands[1], ImmediateOperand)
                ):
                    raise MachineLiftError(f"{source.address:#x}: malformed MOV immediate operands")
                destination, immediate = source.operands
                self.registers[destination.register] = self.constant(
                    immediate.value, source, dtype="int64",
                )
            elif source.token is X86InstructionToken.PUSH_R64:
                if len(source.operands) != 1 or not isinstance(source.operands[0], RegisterOperand):
                    raise MachineLiftError(f"{source.address:#x}: malformed PUSH operand")
                pushed = self.value(source.operands[0].register, source)
                eight = self.constant(8, source, dtype="int64")
                stack_pointer = self.emit(
                    Handler.Sub,
                    (self.value(X86Register.RSP, source), eight),
                    source,
                    dtype="int64",
                    machine_state="register",
                    machine_register="RSP",
                )
                self.registers[X86Register.RSP] = stack_pointer
                self.memory = self.emit(
                    Handler.Store,
                    (self.memory, stack_pointer, pushed),
                    source,
                    dtype="memory",
                    machine_state="stack-push",
                    width=64,
                )
            elif source.token is X86InstructionToken.AND_RM64_IMM8:
                if len(source.operands) != 2 or not isinstance(source.operands[1], ImmediateOperand):
                    raise MachineLiftError(f"{source.address:#x}: malformed AND operands")
                destination, immediate = source.operands
                if isinstance(destination, RegisterOperand):
                    left = self.value(destination.register, source)
                elif isinstance(destination, EffectiveAddressOperand):
                    left = self.load_memory(destination, source)
                else:
                    raise MachineLiftError(f"{source.address:#x}: invalid AND destination")
                right = self.constant(immediate.value, source, dtype="int64")
                result = self.emit(
                    Handler.And, (left, right), source, dtype="int64",
                    flags_effect="written-not-yet-materialized",
                )
                if isinstance(destination, RegisterOperand):
                    self.registers[destination.register] = result
                else:
                    self.store_memory(destination, result, source)
            elif source.token is X86InstructionToken.CMP_R64_RM64:
                if len(source.operands) != 2 or not isinstance(source.operands[0], RegisterOperand):
                    raise MachineLiftError(f"{source.address:#x}: malformed CMP operands")
                left_operand, right_operand = source.operands
                left = self.value(left_operand.register, source)
                if isinstance(right_operand, RegisterOperand):
                    right = self.value(right_operand.register, source)
                elif isinstance(right_operand, EffectiveAddressOperand):
                    right = self.load_memory(right_operand, source)
                else:
                    raise MachineLiftError(f"{source.address:#x}: invalid CMP source")
                self.pending_condition = self.emit(
                    Handler.Ne,
                    (left, right),
                    source,
                    dtype="bool",
                    machine_flags="ZF==0",
                )
            elif source.token is X86InstructionToken.JNE_REL32:
                if (
                    len(source.operands) != 1
                    or not isinstance(source.operands[0], RelativeAddressOperand)
                    or self.pending_condition is None
                ):
                    raise MachineLiftError(
                        f"{source.address:#x}: JNE requires a decoded relative target and CMP predicate"
                    )
                target = source.operands[0]
                self.instructions.append(Instr(
                    Handler.CondBr.value,
                    [self.pending_condition],
                    None,
                    attributes={
                        **self.provenance(source),
                        "true_target_address": target.target_address,
                        "false_target_address": source.address + len(source.encoded),
                        "relative_displacement": target.displacement,
                    },
                ))
            elif source.token is X86InstructionToken.RET_NEAR:
                result = self.value(X86Register.RAX, source)
                self.instructions.append(Instr(
                    Handler.Ret.value,
                    [result],
                    None,
                    attributes=self.provenance(source),
                ))
            elif source.token in {
                X86InstructionToken.SUB_R64_IMM8,
                X86InstructionToken.ADD_R64_IMM8,
            }:
                if (
                    len(source.operands) != 2
                    or not isinstance(source.operands[0], RegisterOperand)
                    or not isinstance(source.operands[1], ImmediateOperand)
                ):
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed stack-arithmetic operands"
                    )
                destination = source.operands[0]
                immediate = source.operands[1]
                literal = self.constant(
                    immediate.value, source, dtype="int64",
                    machine_operand_role="encoded-immediate",
                )
                handler = (
                    Handler.Sub
                    if source.token is X86InstructionToken.SUB_R64_IMM8
                    else Handler.Add
                )
                self.registers[destination.register] = self.emit(
                    handler,
                    (self.value(destination.register, source), literal),
                    source,
                    dtype="int64",
                    machine_state="register",
                    machine_register=destination.register.name,
                    flags_effect="written-not-yet-materialized",
                )
            elif source.token is X86InstructionToken.CALL_REL32:
                if (
                    len(source.operands) != 1
                    or not isinstance(source.operands[0], RelativeAddressOperand)
                ):
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed relative-call operand"
                    )
                target = source.operands[0]
                abi_registers = (
                    X86Register.RCX,
                    X86Register.RDX,
                    X86Register.R8,
                    X86Register.R9,
                )
                arguments = [
                    self.registers[register]
                    for register in abi_registers
                    if register in self.registers
                ]
                arguments.append(self.value(X86Register.RSP, source))
                self.registers[X86Register.RAX] = self.emit(
                    Handler.Call,
                    arguments,
                    source,
                    dtype="int64",
                    callee_address=target.target_address,
                    relative_displacement=target.displacement,
                    calling_convention="windows-x64",
                    implicit_effects=(
                        "push-return-address",
                        "volatile-register-clobber",
                        "stack-memory-read-write",
                    ),
                    unresolved_volatile_registers=(
                        "RCX", "RDX", "R8", "R9", "R10", "R11",
                    ),
                )
            elif source.token is X86InstructionToken.JMP_REL32:
                if (
                    len(source.operands) != 1
                    or not isinstance(source.operands[0], RelativeAddressOperand)
                ):
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed relative-jump operand"
                    )
                target = source.operands[0]
                self.instructions.append(Instr(
                    Handler.Br.value,
                    [],
                    None,
                    attributes={
                        **self.provenance(source),
                        "target_address": target.target_address,
                        "relative_displacement": target.displacement,
                        "machine_control_transfer": "tail-jump",
                    },
                ))
            else:
                raise MachineLiftError(
                    f"{source.address:#x}: no SSA lowering for machine token "
                    f"{int(source.token)}"
                )
        if not self.instructions or self.instructions[-1].op not in {
            Handler.Ret.value,
            Handler.Br.value,
            Handler.CondBr.value,
        }:
            raise MachineLiftError(
                "machine vocabulary did not terminate in return/branch SSA"
            )
        return Function(
            self.name,
            self.args,
            {"entry": BasicBlock("entry", self.instructions)},
            metadata={
                "lifted_from": "x86_64-reference-vocabulary",
                "argument_names": tuple(self.argument_names),
                "argument_registers": tuple(item.name.lower() for item in self.argument_registers),
                "machine_state_model": "register-and-versioned-memory-ssa",
                "machine_state_arguments": ("__machine_rsp", "__machine_memory"),
                "machine_state_shortfalls": (
                    "flags-not-materialized",
                    "volatile-call-results-not-materialized",
                ),
            },
        )


class _StructuredX86CFGLifter(_StructuredX86Lifter):
    """Raise a completely decoded bounded x86 function as an explicit CFG."""

    _CONDITIONAL_TOKENS = frozenset({
        X86InstructionToken.JNE_REL32,
        X86InstructionToken.JNE_REL8,
    })

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.initial_registers = dict(self.registers)
        self.vector_registers: dict[X86VectorRegister, SSAValue] = {}
        self.initial_vector_registers: dict[X86VectorRegister, SSAValue] = {}
        self.initial_memory = self.memory
        self.initial_flags: dict[str, SSAValue] = {}
        self.flags: dict[str, SSAValue] = {}
        self.initial_mxcsr: SSAValue | None = None
        self.mxcsr: SSAValue | None = None
        self.external_fallthrough_addresses: tuple[int, ...] = ()

    _FLAG_NAMES = ("CF", "PF", "AF", "ZF", "SF", "OF", "DF")

    def initial_mxcsr_value(self) -> SSAValue:
        if self.initial_mxcsr is None:
            self.initial_mxcsr = self.fresh(dtype="int32")
            self.args.append(self.initial_mxcsr)
            self.argument_names.append("__machine_mxcsr")
        return self.initial_mxcsr

    def mxcsr_value(self) -> SSAValue:
        if self.mxcsr is None:
            self.mxcsr = self.initial_mxcsr_value()
        return self.mxcsr

    def initial_vector_value(self, register: X86VectorRegister) -> SSAValue:
        initial = self.initial_vector_registers.get(register)
        if initial is None:
            initial = self.fresh(dtype=self.vector_state_dtype)
            self.initial_vector_registers[register] = initial
            self.args.append(initial)
            self.argument_names.append(f"__machine_{register.name.lower()}")
        return initial

    def vector_value(
        self, register: X86VectorRegister, source: DecodedInstruction,
    ) -> SSAValue:
        active = self.vector_registers.get(register)
        if active is None:
            active = self.initial_vector_value(register)
            self.vector_registers[register] = active
        return active

    def read_vector_operand(
        self,
        operand: VectorRegisterOperand | RegisterOperand | EffectiveAddressOperand,
        source: DecodedInstruction,
        *,
        width: int,
    ) -> SSAValue:
        if isinstance(operand, VectorRegisterOperand):
            value = self.vector_value(operand.register, source)
            if width == 256:
                return value
            return self.emit(
                Handler.And,
                (
                    value,
                    self.constant(
                        (1 << width) - 1,
                        source, dtype=self.vector_state_dtype,
                    ),
                ),
                source, dtype=self.vector_state_dtype,
                machine_vector_low_width=width,
            )
        if isinstance(operand, RegisterOperand):
            value = self.read_operand(operand, source, width=width)
            return self.emit(
                Handler.ZExt, (value,), source, dtype="int128",
                from_width=width, to_width=128,
                machine_vector_scalar_transfer=True,
            )
        pointer = self.effective_address(operand, source, dtype="int64")
        return self.emit(
            Handler.Load, (self.memory, pointer), source,
            dtype="int128", machine_state="memory", width=width,
            machine_vector_load=True,
        )

    def write_vector_operand(
        self,
        operand: VectorRegisterOperand | EffectiveAddressOperand,
        value: SSAValue,
        source: DecodedInstruction,
        *,
        width: int,
        preserve_upper: bool = False,
    ) -> None:
        if isinstance(operand, VectorRegisterOperand):
            if width == 256:
                result = value
            elif width == 128 and not preserve_upper:
                prior = self.vector_value(operand.register, source)
                retained = self.emit(
                    Handler.And,
                    (
                        prior,
                        self.constant(
                            ((1 << 256) - 1) ^ ((1 << 128) - 1),
                            source, dtype=self.vector_state_dtype,
                        ),
                    ),
                    source, dtype=self.vector_state_dtype,
                    machine_vector_preserve_ymm_upper=True,
                )
                low = self.emit(
                    Handler.And,
                    (
                        value,
                        self.constant(
                            (1 << 128) - 1,
                            source, dtype=self.vector_state_dtype,
                        ),
                    ),
                    source, dtype=self.vector_state_dtype,
                    machine_vector_low_width=128,
                )
                result = self.emit(
                    Handler.Or, (retained, low), source,
                    dtype=self.vector_state_dtype,
                    machine_vector_legacy_xmm_write=True,
                )
            elif preserve_upper:
                prior = self.vector_value(operand.register, source)
                low_mask = (1 << width) - 1
                retained = self.emit(
                    Handler.And,
                    (
                        prior,
                        self.constant(
                            ((1 << 128) - 1) ^ low_mask,
                            source, dtype="int128",
                        ),
                    ),
                    source, dtype="int128", machine_vector_preserve_upper=True,
                )
                low = self.emit(
                    Handler.And,
                    (value, self.constant(low_mask, source, dtype="int128")),
                    source, dtype="int128", machine_vector_low_width=width,
                )
                result = self.emit(
                    Handler.Or, (retained, low), source, dtype="int128",
                    machine_vector_insert_low=width,
                )
            else:
                result = self.emit(
                    Handler.And,
                    (
                        value,
                        self.constant((1 << width) - 1, source, dtype="int128"),
                    ),
                    source, dtype="int128", machine_vector_zero_upper=width,
                )
            self.vector_registers[operand.register] = result
            return
        pointer = self.effective_address(operand, source, dtype="int64")
        self.memory = self.emit(
            Handler.Store, (self.memory, pointer, value), source,
            dtype="memory", machine_state="memory", width=width,
            machine_vector_store=True,
        )

    def initial_flag(self, name: str) -> SSAValue:
        flag = str(name).upper()
        initial = self.initial_flags.get(flag)
        if initial is None:
            initial = self.fresh(dtype="bool")
            self.initial_flags[flag] = initial
            self.args.append(initial)
            self.argument_names.append(f"__machine_{flag.lower()}")
        return initial

    def flag(self, name: str) -> SSAValue:
        flag = str(name).upper()
        active = self.flags.get(flag)
        if active is None:
            active = self.initial_flag(flag)
            self.flags[flag] = active
        return active

    def bool_constant(self, value: bool, source: DecodedInstruction) -> SSAValue:
        return self.constant(int(bool(value)), source, dtype="bool")

    def _bool(self, handler: Handler, args, source: DecodedInstruction, **attributes) -> SSAValue:
        return self.emit(handler, args, source, dtype="bool", **attributes)

    @staticmethod
    def operand_width(source: DecodedInstruction, index: int) -> int:
        operand = source.operands[index]
        width = getattr(operand, "width", None)
        if width is not None:
            return int(width)
        widths = re.findall(
            r"(?:^|_)(?:XMMM|RM|R|M)(128|64|32|16|8)(?:_|$)",
            source.token.name,
        )
        if widths:
            # Extension forms spell both destination and source widths
            # (MOVSXD_R64_RM32, MOVZX_R32_RM8). The destination is first and
            # the source is last; ordinary forms carry one shared width.
            return int(widths[-1] if index else widths[0])
        raise MachineLiftError(
            f"{source.address:#x}: cannot derive operand width for {source.token.name}"
        )

    def truncate_bits(
        self, value: SSAValue, width: int, source: DecodedInstruction,
    ) -> SSAValue:
        mask = self.constant((1 << int(width)) - 1, source, dtype="int64")
        return self.emit(
            Handler.And, (value, mask), source, dtype="int64",
            machine_width=int(width), machine_modular_truncation=True,
        )

    def logical_flags(
        self, result: SSAValue, width: int, source: DecodedInstruction,
    ) -> None:
        result = self.truncate_bits(result, width, source)
        zero = self.constant(0, source, dtype="int64")
        sign = self.constant(1 << (width - 1), source, dtype="int64")
        self.flags["CF"] = self.bool_constant(False, source)
        self.flags["ZF"] = self._bool(Handler.Eq, (result, zero), source, machine_flag="ZF")
        signed = self.emit(Handler.And, (result, sign), source, dtype="int64")
        self.flags["SF"] = self._bool(Handler.Ne, (signed, zero), source, machine_flag="SF")
        self.flags["OF"] = self.bool_constant(False, source)
        # XOR-fold to the low parity bit; PF is one for even parity.
        parity = result
        for shift in (4, 2, 1):
            shifted = self.emit(
                Handler.Shr,
                (parity, self.constant(shift, source, dtype="int64")),
                source, dtype="int64",
            )
            parity = self.emit(Handler.Xor, (parity, shifted), source, dtype="int64")
        low = self.emit(
            Handler.And,
            (parity, self.constant(1, source, dtype="int64")),
            source, dtype="int64",
        )
        self.flags["PF"] = self._bool(Handler.Eq, (low, zero), source, machine_flag="PF")

    def arithmetic_flags(
        self,
        left: SSAValue,
        right: SSAValue,
        raw: SSAValue,
        width: int,
        source: DecodedInstruction,
        *,
        subtract: bool,
        preserve_cf: bool = False,
    ) -> None:
        mask_value = (1 << int(width)) - 1
        left = self.truncate_bits(left, width, source)
        right = self.truncate_bits(right, width, source)
        result = self.truncate_bits(raw, width, source)
        prior_cf = self.flag("CF") if preserve_cf else None
        self.logical_flags(result, width, source)
        if subtract:
            carry = None if preserve_cf else self._bool(
                Handler.ULt, (left, right), source,
                machine_flag="CF", comparison_unsigned=True,
            )
            first = self.emit(Handler.Xor, (left, right), source, dtype="int64")
        else:
            carry = None if preserve_cf else self._bool(
                Handler.ULt, (result, left), source,
                machine_flag="CF", comparison_unsigned=True,
            )
            xor_lr = self.emit(Handler.Xor, (left, right), source, dtype="int64")
            first = self.emit(
                Handler.Xor,
                (xor_lr, self.constant(mask_value, source, dtype="int64")),
                source, dtype="int64", machine_bitwise_not_width=width,
            )
        second = self.emit(Handler.Xor, (left, result), source, dtype="int64")
        overflow_bits = self.emit(Handler.And, (first, second), source, dtype="int64")
        sign = self.constant(1 << (width - 1), source, dtype="int64")
        overflow_bit = self.emit(Handler.And, (overflow_bits, sign), source, dtype="int64")
        zero = self.constant(0, source, dtype="int64")
        self.flags["OF"] = self._bool(
            Handler.Ne, (overflow_bit, zero), source, machine_flag="OF",
        )
        nibble = self.emit(Handler.Xor, (left, right), source, dtype="int64")
        nibble = self.emit(Handler.Xor, (nibble, result), source, dtype="int64")
        auxiliary = self.emit(
            Handler.And,
            (nibble, self.constant(0x10, source, dtype="int64")),
            source, dtype="int64",
        )
        self.flags["AF"] = self._bool(
            Handler.Ne, (auxiliary, zero), source, machine_flag="AF",
        )
        self.flags["CF"] = prior_cf if preserve_cf else carry

    def condition(self, source: DecodedInstruction) -> SSAValue:
        name = source.token.name.split("_", 1)[0]
        for prefix in ("CMOV", "SET", "J"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        cf, zf, sf, of = (self.flag(item) for item in ("CF", "ZF", "SF", "OF"))
        true = self.bool_constant(True, source)
        if name in {"E", "Z"}:
            return zf
        if name in {"NE", "NZ"}:
            return self._bool(Handler.LNot, (zf,), source, machine_condition=name)
        if name in {"B", "C", "NAE"}:
            return cf
        if name in {"AE", "NB", "NC"}:
            return self._bool(Handler.LNot, (cf,), source, machine_condition=name)
        if name in {"BE", "NA"}:
            return self._bool(Handler.LOr, (cf, zf), source, machine_condition=name)
        if name in {"A", "NBE"}:
            either = self._bool(Handler.LOr, (cf, zf), source)
            return self._bool(Handler.LNot, (either,), source, machine_condition=name)
        if name == "S":
            return sf
        if name in {"P", "PE"}:
            return self.flag("PF")
        if name in {"NP", "PO"}:
            return self._bool(
                Handler.LNot, (self.flag("PF"),), source,
                machine_condition=name,
            )
        if name == "NS":
            return self._bool(Handler.LNot, (sf,), source, machine_condition=name)
        if name == "O":
            return of
        if name == "NO":
            return self._bool(Handler.LNot, (of,), source, machine_condition=name)
        unequal = self._bool(Handler.Ne, (sf, of), source, machine_condition="SF!=OF")
        if name in {"L", "NGE"}:
            return unequal
        if name in {"GE", "NL"}:
            return self._bool(Handler.LNot, (unequal,), source, machine_condition=name)
        if name in {"LE", "NG"}:
            return self._bool(Handler.LOr, (zf, unequal), source, machine_condition=name)
        if name in {"G", "NLE"}:
            either = self._bool(Handler.LOr, (zf, unequal), source)
            return self._bool(Handler.LNot, (either,), source, machine_condition=name)
        raise MachineLiftError(
            f"{source.address:#x}: unsupported machine condition {name!r}"
        )

    def finish_cyclic(
        self,
        block_sources: Mapping[str, list[DecodedInstruction]],
        successors: Mapping[str, list[str]],
        conditional_destinations: Mapping[
            str,
            tuple[
                tuple[str | None, int | None],
                tuple[str | None, int | None],
            ],
        ],
        external_targets: set[int],
        external_fallthroughs: Mapping[str, int],
        graph: nx.DiGraph,
        entry_label: str,
    ) -> Function:
        """Lower a cyclic CFG with preallocated full machine-state Phis."""

        preheader = "__machine_preheader"
        if preheader in block_sources:
            raise MachineLiftError("machine CFG collides with reserved preheader label")

        # Make the complete architectural input explicit before allocating
        # block state. This is conservative; ordinary SSA DCE can later remove
        # GPR/flag Phis unused by the function.
        initial_registers = {
            register: self.initial_value(register) for register in X86Register
        }
        initial_vector_registers = {
            register: self.initial_vector_value(register)
            for register in self.initial_vector_registers
        }
        initial_mxcsr = self.initial_mxcsr
        initial_flags = {
            flag: self.initial_flag(flag) for flag in self._FLAG_NAMES
        }
        initial_memory = self.initial_memory
        predecessors = {
            label: list(graph.predecessors(label)) for label in block_sources
        }
        predecessors[entry_label].insert(0, preheader)

        block_inputs: dict[
            str, tuple[
                dict[X86Register, SSAValue],
                dict[X86VectorRegister, SSAValue],
                SSAValue | None, SSAValue, dict[str, SSAValue],
            ]
        ] = {}
        block_phis: dict[str, list[Instr]] = {}
        for label in block_sources:
            register_values = {
                register: self.fresh(dtype="int64") for register in X86Register
            }
            vector_values = {
                register: self.fresh(dtype=self.vector_state_dtype)
                for register in initial_vector_registers
            }
            mxcsr_value = (
                self.fresh(dtype="int32") if initial_mxcsr is not None else None
            )
            memory_value = self.fresh(dtype="memory")
            flag_values = {
                flag: self.fresh(dtype="bool") for flag in self._FLAG_NAMES
            }
            incoming_blocks = tuple(predecessors[label])
            phis = [
                Instr(
                    Handler.Phi.value, [], value,
                    attributes={
                        "incoming_blocks": incoming_blocks,
                        "machine_state": "register",
                        "machine_register": register.name,
                    },
                )
                for register, value in register_values.items()
            ]
            phis.extend(
                Instr(
                    Handler.Phi.value, [], value,
                    attributes={
                        "incoming_blocks": incoming_blocks,
                        "machine_state": "vector-register",
                        "machine_register": register.name,
                    },
                )
                for register, value in vector_values.items()
            )
            if mxcsr_value is not None:
                phis.append(Instr(
                    Handler.Phi.value, [], mxcsr_value,
                    attributes={
                        "incoming_blocks": incoming_blocks,
                        "machine_state": "mxcsr",
                    },
                ))
            phis.append(Instr(
                Handler.Phi.value, [], memory_value,
                attributes={
                    "incoming_blocks": incoming_blocks,
                    "machine_state": "memory",
                },
            ))
            phis.extend(
                Instr(
                    Handler.Phi.value, [], value,
                    attributes={
                        "incoming_blocks": incoming_blocks,
                        "machine_state": "flags",
                        "machine_flag": flag,
                    },
                )
                for flag, value in flag_values.items()
            )
            block_inputs[label] = (
                register_values, vector_values, mxcsr_value,
                memory_value, flag_values,
            )
            block_phis[label] = phis

        states: dict[
            str, tuple[
                dict[X86Register, SSAValue],
                dict[X86VectorRegister, SSAValue],
                SSAValue | None, SSAValue, dict[str, SSAValue],
            ]
        ] = {
            preheader: (
                initial_registers, initial_vector_registers,
                initial_mxcsr, initial_memory, initial_flags,
            ),
        }
        blocks: dict[str, BasicBlock] = {
            preheader: BasicBlock(
                preheader,
                [Instr(
                    Handler.Br.value, [], None,
                    attributes={"target": entry_label, "machine_preheader": True},
                )],
                [entry_label],
            ),
        }
        for label, sources in block_sources.items():
            registers, vector_registers, mxcsr, memory, flags = block_inputs[label]
            self.registers = dict(registers)
            self.vector_registers = dict(vector_registers)
            self.mxcsr = mxcsr
            self.memory = memory
            self.flags = dict(flags)
            self.pending_condition = None
            self.instructions = list(block_phis[label])
            for source in sources:
                self.lower_one(source)
            if self.instructions and self.instructions[-1].op == Handler.CondBr.value:
                true_destination, false_destination = conditional_destinations[label]
                self.instructions[-1].attributes.update({
                    "true_target": true_destination[0],
                    "false_target": false_destination[0],
                    "true_target_address": true_destination[1],
                    "false_target_address": false_destination[1],
                    "machine_external_control": bool(
                        true_destination[1] is not None
                        or false_destination[1] is not None
                    ),
                })
            elif self.instructions and self.instructions[-1].op == Handler.Br.value:
                if successors[label]:
                    self.instructions[-1].attributes["target"] = successors[label][0]
            elif not self.instructions or self.instructions[-1].op not in {
                Handler.Ret.value, Handler.IndirectBr.value, Handler.Trap.value,
            }:
                external_fallthrough = external_fallthroughs.get(label)
                if external_fallthrough is not None:
                    self.instructions.append(Instr(
                        Handler.Br.value, [], None,
                        attributes={
                            "target_address": int(external_fallthrough),
                            "machine_address": int(sources[-1].address),
                            "machine_control_transfer": "cross-region-fallthrough",
                        },
                    ))
                elif len(successors[label]) != 1:
                    raise MachineLiftError(f"machine block {label} has no terminator")
                else:
                    self.instructions.append(Instr(
                    Handler.Br.value, [], None,
                    attributes={
                        "target": successors[label][0],
                        "synthetic_machine_fallthrough": True,
                    },
                    ))
            blocks[label] = BasicBlock(
                label, list(self.instructions), list(successors[label]),
            )
            states[label] = (
                dict(self.registers), dict(self.vector_registers),
                self.mxcsr, self.memory, dict(self.flags),
            )

        # Complete the forward and backedge arguments only after every block
        # has a final outgoing state.
        for label, phis in block_phis.items():
            incoming_blocks = predecessors[label]
            cursor = 0
            for register in X86Register:
                phis[cursor].args[:] = [
                    states[pred][0][register] for pred in incoming_blocks
                ]
                cursor += 1
            for register in initial_vector_registers:
                phis[cursor].args[:] = [
                    states[pred][1][register] for pred in incoming_blocks
                ]
                cursor += 1
            if initial_mxcsr is not None:
                phis[cursor].args[:] = [
                    states[pred][2] for pred in incoming_blocks
                ]
                cursor += 1
            phis[cursor].args[:] = [states[pred][3] for pred in incoming_blocks]
            cursor += 1
            for flag in self._FLAG_NAMES:
                phis[cursor].args[:] = [
                    states[pred][4][flag] for pred in incoming_blocks
                ]
                cursor += 1

        return Function(
            self.name, self.args, blocks,
            metadata={
                "lifted_from": "x86_64-reference-vocabulary",
                "entry_block": preheader,
                "argument_names": tuple(self.argument_names),
                "argument_registers": tuple(
                    item.name.lower() for item in self.argument_registers
                ),
                "machine_state_model": "full-register-memory-flags-ssa",
                "machine_state_arguments": tuple(self.argument_names),
                "machine_instruction_count": len(self.decoded),
                "machine_loop_state_phis": True,
                "machine_external_control_targets": tuple(
                    sorted(external_targets)
                ),
                "requires_machine_address_linking": bool(external_targets),
                "requires_dynamic_target_linking": any(
                    item.semantic is MachineSemanticToken.INDIRECT_JUMP
                    for item in self.decoded
                ),
            },
        )

    def initial_value(self, register: X86Register) -> SSAValue:
        initial = self.initial_registers.get(register)
        if initial is None:
            initial = self.fresh(dtype="int64")
            self.initial_registers[register] = initial
            self.args.append(initial)
            self.argument_registers.append(register)
            self.argument_names.append(f"__machine_{register.name.lower()}")
        return initial

    def value(self, register: X86Register, source: DecodedInstruction) -> SSAValue:
        current = self.registers.get(register)
        if current is not None:
            return current
        initial = self.initial_value(register)
        self.registers[register] = initial
        return initial

    def read_operand(
        self,
        operand: RegisterOperand | EffectiveAddressOperand | ImmediateOperand,
        source: DecodedInstruction,
        *,
        width: int = 64,
    ) -> SSAValue:
        if isinstance(operand, RegisterOperand):
            value = self.value(operand.register, source)
            if width < 64:
                return self.emit(
                    Handler.Trunc, (value,), source, dtype=f"int{width}",
                    machine_width=width,
                )
            return value
        if isinstance(operand, ImmediateOperand):
            value = int(operand.value)
            if operand.signed and operand.width < width:
                sign = 1 << (operand.width - 1)
                value = (value & (sign - 1)) - (value & sign)
            return self.constant(
                value, source, dtype=f"int{width}",
                machine_operand_role="encoded-immediate",
            )
        pointer = self.effective_address(operand, source, dtype="int64")
        return self.emit(
            Handler.Load,
            (self.memory, pointer),
            source,
            dtype="int32" if width == 32 else "int64",
            machine_state="memory",
            width=width,
        )

    def write_operand(
        self,
        operand: RegisterOperand | EffectiveAddressOperand,
        value: SSAValue,
        source: DecodedInstruction,
        *,
        width: int | None = None,
    ) -> None:
        active_width = int(width or getattr(operand, "width", 64))
        if isinstance(operand, RegisterOperand):
            if active_width == 64:
                self.copy_register(operand, value, source)
            elif active_width == 32:
                self.registers[operand.register] = self.zero_extend_32(value, source)
            else:
                old = self.value(operand.register, source)
                low_mask = (1 << active_width) - 1
                retained = self.emit(
                    Handler.And,
                    (old, self.constant(((1 << 64) - 1) ^ low_mask, source, dtype="int64")),
                    source, dtype="int64", machine_subregister_preserve=True,
                )
                low = self.emit(
                    Handler.And,
                    (value, self.constant(low_mask, source, dtype="int64")),
                    source, dtype="int64", machine_subregister_width=active_width,
                )
                self.registers[operand.register] = self.emit(
                    Handler.Or, (retained, low), source, dtype="int64",
                    machine_subregister_write=True,
                )
        else:
            pointer = self.effective_address(operand, source, dtype="int64")
            self.memory = self.emit(
                Handler.Store, (self.memory, pointer, value), source,
                dtype="memory", machine_state="memory", width=active_width,
            )

    def zero_extend_32(self, value: SSAValue, source: DecodedInstruction) -> SSAValue:
        narrowed = value
        if value.dtype != "int32":
            narrowed = self.emit(
                Handler.Trunc, (value,), source, dtype="int32", machine_width=32,
            )
        return self.emit(
            Handler.ZExt,
            (narrowed,),
            source,
            dtype="int64",
            from_width=32,
            to_width=64,
            x86_zero_extend_register_write=True,
        )

    def lower_call(
        self,
        source: DecodedInstruction,
        target: RelativeAddressOperand | RegisterOperand | EffectiveAddressOperand,
    ) -> None:
        if isinstance(target, RelativeAddressOperand):
            callee = self.constant(target.target_address, source, dtype="int64")
            call_kind = "direct-relative"
        else:
            callee = self.read_operand(target, source, width=64)
            call_kind = "indirect-register" if isinstance(target, RegisterOperand) else "indirect-memory"
        abi_registers = (
            X86Register.RCX, X86Register.RDX, X86Register.R8, X86Register.R9,
        )
        call_args = [callee, self.memory, self.value(X86Register.RSP, source)]
        call_args.extend(self.value(register, source) for register in abi_registers)
        call_state = self.emit(
            Handler.Call,
            call_args,
            source,
            dtype="machine_call_state",
            call_kind=call_kind,
            calling_convention="windows-x64",
            callee_address=(target.target_address if isinstance(target, RelativeAddressOperand) else None),
            implicit_effects=(
                "push-return-address",
                "volatile-register-clobber",
                "stack-memory-read-write",
            ),
            result_model="opaque-call-state-with-load-projections",
            indirect_operand=(
                "register" if isinstance(target, RegisterOperand) else
                "rip-relative-memory" if isinstance(target, EffectiveAddressOperand)
                and target.rip_relative else
                "based-memory" if isinstance(target, EffectiveAddressOperand) else None
            ),
            indirect_register=(
                target.register.name if isinstance(target, RegisterOperand) else None
            ),
            indirect_base_register=(
                target.base.name if isinstance(target, EffectiveAddressOperand)
                and target.base is not None else None
            ),
            indirect_displacement=(
                int(target.displacement)
                if isinstance(target, EffectiveAddressOperand) else None
            ),
            indirect_slot_address=(
                int(source.address) + len(source.encoded) + int(target.displacement)
                if isinstance(target, EffectiveAddressOperand) and target.rip_relative
                else None
            ),
        )
        volatile = (
            X86Register.RAX, X86Register.RCX, X86Register.RDX,
            X86Register.R8, X86Register.R9, X86Register.R10, X86Register.R11,
        )
        for selector, register in enumerate(volatile):
            field = self.constant(selector, source, dtype="int32")
            self.registers[register] = self.emit(
                Handler.Load,
                (call_state, field),
                source,
                dtype="int64",
                machine_call_projection=register.name,
            )
        memory_field = self.constant(len(volatile), source, dtype="int32")
        self.memory = self.emit(
            Handler.Load,
            (call_state, memory_field),
            source,
            dtype="memory",
            machine_call_projection="MEMORY",
        )
        self.pending_condition = None

    def lower_one(self, source: DecodedInstruction) -> None:
        token = source.token
        operands = source.operands
        semantic = MachineSemanticToken(source.semantic)
        if semantic is MachineSemanticToken.NO_OPERATION:
            return
        if semantic in {
            MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
            MachineSemanticToken.REGISTER_OR_MEMORY_READ,
            MachineSemanticToken.REGISTER_WRITE_IMMEDIATE,
        }:
            if len(operands) != 2 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(
                operands[1], (RegisterOperand, EffectiveAddressOperand, ImmediateOperand)
            ):
                raise MachineLiftError(f"{source.address:#x}: malformed MOV operands")
            width = self.operand_width(source, 0)
            moved = self.read_operand(operands[1], source, width=width)
            self.write_operand(operands[0], moved, source, width=width)
            return
        if semantic in {
            MachineSemanticToken.SIGN_EXTEND,
            MachineSemanticToken.ZERO_EXTEND,
        }:
            if len(operands) != 2 or not isinstance(operands[0], RegisterOperand) or not isinstance(
                operands[1], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(f"{source.address:#x}: malformed extension operands")
            target_width = self.operand_width(source, 0)
            source_width = self.operand_width(source, 1)
            value = self.read_operand(operands[1], source, width=source_width)
            handler = (
                Handler.SExt
                if semantic is MachineSemanticToken.SIGN_EXTEND
                else Handler.ZExt
            )
            extended = self.emit(
                handler, (value,), source, dtype=f"int{target_width}",
                from_width=source_width, to_width=target_width,
            )
            self.write_operand(operands[0], extended, source, width=target_width)
            return
        if semantic is MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR:
            if token is X86InstructionToken.CDQ:
                accumulator = self.read_operand(
                    RegisterOperand(X86Register.RAX, width=32),
                    source, width=32,
                )
                sign = self.emit(
                    Handler.And,
                    (accumulator, self.constant(1 << 31, source, dtype="int32")),
                    source, dtype="int32",
                )
                negative = self._bool(
                    Handler.Ne,
                    (sign, self.constant(0, source, dtype="int32")),
                    source, machine_accumulator_sign=True,
                )
                high = self.emit(
                    Handler.Select,
                    (
                        negative,
                        self.constant((1 << 32) - 1, source, dtype="int32"),
                        self.constant(0, source, dtype="int32"),
                    ),
                    source, dtype="int32", machine_cdq_high_half=True,
                )
                self.write_operand(
                    RegisterOperand(X86Register.RDX, width=32),
                    high, source, width=32,
                )
                return
            if token is X86InstructionToken.CDQE:
                source_value = self.read_operand(
                    RegisterOperand(X86Register.RAX, width=32),
                    source, width=32,
                )
                extended = self.emit(
                    Handler.SExt, (source_value,), source, dtype="int64",
                    from_width=32, to_width=64,
                )
                self.write_operand(
                    RegisterOperand(X86Register.RAX, width=64),
                    extended, source, width=64,
                )
                return
            if token is X86InstructionToken.CQO:
                accumulator = self.read_operand(
                    RegisterOperand(X86Register.RAX, width=64),
                    source, width=64,
                )
                sign = self.emit(
                    Handler.And,
                    (
                        accumulator,
                        self.constant(1 << 63, source, dtype="int64"),
                    ),
                    source, dtype="int64",
                )
                negative = self._bool(
                    Handler.Ne,
                    (sign, self.constant(0, source, dtype="int64")),
                    source, machine_accumulator_sign=True,
                )
                high = self.emit(
                    Handler.Select,
                    (
                        negative,
                        self.constant((1 << 64) - 1, source, dtype="int64"),
                        self.constant(0, source, dtype="int64"),
                    ),
                    source, dtype="int64", machine_cqo_high_half=True,
                )
                self.write_operand(
                    RegisterOperand(X86Register.RDX, width=64),
                    high, source, width=64,
                )
                return
            raise MachineLiftError(
                f"{source.address:#x}: accumulator extension {token.name} "
                "is not legalized"
            )
        if semantic is MachineSemanticToken.VECTOR_MOVE:
            if (
                token in {
                    X86InstructionToken.MOVD_RM32_XMM,
                    X86InstructionToken.MOVQ_RM64_XMM,
                }
                and len(operands) == 2
                and isinstance(operands[0], (RegisterOperand, EffectiveAddressOperand))
                and isinstance(operands[1], VectorRegisterOperand)
            ):
                width = 32 if token is X86InstructionToken.MOVD_RM32_XMM else 64
                moved = self.read_vector_operand(
                    operands[1], source, width=width,
                )
                self.write_operand(operands[0], moved, source, width=width)
                return
            if len(operands) != 2 or not isinstance(
                operands[0], (VectorRegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed vector move operands"
                )
            width = (
                64
                if token is X86InstructionToken.MOVSD_XMM_XMMM64
                else 128
            )
            moved = self.read_vector_operand(
                operands[1], source, width=width,
            )
            self.write_vector_operand(
                operands[0], moved, source, width=width,
                # Legacy MOVSD changes the low scalar lane while retaining
                # the destination's high 64 bits. Full-width moves replace it.
                preserve_upper=(width < 128),
            )
            return
        if semantic is MachineSemanticToken.STRING_COMPARE:
            expected_operands = (
                RegisterOperand(X86Register.RDI, width=64),
                RegisterOperand(X86Register.RAX, width=8),
            )
            if (
                token is not X86InstructionToken.SCASB
                or operands != expected_operands
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: unsupported string-compare form "
                    f"{token.name}"
                )
            destination = self.value(X86Register.RDI, source)
            memory_value = self.emit(
                Handler.Load,
                (self.memory, destination),
                source,
                dtype="int8",
                machine_state="memory",
                width=8,
                machine_string_compare="scasb",
            )
            accumulator = self.read_operand(
                RegisterOperand(X86Register.RAX, width=8),
                source,
                width=8,
            )
            raw = self.emit(
                Handler.Sub,
                (accumulator, memory_value),
                source,
                dtype="int8",
                machine_string_compare="scasb",
            )
            self.arithmetic_flags(
                accumulator, memory_value, raw, 8, source, subtract=True,
            )
            stride = self.emit(
                Handler.Select,
                (
                    self.flag("DF"),
                    self.constant(-1, source, dtype="int64"),
                    self.constant(1, source, dtype="int64"),
                ),
                source,
                dtype="int64",
                machine_direction_stride=True,
                machine_string_compare="scasb",
            )
            self.registers[X86Register.RDI] = self.emit(
                Handler.Add,
                (destination, stride),
                source,
                dtype="int64",
                machine_register="RDI",
                modular_width=64,
                machine_string_compare="scasb",
            )
            return
        if semantic is MachineSemanticToken.STRING_STORE:
            if token is X86InstructionToken.STOSB:
                expected = (
                    RegisterOperand(X86Register.RDI, width=64),
                    RegisterOperand(X86Register.RAX, width=8),
                )
                if operands != expected:
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed STOSB operands"
                    )
                destination = self.value(X86Register.RDI, source)
                value = self.read_operand(operands[1], source, width=8)
                self.memory = self.emit(
                    Handler.Store, (self.memory, destination, value), source,
                    dtype="memory", machine_state="memory", width=8,
                    machine_string_store="stosb",
                )
                stride = self.emit(
                    Handler.Select,
                    (
                        self.flag("DF"),
                        self.constant(-1, source, dtype="int64"),
                        self.constant(1, source, dtype="int64"),
                    ),
                    source, dtype="int64", machine_direction_stride=True,
                )
                self.registers[X86Register.RDI] = self.emit(
                    Handler.Add, (destination, stride), source,
                    dtype="int64", machine_register="RDI",
                    modular_width=64, machine_string_store="stosb",
                )
                return
            if token is X86InstructionToken.REP_STOSB:
                expected = (
                    RegisterOperand(X86Register.RDI, width=64),
                    RegisterOperand(X86Register.RAX, width=8),
                    RegisterOperand(X86Register.RCX, width=64),
                )
                if operands != expected:
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed REP STOSB operands"
                    )
                destination = self.value(X86Register.RDI, source)
                count = self.value(X86Register.RCX, source)
                value = self.read_operand(operands[1], source, width=8)
                stride = self.emit(
                    Handler.Select,
                    (
                        self.flag("DF"),
                        self.constant(-1, source, dtype="int64"),
                        self.constant(1, source, dtype="int64"),
                    ),
                    source, dtype="int64", machine_direction_stride=True,
                )
                self.memory = self.emit(
                    Handler.StridedStoreFill,
                    (self.memory, destination, count, value, stride), source,
                    dtype="memory", machine_state="memory", element_width=8,
                    stride_unit="bytes", operation="rep-stosb",
                    iterative=True, zero_count_is_noop=True,
                )
                displacement = self.emit(
                    Handler.Mul, (count, stride), source, dtype="int64",
                    machine_string_displacement=True,
                )
                self.registers[X86Register.RDI] = self.emit(
                    Handler.Add, (destination, displacement), source,
                    dtype="int64", machine_register="RDI", modular_width=64,
                )
                self.registers[X86Register.RCX] = self.constant(
                    0, source, dtype="int64",
                )
                return
            expected_operands = (
                RegisterOperand(X86Register.RDI, width=64),
                RegisterOperand(X86Register.RAX, width=16),
                RegisterOperand(X86Register.RCX, width=64),
            )
            if (
                token is not X86InstructionToken.REP_STOSW
                or operands != expected_operands
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: unsupported string-store form {token.name}"
                )
            destination = self.value(X86Register.RDI, source)
            count = self.value(X86Register.RCX, source)
            value = self.emit(
                Handler.Trunc,
                (self.value(X86Register.RAX, source),),
                source, dtype="int16", machine_string_element_width=16,
            )
            stride = self.emit(
                Handler.Select,
                (
                    self.flag("DF"),
                    self.constant(-2, source, dtype="int64"),
                    self.constant(2, source, dtype="int64"),
                ),
                source, dtype="int64", machine_direction_stride=True,
            )
            self.memory = self.emit(
                Handler.StridedStoreFill,
                (self.memory, destination, count, value, stride),
                source, dtype="memory", machine_state="memory",
                element_width=16, stride_unit="bytes",
                operation="rep-stosw", iterative=True,
                zero_count_is_noop=True,
            )
            displacement = self.emit(
                Handler.Mul, (count, stride), source, dtype="int64",
                machine_string_displacement=True,
            )
            self.registers[X86Register.RDI] = self.emit(
                Handler.Add, (destination, displacement), source,
                dtype="int64", machine_register="RDI",
                modular_width=64,
            )
            self.registers[X86Register.RCX] = self.constant(
                0, source, dtype="int64",
            )
            return
        if semantic is MachineSemanticToken.STRING_MOVE:
            expected_operands = (
                RegisterOperand(X86Register.RDI, width=64),
                RegisterOperand(X86Register.RSI, width=64),
                RegisterOperand(X86Register.RCX, width=64),
            )
            if (
                token is not X86InstructionToken.REP_MOVSQ
                or operands != expected_operands
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: unsupported string-move form {token.name}"
                )
            destination = self.value(X86Register.RDI, source)
            source_address = self.value(X86Register.RSI, source)
            count = self.value(X86Register.RCX, source)
            stride = self.emit(
                Handler.Select,
                (
                    self.flag("DF"),
                    self.constant(-8, source, dtype="int64"),
                    self.constant(8, source, dtype="int64"),
                ),
                source, dtype="int64", machine_direction_stride=True,
            )
            self.memory = self.emit(
                Handler.StridedMemoryCopy,
                (self.memory, destination, source_address, count, stride),
                source, dtype="memory", machine_state="memory",
                element_width=64, stride_unit="bytes",
                operation="rep-movsq", iterative=True,
                ordered_overlap_semantics=True, zero_count_is_noop=True,
            )
            displacement = self.emit(
                Handler.Mul, (count, stride), source, dtype="int64",
                machine_string_displacement=True,
            )
            self.registers[X86Register.RSI] = self.emit(
                Handler.Add, (source_address, displacement), source,
                dtype="int64", machine_register="RSI", modular_width=64,
            )
            self.registers[X86Register.RDI] = self.emit(
                Handler.Add, (destination, displacement), source,
                dtype="int64", machine_register="RDI", modular_width=64,
            )
            self.registers[X86Register.RCX] = self.constant(
                0, source, dtype="int64",
            )
            return
        if semantic is MachineSemanticToken.VECTOR_XOR:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed vector XOR operands"
                )
            left = self.read_vector_operand(operands[0], source, width=128)
            right = self.read_vector_operand(operands[1], source, width=128)
            result = self.emit(
                Handler.Xor, (left, right), source, dtype="int128",
                machine_vector_bit_pattern=True, width=128,
            )
            self.write_vector_operand(
                operands[0], result, source, width=128,
            )
            return
        if semantic is MachineSemanticToken.VECTOR_AND:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed vector AND operands"
                )
            left = self.read_vector_operand(operands[0], source, width=128)
            right = self.read_vector_operand(operands[1], source, width=128)
            result = self.emit(
                Handler.And, (left, right), source, dtype="int128",
                machine_vector_bit_pattern=True, width=128,
            )
            self.write_vector_operand(
                operands[0], result, source, width=128,
            )
            return
        if semantic is MachineSemanticToken.VECTOR_INSERT_128_LANE:
            if (
                len(operands) != 4
                or not isinstance(operands[0], VectorRegisterOperand)
                or not isinstance(operands[1], VectorRegisterOperand)
                or not isinstance(
                    operands[2], (VectorRegisterOperand, EffectiveAddressOperand),
                )
                or not isinstance(operands[3], ImmediateOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed VINSERTF128 operands"
                )
            base = self.read_vector_operand(operands[1], source, width=256)
            inserted = self.read_vector_operand(operands[2], source, width=128)
            shift = 128 if int(operands[3].value) & 1 else 0
            lane_mask = ((1 << 128) - 1) << shift
            retained = self.emit(
                Handler.And,
                (
                    base,
                    self.constant(
                        ((1 << 256) - 1) ^ lane_mask,
                        source, dtype="int256",
                    ),
                ),
                source, dtype="int256", machine_vector_lane_retained=True,
                lane_width=128, lane_index=shift // 128,
            )
            lane = inserted
            if shift:
                lane = self.emit(
                    Handler.Shl,
                    (lane, self.constant(shift, source, dtype="int64")),
                    source, dtype="int256", machine_vector_lane_shift=True,
                )
            result = self.emit(
                Handler.Or, (retained, lane), source, dtype="int256",
                machine_vector_insert_lane=True, lane_width=128,
                lane_index=shift // 128, vector_width=256,
            )
            self.write_vector_operand(operands[0], result, source, width=256)
            return
        if semantic is MachineSemanticToken.VECTOR_SHIFT_RIGHT_LOGICAL:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(operands[1], ImmediateOperand):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed vector byte-shift operands"
                )
            value = self.read_vector_operand(operands[0], source, width=128)
            byte_count = int(operands[1].value)
            if byte_count >= 16:
                result = self.constant(0, source, dtype="int128")
            else:
                result = self.emit(
                    Handler.Shr,
                    (
                        value,
                        self.constant(byte_count * 8, source, dtype="int64"),
                    ),
                    source, dtype="int128", machine_vector_byte_shift=True,
                    vector_width=128, zero_fill=True,
                )
            self.write_vector_operand(
                operands[0], result, source, width=128,
            )
            return
        if semantic in {
            MachineSemanticToken.VECTOR_UNPACK_LOW_QWORDS,
            MachineSemanticToken.VECTOR_UNPACK_LOW_BYTES,
            MachineSemanticToken.VECTOR_UNPACK_LOW_WORDS,
        }:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed vector unpack operands"
                )
            lane_width = {
                X86InstructionToken.PUNPCKLBW_XMM_XMMM128: 8,
                X86InstructionToken.PUNPCKLWD_XMM_XMMM128: 16,
                X86InstructionToken.PUNPCKLQDQ_XMM_XMMM128: 64,
            }.get(token)
            if lane_width is None:
                raise MachineLiftError(
                    f"{source.address:#x}: no unpack lane contract for {token.name}"
                )
            left = self.read_vector_operand(operands[0], source, width=128)
            right = self.read_vector_operand(operands[1], source, width=128)
            result = self.emit(
                Handler.VectorUnpackLow, (left, right), source,
                dtype="int128", lane_width=lane_width, vector_width=128,
                machine_vector_bit_pattern=True,
            )
            self.write_vector_operand(
                operands[0], result, source, width=128,
            )
            return
        if semantic is MachineSemanticToken.VECTOR_ADD_QWORDS:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed packed vector add operands"
                )
            left = self.read_vector_operand(operands[0], source, width=128)
            right = self.read_vector_operand(operands[1], source, width=128)
            result = self.emit(
                Handler.VectorAddModulo, (left, right), source,
                dtype="int128", lane_width=64, vector_width=128,
                machine_vector_bit_pattern=True,
            )
            self.write_vector_operand(
                operands[0], result, source, width=128,
            )
            return
        if semantic is MachineSemanticToken.VECTOR_COMPARE_EQUAL_QWORDS:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed packed equality operands"
                )
            left = self.read_vector_operand(operands[0], source, width=128)
            right = self.read_vector_operand(operands[1], source, width=128)
            result = self.emit(
                Handler.VectorCompareEqualMask, (left, right), source,
                dtype="int128", lane_width=64, vector_width=128,
                true_lane_mask=(1 << 64) - 1, false_lane_mask=0,
                machine_vector_bit_pattern=True,
            )
            self.write_vector_operand(
                operands[0], result, source, width=128,
            )
            return
        if semantic is MachineSemanticToken.VECTOR_SUBTRACT_QWORDS:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed packed vector subtract operands"
                )
            left = self.read_vector_operand(operands[0], source, width=128)
            right = self.read_vector_operand(operands[1], source, width=128)
            result = self.emit(
                Handler.VectorSubtractModulo, (left, right), source,
                dtype="int128", lane_width=64, vector_width=128,
                machine_vector_bit_pattern=True,
            )
            self.write_vector_operand(
                operands[0], result, source, width=128,
            )
            return
        if semantic is MachineSemanticToken.VECTOR_SHUFFLE_DWORDS:
            if len(operands) != 3 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(operands[2], ImmediateOperand):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed packed dword shuffle operands"
                )
            value = self.read_vector_operand(operands[1], source, width=128)
            control = int(operands[2].value) & 0xFF
            lanes = tuple((control >> (2 * index)) & 0x3 for index in range(4))
            result = self.emit(
                Handler.VectorShuffle, (value,), source,
                dtype="int128", lane_width=32, vector_width=128,
                lane_indices=lanes, machine_vector_bit_pattern=True,
            )
            self.write_vector_operand(operands[0], result, source, width=128)
            return
        if semantic is MachineSemanticToken.VECTOR_SIGNED_INT32_TO_FLOAT64:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed packed int32-to-binary64 operands"
                )
            packed_source = self.read_vector_operand(
                operands[1], source, width=64,
            )
            current_mxcsr = self.mxcsr_value()
            result = self.emit(
                Handler.VectorSInt32ToFloat64Bits,
                (packed_source,), source,
                dtype="int128", source_lane_width=32,
                target_lane_width=64, lane_count=2,
                target_format="ieee754-binary64", exact_conversion=True,
                integer_only_encoding=True, host_float_independent=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRVectorSInt32ToFloat64,
                (current_mxcsr, packed_source), source,
                dtype="int32", lane_count=2, exact_conversion=True,
                host_float_independent=True,
            )
            self.write_vector_operand(operands[0], result, source, width=128)
            return
        if semantic is MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT64:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed signed float conversion"
                )
            integer = self.read_operand(operands[1], source, width=64)
            current_mxcsr = self.mxcsr_value()
            encoded = self.emit(
                Handler.SInt64ToFloat64Bits,
                (integer, current_mxcsr), source,
                dtype="int64", source_width=64, target_format="binary64",
                rounding_source="mxcsr", integer_only_encoding=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRPrecision,
                (current_mxcsr, integer), source,
                dtype="int32", may_trap=True,
                exception="precision", status_bit=5, mask_bit=12,
                exact_magnitude_bits=53,
            )
            self.write_vector_operand(
                operands[0], encoded, source, width=64,
                preserve_upper=True,
            )
            return
        if semantic is MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT32:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed signed binary32 conversion"
                )
            integer = self.read_operand(operands[1], source, width=64)
            current_mxcsr = self.mxcsr_value()
            encoded = self.emit(
                Handler.SInt64ToFloat32Bits,
                (integer, current_mxcsr), source,
                dtype="int32", source_width=64, target_format="binary32",
                rounding_source="mxcsr", integer_only_encoding=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRPrecision,
                (current_mxcsr, integer), source,
                dtype="int32", may_trap=True,
                exception="precision", status_bit=5, mask_bit=12,
                exact_magnitude_bits=24,
            )
            self.write_vector_operand(
                operands[0], encoded, source, width=32,
                preserve_upper=True,
            )
            return
        if semantic is MachineSemanticToken.VECTOR_MOVE_LOW_ZERO_UPPER:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (
                    VectorRegisterOperand, RegisterOperand,
                    EffectiveAddressOperand,
                )
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed vector low move operands"
                )
            width = 32 if token is X86InstructionToken.MOVD_XMM_RM32 else 64
            moved = self.read_vector_operand(
                operands[1], source, width=width,
            )
            self.write_vector_operand(
                operands[0], moved, source, width=width,
                preserve_upper=False,
            )
            return
        if semantic in {
            MachineSemanticToken.SCALAR_FLOAT64_COMPARE_UNORDERED,
            MachineSemanticToken.SCALAR_FLOAT64_COMPARE_ORDERED,
            MachineSemanticToken.SCALAR_FLOAT32_COMPARE_ORDERED,
        }:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed scalar float comparison"
                )
            binary32 = semantic is MachineSemanticToken.SCALAR_FLOAT32_COMPARE_ORDERED
            width = 32 if binary32 else 64
            left = self.read_vector_operand(operands[0], source, width=width)
            right = self.read_vector_operand(operands[1], source, width=width)
            left_nan = self._bool(
                Handler.Float32IsNaNBits if binary32 else Handler.Float64IsNaNBits,
                (left,), source,
                machine_float_bits=True,
            )
            right_nan = self._bool(
                Handler.Float32IsNaNBits if binary32 else Handler.Float64IsNaNBits,
                (right,), source,
                machine_float_bits=True,
            )
            unordered = self._bool(
                Handler.LOr, (left_nan, right_nan), source,
                machine_float_unordered=True,
            )
            if semantic in {
                MachineSemanticToken.SCALAR_FLOAT64_COMPARE_ORDERED,
                MachineSemanticToken.SCALAR_FLOAT32_COMPARE_ORDERED,
            }:
                invalid = unordered
            else:
                left_snan = self._bool(
                    Handler.Float64IsSignalingNaNBits, (left,), source,
                    machine_float_bits=True,
                )
                right_snan = self._bool(
                    Handler.Float64IsSignalingNaNBits, (right,), source,
                    machine_float_bits=True,
                )
                invalid = self._bool(
                    Handler.LOr, (left_snan, right_snan), source,
                    machine_float_signaling_nan=True,
                )
            self.mxcsr = self.emit(
                Handler.MXCSRInvalid,
                (self.mxcsr_value(), invalid), source,
                dtype="int32", may_trap=True,
                exception="invalid-operation", status_bit=0, mask_bit=7,
            )
            less = self._bool(
                Handler.Float32BitsLt if binary32 else Handler.Float64BitsLt,
                (left, right), source,
                machine_float_bits=True,
            )
            equal = self._bool(
                Handler.Float32BitsEq if binary32 else Handler.Float64BitsEq,
                (left, right), source,
                machine_float_bits=True,
            )
            false = self.bool_constant(False, source)
            true = self.bool_constant(True, source)
            # UCOMISD/COMISD: unordered=111, greater=000, less=001,
            # equal=100 in the CF/PF/ZF ordering. The explicit comparisons
            # document the mutually exclusive ordinary cases.
            ordered_cf = self.emit(
                Handler.Select, (less, true, false), source,
                dtype="bool", machine_float_case="less",
            )
            ordered_zf = self.emit(
                Handler.Select, (equal, true, false), source,
                dtype="bool", machine_float_case="equal",
            )
            self.flags["CF"] = self.emit(
                Handler.Select, (unordered, true, ordered_cf), source,
                dtype="bool", machine_flag="CF",
            )
            self.flags["PF"] = self.emit(
                Handler.Select, (unordered, true, false), source,
                dtype="bool", machine_flag="PF",
            )
            self.flags["ZF"] = self.emit(
                Handler.Select, (unordered, true, ordered_zf), source,
                dtype="bool", machine_flag="ZF",
            )
            self.flags["OF"] = false
            self.flags["SF"] = false
            self.flags["AF"] = false
            return
        if semantic is MachineSemanticToken.SCALAR_FLOAT64_ADD:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed scalar float addition"
                )
            left = self.read_vector_operand(operands[0], source, width=64)
            right = self.read_vector_operand(operands[1], source, width=64)
            current_mxcsr = self.mxcsr_value()
            encoded = self.emit(
                Handler.Float64AddBits,
                (left, right, current_mxcsr), source,
                dtype="int64", format="ieee754-binary64",
                rounding_source="mxcsr", daz_source="mxcsr-bit-6",
                ftz_source="mxcsr-bit-15", host_float_independent=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRFloat64Add,
                (current_mxcsr, left, right), source,
                dtype="int32", may_trap=True,
                exception_status_bits=(0, 1, 2, 3, 4, 5),
                exception_mask_bits=(7, 8, 9, 10, 11, 12),
                exception_order=(
                    "invalid", "denormal", "divide-by-zero",
                    "overflow", "underflow", "precision",
                ),
                trap_before_destination_write=True,
                host_float_independent=True,
            )
            self.write_vector_operand(
                operands[0], encoded, source, width=64,
                preserve_upper=True,
            )
            return
        if semantic is MachineSemanticToken.SCALAR_FLOAT32_ADD:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed scalar binary32 addition"
                )
            left = self.read_vector_operand(operands[0], source, width=32)
            right = self.read_vector_operand(operands[1], source, width=32)
            current_mxcsr = self.mxcsr_value()
            encoded = self.emit(
                Handler.Float32AddBits,
                (left, right, current_mxcsr), source,
                dtype="int32", format="ieee754-binary32",
                rounding_source="mxcsr", daz_source="mxcsr-bit-6",
                ftz_source="mxcsr-bit-15", host_float_independent=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRFloat32Add,
                (current_mxcsr, left, right), source,
                dtype="int32", may_trap=True,
                exception_status_bits=(0, 1, 2, 3, 4, 5),
                exception_mask_bits=(7, 8, 9, 10, 11, 12),
                exception_order=(
                    "invalid", "denormal", "divide-by-zero",
                    "overflow", "underflow", "precision",
                ),
                trap_before_destination_write=True,
                host_float_independent=True,
            )
            self.write_vector_operand(
                operands[0], encoded, source, width=32,
                preserve_upper=True,
            )
            return
        if semantic is MachineSemanticToken.SCALAR_FLOAT32_DIVIDE:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed scalar binary32 division"
                )
            left = self.read_vector_operand(operands[0], source, width=32)
            right = self.read_vector_operand(operands[1], source, width=32)
            current_mxcsr = self.mxcsr_value()
            encoded = self.emit(
                Handler.Float32DivideBits,
                (left, right, current_mxcsr), source,
                dtype="int32", format="ieee754-binary32",
                rounding_source="mxcsr", daz_source="mxcsr-bit-6",
                ftz_source="mxcsr-bit-15", host_float_independent=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRFloat32Divide,
                (current_mxcsr, left, right), source,
                dtype="int32", may_trap=True,
                exception_status_bits=(0, 1, 2, 3, 4, 5),
                exception_mask_bits=(7, 8, 9, 10, 11, 12),
                exception_order=(
                    "invalid", "denormal", "divide-by-zero",
                    "overflow", "underflow", "precision",
                ),
                trap_before_destination_write=True,
                host_float_independent=True,
            )
            self.write_vector_operand(
                operands[0], encoded, source, width=32,
                preserve_upper=True,
            )
            return
        if semantic is MachineSemanticToken.SCALAR_FLOAT64_DIVIDE:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed scalar binary64 division"
                )
            left = self.read_vector_operand(operands[0], source, width=64)
            right = self.read_vector_operand(operands[1], source, width=64)
            current_mxcsr = self.mxcsr_value()
            encoded = self.emit(
                Handler.Float64DivideBits,
                (left, right, current_mxcsr), source,
                dtype="int64", format="ieee754-binary64",
                rounding_source="mxcsr", daz_source="mxcsr-bit-6",
                ftz_source="mxcsr-bit-15", host_float_independent=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRFloat64Divide,
                (current_mxcsr, left, right), source,
                dtype="int32", may_trap=True,
                exception_status_bits=(0, 1, 2, 3, 4, 5),
                exception_mask_bits=(7, 8, 9, 10, 11, 12),
                exception_order=(
                    "invalid", "denormal", "divide-by-zero",
                    "overflow", "underflow", "precision",
                ),
                trap_before_destination_write=True,
                host_float_independent=True,
            )
            self.write_vector_operand(
                operands[0], encoded, source, width=64,
                preserve_upper=True,
            )
            return
        if semantic is MachineSemanticToken.SCALAR_FLOAT64_SUBTRACT:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed scalar binary64 subtraction"
                )
            left = self.read_vector_operand(operands[0], source, width=64)
            right = self.read_vector_operand(operands[1], source, width=64)
            current_mxcsr = self.mxcsr_value()
            encoded = self.emit(
                Handler.Float64SubtractBits,
                (left, right, current_mxcsr), source,
                dtype="int64", format="ieee754-binary64",
                rounding_source="mxcsr", daz_source="mxcsr-bit-6",
                ftz_source="mxcsr-bit-15", host_float_independent=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRFloat64Subtract,
                (current_mxcsr, left, right), source,
                dtype="int32", may_trap=True,
                exception_status_bits=(0, 1, 2, 3, 4, 5),
                exception_mask_bits=(7, 8, 9, 10, 11, 12),
                exception_order=(
                    "invalid", "denormal", "divide-by-zero",
                    "overflow", "underflow", "precision",
                ),
                trap_before_destination_write=True,
                host_float_independent=True,
            )
            self.write_vector_operand(
                operands[0], encoded, source, width=64,
                preserve_upper=True,
            )
            return
        if semantic is MachineSemanticToken.SCALAR_FLOAT64_TO_SIGNED_INT64_TRUNCATE:
            if len(operands) != 2 or not isinstance(
                operands[0], RegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed truncating binary64 conversion"
                )
            encoded_source = self.read_vector_operand(
                operands[1], source, width=64,
            )
            current_mxcsr = self.mxcsr_value()
            result = self.emit(
                Handler.Float64ToSInt64TruncBits,
                (encoded_source,), source,
                dtype="int64", source_format="ieee754-binary64",
                rounding="toward-zero", ignores_mxcsr_rounding=True,
                invalid_result=-(1 << 63), host_float_independent=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRFloat64ToSIntInvalid,
                (current_mxcsr, encoded_source), source,
                dtype="int32", may_trap=True,
                exception="invalid-operation", status_bit=0, mask_bit=7,
                invalid_cases=("nan", "infinity", "signed-int64-out-of-range"),
                trap_before_destination_write=True,
                host_float_independent=True,
            )
            self.write_operand(operands[0], result, source, width=64)
            return
        if semantic is MachineSemanticToken.SCALAR_FLOAT64_TO_SIGNED_INT32_TRUNCATE:
            if len(operands) != 2 or not isinstance(
                operands[0], RegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed truncating binary64-to-int32 conversion"
                )
            encoded_source = self.read_vector_operand(
                operands[1], source, width=64,
            )
            current_mxcsr = self.mxcsr_value()
            result = self.emit(
                Handler.Float64ToSInt32TruncBits,
                (encoded_source,), source,
                dtype="int32", source_format="ieee754-binary64",
                rounding="toward-zero", ignores_mxcsr_rounding=True,
                invalid_result=-(1 << 31), host_float_independent=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRFloat64ToSIntInvalid,
                (current_mxcsr, encoded_source), source,
                dtype="int32", may_trap=True,
                exception="invalid-operation", status_bit=0, mask_bit=7,
                invalid_cases=("nan", "infinity", "signed-int32-out-of-range"),
                trap_before_destination_write=True,
                host_float_independent=True,
            )
            self.write_operand(operands[0], result, source, width=32)
            return
        if semantic is MachineSemanticToken.SCALAR_FLOAT64_MULTIPLY:
            if len(operands) != 2 or not isinstance(
                operands[0], VectorRegisterOperand
            ) or not isinstance(
                operands[1], (VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed scalar float multiplication"
                )
            left = self.read_vector_operand(operands[0], source, width=64)
            right = self.read_vector_operand(operands[1], source, width=64)
            current_mxcsr = self.mxcsr_value()
            encoded = self.emit(
                Handler.Float64MultiplyBits,
                (left, right, current_mxcsr), source,
                dtype="int64", format="ieee754-binary64",
                rounding_source="mxcsr", daz_source="mxcsr-bit-6",
                ftz_source="mxcsr-bit-15", host_float_independent=True,
            )
            self.mxcsr = self.emit(
                Handler.MXCSRFloat64Multiply,
                (current_mxcsr, left, right), source,
                dtype="int32", may_trap=True,
                exception_status_bits=(0, 1, 2, 3, 4, 5),
                exception_mask_bits=(7, 8, 9, 10, 11, 12),
                exception_order=(
                    "invalid", "denormal", "divide-by-zero",
                    "overflow", "underflow", "precision",
                ),
                trap_before_destination_write=True,
                host_float_independent=True,
            )
            self.write_vector_operand(
                operands[0], encoded, source, width=64,
                preserve_upper=True,
            )
            return
        if semantic is MachineSemanticToken.INTEGER_MULTIPLY:
            if len(operands) == 1:
                (right_operand,) = operands
                if token not in {
                    X86InstructionToken.IMUL_RM32,
                    X86InstructionToken.IMUL_RM64,
                } or not isinstance(
                    right_operand, (RegisterOperand, EffectiveAddressOperand)
                ):
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed accumulator-form IMUL"
                    )
                width = 32 if token is X86InstructionToken.IMUL_RM32 else 64
                accumulator_operand = RegisterOperand(X86Register.RAX, width=width)
                high_operand = RegisterOperand(X86Register.RDX, width=width)
                left = self.read_operand(accumulator_operand, source, width=width)
                right = self.read_operand(right_operand, source, width=width)
                low = self.emit(
                    Handler.SMulLow, (left, right), source, dtype=f"int{width}",
                    width=width, signed_fixed_width=True,
                )
                high = self.emit(
                    Handler.SMulHigh, (left, right), source, dtype=f"int{width}",
                    width=width, signed_fixed_width=True,
                )
                overflow = self._bool(
                    Handler.SMulOverflow, (left, right), source,
                    width=width, machine_flag="CF/OF", signed_fixed_width=True,
                )
                self.flags["CF"] = overflow
                self.flags["OF"] = overflow
                self.write_operand(accumulator_operand, low, source, width=width)
                self.write_operand(high_operand, high, source, width=width)
                return
            if len(operands) == 2:
                destination, right_operand = operands
                if not isinstance(
                    destination, (RegisterOperand, EffectiveAddressOperand)
                ) or not isinstance(
                    right_operand, (RegisterOperand, EffectiveAddressOperand)
                ):
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed two-operand IMUL"
                    )
                width = self.operand_width(source, 0)
                left = self.read_operand(destination, source, width=width)
                right = self.read_operand(right_operand, source, width=width)
            elif len(operands) == 3:
                destination, left_operand, immediate = operands
                if not isinstance(destination, RegisterOperand) or not isinstance(
                    left_operand, (RegisterOperand, EffectiveAddressOperand)
                ) or not isinstance(immediate, ImmediateOperand):
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed three-operand IMUL"
                    )
                width = self.operand_width(source, 0)
                left = self.read_operand(left_operand, source, width=width)
                right = self.read_operand(immediate, source, width=width)
            else:
                raise MachineLiftError(
                    f"{source.address:#x}: IMUL requires two or three operands"
                )
            product = self.emit(
                Handler.SMulLow, (left, right), source,
                dtype=f"int{width}", width=width,
                signed_fixed_width=True,
            )
            overflow = self._bool(
                Handler.SMulOverflow, (left, right), source,
                width=width, machine_flag="CF/OF",
                signed_fixed_width=True,
            )
            self.flags["CF"] = overflow
            self.flags["OF"] = overflow
            # PF/AF/ZF/SF are architecturally undefined after IMUL.  The
            # deterministic machine model preserves their incoming values.
            self.write_operand(destination, product, source, width=width)
            return
        if semantic is MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED:
            if len(operands) != 1 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed accumulator-form MUL"
                )
            width = self.operand_width(source, 0)
            if width not in (32, 64):
                raise MachineLiftError(
                    f"{source.address:#x}: MUL width {width} is not legalized"
                )
            accumulator_operand = RegisterOperand(X86Register.RAX, width=width)
            high_operand = RegisterOperand(X86Register.RDX, width=width)
            accumulator = self.read_operand(
                accumulator_operand, source, width=width,
            )
            multiplier = self.read_operand(operands[0], source, width=width)
            low = self.emit(
                Handler.UMulLow, (accumulator, multiplier), source,
                dtype=f"int{width}", width=width,
                unsigned_fixed_width=True,
            )
            high = self.emit(
                Handler.UMulHigh, (accumulator, multiplier), source,
                dtype=f"int{width}", width=width,
                unsigned_fixed_width=True,
            )
            overflow = self._bool(
                Handler.Ne,
                (high, self.constant(0, source, dtype=f"int{width}")),
                source, machine_flag="CF/OF", unsigned_multiply_high=True,
            )
            self.flags["CF"] = overflow
            self.flags["OF"] = overflow
            # AF/PF/SF/ZF are undefined; retain their deterministic incoming
            # representatives just as the machine interpreter does.
            self.write_operand(accumulator_operand, low, source, width=width)
            self.write_operand(high_operand, high, source, width=width)
            return
        if semantic in {
            MachineSemanticToken.INTEGER_DIVIDE,
            MachineSemanticToken.INTEGER_DIVIDE_SIGNED,
        }:
            if len(operands) != 1 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed accumulator-form divide"
                )
            width = self.operand_width(source, 0)
            if width not in (32, 64):
                raise MachineLiftError(
                    f"{source.address:#x}: divide width {width} is not legalized"
                )
            signed = semantic is MachineSemanticToken.INTEGER_DIVIDE_SIGNED
            low_operand = RegisterOperand(X86Register.RAX, width=width)
            high_operand = RegisterOperand(X86Register.RDX, width=width)
            low = self.read_operand(low_operand, source, width=width)
            high = self.read_operand(high_operand, source, width=width)
            divisor = self.read_operand(operands[0], source, width=width)
            guard = self.emit(
                Handler.WideDivCheck, (high, low, divisor), source,
                dtype="machine_divide_guard", width=width, signed=signed,
                dividend_order="high:low", may_trap=True,
                traps=("zero-divisor", "quotient-overflow"),
            )
            quotient = self.emit(
                Handler.WideDivQuotient, (high, low, divisor, guard), source,
                dtype=f"int{width}", width=width, signed=signed,
                dividend_order="high:low", truncation=(
                    "toward-zero" if signed else "unsigned"
                ),
            )
            remainder = self.emit(
                Handler.WideDivRemainder, (high, low, divisor, guard), source,
                dtype=f"int{width}", width=width, signed=signed,
                dividend_order="high:low", remainder_sign=(
                    "dividend" if signed else "nonnegative"
                ),
            )
            # DIV/IDIV leave arithmetic flags undefined.  Preserve the
            # incoming deterministic representatives, and update only the
            # architectural implicit destinations after the checked guard.
            self.write_operand(low_operand, quotient, source, width=width)
            self.write_operand(high_operand, remainder, source, width=width)
            return
        if semantic is MachineSemanticToken.BIT_SCAN_REVERSE:
            if len(operands) != 2 or not isinstance(
                operands[0], RegisterOperand
            ) or not isinstance(
                operands[1], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed BSR operands"
                )
            destination, source_operand = operands
            width = self.operand_width(source, 0)
            previous = self.read_operand(destination, source, width=width)
            value = self.read_operand(source_operand, source, width=width)
            zero = self.constant(0, source, dtype=f"int{width}")
            is_zero = self._bool(
                Handler.Eq, (value, zero), source,
                machine_flag="ZF", machine_bit_scan_reverse=True,
            )
            candidate = self.emit(
                Handler.MsbIndex, (value,), source, dtype=f"int{width}",
                width=width, zero_totalized=True,
                machine_bit_scan_reverse=True,
            )
            selected = self.emit(
                Handler.Select, (is_zero, previous, candidate), source,
                dtype=f"int{width}",
                machine_undefined_destination="preserve-prior",
            )
            self.flags["ZF"] = is_zero
            # CF/PF/AF/SF/OF are undefined; preserve the deterministic
            # machine representative already carried in self.flags.
            self.write_operand(destination, selected, source, width=width)
            return
        if semantic in {
            MachineSemanticToken.INTEGER_ADD,
            MachineSemanticToken.INTEGER_SUBTRACT,
            MachineSemanticToken.BITWISE_AND,
            MachineSemanticToken.BITWISE_OR,
            MachineSemanticToken.BITWISE_XOR,
        }:
            if len(operands) != 2 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(
                operands[1], (RegisterOperand, EffectiveAddressOperand, ImmediateOperand)
            ):
                raise MachineLiftError(f"{source.address:#x}: malformed binary operands")
            width = self.operand_width(source, 0)
            left = self.read_operand(operands[0], source, width=width)
            right = self.read_operand(operands[1], source, width=width)
            handler = {
                MachineSemanticToken.INTEGER_ADD: Handler.Add,
                MachineSemanticToken.INTEGER_SUBTRACT: Handler.Sub,
                MachineSemanticToken.BITWISE_AND: Handler.And,
                MachineSemanticToken.BITWISE_OR: Handler.Or,
                MachineSemanticToken.BITWISE_XOR: Handler.Xor,
            }[semantic]
            raw = self.emit(handler, (left, right), source, dtype=f"int{width}")
            result = self.truncate_bits(raw, width, source)
            if semantic in {MachineSemanticToken.INTEGER_ADD, MachineSemanticToken.INTEGER_SUBTRACT}:
                self.arithmetic_flags(
                    left, right, raw, width, source,
                    subtract=semantic is MachineSemanticToken.INTEGER_SUBTRACT,
                )
            else:
                self.logical_flags(result, width, source)
            self.write_operand(operands[0], result, source, width=width)
            return
        if semantic is MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW:
            if len(operands) != 2 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(
                operands[1], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(f"{source.address:#x}: malformed SBB operands")
            width = self.operand_width(source, 0)
            left = self.read_operand(operands[0], source, width=width)
            right = self.read_operand(operands[1], source, width=width)
            incoming_cf = self.flag("CF")
            borrow = self.emit(
                Handler.Select,
                (
                    incoming_cf,
                    self.constant(1, source, dtype="int64"),
                    self.constant(0, source, dtype="int64"),
                ),
                source, dtype="int64", machine_borrow_input=True,
            )
            effective_raw = self.emit(
                Handler.Add, (right, borrow), source, dtype=f"int{width}",
            )
            effective = self.truncate_bits(effective_raw, width, source)
            raw = self.emit(
                Handler.Sub, (left, effective), source, dtype=f"int{width}",
            )
            result = self.truncate_bits(raw, width, source)
            ordinary_borrow = self._bool(
                Handler.ULt, (left, right), source,
                comparison_unsigned=True, machine_borrow_without_carry=True,
            )
            self.arithmetic_flags(
                left, effective, raw, width, source, subtract=True,
            )
            # SBB's signed-overflow and auxiliary-carry definitions use the
            # authored source plus the final result.  Substituting the
            # width-truncated ``source + CF`` loses the wrap case
            # (for example INT_MIN - INT_MAX - 1).
            xor_lr = self.emit(
                Handler.Xor, (left, right), source, dtype="int64",
                machine_subtract_with_borrow=True,
            )
            xor_result = self.emit(
                Handler.Xor, (left, result), source, dtype="int64",
                machine_subtract_with_borrow=True,
            )
            overflow_bits = self.emit(
                Handler.And, (xor_lr, xor_result), source, dtype="int64",
                machine_subtract_with_borrow=True,
            )
            overflow_bit = self.emit(
                Handler.And,
                (
                    overflow_bits,
                    self.constant(1 << (width - 1), source, dtype="int64"),
                ),
                source, dtype="int64",
            )
            self.flags["OF"] = self._bool(
                Handler.Ne,
                (overflow_bit, self.constant(0, source, dtype="int64")),
                source, machine_flag="OF", machine_subtract_with_borrow=True,
            )
            auxiliary_bits = self.emit(
                Handler.Xor, (xor_lr, result), source, dtype="int64",
                machine_subtract_with_borrow=True,
            )
            auxiliary_bit = self.emit(
                Handler.And,
                (
                    auxiliary_bits,
                    self.constant(0x10, source, dtype="int64"),
                ),
                source, dtype="int64",
            )
            self.flags["AF"] = self._bool(
                Handler.Ne,
                (auxiliary_bit, self.constant(0, source, dtype="int64")),
                source, machine_flag="AF", machine_subtract_with_borrow=True,
            )
            equal_before_borrow = self._bool(
                Handler.Eq, (left, right), source,
                machine_borrow_boundary=True,
            )
            borrowed_equal = self._bool(
                Handler.LAnd, (incoming_cf, equal_before_borrow), source,
                machine_borrow_boundary=True,
            )
            self.flags["CF"] = self._bool(
                Handler.LOr, (borrowed_equal, ordinary_borrow), source,
                machine_flag="CF", machine_subtract_with_borrow=True,
            )
            self.write_operand(operands[0], result, source, width=width)
            return
        if semantic is MachineSemanticToken.BITWISE_NOT:
            if len(operands) != 1 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(f"{source.address:#x}: malformed NOT operand")
            width = self.operand_width(source, 0)
            value = self.read_operand(operands[0], source, width=width)
            inverted = self.emit(
                Handler.Xor,
                (
                    value,
                    self.constant((1 << width) - 1, source, dtype="int64"),
                ),
                source, dtype=f"int{width}", flags_effect="unchanged",
                machine_bitwise_not_width=width,
            )
            self.write_operand(operands[0], inverted, source, width=width)
            return
        if semantic in {
            MachineSemanticToken.INTEGER_INCREMENT,
            MachineSemanticToken.INTEGER_DECREMENT,
            MachineSemanticToken.INTEGER_NEGATE,
        }:
            if len(operands) != 1 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed unary arithmetic operand"
                )
            width = self.operand_width(source, 0)
            value = self.read_operand(operands[0], source, width=width)
            one = self.constant(1, source, dtype="int64")
            zero = self.constant(0, source, dtype="int64")
            if semantic is MachineSemanticToken.INTEGER_INCREMENT:
                raw = self.emit(Handler.Add, (value, one), source, dtype=f"int{width}")
                self.arithmetic_flags(
                    value, one, raw, width, source,
                    subtract=False, preserve_cf=True,
                )
            elif semantic is MachineSemanticToken.INTEGER_DECREMENT:
                raw = self.emit(Handler.Sub, (value, one), source, dtype=f"int{width}")
                self.arithmetic_flags(
                    value, one, raw, width, source,
                    subtract=True, preserve_cf=True,
                )
            else:
                raw = self.emit(Handler.Sub, (zero, value), source, dtype=f"int{width}")
                self.arithmetic_flags(
                    zero, value, raw, width, source, subtract=True,
                )
            result = self.truncate_bits(raw, width, source)
            self.write_operand(operands[0], result, source, width=width)
            return
        if semantic in {
            MachineSemanticToken.INTEGER_COMPARE,
            MachineSemanticToken.INTEGER_TEST,
        }:
            if len(operands) != 2 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(
                operands[1], (RegisterOperand, EffectiveAddressOperand, ImmediateOperand)
            ):
                raise MachineLiftError(f"{source.address:#x}: malformed comparison operands")
            width = self.operand_width(source, 0)
            left = self.read_operand(operands[0], source, width=width)
            right = self.read_operand(operands[1], source, width=width)
            if semantic is MachineSemanticToken.INTEGER_COMPARE:
                raw = self.emit(Handler.Sub, (left, right), source, dtype=f"int{width}")
                self.arithmetic_flags(left, right, raw, width, source, subtract=True)
            else:
                raw = self.emit(Handler.And, (left, right), source, dtype=f"int{width}")
                self.logical_flags(raw, width, source)
            return
        if semantic is MachineSemanticToken.CONDITIONAL_MOVE:
            if len(operands) != 2 or not isinstance(
                operands[0], RegisterOperand
            ) or not isinstance(
                operands[1], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed conditional-move operands"
                )
            width = self.operand_width(source, 0)
            prior = self.read_operand(operands[0], source, width=width)
            candidate = self.read_operand(operands[1], source, width=width)
            selected = self.emit(
                Handler.Select,
                (self.condition(source), candidate, prior),
                source,
                dtype=f"int{width}",
                machine_condition=source.token.name.split("_", 1)[0],
            )
            self.write_operand(operands[0], selected, source, width=width)
            return
        if semantic is MachineSemanticToken.CONDITIONAL_SET:
            if len(operands) != 1 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed conditional-set operand"
                )
            selected = self.emit(
                Handler.Select,
                (
                    self.condition(source),
                    self.constant(1, source, dtype="int8"),
                    self.constant(0, source, dtype="int8"),
                ),
                source,
                dtype="int8",
                machine_condition=source.token.name.split("_", 1)[0],
            )
            self.write_operand(operands[0], selected, source, width=8)
            return
        if semantic is MachineSemanticToken.BYTE_SWAP:
            if len(operands) != 1 or not isinstance(operands[0], RegisterOperand):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed byte-swap operand"
                )
            width = self.operand_width(source, 0)
            value = self.read_operand(operands[0], source, width=width)
            swapped = self.emit(
                Handler.ByteSwap, (value,), source, dtype=f"int{width}",
                width=width, byte_order="reverse",
            )
            self.write_operand(operands[0], swapped, source, width=width)
            return
        if semantic is MachineSemanticToken.ATOMIC_COMPARE_EXCHANGE:
            if len(operands) != 2 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(operands[1], RegisterOperand):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed compare-exchange operands"
                )
            width = self.operand_width(source, 0)
            accumulator_operand = RegisterOperand(X86Register.RAX, width=width)
            expected = self.read_operand(accumulator_operand, source, width=width)
            desired = self.read_operand(operands[1], source, width=width)
            destination = operands[0]
            pointer = None
            if isinstance(destination, EffectiveAddressOperand):
                pointer = self.effective_address(destination, source)
                observed_input = self.read_operand(destination, source, width=width)
            else:
                observed_input = self.read_operand(destination, source, width=width)
            atomic_args = (
                (self.memory, pointer, expected, desired)
                if pointer is not None else (observed_input, expected, desired)
            )
            observed = self.emit(
                Handler.AtomicCompareExchangeObserved, atomic_args, source,
                dtype=f"int{width}", width=width, ordering="sequentially-consistent",
                locked=0xF0 in source.legacy_prefixes,
            )
            success = self._bool(
                Handler.AtomicCompareExchangeSuccess,
                (observed, expected), source,
                width=width, machine_flag="ZF",
            )
            if pointer is not None:
                self.memory = self.emit(
                    Handler.AtomicCompareExchangeMemory,
                    (self.memory, pointer, expected, desired, observed, success), source,
                    dtype="memory", width=width,
                    ordering="sequentially-consistent", locked=True,
                )
            else:
                selected = self.emit(
                    Handler.Select, (success, desired, observed), source,
                    dtype=f"int{width}", atomic_compare_exchange=True,
                )
                self.write_operand(destination, selected, source, width=width)
            accumulator = self.emit(
                Handler.Select, (success, expected, observed), source,
                dtype=f"int{width}", atomic_compare_exchange_accumulator=True,
            )
            self.write_operand(accumulator_operand, accumulator, source, width=width)
            comparison = self.emit(
                Handler.Sub, (expected, observed), source, dtype=f"int{width}",
            )
            self.arithmetic_flags(
                expected, observed, comparison, width, source, subtract=True,
            )
            self.flags["ZF"] = success
            return
        if semantic is MachineSemanticToken.ATOMIC_EXCHANGE_ADD:
            if len(operands) != 2 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(operands[1], RegisterOperand):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed exchange-add operands"
                )
            width = self.operand_width(source, 0)
            destination, source_operand = operands
            addend = self.read_operand(source_operand, source, width=width)
            pointer = (
                self.effective_address(destination, source)
                if isinstance(destination, EffectiveAddressOperand) else None
            )
            if pointer is not None:
                observed = self.emit(
                    Handler.AtomicExchangeAddObserved,
                    (self.memory, pointer, addend), source,
                    dtype=f"int{width}", width=width,
                    ordering="sequentially-consistent", locked=True,
                )
            else:
                observed = self.read_operand(destination, source, width=width)
            result = self.emit(
                Handler.Add, (observed, addend), source,
                dtype=f"int{width}", modular_width=width,
                atomic_exchange_add=True,
            )
            if pointer is not None:
                self.memory = self.emit(
                    Handler.AtomicExchangeAddMemory,
                    (self.memory, pointer, addend, observed, result), source,
                    dtype="memory", width=width,
                    ordering="sequentially-consistent", locked=True,
                )
            else:
                self.write_operand(destination, result, source, width=width)
            self.write_operand(source_operand, observed, source, width=width)
            self.arithmetic_flags(
                observed, addend, result, width, source, subtract=False,
            )
            return
        if semantic is MachineSemanticToken.ATOMIC_ADD:
            if (
                token is not X86InstructionToken.LOCK_ADD_RM8_R8
                or len(operands) != 2
                or not isinstance(operands[0], EffectiveAddressOperand)
                or not isinstance(operands[1], RegisterOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed locked-add operands"
                )
            destination, source_operand = operands
            width = 8
            pointer = self.effective_address(destination, source)
            addend = self.read_operand(source_operand, source, width=width)
            # The repository atomic exchange-add pair expresses the same
            # indivisible memory transition needed by LOCK ADD.  Unlike XADD,
            # LOCK ADD does not publish the observed value into the source
            # register; only flags and the versioned memory state change.
            observed = self.emit(
                Handler.AtomicExchangeAddObserved,
                (self.memory, pointer, addend),
                source,
                dtype="int8",
                width=width,
                ordering="sequentially-consistent",
                locked=True,
                machine_atomic_operation="lock-add",
            )
            raw = self.emit(
                Handler.Add,
                (observed, addend),
                source,
                dtype="int8",
                modular_width=width,
                machine_atomic_operation="lock-add",
            )
            result = self.truncate_bits(raw, width, source)
            self.memory = self.emit(
                Handler.AtomicExchangeAddMemory,
                (self.memory, pointer, addend, observed, result),
                source,
                dtype="memory",
                width=width,
                ordering="sequentially-consistent",
                locked=True,
                machine_atomic_operation="lock-add",
                source_register_unchanged=True,
            )
            self.arithmetic_flags(
                observed, addend, raw, width, source, subtract=False,
            )
            return
        if semantic is MachineSemanticToken.ATOMIC_INCREMENT:
            if (
                token is not X86InstructionToken.LOCK_INC_RM32
                or len(operands) != 1
                or not isinstance(operands[0], EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed locked-increment operand"
                )
            width = 32
            pointer = self.effective_address(operands[0], source)
            one = self.constant(1, source, dtype="int64")
            observed = self.emit(
                Handler.AtomicExchangeAddObserved,
                (self.memory, pointer, one), source,
                dtype="int32", width=width,
                ordering="sequentially-consistent", locked=True,
                machine_atomic_operation="lock-inc",
            )
            raw = self.emit(
                Handler.Add, (observed, one), source,
                dtype="int32", modular_width=width,
                machine_atomic_operation="lock-inc",
            )
            result = self.truncate_bits(raw, width, source)
            self.memory = self.emit(
                Handler.AtomicExchangeAddMemory,
                (self.memory, pointer, one, observed, result), source,
                dtype="memory", width=width,
                ordering="sequentially-consistent", locked=True,
                machine_atomic_operation="lock-inc",
            )
            self.arithmetic_flags(
                observed, one, raw, width, source,
                subtract=False, preserve_cf=True,
            )
            return
        if semantic is MachineSemanticToken.EXCHANGE:
            if len(operands) != 2 or not all(isinstance(
                operand, (RegisterOperand, EffectiveAddressOperand)
            ) for operand in operands):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed exchange operands"
                )
            width = self.operand_width(source, 0)
            left = self.read_operand(operands[0], source, width=width)
            right = self.read_operand(operands[1], source, width=width)
            self.write_operand(operands[0], right, source, width=width)
            self.write_operand(operands[1], left, source, width=width)
            return
        if semantic in {
            MachineSemanticToken.BIT_TEST,
            MachineSemanticToken.BIT_TEST_RESET,
            MachineSemanticToken.BIT_TEST_COMPLEMENT,
        }:
            if len(operands) != 2 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(
                operands[1], (RegisterOperand, ImmediateOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed bit-test operands"
                )
            width = self.operand_width(source, 0)
            destination = operands[0]
            source_index = operands[1]
            if (
                isinstance(destination, EffectiveAddressOperand)
                and isinstance(source_index, RegisterOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: dynamic memory bit-test awaits "
                    "explicit signed cross-word address lowering"
                )
            pointer = None
            if isinstance(destination, EffectiveAddressOperand):
                assert isinstance(source_index, ImmediateOperand)
                authored_index = int(source_index.value)
                word_offset, bit_index = divmod(authored_index, width)
                pointer = self.effective_address(
                    destination, source, dtype="int64",
                )
                if word_offset:
                    pointer = self.emit(
                        Handler.Add,
                        (
                            pointer,
                            self.constant(
                                word_offset * (width // 8), source,
                                dtype="int64",
                            ),
                        ),
                        source, dtype="int64", machine_bit_word_offset=True,
                    )
                value = self.emit(
                    Handler.Load, (self.memory, pointer), source,
                    dtype=f"int{width}", machine_state="memory", width=width,
                )
                bit = self.constant(bit_index, source, dtype="int64")
            else:
                value = self.read_operand(destination, source, width=width)
                if isinstance(source_index, ImmediateOperand):
                    bit = self.constant(
                        int(source_index.value) % width,
                        source, dtype="int64",
                    )
                else:
                    raw_index = self.read_operand(
                        source_index, source, width=source_index.width,
                    )
                    bit = self.emit(
                        Handler.And,
                        (
                            raw_index,
                            self.constant(width - 1, source, dtype="int64"),
                        ),
                        source, dtype="int64", machine_bit_index_mask=True,
                    )
            shifted = self.emit(
                Handler.Shr,
                (value, bit), source, dtype="int64",
                machine_bit_test_index=True,
            )
            low = self.emit(
                Handler.And,
                (shifted, self.constant(1, source, dtype="int64")),
                source, dtype="int64",
            )
            self.flags["CF"] = self._bool(
                Handler.Ne,
                (low, self.constant(0, source, dtype="int64")),
                source, machine_flag="CF",
            )
            mutation = (
                "set" if token.name.startswith("BTS_")
                else "reset" if token.name.startswith("BTR_")
                else "complement" if token.name.startswith("BTC_")
                else None
            )
            if mutation is not None:
                bit_mask = self.emit(
                    Handler.Shl,
                    (self.constant(1, source, dtype="int64"), bit),
                    source, dtype="int64", machine_bit_mask=True,
                )
                if mutation == "set":
                    result = self.emit(
                        Handler.Or, (value, bit_mask), source,
                        dtype=f"int{width}", machine_bit_set=True,
                    )
                elif mutation == "reset":
                    inverse = self.emit(
                        Handler.Xor,
                        (
                            bit_mask,
                            self.constant(
                                (1 << width) - 1, source, dtype="int64",
                            ),
                        ),
                        source, dtype="int64", machine_bit_mask_inverse=True,
                    )
                    result = self.emit(
                        Handler.And, (value, inverse), source,
                        dtype=f"int{width}", machine_bit_reset=True,
                    )
                else:
                    result = self.emit(
                        Handler.Xor, (value, bit_mask), source,
                        dtype=f"int{width}", machine_bit_complement=True,
                    )
                if pointer is None:
                    self.write_operand(destination, result, source, width=width)
                else:
                    self.memory = self.emit(
                        Handler.Store, (self.memory, pointer, result), source,
                        dtype="memory", machine_state="memory", width=width,
                    )
            return
        if semantic in {
            MachineSemanticToken.ROTATE_LEFT,
            MachineSemanticToken.ROTATE_RIGHT,
        }:
            if len(operands) != 2 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(
                operands[1], (ImmediateOperand, RegisterOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed rotate operands"
                )
            width = self.operand_width(source, 0)
            if isinstance(operands[1], RegisterOperand):
                prior_flags = {
                    flag: self.flag(flag) for flag in self._FLAG_NAMES
                }
                value = self.read_operand(operands[0], source, width=width)
                raw_count = self.read_operand(operands[1], source, width=8)
                count = self.emit(
                    Handler.And,
                    (
                        raw_count,
                        self.constant(
                            0x3F if width == 64 else 0x1F,
                            source, dtype="int64",
                        ),
                    ),
                    source, dtype="int64", machine_masked_rotate_count=True,
                )
                zero = self.constant(0, source, dtype="int64")
                one = self.constant(1, source, dtype="int64")
                active = self._bool(
                    Handler.Ne, (count, zero), source,
                    machine_rotate_count_nonzero=True,
                )
                safe_count = self.emit(
                    Handler.Select, (active, count, one), source,
                    dtype="int64", machine_rotate_safe_count=True,
                )
                complement = self.emit(
                    Handler.Sub,
                    (self.constant(width, source, dtype="int64"), safe_count),
                    source, dtype="int64", machine_rotate_complement=True,
                )
                left_count, right_count = (
                    (safe_count, complement)
                    if semantic is MachineSemanticToken.ROTATE_LEFT
                    else (complement, safe_count)
                )
                direction = (
                    "left" if semantic is MachineSemanticToken.ROTATE_LEFT
                    else "right"
                )
                left = self.emit(
                    Handler.Shl, (value, left_count), source,
                    dtype=f"int{width}", machine_rotate=direction,
                )
                right = self.emit(
                    Handler.Shr, (value, right_count), source,
                    dtype=f"int{width}", machine_rotate=direction,
                )
                combined = self.emit(
                    Handler.Or, (left, right), source,
                    dtype=f"int{width}", machine_rotate=direction,
                )
                rotated = self.truncate_bits(combined, width, source)
                result = self.emit(
                    Handler.Select, (active, rotated, value), source,
                    dtype=f"int{width}", machine_rotate_zero_preserves=True,
                )
                carry_position = (
                    0 if semantic is MachineSemanticToken.ROTATE_LEFT
                    else width - 1
                )
                carry_source = self.emit(
                    Handler.Shr,
                    (rotated, self.constant(carry_position, source, dtype="int64")),
                    source, dtype="int64", machine_rotate_carry=True,
                )
                carry_bit = self.emit(
                    Handler.And,
                    (carry_source, self.constant(1, source, dtype="int64")),
                    source, dtype="int64", machine_rotate_carry=True,
                )
                rotated_cf = self._bool(
                    Handler.Ne, (carry_bit, zero), source,
                    machine_flag="CF", machine_rotate=direction,
                )
                self.flags["CF"] = self._bool(
                    Handler.Select, (active, rotated_cf, prior_flags["CF"]),
                    source, machine_flag="CF", machine_rotate=direction,
                )
                count_is_one = self._bool(
                    Handler.Eq, (count, one), source,
                    machine_rotate_count_one=True,
                )
                msb = self.emit(
                    Handler.And,
                    (rotated, self.constant(1 << (width - 1), source, dtype="int64")),
                    source, dtype="int64", machine_rotate_sign=True,
                )
                sign = self._bool(
                    Handler.Ne, (msb, zero), source,
                    machine_rotate_sign=True,
                )
                if semantic is MachineSemanticToken.ROTATE_LEFT:
                    overflow_right = rotated_cf
                else:
                    next_msb = self.emit(
                        Handler.And,
                        (rotated, self.constant(1 << (width - 2), source, dtype="int64")),
                        source, dtype="int64", machine_rotate_next_sign=True,
                    )
                    overflow_right = self._bool(
                        Handler.Ne, (next_msb, zero), source,
                        machine_rotate_next_sign=True,
                    )
                rotated_of = self._bool(
                    Handler.Ne, (sign, overflow_right), source,
                    machine_flag="OF", machine_rotate=direction,
                )
                self.flags["OF"] = self._bool(
                    Handler.Select,
                    (count_is_one, rotated_of, prior_flags["OF"]),
                    source, machine_flag="OF", machine_rotate=direction,
                )
                self.write_operand(operands[0], result, source, width=width)
                return
            count = (
                int(operands[1].value)
                & (0x3F if width == 64 else 0x1F)
            ) % width
            # A zero effective count changes neither destination nor flags.
            if count == 0:
                return
            value = self.read_operand(operands[0], source, width=width)
            left_count, right_count = (
                (count, width - count)
                if semantic is MachineSemanticToken.ROTATE_LEFT
                else (width - count, count)
            )
            direction = (
                "left" if semantic is MachineSemanticToken.ROTATE_LEFT
                else "right"
            )
            left = self.emit(
                Handler.Shl,
                (value, self.constant(left_count, source, dtype="int64")),
                source, dtype=f"int{width}", machine_rotate=direction,
            )
            right = self.emit(
                Handler.Shr,
                (value, self.constant(right_count, source, dtype="int64")),
                source, dtype=f"int{width}", machine_rotate=direction,
            )
            combined = self.emit(
                Handler.Or, (left, right), source, dtype=f"int{width}",
                machine_rotate=direction,
            )
            result = self.truncate_bits(combined, width, source)
            carry_position = (
                0
                if semantic is MachineSemanticToken.ROTATE_LEFT
                else width - 1
            )
            carry_source = self.emit(
                Handler.Shr,
                (
                    result,
                    self.constant(carry_position, source, dtype="int64"),
                ),
                source, dtype="int64", machine_rotate_carry=True,
            )
            carry_bit = self.emit(
                Handler.And,
                (carry_source, self.constant(1, source, dtype="int64")),
                source, dtype="int64", machine_rotate_carry=True,
            )
            zero = self.constant(0, source, dtype="int64")
            self.flags["CF"] = self._bool(
                Handler.Ne, (carry_bit, zero), source,
                machine_flag="CF", machine_rotate=direction,
            )
            if count == 1:
                most_significant = self.emit(
                    Handler.And,
                    (
                        result,
                        self.constant(1 << (width - 1), source, dtype="int64"),
                    ),
                    source, dtype="int64", machine_rotate_sign=True,
                )
                sign = self._bool(
                    Handler.Ne, (most_significant, zero), source,
                    machine_rotate_sign=True,
                )
                if semantic is MachineSemanticToken.ROTATE_LEFT:
                    overflow_right = self.flags["CF"]
                else:
                    next_significant = self.emit(
                        Handler.And,
                        (
                            result,
                            self.constant(
                                1 << (width - 2), source, dtype="int64",
                            ),
                        ),
                        source, dtype="int64", machine_rotate_next_sign=True,
                    )
                    overflow_right = self._bool(
                        Handler.Ne, (next_significant, zero), source,
                        machine_rotate_next_sign=True,
                    )
                self.flags["OF"] = self._bool(
                    Handler.Ne, (sign, overflow_right), source,
                    machine_flag="OF", machine_rotate=direction,
                )
            # For counts greater than one OF is architecturally undefined;
            # retain its incoming deterministic VM value. PF/AF/ZF/SF are
            # never modified by ROL and therefore remain in the state map.
            self.write_operand(operands[0], result, source, width=width)
            return
        if semantic in {
            MachineSemanticToken.SHIFT_RIGHT_LOGICAL,
            MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC,
            MachineSemanticToken.SHIFT_LEFT,
        }:
            if len(operands) != 2 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(
                operands[1], (ImmediateOperand, RegisterOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed shift operands"
                )
            width = self.operand_width(source, 0)
            if isinstance(operands[1], RegisterOperand):
                prior_flags = {
                    flag: self.flag(flag) for flag in self._FLAG_NAMES
                }
                value = self.read_operand(operands[0], source, width=width)
                raw_count = self.read_operand(operands[1], source, width=8)
                count = self.emit(
                    Handler.And,
                    (
                        raw_count,
                        self.constant(
                            0x3F if width == 64 else 0x1F,
                            source, dtype="int64",
                        ),
                    ),
                    source, dtype="int64", machine_masked_shift_count=True,
                )
                zero = self.constant(0, source, dtype="int64")
                one = self.constant(1, source, dtype="int64")
                active = self._bool(
                    Handler.Ne, (count, zero), source,
                    machine_shift_count_nonzero=True,
                )
                safe_count = self.emit(
                    Handler.Select, (active, count, one), source,
                    dtype="int64", machine_shift_safe_count=True,
                )
                handler = (
                    Handler.Shl if semantic is MachineSemanticToken.SHIFT_LEFT
                    else (
                        Handler.Shr
                        if semantic is MachineSemanticToken.SHIFT_RIGHT_LOGICAL
                        else Handler.AShr
                    )
                )
                raw = self.emit(
                    handler, (value, safe_count), source, dtype=f"int{width}",
                )
                candidate = self.truncate_bits(raw, width, source)
                self.logical_flags(candidate, width, source)
                if semantic is MachineSemanticToken.SHIFT_LEFT:
                    carry_position = self.emit(
                        Handler.Sub,
                        (
                            self.constant(width, source, dtype="int64"),
                            safe_count,
                        ),
                        source, dtype="int64",
                    )
                else:
                    carry_position = self.emit(
                        Handler.Sub, (safe_count, one), source, dtype="int64",
                    )
                carry_source = self.emit(
                    Handler.Shr, (value, carry_position), source, dtype="int64",
                )
                carry_bit = self.emit(
                    Handler.And, (carry_source, one), source, dtype="int64",
                )
                candidate_cf = self._bool(
                    Handler.Ne, (carry_bit, zero), source, machine_flag="CF",
                )
                count_is_one = self._bool(
                    Handler.Eq, (count, one), source,
                    machine_shift_count_one=True,
                )
                if semantic is MachineSemanticToken.SHIFT_LEFT:
                    sign_bits = self.emit(
                        Handler.And,
                        (
                            candidate,
                            self.constant(
                                1 << (width - 1), source, dtype="int64",
                            ),
                        ),
                        source, dtype="int64",
                    )
                    result_sign = self._bool(
                        Handler.Ne, (sign_bits, zero), source,
                    )
                    one_count_of = self._bool(
                        Handler.Ne, (result_sign, candidate_cf), source,
                        machine_flag="OF",
                    )
                elif semantic is MachineSemanticToken.SHIFT_RIGHT_LOGICAL:
                    sign_bits = self.emit(
                        Handler.And,
                        (
                            value,
                            self.constant(
                                1 << (width - 1), source, dtype="int64",
                            ),
                        ),
                        source, dtype="int64",
                    )
                    one_count_of = self._bool(
                        Handler.Ne, (sign_bits, zero), source,
                        machine_flag="OF",
                    )
                else:
                    one_count_of = self.bool_constant(False, source)
                candidate_of = self.emit(
                    Handler.Select,
                    (count_is_one, one_count_of, prior_flags["OF"]),
                    source, dtype="bool", machine_shift_undefined_of_preserved=True,
                )
                proposed_flags = dict(self.flags)
                proposed_flags["CF"] = candidate_cf
                proposed_flags["OF"] = candidate_of
                self.flags = {
                    flag: self.emit(
                        Handler.Select,
                        (active, proposed_flags[flag], prior_flags[flag]),
                        source, dtype="bool",
                        machine_zero_shift_preserves_flag=flag,
                    )
                    for flag in self._FLAG_NAMES
                }
                selected = self.emit(
                    Handler.Select, (active, candidate, value), source,
                    dtype=f"int{width}",
                    machine_zero_shift_preserves_destination=True,
                )
                self.write_operand(operands[0], selected, source, width=width)
                return
            count = int(operands[1].value) & (0x3F if width == 64 else 0x1F)
            if count == 0:
                return
            value = self.read_operand(operands[0], source, width=width)
            amount = self.constant(count, source, dtype="int64")
            handler = (
                Handler.Shl if semantic is MachineSemanticToken.SHIFT_LEFT
                else (
                    Handler.Shr
                    if semantic is MachineSemanticToken.SHIFT_RIGHT_LOGICAL
                    else Handler.AShr
                )
            )
            raw = self.emit(handler, (value, amount), source, dtype=f"int{width}")
            result = self.truncate_bits(raw, width, source)
            self.logical_flags(result, width, source)
            carry_shift = (
                width - count
                if semantic is MachineSemanticToken.SHIFT_LEFT
                else count - 1
            )
            carry_source = self.emit(
                Handler.Shr,
                (value, self.constant(carry_shift, source, dtype="int64")),
                source, dtype="int64",
            )
            carry_bit = self.emit(
                Handler.And,
                (carry_source, self.constant(1, source, dtype="int64")),
                source, dtype="int64",
            )
            self.flags["CF"] = self._bool(
                Handler.Ne,
                (carry_bit, self.constant(0, source, dtype="int64")),
                source, machine_flag="CF",
            )
            if count == 1:
                if semantic is MachineSemanticToken.SHIFT_LEFT:
                    result_sign = self.emit(
                        Handler.And,
                        (result, self.constant(1 << (width - 1), source, dtype="int64")),
                        source, dtype="int64",
                    )
                    result_sign = self._bool(
                        Handler.Ne,
                        (result_sign, self.constant(0, source, dtype="int64")),
                        source,
                    )
                    self.flags["OF"] = self._bool(
                        Handler.Ne, (result_sign, self.flags["CF"]), source,
                        machine_flag="OF",
                    )
                elif semantic is MachineSemanticToken.SHIFT_RIGHT_LOGICAL:
                    sign = self.emit(
                        Handler.And,
                        (value, self.constant(1 << (width - 1), source, dtype="int64")),
                        source, dtype="int64",
                    )
                    self.flags["OF"] = self._bool(
                        Handler.Ne,
                        (sign, self.constant(0, source, dtype="int64")),
                        source, machine_flag="OF",
                    )
                else:
                    self.flags["OF"] = self.bool_constant(False, source)
            self.write_operand(operands[0], result, source, width=width)
            return
        if semantic is MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP:
            if len(operands) != 1 or not isinstance(operands[0], RelativeAddressOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed conditional jump")
            target = operands[0]
            self.instructions.append(Instr(
                Handler.CondBr.value, [self.condition(source)], None,
                attributes={
                    **self.provenance(source),
                    "true_target_address": target.target_address,
                    "false_target_address": source.address + len(source.encoded),
                },
            ))
            return
        if semantic is MachineSemanticToken.DIRECT_RELATIVE_JUMP:
            if len(operands) != 1 or not isinstance(operands[0], RelativeAddressOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed direct jump")
            target = operands[0]
            self.instructions.append(Instr(
                Handler.Br.value, [], None,
                attributes={
                    **self.provenance(source),
                    "target_address": target.target_address,
                    "relative_displacement": target.displacement,
                    "machine_control_transfer": "direct-relative",
                },
            ))
            return
        if semantic is MachineSemanticToken.INDIRECT_JUMP:
            if len(operands) != 1 or not isinstance(
                operands[0], (RegisterOperand, EffectiveAddressOperand)
            ):
                raise MachineLiftError(
                    f"{source.address:#x}: malformed indirect jump"
                )
            target = self.read_operand(operands[0], source, width=64)
            registers = tuple(
                self.value(register, source) for register in X86Register
            )
            vector_registers = tuple(
                self.vector_value(register, source)
                for register in X86VectorRegister
            )
            mxcsr = self.mxcsr_value()
            flags = tuple(self.flag(flag) for flag in self._FLAG_NAMES)
            state_layout = (
                "target", "memory",
                *(f"register.{register.name.lower()}" for register in X86Register),
                *(
                    f"vector-register.{register.name.lower()}"
                    for register in X86VectorRegister
                ),
                "system.amd64.mxcsr",
                *(f"flag.{flag.lower()}" for flag in self._FLAG_NAMES),
            )
            self.instructions.append(Instr(
                Handler.IndirectBr.value,
                [
                    target, self.memory, *registers, *vector_registers,
                    mxcsr, *flags,
                ],
                None,
                attributes={
                    **self.provenance(source),
                    "target_source": (
                        "register" if isinstance(operands[0], RegisterOperand)
                        else "memory"
                    ),
                    "indirect_operand": (
                        "register" if isinstance(operands[0], RegisterOperand)
                        else "rip-relative-memory"
                        if operands[0].rip_relative else "based-memory"
                    ),
                    "indirect_register": (
                        operands[0].register.name
                        if isinstance(operands[0], RegisterOperand) else None
                    ),
                    "indirect_base_register": (
                        operands[0].base.name
                        if isinstance(operands[0], EffectiveAddressOperand)
                        and operands[0].base is not None else None
                    ),
                    "indirect_displacement": (
                        int(operands[0].displacement)
                        if isinstance(operands[0], EffectiveAddressOperand) else None
                    ),
                    "indirect_slot_address": (
                        int(source.address) + len(source.encoded)
                        + int(operands[0].displacement)
                        if isinstance(operands[0], EffectiveAddressOperand)
                        and operands[0].rip_relative else None
                    ),
                    "state_layout": state_layout,
                    "calling_convention": "windows-x64-tail-transfer",
                    "complete_machine_state": True,
                    "requires_target_linking": True,
                },
            ))
            return
        if semantic is MachineSemanticToken.RETURN:
            self.instructions.append(Instr(
                Handler.Ret.value, [self.value(X86Register.RAX, source)], None,
                attributes=self.provenance(source),
            ))
            return
        if semantic in {
            MachineSemanticToken.BREAKPOINT_TRAP,
            MachineSemanticToken.SOFTWARE_INTERRUPT,
        }:
            vector = 3
            if semantic is MachineSemanticToken.SOFTWARE_INTERRUPT:
                if len(operands) != 1 or not isinstance(
                    operands[0], ImmediateOperand
                ):
                    raise MachineLiftError(
                        f"{source.address:#x}: malformed software interrupt"
                    )
                vector = int(operands[0].value) & 0xFF
            self.instructions.append(Instr(
                Handler.Trap.value, [], None,
                attributes={
                    **self.provenance(source),
                    "trap_kind": (
                        "breakpoint" if semantic is MachineSemanticToken.BREAKPOINT_TRAP
                        else "software-interrupt"
                    ),
                    "interrupt_vector": vector,
                    "non_returning": True,
                    "may_trap": True,
                },
            ))
            return
        if token in {X86InstructionToken.LEA_R32_M, X86InstructionToken.LEA_R64_M}:
            destination, address = operands
            if not isinstance(destination, RegisterOperand) or not isinstance(address, EffectiveAddressOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed LEA operands")
            value = self.effective_address(address, source, dtype="int64")
            if token is X86InstructionToken.LEA_R32_M:
                value = self.zero_extend_32(value, source)
            self.registers[destination.register] = value
        elif token in {X86InstructionToken.MOV_R64_RM64, X86InstructionToken.MOV_R32_RM32}:
            destination, source_operand = operands
            if not isinstance(destination, RegisterOperand) or not isinstance(source_operand, (RegisterOperand, EffectiveAddressOperand)):
                raise MachineLiftError(f"{source.address:#x}: malformed MOV read operands")
            width = 32 if token is X86InstructionToken.MOV_R32_RM32 else 64
            value = self.read_operand(source_operand, source, width=width)
            if width == 32:
                value = self.zero_extend_32(value, source)
            self.copy_register(destination, value, source)
        elif token is X86InstructionToken.MOV_RM64_R64:
            destination, source_operand = operands
            if not isinstance(destination, (RegisterOperand, EffectiveAddressOperand)) or not isinstance(source_operand, RegisterOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed MOV write operands")
            self.write_operand(destination, self.value(source_operand.register, source), source)
        elif token is X86InstructionToken.MOV_R64_IMM64:
            destination, immediate = operands
            if not isinstance(destination, RegisterOperand) or not isinstance(immediate, ImmediateOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed MOV immediate operands")
            self.registers[destination.register] = self.constant(immediate.value, source, dtype="int64")
        elif token is X86InstructionToken.PUSH_R64:
            (operand,) = operands
            if not isinstance(operand, RegisterOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed PUSH operand")
            rsp = self.emit(
                Handler.Sub,
                (self.value(X86Register.RSP, source), self.constant(8, source, dtype="int64")),
                source,
                dtype="int64",
                machine_register="RSP",
            )
            self.registers[X86Register.RSP] = rsp
            self.memory = self.emit(
                Handler.Store, (self.memory, rsp, self.value(operand.register, source)),
                source, dtype="memory", machine_state="stack-push", width=64,
            )
        elif token is X86InstructionToken.POP_R64:
            (operand,) = operands
            if not isinstance(operand, RegisterOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed POP operand")
            rsp = self.value(X86Register.RSP, source)
            popped = self.emit(
                Handler.Load, (self.memory, rsp), source,
                dtype="int64", machine_state="stack-pop", width=64,
            )
            self.copy_register(operand, popped, source)
            self.registers[X86Register.RSP] = self.emit(
                Handler.Add, (rsp, self.constant(8, source, dtype="int64")), source,
                dtype="int64", machine_register="RSP",
            )
        elif token in {X86InstructionToken.SUB_R64_IMM8, X86InstructionToken.ADD_R64_IMM8}:
            destination, immediate = operands
            if not isinstance(destination, RegisterOperand) or not isinstance(immediate, ImmediateOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed immediate arithmetic operands")
            handler = Handler.Sub if token is X86InstructionToken.SUB_R64_IMM8 else Handler.Add
            self.registers[destination.register] = self.emit(
                handler,
                (
                    self.value(destination.register, source),
                    self.constant(
                        immediate.value, source, dtype="int64",
                        machine_operand_role="encoded-immediate",
                    ),
                ),
                source,
                dtype="int64",
                machine_register=destination.register.name,
                flags_effect="written-not-materialized",
            )
            self.pending_condition = None
        elif token is X86InstructionToken.AND_RM64_IMM8:
            destination, immediate = operands
            if not isinstance(destination, (RegisterOperand, EffectiveAddressOperand)) or not isinstance(immediate, ImmediateOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed AND immediate operands")
            result = self.emit(
                Handler.And,
                (self.read_operand(destination, source), self.constant(immediate.value, source, dtype="int64")),
                source,
                dtype="int64",
                flags_effect="written-not-materialized",
            )
            self.write_operand(destination, result, source)
            self.pending_condition = None
        elif token in {X86InstructionToken.XOR_RM64_R64, X86InstructionToken.XOR_R64_RM64, X86InstructionToken.AND_R64_RM64}:
            destination, right_operand = operands
            if not isinstance(destination, (RegisterOperand, EffectiveAddressOperand)) or not isinstance(right_operand, (RegisterOperand, EffectiveAddressOperand)):
                raise MachineLiftError(f"{source.address:#x}: malformed binary operands")
            handler = Handler.And if token is X86InstructionToken.AND_R64_RM64 else Handler.Xor
            result = self.emit(
                handler,
                (self.read_operand(destination, source), self.read_operand(right_operand, source)),
                source,
                dtype="int64",
                flags_effect="written-not-materialized",
            )
            self.write_operand(destination, result, source)
            self.pending_condition = None
        elif token is X86InstructionToken.SHL_R64_IMM8:
            destination, immediate = operands
            if not isinstance(destination, (RegisterOperand, EffectiveAddressOperand)) or not isinstance(immediate, ImmediateOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed SHL operands")
            count = immediate.value & 0x3F
            result = self.emit(
                Handler.Shl,
                (self.read_operand(destination, source), self.constant(count, source, dtype="int64")),
                source,
                dtype="int64",
                x86_masked_shift_count=count,
                flags_effect="written-not-materialized",
            )
            self.write_operand(destination, result, source)
            self.pending_condition = None
        elif token is X86InstructionToken.NOT_RM64:
            (destination,) = operands
            if not isinstance(destination, (RegisterOperand, EffectiveAddressOperand)):
                raise MachineLiftError(f"{source.address:#x}: malformed NOT operand")
            self.write_operand(
                destination,
                self.emit(Handler.Not, (self.read_operand(destination, source),), source, dtype="int64", flags_effect="unchanged"),
                source,
            )
        elif token is X86InstructionToken.CMP_R64_RM64:
            left, right = operands
            if not isinstance(left, RegisterOperand) or not isinstance(right, (RegisterOperand, EffectiveAddressOperand)):
                raise MachineLiftError(f"{source.address:#x}: malformed CMP operands")
            self.pending_condition = self.emit(
                Handler.Ne,
                (self.value(left.register, source), self.read_operand(right, source)),
                source,
                dtype="bool",
                machine_flags="ZF==0",
            )
        elif token in self._CONDITIONAL_TOKENS:
            (target,) = operands
            if not isinstance(target, RelativeAddressOperand) or self.pending_condition is None:
                raise MachineLiftError(f"{source.address:#x}: JNE requires a local CMP predicate")
            self.instructions.append(Instr(
                Handler.CondBr.value,
                [self.pending_condition],
                None,
                attributes={
                    **self.provenance(source),
                    "true_target_address": target.target_address,
                    "false_target_address": source.address + len(source.encoded),
                },
            ))
            self.pending_condition = None
        elif token in {X86InstructionToken.CALL_REL32, X86InstructionToken.CALL_RM64}:
            if len(operands) != 1 or not isinstance(operands[0], (RelativeAddressOperand, RegisterOperand, EffectiveAddressOperand)):
                raise MachineLiftError(f"{source.address:#x}: malformed CALL operand")
            self.lower_call(source, operands[0])
        elif token is X86InstructionToken.JMP_REL32:
            (target,) = operands
            if not isinstance(target, RelativeAddressOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed JMP operand")
            self.instructions.append(Instr(
                Handler.Br.value, [], None,
                attributes={**self.provenance(source), "target_address": target.target_address},
            ))
        elif token is X86InstructionToken.RET_NEAR:
            self.instructions.append(Instr(
                Handler.Ret.value, [self.value(X86Register.RAX, source)], None,
                attributes=self.provenance(source),
            ))
        elif token is X86InstructionToken.IMUL_R32_RM32:
            destination, right = operands
            if not isinstance(destination, RegisterOperand) or not isinstance(right, RegisterOperand):
                raise MachineLiftError(f"{source.address:#x}: malformed IMUL operands")
            product = self.emit(
                Handler.Mul,
                (self.read_operand(destination, source, width=32), self.read_operand(right, source, width=32)),
                source,
                dtype="int32",
            )
            self.registers[destination.register] = self.zero_extend_32(product, source)
        else:
            raise MachineLiftError(f"{source.address:#x}: no CFG SSA lowering for {token.name}")

    @staticmethod
    def _target(source: DecodedInstruction) -> int | None:
        if source.semantic in {
            MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
            MachineSemanticToken.DIRECT_RELATIVE_JUMP,
        }:
            target = source.operands[0]
            if isinstance(target, RelativeAddressOperand):
                return target.target_address
        return None

    def finish(self) -> Function:
        if not self.decoded:
            raise MachineLiftError("cannot build a CFG from no instructions")
        # Discover the vector-register slice before CFG state allocation so
        # loops and merges receive the same explicit XMM Phi treatment as
        # general registers. Unused XMM registers are not added to the ABI.
        for source in self.decoded:
            for operand in source.operands:
                if isinstance(operand, VectorRegisterOperand):
                    self.initial_vector_value(operand.register)
            if source.semantic is MachineSemanticToken.INDIRECT_JUMP:
                # IndirectBr exports a complete architectural continuation,
                # including state not otherwise referenced in this region.
                for register in X86VectorRegister:
                    self.initial_vector_value(register)
                self.initial_mxcsr_value()
            if source.semantic in {
                MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT64,
                MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT32,
                MachineSemanticToken.SCALAR_FLOAT64_COMPARE_UNORDERED,
                MachineSemanticToken.SCALAR_FLOAT64_COMPARE_ORDERED,
                MachineSemanticToken.SCALAR_FLOAT32_COMPARE_ORDERED,
                MachineSemanticToken.SCALAR_FLOAT64_ADD,
                MachineSemanticToken.SCALAR_FLOAT32_ADD,
                MachineSemanticToken.SCALAR_FLOAT32_DIVIDE,
                MachineSemanticToken.SCALAR_FLOAT64_DIVIDE,
                MachineSemanticToken.SCALAR_FLOAT64_SUBTRACT,
                MachineSemanticToken.SCALAR_FLOAT64_TO_SIGNED_INT64_TRUNCATE,
                MachineSemanticToken.SCALAR_FLOAT64_TO_SIGNED_INT32_TRUNCATE,
                MachineSemanticToken.VECTOR_SIGNED_INT32_TO_FLOAT64,
                MachineSemanticToken.SCALAR_FLOAT64_MULTIPLY,
            }:
                self.initial_mxcsr_value()
        by_address = {item.address: item for item in self.decoded}
        if len(by_address) != len(self.decoded):
            raise MachineLiftError("duplicate machine instruction address")
        region_end = self.decoded[-1].address + len(self.decoded[-1].encoded)
        leaders = {self.decoded[0].address}
        for source in self.decoded:
            target = self._target(source)
            if target in by_address:
                leaders.add(target)
            if source.semantic is MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP:
                fallthrough = source.address + len(source.encoded)
                if fallthrough < region_end:
                    leaders.add(fallthrough)
        ordered_leaders = sorted(leaders)
        block_sources: dict[str, list[DecodedInstruction]] = {}
        entry_address = ordered_leaders[0]
        address_to_label = {
            address: ("entry" if address == entry_address else f"block_{address:016x}")
            for address in ordered_leaders
        }
        for index, address in enumerate(ordered_leaders):
            end = ordered_leaders[index + 1] if index + 1 < len(ordered_leaders) else region_end
            items = [item for item in self.decoded if address <= item.address < end]
            if not items or items[0].address != address:
                raise MachineLiftError(f"empty machine basic block at {address:#x}")
            block_sources[address_to_label[address]] = items

        successors: dict[str, list[str]] = {}
        external_fallthroughs: dict[str, int] = {}
        conditional_destinations: dict[
            str,
            tuple[tuple[str | None, int | None], tuple[str | None, int | None]],
        ] = {}
        external_targets: set[int] = set()
        graph = nx.DiGraph()
        graph.add_nodes_from(block_sources)
        for index, address in enumerate(ordered_leaders):
            label = address_to_label[address]
            last = block_sources[label][-1]
            labels: list[str] = []
            if last.semantic is MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP:
                target = self._target(last)
                fallthrough = last.address + len(last.encoded)
                ordered_destinations = []
                for destination in (target, fallthrough):
                    if destination in address_to_label:
                        destination_label = address_to_label[destination]
                        labels.append(destination_label)
                        ordered_destinations.append((destination_label, None))
                    else:
                        external_address = None if destination is None else int(destination)
                        if external_address is not None:
                            external_targets.add(external_address)
                        ordered_destinations.append((None, external_address))
                conditional_destinations[label] = tuple(ordered_destinations)
            elif last.semantic is MachineSemanticToken.DIRECT_RELATIVE_JUMP:
                target = self._target(last)
                if target in address_to_label:
                    labels = [address_to_label[target]]
                elif target is not None:
                    external_targets.add(int(target))
            elif last.semantic in {
                MachineSemanticToken.INDIRECT_JUMP,
                MachineSemanticToken.BREAKPOINT_TRAP,
                MachineSemanticToken.SOFTWARE_INTERRUPT,
            }:
                labels = []
            elif last.token is not X86InstructionToken.RET_NEAR and index + 1 < len(ordered_leaders):
                labels = [address_to_label[ordered_leaders[index + 1]]]
            elif (
                last.address + len(last.encoded)
                in self.external_fallthrough_addresses
            ):
                destination = last.address + len(last.encoded)
                external_fallthroughs[label] = int(destination)
                external_targets.add(int(destination))
            successors[label] = labels
            graph.add_edges_from((label, item) for item in labels)
        entry_label = address_to_label[ordered_leaders[0]]
        if not nx.is_directed_acyclic_graph(graph):
            return self.finish_cyclic(
                block_sources, successors, conditional_destinations,
                external_targets, external_fallthroughs, graph, entry_label,
            )
        states: dict[
            str, tuple[
                dict[X86Register, SSAValue],
                dict[X86VectorRegister, SSAValue],
                SSAValue | None, SSAValue, dict[str, SSAValue],
            ]
        ] = {}
        blocks: dict[str, BasicBlock] = {}
        for label in nx.topological_sort(graph):
            predecessors = list(graph.predecessors(label))
            phi_instrs: list[Instr] = []
            if not predecessors:
                if label != entry_label:
                    raise MachineLiftError(f"unreachable machine block {label}")
                self.registers = dict(self.initial_registers)
                self.vector_registers = dict(self.initial_vector_registers)
                self.mxcsr = self.initial_mxcsr
                self.memory = self.initial_memory
                self.flags = dict(self.initial_flags)
            else:
                register_keys = set(self.initial_registers)
                for predecessor in predecessors:
                    register_keys.update(states[predecessor][0])
                merged: dict[X86Register, SSAValue] = {}
                for register in sorted(register_keys, key=int):
                    initial = self.initial_value(register)
                    incoming = [states[pred][0].get(register, initial) for pred in predecessors]
                    if all(value.id == incoming[0].id for value in incoming[1:]):
                        merged[register] = incoming[0]
                    else:
                        result = self.fresh(dtype=incoming[0].dtype or "int64")
                        phi_instrs.append(Instr(
                            Handler.Phi.value, incoming, result,
                            attributes={
                                "incoming_blocks": tuple(predecessors),
                                "machine_state": "register",
                                "machine_register": register.name,
                            },
                        ))
                        merged[register] = result
                incoming_mxcsr = [states[pred][2] for pred in predecessors]
                if self.initial_mxcsr is None:
                    merged_mxcsr = None
                elif all(
                    value is not None
                    and value.id == incoming_mxcsr[0].id
                    for value in incoming_mxcsr[1:]
                ):
                    merged_mxcsr = incoming_mxcsr[0]
                else:
                    assert all(value is not None for value in incoming_mxcsr)
                    merged_mxcsr = self.fresh(dtype="int32")
                    phi_instrs.append(Instr(
                        Handler.Phi.value, list(incoming_mxcsr), merged_mxcsr,
                        attributes={
                            "incoming_blocks": tuple(predecessors),
                            "machine_state": "mxcsr",
                        },
                    ))
                incoming_memory = [states[pred][3] for pred in predecessors]
                if all(value.id == incoming_memory[0].id for value in incoming_memory[1:]):
                    merged_memory = incoming_memory[0]
                else:
                    merged_memory = self.fresh(dtype="memory")
                    phi_instrs.append(Instr(
                        Handler.Phi.value, incoming_memory, merged_memory,
                        attributes={"incoming_blocks": tuple(predecessors), "machine_state": "memory"},
                    ))
                self.registers = merged
                vector_keys = set(self.initial_vector_registers)
                for predecessor in predecessors:
                    vector_keys.update(states[predecessor][1])
                merged_vectors: dict[X86VectorRegister, SSAValue] = {}
                for register in sorted(vector_keys, key=int):
                    initial = self.initial_vector_value(register)
                    incoming = [
                        states[pred][1].get(register, initial)
                        for pred in predecessors
                    ]
                    if all(value.id == incoming[0].id for value in incoming[1:]):
                        merged_vectors[register] = incoming[0]
                    else:
                        result = self.fresh(dtype=self.vector_state_dtype)
                        phi_instrs.append(Instr(
                            Handler.Phi.value, incoming, result,
                            attributes={
                                "incoming_blocks": tuple(predecessors),
                                "machine_state": "vector-register",
                                "machine_register": register.name,
                            },
                        ))
                        merged_vectors[register] = result
                self.vector_registers = merged_vectors
                self.mxcsr = merged_mxcsr
                self.memory = merged_memory
                flag_keys = set(self.initial_flags)
                for predecessor in predecessors:
                    flag_keys.update(states[predecessor][4])
                merged_flags: dict[str, SSAValue] = {}
                for flag in sorted(flag_keys):
                    initial = self.initial_flag(flag)
                    incoming = [
                        states[pred][4].get(flag, initial) for pred in predecessors
                    ]
                    if all(value.id == incoming[0].id for value in incoming[1:]):
                        merged_flags[flag] = incoming[0]
                    else:
                        result = self.fresh(dtype="bool")
                        phi_instrs.append(Instr(
                            Handler.Phi.value, incoming, result,
                            attributes={
                                "incoming_blocks": tuple(predecessors),
                                "machine_state": "flags",
                                "machine_flag": flag,
                            },
                        ))
                        merged_flags[flag] = result
                self.flags = merged_flags
            self.pending_condition = None
            self.instructions = phi_instrs
            for source in block_sources[label]:
                self.lower_one(source)
            if self.instructions and self.instructions[-1].op == Handler.CondBr.value:
                true_destination, false_destination = conditional_destinations[label]
                self.instructions[-1].attributes.update({
                    "true_target": true_destination[0],
                    "false_target": false_destination[0],
                    "true_target_address": true_destination[1],
                    "false_target_address": false_destination[1],
                    "machine_external_control": bool(
                        true_destination[1] is not None
                        or false_destination[1] is not None
                    ),
                })
            elif (
                self.instructions
                and self.instructions[-1].op == Handler.Br.value
                and successors[label]
            ):
                self.instructions[-1].attributes["target"] = successors[label][0]
            if not self.instructions or self.instructions[-1].op not in {
                Handler.Ret.value, Handler.Br.value, Handler.CondBr.value,
                Handler.IndirectBr.value, Handler.Trap.value,
            }:
                labels = successors[label]
                external_fallthrough = external_fallthroughs.get(label)
                if external_fallthrough is not None:
                    self.instructions.append(Instr(
                        Handler.Br.value, [], None,
                        attributes={
                            "target_address": int(external_fallthrough),
                            "machine_address": int(block_sources[label][-1].address),
                            "machine_control_transfer": "cross-region-fallthrough",
                        },
                    ))
                elif len(labels) != 1:
                    raise MachineLiftError(f"machine block {label} has no terminator")
                else:
                    self.instructions.append(Instr(
                    Handler.Br.value, [], None,
                    attributes={"target": labels[0], "synthetic_machine_fallthrough": True},
                    ))
            blocks[label] = BasicBlock(label, list(self.instructions), list(successors[label]))
            states[label] = (
                dict(self.registers), dict(self.vector_registers),
                self.mxcsr, self.memory, dict(self.flags),
            )

        return Function(
            self.name,
            self.args,
            blocks,
            metadata={
                "lifted_from": "x86_64-reference-vocabulary",
                "entry_block": entry_label,
                "argument_names": tuple(self.argument_names),
                "argument_registers": tuple(item.name.lower() for item in self.argument_registers),
                "machine_state_model": "register-memory-call-state-ssa",
                "machine_state_arguments": tuple(self.argument_names),
                "machine_instruction_count": len(self.decoded),
                "machine_external_control_targets": tuple(sorted(external_targets)),
                "requires_machine_address_linking": bool(external_targets),
                "requires_dynamic_target_linking": any(
                    item.semantic is MachineSemanticToken.INDIRECT_JUMP
                    for item in self.decoded
                ),
            },
        )


def _materialize_binary_region(binary_region) -> bytes | bytearray | memoryview | list[object]:
    if isinstance(binary_region, AbstractTensor):
        shape = tuple(binary_region.shape)
        if len(shape) != 1:
            raise VocabularyDecodeError(
                f"binary AbstractTensor must be rank one, received shape {shape}"
            )
        return list(binary_region.tolist())
    if isinstance(binary_region, (bytes, bytearray, memoryview)):
        return binary_region
    try:
        return list(binary_region)
    except TypeError as error:
        raise VocabularyDecodeError(
            "binary region must be bytes, a rank-one AbstractTensor, or an integer iterable"
        ) from error


def _region_values(binary_region) -> list[object]:
    if isinstance(binary_region, memoryview):
        return list(binary_region.cast("B"))
    if isinstance(binary_region, (bytes, bytearray)):
        return list(binary_region)
    return list(binary_region)


def _abstract_tensor_statistics(
    byte_values: Sequence[int],
    instructions: Sequence[DecodedInstruction],
) -> tuple[int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Use backend-neutral tensor algebra for bulk validation and counting."""

    if byte_values:
        byte_tensor = AbstractTensor.get_tensor(list(byte_values), dtype="int64")
        valid = byte_tensor.greater_equal(0) & byte_tensor.less_equal(0xFF)
        valid_byte_count = int(valid.to_dtype("int64").sum().item())
        byte_sum = int(byte_tensor.sum().item())
    else:
        valid_byte_count = 0
        byte_sum = 0

    token_values = [int(item.token) for item in instructions]
    semantic_values = [int(item.semantic) for item in instructions]

    def counts(values: list[int], domain: type[MachineSemanticToken] | type[X86InstructionToken]):
        if not values:
            return tuple((int(token), 0) for token in domain)
        tensor = AbstractTensor.get_tensor(values, dtype="int64")
        return tuple(
            (
                int(token),
                int(tensor.equal(int(token)).to_dtype("int64").sum().item()),
            )
            for token in domain
        )

    return (
        byte_sum,
        valid_byte_count,
        counts(token_values, X86InstructionToken),
        counts(semantic_values, MachineSemanticToken),
    )


def _statistics_with_tensor_fallback(
    byte_values: Sequence[object],
    instructions: Sequence[DecodedInstruction],
) -> tuple[
    int,
    int,
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    bool,
    str | None,
]:
    try:
        byte_sum, valid, instruction_counts, semantic_counts = (
            _abstract_tensor_statistics(byte_values, instructions)  # type: ignore[arg-type]
        )
        return byte_sum, valid, instruction_counts, semantic_counts, True, None
    except Exception as error:
        # Statistics must never prevent a safe decode result from being
        # returned. Keep the backend failure observable and compute the same
        # reductions exactly over the already validated host values.
        integers = [operator.index(value) for value in byte_values]
        instruction_counter = Counter(int(item.token) for item in instructions)
        semantic_counter = Counter(int(item.semantic) for item in instructions)
        return (
            sum(integers),
            sum(0 <= value <= 0xFF for value in integers),
            tuple((int(token), instruction_counter[int(token)]) for token in X86InstructionToken),
            tuple((int(token), semantic_counter[int(token)]) for token in MachineSemanticToken),
            False,
            f"{type(error).__name__}: {error}",
        )


def raise_binary_region_to_ssa(
    binary_region,
    *,
    maximum_file_size: int,
    size: int | None = None,
    base_address: int = 0,
    name: str = "lifted_binary",
    argument_registers: Sequence[X86Register | str] = ("ecx", "edx", "r8d", "r9d"),
    argument_names: Sequence[str] = ("arg0", "arg1", "arg2", "arg3"),
    decoder: X86ReferenceDecoder | None = None,
    allow_trailing_after_terminal: bool = False,
    full_vocabulary_report: bool = False,
    audit_preview_bytes: int = 16,
    cfg_decode: bool = False,
    cfg_entry_addresses: Sequence[int] = (),
) -> BinaryToSSAResult:
    """Raise one bounded binary region to SSA and report vocabulary coverage.

    This is the complete ingestion path: materialize the user region, enforce
    its maximum accepted capacity, decode handwritten ModRM/SIB forms, lower
    numeric instruction tokens, and return coverage statistics. A decode
    failure preserves only the proven instruction prefix and returns no
    executable SSA function. With ``full_vocabulary_report=True``, a second
    bytewise diagnostic pass inventories the whole region. Instructions found
    after its first gap are explicitly candidates, not proven boundaries, and
    every diagnostic gap still prevents SSA construction.
    """

    if isinstance(maximum_file_size, bool):
        raise ValueError("maximum_file_size must be a non-negative integer")
    try:
        maximum = operator.index(maximum_file_size)
    except TypeError as error:
        raise ValueError("maximum_file_size must be a non-negative integer") from error
    if maximum < 0:
        raise ValueError("maximum_file_size must be a non-negative integer")
    if isinstance(base_address, bool):
        raise ValueError("base_address must be an integer")
    try:
        origin = operator.index(base_address)
    except TypeError as error:
        raise ValueError("base_address must be an integer") from error

    try:
        materialized = _materialize_binary_region(binary_region)
        values = _region_values(materialized)
    except (TypeError, ValueError, VocabularyDecodeError) as error:
        failure = VocabularyFailure("input", 0, origin, b"", str(error))
        statistics = VocabularyStatistics(
            0, 0, 0, 0, 1, 0.0, 0, 0,
            tuple((int(token), 0) for token in X86InstructionToken),
            tuple((int(token), 0) for token in MachineSemanticToken),
            False, False, None,
        )
        return BinaryToSSAResult(None, (), (failure,), statistics)

    capacity = len(values)
    if capacity > maximum:
        failure = VocabularyFailure(
            "maximum_file_size",
            0,
            origin,
            bytes(int(value) for value in values[:8] if isinstance(value, int) and 0 <= value <= 255),
            f"binary region capacity {capacity} exceeds maximum_file_size {maximum}",
        )
        statistics = VocabularyStatistics(
            capacity, 0, 0, 0, 1, 0.0, 0, 0,
            tuple((int(token), 0) for token in X86InstructionToken),
            tuple((int(token), 0) for token in MachineSemanticToken),
            False, False, None,
        )
        return BinaryToSSAResult(None, (), (failure,), statistics)

    active_decoder = decoder or X86ReferenceDecoder()
    try:
        if cfg_decode:
            report = active_decoder.decode_cfg_report(
                materialized, size=size, base_address=origin,
                entry_addresses=cfg_entry_addresses,
            )
        else:
            report = active_decoder.decode_report(
                materialized,
                size=size,
                base_address=origin,
                stop_at_return=not full_vocabulary_report,
                allow_trailing_after_terminal=allow_trailing_after_terminal,
            )
    except VocabularyDecodeError as error:
        failure = VocabularyFailure("input", 0, origin, b"", str(error))
        report = DecodeReport((), (failure,), capacity, 0, 0, False)

    vocabulary_audit: VocabularyAuditReport | None = None
    if full_vocabulary_report:
        try:
            vocabulary_audit = active_decoder.audit_region(
                materialized,
                size=size,
                base_address=origin,
                preview_bytes=audit_preview_bytes,
            )
        except (ValueError, VocabularyDecodeError) as error:
            failures = list(report.failures)
            failures.append(VocabularyFailure(
                "diagnostic_audit", 0, origin, b"", str(error),
            ))
        else:
            failures = list(report.failures)
            if not cfg_decode:
                failures.extend(vocabulary_audit.gap_failures)
    else:
        failures = list(report.failures)
    function: Function | None = None
    if not failures:
        try:
            lifter_type = (
                _StructuredX86CFGLifter
                if full_vocabulary_report
                else _StructuredX86Lifter
            )
            lifter = lifter_type(
                name,
                report.instructions,
                argument_registers,
                argument_names,
            )
            if isinstance(lifter, _StructuredX86CFGLifter):
                lifter.external_fallthrough_addresses = tuple(
                    report.external_fallthrough_addresses
                )
            function = _stamp_machine_group_fingerprints(lifter.finish())
            if report.unreachable_spans:
                function.metadata["machine_unreachable_spans"] = tuple(
                    (origin + start, origin + end)
                    for start, end in report.unreachable_spans
                )
                function.metadata["machine_unreachable_byte_count"] = sum(
                    end - start for start, end in report.unreachable_spans
                )
        except (MachineLiftError, ValueError) as error:
            source = report.instructions[-1] if report.instructions else None
            failures.append(VocabularyFailure(
                "lowering",
                0 if source is None else source.address - origin,
                origin if source is None else source.address,
                b"" if source is None else source.encoded,
                str(error),
            ))
            # Decoding is complete, so retain the whole body in the explicit
            # machine-state SSA dialect even when ordinary SSA legalization is
            # incomplete.  The lowering failure remains visible and prevents
            # a false completeness claim.
            function = decoded_function_to_machine_ssa(
                name, report.instructions,
                external_fallthrough_address=(
                    report.external_fallthrough_addresses[0]
                    if len(report.external_fallthrough_addresses) == 1
                    else None
                ),
            )
            if report.unreachable_spans:
                function.metadata["machine_unreachable_spans"] = tuple(
                    (origin + start, origin + end)
                    for start, end in report.unreachable_spans
                )
                function.metadata["machine_unreachable_byte_count"] = sum(
                    end - start for start, end in report.unreachable_spans
                )

    accepted_values = values[:report.accepted_size]
    (
        byte_sum,
        valid_count,
        instruction_counts,
        semantic_counts,
        tensor_math_used,
        tensor_math_error,
    ) = (
        _statistics_with_tensor_fallback(accepted_values, report.instructions)
    )
    statistics = VocabularyStatistics(
        region_capacity=report.region_capacity,
        accepted_size=report.accepted_size,
        decoded_bytes=report.decoded_bytes,
        instruction_count=len(report.instructions),
        failed_vocabulary_count=len(failures),
        byte_coverage=(
            1.0 if report.accepted_size == 0 and not failures
            else (report.decoded_bytes / report.accepted_size if report.accepted_size else 0.0)
        ),
        byte_sum=byte_sum,
        valid_byte_count=valid_count,
        instruction_token_counts=instruction_counts,
        semantic_token_counts=semantic_counts,
        stopped_at_return=report.stopped_at_return,
        tensor_math_used=tensor_math_used,
        tensor_math_error=tensor_math_error,
        diagnostic_known_bytes=(0 if vocabulary_audit is None else vocabulary_audit.known_bytes),
        diagnostic_missing_bytes=(0 if vocabulary_audit is None else vocabulary_audit.missing_bytes),
        diagnostic_gap_count=(0 if vocabulary_audit is None else len(vocabulary_audit.gap_failures)),
        diagnostic_candidate_instruction_count=(
            0 if vocabulary_audit is None else len(vocabulary_audit.candidate_instructions)
        ),
    )
    return BinaryToSSAResult(
        function,
        report.instructions,
        tuple(failures),
        statistics,
        vocabulary_audit,
    )


def emit_scalar_c(function: Function, *, name: str | None = None) -> str:
    """Emit the straight-line scalar subset accepted by the prototype."""

    if set(function.blocks) != {"entry"}:
        raise MachineLiftError("scalar C emission requires one entry block")
    argument_names = tuple(function.metadata.get("argument_names", ()))
    if len(argument_names) != len(function.args):
        argument_names = tuple(f"arg{index}" for index in range(len(function.args)))
    expressions: dict[int, str] = {
        value.id: argument_names[index]
        for index, value in enumerate(function.args)
    }
    returned: str | None = None
    binary = {
        Handler.Add.value: "+",
        Handler.Sub.value: "-",
        Handler.Mul.value: "*",
    }
    for instruction in function.blocks["entry"].instrs:
        if instruction.op == Handler.Const.value:
            expressions[instruction.res.id] = str(int(instruction.attributes["value"]))
        elif instruction.op in binary:
            if instruction.res is None or len(instruction.args) != 2:
                raise MachineLiftError(f"malformed scalar {instruction.op}")
            left, right = (expressions[value.id] for value in instruction.args)
            expressions[instruction.res.id] = f"({left} {binary[instruction.op]} {right})"
        elif instruction.op == Handler.Ret.value:
            if len(instruction.args) != 1:
                raise MachineLiftError("scalar C return requires one value")
            returned = expressions[instruction.args[0].id]
        else:
            raise MachineLiftError(f"no scalar C spelling for {instruction.op}")
    if returned is None:
        raise MachineLiftError("scalar C emission requires a return")
    parameters = ", ".join(
        f"int32_t {argument}" for argument in argument_names
    )
    function_name = name or function.name
    return (
        "#include <stdint.h>\n"
        f"__declspec(dllexport) int32_t {function_name}({parameters}) {{\n"
        f"    return {returned};\n"
        "}\n"
    )


def c_function_token_multigraph(
    source: str,
    function_name: str,
    *,
    operation_tokens: Mapping[str, int],
    atlas: TokenPathAtlas | None = None,
) -> tuple[nx.MultiDiGraph, TokenPathAtlas]:
    """Parse a scalar C function into numeric token/dataflow topology.

    This controlled frontend accepts parameters, binary expressions, constants,
    and one return statement. ``pycparser`` supplies syntax; graph identity and
    expression identity are integer tokens. Preprocessor lines and MSVC-style
    ``__declspec`` are removed only so the same compilable Windows fixture can
    enter the portable parser.
    """

    from pycparser import c_ast, c_parser

    parser_source = re.sub(r"(?m)^\s*#.*$", "", str(source))
    parser_source = re.sub(r"__declspec\s*\([^)]*\)", "", parser_source)
    parser_source = re.sub(r"\b(?:u?int(?:8|16|32|64)_t)\b", "int", parser_source)
    syntax = c_parser.CParser().parse(parser_source)
    definition = next((
        item for item in syntax.ext
        if isinstance(item, c_ast.FuncDef)
        and item.decl.name == function_name
    ), None)
    if definition is None:
        raise MachineLiftError(f"C source has no function {function_name!r}")

    token_atlas = atlas or TokenPathAtlas()
    graph = nx.MultiDiGraph()
    next_node = 0
    parameters: dict[str, tuple[int, int]] = {}
    declarations = definition.decl.type.args.params if definition.decl.type.args else ()
    for position, declaration in enumerate(declarations):
        node = next_node
        next_node += 1
        token_id = int(operation_tokens["argument"])
        expression_token = token_atlas.consume((token_id, position))
        graph.add_node(
            node,
            token_id=token_id,
            diagnostic=declaration.name,
            position=position,
            expression_token=expression_token,
        )
        parameters[declaration.name] = (node, expression_token)

    binary_handlers = {
        "+": Handler.Add.value,
        "-": Handler.Sub.value,
        "*": Handler.Mul.value,
        "/": Handler.Div.value,
    }

    def lower(expression) -> tuple[int, int]:
        nonlocal next_node
        if isinstance(expression, c_ast.ID):
            try:
                return parameters[expression.name]
            except KeyError as error:
                raise MachineLiftError(
                    f"C expression reads unknown value {expression.name!r}"
                ) from error
        if isinstance(expression, c_ast.Constant):
            token_id = int(operation_tokens[Handler.Const.value])
            node = next_node
            next_node += 1
            expression_token = token_atlas.consume(
                (token_id, int(expression.value, 0)),
            )
            graph.add_node(
                node,
                token_id=token_id,
                diagnostic="Const",
                value=int(expression.value, 0),
                expression_token=expression_token,
            )
            return node, expression_token
        if isinstance(expression, c_ast.BinaryOp):
            try:
                operation = binary_handlers[expression.op]
                token_id = int(operation_tokens[operation])
            except KeyError as error:
                raise MachineLiftError(
                    f"C token vocabulary does not support {expression.op!r}"
                ) from error
            left_node, left_token = lower(expression.left)
            right_node, right_token = lower(expression.right)
            node = next_node
            next_node += 1
            expression_token = token_atlas.consume(
                (token_id, left_token, right_token),
            )
            graph.add_node(
                node,
                token_id=token_id,
                diagnostic=operation,
                expression=expression_token,
                expression_token=expression_token,
            )
            graph.add_edge(left_node, node, position=0)
            graph.add_edge(right_node, node, position=1)
            return node, expression_token
        raise MachineLiftError(
            f"unsupported C expression node {type(expression).__name__}"
        )

    statements = definition.body.block_items or ()
    returns = [item for item in statements if isinstance(item, c_ast.Return)]
    if len(returns) != 1 or len(statements) != 1:
        raise MachineLiftError("controlled C frontend requires exactly one return")
    value_node, value_token = lower(returns[0].expr)
    return_node = next_node
    return_token = int(operation_tokens[Handler.Ret.value])
    graph.add_node(
        return_node,
        token_id=return_token,
        diagnostic=Handler.Ret.value,
        expression_token=token_atlas.consume((return_token, value_token)),
    )
    graph.add_edge(value_node, return_node, position=0)
    return graph, token_atlas


def ssa_dataflow_multigraph(
    function: Function,
    *,
    operation_tokens: Mapping[str, int] | None = None,
) -> nx.MultiDiGraph:
    """Project SSA dependencies while retaining repeated operand edges.

    ``operation_tokens`` is the bridge to an atlas/token vocabulary. When it
    is supplied, every graph node receives an integer ``token_id``; operation
    strings remain diagnostic annotations and are not comparison identity.
    The caller owns token numbering because the repository's canonical and
    Nodus atlas vocabularies, rather than this lifting prototype, are the
    authorities for stable IDs.
    """

    graph = nx.MultiDiGraph()
    for index, argument in enumerate(function.args):
        attributes = {"kind": "argument", "position": index}
        if operation_tokens is not None:
            attributes["token_id"] = int(operation_tokens["argument"])
        graph.add_node(argument.id, **attributes)
    next_structural_id = -1
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.res is None:
                node = next_structural_id
                next_structural_id -= 1
            else:
                node = instruction.res.id
            attributes = {"kind": instruction.op}
            if operation_tokens is not None:
                try:
                    attributes["token_id"] = int(operation_tokens[instruction.op])
                except KeyError as error:
                    raise MachineLiftError(
                        f"operation token vocabulary has no {instruction.op!r}"
                    ) from error
            graph.add_node(node, **attributes)
            for position, argument in enumerate(instruction.args):
                graph.add_edge(argument.id, node, position=position)
    return graph


def quotient_common_subexpressions(
    graph: nx.MultiDiGraph,
    *,
    expression_attribute: str = "expression",
) -> nx.MultiDiGraph:
    """Contract nodes that declare the same common-subexpression identity.

    This is a graph rewrite witness for CSE, not an optimizer. Nodes without an
    ``expression_attribute`` remain distinct. When equivalent computations are
    merged, duplicate definition edges with the same operand position collapse,
    while outgoing parallel edges remain distinct uses of the shared result.
    """

    result = nx.MultiDiGraph()
    representative: dict[object, object] = {}
    expressions: dict[object, object] = {}
    for node, attributes in graph.nodes(data=True):
        expression = attributes.get(expression_attribute)
        if expression is None:
            representative[node] = node
            result.add_node(node, **attributes)
            continue
        canonical = expressions.setdefault(expression, node)
        representative[node] = canonical
        if canonical == node:
            result.add_node(node, **attributes)

    definition_edges: set[tuple[object, object, object]] = set()
    contracted_targets = {
        representative[node]
        for node in graph
        if representative[node] != node
    }
    for source, target, attributes in graph.edges(data=True):
        new_source = representative[source]
        new_target = representative[target]
        # Equivalent computation nodes have equivalent operand definitions.
        # Keep one edge per source/target/operand role, but retain every
        # outgoing edge because each represents a separate use.
        if new_target in contracted_targets or target != new_target:
            identity = (
                new_source,
                new_target,
                attributes.get("position"),
            )
            if identity in definition_edges:
                continue
            definition_edges.add(identity)
        result.add_edge(new_source, new_target, **attributes)
    return result


def topology_profile(graph: nx.MultiDiGraph | nx.DiGraph) -> TopologyProfile:
    """Return label-independent structural invariants for graph comparison."""

    simple = nx.DiGraph(graph)
    undirected = simple.to_undirected()
    components = nx.number_connected_components(undirected) if simple else 0
    cycle_rank = simple.number_of_edges() - simple.number_of_nodes() + components
    degrees = Counter((graph.in_degree(node), graph.out_degree(node)) for node in graph)
    return TopologyProfile(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        components=components,
        cycle_rank=max(0, cycle_rank),
        sources=sum(graph.in_degree(node) == 0 for node in graph),
        sinks=sum(graph.out_degree(node) == 0 for node in graph),
        branches=sum(graph.out_degree(node) > 1 for node in graph),
        merges=sum(graph.in_degree(node) > 1 for node in graph),
        degrees=tuple(sorted(degrees.items())),
    )


def _ratio(left: int, right: int) -> float:
    return 1.0 if left == right == 0 else min(left, right) / max(left, right)


def topology_similarity(
    left: nx.MultiDiGraph | nx.DiGraph,
    right: nx.MultiDiGraph | nx.DiGraph,
) -> float:
    """Similarity of topology, intentionally independent of node identity.

    The score is not graph isomorphism.  It averages stable global invariants
    and a multiset Jaccard score over in/out-degree pairs, allowing compiler
    optimizations to merge or duplicate individual operations.
    """

    a = topology_profile(left)
    b = topology_profile(right)
    scalar_scores = [
        _ratio(getattr(a, field), getattr(b, field))
        for field in (
            "nodes", "edges", "components", "cycle_rank", "sources",
            "sinks", "branches", "merges",
        )
    ]
    left_degrees = Counter(dict(a.degrees))
    right_degrees = Counter(dict(b.degrees))
    intersection = sum((left_degrees & right_degrees).values())
    union = sum((left_degrees | right_degrees).values())
    degree_score = 1.0 if union == 0 else intersection / union
    return sum((*scalar_scores, degree_score)) / (len(scalar_scores) + 1)


__all__ = [
    "BinaryToSSAResult",
    "MachineFunction",
    "MachineInstruction",
    "MachineLiftError",
    "TopologyProfile",
    "TokenPathAtlas",
    "VocabularyStatistics",
    "c_function_token_multigraph",
    "disassemble_gnu_object",
    "emit_scalar_c",
    "lift_x86_64_affine_function",
    "parse_objdump_function",
    "quotient_common_subexpressions",
    "raise_binary_region_to_ssa",
    "ssa_dataflow_multigraph",
    "topology_profile",
    "topology_similarity",
]
