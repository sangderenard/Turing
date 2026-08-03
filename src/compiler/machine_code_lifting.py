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
    VocabularyAuditReport,
    VocabularyDecodeError,
    VocabularyFailure,
    X86InstructionToken,
    X86ReferenceDecoder,
    X86Register,
)
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
    ) -> SSAValue:
        result = self.fresh(dtype=dtype)
        self.instructions.append(Instr(
            Handler.Const.value,
            [],
            result,
            attributes={**self.provenance(source), "value": int(value)},
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
                literal = self.constant(immediate.value, source, dtype="int64")
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
        self.initial_memory = self.memory

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
        operand: RegisterOperand | EffectiveAddressOperand,
        source: DecodedInstruction,
        *,
        width: int = 64,
    ) -> SSAValue:
        if isinstance(operand, RegisterOperand):
            value = self.value(operand.register, source)
            if width == 32:
                return self.emit(
                    Handler.Trunc, (value,), source, dtype="int32",
                    machine_width=32,
                )
            return value
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
    ) -> None:
        if isinstance(operand, RegisterOperand):
            self.copy_register(operand, value, source)
        else:
            self.store_memory(operand, value, source)

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
                (self.value(destination.register, source), self.constant(immediate.value, source, dtype="int64")),
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
        if source.token in _StructuredX86CFGLifter._CONDITIONAL_TOKENS or source.token is X86InstructionToken.JMP_REL32:
            target = source.operands[0]
            if isinstance(target, RelativeAddressOperand):
                return target.target_address
        return None

    def finish(self) -> Function:
        if not self.decoded:
            raise MachineLiftError("cannot build a CFG from no instructions")
        by_address = {item.address: item for item in self.decoded}
        if len(by_address) != len(self.decoded):
            raise MachineLiftError("duplicate machine instruction address")
        region_end = self.decoded[-1].address + len(self.decoded[-1].encoded)
        leaders = {self.decoded[0].address}
        for source in self.decoded:
            target = self._target(source)
            if target is not None and self.decoded[0].address <= target < region_end:
                if target not in by_address:
                    raise MachineLiftError(f"branch target {target:#x} is not an instruction boundary")
                leaders.add(target)
            if source.token in self._CONDITIONAL_TOKENS:
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
        graph = nx.DiGraph()
        graph.add_nodes_from(block_sources)
        for index, address in enumerate(ordered_leaders):
            label = address_to_label[address]
            last = block_sources[label][-1]
            labels: list[str] = []
            if last.token in self._CONDITIONAL_TOKENS:
                target = self._target(last)
                fallthrough = last.address + len(last.encoded)
                if target not in address_to_label or fallthrough not in address_to_label:
                    raise MachineLiftError(f"{last.address:#x}: conditional successor leaves bounded function")
                labels = [address_to_label[target], address_to_label[fallthrough]]
            elif last.token is X86InstructionToken.JMP_REL32:
                target = self._target(last)
                if target in address_to_label:
                    labels = [address_to_label[target]]
            elif last.token is not X86InstructionToken.RET_NEAR and index + 1 < len(ordered_leaders):
                labels = [address_to_label[ordered_leaders[index + 1]]]
            successors[label] = labels
            graph.add_edges_from((label, item) for item in labels)
        if not nx.is_directed_acyclic_graph(graph):
            raise MachineLiftError("looping machine CFG requires loop-header Phi construction")

        entry_label = address_to_label[ordered_leaders[0]]
        states: dict[str, tuple[dict[X86Register, SSAValue], SSAValue]] = {}
        blocks: dict[str, BasicBlock] = {}
        for label in nx.topological_sort(graph):
            predecessors = list(graph.predecessors(label))
            phi_instrs: list[Instr] = []
            if not predecessors:
                if label != entry_label:
                    raise MachineLiftError(f"unreachable machine block {label}")
                self.registers = dict(self.initial_registers)
                self.memory = self.initial_memory
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
                incoming_memory = [states[pred][1] for pred in predecessors]
                if all(value.id == incoming_memory[0].id for value in incoming_memory[1:]):
                    merged_memory = incoming_memory[0]
                else:
                    merged_memory = self.fresh(dtype="memory")
                    phi_instrs.append(Instr(
                        Handler.Phi.value, incoming_memory, merged_memory,
                        attributes={"incoming_blocks": tuple(predecessors), "machine_state": "memory"},
                    ))
                self.registers = merged
                self.memory = merged_memory
            self.pending_condition = None
            self.instructions = phi_instrs
            for source in block_sources[label]:
                self.lower_one(source)
            if self.instructions and self.instructions[-1].op == Handler.CondBr.value:
                self.instructions[-1].attributes.update({
                    "true_target": successors[label][0],
                    "false_target": successors[label][1],
                })
            elif (
                self.instructions
                and self.instructions[-1].op == Handler.Br.value
                and successors[label]
            ):
                self.instructions[-1].attributes["target"] = successors[label][0]
            if not self.instructions or self.instructions[-1].op not in {
                Handler.Ret.value, Handler.Br.value, Handler.CondBr.value,
            }:
                labels = successors[label]
                if len(labels) != 1:
                    raise MachineLiftError(f"machine block {label} has no terminator")
                self.instructions.append(Instr(
                    Handler.Br.value, [], None,
                    attributes={"target": labels[0], "synthetic_machine_fallthrough": True},
                ))
            blocks[label] = BasicBlock(label, list(self.instructions), list(successors[label]))
            states[label] = (dict(self.registers), self.memory)

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
                "machine_state_arguments": ("__machine_rsp", "__machine_memory"),
                "machine_instruction_count": len(self.decoded),
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
            function = lifter_type(
                name,
                report.instructions,
                argument_registers,
                argument_names,
            ).finish()
        except (MachineLiftError, ValueError) as error:
            source = report.instructions[-1] if report.instructions else None
            failures.append(VocabularyFailure(
                "lowering",
                0 if source is None else source.address - origin,
                origin if source is None else source.address,
                b"" if source is None else source.encoded,
                str(error),
            ))

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
