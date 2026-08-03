"""Bounded binary-container ingestion and cross-layer equivalence tables.

This module owns file/container structure. ISA instruction framing remains in
``machine_reference_vocabulary`` and SSA construction remains in
``machine_code_lifting``. Keeping those grammars separate prevents PE metadata
bytes from being mistaken for executable x86 bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
import operator
from types import MappingProxyType
from typing import Iterable

from ..transmogrifier.ssa_registry import Handler
from .machine_code_lifting import (
    BinaryToSSAResult,
    VocabularyStatistics,
    raise_binary_region_to_ssa,
)
from .machine_reference_vocabulary import (
    AuditConfidence,
    DecodeReport,
    DecodedInstruction,
    EffectiveAddressOperand,
    ImmediateOperand,
    MachineSemanticToken,
    RegisterOperand,
    RelativeAddressOperand,
    VocabularyAuditReport,
    VocabularyFailure,
    X86InstructionToken,
    X86ReferenceDecoder,
    X86Register,
)
from .x86_tensor_read_head import (
    EncodingFlag,
    PrefixAction,
    ReadFailure,
    ReadPhase,
    ReadStatus,
    X86EncodingRow,
    X86ReadBatch,
    X86ReadHeadConfig,
    X86ReadHeadState,
    X86TensorReadHead,
    controlled_x86_64_read_head_config,
)


class BinaryFormatError(ValueError):
    """A binary container violates the bounded format vocabulary."""


class BinaryLayer(IntEnum):
    FILE_CONTAINER = 0
    PE_STRUCTURE = 1
    ISA_ENCODING = 2
    MACHINE_SEMANTIC = 3
    REPOSITORY_SSA = 4


class PEVocabularyToken(IntEnum):
    DOS_SIGNATURE = 0
    PE_SIGNATURE = 1
    COFF_HEADER = 2
    OPTIONAL_HEADER_PE32 = 3
    OPTIONAL_HEADER_PE32_PLUS = 4
    SECTION_TABLE = 5
    EXECUTABLE_SECTION = 6
    ENTRY_POINT = 7


class PEMachine(IntEnum):
    I386 = 0x014C
    AMD64 = 0x8664


class PESectionFlag(IntFlag):
    CODE = 0x00000020
    INITIALIZED_DATA = 0x00000040
    UNINITIALIZED_DATA = 0x00000080
    EXECUTE = 0x20000000
    READ = 0x40000000
    WRITE = 0x80000000


@dataclass(frozen=True, slots=True)
class BinaryEquivalence:
    """One explicit correspondence between adjacent representation layers."""

    source_layer: BinaryLayer
    source_token: int
    encoded_form: str
    canonical_meaning: str
    target_layer: BinaryLayer
    target_tokens: tuple[str, ...]
    coverage: str
    constraints: str


PE_EQUIVALENCE_TABLE: tuple[BinaryEquivalence, ...] = (
    BinaryEquivalence(
        BinaryLayer.FILE_CONTAINER,
        int(PEVocabularyToken.DOS_SIGNATURE),
        "file[0:2] == 4d 5a",
        "DOS-compatible PE envelope",
        BinaryLayer.PE_STRUCTURE,
        ("e_lfanew",),
        "exact",
        "e_lfanew must address a complete PE signature and COFF header",
    ),
    BinaryEquivalence(
        BinaryLayer.PE_STRUCTURE,
        int(PEVocabularyToken.PE_SIGNATURE),
        "50 45 00 00",
        "PE header begins at e_lfanew",
        BinaryLayer.PE_STRUCTURE,
        ("COFF_HEADER", "OPTIONAL_HEADER", "SECTION_TABLE"),
        "exact",
        "all referenced ranges remain inside the accepted file region",
    ),
    BinaryEquivalence(
        BinaryLayer.PE_STRUCTURE,
        int(PEVocabularyToken.COFF_HEADER),
        "20 bytes after PE signature",
        "architecture, section count, and optional-header extent",
        BinaryLayer.PE_STRUCTURE,
        ("PEMachine", "section_count", "optional_header_size"),
        "exact",
        "machine and section-count values must be vocabulary-backed and bounded",
    ),
    BinaryEquivalence(
        BinaryLayer.PE_STRUCTURE,
        int(PEVocabularyToken.OPTIONAL_HEADER_PE32),
        "optional_magic == 0x10b",
        "32-bit PE image metadata",
        BinaryLayer.ISA_ENCODING,
        ("I386", "entry_point_RVA"),
        "container-only",
        "parsed and mapped; current instruction vocabulary does not lift I386 mode",
    ),
    BinaryEquivalence(
        BinaryLayer.PE_STRUCTURE,
        int(PEVocabularyToken.OPTIONAL_HEADER_PE32_PLUS),
        "optional_magic == 0x20b",
        "64-bit PE image metadata",
        BinaryLayer.ISA_ENCODING,
        ("AMD64", "entry_point_RVA"),
        "conditional",
        "COFF machine must be AMD64 before invoking the x86-64 vocabulary",
    ),
    BinaryEquivalence(
        BinaryLayer.PE_STRUCTURE,
        int(PEVocabularyToken.SECTION_TABLE),
        "N contiguous 40-byte section records",
        "RVA and file-backed region mapping",
        BinaryLayer.PE_STRUCTURE,
        ("PESection",),
        "exact",
        "section count is capped and nonempty raw ranges may not overlap",
    ),
    BinaryEquivalence(
        BinaryLayer.PE_STRUCTURE,
        int(PEVocabularyToken.EXECUTABLE_SECTION),
        "section.characteristics & 0x20000000",
        "file-backed executable byte region",
        BinaryLayer.ISA_ENCODING,
        ("bounded_machine_code_region",),
        "exact",
        "entry-point delta must be smaller than SizeOfRawData",
    ),
    BinaryEquivalence(
        BinaryLayer.PE_STRUCTURE,
        int(PEVocabularyToken.ENTRY_POINT),
        "AddressOfEntryPoint RVA",
        "initial executable address",
        BinaryLayer.ISA_ENCODING,
        ("entry_file_offset",),
        "exact",
        "RVA must map uniquely into an executable section's raw bytes",
    ),
)


X86_SSA_EQUIVALENCE_TABLE: tuple[BinaryEquivalence, ...] = (
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.IMUL_R32_RM32),
        "0f af /r",
        MachineSemanticToken.INTEGER_MULTIPLY.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Mul.value,),
        "register-source subset",
        "REX.W and memory-source semantics are rejected",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.LEA_R32_M),
        "8d /r",
        MachineSemanticToken.EFFECTIVE_ADDRESS.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Const.value, Handler.Mul.value, Handler.Add.value),
        "ModRM/SIB address algebra",
        "32-bit destination with 64-bit address calculation",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.RET_NEAR),
        "c3",
        MachineSemanticToken.RETURN.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Ret.value,),
        "exact for the controlled straight-line ABI",
        "return value is read from the bound RAX register state",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.SUB_R64_IMM8),
        "REX.W 83 /5 ib",
        MachineSemanticToken.INTEGER_SUBTRACT.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Sub.value,),
        "decoded; SSA lowering deferred",
        "requires explicit 64-bit register/flags state; memory destination rejected",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.CALL_REL32),
        "e8 cd",
        MachineSemanticToken.DIRECT_RELATIVE_CALL.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Call.value,),
        "decoded; SSA equivalence is not yet sufficient for lowering",
        "must model return-address push, target resolution, ABI arguments, and clobbers",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.ADD_R64_IMM8),
        "REX.W 83 /0 ib",
        MachineSemanticToken.INTEGER_ADD.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Add.value,),
        "decoded; SSA lowering deferred",
        "requires explicit 64-bit register/flags state; memory destination rejected",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.JMP_REL32),
        "e9 cd",
        MachineSemanticToken.DIRECT_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Br.value,),
        "decoded; control-flow lowering deferred",
        "resolved target must become a CFG edge or an explicit tail-call boundary",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.MOV_RM64_R64),
        "REX.W 89 /r",
        MachineSemanticToken.REGISTER_OR_MEMORY_WRITE.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Store.value,),
        "decoded for register and ModRM/SIB memory destinations",
        "memory destinations require versioned memory SSA before lowering",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.PUSH_R64),
        "50+rd with optional REX.B",
        MachineSemanticToken.STACK_PUSH.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Store.value, Handler.Sub.value),
        "decoded register-in-opcode family",
        "lowering requires versioned RSP and stack-memory state",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.MOV_R64_RM64),
        "REX.W 8b /r",
        MachineSemanticToken.REGISTER_OR_MEMORY_READ.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Load.value,),
        "decoded for register and ModRM/SIB memory sources",
        "register copies lower directly; memory sources require versioned memory SSA",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.AND_RM64_IMM8),
        "REX.W 83 /4 ib",
        MachineSemanticToken.BITWISE_AND.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.And.value,),
        "decoded for register or ModRM/SIB memory destinations",
        "memory read-modify-write requires versioned memory SSA and flags state",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.MOV_R64_IMM64),
        "REX.W B8+rd io",
        MachineSemanticToken.REGISTER_WRITE_IMMEDIATE.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Const.value,),
        "decoded register-in-opcode imm64 family",
        "unsigned imm64 is retained as a bit pattern",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.CMP_R64_RM64),
        "REX.W 3b /r",
        MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC,
        ("integer_subtract_flags",),
        "decoded for register and ModRM/SIB sources",
        "comparison becomes a predicate only when paired with a flag-consuming operation",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.JNE_REL32),
        "0f 85 cd",
        MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.CondBr.value,),
        "decoded as a two-successor basic-block boundary",
        "requires the preceding flags producer or an explicit incoming flags value",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.LEA_R64_M),
        "REX.W 8d /r",
        MachineSemanticToken.EFFECTIVE_ADDRESS.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Const.value, Handler.Mul.value, Handler.Add.value),
        "64-bit ModRM/SIB address algebra",
        "does not dereference the computed address",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.CALL_RM64),
        "ff /2",
        MachineSemanticToken.INDIRECT_CALL.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Load.value, Handler.Call.value),
        "callee value comes from a register or ModRM/SIB memory source",
        "Windows x64 volatile state and memory effects remain explicit call metadata",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.MOV_R32_RM32),
        "8b /r",
        MachineSemanticToken.REGISTER_OR_MEMORY_READ.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Load.value, Handler.Cast.value),
        "32-bit load/copy followed by architectural zero-extension",
        "memory access width is 32 bits",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.XOR_RM64_R64),
        "REX.W 31 /r",
        MachineSemanticToken.BITWISE_XOR.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Load.value, Handler.Xor.value, Handler.Store.value),
        "64-bit register or memory read-modify-write",
        "writes arithmetic flags; current cmd path consumes only later CMP flags",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.XOR_R64_RM64),
        "REX.W 33 /r",
        MachineSemanticToken.BITWISE_XOR.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Load.value, Handler.Xor.value),
        "64-bit register destination",
        "writes arithmetic flags; current cmd path consumes only later CMP flags",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.SHL_R64_IMM8),
        "REX.W c1 /4 ib",
        MachineSemanticToken.SHIFT_LEFT.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Shl.value,),
        "64-bit immediate-count left shift",
        "x86 masks the immediate count to six bits",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.AND_R64_RM64),
        "REX.W 23 /r",
        MachineSemanticToken.BITWISE_AND.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Load.value, Handler.And.value),
        "64-bit register destination",
        "writes arithmetic flags; current cmd path consumes only later CMP flags",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.JNE_REL8),
        "75 cb",
        MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.CondBr.value,),
        "decoded as a two-successor basic-block boundary",
        "requires the preceding flags producer or an explicit incoming flags value",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.NOT_RM64),
        "REX.W f7 /2",
        MachineSemanticToken.BITWISE_NOT.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Not.value,),
        "64-bit bitwise complement",
        "does not modify x86 flags",
    ),
    BinaryEquivalence(
        BinaryLayer.ISA_ENCODING,
        int(X86InstructionToken.POP_R64),
        "58+rd with optional REX.B",
        MachineSemanticToken.STACK_POP.name,
        BinaryLayer.REPOSITORY_SSA,
        (Handler.Load.value, Handler.Add.value),
        "64-bit stack load followed by RSP increment",
        "memory remains versioned but is not modified by POP",
    ),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.NOP_RM),
        "0f 1f /0", MachineSemanticToken.NO_OPERATION.name,
        BinaryLayer.MACHINE_SEMANTIC, ("machine_noop",), "padding form", "no state change"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.INT3),
        "cc", MachineSemanticToken.BREAKPOINT_TRAP.name,
        BinaryLayer.MACHINE_SEMANTIC, ("breakpoint_trap",), "exact one-byte trap", "terminates normal execution"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.NOP),
        "90", MachineSemanticToken.NO_OPERATION.name,
        BinaryLayer.MACHINE_SEMANTIC, ("machine_noop",), "exact one-byte padding", "no state change"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_R64_IMM32),
        "REX.W 81 /5 id", MachineSemanticToken.INTEGER_SUBTRACT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "register destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XOR_R32_RM32),
        "33 /r", MachineSemanticToken.BITWISE_XOR.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Xor.value, Handler.ZExt.value), "32-bit destination", "zero-extends register result"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.TEST_RM32_R32),
        "85 /r", MachineSemanticToken.INTEGER_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_test_flags",), "32-bit non-writing AND", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.TEST_RM64_R64),
        "REX.W 85 /r", MachineSemanticToken.INTEGER_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_test_flags",), "64-bit non-writing AND", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JE_REL8),
        "74 cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short ZF branch", "requires explicit flags producer"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JE_REL32),
        "0f 84 cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near ZF branch", "requires explicit flags producer"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JS_REL32),
        "0f 88 cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near SF branch", "requires explicit flags producer"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_R32_IMM32),
        "B8+rd id", MachineSemanticToken.REGISTER_WRITE_IMMEDIATE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Const.value, Handler.ZExt.value), "32-bit immediate register write", "zero-extends destination"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.NEG_RM8),
        "f6 /3", MachineSemanticToken.INTEGER_NEGATE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Neg.value,), "byte register or memory", "high-byte aliases remain distinct vocabulary"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SBB_R64_RM64),
        "REX.W 1b /r", MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "64-bit destination plus carry input", "requires explicit CF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVNE_R64_RM64),
        "REX.W 0f 45 /r", MachineSemanticToken.CONDITIONAL_MOVE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit conditional move", "requires explicit ZF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_RM32_R32),
        "89 /r", MachineSemanticToken.REGISTER_OR_MEMORY_WRITE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Store.value, Handler.Trunc.value), "32-bit register or memory write", "memory access width is 32 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.INC_RM64),
        "REX.W ff /0", MachineSemanticToken.INTEGER_INCREMENT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "64-bit register or memory increment", "preserves CF while writing other arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_RM64_IMM8),
        "REX.W 83 /1 ib", MachineSemanticToken.BITWISE_OR.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "64-bit register or memory destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_RM32_IMM32),
        "c7 /0 id", MachineSemanticToken.REGISTER_OR_MEMORY_WRITE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Const.value, Handler.Store.value), "32-bit immediate destination", "memory width is 32 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_RM8_IMM8),
        "c6 /0 ib", MachineSemanticToken.REGISTER_OR_MEMORY_WRITE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Const.value, Handler.Store.value), "8-bit immediate destination", "memory width is 8 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_R64_RM64),
        "REX.W 2b /r", MachineSemanticToken.INTEGER_SUBTRACT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "64-bit register destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_RM8_R8),
        "88 /r", MachineSemanticToken.REGISTER_OR_MEMORY_WRITE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Store.value, Handler.Trunc.value), "8-bit register or memory write", "legacy high-byte registers are separate forms"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RM64_IMM8),
        "REX.W 83 /7 ib", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "64-bit immediate comparison", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RM32_IMM8),
        "83 /7 ib", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "32-bit immediate comparison", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RAX_IMM32),
        "REX.W 3d id", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "sign-extended accumulator comparison", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_EAX_IMM32),
        "3d id", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "32-bit accumulator comparison", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.TEST_RM8_IMM8),
        "f6 /0 ib", MachineSemanticToken.INTEGER_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_test_flags",), "8-bit immediate non-writing AND", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.TEST_RM8_R8),
        "84 /r", MachineSemanticToken.INTEGER_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_test_flags",), "8-bit non-writing AND", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVSXD_R64_RM32),
        "REX.W 63 /r", MachineSemanticToken.SIGN_EXTEND.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.SExt.value,), "32-to-64-bit sign extension", "memory source width is 32 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_R8_RM8),
        "8a /r", MachineSemanticToken.REGISTER_OR_MEMORY_READ.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Load.value,), "8-bit register destination", "legacy high-byte registers are separate forms"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVNE_R32_RM32),
        "0f 45 /r", MachineSemanticToken.CONDITIONAL_MOVE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Select.value, Handler.ZExt.value), "32-bit conditional move", "requires explicit ZF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_RM64_IMM32),
        "REX.W c7 /0 id", MachineSemanticToken.REGISTER_OR_MEMORY_WRITE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Const.value, Handler.Store.value), "sign-extended immediate destination", "memory width is 64 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVZX_R32_RM16),
        "0f b7 /r", MachineSemanticToken.ZERO_EXTEND.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Load.value, Handler.ZExt.value), "16-to-32-bit zero extension", "register write zero-extends to 64 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JMP_REL8),
        "eb cb", MachineSemanticToken.DIRECT_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Br.value,), "short direct branch", "resolved target must be an instruction boundary"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_RM32_IMM8),
        "83 /4 ib", MachineSemanticToken.BITWISE_AND.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "32-bit immediate destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_R64_IMM32),
        "REX.W 81 /0 id", MachineSemanticToken.INTEGER_ADD.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "64-bit immediate destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_RM32_IMM32),
        "81 /4 id", MachineSemanticToken.BITWISE_AND.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "32-bit immediate destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.TEST_RM16_R16),
        "66 85 /r", MachineSemanticToken.INTEGER_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_test_flags",), "16-bit non-writing AND", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RM16_R16),
        "66 39 /r", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "16-bit register or memory comparison", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RM16_IMM8),
        "66 83 /7 ib", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "16-bit sign-extended immediate comparison", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_RM16_R16),
        "66 89 /r", MachineSemanticToken.REGISTER_OR_MEMORY_WRITE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Store.value, Handler.Trunc.value), "16-bit register or memory write", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JS_REL8),
        "78 cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short SF branch", "requires explicit flags producer"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RM8_R8),
        "38 /r", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "8-bit register or memory comparison", "legacy high-byte aliases remain distinct vocabulary"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_RM32_IMM8),
        "83 /5 ib", MachineSemanticToken.INTEGER_SUBTRACT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "32-bit sign-extended immediate destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.INC_RM32),
        "ff /0", MachineSemanticToken.INTEGER_INCREMENT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "32-bit register or memory increment", "preserves CF while writing other arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BTR_RM32_IMM8),
        "0f ba /6 ib", MachineSemanticToken.BIT_TEST_RESET.name,
        BinaryLayer.MACHINE_SEMANTIC, ("bit_test_reset",), "32-bit bit selection and reset", "writes selected prior bit to CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RM32_R32),
        "39 /r", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "32-bit register or memory comparison", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_R64_RM64),
        "REX.W 03 /r", MachineSemanticToken.INTEGER_ADD.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "64-bit register destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.TEST_EAX_IMM32),
        "a9 id", MachineSemanticToken.INTEGER_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_test_flags",), "32-bit accumulator non-writing AND", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_RM32_R32),
        "09 /r", MachineSemanticToken.BITWISE_OR.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "32-bit register or memory destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JAE_REL32),
        "0f 83 cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near unsigned-above-or-equal branch", "requires explicit CF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVSX_R32_RM16),
        "0f bf /r", MachineSemanticToken.SIGN_EXTEND.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Load.value, Handler.SExt.value), "16-to-32-bit sign extension", "memory source width is 16 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_R32_RM32),
        "3b /r", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "32-bit register destination comparison", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.NEG_RM32),
        "f7 /3", MachineSemanticToken.INTEGER_NEGATE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Neg.value,), "32-bit register or memory negation", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JA_REL8),
        "77 cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short unsigned-above branch", "requires explicit CF and ZF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JBE_REL32),
        "0f 86 cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near unsigned-below-or-equal branch", "requires explicit CF and ZF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JAE_REL8),
        "73 cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short unsigned-above-or-equal branch", "requires explicit CF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVZX_R32_RM8),
        "0f b6 /r", MachineSemanticToken.ZERO_EXTEND.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Load.value, Handler.ZExt.value), "8-to-32-bit zero extension", "legacy high-byte sources remain distinct vocabulary"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_RM32_IMM8),
        "83 /1 ib", MachineSemanticToken.BITWISE_OR.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "32-bit sign-extended immediate destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETE_RM8),
        "0f 94 /0", MachineSemanticToken.CONDITIONAL_SET.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Select.value, Handler.Trunc.value), "write one byte from ZF", "legacy high-byte destinations remain distinct vocabulary"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_R16_RM16),
        "66 3b /r", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "16-bit register-first comparison", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JA_REL32),
        "0f 87 cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near unsigned-above branch", "requires explicit CF and ZF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SBB_R32_RM32),
        "1b /r", MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "32-bit destination plus carry input", "requires explicit CF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RM64_R64),
        "REX.W 39 /r", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "64-bit register or memory comparison", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETB_RM8),
        "0f 92 /0", MachineSemanticToken.CONDITIONAL_SET.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Select.value, Handler.Trunc.value), "write one byte from CF", "legacy high-byte destinations remain distinct vocabulary"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_R32_RM32),
        "2b /r", MachineSemanticToken.INTEGER_SUBTRACT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "32-bit register destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.TEST_AL_IMM8),
        "a8 ib", MachineSemanticToken.INTEGER_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_test_flags",), "8-bit accumulator non-writing AND", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RM32_IMM32),
        "81 /7 id", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "32-bit immediate comparison", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JBE_REL8),
        "76 cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short unsigned-below-or-equal branch", "requires explicit CF and ZF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHR_RM32_1),
        "d1 /5", MachineSemanticToken.SHIFT_RIGHT_LOGICAL.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Shr.value,), "32-bit logical shift by one", "writes shift flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.TEST_RM32_IMM32),
        "f7 /0 id", MachineSemanticToken.INTEGER_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_test_flags",), "32-bit immediate non-writing AND", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SAR_RM64_1),
        "REX.W d1 /7", MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Shr.value,), "64-bit arithmetic shift by one", "requires signed right-shift semantics and writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_R32_RM32),
        "23 /r", MachineSemanticToken.BITWISE_AND.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "32-bit register destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_RM64_R64),
        "REX.W 01 /r", MachineSemanticToken.INTEGER_ADD.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "64-bit register or memory destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_RM32_R32),
        "01 /r", MachineSemanticToken.INTEGER_ADD.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "32-bit register or memory destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.DIV_RM32),
        "f7 /6", MachineSemanticToken.INTEGER_DIVIDE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Div.value,), "unsigned EDX:EAX division by 32-bit source", "produces quotient and remainder and may trap"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BT_RM32_IMM8),
        "0f ba /4 ib", MachineSemanticToken.BIT_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("bit_test",), "32-bit selected-bit read", "writes selected bit to CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHR_RM32_IMM8),
        "c1 /5 ib", MachineSemanticToken.SHIFT_RIGHT_LOGICAL.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Shr.value,), "32-bit logical immediate shift", "effective count is architecturally masked and flags are written"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JGE_REL8),
        "7d cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short signed-greater-or-equal branch", "requires explicit SF and OF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.DEC_RM32),
        "ff /1", MachineSemanticToken.INTEGER_DECREMENT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "32-bit register or memory decrement", "preserves CF while writing other arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JMP_RM64),
        "ff /4", MachineSemanticToken.INDIRECT_JUMP.name,
        BinaryLayer.MACHINE_SEMANTIC, ("indirect_control_target",), "64-bit register or memory target", "target resolution must preserve an open control edge"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVE_R64_RM64),
        "REX.W 0f 44 /r", MachineSemanticToken.CONDITIONAL_MOVE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit conditional move", "requires explicit ZF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JB_REL32),
        "0f 82 cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near unsigned-below branch", "requires explicit CF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_RM32_IMM8),
        "83 /0 ib", MachineSemanticToken.INTEGER_ADD.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "32-bit sign-extended immediate destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_R16_RM16),
        "66 2b /r", MachineSemanticToken.INTEGER_SUBTRACT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "16-bit register destination", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.NEG_RM16),
        "66 f7 /3", MachineSemanticToken.INTEGER_NEGATE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Neg.value,), "16-bit register or memory negation", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JG_REL32),
        "0f 8f cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near signed-greater branch", "requires explicit ZF, SF, and OF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RM8_IMM8),
        "80 /7 ib", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "8-bit immediate comparison", "legacy high-byte destinations remain distinct vocabulary"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_RM16_IMM8),
        "66 83 /5 ib", MachineSemanticToken.INTEGER_SUBTRACT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "16-bit sign-extended immediate destination", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_R32_RM32),
        "03 /r", MachineSemanticToken.INTEGER_ADD.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "32-bit register destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JB_REL8),
        "72 cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short unsigned-below branch", "requires explicit CF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHR_RM64_1),
        "REX.W d1 /5", MachineSemanticToken.SHIFT_RIGHT_LOGICAL.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Shr.value,), "64-bit logical shift by one", "writes shift flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JGE_REL32),
        "0f 8d cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near signed-greater-or-equal branch", "requires explicit SF and OF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.NEG_RM64),
        "REX.W f7 /3", MachineSemanticToken.INTEGER_NEGATE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Neg.value,), "64-bit register or memory negation", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_AL_IMM8),
        "3c ib", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "8-bit accumulator comparison", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETNE_RM8),
        "0f 95 /0", MachineSemanticToken.CONDITIONAL_SET.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Select.value, Handler.Trunc.value), "write one byte from inverted ZF", "legacy high-byte destinations remain distinct vocabulary"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_RM8_IMM8),
        "80 /4 ib", MachineSemanticToken.BITWISE_AND.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "8-bit immediate destination", "legacy high-byte destinations remain distinct vocabulary"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BTS_RM32_IMM8),
        "0f ba /5 ib", MachineSemanticToken.BIT_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("bit_test_set",), "32-bit selected-bit read and set", "writes selected prior bit to CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JLE_REL8),
        "7e cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short signed-less-or-equal branch", "requires explicit ZF, SF, and OF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_R32_RM32),
        "0b /r", MachineSemanticToken.BITWISE_OR.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "32-bit register destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVE_R32_RM32),
        "0f 44 /r", MachineSemanticToken.CONDITIONAL_MOVE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit conditional move", "requires explicit ZF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_R8_IMM8),
        "b0+rb ib", MachineSemanticToken.REGISTER_WRITE_IMMEDIATE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Const.value,), "8-bit immediate register write", "legacy AH-CH-DH-BH and REX low-byte identities remain distinct operands"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVA_R32_RM32),
        "0f 47 /r", MachineSemanticToken.CONDITIONAL_MOVE.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit unsigned-above conditional move", "requires explicit CF and ZF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JG_REL8),
        "7f cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short signed-greater branch", "requires explicit ZF, SF, and OF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_EAX_IMM32),
        "25 id", MachineSemanticToken.BITWISE_AND.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "32-bit accumulator immediate AND", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XOR_RM32_IMM8),
        "83 /6 ib", MachineSemanticToken.BITWISE_XOR.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Xor.value,), "32-bit sign-extended immediate destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_RM32_IMM32),
        "81 /0 id", MachineSemanticToken.INTEGER_ADD.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "32-bit immediate destination", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CDQE),
        "REX.W 98", MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.SExt.value,), "EAX-to-RAX sign extension", "fixed accumulator operands"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XOR_R8_RM8),
        "32 /r", MachineSemanticToken.BITWISE_XOR.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Xor.value,), "8-bit register destination", "legacy high-byte sources remain distinct vocabulary"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BT_RM64_R64),
        "REX.W 0f a3 /r", MachineSemanticToken.BIT_TEST.name,
        BinaryLayer.MACHINE_SEMANTIC, ("bit_test",), "64-bit selected-bit read", "writes selected bit to CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MUL_RM64),
        "REX.W f7 /4", MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Mul.value,), "unsigned RDX:RAX product", "produces two architectural destinations and writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.DEC_RM64),
        "REX.W ff /1", MachineSemanticToken.INTEGER_DECREMENT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "64-bit register or memory decrement", "preserves CF while writing other arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.NOP_66),
        "66 90", MachineSemanticToken.NO_OPERATION.name,
        BinaryLayer.MACHINE_SEMANTIC, ("machine_noop",), "two-byte operand-size-prefix NOP", "no state change"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_RM64_IMM32),
        "REX.W 81 /7 id", MachineSemanticToken.INTEGER_COMPARE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "64-bit sign-extended immediate comparison", "writes flags only"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JL_REL8),
        "7c cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short signed-less branch", "requires explicit SF and OF state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHL_RM32_CL),
        "d3 /4", MachineSemanticToken.SHIFT_LEFT.name,
        BinaryLayer.REPOSITORY_SSA, (Handler.Shl.value,), "32-bit shift by CL", "effective count is architecturally masked and flags are written"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XORPS_XMM_XMMM128),
        "0f 57 /r", MachineSemanticToken.VECTOR_XOR.name,
        BinaryLayer.MACHINE_SEMANTIC, ("vector_xor_128",), "128-bit XMM bitwise XOR", "register or 128-bit memory source"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVUPS_XMM_XMMM128),
        "0f 10 /r", MachineSemanticToken.VECTOR_MOVE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("vector_load_128",), "unaligned 128-bit XMM load or copy", "memory source width is 128 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVUPS_XMMM128_XMM),
        "0f 11 /r", MachineSemanticToken.VECTOR_MOVE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("vector_store_128",), "unaligned 128-bit XMM store or copy", "memory destination width is 128 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVDQU_XMM_XMMM128),
        "f3 0f 6f /r", MachineSemanticToken.VECTOR_MOVE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("vector_load_128",), "unaligned integer XMM load or copy", "mandatory f3 prefix selects the form"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVDQU_XMMM128_XMM),
        "f3 0f 7f /r", MachineSemanticToken.VECTOR_MOVE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("vector_store_128",), "unaligned integer XMM store or copy", "mandatory f3 prefix selects the form"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVDQA_XMMM128_XMM),
        "66 0f 7f /r", MachineSemanticToken.VECTOR_MOVE.name,
        BinaryLayer.MACHINE_SEMANTIC, ("vector_store_128",), "aligned integer XMM store or copy", "mandatory 66 prefix selects the form"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_AL_IMM8), "24 ib", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "8-bit accumulator immediate AND", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_RM16_IMM8), "66 83 /1 ib", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "16-bit immediate destination", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JLE_REL32), "0f 8e cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name, BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near signed-less-or-equal branch", "requires ZF, SF, and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMP_R8_RM8), "3a /r", MachineSemanticToken.INTEGER_COMPARE.name, BinaryLayer.MACHINE_SEMANTIC, ("integer_subtract_flags",), "8-bit register-first comparison", "legacy high-byte sources remain distinct"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JL_REL32), "0f 8c cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name, BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near signed-less branch", "requires SF and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.REP_STOSW), "66 f3 ab", MachineSemanticToken.STRING_STORE.name, BinaryLayer.MACHINE_SEMANTIC, ("repeat_store_word",), "RCX-counted AX stores through RDI", "direction flag and memory state are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVO_R64_RM64), "REX.W 0f 40 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit overflow conditional move", "requires OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.NOP_RM_66), "66... 0f 1f /0", MachineSemanticToken.NO_OPERATION.name, BinaryLayer.MACHINE_SEMANTIC, ("machine_noop",), "operand-size-prefixed multi-byte NOP", "redundant 66 prefixes are retained in provenance"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JNS_REL8), "79 cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name, BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "short nonnegative branch", "requires SF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BTS_RM32_R32), "0f ab /r", MachineSemanticToken.BIT_TEST.name, BinaryLayer.MACHINE_SEMANTIC, ("bit_test_set",), "32-bit selected-bit read and set", "writes prior bit to CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETA_RM8), "0f 97 /0", MachineSemanticToken.CONDITIONAL_SET.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value, Handler.Trunc.value), "write byte from unsigned-above predicate", "requires CF and ZF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MUL_RM32), "f7 /4", MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED.name, BinaryLayer.REPOSITORY_SSA, (Handler.Mul.value,), "unsigned EDX:EAX product", "produces two destinations"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_RM64_R64), "REX.W 21 /r", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "64-bit destination AND", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.NOT_RM32), "f7 /2", MachineSemanticToken.BITWISE_NOT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Not.value,), "32-bit complement", "does not write flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_R16_RM16), "66 23 /r", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "16-bit register destination", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVDQA_XMM_XMMM128), "66 0f 6f /r", MachineSemanticToken.VECTOR_MOVE.name, BinaryLayer.MACHINE_SEMANTIC, ("vector_load_128",), "aligned integer XMM load or copy", "mandatory 66 prefix selects form"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_R64_RM64_FS), "65 REX.W 8b /r", MachineSemanticToken.REGISTER_OR_MEMORY_READ.name, BinaryLayer.MACHINE_SEMANTIC, ("fs_segment_load_64",), "64-bit FS-relative load or copy", "FS base participates in effective address"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ROL_RM64_IMM8), "REX.W c1 /0 ib", MachineSemanticToken.ROTATE_LEFT.name, BinaryLayer.MACHINE_SEMANTIC, ("rotate_left",), "64-bit immediate rotate", "count is masked and flags depend on count"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.IMUL_R64_RM64_IMM8), "REX.W 6b /r ib", MachineSemanticToken.INTEGER_MULTIPLY.name, BinaryLayer.REPOSITORY_SSA, (Handler.Mul.value,), "64-bit signed multiply with immediate", "writes CF and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVAPS_XMMM128_XMM), "0f 29 /r", MachineSemanticToken.VECTOR_MOVE.name, BinaryLayer.MACHINE_SEMANTIC, ("vector_store_128",), "aligned XMM store or copy", "memory destination requires alignment"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVAPS_XMM_XMMM128), "0f 28 /r", MachineSemanticToken.VECTOR_MOVE.name, BinaryLayer.MACHINE_SEMANTIC, ("vector_load_128",), "aligned XMM load or copy", "memory source requires alignment"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BT_RM32_R32), "0f a3 /r", MachineSemanticToken.BIT_TEST.name, BinaryLayer.MACHINE_SEMANTIC, ("bit_test",), "32-bit selected-bit read", "writes selected bit to CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETG_RM8), "0f 9f /0", MachineSemanticToken.CONDITIONAL_SET.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value, Handler.Trunc.value), "write byte from signed-greater predicate", "requires ZF, SF, and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_R16_RM16), "66 03 /r", MachineSemanticToken.INTEGER_ADD.name, BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "16-bit register destination", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.TEST_RM16_IMM16), "66 f7 /0 iw", MachineSemanticToken.INTEGER_TEST.name, BinaryLayer.MACHINE_SEMANTIC, ("integer_test_flags",), "16-bit immediate non-writing AND", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVE_R16_RM16), "66 0f 44 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "16-bit conditional move", "requires ZF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ROL_RM8_IMM8), "c0 /0 ib", MachineSemanticToken.ROTATE_LEFT.name, BinaryLayer.MACHINE_SEMANTIC, ("rotate_left",), "8-bit immediate rotate", "count and flag semantics are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SCASB), "ae", MachineSemanticToken.STRING_COMPARE.name, BinaryLayer.MACHINE_SEMANTIC, ("scan_byte",), "compare AL against byte at RDI", "DF, flags, and RDI state are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.PSRLDQ_XMM_IMM8), "66 0f 73 /3 ib", MachineSemanticToken.VECTOR_SHIFT_RIGHT_LOGICAL.name, BinaryLayer.MACHINE_SEMANTIC, ("vector_byte_shift_right",), "128-bit XMM byte shift", "mandatory prefix and register-only destination"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_RM16_IMM8), "66 83 /0 ib", MachineSemanticToken.INTEGER_ADD.name, BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "16-bit immediate add", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETLE_RM8), "0f 9e /0", MachineSemanticToken.CONDITIONAL_SET.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value, Handler.Trunc.value), "signed-less-or-equal byte", "requires ZF, SF, OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMPXCHG_RM64_R64), "f0 REX.W 0f b1 /r", MachineSemanticToken.ATOMIC_COMPARE_EXCHANGE.name, BinaryLayer.MACHINE_SEMANTIC, ("atomic_compare_exchange_64",), "locked 64-bit compare exchange", "memory ordering, RAX, destination, source, and flags are effects"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ROR_RM64_IMM8), "REX.W c1 /1 ib", MachineSemanticToken.ROTATE_RIGHT.name, BinaryLayer.MACHINE_SEMANTIC, ("rotate_right",), "64-bit immediate rotate", "count and flag semantics are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVAE_R64_RM64), "REX.W 0f 43 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit unsigned-above-or-equal move", "requires CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVB_R64_RM64), "REX.W 0f 42 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit unsigned-below move", "requires CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVB_R16_RM16), "66 0f 42 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "16-bit unsigned-below move", "requires CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVB_R32_RM32), "0f 42 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit unsigned-below move", "requires CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BTR_RM32_R32), "0f b3 /r", MachineSemanticToken.BIT_TEST_RESET.name, BinaryLayer.MACHINE_SEMANTIC, ("bit_test_reset",), "32-bit selected-bit reset", "writes prior bit to CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_RM32_IMM32), "81 /1 id", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "32-bit immediate OR", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHL_RM16_IMM8), "66 c1 /4 ib", MachineSemanticToken.SHIFT_LEFT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shl.value,), "16-bit immediate shift", "count is masked and flags are written"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHL_RM32_IMM8), "c1 /4 ib", MachineSemanticToken.SHIFT_LEFT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shl.value,), "32-bit immediate shift", "count is masked and flags are written"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVBE_R32_RM32), "0f 46 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit unsigned-below-or-equal move", "requires CF and ZF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.REP_MOVSQ), "f3 REX.W a5", MachineSemanticToken.STRING_MOVE.name, BinaryLayer.MACHINE_SEMANTIC, ("repeat_move_qword",), "RCX-counted RSI-to-RDI move", "DF and versioned memory are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETNS_RM8), "0f 99 /0", MachineSemanticToken.CONDITIONAL_SET.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value, Handler.Trunc.value), "nonnegative predicate byte", "requires SF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVSX_R64_RM16), "REX.W 0f bf /r", MachineSemanticToken.SIGN_EXTEND.name, BinaryLayer.REPOSITORY_SSA, (Handler.SExt.value,), "16-to-64-bit sign extension", "memory source width is 16 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVLE_R32_RM32), "0f 4e /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit signed-less-or-equal move", "requires ZF, SF, OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_RM32_R32), "21 /r", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "32-bit destination AND", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.IMUL_R32_RM32_IMM8), "6b /r ib", MachineSemanticToken.INTEGER_MULTIPLY.name, BinaryLayer.REPOSITORY_SSA, (Handler.Mul.value,), "32-bit signed multiply with immediate", "writes CF and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHR_RM64_IMM8), "REX.W c1 /5 ib", MachineSemanticToken.SHIFT_RIGHT_LOGICAL.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shr.value,), "64-bit logical shift", "count is masked and flags are written"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_R8_RM8), "0a /r", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "8-bit register destination", "legacy high-byte sources remain distinct"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XADD_RM32_R32), "f0 0f c1 /r", MachineSemanticToken.ATOMIC_EXCHANGE_ADD.name, BinaryLayer.MACHINE_SEMANTIC, ("atomic_exchange_add_32",), "locked exchange-add", "memory order, both destinations, and flags are effects"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_EAX_IMM32), "0d id", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "32-bit accumulator immediate OR", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_R8_RM8), "22 /r", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "8-bit register destination", "legacy high-byte sources remain distinct"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.LOCK_ADD_RM8_R8), "f0 00 /r", MachineSemanticToken.ATOMIC_ADD.name, BinaryLayer.MACHINE_SEMANTIC, ("atomic_add_8",), "locked byte add", "memory order, destination, and flags are effects"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVQ_RM64_XMM), "66 REX.W 0f 7e /r", MachineSemanticToken.VECTOR_MOVE.name, BinaryLayer.MACHINE_SEMANTIC, ("xmm_low64_store",), "low XMM qword to GPR or memory", "destination width is 64 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XCHG_RM64_R64), "REX.W 87 /r", MachineSemanticToken.EXCHANGE.name, BinaryLayer.MACHINE_SEMANTIC, ("exchange_64",), "64-bit exchange", "memory form is implicitly atomic"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_RM16_IMM16), "66 81 /4 iw", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "16-bit immediate AND", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BTC_RM32_IMM8), "0f ba /7 ib", MachineSemanticToken.BIT_TEST_COMPLEMENT.name, BinaryLayer.MACHINE_SEMANTIC, ("bit_test_complement",), "32-bit selected-bit complement", "writes prior bit to CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.DIV_RM64), "REX.W f7 /6", MachineSemanticToken.INTEGER_DIVIDE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Div.value,), "unsigned RDX:RAX division", "quotient, remainder, and trap are effects"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_RM16_R16), "66 09 /r", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "16-bit destination OR", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.LOCK_DEC_RM32), "f0 ff /1", MachineSemanticToken.INTEGER_DECREMENT.name, BinaryLayer.MACHINE_SEMANTIC, ("atomic_decrement_32",), "locked decrement", "memory order and flags are effects"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVSD_XMM_XMMM64), "f2 0f 10 /r", MachineSemanticToken.VECTOR_MOVE.name, BinaryLayer.MACHINE_SEMANTIC, ("scalar_float64_load",), "scalar double load or XMM copy", "mandatory f2 prefix selects form"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVSD_XMMM64_XMM), "f2 0f 11 /r", MachineSemanticToken.VECTOR_MOVE.name, BinaryLayer.MACHINE_SEMANTIC, ("scalar_float64_store",), "scalar double store or XMM copy", "mandatory f2 prefix selects form"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_RAX_IMM32), "REX.W 05 id", MachineSemanticToken.INTEGER_ADD.name, BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "64-bit accumulator sign-extended immediate add", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_R64_RM64), "REX.W 0b /r", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "64-bit register destination", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.INT_IMM8), "cd ib", MachineSemanticToken.SOFTWARE_INTERRUPT.name, BinaryLayer.MACHINE_SEMANTIC, ("software_interrupt",), "immediate interrupt vector", "control, privilege, stack, and flags effects are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.INC_RM16), "66 ff /0", MachineSemanticToken.INTEGER_INCREMENT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "16-bit increment", "preserves CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHR_RM8_IMM8), "c0 /5 ib", MachineSemanticToken.SHIFT_RIGHT_LOGICAL.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shr.value,), "8-bit logical shift", "count and flags are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CQO), "REX.W 99", MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR.name, BinaryLayer.REPOSITORY_SSA, (Handler.SExt.value,), "RAX sign extension into RDX:RAX", "produces the high dividend half"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.IDIV_RM64), "REX.W f7 /7", MachineSemanticToken.INTEGER_DIVIDE_SIGNED.name, BinaryLayer.REPOSITORY_SSA, (Handler.Div.value,), "signed RDX:RAX division", "quotient, remainder, and trap are effects"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVAE_R32_RM32), "0f 43 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit unsigned-above-or-equal move", "requires CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVL_R32_RM32), "0f 4c /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit signed-less move", "requires SF and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XOR_RM32_R32), "31 /r", MachineSemanticToken.BITWISE_XOR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Xor.value,), "32-bit destination XOR", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHL_RM64_CL), "REX.W d3 /4", MachineSemanticToken.SHIFT_LEFT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shl.value,), "64-bit shift by CL", "count is masked and flags are written"),
)


BINARY_EQUIVALENCE_TABLE: tuple[BinaryEquivalence, ...] = (
    *PE_EQUIVALENCE_TABLE,
    *X86_SSA_EQUIVALENCE_TABLE,
)


def _validate_equivalence_table(table: Iterable[BinaryEquivalence]) -> None:
    identities: set[tuple[BinaryLayer, int]] = set()
    for row in table:
        identity = (row.source_layer, row.source_token)
        if identity in identities:
            raise RuntimeError(f"duplicate binary equivalence identity {identity}")
        if not row.encoded_form or not row.canonical_meaning or not row.target_tokens:
            raise RuntimeError(f"incomplete binary equivalence row {identity}")
        identities.add(identity)


_validate_equivalence_table(BINARY_EQUIVALENCE_TABLE)

_covered_pe_tokens = frozenset(row.source_token for row in PE_EQUIVALENCE_TABLE)
if _covered_pe_tokens != frozenset(int(token) for token in PEVocabularyToken):
    raise RuntimeError("PE equivalence table does not cover every PE vocabulary token")
_covered_x86_tokens = frozenset(row.source_token for row in X86_SSA_EQUIVALENCE_TABLE)
if _covered_x86_tokens != frozenset(int(token) for token in X86InstructionToken):
    raise RuntimeError("x86 equivalence table does not cover every instruction token")

EQUIVALENCE_BY_SOURCE = MappingProxyType({
    (row.source_layer, row.source_token): row
    for row in BINARY_EQUIVALENCE_TABLE
})


def equivalences_targeting(
    target_layer: BinaryLayer,
    target_token: str,
) -> tuple[BinaryEquivalence, ...]:
    """Return every declared source representation for one target token."""

    return tuple(
        row for row in BINARY_EQUIVALENCE_TABLE
        if row.target_layer is target_layer and target_token in row.target_tokens
    )


@dataclass(frozen=True, slots=True)
class PESection:
    name: str
    virtual_address: int
    virtual_size: int
    raw_offset: int
    raw_size: int
    characteristics: int

    @property
    def executable(self) -> bool:
        return bool(self.characteristics & int(PESectionFlag.EXECUTE))

    @property
    def raw_end(self) -> int:
        return self.raw_offset + self.raw_size

    def contains_rva(self, rva: int) -> bool:
        span = max(self.virtual_size, self.raw_size)
        return span > 0 and self.virtual_address <= rva < self.virtual_address + span

    def file_offset_for_rva(self, rva: int) -> int | None:
        if not self.contains_rva(rva):
            return None
        delta = rva - self.virtual_address
        if delta >= self.raw_size:
            return None
        return self.raw_offset + delta


@dataclass(frozen=True, slots=True)
class PERuntimeFunction:
    """One AMD64 exception-directory function range."""

    begin_rva: int
    end_rva: int
    unwind_info_rva: int

    def contains_rva(self, rva: int) -> bool:
        return self.begin_rva <= rva < self.end_rva


@dataclass(frozen=True, slots=True)
class PEImage:
    machine: PEMachine
    pe32_plus: bool
    image_base: int
    entrypoint_rva: int
    entrypoint_file_offset: int
    entrypoint_section_index: int
    sections: tuple[PESection, ...]
    runtime_functions: tuple[PERuntimeFunction, ...]
    encoded: bytes

    @property
    def entrypoint_section(self) -> PESection:
        return self.sections[self.entrypoint_section_index]

    def section_for_rva(self, rva: int) -> PESection | None:
        matches = tuple(section for section in self.sections if section.contains_rva(rva))
        return matches[0] if len(matches) == 1 else None

    def file_offset_for_rva(self, rva: int) -> int | None:
        section = self.section_for_rva(rva)
        return None if section is None else section.file_offset_for_rva(rva)

    def runtime_function_for_rva(self, rva: int) -> PERuntimeFunction | None:
        matches = tuple(
            function for function in self.runtime_functions if function.contains_rva(rva)
        )
        return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True, slots=True)
class PEStatistics:
    file_size: int
    section_count: int
    executable_section_count: int
    executable_raw_bytes: int
    entrypoint_rva: int
    entrypoint_file_offset: int
    runtime_function_count: int


@dataclass(frozen=True, slots=True)
class PEToSSAResult:
    image: PEImage
    statistics: PEStatistics
    code_region_offset: int
    code_region_size: int
    lifting: BinaryToSSAResult


def _strict_file_bytes(binary_region) -> bytes:
    if isinstance(binary_region, bytes):
        return binary_region
    if isinstance(binary_region, bytearray):
        return bytes(binary_region)
    if isinstance(binary_region, memoryview):
        try:
            return binary_region.cast("B").tobytes()
        except (TypeError, ValueError) as error:
            raise BinaryFormatError(
                "binary memoryview must be contiguous and byte-addressable"
            ) from error
    values: list[int] = []
    try:
        iterator = iter(binary_region)
    except TypeError as error:
        raise BinaryFormatError("binary file must be bytes or an integer iterable") from error
    for offset, value in enumerate(iterator):
        if isinstance(value, bool):
            raise BinaryFormatError(f"boolean at binary file offset {offset}")
        try:
            byte = operator.index(value)
        except TypeError as error:
            raise BinaryFormatError(
                f"non-integer at binary file offset {offset}: {value!r}"
            ) from error
        if not 0 <= byte <= 0xFF:
            raise BinaryFormatError(
                f"value outside [0, 255] at binary file offset {offset}: {byte}"
            )
        values.append(byte)
    return bytes(values)


def _need(data: bytes, offset: int, size: int, label: str) -> None:
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise BinaryFormatError(
            f"truncated {label}: need [{offset}, {offset + size}) inside file size {len(data)}"
        )


def _u16(data: bytes, offset: int, label: str) -> int:
    _need(data, offset, 2, label)
    return int.from_bytes(data[offset:offset + 2], "little")


def _u32(data: bytes, offset: int, label: str) -> int:
    _need(data, offset, 4, label)
    return int.from_bytes(data[offset:offset + 4], "little")


def _u64(data: bytes, offset: int, label: str) -> int:
    _need(data, offset, 8, label)
    return int.from_bytes(data[offset:offset + 8], "little")


def parse_pe_image(
    binary_region,
    *,
    maximum_file_size: int,
    maximum_sections: int = 96,
) -> tuple[PEImage, PEStatistics]:
    """Parse the PE envelope and map its entry-point RVA to file bytes."""

    if isinstance(maximum_file_size, bool) or isinstance(maximum_sections, bool):
        raise ValueError("PE limits must be non-negative integers")
    try:
        file_limit = operator.index(maximum_file_size)
        section_limit = operator.index(maximum_sections)
    except TypeError as error:
        raise ValueError("PE limits must be non-negative integers") from error
    if file_limit < 0 or section_limit < 0:
        raise ValueError("PE limits must be non-negative integers")

    data = _strict_file_bytes(binary_region)
    if len(data) > file_limit:
        raise BinaryFormatError(
            f"PE file size {len(data)} exceeds maximum_file_size {file_limit}"
        )
    _need(data, 0, 0x40, "DOS header")
    if data[:2] != b"MZ":
        raise BinaryFormatError("missing DOS MZ signature")
    pe_offset = _u32(data, 0x3C, "DOS e_lfanew")
    _need(data, pe_offset, 24, "PE signature and COFF header")
    if data[pe_offset:pe_offset + 4] != b"PE\x00\x00":
        raise BinaryFormatError(f"missing PE signature at file offset {pe_offset}")

    coff = pe_offset + 4
    machine_value = _u16(data, coff, "COFF machine")
    try:
        machine = PEMachine(machine_value)
    except ValueError as error:
        raise BinaryFormatError(f"unsupported PE machine {machine_value:#06x}") from error
    section_count = _u16(data, coff + 2, "COFF section count")
    if section_count == 0 or section_count > section_limit:
        raise BinaryFormatError(
            f"PE section count {section_count} is outside [1, {section_limit}]"
        )
    optional_size = _u16(data, coff + 16, "COFF optional-header size")
    optional = coff + 20
    _need(data, optional, optional_size, "PE optional header")
    if optional_size < 64:
        raise BinaryFormatError(f"PE optional header is too small: {optional_size}")
    magic = _u16(data, optional, "PE optional-header magic")
    if magic == 0x20B:
        pe32_plus = True
        image_base = _u64(data, optional + 24, "PE32+ image base")
    elif magic == 0x10B:
        pe32_plus = False
        image_base = _u32(data, optional + 28, "PE32 image base")
    else:
        raise BinaryFormatError(f"unsupported PE optional-header magic {magic:#06x}")
    entrypoint_rva = _u32(data, optional + 16, "PE entry-point RVA")
    directory_count_offset = optional + (108 if pe32_plus else 92)
    directory_table_offset = optional + (112 if pe32_plus else 96)
    directory_count = 0
    if optional_size >= directory_count_offset - optional + 4:
        directory_count = _u32(data, directory_count_offset, "PE data-directory count")
    visible_directories = min(directory_count, 16)
    if visible_directories:
        if (
            directory_table_offset + visible_directories * 8
            > optional + optional_size
        ):
            raise BinaryFormatError(
                "PE data-directory count extends beyond the optional header"
            )
        _need(
            data,
            directory_table_offset,
            visible_directories * 8,
            "PE data directories",
        )
    exception_rva = 0
    exception_size = 0
    if visible_directories > 3:
        exception_entry = directory_table_offset + 3 * 8
        exception_rva = _u32(data, exception_entry, "PE exception-directory RVA")
        exception_size = _u32(data, exception_entry + 4, "PE exception-directory size")

    section_table = optional + optional_size
    _need(data, section_table, section_count * 40, "PE section table")
    sections: list[PESection] = []
    raw_ranges: list[tuple[int, int, str]] = []
    for index in range(section_count):
        offset = section_table + index * 40
        raw_name = data[offset:offset + 8].split(b"\x00", 1)[0]
        name = raw_name.decode("ascii", errors="backslashreplace") or f"section_{index}"
        section = PESection(
            name=name,
            virtual_size=_u32(data, offset + 8, f"{name} virtual size"),
            virtual_address=_u32(data, offset + 12, f"{name} virtual address"),
            raw_size=_u32(data, offset + 16, f"{name} raw size"),
            raw_offset=_u32(data, offset + 20, f"{name} raw offset"),
            characteristics=_u32(data, offset + 36, f"{name} characteristics"),
        )
        if section.raw_size:
            _need(data, section.raw_offset, section.raw_size, f"{name} raw data")
            raw_ranges.append((section.raw_offset, section.raw_end, name))
        sections.append(section)

    for left, right in zip(sorted(raw_ranges), sorted(raw_ranges)[1:]):
        if left[1] > right[0]:
            raise BinaryFormatError(
                f"overlapping PE raw sections {left[2]!r} and {right[2]!r}"
            )

    mappings = [
        (index, section, section.file_offset_for_rva(entrypoint_rva))
        for index, section in enumerate(sections)
        if section.contains_rva(entrypoint_rva)
    ]
    if len(mappings) != 1:
        raise BinaryFormatError(
            f"entry-point RVA {entrypoint_rva:#x} maps to {len(mappings)} sections"
        )
    entry_index, entry_section, entry_offset = mappings[0]
    if entry_offset is None:
        raise BinaryFormatError(
            f"entry-point RVA {entrypoint_rva:#x} has no file-backed byte"
        )
    if not entry_section.executable:
        raise BinaryFormatError(
            f"entry point maps to non-executable section {entry_section.name!r}"
        )

    runtime_functions: list[PERuntimeFunction] = []
    if exception_rva or exception_size:
        if not exception_rva or not exception_size:
            raise BinaryFormatError("PE exception directory has only one nonzero field")
        if exception_size % 12:
            raise BinaryFormatError(
                f"AMD64 exception-directory size {exception_size} is not divisible by 12"
            )
        exception_sections = [
            section for section in sections
            if section.file_offset_for_rva(exception_rva) is not None
        ]
        if len(exception_sections) != 1:
            raise BinaryFormatError("PE exception directory does not map uniquely to file bytes")
        exception_section = exception_sections[0]
        exception_offset = exception_section.file_offset_for_rva(exception_rva)
        assert exception_offset is not None
        if exception_offset + exception_size > exception_section.raw_end:
            raise BinaryFormatError("PE exception directory crosses its raw section boundary")
        _need(data, exception_offset, exception_size, "AMD64 exception directory")
        previous_begin = -1
        for record_offset in range(exception_offset, exception_offset + exception_size, 12):
            begin_rva = _u32(data, record_offset, "runtime-function begin RVA")
            end_rva = _u32(data, record_offset + 4, "runtime-function end RVA")
            unwind_rva = _u32(data, record_offset + 8, "runtime-function unwind RVA")
            if begin_rva >= end_rva:
                raise BinaryFormatError(
                    f"invalid runtime-function range [{begin_rva:#x}, {end_rva:#x})"
                )
            if begin_rva < previous_begin:
                raise BinaryFormatError("AMD64 runtime-function table is not sorted")
            previous_begin = begin_rva
            begin_section = next((
                section for section in sections
                if section.executable and section.file_offset_for_rva(begin_rva) is not None
            ), None)
            end_section = next((
                section for section in sections
                if section.executable and section.file_offset_for_rva(end_rva - 1) is not None
            ), None)
            if begin_section is None or end_section is not begin_section:
                raise BinaryFormatError(
                    f"runtime function [{begin_rva:#x}, {end_rva:#x}) is not file-backed executable code"
                )
            runtime_functions.append(PERuntimeFunction(begin_rva, end_rva, unwind_rva))

    image = PEImage(
        machine=machine,
        pe32_plus=pe32_plus,
        image_base=image_base,
        entrypoint_rva=entrypoint_rva,
        entrypoint_file_offset=entry_offset,
        entrypoint_section_index=entry_index,
        sections=tuple(sections),
        runtime_functions=tuple(runtime_functions),
        encoded=data,
    )
    executable = tuple(section for section in sections if section.executable)
    statistics = PEStatistics(
        file_size=len(data),
        section_count=len(sections),
        executable_section_count=len(executable),
        executable_raw_bytes=sum(section.raw_size for section in executable),
        entrypoint_rva=entrypoint_rva,
        entrypoint_file_offset=entry_offset,
        runtime_function_count=len(runtime_functions),
    )
    return image, statistics


def pe_runtime_function_region(
    image: PEImage,
    rva: int,
    *,
    maximum_function_size: int,
) -> tuple[PERuntimeFunction, int, bytes]:
    """Return one exact `.pdata`-bounded AMD64 function byte region."""

    if isinstance(maximum_function_size, bool):
        raise ValueError("maximum_function_size must be a positive integer")
    try:
        limit = operator.index(maximum_function_size)
    except TypeError as error:
        raise ValueError("maximum_function_size must be a positive integer") from error
    if limit <= 0:
        raise ValueError("maximum_function_size must be a positive integer")
    function = image.runtime_function_for_rva(int(rva))
    if function is None:
        raise BinaryFormatError(f"no unique runtime-function record contains RVA {rva:#x}")
    size = function.end_rva - function.begin_rva
    if size > limit:
        raise BinaryFormatError(
            f"runtime function size {size} exceeds maximum_function_size {limit}"
        )
    offset = image.file_offset_for_rva(function.begin_rva)
    if offset is None:
        raise BinaryFormatError("runtime-function begin RVA is not file backed")
    _need(image.encoded, offset, size, "runtime-function bytes")
    return function, offset, image.encoded[offset:offset + size]


def raise_pe_entrypoint_to_ssa(
    binary_region,
    *,
    maximum_file_size: int,
    maximum_code_region_size: int,
    name: str = "lifted_pe_entrypoint",
    argument_registers=("ecx", "edx", "r8d", "r9d"),
    argument_names=("arg0", "arg1", "arg2", "arg3"),
    full_vocabulary_report: bool = False,
    audit_preview_bytes: int = 16,
) -> PEToSSAResult:
    """Parse PE, bound its entry-point bytes, and invoke the x86→SSA raiser."""

    if isinstance(maximum_code_region_size, bool):
        raise ValueError("maximum_code_region_size must be a positive integer")
    try:
        code_limit = operator.index(maximum_code_region_size)
    except TypeError as error:
        raise ValueError("maximum_code_region_size must be a positive integer") from error
    if code_limit <= 0:
        raise ValueError("maximum_code_region_size must be a positive integer")

    image, statistics = parse_pe_image(
        binary_region,
        maximum_file_size=maximum_file_size,
    )
    if image.machine is not PEMachine.AMD64 or not image.pe32_plus:
        raise BinaryFormatError(
            "the current ISA vocabulary requires an AMD64 PE32+ image"
        )
    section = image.entrypoint_section
    available = section.raw_end - image.entrypoint_file_offset
    code_size = min(available, code_limit)
    code = image.encoded[
        image.entrypoint_file_offset:image.entrypoint_file_offset + code_size
    ]
    lifting = raise_binary_region_to_ssa(
        code,
        maximum_file_size=code_limit,
        base_address=image.image_base + image.entrypoint_rva,
        name=name,
        argument_registers=argument_registers,
        argument_names=argument_names,
        allow_trailing_after_terminal=True,
        full_vocabulary_report=full_vocabulary_report,
        audit_preview_bytes=audit_preview_bytes,
    )
    return PEToSSAResult(
        image=image,
        statistics=statistics,
        code_region_offset=image.entrypoint_file_offset,
        code_region_size=code_size,
        lifting=lifting,
    )


__all__ = [
    "AuditConfidence",
    "BINARY_EQUIVALENCE_TABLE",
    "EQUIVALENCE_BY_SOURCE",
    "EncodingFlag",
    "BinaryEquivalence",
    "BinaryFormatError",
    "BinaryLayer",
    "BinaryToSSAResult",
    "DecodeReport",
    "DecodedInstruction",
    "EffectiveAddressOperand",
    "ImmediateOperand",
    "MachineSemanticToken",
    "PEMachine",
    "PERuntimeFunction",
    "PESection",
    "PESectionFlag",
    "PEStatistics",
    "PEToSSAResult",
    "PEVocabularyToken",
    "PE_EQUIVALENCE_TABLE",
    "PEImage",
    "RegisterOperand",
    "PrefixAction",
    "ReadFailure",
    "ReadPhase",
    "ReadStatus",
    "RelativeAddressOperand",
    "VocabularyFailure",
    "VocabularyAuditReport",
    "VocabularyStatistics",
    "X86_SSA_EQUIVALENCE_TABLE",
    "X86InstructionToken",
    "X86EncodingRow",
    "X86ReadBatch",
    "X86ReadHeadConfig",
    "X86ReadHeadState",
    "X86ReferenceDecoder",
    "X86Register",
    "X86TensorReadHead",
    "controlled_x86_64_read_head_config",
    "equivalences_targeting",
    "parse_pe_image",
    "pe_runtime_function_region",
    "raise_binary_region_to_ssa",
    "raise_pe_entrypoint_to_ssa",
]
