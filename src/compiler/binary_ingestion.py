"""Bounded binary-container ingestion and cross-layer equivalence tables.

This module owns file/container structure. ISA instruction framing remains in
``machine_reference_vocabulary`` and SSA construction remains in
``machine_code_lifting``. Keeping those grammars separate prevents PE metadata
bytes from being mistaken for executable x86 bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from hashlib import sha256
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
    ReadHeadDirection,
    ReadFailure,
    ReadPhase,
    ReadStatus,
    X86EncodingRow,
    X86EncodingFields,
    X86ReadBatch,
    X86ReadHeadConfig,
    X86ReadHeadState,
    X86ReversibleReadHead,
    X86TensorReadHead,
    controlled_x86_64_read_head_config,
    controlled_x86_64_read_head_profile,
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


@dataclass(frozen=True, slots=True)
class ReverseBinarySelection:
    """An elective, proof-gated repository-SSA to ISA selection.

    These records do not participate in ingestion and never rewrite SSA on
    their own.  A host lowering may request candidates only after proving all
    required facts.  This keeps exact machine-state recovery distinct from a
    later decision to select a denser machine encoding.
    """

    source_tokens: tuple[str, ...]
    target_token: int
    encoded_form: str
    lane_count: int
    lane_width: int
    required_facts: frozenset[str]
    preserved_state: frozenset[str]
    canonical_meaning: str


@dataclass(frozen=True, slots=True)
class ReverseBinarySelectionPlan:
    """One audited decision to retain or newly select a PE instruction.

    Exact-byte retention is available only when every SSA instruction in the
    group carries the same decoded address/token/bytes provenance and those
    bytes decode back to the selected token.  Transformed SSA can still select
    an encoding template, but cannot claim byte identity.
    """

    selection: ReverseBinarySelection
    source_operations: tuple[str, ...]
    source_value_ids: tuple[int, ...]
    machine_address: int | None
    encoded: bytes | None
    mode: str
    witness: str


SSA_PE_REVERSE_SELECTION_TABLE: tuple[ReverseBinarySelection, ...] = (
    ReverseBinarySelection(
        (Handler.Add.value,),
        int(X86InstructionToken.ADD_R64_IMM8),
        "REX.W 83 /0 ib", 1, 64,
        frozenset({
            "register-or-memory-destination", "signed-immediate-8",
            "width-64", "modulo-2^64", "all-add-flags-exact",
        }),
        frozenset({"xmm", "mxcsr"}),
        "64-bit modular add of a sign-extended immediate with exact integer flags",
    ),
    ReverseBinarySelection(
        (Handler.VectorAddModulo.value,),
        int(X86InstructionToken.PADDQ_XMM_XMMM128),
        "66 0f d4 /r", 2, 64,
        frozenset({
            "two-independent-lanes",
            "modulo-2^64",
            "no-cross-lane-carry",
            "xmm-destination-available",
        }),
        frozenset({"rflags", "mxcsr", "upper-xmm-outside-128"}),
        "two independent repository-SSA uint64 modular additions",
    ),
    ReverseBinarySelection(
        (Handler.VectorSubtractModulo.value,),
        int(X86InstructionToken.PSUBQ_XMM_XMMM128),
        "66 0f fb /r", 2, 64,
        frozenset({
            "two-independent-lanes",
            "modulo-2^64",
            "no-cross-lane-borrow",
            "xmm-destination-available",
        }),
        frozenset({"rflags", "mxcsr", "upper-xmm-outside-128"}),
        "two independent repository-SSA uint64 modular subtractions",
    ),
    ReverseBinarySelection(
        (Handler.StridedMemoryCopy.value,),
        int(X86InstructionToken.REP_MOVSQ),
        "f3 REX.W a5", 1, 64,
        frozenset({
            "source-register-rsi", "destination-register-rdi",
            "count-register-rcx", "direction-flag-df",
            "ordered-overlap-semantics", "qword-elements",
        }),
        frozenset({"rflags-except-df", "xmm", "mxcsr"}),
        "ordered RCX-counted qword copy from RSI to RDI with DF stride",
    ),
    ReverseBinarySelection(
        (
            Handler.AtomicExchangeAddObserved.value,
            Handler.AtomicExchangeAddMemory.value,
        ),
        int(X86InstructionToken.XADD_RM32_R32),
        "f0 0f c1 /r", 1, 32,
        frozenset({
            "memory-destination", "register-source", "width-32",
            "sequentially-consistent", "locked",
            "source-receives-observed", "all-add-flags-exact",
        }),
        frozenset({"xmm", "mxcsr"}),
        "locked 32-bit atomic exchange-add with observed source result",
    ),
    ReverseBinarySelection(
        (
            Handler.Shr.value, Handler.And.value,
            Handler.Shl.value, Handler.Xor.value,
        ),
        int(X86InstructionToken.BTC_RM32_IMM8),
        "0f ba /7 ib", 1, 32,
        frozenset({
            "width-32", "immediate-bit-index", "destination-bit-complement",
            "cf-is-prior-bit", "other-flags-preserved",
        }),
        frozenset({"of", "sf", "zf", "af", "pf", "xmm", "mxcsr"}),
        "32-bit selected-bit complement with prior bit copied to CF",
    ),
)


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
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.PSRLDQ_XMM_IMM8), "66 0f 73 /3 ib", MachineSemanticToken.VECTOR_SHIFT_RIGHT_LOGICAL.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shr.value,), "128-bit XMM byte shift", "count is scaled by eight; counts at least sixteen produce zero; flags and MXCSR unchanged"),
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
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.REP_MOVSQ), "f3 REX.W a5", MachineSemanticToken.STRING_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.StridedMemoryCopy.value, Handler.Add.value, Handler.Mul.value), "RCX-counted RSI-to-RDI move", "ordered overlap behavior, DF, RCX, RSI, RDI, and versioned memory are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETNS_RM8), "0f 99 /0", MachineSemanticToken.CONDITIONAL_SET.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value, Handler.Trunc.value), "nonnegative predicate byte", "requires SF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVSX_R64_RM16), "REX.W 0f bf /r", MachineSemanticToken.SIGN_EXTEND.name, BinaryLayer.REPOSITORY_SSA, (Handler.SExt.value,), "16-to-64-bit sign extension", "memory source width is 16 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVLE_R32_RM32), "0f 4e /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit signed-less-or-equal move", "requires ZF, SF, OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_RM32_R32), "21 /r", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "32-bit destination AND", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.IMUL_R32_RM32_IMM8), "6b /r ib", MachineSemanticToken.INTEGER_MULTIPLY.name, BinaryLayer.REPOSITORY_SSA, (Handler.Mul.value,), "32-bit signed multiply with immediate", "writes CF and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHR_RM64_IMM8), "REX.W c1 /5 ib", MachineSemanticToken.SHIFT_RIGHT_LOGICAL.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shr.value,), "64-bit logical shift", "count is masked and flags are written"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_R8_RM8), "0a /r", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "8-bit register destination", "legacy high-byte sources remain distinct"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XADD_RM32_R32), "f0 0f c1 /r", MachineSemanticToken.ATOMIC_EXCHANGE_ADD.name, BinaryLayer.REPOSITORY_SSA, (Handler.AtomicExchangeAddObserved.value, Handler.AtomicExchangeAddMemory.value, Handler.Add.value), "locked exchange-add", "sequential consistency, observed value, result memory, source register, and arithmetic flags are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_EAX_IMM32), "0d id", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "32-bit accumulator immediate OR", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_R8_RM8), "22 /r", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "8-bit register destination", "legacy high-byte sources remain distinct"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.LOCK_ADD_RM8_R8), "f0 00 /r", MachineSemanticToken.ATOMIC_ADD.name, BinaryLayer.MACHINE_SEMANTIC, ("atomic_add_8",), "locked byte add", "memory order, destination, and flags are effects"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVQ_RM64_XMM), "66 REX.W 0f 7e /r", MachineSemanticToken.VECTOR_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Trunc.value, Handler.Store.value), "low XMM qword to GPR or memory", "low 64 encoded bits transfer to the authored GPR or versioned memory destination; flags and MXCSR unchanged"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XCHG_RM64_R64), "REX.W 87 /r", MachineSemanticToken.EXCHANGE.name, BinaryLayer.MACHINE_SEMANTIC, ("exchange_64",), "64-bit exchange", "memory form is implicitly atomic"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_RM16_IMM16), "66 81 /4 iw", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "16-bit immediate AND", "writes flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BTC_RM32_IMM8), "0f ba /7 ib", MachineSemanticToken.BIT_TEST_COMPLEMENT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shr.value, Handler.And.value, Handler.Shl.value, Handler.Xor.value), "32-bit selected-bit complement", "addressed word, selected bit mutation, and prior bit in CF are explicit"),
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
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.NOT_RM8), "f6 /2", MachineSemanticToken.BITWISE_NOT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Not.value,), "8-bit complement", "does not write flags; legacy high-byte registers remain distinct"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SAR_RM64_IMM8), "REX.W c1 /7 ib", MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC.name, BinaryLayer.REPOSITORY_SSA, (Handler.AShr.value,), "64-bit immediate arithmetic shift", "count is masked and flags are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JNS_REL32), "0f 89 cd", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name, BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "near nonnegative branch", "requires SF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVSX_R64_RM8), "REX.W 0f be /r", MachineSemanticToken.SIGN_EXTEND.name, BinaryLayer.REPOSITORY_SSA, (Handler.SExt.value,), "8-to-64-bit sign extension", "memory source width is 8 bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.IMUL_R64_RM64), "REX.W 0f af /r", MachineSemanticToken.INTEGER_MULTIPLY.name, BinaryLayer.MACHINE_SEMANTIC, ("signed_multiply_low_64",), "two-operand signed multiply", "exact CF/OF fit test needs the full signed 128-bit product"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_RM8_IMM8), "80 /1 ib", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "byte immediate OR", "legacy high-byte destinations remain distinct"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_RM32_IMM32), "81 /5 id", MachineSemanticToken.INTEGER_SUBTRACT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "32-bit immediate subtract", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XOR_RM64_IMM8), "REX.W 83 /6 ib", MachineSemanticToken.BITWISE_XOR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Xor.value,), "64-bit sign-extended immediate XOR", "writes logical flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVG_R32_RM32), "0f 4f /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit signed-greater conditional move", "requires ZF, SF, and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVG_R64_RM64), "REX.W 0f 4f /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit signed-greater conditional move", "requires ZF, SF, and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVNS_R32_RM32), "0f 49 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit nonnegative conditional move", "requires SF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_EAX_IMM32), "05 id", MachineSemanticToken.INTEGER_ADD.name, BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "32-bit accumulator immediate add", "writes arithmetic flags and zero-extends EAX"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_EAX_IMM32), "2d id", MachineSemanticToken.INTEGER_SUBTRACT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "32-bit accumulator immediate subtract", "writes arithmetic flags and zero-extends EAX"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_AL_IMM8), "0c ib", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "AL immediate OR", "preserves the upper 56 bits of RAX"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_RM8_IMM8), "80 /5 ib", MachineSemanticToken.INTEGER_SUBTRACT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "byte immediate subtract", "legacy high-byte destinations remain distinct"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.OR_R16_RM16), "66 0b /r", MachineSemanticToken.BITWISE_OR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Or.value,), "16-bit register-destination OR", "operand-size override is required"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_RM64_R64), "REX.W 29 /r", MachineSemanticToken.INTEGER_SUBTRACT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "64-bit register-to-memory subtract", "writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVLE_R64_RM64), "REX.W 0f 4e /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit signed-less-or-equal conditional move", "requires ZF, SF, and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETAE_RM8), "0f 93 /0", MachineSemanticToken.CONDITIONAL_SET.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value, Handler.Trunc.value), "unsigned-above-or-equal byte result", "requires CF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XCHG_RM32_R32), "87 /r", MachineSemanticToken.EXCHANGE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Load.value, Handler.Store.value), "32-bit register or memory exchange", "memory form is implicitly atomic"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CDQ), "99", MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR.name, BinaryLayer.MACHINE_SEMANTIC, ("sign_extend_eax_into_edx_eax",), "sign-extend EAX into EDX:EAX", "produces the high dividend half"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.IDIV_RM32), "f7 /7", MachineSemanticToken.INTEGER_DIVIDE_SIGNED.name, BinaryLayer.MACHINE_SEMANTIC, ("signed_divide_edx_eax",), "signed EDX:EAX division", "quotient, remainder, and divide-error trap are effects"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BSR_R32_RM32), "0f bd /r", MachineSemanticToken.BIT_SCAN_REVERSE.name, BinaryLayer.MACHINE_SEMANTIC, ("bit_scan_reverse_32",), "index of the highest set bit", "zero sets ZF and leaves destination architecturally undefined"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CVTSI2SD_XMM_RM64), "f2 REX.W 0f 2a /r", MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT64.name, BinaryLayer.MACHINE_SEMANTIC, ("signed_int64_to_scalar_float64",), "signed integer to low double lane", "legacy form preserves the upper XMM lane"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVQ_XMM_RM64), "66 REX.W 0f 6e /r", MachineSemanticToken.VECTOR_MOVE_LOW_ZERO_UPPER.name, BinaryLayer.MACHINE_SEMANTIC, ("move_qword_to_xmm_zero_upper",), "integer qword to low XMM lane", "upper XMM lane is cleared"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVSX_R32_RM8), "0f be /r", MachineSemanticToken.SIGN_EXTEND.name, BinaryLayer.REPOSITORY_SSA, (Handler.SExt.value,), "8-to-32-bit sign extension", "32-bit destination write clears its upper half"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHR_RM64_CL), "REX.W d3 /5", MachineSemanticToken.SHIFT_RIGHT_LOGICAL.name, BinaryLayer.MACHINE_SEMANTIC, ("shift_right_logical_dynamic",), "64-bit logical shift by masked CL", "zero count preserves destination and all flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETL_RM8), "0f 9c /0", MachineSemanticToken.CONDITIONAL_SET.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "signed-less byte result", "requires SF and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SETS_RM8), "0f 98 /0", MachineSemanticToken.CONDITIONAL_SET.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "negative-sign byte result", "requires SF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVNS_R64_RM64), "REX.W 0f 49 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit nonnegative conditional move", "requires SF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVA_R64_RM64), "REX.W 0f 47 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit unsigned-above conditional move", "requires CF and ZF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHR_RM32_CL), "d3 /5", MachineSemanticToken.SHIFT_RIGHT_LOGICAL.name, BinaryLayer.MACHINE_SEMANTIC, ("shift_right_logical_dynamic_32",), "32-bit logical shift by masked CL", "zero count preserves destination and all flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SAR_RM32_IMM8), "c1 /7 ib", MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC.name, BinaryLayer.REPOSITORY_SSA, (Handler.AShr.value,), "32-bit arithmetic shift by immediate", "masked nonzero immediate has explicit CF/OF/ZF/SF/PF semantics"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_AX_IMM16), "66 25 iw", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "16-bit accumulator immediate AND", "preserves the upper 48 bits of RAX"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOV_RM16_IMM16), "66 c7 /0 iw", MachineSemanticToken.REGISTER_WRITE_IMMEDIATE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Store.value,), "16-bit immediate register or memory write", "register form preserves upper bits"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.XOR_RM64_IMM32), "REX.W 81 /6 id", MachineSemanticToken.BITWISE_XOR.name, BinaryLayer.REPOSITORY_SSA, (Handler.Xor.value,), "64-bit XOR with sign-extended imm32", "writes logical flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BSR_R64_RM64), "REX.W 0f bd /r", MachineSemanticToken.BIT_SCAN_REVERSE.name, BinaryLayer.MACHINE_SEMANTIC, ("bit_scan_reverse_64",), "index of the highest set bit", "zero sets ZF and leaves destination architecturally undefined"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.DEC_RM8), "fe /1", MachineSemanticToken.INTEGER_DECREMENT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "byte decrement", "preserves CF and updates arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.INC_RM8), "fe /0", MachineSemanticToken.INTEGER_INCREMENT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "byte increment", "preserves CF and updates arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_R8_RM8), "2a /r", MachineSemanticToken.INTEGER_SUBTRACT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "byte register-destination subtract", "updates arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.UCOMISD_XMM_XMMM64), "66 0f 2e /r", MachineSemanticToken.SCALAR_FLOAT64_COMPARE_UNORDERED.name, BinaryLayer.MACHINE_SEMANTIC, ("scalar_float64_compare_unordered",), "unordered scalar double comparison", "sets ZF/PF/CF and clears OF/SF/AF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVQ_XMM_XMMM64), "f3 0f 7e /r", MachineSemanticToken.VECTOR_MOVE_LOW_ZERO_UPPER.name, BinaryLayer.MACHINE_SEMANTIC, ("move_qword_to_xmm_zero_upper",), "qword load into low XMM lane", "clears the upper lane"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADDSD_XMM_XMMM64), "f2 0f 58 /r", MachineSemanticToken.SCALAR_FLOAT64_ADD.name, BinaryLayer.MACHINE_SEMANTIC, ("scalar_float64_add",), "scalar double addition", "preserves the destination upper lane"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.PUNPCKLQDQ_XMM_XMMM128), "66 0f 6c /r", MachineSemanticToken.VECTOR_UNPACK_LOW_QWORDS.name, BinaryLayer.MACHINE_SEMANTIC, ("vector_unpack_low_qwords",), "interleave low qwords", "destination low qword is retained below source low qword"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SHL_RM8_IMM8), "c0 /4 ib", MachineSemanticToken.SHIFT_LEFT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shl.value,), "byte logical left shift", "masked immediate has explicit flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_RM8_IMM8), "80 /0 ib", MachineSemanticToken.INTEGER_ADD.name, BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "byte immediate add", "updates arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.JP_REL8), "7a cb", MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP.name, BinaryLayer.REPOSITORY_SSA, (Handler.CondBr.value,), "branch when parity flag is set", "requires explicit PF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.PUNPCKLBW_XMM_XMMM128), "66 0f 60 /r", MachineSemanticToken.VECTOR_UNPACK_LOW_BYTES.name, BinaryLayer.MACHINE_SEMANTIC, ("vector_unpack_low_bytes",), "interleave low bytes", "eight lanes from each operand form sixteen destination bytes"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MULSD_XMM_XMMM64), "f2 0f 59 /r", MachineSemanticToken.SCALAR_FLOAT64_MULTIPLY.name, BinaryLayer.MACHINE_SEMANTIC, ("scalar_float64_multiply",), "scalar double multiplication", "MXCSR rounding and exceptions are explicit machine state"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.PADDQ_XMM_XMMM128), "66 0f d4 /r", MachineSemanticToken.VECTOR_ADD_QWORDS.name, BinaryLayer.MACHINE_SEMANTIC, ("vector_add_qwords",), "packed qword addition", "two independent modular 64-bit lanes"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.IMUL_R64_RM64_IMM32), "REX.W 69 /r id", MachineSemanticToken.INTEGER_MULTIPLY.name, BinaryLayer.MACHINE_SEMANTIC, ("signed_multiply_64_imm32",), "signed 64-bit multiply with sign-extended imm32", "CF/OF indicate whether the full product fits the retained destination"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.COMISD_XMM_XMMM64), "66 0f 2f /r", MachineSemanticToken.SCALAR_FLOAT64_COMPARE_ORDERED.name, BinaryLayer.MACHINE_SEMANTIC, ("scalar_float64_compare_ordered",), "ordered scalar double comparison", "NaN updates MXCSR invalid state and comparison flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.PUNPCKLWD_XMM_XMMM128), "66 0f 61 /r", MachineSemanticToken.VECTOR_UNPACK_LOW_WORDS.name, BinaryLayer.MACHINE_SEMANTIC, ("vector_unpack_low_words",), "interleave low words", "four lanes from each operand form eight destination words"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVD_XMM_RM32), "66 0f 6e /r", MachineSemanticToken.VECTOR_MOVE_LOW_ZERO_UPPER.name, BinaryLayer.MACHINE_SEMANTIC, ("move_dword_to_xmm_zero_upper",), "integer dword into low XMM lane", "clears bits 32 through 127"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVGE_R64_RM64), "REX.W 0f 4d /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit signed-greater-or-equal conditional move", "requires SF and OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_AL_IMM8), "2c ib", MachineSemanticToken.INTEGER_SUBTRACT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "8-bit accumulator immediate subtract", "preserves upper RAX bits and writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_R8_RM8), "02 /r", MachineSemanticToken.INTEGER_ADD.name, BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "8-bit register-destination add", "legacy high-byte operands remain distinct and arithmetic flags are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.IMUL_RM64), "REX.W f7 /5", MachineSemanticToken.INTEGER_MULTIPLY.name, BinaryLayer.REPOSITORY_SSA, (Handler.SMulLow.value,), "signed accumulator-form 64-bit multiply", "produces the full signed product in RDX:RAX and defines CF/OF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVBE_R64_RM64), "REX.W 0f 46 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit unsigned-below-or-equal conditional move", "requires CF and ZF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_RM16_R16), "66 01 /r", MachineSemanticToken.INTEGER_ADD.name, BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "16-bit register-to-register-or-memory add", "preserves upper register bits and writes arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CVTSI2SS_XMM_RM64), "f3 REX.W 0f 2a /r", MachineSemanticToken.SIGNED_INTEGER_TO_SCALAR_FLOAT32.name, BinaryLayer.REPOSITORY_SSA, (Handler.SInt64ToFloat32Bits.value,), "signed integer to low binary32 XMM lane", "MXCSR rounding and precision exception are explicit; upper 96 bits are preserved"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADDSS_XMM_XMMM32), "f3 0f 58 /r", MachineSemanticToken.SCALAR_FLOAT32_ADD.name, BinaryLayer.REPOSITORY_SSA, (Handler.Float32AddBits.value, Handler.MXCSRFloat32Add.value), "scalar binary32 addition into low XMM lane", "encoded IEEE result and MXCSR status/trap transition are explicit; upper 96 bits are preserved"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.DIVSS_XMM_XMMM32), "f3 0f 5e /r", MachineSemanticToken.SCALAR_FLOAT32_DIVIDE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Float32DivideBits.value, Handler.MXCSRFloat32Divide.value), "scalar binary32 division into low XMM lane", "encoded IEEE result and ordered MXCSR status/trap transition are explicit; upper 96 bits are preserved"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.COMISS_XMM_XMMM32), "0f 2f /r", MachineSemanticToken.SCALAR_FLOAT32_COMPARE_ORDERED.name, BinaryLayer.REPOSITORY_SSA, (Handler.Float32IsNaNBits.value, Handler.Float32BitsLt.value, Handler.Float32BitsEq.value, Handler.MXCSRInvalid.value), "ordered scalar binary32 comparison into flags", "encoded comparisons, invalid status/trap, and CF/PF/ZF with OF/SF/AF clearing are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVS_R64_RM64), "REX.W 0f 48 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "64-bit sign-flag conditional move", "reads SF and preserves destination when clear"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.DIVSD_XMM_XMMM64), "f2 0f 5e /r", MachineSemanticToken.SCALAR_FLOAT64_DIVIDE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Float64DivideBits.value, Handler.MXCSRFloat64Divide.value), "scalar binary64 division into low XMM lane", "encoded IEEE result and ordered MXCSR status/trap transition are explicit; upper 64 bits are preserved"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMPXCHG_RM32_R32), "f0 0f b1 /r", MachineSemanticToken.ATOMIC_COMPARE_EXCHANGE.name, BinaryLayer.REPOSITORY_SSA, (Handler.AtomicCompareExchangeObserved.value, Handler.AtomicCompareExchangeSuccess.value, Handler.AtomicCompareExchangeMemory.value), "locked 32-bit compare exchange", "memory ordering, EAX, destination, source, and arithmetic flags remain explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.AND_RM64_IMM32), "REX.W 81 /4 id", MachineSemanticToken.BITWISE_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "64-bit AND with sign-extended imm32", "destination and logical flags are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.BSWAP_R32), "0f c8+rd", MachineSemanticToken.BYTE_SWAP.name, BinaryLayer.REPOSITORY_SSA, (Handler.ByteSwap.value,), "reverse four bytes in a 32-bit register", "32-bit destination write zeroes the upper register half and leaves flags unchanged"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ROR_RM16_IMM8), "66 c1 /1 ib", MachineSemanticToken.ROTATE_RIGHT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shl.value, Handler.Shr.value, Handler.Or.value), "16-bit immediate right rotate", "effective count and CF/OF rules are explicit; other flags preserved"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUBSD_XMM_XMMM64), "f2 0f 5c /r", MachineSemanticToken.SCALAR_FLOAT64_SUBTRACT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Float64SubtractBits.value, Handler.MXCSRFloat64Subtract.value), "scalar binary64 subtraction into low XMM lane", "encoded IEEE result and ordered MXCSR status/trap transition are explicit; upper 64 bits are preserved"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVS_R32_RM32), "0f 48 /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit sign-flag conditional move", "reads SF; selected 32-bit destination write zeroes its upper register half"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CVTTSD2SI_R64_XMMM64), "f2 REX.W 0f 2c /r", MachineSemanticToken.SCALAR_FLOAT64_TO_SIGNED_INT64_TRUNCATE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Float64ToSInt64TruncBits.value, Handler.MXCSRFloat64ToSIntInvalid.value), "truncate encoded scalar binary64 toward zero into signed int64", "rounding control is ignored; NaN/infinity/out-of-range produce integer-indefinite and explicit MXCSR invalid status/trap"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SUB_RM32_R32), "29 /r", MachineSemanticToken.INTEGER_SUBTRACT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "32-bit register-source subtraction", "destination, arithmetic flags, and 32-bit register zero-extension are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.PCMPEQQ_XMM_XMMM128), "66 0f 38 29 /r", MachineSemanticToken.VECTOR_COMPARE_EQUAL_QWORDS.name, BinaryLayer.REPOSITORY_SSA, (Handler.VectorCompareEqualMask.value,), "two independent signedness-neutral qword equality masks", "each true lane becomes all 64 one-bits; false becomes zero; flags and MXCSR unchanged"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SAR_RM32_1), "d1 /7", MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shr.value,), "32-bit arithmetic right shift by one", "sign-fill and CF/OF behavior are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.PSUBQ_XMM_XMMM128), "66 0f fb /r", MachineSemanticToken.VECTOR_SUBTRACT_QWORDS.name, BinaryLayer.REPOSITORY_SSA, (Handler.VectorSubtractModulo.value,), "two independent modular 64-bit lane subtractions", "borrows never cross lane boundaries; flags and MXCSR unchanged"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.SBB_R8_RM8), "1a /r", MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW.name, BinaryLayer.REPOSITORY_SSA, (Handler.Sub.value,), "8-bit subtract source and incoming carry from destination", "result width, CF/OF/SF/ZF/AF/PF, and preserved upper register bits are explicit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ADD_AL_IMM8), "04 ib", MachineSemanticToken.INTEGER_ADD.name, BinaryLayer.REPOSITORY_SSA, (Handler.Add.value,), "8-bit accumulator immediate addition", "preserves upper RAX bits and writes exact arithmetic flags"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.IMUL_R32_RM32_IMM32), "69 /r id", MachineSemanticToken.INTEGER_MULTIPLY.name, BinaryLayer.REPOSITORY_SSA, (Handler.SMulLow.value, Handler.SMulOverflow.value), "32-bit signed multiply with immediate", "CF/OF report truncation; other arithmetic flags remain architecturally undefined/preserved"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ANDPS_XMM_XMMM128), "0f 54 /r", MachineSemanticToken.VECTOR_AND.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value,), "128-bit XMM bit-pattern conjunction", "all 128 encoded bits participate; flags and MXCSR unchanged"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CMOVGE_R32_RM32), "0f 4d /r", MachineSemanticToken.CONDITIONAL_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Select.value,), "32-bit signed-greater-or-equal conditional move", "condition is SF==OF; selected 32-bit destination write zeroes upper half"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.IMUL_RM32), "f7 /5", MachineSemanticToken.INTEGER_MULTIPLY.name, BinaryLayer.REPOSITORY_SSA, (Handler.SMulLow.value, Handler.SMulHigh.value, Handler.SMulOverflow.value), "signed accumulator-form 32-bit multiply", "full signed product is written to EDX:EAX and CF/OF report sign-extension fit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.PSHUFD_XMM_XMMM128_IMM8), "66 0f 70 /r ib", MachineSemanticToken.VECTOR_SHUFFLE_DWORDS.name, BinaryLayer.REPOSITORY_SSA, (Handler.VectorShuffle.value,), "four-lane 32-bit source permutation", "each destination lane selects the source lane named by its two control bits; flags and MXCSR unchanged"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CVTTSD2SI_R32_XMMM64), "f2 0f 2c /r", MachineSemanticToken.SCALAR_FLOAT64_TO_SIGNED_INT32_TRUNCATE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Float64ToSInt32TruncBits.value, Handler.MXCSRFloat64ToSIntInvalid.value), "truncate encoded scalar binary64 toward zero into signed int32", "rounding control is ignored; NaN/infinity/out-of-range produce 0x80000000 and explicit MXCSR invalid status/trap"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.CVTDQ2PD_XMM_XMMM64), "f3 0f e6 /r", MachineSemanticToken.VECTOR_SIGNED_INT32_TO_FLOAT64.name, BinaryLayer.REPOSITORY_SSA, (Handler.VectorSInt32ToFloat64Bits.value, Handler.MXCSRVectorSInt32ToFloat64.value), "two signed int32 lanes to two encoded binary64 lanes", "both int32 inputs are exactly representable in binary64; MXCSR state transition is explicit and host floating arithmetic is unused"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.MOVD_RM32_XMM), "66 0f 7e /r", MachineSemanticToken.VECTOR_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value, Handler.Store.value), "low XMM dword to GPR or memory", "low 32 encoded bits transfer to the authored destination; GPR write zeroes upper 32 bits; flags and MXCSR unchanged"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.ROR_RM64_CL), "REX.W d3 /1", MachineSemanticToken.ROTATE_RIGHT.name, BinaryLayer.REPOSITORY_SSA, (Handler.Shl.value, Handler.Shr.value, Handler.Or.value, Handler.Select.value), "64-bit right rotate by masked CL", "zero count preserves destination and all flags; one-bit count defines OF; CF follows the rotated high bit"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.LOCK_INC_RM32), "f0 ff /0", MachineSemanticToken.ATOMIC_INCREMENT.name, BinaryLayer.REPOSITORY_SSA, (Handler.AtomicExchangeAddObserved.value, Handler.AtomicExchangeAddMemory.value), "locked 32-bit memory increment", "sequentially consistent atomic read-modify-write preserves CF and updates OF/SF/ZF/AF/PF"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.VINSERTF128_YMM_YMM_XMMM128_IMM8), "VEX.256.66.0f3a.W0 18 /r ib", MachineSemanticToken.VECTOR_INSERT_128_LANE.name, BinaryLayer.REPOSITORY_SSA, (Handler.And.value, Handler.Shl.value, Handler.Or.value), "insert one 128-bit source lane into a selected half of a 256-bit destination", "guest AVX is decomposed into exact repository bit operations and is never a host pass-through requirement"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.VMOVDQA_YMMM256_YMM), "VEX.256.66.0f.WIG 7f /r", MachineSemanticToken.VECTOR_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Store.value,), "move one 256-bit YMM bit pattern to register or memory", "guest AVX store is explicit repository memory state and never requires host SIMD execution"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.STOSB), "aa", MachineSemanticToken.STRING_STORE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Store.value, Handler.Select.value, Handler.Add.value), "store AL at RDI and advance by direction flag", "single iteration does not read or write RCX"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.VMOVNTDQ_M256_YMM), "VEX.256.66.0f.WIG e7 /r", MachineSemanticToken.VECTOR_MOVE.name, BinaryLayer.REPOSITORY_SSA, (Handler.Store.value,), "non-temporal 256-bit YMM bit-pattern store", "non-temporal is an ordering/cache hint; repository memory state receives the exact 256 bits without host AVX forwarding"),
    BinaryEquivalence(BinaryLayer.ISA_ENCODING, int(X86InstructionToken.REP_STOSB), "f3 aa", MachineSemanticToken.STRING_STORE.name, BinaryLayer.REPOSITORY_SSA, (Handler.StridedStoreFill.value, Handler.Select.value, Handler.Mul.value, Handler.Add.value), "repeat AL byte fill from RDI for RCX elements", "explicit iterative memory operation consumes RCX and advances RDI under DF"),
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


def eligible_reverse_selections(
    source_tokens: Iterable[str],
    *,
    proven_facts: Iterable[str] = (),
    allow_multi_lane: bool = False,
) -> tuple[ReverseBinarySelection, ...]:
    """Return explicitly enabled, fully proved SSA-to-PE selections.

    The opt-in flag is intentionally false by default.  Token equality is
    exact and every semantic/state fact on the record must be supplied, so a
    caller cannot collapse scalar or machine-state operations merely because
    their arithmetic spelling happens to match.
    """

    if not allow_multi_lane:
        return ()
    tokens = tuple(str(token) for token in source_tokens)
    facts = frozenset(str(fact) for fact in proven_facts)
    return tuple(
        row for row in SSA_PE_REVERSE_SELECTION_TABLE
        if row.source_tokens == tokens and row.required_facts <= facts
    )


def plan_reverse_selection(
    instructions: Iterable[object],
    *,
    proven_facts: Iterable[str] = (),
    allow_multi_lane: bool = False,
) -> ReverseBinarySelectionPlan | None:
    """Produce a proof-gated reverse plan for one authored SSA group."""

    group = tuple(instructions)
    operations = tuple(str(getattr(item, "op")) for item in group)
    facts = frozenset(str(fact) for fact in proven_facts)

    def ordered_subsequence(needles: tuple[str, ...]) -> bool:
        cursor = iter(operations)
        return all(any(item == needle for item in cursor) for needle in needles)

    candidates = tuple(
        row for row in SSA_PE_REVERSE_SELECTION_TABLE
        if allow_multi_lane
        and row.required_facts <= facts
        and ordered_subsequence(row.source_tokens)
    )
    if len(candidates) != 1:
        return None
    selection = candidates[0]
    result_ids = tuple(
        int(item.res.id) for item in group
        if getattr(item, "res", None) is not None
    )
    provenance = tuple(
        (
            getattr(item, "attributes", {}).get("machine_address"),
            getattr(item, "attributes", {}).get("machine_token"),
            getattr(item, "attributes", {}).get("machine_bytes"),
        )
        for item in group
    )
    exact = None
    address = None
    if provenance and len(set(provenance)) == 1:
        raw_address, raw_token, raw_encoded = provenance[0]
        if raw_address is not None and raw_encoded:
            try:
                encoded = bytes.fromhex(str(raw_encoded))
                decoded, end = X86ReferenceDecoder().decode_one(
                    memoryview(encoded), 0, base_address=int(raw_address),
                )
            except (ValueError, VocabularyDecodeError):
                pass
            else:
                if (
                    end == len(encoded)
                    and int(decoded.token) == int(selection.target_token)
                    and int(raw_token) == int(decoded.token)
                ):
                    head = X86TensorReadHead.from_profile(
                        controlled_x86_64_read_head_profile(),
                    )
                    exact = head.rewrite_instruction(
                        int(decoded.token), decoded.encoded,
                    )
                    address = int(raw_address)
    mode = "exact-retention" if exact is not None else "template-selection"
    digest = sha256()
    digest.update(mode.encode("ascii"))
    digest.update(repr(operations).encode("utf-8"))
    digest.update(repr(tuple(sorted(facts))).encode("utf-8"))
    digest.update(str(selection.target_token).encode("ascii"))
    digest.update(exact or b"")
    return ReverseBinarySelectionPlan(
        selection, operations, result_ids, address, exact, mode,
        digest.hexdigest(),
    )


def write_reverse_selection(
    plan: ReverseBinarySelectionPlan,
    fields: X86EncodingFields | None = None,
) -> bytes:
    """Write a proof-gated selection through the bidirectional x86 head.

    Exact retention revalidates and rewrites the preserved instruction fields.
    A transformed selection must provide its newly allocated operand fields;
    the API never guesses registers, addresses, or immediates from SSA names.
    """

    head = X86TensorReadHead.from_profile(controlled_x86_64_read_head_profile())
    if fields is None:
        if plan.encoded is None:
            raise ValueError("transformed reverse selection requires encoding fields")
        return head.rewrite_instruction(plan.selection.target_token, plan.encoded)
    return head.write_instruction(plan.selection.target_token, fields)


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
class PEImportSymbol:
    """One PE import-address-table slot and its stable external identity."""

    library: str
    name: str | None
    ordinal: int | None
    iat_rva: int

    @property
    def display_name(self) -> str:
        symbol = self.name if self.name is not None else f"ordinal:{self.ordinal}"
        return f"{self.library}!{symbol}"


@dataclass(frozen=True, slots=True)
class PEDelayImportSymbol:
    """One delayed IAT slot lowered into the deterministic link plan."""

    library: str
    name: str | None
    ordinal: int | None
    iat_rva: int
    module_handle_rva: int

    @property
    def display_name(self) -> str:
        symbol = self.name if self.name is not None else f"ordinal:{self.ordinal}"
        return f"{self.library}!{symbol}"


@dataclass(frozen=True, slots=True)
class PEBaseRelocation:
    """One loader relocation expressed against the image RVA namespace."""

    type: int
    rva: int


@dataclass(frozen=True, slots=True)
class PEExportSymbol:
    """One named or ordinal PE export, including a possible forwarder."""

    name: str | None
    ordinal: int
    rva: int | None
    forwarder: str | None = None

    @property
    def display_name(self) -> str:
        return self.name if self.name is not None else f"ordinal:{self.ordinal}"


@dataclass(frozen=True, slots=True)
class PETLSDirectory:
    """Bounded initial-thread TLS template and process-attach callbacks."""

    raw_data_start_rva: int
    raw_data_end_rva: int
    index_rva: int
    callbacks: tuple[int, ...]
    zero_fill_size: int
    characteristics: int
    template: bytes


@dataclass(frozen=True, slots=True)
class PEImage:
    machine: PEMachine
    coff_characteristics: int
    pe32_plus: bool
    image_base: int
    entrypoint_rva: int
    entrypoint_file_offset: int
    entrypoint_section_index: int
    sections: tuple[PESection, ...]
    runtime_functions: tuple[PERuntimeFunction, ...]
    imports: tuple[PEImportSymbol, ...]
    delay_imports: tuple[PEDelayImportSymbol, ...]
    export_name: str | None
    exports: tuple[PEExportSymbol, ...]
    base_relocations: tuple[PEBaseRelocation, ...]
    tls_directory: PETLSDirectory | None
    encoded: bytes

    @property
    def entrypoint_section(self) -> PESection:
        return self.sections[self.entrypoint_section_index]

    @property
    def is_dll(self) -> bool:
        return bool(self.coff_characteristics & 0x2000)

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

    def export_by_name(self, name: str) -> PEExportSymbol | None:
        """Resolve a case-sensitive PE export name without host loading."""

        matches = tuple(item for item in self.exports if item.name == str(name))
        if len(matches) > 1:
            destinations = {(item.rva, item.forwarder) for item in matches}
            if len(destinations) > 1:
                raise BinaryFormatError(f"conflicting PE export name {name!r}")
        return matches[0] if matches else None

    def export_by_ordinal(self, ordinal: int) -> PEExportSymbol | None:
        """Resolve an ordinal while allowing aliases with one destination."""

        matches = tuple(item for item in self.exports if item.ordinal == int(ordinal))
        if len(matches) > 1:
            destinations = {(item.rva, item.forwarder) for item in matches}
            if len(destinations) > 1:
                raise BinaryFormatError(f"conflicting PE export ordinal {ordinal}")
        return matches[0] if matches else None


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
    coff_characteristics = _u16(data, coff + 18, "COFF characteristics")
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
    export_rva = 0
    export_size = 0
    import_rva = 0
    import_size = 0
    relocation_rva = 0
    relocation_size = 0
    delay_import_rva = 0
    delay_import_size = 0
    tls_rva = 0
    tls_size = 0
    if visible_directories > 0:
        export_entry = directory_table_offset
        export_rva = _u32(data, export_entry, "PE export-directory RVA")
        export_size = _u32(data, export_entry + 4, "PE export-directory size")
    if visible_directories > 1:
        import_entry = directory_table_offset + 1 * 8
        import_rva = _u32(data, import_entry, "PE import-directory RVA")
        import_size = _u32(data, import_entry + 4, "PE import-directory size")
    if visible_directories > 3:
        exception_entry = directory_table_offset + 3 * 8
        exception_rva = _u32(data, exception_entry, "PE exception-directory RVA")
        exception_size = _u32(data, exception_entry + 4, "PE exception-directory size")
    if visible_directories > 5:
        relocation_entry = directory_table_offset + 5 * 8
        relocation_rva = _u32(data, relocation_entry, "PE relocation-directory RVA")
        relocation_size = _u32(
            data, relocation_entry + 4, "PE relocation-directory size",
        )
    if visible_directories > 9:
        tls_entry = directory_table_offset + 9 * 8
        tls_rva = _u32(data, tls_entry, "PE TLS-directory RVA")
        tls_size = _u32(data, tls_entry + 4, "PE TLS-directory size")
    if visible_directories > 13:
        delay_entry = directory_table_offset + 13 * 8
        delay_import_rva = _u32(data, delay_entry, "PE delay-import RVA")
        delay_import_size = _u32(data, delay_entry + 4, "PE delay-import size")

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

    def file_offset_for_import_rva(rva: int, label: str) -> int:
        matches = [section.file_offset_for_rva(rva) for section in sections]
        offsets = [offset for offset in matches if offset is not None]
        if len(offsets) != 1:
            raise BinaryFormatError(f"{label} RVA {rva:#x} does not map uniquely to file bytes")
        return offsets[0]

    def import_ascii(rva: int, label: str) -> str:
        offset = file_offset_for_import_rva(rva, label)
        end = data.find(b"\x00", offset, min(len(data), offset + 4096))
        if end < 0:
            raise BinaryFormatError(f"unterminated {label} string at RVA {rva:#x}")
        try:
            return data[offset:end].decode("ascii")
        except UnicodeDecodeError as error:
            raise BinaryFormatError(f"non-ASCII {label} string at RVA {rva:#x}") from error

    tls_directory: PETLSDirectory | None = None
    if tls_rva or tls_size:
        if not tls_rva or not tls_size:
            raise BinaryFormatError("PE TLS directory has only one nonzero field")
        structure_size = 40 if pe32_plus else 24
        if tls_size < structure_size:
            raise BinaryFormatError("PE TLS directory is smaller than its structure")
        tls_offset = file_offset_for_import_rva(tls_rva, "TLS directory")
        tls_section = next(
            section for section in sections
            if section.file_offset_for_rva(tls_rva) == tls_offset
        )
        if tls_offset + tls_size > tls_section.raw_end:
            raise BinaryFormatError("PE TLS directory crosses its raw section boundary")
        _need(data, tls_offset, structure_size, "PE TLS directory")
        if pe32_plus:
            start_va = _u64(data, tls_offset, "PE TLS raw-data start VA")
            end_va = _u64(data, tls_offset + 8, "PE TLS raw-data end VA")
            index_va = _u64(data, tls_offset + 16, "PE TLS index VA")
            callbacks_va = _u64(data, tls_offset + 24, "PE TLS callbacks VA")
            zero_fill = _u32(data, tls_offset + 32, "PE TLS zero-fill size")
            characteristics = _u32(data, tls_offset + 36, "PE TLS characteristics")
            pointer_size = 8
        else:
            start_va = _u32(data, tls_offset, "PE TLS raw-data start VA")
            end_va = _u32(data, tls_offset + 4, "PE TLS raw-data end VA")
            index_va = _u32(data, tls_offset + 8, "PE TLS index VA")
            callbacks_va = _u32(data, tls_offset + 12, "PE TLS callbacks VA")
            zero_fill = _u32(data, tls_offset + 16, "PE TLS zero-fill size")
            characteristics = _u32(data, tls_offset + 20, "PE TLS characteristics")
            pointer_size = 4

        def tls_va_to_rva(value: int, label: str, *, allow_zero: bool = False) -> int:
            if not value and allow_zero:
                return 0
            if value < image_base:
                raise BinaryFormatError(f"{label} {value:#x} precedes image base")
            result = value - image_base
            if result >= 1 << 32:
                raise BinaryFormatError(f"{label} {value:#x} exceeds the image RVA namespace")
            return result

        if bool(start_va) != bool(end_va):
            raise BinaryFormatError("PE TLS raw-data range has only one nonzero endpoint")
        start_rva = tls_va_to_rva(start_va, "PE TLS raw-data start", allow_zero=True)
        end_rva = tls_va_to_rva(end_va, "PE TLS raw-data end", allow_zero=True)
        if end_rva < start_rva:
            raise BinaryFormatError("PE TLS raw-data range is reversed")
        template_size = end_rva - start_rva
        if template_size + zero_fill > 64 * 1024 * 1024:
            raise BinaryFormatError("PE TLS initial allocation exceeds 64 MiB")
        if template_size:
            template_offset = file_offset_for_import_rva(start_rva, "TLS template")
            template_section = next(
                section for section in sections
                if section.file_offset_for_rva(start_rva) == template_offset
            )
            if template_offset + template_size > template_section.raw_end:
                raise BinaryFormatError("PE TLS template crosses its raw section boundary")
            _need(data, template_offset, template_size, "PE TLS template")
            template = data[template_offset:template_offset + template_size]
        else:
            template = b""
        index_rva = tls_va_to_rva(index_va, "PE TLS index")
        if not any(section.contains_rva(index_rva) for section in sections):
            raise BinaryFormatError("PE TLS index is not mapped")
        callbacks: list[int] = []
        callbacks_rva = tls_va_to_rva(
            callbacks_va, "PE TLS callback array", allow_zero=True,
        )
        if callbacks_rva:
            for callback_index in range(4096):
                pointer_rva = callbacks_rva + callback_index * pointer_size
                pointer_offset = file_offset_for_import_rva(
                    pointer_rva, "TLS callback pointer",
                )
                callback_va = (
                    _u64(data, pointer_offset, "PE64 TLS callback")
                    if pe32_plus else _u32(data, pointer_offset, "PE32 TLS callback")
                )
                if not callback_va:
                    break
                callback_rva = tls_va_to_rva(callback_va, "PE TLS callback")
                section = next((
                    item for item in sections
                    if item.executable and item.contains_rva(callback_rva)
                ), None)
                if section is None:
                    raise BinaryFormatError(
                        f"PE TLS callback RVA {callback_rva:#x} is not executable"
                    )
                callbacks.append(callback_rva)
            else:
                raise BinaryFormatError("PE TLS callback table exceeds 4096 entries")
        tls_directory = PETLSDirectory(
            start_rva, end_rva, index_rva, tuple(callbacks),
            zero_fill, characteristics, bytes(template),
        )

    export_name: str | None = None
    exports: list[PEExportSymbol] = []
    if export_rva or export_size:
        if not export_rva or not export_size:
            raise BinaryFormatError("PE export directory has only one nonzero field")
        if export_size < 40:
            raise BinaryFormatError("PE export directory is smaller than its header")
        export_offset = file_offset_for_import_rva(export_rva, "export directory")
        export_end = export_offset + export_size
        export_section = next(
            section for section in sections
            if section.file_offset_for_rva(export_rva) == export_offset
        )
        if export_end > export_section.raw_end:
            raise BinaryFormatError("PE export directory crosses its raw section boundary")
        _need(data, export_offset, export_size, "PE export directory")
        export_name_rva = _u32(data, export_offset + 12, "PE export module name")
        ordinal_base = _u32(data, export_offset + 16, "PE export ordinal base")
        function_count = _u32(data, export_offset + 20, "PE export function count")
        name_count = _u32(data, export_offset + 24, "PE export name count")
        function_table_rva = _u32(data, export_offset + 28, "PE export address table")
        name_table_rva = _u32(data, export_offset + 32, "PE export name table")
        ordinal_table_rva = _u32(data, export_offset + 36, "PE export ordinal table")
        if function_count > 65536 or name_count > 65536:
            raise BinaryFormatError("PE export table exceeds 65536 entries")
        if name_count > function_count:
            raise BinaryFormatError("PE export name count exceeds function count")
        export_name = import_ascii(export_name_rva, "export module name")
        function_table = (
            file_offset_for_import_rva(function_table_rva, "export address table")
            if function_count else 0
        )
        name_table = (
            file_offset_for_import_rva(name_table_rva, "export name pointer table")
            if name_count else 0
        )
        ordinal_table = (
            file_offset_for_import_rva(ordinal_table_rva, "export ordinal table")
            if name_count else 0
        )
        _need(data, function_table, function_count * 4, "PE export address table")
        _need(data, name_table, name_count * 4, "PE export name pointer table")
        _need(data, ordinal_table, name_count * 2, "PE export ordinal table")
        names_by_index: dict[int, list[str]] = {}
        for name_index in range(name_count):
            ordinal_index = _u16(
                data, ordinal_table + name_index * 2, "PE export ordinal index",
            )
            if ordinal_index >= function_count:
                raise BinaryFormatError(
                    f"PE export ordinal index {ordinal_index} exceeds function table"
                )
            symbol_name_rva = _u32(
                data, name_table + name_index * 4, "PE export symbol name RVA",
            )
            names_by_index.setdefault(ordinal_index, []).append(
                import_ascii(symbol_name_rva, "export symbol name")
            )
        export_rva_end = export_rva + export_size
        for function_index in range(function_count):
            function_rva = _u32(
                data, function_table + function_index * 4, "PE export function RVA",
            )
            if not function_rva:
                continue
            if export_rva <= function_rva < export_rva_end:
                forwarder = import_ascii(function_rva, "export forwarder")
                target_rva = None
            else:
                if not any(section.contains_rva(function_rva) for section in sections):
                    raise BinaryFormatError(
                        f"PE export target RVA {function_rva:#x} is not mapped"
                    )
                forwarder = None
                target_rva = function_rva
            names = names_by_index.get(function_index, [None])
            exports.extend(
                PEExportSymbol(
                    name=name,
                    ordinal=ordinal_base + function_index,
                    rva=target_rva,
                    forwarder=forwarder,
                )
                for name in names
            )

    imports: list[PEImportSymbol] = []
    if import_rva or import_size:
        if not import_rva or not import_size:
            raise BinaryFormatError("PE import directory has only one nonzero field")
        import_offset = file_offset_for_import_rva(import_rva, "import directory")
        import_end = import_offset + import_size
        _need(data, import_offset, import_size, "PE import directory")
        descriptor_offset = import_offset
        descriptor_count = 0
        pointer_size = 8 if pe32_plus else 4
        ordinal_mask = 1 << (pointer_size * 8 - 1)
        while descriptor_offset + 20 <= import_end:
            fields = tuple(_u32(data, descriptor_offset + index * 4, "PE import descriptor") for index in range(5))
            if not any(fields):
                break
            original_thunk_rva, _, _, library_name_rva, first_thunk_rva = fields
            if not library_name_rva or not first_thunk_rva:
                raise BinaryFormatError("PE import descriptor lacks library name or first thunk")
            library = import_ascii(library_name_rva, "import library")
            lookup_rva = original_thunk_rva or first_thunk_rva
            for symbol_index in range(65536):
                thunk_rva = lookup_rva + symbol_index * pointer_size
                thunk_offset = file_offset_for_import_rva(thunk_rva, "import lookup thunk")
                thunk = (
                    _u64(data, thunk_offset, "PE64 import lookup thunk")
                    if pe32_plus else _u32(data, thunk_offset, "PE32 import lookup thunk")
                )
                if thunk == 0:
                    break
                if thunk & ordinal_mask:
                    name = None
                    ordinal = thunk & 0xFFFF
                else:
                    name_rva = thunk & (ordinal_mask - 1)
                    name_offset = file_offset_for_import_rva(name_rva, "import hint/name")
                    _need(data, name_offset, 2, "PE import hint")
                    name = import_ascii(name_rva + 2, "import symbol")
                    ordinal = None
                imports.append(PEImportSymbol(
                    library=library,
                    name=name,
                    ordinal=ordinal,
                    iat_rva=first_thunk_rva + symbol_index * pointer_size,
                ))
            else:
                raise BinaryFormatError("PE import thunk table exceeds 65536 symbols")
            descriptor_count += 1
            if descriptor_count > 4096:
                raise BinaryFormatError("PE import directory exceeds 4096 descriptors")
            descriptor_offset += 20
        else:
            raise BinaryFormatError("PE import directory has no terminating descriptor")

    delay_imports: list[PEDelayImportSymbol] = []
    if delay_import_rva or delay_import_size:
        if not delay_import_rva or not delay_import_size:
            raise BinaryFormatError("PE delay-import directory has only one nonzero field")
        delay_offset = file_offset_for_import_rva(
            delay_import_rva, "delay-import directory",
        )
        delay_end = delay_offset + delay_import_size
        delay_section = next(
            section for section in sections
            if section.file_offset_for_rva(delay_import_rva) == delay_offset
        )
        if delay_end > delay_section.raw_end:
            raise BinaryFormatError("PE delay-import directory crosses its raw section boundary")
        _need(data, delay_offset, delay_import_size, "PE delay-import directory")
        descriptor_offset = delay_offset
        descriptor_count = 0
        pointer_size = 8 if pe32_plus else 4
        ordinal_mask = 1 << (pointer_size * 8 - 1)
        while descriptor_offset + 32 <= delay_end:
            fields = tuple(
                _u32(data, descriptor_offset + index * 4, "PE delay-import descriptor")
                for index in range(8)
            )
            if not any(fields):
                break
            attributes, name_value, module_handle_value, iat_value, int_value, _, _, _ = fields
            if attributes & ~1:
                raise BinaryFormatError(
                    f"unsupported PE delay-import attributes {attributes:#x}"
                )

            def delay_rva(value: int, label: str) -> int:
                if not value:
                    return 0
                result = value if attributes & 1 else value - image_base
                if result <= 0 or result >= 1 << 32:
                    raise BinaryFormatError(f"invalid {label} {value:#x}")
                return result

            library_name_rva = delay_rva(name_value, "delay-import library RVA")
            module_handle_rva = delay_rva(
                module_handle_value, "delay-import module-handle RVA",
            )
            first_thunk_rva = delay_rva(iat_value, "delay-import IAT RVA")
            lookup_rva = delay_rva(int_value, "delay-import name table RVA")
            if not library_name_rva or not first_thunk_rva or not lookup_rva:
                raise BinaryFormatError(
                    "PE delay-import descriptor lacks library, IAT, or name table"
                )
            library = import_ascii(library_name_rva, "delay-import library")
            for symbol_index in range(65536):
                thunk_rva = lookup_rva + symbol_index * pointer_size
                thunk_offset = file_offset_for_import_rva(
                    thunk_rva, "delay-import lookup thunk",
                )
                thunk = (
                    _u64(data, thunk_offset, "PE64 delay-import lookup thunk")
                    if pe32_plus else _u32(data, thunk_offset, "PE32 delay-import lookup thunk")
                )
                if thunk == 0:
                    break
                if thunk & ordinal_mask:
                    name = None
                    ordinal = thunk & 0xFFFF
                else:
                    name_rva = thunk & (ordinal_mask - 1)
                    name_offset = file_offset_for_import_rva(
                        name_rva, "delay-import hint/name",
                    )
                    _need(data, name_offset, 2, "PE delay-import hint")
                    name = import_ascii(name_rva + 2, "delay-import symbol")
                    ordinal = None
                delay_imports.append(PEDelayImportSymbol(
                    library, name, ordinal,
                    first_thunk_rva + symbol_index * pointer_size,
                    module_handle_rva,
                ))
            else:
                raise BinaryFormatError("PE delay-import table exceeds 65536 symbols")
            descriptor_count += 1
            if descriptor_count > 4096:
                raise BinaryFormatError("PE delay-import directory exceeds 4096 descriptors")
            descriptor_offset += 32
        else:
            raise BinaryFormatError("PE delay-import directory has no terminating descriptor")

    base_relocations: list[PEBaseRelocation] = []
    if relocation_rva or relocation_size:
        if not relocation_rva or not relocation_size:
            raise BinaryFormatError("PE relocation directory has only one nonzero field")
        relocation_offset = file_offset_for_import_rva(
            relocation_rva, "base relocation directory",
        )
        relocation_end = relocation_offset + relocation_size
        relocation_section = next(
            section for section in sections
            if section.file_offset_for_rva(relocation_rva) == relocation_offset
        )
        if relocation_end > relocation_section.raw_end:
            raise BinaryFormatError(
                "PE base relocation directory crosses its raw section boundary"
            )
        _need(data, relocation_offset, relocation_size, "PE base relocation directory")
        cursor = relocation_offset
        while cursor < relocation_end:
            _need(data, cursor, 8, "PE base relocation block")
            page_rva = _u32(data, cursor, "PE base relocation page RVA")
            block_size = _u32(data, cursor + 4, "PE base relocation block size")
            if block_size < 8 or block_size % 2:
                raise BinaryFormatError(
                    f"invalid PE base relocation block size {block_size}"
                )
            if block_size > relocation_end - cursor:
                raise BinaryFormatError("PE base relocation block exceeds its directory")
            for entry_offset in range(cursor + 8, cursor + block_size, 2):
                entry = _u16(data, entry_offset, "PE base relocation entry")
                relocation_type = entry >> 12
                if relocation_type == 0:  # IMAGE_REL_BASED_ABSOLUTE padding
                    continue
                target_rva = page_rva + (entry & 0x0FFF)
                if not any(section.contains_rva(target_rva) for section in sections):
                    raise BinaryFormatError(
                        f"PE base relocation target RVA {target_rva:#x} is not mapped"
                    )
                base_relocations.append(PEBaseRelocation(
                    type=relocation_type,
                    rva=target_rva,
                ))
            cursor += block_size
        if cursor != relocation_end:
            raise BinaryFormatError("PE base relocation directory is not block aligned")

    image = PEImage(
        machine=machine,
        coff_characteristics=coff_characteristics,
        pe32_plus=pe32_plus,
        image_base=image_base,
        entrypoint_rva=entrypoint_rva,
        entrypoint_file_offset=entry_offset,
        entrypoint_section_index=entry_index,
        sections=tuple(sections),
        runtime_functions=tuple(runtime_functions),
        imports=tuple(imports),
        delay_imports=tuple(delay_imports),
        export_name=export_name,
        exports=tuple(exports),
        base_relocations=tuple(base_relocations),
        tls_directory=tls_directory,
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
    "PEBaseRelocation",
    "PEDelayImportSymbol",
    "PEExportSymbol",
    "PEMachine",
    "PEImportSymbol",
    "PERuntimeFunction",
    "PESection",
    "PESectionFlag",
    "PEStatistics",
    "PETLSDirectory",
    "PEToSSAResult",
    "PEVocabularyToken",
    "PE_EQUIVALENCE_TABLE",
    "PEImage",
    "RegisterOperand",
    "PrefixAction",
    "ReadHeadDirection",
    "ReadFailure",
    "ReadPhase",
    "ReadStatus",
    "RelativeAddressOperand",
    "VocabularyFailure",
    "VocabularyAuditReport",
    "VocabularyStatistics",
    "X86_SSA_EQUIVALENCE_TABLE",
    "SSA_PE_REVERSE_SELECTION_TABLE",
    "ReverseBinarySelection",
    "ReverseBinarySelectionPlan",
    "X86InstructionToken",
    "X86EncodingFields",
    "X86EncodingRow",
    "X86ReadBatch",
    "X86ReadHeadConfig",
    "X86ReadHeadState",
    "X86ReversibleReadHead",
    "X86ReferenceDecoder",
    "X86Register",
    "X86TensorReadHead",
    "controlled_x86_64_read_head_config",
    "controlled_x86_64_read_head_profile",
    "equivalences_targeting",
    "eligible_reverse_selections",
    "plan_reverse_selection",
    "write_reverse_selection",
    "parse_pe_image",
    "pe_runtime_function_region",
    "raise_binary_region_to_ssa",
    "raise_pe_entrypoint_to_ssa",
]
