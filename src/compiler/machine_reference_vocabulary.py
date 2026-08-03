"""Executable x86-64 byte vocabulary for the machine-lifting frontend.

The vocabulary owns instruction identity and byte decoding.  It does not use
assembly text as an intermediate representation: callers provide a bounded
binary region and receive numeric instruction tokens with structured operands
and the exact bytes consumed from that region.

The current controlled vocabulary includes instruction forms for:

* ``IMUL r32, r/m32`` (``0F AF /r``), currently register-source semantics;
* ``LEA r32, m`` (``8D /r``), with ModRM/SIB/displacement decoding;
* near ``RET`` (``C3``);
* ``SUB r64, imm8`` (``REX.W 83 /5 ib``), register destination only;
* direct near ``CALL rel32`` (``E8 cd``);
* ``ADD r64, imm8`` (``REX.W 83 /0 ib``), register destination only;
* direct near ``JMP rel32`` (``E9 cd``);
* ``MOV r/m64,r64``, ``MOV r64,r/m64``, and ``MOV r64,imm64``;
* register ``PUSH`` via the masked ``50+rd`` opcode family;
* ``AND r/m64,imm8`` (``REX.W 83 /4 ib``);
* ``CMP r64,r/m64`` and near ``JNE rel32`` basic-block termination.
* 64-bit ``LEA``, indirect ``CALL r/m64``, and 32-bit ``MOV`` zero-extension;
* 64-bit ``XOR``, register-source ``AND``, immediate ``SHL``, and ``NOT``;
* short ``JNE rel8`` and register ``POP``.

Unsupported prefixes, opcodes, addressing forms, truncation, and trailing
bytes fail closed at their original binary address.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import operator
from typing import Callable, Iterable, TypeAlias


class VocabularyDecodeError(ValueError):
    """The binary region is not completely described by this vocabulary."""


@dataclass(frozen=True, slots=True)
class VocabularyFailure:
    """One precise point where the reference vocabulary stopped being total."""

    category: str
    region_offset: int
    address: int
    encoded_preview: bytes
    reason: str


@dataclass(frozen=True, slots=True)
class DecodeReport:
    """Fail-closed decode result, including the safely decoded prefix."""

    instructions: tuple["DecodedInstruction", ...]
    failures: tuple[VocabularyFailure, ...]
    region_capacity: int
    accepted_size: int
    decoded_bytes: int
    stopped_at_return: bool
    stopped_at_control_transfer: bool = False

    @property
    def complete(self) -> bool:
        return not self.failures and self.decoded_bytes == self.accepted_size


class AuditConfidence(IntEnum):
    PROVEN_PREFIX = 0
    BYTEWISE_RESYNCHRONIZATION_CANDIDATE = 1


@dataclass(frozen=True, slots=True)
class VocabularyAuditReport:
    """Whole-region byte coverage without claiming unknown x86 boundaries."""

    accepted_size: int
    candidate_instructions: tuple["DecodedInstruction", ...]
    gap_failures: tuple[VocabularyFailure, ...]
    known_bytes: int
    missing_bytes: int
    signature_counts: tuple[tuple[bytes, int], ...]
    confidence: AuditConfidence = AuditConfidence.BYTEWISE_RESYNCHRONIZATION_CANDIDATE

    @property
    def complete(self) -> bool:
        return self.missing_bytes == 0


class X86InstructionToken(IntEnum):
    """Append-only identities for instruction forms, not mnemonic strings."""

    IMUL_R32_RM32 = 0
    LEA_R32_M = 1
    RET_NEAR = 2
    SUB_R64_IMM8 = 3
    CALL_REL32 = 4
    ADD_R64_IMM8 = 5
    JMP_REL32 = 6
    MOV_RM64_R64 = 7
    PUSH_R64 = 8
    MOV_R64_RM64 = 9
    AND_RM64_IMM8 = 10
    MOV_R64_IMM64 = 11
    CMP_R64_RM64 = 12
    JNE_REL32 = 13
    LEA_R64_M = 14
    CALL_RM64 = 15
    MOV_R32_RM32 = 16
    XOR_RM64_R64 = 17
    XOR_R64_RM64 = 18
    SHL_R64_IMM8 = 19
    AND_R64_RM64 = 20
    JNE_REL8 = 21
    NOT_RM64 = 22
    POP_R64 = 23
    NOP_RM = 24
    INT3 = 25
    NOP = 26
    SUB_R64_IMM32 = 27
    XOR_R32_RM32 = 28
    TEST_RM32_R32 = 29
    TEST_RM64_R64 = 30
    JE_REL8 = 31
    JE_REL32 = 32
    JS_REL32 = 33
    MOV_R32_IMM32 = 34
    NEG_RM8 = 35
    SBB_R64_RM64 = 36
    CMOVNE_R64_RM64 = 37
    MOV_RM32_R32 = 38
    INC_RM64 = 39
    OR_RM64_IMM8 = 40
    MOV_RM32_IMM32 = 41
    MOV_RM8_IMM8 = 42
    SUB_R64_RM64 = 43
    MOV_RM8_R8 = 44
    CMP_RM64_IMM8 = 45
    CMP_RM32_IMM8 = 46
    CMP_RAX_IMM32 = 47
    CMP_EAX_IMM32 = 48
    TEST_RM8_IMM8 = 49
    TEST_RM8_R8 = 50
    MOVSXD_R64_RM32 = 51
    MOV_R8_RM8 = 52
    CMOVNE_R32_RM32 = 53
    MOV_RM64_IMM32 = 54
    MOVZX_R32_RM16 = 55
    JMP_REL8 = 56
    AND_RM32_IMM8 = 57
    ADD_R64_IMM32 = 58
    AND_RM32_IMM32 = 59
    TEST_RM16_R16 = 60
    CMP_RM16_R16 = 61
    CMP_RM16_IMM8 = 62
    MOV_RM16_R16 = 63
    JS_REL8 = 64
    CMP_RM8_R8 = 65
    SUB_RM32_IMM8 = 66
    INC_RM32 = 67
    BTR_RM32_IMM8 = 68
    CMP_RM32_R32 = 69
    ADD_R64_RM64 = 70
    TEST_EAX_IMM32 = 71
    OR_RM32_R32 = 72
    JAE_REL32 = 73
    MOVSX_R32_RM16 = 74
    CMP_R32_RM32 = 75
    NEG_RM32 = 76
    JA_REL8 = 77
    JBE_REL32 = 78
    JAE_REL8 = 79
    MOVZX_R32_RM8 = 80
    OR_RM32_IMM8 = 81
    SETE_RM8 = 82
    CMP_R16_RM16 = 83
    JA_REL32 = 84
    SBB_R32_RM32 = 85
    CMP_RM64_R64 = 86
    SETB_RM8 = 87
    SUB_R32_RM32 = 88
    TEST_AL_IMM8 = 89
    CMP_RM32_IMM32 = 90
    JBE_REL8 = 91
    SHR_RM32_1 = 92
    TEST_RM32_IMM32 = 93
    SAR_RM64_1 = 94
    AND_R32_RM32 = 95
    ADD_RM64_R64 = 96
    ADD_RM32_R32 = 97
    DIV_RM32 = 98
    BT_RM32_IMM8 = 99
    SHR_RM32_IMM8 = 100
    JGE_REL8 = 101
    DEC_RM32 = 102
    JMP_RM64 = 103
    CMOVE_R64_RM64 = 104
    JB_REL32 = 105
    ADD_RM32_IMM8 = 106
    SUB_R16_RM16 = 107
    NEG_RM16 = 108
    JG_REL32 = 109
    CMP_RM8_IMM8 = 110
    SUB_RM16_IMM8 = 111
    ADD_R32_RM32 = 112
    JB_REL8 = 113
    SHR_RM64_1 = 114
    JGE_REL32 = 115
    NEG_RM64 = 116
    CMP_AL_IMM8 = 117
    SETNE_RM8 = 118
    AND_RM8_IMM8 = 119
    BTS_RM32_IMM8 = 120
    JLE_REL8 = 121
    OR_R32_RM32 = 122
    CMOVE_R32_RM32 = 123
    MOV_R8_IMM8 = 124
    CMOVA_R32_RM32 = 125
    JG_REL8 = 126
    AND_EAX_IMM32 = 127
    XOR_RM32_IMM8 = 128
    ADD_RM32_IMM32 = 129
    CDQE = 130
    XOR_R8_RM8 = 131
    BT_RM64_R64 = 132
    MUL_RM64 = 133
    DEC_RM64 = 134
    NOP_66 = 135
    CMP_RM64_IMM32 = 136
    JL_REL8 = 137
    SHL_RM32_CL = 138
    XORPS_XMM_XMMM128 = 139
    MOVUPS_XMM_XMMM128 = 140
    MOVUPS_XMMM128_XMM = 141
    MOVDQU_XMM_XMMM128 = 142
    MOVDQU_XMMM128_XMM = 143
    MOVDQA_XMMM128_XMM = 144
    AND_AL_IMM8 = 145
    OR_RM16_IMM8 = 146
    JLE_REL32 = 147
    CMP_R8_RM8 = 148
    JL_REL32 = 149
    REP_STOSW = 150
    CMOVO_R64_RM64 = 151
    NOP_RM_66 = 152
    JNS_REL8 = 153
    BTS_RM32_R32 = 154
    SETA_RM8 = 155
    MUL_RM32 = 156
    AND_RM64_R64 = 157
    NOT_RM32 = 158
    AND_R16_RM16 = 159
    MOVDQA_XMM_XMMM128 = 160
    MOV_R64_RM64_FS = 161
    ROL_RM64_IMM8 = 162
    IMUL_R64_RM64_IMM8 = 163
    MOVAPS_XMMM128_XMM = 164
    MOVAPS_XMM_XMMM128 = 165
    BT_RM32_R32 = 166
    SETG_RM8 = 167
    ADD_R16_RM16 = 168
    TEST_RM16_IMM16 = 169
    CMOVE_R16_RM16 = 170
    ROL_RM8_IMM8 = 171
    SCASB = 172
    PSRLDQ_XMM_IMM8 = 173
    ADD_RM16_IMM8 = 174
    SETLE_RM8 = 175
    CMPXCHG_RM64_R64 = 176
    ROR_RM64_IMM8 = 177
    CMOVAE_R64_RM64 = 178
    CMOVB_R64_RM64 = 179
    CMOVB_R16_RM16 = 180
    CMOVB_R32_RM32 = 181
    BTR_RM32_R32 = 182
    OR_RM32_IMM32 = 183
    SHL_RM16_IMM8 = 184
    SHL_RM32_IMM8 = 185
    CMOVBE_R32_RM32 = 186
    REP_MOVSQ = 187
    SETNS_RM8 = 188
    MOVSX_R64_RM16 = 189
    CMOVLE_R32_RM32 = 190
    AND_RM32_R32 = 191
    IMUL_R32_RM32_IMM8 = 192
    SHR_RM64_IMM8 = 193
    OR_R8_RM8 = 194
    XADD_RM32_R32 = 195
    OR_EAX_IMM32 = 196
    AND_R8_RM8 = 197
    LOCK_ADD_RM8_R8 = 198
    MOVQ_RM64_XMM = 199
    XCHG_RM64_R64 = 200
    AND_RM16_IMM16 = 201
    BTC_RM32_IMM8 = 202
    DIV_RM64 = 203
    OR_RM16_R16 = 204
    LOCK_DEC_RM32 = 205
    MOVSD_XMM_XMMM64 = 206
    MOVSD_XMMM64_XMM = 207
    ADD_RAX_IMM32 = 208
    OR_R64_RM64 = 209
    INT_IMM8 = 210
    INC_RM16 = 211
    SHR_RM8_IMM8 = 212
    CQO = 213
    IDIV_RM64 = 214
    CMOVAE_R32_RM32 = 215
    CMOVL_R32_RM32 = 216
    XOR_RM32_R32 = 217
    SHL_RM64_CL = 218


class MachineSemanticToken(IntEnum):
    """Machine-state transformations selected by decoded instruction forms."""

    INTEGER_MULTIPLY = 0
    EFFECTIVE_ADDRESS = 1
    RETURN = 2
    INTEGER_SUBTRACT = 3
    DIRECT_RELATIVE_CALL = 4
    INTEGER_ADD = 5
    DIRECT_RELATIVE_JUMP = 6
    REGISTER_OR_MEMORY_WRITE = 7
    STACK_PUSH = 8
    REGISTER_OR_MEMORY_READ = 9
    BITWISE_AND = 10
    REGISTER_WRITE_IMMEDIATE = 11
    INTEGER_COMPARE = 12
    CONDITIONAL_RELATIVE_JUMP = 13
    INDIRECT_CALL = 14
    BITWISE_XOR = 15
    SHIFT_LEFT = 16
    BITWISE_NOT = 17
    STACK_POP = 18
    NO_OPERATION = 19
    BREAKPOINT_TRAP = 20
    INTEGER_TEST = 21
    INTEGER_NEGATE = 22
    INTEGER_SUBTRACT_WITH_BORROW = 23
    CONDITIONAL_MOVE = 24
    INTEGER_INCREMENT = 25
    BITWISE_OR = 26
    SIGN_EXTEND = 27
    ZERO_EXTEND = 28
    BIT_TEST_RESET = 29
    CONDITIONAL_SET = 30
    SHIFT_RIGHT_LOGICAL = 31
    SHIFT_RIGHT_ARITHMETIC = 32
    INTEGER_DIVIDE = 33
    BIT_TEST = 34
    INTEGER_DECREMENT = 35
    INDIRECT_JUMP = 36
    INTEGER_MULTIPLY_UNSIGNED = 37
    SIGN_EXTEND_ACCUMULATOR = 38
    VECTOR_XOR = 39
    VECTOR_MOVE = 40
    STRING_STORE = 41
    ROTATE_LEFT = 42
    STRING_COMPARE = 43
    VECTOR_SHIFT_RIGHT_LOGICAL = 44
    ATOMIC_COMPARE_EXCHANGE = 45
    ROTATE_RIGHT = 46
    STRING_MOVE = 47
    ATOMIC_EXCHANGE_ADD = 48
    ATOMIC_ADD = 49
    EXCHANGE = 50
    BIT_TEST_COMPLEMENT = 51
    SOFTWARE_INTERRUPT = 52
    INTEGER_DIVIDE_SIGNED = 53


class X86Register(IntEnum):
    RAX = 0
    RCX = 1
    RDX = 2
    RBX = 3
    RSP = 4
    RBP = 5
    RSI = 6
    RDI = 7
    R8 = 8
    R9 = 9
    R10 = 10
    R11 = 11
    R12 = 12
    R13 = 13
    R14 = 14
    R15 = 15


class X86HighByteRegister(IntEnum):
    AH = 0
    CH = 1
    DH = 2
    BH = 3


class X86VectorRegister(IntEnum):
    XMM0 = 0
    XMM1 = 1
    XMM2 = 2
    XMM3 = 3
    XMM4 = 4
    XMM5 = 5
    XMM6 = 6
    XMM7 = 7
    XMM8 = 8
    XMM9 = 9
    XMM10 = 10
    XMM11 = 11
    XMM12 = 12
    XMM13 = 13
    XMM14 = 14
    XMM15 = 15


@dataclass(frozen=True, slots=True)
class RegisterOperand:
    register: X86Register
    width: int


@dataclass(frozen=True, slots=True)
class HighByteRegisterOperand:
    register: X86HighByteRegister
    width: int = 8


@dataclass(frozen=True, slots=True)
class VectorRegisterOperand:
    register: X86VectorRegister
    width: int = 128


@dataclass(frozen=True, slots=True)
class EffectiveAddressOperand:
    """One decoded 64-bit addressing expression.

    ``base + index * scale + displacement`` is represented without rendering
    or simplifying it. ``base=None`` covers displacement-only/RIP-relative
    encodings; ``rip_relative`` distinguishes those two meanings.
    """

    base: X86Register | None
    index: X86Register | None
    scale: int
    displacement: int
    address_width: int = 64
    rip_relative: bool = False


@dataclass(frozen=True, slots=True)
class ImmediateOperand:
    value: int
    width: int
    signed: bool


@dataclass(frozen=True, slots=True)
class RelativeAddressOperand:
    """Signed next-instruction-relative displacement and resolved address."""

    displacement: int
    width: int
    target_address: int


MachineOperand: TypeAlias = (
    RegisterOperand
    | HighByteRegisterOperand
    | VectorRegisterOperand
    | EffectiveAddressOperand
    | ImmediateOperand
    | RelativeAddressOperand
)


@dataclass(frozen=True, slots=True)
class DecodedInstruction:
    address: int
    token: X86InstructionToken
    semantic: MachineSemanticToken
    operands: tuple[MachineOperand, ...]
    encoded: bytes
    rex: int | None = None
    legacy_prefixes: tuple[int, ...] = ()


OperandDecoder: TypeAlias = Callable[
    [memoryview, int, int, int | None],
    tuple[tuple[MachineOperand, ...], int],
]


@dataclass(frozen=True, slots=True)
class InstructionSpec:
    token: X86InstructionToken
    semantic: MachineSemanticToken
    opcode: bytes
    decode_operands: OperandDecoder
    opcode_mask: bytes | None = None
    modrm_extension: int | None = None
    allow_rex: bool = True
    allow_rex_w: bool = False
    require_rex_w: bool = False
    allowed_legacy_prefixes: frozenset[int] = frozenset()
    required_legacy_prefixes: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class _ModRM:
    reg: RegisterOperand
    rm: MachineOperand
    next_offset: int
    mod: int


def _need(region: memoryview, offset: int, count: int, address: int) -> None:
    if offset < 0 or count < 0 or offset + count > len(region):
        raise VocabularyDecodeError(
            f"{address:#x}: truncated instruction; need {count} byte(s) "
            f"at region offset {offset}"
        )


def _signed(region: memoryview, offset: int, width: int, address: int) -> int:
    _need(region, offset, width, address)
    return int.from_bytes(region[offset:offset + width], "little", signed=True)


def _gpr(code: int) -> X86Register:
    try:
        return X86Register(int(code))
    except ValueError as error:
        raise VocabularyDecodeError(f"invalid x86 register code {code}") from error


def _xmm(code: int) -> X86VectorRegister:
    try:
        return X86VectorRegister(int(code))
    except ValueError as error:
        raise VocabularyDecodeError(f"invalid x86 vector register code {code}") from error


def _decode_modrm(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
    *,
    register_width: int = 32,
) -> _ModRM:
    """Decode ModRM/SIB once for every instruction form that uses ``/r``."""

    _need(region, offset, 1, address)
    modrm = int(region[offset])
    offset += 1
    mod = modrm >> 6
    reg_code = ((modrm >> 3) & 0x7) | (0x8 if rex and rex & 0x4 else 0)
    rm_low = modrm & 0x7
    rm_code = rm_low | (0x8 if rex and rex & 0x1 else 0)
    reg = RegisterOperand(_gpr(reg_code), register_width)

    if mod == 0x3:
        return _ModRM(
            reg,
            RegisterOperand(_gpr(rm_code), register_width),
            offset,
            mod,
        )

    base: X86Register | None
    index: X86Register | None = None
    scale = 1
    displacement = 0
    rip_relative = False

    if rm_low == 0x4:
        _need(region, offset, 1, address)
        sib = int(region[offset])
        offset += 1
        scale = 1 << (sib >> 6)
        index_low = (sib >> 3) & 0x7
        base_low = sib & 0x7
        index_code = index_low | (0x8 if rex and rex & 0x2 else 0)
        base_code = base_low | (0x8 if rex and rex & 0x1 else 0)
        # Index code 4 without REX.X means the SIB has no index term.
        index = None if index_low == 0x4 and not (rex and rex & 0x2) else _gpr(index_code)
        if mod == 0 and base_low == 0x5:
            base = None
            displacement = _signed(region, offset, 4, address)
            offset += 4
        else:
            base = _gpr(base_code)
    elif mod == 0 and rm_low == 0x5:
        # In 64-bit addressing mode this is RIP + disp32.
        base = None
        rip_relative = True
        displacement = _signed(region, offset, 4, address)
        offset += 4
    else:
        base = _gpr(rm_code)

    if mod == 0x1:
        displacement = _signed(region, offset, 1, address)
        offset += 1
    elif mod == 0x2:
        displacement = _signed(region, offset, 4, address)
        offset += 4

    return _ModRM(
        reg,
        EffectiveAddressOperand(
            base=base,
            index=index,
            scale=scale,
            displacement=displacement,
            rip_relative=rip_relative,
        ),
        offset,
        mod,
    )


def _decode_imul_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex)
    if not isinstance(decoded.rm, RegisterOperand):
        raise VocabularyDecodeError(
            f"{address:#x}: IMUL memory-source decoding exists, but memory-state "
            "semantics are not in the current lifting vocabulary"
        )
    return (decoded.reg, decoded.rm), decoded.next_offset


def _decode_lea_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex)
    if not isinstance(decoded.rm, EffectiveAddressOperand):
        raise VocabularyDecodeError(
            f"{address:#x}: LEA requires a memory addressing form, not a register"
        )
    return (decoded.reg, decoded.rm), decoded.next_offset


def _decode_lea_r64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    if not isinstance(decoded.rm, EffectiveAddressOperand):
        raise VocabularyDecodeError(
            f"{address:#x}: LEA requires a memory addressing form, not a register"
        )
    return (decoded.reg, decoded.rm), decoded.next_offset


def _decode_mov_rm64_r64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    # Opcode 89 encodes r64 in ModRM.reg and the destination in ModRM.r/m.
    return (decoded.rm, decoded.reg), decoded.next_offset


def _decode_mov_r64_rm64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    # Opcode 8B encodes the destination in ModRM.reg and source in ModRM.r/m.
    return (decoded.reg, decoded.rm), decoded.next_offset


def _decode_mov_r32_rm32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    return (decoded.reg, decoded.rm), decoded.next_offset


def _decode_binary_rm64_r64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    return (decoded.rm, decoded.reg), decoded.next_offset


def _decode_binary_r64_rm64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    return (decoded.reg, decoded.rm), decoded.next_offset


def _decode_binary_r32_rm32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    return (decoded.reg, decoded.rm), decoded.next_offset


def _decode_vector_binary_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
    *,
    destination_in_reg: bool,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=128)
    vector_reg = VectorRegisterOperand(_xmm(int(decoded.reg.register)))
    vector_rm: MachineOperand
    if isinstance(decoded.rm, RegisterOperand):
        vector_rm = VectorRegisterOperand(_xmm(int(decoded.rm.register)))
    else:
        vector_rm = decoded.rm
    if destination_in_reg:
        return (vector_reg, vector_rm), decoded.next_offset
    return (vector_rm, vector_reg), decoded.next_offset


def _decode_vector_reg_rm_operands(region, offset, address, rex):
    return _decode_vector_binary_operands(
        region, offset, address, rex, destination_in_reg=True,
    )


def _decode_vector_rm_reg_operands(region, offset, address, rex):
    return _decode_vector_binary_operands(
        region, offset, address, rex, destination_in_reg=False,
    )


def _decode_binary_rm32_r32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    return (decoded.rm, decoded.reg), decoded.next_offset


def _decode_binary_rm16_r16_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=16)
    return (decoded.rm, decoded.reg), decoded.next_offset


def _decode_binary_r16_rm16_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=16)
    return (decoded.reg, decoded.rm), decoded.next_offset


def _decode_binary_rm8_r8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    modrm = int(region[offset])
    decoded = _decode_modrm(region, offset, address, rex, register_width=8)
    if rex is None and ((modrm >> 3) & 0x7) >= 4:
        raise VocabularyDecodeError(
            f"{address:#x}: legacy high-byte source needs a distinct register token"
        )
    if rex is None and decoded.mod == 3 and (modrm & 0x7) >= 4:
        raise VocabularyDecodeError(
            f"{address:#x}: legacy high-byte destination needs a distinct register token"
        )
    return (decoded.rm, decoded.reg), decoded.next_offset


def _decode_binary_r8_rm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _decode_binary_rm8_r8_operands(region, offset, address, rex)
    decoded = _decode_modrm(region, offset, address, rex, register_width=8)
    return (decoded.reg, decoded.rm), decoded.next_offset


def _decode_nop_rm_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    return (), decoded.next_offset


def _decode_no_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    return (), offset


def _decode_push_r64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset - 1, 1, address)
    opcode = int(region[offset - 1])
    register_code = (opcode & 0x7) | (0x8 if rex and rex & 0x1 else 0)
    return (RegisterOperand(_gpr(register_code), width=64),), offset


def _decode_pop_r64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset - 1, 1, address)
    opcode = int(region[offset - 1])
    register_code = (opcode & 0x7) | (0x8 if rex and rex & 0x1 else 0)
    return (RegisterOperand(_gpr(register_code), width=64),), offset


def _decode_sub_r64_imm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    modrm = int(region[offset])
    extension = (modrm >> 3) & 0x7
    if extension != 0x5:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode 83 requires ModRM /5 for SUB, received /{extension}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of the group-1 /5 opcode extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    if not isinstance(decoded.rm, RegisterOperand):
        raise VocabularyDecodeError(
            f"{address:#x}: SUB memory-destination state is outside this token"
        )
    immediate = _signed(region, decoded.next_offset, 1, address)
    return (
        decoded.rm,
        ImmediateOperand(immediate, width=8, signed=True),
    ), decoded.next_offset + 1


def _decode_add_r64_imm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    modrm = int(region[offset])
    extension = (modrm >> 3) & 0x7
    if extension != 0x0:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode 83 requires ModRM /0 for ADD, received /{extension}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of the group-1 /0 opcode extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    immediate = _signed(region, decoded.next_offset, 1, address)
    return (
        decoded.rm,
        ImmediateOperand(immediate, width=8, signed=True),
    ), decoded.next_offset + 1


def _decode_and_rm64_imm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    extension = (int(region[offset]) >> 3) & 0x7
    if extension != 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode 83 requires ModRM /4 for AND, received /{extension}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of the group-1 /4 opcode extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    immediate = _signed(region, decoded.next_offset, 1, address)
    return (
        decoded.rm,
        ImmediateOperand(immediate, width=8, signed=True),
    ), decoded.next_offset + 1


def _decode_sub_r64_imm32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    extension = (int(region[offset]) >> 3) & 0x7
    if extension != 5:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode 81 requires ModRM /5 for SUB, received /{extension}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of the group-1 /5 opcode extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    if not isinstance(decoded.rm, RegisterOperand):
        raise VocabularyDecodeError(
            f"{address:#x}: SUB imm32 memory destination is outside this token"
        )
    immediate = _signed(region, decoded.next_offset, 4, address)
    return (
        decoded.rm,
        ImmediateOperand(immediate, width=32, signed=True),
    ), decoded.next_offset + 4


def _decode_mov_r32_imm32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset - 1, 1, address)
    opcode = int(region[offset - 1])
    register_code = (opcode & 0x7) | (0x8 if rex and rex & 0x1 else 0)
    _need(region, offset, 4, address)
    immediate = int.from_bytes(region[offset:offset + 4], "little", signed=False)
    return (
        RegisterOperand(_gpr(register_code), width=32),
        ImmediateOperand(immediate, width=32, signed=False),
    ), offset + 4


def _decode_mov_r8_imm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset - 1, 1, address)
    opcode_code = int(region[offset - 1]) & 0x7
    _need(region, offset, 1, address)
    if rex is None and opcode_code >= 4:
        destination: MachineOperand = HighByteRegisterOperand(
            X86HighByteRegister(opcode_code - 4),
        )
    else:
        register_code = opcode_code | (0x8 if rex and rex & 0x1 else 0)
        destination = RegisterOperand(_gpr(register_code), width=8)
    return (
        destination,
        ImmediateOperand(int(region[offset]), width=8, signed=False),
    ), offset + 1


def _decode_group_rm_imm(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
    *,
    extension: int,
    operand_width: int,
    immediate_width: int,
    signed: bool,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    received = (int(region[offset]) >> 3) & 0x7
    if received != extension:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode group requires ModRM /{extension}, received /{received}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of an opcode-group extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=operand_width)
    if (
        operand_width == 8
        and rex is None
        and isinstance(decoded.rm, RegisterOperand)
        and (int(region[offset]) & 0x7) >= 4
    ):
        raise VocabularyDecodeError(
            f"{address:#x}: legacy high-byte destination needs a distinct register token"
        )
    _need(region, decoded.next_offset, immediate_width, address)
    immediate = int.from_bytes(
        region[decoded.next_offset:decoded.next_offset + immediate_width],
        "little",
        signed=signed,
    )
    return (
        decoded.rm,
        ImmediateOperand(immediate, width=immediate_width * 8, signed=signed),
    ), decoded.next_offset + immediate_width


def _decode_or_rm64_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=1, operand_width=64, immediate_width=1, signed=True,
    )


def _decode_mov_rm32_imm32_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=0, operand_width=32, immediate_width=4, signed=False,
    )


def _decode_mov_rm8_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=0, operand_width=8, immediate_width=1, signed=False,
    )


def _decode_cmp_rm64_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=7, operand_width=64, immediate_width=1, signed=True,
    )


def _decode_cmp_rm32_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=7, operand_width=32, immediate_width=1, signed=True,
    )


def _decode_test_rm8_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=0, operand_width=8, immediate_width=1, signed=False,
    )


def _decode_accumulator_imm32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
    *,
    width: int,
) -> tuple[tuple[MachineOperand, ...], int]:
    immediate = _signed(region, offset, 4, address)
    return (
        RegisterOperand(X86Register.RAX, width=width),
        ImmediateOperand(immediate, width=32, signed=True),
    ), offset + 4


def _decode_cmp_rax_imm32_operands(region, offset, address, rex):
    return _decode_accumulator_imm32_operands(
        region, offset, address, rex, width=64,
    )


def _decode_cmp_eax_imm32_operands(region, offset, address, rex):
    return _decode_accumulator_imm32_operands(
        region, offset, address, rex, width=32,
    )


def _decode_test_eax_imm32_operands(region, offset, address, rex):
    return _decode_accumulator_imm32_operands(
        region, offset, address, rex, width=32,
    )


def _decode_test_al_imm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    return (
        RegisterOperand(X86Register.RAX, width=8),
        ImmediateOperand(int(region[offset]), width=8, signed=False),
    ), offset + 1


def _decode_cmp_al_imm8_operands(region, offset, address, rex):
    return _decode_test_al_imm8_operands(region, offset, address, rex)


def _decode_and_al_imm8_operands(region, offset, address, rex):
    return _decode_test_al_imm8_operands(region, offset, address, rex)


def _decode_and_eax_imm32_operands(region, offset, address, rex):
    return _decode_accumulator_imm32_operands(
        region, offset, address, rex, width=32,
    )


def _decode_or_eax_imm32_operands(region, offset, address, rex):
    return _decode_accumulator_imm32_operands(
        region, offset, address, rex, width=32,
    )


def _decode_add_rax_imm32_operands(region, offset, address, rex):
    return _decode_accumulator_imm32_operands(
        region, offset, address, rex, width=64,
    )


def _decode_movsxd_r64_rm32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    source: MachineOperand = decoded.rm
    if isinstance(source, RegisterOperand):
        source = RegisterOperand(source.register, width=32)
    return (decoded.reg, source), decoded.next_offset


def _decode_movzx_r32_rm16_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    source: MachineOperand = decoded.rm
    if isinstance(source, RegisterOperand):
        source = RegisterOperand(source.register, width=16)
    return (decoded.reg, source), decoded.next_offset


def _decode_movsx_r32_rm16_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    source: MachineOperand = decoded.rm
    if isinstance(source, RegisterOperand):
        source = RegisterOperand(source.register, width=16)
    return (decoded.reg, source), decoded.next_offset


def _decode_movsx_r64_rm16_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    source: MachineOperand = decoded.rm
    if isinstance(source, RegisterOperand):
        source = RegisterOperand(source.register, width=16)
    return (decoded.reg, source), decoded.next_offset


def _decode_movzx_r32_rm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    modrm = int(region[offset])
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    source: MachineOperand = decoded.rm
    if isinstance(source, RegisterOperand):
        if rex is None and (modrm & 0x7) >= 4:
            raise VocabularyDecodeError(
                f"{address:#x}: legacy high-byte source needs a distinct register token"
            )
        source = RegisterOperand(source.register, width=8)
    return (decoded.reg, source), decoded.next_offset


def _decode_mov_rm64_imm32_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=0, operand_width=64, immediate_width=4, signed=True,
    )


def _decode_and_rm32_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=4, operand_width=32, immediate_width=1, signed=True,
    )


def _decode_add_r64_imm32_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=0, operand_width=64, immediate_width=4, signed=True,
    )


def _decode_and_rm32_imm32_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=4, operand_width=32, immediate_width=4, signed=False,
    )


def _decode_or_rm32_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=1, operand_width=32, immediate_width=1, signed=True,
    )


def _decode_add_rm32_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=0, operand_width=32, immediate_width=1, signed=True,
    )


def _decode_add_rm16_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=0, operand_width=16, immediate_width=1, signed=True,
    )


def _decode_cmp_rm8_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=7, operand_width=8, immediate_width=1, signed=True,
    )


def _decode_sub_rm16_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=5, operand_width=16, immediate_width=1, signed=True,
    )


def _decode_and_rm8_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=4, operand_width=8, immediate_width=1, signed=False,
    )


def _decode_or_rm16_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=1, operand_width=16, immediate_width=1, signed=True,
    )


def _decode_test_rm16_imm16_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=0, operand_width=16, immediate_width=2, signed=False,
    )


def _decode_and_rm16_imm16_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=4, operand_width=16, immediate_width=2, signed=False,
    )


def _decode_cmp_rm32_imm32_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=7, operand_width=32, immediate_width=4, signed=False,
    )


def _decode_cmp_rm64_imm32_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=7, operand_width=64, immediate_width=4, signed=True,
    )


def _decode_add_rm32_imm32_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=0, operand_width=32, immediate_width=4, signed=False,
    )


def _decode_or_rm32_imm32_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=1, operand_width=32, immediate_width=4, signed=False,
    )


def _decode_xor_rm32_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=6, operand_width=32, immediate_width=1, signed=True,
    )


def _decode_test_rm32_imm32_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=0, operand_width=32, immediate_width=4, signed=False,
    )


def _decode_cmp_rm16_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=7, operand_width=16, immediate_width=1, signed=True,
    )


def _decode_sub_rm32_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=5, operand_width=32, immediate_width=1, signed=True,
    )


def _decode_btr_rm32_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=6, operand_width=32, immediate_width=1, signed=False,
    )


def _decode_bt_rm32_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=4, operand_width=32, immediate_width=1, signed=False,
    )


def _decode_bts_rm32_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=5, operand_width=32, immediate_width=1, signed=False,
    )


def _decode_btr_rm32_r32_operands(region, offset, address, rex):
    return _decode_bt_rm32_r32_operands(region, offset, address, rex)


def _decode_btc_rm32_imm8_operands(region, offset, address, rex):
    return _decode_group_rm_imm(
        region, offset, address, rex,
        extension=7, operand_width=32, immediate_width=1, signed=False,
    )


def _decode_bt_rm64_r64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    return (decoded.rm, decoded.reg), decoded.next_offset


def _decode_bt_rm32_r32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    return (decoded.rm, decoded.reg), decoded.next_offset


def _decode_shift_group(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
    *,
    extension: int,
    operand_width: int,
    immediate: bool,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    received = (int(region[offset]) >> 3) & 0x7
    if received != extension:
        raise VocabularyDecodeError(
            f"{address:#x}: shift group requires ModRM /{extension}, received /{received}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of a shift opcode-group extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=operand_width)
    if immediate:
        _need(region, decoded.next_offset, 1, address)
        count = int(region[decoded.next_offset])
        end = decoded.next_offset + 1
    else:
        count = 1
        end = decoded.next_offset
    return (
        decoded.rm,
        ImmediateOperand(count, width=8, signed=False),
    ), end


def _decode_shr_rm32_1_operands(region, offset, address, rex):
    return _decode_shift_group(
        region, offset, address, rex,
        extension=5, operand_width=32, immediate=False,
    )


def _decode_sar_rm64_1_operands(region, offset, address, rex):
    return _decode_shift_group(
        region, offset, address, rex,
        extension=7, operand_width=64, immediate=False,
    )


def _decode_shr_rm32_imm8_operands(region, offset, address, rex):
    return _decode_shift_group(
        region, offset, address, rex,
        extension=5, operand_width=32, immediate=True,
    )


def _decode_shr_rm64_1_operands(region, offset, address, rex):
    return _decode_shift_group(
        region, offset, address, rex,
        extension=5, operand_width=64, immediate=False,
    )


def _decode_shl_rm32_cl_operands(region, offset, address, rex):
    operands, end = _decode_shift_group(
        region, offset, address, rex,
        extension=4, operand_width=32, immediate=False,
    )
    return (
        operands[0],
        RegisterOperand(X86Register.RCX, width=8),
    ), end


def _decode_shl_rm64_cl_operands(region, offset, address, rex):
    operands, end = _decode_shift_group(
        region, offset, address, rex,
        extension=4, operand_width=64, immediate=False,
    )
    return (
        operands[0],
        RegisterOperand(X86Register.RCX, width=8),
    ), end


def _decode_rol_rm64_imm8_operands(region, offset, address, rex):
    return _decode_shift_group(
        region, offset, address, rex,
        extension=0, operand_width=64, immediate=True,
    )


def _decode_rol_rm8_imm8_operands(region, offset, address, rex):
    operands, end = _decode_shift_group(
        region, offset, address, rex,
        extension=0, operand_width=8, immediate=True,
    )
    if (
        rex is None
        and isinstance(operands[0], RegisterOperand)
        and (int(region[offset]) & 0x7) >= 4
    ):
        raise VocabularyDecodeError(
            f"{address:#x}: legacy high-byte destination needs a distinct register token"
        )
    return operands, end


def _decode_shr_rm8_imm8_operands(region, offset, address, rex):
    operands, end = _decode_shift_group(
        region, offset, address, rex,
        extension=5, operand_width=8, immediate=True,
    )
    if (
        rex is None
        and isinstance(operands[0], RegisterOperand)
        and (int(region[offset]) & 0x7) >= 4
    ):
        raise VocabularyDecodeError(
            f"{address:#x}: legacy high-byte destination needs a distinct register token"
        )
    return operands, end


def _decode_ror_rm64_imm8_operands(region, offset, address, rex):
    return _decode_shift_group(
        region, offset, address, rex,
        extension=1, operand_width=64, immediate=True,
    )


def _decode_shl_rm16_imm8_operands(region, offset, address, rex):
    return _decode_shift_group(
        region, offset, address, rex,
        extension=4, operand_width=16, immediate=True,
    )


def _decode_shl_rm32_imm8_operands(region, offset, address, rex):
    return _decode_shift_group(
        region, offset, address, rex,
        extension=4, operand_width=32, immediate=True,
    )


def _decode_shr_rm64_imm8_operands(region, offset, address, rex):
    return _decode_shift_group(
        region, offset, address, rex,
        extension=5, operand_width=64, immediate=True,
    )


def _decode_unary_group_rm(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
    *,
    extension: int,
    operand_width: int,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    received = (int(region[offset]) >> 3) & 0x7
    if received != extension:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode group requires ModRM /{extension}, received /{received}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of an opcode-group extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=operand_width)
    return (decoded.rm,), decoded.next_offset


def _decode_div_rm32_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=6, operand_width=32,
    )


def _decode_neg_rm16_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=3, operand_width=16,
    )


def _decode_neg_rm64_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=3, operand_width=64,
    )


def _decode_mul_rm64_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=4, operand_width=64,
    )


def _decode_mul_rm32_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=4, operand_width=32,
    )


def _decode_div_rm64_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=6, operand_width=64,
    )


def _decode_idiv_rm64_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=7, operand_width=64,
    )


def _decode_not_rm32_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=2, operand_width=32,
    )


def _decode_dec_rm64_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=1, operand_width=64,
    )


def _decode_inc_rm16_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=0, operand_width=16,
    )


def _decode_cdqe_operands(region, offset, address, rex):
    return (
        RegisterOperand(X86Register.RAX, width=64),
        RegisterOperand(X86Register.RAX, width=32),
    ), offset


def _decode_cqo_operands(region, offset, address, rex):
    return (
        RegisterOperand(X86Register.RDX, width=64),
        RegisterOperand(X86Register.RAX, width=64),
    ), offset


def _decode_int_imm8_operands(region, offset, address, rex):
    _need(region, offset, 1, address)
    return (ImmediateOperand(int(region[offset]), width=8, signed=False),), offset + 1


def _decode_rep_stosw_operands(region, offset, address, rex):
    return (
        RegisterOperand(X86Register.RDI, width=64),
        RegisterOperand(X86Register.RAX, width=16),
        RegisterOperand(X86Register.RCX, width=64),
    ), offset


def _decode_scasb_operands(region, offset, address, rex):
    return (
        RegisterOperand(X86Register.RDI, width=64),
        RegisterOperand(X86Register.RAX, width=8),
    ), offset


def _decode_rep_movsq_operands(region, offset, address, rex):
    return (
        RegisterOperand(X86Register.RDI, width=64),
        RegisterOperand(X86Register.RSI, width=64),
        RegisterOperand(X86Register.RCX, width=64),
    ), offset


def _decode_imul_r64_rm64_imm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    immediate = _signed(region, decoded.next_offset, 1, address)
    return (
        decoded.reg,
        decoded.rm,
        ImmediateOperand(immediate, width=8, signed=True),
    ), decoded.next_offset + 1


def _decode_imul_r32_rm32_imm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    immediate = _signed(region, decoded.next_offset, 1, address)
    return (
        decoded.reg,
        decoded.rm,
        ImmediateOperand(immediate, width=8, signed=True),
    ), decoded.next_offset + 1


def _decode_psrldq_xmm_imm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=128)
    if decoded.mod != 3 or not isinstance(decoded.rm, RegisterOperand):
        raise VocabularyDecodeError(f"{address:#x}: PSRLDQ requires an XMM register")
    _need(region, decoded.next_offset, 1, address)
    return (
        VectorRegisterOperand(_xmm(int(decoded.rm.register))),
        ImmediateOperand(int(region[decoded.next_offset]), width=8, signed=False),
    ), decoded.next_offset + 1


def _decode_movq_rm64_xmm_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    source = VectorRegisterOperand(_xmm(int(decoded.reg.register)))
    return (decoded.rm, source), decoded.next_offset


def _decode_dec_rm32_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=1, operand_width=32,
    )


def _decode_jmp_rm64_operands(region, offset, address, rex):
    return _decode_unary_group_rm(
        region, offset, address, rex, extension=4, operand_width=64,
    )


def _decode_mov_r64_imm64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset - 1, 1, address)
    opcode = int(region[offset - 1])
    register_code = (opcode & 0x7) | (0x8 if rex and rex & 0x1 else 0)
    _need(region, offset, 8, address)
    immediate = int.from_bytes(region[offset:offset + 8], "little", signed=False)
    return (
        RegisterOperand(_gpr(register_code), width=64),
        ImmediateOperand(immediate, width=64, signed=False),
    ), offset + 8


def _decode_call_rel32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    displacement = _signed(region, offset, 4, address)
    end = offset + 4
    # E8 has a one-byte opcode and this form rejects prefixes, so its next
    # instruction is exactly address + 5.
    target = address + 5 + displacement
    return (
        RelativeAddressOperand(displacement, width=32, target_address=target),
    ), end


def _decode_jcc_rel32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    displacement = _signed(region, offset, 4, address)
    end = offset + 4
    # The supported 0F 8x near-Jcc form has a two-byte opcode.
    target = address + 6 + displacement
    return (
        RelativeAddressOperand(displacement, width=32, target_address=target),
    ), end


def _decode_jcc_rel8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    displacement = _signed(region, offset, 1, address)
    return (
        RelativeAddressOperand(displacement, width=8, target_address=address + 2 + displacement),
    ), offset + 1


def _decode_call_rm64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    extension = (int(region[offset]) >> 3) & 0x7
    if extension != 2:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode FF requires ModRM /2 for near CALL, received /{extension}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of the group-5 /2 opcode extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    return (decoded.rm,), decoded.next_offset


def _decode_shift_rm64_imm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    extension = (int(region[offset]) >> 3) & 0x7
    if extension != 4:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode C1 requires ModRM /4 for SHL, received /{extension}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of the group-2 /4 opcode extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    _need(region, decoded.next_offset, 1, address)
    immediate = int.from_bytes(region[decoded.next_offset:decoded.next_offset + 1], "little")
    return (decoded.rm, ImmediateOperand(immediate, width=8, signed=False)), decoded.next_offset + 1


def _decode_not_rm64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    extension = (int(region[offset]) >> 3) & 0x7
    if extension != 2:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode F7 requires ModRM /2 for NOT, received /{extension}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of the group-3 /2 opcode extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    return (decoded.rm,), decoded.next_offset


def _decode_neg_rm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    modrm = int(region[offset])
    extension = (modrm >> 3) & 0x7
    if extension != 3:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode F6 requires ModRM /3 for NEG, received /{extension}"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=8)
    if isinstance(decoded.rm, RegisterOperand):
        low = modrm & 0x7
        if rex is None and low >= 4:
            raise VocabularyDecodeError(
                f"{address:#x}: legacy high-byte register needs a distinct register token"
            )
    return (decoded.rm,), decoded.next_offset


def _decode_inc_rm64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    extension = (int(region[offset]) >> 3) & 0x7
    if extension != 0:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode FF requires ModRM /0 for INC, received /{extension}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of the group-5 /0 opcode extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    return (decoded.rm,), decoded.next_offset


def _decode_inc_rm32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    extension = (int(region[offset]) >> 3) & 0x7
    if extension != 0:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode FF requires ModRM /0 for INC, received /{extension}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of the group-5 /0 opcode extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    return (decoded.rm,), decoded.next_offset


def _decode_neg_rm32_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    extension = (int(region[offset]) >> 3) & 0x7
    if extension != 3:
        raise VocabularyDecodeError(
            f"{address:#x}: opcode F7 requires ModRM /3 for NEG, received /{extension}"
        )
    if rex is not None and rex & 0x4:
        raise VocabularyDecodeError(
            f"{address:#x}: REX.R is not part of the group-3 /3 opcode extension"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=32)
    return (decoded.rm,), decoded.next_offset


def _decode_sete_rm8_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    _need(region, offset, 1, address)
    modrm = int(region[offset])
    extension = (modrm >> 3) & 0x7
    if extension != 0:
        raise VocabularyDecodeError(
            f"{address:#x}: SETE requires reserved ModRM.reg bits 000, received {extension:03b}"
        )
    decoded = _decode_modrm(region, offset, address, rex, register_width=8)
    if rex is None and decoded.mod == 3 and (modrm & 0x7) >= 4:
        raise VocabularyDecodeError(
            f"{address:#x}: legacy high-byte destination needs a distinct register token"
        )
    return (decoded.rm,), decoded.next_offset


def _decode_cmp_r64_rm64_operands(
    region: memoryview,
    offset: int,
    address: int,
    rex: int | None,
) -> tuple[tuple[MachineOperand, ...], int]:
    decoded = _decode_modrm(region, offset, address, rex, register_width=64)
    return (decoded.reg, decoded.rm), decoded.next_offset


X86_64_REFERENCE_VOCABULARY: tuple[InstructionSpec, ...] = (
    InstructionSpec(
        X86InstructionToken.IMUL_R32_RM32,
        MachineSemanticToken.INTEGER_MULTIPLY,
        b"\x0f\xaf",
        _decode_imul_operands,
    ),
    InstructionSpec(
        X86InstructionToken.LEA_R32_M,
        MachineSemanticToken.EFFECTIVE_ADDRESS,
        b"\x8d",
        _decode_lea_operands,
    ),
    InstructionSpec(
        X86InstructionToken.RET_NEAR,
        MachineSemanticToken.RETURN,
        b"\xc3",
        _decode_no_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.SUB_R64_IMM8,
        MachineSemanticToken.INTEGER_SUBTRACT,
        b"\x83",
        _decode_sub_r64_imm8_operands,
        modrm_extension=5,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.CALL_REL32,
        MachineSemanticToken.DIRECT_RELATIVE_CALL,
        b"\xe8",
        _decode_call_rel32_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.ADD_R64_IMM8,
        MachineSemanticToken.INTEGER_ADD,
        b"\x83",
        _decode_add_r64_imm8_operands,
        modrm_extension=0,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.JMP_REL32,
        MachineSemanticToken.DIRECT_RELATIVE_JUMP,
        b"\xe9",
        _decode_call_rel32_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_RM64_R64,
        MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
        b"\x89",
        _decode_mov_rm64_r64_operands,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.PUSH_R64,
        MachineSemanticToken.STACK_PUSH,
        b"\x50",
        _decode_push_r64_operands,
        opcode_mask=b"\xf8",
        allow_rex=True,
        allow_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_R64_RM64,
        MachineSemanticToken.REGISTER_OR_MEMORY_READ,
        b"\x8b",
        _decode_mov_r64_rm64_operands,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.AND_RM64_IMM8,
        MachineSemanticToken.BITWISE_AND,
        b"\x83",
        _decode_and_rm64_imm8_operands,
        modrm_extension=4,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_R64_IMM64,
        MachineSemanticToken.REGISTER_WRITE_IMMEDIATE,
        b"\xb8",
        _decode_mov_r64_imm64_operands,
        opcode_mask=b"\xf8",
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_R64_RM64,
        MachineSemanticToken.INTEGER_COMPARE,
        b"\x3b",
        _decode_cmp_r64_rm64_operands,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.JNE_REL32,
        MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x85",
        _decode_jcc_rel32_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.LEA_R64_M,
        MachineSemanticToken.EFFECTIVE_ADDRESS,
        b"\x8d",
        _decode_lea_r64_operands,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.CALL_RM64,
        MachineSemanticToken.INDIRECT_CALL,
        b"\xff",
        _decode_call_rm64_operands,
        modrm_extension=2,
        allow_rex=True,
        allow_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_R32_RM32,
        MachineSemanticToken.REGISTER_OR_MEMORY_READ,
        b"\x8b",
        _decode_mov_r32_rm32_operands,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.XOR_RM64_R64,
        MachineSemanticToken.BITWISE_XOR,
        b"\x31",
        _decode_binary_rm64_r64_operands,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.XOR_R64_RM64,
        MachineSemanticToken.BITWISE_XOR,
        b"\x33",
        _decode_binary_r64_rm64_operands,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.SHL_R64_IMM8,
        MachineSemanticToken.SHIFT_LEFT,
        b"\xc1",
        _decode_shift_rm64_imm8_operands,
        modrm_extension=4,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.AND_R64_RM64,
        MachineSemanticToken.BITWISE_AND,
        b"\x23",
        _decode_binary_r64_rm64_operands,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.JNE_REL8,
        MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x75",
        _decode_jcc_rel8_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.NOT_RM64,
        MachineSemanticToken.BITWISE_NOT,
        b"\xf7",
        _decode_not_rm64_operands,
        modrm_extension=2,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.POP_R64,
        MachineSemanticToken.STACK_POP,
        b"\x58",
        _decode_pop_r64_operands,
        opcode_mask=b"\xf8",
        allow_rex=True,
        allow_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.NOP_RM,
        MachineSemanticToken.NO_OPERATION,
        b"\x0f\x1f",
        _decode_nop_rm_operands,
        modrm_extension=0,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.INT3,
        MachineSemanticToken.BREAKPOINT_TRAP,
        b"\xcc",
        _decode_no_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.NOP,
        MachineSemanticToken.NO_OPERATION,
        b"\x90",
        _decode_no_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.SUB_R64_IMM32,
        MachineSemanticToken.INTEGER_SUBTRACT,
        b"\x81",
        _decode_sub_r64_imm32_operands,
        modrm_extension=5,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.XOR_R32_RM32,
        MachineSemanticToken.BITWISE_XOR,
        b"\x33",
        _decode_binary_r32_rm32_operands,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.TEST_RM32_R32,
        MachineSemanticToken.INTEGER_TEST,
        b"\x85",
        _decode_binary_rm32_r32_operands,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.TEST_RM64_R64,
        MachineSemanticToken.INTEGER_TEST,
        b"\x85",
        _decode_binary_rm64_r64_operands,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.JE_REL8,
        MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x74",
        _decode_jcc_rel8_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.JE_REL32,
        MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x84",
        _decode_jcc_rel32_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.JS_REL32,
        MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x88",
        _decode_jcc_rel32_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_R32_IMM32,
        MachineSemanticToken.REGISTER_WRITE_IMMEDIATE,
        b"\xb8",
        _decode_mov_r32_imm32_operands,
        opcode_mask=b"\xf8",
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.NEG_RM8,
        MachineSemanticToken.INTEGER_NEGATE,
        b"\xf6",
        _decode_neg_rm8_operands,
        modrm_extension=3,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SBB_R64_RM64,
        MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW,
        b"\x1b",
        _decode_binary_r64_rm64_operands,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMOVNE_R64_RM64,
        MachineSemanticToken.CONDITIONAL_MOVE,
        b"\x0f\x45",
        _decode_binary_r64_rm64_operands,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_RM32_R32,
        MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
        b"\x89",
        _decode_binary_rm32_r32_operands,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.INC_RM64,
        MachineSemanticToken.INTEGER_INCREMENT,
        b"\xff",
        _decode_inc_rm64_operands,
        modrm_extension=0,
        allow_rex=True,
        allow_rex_w=True,
        require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.OR_RM64_IMM8, MachineSemanticToken.BITWISE_OR,
        b"\x83", _decode_or_rm64_imm8_operands, modrm_extension=1,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_RM32_IMM32, MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
        b"\xc7", _decode_mov_rm32_imm32_operands, modrm_extension=0,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_RM8_IMM8, MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
        b"\xc6", _decode_mov_rm8_imm8_operands, modrm_extension=0,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SUB_R64_RM64, MachineSemanticToken.INTEGER_SUBTRACT,
        b"\x2b", _decode_binary_r64_rm64_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_RM8_R8, MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
        b"\x88", _decode_binary_rm8_r8_operands,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RM64_IMM8, MachineSemanticToken.INTEGER_COMPARE,
        b"\x83", _decode_cmp_rm64_imm8_operands, modrm_extension=7,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RM32_IMM8, MachineSemanticToken.INTEGER_COMPARE,
        b"\x83", _decode_cmp_rm32_imm8_operands, modrm_extension=7,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RAX_IMM32, MachineSemanticToken.INTEGER_COMPARE,
        b"\x3d", _decode_cmp_rax_imm32_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_EAX_IMM32, MachineSemanticToken.INTEGER_COMPARE,
        b"\x3d", _decode_cmp_eax_imm32_operands,
        allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.TEST_RM8_IMM8, MachineSemanticToken.INTEGER_TEST,
        b"\xf6", _decode_test_rm8_imm8_operands, modrm_extension=0,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.TEST_RM8_R8, MachineSemanticToken.INTEGER_TEST,
        b"\x84", _decode_binary_rm8_r8_operands,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOVSXD_R64_RM32, MachineSemanticToken.SIGN_EXTEND,
        b"\x63", _decode_movsxd_r64_rm32_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_R8_RM8, MachineSemanticToken.REGISTER_OR_MEMORY_READ,
        b"\x8a", _decode_binary_r8_rm8_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMOVNE_R32_RM32, MachineSemanticToken.CONDITIONAL_MOVE,
        b"\x0f\x45", _decode_binary_r32_rm32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_RM64_IMM32, MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
        b"\xc7", _decode_mov_rm64_imm32_operands, modrm_extension=0,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOVZX_R32_RM16, MachineSemanticToken.ZERO_EXTEND,
        b"\x0f\xb7", _decode_movzx_r32_rm16_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.JMP_REL8, MachineSemanticToken.DIRECT_RELATIVE_JUMP,
        b"\xeb", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.AND_RM32_IMM8, MachineSemanticToken.BITWISE_AND,
        b"\x83", _decode_and_rm32_imm8_operands, modrm_extension=4,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.ADD_R64_IMM32, MachineSemanticToken.INTEGER_ADD,
        b"\x81", _decode_add_r64_imm32_operands, modrm_extension=0,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.AND_RM32_IMM32, MachineSemanticToken.BITWISE_AND,
        b"\x81", _decode_and_rm32_imm32_operands, modrm_extension=4,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.TEST_RM16_R16, MachineSemanticToken.INTEGER_TEST,
        b"\x85", _decode_binary_rm16_r16_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RM16_R16, MachineSemanticToken.INTEGER_COMPARE,
        b"\x39", _decode_binary_rm16_r16_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RM16_IMM8, MachineSemanticToken.INTEGER_COMPARE,
        b"\x83", _decode_cmp_rm16_imm8_operands, modrm_extension=7,
        allow_rex=True, allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.MOV_RM16_R16, MachineSemanticToken.REGISTER_OR_MEMORY_WRITE,
        b"\x89", _decode_binary_rm16_r16_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.JS_REL8, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x78", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RM8_R8, MachineSemanticToken.INTEGER_COMPARE,
        b"\x38", _decode_binary_rm8_r8_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SUB_RM32_IMM8, MachineSemanticToken.INTEGER_SUBTRACT,
        b"\x83", _decode_sub_rm32_imm8_operands, modrm_extension=5,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.INC_RM32, MachineSemanticToken.INTEGER_INCREMENT,
        b"\xff", _decode_inc_rm32_operands, modrm_extension=0,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.BTR_RM32_IMM8, MachineSemanticToken.BIT_TEST_RESET,
        b"\x0f\xba", _decode_btr_rm32_imm8_operands, modrm_extension=6,
        allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RM32_R32, MachineSemanticToken.INTEGER_COMPARE,
        b"\x39", _decode_binary_rm32_r32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.ADD_R64_RM64, MachineSemanticToken.INTEGER_ADD,
        b"\x03", _decode_binary_r64_rm64_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.TEST_EAX_IMM32, MachineSemanticToken.INTEGER_TEST,
        b"\xa9", _decode_test_eax_imm32_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.OR_RM32_R32, MachineSemanticToken.BITWISE_OR,
        b"\x09", _decode_binary_rm32_r32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.JAE_REL32, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x83", _decode_jcc_rel32_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.MOVSX_R32_RM16, MachineSemanticToken.SIGN_EXTEND,
        b"\x0f\xbf", _decode_movsx_r32_rm16_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_R32_RM32, MachineSemanticToken.INTEGER_COMPARE,
        b"\x3b", _decode_binary_r32_rm32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.NEG_RM32, MachineSemanticToken.INTEGER_NEGATE,
        b"\xf7", _decode_neg_rm32_operands, modrm_extension=3, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.JA_REL8, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x77", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.JBE_REL32, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x86", _decode_jcc_rel32_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.JAE_REL8, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x73", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.MOVZX_R32_RM8, MachineSemanticToken.ZERO_EXTEND,
        b"\x0f\xb6", _decode_movzx_r32_rm8_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.OR_RM32_IMM8, MachineSemanticToken.BITWISE_OR,
        b"\x83", _decode_or_rm32_imm8_operands, modrm_extension=1, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SETE_RM8, MachineSemanticToken.CONDITIONAL_SET,
        b"\x0f\x94", _decode_sete_rm8_operands, modrm_extension=0, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_R16_RM16, MachineSemanticToken.INTEGER_COMPARE,
        b"\x3b", _decode_binary_r16_rm16_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.JA_REL32, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x87", _decode_jcc_rel32_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.SBB_R32_RM32, MachineSemanticToken.INTEGER_SUBTRACT_WITH_BORROW,
        b"\x1b", _decode_binary_r32_rm32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RM64_R64, MachineSemanticToken.INTEGER_COMPARE,
        b"\x39", _decode_binary_rm64_r64_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.SETB_RM8, MachineSemanticToken.CONDITIONAL_SET,
        b"\x0f\x92", _decode_sete_rm8_operands, modrm_extension=0, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SUB_R32_RM32, MachineSemanticToken.INTEGER_SUBTRACT,
        b"\x2b", _decode_binary_r32_rm32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.TEST_AL_IMM8, MachineSemanticToken.INTEGER_TEST,
        b"\xa8", _decode_test_al_imm8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RM32_IMM32, MachineSemanticToken.INTEGER_COMPARE,
        b"\x81", _decode_cmp_rm32_imm32_operands, modrm_extension=7, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.JBE_REL8, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x76", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.SHR_RM32_1, MachineSemanticToken.SHIFT_RIGHT_LOGICAL,
        b"\xd1", _decode_shr_rm32_1_operands, modrm_extension=5, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.TEST_RM32_IMM32, MachineSemanticToken.INTEGER_TEST,
        b"\xf7", _decode_test_rm32_imm32_operands, modrm_extension=0, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SAR_RM64_1, MachineSemanticToken.SHIFT_RIGHT_ARITHMETIC,
        b"\xd1", _decode_sar_rm64_1_operands, modrm_extension=7,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.AND_R32_RM32, MachineSemanticToken.BITWISE_AND,
        b"\x23", _decode_binary_r32_rm32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.ADD_RM64_R64, MachineSemanticToken.INTEGER_ADD,
        b"\x01", _decode_binary_rm64_r64_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.ADD_RM32_R32, MachineSemanticToken.INTEGER_ADD,
        b"\x01", _decode_binary_rm32_r32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.DIV_RM32, MachineSemanticToken.INTEGER_DIVIDE,
        b"\xf7", _decode_div_rm32_operands, modrm_extension=6, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.BT_RM32_IMM8, MachineSemanticToken.BIT_TEST,
        b"\x0f\xba", _decode_bt_rm32_imm8_operands, modrm_extension=4, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SHR_RM32_IMM8, MachineSemanticToken.SHIFT_RIGHT_LOGICAL,
        b"\xc1", _decode_shr_rm32_imm8_operands, modrm_extension=5, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.JGE_REL8, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x7d", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.DEC_RM32, MachineSemanticToken.INTEGER_DECREMENT,
        b"\xff", _decode_dec_rm32_operands, modrm_extension=1, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.JMP_RM64, MachineSemanticToken.INDIRECT_JUMP,
        b"\xff", _decode_jmp_rm64_operands, modrm_extension=4,
        allow_rex=True, allow_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMOVE_R64_RM64, MachineSemanticToken.CONDITIONAL_MOVE,
        b"\x0f\x44", _decode_binary_r64_rm64_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.JB_REL32, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x82", _decode_jcc_rel32_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.ADD_RM32_IMM8, MachineSemanticToken.INTEGER_ADD,
        b"\x83", _decode_add_rm32_imm8_operands, modrm_extension=0, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SUB_R16_RM16, MachineSemanticToken.INTEGER_SUBTRACT,
        b"\x2b", _decode_binary_r16_rm16_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.NEG_RM16, MachineSemanticToken.INTEGER_NEGATE,
        b"\xf7", _decode_neg_rm16_operands, modrm_extension=3, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.JG_REL32, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x8f", _decode_jcc_rel32_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RM8_IMM8, MachineSemanticToken.INTEGER_COMPARE,
        b"\x80", _decode_cmp_rm8_imm8_operands, modrm_extension=7, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SUB_RM16_IMM8, MachineSemanticToken.INTEGER_SUBTRACT,
        b"\x83", _decode_sub_rm16_imm8_operands, modrm_extension=5, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.ADD_R32_RM32, MachineSemanticToken.INTEGER_ADD,
        b"\x03", _decode_binary_r32_rm32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.JB_REL8, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x72", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.SHR_RM64_1, MachineSemanticToken.SHIFT_RIGHT_LOGICAL,
        b"\xd1", _decode_shr_rm64_1_operands, modrm_extension=5,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.JGE_REL32, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x8d", _decode_jcc_rel32_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.NEG_RM64, MachineSemanticToken.INTEGER_NEGATE,
        b"\xf7", _decode_neg_rm64_operands, modrm_extension=3,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_AL_IMM8, MachineSemanticToken.INTEGER_COMPARE,
        b"\x3c", _decode_cmp_al_imm8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.SETNE_RM8, MachineSemanticToken.CONDITIONAL_SET,
        b"\x0f\x95", _decode_sete_rm8_operands, modrm_extension=0, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.AND_RM8_IMM8, MachineSemanticToken.BITWISE_AND,
        b"\x80", _decode_and_rm8_imm8_operands, modrm_extension=4, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.BTS_RM32_IMM8, MachineSemanticToken.BIT_TEST,
        b"\x0f\xba", _decode_bts_rm32_imm8_operands, modrm_extension=5, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.JLE_REL8, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x7e", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.OR_R32_RM32, MachineSemanticToken.BITWISE_OR,
        b"\x0b", _decode_binary_r32_rm32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMOVE_R32_RM32, MachineSemanticToken.CONDITIONAL_MOVE,
        b"\x0f\x44", _decode_binary_r32_rm32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOV_R8_IMM8, MachineSemanticToken.REGISTER_WRITE_IMMEDIATE,
        b"\xb0", _decode_mov_r8_imm8_operands, opcode_mask=b"\xf8", allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.CMOVA_R32_RM32, MachineSemanticToken.CONDITIONAL_MOVE,
        b"\x0f\x47", _decode_binary_r32_rm32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.JG_REL8, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x7f", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.AND_EAX_IMM32, MachineSemanticToken.BITWISE_AND,
        b"\x25", _decode_and_eax_imm32_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.XOR_RM32_IMM8, MachineSemanticToken.BITWISE_XOR,
        b"\x83", _decode_xor_rm32_imm8_operands, modrm_extension=6, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.ADD_RM32_IMM32, MachineSemanticToken.INTEGER_ADD,
        b"\x81", _decode_add_rm32_imm32_operands, modrm_extension=0, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.CDQE, MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR,
        b"\x98", _decode_cdqe_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.XOR_R8_RM8, MachineSemanticToken.BITWISE_XOR,
        b"\x32", _decode_binary_r8_rm8_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.BT_RM64_R64, MachineSemanticToken.BIT_TEST,
        b"\x0f\xa3", _decode_bt_rm64_r64_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.MUL_RM64, MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED,
        b"\xf7", _decode_mul_rm64_operands, modrm_extension=4,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.DEC_RM64, MachineSemanticToken.INTEGER_DECREMENT,
        b"\xff", _decode_dec_rm64_operands, modrm_extension=1,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.NOP_66, MachineSemanticToken.NO_OPERATION,
        b"\x90", _decode_no_operands, allow_rex=False,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.CMP_RM64_IMM32, MachineSemanticToken.INTEGER_COMPARE,
        b"\x81", _decode_cmp_rm64_imm32_operands, modrm_extension=7,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.JL_REL8, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x7c", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.SHL_RM32_CL, MachineSemanticToken.SHIFT_LEFT,
        b"\xd3", _decode_shl_rm32_cl_operands, modrm_extension=4, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.XORPS_XMM_XMMM128, MachineSemanticToken.VECTOR_XOR,
        b"\x0f\x57", _decode_vector_reg_rm_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOVUPS_XMM_XMMM128, MachineSemanticToken.VECTOR_MOVE,
        b"\x0f\x10", _decode_vector_reg_rm_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOVUPS_XMMM128_XMM, MachineSemanticToken.VECTOR_MOVE,
        b"\x0f\x11", _decode_vector_rm_reg_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOVDQU_XMM_XMMM128, MachineSemanticToken.VECTOR_MOVE,
        b"\x0f\x6f", _decode_vector_reg_rm_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0xf3}),
        required_legacy_prefixes=frozenset({0xf3}),
    ),
    InstructionSpec(
        X86InstructionToken.MOVDQU_XMMM128_XMM, MachineSemanticToken.VECTOR_MOVE,
        b"\x0f\x7f", _decode_vector_rm_reg_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0xf3}),
        required_legacy_prefixes=frozenset({0xf3}),
    ),
    InstructionSpec(
        X86InstructionToken.MOVDQA_XMMM128_XMM, MachineSemanticToken.VECTOR_MOVE,
        b"\x0f\x7f", _decode_vector_rm_reg_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.AND_AL_IMM8, MachineSemanticToken.BITWISE_AND,
        b"\x24", _decode_and_al_imm8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.OR_RM16_IMM8, MachineSemanticToken.BITWISE_OR,
        b"\x83", _decode_or_rm16_imm8_operands, modrm_extension=1, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.JLE_REL32, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x8e", _decode_jcc_rel32_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.CMP_R8_RM8, MachineSemanticToken.INTEGER_COMPARE,
        b"\x3a", _decode_binary_r8_rm8_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.JL_REL32, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x0f\x8c", _decode_jcc_rel32_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.REP_STOSW, MachineSemanticToken.STRING_STORE,
        b"\xab", _decode_rep_stosw_operands, allow_rex=False,
        allowed_legacy_prefixes=frozenset({0x66, 0xf3}),
        required_legacy_prefixes=frozenset({0x66, 0xf3}),
    ),
    InstructionSpec(
        X86InstructionToken.CMOVO_R64_RM64, MachineSemanticToken.CONDITIONAL_MOVE,
        b"\x0f\x40", _decode_binary_r64_rm64_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.NOP_RM_66, MachineSemanticToken.NO_OPERATION,
        b"\x0f\x1f", _decode_nop_rm_operands, modrm_extension=0, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.JNS_REL8, MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        b"\x79", _decode_jcc_rel8_operands, allow_rex=False,
    ),
    InstructionSpec(
        X86InstructionToken.BTS_RM32_R32, MachineSemanticToken.BIT_TEST,
        b"\x0f\xab", _decode_bt_rm32_r32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SETA_RM8, MachineSemanticToken.CONDITIONAL_SET,
        b"\x0f\x97", _decode_sete_rm8_operands, modrm_extension=0, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.MUL_RM32, MachineSemanticToken.INTEGER_MULTIPLY_UNSIGNED,
        b"\xf7", _decode_mul_rm32_operands, modrm_extension=4, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.AND_RM64_R64, MachineSemanticToken.BITWISE_AND,
        b"\x21", _decode_binary_rm64_r64_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.NOT_RM32, MachineSemanticToken.BITWISE_NOT,
        b"\xf7", _decode_not_rm32_operands, modrm_extension=2, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.AND_R16_RM16, MachineSemanticToken.BITWISE_AND,
        b"\x23", _decode_binary_r16_rm16_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.MOVDQA_XMM_XMMM128, MachineSemanticToken.VECTOR_MOVE,
        b"\x0f\x6f", _decode_vector_reg_rm_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.MOV_R64_RM64_FS, MachineSemanticToken.REGISTER_OR_MEMORY_READ,
        b"\x8b", _decode_mov_r64_rm64_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
        allowed_legacy_prefixes=frozenset({0x65}),
        required_legacy_prefixes=frozenset({0x65}),
    ),
    InstructionSpec(
        X86InstructionToken.ROL_RM64_IMM8, MachineSemanticToken.ROTATE_LEFT,
        b"\xc1", _decode_rol_rm64_imm8_operands, modrm_extension=0,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.IMUL_R64_RM64_IMM8, MachineSemanticToken.INTEGER_MULTIPLY,
        b"\x6b", _decode_imul_r64_rm64_imm8_operands,
        allow_rex=True, allow_rex_w=True, require_rex_w=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOVAPS_XMMM128_XMM, MachineSemanticToken.VECTOR_MOVE,
        b"\x0f\x29", _decode_vector_rm_reg_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.MOVAPS_XMM_XMMM128, MachineSemanticToken.VECTOR_MOVE,
        b"\x0f\x28", _decode_vector_reg_rm_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.BT_RM32_R32, MachineSemanticToken.BIT_TEST,
        b"\x0f\xa3", _decode_bt_rm32_r32_operands, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.SETG_RM8, MachineSemanticToken.CONDITIONAL_SET,
        b"\x0f\x9f", _decode_sete_rm8_operands, modrm_extension=0, allow_rex=True,
    ),
    InstructionSpec(
        X86InstructionToken.ADD_R16_RM16, MachineSemanticToken.INTEGER_ADD,
        b"\x03", _decode_binary_r16_rm16_operands, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(
        X86InstructionToken.TEST_RM16_IMM16, MachineSemanticToken.INTEGER_TEST,
        b"\xf7", _decode_test_rm16_imm16_operands, modrm_extension=0, allow_rex=True,
        allowed_legacy_prefixes=frozenset({0x66}),
        required_legacy_prefixes=frozenset({0x66}),
    ),
    InstructionSpec(X86InstructionToken.CMOVE_R16_RM16, MachineSemanticToken.CONDITIONAL_MOVE, b"\x0f\x44", _decode_binary_r16_rm16_operands, allow_rex=True, allowed_legacy_prefixes=frozenset({0x66}), required_legacy_prefixes=frozenset({0x66})),
    InstructionSpec(X86InstructionToken.ROL_RM8_IMM8, MachineSemanticToken.ROTATE_LEFT, b"\xc0", _decode_rol_rm8_imm8_operands, modrm_extension=0, allow_rex=True),
    InstructionSpec(X86InstructionToken.SCASB, MachineSemanticToken.STRING_COMPARE, b"\xae", _decode_scasb_operands, allow_rex=False),
    InstructionSpec(X86InstructionToken.PSRLDQ_XMM_IMM8, MachineSemanticToken.VECTOR_SHIFT_RIGHT_LOGICAL, b"\x0f\x73", _decode_psrldq_xmm_imm8_operands, modrm_extension=3, allow_rex=True, allowed_legacy_prefixes=frozenset({0x66}), required_legacy_prefixes=frozenset({0x66})),
    InstructionSpec(X86InstructionToken.ADD_RM16_IMM8, MachineSemanticToken.INTEGER_ADD, b"\x83", _decode_add_rm16_imm8_operands, modrm_extension=0, allow_rex=True, allowed_legacy_prefixes=frozenset({0x66}), required_legacy_prefixes=frozenset({0x66})),
    InstructionSpec(X86InstructionToken.SETLE_RM8, MachineSemanticToken.CONDITIONAL_SET, b"\x0f\x9e", _decode_sete_rm8_operands, modrm_extension=0, allow_rex=True),
    InstructionSpec(X86InstructionToken.CMPXCHG_RM64_R64, MachineSemanticToken.ATOMIC_COMPARE_EXCHANGE, b"\x0f\xb1", _decode_binary_rm64_r64_operands, allow_rex=True, allow_rex_w=True, require_rex_w=True, allowed_legacy_prefixes=frozenset({0xf0}), required_legacy_prefixes=frozenset({0xf0})),
    InstructionSpec(X86InstructionToken.ROR_RM64_IMM8, MachineSemanticToken.ROTATE_RIGHT, b"\xc1", _decode_ror_rm64_imm8_operands, modrm_extension=1, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.CMOVAE_R64_RM64, MachineSemanticToken.CONDITIONAL_MOVE, b"\x0f\x43", _decode_binary_r64_rm64_operands, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.CMOVB_R64_RM64, MachineSemanticToken.CONDITIONAL_MOVE, b"\x0f\x42", _decode_binary_r64_rm64_operands, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.CMOVB_R16_RM16, MachineSemanticToken.CONDITIONAL_MOVE, b"\x0f\x42", _decode_binary_r16_rm16_operands, allow_rex=True, allowed_legacy_prefixes=frozenset({0x66}), required_legacy_prefixes=frozenset({0x66})),
    InstructionSpec(X86InstructionToken.CMOVB_R32_RM32, MachineSemanticToken.CONDITIONAL_MOVE, b"\x0f\x42", _decode_binary_r32_rm32_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.BTR_RM32_R32, MachineSemanticToken.BIT_TEST_RESET, b"\x0f\xb3", _decode_btr_rm32_r32_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.OR_RM32_IMM32, MachineSemanticToken.BITWISE_OR, b"\x81", _decode_or_rm32_imm32_operands, modrm_extension=1, allow_rex=True),
    InstructionSpec(X86InstructionToken.SHL_RM16_IMM8, MachineSemanticToken.SHIFT_LEFT, b"\xc1", _decode_shl_rm16_imm8_operands, modrm_extension=4, allow_rex=True, allowed_legacy_prefixes=frozenset({0x66}), required_legacy_prefixes=frozenset({0x66})),
    InstructionSpec(X86InstructionToken.SHL_RM32_IMM8, MachineSemanticToken.SHIFT_LEFT, b"\xc1", _decode_shl_rm32_imm8_operands, modrm_extension=4, allow_rex=True),
    InstructionSpec(X86InstructionToken.CMOVBE_R32_RM32, MachineSemanticToken.CONDITIONAL_MOVE, b"\x0f\x46", _decode_binary_r32_rm32_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.REP_MOVSQ, MachineSemanticToken.STRING_MOVE, b"\xa5", _decode_rep_movsq_operands, allow_rex=True, allow_rex_w=True, require_rex_w=True, allowed_legacy_prefixes=frozenset({0xf3}), required_legacy_prefixes=frozenset({0xf3})),
    InstructionSpec(X86InstructionToken.SETNS_RM8, MachineSemanticToken.CONDITIONAL_SET, b"\x0f\x99", _decode_sete_rm8_operands, modrm_extension=0, allow_rex=True),
    InstructionSpec(X86InstructionToken.MOVSX_R64_RM16, MachineSemanticToken.SIGN_EXTEND, b"\x0f\xbf", _decode_movsx_r64_rm16_operands, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.CMOVLE_R32_RM32, MachineSemanticToken.CONDITIONAL_MOVE, b"\x0f\x4e", _decode_binary_r32_rm32_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.AND_RM32_R32, MachineSemanticToken.BITWISE_AND, b"\x21", _decode_binary_rm32_r32_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.IMUL_R32_RM32_IMM8, MachineSemanticToken.INTEGER_MULTIPLY, b"\x6b", _decode_imul_r32_rm32_imm8_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.SHR_RM64_IMM8, MachineSemanticToken.SHIFT_RIGHT_LOGICAL, b"\xc1", _decode_shr_rm64_imm8_operands, modrm_extension=5, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.OR_R8_RM8, MachineSemanticToken.BITWISE_OR, b"\x0a", _decode_binary_r8_rm8_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.XADD_RM32_R32, MachineSemanticToken.ATOMIC_EXCHANGE_ADD, b"\x0f\xc1", _decode_binary_rm32_r32_operands, allow_rex=True, allowed_legacy_prefixes=frozenset({0xf0}), required_legacy_prefixes=frozenset({0xf0})),
    InstructionSpec(X86InstructionToken.OR_EAX_IMM32, MachineSemanticToken.BITWISE_OR, b"\x0d", _decode_or_eax_imm32_operands, allow_rex=False),
    InstructionSpec(X86InstructionToken.AND_R8_RM8, MachineSemanticToken.BITWISE_AND, b"\x22", _decode_binary_r8_rm8_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.LOCK_ADD_RM8_R8, MachineSemanticToken.ATOMIC_ADD, b"\x00", _decode_binary_rm8_r8_operands, allow_rex=True, allowed_legacy_prefixes=frozenset({0xf0}), required_legacy_prefixes=frozenset({0xf0})),
    InstructionSpec(X86InstructionToken.MOVQ_RM64_XMM, MachineSemanticToken.VECTOR_MOVE, b"\x0f\x7e", _decode_movq_rm64_xmm_operands, allow_rex=True, allow_rex_w=True, require_rex_w=True, allowed_legacy_prefixes=frozenset({0x66}), required_legacy_prefixes=frozenset({0x66})),
    InstructionSpec(X86InstructionToken.XCHG_RM64_R64, MachineSemanticToken.EXCHANGE, b"\x87", _decode_binary_rm64_r64_operands, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.AND_RM16_IMM16, MachineSemanticToken.BITWISE_AND, b"\x81", _decode_and_rm16_imm16_operands, modrm_extension=4, allow_rex=True, allowed_legacy_prefixes=frozenset({0x66}), required_legacy_prefixes=frozenset({0x66})),
    InstructionSpec(X86InstructionToken.BTC_RM32_IMM8, MachineSemanticToken.BIT_TEST_COMPLEMENT, b"\x0f\xba", _decode_btc_rm32_imm8_operands, modrm_extension=7, allow_rex=True),
    InstructionSpec(X86InstructionToken.DIV_RM64, MachineSemanticToken.INTEGER_DIVIDE, b"\xf7", _decode_div_rm64_operands, modrm_extension=6, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.OR_RM16_R16, MachineSemanticToken.BITWISE_OR, b"\x09", _decode_binary_rm16_r16_operands, allow_rex=True, allowed_legacy_prefixes=frozenset({0x66}), required_legacy_prefixes=frozenset({0x66})),
    InstructionSpec(X86InstructionToken.LOCK_DEC_RM32, MachineSemanticToken.INTEGER_DECREMENT, b"\xff", _decode_dec_rm32_operands, modrm_extension=1, allow_rex=True, allowed_legacy_prefixes=frozenset({0xf0}), required_legacy_prefixes=frozenset({0xf0})),
    InstructionSpec(X86InstructionToken.MOVSD_XMM_XMMM64, MachineSemanticToken.VECTOR_MOVE, b"\x0f\x10", _decode_vector_reg_rm_operands, allow_rex=True, allowed_legacy_prefixes=frozenset({0xf2}), required_legacy_prefixes=frozenset({0xf2})),
    InstructionSpec(X86InstructionToken.MOVSD_XMMM64_XMM, MachineSemanticToken.VECTOR_MOVE, b"\x0f\x11", _decode_vector_rm_reg_operands, allow_rex=True, allowed_legacy_prefixes=frozenset({0xf2}), required_legacy_prefixes=frozenset({0xf2})),
    InstructionSpec(X86InstructionToken.ADD_RAX_IMM32, MachineSemanticToken.INTEGER_ADD, b"\x05", _decode_add_rax_imm32_operands, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.OR_R64_RM64, MachineSemanticToken.BITWISE_OR, b"\x0b", _decode_binary_r64_rm64_operands, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.INT_IMM8, MachineSemanticToken.SOFTWARE_INTERRUPT, b"\xcd", _decode_int_imm8_operands, allow_rex=False),
    InstructionSpec(X86InstructionToken.INC_RM16, MachineSemanticToken.INTEGER_INCREMENT, b"\xff", _decode_inc_rm16_operands, modrm_extension=0, allow_rex=True, allowed_legacy_prefixes=frozenset({0x66}), required_legacy_prefixes=frozenset({0x66})),
    InstructionSpec(X86InstructionToken.SHR_RM8_IMM8, MachineSemanticToken.SHIFT_RIGHT_LOGICAL, b"\xc0", _decode_shr_rm8_imm8_operands, modrm_extension=5, allow_rex=True),
    InstructionSpec(X86InstructionToken.CQO, MachineSemanticToken.SIGN_EXTEND_ACCUMULATOR, b"\x99", _decode_cqo_operands, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.IDIV_RM64, MachineSemanticToken.INTEGER_DIVIDE_SIGNED, b"\xf7", _decode_idiv_rm64_operands, modrm_extension=7, allow_rex=True, allow_rex_w=True, require_rex_w=True),
    InstructionSpec(X86InstructionToken.CMOVAE_R32_RM32, MachineSemanticToken.CONDITIONAL_MOVE, b"\x0f\x43", _decode_binary_r32_rm32_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.CMOVL_R32_RM32, MachineSemanticToken.CONDITIONAL_MOVE, b"\x0f\x4c", _decode_binary_r32_rm32_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.XOR_RM32_R32, MachineSemanticToken.BITWISE_XOR, b"\x31", _decode_binary_rm32_r32_operands, allow_rex=True),
    InstructionSpec(X86InstructionToken.SHL_RM64_CL, MachineSemanticToken.SHIFT_LEFT, b"\xd3", _decode_shl_rm64_cl_operands, modrm_extension=4, allow_rex=True, allow_rex_w=True, require_rex_w=True),
)


_LEGACY_PREFIXES = frozenset({
    0x26, 0x2E, 0x36, 0x3E, 0x64, 0x65,
    0x66, 0x67, 0xF0, 0xF2, 0xF3,
})


def _strict_region_bytes(
    binary_region: bytes | bytearray | memoryview | Iterable[int],
) -> tuple[memoryview, int]:
    """Normalize a region without truncating, wrapping, or guessing values."""

    if isinstance(binary_region, memoryview):
        try:
            raw = binary_region.cast("B")
        except (TypeError, ValueError) as error:
            raise VocabularyDecodeError(
                "binary memoryview must be contiguous and byte-addressable"
            ) from error
        return raw, len(raw)
    if isinstance(binary_region, (bytes, bytearray)):
        raw = memoryview(binary_region)
        return raw, len(raw)

    values: list[int] = []
    try:
        iterator = iter(binary_region)
    except TypeError as error:
        raise VocabularyDecodeError("binary region must be bytes or an integer iterable") from error
    for offset, value in enumerate(iterator):
        if isinstance(value, bool):
            raise VocabularyDecodeError(
                f"binary value at region offset {offset} is boolean, not a byte integer"
            )
        try:
            byte = operator.index(value)
        except TypeError as error:
            raise VocabularyDecodeError(
                f"binary value at region offset {offset} is not an integer: {value!r}"
            ) from error
        if not 0 <= byte <= 0xFF:
            raise VocabularyDecodeError(
                f"binary value at region offset {offset} is outside [0, 255]: {byte}"
            )
        values.append(byte)
    raw = memoryview(bytes(values))
    return raw, len(raw)


class X86ReferenceDecoder:
    """Decode a bounded region exclusively through ``InstructionSpec`` data."""

    def __init__(
        self,
        vocabulary: Iterable[InstructionSpec] = X86_64_REFERENCE_VOCABULARY,
    ) -> None:
        self.vocabulary = tuple(vocabulary)
        if not self.vocabulary:
            raise ValueError("an instruction vocabulary cannot be empty")
        self._by_first_byte: dict[int, tuple[InstructionSpec, ...]] = {}
        tokens: set[X86InstructionToken] = set()
        encodings: set[tuple[bytes, bytes, int | None, bool, bool, bool, tuple[int, ...], tuple[int, ...]]] = set()
        for spec in self.vocabulary:
            if not spec.opcode:
                raise ValueError(f"instruction token {spec.token} has no opcode")
            if spec.token in tokens:
                raise ValueError(f"duplicate instruction token {spec.token!r}")
            if spec.modrm_extension is not None and not 0 <= spec.modrm_extension <= 7:
                raise ValueError(
                    f"invalid ModRM extension /{spec.modrm_extension} for {spec.token!r}"
                )
            if not spec.required_legacy_prefixes <= spec.allowed_legacy_prefixes:
                raise ValueError(
                    f"required legacy prefixes are not allowed for {spec.token!r}"
                )
            mask = spec.opcode_mask or bytes([0xFF] * len(spec.opcode))
            if len(mask) != len(spec.opcode):
                raise ValueError(f"opcode mask width differs for {spec.token!r}")
            if any(value == 0 for value in mask):
                raise ValueError(f"opcode masks must constrain every byte for {spec.token!r}")
            encoding = (
                spec.opcode,
                mask,
                spec.modrm_extension,
                spec.allow_rex,
                spec.allow_rex_w,
                spec.require_rex_w,
                tuple(sorted(spec.allowed_legacy_prefixes)),
                tuple(sorted(spec.required_legacy_prefixes)),
            )
            if encoding in encodings:
                suffix = "" if spec.modrm_extension is None else f" /{spec.modrm_extension}"
                raise ValueError(f"duplicate encoding {spec.opcode.hex(' ')}{suffix}")
            sibling_extensions = {
                extension
                for (
                    opcode,
                    sibling_mask,
                    extension,
                    allow_rex,
                    allow_rex_w,
                    require_rex_w,
                    allowed_legacy,
                    required_legacy,
                ) in encodings
                if opcode == spec.opcode
                and sibling_mask == mask
                and allow_rex == spec.allow_rex
                and allow_rex_w == spec.allow_rex_w
                and require_rex_w == spec.require_rex_w
                and allowed_legacy == tuple(sorted(spec.allowed_legacy_prefixes))
                and required_legacy == tuple(sorted(spec.required_legacy_prefixes))
            }
            if sibling_extensions and (
                spec.modrm_extension is None or None in sibling_extensions
            ):
                raise ValueError(
                    f"ambiguous plain/group opcode vocabulary for {spec.opcode.hex(' ')}"
                )
            tokens.add(spec.token)
            encodings.add(encoding)
            for first_byte in range(256):
                if first_byte & mask[0] != spec.opcode[0] & mask[0]:
                    continue
                bucket = self._by_first_byte.setdefault(first_byte, ())
                self._by_first_byte[first_byte] = tuple(sorted(
                    (*bucket, spec), key=lambda item: len(item.opcode), reverse=True,
                ))

    def decode_one(
        self,
        region: memoryview,
        offset: int,
        *,
        base_address: int = 0,
    ) -> tuple[DecodedInstruction, int]:
        start = int(offset)
        address = int(base_address) + start
        _need(region, start, 1, address)
        cursor = start
        rex: int | None = None
        legacy_prefixes: list[int] = []
        first = int(region[cursor])
        while first in _LEGACY_PREFIXES:
            legacy_prefixes.append(first)
            cursor += 1
            _need(region, cursor, 1, address)
            first = int(region[cursor])
        if 0x40 <= first <= 0x4F:
            rex = first
            cursor += 1
            _need(region, cursor, 1, address)
            first = int(region[cursor])
            if 0x40 <= first <= 0x4F:
                raise VocabularyDecodeError(
                    f"{address:#x}: multiple REX prefixes are outside the vocabulary"
                )
            if first in _LEGACY_PREFIXES:
                raise VocabularyDecodeError(
                    f"{address:#x}: legacy prefix {first:#04x} after REX is invalid"
                )

        candidates = self._by_first_byte.get(first, ())
        def matches(item: InstructionSpec) -> bool:
            observed_legacy = frozenset(legacy_prefixes)
            if not observed_legacy <= item.allowed_legacy_prefixes:
                return False
            if not item.required_legacy_prefixes <= observed_legacy:
                return False
            if rex is not None and not item.allow_rex:
                return False
            if rex is not None and rex & 0x08 and not item.allow_rex_w:
                return False
            if item.require_rex_w and (rex is None or not rex & 0x08):
                return False
            operand_offset = cursor + len(item.opcode)
            mask = item.opcode_mask or bytes([0xFF] * len(item.opcode))
            if (
                operand_offset > len(region)
                or any(
                    actual & masked != expected & masked
                    for actual, expected, masked in zip(
                        region[cursor:operand_offset], item.opcode, mask,
                    )
                )
            ):
                return False
            if item.modrm_extension is None:
                return True
            if operand_offset >= len(region):
                return False
            return ((int(region[operand_offset]) >> 3) & 0x7) == item.modrm_extension

        spec = next((item for item in candidates if matches(item)), None)
        if spec is None:
            available = bytes(region[cursor:])
            truncated = next((
                item for item in candidates
                if len(available) < len(item.opcode)
                and all(
                    actual & masked == expected & masked
                    for actual, expected, masked in zip(
                        available,
                        item.opcode,
                        item.opcode_mask or bytes([0xFF] * len(item.opcode)),
                    )
                )
            ), None)
            if truncated is not None:
                raise VocabularyDecodeError(
                    f"{address:#x}: truncated opcode for token {truncated.token.name}; "
                    f"need {len(truncated.opcode)} opcode byte(s), have {len(available)}"
                )
            opcode_matches = tuple(
                item for item in candidates
                if cursor + len(item.opcode) <= len(region)
                and all(
                    actual & masked == expected & masked
                    for actual, expected, masked in zip(
                        region[cursor:cursor + len(item.opcode)],
                        item.opcode,
                        item.opcode_mask or bytes([0xFF] * len(item.opcode)),
                    )
                )
            )
            if opcode_matches and all(
                item.modrm_extension is not None for item in opcode_matches
            ):
                modrm_offset = cursor + len(opcode_matches[0].opcode)
                _need(region, modrm_offset, 1, address)
                extension = (int(region[modrm_offset]) >> 3) & 0x7
                raise VocabularyDecodeError(
                    f"{address:#x}: no instruction token for opcode "
                    f"{opcode_matches[0].opcode.hex(' ')} ModRM /{extension}"
                )
            preview = bytes(region[cursor:min(len(region), cursor + 4)]).hex(" ")
            raise VocabularyDecodeError(
                f"{address:#x}: no instruction token for bytes {preview or '<eof>'}"
            )
        if rex is not None and not spec.allow_rex:
            raise VocabularyDecodeError(
                f"{address:#x}: REX prefix is outside token {spec.token.name}"
            )
        if rex is not None and rex & 0x08 and not spec.allow_rex_w:
            raise VocabularyDecodeError(
                f"{address:#x}: REX.W changes operand width and is outside "
                f"token {spec.token.name}"
            )
        if spec.require_rex_w and (rex is None or not rex & 0x08):
            raise VocabularyDecodeError(
                f"{address:#x}: token {spec.token.name} requires REX.W"
            )

        operand_offset = cursor + len(spec.opcode)
        operands, end = spec.decode_operands(region, operand_offset, address, rex)
        encoded = bytes(region[start:end])
        return DecodedInstruction(
            address=address,
            token=spec.token,
            semantic=spec.semantic,
            operands=operands,
            encoded=encoded,
            rex=rex,
            legacy_prefixes=tuple(legacy_prefixes),
        ), end

    def decode_region(
        self,
        binary_region: bytes | bytearray | memoryview | Iterable[int],
        *,
        size: int | None = None,
        base_address: int = 0,
        stop_at_return: bool = True,
        allow_trailing_after_terminal: bool = False,
    ) -> tuple[DecodedInstruction, ...]:
        """Decode exactly the accepted prefix of a user-provided byte region."""

        raw, capacity = _strict_region_bytes(binary_region)
        if isinstance(size, bool):
            raise VocabularyDecodeError("binary size must be an integer, not boolean")
        try:
            accepted = capacity if size is None else operator.index(size)
        except TypeError as error:
            raise VocabularyDecodeError("binary size must be an integer") from error
        if accepted < 0 or accepted > len(raw):
            raise VocabularyDecodeError(
                f"binary size {accepted} is outside region capacity {len(raw)}"
            )
        region = raw[:accepted]
        decoded: list[DecodedInstruction] = []
        offset = 0
        while offset < len(region):
            instruction, offset = self.decode_one(
                region, offset, base_address=base_address,
            )
            decoded.append(instruction)
            if instruction.semantic in {
                MachineSemanticToken.RETURN,
                MachineSemanticToken.DIRECT_RELATIVE_JUMP,
                MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
            } and stop_at_return:
                if offset != len(region) and not allow_trailing_after_terminal:
                    trailing = bytes(region[offset:]).hex(" ")
                    raise VocabularyDecodeError(
                        f"{base_address + offset:#x}: trailing bytes after return: "
                        f"{trailing}"
                    )
                break
        if not decoded:
            raise VocabularyDecodeError("binary region contains no instructions")
        if stop_at_return and decoded[-1].semantic not in {
            MachineSemanticToken.RETURN,
            MachineSemanticToken.DIRECT_RELATIVE_JUMP,
            MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        }:
            raise VocabularyDecodeError(
                "binary region ended without a terminal control-transfer token"
            )
        return tuple(decoded)

    def decode_report(
        self,
        binary_region: bytes | bytearray | memoryview | Iterable[int],
        *,
        size: int | None = None,
        base_address: int = 0,
        stop_at_return: bool = True,
        allow_trailing_after_terminal: bool = False,
    ) -> DecodeReport:
        """Decode as far as instruction boundaries remain trustworthy.

        An x86 decoder cannot safely scan past an unknown variable-length
        instruction. Consequently the failure tuple has at most one terminal
        entry: it identifies the first byte for which vocabulary coverage was
        insufficient, while ``instructions`` preserves the valid prefix.
        """

        raw, capacity = _strict_region_bytes(binary_region)
        if isinstance(size, bool):
            raise VocabularyDecodeError("binary size must be an integer, not boolean")
        try:
            accepted = capacity if size is None else operator.index(size)
        except TypeError as error:
            raise VocabularyDecodeError("binary size must be an integer") from error
        if accepted < 0 or accepted > capacity:
            raise VocabularyDecodeError(
                f"binary size {accepted} is outside region capacity {capacity}"
            )

        region = raw[:accepted]
        decoded: list[DecodedInstruction] = []
        failures: list[VocabularyFailure] = []
        offset = 0
        stopped_at_return = False
        stopped_at_control_transfer = False
        while offset < accepted:
            start = offset
            try:
                instruction, offset = self.decode_one(
                    region, start, base_address=base_address,
                )
            except VocabularyDecodeError as error:
                failures.append(VocabularyFailure(
                    category="decode",
                    region_offset=start,
                    address=int(base_address) + start,
                    encoded_preview=bytes(region[start:min(accepted, start + 8)]),
                    reason=str(error),
                ))
                break
            decoded.append(instruction)
            if instruction.semantic in {
                MachineSemanticToken.RETURN,
                MachineSemanticToken.DIRECT_RELATIVE_JUMP,
                MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
            } and stop_at_return:
                stopped_at_return = instruction.semantic is MachineSemanticToken.RETURN
                stopped_at_control_transfer = True
                if offset != accepted and not allow_trailing_after_terminal:
                    failures.append(VocabularyFailure(
                        category="trailing_bytes",
                        region_offset=offset,
                        address=int(base_address) + offset,
                        encoded_preview=bytes(region[offset:min(accepted, offset + 8)]),
                        reason=f"{base_address + offset:#x}: trailing bytes after return",
                    ))
                break

        if not failures and not decoded:
            failures.append(VocabularyFailure(
                category="empty_region",
                region_offset=0,
                address=int(base_address),
                encoded_preview=b"",
                reason="binary region contains no instructions",
            ))
        elif (
            not failures
            and stop_at_return
            and decoded[-1].semantic not in {
                MachineSemanticToken.RETURN,
                MachineSemanticToken.DIRECT_RELATIVE_JUMP,
                MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
            }
        ):
            failures.append(VocabularyFailure(
                category="missing_return",
                region_offset=offset,
                address=int(base_address) + offset,
                encoded_preview=b"",
                reason="binary region ended without a terminal control-transfer token",
            ))

        return DecodeReport(
            instructions=tuple(decoded),
            failures=tuple(failures),
            region_capacity=capacity,
            accepted_size=accepted,
            decoded_bytes=offset,
            stopped_at_return=stopped_at_return,
            stopped_at_control_transfer=stopped_at_control_transfer,
        )

    def audit_region(
        self,
        binary_region: bytes | bytearray | memoryview | Iterable[int],
        *,
        size: int | None = None,
        base_address: int = 0,
        preview_bytes: int = 16,
    ) -> VocabularyAuditReport:
        """Classify every byte, using bytewise recovery only for diagnostics.

        A successful candidate after a gap is not promoted to a proven x86
        boundary: it may begin inside an unknown instruction's operand field.
        The report is nevertheless complete as a byte-coverage census and is
        suitable for prioritizing vocabulary growth. Any gap must still make
        semantic lifting fail closed.
        """

        if preview_bytes <= 0:
            raise ValueError("preview_bytes must be positive")
        raw, capacity = _strict_region_bytes(binary_region)
        if isinstance(size, bool):
            raise VocabularyDecodeError("binary size must be an integer, not boolean")
        try:
            accepted = capacity if size is None else operator.index(size)
        except TypeError as error:
            raise VocabularyDecodeError("binary size must be an integer") from error
        if accepted < 0 or accepted > capacity:
            raise VocabularyDecodeError(
                f"binary size {accepted} is outside region capacity {capacity}"
            )
        region = raw[:accepted]
        candidates: list[DecodedInstruction] = []
        gaps: list[VocabularyFailure] = []
        signatures: dict[bytes, int] = {}
        known_bytes = 0
        offset = 0
        gap_start: int | None = None
        gap_reason = ""

        def close_gap(end: int) -> None:
            nonlocal gap_start, gap_reason
            if gap_start is None:
                return
            encoded = bytes(region[gap_start:end])
            signature = encoded[:4]
            signatures[signature] = signatures.get(signature, 0) + 1
            gaps.append(VocabularyFailure(
                category="diagnostic_gap",
                region_offset=gap_start,
                address=int(base_address) + gap_start,
                encoded_preview=encoded[:preview_bytes],
                reason=(
                    f"unclassified byte span [{gap_start}, {end}) of length "
                    f"{end - gap_start}; first decoder error: {gap_reason}"
                ),
            ))
            gap_start = None
            gap_reason = ""

        while offset < accepted:
            try:
                instruction, end = self.decode_one(
                    region, offset, base_address=base_address,
                )
            except VocabularyDecodeError as error:
                if gap_start is None:
                    gap_start = offset
                    gap_reason = str(error)
                offset += 1
                continue
            close_gap(offset)
            candidates.append(instruction)
            known_bytes += end - offset
            offset = end
        close_gap(accepted)
        return VocabularyAuditReport(
            accepted_size=accepted,
            candidate_instructions=tuple(candidates),
            gap_failures=tuple(gaps),
            known_bytes=known_bytes,
            missing_bytes=accepted - known_bytes,
            signature_counts=tuple(sorted(signatures.items())),
        )


__all__ = [
    "DecodedInstruction",
    "DecodeReport",
    "AuditConfidence",
    "EffectiveAddressOperand",
    "ImmediateOperand",
    "InstructionSpec",
    "MachineOperand",
    "MachineSemanticToken",
    "RegisterOperand",
    "RelativeAddressOperand",
    "VocabularyDecodeError",
    "VocabularyFailure",
    "VocabularyAuditReport",
    "X86InstructionToken",
    "X86ReferenceDecoder",
    "X86Register",
    "X86_64_REFERENCE_VOCABULARY",
]
