"""Configurable x86 read-head state machine over ``AbstractTensor``.

The read head is an encoding engine, not an instruction vocabulary. Callers
provide tensor-resident tables describing opcode maps, ModRM opcode groups,
immediate fields, prefix policy, and terminal forms. Each transition advances
all active lanes by at most one byte using tensor masks. No assembly strings,
host disassembler, or semantic lowering is involved.

The host may loop over ``transition`` to schedule work, but cursor movement,
table lookup, field accumulation, and lane state are AbstractTensor values so
the same state machine can be captured by repository backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum, IntFlag
from types import MappingProxyType
from typing import Callable, ClassVar, Iterable, Mapping, Sequence

from ..common.tensors import AbstractTensor


class ReadPhase(IntEnum):
    PREFIX = 0
    OPCODE = 1
    MODRM = 2
    SIB = 3
    DISPLACEMENT = 4
    IMMEDIATE = 5
    EMIT = 6
    HALT = 7


class ReadStatus(IntEnum):
    ACTIVE = 0
    EMITTED = 1
    FAILED = 2
    HALTED = 3


class ReadFailure(IntEnum):
    NONE = 0
    TRUNCATED = 1
    UNKNOWN_OPCODE = 2
    UNKNOWN_OPCODE_GROUP = 3
    FORBIDDEN_PREFIX = 4
    REQUIRED_PREFIX_MISSING = 5
    INSTRUCTION_TOO_LONG = 6
    CONFIGURATION = 7
    STEP_LIMIT = 8
    INVALID_MODRM = 9


class ReadHeadExecutionMode(IntEnum):
    DECODE = 0
    TRACE = 1
    EMULATE = 2


class ReadHeadDirection(IntEnum):
    BACKWARD = -1
    FORWARD = 1


class PrefixAction(IntEnum):
    NONE = 0
    LEGACY = 1
    REX = 2
    FORBIDDEN = 3


class EncodingFlag(IntFlag):
    HAS_MODRM = 1 << 0
    IMMEDIATE_SIGNED = 1 << 1
    IMMEDIATE_RELATIVE = 1 << 2
    TERMINAL = 1 << 3
    MODRM_MEMORY_ONLY = 1 << 4


class X86OperandForm(IntEnum):
    """Bidirectional operand-field grammar owned by an encoding row."""

    NONE = 0
    REG_RM = 1
    RM_REG = 2
    RM_IMMEDIATE = 3
    OPCODE_REGISTER = 4
    OPCODE_REGISTER_IMMEDIATE = 5
    RELATIVE = 6
    RM = 7
    REG_RM_IMMEDIATE = 8
    IMMEDIATE = 9


@dataclass(frozen=True, slots=True)
class X86EncodingRow:
    """Host-authored row compiled into tensor lookup tables."""

    token: int
    opcode_map: int
    opcode: int
    opcode_mask: int = 0xFF
    modrm_extension: int | None = None
    has_modrm: bool = False
    immediate_bytes: int = 0
    immediate_signed: bool = False
    immediate_relative: bool = False
    terminal: bool = False
    allow_rex: bool = True
    allowed_rex_mask: int = 0x0F
    required_rex_mask: int = 0
    allowed_legacy_mask: int = 0
    required_legacy_mask: int = 0
    operand_form: X86OperandForm = X86OperandForm.NONE
    modrm_memory_only: bool = False

    def __post_init__(self) -> None:
        if self.token < 0:
            raise ValueError("instruction token must be non-negative")
        if self.opcode_map < 0:
            raise ValueError("opcode map must be non-negative")
        if not 0 <= self.opcode <= 0xFF:
            raise ValueError("opcode must be a byte")
        if not 0 < self.opcode_mask <= 0xFF:
            raise ValueError("opcode mask must be a nonzero byte")
        if self.modrm_extension is not None and not 0 <= self.modrm_extension <= 7:
            raise ValueError("ModRM opcode-group extension must be in [0, 7]")
        if self.modrm_extension is not None and not self.has_modrm:
            raise ValueError("opcode-group rows require ModRM")
        if self.modrm_memory_only and not self.has_modrm:
            raise ValueError("memory-only ModRM constraint requires ModRM")
        if self.immediate_bytes not in (0, 1, 2, 4, 8):
            raise ValueError("immediate width must be 0, 1, 2, 4, or 8 bytes")
        if self.allowed_rex_mask & ~0x0F or self.required_rex_mask & ~0x0F:
            raise ValueError("REX masks use only W/R/X/B bits")
        if self.required_rex_mask & ~self.allowed_rex_mask:
            raise ValueError("required REX bits must also be allowed")
        if self.required_rex_mask and not self.allow_rex:
            raise ValueError("a required REX bit needs allow_rex=True")
        if self.required_legacy_mask & ~self.allowed_legacy_mask:
            raise ValueError("required legacy prefixes must also be allowed")
        if self.operand_form in {
            X86OperandForm.REG_RM, X86OperandForm.RM_REG,
            X86OperandForm.RM_IMMEDIATE, X86OperandForm.RM,
        } and not self.has_modrm:
            raise ValueError("ModRM operand forms require has_modrm=True")
        if self.operand_form is X86OperandForm.RM_IMMEDIATE and not self.immediate_bytes:
            raise ValueError("immediate operand form requires immediate bytes")
        if self.operand_form in {
            X86OperandForm.REG_RM_IMMEDIATE, X86OperandForm.IMMEDIATE,
        } and not self.immediate_bytes:
            raise ValueError("immediate operand form requires immediate bytes")


@dataclass(frozen=True, slots=True)
class X86EncodingFields:
    """Complete variable fields needed to invert one encoding row."""

    rex: int | None = None
    legacy_prefixes: tuple[int, ...] | None = None
    opcode_low_bits: int = 0
    modrm: int | None = None
    sib: int | None = None
    displacement: bytes = b""
    immediate: int | None = None


@dataclass(frozen=True, slots=True)
class X86AllocatedInstruction:
    """A selected token plus explicit physical operand placements."""

    token: int
    operands: tuple[object, ...]
    address: int | None = None


@dataclass(frozen=True, slots=True)
class X86ReadHeadConfig:
    """Tensor-resident transition vocabulary shared by every read lane."""

    opcode_token: AbstractTensor
    opcode_group: AbstractTensor
    legacy_selector: AbstractTensor
    group_token: AbstractTensor
    escape_map: AbstractTensor
    prefix_action: AbstractTensor
    prefix_bit: AbstractTensor
    flags: AbstractTensor
    immediate_bytes: AbstractTensor
    allowed_rex_mask: AbstractTensor
    allow_rex_prefix: AbstractTensor
    required_rex_mask: AbstractTensor
    allowed_legacy_mask: AbstractTensor
    required_legacy_mask: AbstractTensor
    maximum_instruction_bytes: int = 15
    profile_name: str = "anonymous"
    encoding_row_count: int = 0

    @property
    def opcode_map_count(self) -> int:
        return int(self.opcode_token.shape[2])

    @property
    def vocabulary_size(self) -> int:
        return int(self.flags.shape[0])

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[X86EncodingRow],
        *,
        escape_maps: dict[tuple[int, int], int] | None = None,
        legacy_prefix_bits: dict[int, int] | None = None,
        forbid_prefixes: Iterable[int] = (),
        rex_prefixes: bool = True,
        maximum_instruction_bytes: int = 15,
        profile_name: str = "anonymous",
        like: AbstractTensor | None = None,
    ) -> "X86ReadHeadConfig":
        """Validate encoding rows and compile dense tensor lookup tables."""

        records = tuple(rows)
        if not records:
            raise ValueError("x86 read-head configuration cannot be empty")
        if maximum_instruction_bytes <= 0:
            raise ValueError("maximum instruction length must be positive")
        escapes = dict({
            (0, 0x0F): 1,
            (1, 0x38): 2,
            (1, 0x3A): 3,
        } if escape_maps is None else escape_maps)
        legacy = dict({
            0xF0: 1 << 0,
            0xF2: 1 << 1,
            0xF3: 1 << 2,
            0x2E: 1 << 3,
            0x36: 1 << 4,
            0x3E: 1 << 5,
            0x26: 1 << 6,
            0x64: 1 << 7,
            0x65: 1 << 8,
            0x66: 1 << 9,
            0x67: 1 << 10,
        } if legacy_prefix_bits is None else legacy_prefix_bits)
        map_count = 1 + max(
            [row.opcode_map for row in records]
            + [source for source, _ in escapes]
            + list(escapes.values())
        )
        vocabulary_size = 1 + max(row.token for row in records)
        # Legacy-prefix identity and REX.W are encoding selectors, not merely
        # post-selection policy.  The same opcode may name distinct tokens
        # under no prefix, 66, F2, F3, or LOCK.
        legacy_masks = {0}
        maximum_legacy_mask = 1 << len(legacy)
        for row in records:
            legacy_masks.update(
                mask for mask in range(maximum_legacy_mask)
                if not (mask & ~row.allowed_legacy_mask)
                and not (row.required_legacy_mask & ~mask)
            )
        ordered_legacy_masks = tuple(sorted(legacy_masks))
        legacy_lanes = {
            mask: lane for lane, mask in enumerate(ordered_legacy_masks)
        }
        legacy_selector = [-1] * maximum_legacy_mask
        for mask, lane in legacy_lanes.items():
            legacy_selector[mask] = lane

        # REX.W is an encoding selector, not merely a post-selection prefix
        # check.  For example B8+rd is MOV r32,imm32 without W and
        # MOV r64,imm64 with W.  Keep both dimensions tensor-resident.
        opcode_token = [
            [[[-1] * 256 for _ in range(map_count)] for _ in range(2)]
            for _ in ordered_legacy_masks
        ]
        opcode_group = [
            [[[-1] * 256 for _ in range(map_count)] for _ in range(2)]
            for _ in ordered_legacy_masks
        ]
        group_token: list[list[int]] = []
        group_ids: dict[tuple[int, int, int, int], int] = {}
        token_metadata: dict[int, tuple[int, int, int, int, int, int, int]] = {}
        encodings: set[tuple[int, int, int, int | None, int, int]] = set()

        for row in records:
            allows_w = bool(row.allow_rex and row.allowed_rex_mask & 0x08)
            requires_w = bool(row.required_rex_mask & 0x08)
            w_lanes = (1,) if requires_w else ((0, 1) if allows_w else (0,))
            row_legacy_masks = tuple(
                mask for mask in ordered_legacy_masks
                if not (mask & ~row.allowed_legacy_mask)
                and not (row.required_legacy_mask & ~mask)
            )
            for legacy_mask in row_legacy_masks:
                for w_lane in w_lanes:
                    identity = (
                        row.opcode_map, row.opcode, row.opcode_mask,
                        row.modrm_extension, w_lane, legacy_mask,
                    )
                    if identity in encodings:
                        raise ValueError(f"duplicate x86 encoding row {identity}")
                    encodings.add(identity)
            metadata = (
                (int(EncodingFlag.HAS_MODRM) if row.has_modrm else 0)
                | (int(EncodingFlag.IMMEDIATE_SIGNED) if row.immediate_signed else 0)
                | (int(EncodingFlag.IMMEDIATE_RELATIVE) if row.immediate_relative else 0)
                | (int(EncodingFlag.TERMINAL) if row.terminal else 0)
                | (int(EncodingFlag.MODRM_MEMORY_ONLY)
                   if row.modrm_memory_only else 0),
                row.immediate_bytes,
                int(row.allow_rex),
                row.allowed_rex_mask,
                row.required_rex_mask,
                row.allowed_legacy_mask,
                row.required_legacy_mask,
            )
            previous = token_metadata.setdefault(row.token, metadata)
            if previous != metadata:
                raise ValueError(f"token {row.token} has inconsistent encoding metadata")

            matching_opcodes = tuple(
                opcode for opcode in range(256)
                if opcode & row.opcode_mask == row.opcode & row.opcode_mask
            )
            for legacy_mask in row_legacy_masks:
                legacy_lane = legacy_lanes[legacy_mask]
                for w_lane in w_lanes:
                    for opcode in matching_opcodes:
                      if row.modrm_extension is None:
                        if opcode_group[legacy_lane][w_lane][row.opcode_map][opcode] >= 0:
                            raise ValueError("plain opcode conflicts with an opcode-group row")
                        if opcode_token[legacy_lane][w_lane][row.opcode_map][opcode] >= 0:
                            raise ValueError("masked opcode overlaps another plain token")
                        opcode_token[legacy_lane][w_lane][row.opcode_map][opcode] = row.token
                      else:
                        if opcode_token[legacy_lane][w_lane][row.opcode_map][opcode] >= 0:
                            raise ValueError("opcode-group row conflicts with a plain opcode")
                        key = (legacy_lane, w_lane, row.opcode_map, opcode)
                        group_id = group_ids.get(key)
                        if group_id is None:
                            group_id = len(group_token)
                            group_ids[key] = group_id
                            group_token.append([-1] * 8)
                            opcode_group[legacy_lane][w_lane][row.opcode_map][opcode] = group_id
                        if group_token[group_id][row.modrm_extension] >= 0:
                            raise ValueError("masked opcode-group extension overlaps another row")
                        group_token[group_id][row.modrm_extension] = row.token

        flags = [0] * vocabulary_size
        immediate_bytes = [0] * vocabulary_size
        allowed_rex = [0] * vocabulary_size
        allow_rex = [0] * vocabulary_size
        required_rex = [0] * vocabulary_size
        allowed_legacy = [0] * vocabulary_size
        required_legacy = [0] * vocabulary_size
        for token, metadata in token_metadata.items():
            (
                flags[token], immediate_bytes[token], allow_rex[token], allowed_rex[token],
                required_rex[token], allowed_legacy[token], required_legacy[token],
            ) = metadata

        escape_table = [[-1] * 256 for _ in range(map_count)]
        for (source_map, octet), target_map in escapes.items():
            if not (0 <= source_map < map_count and 0 <= target_map < map_count):
                raise ValueError("escape opcode map is outside configured maps")
            if not 0 <= octet <= 0xFF:
                raise ValueError("escape opcode must be a byte")
            escape_table[source_map][octet] = target_map

        prefix_action = [int(PrefixAction.NONE)] * 256
        prefix_bit = [0] * 256
        for octet, bit in legacy.items():
            if not 0 <= octet <= 0xFF or bit <= 0:
                raise ValueError("legacy prefix entries require a byte and positive bit")
            prefix_action[octet] = int(PrefixAction.LEGACY)
            prefix_bit[octet] = bit
        if rex_prefixes:
            for octet in range(0x40, 0x50):
                prefix_action[octet] = int(PrefixAction.REX)
        for octet in forbid_prefixes:
            if not 0 <= int(octet) <= 0xFF:
                raise ValueError("forbidden prefix must be a byte")
            prefix_action[int(octet)] = int(PrefixAction.FORBIDDEN)

        def tensor(data) -> AbstractTensor:
            return AbstractTensor.get_tensor(data, dtype="int64", like=like)

        return cls(
            opcode_token=tensor(opcode_token),
            opcode_group=tensor(opcode_group),
            legacy_selector=tensor(legacy_selector),
            group_token=tensor(group_token or [[-1] * 8]),
            escape_map=tensor(escape_table),
            prefix_action=tensor(prefix_action),
            prefix_bit=tensor(prefix_bit),
            flags=tensor(flags),
            immediate_bytes=tensor(immediate_bytes),
            allowed_rex_mask=tensor(allowed_rex),
            allow_rex_prefix=tensor(allow_rex),
            required_rex_mask=tensor(required_rex),
            allowed_legacy_mask=tensor(allowed_legacy),
            required_legacy_mask=tensor(required_legacy),
            maximum_instruction_bytes=int(maximum_instruction_bytes),
            profile_name=str(profile_name),
            encoding_row_count=len(records),
        )


@dataclass(frozen=True, slots=True)
class X86ReadHeadProfile:
    """Immutable, composable source configuration for a read-head program."""

    name: str
    rows: tuple[X86EncodingRow, ...]
    escape_maps: Mapping[tuple[int, int], int] = field(
        default_factory=lambda: {
            (0, 0x0F): 1,
            (1, 0x38): 2,
            (1, 0x3A): 3,
        },
    )
    legacy_prefix_bits: Mapping[int, int] | None = None
    forbid_prefixes: tuple[int, ...] = ()
    rex_prefixes: bool = True
    maximum_instruction_bytes: int = 15
    token_names: Mapping[int, str] = field(default_factory=dict)
    source_tokens: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tokens = {int(row.token) for row in self.rows}
        unknown_names = set(map(int, self.token_names)) - tokens
        unknown_sources = set(map(int, self.source_tokens)) - tokens
        if unknown_names or unknown_sources:
            raise ValueError(
                "profile token metadata names absent rows: "
                f"names={sorted(unknown_names)!r} "
                f"sources={sorted(unknown_sources)!r}"
            )
        names = tuple(str(name) for name in self.token_names.values())
        if len(names) != len(set(names)):
            raise ValueError("profile token names must be unique")

    def token_name(self, token: int) -> str:
        """Return this code set's name without consulting a global enum."""

        return str(self.token_names.get(int(token), f"token_{int(token)}"))

    def source_token(self, token: int) -> int:
        """Return the originating vocabulary token after procedural remaps."""

        return int(self.source_tokens.get(int(token), int(token)))

    def remap(
        self,
        name: str,
        *,
        token_transform: Callable[[int], int],
        name_transform: Callable[[str], str] = lambda value: value,
    ) -> "X86ReadHeadProfile":
        """Create an isomorphic code set with procedurally replaced identity.

        Encoding grammar is copied exactly. Only emitted token identity and
        its diagnostic name change, so read, field recovery, and write remain
        a closed operation within the selected profile. This allows multiple
        otherwise identical vocabularies to coexist in disjoint token spaces.
        """

        mapping = {
            int(row.token): int(token_transform(int(row.token)))
            for row in self.rows
        }
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("token transform is not one-to-one")
        if any(token < 0 for token in mapping.values()):
            raise ValueError("remapped instruction tokens must be non-negative")
        rows = tuple(
            replace(row, token=mapping[int(row.token)]) for row in self.rows
        )
        token_names = {
            mapping[old]: str(name_transform(self.token_name(old)))
            for old in mapping
        }
        source_tokens = {
            mapping[old]: self.source_token(old) for old in mapping
        }
        return X86ReadHeadProfile(
            name=str(name),
            rows=rows,
            escape_maps=MappingProxyType(dict(self.escape_maps)),
            legacy_prefix_bits=(
                None if self.legacy_prefix_bits is None
                else MappingProxyType(dict(self.legacy_prefix_bits))
            ),
            forbid_prefixes=self.forbid_prefixes,
            rex_prefixes=self.rex_prefixes,
            maximum_instruction_bytes=self.maximum_instruction_bytes,
            token_names=MappingProxyType(token_names),
            source_tokens=MappingProxyType(source_tokens),
        )

    def namespace(
        self,
        namespace: str,
        *,
        token_base: int,
        separator: str = "::",
    ) -> "X86ReadHeadProfile":
        """Place this whole code set in a caller-selected disjoint token range."""

        ordered = sorted({int(row.token) for row in self.rows})
        dense = {token: int(token_base) + index for index, token in enumerate(ordered)}
        return self.remap(
            f"{namespace}:{self.name}",
            token_transform=dense.__getitem__,
            name_transform=lambda value: f"{namespace}{separator}{value}",
        )

    def extend(
        self,
        name: str,
        rows: Iterable[X86EncodingRow],
        *,
        escape_maps: Mapping[tuple[int, int], int] | None = None,
    ) -> "X86ReadHeadProfile":
        added_rows = tuple(rows)
        merged_escapes = dict(self.escape_maps)
        merged_escapes.update(dict(escape_maps or {}))
        return X86ReadHeadProfile(
            name=str(name),
            rows=(*self.rows, *added_rows),
            escape_maps=MappingProxyType(merged_escapes),
            legacy_prefix_bits=self.legacy_prefix_bits,
            forbid_prefixes=self.forbid_prefixes,
            rex_prefixes=self.rex_prefixes,
            maximum_instruction_bytes=self.maximum_instruction_bytes,
            token_names=MappingProxyType({
                **dict(self.token_names),
                **{
                    int(row.token): f"token_{int(row.token)}"
                    for row in added_rows
                    if int(row.token) not in self.token_names
                },
            }),
            source_tokens=MappingProxyType({
                **dict(self.source_tokens),
                **{
                    int(row.token): int(row.token)
                    for row in added_rows
                    if int(row.token) not in self.source_tokens
                },
            }),
        )

    def encode(self, token: int, fields: X86EncodingFields = X86EncodingFields()) -> bytes:
        """Invert one unambiguous row from complete architectural fields."""

        rows = tuple(row for row in self.rows if int(row.token) == int(token))
        if len(rows) != 1:
            raise ValueError(
                f"token {token} requires exactly one encoding row, found {len(rows)}"
            )
        row = rows[0]
        legacy_bits = dict({
            0xF0: 1 << 0, 0xF2: 1 << 1, 0xF3: 1 << 2,
            0x2E: 1 << 3, 0x36: 1 << 4, 0x3E: 1 << 5,
            0x26: 1 << 6, 0x64: 1 << 7, 0x65: 1 << 8,
            0x66: 1 << 9, 0x67: 1 << 10,
        } if self.legacy_prefix_bits is None else self.legacy_prefix_bits)
        if fields.legacy_prefixes is None:
            legacy_prefixes = tuple(
                octet for octet, bit in legacy_bits.items()
                if row.required_legacy_mask & bit
            )
        else:
            legacy_prefixes = fields.legacy_prefixes
        legacy_mask = 0
        for prefix in legacy_prefixes:
            bit = legacy_bits.get(int(prefix))
            if bit is None or not (row.allowed_legacy_mask & bit):
                raise ValueError(f"legacy prefix {int(prefix):#x} is forbidden for token {token}")
            legacy_mask |= bit
        if legacy_mask & ~row.allowed_legacy_mask:
            raise ValueError("legacy prefix mask exceeds encoding row")
        if row.required_legacy_mask & ~legacy_mask:
            raise ValueError("required legacy prefix is missing")
        rex = fields.rex
        if rex is None and row.required_rex_mask:
            rex = 0x40 | row.required_rex_mask
        if rex is not None:
            if not 0x40 <= int(rex) <= 0x4F or not row.allow_rex:
                raise ValueError("invalid or forbidden REX prefix")
            rex_bits = int(rex) - 0x40
            if rex_bits & ~row.allowed_rex_mask:
                raise ValueError("REX bits exceed encoding row")
            if row.required_rex_mask & ~rex_bits:
                raise ValueError("required REX bits are missing")
        elif row.required_rex_mask:
            raise ValueError("required REX prefix is missing")
        low_mask = (~row.opcode_mask) & 0xFF
        if fields.opcode_low_bits & ~low_mask:
            raise ValueError("opcode low bits exceed masked opcode field")
        opcode = (row.opcode & row.opcode_mask) | int(fields.opcode_low_bits)
        modrm = fields.modrm
        if row.has_modrm != (modrm is not None):
            raise ValueError("ModRM presence does not match encoding row")
        if modrm is not None:
            if not 0 <= int(modrm) <= 0xFF:
                raise ValueError("ModRM must be a byte")
            if row.modrm_extension is not None and (int(modrm) >> 3) & 7 != row.modrm_extension:
                raise ValueError("ModRM extension does not match encoding row")
            mod, rm = int(modrm) >> 6, int(modrm) & 7
            if row.modrm_memory_only and mod == 3:
                raise ValueError("encoding row requires a memory ModRM operand")
            needs_sib = mod != 3 and rm == 4
            if needs_sib != (fields.sib is not None):
                raise ValueError("SIB presence does not match ModRM")
            expected_displacement = 1 if mod == 1 else 4 if mod == 2 else 0
            if mod == 0 and rm == 5:
                expected_displacement = 4
            if needs_sib and mod == 0 and int(fields.sib) & 7 == 5:
                expected_displacement = 4
            if len(fields.displacement) != expected_displacement:
                raise ValueError("displacement width does not match ModRM/SIB")
        elif fields.sib is not None or fields.displacement:
            raise ValueError("operand bytes supplied for non-ModRM encoding")
        if row.immediate_bytes:
            if fields.immediate is None:
                raise ValueError("required immediate is missing")
            modulus = 1 << (row.immediate_bytes * 8)
            immediate = int(fields.immediate)
            lower = -(modulus // 2) if row.immediate_signed else 0
            upper = modulus // 2 - 1 if row.immediate_signed else modulus - 1
            if not lower <= immediate <= upper:
                raise ValueError("immediate is outside encoding width")
            immediate_bytes = (immediate % modulus).to_bytes(row.immediate_bytes, "little")
        else:
            if fields.immediate is not None:
                raise ValueError("unexpected immediate")
            immediate_bytes = b""
        map_prefix = {
            0: b"", 1: b"\x0f", 2: b"\x0f\x38", 3: b"\x0f\x3a",
        }.get(row.opcode_map)
        if map_prefix is None:
            raise ValueError(f"opcode map {row.opcode_map} has no inverse byte path")
        encoded = b"".join((
            bytes(legacy_prefixes),
            b"" if rex is None else bytes((int(rex),)),
            map_prefix, bytes((opcode,)),
            b"" if modrm is None else bytes((int(modrm),)),
            b"" if fields.sib is None else bytes((int(fields.sib),)),
            bytes(fields.displacement), immediate_bytes,
        ))
        if len(encoded) > self.maximum_instruction_bytes:
            raise ValueError("encoded instruction exceeds maximum length")
        return encoded

    def fields_from_encoded(self, token: int, encoded: bytes) -> X86EncodingFields:
        """Recover every variable field for an exact write-head round trip."""

        rows = tuple(row for row in self.rows if int(row.token) == int(token))
        if len(rows) != 1:
            raise ValueError(
                f"token {token} requires exactly one encoding row, found {len(rows)}"
            )
        row = rows[0]
        legacy_bits = dict({
            0xF0: 1 << 0, 0xF2: 1 << 1, 0xF3: 1 << 2,
            0x2E: 1 << 3, 0x36: 1 << 4, 0x3E: 1 << 5,
            0x26: 1 << 6, 0x64: 1 << 7, 0x65: 1 << 8,
            0x66: 1 << 9, 0x67: 1 << 10,
        } if self.legacy_prefix_bits is None else self.legacy_prefix_bits)
        cursor = 0
        prefixes: list[int] = []
        while cursor < len(encoded) and encoded[cursor] in legacy_bits:
            prefixes.append(encoded[cursor])
            cursor += 1
        rex = None
        if cursor < len(encoded) and 0x40 <= encoded[cursor] <= 0x4F:
            rex = encoded[cursor]
            cursor += 1
        map_prefix = {
            0: b"", 1: b"\x0f", 2: b"\x0f\x38", 3: b"\x0f\x3a",
        }.get(row.opcode_map)
        if map_prefix is None:
            raise ValueError(f"opcode map {row.opcode_map} has no inverse byte path")
        if encoded[cursor:cursor + len(map_prefix)] != map_prefix:
            raise ValueError("encoded opcode map does not match row")
        cursor += len(map_prefix)
        if cursor >= len(encoded):
            raise ValueError("encoded instruction has no opcode")
        opcode = encoded[cursor]
        cursor += 1
        if opcode & row.opcode_mask != row.opcode & row.opcode_mask:
            raise ValueError("encoded opcode does not match row")
        opcode_low_bits = opcode & ((~row.opcode_mask) & 0xFF)
        modrm = None
        sib = None
        displacement = b""
        if row.has_modrm:
            if cursor >= len(encoded):
                raise ValueError("encoded instruction has no ModRM")
            modrm = encoded[cursor]
            cursor += 1
            mod, rm = modrm >> 6, modrm & 7
            if mod != 3 and rm == 4:
                if cursor >= len(encoded):
                    raise ValueError("encoded instruction has no SIB")
                sib = encoded[cursor]
                cursor += 1
            displacement_width = 1 if mod == 1 else 4 if mod == 2 else 0
            if mod == 0 and rm == 5:
                displacement_width = 4
            if sib is not None and mod == 0 and sib & 7 == 5:
                displacement_width = 4
            if cursor + displacement_width > len(encoded):
                raise ValueError("encoded instruction has truncated displacement")
            displacement = bytes(encoded[cursor:cursor + displacement_width])
            cursor += displacement_width
        immediate = None
        if row.immediate_bytes:
            end = cursor + row.immediate_bytes
            if end > len(encoded):
                raise ValueError("encoded instruction has truncated immediate")
            immediate = int.from_bytes(
                encoded[cursor:end], "little", signed=row.immediate_signed,
            )
            cursor = end
        if cursor != len(encoded):
            raise ValueError("encoded instruction contains trailing bytes")
        fields = X86EncodingFields(
            rex=rex,
            legacy_prefixes=tuple(prefixes),
            opcode_low_bits=opcode_low_bits,
            modrm=modrm,
            sib=sib,
            displacement=displacement,
            immediate=immediate,
        )
        if self.encode(token, fields) != encoded:
            raise ValueError("encoding fields do not exactly reproduce instruction")
        return fields

    def fields_from_operands(
        self,
        token: int,
        operands: Sequence[object],
        *,
        address: int | None = None,
    ) -> X86EncodingFields:
        """Encode explicit AMD64 operand placement into architectural fields.

        This is the allocation boundary: it consumes physical registers and
        effective addresses, never SSA value names or inferred placement.
        """

        from .machine_reference_vocabulary import (
            EffectiveAddressOperand, ImmediateOperand, RegisterOperand,
            RelativeAddressOperand, VectorRegisterOperand, X86InstructionToken,
        )

        try:
            kind = X86InstructionToken(self.source_token(int(token)))
        except ValueError as error:
            raise ValueError(
                f"{self.token_name(token)} has no allocated-operand writer"
            ) from error
        rows = tuple(row for row in self.rows if row.token == int(token))
        if len(rows) != 1:
            raise ValueError(f"token {token} requires one encoding row")
        row = rows[0]
        display_name = self.token_name(token)
        form = X86OperandForm(row.operand_form)
        items = tuple(operands)
        if form is X86OperandForm.NONE:
            if items:
                raise ValueError(f"{display_name} has no allocated operands")
            return X86EncodingFields()
        if form is X86OperandForm.REG_RM:
            if len(items) != 2 or not isinstance(items[0], (RegisterOperand, VectorRegisterOperand)) or not isinstance(items[1], (RegisterOperand, VectorRegisterOperand, EffectiveAddressOperand)):
                raise ValueError(f"{display_name} requires register, register/memory operands")
            if kind is X86InstructionToken.LEA_R32_M and not isinstance(items[1], EffectiveAddressOperand):
                raise ValueError("LEA requires an effective-address source")
            return self._modrm_fields(token, items[0], items[1])
        if form is X86OperandForm.RM_REG:
            if len(items) != 2 or not isinstance(items[0], (RegisterOperand, EffectiveAddressOperand)) or not isinstance(items[1], RegisterOperand):
                raise ValueError(f"{display_name} requires register/memory, register operands")
            return self._modrm_fields(token, items[1], items[0])
        if form is X86OperandForm.RM_IMMEDIATE:
            if len(items) != 2 or not isinstance(items[0], (RegisterOperand, VectorRegisterOperand, EffectiveAddressOperand)) or not isinstance(items[1], ImmediateOperand):
                raise ValueError(f"{display_name} requires register/memory, immediate operands")
            fields = self._modrm_fields(token, None, items[0])
            return X86EncodingFields(
                rex=fields.rex, legacy_prefixes=fields.legacy_prefixes,
                opcode_low_bits=fields.opcode_low_bits, modrm=fields.modrm,
                sib=fields.sib, displacement=fields.displacement,
                immediate=int(items[1].value),
            )
        if form is X86OperandForm.RM:
            if len(items) != 1 or not isinstance(
                items[0], (RegisterOperand, VectorRegisterOperand, EffectiveAddressOperand)
            ):
                raise ValueError(f"{display_name} requires one register/memory operand")
            return self._modrm_fields(token, None, items[0])
        if form is X86OperandForm.REG_RM_IMMEDIATE:
            if len(items) != 3 or not isinstance(
                items[0], (RegisterOperand, VectorRegisterOperand)
            ) or not isinstance(
                items[1], (RegisterOperand, VectorRegisterOperand, EffectiveAddressOperand)
            ) or not isinstance(items[2], ImmediateOperand):
                raise ValueError(
                    f"{display_name} requires register, register/memory, immediate operands"
                )
            fields = self._modrm_fields(token, items[0], items[1])
            return X86EncodingFields(
                rex=fields.rex, legacy_prefixes=fields.legacy_prefixes,
                opcode_low_bits=fields.opcode_low_bits, modrm=fields.modrm,
                sib=fields.sib, displacement=fields.displacement,
                immediate=int(items[2].value),
            )
        if form is X86OperandForm.IMMEDIATE:
            immediates = tuple(item for item in items if isinstance(item, ImmediateOperand))
            if len(immediates) != 1:
                raise ValueError(f"{display_name} requires one encoded immediate operand")
            return X86EncodingFields(immediate=int(immediates[0].value))
        if form is X86OperandForm.RELATIVE:
            if len(items) != 1 or not isinstance(items[0], RelativeAddressOperand):
                raise ValueError(f"{display_name} requires one relative operand")
            displacement = int(items[0].displacement)
            if address is not None:
                rows = tuple(row for row in self.rows if row.token == int(token))
                if len(rows) != 1:
                    raise ValueError("relative token needs one encoding row")
                length = len(self.encode(token, X86EncodingFields(immediate=0)))
                displacement = int(items[0].target_address) - (int(address) + length)
            return X86EncodingFields(immediate=displacement)
        if form is X86OperandForm.OPCODE_REGISTER:
            if len(items) != 1 or not isinstance(items[0], RegisterOperand):
                raise ValueError(f"{display_name} requires one opcode-selected register")
            code = int(items[0].register)
            return X86EncodingFields(
                rex=(0x41 if code >= 8 else None), opcode_low_bits=code & 7,
            )
        if form is X86OperandForm.OPCODE_REGISTER_IMMEDIATE:
            if len(items) != 2 or not isinstance(items[0], RegisterOperand) or not isinstance(items[1], ImmediateOperand):
                raise ValueError(f"{display_name} requires register, immediate operands")
            code = int(items[0].register)
            rex = (
                0x40 | row.required_rex_mask | (1 if code >= 8 else 0)
                if row.required_rex_mask or code >= 8 else None
            )
            return X86EncodingFields(
                rex=rex,
                opcode_low_bits=code & 7, immediate=int(items[1].value),
            )
        raise ValueError(f"{display_name} has no allocated-operand writer")

    def _modrm_fields(self, token: int, reg_operand, rm_operand) -> X86EncodingFields:
        from .machine_reference_vocabulary import EffectiveAddressOperand

        rows = tuple(row for row in self.rows if row.token == int(token))
        if len(rows) != 1:
            raise ValueError(f"token {token} requires one encoding row")
        row = rows[0]
        rex_bits = row.required_rex_mask
        reg_code = row.modrm_extension if reg_operand is None else int(reg_operand.register)
        assert reg_code is not None
        if reg_code >= 8:
            rex_bits |= 0x04
        reg = reg_code & 7
        sib = None
        displacement = b""
        if not isinstance(rm_operand, EffectiveAddressOperand):
            rm_code = int(rm_operand.register)
            if rm_code >= 8:
                rex_bits |= 0x01
            modrm = 0xC0 | (reg << 3) | (rm_code & 7)
        else:
            base_code = None if rm_operand.base is None else int(rm_operand.base)
            index_code = None if rm_operand.index is None else int(rm_operand.index)
            if rm_operand.scale not in (1, 2, 4, 8):
                raise ValueError("effective-address scale must be 1, 2, 4, or 8")
            if index_code is not None and index_code & 7 == 4 and index_code < 8:
                raise ValueError("RSP cannot be an effective-address index")
            value = int(rm_operand.displacement)
            if rm_operand.rip_relative:
                if base_code is not None or index_code is not None:
                    raise ValueError("RIP-relative address cannot have base or index")
                mod, rm = 0, 5
                displacement = value.to_bytes(4, "little", signed=True)
            else:
                needs_sib = base_code is None or index_code is not None or (base_code & 7) == 4
                if base_code is None:
                    mod = 0
                    displacement = value.to_bytes(4, "little", signed=True)
                elif value == 0 and (base_code & 7) != 5:
                    mod = 0
                elif -128 <= value <= 127:
                    mod = 1
                    displacement = value.to_bytes(1, "little", signed=True)
                else:
                    mod = 2
                    displacement = value.to_bytes(4, "little", signed=True)
                rm = 4 if needs_sib else base_code & 7
                if needs_sib:
                    scale_bits = {1: 0, 2: 1, 4: 2, 8: 3}[rm_operand.scale]
                    index_bits = 4 if index_code is None else index_code & 7
                    base_bits = 5 if base_code is None else base_code & 7
                    sib = (scale_bits << 6) | (index_bits << 3) | base_bits
                    if index_code is not None and index_code >= 8:
                        rex_bits |= 0x02
                    if base_code is not None and base_code >= 8:
                        rex_bits |= 0x01
                elif base_code is not None and base_code >= 8:
                    rex_bits |= 0x01
            modrm = (mod << 6) | (reg << 3) | rm
        rex = 0x40 | rex_bits if rex_bits or row.required_rex_mask else None
        return X86EncodingFields(
            rex=rex, modrm=modrm, sib=sib, displacement=displacement,
        )

    def compile(self, *, like: AbstractTensor | None = None) -> X86ReadHeadConfig:
        return X86ReadHeadConfig.from_rows(
            self.rows,
            escape_maps=dict(self.escape_maps),
            legacy_prefix_bits=(
                None if self.legacy_prefix_bits is None
                else dict(self.legacy_prefix_bits)
            ),
            forbid_prefixes=self.forbid_prefixes,
            rex_prefixes=self.rex_prefixes,
            maximum_instruction_bytes=self.maximum_instruction_bytes,
            profile_name=self.name,
            like=like,
        )


@dataclass(frozen=True, slots=True)
class X86ReadHeadCodeSetBank:
    """Several profile-insistent read/write heads with disjoint identities.

    A bank does not merge opcode tables: identical bytes may intentionally
    mean different tokens in different code sets. The caller selects the code
    set for each stream, and every operation is then performed exclusively by
    that profile's tensor tables and inverse grammar.
    """

    profiles: Mapping[str, X86ReadHeadProfile]

    def __post_init__(self) -> None:
        normalized = {str(name): profile for name, profile in self.profiles.items()}
        if not normalized:
            raise ValueError("code-set bank cannot be empty")
        owners: dict[int, str] = {}
        for name, profile in normalized.items():
            for row in profile.rows:
                token = int(row.token)
                previous = owners.setdefault(token, name)
                if previous != name:
                    raise ValueError(
                        f"token {token} is shared by code sets "
                        f"{previous!r} and {name!r}; namespace them first"
                    )
        object.__setattr__(self, "profiles", MappingProxyType(normalized))

    def profile(self, code_set: str) -> X86ReadHeadProfile:
        try:
            return self.profiles[str(code_set)]
        except KeyError as error:
            raise KeyError(f"unknown x86 code set {code_set!r}") from error

    def head(
        self,
        code_set: str,
        *,
        like: AbstractTensor | None = None,
    ) -> "X86TensorReadHead":
        return X86TensorReadHead.from_profile(
            self.profile(code_set), like=like,
        )

    def encode(
        self,
        code_set: str,
        token: int,
        fields: X86EncodingFields = X86EncodingFields(),
    ) -> bytes:
        return self.profile(code_set).encode(token, fields)

    def rewrite_instruction(
        self, code_set: str, token: int, encoded: bytes,
    ) -> bytes:
        return self.head(code_set).rewrite_instruction(token, encoded)

    def run(
        self,
        batches: Mapping[
            str, "X86ReadBatch | X86PreparedReadBatch"
        ],
        *,
        mode: ReadHeadExecutionMode = ReadHeadExecutionMode.DECODE,
        maximum_microsteps: int = 1_000_000,
    ) -> Mapping[str, "X86ReadHeadRunResult"]:
        """Advance several explicitly selected code sets in one orchestration.

        Each stream retains its own tensor lookup tables and emitted token
        namespace. No byte grammar is merged and no global token enum selects
        a meaning on behalf of the supplied profile.
        """

        requested = {str(name): batch for name, batch in batches.items()}
        unknown = set(requested) - set(self.profiles)
        if unknown:
            raise KeyError(f"unknown x86 code sets {sorted(unknown)!r}")
        return MappingProxyType({
            name: self.head(name).run(
                batch, mode=mode, maximum_microsteps=maximum_microsteps,
            )
            for name, batch in requested.items()
        })

    def token_owner(self, token: int) -> str:
        matches = tuple(
            name for name, profile in self.profiles.items()
            if any(int(row.token) == int(token) for row in profile.rows)
        )
        if len(matches) != 1:
            raise KeyError(f"token {token} has no unique code-set owner")
        return matches[0]

@dataclass(frozen=True, slots=True)
class X86ReadBatch:
    """Padded byte lanes plus their actual lengths and virtual origins."""

    octets: AbstractTensor
    valid_lengths: AbstractTensor
    base_addresses: AbstractTensor

    def __post_init__(self) -> None:
        if self.octets.ndims() != 2:
            raise ValueError("x86 read batch octets must have shape (lanes, capacity)")
        lanes = int(self.octets.shape[0])
        if int(self.octets.shape[1]) <= 0:
            raise ValueError("x86 read batch capacity must be positive")
        if tuple(self.valid_lengths.shape) != (lanes,):
            raise ValueError("valid_lengths must have one value per lane")
        if tuple(self.base_addresses.shape) != (lanes,):
            raise ValueError("base_addresses must have one value per lane")
        valid_octets = (
            ((self.octets % 1) == 0)
            & (self.octets >= 0)
            & (self.octets <= 0xFF)
        )
        if not bool(valid_octets.all().item()):
            raise ValueError("x86 read batch contains a non-byte value")
        capacity = int(self.octets.shape[1])
        lengths_valid = (
            ((self.valid_lengths % 1) == 0)
            & (self.valid_lengths >= 0)
            & (self.valid_lengths <= capacity)
        )
        if not bool(lengths_valid.all().item()):
            raise ValueError("valid_lengths is outside the padded lane capacity")

    def prepare(self) -> "X86PreparedReadBatch":
        """Cache invariant gathers used by every tensor microstep."""

        lanes, capacity = (int(size) for size in self.octets.shape)
        lane = AbstractTensor.arange(lanes, cls=type(self.octets)).to_dtype("int64")
        return X86PreparedReadBatch(
            octets=self.octets,
            valid_lengths=self.valid_lengths,
            base_addresses=self.base_addresses,
            flat_octets=self.octets.reshape((-1,)),
            lane_offsets=lane * capacity,
        )


@dataclass(frozen=True, slots=True)
class X86PreparedReadBatch:
    """A read batch with shape-invariant indexing tensors precomputed once."""

    octets: AbstractTensor
    valid_lengths: AbstractTensor
    base_addresses: AbstractTensor
    flat_octets: AbstractTensor
    lane_offsets: AbstractTensor


@dataclass(frozen=True, slots=True)
class X86ReadHeadRunResult:
    final_state: "X86ReadHeadState"
    emission_states: tuple["X86ReadHeadState", ...]
    microsteps: int
    mode: ReadHeadExecutionMode


@dataclass(frozen=True, slots=True)
class X86ReadHeadState:
    """All mutable decoder state, one scalar per parallel read lane."""

    cursor: AbstractTensor
    instruction_start: AbstractTensor
    phase: AbstractTensor
    status: AbstractTensor
    failure: AbstractTensor
    token: AbstractTensor
    opcode_map: AbstractTensor
    opcode: AbstractTensor
    rex: AbstractTensor
    rex_present: AbstractTensor
    legacy_prefixes: AbstractTensor
    modrm: AbstractTensor
    sib: AbstractTensor
    displacement: AbstractTensor
    immediate: AbstractTensor
    relative_target: AbstractTensor
    field_accumulator: AbstractTensor
    field_multiplier: AbstractTensor
    field_remaining: AbstractTensor
    field_width: AbstractTensor

    REGISTER_NAMES: ClassVar[tuple[str, ...]] = (
        "cursor", "instruction_start", "phase", "status", "failure",
        "token", "opcode_map", "opcode", "rex", "rex_present",
        "legacy_prefixes", "modrm", "sib", "displacement", "immediate",
        "relative_target", "field_accumulator", "field_multiplier",
        "field_remaining", "field_width",
    )

    def register_tensor(self) -> AbstractTensor:
        """Return every mutable head register as ``(cores, registers)``.

        This is the stable observation ABI for debuggers and shader-backed
        displays.  It deliberately remains an AbstractTensor operation so a
        captured read-head graph can publish register state without a host
        round trip.
        """

        return AbstractTensor.stack(
            [getattr(self, name) for name in self.REGISTER_NAMES], dim=1,
        )

    def register_contents(self) -> tuple[dict[str, int], ...]:
        """Materialize named per-core values for host-side examination."""

        rows = self.register_tensor().tolist()
        return tuple(
            {name: int(value) for name, value in zip(self.REGISTER_NAMES, row)}
            for row in rows
        )

    @classmethod
    def initial(cls, batch: X86ReadBatch) -> "X86ReadHeadState":
        lanes = int(batch.octets.shape[0])
        zero = AbstractTensor.zeros((lanes,), dtype="int64", cls=type(batch.octets))
        negative = zero - 1
        # Every field below starts at the same numeric value (0 or -1), but
        # each is an independent state slot that later diverges under its
        # own _select(...) updates.  Handing multiple fields a reference to
        # the *same* tensor object -- rather than each its own tensor
        # holding that value -- makes them indistinguishable primitives to
        # anything that tracks value identity (the AOT discovery capture in
        # particular), which then can't tell "two fields that start equal"
        # from "one field read twice".  Cloning gives each field its own
        # object without changing what value it starts at.
        return cls(
            cursor=zero.clone(),
            instruction_start=zero.clone(),
            phase=zero + int(ReadPhase.PREFIX),
            status=zero + int(ReadStatus.ACTIVE),
            failure=zero + int(ReadFailure.NONE),
            token=negative.clone(),
            opcode_map=zero.clone(),
            opcode=negative.clone(),
            rex=zero.clone(),
            rex_present=zero.clone(),
            legacy_prefixes=zero.clone(),
            modrm=negative.clone(),
            sib=negative.clone(),
            displacement=zero.clone(),
            immediate=zero.clone(),
            relative_target=zero.clone(),
            field_accumulator=zero.clone(),
            field_multiplier=zero + 1,
            field_remaining=zero.clone(),
            field_width=zero.clone(),
        )


def _select(mask: AbstractTensor, yes, no):
    return AbstractTensor.where(mask, yes, no)


def _table_1d(table: AbstractTensor, index: AbstractTensor) -> AbstractTensor:
    return table.gather(index.to_dtype("int64"), dim=0)


def _table_2d(
    table: AbstractTensor,
    row: AbstractTensor,
    column: AbstractTensor,
) -> AbstractTensor:
    width = int(table.shape[1])
    flat_index = row.to_dtype("int64") * width + column.to_dtype("int64")
    return table.reshape((-1,)).gather(flat_index, dim=0)


def _bit_is_set(value: AbstractTensor, bit: int) -> AbstractTensor:
    """Integer bit test expressed through portable AbstractTensor arithmetic."""

    return ((value // int(bit)) % 2) > 0


def _add_mask_bit(mask: AbstractTensor, bit: AbstractTensor) -> AbstractTensor:
    """Union one power-of-two bit without relying on logical ``__or__``."""

    present = ((mask // bit.maximum(1)) % 2) > 0
    return mask + _select(present, 0, bit)


def _mask_has_forbidden_bits(
    observed: AbstractTensor,
    allowed: AbstractTensor,
    bits: tuple[int, ...],
) -> AbstractTensor:
    invalid = observed < 0
    for bit in bits:
        invalid = invalid | (
            _bit_is_set(observed, bit) & (_bit_is_set(allowed, bit) == 0)
        )
    return invalid


def _mask_missing_required_bits(
    observed: AbstractTensor,
    required: AbstractTensor,
    bits: tuple[int, ...],
) -> AbstractTensor:
    missing = observed < 0
    for bit in bits:
        missing = missing | (
            _bit_is_set(required, bit) & (_bit_is_set(observed, bit) == 0)
        )
    return missing


class X86TensorReadHead:
    """Pure tensor transition kernel for parallel variable-length x86 reads."""

    def __init__(
        self,
        config: X86ReadHeadConfig,
        *,
        profile: X86ReadHeadProfile | None = None,
    ) -> None:
        self.config = config
        self.profile = profile
        # Reshape is metadata work but some backends materialize it. Cache all
        # immutable table views once instead of rebuilding them per microstep.
        self._opcode_token_flat = config.opcode_token.reshape((-1,))
        self._opcode_group_flat = config.opcode_group.reshape((-1,))
        self._legacy_selector_flat = config.legacy_selector.reshape((-1,))
        self._group_token_flat = config.group_token.reshape((-1,))
        self._escape_map_flat = config.escape_map.reshape((-1,))

    @classmethod
    def from_profile(
        cls,
        profile: X86ReadHeadProfile,
        *,
        like: AbstractTensor | None = None,
    ) -> "X86TensorReadHead":
        """Compile one shared source profile into a bidirectional head."""

        return cls(profile.compile(like=like), profile=profile)

    def write_instruction(
        self,
        token: int,
        fields: X86EncodingFields = X86EncodingFields(),
    ) -> bytes:
        """Emit one instruction through the same profile used for reading."""

        if self.profile is None:
            raise RuntimeError("write_instruction requires a source read-head profile")
        return self.profile.encode(token, fields)

    def encoding_fields(self, token: int, encoded: bytes) -> X86EncodingFields:
        """Recover fields using this head's own inverse encoding table."""

        if self.profile is None:
            raise RuntimeError("encoding_fields requires a source read-head profile")
        return self.profile.fields_from_encoded(token, encoded)

    def rewrite_instruction(self, token: int, encoded: bytes) -> bytes:
        """Decode framing fields and write them back without a byte template."""

        return self.write_instruction(token, self.encoding_fields(token, encoded))

    def write_allocated(self, instruction: X86AllocatedInstruction) -> bytes:
        """Write a selected instruction after explicit physical allocation."""

        if self.profile is None:
            raise RuntimeError("write_allocated requires a source read-head profile")
        fields = self.profile.fields_from_operands(
            instruction.token, instruction.operands, address=instruction.address,
        )
        return self.write_instruction(instruction.token, fields)

    def _peek(
        self,
        batch: X86ReadBatch | X86PreparedReadBatch,
        state: X86ReadHeadState,
    ) -> tuple[AbstractTensor, AbstractTensor]:
        lanes, capacity = (int(size) for size in batch.octets.shape)
        in_bounds = state.cursor < batch.valid_lengths
        safe_cursor = state.cursor.maximum(0).minimum(max(0, capacity - 1))
        if isinstance(batch, X86PreparedReadBatch):
            flat_octets = batch.flat_octets
            lane_offsets = batch.lane_offsets
        else:
            lane = AbstractTensor.arange(lanes, cls=type(batch.octets))
            flat_octets = batch.octets.reshape((-1,))
            lane_offsets = lane * capacity
        flat_index = lane_offsets + safe_cursor
        octet = flat_octets.gather(
            flat_index.to_dtype("int64"), dim=0,
        )
        return octet.to_dtype("int64"), in_bounds

    def transition(
        self,
        batch: X86ReadBatch | X86PreparedReadBatch,
        state: X86ReadHeadState,
    ) -> X86ReadHeadState:
        """Advance every active lane by one state-machine microstep."""

        byte, in_bounds = self._peek(batch, state)
        active = state.status == int(ReadStatus.ACTIVE)
        needs_byte = active & (state.phase != int(ReadPhase.EMIT))
        truncated = needs_byte & ~in_bounds
        status = _select(truncated, int(ReadStatus.FAILED), state.status)
        failure = _select(truncated, int(ReadFailure.TRUNCATED), state.failure)
        live = active & in_bounds

        cursor = state.cursor
        instruction_start = state.instruction_start
        phase = state.phase
        token = state.token
        opcode_map = state.opcode_map
        opcode = state.opcode
        rex = state.rex
        rex_present = state.rex_present
        legacy = state.legacy_prefixes
        modrm = state.modrm
        sib = state.sib
        displacement = state.displacement
        immediate = state.immediate
        relative_target = state.relative_target
        accumulator = state.field_accumulator
        multiplier = state.field_multiplier
        remaining = state.field_remaining
        field_width = state.field_width

        prefix_lane = live & (phase == int(ReadPhase.PREFIX))
        prefix_action = _table_1d(self.config.prefix_action, byte)
        legacy_lane = prefix_lane & (prefix_action == int(PrefixAction.LEGACY))
        rex_lane = prefix_lane & (prefix_action == int(PrefixAction.REX))
        forbidden_lane = prefix_lane & (prefix_action == int(PrefixAction.FORBIDDEN))
        misordered_prefix = prefix_lane & (rex_present > 0) & (legacy_lane | rex_lane)
        opcode_ready = prefix_lane & (prefix_action == int(PrefixAction.NONE))
        cursor = _select(legacy_lane | rex_lane, cursor + 1, cursor)
        legacy = _select(
            legacy_lane,
            _add_mask_bit(legacy, _table_1d(self.config.prefix_bit, byte)),
            legacy,
        )
        rex = _select(rex_lane, byte - 0x40, rex)
        rex_present = _select(rex_lane, 1, rex_present)
        phase = _select(opcode_ready, int(ReadPhase.OPCODE), phase)
        status = _select(
            forbidden_lane | misordered_prefix, int(ReadStatus.FAILED), status,
        )
        failure = _select(
            forbidden_lane | misordered_prefix,
            int(ReadFailure.FORBIDDEN_PREFIX),
            failure,
        )

        opcode_lane = live & (state.phase == int(ReadPhase.OPCODE))
        next_map = self._table_2d_flat(
            self._escape_map_flat, self.config.escape_map, opcode_map, byte,
        )
        escape_lane = opcode_lane & (next_map >= 0)
        opcode_map = _select(escape_lane, next_map, opcode_map)
        cursor = _select(escape_lane, cursor + 1, cursor)
        final_opcode_lane = opcode_lane & ~escape_lane
        rex_w = _bit_is_set(rex, 0x08) & (rex_present > 0)
        legacy_lane_index = _table_1d(
            self.config.legacy_selector, legacy.maximum(0),
        )
        known_legacy = legacy_lane_index >= 0
        safe_legacy_lane = legacy_lane_index.maximum(0)
        direct_token = self._table_4d_flat(
            self._opcode_token_flat, self.config.opcode_token,
            safe_legacy_lane, rex_w, opcode_map, byte,
        )
        group_id = self._table_4d_flat(
            self._opcode_group_flat, self.config.opcode_group,
            safe_legacy_lane, rex_w, opcode_map, byte,
        )
        known_opcode = known_legacy & ((direct_token >= 0) | (group_id >= 0))
        # Preserve the distinction between an unknown opcode and a known
        # opcode selected under a different mandatory legacy-prefix lane.
        known_in_other_legacy_lane = known_opcode
        for lane in range(int(self.config.opcode_token.shape[0])):
            lane_index = safe_legacy_lane * 0 + lane
            alternate_direct = self._table_4d_flat(
                self._opcode_token_flat, self.config.opcode_token,
                lane_index, rex_w, opcode_map, byte,
            )
            alternate_group = self._table_4d_flat(
                self._opcode_group_flat, self.config.opcode_group,
                lane_index, rex_w, opcode_map, byte,
            )
            known_in_other_legacy_lane = known_in_other_legacy_lane | (
                (alternate_direct >= 0) | (alternate_group >= 0)
            )
        legacy_selection_failure = (
            final_opcode_lane & ~known_opcode & known_in_other_legacy_lane
        )
        unknown_opcode = (
            final_opcode_lane & ~known_opcode & ~known_in_other_legacy_lane
        )
        opcode = _select(final_opcode_lane, byte, opcode)
        cursor = _select(final_opcode_lane & known_opcode, cursor + 1, cursor)
        token = _select(final_opcode_lane & (direct_token >= 0), direct_token, token)
        direct_phase = self._phase_after_token(direct_token)
        phase = _select(
            final_opcode_lane & known_opcode,
            _select(
                group_id >= 0,
                int(ReadPhase.MODRM),
                direct_phase,
            ),
            phase,
        )
        direct_immediate = (
            final_opcode_lane
            & (direct_token >= 0)
            & (direct_phase == int(ReadPhase.IMMEDIATE))
        )
        direct_immediate_width = _table_1d(
            self.config.immediate_bytes, direct_token.maximum(0),
        )
        remaining = _select(direct_immediate, direct_immediate_width, remaining)
        field_width = _select(direct_immediate, direct_immediate_width, field_width)
        accumulator = _select(direct_immediate, 0, accumulator)
        multiplier = _select(direct_immediate, 1, multiplier)
        direct_bad_prefix, direct_missing_prefix = self._prefix_errors(
            direct_token, rex, rex_present, legacy,
        )
        direct_prefix_failure = final_opcode_lane & (direct_token >= 0) & (
            direct_bad_prefix | direct_missing_prefix
        )
        status = _select(direct_prefix_failure, int(ReadStatus.FAILED), status)
        failure = _select(
            final_opcode_lane & (direct_token >= 0) & direct_bad_prefix,
            int(ReadFailure.FORBIDDEN_PREFIX),
            failure,
        )
        failure = _select(
            final_opcode_lane & (direct_token >= 0) & direct_missing_prefix,
            int(ReadFailure.REQUIRED_PREFIX_MISSING),
            failure,
        )
        status = _select(
            legacy_selection_failure, int(ReadStatus.FAILED), status,
        )
        failure = _select(
            legacy_selection_failure,
            int(ReadFailure.REQUIRED_PREFIX_MISSING), failure,
        )
        status = _select(unknown_opcode, int(ReadStatus.FAILED), status)
        failure = _select(unknown_opcode, int(ReadFailure.UNKNOWN_OPCODE), failure)

        modrm_lane = live & (state.phase == int(ReadPhase.MODRM))
        state_rex_w = _bit_is_set(state.rex, 0x08) & (state.rex_present > 0)
        state_legacy_lane = _table_1d(
            self.config.legacy_selector, state.legacy_prefixes.maximum(0),
        ).maximum(0)
        group_for_opcode = self._table_4d_flat(
            self._opcode_group_flat, self.config.opcode_group,
            state_legacy_lane, state_rex_w,
            state.opcode_map, state.opcode.maximum(0),
        )
        extension = (byte // 8) % 8
        group_token = self._table_2d_flat(
            self._group_token_flat, self.config.group_token,
            group_for_opcode.maximum(0),
            extension,
        )
        resolving_group = modrm_lane & (state.token < 0)
        missing_group = resolving_group & (group_token < 0)
        resolved_token = _select(resolving_group, group_token, state.token)
        token = _select(modrm_lane & ~missing_group, resolved_token, token)
        modrm = _select(modrm_lane & ~missing_group, byte, modrm)
        cursor = _select(modrm_lane & ~missing_group, cursor + 1, cursor)
        mod = byte // 64
        rm = byte % 8
        memory_only = _bit_is_set(
            _table_1d(self.config.flags, resolved_token.maximum(0)),
            int(EncodingFlag.MODRM_MEMORY_ONLY),
        )
        invalid_modrm = modrm_lane & ~missing_group & memory_only & (mod == 3)
        status = _select(invalid_modrm, int(ReadStatus.FAILED), status)
        failure = _select(
            invalid_modrm, int(ReadFailure.INVALID_MODRM), failure,
        )
        has_sib = (mod != 3) & (rm == 4)
        displacement_width = _select(
            mod == 1,
            1,
            _select((mod == 2) | ((mod == 0) & (rm == 5)), 4, 0),
        )
        after_operands = self._phase_after_operands(resolved_token)
        post_modrm_phase = _select(
            has_sib,
            int(ReadPhase.SIB),
            _select(
                displacement_width > 0,
                int(ReadPhase.DISPLACEMENT),
                after_operands,
            ),
        )
        valid_modrm = modrm_lane & ~missing_group & ~invalid_modrm
        phase = _select(valid_modrm, post_modrm_phase, phase)
        remaining = _select(
            valid_modrm & ~has_sib & (displacement_width > 0),
            displacement_width,
            remaining,
        )
        field_width = _select(
            valid_modrm & ~has_sib & (displacement_width > 0),
            displacement_width,
            field_width,
        )
        accumulator = _select(modrm_lane, 0, accumulator)
        multiplier = _select(modrm_lane, 1, multiplier)
        group_immediate = (
            modrm_lane
            & ~missing_group
            & ~has_sib
            & (displacement_width == 0)
            & (after_operands == int(ReadPhase.IMMEDIATE))
        )
        group_immediate_width = _table_1d(
            self.config.immediate_bytes, resolved_token.maximum(0),
        )
        remaining = _select(group_immediate, group_immediate_width, remaining)
        field_width = _select(group_immediate, group_immediate_width, field_width)
        group_bad_prefix, group_missing_prefix = self._prefix_errors(
            resolved_token, rex, rex_present, legacy,
        )
        group_prefix_failure = modrm_lane & ~missing_group & (
            group_bad_prefix | group_missing_prefix
        )
        status = _select(group_prefix_failure, int(ReadStatus.FAILED), status)
        failure = _select(
            modrm_lane & ~missing_group & group_bad_prefix,
            int(ReadFailure.FORBIDDEN_PREFIX),
            failure,
        )
        failure = _select(
            modrm_lane & ~missing_group & group_missing_prefix,
            int(ReadFailure.REQUIRED_PREFIX_MISSING),
            failure,
        )
        status = _select(missing_group, int(ReadStatus.FAILED), status)
        failure = _select(
            missing_group, int(ReadFailure.UNKNOWN_OPCODE_GROUP), failure,
        )

        sib_lane = live & (state.phase == int(ReadPhase.SIB))
        sib = _select(sib_lane, byte, sib)
        cursor = _select(sib_lane, cursor + 1, cursor)
        sib_base = byte % 8
        sib_disp_width = _select(
            ((state.modrm // 64) == 0) & (sib_base == 5), 4, 0,
        )
        after_sib = self._phase_after_operands(state.token)
        phase = _select(
            sib_lane,
            _select(
                sib_disp_width > 0,
                int(ReadPhase.DISPLACEMENT),
                after_sib,
            ),
            phase,
        )
        remaining = _select(sib_lane, sib_disp_width, remaining)
        field_width = _select(sib_lane, sib_disp_width, field_width)
        accumulator = _select(sib_lane, 0, accumulator)
        multiplier = _select(sib_lane, 1, multiplier)
        sib_immediate = (
            sib_lane
            & (sib_disp_width == 0)
            & (after_sib == int(ReadPhase.IMMEDIATE))
        )
        sib_immediate_width = _table_1d(
            self.config.immediate_bytes, state.token.maximum(0),
        )
        remaining = _select(sib_immediate, sib_immediate_width, remaining)
        field_width = _select(sib_immediate, sib_immediate_width, field_width)

        displacement_lane = live & (state.phase == int(ReadPhase.DISPLACEMENT))
        displacement_acc = accumulator + byte * multiplier
        displacement_done = displacement_lane & (remaining == 1)
        cursor = _select(displacement_lane, cursor + 1, cursor)
        accumulator = _select(displacement_lane, displacement_acc, accumulator)
        multiplier = _select(displacement_lane, multiplier * 256, multiplier)
        remaining = _select(displacement_lane, remaining - 1, remaining)
        signed_displacement = self._sign_extend(displacement_acc, field_width)
        displacement = _select(displacement_done, signed_displacement, displacement)
        after_displacement = self._phase_after_operands(state.token)
        phase = _select(
            displacement_done, after_displacement, phase,
        )
        displacement_immediate = displacement_done & (
            after_displacement == int(ReadPhase.IMMEDIATE)
        )
        displacement_immediate_width = _table_1d(
            self.config.immediate_bytes, state.token.maximum(0),
        )
        remaining = _select(
            displacement_immediate, displacement_immediate_width, remaining,
        )
        field_width = _select(
            displacement_immediate, displacement_immediate_width, field_width,
        )
        accumulator = _select(displacement_immediate, 0, accumulator)
        multiplier = _select(displacement_immediate, 1, multiplier)

        immediate_lane = live & (state.phase == int(ReadPhase.IMMEDIATE))
        immediate_acc = accumulator + byte * multiplier
        immediate_done = immediate_lane & (remaining == 1)
        cursor = _select(immediate_lane, cursor + 1, cursor)
        accumulator = _select(immediate_lane, immediate_acc, accumulator)
        multiplier = _select(immediate_lane, multiplier * 256, multiplier)
        remaining = _select(immediate_lane, remaining - 1, remaining)
        token_flags = _table_1d(self.config.flags, state.token.maximum(0))
        immediate_signed = _bit_is_set(
            token_flags, int(EncodingFlag.IMMEDIATE_SIGNED),
        )
        signed_immediate = self._sign_extend(immediate_acc, field_width)
        final_immediate = _select(immediate_signed, signed_immediate, immediate_acc)
        immediate = _select(immediate_done, final_immediate, immediate)
        relative = _bit_is_set(
            token_flags, int(EncodingFlag.IMMEDIATE_RELATIVE),
        )
        relative_target = _select(
            immediate_done & relative,
            batch.base_addresses + cursor + final_immediate,
            relative_target,
        )
        phase = _select(immediate_done, int(ReadPhase.EMIT), phase)

        emitted = active & (state.phase == int(ReadPhase.EMIT))
        status = _select(emitted, int(ReadStatus.EMITTED), status)
        terminal_flags = _table_1d(self.config.flags, state.token.maximum(0))
        terminal = emitted & _bit_is_set(
            terminal_flags, int(EncodingFlag.TERMINAL),
        )
        phase = _select(terminal, int(ReadPhase.HALT), phase)
        status = _select(terminal, int(ReadStatus.HALTED), status)

        overlong = active & (
            (cursor - instruction_start) > self.config.maximum_instruction_bytes
        )
        status = _select(overlong, int(ReadStatus.FAILED), status)
        failure = _select(
            overlong, int(ReadFailure.INSTRUCTION_TOO_LONG), failure,
        )

        return X86ReadHeadState(
            cursor, instruction_start, phase, status, failure, token,
            opcode_map, opcode, rex, rex_present, legacy, modrm, sib, displacement,
            immediate, relative_target, accumulator, multiplier, remaining,
            field_width,
        )

    @staticmethod
    def _table_2d_flat(flat, table, row, column):
        width = int(table.shape[1])
        index = row.to_dtype("int64") * width + column.to_dtype("int64")
        return flat.gather(index, dim=0)

    @staticmethod
    def _table_3d_flat(flat, table, plane, row, column):
        rows = int(table.shape[1])
        width = int(table.shape[2])
        index = (
            plane.to_dtype("int64") * rows * width
            + row.to_dtype("int64") * width
            + column.to_dtype("int64")
        )
        return flat.gather(index, dim=0)

    @staticmethod
    def _table_4d_flat(flat, table, outer, plane, row, column):
        planes = int(table.shape[1])
        rows = int(table.shape[2])
        width = int(table.shape[3])
        index = (
            outer.to_dtype("int64") * planes * rows * width
            + plane.to_dtype("int64") * rows * width
            + row.to_dtype("int64") * width
            + column.to_dtype("int64")
        )
        return flat.gather(index, dim=0)

    def run(
        self,
        batch: X86ReadBatch | X86PreparedReadBatch,
        *,
        mode: ReadHeadExecutionMode = ReadHeadExecutionMode.DECODE,
        maximum_microsteps: int = 1_000_000,
        executor: Callable[[X86ReadHeadState], None] | None = None,
    ) -> X86ReadHeadRunResult:
        """Orchestrate decoding, tracing, or caller-supplied emulation effects.

        Emulation is deliberately fail-closed: an executor must consume every
        emitted state. This read head schedules instruction events but does not
        silently invent machine-state semantics.
        """

        if maximum_microsteps <= 0:
            raise ValueError("maximum_microsteps must be positive")
        if mode is ReadHeadExecutionMode.EMULATE and executor is None:
            raise ValueError("emulation mode requires an instruction executor")
        prepared = batch.prepare() if isinstance(batch, X86ReadBatch) else batch
        state = X86ReadHeadState.initial(prepared)  # type: ignore[arg-type]
        emissions: list[X86ReadHeadState] = []
        for step in range(1, maximum_microsteps + 1):
            state = self.transition(prepared, state)
            emitted = state.status == int(ReadStatus.EMITTED)
            terminal_emitted = state.status == int(ReadStatus.HALTED)
            event = emitted | terminal_emitted
            if bool(event.any().item()):
                if mode is not ReadHeadExecutionMode.DECODE:
                    emissions.append(state)
                if executor is not None:
                    executor(state)
                state = self.acknowledge(state)
            active = state.status == int(ReadStatus.ACTIVE)
            if not bool(active.any().item()):
                return X86ReadHeadRunResult(state, tuple(emissions), step, mode)
        raise RuntimeError(
            f"x86 read-head exceeded {maximum_microsteps} tensor microsteps"
        )

    def transition_block(
        self,
        batch: X86ReadBatch | X86PreparedReadBatch,
        state: X86ReadHeadState,
        *,
        microsteps: int,
    ) -> X86ReadHeadState:
        """Build a fixed tensor transition block suitable for AOT capture.

        Emitted lanes remain emitted; callers acknowledge them between blocks.
        No per-step host synchronization occurs inside this method.
        """

        if microsteps <= 0:
            raise ValueError("microsteps must be positive")
        prepared = batch.prepare() if isinstance(batch, X86ReadBatch) else batch
        current = state
        for _ in range(microsteps):
            current = self.transition(prepared, current)
        return current

    def acknowledge(
        self,
        state: X86ReadHeadState,
    ) -> X86ReadHeadState:
        """Reset emitted nonterminal lanes at their current cursor."""

        emitted = state.status == int(ReadStatus.EMITTED)
        zero = state.cursor * 0
        negative = zero - 1
        return X86ReadHeadState(
            state.cursor,
            _select(emitted, state.cursor, state.instruction_start),
            _select(emitted, int(ReadPhase.PREFIX), state.phase),
            _select(emitted, int(ReadStatus.ACTIVE), state.status),
            _select(emitted, int(ReadFailure.NONE), state.failure),
            _select(emitted, negative, state.token),
            _select(emitted, zero, state.opcode_map),
            _select(emitted, negative, state.opcode),
            _select(emitted, zero, state.rex),
            _select(emitted, zero, state.rex_present),
            _select(emitted, zero, state.legacy_prefixes),
            _select(emitted, negative, state.modrm),
            _select(emitted, negative, state.sib),
            _select(emitted, zero, state.displacement),
            _select(emitted, zero, state.immediate),
            _select(emitted, zero, state.relative_target),
            _select(emitted, zero, state.field_accumulator),
            _select(emitted, zero + 1, state.field_multiplier),
            _select(emitted, zero, state.field_remaining),
            _select(emitted, zero, state.field_width),
        )

    def _phase_after_token(self, token: AbstractTensor) -> AbstractTensor:
        safe_token = token.maximum(0)
        flags = _table_1d(self.config.flags, safe_token)
        immediate_width = _table_1d(self.config.immediate_bytes, safe_token)
        return _select(
            _bit_is_set(flags, int(EncodingFlag.HAS_MODRM)),
            int(ReadPhase.MODRM),
            _select(
                immediate_width > 0,
                int(ReadPhase.IMMEDIATE),
                int(ReadPhase.EMIT),
            ),
        )

    def _phase_after_operands(self, token: AbstractTensor) -> AbstractTensor:
        safe_token = token.maximum(0)
        immediate_width = _table_1d(self.config.immediate_bytes, safe_token)
        return _select(
            immediate_width > 0,
            int(ReadPhase.IMMEDIATE),
            int(ReadPhase.EMIT),
        )

    def _prefix_errors(
        self,
        token: AbstractTensor,
        rex: AbstractTensor,
        rex_present: AbstractTensor,
        legacy: AbstractTensor,
    ) -> tuple[AbstractTensor, AbstractTensor]:
        safe_token = token.maximum(0)
        allow_rex = _table_1d(self.config.allow_rex_prefix, safe_token)
        allowed_rex = _table_1d(self.config.allowed_rex_mask, safe_token)
        required_rex = _table_1d(self.config.required_rex_mask, safe_token)
        allowed_legacy = _table_1d(self.config.allowed_legacy_mask, safe_token)
        required_legacy = _table_1d(self.config.required_legacy_mask, safe_token)
        forbidden_rex = ((rex_present > 0) & (allow_rex == 0)) | (
            _mask_has_forbidden_bits(rex, allowed_rex, (1, 2, 4, 8))
        )
        forbidden_legacy = _mask_has_forbidden_bits(
            legacy, allowed_legacy, tuple(1 << shift for shift in range(11)),
        )
        required_missing = _mask_missing_required_bits(
            rex, required_rex, (1, 2, 4, 8),
        ) | _mask_missing_required_bits(
            legacy, required_legacy, tuple(1 << shift for shift in range(11)),
        )
        return forbidden_rex | forbidden_legacy, required_missing

    @staticmethod
    def _sign_extend(value: AbstractTensor, width: AbstractTensor) -> AbstractTensor:
        """Sign-extend 1/2/4-byte fields using tensor-selected constants."""

        cls = type(value)
        thresholds = AbstractTensor.get_tensor(
            [0, 1 << 7, 1 << 15, 0, 1 << 31, 0, 0, 0, 0],
            dtype="int64",
            cls=cls,
        )
        moduli = AbstractTensor.get_tensor(
            [0, 1 << 8, 1 << 16, 0, 1 << 32, 0, 0, 0, 0],
            dtype="int64",
            cls=cls,
        )
        threshold = _table_1d(thresholds, width.to_dtype("int64"))
        modulus = _table_1d(moduli, width.to_dtype("int64"))
        return _select((threshold > 0) & (value >= threshold), value - modulus, value)


@dataclass(slots=True)
class X86ReversibleReadHead:
    """Journaled virtual multicore around the pure tensor transition kernel.

    One batch lane is one independently advancing core.  Forward cycles apply
    the same tensor kernel to every core concurrently.  Backward cycles restore
    the exact prior immutable state, including partial instruction fields; no
    attempt is made to guess an algebraic inverse for a lossy decoder step.

    Rewinding and then moving forward truncates the abandoned future, giving
    callers an explicit branch point suitable for graph inspection or forking.
    The retained tensor states keep the forward computational graph intact.
    """

    head: X86TensorReadHead
    batch: X86PreparedReadBatch
    _history: list[X86ReadHeadState]
    _position: int = 0

    @classmethod
    def create(
        cls,
        head: X86TensorReadHead,
        batch: X86ReadBatch | X86PreparedReadBatch,
    ) -> "X86ReversibleReadHead":
        prepared = batch.prepare() if isinstance(batch, X86ReadBatch) else batch
        initial = X86ReadHeadState.initial(prepared)  # type: ignore[arg-type]
        return cls(head=head, batch=prepared, _history=[initial])

    @property
    def state(self) -> X86ReadHeadState:
        return self._history[self._position]

    @property
    def core_count(self) -> int:
        return int(self.batch.octets.shape[0])

    @property
    def history_position(self) -> int:
        return self._position

    @property
    def history_length(self) -> int:
        return len(self._history)

    def _append(self, state: X86ReadHeadState) -> X86ReadHeadState:
        del self._history[self._position + 1:]
        self._history.append(state)
        self._position += 1
        return state

    def transition(self, direction: ReadHeadDirection = ReadHeadDirection.FORWARD) -> X86ReadHeadState:
        """Move one reversible microstep in the requested direction."""

        if direction is ReadHeadDirection.BACKWARD:
            if self._position == 0:
                raise IndexError("read head is already at the beginning of history")
            self._position -= 1
            return self.state
        if direction is not ReadHeadDirection.FORWARD:
            raise ValueError(f"unsupported read-head direction {direction!r}")
        return self._append(self.head.transition(self.batch, self.state))

    def acknowledge(self) -> X86ReadHeadState:
        """Journal acknowledgement as its own reversible state change."""

        return self._append(self.head.acknowledge(self.state))

    def seek_history(self, position: int) -> X86ReadHeadState:
        """Move to an already-recorded examination point without recomputing."""

        if not 0 <= position < len(self._history):
            raise IndexError("read-head history position is out of range")
        self._position = int(position)
        return self.state

    def fork(self) -> "X86ReversibleReadHead":
        """Create an independent execution thread at the current state."""

        return X86ReversibleReadHead(
            self.head, self.batch, list(self._history[:self._position + 1]),
            self._position,
        )

    def register_tensor(self) -> AbstractTensor:
        return self.state.register_tensor()

    def register_contents(self) -> tuple[dict[str, int], ...]:
        return self.state.register_contents()


def controlled_x86_64_read_head_profile() -> X86ReadHeadProfile:
    """Derive tensor/read-write rows from the authoritative ISA vocabulary."""

    from .machine_reference_vocabulary import X86_64_REFERENCE_VOCABULARY

    prefix_bits = {
        0xF0: 1 << 0, 0xF2: 1 << 1, 0xF3: 1 << 2,
        0x2E: 1 << 3, 0x36: 1 << 4, 0x3E: 1 << 5,
        0x26: 1 << 6, 0x64: 1 << 7, 0x65: 1 << 8,
        0x66: 1 << 9, 0x67: 1 << 10,
    }
    rows: list[X86EncodingRow] = []
    for spec in X86_64_REFERENCE_VOCABULARY:
        if spec.reversible_operand_form is None:
            continue
        opcode = bytes(spec.opcode)
        if opcode.startswith(b"\x0f\x38"):
            if len(opcode) != 3:
                raise ValueError(f"invalid 0F38 opcode path for {spec.token.name}")
            opcode_map, opcode_byte = 2, opcode[2]
            mask = 0xFF if spec.opcode_mask is None else spec.opcode_mask[2]
        elif opcode.startswith(b"\x0f\x3a"):
            if len(opcode) != 3:
                raise ValueError(f"invalid 0F3A opcode path for {spec.token.name}")
            opcode_map, opcode_byte = 3, opcode[2]
            mask = 0xFF if spec.opcode_mask is None else spec.opcode_mask[2]
        elif opcode.startswith(b"\x0f"):
            if len(opcode) != 2:
                raise ValueError(f"invalid 0F opcode path for {spec.token.name}")
            opcode_map, opcode_byte = 1, opcode[1]
            mask = 0xFF if spec.opcode_mask is None else spec.opcode_mask[1]
        else:
            if len(opcode) != 1:
                raise ValueError(
                    f"reversible token {spec.token.name} has invalid opcode path"
                )
            opcode_map, opcode_byte = 0, opcode[0]
            mask = 0xFF if spec.opcode_mask is None else spec.opcode_mask[0]
        allowed_legacy = sum(
            prefix_bits[prefix] for prefix in spec.allowed_legacy_prefixes
        )
        required_legacy = sum(
            prefix_bits[prefix] for prefix in spec.required_legacy_prefixes
        )
        required_rex = 0x08 if spec.require_rex_w else 0
        rows.append(X86EncodingRow(
            token=int(spec.token),
            opcode_map=opcode_map,
            opcode=opcode_byte,
            opcode_mask=mask,
            modrm_extension=spec.modrm_extension,
            modrm_memory_only=bool(spec.reversible_modrm_memory_only),
            has_modrm=(
                spec.modrm_extension is not None
                or spec.reversible_operand_form in {
                    "REG_RM", "RM_REG", "RM_IMMEDIATE", "RM",
                    "REG_RM_IMMEDIATE",
                }
            ),
            immediate_bytes=spec.reversible_immediate_bytes,
            immediate_signed=spec.reversible_immediate_signed,
            immediate_relative=spec.reversible_immediate_relative,
            terminal=spec.reversible_terminal,
            allow_rex=spec.allow_rex,
            allowed_rex_mask=spec.reversible_allowed_rex_mask,
            required_rex_mask=required_rex,
            allowed_legacy_mask=allowed_legacy,
            required_legacy_mask=required_legacy,
            operand_form=X86OperandForm[spec.reversible_operand_form],
        ))
    return X86ReadHeadProfile(
        name="controlled-x86-64",
        rows=tuple(rows),
        token_names=MappingProxyType({
            int(spec.token): spec.token.name
            for spec in X86_64_REFERENCE_VOCABULARY
            if spec.reversible_operand_form is not None
        }),
        source_tokens=MappingProxyType({
            int(spec.token): int(spec.token)
            for spec in X86_64_REFERENCE_VOCABULARY
            if spec.reversible_operand_form is not None
        }),
    )


def controlled_x86_64_read_head_config(
    *,
    like: AbstractTensor | None = None,
) -> X86ReadHeadConfig:
    """Compile the repository's current controlled vocabulary into tensors."""

    return controlled_x86_64_read_head_profile().compile(like=like)


__all__ = [
    "EncodingFlag",
    "PrefixAction",
    "ReadFailure",
    "ReadHeadExecutionMode",
    "ReadHeadDirection",
    "ReadPhase",
    "ReadStatus",
    "X86EncodingRow",
    "X86OperandForm",
    "X86EncodingFields",
    "X86AllocatedInstruction",
    "X86ReadBatch",
    "X86PreparedReadBatch",
    "X86ReadHeadConfig",
    "X86ReadHeadCodeSetBank",
    "X86ReadHeadProfile",
    "X86ReadHeadState",
    "X86ReadHeadRunResult",
    "X86ReversibleReadHead",
    "X86TensorReadHead",
    "controlled_x86_64_read_head_profile",
    "controlled_x86_64_read_head_config",
]
