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

from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from types import MappingProxyType
from typing import Callable, ClassVar, Iterable, Mapping

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
        if self.immediate_bytes not in (0, 1, 2, 4, 8):
            raise ValueError("immediate width must be 0, 1, 2, 4, or 8 bytes")
        if self.allowed_rex_mask & ~0x0F or self.required_rex_mask & ~0x0F:
            raise ValueError("REX masks use only W/R/X/B bits")
        if self.required_rex_mask & ~self.allowed_rex_mask:
            raise ValueError("required REX bits must also be allowed")
        if self.required_rex_mask and not self.allow_rex:
            raise ValueError("a required REX bit needs allow_rex=True")


@dataclass(frozen=True, slots=True)
class X86ReadHeadConfig:
    """Tensor-resident transition vocabulary shared by every read lane."""

    opcode_token: AbstractTensor
    opcode_group: AbstractTensor
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
    maximum_instruction_bytes: int = 15
    profile_name: str = "anonymous"
    encoding_row_count: int = 0

    @property
    def opcode_map_count(self) -> int:
        return int(self.opcode_token.shape[0])

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
        escapes = dict({(0, 0x0F): 1} if escape_maps is None else escape_maps)
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
        opcode_token = [[-1] * 256 for _ in range(map_count)]
        opcode_group = [[-1] * 256 for _ in range(map_count)]
        group_token: list[list[int]] = []
        group_ids: dict[tuple[int, int], int] = {}
        token_metadata: dict[int, tuple[int, int, int, int, int, int]] = {}
        encodings: set[tuple[int, int, int, int | None]] = set()

        for row in records:
            identity = (
                row.opcode_map, row.opcode, row.opcode_mask, row.modrm_extension,
            )
            if identity in encodings:
                raise ValueError(f"duplicate x86 encoding row {identity}")
            encodings.add(identity)
            metadata = (
                (int(EncodingFlag.HAS_MODRM) if row.has_modrm else 0)
                | (int(EncodingFlag.IMMEDIATE_SIGNED) if row.immediate_signed else 0)
                | (int(EncodingFlag.IMMEDIATE_RELATIVE) if row.immediate_relative else 0)
                | (int(EncodingFlag.TERMINAL) if row.terminal else 0),
                row.immediate_bytes,
                int(row.allow_rex),
                row.allowed_rex_mask,
                row.required_rex_mask,
                row.allowed_legacy_mask,
            )
            previous = token_metadata.setdefault(row.token, metadata)
            if previous != metadata:
                raise ValueError(f"token {row.token} has inconsistent encoding metadata")

            matching_opcodes = tuple(
                opcode for opcode in range(256)
                if opcode & row.opcode_mask == row.opcode & row.opcode_mask
            )
            for opcode in matching_opcodes:
                if row.modrm_extension is None:
                    if opcode_group[row.opcode_map][opcode] >= 0:
                        raise ValueError("plain opcode conflicts with an opcode-group row")
                    if opcode_token[row.opcode_map][opcode] >= 0:
                        raise ValueError("masked opcode overlaps another plain token")
                    opcode_token[row.opcode_map][opcode] = row.token
                else:
                    if opcode_token[row.opcode_map][opcode] >= 0:
                        raise ValueError("opcode-group row conflicts with a plain opcode")
                    key = (row.opcode_map, opcode)
                    group_id = group_ids.get(key)
                    if group_id is None:
                        group_id = len(group_token)
                        group_ids[key] = group_id
                        group_token.append([-1] * 8)
                        opcode_group[row.opcode_map][opcode] = group_id
                    if group_token[group_id][row.modrm_extension] >= 0:
                        raise ValueError("masked opcode-group extension overlaps another row")
                    group_token[group_id][row.modrm_extension] = row.token

        flags = [0] * vocabulary_size
        immediate_bytes = [0] * vocabulary_size
        allowed_rex = [0] * vocabulary_size
        allow_rex = [0] * vocabulary_size
        required_rex = [0] * vocabulary_size
        allowed_legacy = [0] * vocabulary_size
        for token, metadata in token_metadata.items():
            (
                flags[token], immediate_bytes[token], allow_rex[token], allowed_rex[token],
                required_rex[token], allowed_legacy[token],
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
        default_factory=lambda: {(0, 0x0F): 1},
    )
    legacy_prefix_bits: Mapping[int, int] | None = None
    forbid_prefixes: tuple[int, ...] = ()
    rex_prefixes: bool = True
    maximum_instruction_bytes: int = 15

    def extend(
        self,
        name: str,
        rows: Iterable[X86EncodingRow],
        *,
        escape_maps: Mapping[tuple[int, int], int] | None = None,
    ) -> "X86ReadHeadProfile":
        merged_escapes = dict(self.escape_maps)
        merged_escapes.update(dict(escape_maps or {}))
        return X86ReadHeadProfile(
            name=str(name),
            rows=(*self.rows, *tuple(rows)),
            escape_maps=MappingProxyType(merged_escapes),
            legacy_prefix_bits=self.legacy_prefix_bits,
            forbid_prefixes=self.forbid_prefixes,
            rex_prefixes=self.rex_prefixes,
            maximum_instruction_bytes=self.maximum_instruction_bytes,
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
        return cls(
            cursor=zero,
            instruction_start=zero,
            phase=zero + int(ReadPhase.PREFIX),
            status=zero + int(ReadStatus.ACTIVE),
            failure=zero + int(ReadFailure.NONE),
            token=negative,
            opcode_map=zero,
            opcode=negative,
            rex=zero,
            rex_present=zero,
            legacy_prefixes=zero,
            modrm=negative,
            sib=negative,
            displacement=zero,
            immediate=zero,
            relative_target=zero,
            field_accumulator=zero,
            field_multiplier=zero + 1,
            field_remaining=zero,
            field_width=zero,
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

    def __init__(self, config: X86ReadHeadConfig) -> None:
        self.config = config
        # Reshape is metadata work but some backends materialize it. Cache all
        # immutable table views once instead of rebuilding them per microstep.
        self._opcode_token_flat = config.opcode_token.reshape((-1,))
        self._opcode_group_flat = config.opcode_group.reshape((-1,))
        self._group_token_flat = config.group_token.reshape((-1,))
        self._escape_map_flat = config.escape_map.reshape((-1,))

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
        direct_token = self._table_2d_flat(
            self._opcode_token_flat, self.config.opcode_token, opcode_map, byte,
        )
        group_id = self._table_2d_flat(
            self._opcode_group_flat, self.config.opcode_group, opcode_map, byte,
        )
        known_opcode = (direct_token >= 0) | (group_id >= 0)
        unknown_opcode = final_opcode_lane & ~known_opcode
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
        status = _select(unknown_opcode, int(ReadStatus.FAILED), status)
        failure = _select(unknown_opcode, int(ReadFailure.UNKNOWN_OPCODE), failure)

        modrm_lane = live & (state.phase == int(ReadPhase.MODRM))
        group_for_opcode = self._table_2d_flat(
            self._opcode_group_flat, self.config.opcode_group,
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
        phase = _select(modrm_lane & ~missing_group, post_modrm_phase, phase)
        remaining = _select(
            modrm_lane & ~missing_group & ~has_sib & (displacement_width > 0),
            displacement_width,
            remaining,
        )
        field_width = _select(
            modrm_lane & ~missing_group & ~has_sib & (displacement_width > 0),
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
        forbidden_rex = ((rex_present > 0) & (allow_rex == 0)) | (
            _mask_has_forbidden_bits(rex, allowed_rex, (1, 2, 4, 8))
        )
        forbidden_legacy = _mask_has_forbidden_bits(
            legacy, allowed_legacy, tuple(1 << shift for shift in range(11)),
        )
        required_missing = _mask_missing_required_bits(
            rex, required_rex, (1, 2, 4, 8),
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


def controlled_x86_64_read_head_config(
    *,
    like: AbstractTensor | None = None,
) -> X86ReadHeadConfig:
    """Compile the repository's current controlled vocabulary into tensors."""

    from .machine_reference_vocabulary import X86InstructionToken

    return X86ReadHeadConfig.from_rows(
        (
            X86EncodingRow(
                int(X86InstructionToken.IMUL_R32_RM32),
                opcode_map=1,
                opcode=0xAF,
                has_modrm=True,
                allowed_rex_mask=0x07,
            ),
            X86EncodingRow(
                int(X86InstructionToken.LEA_R32_M),
                opcode_map=0,
                opcode=0x8D,
                has_modrm=True,
                allowed_rex_mask=0x07,
            ),
            X86EncodingRow(
                int(X86InstructionToken.RET_NEAR),
                opcode_map=0,
                opcode=0xC3,
                terminal=True,
                allow_rex=False,
                allowed_rex_mask=0,
            ),
            X86EncodingRow(
                int(X86InstructionToken.SUB_R64_IMM8),
                opcode_map=0,
                opcode=0x83,
                modrm_extension=5,
                has_modrm=True,
                immediate_bytes=1,
                immediate_signed=True,
                allowed_rex_mask=0x0B,
                required_rex_mask=0x08,
            ),
            X86EncodingRow(
                int(X86InstructionToken.CALL_REL32),
                opcode_map=0,
                opcode=0xE8,
                immediate_bytes=4,
                immediate_signed=True,
                immediate_relative=True,
                allow_rex=False,
                allowed_rex_mask=0,
            ),
            X86EncodingRow(
                int(X86InstructionToken.ADD_R64_IMM8),
                opcode_map=0,
                opcode=0x83,
                modrm_extension=0,
                has_modrm=True,
                immediate_bytes=1,
                immediate_signed=True,
                allowed_rex_mask=0x0B,
                required_rex_mask=0x08,
            ),
            X86EncodingRow(
                int(X86InstructionToken.JMP_REL32),
                opcode_map=0,
                opcode=0xE9,
                immediate_bytes=4,
                immediate_signed=True,
                immediate_relative=True,
                terminal=True,
                allow_rex=False,
                allowed_rex_mask=0,
            ),
            X86EncodingRow(
                int(X86InstructionToken.MOV_RM64_R64),
                opcode_map=0,
                opcode=0x89,
                has_modrm=True,
                allowed_rex_mask=0x0F,
                required_rex_mask=0x08,
            ),
            X86EncodingRow(
                int(X86InstructionToken.PUSH_R64),
                opcode_map=0,
                opcode=0x50,
                opcode_mask=0xF8,
                allowed_rex_mask=0x0F,
            ),
            X86EncodingRow(
                int(X86InstructionToken.MOV_R64_RM64),
                opcode_map=0,
                opcode=0x8B,
                has_modrm=True,
                allowed_rex_mask=0x0F,
                required_rex_mask=0x08,
            ),
            X86EncodingRow(
                int(X86InstructionToken.AND_RM64_IMM8),
                opcode_map=0,
                opcode=0x83,
                modrm_extension=4,
                has_modrm=True,
                immediate_bytes=1,
                immediate_signed=True,
                allowed_rex_mask=0x0B,
                required_rex_mask=0x08,
            ),
            X86EncodingRow(
                int(X86InstructionToken.MOV_R64_IMM64),
                opcode_map=0,
                opcode=0xB8,
                opcode_mask=0xF8,
                immediate_bytes=8,
                allowed_rex_mask=0x09,
                required_rex_mask=0x08,
            ),
            X86EncodingRow(
                int(X86InstructionToken.CMP_R64_RM64),
                opcode_map=0,
                opcode=0x3B,
                has_modrm=True,
                allowed_rex_mask=0x0F,
                required_rex_mask=0x08,
            ),
            X86EncodingRow(
                int(X86InstructionToken.JNE_REL32),
                opcode_map=1,
                opcode=0x85,
                immediate_bytes=4,
                immediate_signed=True,
                immediate_relative=True,
                terminal=True,
                allow_rex=False,
                allowed_rex_mask=0,
            ),
        ),
        like=like,
    )


__all__ = [
    "EncodingFlag",
    "PrefixAction",
    "ReadFailure",
    "ReadHeadExecutionMode",
    "ReadHeadDirection",
    "ReadPhase",
    "ReadStatus",
    "X86EncodingRow",
    "X86ReadBatch",
    "X86PreparedReadBatch",
    "X86ReadHeadConfig",
    "X86ReadHeadProfile",
    "X86ReadHeadState",
    "X86ReadHeadRunResult",
    "X86ReversibleReadHead",
    "X86TensorReadHead",
    "controlled_x86_64_read_head_config",
]
