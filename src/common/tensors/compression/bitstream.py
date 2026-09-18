"""Variable-length codeword compaction using AbstractTensor primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..abstraction import AbstractTensor
from .huffman import CanonicalHuffmanTable, HuffmanCodewords


@dataclass(frozen=True)
class PackedBitstream:
    """Byte-aligned tensor payload with exact bit and symbol provenance."""

    octets: AbstractTensor
    valid_bits: AbstractTensor
    symbol_offsets: AbstractTensor
    symbol_lengths: AbstractTensor

    @property
    def byte_count(self) -> int:
        """Return the serialized byte count at the explicit I/O boundary."""
        return (int(self.valid_bits.item()) + 7) // 8

    def to_bytes(self) -> bytes:
        """Materialize bytes only when crossing from tensors into file I/O."""
        return tensor_octets_to_bytes(self.octets, count=self.byte_count)


@dataclass(frozen=True)
class ResidentBytePacket:
    """Fixed-capacity octets with a device-resident logical byte count.

    Variable serialized length is numerical program state.  It must not become
    a Python slice bound observed during discovery, because doing so compiles
    one captured packet instead of the reusable program.  Shell stream
    descriptors consume ``byte_count`` while ``octets`` retains a stable arena
    allocation.
    """

    octets: AbstractTensor
    byte_count: AbstractTensor

    def __post_init__(self) -> None:
        if not isinstance(self.octets, AbstractTensor):
            raise TypeError("resident packet octets must be an AbstractTensor")
        if self.octets.ndims() != 1:
            raise ValueError("resident packet octets must be one-dimensional")
        if not isinstance(self.byte_count, AbstractTensor):
            raise TypeError(
                "resident packet byte_count must be an AbstractTensor"
            )

    @property
    def capacity(self) -> int:
        return int(self.octets.shape[0])

    def to_bytes(self) -> bytes:
        """Materialize only at the terminal host/file boundary."""

        return tensor_octets_to_bytes(
            self.octets,
            count=int(self.byte_count.item()),
        )


def resident_byte_packet(
    octets: AbstractTensor,
    byte_count: AbstractTensor | int | None = None,
) -> ResidentBytePacket:
    """Attach an explicit logical length to one resident octet allocation."""

    if not isinstance(octets, AbstractTensor):
        raise TypeError("octets must be an AbstractTensor")
    if byte_count is None:
        return ResidentBytePacket(
            octets,
            AbstractTensor.full(
                (1,),
                int(octets.shape[0]),
                dtype="int64",
                cls=type(octets),
            ),
        )
    if not isinstance(byte_count, AbstractTensor):
        return ResidentBytePacket(
            octets,
            AbstractTensor.full(
                (1,),
                int(byte_count),
                dtype="int64",
                cls=type(octets),
            ),
        )
    return ResidentBytePacket(octets, byte_count)


def concatenate_resident_byte_packets(
    packets,
) -> ResidentBytePacket:
    """Concatenate packet prefixes without host reads or dynamic shapes."""

    packets = tuple(packets)
    if not packets:
        raise ValueError("at least one resident byte packet is required")
    if not all(isinstance(packet, ResidentBytePacket) for packet in packets):
        raise TypeError("all packet parts must be ResidentBytePacket values")
    exemplar = packets[0].octets
    total_capacity = sum(packet.octets.shape[0] for packet in packets)
    output_positions = AbstractTensor.arange(
        total_capacity,
        cls=type(exemplar),
    )
    output = AbstractTensor.zeros(
        (total_capacity,),
        dtype=exemplar.dtype,
        cls=type(exemplar),
    )
    # This is loop-carried numerical state, so express its initialization as
    # an ordinary AbstractTensor operation.  A scalar wrapped through
    # ``ensure_tensor`` is a lifecycle conversion; reducing that conversion as
    # an identity can detach the source literal from the loop's resident
    # initial-value edge.  ``zeros`` gives every backend and the control-flow
    # planner an explicit producer without specializing from discovery data.
    prefix = AbstractTensor.zeros(
        (1,),
        dtype="int64",
        cls=type(exemplar),
    )
    for packet in packets:
        local = output_positions - prefix
        valid = (local >= 0) & (local < packet.byte_count)
        packet_capacity = packet.octets.shape[0]
        safe = local.maximum(0).minimum(max(0, packet_capacity - 1))
        output = output + packet.octets[safe.to_dtype("int64")] * valid
        prefix = prefix + packet.byte_count
    # Packet arithmetic, rather than the compiler or structural constructor,
    # owns the public integer-count contract.
    return resident_byte_packet(output, prefix.to_dtype("int64"))


def tensor_octets_to_bytes(
    octets: AbstractTensor,
    *,
    count: int | None = None,
) -> bytes:
    """Materialize an octet tensor at the explicit binary-I/O boundary.

    Compression code keeps byte values as tensors through packing, carry
    handling, and format-specific byte transforms.  This helper is the one
    intentional transition from numerical storage to Python's immutable byte
    container.
    """
    if not isinstance(octets, AbstractTensor):
        raise TypeError("octets must be an AbstractTensor")
    if octets.ndims() != 1:
        raise ValueError("octets must be one-dimensional")
    limit = octets.shape[0] if count is None else int(count)
    if limit < 0 or limit > octets.shape[0]:
        raise ValueError("octet count is outside the tensor payload")
    values = octets[:limit].tolist()
    serialized: list[int] = []
    for index, value in enumerate(values):
        try:
            numeric = float(value)
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"octet {index} is not a finite integer byte value: {value!r}"
            ) from error
        if not math.isfinite(numeric) or numeric != integer:
            raise ValueError(
                f"octet {index} is not an integer byte value: {value!r}"
            )
        if integer < 0 or integer > 255:
            raise ValueError(
                f"octet {index} is outside [0, 255]: {value!r}"
            )
        serialized.append(integer)
    return bytes(serialized)


tensor_octets_to_bytes.__process_graph_boundary__ = "host_materialization"


@dataclass(frozen=True)
class UnpackedBitstream:
    """Parallel MSB-first bits recovered exactly from packed octets."""

    bits: AbstractTensor
    valid: AbstractTensor
    valid_bits: AbstractTensor


@dataclass(frozen=True)
class DecodedHuffmanStream:
    """Fixed-capacity decoded symbols with an explicit validity mask."""

    symbols: AbstractTensor
    valid: AbstractTensor
    symbol_count: AbstractTensor
    valid_bits: AbstractTensor

    def tolist(self) -> list[int]:
        """Materialize only decoded symbols at the explicit host boundary."""
        count = int(self.symbol_count.item())
        return [int(value) for value in self.symbols[:count].tolist()]


def compact_codewords(codewords: HuffmanCodewords) -> PackedBitstream:
    """Compact padded codewords into a continuous MSB-first byte tensor.

    Prefix sums assign every symbol a destination bit offset. Existing tensor
    scatter places the valid bits into a fixed-capacity stream; invalid padded
    positions target a scratch cell with zero payload. Finally, an 8-wide dot
    product converts bits into octets.
    """
    lengths = codewords.lengths.flatten()
    bits = codewords.bits.reshape(-1, codewords.max_bits)
    valid = codewords.valid.reshape(-1, codewords.max_bits)
    symbol_count = lengths.shape[0]
    capacity_bits = symbol_count * codewords.max_bits

    offsets = lengths.cumsum(dim=0) - lengths
    positions = AbstractTensor.arange(
        codewords.max_bits, cls=type(lengths)
    )
    destinations = offsets.unsqueeze(1) + positions
    numeric_valid = valid.to_dtype("int64")
    scratch = capacity_bits
    safe_destinations = (
        destinations * numeric_valid
        + scratch * (1 - numeric_valid)
    ).to_dtype("int64")
    contributions = bits * numeric_valid

    stream_with_scratch = AbstractTensor.zeros(
        (capacity_bits + 1,), cls=type(bits)
    )
    stream_with_scratch = AbstractTensor.scatter(
        stream_with_scratch,
        safe_destinations.flatten(),
        contributions.flatten(),
        dim=0,
    )
    stream = stream_with_scratch[:capacity_bits]

    pad_bits = (-capacity_bits) % 8
    if pad_bits:
        stream = AbstractTensor.cat(
            (
                stream,
                AbstractTensor.zeros((pad_bits,), cls=type(stream)),
            ),
            dim=0,
        )
    byte_weights = stream.ensure_tensor(
        (128, 64, 32, 16, 8, 4, 2, 1)
    )
    octets = (
        stream.reshape(-1, 8) * byte_weights.unsqueeze(0)
    ).sum(dim=1).to_dtype("int64")
    return PackedBitstream(
        octets=octets,
        valid_bits=lengths.sum(),
        symbol_offsets=offsets,
        symbol_lengths=lengths,
    )


def unpack_octets(stream: PackedBitstream) -> UnpackedBitstream:
    """Expand packed octets into MSB-first bits with an exact validity mask."""
    positions = AbstractTensor.arange(8, cls=type(stream.octets))
    divisors = 2 ** (7 - positions)
    bits = (
        (stream.octets.unsqueeze(1) // divisors.unsqueeze(0)) % 2
    ).flatten()
    bit_positions = AbstractTensor.arange(bits.shape[0], cls=type(bits))
    valid = (stream.valid_bits - bit_positions) > 0
    return UnpackedBitstream(
        bits=bits,
        valid=valid,
        valid_bits=stream.valid_bits,
    )


def decode_with_provenance(
    table: CanonicalHuffmanTable,
    stream: PackedBitstream,
) -> AbstractTensor:
    """Recover symbols exactly using retained codeword boundaries.

    This is intentionally the provenance-guided decoder. ``compact_codewords``
    preserves each symbol's bit offset and length, so decoding can remain a
    parallel tensor program instead of introducing a serial prefix-tree walk.
    A later wire decoder may discover those boundaries from bytes alone.
    """
    unpacked = unpack_octets(stream)
    offsets = stream.symbol_offsets.flatten()
    lengths = stream.symbol_lengths.flatten()
    if offsets.shape != lengths.shape:
        raise ValueError("symbol offsets and lengths must have equal shape")
    if bool((lengths <= 0).any().item()):
        raise ValueError("symbol lengths must be positive")
    if bool((lengths > table.max_bits).any().item()):
        raise ValueError("symbol length exceeds the Huffman table limit")

    positions = AbstractTensor.arange(table.max_bits, cls=type(lengths))
    source_indices = (
        offsets.unsqueeze(1) + positions.unsqueeze(0)
    ).to_dtype("int64")
    if bool((source_indices >= unpacked.bits.shape[0]).any().item()):
        raise ValueError("symbol provenance extends beyond the packed payload")

    gathered_bits = unpacked.bits[source_indices]
    valid = (lengths.unsqueeze(1) - positions.unsqueeze(0)) > 0
    exponents = (
        lengths.unsqueeze(1) - 1 - positions.unsqueeze(0)
    ).maximum(0)
    codes = (
        gathered_bits
        * valid.to_dtype("int64")
        * (2 ** exponents)
    ).sum(dim=1)

    # Predicate operators are valuewise in AbstractTensor. Arithmetic creates
    # the desired broadcasted matrices before each scalar comparison.
    same_code = (
        codes.unsqueeze(1) - table.codes.unsqueeze(0)
    ) == 0
    same_length = (
        lengths.unsqueeze(1) - table.lengths.unsqueeze(0)
    ) == 0
    present = table.lengths.unsqueeze(0) > 0
    matches = same_code & same_length & present
    match_counts = matches.to_dtype("int64").sum(dim=1)
    if not bool((match_counts == 1).all().item()):
        raise ValueError("packed payload does not identify exactly one symbol")

    alphabet = table.alphabet
    return (
        matches.to_dtype("int64") * alphabet.unsqueeze(0)
    ).sum(dim=1)


def decode_huffman_octets(
    table: CanonicalHuffmanTable,
    octets: AbstractTensor,
    valid_bits: AbstractTensor,
) -> DecodedHuffmanStream:
    """Decode a canonical Huffman stream without symbol-boundary provenance.

    The decoder is a tensor state machine. Each bit advances a code accumulator
    and code length; parallel comparison against the whole canonical table
    identifies completed symbols. Emissions are compacted once through tensor
    prefix sums and scatter, avoiding a host-side prefix tree or output list.
    """
    if not isinstance(octets, AbstractTensor):
        raise TypeError("octets must be an AbstractTensor")
    if not isinstance(valid_bits, AbstractTensor):
        raise TypeError("valid_bits must be an AbstractTensor")
    if octets.ndims() != 1:
        raise ValueError("octets must be one-dimensional")
    if not bool(
        (((octets % 1) == 0) & (octets >= 0) & (octets <= 255)).all().item()
    ):
        raise ValueError("octets must contain integer byte values")

    bit_count = int(valid_bits.item())
    capacity = octets.shape[0] * 8
    if bit_count < 0 or bit_count > capacity:
        raise ValueError("valid bit count exceeds the octet payload")
    if bit_count == 0:
        empty = AbstractTensor.zeros((0,), cls=type(octets))
        return DecodedHuffmanStream(
            symbols=empty,
            valid=empty > 0,
            symbol_count=valid_bits * 0,
            valid_bits=valid_bits,
        )

    positions = AbstractTensor.arange(8, cls=type(octets))
    divisors = 2 ** (7 - positions)
    bits = (
        (octets.unsqueeze(1) // divisors.unsqueeze(0)) % 2
    ).flatten()[:bit_count]

    code = valid_bits * 0
    length = valid_bits * 0
    dead = valid_bits * 0
    emitted_symbols = []
    emitted_masks = []
    present = table.lengths > 0
    alphabet = table.alphabet
    for position in range(bit_count):
        candidate_code = code * 2 + bits[position]
        candidate_length = length + 1
        matches = (
            ((candidate_code - table.codes) == 0)
            & ((candidate_length - table.lengths) == 0)
            & present
        )
        match_count = matches.to_dtype("int64").sum()
        hit = match_count > 0
        ambiguous = match_count > 1
        dead = (
            (dead > 0)
            | ambiguous
            | ((candidate_length >= table.max_bits) & hit.logical_not())
        ).to_dtype("int64")
        numeric_hit = hit.to_dtype("int64")
        symbol = (
            matches.to_dtype("int64") * alphabet
        ).sum()
        emitted_symbols.append(symbol * numeric_hit)
        emitted_masks.append(numeric_hit)
        code = candidate_code * (1 - numeric_hit)
        length = candidate_length * (1 - numeric_hit)

    if bool((dead > 0).item()):
        raise ValueError("invalid or overlong Huffman prefix")
    if bool((length != 0).item()):
        raise ValueError("Huffman payload ends in an incomplete codeword")

    symbol_at_bit = AbstractTensor.stack(emitted_symbols, dim=0).reshape(-1)
    hit_at_bit = AbstractTensor.stack(emitted_masks, dim=0).reshape(-1)
    symbol_count = hit_at_bit.sum()
    ranks = hit_at_bit.cumsum(dim=0) - 1
    scratch = bit_count
    destinations = (
        ranks * hit_at_bit + scratch * (1 - hit_at_bit)
    ).to_dtype("int64")
    compact_with_scratch = AbstractTensor.zeros(
        (bit_count + 1,), cls=type(octets)
    )
    compact_with_scratch = AbstractTensor.scatter(
        compact_with_scratch,
        destinations,
        symbol_at_bit * hit_at_bit,
        dim=0,
    )
    symbols = compact_with_scratch[:bit_count]
    slots = AbstractTensor.arange(bit_count, cls=type(symbols))
    valid = (symbol_count - slots) > 0
    return DecodedHuffmanStream(
        symbols=symbols,
        valid=valid,
        symbol_count=symbol_count,
        valid_bits=valid_bits,
    )


__all__ = [
    "PackedBitstream",
    "ResidentBytePacket",
    "DecodedHuffmanStream",
    "UnpackedBitstream",
    "concatenate_resident_byte_packets",
    "compact_codewords",
    "decode_huffman_octets",
    "decode_with_provenance",
    "tensor_octets_to_bytes",
    "resident_byte_packet",
    "unpack_octets",
]
