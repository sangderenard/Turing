"""Self-describing canonical Huffman coding of an octet string.

A packet is the canonical table followed by the codeword stream:

```text
version (1) | raw length, valid bits (varints) | [256 code lengths (128)] | stream
```

Code lengths are packed two to an octet (high nibble first) and limited to
``MAX_BITS``, so the table is the 128-octet header JPEG's DHT idea implies
and no tree is stored.  A packet coded with a table both sides already hold
(``lengths=``, version 2) omits it: a stream of small packets shares one
table instead of paying for one each.  Encoding counts the octet alphabet with one summing
scatter, builds a length-limited optimal table (Package-Merge), expands every
octet to its codeword in parallel and compacts the codewords
(``compact_codewords``).

Decoding never walks a prefix tree.  Every bit position's ``MAX_BITS``-wide
window is formed at once and looked up in the table's left-aligned prefix
ranges, giving the symbol and code length that WOULD start there.  The real
symbol starts are the orbit of position 0 under ``i -> i + length(i)``; that
orbit is read in ``log2(symbols)`` rounds of pointer doubling, so the whole
decode is a fixed number of gathers, independent of the stream's length.
Positions with no code, an orbit that overruns the stream, or one that does
not end exactly on the valid bit count refuse as corrupt.
"""

from __future__ import annotations

from ..abstraction import AbstractTensor
from .bitstream import compact_codewords, tensor_octets_to_bytes
from .huffman import CanonicalHuffmanTable, length_limited_huffman_code_lengths

#: Packet layout versions: the table in the packet, or the caller's.
VERSION = 1
SHARED_VERSION = 2
#: Longest code.  2**MAX_BITS bounds the decode lookup; 256 symbols need 8.
MAX_BITS = 12
ALPHABET = 256
_TABLE = ALPHABET // 2


def _octets(data: bytes) -> AbstractTensor:
    return AbstractTensor.tensor(list(data)).to_dtype("int64")


def _int(value: AbstractTensor) -> int:
    return int(value.item())


def _code_lengths(octets: AbstractTensor) -> AbstractTensor:
    frequencies = AbstractTensor.scatter(
        AbstractTensor.zeros((ALPHABET,), cls=type(octets)),
        octets,
        octets * 0 + 1,
        dim=0,
    ).to_dtype("int64")
    if _int((frequencies > 0).to_dtype("int64").sum()) == 1:
        # One distinct octet: a one-bit code still counts its repeats.
        return (frequencies > 0).to_dtype("int64")
    return length_limited_huffman_code_lengths(
        frequencies, max_bits=MAX_BITS,
    ).to_dtype("int64")


def _varint(out: bytearray, value: int) -> None:
    while True:
        low, value = value & 0x7F, value >> 7
        out.append(low | (0x80 if value else 0))
        if not value:
            return


def _read_varint(packet: bytes, at: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if at >= len(packet) or shift > 63:
            raise ValueError("octet Huffman packet header is truncated")
        octet = packet[at]
        at += 1
        value |= (octet & 0x7F) << shift
        shift += 7
        if not octet & 0x80:
            return value, at


def shared_code_lengths(sample: bytes) -> tuple[int, ...]:
    """A table for a stream of packets resembling ``sample``.

    Every octet gets a code (counts start at one), so any later packet can
    be coded with it; the observed frequencies set the lengths.
    """

    octets = _octets(bytes(sample)) if sample else None
    frequencies = AbstractTensor.zeros((ALPHABET,)).to_dtype("int64") + 1
    if octets is not None:
        frequencies = AbstractTensor.scatter(
            frequencies, octets, octets * 0 + 1, dim=0,
        ).to_dtype("int64")
    lengths = length_limited_huffman_code_lengths(frequencies, max_bits=MAX_BITS)
    return tuple(int(value) for value in lengths.tolist())


def encode_octets(data: bytes, lengths: tuple[int, ...] | None = None) -> bytes:
    """Canonical Huffman packet of ``data``; ``lengths`` codes it with a
    shared table (every octet of ``data`` must have a code in it)."""

    data = bytes(data)
    header = bytearray((VERSION if lengths is None else SHARED_VERSION,))
    _varint(header, len(data))
    if not data:
        _varint(header, 0)
        return bytes(header)
    octets = _octets(data)
    code_lengths = (
        _code_lengths(octets) if lengths is None
        else _octets(bytes(lengths))
    )
    table = CanonicalHuffmanTable.from_code_lengths(
        code_lengths, max_bits=MAX_BITS,
    )
    stream = compact_codewords(table.encode_codewords(octets))
    _varint(header, _int(stream.valid_bits))
    if lengths is None:
        pairs = code_lengths.reshape(-1, 2)
        header += tensor_octets_to_bytes(
            (pairs[:, 0] * 16 + pairs[:, 1]).to_dtype("int64")
        )
    return bytes(header) + stream.to_bytes()


def decode_octets(packet: bytes, lengths: tuple[int, ...] | None = None) -> bytes:
    """Inverse of :func:`encode_octets` with the same ``lengths``; refuses a
    corrupt packet."""

    packet = bytes(packet)
    expected = VERSION if lengths is None else SHARED_VERSION
    if not packet or packet[0] != expected:
        raise ValueError("octet Huffman packet has no valid header")
    count, at = _read_varint(packet, 1)
    bit_count, at = _read_varint(packet, at)
    if count == 0:
        if bit_count or len(packet) != at:
            raise ValueError("empty octet Huffman packet carries a stream")
        return b""
    table_end = at + (_TABLE if lengths is None else 0)
    payload = packet[table_end:]
    if (
        table_end > len(packet)
        or bit_count > 8 * len(payload) or len(payload) != (bit_count + 7) // 8
    ):
        raise ValueError("octet Huffman stream length disagrees with its header")
    if lengths is None:
        nibbles = _octets(packet[at:table_end])
        lengths = AbstractTensor.stack(
            (nibbles // 16, nibbles % 16), dim=1,
        ).reshape(-1)
    else:
        lengths = _octets(bytes(lengths))
    table = CanonicalHuffmanTable.from_code_lengths(lengths, max_bits=MAX_BITS)

    # Each code owns the left-aligned window range
    # [code << (MAX_BITS - length), (code + 1) << (MAX_BITS - length)).
    present = (lengths > 0).to_dtype("int64")
    span = (2 ** (MAX_BITS - lengths.maximum(1))) * present
    start = table.codes * span
    # The ranges are disjoint, so a summing scatter of +owner at each start
    # and -owner at each end, then a prefix sum, labels every window with its
    # owner + 1 (0 where no code covers it).
    alphabet = AbstractTensor.arange(ALPHABET, cls=type(lengths)).to_dtype("int64")
    edges = AbstractTensor.cat((start, start + span), dim=0).to_dtype("int64")

    def label(values):
        weight = (values * present).to_dtype("int64")
        marks = AbstractTensor.scatter(
            AbstractTensor.zeros((2 ** MAX_BITS + 1,), cls=type(lengths)),
            edges, AbstractTensor.cat((weight, weight * -1), dim=0), dim=0,
        )
        return marks.cumsum(dim=0)[: 2 ** MAX_BITS].to_dtype("int64")

    window_symbol = label(alphabet + 1) - 1
    window_length = label(lengths)

    # Every bit position's window, the stream padded with MAX_BITS zeros.
    divisors = 2 ** (7 - AbstractTensor.arange(8, cls=type(lengths)))
    bits = ((_octets(payload).unsqueeze(1) // divisors.unsqueeze(0)) % 2).flatten()
    bits = AbstractTensor.cat(
        (bits[:bit_count], AbstractTensor.zeros((MAX_BITS,), cls=type(bits))),
        dim=0,
    ).to_dtype("int64")
    value = bits[:bit_count] * 0
    for offset in range(MAX_BITS):
        value = value * 2 + bits[offset:offset + bit_count]
    symbol_at = window_symbol[value]
    length_at = window_length[value]

    # ``jump`` over positions 0..bit_count; bit_count is the absorbing end.
    # A position with no code jumps past the end, so it can never land on it.
    positions = AbstractTensor.arange(bit_count, cls=type(lengths))
    missing = (length_at == 0).to_dtype("int64")
    landing = positions + length_at + missing * (bit_count + 1)
    overrun = (landing > bit_count).to_dtype("int64")
    sentinel = bit_count + 1
    jump = AbstractTensor.cat((
        landing * (1 - overrun) + sentinel * overrun,
        AbstractTensor.tensor([bit_count, sentinel]).to_dtype("int64"),
    ), dim=0).to_dtype("int64")
    ordinals = AbstractTensor.arange(count, cls=type(lengths)).to_dtype("int64")
    starts = ordinals * 0
    end = AbstractTensor.tensor([0]).to_dtype("int64")
    power = 1
    while power <= count:
        taken = (ordinals // power) % 2
        starts = starts + taken * (jump[starts] - starts)
        if count & power:
            end = jump[end]
        jump = jump[jump]
        power *= 2
    if _int(end[0]) != bit_count or _int((starts - bit_count).max()) >= 0:
        raise ValueError("octet Huffman stream is corrupt")
    return tensor_octets_to_bytes(symbol_at[starts].to_dtype("int64"))


__all__ = [
    "MAX_BITS", "SHARED_VERSION", "VERSION", "decode_octets", "encode_octets",
    "shared_code_lengths",
]
