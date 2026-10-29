"""Self-describing canonical Huffman coding of an octet string.

A packet is the canonical table followed by the codeword stream:

```text
version (1) | raw length (8) | valid bits (8) | 256 code lengths (128) | stream
```

Code lengths are packed two to an octet (high nibble first) and limited to
``MAX_BITS``, so the table is the 128-octet header JPEG's DHT idea implies
and no tree is stored.  Encoding counts the octet alphabet with one summing
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

#: Packet layout version.
VERSION = 1
#: Longest code.  2**MAX_BITS bounds the decode lookup; 256 symbols need 8.
MAX_BITS = 12
ALPHABET = 256
_HEADER = 1 + 8 + 8 + ALPHABET // 2


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


def encode_octets(data: bytes) -> bytes:
    """Canonical Huffman packet of ``data``."""

    data = bytes(data)
    header = bytearray((VERSION,))
    header += len(data).to_bytes(8, "little")
    if not data:
        return bytes(header + (0).to_bytes(8, "little") + bytes(ALPHABET // 2))
    octets = _octets(data)
    lengths = _code_lengths(octets)
    table = CanonicalHuffmanTable.from_code_lengths(lengths, max_bits=MAX_BITS)
    stream = compact_codewords(table.encode_codewords(octets))
    header += _int(stream.valid_bits).to_bytes(8, "little")
    pairs = lengths.reshape(-1, 2)
    header += tensor_octets_to_bytes(
        (pairs[:, 0] * 16 + pairs[:, 1]).to_dtype("int64")
    )
    return bytes(header) + stream.to_bytes()


def decode_octets(packet: bytes) -> bytes:
    """Inverse of :func:`encode_octets`; refuses a corrupt packet."""

    packet = bytes(packet)
    if len(packet) < _HEADER or packet[0] != VERSION:
        raise ValueError("octet Huffman packet has no valid header")
    count = int.from_bytes(packet[1:9], "little")
    bit_count = int.from_bytes(packet[9:17], "little")
    payload = packet[_HEADER:]
    if count == 0:
        if bit_count or payload:
            raise ValueError("empty octet Huffman packet carries a stream")
        return b""
    if bit_count > 8 * len(payload) or len(payload) != (bit_count + 7) // 8:
        raise ValueError("octet Huffman stream length disagrees with its header")
    nibbles = _octets(packet[17:_HEADER])
    lengths = AbstractTensor.stack(
        (nibbles // 16, nibbles % 16), dim=1,
    ).reshape(-1)
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


__all__ = ["MAX_BITS", "VERSION", "decode_octets", "encode_octets"]
