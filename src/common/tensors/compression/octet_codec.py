"""General octet/text compression: LZ77 matches, then canonical Huffman.

An LZ packet (``octet_lz``) is a header, a command stream (literal counts,
match lengths and distances as varints) and the literal octets.  The two
streams have very different statistics -- small integers against text -- so
each gets its own canonical Huffman table (``octet_huffman``), as DEFLATE
codes its literal/length and distance alphabets apart::

    LZ header | command packet length (varint) | Huffman(commands) | Huffman(literals)

Every stage is a fixed sequence of whole-tensor AbstractTensor operations
(sorts, prefix sums, scatters and pointer-doubling gathers); none walks the
input octet by octet.

A stream of small packets -- the tiles of one table -- shares an
:class:`OctetCodebook`: a dictionary every packet may copy from as though it
preceded it, and one Huffman table per stream.  Each packet then pays neither
for its own tables nor for re-introducing text the stream has already seen.
"""

from __future__ import annotations

from dataclasses import dataclass

from .octet_huffman import decode_octets, encode_octets, shared_code_lengths
from .octet_lz import _header, _read_header, lz_decode, lz_encode

#: The longest shared dictionary: the tail of the training sample.
DICTIONARY_LIMIT = 8 << 10


def _split(packet: bytes) -> tuple[bytes, bytes, bytes]:
    (_n, _groups, command_bytes), at = _read_header(packet, 3)
    return packet[:at], packet[at:at + command_bytes], packet[at + command_bytes:]


@dataclass(frozen=True, slots=True)
class OctetCodebook:
    """What every packet of one stream is coded against."""

    dictionary: bytes
    command_lengths: tuple[int, ...]
    literal_lengths: tuple[int, ...]

    @classmethod
    def train(cls, sample: bytes) -> "OctetCodebook":
        """A codebook for packets resembling ``sample``.

        The tail of ``sample`` is the dictionary; the tables are fitted to
        what later packets carry: ``sample``'s head coded against it.
        """

        sample = bytes(sample)
        dictionary = sample[-DICTIONARY_LIMIT:]
        half = max(1, len(sample) // 2)
        _header_part, commands, literals = _split(
            lz_encode(sample[half:], sample[:half][-DICTIONARY_LIMIT:])
        )
        return cls(
            dictionary,
            shared_code_lengths(commands),
            shared_code_lengths(literals),
        )


def compress_octets(data: bytes, codebook: OctetCodebook | None = None) -> bytes:
    dictionary = b"" if codebook is None else codebook.dictionary
    header, commands, literals = _split(lz_encode(data, dictionary))
    coded_commands = encode_octets(
        commands, None if codebook is None else codebook.command_lengths,
    )
    coded_literals = encode_octets(
        literals, None if codebook is None else codebook.literal_lengths,
    )
    return header + _header(len(coded_commands)) + coded_commands + coded_literals


def decompress_octets(
    packet: bytes, codebook: OctetCodebook | None = None,
) -> bytes:
    packet = bytes(packet)
    (n, groups, command_bytes), at = _read_header(packet, 3)
    (coded_length,), used = _read_header(packet[at:], 1)
    start = at + used
    if start + coded_length > len(packet):
        raise ValueError("octet codec packet is truncated")
    commands = decode_octets(
        packet[start:start + coded_length],
        None if codebook is None else codebook.command_lengths,
    )
    literals = decode_octets(
        packet[start + coded_length:],
        None if codebook is None else codebook.literal_lengths,
    )
    if len(commands) != command_bytes:
        raise ValueError("octet codec command stream disagrees with its header")
    return lz_decode(
        _header(n, groups, command_bytes) + commands + literals,
        b"" if codebook is None else codebook.dictionary,
    )


__all__ = [
    "DICTIONARY_LIMIT", "OctetCodebook", "compress_octets", "decompress_octets",
]
