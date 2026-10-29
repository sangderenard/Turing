"""General octet/text compression: LZ77 matches, then canonical Huffman.

``compress_octets`` is ``octet_huffman.encode_octets(octet_lz.lz_encode(x))``
and ``decompress_octets`` its inverse.  Both stages are fixed sequences of
whole-tensor AbstractTensor operations (sorts, prefix sums, scatters and
pointer-doubling gathers); neither walks the input octet by octet.
"""

from __future__ import annotations

from .octet_huffman import decode_octets, encode_octets
from .octet_lz import lz_decode, lz_encode


def compress_octets(data: bytes) -> bytes:
    return encode_octets(lz_encode(data))


def decompress_octets(packet: bytes) -> bytes:
    return lz_decode(decode_octets(packet))


__all__ = ["compress_octets", "decompress_octets"]
