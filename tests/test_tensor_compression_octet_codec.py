"""AbstractTensor octet codec (LZ77 + canonical Huffman) and the tile row
codec that replaced pickle in ``CompressedTable``."""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass

import pytest

from src.common.tensors.compression.octet_codec import (
    compress_octets, decompress_octets,
)
from src.common.tensors.compression.octet_huffman import decode_octets, encode_octets
from src.common.tensors.compression.octet_lz import lz_decode, lz_encode
from src.compiler.tile_row_codec import decode_rows, encode_rows


def _samples():
    generator = random.Random(7)
    yield from (b"", b"a", b"abcd", b"a" * 50, b"abcabcabcabcx", bytes(range(256)))
    yield bytes(generator.randrange(256) for _ in range(3000))
    yield b"".join(("repeated fact " * 40 + str(i)).encode() for i in range(32))
    for _ in range(60):
        alphabet = generator.choice((b"ab", b"abc", bytes(range(256))))
        yield bytes(
            generator.choice(alphabet) for _ in range(generator.randrange(80))
        )


@pytest.mark.parametrize("encode, decode", [
    (lz_encode, lz_decode),
    (encode_octets, decode_octets),
    (compress_octets, decompress_octets),
])
def test_every_stage_round_trips_exactly(encode, decode):
    for data in _samples():
        assert decode(encode(data)) == data


def test_repeated_text_compresses_and_noise_does_not_grow_much():
    text = b"".join(("repeated fact " * 40 + str(i)).encode() for i in range(32))
    assert len(compress_octets(text)) * 40 < len(text)
    noise = bytes(random.Random(3).randrange(256) for _ in range(4000))
    assert len(compress_octets(noise)) < len(noise) + 200


def test_a_damaged_packet_refuses():
    packet = bytearray(compress_octets(b"stable repeated text " * 50))
    packet[-3] ^= 0xFF
    with pytest.raises(ValueError):
        decompress_octets(bytes(packet))
    with pytest.raises(ValueError):
        lz_decode(b"\x05" + bytes(23))


class Colour(enum.Enum):
    RED = "red"


@dataclass(frozen=True, slots=True)
class Fact:
    name: str
    value: object


class Plain:
    def __init__(self, n):
        self.n = n

    def __eq__(self, other):
        return type(other) is Plain and other.n == self.n


def test_rows_round_trip_every_value_shape():
    rows = (
        (("scope", 1), Fact("x", (1.5, -2**80, None, True, b"\x00", [1, {2: "y"}]))),
        (Colour.RED, frozenset({1, 2})),
        (Fact, Plain(3)),
        ("text", "text"),
    )
    assert decode_rows(encode_rows(rows)) == rows


def test_rows_refuse_what_has_no_exact_inverse():
    with pytest.raises(TypeError, match="module-level name"):
        encode_rows(((lambda: 0, 1),))
    cycle = []
    cycle.append(cycle)
    with pytest.raises(TypeError, match="refers to itself"):
        encode_rows((cycle,))


def _named(name):
    return {"red": Colour.RED}[name]


def test_a_reducer_writes_a_value_by_name():
    rows = ((Colour.RED, 1),)
    reducers = {Colour: lambda value: (_named, (value.value,))}
    assert decode_rows(encode_rows(rows, reducers)) == rows


def test_a_shared_codebook_round_trips_and_shrinks_a_stream_of_tiles():
    from src.common.tensors.compression.octet_codec import OctetCodebook

    def tile(start):
        return encode_rows(tuple(
            (("scope", i), ("repeated fact " * 40 + str(i), i))
            for i in range(start, start + 32)
        ))

    codebook = OctetCodebook.train(tile(0) + tile(32) + tile(64))
    later = tile(96)
    shared = compress_octets(later, codebook)
    assert decompress_octets(shared, codebook) == later
    assert len(shared) < len(compress_octets(later))
    with pytest.raises(ValueError):
        decompress_octets(shared)


def test_a_table_trains_its_codebook_after_its_first_tiles():
    from src.compiler.compressed_tiles import CODEBOOK_AFTER_TILES, CompressedTable

    rows = {i: "node %d stage ingest" % i for i in range(32 * (CODEBOOK_AFTER_TILES + 2))}
    table = CompressedTable(rows.items(), tile_rows=32)
    tiles = table._tiles
    assert all(tile.codebook is None for tile in tiles[:CODEBOOK_AFTER_TILES])
    assert all(tile.codebook is table._codebook for tile in tiles[CODEBOOK_AFTER_TILES:])
    assert dict(table.items()) == rows
