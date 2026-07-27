import pytest
from itertools import product

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.compression.huffman import CanonicalHuffmanTable
from src.common.tensors.compression.huffman import huffman_code_lengths
from src.common.tensors.compression.huffman import (
    length_limited_huffman_code_lengths,
)
from src.common.tensors.compression.bitstream import (
    compact_codewords,
    decode_huffman_octets,
    decode_with_provenance,
    unpack_octets,
)
from src.common.tensors.compression.jpeg.huffman import (
    jpeg_standard_dc_luminance,
)


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_jpeg_dc_luminance_canonical_codes_are_backend_independent(backend):
    with AT.use_backend(backend):
        like = AT.tensor([0])
        table = jpeg_standard_dc_luminance(like)
        symbols = AT.tensor(list(range(12)))
        codes, lengths = table.lookup(symbols)

    assert lengths.tolist() == [2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9]
    assert codes.tolist() == [
        0b00,
        0b010,
        0b011,
        0b100,
        0b101,
        0b110,
        0b1110,
        0b11110,
        0b111110,
        0b1111110,
        0b11111110,
        0b111111110,
    ]


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_parallel_codeword_expansion_is_msb_first(backend):
    with AT.use_backend(backend):
        table = jpeg_standard_dc_luminance(AT.tensor([0]))
        encoded = table.encode_codewords(AT.tensor([0, 1, 6, 11]))

    assert encoded.lengths.tolist() == [2, 3, 4, 9]
    rows = encoded.bits.tolist()
    assert rows[0][:2] == [0, 0]
    assert rows[1][:3] == [0, 1, 0]
    assert rows[2][:4] == [1, 1, 1, 0]
    assert rows[3][:9] == [1, 1, 1, 1, 1, 1, 1, 1, 0]
    assert encoded.valid.sum().item() == 18


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_codewords_compact_to_real_msb_first_octets(backend):
    with AT.use_backend(backend):
        table = jpeg_standard_dc_luminance(AT.tensor([0]))
        codewords = table.encode_codewords(AT.tensor([0, 1, 6, 11]))
        packed = compact_codewords(codewords)

    assert packed.valid_bits.item() == 18
    assert packed.symbol_offsets.tolist() == [0, 2, 5, 9]
    assert packed.to_bytes() == bytes((0x17, 0x7F, 0x80))


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_packed_huffman_round_trip_is_exact(backend):
    expected = [11, 0, 3, 6, 1, 10, 2, 2, 9]
    with AT.use_backend(backend):
        table = jpeg_standard_dc_luminance(AT.tensor([0]))
        packed = compact_codewords(
            table.encode_codewords(AT.tensor(expected))
        )
        unpacked = unpack_octets(packed)
        decoded = decode_with_provenance(table, packed)

    assert unpacked.valid.sum().item() == packed.valid_bits.item()
    assert decoded.tolist() == expected


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_frequency_table_round_trip_is_exact(backend):
    expected = [5, 0, 5, 2, 1, 4, 3, 5, 2, 0]
    with AT.use_backend(backend):
        table = CanonicalHuffmanTable.from_frequencies(
            AT.tensor([5, 9, 12, 13, 16, 45])
        )
        packed = compact_codewords(
            table.encode_codewords(AT.tensor(expected))
        )
        decoded = decode_with_provenance(table, packed)

    assert decoded.tolist() == expected


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_wire_decoder_needs_only_octets_and_valid_bit_count(backend):
    expected = [11, 0, 3, 6, 1, 10, 2, 2, 9]
    with AT.use_backend(backend):
        table = jpeg_standard_dc_luminance(AT.tensor([0]))
        packed = compact_codewords(
            table.encode_codewords(AT.tensor(expected))
        )
        decoded = decode_huffman_octets(
            table,
            packed.octets[:packed.byte_count],
            packed.valid_bits,
        )

    assert decoded.symbol_count.item() == len(expected)
    assert decoded.valid.sum().item() == len(expected)
    assert decoded.tolist() == expected


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_explicit_integer_alphabet_round_trip(backend):
    alphabet = [-7, 42, 1000, 3]
    expected = [1000, -7, 42, 1000, 3, -7]
    with AT.use_backend(backend):
        table = CanonicalHuffmanTable.from_frequencies(
            AT.tensor([5, 2, 9, 1]),
            symbols=AT.tensor(alphabet),
        )
        packed = compact_codewords(
            table.encode_codewords(AT.tensor(expected))
        )
        decoded = decode_huffman_octets(
            table,
            packed.octets[:packed.byte_count],
            packed.valid_bits,
        )

    assert decoded.tolist() == expected


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_one_symbol_source_has_a_decodable_one_bit_code(backend):
    expected = [99, 99, 99, 99, 99]
    with AT.use_backend(backend):
        table = CanonicalHuffmanTable.from_frequencies(
            AT.tensor([0, 12, 0]),
            symbols=AT.tensor([5, 99, 1000]),
        )
        packed = compact_codewords(
            table.encode_codewords(AT.tensor(expected))
        )
        decoded = decode_huffman_octets(
            table,
            packed.octets[:packed.byte_count],
            packed.valid_bits,
        )

    assert table.lengths.tolist() == [0, 1, 0]
    assert decoded.tolist() == expected


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_optional_length_limit_builds_a_valid_optimal_tree(backend):
    frequencies = [1, 1, 2, 3, 5, 8, 13, 21]
    with AT.use_backend(backend):
        lengths = length_limited_huffman_code_lengths(
            AT.tensor(frequencies), max_bits=4
        )
        table = CanonicalHuffmanTable.from_frequencies(
            AT.tensor(frequencies), max_bits=4
        )

    assert max(lengths.tolist()) == 4
    assert table.lengths.tolist() == lengths.tolist()
    assert sum(
        2 ** (4 - int(length)) for length in lengths.tolist()
    ) == 2 ** 4


@pytest.mark.parametrize(
    ("frequencies", "max_bits"),
    [
        ([1, 1, 1], 2),
        ([1, 2, 3, 9], 3),
        ([1, 1, 2, 3, 5], 3),
        ([1, 1, 1, 1, 1, 100], 3),
    ],
)
def test_length_limited_cost_matches_exhaustive_small_optimum(
    frequencies, max_bits
):
    with AT.use_backend("numpy"):
        lengths = length_limited_huffman_code_lengths(
            AT.tensor(frequencies), max_bits=max_bits
        ).tolist()

    target_units = 2 ** max_bits
    feasible = (
        candidate
        for candidate in product(
            range(1, max_bits + 1), repeat=len(frequencies)
        )
        if sum(2 ** (max_bits - length) for length in candidate)
        == target_units
    )
    optimal_cost = min(
        sum(weight * length for weight, length in zip(frequencies, candidate))
        for candidate in feasible
    )
    actual_cost = sum(
        weight * int(length) for weight, length in zip(frequencies, lengths)
    )
    assert actual_cost == optimal_cost


def test_impossible_length_limit_is_rejected():
    with AT.use_backend("numpy"):
        with pytest.raises(ValueError, match="cannot fit"):
            length_limited_huffman_code_lengths(
                AT.tensor([1, 1, 1, 1, 1]), max_bits=2
            )


def test_wire_decoder_rejects_truncated_codeword():
    with AT.use_backend("numpy"):
        table = jpeg_standard_dc_luminance(AT.tensor([0]))
        packed = compact_codewords(
            table.encode_codewords(AT.tensor([11]))
        )
        with pytest.raises(ValueError, match="incomplete"):
            decode_huffman_octets(
                table,
                packed.octets[:packed.byte_count],
                AT.tensor(8),
            )


def test_wire_decoder_rejects_unknown_prefix():
    with AT.use_backend("numpy"):
        table = CanonicalHuffmanTable.from_code_lengths(
            AT.tensor([1]), max_bits=1
        )
        with pytest.raises(ValueError, match="invalid"):
            decode_huffman_octets(
                table,
                AT.tensor([0x80]),
                AT.tensor(1),
            )


def test_kraft_overflow_is_rejected():
    with AT.use_backend("numpy"):
        with pytest.raises(ValueError, match="Kraft"):
            CanonicalHuffmanTable.from_code_lengths(
                AT.tensor([1, 1, 1]), max_bits=2
            )


def test_missing_symbol_is_rejected():
    with AT.use_backend("numpy"):
        table = CanonicalHuffmanTable.from_code_lengths(
            AT.tensor([1, 0]), max_bits=2
        )
        with pytest.raises(ValueError, match="no code"):
            table.lookup(AT.tensor([1]))


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_frequency_tree_is_built_in_abstract_tensor(backend):
    with AT.use_backend(backend):
        frequencies = AT.tensor([5, 9, 12, 13, 16, 45])
        lengths = huffman_code_lengths(frequencies)
        table = CanonicalHuffmanTable.from_frequencies(frequencies)

    assert lengths.tolist() == [4, 4, 3, 3, 3, 1]
    assert table.lengths.tolist() == [4, 4, 3, 3, 3, 1]
