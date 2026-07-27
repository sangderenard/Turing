import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.compression.bitstream import (
    compact_codewords,
    decode_huffman_octets,
)
from src.common.tensors.compression.coefficient_events import (
    collect_block_coefficient_events,
    reconstruct_block_coefficients,
)
from src.common.tensors.compression.entropy_symbols import (
    ac_entropy_tokens,
    coefficient_events_to_entropy_symbols,
    entropy_symbols_to_coefficient_events,
)
from src.common.tensors.compression.entropy_scan import (
    decode_entropy_octets,
    encode_entropy_sequence,
)
from src.common.tensors.compression.huffman import CanonicalHuffmanTable


COEFFICIENTS = [
    [[10, 0, 0, -3, 0, 2, 0, 0]],
    [[13, 0, 0, 0, 0, 0, 0, 0]],
    [[8, 4, 0, 0, 0, -1, 0, 6]],
]


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_coefficient_events_map_to_reversible_entropy_symbols(backend):
    with AT.use_backend(backend):
        original = AT.tensor(COEFFICIENTS)
        events = collect_block_coefficient_events(
            original, max_magnitude_bits=8
        )
        streams = coefficient_events_to_entropy_symbols(events)
        recovered_events = entropy_symbols_to_coefficient_events(streams)
        reconstructed = reconstruct_block_coefficients(recovered_events)

    assert streams.dc.symbols.tolist() == [4, 2, 3]
    assert streams.ac.symbols.tolist() == [
        [21, 12, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [4, 29, 13, 0, 0, 0, 0, 0],
    ]
    assert streams.ac.valid.tolist() == [
        [True, True, True, False, False, False, False, False],
        [True, False, False, False, False, False, False, False],
        [True, True, True, True, False, False, False, False],
    ]
    assert reconstructed.tolist() == COEFFICIENTS


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_ac_entropy_symbols_feed_general_huffman_end_to_end(backend):
    with AT.use_backend(backend):
        events = collect_block_coefficient_events(
            AT.tensor(COEFFICIENTS), max_magnitude_bits=8
        )
        streams = coefficient_events_to_entropy_symbols(events)
        compact = streams.ac.compact()
        alphabet = AT.arange(streams.ac_alphabet_size)
        table = CanonicalHuffmanTable.from_samples(
            compact.symbols,
            alphabet,
            valid=compact.valid,
            max_bits=4,
        )
        valid_symbols = compact.symbols[
            :int(compact.count.item())
        ]
        packed = compact_codewords(
            table.encode_codewords(valid_symbols)
        )
        decoded = decode_huffman_octets(
            table,
            packed.octets[:packed.byte_count],
            packed.valid_bits,
        )

    assert decoded.tolist() == streams.ac.to_symbol_list()


def test_entropy_symbol_alphabet_includes_eob_and_all_run_categories():
    with AT.use_backend("numpy"):
        events = collect_block_coefficient_events(
            AT.tensor(COEFFICIENTS), max_magnitude_bits=8
        )
        streams = coefficient_events_to_entropy_symbols(events)

    assert streams.ac_category_radix == 9
    assert streams.ac_alphabet_size == 64


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_huffman_and_amplitude_bits_share_one_decodable_scan(backend):
    with AT.use_backend(backend):
        events = collect_block_coefficient_events(
            AT.tensor(COEFFICIENTS), max_magnitude_bits=8
        )
        streams = coefficient_events_to_entropy_symbols(events)

        dc_alphabet = AT.arange(streams.max_magnitude_bits + 1)
        dc_table = CanonicalHuffmanTable.from_samples(
            streams.dc.symbols,
            dc_alphabet,
            valid=streams.dc.valid,
            max_bits=3,
        )
        dc_scan = encode_entropy_sequence(dc_table, streams.dc)
        decoded_dc = decode_entropy_octets(
            dc_table,
            dc_alphabet,
            dc_scan.octets[:dc_scan.byte_count],
            dc_scan.valid_bits,
        )

        ac_compact = streams.ac.compact()
        ac_alphabet = AT.arange(streams.ac_alphabet_size)
        ac_table = CanonicalHuffmanTable.from_samples(
            ac_compact.symbols,
            ac_alphabet,
            valid=ac_compact.valid,
            max_bits=4,
        )
        ac_scan = encode_entropy_sequence(ac_table, streams.ac)
        decoded_ac = decode_entropy_octets(
            ac_table,
            streams.ac_payload_lengths(ac_alphabet),
            ac_scan.octets[:ac_scan.byte_count],
            ac_scan.valid_bits,
        )
        expected_dc = streams.dc.compact()
        expected_ac = streams.ac.compact()

    dc_count = int(decoded_dc.count.item())
    ac_count = int(decoded_ac.count.item())
    assert decoded_dc.symbols[:dc_count].tolist() == (
        expected_dc.symbols[:dc_count].tolist()
    )
    assert decoded_dc.payloads[:dc_count].tolist() == (
        expected_dc.payloads[:dc_count].tolist()
    )
    assert decoded_dc.payload_lengths[:dc_count].tolist() == (
        expected_dc.payload_lengths[:dc_count].tolist()
    )
    assert decoded_ac.symbols[:ac_count].tolist() == (
        expected_ac.symbols[:ac_count].tolist()
    )
    assert decoded_ac.payloads[:ac_count].tolist() == (
        expected_ac.payloads[:ac_count].tolist()
    )
    assert decoded_ac.payload_lengths[:ac_count].tolist() == (
        expected_ac.payload_lengths[:ac_count].tolist()
    )


def test_interleaved_scan_rejects_truncated_amplitude():
    with AT.use_backend("numpy"):
        events = collect_block_coefficient_events(
            AT.tensor(COEFFICIENTS), max_magnitude_bits=8
        )
        streams = coefficient_events_to_entropy_symbols(events)
        alphabet = AT.arange(streams.max_magnitude_bits + 1)
        table = CanonicalHuffmanTable.from_samples(
            streams.dc.symbols,
            alphabet,
            valid=streams.dc.valid,
            max_bits=3,
        )
        scan = encode_entropy_sequence(table, streams.dc)
        with pytest.raises(ValueError, match="incomplete"):
            decode_entropy_octets(
                table,
                alphabet,
                scan.octets[:scan.byte_count],
                scan.valid_bits - 1,
            )


def test_actual_ac_token_region_runs_directly_on_glsl():
    from src.common.tensors.accelerator_backends.glsl_backend import (
        GLContextUnavailable,
        require_gl_context,
    )

    try:
        require_gl_context()
    except GLContextUnavailable as exc:
        pytest.skip(f"no OpenGL compute context: {exc}")

    with AT.use_backend("glsl"):
        runs = AT.tensor([2, 1, 0, 3])
        categories = AT.tensor([2, 2, 3, 1])
        valid = AT.tensor([1, 1, 0, 1])
        tokens = ac_entropy_tokens(
            runs, categories, valid, category_radix=9
        )

    assert tokens.tolist() == [21, 12, 0, 29]


def test_complete_entropy_scan_runs_directly_on_glsl_backend():
    from src.common.tensors.accelerator_backends.glsl_backend import (
        GLContextUnavailable,
        require_gl_context,
    )
    try:
        require_gl_context()
    except GLContextUnavailable as exc:
        pytest.skip(f"no OpenGL compute context: {exc}")

    with AT.use_backend("glsl"):
        events = collect_block_coefficient_events(
            AT.tensor(COEFFICIENTS), max_magnitude_bits=8
        )
        streams = coefficient_events_to_entropy_symbols(events)
        compact = streams.ac.compact()
        alphabet = AT.arange(streams.ac_alphabet_size)
        table = CanonicalHuffmanTable.from_samples(
            compact.symbols,
            alphabet,
            valid=compact.valid,
            max_bits=4,
        )
        scan = encode_entropy_sequence(table, streams.ac)
        decoded = decode_entropy_octets(
            table,
            streams.ac_payload_lengths(alphabet),
            scan.octets[:scan.byte_count],
            scan.valid_bits,
        )

    assert scan.to_bytes() == bytes.fromhex("e58245b0")
    assert int(scan.valid_bits.item()) == 31
    assert decoded.to_symbol_list() == streams.ac.to_symbol_list()
