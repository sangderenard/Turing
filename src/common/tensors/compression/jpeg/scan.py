"""Baseline JPEG scan adaptation over neutral coefficient events."""

from __future__ import annotations

from dataclasses import dataclass

from ...abstraction import AbstractTensor
from ..bitstream import PackedBitstream, compact_codewords
from ..coefficient_events import (
    BlockCoefficientEvents,
    concatenate_block_coefficient_events,
)
from ..entropy_scan import interleave_codewords_and_payloads
from ..entropy_symbols import EntropySymbolSequence
from ..huffman import HuffmanCodewords
from .huffman import (
    jpeg_standard_ac_chrominance,
    jpeg_standard_ac_luminance,
    jpeg_standard_dc_chrominance,
    jpeg_standard_dc_luminance,
)


JPEG_AC_CAPACITY = 64
JPEG_ZRL = 0xF0


def jpeg_ac_entropy_symbols(
    events: BlockCoefficientEvents,
) -> EntropySymbolSequence:
    """Translate neutral AC events into baseline JPEG run/category symbols."""
    ac_count = events.coefficient_count - 1
    if ac_count != 63:
        raise ValueError("baseline JPEG requires 64-coefficient blocks")
    runs = events.ac_zero_runs
    event_valid = events.ac_valid.to_dtype("int64")
    zrl_count = runs // 16
    remainder = runs % 16
    subslots = AbstractTensor.arange(4, cls=type(runs))
    before_actual = (
        zrl_count.unsqueeze(2) - subslots.reshape(1, 1, 4)
    ) > 0
    is_actual = (
        zrl_count.unsqueeze(2) - subslots.reshape(1, 1, 4)
    ) == 0
    candidate_valid = (
        (
            zrl_count.unsqueeze(2) - subslots.reshape(1, 1, 4)
        ) >= 0
    ).to_dtype("int64") * event_valid.unsqueeze(2)
    zrl_mask = before_actual.to_dtype("int64") * event_valid.unsqueeze(2)
    actual_mask = is_actual.to_dtype("int64") * event_valid.unsqueeze(2)
    actual_symbol = (
        remainder * 16 + events.ac.categories
    ).unsqueeze(2)
    candidate_symbols = (
        JPEG_ZRL * zrl_mask + actual_symbol * actual_mask
    )
    candidate_payloads = (
        events.ac.payloads.unsqueeze(2) * actual_mask
    )
    candidate_lengths = (
        events.ac.categories.unsqueeze(2) * actual_mask
    )

    block_count = runs.shape[0]
    candidate_width = ac_count * 4
    flat_symbols = candidate_symbols.reshape(block_count, candidate_width)
    flat_payloads = candidate_payloads.reshape(block_count, candidate_width)
    flat_lengths = candidate_lengths.reshape(block_count, candidate_width)
    flat_valid = candidate_valid.reshape(block_count, candidate_width)
    eob_valid = (events.trailing_zeros > 0).to_dtype("int64").unsqueeze(1)
    zero_column = AbstractTensor.zeros(
        (block_count, 1), cls=type(flat_symbols)
    )
    symbols = AbstractTensor.cat((flat_symbols, zero_column), dim=1)
    payloads = AbstractTensor.cat((flat_payloads, zero_column), dim=1)
    lengths = AbstractTensor.cat((flat_lengths, zero_column), dim=1)
    validity = AbstractTensor.cat((flat_valid, eob_valid), dim=1)

    token_counts = validity.sum(dim=1)
    ranks = validity.cumsum(dim=1) - 1
    source_width = validity.shape[1]
    block_offsets = (
        AbstractTensor.arange(block_count, cls=type(validity))
        * JPEG_AC_CAPACITY
    )
    scratch = block_count * JPEG_AC_CAPACITY
    destinations = (
        (block_offsets.unsqueeze(1) + ranks) * validity
        + scratch * (1 - validity)
    ).to_dtype("int64")

    def compact_rows(field: AbstractTensor) -> AbstractTensor:
        target = AbstractTensor.zeros(
            (scratch + 1,), cls=type(field)
        )
        target = AbstractTensor.scatter(
            target,
            destinations.flatten(),
            (field * validity).flatten(),
            dim=0,
        )
        return target[:scratch].reshape(
            block_count, JPEG_AC_CAPACITY
        )

    slots = AbstractTensor.arange(
        JPEG_AC_CAPACITY, cls=type(validity)
    )
    compact_valid = (token_counts.unsqueeze(1) - slots.unsqueeze(0)) > 0
    return EntropySymbolSequence(
        symbols=compact_rows(symbols),
        payloads=compact_rows(payloads),
        payload_lengths=compact_rows(lengths),
        valid=compact_valid,
    )


def _encode_block_codewords(
    events: BlockCoefficientEvents,
    *,
    dc_table,
    ac_table,
) -> HuffmanCodewords:
    """Return fixed-shape DC/AC bit rows while retaining block boundaries."""
    if events.coefficient_count != 64:
        raise ValueError("baseline JPEG requires 64-coefficient blocks")
    dc_valid = (events.dc.categories * 0 + 1) > 0
    dc_sequence = EntropySymbolSequence(
        symbols=events.dc.categories,
        payloads=events.dc.payloads,
        payload_lengths=events.dc.categories,
        valid=dc_valid,
    )
    ac_sequence = jpeg_ac_entropy_symbols(events)

    dc_codewords = dc_table.encode_codewords(
        dc_sequence.symbols, validate=False
    )
    dc_rows = interleave_codewords_and_payloads(
        dc_codewords,
        dc_sequence.payloads,
        dc_sequence.payload_lengths,
        max_payload_bits=11,
        validate=False,
    )

    block_count = events.dc.categories.shape[0]
    ac_symbols = ac_sequence.symbols.flatten()
    ac_codewords = ac_table.encode_codewords(
        ac_symbols, validate=False
    )
    ac_rows = interleave_codewords_and_payloads(
        ac_codewords,
        ac_sequence.payloads.flatten(),
        ac_sequence.payload_lengths.flatten(),
        max_payload_bits=11,
        validate=False,
    )
    row_width = dc_rows.max_bits
    if ac_rows.max_bits != row_width:
        raise RuntimeError("DC and AC entropy rows did not share a width")

    ac_symbol_valid = ac_sequence.valid.flatten().to_dtype("int64")
    ac_bits = (
        ac_rows.bits.reshape(-1, row_width)
        * ac_symbol_valid.unsqueeze(1)
    )
    ac_lengths = ac_rows.lengths * ac_symbol_valid
    ac_bit_valid = (
        ac_rows.valid.reshape(-1, row_width)
        & ac_sequence.valid.flatten().unsqueeze(1)
    )

    block_bits = AbstractTensor.cat(
        (
            dc_rows.bits.reshape(block_count, 1, row_width),
            ac_bits.reshape(
                block_count, JPEG_AC_CAPACITY, row_width
            ),
        ),
        dim=1,
    )
    block_lengths = AbstractTensor.cat(
        (
            dc_rows.lengths.reshape(block_count, 1),
            ac_lengths.reshape(block_count, JPEG_AC_CAPACITY),
        ),
        dim=1,
    )
    block_valid = AbstractTensor.cat(
        (
            dc_rows.valid.reshape(block_count, 1, row_width),
            ac_bit_valid.reshape(
                block_count, JPEG_AC_CAPACITY, row_width
            ),
        ),
        dim=1,
    )
    flattened_lengths = block_lengths.flatten()
    return HuffmanCodewords(
        codes=AbstractTensor.zeros(
            flattened_lengths.shape, cls=type(flattened_lengths)
        ),
        lengths=flattened_lengths,
        bits=block_bits.reshape(-1, row_width),
        valid=block_valid.reshape(-1, row_width),
        max_bits=row_width,
    )


def encode_baseline_luma_scan(
    events: BlockCoefficientEvents,
    *,
    dc_table=None,
    ac_table=None,
) -> PackedBitstream:
    """Encode block-interleaved DC and AC data using standard luma tables."""
    if dc_table is None:
        dc_table = jpeg_standard_dc_luminance(events.dc.categories)
    if ac_table is None:
        ac_table = jpeg_standard_ac_luminance(events.ac.categories)
    codewords = _encode_block_codewords(
        events,
        dc_table=dc_table,
        ac_table=ac_table,
    )
    return compact_codewords(codewords)


def encode_baseline_color_scan(
    y_events: BlockCoefficientEvents,
    cb_events: BlockCoefficientEvents,
    cr_events: BlockCoefficientEvents,
    *,
    luma_dc_table=None,
    luma_ac_table=None,
    chroma_dc_table=None,
    chroma_ac_table=None,
) -> PackedBitstream:
    """Encode a 4:4:4 scan in MCU order: Y block, Cb block, Cr block."""
    block_count = y_events.dc.categories.shape[0]
    if (
        cb_events.dc.categories.shape[0] != block_count
        or cr_events.dc.categories.shape[0] != block_count
    ):
        raise ValueError("4:4:4 JPEG components must have equal block counts")
    return encode_baseline_color_component_scan(
        y_events,
        concatenate_block_coefficient_events((cb_events, cr_events)),
        luma_dc_table=luma_dc_table,
        luma_ac_table=luma_ac_table,
        chroma_dc_table=chroma_dc_table,
        chroma_ac_table=chroma_ac_table,
    )


def encode_baseline_color_component_scan(
    y_events: BlockCoefficientEvents,
    chroma_events: BlockCoefficientEvents,
    *,
    luma_dc_table=None,
    luma_ac_table=None,
    chroma_dc_table=None,
    chroma_ac_table=None,
) -> PackedBitstream:
    """Encode luma blocks followed by contiguous Cb and Cr event blocks."""

    block_count = y_events.dc.categories.shape[0]
    if chroma_events.dc.categories.shape[0] != block_count * 2:
        raise ValueError("4:4:4 chroma events must contain Cb then Cr blocks")
    if luma_dc_table is None:
        luma_dc_table = jpeg_standard_dc_luminance(y_events.dc.categories)
    if luma_ac_table is None:
        luma_ac_table = jpeg_standard_ac_luminance(y_events.ac.categories)
    if chroma_dc_table is None:
        chroma_dc_table = jpeg_standard_dc_chrominance(
            chroma_events.dc.categories
        )
    if chroma_ac_table is None:
        chroma_ac_table = jpeg_standard_ac_chrominance(
            chroma_events.ac.categories
        )

    y_words = _encode_block_codewords(
        y_events,
        dc_table=luma_dc_table,
        ac_table=luma_ac_table,
    )
    chroma_words = _encode_block_codewords(
        chroma_events,
        dc_table=chroma_dc_table,
        ac_table=chroma_ac_table,
    )
    chroma_symbol_count = block_count * (JPEG_AC_CAPACITY + 1)
    cb_slice = slice(0, chroma_symbol_count)
    cr_slice = slice(chroma_symbol_count, chroma_symbol_count * 2)

    def chroma_component_words(symbol_slice) -> HuffmanCodewords:
        return HuffmanCodewords(
            codes=chroma_words.codes[symbol_slice],
            lengths=chroma_words.lengths[symbol_slice],
            bits=chroma_words.bits[symbol_slice],
            valid=chroma_words.valid[symbol_slice],
            max_bits=chroma_words.max_bits,
        )

    cb_words = chroma_component_words(cb_slice)
    cr_words = chroma_component_words(cr_slice)
    if not (
        y_words.max_bits == cb_words.max_bits == cr_words.max_bits
    ):
        raise RuntimeError("JPEG component entropy rows have unequal widths")

    symbols_per_block = JPEG_AC_CAPACITY + 1
    row_width = y_words.max_bits

    def block_field(field: AbstractTensor, final_width: int = 0):
        shape = (block_count, symbols_per_block)
        if final_width:
            shape = shape + (final_width,)
        return field.reshape(shape)

    lengths = AbstractTensor.stack(
        (
            block_field(y_words.lengths),
            block_field(cb_words.lengths),
            block_field(cr_words.lengths),
        ),
        dim=1,
    ).flatten()
    bits = AbstractTensor.stack(
        (
            block_field(y_words.bits, row_width),
            block_field(cb_words.bits, row_width),
            block_field(cr_words.bits, row_width),
        ),
        dim=1,
    ).reshape(-1, row_width)
    valid = AbstractTensor.stack(
        (
            block_field(y_words.valid, row_width),
            block_field(cb_words.valid, row_width),
            block_field(cr_words.valid, row_width),
        ),
        dim=1,
    ).reshape(-1, row_width)
    interleaved = HuffmanCodewords(
        codes=AbstractTensor.zeros(lengths.shape, cls=type(lengths)),
        lengths=lengths,
        bits=bits,
        valid=valid,
        max_bits=row_width,
    )
    return compact_codewords(interleaved)


__all__ = [
    "JPEG_AC_CAPACITY",
    "JPEG_ZRL",
    "encode_baseline_color_component_scan",
    "encode_baseline_color_scan",
    "encode_baseline_luma_scan",
    "jpeg_ac_entropy_symbols",
]
