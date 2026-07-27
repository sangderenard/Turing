"""Self-contained Huffman-symbol and raw-payload entropy scans."""

from __future__ import annotations

from ..abstraction import AbstractTensor
from ..autograd import autograd
from .bitstream import PackedBitstream, compact_codewords
from .entropy_symbols import EntropySymbolSequence
from .huffman import CanonicalHuffmanTable, HuffmanCodewords


def _truth(value: AbstractTensor) -> bool:
    return bool(value.item())


def interleave_codewords_and_payloads(
    codewords: HuffmanCodewords,
    payloads: AbstractTensor,
    payload_lengths: AbstractTensor,
    *,
    max_payload_bits: int | None = None,
    validate: bool = True,
) -> HuffmanCodewords:
    """Place each raw payload immediately after its Huffman codeword."""
    if not isinstance(payloads, AbstractTensor):
        raise TypeError("payloads must be an AbstractTensor")
    if not isinstance(payload_lengths, AbstractTensor):
        raise TypeError("payload_lengths must be an AbstractTensor")
    if payloads.shape != codewords.lengths.shape:
        raise ValueError("payloads must align with Huffman symbols")
    if payload_lengths.shape != codewords.lengths.shape:
        raise ValueError("payload lengths must align with Huffman symbols")
    observed_payload_bits = (
        int(payload_lengths.max().item()) if validate else None
    )
    if validate:
        if not _truth(((payloads % 1) == 0).all()):
            raise ValueError("entropy payloads must be integers")
        if not _truth(((payload_lengths % 1) == 0).all()):
            raise ValueError("entropy payload lengths must be integers")
        if not _truth((payload_lengths >= 0).all()):
            raise ValueError("entropy payload lengths cannot be negative")
        payload_present = payload_lengths > 0
        payload_fits = (
            (payloads >= 0)
            & (
                (payloads < (2 ** payload_lengths.maximum(1)))
                | ~payload_present
            )
            & ((payloads == 0) | payload_present)
        )
        if not _truth(payload_fits.all()):
            raise ValueError("payload does not fit its declared bit length")
    payload_limit = (
        observed_payload_bits
        if max_payload_bits is None
        else max_payload_bits
    )
    if validate and payload_limit < observed_payload_bits:
        raise ValueError("max_payload_bits is smaller than a payload")
    if payload_limit < 0:
        raise ValueError("max_payload_bits cannot be negative")
    if payload_limit == 0:
        return codewords

    lengths = codewords.lengths.flatten()
    payload_values = payloads.flatten()
    payload_widths = payload_lengths.flatten()
    symbol_count = lengths.shape[0]
    row_width = codewords.max_bits + payload_limit
    row_offsets = (
        AbstractTensor.arange(symbol_count, cls=type(lengths))
        * row_width
    )
    capacity = symbol_count * row_width
    scratch = capacity
    target = AbstractTensor.zeros(
        (capacity + 1,), cls=type(codewords.bits)
    )

    code_positions = AbstractTensor.arange(
        codewords.max_bits, cls=type(lengths)
    )
    code_valid = codewords.valid.reshape(
        symbol_count, codewords.max_bits
    ).to_dtype("int64")
    code_destinations = (
        row_offsets.unsqueeze(1) + code_positions.unsqueeze(0)
    )
    safe_code_destinations = (
        code_destinations * code_valid
        + scratch * (1 - code_valid)
    ).to_dtype("int64")
    code_bits = codewords.bits.reshape(
        symbol_count, codewords.max_bits
    ) * code_valid
    with autograd.no_grad():
        target = AbstractTensor.scatter(
            target,
            safe_code_destinations.flatten(),
            code_bits.flatten(),
            dim=0,
        )

    payload_positions = AbstractTensor.arange(
        payload_limit, cls=type(payload_widths)
    )
    payload_valid = (
        payload_widths.unsqueeze(1) - payload_positions.unsqueeze(0)
    ) > 0
    payload_exponents = (
        payload_widths.unsqueeze(1)
        - 1
        - payload_positions.unsqueeze(0)
    ) * payload_valid.to_dtype("int64")
    payload_bits = (
        (payload_values.unsqueeze(1) // (2 ** payload_exponents)) % 2
    ) * payload_valid.to_dtype("int64")
    payload_destinations = (
        row_offsets.unsqueeze(1)
        + lengths.unsqueeze(1)
        + payload_positions.unsqueeze(0)
    )
    numeric_payload_valid = payload_valid.to_dtype("int64")
    safe_payload_destinations = (
        payload_destinations * numeric_payload_valid
        + scratch * (1 - numeric_payload_valid)
    ).to_dtype("int64")
    with autograd.no_grad():
        target = AbstractTensor.scatter(
            target,
            safe_payload_destinations.flatten(),
            payload_bits.flatten(),
            dim=0,
        )

    total_lengths = lengths + payload_widths
    positions = AbstractTensor.arange(row_width, cls=type(lengths))
    combined_valid = (
        total_lengths.unsqueeze(1) - positions.unsqueeze(0)
    ) > 0
    return HuffmanCodewords(
        codes=codewords.codes.flatten(),
        lengths=total_lengths,
        bits=target[:capacity].reshape(symbol_count, row_width),
        valid=combined_valid,
        max_bits=row_width,
    )


def encode_entropy_sequence(
    table: CanonicalHuffmanTable,
    sequence: EntropySymbolSequence,
    *,
    max_payload_bits: int | None = None,
) -> PackedBitstream:
    """Encode valid symbols and their payloads into one continuous bitstream."""
    compact = sequence.compact()
    count = int(compact.count.item())
    symbols = compact.symbols[:count]
    payloads = compact.payloads[:count]
    payload_lengths = compact.payload_lengths[:count]
    codewords = table.encode_codewords(symbols)
    combined = interleave_codewords_and_payloads(
        codewords,
        payloads,
        payload_lengths,
        max_payload_bits=max_payload_bits,
    )
    return compact_codewords(combined)


def decode_entropy_octets(
    table: CanonicalHuffmanTable,
    payload_lengths: AbstractTensor,
    octets: AbstractTensor,
    valid_bits: AbstractTensor,
) -> EntropySymbolSequence:
    """Decode interleaved Huffman symbols and raw payloads without provenance."""
    if payload_lengths.shape != table.lengths.shape:
        raise ValueError("payload length table must align with Huffman table")
    if not _truth(((payload_lengths % 1) == 0).all()):
        raise ValueError("payload lengths must be integers")
    if not _truth((payload_lengths >= 0).all()):
        raise ValueError("payload lengths cannot be negative")
    if octets.ndims() != 1:
        raise ValueError("octets must be one-dimensional")
    if not _truth(
        (((octets % 1) == 0) & (octets >= 0) & (octets <= 255)).all()
    ):
        raise ValueError("octets must contain integer byte values")

    bit_count = int(valid_bits.item())
    if bit_count < 0 or bit_count > octets.shape[0] * 8:
        raise ValueError("valid bit count exceeds the octet payload")
    if bit_count == 0:
        empty = AbstractTensor.zeros((0,), cls=type(octets))
        return EntropySymbolSequence(
            symbols=empty,
            payloads=empty,
            payload_lengths=empty,
            valid=empty > 0,
        )

    bit_positions = AbstractTensor.arange(8, cls=type(octets))
    bits = (
        (
            octets.unsqueeze(1)
            // (2 ** (7 - bit_positions)).unsqueeze(0)
        )
        % 2
    ).flatten()[:bit_count]

    zero = valid_bits * 0
    code = zero
    code_length = zero
    payload_remaining = zero
    payload_value = zero
    current_symbol = zero
    current_payload_length = zero
    dead = zero
    emitted_symbols = []
    emitted_payloads = []
    emitted_lengths = []
    emitted_masks = []
    present = table.lengths > 0
    alphabet = table.alphabet

    for position in range(bit_count):
        bit = bits[position]
        in_payload = (payload_remaining > 0).to_dtype("int64")
        in_code = 1 - in_payload

        candidate_code = code * 2 + bit
        candidate_length = code_length + 1
        matches = (
            (
                ((candidate_code - table.codes) == 0).to_dtype("int64")
                * ((candidate_length - table.lengths) == 0).to_dtype("int64")
                * present.to_dtype("int64")
                * in_code
            )
            > 0
        )
        match_count = matches.to_dtype("int64").sum()
        hit = (match_count > 0).to_dtype("int64")
        hit_symbol = (
            matches.to_dtype("int64") * alphabet
        ).sum()
        hit_payload_length = (
            matches.to_dtype("int64") * payload_lengths
        ).sum()

        payload_candidate = payload_value * 2 + bit
        payload_finish = (
            in_payload
            * (payload_remaining == 1).to_dtype("int64")
        )
        zero_payload_finish = (
            hit
            * (hit_payload_length == 0).to_dtype("int64")
        )
        emit = (
            (payload_finish + zero_payload_finish) > 0
        ).to_dtype("int64")
        emitted_symbols.append(
            current_symbol * payload_finish
            + hit_symbol * zero_payload_finish
        )
        emitted_payloads.append(payload_candidate * payload_finish)
        emitted_lengths.append(
            current_payload_length * payload_finish
        )
        emitted_masks.append(emit)

        continue_payload = in_payload * (1 - payload_finish)
        start_payload = (
            in_code
            * hit
            * (hit_payload_length > 0).to_dtype("int64")
        )
        continue_code = in_code * (1 - hit)
        payload_remaining = (
            (payload_remaining - 1) * continue_payload
            + hit_payload_length * start_payload
        )
        payload_value = payload_candidate * continue_payload
        current_symbol = (
            current_symbol * continue_payload
            + hit_symbol * start_payload
        )
        current_payload_length = (
            current_payload_length * continue_payload
            + hit_payload_length * start_payload
        )
        code = candidate_code * continue_code
        code_length = candidate_length * continue_code
        prefix_failure = (
            in_code
            * (candidate_length >= table.max_bits).to_dtype("int64")
            * (1 - hit)
        )
        ambiguity = (match_count > 1).to_dtype("int64")
        dead = (
            (dead > 0) | (prefix_failure > 0) | (ambiguity > 0)
        ).to_dtype("int64")

    if _truth(dead > 0):
        raise ValueError("invalid or ambiguous Huffman prefix")
    if _truth((code_length != 0) | (payload_remaining != 0)):
        raise ValueError("entropy payload ends in an incomplete symbol")

    symbols_at_bit = AbstractTensor.stack(
        emitted_symbols, dim=0
    ).reshape(-1)
    payloads_at_bit = AbstractTensor.stack(
        emitted_payloads, dim=0
    ).reshape(-1)
    lengths_at_bit = AbstractTensor.stack(
        emitted_lengths, dim=0
    ).reshape(-1)
    hit_at_bit = AbstractTensor.stack(
        emitted_masks, dim=0
    ).reshape(-1)
    count = hit_at_bit.sum()
    ranks = hit_at_bit.cumsum(dim=0) - 1
    scratch = bit_count
    destinations = (
        ranks * hit_at_bit + scratch * (1 - hit_at_bit)
    ).to_dtype("int64")

    def compact_emissions(field: AbstractTensor) -> AbstractTensor:
        target = AbstractTensor.zeros(
            (bit_count + 1,), cls=type(field)
        )
        with autograd.no_grad():
            target = AbstractTensor.scatter(
                target,
                destinations,
                field * hit_at_bit,
                dim=0,
            )
        return target[:bit_count]

    slots = AbstractTensor.arange(bit_count, cls=type(octets))
    return EntropySymbolSequence(
        symbols=compact_emissions(symbols_at_bit),
        payloads=compact_emissions(payloads_at_bit),
        payload_lengths=compact_emissions(lengths_at_bit),
        valid=(count - slots) > 0,
    )


__all__ = [
    "decode_entropy_octets",
    "encode_entropy_sequence",
    "interleave_codewords_and_payloads",
]
