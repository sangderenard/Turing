"""JPEG DHT table adaptation without leaving AbstractTensor."""

from __future__ import annotations

from ...abstraction import AbstractTensor
from ...autograd import autograd
from ..huffman import CanonicalHuffmanTable


# ITU-T T.81 Annex K default luminance DC table. The sixteen entries are the
# number of symbols having code lengths 1..16.
JPEG_DC_LUMINANCE_COUNTS = (
    0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
)
JPEG_DC_LUMINANCE_SYMBOLS = tuple(range(12))

JPEG_AC_LUMINANCE_COUNTS = (
    0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D,
)

JPEG_AC_LUMINANCE_SYMBOLS = (
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
)

JPEG_DC_CHROMINANCE_COUNTS = (
    0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
)
JPEG_DC_CHROMINANCE_SYMBOLS = tuple(range(12))

JPEG_AC_CHROMINANCE_COUNTS = (
    0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77,
)

JPEG_AC_CHROMINANCE_SYMBOLS = (
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34,
    0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
    0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
    0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
)


def jpeg_huffman_table(
    length_counts: tuple[int, ...],
    ordered_symbols: AbstractTensor,
    *,
    alphabet_size: int = 256,
) -> CanonicalHuffmanTable:
    """Convert JPEG's ordered DHT count/symbol form into a canonical table."""
    if len(length_counts) != 16:
        raise ValueError("JPEG DHT requires sixteen code-length counts")
    if any(count < 0 for count in length_counts):
        raise ValueError("JPEG DHT counts cannot be negative")
    if sum(length_counts) != ordered_symbols.shape[0]:
        raise ValueError("JPEG DHT counts do not match the symbol list")
    if alphabet_size < 1:
        raise ValueError("alphabet_size must be positive")
    parts = [
        AbstractTensor.full(
            (count,), width, cls=type(ordered_symbols)
        )
        for width, count in enumerate(length_counts, start=1)
        if count
    ]
    ordered_lengths = AbstractTensor.cat(parts, dim=0)
    ordered = CanonicalHuffmanTable.from_code_lengths(
        ordered_lengths,
        max_bits=16,
        validate=False,
    )
    # JPEG transmits symbols in canonical-code order, but its symbol space is
    # dense and bounded.  Materialize that order once into dense lookup tensors
    # so encoding a frame can gather by symbol value.  Keeping
    # ``ordered_symbols`` on every table would turn each lookup into an
    # O(samples * alphabet) comparison matrix, which is especially punishing
    # for accelerator backends and obscures the direct-index operation already
    # present in the general CanonicalHuffmanTable.
    dense_codes = AbstractTensor.zeros(
        (alphabet_size,), cls=type(ordered_symbols)
    )
    dense_lengths = AbstractTensor.zeros(
        (alphabet_size,), cls=type(ordered_symbols)
    )
    indices = ordered_symbols.to_dtype("int64")
    with autograd.no_grad():
        dense_codes = AbstractTensor.scatter(
            dense_codes, indices, ordered.codes, dim=0
        )
        dense_lengths = AbstractTensor.scatter(
            dense_lengths, indices, ordered.lengths, dim=0
        )
    return CanonicalHuffmanTable(
        codes=dense_codes,
        lengths=dense_lengths,
        max_bits=16,
    )


def jpeg_standard_dc_luminance(like: AbstractTensor) -> CanonicalHuffmanTable:
    """Return the standard JPEG luminance-DC table on ``like``'s backend."""
    symbols = like.ensure_tensor(JPEG_DC_LUMINANCE_SYMBOLS)
    return jpeg_huffman_table(
        JPEG_DC_LUMINANCE_COUNTS,
        symbols,
        alphabet_size=12,
    )


def jpeg_standard_ac_luminance(like: AbstractTensor) -> CanonicalHuffmanTable:
    """Return the standard JPEG luminance-AC table on ``like``'s backend."""
    symbols = like.ensure_tensor(JPEG_AC_LUMINANCE_SYMBOLS)
    return jpeg_huffman_table(
        JPEG_AC_LUMINANCE_COUNTS,
        symbols,
        alphabet_size=256,
    )


def jpeg_standard_dc_chrominance(
    like: AbstractTensor,
) -> CanonicalHuffmanTable:
    """Return the standard JPEG chrominance-DC table."""
    symbols = like.ensure_tensor(JPEG_DC_CHROMINANCE_SYMBOLS)
    return jpeg_huffman_table(
        JPEG_DC_CHROMINANCE_COUNTS,
        symbols,
        alphabet_size=12,
    )


def jpeg_standard_ac_chrominance(
    like: AbstractTensor,
) -> CanonicalHuffmanTable:
    """Return the standard JPEG chrominance-AC table."""
    symbols = like.ensure_tensor(JPEG_AC_CHROMINANCE_SYMBOLS)
    return jpeg_huffman_table(
        JPEG_AC_CHROMINANCE_COUNTS,
        symbols,
        alphabet_size=256,
    )


__all__ = [
    "JPEG_AC_CHROMINANCE_COUNTS",
    "JPEG_AC_CHROMINANCE_SYMBOLS",
    "JPEG_AC_LUMINANCE_COUNTS",
    "JPEG_AC_LUMINANCE_SYMBOLS",
    "JPEG_DC_CHROMINANCE_COUNTS",
    "JPEG_DC_CHROMINANCE_SYMBOLS",
    "JPEG_DC_LUMINANCE_COUNTS",
    "JPEG_DC_LUMINANCE_SYMBOLS",
    "jpeg_huffman_table",
    "jpeg_standard_ac_chrominance",
    "jpeg_standard_ac_luminance",
    "jpeg_standard_dc_chrominance",
    "jpeg_standard_dc_luminance",
]
