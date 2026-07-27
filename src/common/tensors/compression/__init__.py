"""Backend-agnostic compression primitives composed from AbstractTensor ops."""

from .huffman import (
    CanonicalHuffmanTable,
    HuffmanCodewords,
    canonical_codes_from_lengths,
    huffman_code_lengths,
    length_limited_huffman_code_lengths,
    symbol_frequencies,
)
from .bitstream import (
    DecodedHuffmanStream,
    PackedBitstream,
    UnpackedBitstream,
    compact_codewords,
    decode_huffman_octets,
    decode_with_provenance,
    unpack_octets,
)
from .block_transform import (
    block_view_2d,
    dct_2d_blocks,
    orthonormal_dct_basis,
)
from .coefficient_events import (
    BlockCoefficientEvents,
    SignedMagnitudeFields,
    collect_block_coefficient_events,
    decode_signed_magnitudes,
    encode_signed_magnitudes,
    reconstruct_block_coefficients,
)
from .entropy_symbols import (
    BlockEntropySymbols,
    EntropySymbolSequence,
    ac_entropy_tokens,
    coefficient_events_to_entropy_symbols,
    entropy_symbols_to_coefficient_events,
)
from .entropy_scan import (
    decode_entropy_octets,
    encode_entropy_sequence,
    interleave_codewords_and_payloads,
)
from .pcm import PCMFormat, RationalAudioScheduler, encode_pcm

__all__ = [
    "CanonicalHuffmanTable",
    "BlockCoefficientEvents",
    "BlockEntropySymbols",
    "DecodedHuffmanStream",
    "HuffmanCodewords",
    "EntropySymbolSequence",
    "PackedBitstream",
    "PCMFormat",
    "RationalAudioScheduler",
    "SignedMagnitudeFields",
    "UnpackedBitstream",
    "canonical_codes_from_lengths",
    "ac_entropy_tokens",
    "huffman_code_lengths",
    "length_limited_huffman_code_lengths",
    "symbol_frequencies",
    "compact_codewords",
    "decode_huffman_octets",
    "decode_entropy_octets",
    "decode_with_provenance",
    "unpack_octets",
    "block_view_2d",
    "dct_2d_blocks",
    "collect_block_coefficient_events",
    "coefficient_events_to_entropy_symbols",
    "decode_signed_magnitudes",
    "encode_signed_magnitudes",
    "encode_entropy_sequence",
    "encode_pcm",
    "entropy_symbols_to_coefficient_events",
    "interleave_codewords_and_payloads",
    "orthonormal_dct_basis",
    "reconstruct_block_coefficients",
]
