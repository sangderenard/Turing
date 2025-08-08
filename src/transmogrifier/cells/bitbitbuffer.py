# Compatibility shim: the BitBitBuffer module has moved to `transmogrifier.bitbitbuffer`.
# This file re-exports the split classes to preserve older import paths.

from ..bitbitbuffer import BitBitBuffer
from ..bitbitbuffer.helpers import (
    BitBitItem,
    BitBitSlice,
    BitBitIndex,
    BitBitIndexer,
    PIDBuffer,
    CellProposal,
    _RawSpan,
)

__all__ = [
    "BitBitBuffer",
    "BitBitItem",
    "BitBitSlice",
    "BitBitIndex",
    "BitBitIndexer",
    "PIDBuffer",
    "CellProposal",
    "_RawSpan",
]