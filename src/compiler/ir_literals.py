"""Literal decoders shared by repository-SSA target emitters."""
from __future__ import annotations

import struct
from typing import Any


def decode_llvm_scalar_literal(value: str) -> Any:
    """Decode the scalar spelling retained by the LLVM-to-SSA importer."""

    dtype, separator, token = str(value).strip().partition(" ")
    if not separator:
        raise ValueError(f"malformed LLVM literal {value!r}")
    token = token.strip()
    if dtype.startswith("i"):
        if token in {"true", "false"}:
            return token == "true"
        return int(token, 0)
    if dtype in {"half", "float", "double"}:
        if token.casefold().startswith("0x"):
            bits = token[2:]
            if len(bits) == 8:
                return struct.unpack(">f", bytes.fromhex(bits))[0]
            if len(bits) == 16:
                return struct.unpack(">d", bytes.fromhex(bits))[0]
            raise ValueError(
                f"unsupported LLVM floating bit pattern {token!r}"
            )
        return float(token)
    raise ValueError(
        f"unsupported LLVM literal type {dtype!r} in {value!r}"
    )


__all__ = ["decode_llvm_scalar_literal"]
