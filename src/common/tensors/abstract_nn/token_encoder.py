"""Lossless deterministic token encoder for compiler-owned identities."""
from __future__ import annotations

import json
from typing import Any, Mapping


def encode_identity_tokens(identity: Mapping[str, Any]) -> int:
    """Encode canonical identity tokens into one reversible integer.

    Canonical JSON supplies the token stream; UTF-8 bytes are encoded as
    base-257 digits (1..256) after a leading sentinel.  This is injective,
    deterministic, and contains no vocabulary cache or hash collision path.
    """

    tokens = json.dumps(
        dict(identity), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    encoded = 1
    for token in tokens:
        encoded = encoded * 257 + token + 1
    return encoded


def decode_identity_tokens(encoded: int) -> dict[str, Any]:
    """Invert :func:`encode_identity_tokens`."""

    digits: list[int] = []
    current = int(encoded)
    while current > 1:
        current, digit = divmod(current, 257)
        if digit == 0:
            raise ValueError("not a token-encoded identity")
        digits.append(digit - 1)
    if current != 1:
        raise ValueError("not a token-encoded identity")
    return json.loads(bytes(reversed(digits)).decode("utf-8"))


__all__ = ["decode_identity_tokens", "encode_identity_tokens"]
