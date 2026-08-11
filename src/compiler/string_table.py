"""A universal string/token table for the lowered common representation.

Every word the program mentions -- a section name pulled from bytes at run time,
a dict key like ``'windows.loader.tls_callback_count'``, a compared literal like
``'node-wasm'`` -- lowers to the SAME token: the FNV-1a 64-bit hash of its bytes.
Because the token is content-addressed, a value extracted at run time and a
compile-time constant collapse to one identity with no shared counter to keep in
sync, and every backend (WASM, C/PE, GLSL, Fortran, LLVM) computes the identical
token from the identical bytes.

This module owns that mapping. ``StringTable`` records token -> string so the
lowered program can carry a table for reverse lookup (display, error messages,
re-emitting a name), and persists it like the topology catalogue so the common
representation accrues across builds. The token itself is
``ir_container_ops.fnv1a_64`` -- the one hash the container keys and the
runtime null-terminated-name primitive already fold with -- so strings, dict
keys, and extracted names all live in one namespace.

Interning strings does NOT by itself solve lowering: a token is a 64-bit value,
which the float64 working kernel holds as reinterpreted bits and compares in i64
(see the backend). What lives here is only the universal, backend-neutral
question of *which token a word is* and *what word a token was*.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .ir_container_ops import fnv1a_64


def string_token(text: str) -> int:
    """The universal token for a word: the FNV-1a 64-bit hash of its UTF-8 bytes
    (signed i64). Identical to a runtime-extracted name's token for equal bytes."""
    return fnv1a_64(text)


#: The sentinel for Python ``None`` -- the "absence" singleton. It is a value
#: like any other in the 64-bit working type (a reserved, content-addressed
#: token distinct from real data), so ``x = None``, a ``None`` field, or a
#: returned ``None`` is expressible and comparable rather than an inexpressible
#: literal. Reserved via the same fnv namespace as every other token, so no
#: real word or byte buffer can collide with it by construction of the marker.
NONE_TOKEN = fnv1a_64("\x00turing.sentinel.None\x00")


class StringTable:
    """Persistent token -> string map for the lowered common representation."""

    def __init__(self, root: str | Path | None = None) -> None:
        base = Path(root) if root is not None else Path.cwd() / "repo-cache"
        self.path = base / "string-table" / "strings.json"
        self._entries: dict[int, str] = {}
        self._load()

    def intern(self, text: str) -> int:
        """Return the word's token, recording token -> text for reverse lookup."""
        token = string_token(text)
        self._entries.setdefault(token, text)
        return token

    def get(self, token: int) -> str | None:
        return self._entries.get(int(token))

    @property
    def entries(self) -> Mapping[int, str]:
        return dict(self._entries)

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        for token, text in data.get("strings", {}).items():
            try:
                self._entries[int(token)] = str(text)
            except (TypeError, ValueError):
                continue

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "abi": "turing.string-table.v1",
            "strings": {str(token): text for token, text in sorted(self._entries.items())},
        }
        temporary = self.path.with_suffix(f".json.{__import__('os').getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)
