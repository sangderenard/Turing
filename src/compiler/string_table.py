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

Interning strings does NOT by itself solve lowering: a token is a typed signed
64-bit identity and the structural ABI must retain it as such. What lives here
is only the universal, backend-neutral
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


class StringTokenCollision(ValueError):
    """Two different words reached one content-addressed token.

    The token is a 64-bit hash, so it can collide. A collision is refused,
    never resolved by keeping whichever word arrived first: that silently
    gave the second word the first word's identity and reverse lookup.
    """


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
        # "TOK0": a second word under one token is refused by name. The book
        # refuses it too, but only inside one compile; this table outlives
        # every book (it persists across builds), so it holds the rule itself.
        incumbent = self._entries.get(token)
        if incumbent is not None and incumbent != text:
            raise StringTokenCollision(
                "StringTable token collision: "
                f"token={int(token)} recorded={incumbent!r} proposed={text!r}"
            )
        self._entries[token] = text
        from .concordance_declarations import (
            INGESTION, STRING_TABLE_ENTRY, STRING_TABLE_INGESTION,
        )
        from .identity_concordance import Mode, Novel, current_identity_book

        current_identity_book().post(
            STRING_TABLE_ENTRY, (int(token),), str(text),
            stage=INGESTION,
            provenance=Novel(STRING_TABLE_INGESTION, ()),
            mode=Mode.CONCORD,
        )
        return token

    def get(self, token: int) -> str | None:
        return self._entries.get(int(token))

    @property
    def entries(self) -> Mapping[int, str]:
        return dict(self._entries)

    def publish_entries(self, *, book, stage) -> tuple:
        """Publish the exact loaded/interned vocabulary as ingestion facts."""

        from .concordance_declarations import STRING_TABLE_ENTRY

        cells = []
        for token, text in sorted(self._entries.items()):
            cell = book.latest_ref(STRING_TABLE_ENTRY, (int(token),))
            if cell is None:
                raise ValueError(
                    "StringTable entry lacks its ingestion concordance cell: "
                    f"token={int(token)} text={text!r}"
                )
            cells.append(cell)
        return tuple(cells)

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        for token, text in data.get("strings", {}).items():
            try:
                if int(token) == string_token(str(text)):
                    self.intern(str(text))
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
