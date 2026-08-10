"""The basic string operators of the common representation.

SSA and the dual IR shell (``dual_ir_shell.DualIRShell``: numeric ``FusedProgram``
+ control ``ControlProgram`` + ``map_ir``) are standard headers the whole pipeline
shares. Strings deserve the same: one backend-neutral vocabulary of basic string
operators, so every backend lowers the SAME small set instead of each re-deriving
a `.split` here and a `==` there. This module is that header -- it defines the two
string value models and the operator set over them; it is not a backend.

Two value models, bridged by ``string_hash``
--------------------------------------------
* **Token** -- a word's 64-bit content identity (``string_table.string_token``,
  FNV-1a). Cheap, comparable, usable as a dict key; carries NO bytes. This is
  what a constant, a dict key, or an ``==`` interns to, and what a runtime name
  extracted from bytes hashes to, so the two collapse to one identity. Held in
  the numeric working type as reinterpreted bits.

* **String ref** -- a heap value carrying the actual bytes: a block
  ``[length:i32][bytes...]`` whose i32 base offset is the value. Needed whenever
  the bytes themselves are operated on (slice, concatenate, index a character,
  split and keep a part). Materialised in the bump heap alongside the container
  maps (``wasm_container``); a constant string ref lives in static data.

``string_hash(ref) -> token`` bridges them: any bytes reduce to their identity,
so an operator that produces a ref (a split part, a slice) can still be compared
or used as a key. The reverse (``token -> bytes``) is only for display and goes
through the ``StringTable`` recorded at compile time, never at run time.

Operator set
------------
Identity / bridge:
  ``string_token(text=const) -> token``                     (constant word)
  ``string_hash(ref) -> token``                             (bytes -> identity)
  ``string_compare(a, b) {op: equal|not_equal} -> bool``    (identity test)

Bytes (produce/consume a string ref):
  ``string_const(text=const) -> ref``                       (constant -> heap/static)
  ``string_length(ref) -> i32``
  ``string_char(ref, i) -> byte``                           (ref[i])
  ``string_find(ref, delim) {from: i32=0} -> i32``          (index of delim, or length)
  ``string_slice(ref, start, stop) -> ref``                 (ref[start:stop])
  ``string_concat(a, b) -> ref``                            (a + b)
  ``string_split_part(ref, delim, part) -> ref``            (ref.split(delim, 1)[part])

Every operator is pure over its inputs and deterministic; the runtime string
values live in the heap the container ABI already owns, so string and dict state
share one arena. A backend supplies a lowering per operator name; the recognition
of source idioms into these operators is a central fold (``ir_string_interning``
for tokens/compare, ``ir_byte_idioms`` for the null-terminated hash, extended
here for the general split), never per-backend.
"""
from __future__ import annotations

# Identity / bridge operators.
STRING_TOKEN = "string_token"        # const word  -> 64-bit content token
STRING_HASH = "string_hash"          # ref         -> token (hash the bytes)
STRING_COMPARE_ATTR = "string_compare"  # tag on equal/not_equal: compare identities

# Bytes operators (a "ref" is an i32 heap offset of [length:i32][bytes...]).
STRING_CONST = "string_const"        # const word  -> ref (static data)
STRING_LENGTH = "string_length"      # ref         -> i32 length
STRING_CHAR = "string_char"          # ref, i      -> byte
STRING_FIND = "string_find"          # ref, delim  -> i32 index of delim, or length
STRING_SLICE = "string_slice"        # ref, a, b   -> ref (ref[a:b])
STRING_CONCAT = "string_concat"      # a, b        -> ref (a + b)
STRING_SPLIT_PART = "string_split_part"  # ref, delim, part -> ref (split(delim,1)[part])

#: The complete basic set, for a backend to check it lowers every operator, and
#: for a fold to know which names are string operators.
STRING_OPERATORS = frozenset({
    STRING_TOKEN, STRING_HASH, STRING_CONST, STRING_LENGTH, STRING_CHAR,
    STRING_FIND, STRING_SLICE, STRING_CONCAT, STRING_SPLIT_PART,
})

# --- String-ref heap block layout (shared with wasm_container's bump heap) -----
#: A string ref points at this block; the length precedes the bytes so a ref
#: alone is enough to read the whole string.
STRING_REF_LENGTH_OFFSET = 0     # i32 length at the block's start
STRING_REF_BYTES_OFFSET = 4      # bytes follow immediately


def string_ref_block_bytes(length: int) -> int:
    """Total heap bytes for a string ref holding ``length`` bytes (4-byte header
    plus the bytes, rounded up to 4 for the next block's alignment)."""
    return ((STRING_REF_BYTES_OFFSET + int(length)) + 3) & ~3
