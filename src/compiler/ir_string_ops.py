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

* **String view (a fat pointer)** -- the actual bytes referenced without owning
  them: an i64 packing ``(byte_ptr:i32 in the high 32, length:i32 in the low
  32)``, held in the working type as reinterpreted bits like a token. The bytes
  live wherever they already are -- inside ``subject``, inside another string,
  or a heap block -- and the view just points at a range of them. This is the
  research answer (JavaScript's sliced strings, Perl's COW): read-only string
  work should COPY NOTHING. ``slice`` and ``split-part`` are pure repacks of a
  sub-range -- O(1), no allocation. Only an operation that must produce NEW
  contiguous bytes (``concat``, an in-place mutation) MATERIALISES: it allocates
  bytes in the same bump heap the container maps use and returns a view of them.
  The default is a view; materialisation is the forced case, and mutation takes
  an explicit in-place flag (the string analogue of the container store's
  in-place aliasing).

``string_hash(view) -> token`` bridges them: any range of bytes reduces to its
identity, so an operator that produces a view (a split part, a slice) can still
be compared or used as a key -- and since decoder strings are almost always
extracted-then-hashed, the view is consumed without ever materialising. The
reverse (``token -> bytes``) is only for display and goes through the
``StringTable`` recorded at compile time, never at run time.

Operator set
------------
Identity / bridge:
  ``string_token(text=const) -> token``                     (constant word)
  ``string_hash(ref) -> token``                             (bytes -> identity)
  ``string_compare(a, b) {op: equal|not_equal} -> bool``    (identity test)

Bytes (a "view" is the i64 fat pointer above; only ``concat``/materialise copy):
  ``string_const(text=const) -> view``                      (constant -> static bytes)
  ``string_length(view) -> i32``                            (the low half)
  ``string_char(view, i) -> byte``                          (view[i])
  ``string_find(view, delim) {from: i32=0} -> i32``         (index of delim, or length)
  ``string_slice(view, start, stop) -> view``               (repack a sub-range; free)
  ``string_concat(a, b) {in_place: bool=false} -> view``    (a + b; MATERIALISES)
  ``string_split_part(view, delim, part) -> view``          (split(delim,1)[part]; free)

Every operator is pure over its inputs and deterministic; a materialised string's
bytes live in the heap the container ABI already owns, so string and dict state
share one arena, and a view points into ``subject`` or that heap without copying.
A backend supplies a lowering per operator name -- the universal path is native
(the WASM backend may additionally choose to defer to its JS host's strings as a
special case). Recognition of source idioms into these operators is a central
fold (``ir_string_interning`` for tokens/compare, ``ir_byte_idioms`` for the
null-terminated hash, extended for the general split), never per-backend.
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

# Fused split-then-hash: the common terminal shape (a split part is taken and
# immediately compared/keyed), so the view is never even named. Inputs [view];
# attrs {delim: byte, part: 0|1}. Result is the part's token.
STRING_SPLIT_PART_HASH = "string_split_part_hash"

#: The complete basic set, for a backend to check it lowers every operator, and
#: for a fold to know which names are string operators.
STRING_OPERATORS = frozenset({
    STRING_TOKEN, STRING_HASH, STRING_CONST, STRING_LENGTH, STRING_CHAR,
    STRING_FIND, STRING_SLICE, STRING_CONCAT, STRING_SPLIT_PART,
})

# --- String view fat-pointer packing ------------------------------------------
#: A string value is an i64: byte pointer in the high 32 bits, length in the low
#: 32. The bytes are wherever they already live (subject, another string, or a
#: materialised heap block); the view just points at a range, so slice/split are
#: pure repacks with no allocation. Materialised bytes live in the same bump heap
#: the container maps use (wasm_container), so string and dict state share one arena.


def pack_string_view(byte_ptr: int, length: int) -> int:
    """The i64 fat pointer for a range of bytes."""
    return ((int(byte_ptr) & 0xFFFFFFFF) << 32) | (int(length) & 0xFFFFFFFF)


def unpack_string_view(fat: int) -> tuple[int, int]:
    """(byte_ptr, length) from a fat pointer."""
    return (fat >> 32) & 0xFFFFFFFF, fat & 0xFFFFFFFF
