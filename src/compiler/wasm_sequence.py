"""Heap byte-sequence primitives for WebAssembly kernels.

The decoder does real byte-string work on the subject bytes. The recurring idiom
is null-terminated name extraction -- ``data[offset:offset+8].split(b"\\x00", 1)[0]``
(binary_ingestion.py: PE section names) -- whose result is a name used downstream
as a dict key or label. Rather than materialise a variable-length byte list, the
whole idiom collapses to the same i64 identity a string key gets: an FNV-1a hash
of the bytes before the first delimiter (bounded by the slice width). Two equal
names hash equal, so dict lookups keyed by names stay consistent with the string
constant path in fused_program_wasm_backend.

``emit_hash_delimited_prefix`` computes that hash inline over ``[start, start+
maxlen)`` of a byte buffer, stopping at the first delimiter byte -- the numeric
kernel counterpart of ``hash(data[start:start+maxlen].split(delim, 1)[0])``.
"""
from __future__ import annotations

from .wasm_binary import CodeBuilder

# Must match fused_program_wasm_backend._fnv1a_64 so a runtime-hashed name and a
# compile-time-hashed string constant collide iff the bytes are equal.
_FNV64_OFFSET_SIGNED = 0xCBF29CE484222325 - 2 ** 64  # fold to signed i64
_FNV64_PRIME = 0x100000001B3

_I32_ADD = 0x6A
_I32_GE_S = 0x4E
_I32_EQ = 0x46
_I32_LOAD8_U = 0x2D
_I64_XOR = 0x85
_I64_MUL = 0x7E


def emit_hash_delimited_prefix(
    builder: CodeBuilder,
    *,
    buf_addr_local: int,
    start_local: int,
    maxlen_local: int,
    delim_local: int,
    result_local: int,
    index_local: int,
    byte_local: int,
) -> None:
    """result = FNV-1a( bytes[start : start + first(delim) or maxlen] ).

    ``buf_addr_local``/``start_local``/``maxlen_local``/``delim_local`` are i32
    locals (a byte buffer base, a byte start offset, a max length, and the
    delimiter byte value 0..255). ``result_local`` is i64; ``index_local`` and
    ``byte_local`` are i32 scratch. Reads one byte at a time, folding until the
    delimiter or the max length -- the exact null-terminated-prefix semantics.
    """

    builder.i64_const(_FNV64_OFFSET_SIGNED).local_set(result_local)
    builder.i32_const(0).local_set(index_local)
    builder.block()          # depth 2: done
    builder.loop()           # depth 1: scan
    # if index >= maxlen: stop.
    builder.local_get(index_local).local_get(maxlen_local).raw(_I32_GE_S)
    builder.br_if(1)
    # byte = load8_u(buf + start + index)
    builder.local_get(buf_addr_local)
    builder.local_get(start_local).raw(_I32_ADD)
    builder.local_get(index_local).raw(_I32_ADD)
    builder.raw(_I32_LOAD8_U, 0x00, 0x00)  # i32.load8_u align=0 offset=0
    builder.local_set(byte_local)
    # if byte == delim: stop.
    builder.local_get(byte_local).local_get(delim_local).raw(_I32_EQ)
    builder.br_if(1)
    # result = (result ^ byte) * FNV_PRIME   (byte zero-extended to i64)
    builder.local_get(result_local)
    builder.local_get(byte_local).raw(0xAD)  # i64.extend_i32_u
    builder.raw(_I64_XOR)
    builder.i64_const(_FNV64_PRIME).raw(_I64_MUL)
    builder.local_set(result_local)
    # index += 1; continue.
    builder.local_get(index_local).i32_const(1).raw(_I32_ADD).local_set(index_local)
    builder.br(0)
    builder.end()            # loop
    builder.end()            # block
