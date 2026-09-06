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
from .ir_container_ops import FNV64_OFFSET, FNV64_PRIME

# Must match the central ir_container_ops.fnv1a_64 so a runtime-hashed name and a
# compile-time-hashed string constant collide iff the bytes are equal.
_FNV64_OFFSET_SIGNED = FNV64_OFFSET - 2 ** 64  # fold to signed i64
_FNV64_PRIME = FNV64_PRIME

_I32_ADD = 0x6A
_I32_SUB = 0x6B
_I32_GE_S = 0x4E
_I32_EQ = 0x46
_I32_NE = 0x47
_I32_LOAD8_U = 0x2D
_I32_LOAD = 0x28
_I64_XOR = 0x85
_I64_MUL = 0x7E
_I64_SHR_U = 0x88
_I64_SHL = 0x86
_I64_OR = 0x84
_I64_EXTEND_I32_U = 0xAD
_I32_WRAP_I64 = 0xA7


def emit_string_unpack(builder: CodeBuilder, *, view_local: int,
                       ptr_local: int, length_local: int) -> None:
    """From a fat-pointer view (i64: byte_ptr<<32 | length) extract the byte
    pointer and length into i32 locals. No memory touched -- a view is a value."""
    builder.local_get(view_local).i64_const(32).raw(_I64_SHR_U).raw(_I32_WRAP_I64)
    builder.local_set(ptr_local)
    builder.local_get(view_local).raw(_I32_WRAP_I64)  # low 32 bits = length
    builder.local_set(length_local)


def emit_string_length(builder: CodeBuilder, *, view_local: int, result_local: int) -> None:
    """``string_length``: result(i32) = the length half of the fat pointer."""
    builder.local_get(view_local).raw(_I32_WRAP_I64).local_set(result_local)


def emit_string_slice(builder: CodeBuilder, *, ptr_local: int, start_local: int,
                      stop_local: int, result_local: int) -> None:
    """``string_slice``: result(i64 view) = the bytes [start, stop) at ptr+start,
    as a fat pointer. A pure repack -- nothing is copied (JS's sliced string)."""
    builder.local_get(ptr_local).local_get(start_local).raw(_I32_ADD)   # newptr
    builder.raw(_I64_EXTEND_I32_U).i64_const(32).raw(_I64_SHL)
    builder.local_get(stop_local).local_get(start_local).raw(_I32_SUB)  # newlen
    builder.raw(_I64_EXTEND_I32_U).raw(_I64_OR)
    builder.local_set(result_local)


def emit_string_find(
    builder: CodeBuilder, *, ptr_local: int, length_local: int, delim_local: int,
    from_local: int, result_local: int, index_local: int, byte_local: int,
) -> None:
    """``string_find``: result(i32) = index of the first ``delim`` byte at ``ptr``
    at/after ``from``, or ``length`` if absent. All locals i32; ``delim`` a byte."""
    builder.local_get(from_local).local_set(index_local)
    builder.block()          # done
    builder.loop()           # scan
    builder.local_get(index_local).local_get(length_local).raw(_I32_GE_S)
    builder.if_()
    builder.local_get(length_local).local_set(result_local)
    builder.br(2)
    builder.end()
    builder.local_get(ptr_local).local_get(index_local).raw(_I32_ADD)
    builder.raw(_I32_LOAD8_U, 0x00, 0x00)
    builder.local_set(byte_local)
    builder.local_get(byte_local).local_get(delim_local).raw(_I32_EQ)
    builder.if_()
    builder.local_get(index_local).local_set(result_local)
    builder.br(2)
    builder.end()
    builder.local_get(index_local).i32_const(1).raw(_I32_ADD).local_set(index_local)
    builder.br(0)
    builder.end()            # loop
    builder.end()            # block


def emit_string_hash_range(
    builder: CodeBuilder, *, ptr_local: int, start_local: int, end_local: int,
    result_local: int, index_local: int, byte_local: int,
) -> None:
    """``string_hash`` over a range: result(i64 token) = FNV-1a of the bytes
    ``[start, end)`` at ``ptr`` -- the token a slice/part interns to, so a view
    can be compared or keyed without ever materialising."""
    builder.i64_const(_FNV64_OFFSET_SIGNED).local_set(result_local)
    builder.local_get(start_local).local_set(index_local)
    builder.block()          # done
    builder.loop()           # fold
    builder.local_get(index_local).local_get(end_local).raw(_I32_GE_S)
    builder.br_if(1)
    builder.local_get(ptr_local).local_get(index_local).raw(_I32_ADD)
    builder.raw(_I32_LOAD8_U, 0x00, 0x00)
    builder.local_set(byte_local)
    builder.local_get(result_local)
    builder.local_get(byte_local).raw(_I64_EXTEND_I32_U)
    builder.raw(_I64_XOR)
    builder.i64_const(_FNV64_PRIME).raw(_I64_MUL)
    builder.local_set(result_local)
    builder.local_get(index_local).i32_const(1).raw(_I32_ADD).local_set(index_local)
    builder.br(0)
    builder.end()            # loop
    builder.end()            # block


def emit_string_split_part_hash(
    builder: CodeBuilder, *, view_local: int, delim_local: int, part: int,
    result_local: int, ptr_local: int, length_local: int, pos_local: int,
    start_local: int, end_local: int, index_local: int, byte_local: int,
) -> None:
    """``string_hash(string_split_part(view, delim, part))`` for ``part`` in
    {0,1}: the token of the prefix before the first ``delim`` (part 0) or the
    suffix after it (part 1). The part is a free sub-range view of the same
    bytes, hashed in place -- no copy. ``result_local`` is i64; the rest i32."""
    if part not in (0, 1):
        raise ValueError("only split(delim, 1)[0] / [1] are lowered")
    emit_string_unpack(builder, view_local=view_local, ptr_local=ptr_local,
                       length_local=length_local)
    builder.i32_const(0).local_set(start_local)  # from = 0 for find
    emit_string_find(builder, ptr_local=ptr_local, length_local=length_local,
                     delim_local=delim_local, from_local=start_local,
                     result_local=pos_local, index_local=index_local,
                     byte_local=byte_local)
    if part == 0:
        builder.i32_const(0).local_set(start_local)
        builder.local_get(pos_local).local_set(end_local)
    else:
        # start = (pos >= length) ? length : pos + 1 ; end = length.
        builder.local_get(length_local)
        builder.local_get(pos_local).i32_const(1).raw(_I32_ADD)
        builder.local_get(pos_local).local_get(length_local).raw(_I32_GE_S)
        builder.select()
        builder.local_set(start_local)
        builder.local_get(length_local).local_set(end_local)
    emit_string_hash_range(builder, ptr_local=ptr_local, start_local=start_local,
                           end_local=end_local, result_local=result_local,
                           index_local=index_local, byte_local=byte_local)


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
