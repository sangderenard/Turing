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


def emit_string_length(builder: CodeBuilder, *, ref_local: int, result_local: int) -> None:
    """``string_length``: result(i32) = the length header at the ref's start."""
    from .ir_string_ops import STRING_REF_LENGTH_OFFSET
    builder.local_get(ref_local)
    builder.raw(_I32_LOAD, 0x02, STRING_REF_LENGTH_OFFSET)  # i32.load align=2
    builder.local_set(result_local)


def emit_string_find(
    builder: CodeBuilder, *, ref_local: int, delim_local: int, from_local: int,
    result_local: int, length_local: int, index_local: int, byte_local: int,
) -> None:
    """``string_find``: result(i32) = the index of the first ``delim`` byte in the
    string ref at or after ``from``, or the string length if absent. ``delim`` is
    a byte value; the scratch locals are i32."""
    from .ir_string_ops import STRING_REF_BYTES_OFFSET
    emit_string_length(builder, ref_local=ref_local, result_local=length_local)
    builder.local_get(from_local).local_set(index_local)
    builder.block()          # done
    builder.loop()           # scan
    # if index >= length: result = length; break
    builder.local_get(index_local).local_get(length_local).raw(_I32_GE_S)
    builder.if_()
    builder.local_get(length_local).local_set(result_local)
    builder.br(2)
    builder.end()
    # byte = load8_u(ref + bytes_offset + index)
    builder.local_get(ref_local).i32_const(STRING_REF_BYTES_OFFSET).raw(_I32_ADD)
    builder.local_get(index_local).raw(_I32_ADD)
    builder.raw(_I32_LOAD8_U, 0x00, 0x00)
    builder.local_set(byte_local)
    # if byte == delim: result = index; break
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
    builder: CodeBuilder, *, ref_local: int, start_local: int, end_local: int,
    result_local: int, index_local: int, byte_local: int,
) -> None:
    """``string_hash`` over a sub-range: result(i64) = FNV-1a of the string ref's
    bytes ``[start, end)`` -- the same token a constant word interns to, so a
    slice/part can be compared or used as a key. Scratch locals are i32."""
    from .ir_string_ops import STRING_REF_BYTES_OFFSET
    builder.i64_const(_FNV64_OFFSET_SIGNED).local_set(result_local)
    builder.local_get(start_local).local_set(index_local)
    builder.block()          # done
    builder.loop()           # fold
    builder.local_get(index_local).local_get(end_local).raw(_I32_GE_S)
    builder.br_if(1)
    builder.local_get(ref_local).i32_const(STRING_REF_BYTES_OFFSET).raw(_I32_ADD)
    builder.local_get(index_local).raw(_I32_ADD)
    builder.raw(_I32_LOAD8_U, 0x00, 0x00)
    builder.local_set(byte_local)
    builder.local_get(result_local)
    builder.local_get(byte_local).raw(0xAD)  # i64.extend_i32_u
    builder.raw(_I64_XOR)
    builder.i64_const(_FNV64_PRIME).raw(_I64_MUL)
    builder.local_set(result_local)
    builder.local_get(index_local).i32_const(1).raw(_I32_ADD).local_set(index_local)
    builder.br(0)
    builder.end()            # loop
    builder.end()            # block


def emit_string_split_part_hash(
    builder: CodeBuilder, *, ref_local: int, delim_local: int, part: int,
    result_local: int, pos_local: int, length_local: int, start_local: int,
    end_local: int, index_local: int, byte_local: int,
) -> None:
    """``string_hash(string_split_part(ref, delim, part))`` for ``part`` in {0,1}:
    the token of the prefix before the first ``delim`` (part 0) or the suffix
    after it (part 1). This is the general delimiter split -- the null-terminated
    name hash is the special case ``part=0``. Scratch locals are i32 except
    ``result_local`` (i64)."""
    if part not in (0, 1):
        raise ValueError("only split(delim, 1)[0] / [1] are lowered")
    # pos = find(delim, from=0); length = len(ref). start_local is scratch for
    # the from-offset here before it becomes the range start below.
    builder.i32_const(0).local_set(start_local)
    emit_string_find(builder, ref_local=ref_local, delim_local=delim_local,
                     from_local=start_local, result_local=pos_local,
                     length_local=length_local, index_local=index_local,
                     byte_local=byte_local)
    if part == 0:
        builder.i32_const(0).local_set(start_local)
        builder.local_get(pos_local).local_set(end_local)
    else:
        # suffix starts just past the delimiter, or at length if the delimiter is
        # absent (pos == length). select(a, b, cond) keeps a when cond != 0:
        # start = (pos >= length) ? length : pos + 1.
        builder.local_get(length_local)                       # a = length
        builder.local_get(pos_local).i32_const(1).raw(_I32_ADD)  # b = pos + 1
        builder.local_get(pos_local).local_get(length_local).raw(_I32_GE_S)  # cond
        builder.select()
        builder.local_set(start_local)
        builder.local_get(length_local).local_set(end_local)
    emit_string_hash_range(builder, ref_local=ref_local, start_local=start_local,
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
