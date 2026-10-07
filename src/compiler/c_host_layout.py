"""The host-facing header of a C module artifact: ``<entry>_layout.h``.

An artifact made from the entry's API_CONTRACT row (``emission_concordance.
post_api_contract``), which is itself DERIVED from the entry's
``program_abi_field_slot`` rows and BUFFER_ORDER.  Nothing here decides a
layout: ``layout_header_source`` prints the contract's table, so the header,
the book and the compiled entry cannot name different buffers.

The header carries

* ``enum turing_dtype`` -- the C lane's storage classes
  (``dtype_layout.C_LANE_STORAGE_CLASSES``), in that order;
* ``turing_layout_entry <entry>_layout[]`` and ``<entry>_layout_count`` -- one
  entry per public buffer in ``void **buffers`` order;
* ``<ENTRY>_BUFFER_COUNT``, ``<ENTRY>_BATCH`` (when the host declared one)
  and ``<ENTRY>_COL_<name>`` buffer-index constants for every named buffer;
* the entry's prototype.

A buffer's name is its ProgramABI slot: ``parameter.field`` for a record
field (``.presence`` appended for an optional field's presence slot),
``parameter`` for a bare parameter.  A buffer no slot names is unnamed.
"""

from __future__ import annotations

import re
from typing import Iterable

from ..transmogrifier.dtype_layout import C_LANE_STORAGE_CLASSES, DTYPES
from .concordance_declarations import LayoutBuffer, LayoutKind, ProgramAbiSlotRole

#: ``turing_layout_entry.kind`` / ``enum turing_layout_kind``.
_KINDS = {LayoutKind.FIELD: 0, LayoutKind.SCALAR: 1, LayoutKind.OTHER: 2}


def column_name(buffer: LayoutBuffer) -> str | None:
    """The name a host binds a buffer by, from its slot; None when no slot
    names it."""

    if buffer.kind is LayoutKind.OTHER:
        return None
    base = (
        str(buffer.parameter) if buffer.field is None
        else f"{buffer.parameter}.{buffer.field}"
    )
    if buffer.role is ProgramAbiSlotRole.PRESENCE:
        base += ".presence"
    return base


def _identifier(text: str) -> str:
    return re.sub(r"\W", "_", str(text))


def _literal(text: str | None) -> str:
    if text is None:
        return "NULL"
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def layout_header_source(
    entry: str, batch: int | None, buffers: Iterable[LayoutBuffer],
    *, fingerprint: str = "",
) -> str:
    """The text of ``<entry>_layout.h`` for one API_CONTRACT table."""

    buffers = tuple(buffers)
    symbol = _identifier(entry)
    upper = symbol.upper()
    dtype_enum = {name: index for index, name in enumerate(C_LANE_STORAGE_CLASSES)}
    constants: dict[str, str] = {}
    for buffer in buffers:
        name = column_name(buffer)
        if name is None:
            continue
        macro = f"{upper}_COL_{_identifier(name)}"
        if macro in constants:
            raise ValueError(
                f"layout header of {entry!r}: columns {constants[macro]!r} and "
                f"{name!r} both spell {macro}"
            )
        constants[macro] = name
    for buffer in buffers:
        if buffer.dtype not in dtype_enum:
            raise ValueError(
                f"layout header of {entry!r}: buffer {buffer.buffer_index} has "
                f"dtype {buffer.dtype!r}, which is not a C lane storage class "
                f"{C_LANE_STORAGE_CLASSES!r}"
            )
        if buffer.itemsize != DTYPES[buffer.dtype].byte_size:
            raise ValueError(
                f"layout header of {entry!r}: buffer {buffer.buffer_index} "
                f"itemsize {buffer.itemsize} disagrees with {buffer.dtype!r}"
            )

    lines = [
        f"/* {entry}_layout.h -- the host-facing layout of {entry}.",
        " * Printed from the entry's API_CONTRACT row on the identity book",
        " * (derived from its program_abi_field_slot rows and BUFFER_ORDER);",
        " * edit the program, not this file.",
        *((f" * contract sha256: {fingerprint}",) if fingerprint else ()),
        " */",
        "#ifndef TURING_LAYOUT_TYPES_H",
        "#define TURING_LAYOUT_TYPES_H",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "enum turing_dtype {",
        *(
            f"    TURING_{name.upper()} = {index},"
            for name, index in dtype_enum.items()
        ),
        "};",
        "",
        "enum turing_layout_kind {",
        "    TURING_LAYOUT_FIELD = 0,",
        "    TURING_LAYOUT_SCALAR = 1,",
        "    TURING_LAYOUT_OTHER = 2,",
        "};",
        "",
        "/* One public buffer of an entry, at buffer_index of void **buffers.",
        " * name: \"parameter.field\" or \"parameter\" (NULL when no ProgramABI slot",
        " * names the buffer); count: the declared elements; capacity: the",
        " * elements the entry may touch (allocate at least this). */",
        "typedef struct turing_layout_entry {",
        "    const char *name;",
        "    const char *parameter;",
        "    const char *field;",
        "    int32_t buffer_index;",
        "    int32_t dtype;",
        "    int32_t kind;",
        "    int32_t written;",
        "    uint64_t count;",
        "    uint64_t capacity;",
        "    uint64_t itemsize;",
        "} turing_layout_entry;",
        "",
        "#endif",
        "",
        f"#ifndef {upper}_LAYOUT_H",
        f"#define {upper}_LAYOUT_H",
        "",
        f"#define {upper}_BUFFER_COUNT {len(buffers)}",
        *((f"#define {upper}_BATCH {int(batch)}",) if batch is not None else ()),
        *(
            f"#define {macro} {next(b.buffer_index for b in buffers if column_name(b) == name)}"
            for macro, name in constants.items()
        ),
        "",
        f"void {symbol}(void **buffers, long long *extents);",
        "",
        f"static const turing_layout_entry {symbol}_layout[] = {{",
        *(
            "    {"
            + ", ".join((
                _literal(column_name(buffer)),
                _literal(None if buffer.parameter is None else str(buffer.parameter)),
                _literal(None if buffer.field is None else str(buffer.field)),
                str(buffer.buffer_index),
                f"TURING_{buffer.dtype.upper()}",
                f"TURING_LAYOUT_{buffer.kind.name}",
                str(int(buffer.written)),
                f"{buffer.count}u",
                f"{buffer.capacity}u",
                f"{buffer.itemsize}u",
            ))
            + "},"
            for buffer in buffers
        ),
        "};",
        f"static const int32_t {symbol}_layout_count = {upper}_BUFFER_COUNT;",
        "",
        f"#endif /* {upper}_LAYOUT_H */",
        "",
    ]
    return "\n".join(lines)
