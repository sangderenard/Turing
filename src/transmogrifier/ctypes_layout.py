"""Struct and union rows from intercepted ctypes types.

The eager program declares its byte layouts as ``ctypes.Structure`` and
``ctypes.Union`` subclasses and ctypes lays them out.  When the same source is
compiled, the compiler intercepts those classes as live objects and asks
ctypes what it already knows -- ``sizeof``, ``alignment``, each field's
``offset`` and ``size`` -- and writes one :class:`SSAStructDescriptor` or
:class:`SSAUnionDescriptor` row per type into the module's type tables.
Nothing here re-derives a layout, parses ``_fields_`` as syntax, or applies a
layout rule of its own: the numbers are read off the class.

Every row is counted in the one memory schema Python delivers, the host's
(:func:`host_schema`): a unit of 8 bits and the host's justification.  Nothing
declares it and no contract may change it; the schema a backend targets is a
separate contract fact, and absent it the same schema, so nothing converts.

Leaf dtypes are the repository spellings declared in
:mod:`src.transmogrifier.dtype_layout`.  A ctypes scalar is identified by its
``_type_`` code (``'d'`` double, ``'q'`` int64, ...), never by its Python class
name, because ctypes aliases those names per platform (``c_int64 is c_long`` on
LP64).  A scalar with no repository dtype (``uint32``, long double, wide char,
bitfields) is refused loudly rather than widened.

A union member that is a scalar is wrapped as a one-field struct row named
``<union identity>.<member>`` so every union member is a struct and the
per-backend recipe (storage member + tail padding) is uniform.
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from .dtype_layout import DTYPES, DTypeLayout
from .ssa import (
    SSALayoutSchema,
    SSAStructDescriptor,
    SSAStructFieldDescriptor,
    SSAStructTable,
    SSAUnionDescriptor,
    SSAUnionTable,
)


class CTypesLayoutError(ValueError):
    """A ctypes type that has no exact repository layout."""


def host_schema() -> SSALayoutSchema:
    """The one memory schema Python and ctypes deliver: the host's.

    ctypes counts in 8-bit units; a member narrower than a unit sits at the
    low end on a little-endian host and the high end on a big-endian one.
    Offsets run up.  Python cannot say what the cache-ideal spans are, so
    none are declared.
    """

    return SSALayoutSchema(
        unit_bits=8,
        justification="low" if sys.byteorder == "little" else "high",
        direction="up",
    )


# ``_type_`` code -> (kind, signed).  Sizes come from ctypes.sizeof, so the
# same code resolves to the platform's actual width (``'l'`` is 8 on LP64).
_SCALAR_CODES: dict[str, tuple[str, bool]] = {
    "?": ("bool", False),
    "b": ("int", True), "B": ("uint", False),
    "h": ("int", True), "H": ("uint", False),
    "i": ("int", True), "I": ("uint", False),
    "l": ("int", True), "L": ("uint", False),
    "q": ("int", True), "Q": ("uint", False),
    "f": ("float", True), "d": ("float", True),
    "P": ("ptr", False),
}


def _scalar_layout(ctype: Any) -> DTypeLayout:
    code = getattr(ctype, "_type_", None)
    if not isinstance(code, str) or code not in _SCALAR_CODES:
        raise CTypesLayoutError(
            f"{ctype!r}: not a ctypes scalar with a repository layout"
        )
    kind, _signed = _SCALAR_CODES[code]
    size = ctypes.sizeof(ctype)
    if kind == "ptr":
        return DTYPES["ptr"]
    for layout in DTYPES.values():
        if layout.kind == kind and layout.byte_size == size:
            return layout
    raise CTypesLayoutError(
        f"{ctype!r} ({kind}, {size} bytes): no repository dtype declares this "
        f"layout in dtype_layout.py"
    )


def scalar_dtype_of(ctype: Any) -> str:
    """The repository dtype a ctypes scalar type is, by its code and size."""

    return _scalar_layout(ctype).name


def _is_struct(ctype: Any) -> bool:
    return isinstance(ctype, type) and issubclass(ctype, ctypes.Structure)


def _is_union(ctype: Any) -> bool:
    return isinstance(ctype, type) and issubclass(ctype, ctypes.Union)


def _is_array(ctype: Any) -> bool:
    return isinstance(ctype, type) and issubclass(ctype, ctypes.Array)


def type_identity(ctype: type) -> str:
    """``module.qualname`` -- the spelling the program ABI uses for a class."""

    return f"{getattr(ctype, '__module__', '')}.{ctype.__qualname__}".strip(".")


@dataclass
class CTypesInterception:
    """Writes struct/union rows for ctypes classes into a module's tables.

    ``mint`` issues row ids (the compiler passes its global issuer).  A class
    seen twice yields the same row; a second, different layout under one
    identity is recorded by the tables as a supersession edge on the identity
    book (stage ``"ctypes_interception"``), never silently overwritten.
    """

    STAGE = "ctypes_interception"

    struct_table: SSAStructTable
    union_table: SSAUnionTable
    mint: Callable[[], int]
    schema: SSALayoutSchema = field(default_factory=host_schema)
    _struct_ids: dict[type, int] = field(default_factory=dict)
    _union_ids: dict[type, int] = field(default_factory=dict)

    # -- public ---------------------------------------------------------

    def intercept(self, ctype: type) -> tuple[str, int]:
        """Register ``ctype`` (a Structure or Union subclass) and every type it
        contains.  Returns ``("struct" | "union", row id)``."""

        if _is_union(ctype):
            return ("union", self._union_row(ctype))
        if _is_struct(ctype):
            return ("struct", self._struct_row(ctype))
        raise CTypesLayoutError(
            f"{ctype!r} is neither a ctypes.Structure nor a ctypes.Union subclass"
        )

    @staticmethod
    def is_layout_type(value: Any) -> bool:
        return _is_struct(value) or _is_union(value)

    # -- rows -----------------------------------------------------------

    def _member(self, owner: type, name: str, ctype: Any) -> SSAStructFieldDescriptor:
        descriptor = getattr(owner, name)
        offset = int(descriptor.offset)
        size = int(descriptor.size)
        if size >> 16:
            # ctypes packs bitfield width into the high bits of ``size``.
            raise CTypesLayoutError(
                f"{type_identity(owner)}.{name}: bitfields have no repository layout"
            )
        count = 1
        element = ctype
        while _is_array(element):
            count *= int(element._length_)
            element = element._type_
        if _is_union(element):
            return SSAStructFieldDescriptor(
                name, self._units(offset, owner, name),
                self._units(ctypes.sizeof(element), owner, name),
                union_id=self._union_row(element), count=count,
            )
        if _is_struct(element):
            return SSAStructFieldDescriptor(
                name, self._units(offset, owner, name),
                self._units(ctypes.sizeof(element), owner, name),
                struct_id=self._struct_row(element), count=count,
            )
        layout = _scalar_layout(element)
        return SSAStructFieldDescriptor(
            name, self._units(offset, owner, name),
            self._units(layout.byte_size, owner, name),
            dtype=layout.name, count=count,
        )

    def _units(self, host_bytes: int, owner: Any, what: str) -> int:
        """Host bytes counted in the schema's units.  ctypes delivers bytes;
        the schema divides them, or the type is refused, never rounded."""

        return self.schema.units(
            int(host_bytes) * 8, f"{type_identity(owner)}.{what}",
        )

    def _struct_row(self, ctype: type) -> int:
        known = self._struct_ids.get(ctype)
        if known is not None:
            return known
        fields = tuple(
            self._member(ctype, str(name), member_type)
            for name, member_type, *_ in ctype._fields_
        )
        row = SSAStructDescriptor(
            self.mint(), type_identity(ctype), self.schema,
            self._units(ctypes.sizeof(ctype), ctype, "sizeof"),
            self._units(ctypes.alignment(ctype), ctype, "alignment"), fields,
        )
        self.struct_table.register(row, stage=self.STAGE)
        self._struct_ids[ctype] = row.struct_id
        return row.struct_id

    def _scalar_member_struct(self, union: type, name: str, ctype: Any) -> int:
        """A scalar union alternative as a one-field struct row."""

        layout = _scalar_layout(ctype)
        identity = f"{type_identity(union)}.{name}"
        existing = self.struct_table.by_identity(identity)
        if existing is not None:
            return existing.struct_id
        size = self._units(layout.byte_size, union, name)
        row = SSAStructDescriptor(
            self.mint(), identity, self.schema, size,
            self._units(layout.alignment, union, name),
            (SSAStructFieldDescriptor(name, 0, size, dtype=layout.name),),
        )
        self.struct_table.register(row, stage=self.STAGE)
        return row.struct_id

    def _union_row(self, ctype: type) -> int:
        known = self._union_ids.get(ctype)
        if known is not None:
            return known
        members: list[SSAStructFieldDescriptor] = []
        alignments: list[int] = []
        for name, member_type, *_ in ctype._fields_:
            name = str(name)
            descriptor = getattr(ctype, name)
            if int(descriptor.size) >> 16:
                raise CTypesLayoutError(
                    f"{type_identity(ctype)}.{name}: bitfields have no repository layout"
                )
            if _is_array(member_type):
                raise CTypesLayoutError(
                    f"{type_identity(ctype)}.{name}: an array union member is not "
                    f"a struct; wrap it in a one-field Structure"
                )
            if _is_union(member_type):
                raise CTypesLayoutError(
                    f"{type_identity(ctype)}.{name}: a union directly inside a "
                    f"union is not a struct member; wrap it in a Structure"
                )
            if _is_struct(member_type):
                struct_id = self._struct_row(member_type)
            else:
                struct_id = self._scalar_member_struct(ctype, name, member_type)
            row = self.struct_table.by_id(struct_id)
            members.append(SSAStructFieldDescriptor(
                name, self._units(descriptor.offset, ctype, name), row.size,
                struct_id=struct_id,
            ))
            alignments.append(row.alignment)
        strictest = max(alignments)
        storage = next(
            member.name for member, alignment in zip(members, alignments)
            if alignment == strictest
        )
        row = SSAUnionDescriptor(
            self.mint(), type_identity(ctype), self.schema,
            self._units(ctypes.sizeof(ctype), ctype, "sizeof"),
            self._units(ctypes.alignment(ctype), ctype, "alignment"),
            tuple(members), storage,
        )
        self.union_table.register(row, stage=self.STAGE)
        self._union_ids[ctype] = row.union_id
        return row.union_id


def member_path_layout(
    struct_table: SSAStructTable,
    union_table: SSAUnionTable,
    kind: str,
    row_id: int,
    path: tuple[str, ...],
) -> tuple[int, str | None, tuple[str, int] | None, int]:
    """Resolve an attribute path against a row.

    Returns ``(offset in units, leaf dtype or None, aggregate (kind, id) or None,
    count)``.  Offsets accumulate along the path; a union member adds zero.
    Raises ``CTypesLayoutError`` on an unknown member or a leaf reached early.
    """

    offset = 0
    current: tuple[str, int] | None = (kind, int(row_id))
    dtype: str | None = None
    count = 1
    for name in path:
        if current is None:
            raise CTypesLayoutError(
                f"member {name!r} requested below leaf of dtype {dtype!r}"
            )
        table_kind, identifier = current
        if table_kind == "union":
            row = union_table.by_id(identifier)
            member = None if row is None else row.member(name)
        else:
            row = struct_table.by_id(identifier)
            member = None if row is None else row.field(name)
        if row is None or member is None:
            raise CTypesLayoutError(
                f"{table_kind} {identifier} has no member {name!r}"
                + ("" if row is None else f" (has {[m.name for m in (row.members if table_kind == 'union' else row.fields)]})")
            )
        offset += member.offset
        count = member.count
        if member.struct_id is not None:
            current = ("struct", member.struct_id)
            dtype = None
        elif member.union_id is not None:
            current = ("union", member.union_id)
            dtype = None
        else:
            current = None
            dtype = member.dtype
    return offset, dtype, current, count
