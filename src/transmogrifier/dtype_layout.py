"""The compiler's dtype/layout authority.

Every scalar spelling and alias resolves to one DTypeLayout. Storage, ABI,
allocation and byte-count views are derived from that layout, rather than
keeping a second vocabulary that can silently turn an integer into double.
Backend tables include only layouts with a declared native representation.

A missing dtype retains the repository's float64 default. An explicit name
without a layout is refused; a width-only LLVM reverse lookup cannot restore
signedness and must never replace the originating dtype at an ABI boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


#: The dtype every lane assumes when a value declares none.
DEFAULT_DTYPE = "float64"


@dataclass(frozen=True)
class DTypeLayout:
    """One dtype's storage layout and its spelling in every lane.

    ``name`` is the canonical repository spelling.  ``aliases`` lists every
    spelling any backend accepts for this dtype (``name`` included); which
    aliases a given lane honours is declared per lookup below, because the
    copies did not share one vocabulary.
    """

    name: str
    #: "float" | "int" | "uint" | "bool" | "ptr"
    kind: str
    byte_size: int
    #: Natural alignment in bytes.  New information: no copy carried it.
    alignment: int
    #: C ABI spelling and the ctypes name a Python caller needs for it.
    c_type: str
    ctypes_name: str
    #: What the C module lane allocates for this dtype.  Differs from
    #: ``c_type`` only for bool, which that lane stores as ``uint8_t``.
    c_storage_type: str
    llvm_type: str
    wasm_type: str | None
    fortran_kind: str | None
    numpy_dtype: str | None
    nodus_code: int | None
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.byte_size <= 0 or self.alignment <= 0:
            raise ValueError(f"{self.name}: byte_size and alignment must be positive")
        if self.byte_size % self.alignment:
            raise ValueError(f"{self.name}: alignment must divide byte_size")
        if self.name not in self.aliases:
            raise ValueError(f"{self.name}: aliases must include the canonical name")


_DECLARED: tuple[DTypeLayout, ...] = (
    DTypeLayout(
        name="bool", kind="bool", byte_size=1, alignment=1,
        c_type="bool", ctypes_name="c_bool", c_storage_type="uint8_t",
        llvm_type="i1", wasm_type=None, fortran_kind="logical(c_bool)",
        numpy_dtype="bool", nodus_code=15,
        aliases=("bool", "i1", "logical"),
    ),
    DTypeLayout(
        name="uint8", kind="uint", byte_size=1, alignment=1,
        c_type="uint8_t", ctypes_name="c_uint8", c_storage_type="uint8_t",
        llvm_type="i8", wasm_type=None, fortran_kind="unsigned(c_int8_t)",
        numpy_dtype="uint8", nodus_code=7,
        aliases=("uint8", "u8"),
    ),
    DTypeLayout(
        name="int8", kind="int", byte_size=1, alignment=1,
        c_type="int8_t", ctypes_name="c_int8", c_storage_type="int8_t",
        llvm_type="i8", wasm_type=None, fortran_kind="integer(c_int8_t)",
        numpy_dtype="int8", nodus_code=3,
        aliases=("int8",),
    ),
    DTypeLayout(
        name="int16", kind="int", byte_size=2, alignment=2,
        c_type="int16_t", ctypes_name="c_int16", c_storage_type="int16_t",
        llvm_type="i16", wasm_type=None, fortran_kind="integer(c_int16_t)",
        numpy_dtype="int16", nodus_code=4,
        aliases=("int16",),
    ),
    DTypeLayout(
        name="uint16", kind="uint", byte_size=2, alignment=2,
        c_type="uint16_t", ctypes_name="c_uint16", c_storage_type="uint16_t",
        llvm_type="i16", wasm_type=None, fortran_kind="unsigned(c_int16_t)",
        numpy_dtype="uint16", nodus_code=12,
        aliases=("uint16",),
    ),
    DTypeLayout(
        name="int32", kind="int", byte_size=4, alignment=4,
        c_type="int32_t", ctypes_name="c_int32", c_storage_type="int32_t",
        llvm_type="i32", wasm_type="i32", fortran_kind="integer(c_int32_t)",
        numpy_dtype="int32", nodus_code=5,
        aliases=("int32", "i32", "int"),
    ),
    DTypeLayout(
        name="uint32", kind="uint", byte_size=4, alignment=4,
        c_type="uint32_t", ctypes_name="c_uint32", c_storage_type="uint32_t",
        llvm_type="i32", wasm_type=None, fortran_kind="unsigned(c_int32_t)",
        numpy_dtype="uint32", nodus_code=13,
        aliases=("uint32", "u32"),
    ),
    DTypeLayout(
        name="int64", kind="int", byte_size=8, alignment=8,
        c_type="int64_t", ctypes_name="c_int64", c_storage_type="int64_t",
        llvm_type="i64", wasm_type="i64", fortran_kind="integer(c_int64_t)",
        numpy_dtype="int64", nodus_code=6,
        # ``opaque_ref`` is a repository dtype (an opaque handle) that every
        # lane lowers as a 64-bit integer.
        aliases=("int64", "i64", "long", "opaque_ref"),
    ),
    DTypeLayout(
        name="uint64", kind="uint", byte_size=8, alignment=8,
        c_type="uint64_t", ctypes_name="c_uint64", c_storage_type="uint64_t",
        llvm_type="i64", wasm_type=None, fortran_kind="unsigned(c_int64_t)",
        numpy_dtype="uint64", nodus_code=14,
        aliases=("uint64", "u64"),
    ),
    # Internal arithmetic domain for an exact join of uint64 and int64.
    # These native scalar spellings are not a public buffer/ctypes contract:
    # NumPy has no corresponding integer dtype, and their host ABI is not
    # portable.  The public layout views below therefore omit both rows.
    DTypeLayout(
        name="int128", kind="int", byte_size=16, alignment=16,
        c_type="__int128", ctypes_name="", c_storage_type="__int128",
        llvm_type="i128", wasm_type=None, fortran_kind="integer(kind=16)",
        numpy_dtype=None, nodus_code=None,
        aliases=("int128",),
    ),
    DTypeLayout(
        name="uint128", kind="uint", byte_size=16, alignment=16,
        c_type="__uint128_t", ctypes_name="", c_storage_type="__uint128_t",
        llvm_type="i128", wasm_type=None, fortran_kind="unsigned(kind=16)",
        numpy_dtype=None, nodus_code=None,
        aliases=("uint128",),
    ),
    DTypeLayout(
        name="float32", kind="float", byte_size=4, alignment=4,
        c_type="float", ctypes_name="c_float", c_storage_type="float",
        llvm_type="float", wasm_type="f32", fortran_kind="real(c_float)",
        numpy_dtype="float32", nodus_code=1,
        aliases=("float32", "f32", "float"),
    ),
    DTypeLayout(
        name="float64", kind="float", byte_size=8, alignment=8,
        c_type="double", ctypes_name="c_double", c_storage_type="double",
        llvm_type="double", wasm_type="f64", fortran_kind="real(c_double)",
        numpy_dtype="float64", nodus_code=2,
        aliases=("float64", "f64", "double"),
    ),
    # The machine-word storage address internal values cross function
    # boundaries as (LLVM ``ptr``, C ``void *``). It has no numeric numpy
    # scalar representation; pointer ABI views retain that distinct kind.
    DTypeLayout(
        name="ptr", kind="ptr", byte_size=8, alignment=8,
        c_type="void *", ctypes_name="c_void_p", c_storage_type="void *",
        llvm_type="ptr", wasm_type=None, fortran_kind=None,
        numpy_dtype=None, nodus_code=16,
        aliases=("ptr",),
    ),
)

#: Canonical name -> layout.
DTYPES: Mapping[str, DTypeLayout] = MappingProxyType(
    {layout.name: layout for layout in _DECLARED}
)

#: Every accepted spelling -> canonical name. Exact spelling; callers that
#: accept case-insensitive user input casefold it before resolving.
ALIASES: Mapping[str, str] = MappingProxyType({
    alias: layout.name for layout in _DECLARED for alias in layout.aliases
})

if len(ALIASES) != sum(len(layout.aliases) for layout in _DECLARED):
    raise RuntimeError("dtype_layout: an alias is claimed by two dtypes")


def resolve(name: object) -> DTypeLayout | None:
    """The layout ``name`` spells, or ``None``.  Exact spelling, no folding."""

    canonical = ALIASES.get(str(name))
    return None if canonical is None else DTYPES[canonical]


def layout_for(name: object) -> DTypeLayout:
    """The layout ``name`` spells; ``KeyError`` for an unknown spelling."""

    layout = resolve(name)
    if layout is None:
        raise KeyError(f"no dtype layout declared for {name!r}")
    return layout


def common_integer_dtype(dtypes) -> str | None:
    """Return the narrowest declared integer layout covering every operand.

    This is a mathematical value-domain join, not NumPy's dtype promotion.
    A signed layout must represent both the negative extent of every signed
    input and the full positive extent of every unsigned input.  In
    particular, ``uint64`` plus ``int64`` joins at the internal ``int128``
    layout; it never passes through binary64 or a narrowing cast.

    The internal 128-bit layouts are not public NumPy/ctypes buffer types.
    Callers still have to respect that boundary when publishing a result.
    """

    names = tuple(str(dtype or "").casefold() for dtype in dtypes)
    if not names or any(name == "opaque_ref" for name in names):
        return None
    layouts = tuple(resolve(name) for name in names)
    if any(layout is None or layout.kind not in {"int", "uint", "bool"}
           for layout in layouts):
        return None
    if all(layout.kind == "bool" for layout in layouts):
        return "bool"

    minimum = 0
    maximum = 0
    for layout in layouts:
        if layout.kind == "bool":
            value_min, value_max = 0, 1
        elif layout.kind == "uint":
            bits = 8 * layout.byte_size
            value_min, value_max = 0, (1 << bits) - 1
        else:
            bits = 8 * layout.byte_size
            value_min, value_max = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        minimum = min(minimum, value_min)
        maximum = max(maximum, value_max)

    candidates = []
    for layout in _SCALAR_LAYOUTS:
        if layout.kind == "uint":
            bits = 8 * layout.byte_size
            lower, upper = 0, (1 << bits) - 1
        elif layout.kind == "int":
            bits = 8 * layout.byte_size
            lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        else:
            continue
        if lower <= minimum and maximum <= upper:
            candidates.append((layout.byte_size, layout.name))
    return min(candidates)[1] if candidates else None


# --------------------------------------------------------------------------
# Layout views. Native backend support is a property of the same record.
# --------------------------------------------------------------------------

_SCALAR_LAYOUTS = tuple(layout for layout in _DECLARED if layout.kind != "ptr")
_SCALAR_NAMES = tuple(alias for layout in _SCALAR_LAYOUTS for alias in layout.aliases)
_PUBLIC_SCALAR_LAYOUTS = tuple(
    layout for layout in _SCALAR_LAYOUTS
    if layout.numpy_dtype is not None and layout.ctypes_name
)
_PUBLIC_SCALAR_NAMES = tuple(
    alias for layout in _PUBLIC_SCALAR_LAYOUTS for alias in layout.aliases
)

C_ABI_NAMES: tuple[str, ...] = _PUBLIC_SCALAR_NAMES
C_LANE_NAMES: tuple[str, ...] = _PUBLIC_SCALAR_NAMES
SHELL_NUMPY_NAMES: tuple[str, ...] = _PUBLIC_SCALAR_NAMES
LLVM_LANE_NAMES: tuple[str, ...] = (*_PUBLIC_SCALAR_NAMES, "ptr")
BYTE_SIZE_NAMES: tuple[str, ...] = tuple(ALIASES)
C_LANE_STORAGE_CLASSES: tuple[str, ...] = tuple(
    layout.name for layout in _PUBLIC_SCALAR_LAYOUTS
)

# LLVM integer spelling states width, not sign. These representative layouts
# are used only by width/byte views. Call and buffer code must retain its exact
# dtype record to distinguish uint32 from int32, or uint64 from int64.
LLVM_LANE_TYPES: tuple[str, ...] = (
    "float64", "float32", "int64", "ptr", "int32", "int16", "int8", "bool",
)
FORTRAN_NAMES: tuple[str, ...] = tuple(
    alias for layout in _SCALAR_LAYOUTS if layout.fortran_kind is not None
    for alias in layout.aliases
)
WASM_NAMES: tuple[str, ...] = tuple(
    alias for layout in _SCALAR_LAYOUTS if layout.wasm_type is not None
    for alias in layout.aliases
)


# --------------------------------------------------------------------------
# 1. compiled_program_api._C_TYPES / _c_type_for
# --------------------------------------------------------------------------

def c_abi_type_table() -> dict[str, tuple[str, str]]:
    """``{dtype spelling: (C ABI type, ctypes name)}`` for the C-ABI vocabulary."""

    return {
        name: (layout_for(name).c_type, layout_for(name).ctypes_name)
        for name in C_ABI_NAMES
    }


def c_abi_type(dtype: object) -> tuple[str, str]:
    """Exact C ABI spelling and ctypes name; refuse an undeclared name."""

    spelling = str(dtype or DEFAULT_DTYPE).casefold()
    layout = layout_for(spelling)
    # ``ptr`` is an internal direct-call ABI value with an established
    # ``void *``/``c_void_p`` spelling, but it is not a scalar buffer dtype.
    # Preserve that direct-call contract while refusing the internal 128-bit
    # arithmetic layouts, which have no ctypes representation.
    if spelling not in C_ABI_NAMES and layout.kind != "ptr":
        raise KeyError(f"no public C ABI layout declared for {dtype!r}")
    return (layout.c_type, layout.ctypes_name)


# --------------------------------------------------------------------------
# 2-4. ssa_c_backend: _numpy_dtype / _dtype_for_c_storage / solved_buffer_type
# --------------------------------------------------------------------------

def c_lane_dtype_for_storage(c_type: str) -> str:
    """Repository dtype stated by a C scalar spelling.

    c632 exposed the old uint8_t -> bool collapse. A shared one-byte storage
    width is not shared value identity: the declared uint8_t type is uint8.
    Bool is recovered from its own C spelling, not from an unsigned byte.
    """

    for layout in _SCALAR_LAYOUTS:
        if layout.c_type == c_type:
            return layout.name
    raise KeyError(f"no C storage layout declared for {c_type!r}")


def c_lane_numpy_dtype(dtype: object) -> str:
    """Exact numpy allocation dtype of the originating scalar layout."""

    layout = layout_for(str(dtype or DEFAULT_DTYPE).casefold())
    if layout.numpy_dtype is None:
        raise KeyError(f"no numpy storage layout declared for {dtype!r}")
    return layout.numpy_dtype


def c_lane_storage_for_llvm_type(llvm_type: object) -> str:
    """C storage spelling for an LLVM width; refuse an undeclared width.

    This view does not recover signedness. An exact originating dtype takes
    precedence at semantic storage/call boundaries.
    """

    layout = by_llvm_type(llvm_type)
    if layout is None:
        raise KeyError(f"no LLVM storage layout declared for {llvm_type!r}")
    return layout.c_storage_type


# --------------------------------------------------------------------------
# 5-6. ssa_llvm_backend: _value_llvm_type / _LLVM_TYPE_BYTES
# --------------------------------------------------------------------------

def llvm_type_for_dtype(dtype: object) -> str:
    """LLVM width stated by the exact dtype; refuse an undeclared name.

    c626/c632 preserved uint32/uint8 in SSA, but the old restricted name view
    emitted double. The physical width is always read from that same layout.
    Scalar float32 promotion remains the consuming emitter's stated policy.
    """

    return layout_for(str(dtype or DEFAULT_DTYPE).casefold()).llvm_type


def by_llvm_type(llvm_type: object) -> DTypeLayout | None:
    """The LLVM-lane layout spelled ``llvm_type`` (exact), or ``None``."""

    spelling = str(llvm_type)
    for name in LLVM_LANE_TYPES:
        if DTYPES[name].llvm_type == spelling:
            return DTYPES[name]
    return None


def llvm_abi_numpy_table() -> dict[str, str]:
    """``{LLVM spelling: numpy dtype}`` of the LLVM lane's public buffers.

    Declared from the same layouts as every other lookup here; ``ptr`` is the
    machine word (``uintp``), ``i1`` is numpy ``bool``.
    """

    return {
        DTYPES[name].llvm_type: (
            "uintp" if name == "ptr" else str(DTYPES[name].numpy_dtype)
        )
        for name in LLVM_LANE_TYPES
    }


def llvm_type_bytes_table() -> dict[str, int]:
    """``{LLVM spelling: element bytes}`` for internal LLVM storage types.

    Public buffer spellings remain governed by ``LLVM_LANE_TYPES`` and its
    NumPy ABI table. Internal scalar widths such as i128 still need exact
    alloca/copy sizes without becoming public buffer types.
    """

    return {
        layout.llvm_type: layout.byte_size
        for layout in _DECLARED
    }


# --------------------------------------------------------------------------
# 7. ssa_fortran_backend._DTYPE_KIND
# --------------------------------------------------------------------------

def fortran_kind_table() -> dict[str, str]:
    """``{dtype spelling: Fortran kind}`` for the Fortran vocabulary."""

    return {name: str(layout_for(name).fortran_kind) for name in FORTRAN_NAMES}


# --------------------------------------------------------------------------
# 8-9. tensor_ssa_lowering dtype_bytes / ssa_backend._dtype_bytes
# --------------------------------------------------------------------------

def dtype_byte_size(dtype: object, default: int = 8) -> int:
    """Exact layout byte count; ``default`` applies only to an absent dtype.

    uint32 used to miss a legacy whitelist and allocate/count eight-byte rows
    even though its authority states four. Explicit names cannot fall through.
    """

    if dtype is None or str(dtype) == "":
        return default
    return layout_for(str(dtype).casefold()).byte_size


# --------------------------------------------------------------------------
# 10. fused_program_wasm_backend._TYPES
# --------------------------------------------------------------------------

def wasm_type_table() -> dict[str, tuple[str, int, str, str]]:
    """``{dtype spelling: (WAT type, element bytes, load, store)}``."""

    table: dict[str, tuple[str, int, str, str]] = {}
    for name in WASM_NAMES:
        layout = layout_for(name)
        wasm = str(layout.wasm_type)
        table[name] = (wasm, layout.byte_size, f"{wasm}.load", f"{wasm}.store")
    return table


# --------------------------------------------------------------------------
# 11. ssa_javascript_backend._dtype_is_int64
# --------------------------------------------------------------------------

def dtype_is_int64(dtype: object) -> bool:
    """Whether ``dtype`` names a 64-bit integer (signed or unsigned).

    Casefolded over every alias; ``None``/empty -> ``False``.
    """

    layout = resolve(str(dtype or "").casefold())
    return layout is not None and layout.kind in ("int", "uint") and layout.byte_size == 8


# --------------------------------------------------------------------------
# 13. fortran_c_shell._NUMPY_DTYPES
# --------------------------------------------------------------------------

def shell_numpy_dtype_table() -> dict[str, str]:
    """``{dtype spelling: numpy dtype spelling}`` for the C shell's vocabulary.

    Spellings, not ``numpy.dtype`` objects: this module imports nothing.  The
    shell wraps each value in ``numpy.dtype`` itself.
    """

    return {name: str(layout_for(name).numpy_dtype) for name in SHELL_NUMPY_NAMES}


def shell_numpy_dtype(dtype: object) -> str | None:
    """Exact numpy dtype spelling; unknown/non-numeric types return None.

    The shell turns absence of a numeric layout into its named refusal.
    """

    layout = resolve(str(dtype).casefold())
    return None if layout is None or layout.numpy_dtype is None else layout.numpy_dtype


__all__ = [
    "DEFAULT_DTYPE",
    "DTypeLayout",
    "DTYPES",
    "ALIASES",
    "resolve",
    "layout_for",
    "common_integer_dtype",
    "by_llvm_type",
    "llvm_abi_numpy_table",
    "C_ABI_NAMES",
    "C_LANE_NAMES",
    "C_LANE_STORAGE_CLASSES",
    "LLVM_LANE_NAMES",
    "LLVM_LANE_TYPES",
    "FORTRAN_NAMES",
    "BYTE_SIZE_NAMES",
    "WASM_NAMES",
    "SHELL_NUMPY_NAMES",
    "c_abi_type_table",
    "c_abi_type",
    "c_lane_dtype_for_storage",
    "c_lane_numpy_dtype",
    "c_lane_storage_for_llvm_type",
    "llvm_type_for_dtype",
    "llvm_type_bytes_table",
    "fortran_kind_table",
    "dtype_byte_size",
    "wasm_type_table",
    "dtype_is_int64",
    "shell_numpy_dtype_table",
    "shell_numpy_dtype",
]
