"""The compiler's one dtype/layout authority.

Every dtype name the compiler accepts is declared here exactly once, as a
frozen :class:`DTypeLayout` carrying its byte size, natural alignment and the
spelling each lane uses for it (C ABI, C module-lane storage, LLVM, WebAssembly,
Fortran kind, numpy dtype, nodus code), together with every alias the backends
accept for it.  The per-backend tables that used to carry these facts are now
lookups into this module (``docs/UNION_TYPE_DESIGN_2026-09-29.md`` §1 "dtype
authority: none" and §3.3).

**Alignment is new information.**  No copy carried an alignment; each one
recorded only a byte size (and the LLVM lane wrote ``align 8`` on everything
it could).  ``alignment`` is declared here for the first time: the byte size
for every scalar, which is 1 for the one-byte types (bool / i1 / uint8).  The
C and LLVM spellings of the types no lane emits today (int8, int16, uint16,
uint64, float32's LLVM ``float``) are likewise declared fresh; they are the
fixed ``<stdint.h>`` / LLVM names, not compiler policy, and no lookup below
returns them because no copy ever did.

**The twelve copies this module replaces** (each lookup reproduces one copy's
vocabulary, case handling and unknown-name fallback exactly; the copies did
not agree with each other, so the per-copy vocabularies below are as much a
part of the record as the byte sizes):

1. ``compiled_program_api._C_TYPES`` -> :func:`c_abi_type_table`,
   :func:`c_abi_type` (C ABI spelling + ctypes name; unknown -> double).
2. ``ssa_c_backend._numpy_dtype`` -> :func:`c_lane_numpy_dtype` (collapses
   to the four storage classes the C module lane allocates; unknown ->
   ``"float64"``; ``uint8`` shares bool's one-byte storage).
3. ``ssa_c_backend._dtype_for_c_storage`` -> :func:`c_lane_dtype_for_storage`
   (C storage spelling -> repository dtype of that storage class).
4. ``ssa_c_backend.solved_buffer_type``'s inline mapping ->
   :func:`c_lane_storage_for_llvm_type` (LLVM spelling -> C storage spelling;
   ``i1`` is stored as ``uint8_t``; unknown -> ``"double"``).
5. ``ssa_llvm_backend._value_llvm_type``'s name mapping ->
   :func:`llvm_type_for_dtype` (unknown -> ``"double"``).
6. ``ssa_llvm_backend._LLVM_TYPE_BYTES`` -> :func:`llvm_type_bytes_table`.
7. ``ssa_fortran_backend._DTYPE_KIND`` -> :func:`fortran_kind_table`.
8. ``tensor_ssa_lowering``'s three inline ``dtype_bytes`` dicts ->
   :func:`dtype_byte_size` (unknown -> 8).
9. ``accelerator_backends.ssa_backend.SSATensorOperations._dtype_bytes`` ->
   :func:`dtype_byte_size`.
10. ``fused_program_wasm_backend._TYPES`` -> :func:`wasm_type_table`.
11. ``ssa_javascript_backend._dtype_is_int64`` -> :func:`dtype_is_int64`.
12. ``ssa_storage_requirements``'s default ``"float64"`` -> :data:`DEFAULT_DTYPE`.

``fortran_c_shell._NUMPY_DTYPES`` (the numpy dtype table of the shell) is the
same kind of copy and is left for a follow-up; it is not edited here.

This module imports nothing from the repository so that every lane can import
it.  The nodus codes mirror ``NodusTensorDType`` in nodus' ``tensor_abi.h``
(``accelerator_backends/nodus_arena.py``), which loads a DLL on import and is
therefore not imported from here.
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
        llvm_type="i8", wasm_type=None, fortran_kind=None,
        numpy_dtype="uint8", nodus_code=7,
        aliases=("uint8", "u8"),
    ),
    DTypeLayout(
        name="int8", kind="int", byte_size=1, alignment=1,
        c_type="int8_t", ctypes_name="c_int8", c_storage_type="int8_t",
        llvm_type="i8", wasm_type=None, fortran_kind=None,
        numpy_dtype="int8", nodus_code=3,
        aliases=("int8",),
    ),
    DTypeLayout(
        name="int16", kind="int", byte_size=2, alignment=2,
        c_type="int16_t", ctypes_name="c_int16", c_storage_type="int16_t",
        llvm_type="i16", wasm_type=None, fortran_kind=None,
        numpy_dtype="int16", nodus_code=4,
        aliases=("int16",),
    ),
    DTypeLayout(
        name="uint16", kind="uint", byte_size=2, alignment=2,
        c_type="uint16_t", ctypes_name="c_uint16", c_storage_type="uint16_t",
        llvm_type="i16", wasm_type=None, fortran_kind=None,
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
        llvm_type="i64", wasm_type=None, fortran_kind=None,
        numpy_dtype="uint64", nodus_code=14,
        aliases=("uint64", "u64"),
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
    # boundaries as (LLVM ``ptr``, C ``void *``).  Not a repository dtype: no
    # dtype-name lookup below resolves to it; it exists for the LLVM-spelling
    # lookups (byte width, C storage) that must know it.
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

#: Every accepted spelling -> canonical name.  Exact spelling; the per-copy
#: lookups apply whatever case folding their copy applied.
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


def _resolve_within(name: str, vocabulary: frozenset[str]) -> DTypeLayout | None:
    return resolve(name) if name in vocabulary else None


def _check_vocabulary(vocabulary: frozenset[str]) -> frozenset[str]:
    unknown = sorted(vocabulary - ALIASES.keys())
    if unknown:
        raise RuntimeError(f"dtype_layout: vocabulary names undeclared dtypes {unknown}")
    return vocabulary


# --------------------------------------------------------------------------
# Per-lane vocabularies.  Each is exactly the key set of the copy it replaces.
# --------------------------------------------------------------------------

#: ``compiled_program_api._C_TYPES`` keys (declaration order kept).
C_ABI_NAMES: tuple[str, ...] = (
    "uint8", "u8",
    "float", "float32", "f32",
    "double", "float64", "f64",
    "int", "int32", "i32",
    "int64", "i64", "opaque_ref",
    "bool", "logical",
)
_C_ABI_VOCABULARY = _check_vocabulary(frozenset(C_ABI_NAMES))

#: The names ``ssa_c_backend._numpy_dtype`` recognised (casefolded).
C_LANE_NAMES: tuple[str, ...] = (
    "bool", "i1", "uint8", "u8",
    "int", "int32", "i32",
    "int64", "i64", "long",
)
_C_LANE_VOCABULARY = _check_vocabulary(frozenset(C_LANE_NAMES))

#: The four storage classes the C module lane allocates, in the order
#: ``_dtype_for_c_storage`` tested them.  A dtype belongs to the class whose
#: ``c_storage_type`` it shares (uint8 -> bool); anything else is float64.
C_LANE_STORAGE_CLASSES: tuple[str, ...] = ("bool", "int32", "int64", "float64")

#: The names ``ssa_llvm_backend._value_llvm_type`` recognised (lowercased).
LLVM_LANE_NAMES: tuple[str, ...] = (
    "bool", "i1",
    "int", "int32", "i32",
    "int64", "i64", "long",
    "opaque_ref",
)
_LLVM_LANE_VOCABULARY = _check_vocabulary(frozenset(LLVM_LANE_NAMES))

#: The layouts whose LLVM spelling the LLVM lane emits -- the key set of
#: ``_LLVM_TYPE_BYTES`` and of ``solved_buffer_type``'s mapping.  Reverse
#: lookups by LLVM spelling search only these (``i64`` is also uint64's
#: spelling, and ``i8``/``i16`` are shared by two dtypes each).
LLVM_LANE_TYPES: tuple[str, ...] = ("float64", "int64", "ptr", "int32", "bool")

#: ``ssa_fortran_backend._DTYPE_KIND`` keys (exact case).
FORTRAN_NAMES: tuple[str, ...] = (
    "double", "i64", "i32", "i1",
    "float64", "float32", "float",
    "int64", "int32", "int",
    "bool", "opaque_ref",
)
_FORTRAN_VOCABULARY = _check_vocabulary(frozenset(FORTRAN_NAMES))

#: Keys of the ``dtype_bytes`` dicts in ``tensor_ssa_lowering`` and of
#: ``ssa_backend._dtype_bytes`` (lowercased).  Note ``f32``/``f64``/``u8``/
#: ``long``/``opaque_ref``/``logical`` were NOT keys and fell to the default.
BYTE_SIZE_NAMES: tuple[str, ...] = (
    "bool", "i1", "int8", "uint8",
    "int16", "uint16",
    "float32", "float", "int32", "i32",
    "float64", "double", "int64", "i64",
)
_BYTE_SIZE_VOCABULARY = _check_vocabulary(frozenset(BYTE_SIZE_NAMES))

#: ``fused_program_wasm_backend._TYPES`` keys (declaration order kept).
WASM_NAMES: tuple[str, ...] = (
    "float64", "f64", "double",
    "float32", "f32", "float",
    "int64", "i64", "long",
    "int32", "i32", "int",
)
_WASM_VOCABULARY = _check_vocabulary(frozenset(WASM_NAMES))

for _name in LLVM_LANE_TYPES + C_LANE_STORAGE_CLASSES:
    if _name not in DTYPES:
        raise RuntimeError(f"dtype_layout: lane names undeclared dtype {_name!r}")
del _name


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
    """C ABI spelling and ctypes name for ``dtype``; unknown -> double.

    Exact spelling (no case folding), ``None``/empty -> :data:`DEFAULT_DTYPE`.
    """

    layout = _resolve_within(str(dtype or DEFAULT_DTYPE), _C_ABI_VOCABULARY)
    if layout is None:
        layout = DTYPES[DEFAULT_DTYPE]
    return (layout.c_type, layout.ctypes_name)


# --------------------------------------------------------------------------
# 2-4. ssa_c_backend: _numpy_dtype / _dtype_for_c_storage / solved_buffer_type
# --------------------------------------------------------------------------

def c_lane_dtype_for_storage(c_type: str) -> str:
    """Repository dtype of the C module lane's storage class spelled ``c_type``.

    ``uint8_t`` -> ``"bool"``, ``int32_t`` -> ``"int32"``, ``int64_t`` ->
    ``"int64"``, anything else -> ``"float64"``.
    """

    for name in C_LANE_STORAGE_CLASSES:
        if DTYPES[name].c_storage_type == c_type:
            return name
    return DEFAULT_DTYPE


def c_lane_numpy_dtype(dtype: object) -> str:
    """numpy dtype the C module lane allocates for ``dtype``.

    Casefolded; collapses to the four storage classes (bool / int32 / int64 /
    float64) by C storage spelling, so ``uint8`` lands in bool's one-byte
    class; every unrecognised name is ``"float64"``.
    """

    name = str(dtype or DEFAULT_DTYPE).casefold()
    layout = _resolve_within(name, _C_LANE_VOCABULARY)
    if layout is None:
        return DEFAULT_DTYPE
    storage_class = DTYPES[c_lane_dtype_for_storage(layout.c_storage_type)]
    return str(storage_class.numpy_dtype)


def c_lane_storage_for_llvm_type(llvm_type: object) -> str:
    """C storage spelling for an LLVM-lane spelling; unknown -> ``"double"``.

    ``i1`` -> ``uint8_t``, ``i32`` -> ``int32_t``, ``i64`` -> ``int64_t``,
    ``double`` -> ``double``, ``ptr`` -> ``void *``.
    """

    layout = by_llvm_type(llvm_type)
    return DTYPES[DEFAULT_DTYPE].c_storage_type if layout is None else layout.c_storage_type


# --------------------------------------------------------------------------
# 5-6. ssa_llvm_backend: _value_llvm_type / _LLVM_TYPE_BYTES
# --------------------------------------------------------------------------

def llvm_type_for_dtype(dtype: object) -> str:
    """LLVM spelling the LLVM lane emits for ``dtype``; unknown -> ``"double"``.

    Lowercased; ``None``/empty -> :data:`DEFAULT_DTYPE`.  The aggregate-output
    ``ptr`` case and the ``physical_dtype`` accounting read stay with the
    caller; this is only the name mapping.
    """

    name = str(dtype or DEFAULT_DTYPE).lower()
    layout = _resolve_within(name, _LLVM_LANE_VOCABULARY)
    return DTYPES[DEFAULT_DTYPE].llvm_type if layout is None else layout.llvm_type


def by_llvm_type(llvm_type: object) -> DTypeLayout | None:
    """The LLVM-lane layout spelled ``llvm_type`` (exact), or ``None``."""

    spelling = str(llvm_type)
    for name in LLVM_LANE_TYPES:
        if DTYPES[name].llvm_type == spelling:
            return DTYPES[name]
    return None


def llvm_type_bytes_table() -> dict[str, int]:
    """``{LLVM spelling: element bytes}`` for the types the LLVM lane emits."""

    return {DTYPES[name].llvm_type: DTYPES[name].byte_size for name in LLVM_LANE_TYPES}


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
    """Element bytes for ``dtype`` under the tensor-lowering vocabulary.

    Lowercased; ``None``/empty -> :data:`DEFAULT_DTYPE`; unrecognised -> ``default``.
    """

    name = str(dtype or DEFAULT_DTYPE).lower()
    layout = _resolve_within(name, _BYTE_SIZE_VOCABULARY)
    return default if layout is None else layout.byte_size


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


__all__ = [
    "DEFAULT_DTYPE",
    "DTypeLayout",
    "DTYPES",
    "ALIASES",
    "resolve",
    "layout_for",
    "by_llvm_type",
    "C_ABI_NAMES",
    "C_LANE_NAMES",
    "C_LANE_STORAGE_CLASSES",
    "LLVM_LANE_NAMES",
    "LLVM_LANE_TYPES",
    "FORTRAN_NAMES",
    "BYTE_SIZE_NAMES",
    "WASM_NAMES",
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
]
