"""Emit Fortran 2008 from Turing SSA.

Fortran is worth targeting for one specific reason, and it is not nostalgia:
**dummy arguments may not alias.**  The standard guarantees it, so a Fortran
compiler vectorizes array loops that an equivalent C or LLVM function cannot,
because there the compiler must assume ``a``, ``b`` and ``out`` might overlap.
The LLVM path in this repository recovers the same freedom only by asserting
``noalias`` explicitly; Fortran gets it from the language.

The emitter targets ``iso_c_binding`` with ``bind(C)``, so generated subroutines
share the calling convention the C and LLVM backends already use and can be
dropped into the same shell ABI.  Arrays are declared ``contiguous`` to state
stride-1 access, which is what allows unit-stride vector loads.

SSA maps onto Fortran cleanly:

* every ``SSAValue`` becomes a local scalar or array temporary;
* a ``BasicBlock`` becomes a labelled section;
* control edges become ``goto``, which Fortran still has and which mirrors
  ``br``/``condbr`` exactly;
* ``Phi`` is eliminated the standard way — assign into the phi's variable in
  each predecessor before branching, rather than inventing a parallel construct.

No Fortran compiler is required to *emit*.  Compilation is a separate, optional
step so the emitter stays usable on machines without gfortran, which is the
common case here.
"""

from __future__ import annotations

import math
import hashlib
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..transmogrifier.ssa import (
    BasicBlock,
    Function,
    Instr,
    IRModule,
    SSAValue,
    SSATensorTable,
)
from .string_table import NONE_TOKEN as _NONE_TOKEN, string_token as _string_token


_FORTRAN_IDENTIFIER_LIMIT = 63


def _fortran_symbol_table(names: Iterable[str]) -> dict[str, str]:
    """Return stable, collision-free Fortran identifiers for authored names.

    The authored name remains the bind(C) symbol and public API identity.
    Only the module-local procedure identifier is shortened, because Fortran
    2008 limits identifiers to 63 characters and compares them case
    insensitively.
    """

    result: dict[str, str] = {}
    occupied: set[str] = set()
    for authored in sorted(map(str, names)):
        cleaned = re.sub(r"[^A-Za-z0-9_]", "_", authored)
        if not cleaned or not cleaned[0].isalpha():
            cleaned = "f_" + cleaned
        candidate = cleaned
        if (
            len(candidate) > _FORTRAN_IDENTIFIER_LIMIT
            or candidate.casefold() in occupied
        ):
            digest = hashlib.sha256(authored.encode("utf-8")).hexdigest()[:12]
            suffix = "__" + digest
            candidate = cleaned[:_FORTRAN_IDENTIFIER_LIMIT - len(suffix)] + suffix
        if candidate.casefold() in occupied:
            raise FortranEmissionError(
                f"Fortran symbol shortening collided for {authored!r}"
            )
        occupied.add(candidate.casefold())
        result[authored] = candidate
    return result

# Fortran intrinsic (or expression template) for each SSA operation.  ``{0}``
# and ``{1}`` are the operand expressions.  Anything absent is reported as an
# unsupported operation rather than guessed at.
_BINARY: dict[str, str] = {
    "Add": "({0} + {1})",
    "Sub": "({0} - {1})",
    "Mul": "({0} * {1})",
    "Div": "({0} / {1})",
    "Pow": "({0} ** {1})",
    "Mod": "modulo({0}, {1})",
    "FloorDiv": "floor({0} / {1})",
    "Eq": "({0} == {1})",
    "Ne": "({0} /= {1})",
    "Lt": "({0} < {1})",
    "Le": "({0} <= {1})",
    "Gt": "({0} > {1})",
    "Ge": "({0} >= {1})",
    # Fortran 2008 bitwise comparison intrinsics compare the bit sequences as
    # unsigned integers without requiring a nonstandard unsigned kind.
    "ULt": "blt({0}, {1})",
    "ULe": "ble({0}, {1})",
    "UGt": "bgt({0}, {1})",
    "UGe": "bge({0}, {1})",
    "LAnd": "({0} .and. {1})",
    "LOr": "({0} .or. {1})",
    "And": "iand({0}, {1})",
    "Or": "ior({0}, {1})",
    "Xor": "ieor({0}, {1})",
    "Shl": "shiftl({0}, {1})",
    "Shr": "shiftr({0}, {1})",
    "AShr": "shifta({0}, {1})",
    "MatMul": "matmul({0}, {1})",
    "Max": "max({0}, {1})",
    "Min": "min({0}, {1})",
    # Canonical lowercase spellings used by FusedProgram steps.
    "add": "({0} + {1})",
    "sub": "({0} - {1})",
    "mul": "({0} * {1})",
    "truediv": "({0} / {1})",
    "pow": "({0} ** {1})",
    "maximum": "max({0}, {1})",
    "minimum": "min({0}, {1})",
    "matmul": "matmul({0}, {1})",
    "mod": "modulo({0}, {1})",
    "floordiv": "floor({0} / {1})",
    "less": "({0} < {1})",
    "less_equal": "({0} <= {1})",
    "greater": "({0} > {1})",
    "greater_equal": "({0} >= {1})",
    "equal": "({0} == {1})",
    "not_equal": "({0} /= {1})",
    "logical_and": "({0} .and. {1})",
    "logical_or": "({0} .or. {1})",
    "bitand": "iand({0}, {1})",
    "bitor": "ior({0}, {1})",
    "bitxor": "ieor({0}, {1})",
    "shl": "shiftl({0}, {1})",
    "shr": "shiftr({0}, {1})",
}

_UNARY: dict[str, str] = {
    "Neg": "(-{0})",
    "Abs": "abs({0})",
    "Not": "not({0})",
    "LNot": "(.not. {0})",
    # Numeric conversions. These arrive named after their LLVM opcodes,
    # because the precompile lowering shares an instruction vocabulary with
    # the LLVM backend, but each is an ordinary Fortran conversion intrinsic.
    # SExt/ZExt widen an integer, which in Fortran is just the integer kind
    # the target is already declared with.
    "SExt": "int({0}, c_int)",
    "ZExt": "int({0}, c_int)",
    "Trunc": "int({0}, c_int)",
    "FpToSi": "int({0}, c_int)",
    "FpToUi": "int({0}, c_int)",
    "SiToFp": "real({0}, c_double)",
    "UiToFp": "real({0}, c_double)",
    "FpExt": "real({0}, c_double)",
    # Narrowing keeps the double working type but passes the VALUE through
    # single precision; spelling it as a plain c_double conversion was an
    # identity that never narrowed anything.
    "FpTrunc": "real(real({0}, c_float), c_double)",
    "neg": "(-{0})",
    "abs": "abs({0})",
    "sqrt": "sqrt({0})",
    "Sqrt": "sqrt({0})",
    # The transcendentals. ``sqrt`` above is algebraic and was already here;
    # these are not producible by finitely many algebraic operations, which is
    # the only reason they were a separate omission rather than an oversight
    # of the same kind. Every one is a Fortran intrinsic -- the inverse
    # hyperbolics since Fortran 2008 -- so nothing here is a new capability,
    # only a registration that was missing.
    #
    # CONCERN, worth auditing rather than assuming: Fortran intrinsics are
    # ELEMENTAL, so ``exp(a)`` over a whole array is a single whole-array
    # operation the compiler is free to vectorise -- exactly the property
    # ``_REDUCTION`` below is written to exploit ("emitted as whole-array
    # intrinsics rather than explicit loops so the compiler picks the
    # schedule"). That only holds if the operand still *is* an array by the
    # time it reaches here. If the SSA arriving at this table has already been
    # scalarised into a per-element loop, these templates faithfully emit a
    # scalar call per element and the batch opportunity is gone -- not because
    # the registration is wrong, but because it was lost upstream. The
    # batch-capable library functions (``unary_double`` and friends) are the
    # ones that would preserve it. So: check whether these callees arrive with
    # array operands before concluding the emitted Fortran is as fast as it
    # can be.
    "exp": "exp({0})",
    "log": "log({0})",
    "sin": "sin({0})",
    "cos": "cos({0})",
    "tan": "tan({0})",
    "asin": "asin({0})",
    "acos": "acos({0})",
    "atan": "atan({0})",
    "sinh": "sinh({0})",
    "cosh": "cosh({0})",
    "tanh": "tanh({0})",
    "asinh": "asinh({0})",
    "acosh": "acosh({0})",
    "atanh": "atanh({0})",
    # FLOOR/CEILING/NINT return INTEGER in Fortran, where the numpy
    # equivalents return a float. Keeping the recorded program's type means
    # converting back, which also stops these from poisoning every intrinsic
    # downstream that then sees mixed INTEGER and REAL operands. Fortran
    # assignment converts on the way into an integer variable, so this is
    # safe even when the result is declared integer.
    "floor": "real(floor({0}), c_double)",
    "ceil": "real(ceiling({0}), c_double)",
    "round": "real(nint({0}), c_double)",
    "sign": "sign(1.0_c_double, {0})",
    "logical_not": "(.not. {0})",
    "trunc": "aint({0})",
    "copy": "{0}",
    "isnan": "ieee_is_nan({0})",
    "isfinite": "ieee_is_finite({0})",
    "isinf": "(.not. ieee_is_finite({0}) .and. .not. ieee_is_nan({0}))",
    # Every other backend in the torture matrix reports a comparison as a
    # plain 0.0/1.0 double, not a native boolean -- callers compare outputs
    # with assert_allclose, not a boolean buffer.  Match that at the
    # boundary rather than exporting a LOGICAL dummy no caller expects.
    "bool_to_float64": "merge(1.0_c_double, 0.0_c_double, {0})",
}

# Reductions collapse an array to a scalar; they are emitted as whole-array
# intrinsics rather than explicit loops so the compiler picks the schedule.
_REDUCTION: dict[str, str] = {
    "sum": "sum({0})",
    "prod": "product({0})",
    "max": "maxval({0})",
    "min": "minval({0})",
    "mean": "(sum({0}) / real(size({0}), c_double))",
    "all": "all({0})",
    "any": "any({0})",
}

# Binary operators whose result shape is meant to differ from their operands',
# so conforming the operands to the result would be wrong.
_SHAPE_CHANGING_BINARY = frozenset({"MatMul", "matmul"})

# Operators that take LOGICAL operands natively, so a boolean reaching them
# must be left alone rather than converted to a number.
_LOGICAL_BINARY = frozenset(
    {
        "LAnd", "LOr", "logical_and", "logical_or",
        "And", "Or", "Xor",
        "Eq", "Ne", "equal", "not_equal",
    }
)
# Unary operations that take a LOGICAL operand, so it must not be converted
# to a number on the way in.
_LOGICAL_UNARY = frozenset({"LNot", "Not", "logical_not", "bool_to_float64"})

# Of those, the ones that also *produce* LOGICAL. bool_to_float64 is the
# conversion itself: it consumes a mask and yields a number, so treating its
# result as logical would convert what was just converted.
_LOGICAL_RESULT_UNARY = frozenset(
    {"LNot", "Not", "logical_not", "isnan", "isfinite", "isinf"}
)

# Comparisons yield LOGICAL; everything else here yields a number.
_COMPARISON = frozenset(
    {
        "Eq", "Ne", "Lt", "Le", "Gt", "Ge",
        "ULt", "ULe", "UGt", "UGe",
        "equal", "not_equal", "less", "less_equal", "greater",
        "greater_equal",
    }
)

# Operations whose Fortran template requires REAL operands whatever the
# result is declared to be.
# Operations that rearrange values without computing new ones, so the type of
# the result is the type of what went in.
_SHAPE_ONLY = frozenset(
    {
        "slice", "reshape", "view", "broadcast_to", "permute", "pad", "stack",
        "concat", "cat", "concatenate", "gather", "scatter", "index_set",
        "repeat", "flatten",
        "squeeze", "unsqueeze", "expand", "clone", "copy", "detach",
        "swapaxes", "transpose",
    }
)

_REAL_OPERAND = frozenset(
    {"sign", "floor", "ceil", "round", "trunc", "sqrt", "exp", "log"}
)

_INTEGER_DTYPES = frozenset(
    {"int", "int8", "int16", "int32", "int64", "i32", "i64", "i1", "bool", "logical"}
)

_LOGICAL_DTYPES = frozenset({"i1", "bool", "logical"})

_DTYPE_KIND: dict[str, str] = {
    "double": "real(c_double)",
    "i64": "integer(c_int64_t)",
    "i32": "integer(c_int32_t)",
    "i1": "logical(c_bool)",
    "float64": "real(c_double)",
    "float32": "real(c_float)",
    "double": "real(c_double)",
    "float": "real(c_float)",
    "int64": "integer(c_int64_t)",
    "int32": "integer(c_int32_t)",
    "int": "integer(c_int32_t)",
    "bool": "logical(c_bool)",
    "opaque_ref": "integer(c_int64_t)",
}

DEFAULT_DTYPE = "float64"



def supported_tensor_operations() -> frozenset[str]:
    """Canonical tensor operations this emitter can spell directly.

    SSA also contains control and memory instructions, but those are not
    tensor-operator vocabulary.  This view is for target selection and is
    derived from the emitter tables themselves so it cannot become a second
    hand-maintained backend list.
    """

    from ..common.tensors.operator_catalog import (
        CANONICAL_ABSTRACT_TENSOR_OPERATORS,
    )
    from .ssa_numeric_operators import TENSOR_SSA_OPERATORS

    registered = (
        frozenset(_BINARY)
        | frozenset(_UNARY)
        | frozenset(_REDUCTION)
        | _SHAPE_ONLY
        | frozenset({
            "tensor_from_list", "where", "fill", "zeros", "zeros_like",
            "empty", "empty_like", "ones", "ones_like", "full", "full_like",
            "arange", "cumsum",
            "boolean_mask_select", "double", "float", "int", "long",
            "long_cast", "to_dtype", "cpu", "tolist",
        })
    )
    scalar = frozenset(_BINARY) | frozenset(_UNARY)
    registered |= frozenset(
        row.name
        for row in TENSOR_SSA_OPERATORS
        if row.is_direct and row.handler.value in scalar
    )
    return frozenset(registered & CANONICAL_ABSTRACT_TENSOR_OPERATORS)


class FortranEmissionError(ValueError):
    """Raised when an SSA construct has no honest Fortran spelling."""


@dataclass(frozen=True)
class FortranShortfall:
    """One SSA operation the emitter cannot express."""

    op: str
    block: str
    reason: str

    def format(self) -> str:
        return f"{self.op} [{self.block}]: {self.reason}"


@dataclass
class FortranSubroutine:
    """Generated Fortran for one SSA function."""

    name: str
    source: str
    shortfalls: tuple[FortranShortfall, ...] = ()
    # The extent parameters this subroutine declares, in argument order, so a
    # caller in the same module can pass exactly what it expects.
    extent_names: tuple[str, ...] = ()
    # Rankless SSA values used as address bases are C pointers.  Any ``extent``
    # operation over one requires an explicit companion length in the ABI.
    dynamic_array_extents: tuple[tuple[int, str], ...] = ()
    # Full runtime dimensions for every shape-dynamic array, including rank
    # two and higher buffers whose allocation cannot be described by one size.
    dynamic_array_dimensions: tuple[tuple[int, tuple[str, ...]], ...] = ()
    # Exact dummy arguments emitted without Fortran ``value``.
    reference_argument_ids: tuple[int, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.shortfalls


def _declaration(value: SSAValue, *, elemental: bool) -> str:
    kind = _DTYPE_KIND.get(value.dtype or DEFAULT_DTYPE)
    if kind is None:
        raise FortranEmissionError(f"no Fortran kind for dtype {value.dtype!r}")
    return kind if elemental else kind


def _name(value: SSAValue) -> str:
    return f"t{value.id}"


def _is_array(value: SSAValue) -> bool:
    return bool(value.shape)


def _element_count(shape: tuple[int, ...]) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total


def dimension_extents(values: Iterable[SSAValue]) -> dict[int, str]:
    """Map each distinct array dimension size across ``values`` to a Fortran
    extent parameter name.

    One name per distinct size, not one name per array: two arrays that
    happen to share a dimension size reuse the same parameter, and a matmul
    chain's differing row/inner/column counts each get their own. This is
    the single source of truth for extent naming -- the emitter uses it to
    declare arrays, and any caller building a shim that must pass matching
    extent arguments (fortran_jit_backend.py) uses the exact same function
    so the two never disagree on names or order.
    """

    sizes: dict[int, str] = {}
    for value in values:
        for size in value.shape:
            size = int(size)
            if size not in sizes:
                sizes[size] = f"extent_{size}"
    return sizes


def _broadcast(
    expression: str,
    shape: tuple[int, ...],
    result_shape: tuple[int, ...],
) -> str | None:
    """Expand ``expression`` from ``shape`` to ``result_shape``.

    numpy broadcasting is fully described by the two shapes: dimensions align
    from the right, and any operand dimension of extent one repeats to meet
    the result.  Fortran spells that repetition ``SPREAD``, which inserts a
    dimension rather than stretching one -- so each extent-one dimension is
    first indexed away (removing it) and then spread back at its own
    position with the result's extent.  Working left to right keeps every
    position valid as dimensions are reinserted.
    """

    rank = len(result_shape)
    if len(shape) > rank:
        return None
    # Align from the right, the way numpy does.
    aligned = (1,) * (rank - len(shape)) + tuple(int(size) for size in shape)
    expanding: list[int] = []
    for position, (size, target) in enumerate(zip(aligned, result_shape)):
        if size == int(target):
            continue
        if size != 1:
            return None
        expanding.append(position)
    if not expanding:
        return None

    # A one-element array is semantically a scalar broadcast. SUM is the
    # legal Fortran scalarisation for an arbitrary array expression; unlike
    # appending ``(1)`` it also works when the producer was inlined.
    if shape and _element_count(shape) == 1:
        result = f"sum({expression})"
        for position, target in enumerate(result_shape):
            result = f"spread({result}, dim={position + 1}, ncopies={int(target)})"
        return result

    # Only the operand's *own* extent-one dimensions need indexing away.
    # The dimensions alignment prepended do not exist on it, and SPREAD
    # introduces them; reshaping to the aligned rank first and then indexing
    # those positions back out would be a no-op -- and an illegal one, since
    # the result of RESHAPE is an expression, which Fortran will not
    # subscript.
    offset = rank - len(shape)
    subscripts = [
        "1" if (offset + position) in expanding else ":"
        for position in range(len(shape))
    ]
    result = (
        f"{expression}({', '.join(subscripts)})" if subscripts else expression
    )
    for position in expanding:
        result = (
            f"spread({result}, dim={position + 1}, "
            f"ncopies={int(result_shape[position])})"
        )
    return result


def _array_literal(
    values: Sequence[Any],
    shape: tuple[int, ...],
    *,
    dtype: str = DEFAULT_DTYPE,
) -> str:
    """A Fortran array constructor for an SSA array constant.

    Fortran expresses this natively: ``[a, b, c]`` is an array constructor,
    and ``reshape`` gives it a rank.  A constant whose elements are all equal
    needs neither -- Fortran broadcasts a scalar across a whole array on
    assignment, which is both the shortest source and the form a compiler
    folds best, so an all-``.false.`` mask of 124416 elements stays one token
    instead of 124416.
    """

    def flatten(items: Sequence[Any]) -> tuple[Any, ...]:
        flattened = []
        for item in items:
            if isinstance(item, (list, tuple)):
                flattened.extend(flatten(item))
            else:
                flattened.append(item)
        return tuple(flattened)

    elements = flatten(values)
    if not elements:
        kind = _DTYPE_KIND.get(dtype or DEFAULT_DTYPE)
        if kind is None:
            raise FortranEmissionError(f"no Fortran kind for dtype {dtype!r}")
        constructor = f"[{kind} ::]"
        if len(shape) <= 1:
            return constructor
        extents = ", ".join(str(int(size)) for size in shape)
        return f"reshape({constructor}, [{extents}])"
    if len(set(elements)) == 1:
        return _literal(elements[0])
    constructor = "[" + ", ".join(_literal(element) for element in elements) + "]"
    if len(shape) <= 1:
        return constructor
    extents = ", ".join(str(int(size)) for size in shape)
    return f"reshape({constructor}, [{extents}])"



def sin_table_declaration() -> str:
    """The shared baked table as a Fortran parameter array."""

    from .fused_program_wasm_backend import lut_for

    values, _achieved, _lower, _upper, _periodic = lut_for("sin")
    items = ", ".join(
        f"{value!r}".replace("e", "d") + "_c_double" for value in values
    )
    return (
        f"    real(c_double), parameter :: turing_sin_table(0:{len(values) - 1})"
        f" = [ {items} ]"
    )


def _table_sin_fortran(argument: str, shift: float) -> str:
    """sin(argument + shift) by interpolating the shared baked table."""

    from .fused_program_wasm_backend import lut_for
    from .bounded_constants import materialize_pi

    values, _achieved, lower, upper, periodic = lut_for("sin")
    intervals = len(values) - 1
    def literal(value: float) -> str:
        return f"{value!r}".replace("e", "d") + "_c_double"
    x = argument if shift == 0.0 else f"({argument} + {literal(shift)})"
    span = upper - lower
    placed = (
        f"({x} - {literal(span)} * floor(({x} - {literal(lower)})"
        f" * {literal(1.0 / span)}))"
        if periodic else x
    )
    t = (
        f"min(max(({placed} - {literal(lower)}) * "
        f"{literal(intervals / span)}, 0.0_c_double), {literal(float(intervals))})"
    )
    index = f"min(int({t}), {intervals - 1})"
    return (
        f"(turing_sin_table({index}) + ({t} - real({index}, c_double)) * "
        f"(turing_sin_table({index} + 1) - turing_sin_table({index})))"
    )


def _series_sin_fortran(argument: str, shift: float) -> str:
    """sin(argument + shift) as one Fortran expression, from the shared series.

    Fortran has no statement expression, so the reduced argument is spelled out
    at each use; the compiler removes the repetition. The coefficients and the
    constant come from bounded_constants, not from a second series stated here.
    """

    from .bounded_constants import sin_series_terms

    coefficients, pi, _bound = sin_series_terms()
    def literal(value: float) -> str:
        return f"{value!r}".replace("e", "d") + "_c_double"
    x = argument if shift == 0.0 else f"({argument} + {literal(shift)})"
    turns = f"nint({x} * {literal(1.0 / pi)})"
    r = f"({x} - {literal(pi)} * real({turns}, c_double))"
    horner = literal(coefficients[0])
    for coefficient in coefficients[1:]:
        horner = f"({horner} * ({r} * {r}) + {literal(coefficient)})"
    series = f"({horner} * {r})"
    return f"merge(-{series}, {series}, mod(abs({turns}), 2) == 1)"


def _literal(value: Any) -> str:
    # None and words are typed signed 64-bit identities.  Realise them at the
    # one point every literal passes through to become Fortran; never project
    # them through f64 bits.
    if value is None:
        return f"{_NONE_TOKEN}_c_int64_t"
    if isinstance(value, (str, bytes)):
        return f"{_string_token(value)}_c_int64_t"
    if isinstance(value, bool):
        return ".true._c_bool" if value else ".false._c_bool"
    if isinstance(value, int):
        # Libraries commonly expose protocol constants through an ``int``
        # subclass whose string form is the authored Python name (for example
        # ``re._constants.SUBPATTERN``).  The target knows the structural
        # integer value, not that Python-side spelling.
        return str(int(value))
    if isinstance(value, float):
        if math.isnan(value):
            return "ieee_value(0.0_c_double, ieee_quiet_nan)"
        if math.isinf(value):
            direction = "ieee_positive_inf" if value > 0 else "ieee_negative_inf"
            return f"ieee_value(0.0_c_double, {direction})"
        text = repr(float(value))
        if "e" in text or "E" in text:
            mantissa, _, exponent = text.partition("e")
            if "." not in mantissa:
                mantissa += ".0"
            return f"{mantissa}e{exponent}_c_double"
        if "." not in text:
            text += ".0"
        return f"{text}_c_double"
    raise FortranEmissionError(f"cannot express literal {value!r} in Fortran")


def _llvm_literal(value: str) -> Any:
    """Decode the scalar spelling retained by the LLVM-to-SSA importer."""

    dtype, separator, token = str(value).strip().partition(" ")
    if not separator:
        raise FortranEmissionError(f"malformed LLVM literal {value!r}")
    token = token.strip()
    if dtype.startswith("i"):
        if token in {"true", "false"}:
            return token == "true"
        return int(token, 0)
    if dtype in {"half", "float", "double"}:
        if token.casefold().startswith("0x"):
            bits = token[2:]
            if len(bits) == 8:
                return struct.unpack(">f", bytes.fromhex(bits))[0]
            if len(bits) == 16:
                return struct.unpack(">d", bytes.fromhex(bits))[0]
            raise FortranEmissionError(
                f"unsupported LLVM floating bit pattern {token!r}"
            )
        return float(token)
    raise FortranEmissionError(
        f"unsupported LLVM literal type {dtype!r} in {value!r}"
    )


class _FunctionEmitter:
    """Translate one SSA :class:`Function` into a Fortran subroutine."""

    def __init__(
        self,
        function: Function,
        *,
        dtype: str = DEFAULT_DTYPE,
        outputs: Sequence[SSAValue] = (),
        callee_extents: Mapping[str, Sequence[str]] | None = None,
        callee_arity: Mapping[str, int] | None = None,
        callee_output_count: Mapping[str, int] | None = None,
        callee_outputs: Mapping[str, Sequence[SSAValue]] | None = None,
        callee_arguments: Mapping[str, Sequence[SSAValue]] | None = None,
        callee_array_arguments: Mapping[str, Sequence[int]] | None = None,
        callee_inout_pairs: Mapping[
            str, Sequence[tuple[int, int]]
        ] | None = None,
        trig_solver: str = "lut",
        array_base_ids: Sequence[int] = (),
        mutated_base_ids: Sequence[int] = (),
        dynamic_array_ranks: Mapping[int, int] | None = None,
        value_dtypes: Mapping[int, str] | None = None,
        tensor_table: SSATensorTable | None = None,
        native_symbol: str | None = None,
        callee_native_symbols: Mapping[str, str] | None = None,
        extent_namespace: str = "",
    ):
        if str(trig_solver) not in {"lut", "continuous"}:
            raise ValueError(
                f"unknown trig solver {trig_solver!r}; expected 'lut' or "
                "'continuous'"
            )
        self.trig_solver = str(trig_solver)
        self.uses_sin_table = False
        self.function = function
        self.native_symbol = str(native_symbol or function.name)
        self.callee_native_symbols = dict(callee_native_symbols or {})
        self.extent_namespace = (
            str(extent_namespace) + "_" if extent_namespace else ""
        )
        self.dtype = dtype
        self.outputs = tuple(outputs)
        self.tensor_table = tensor_table or SSATensorTable()
        # Value ids used as an address base anywhere in the module (dynamic
        # arrays), and those a store mutates -- shared across a method's
        # functions by id, so every function that names one declares it as the
        # assumed-size array it is, with matching intent, and caller and callee
        # agree on rank without positional signature matching.
        self.array_base_ids = {int(value_id) for value_id in array_base_ids}
        self.array_base_ids.update(
            int(argument.id)
            for argument in self.function.args
            if (
                str((argument.accounting or {}).get("program_abi_storage"))
                == "span"
                or int((argument.accounting or {}).get(
                    "program_abi_rank", 0
                )) > 0
                or int((argument.accounting or {}).get(
                    "ssa_call_rank", 0
                )) > 0
            )
        )
        self.mutated_base_ids = {int(value_id) for value_id in mutated_base_ids}
        self.callee_extents = dict(callee_extents or {})
        self.callee_arity = dict(callee_arity or {})
        self.callee_output_count = dict(callee_output_count or {})
        self.callee_outputs = {
            str(name): tuple(values)
            for name, values in (callee_outputs or {}).items()
        }
        self.callee_arguments = {
            str(name): tuple(values)
            for name, values in (callee_arguments or {}).items()
        }
        self.callee_array_arguments = {
            str(name): frozenset(int(position) for position in positions)
            for name, positions in (callee_array_arguments or {}).items()
        }
        self.callee_inout_pairs = {
            str(name): tuple(pairs)
            for name, pairs in (callee_inout_pairs or {}).items()
        }
        self._value_types: dict[int, SSAValue] = {}
        self._infer_control_value_types()
        for value_id, value_dtype in (value_dtypes or {}).items():
            current = self._value_types.get(int(value_id))
            if current is None:
                continue
            self._value_types[int(value_id)] = SSAValue(
                current.id,
                str(value_dtype),
                tuple(current.shape),
                current.device,
                dict(current.accounting or {}),
            )
        self.outputs = tuple(self._typed(value) for value in self.outputs)
        self.dynamic_array_extents = {
            int(instr.args[0].id): (
                f"extent_dynamic_{self.extent_namespace}"
                f"{int(instr.args[0].id)}"
            )
            for block in self.function.blocks.values()
            for instr in block.instrs
            if (
                (instr.attributes.get("tensor_operation") or instr.op)
                == "extent"
                and instr.args
                and int(instr.args[0].id) in self.array_base_ids
                and not self._typed(instr.args[0]).shape
            )
        }
        # A contracted Python span can deliberately remain shape-dynamic while
        # still carrying an exact rank.  Fortran assumed-size dummies preserve
        # that ABI without a descriptor, but every dimension except the last
        # needs an explicit runtime extent: ``a(n, *)`` for rank two, etc.
        indexed_ranks: dict[int, int] = {}
        for block in self.function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op in {"GetElementPtr", "getelementptr"}
                    and instruction.args
                ):
                    base_id = int(instruction.args[0].id)
                    indexed_ranks[base_id] = max(
                        indexed_ranks.get(base_id, 0),
                        len(instruction.args) - 1,
                    )
        self.dynamic_array_ranks: dict[int, int] = {
            int(value_id): int(rank)
            for value_id, rank in dict(dynamic_array_ranks or {}).items()
            if int(rank) > 0
        }
        for argument in self.function.args:
            argument_id = int(argument.id)
            if argument_id not in self.array_base_ids:
                continue
            self.dynamic_array_ranks[argument_id] = max(
                1,
                self.dynamic_array_ranks.get(argument_id, 0),
                indexed_ranks.get(argument_id, 0),
                len(tuple(self._typed(argument).shape or ())),
                int((argument.accounting or {}).get("program_abi_rank", 0)),
                int((argument.accounting or {}).get("ssa_call_rank", 0)),
            )
        self.dynamic_array_leading_extents = {
            value_id: (
                (self.dynamic_array_extents[value_id],)
                if rank == 1 and value_id in self.dynamic_array_extents
                else tuple(
                    f"extent_dynamic_{self.extent_namespace}{value_id}_{axis}"
                    for axis in range(1, rank + 1)
                )
            )
            for value_id, rank in self.dynamic_array_ranks.items()
            if (
                rank > 0
                and not tuple(
                    self._value_types[value_id].shape
                    if value_id in self._value_types
                    else ()
                )
            )
        }
        self.shortfalls: list[FortranShortfall] = []
        for descriptor in self.tensor_table.tensors.values():
            if descriptor.metadata_state == "unresolved":
                self.shortfalls.append(FortranShortfall(
                    "ssa_tensor_table",
                    "entry",
                    "tensor metadata is unresolved before target emission: "
                    f"tensor_id={descriptor.tensor_id}, "
                    f"data_value_id={descriptor.data_value_id}",
                ))
            elif descriptor.metadata_state == "dynamic":
                self.array_base_ids.add(int(descriptor.data_value_id))
                self.array_base_ids.add(int(descriptor.shape_value_id))
                if descriptor.strides_value_id is not None:
                    self.array_base_ids.add(int(descriptor.strides_value_id))
        self._branch_targets: set[str] = set()
        # Single-use temporaries are substituted into their consumer so one SSA
        # chain becomes one Fortran array expression.  Emitting a statement per
        # step would materialise an array temporary per step, turning a fused
        # traversal into N passes over memory — which is precisely the cost the
        # whole-array form exists to avoid.
        self._inlined: dict[int, str] = {}
        self._use_sites: dict[int, list[tuple[str, int]]] = {}
        self._locals: dict[int, SSAValue] = {}
        self._phi_targets: dict[str, list[tuple[str, SSAValue, SSAValue]]] = {}
        self._loop_variables: list[str] = []
        # Values emitted as part of a multi-instruction group (a region call,
        # an indexed store) rather than one at a time.
        self._consumed: set[int] = set()
        self._address_producers: dict[int, tuple[SSAValue, tuple[SSAValue, ...]]] = {}
        self._mutated_arrays: set[int] = set()
        self._producers: dict[int, Instr] = {}
        # Collection value id -> the induction that indexes it.
        self._collections: dict[int, SSAValue] = {}

    def _prefer_value_type(self, value: SSAValue) -> bool:
        """Retain the richest known type for one SSA identity."""

        current = self._value_types.get(int(value.id))
        if current is None or (
            not tuple(current.shape) and tuple(value.shape)
        ):
            self._value_types[int(value.id)] = value
            return True
        return False

    def _typed(self, value: SSAValue) -> SSAValue:
        return self._value_types.get(int(value.id), value)

    def _infer_control_value_types(self) -> None:
        """Propagate resident arena rank through aggregate calls and Phis."""

        for value in (*self.function.args, *self.outputs):
            self._prefer_value_type(value)
        for block in self.function.blocks.values():
            for instr in block.instrs:
                for value in instr.args:
                    self._prefer_value_type(value)
                if instr.res is not None:
                    self._prefer_value_type(instr.res)

        # An indexed load has the element dtype of its address base even when
        # the structural linker left the load occurrence itself untyped.
        for block in self.function.blocks.values():
            address_bases: dict[int, SSAValue] = {}
            for instr in block.instrs:
                if (
                    instr.op in {"GetElementPtr", "getelementptr"}
                    and instr.res is not None
                    and instr.args
                    and instr.attributes.get("aggregate_index") is None
                ):
                    address_bases[int(instr.res.id)] = self._typed(instr.args[0])
                elif (
                    instr.op in {"Load", "load"}
                    and instr.res is not None
                    and instr.args
                    and int(instr.args[0].id) in address_bases
                ):
                    base = address_bases[int(instr.args[0].id)]
                    current = self._typed(instr.res)
                    self._value_types[int(instr.res.id)] = SSAValue(
                        instr.res.id,
                        base.dtype or current.dtype,
                        tuple(current.shape),
                        base.device or current.device,
                        {
                            **dict(current.accounting or {}),
                            **dict(base.accounting or {}),
                        },
                    )

        # A region result paired with an input denotes that input's arena.
        for block in self.function.blocks.values():
            calls: dict[int, tuple[str, Sequence[SSAValue]]] = {}
            addresses: dict[int, tuple[str, Sequence[SSAValue], int]] = {}
            for instr in block.instrs:
                if (
                    instr.op in ("Call", "call")
                    and instr.res is not None
                    and instr.attributes.get("result_convention")
                    == "ssa.aggregate"
                ):
                    calls[instr.res.id] = (
                        str(instr.attributes.get("callee") or ""),
                        instr.args,
                    )
                elif (
                    instr.op == "GetElementPtr"
                    and instr.res is not None
                    and instr.args
                    and instr.args[0].id in calls
                ):
                    position = instr.attributes.get("aggregate_index")
                    if position is not None:
                        callee, arguments = calls[instr.args[0].id]
                        addresses[instr.res.id] = (
                            callee,
                            arguments,
                            int(position),
                        )
                elif (
                    instr.op in ("Load", "load")
                    and instr.res is not None
                    and instr.args
                    and instr.args[0].id in addresses
                ):
                    callee, arguments, output_index = addresses[
                        instr.args[0].id
                    ]
                    contract_outputs = self.callee_outputs.get(callee, ())
                    if output_index < len(contract_outputs):
                        contract = contract_outputs[output_index]
                        current = self._typed(instr.res)
                        self._value_types[instr.res.id] = SSAValue(
                            instr.res.id,
                            contract.dtype or current.dtype,
                            tuple(contract.shape or current.shape),
                            contract.device or current.device,
                            {
                                **dict(current.accounting or {}),
                                **dict(contract.accounting or {}),
                            },
                        )
                    for input_index, paired_output in self.callee_inout_pairs.get(
                        callee, ()
                    ):
                        if (
                            int(paired_output) == output_index
                            and int(input_index) < len(arguments)
                        ):
                            source = self._typed(arguments[int(input_index)])
                            self._value_types[instr.res.id] = SSAValue(
                                instr.res.id,
                                source.dtype,
                                tuple(source.shape),
                                source.device,
                                dict(source.accounting),
                            )

        # Loop-carried values can form Phi chains, so settle to a fixed point.
        changed = True
        while changed:
            changed = False
            for block in self.function.blocks.values():
                for instr in block.instrs:
                    if instr.op not in ("Phi", "phi") or instr.res is None:
                        continue
                    candidates = [self._typed(value) for value in instr.args]
                    candidates.extend(
                        self._typed(value)
                        for _, value in (instr.attributes.get("incoming") or ())
                    )
                    shaped = next(
                        (value for value in candidates if tuple(value.shape)),
                        None,
                    )
                    if shaped is None or tuple(self._typed(instr.res).shape):
                        continue
                    self._value_types[instr.res.id] = SSAValue(
                        instr.res.id,
                        shaped.dtype,
                        tuple(shaped.shape),
                        shaped.device,
                        dict(shaped.accounting),
                    )
                    changed = True

    # -- expression construction ------------------------------------------
    def _operand(self, value: SSAValue) -> str:
        inlined = self._inlined.get(value.id)
        return inlined if inlined is not None else _name(value)

    def _call_operand(self, value: SSAValue, *, array_expected: bool) -> str:
        """Render one call operand, folding a repository address to a section.

        SSA represents a selected nested row as ``GetElementPtr(child, off)``.
        Fortran has no first-class pointer arithmetic, but an array element or
        section is a valid actual argument for an assumed-size dummy.  Folding
        the address here preserves the same caller-owned storage ABI without a
        temporary copy or runtime object.
        """

        if array_expected:
            producer = self._address_producers.get(int(value.id))
            if producer is not None:
                collection, positions = producer
                self._consumed.add(int(value.id))
                subscripts = ", ".join(
                    (
                        f"{self._operand(position)} + 1"
                        if str(self._typed(position).dtype) in _INTEGER_DTYPES
                        else f"int({self._operand(position)}) + 1"
                    )
                    for position in positions
                )
                return f"{self._operand(collection)}({subscripts})"
        return self._operand(value)

    def _collect_use_sites(self) -> None:
        for block in self.function.blocks.values():
            for index, instr in enumerate(block.instrs):
                for argument in instr.args:
                    self._use_sites.setdefault(argument.id, []).append(
                        (block.name, index)
                    )

    def _may_inline(self, instr: Instr, block: BasicBlock) -> bool:
        """Whether this result can be folded into its consumer.

        Only a value used exactly once, inside the same block, and not visible
        outside the subroutine may be substituted.  Crossing a block boundary
        would move the computation across control flow.
        """

        result = instr.res
        # A shaped constant must remain an array designator.  Scalar
        # assignment can initialise it compactly, whereas substituting the
        # literal into a pointer/assumed-size call argument loses its rank.
        if instr.op in ("Const", "const") and tuple(result.shape):
            return False
        if result.id in self._bound_ids:
            return False
        if instr.op in ("Phi", "phi"):
            return False
        uses = self._use_sites.get(result.id, ())
        if len(uses) != 1:
            return False
        if uses[0][0] != block.name:
            return False
        return not self._is_subscripted_by(result, uses[0])

    def _is_subscripted_by(
        self, value: SSAValue, use: tuple[str, int]
    ) -> bool:
        """Whether this value's one consumer will subscript it.

        Fortran subscripts an array designator, never an arbitrary
        expression: ``(a + b)(1, :)`` is a syntax error.  So a value that its
        consumer has to index -- to take a section, or to conform a shape by
        indexing an extent-one dimension away -- must stay a named temporary
        even though it is used once.  Inlining it would be a folding
        optimisation that produces source no compiler accepts.
        """

        block_name, position = use
        block = self.function.blocks.get(block_name)
        if block is None or position >= len(block.instrs):
            return False
        consumer = block.instrs[position]
        operation = (
            consumer.attributes.get("tensor_operation") or consumer.op
        )
        if operation in (
            "slice", "gather", "scatter", "index_set", "pad", "cumsum",
            "repeat",
        ):
            return True
        if consumer.res is None:
            return False
        if (
            operation in _SHAPE_CHANGING_BINARY
            and len(consumer.res.shape) > 2
        ):
            # A batched matmul indexes each operand per batch element.
            return True
        # An elementwise operand of a different shape is conformed by
        # indexing (a broadcast) rather than by an intrinsic call.
        if operation in _BINARY and operation not in _SHAPE_CHANGING_BINARY:
            shape = tuple(value.shape)
            result_shape = tuple(consumer.res.shape)
            if shape and shape != result_shape:
                return True
        return False

    def _structural(
        self, instr: Instr, args: list[str], op: str | None = None
    ) -> str | None:
        """Shape-and-layout ops, as native Fortran array expressions.

        These are not elementwise, so they are absent from the intrinsic
        tables; Fortran still says all of them directly -- array sections,
        array constructors, ``reshape``. Anything whose Fortran form would
        depend on a memory layout this emitter cannot confirm returns None and
        is reported as a shortfall rather than guessed at.
        """

        # The precompile lowering wraps ops in Handler.Call and records the
        # canonical name under "tensor_operation"; the tape JIT builds the
        # instruction with that name as its opcode directly. Both name the
        # same operation, so both resolve here rather than in two emitters.
        operation = instr.attributes.get("tensor_operation") or op or instr.op
        if instr.res is None:
            return None
        attributes = instr.attributes
        shape = instr.res.shape
        rank = len(shape)

        if operation in (
            "cpu", "detach", "tolist", "clone", "copy", "contiguous",
        ) and len(args) == 1:
            # These operations cross Python/backend representation boundaries.
            # Inside an already-resident Fortran program the value is in its
            # target representation, so the numerical value is unchanged.
            return args[0]

        if (
            operation
            in {"double", "float", "int", "long", "long_cast", "to_dtype"}
            and len(args) == 1
        ):
            requested = str(
                attributes.get("dtype")
                or attributes.get("target_dtype")
                or operation
            ).lower()
            if requested in {"double", "float", "float32", "float64"}:
                kind = "c_float" if requested in {"float", "float32"} else "c_double"
                return f"real({args[0]}, {kind})"
            if requested in {"int", "int32", "long", "long_cast", "int64"}:
                kind = (
                    "c_int64_t"
                    if requested in {"long", "long_cast", "int64"}
                    else "c_int32_t"
                )
                return f"int({args[0]}, {kind})"
            return None

        if operation == "boolean_mask_select" and len(args) == 2:
            return f"pack({args[0]}, {args[1]})"

        reduction_axis = attributes.get("dim", attributes.get("axis"))
        if operation == "extent" and len(args) == 1:
            # Control lowering can refer to a resident collection through a
            # scalar-shaped SSA spelling even when the same value identity
            # has a ranked definition at the function boundary or across a
            # region call.  Rank inference has already unified those
            # identities; consult it here instead of the lossy occurrence.
            source_rank = len(self._typed(instr.args[0]).shape)
            dim = int(attributes.get("dim", 0))
            dynamic_extent = self.dynamic_array_extents.get(
                int(instr.args[0].id)
            )
            if attributes.get("extent_kind") == "numel":
                if source_rank > 0:
                    return f"size({args[0]})"
                if dynamic_extent is not None:
                    return dynamic_extent
                return None
            if source_rank == 0 and dim == 0 and dynamic_extent is not None:
                return dynamic_extent
            if source_rank == 0 or not (-source_rank <= dim < source_rank):
                return None
            return f"size({args[0]}, {(dim % source_rank) + 1})"
        if (
            operation in _REDUCTION
            and reduction_axis is not None
            and len(args) == 1
        ):
            # Fortran reduces along one dimension natively. Arrays are
            # declared in SSA dimension order, so the axis needs no
            # translation; sum(a, dim=k) drops that dimension, and keepdim
            # asks for it back as an extent of one.
            source_rank = len(instr.args[0].shape)
            if isinstance(reduction_axis, (tuple, list)):
                if len(reduction_axis) != 1:
                    return None
                reduction_axis = reduction_axis[0]
            axis = (int(reduction_axis) % source_rank) + 1
            reduced = _REDUCTION[operation].format(
                f"{args[0]}, dim={axis}"
            )
            if len(shape) == source_rank:
                if source_rank == 1:
                    return f"[{reduced}]"
                extents = ", ".join(str(int(size)) for size in shape)
                return f"reshape({reduced}, [{extents}])"
            return reduced

        if (
            operation in _REDUCTION
            and len(args) == 1
            and len(instr.args[0].shape) == 0
        ):
            # Reducing a scalar is the identity.  NumPy permits it (and may
            # retain a singleton result shape); Fortran's reduction
            # intrinsics require an array argument, while scalar assignment
            # already broadcasts into a singleton destination.
            return args[0]

        if operation in (
            "reshape", "view", "flatten", "squeeze", "unsqueeze",
        ) and len(args) == 1:
            # A reshape is defined by row-major element order, but Fortran
            # traverses column-major, so both ends need stating.
            #
            # Destination: ORDER=[n..1] makes the last dimension vary
            # fastest, which is the row-major fill.
            #
            # Source: a rank>1 operand's Fortran element order is
            # column-major, which is *not* the order the reshape reads it
            # in. Reversing its dimensions first yields an array whose
            # column-major traversal is the operand's row-major traversal.
            # For a rank-1 operand the two coincide and this collapses away.
            source = args[0]
            source_shape = tuple(instr.args[0].shape)
            source_rank = len(source_shape)
            if rank == 0:
                if source_rank == 0:
                    return source
                indices = ", ".join("1" for _ in source_shape)
                return f"{source}({indices})"
            if source_rank == 0:
                source = f"[{source}]"
            if source_rank > 1:
                reversed_extents = ", ".join(
                    str(int(size)) for size in reversed(source_shape)
                )
                source_order = ", ".join(
                    str(value) for value in range(source_rank, 0, -1)
                )
                source = (
                    f"reshape({source}, [{reversed_extents}], "
                    f"order=[{source_order}])"
                )
            if rank <= 1:
                return f"reshape({source}, [{_element_count(shape)}])"
            extents = ", ".join(str(int(size)) for size in shape)
            order = ", ".join(str(value) for value in range(rank, 0, -1))
            return f"reshape({source}, [{extents}], order=[{order}])"

        if operation in ("broadcast_to", "expand") and len(args) == 1:
            source_shape = tuple(instr.args[0].shape)
            target_shape = tuple(shape)
            if source_shape == target_shape:
                return args[0]
            return _broadcast(args[0], source_shape, target_shape)

        if operation == "repeat" and len(args) == 1:
            source_shape = tuple(map(int, instr.args[0].shape))
            target_shape = tuple(map(int, shape))
            if len(source_shape) != len(target_shape):
                return None
            raw_repeats = attributes.get("repeats", 1)
            dim = attributes.get("dim")
            if dim is not None and not isinstance(raw_repeats, (tuple, list)):
                repeat_counts = [1] * rank
                repeat_counts[int(dim) % rank] = int(raw_repeats)
            else:
                repeat_counts = list(map(int, (
                    raw_repeats
                    if isinstance(raw_repeats, (tuple, list))
                    else (raw_repeats,)
                )))
                if len(repeat_counts) != rank:
                    return None
            if any(count <= 0 for count in repeat_counts):
                return None
            if any(
                target != source * count
                for source, target, count in zip(
                    source_shape, target_shape, repeat_counts
                )
            ):
                return None
            subscripts = []
            for source_extent, target_extent, count in zip(
                source_shape, target_shape, repeat_counts
            ):
                if count == 1:
                    subscripts.append(":")
                    continue
                index = self._loop_variable()
                subscripts.append(
                    f"[(mod({index} - 1, {source_extent}) + 1, "
                    f"{index} = 1, {target_extent})]"
                )
            return f"{args[0]}({', '.join(subscripts)})"

        if operation in {
            "Fill", "fill", "zeros", "zeros_like", "empty", "empty_like",
            "ones", "ones_like", "full", "full_like",
        }:
            # Handler.Fill is the compiler's backend-neutral span operation.
            # The result storage is already declared (or supplied by the
            # caller when it is public), so initialization is one scalar
            # whole-array assignment, never an element constructor.
            default = 1 if operation in {"ones", "ones_like"} else 0
            fill = attributes.get("fill_value", attributes.get("value", default))
            if fill is None:
                return None
            if instr.res.dtype in ("float64", "float32", "double"):
                fill = float(fill)
            return _literal(fill)

        if operation == "arange":
            start = int(attributes.get("start", 0))
            step = int(attributes.get("step", 1))
            end = int(attributes["end"])
            index = self._loop_variable()
            # An implied-do array constructor: the Fortran form of arange,
            # with no materialised element list regardless of length.
            return (
                f"[({index}, {index} = {start}, {end - 1}, {step})]"
            )

        if operation == "slice" and attributes.get("slice_kind") == "index_select":
            # A gather. Fortran applies a vector subscript along one
            # dimension directly, so no loop and no temporary are needed.
            if len(args) != 2:
                return None
            source = instr.args[0]
            source_rank = len(source.shape)
            dim = int(attributes.get("dim", 0)) % source_rank
            subscripts = [":"] * source_rank
            # SSA indices are 0-based; Fortran subscripts start at 1.
            subscripts[dim] = f"{args[1]} + 1"
            return f"{args[0]}({', '.join(subscripts)})"

        if operation == "slice":
            if attributes.get("slice_kind") != "axis" or len(args) != 1:
                return None
            source = instr.args[0]
            source_rank = len(source.shape)
            dim = int(attributes.get("dim", 0))
            start = int(attributes.get("start", 0))
            step = int(attributes.get("step", 1))
            count = int(attributes.get("count", 1))
            # Arrays are declared in SSA dimension order (see dims() in
            # emit()), so Fortran subscript k+1 is SSA dim k.
            axis = (dim % source_rank) + 1
            subscripts = [":"] * source_rank
            if count == 1 and rank == source_rank - 1:
                # A single index drops the rank, matching the result shape.
                subscripts[axis - 1] = str(start + 1)
            else:
                stop = start + count * step
                subscripts[axis - 1] = (
                    f"{start + 1}:{stop}:{step}" if step != 1
                    else f"{start + 1}:{stop}"
                )
            return f"{args[0]}({', '.join(subscripts)})"

        if operation == "permute":
            dims = list(attributes.get("dims") or ())
            if len(dims) != rank or sorted(dims) != list(range(rank)):
                return None
            # RESHAPE fills the result so that its dimension ORDER(k) takes
            # the source's k-th dimension; for a permutation that makes ORDER
            # the inverse of the requested dims. Arrays are declared in SSA
            # dimension order, so this is a plain 1-based inverse with no
            # reversal.
            inverse = [0] * rank
            for position, source_dim in enumerate(dims):
                inverse[source_dim] = position + 1
            order = ", ".join(str(value) for value in inverse)
            extents = ", ".join(str(int(size)) for size in shape)
            return f"reshape({args[0]}, [{extents}], order=[{order}])"

        if operation in ("transpose", "swapaxes") and len(args) == 1:
            source_rank = len(self._typed(instr.args[0]).shape)
            dim0 = int(attributes.get("dim0", attributes.get("axis1", 0)))
            dim1 = int(attributes.get("dim1", attributes.get("axis2", 1)))
            if source_rank == 2 and {dim0 % 2, dim1 % 2} == {0, 1}:
                return f"transpose({args[0]})"

        return None

    def _is_logical(self, value: SSAValue) -> bool:
        """Whether this value's *emitted* expression is LOGICAL.

        The declared dtype alone is not enough. A value can be recorded as
        ``bool`` and still be produced by arithmetic -- numpy makes no
        distinction, so ``mask + 0`` keeps the boolean dtype while plainly
        being a number. Emitting MERGE against that gives a non-LOGICAL mask,
        which Fortran rejects. An inlined value's real type is decided by the
        instruction that produced it.
        """

        value = self._typed(value)
        # LLVM's i1 is intrinsically a logical value.  Check the recorded
        # scalar kind before examining an inlined producer: an imported
        # ``fcmp`` is represented as an ordinary Call and would otherwise be
        # misclassified merely because its expression was inlined.
        if str(getattr(value, "dtype", "") or "") in _LOGICAL_DTYPES:
            return True
        producer = self._producers.get(value.id)
        if producer is not None and value.id in self._inlined:
            operation = (
                producer.attributes.get("tensor_operation") or producer.op
            )
            if producer.op in ("Const", "const"):
                return self._instruction_is_logical(producer)
            if operation in _SHAPE_ONLY and producer.args:
                return self._is_logical(producer.args[0])
            if (
                operation in _COMPARISON
                or operation in _LOGICAL_BINARY
                or operation in _LOGICAL_RESULT_UNARY
            ):
                return True
            # Produced by arithmetic: a number, whatever the dtype claims.
            return False
        if str(getattr(value, "dtype", "") or "") not in _LOGICAL_DTYPES:
            return False
        if producer is None or value.id not in self._inlined:
            # A declared LOGICAL variable, or an argument: the declaration is
            # the truth.
            return True
        operation = (
            producer.attributes.get("tensor_operation") or producer.op
        )
        if operation in _SHAPE_ONLY and producer.args:
            # Rearranging a mask does not stop it being a mask.
            return self._is_logical(producer.args[0])
        return (
            operation in _COMPARISON
            or operation in _LOGICAL_BINARY
            or operation in _LOGICAL_RESULT_UNARY
        )

    def _batched_matmul(self, instr: Instr, operation: str) -> list[str] | None:
        """A matmul over leading batch dimensions, as nested loops.

        Fortran's MATMUL takes rank-1 and rank-2 arguments only, while the
        recorded program follows numpy: the last two dimensions are the
        matrix and everything before them is a batch that broadcasts.  So the
        batch has to become real loops around a rank-2 MATMUL -- there is no
        single-expression form, and emitting one anyway is what produced
        "'matrix_b' argument of 'matmul' must be of rank 1".
        """

        if operation not in ("matmul", "MatMul") or len(instr.args) != 2:
            return None
        result_shape = tuple(instr.res.shape)
        batch_rank = len(result_shape) - 2
        if batch_rank <= 0:
            # Rank 2 or less is exactly what MATMUL already accepts.
            return None
        if any(len(argument.shape) < 2 for argument in instr.args):
            return None

        indices = [self._loop_variable() for _ in range(batch_rank)]

        def batch_subscripts(value: SSAValue) -> str:
            """This operand's own batch subscripts, aligned from the right.

            An operand carrying fewer dimensions is broadcast across the
            batch, and one whose batch extent is 1 repeats -- the same rule
            the elementwise conforming path follows.
            """

            own_batch = len(value.shape) - 2
            if own_batch <= 0:
                return ""
            offset = batch_rank - own_batch
            parts = []
            for position in range(own_batch):
                extent = int(value.shape[position])
                parts.append(
                    "1" if extent == 1 else indices[offset + position]
                )
            return ", ".join(parts) + ", "

        left, right = instr.args
        target = _name(instr.res)
        statements: list[str] = []
        for depth, index in enumerate(indices):
            pad = "    " + "  " * depth
            statements.append(
                f"{pad}do {index} = 1, {int(result_shape[depth])}"
            )
        pad = "    " + "  " * batch_rank
        statements.append(
            f"{pad}{target}({', '.join(indices)}, :, :) = matmul("
            f"{self._operand(left)}({batch_subscripts(left)}:, :), "
            f"{self._operand(right)}({batch_subscripts(right)}:, :))"
        )
        for depth in range(batch_rank - 1, -1, -1):
            statements.append("    " + "  " * depth + "end do")
        return statements

    def _instruction_is_logical(self, instr: Instr) -> bool:
        """Whether this instruction's expression evaluates to LOGICAL.

        Asked of the instruction rather than its result, because the result
        is about to be assigned to a declared variable and so is never
        inlined -- the value-level test would look at the declaration and
        answer with what we are trying to check.
        """

        operation = instr.attributes.get("tensor_operation") or instr.op
        if instr.res is not None and str(instr.res.dtype or "") in _LOGICAL_DTYPES:
            return True
        if operation in _SHAPE_ONLY and instr.args:
            return self._is_logical(instr.args[0])
        # Tested against instr.op, not `operation`: a constant carries
        # tensor_operation="tensor_from_list", which shadows the Const op
        # name that `operation` would otherwise report.
        if instr.op in ("Const", "const"):
            # A boolean constant emits a LOGICAL literal (_literal writes
            # .true._c_bool), so it is already LOGICAL whatever its recorded
            # dtype says. Missing this wrapped the literal as
            # ((.true._c_bool) /= 0), which gfortran rejects for comparing
            # LOGICAL against INTEGER.
            for key in ("constant", "value"):
                candidate = instr.attributes.get(key)
                if isinstance(candidate, bool):
                    return True
            # An array constant keeps its elements under "values" and leaves
            # the scalar keys None, so a boolean mask constant is only
            # visible here.
            values = instr.attributes.get("values")
            if isinstance(values, bool):
                return True
            if isinstance(values, (list, tuple)) and values and all(
                isinstance(element, bool) for element in values
            ):
                return True
        if operation in {"And", "Or", "Xor"}:
            # These repository ops are overloaded: logical operands mean
            # boolean conjunction/disjunction, integer operands mean bitwise
            # arithmetic.  The expression emitter already makes this same
            # distinction; assignment coercion must not subsequently treat an
            # integer IOR as a mask and wrap it in MERGE.
            return bool(instr.args) and all(
                self._is_logical(value) for value in instr.args
            )
        return (
            operation in _COMPARISON
            or operation in _LOGICAL_BINARY
            or operation in _LOGICAL_RESULT_UNARY
        )

    def _collect_producers(self) -> None:
        for block in self.function.blocks.values():
            for instr in block.instrs:
                if instr.res is not None:
                    self._producers[instr.res.id] = instr

    def _numeric(self, instr: Instr, args: list[str]) -> list[str]:
        """Give a LOGICAL operand a numeric value where one is required.

        A comparison produces LOGICAL, and the recorded program then uses it
        in arithmetic the way numpy does, where ``True`` is 1.  Fortran keeps
        the types apart and rejects LOGICAL at every numeric intrinsic and
        operator, so the conversion has to be written out -- ``MERGE`` is how
        Fortran says it, and it is what this emitter already uses for a
        boolean leaving through a real buffer.
        """

        operation = instr.attributes.get("tensor_operation") or instr.op
        if operation in _REAL_OPERAND:
            # These templates name a REAL constant of their own (sign) or
            # feed a REAL-only intrinsic, so an integer operand is a kind
            # mismatch regardless of what the result is declared as.
            target_real = True
        elif operation in _COMPARISON:
            # A comparison's result is LOGICAL, which says nothing about how
            # its operands should promote -- they promote to each other.
            target_real = any(
                str(getattr(value, "dtype", None) or self.dtype)
                not in _INTEGER_DTYPES
                for value in instr.args
            )
        else:
            result_dtype = str(getattr(instr.res, "dtype", None) or self.dtype)
            target_real = result_dtype not in _INTEGER_DTYPES
        converted = list(args)
        for position, value in enumerate(instr.args):
            if position >= len(converted):
                continue
            if self._is_logical(value):
                converted[position] = _UNARY["bool_to_float64"].format(
                    converted[position]
                )
                continue
            # numpy promotes mixed integer/real operands silently; Fortran
            # rejects them, so the promotion is written out.
            dtype = str(getattr(value, "dtype", None) or self.dtype)
            operand_real = dtype not in _INTEGER_DTYPES
            if operand_real == target_real:
                continue
            converted[position] = (
                f"real({converted[position]}, c_double)"
                if target_real
                else f"int({converted[position]}, c_int)"
            )
        return converted

    def _coerce_to_result(
        self,
        instr: Instr,
        value: SSAValue,
        expression: str,
    ) -> str:
        """Convert one expression to the instruction result's scalar kind.

        Fortran's ``merge`` requires its true and false sources to have the
        same type and kind.  The tensor IR follows numpy promotion rules, so
        a ``where`` may quite correctly select between an integer and a real,
        or between a logical and a number.  Apply that promotion to each
        branch before spelling the intrinsic.
        """

        return self._coerce_value_to_target(instr.res, value, expression)

    def _coerce_value_to_target(
        self,
        target: SSAValue,
        value: SSAValue,
        expression: str,
    ) -> str:
        """Convert a scalar expression to an SSA target's inferred kind."""

        result_dtype = str(self._typed(target).dtype or self.dtype)
        target_logical = result_dtype in _LOGICAL_DTYPES
        source_logical = self._is_logical(value)
        if target_logical:
            if source_logical:
                return expression
            source_dtype = str(getattr(value, "dtype", None) or self.dtype)
            zero = (
                "0_c_int64_t" if source_dtype.endswith("int64")
                else "0_c_int32_t" if source_dtype.endswith(("int32", "int"))
                else "0.0_c_double"
            )
            return f"({expression} /= {zero})"
        if source_logical:
            numeric = _UNARY["bool_to_float64"].format(expression)
            return (
                numeric
                if result_dtype not in _INTEGER_DTYPES
                else f"int({numeric}, c_int64_t)"
            )
        source_dtype = str(getattr(value, "dtype", None) or self.dtype)
        source_real = source_dtype not in _INTEGER_DTYPES
        target_real = result_dtype not in _INTEGER_DTYPES
        if source_real == target_real:
            return expression
        return (
            f"real({expression}, c_double)"
            if target_real
            else f"int({expression}, c_int64_t)"
        )

    def _coerce_call_operand(
        self, value: SSAValue, expression: str, target: SSAValue | None
    ) -> str:
        """Spell the explicit scalar conversion required by a native formal."""

        if target is None:
            return expression
        source_dtype = str(self._typed(value).dtype or self.dtype)
        target_dtype = str(target.dtype or self.dtype)
        source_logical = source_dtype in _LOGICAL_DTYPES
        target_logical = target_dtype in _LOGICAL_DTYPES
        if source_logical and target_logical:
            return f"logical({expression}, kind=c_bool)"
        if source_logical != target_logical:
            if target_logical:
                zero = (
                    "0_c_int64_t" if source_dtype.endswith("int64")
                    else "0_c_int32_t" if source_dtype.endswith(("int32", "int"))
                    else "0.0_c_double"
                )
                return f"({expression} /= {zero})"
            numeric = _UNARY["bool_to_float64"].format(expression)
            return (
                f"int({numeric}, c_int64_t)"
                if target_dtype in _INTEGER_DTYPES
                else numeric
            )
        source_integer = source_dtype in _INTEGER_DTYPES
        target_integer = target_dtype in _INTEGER_DTYPES
        if target_integer:
            kind = (
                "c_int64_t" if target_dtype.endswith("64")
                else "c_int32_t"
            )
            return (
                expression
                if source_dtype == target_dtype
                else f"int({expression}, {kind})"
            )
        if source_integer and not target_integer:
            return f"real({expression}, c_double)"
        return expression

    def _conform(self, instr: Instr, args: list[str]) -> list[str] | None:
        """Make an elementwise op's operands conform to its result shape.

        numpy broadcasts, so the recorded program freely combines a
        ``(2304,)`` with a ``(1, 48, 48)`` or a one-element array with a
        whole field.  Fortran's whole-array operators require identical
        shapes, with exactly one exception: a scalar combines with any array.
        Emitting the operands unchanged produces "Incompatible ranks", so
        each is restated at the result's shape.
        """

        if instr.res is None:
            return args
        result_shape = tuple(instr.res.shape)
        result_rank = self.dynamic_array_ranks.get(
            int(instr.res.id), len(result_shape)
        )
        result_count = _element_count(result_shape)
        conformed: list[str] = []
        for position, expression in enumerate(args):
            if position >= len(instr.args):
                # A scalar recorded as an attribute rather than an SSA value
                # (right_scalar/left_scalar) was appended to args. It is
                # already a Fortran scalar and conforms to anything.
                conformed.append(expression)
                continue
            value = instr.args[position]
            shape = tuple(value.shape)
            source_rank = self.dynamic_array_ranks.get(
                int(value.id), len(shape)
            )
            if result_rank == 0 and source_rank > 0 and not shape:
                subscripts = ", ".join("1" for _ in range(source_rank))
                conformed.append(f"{expression}({subscripts})")
                continue
            if shape == result_shape or not shape:
                conformed.append(expression)
                continue
            count = _element_count(shape)
            if count == result_count:
                # Same elements, different rank: preserve the recorded
                # row-major walk while restating the shape. A plain Fortran
                # RESHAPE consumes and fills in column-major order; that
                # silently transposes/chops non-square reshape -> permute ->
                # flatten programs. Reverse the source traversal first, then
                # make the destination's last dimension vary fastest, exactly
                # as the explicit reshape structural lowering does above.
                if not result_shape:
                    conformed.append(f"{expression}({', '.join(['1'] * len(shape))})")
                    continue
                source = expression
                if len(shape) > 1:
                    source_extents = ", ".join(
                        str(int(size)) for size in reversed(shape)
                    )
                    source_order = ", ".join(
                        str(value) for value in range(len(shape), 0, -1)
                    )
                    source = (
                        f"reshape({source}, [{source_extents}], "
                        f"order=[{source_order}])"
                    )
                extents = ", ".join(str(int(size)) for size in result_shape)
                if len(result_shape) > 1:
                    result_order = ", ".join(
                        str(value)
                        for value in range(len(result_shape), 0, -1)
                    )
                    conformed.append(
                        f"reshape({source}, [{extents}], "
                        f"order=[{result_order}])"
                    )
                else:
                    conformed.append(f"reshape({source}, [{extents}])")
                continue
            if count == 1:
                # A one-element array acting as a scalar: index it, so
                # Fortran's scalar-to-array broadcast applies.
                subscripts = ", ".join(["1"] * len(shape))
                conformed.append(f"{expression}({subscripts})")
                continue
            broadcast = _broadcast(expression, shape, result_shape)
            if broadcast is None:
                return None
            conformed.append(broadcast)
        return conformed

    def _region_call(
        self, block: BasicBlock, index: int
    ) -> list[str] | None:
        """A scheduled region's call, as one Fortran ``call`` statement.

        The lowering states a region call the way a pointer-based backend
        consumes it: return an aggregate, then walk into it per output.

            %agg = Call region(...)      result_convention='ssa.aggregate'
            %c   = Const <k>
            %ptr = GetElementPtr %agg, %c
            %out = Load %ptr

        Fortran has no pointer arithmetic and returns through arguments, so
        ``GetElementPtr`` has no standalone meaning here -- only the group
        does. Recognising the group turns all of it into one call whose
        outputs are actual arguments, which is what the callee already
        declares (its own ``intent(out)`` dummies).
        """

        instr = block.instrs[index]
        if instr.op not in ("Call", "call"):
            return None
        if instr.attributes.get("result_convention") != "ssa.aggregate":
            return None
        callee = instr.attributes.get("callee")
        if not callee:
            return None
        aggregate = instr.res
        outputs: dict[int, SSAValue] = {}
        consumed: list[SSAValue] = []
        if aggregate is not None:
            addresses: dict[int, int] = {}
            for follower in block.instrs[index + 1:]:
                if follower.res is None:
                    continue
                if follower.op == "GetElementPtr" and follower.args and (
                    follower.args[0].id == aggregate.id
                ):
                    position = follower.attributes.get("aggregate_index")
                    if position is None:
                        return None
                    addresses[follower.res.id] = int(position)
                    consumed.append(follower.res)
                elif follower.op == "Load" and follower.args and (
                    follower.args[0].id in addresses
                ):
                    outputs[addresses[follower.args[0].id]] = follower.res
                    consumed.append(follower.res)
            if not outputs:
                return None
            consumed.append(aggregate)

        # Repository outputs are identities, not tuple-slot occurrences.  A
        # source return may deliberately repeat one value in more than one
        # aggregate position; module signature construction canonicalizes
        # those positions by SSA id.  Apply the same canonicalization at the
        # call site so a repeated projection does not become an extra native
        # argument (the caller linker already makes both projections name the
        # same SSA value).
        ordered_outputs = []
        seen_output_ids: set[int] = set()
        for position in sorted(outputs):
            value = outputs[position]
            value_id = int(value.id)
            if value_id in seen_output_ids:
                continue
            seen_output_ids.add(value_id)
            ordered_outputs.append(value)
        expected_outputs = int(self.callee_output_count.get(str(callee), 0))
        if expected_outputs and len(ordered_outputs) != expected_outputs:
            return None
        for value in ordered_outputs:
            self._locals[value.id] = self._typed(value)
        for value in consumed:
            self._consumed.add(value.id)

        # The callee is emitted by this same module, so its argument order is
        # known: its own extents first, then feeds, then outputs. The extents
        # must be the ones it actually declares -- rederiving them from the
        # call site gives a different set whenever the callee has interior
        # arrays whose sizes never appear in its signature.
        declared = self.callee_extents.get(str(callee))
        if declared is not None:
            extents = list(declared)
        else:
            call_values = [*instr.args, *ordered_outputs]
            extents = sorted(dimension_extents(call_values).values())
        array_positions = self.callee_array_arguments.get(
            str(callee), frozenset()
        )
        formal_arguments = self.callee_arguments.get(str(callee), ())
        input_arguments = []
        for argument_index, value in enumerate(instr.args):
            array_expected = argument_index in array_positions
            operand = self._call_operand(
                value, array_expected=array_expected
            )
            if not array_expected:
                operand = self._coerce_call_operand(
                    value,
                    operand,
                    formal_arguments[argument_index]
                    if argument_index < len(formal_arguments)
                    else None,
                )
            input_arguments.append(operand)
        output_arguments = [_name(value) for value in ordered_outputs]
        prelude = []
        aliased_output_indices = set()
        for input_index, output_index in self.callee_inout_pairs.get(
            str(callee), ()
        ):
            if (
                input_index >= len(input_arguments)
                or output_index >= len(output_arguments)
            ):
                return None
            output_argument = output_arguments[output_index]
            source_value = self._typed(instr.args[input_index])
            target_value = self._typed(ordered_outputs[output_index])
            source_expression = input_arguments[input_index]
            source_logical = self._is_logical(source_value)
            target_logical = str(target_value.dtype or "") in _LOGICAL_DTYPES
            if target_logical and not source_logical:
                source_dtype = str(source_value.dtype or self.dtype)
                zero = (
                    "0_c_int64_t" if source_dtype.endswith("int64")
                    else "0_c_int32_t" if source_dtype.endswith(("int32", "int"))
                    else "0.0_c_double"
                )
                source_expression = f"({source_expression} /= {zero})"
            elif source_logical and not target_logical:
                source_expression = _UNARY["bool_to_float64"].format(
                    source_expression
                )
                if str(target_value.dtype or self.dtype) in _INTEGER_DTYPES:
                    source_expression = f"int({source_expression}, c_int64_t)"
            prelude.append(
                f"    {output_argument} = {source_expression}"
            )
            input_arguments[input_index] = output_argument
            aliased_output_indices.add(int(output_index))
        arguments = [
            *extents,
            *input_arguments,
            *(
                argument
                for index, argument in enumerate(output_arguments)
                if index not in aliased_output_indices
            ),
        ]
        native_callee = self.callee_native_symbols.get(str(callee), str(callee))
        return [*prelude, f"    call {native_callee}({', '.join(arguments)})"]

    def _ssa_call(self, instr: Instr) -> list[str] | None:
        """Emit an ordinary call to another fully present SSA function."""

        if instr.op not in ("Call", "call"):
            return None
        if instr.attributes.get("tensor_operation") is not None:
            return None
        callee = str(instr.attributes.get("callee") or "")
        if not callee or callee not in self.callee_arity:
            return None
        if len(instr.args) != self.callee_arity[callee]:
            return None
        output_count = int(self.callee_output_count.get(callee, 0))
        output_argument = instr.attributes.get("ssa_output_argument")
        array_positions = self.callee_array_arguments.get(callee, frozenset())
        formal_arguments = self.callee_arguments.get(callee, ())
        call_operands = []
        for position, value in enumerate(instr.args):
            operand = self._call_operand(
                value, array_expected=position in array_positions
            )
            typed = self._typed(value)
            if position not in array_positions and tuple(typed.shape):
                if _element_count(tuple(typed.shape)) != 1:
                    return None
                operand += "(" + ", ".join("1" for _ in typed.shape) + ")"
            if position not in array_positions:
                operand = self._coerce_call_operand(
                    value,
                    operand,
                    formal_arguments[position]
                    if position < len(formal_arguments)
                    else None,
                )
            call_operands.append(operand)
        arguments = [
            *self.callee_extents.get(callee, ()),
            *call_operands,
        ]
        if output_argument is not None:
            position = int(output_argument)
            if (
                instr.res is None
                or position < 0
                or position >= len(instr.args)
                or int(instr.args[position].id) != int(instr.res.id)
            ):
                return None
            self._locals[instr.res.id] = self._typed(instr.res)
        elif output_count == 1 and instr.res is not None:
            self._locals[instr.res.id] = self._typed(instr.res)
            arguments.append(_name(instr.res))
        elif output_count != 0 or instr.res is not None:
            return None
        native_callee = self.callee_native_symbols.get(str(callee), str(callee))
        return [f"    call {native_callee}({', '.join(arguments)})"]

    #: Value-precision cast kernels arrive as ``Call`` instructions whose
    #: callee is the C kernel symbol (attached by the shared tensor SSA
    #: lowering) with operands ``(source, out, count)``.  They are not SSA
    #: functions in the module, so ``_ssa_call`` cannot serve them -- but each
    #: is one ELEMENTAL Fortran expression, so the whole call is a single
    #: whole-array assignment and the count operand never needs to exist.
    #: The reference semantics is the numpy backend's ``_cast_`` map.
    _CAST_KERNEL_SPELLINGS = {
        "cast_double_to_float_values": "real(real({0}, c_float), c_double)",
        "cast_double_to_double_values": "({0})",
        "cast_double_to_int_values": "real(int({0}, c_int64_t), c_double)",
        "cast_double_to_bool_values": (
            "merge(1.0_c_double, 0.0_c_double, ({0}) /= 0.0_c_double)"
        ),
    }

    def _cast_kernel_call(self, instr: Instr) -> list[str] | None:
        if instr.op not in ("Call", "call"):
            return None
        spelling = self._CAST_KERNEL_SPELLINGS.get(
            str(instr.attributes.get("callee") or "")
        )
        if spelling is None or instr.res is None:
            return None
        output_argument = instr.attributes.get("ssa_output_argument")
        if (
            output_argument is None
            or len(instr.args) < 2
            or int(output_argument) < 0
            or int(output_argument) >= len(instr.args)
            or int(instr.args[int(output_argument)].id) != int(instr.res.id)
        ):
            return None
        source = self._operand(instr.args[0])
        self._locals[instr.res.id] = self._typed(instr.res)
        return [f"    {_name(instr.res)} = {spelling.format(source)}"]

    def _indexed_store(self, instr: Instr) -> list[str] | None:
        """``collection[i] = value``, without materialising an address.

        A ``GetElementPtr``/``Store`` pair is how a pointer-based backend
        writes one iteration's value into a resident collection.  Fortran
        indexes the array directly, so the address never becomes a value and
        the pair collapses into one assignment.
        """

        if instr.op not in ("Store", "store") or len(instr.args) != 2:
            return None
        source, address = instr.args
        producer = self._address_producers.get(address.id)
        if producer is None:
            if int(address.id) not in self.array_base_ids:
                return None
            return [
                f"    {self._operand(address)}(1) = {self._operand(source)}"
            ]
        collection, positions = producer
        self._consumed.add(address.id)
        # SSA induction values are 0-based; Fortran subscripts start at 1. A
        # subscript must be an integer, but an index carried in the f64 working
        # type is a real, so truncate it explicitly rather than lean on the
        # legacy real-index extension.
        subscripts = ", ".join(
            (
                f"{self._operand(position)} + 1"
                if str(self._typed(position).dtype or "") in _INTEGER_DTYPES
                else f"int({self._operand(position)}) + 1"
            )
            for position in positions
        )
        source_expression = self._operand(source)
        collection_dtype = str(self._typed(collection).dtype or self.dtype)
        source_dtype = str(self._typed(source).dtype or self.dtype)
        if source_dtype == "opaque_ref" and collection_dtype not in _INTEGER_DTYPES:
            # Object scalar arenas are fixed-width f64 words. Preserve an
            # opaque handle's bits exactly instead of numerically converting
            # i64 through a lossy real value.
            source_expression = f"transfer({source_expression}, 0.0_c_double)"
        else:
            source_expression = self._coerce_value_to_target(
                collection, source, source_expression
            )
        return [
            f"    {self._operand(collection)}({subscripts})"
            f" = {source_expression}"
        ]

    def _collect_address_producers(self) -> None:
        """Index every ``GetElementPtr`` that addresses a collection slot."""

        for block in self.function.blocks.values():
            for instr in block.instrs:
                if instr.op not in ("GetElementPtr", "getelementptr") or instr.res is None:
                    continue
                # Aggregate result extraction is handled by _region_call and
                # carries its index as metadata. Ordinary tensor addressing
                # carries the collection followed by one index per rank.
                if len(instr.args) < 2:
                    continue
                collection = instr.args[0]
                positions = tuple(instr.args[1:])
                self._address_producers[instr.res.id] = (collection, positions)
                if instr.attributes.get("binding") == "collection_publication":
                    # A collection is written once per iteration, so it holds
                    # one element per trip and its extent comes from the loop.
                    self._collections[collection.id] = positions[0]

        stored_addresses = {
            instr.args[1].id
            for block in self.function.blocks.values()
            for instr in block.instrs
            if instr.op in ("Store", "store") and len(instr.args) == 2
        }
        self._mutated_arrays = {
            collection.id
            for address_id, (collection, _positions) in self._address_producers.items()
            if address_id in stored_addresses
        }

    def _collection_extent(self, induction: SSAValue) -> str | None:
        """The trip count governing a collection, as a Fortran expression.

        The loop's own exit test states it -- ``induction < bound`` in the
        header -- so the bound is read from there rather than assumed. It is
        typically a control uniform, which is already a dummy argument, so
        the collection can be declared explicit-shape over it.
        """

        for block in self.function.blocks.values():
            for instr in block.instrs:
                operation = (
                    instr.attributes.get("tensor_operation") or instr.op
                )
                if operation not in ("Lt", "Le", "less", "less_equal"):
                    continue
                if len(instr.args) != 2 or instr.args[0].id != induction.id:
                    continue
                return _name(instr.args[1])
        return None

    def _loop_variable(self) -> str:
        """An integer loop index, declared alongside the locals."""

        name = f"i_loop{len(self._loop_variables)}"
        self._loop_variables.append(name)
        return name

    def _statements(self, instr: Instr) -> list[str] | None:
        """Ops that are a Fortran *statement* rather than one expression.

        A running sum or an indexed assignment has no single-expression
        intrinsic form, so returning None from ``_expression`` and reporting a
        shortfall would be wrong -- Fortran expresses both directly, just as
        statements. Everything a single array expression can say stays in
        ``_expression``.
        """

        operation = instr.attributes.get("tensor_operation") or instr.op
        if instr.res is None:
            return None
        batched = self._batched_matmul(instr, operation)
        if batched is not None:
            return batched
        if operation not in (
            "cumsum", "gather", "scatter", "index_set", "pad", "stack",
            "concat", "concatenate",
        ):
            return None
        target = _name(instr.res)

        if operation == "pad":
            if len(instr.args) != 1:
                return None
            source = self._operand(instr.args[0])
            source_shape = tuple(int(value) for value in instr.args[0].shape)
            rank = len(source_shape)
            pad = instr.attributes.get("pad", 0)
            if isinstance(pad, int):
                widths = [(int(pad), int(pad))] * rank
            elif (
                isinstance(pad, (tuple, list))
                and len(pad) == rank
                and all(
                    isinstance(pair, (tuple, list)) and len(pair) == 2
                    for pair in pad
                )
            ):
                widths = [tuple(map(int, pair)) for pair in pad]
            elif isinstance(pad, (tuple, list)) and len(pad) % 2 == 0:
                widths = [(0, 0)] * (rank - len(pad) // 2)
                widths.extend(
                    (int(pad[-2 * (axis + 1)]), int(pad[-2 * (axis + 1) + 1]))
                    for axis in range(len(pad) // 2)
                )
            else:
                return None
            mode = str(instr.attributes.get("mode", "constant"))
            if mode == "constant":
                value = _literal(instr.attributes.get("value", 0.0))
                sections = [
                    f"{left + 1}:{left + extent}"
                    for extent, (left, _right) in zip(source_shape, widths)
                ]
                return [
                    f"    {target} = {value}",
                    f"    {target}({', '.join(sections)}) = {source}",
                ]
            if mode != "edge":
                return None
            indices = [self._loop_variable() for _ in range(rank)]
            statements = []
            indent = "    "
            for index, extent, (left, right) in zip(
                indices, source_shape, widths
            ):
                statements.append(
                    f"{indent}do {index} = 1, {left + extent + right}"
                )
                indent += "  "
            source_indices = [
                f"min(max({index} - {left}, 1), {extent})"
                for index, extent, (left, _right) in zip(
                    indices, source_shape, widths
                )
            ]
            statements.append(
                f"{indent}{target}({', '.join(indices)}) = "
                f"{source}({', '.join(source_indices)})"
            )
            for _index in reversed(indices):
                indent = indent[:-2]
                statements.append(f"{indent}end do")
            return statements

        if operation == "gather":
            if len(instr.args) != 2 or len(instr.res.shape) != 1:
                return None
            source, indices = (self._operand(arg) for arg in instr.args)
            return [f"    {target} = {source}({indices} + 1)"]

        if operation == "index_set":
            if len(instr.args) != 2:
                return None
            base = self._operand(instr.args[0])
            value = self._operand(instr.args[1])
            rank = len(instr.res.shape)
            index = instr.attributes.get("slices")
            items = list(index) if isinstance(index, tuple) else [index]
            items.extend([slice(None)] * (rank - len(items)))
            if len(items) != rank:
                return None
            subscripts = []
            for axis, item in enumerate(items):
                extent = int(instr.res.shape[axis])
                if isinstance(item, int):
                    normalized = item + extent if item < 0 else item
                    subscripts.append(str(normalized + 1))
                elif isinstance(item, slice):
                    start, stop, step = item.indices(extent)
                    if step > 0:
                        subscripts.append(
                            f"{start + 1}:{stop}"
                            + ("" if step == 1 else f":{step}")
                        )
                    else:
                        # Python's stop is exclusive; for a descending
                        # Fortran triplet its inclusive endpoint is stop+2 in
                        # one-based coordinates.
                        subscripts.append(
                            f"{start + 1}:{stop + 2}:{step}"
                        )
                else:
                    return None
            if (
                tuple(instr.args[1].shape)
                and _element_count(tuple(instr.args[1].shape)) == 1
            ):
                value = f"sum({value})"
            return [
                f"    {target} = {base}",
                f"    {target}({', '.join(subscripts)}) = {value}",
            ]

        if operation in ("concat", "concatenate", "cat"):
            # Concatenation writes each source into its own run of the joined
            # dimension. Fortran has no general concat intrinsic, but a
            # section assignment per source says it exactly, at any rank.
            rank = len(instr.res.shape)
            if rank == 0:
                return None
            dim = int(instr.attributes.get("dim", 0)) % rank
            statements = []
            offset = 0
            for argument in instr.args:
                extent = int(argument.shape[dim]) if argument.shape else 1
                subscripts = [":"] * rank
                subscripts[dim] = f"{offset + 1}:{offset + extent}"
                statements.append(
                    f"    {target}({', '.join(subscripts)}) = "
                    f"{self._operand(argument)}"
                )
                offset += extent
            return statements

        if operation == "stack":
            # Stacking adds a new dimension, so each source fills one slice of
            # it -- a sequence of section assignments, which is exactly what
            # Fortran writes. Arrays are declared in SSA dimension order, so
            # the new axis sits where the op says it does.
            rank = len(instr.res.shape)
            if rank == 0:
                return None
            dim = int(instr.attributes.get("dim", 0)) % rank
            statements = []
            for position, argument in enumerate(instr.args, start=1):
                subscripts = [":"] * rank
                subscripts[dim] = str(position)
                statements.append(
                    f"    {target}({', '.join(subscripts)}) = "
                    f"{self._operand(argument)}"
                )
            return statements

        if operation == "cumsum":
            source = self._operand(instr.args[0])
            rank = len(instr.res.shape)
            if rank == 0:
                return None
            dim = int(instr.attributes.get("dim", 0)) % rank
            # Arrays are declared in SSA dimension order (see dims() in
            # emit()), so Fortran subscript k+1 is SSA dim k.
            axis = dim + 1
            index = self._loop_variable()
            def section(position: str) -> str:
                subscripts = [":"] * rank
                subscripts[axis - 1] = position
                return f"{target}({', '.join(subscripts)})"
            return [
                f"    {target} = {source}",
                f"    do {index} = 2, size({target}, {axis})",
                f"      {section(index)} = {section(index)} + "
                f"{section(f'{index} - 1')}",
                "    end do",
            ]

        # scatter: copy, then assign through the index vector. Fortran applies
        # a vector subscript elementwise, so no loop is needed.
        if len(instr.args) != 3:
            return None
        base, indices, values = (self._operand(a) for a in instr.args)
        if len(instr.res.shape) != 1:
            return None
        return [
            f"    {target} = {base}",
            # SSA indices are 0-based; Fortran subscripts start at 1.
            f"    {target}({indices} + 1) = {values}",
        ]

    def _expression(self, instr: Instr) -> str | None:
        op = instr.op
        if op in ("Call", "call"):
            # precompile_to_ssa.lower_fused_program_to_ssa wraps almost
            # every tensor op in Handler.Call (it names the C/LLVM kernel
            # symbol as "callee" for those backends), but it preserves the
            # original canonical op name under "tensor_operation" precisely
            # so a target that doesn't dispatch by callee symbol -- this one
            # -- can still recognise the operation. Without this, every
            # instruction coming from that lowering path reports as an
            # unsupported "Call" shortfall, even ops this emitter already
            # knows how to express (add, sin, sum, matmul, ...).
            tensor_operation = instr.attributes.get("tensor_operation")
            if tensor_operation is not None:
                op = str(tensor_operation)
            else:
                callee = str(instr.attributes.get("callee") or "")
                intrinsic = {
                    "pow": "{0} ** {1}",
                    "llvm.sqrt.f64": "sqrt({0})",
                    "llvm.fabs.f64": "abs({0})",
                    "llvm.round.f64": "anint({0})",
                    "llvm.trunc.f64": "aint({0})",
                    "llvm.floor.f64": "floor({0}, kind=c_int64_t)",
                    "llvm.ceil.f64": "ceiling({0}, kind=c_int64_t)",
                }.get(callee)
                if intrinsic is None and len(instr.args) == 1:
                    # A call whose callee is simply the operation's own name,
                    # carrying no ``tensor_operation`` to recognise it by.
                    # These are the primitive-library functions the SSA
                    # lowering emits -- ``exp`` calling ``exp`` -- and the
                    # name is the whole identity. Consulting ``_UNARY`` here
                    # means one registration serves both spellings instead of
                    # this dict having to restate every intrinsic it already
                    # holds.
                    intrinsic = _UNARY.get(callee)
                if intrinsic is not None:
                    return intrinsic.format(*(self._operand(a) for a in instr.args))
                if callee in {"llvm.fcmp.ord", "llvm.fcmp.uno"} and len(instr.args) == 2:
                    left, right = (self._operand(a) for a in instr.args)
                    if callee.endswith(".ord"):
                        return (
                            f"((.not. ieee_is_nan({left})) .and. "
                            f"(.not. ieee_is_nan({right})))"
                        )
                    return f"(ieee_is_nan({left}) .or. ieee_is_nan({right}))"
        args = [self._operand(a) for a in instr.args]
        constant = instr.attributes.get("constant", None)

        if op in {"SiToFp", "UiToFp"} and len(instr.args) == 1 and self._is_logical(instr.args[0]):
            return _UNARY["bool_to_float64"].format(args[0])

        if op in {"LAnd", "LOr", "logical_and", "logical_or"} and len(args) == 2:
            logical_args = []
            for value, expression in zip(instr.args, args):
                if (
                    self.dynamic_array_ranks.get(int(instr.res.id), 0) == 0
                    and self.dynamic_array_ranks.get(int(value.id), 0) > 0
                ):
                    rank = self.dynamic_array_ranks[int(value.id)]
                    expression += "(" + ", ".join(
                        "1" for _ in range(rank)
                    ) + ")"
                if self._is_logical(value):
                    logical_args.append(expression)
                    continue
                dtype = str(self._typed(value).dtype or self.dtype)
                zero = (
                    "0_c_int64_t" if dtype.endswith("int64")
                    else "0_c_int32_t" if dtype.endswith(("int32", "int"))
                    else "0.0_c_double"
                )
                logical_args.append(f"({expression} /= {zero})")
            operator = ".and." if op in {"LAnd", "logical_and"} else ".or."
            return f"({logical_args[0]} {operator} {logical_args[1]})"

        if op in {"And", "Or", "Xor"} and len(instr.args) == 2 and all(
            self._is_logical(value) for value in instr.args
        ):
            logical_operator = {
                "And": ".and.", "Or": ".or.", "Xor": ".neqv."
            }[op]
            return f"({args[0]} {logical_operator} {args[1]})"

        if op in {"And", "Or", "Xor"} and len(args) == 2:
            expression = _BINARY[op].format(*(
                f"int({argument}, c_int64_t)" for argument in args
            ))
            if str(instr.res.dtype or self.dtype) in _INTEGER_DTYPES:
                return f"int({expression}, c_int)"
            return f"real({expression}, c_double)"

        if op == "invert" and len(args) == 1:
            source_dtype = str(self._typed(instr.args[0]).dtype).casefold()
            if source_dtype in _LOGICAL_DTYPES:
                return f"(.not. {args[0]})"
            if "int" in source_dtype:
                return f"not({args[0]})"
            # Source ``~`` is unambiguously integer even when an upstream
            # source region has not yet refined its scalar dtype.  Preserve
            # integer bit semantics and convert back only when the result ABI
            # remains the generic scalar dtype.
            expression = f"not(int({args[0]}, c_int64_t))"
            if str(instr.res.dtype or self.dtype) in _INTEGER_DTYPES:
                return expression
            return f"real({expression}, c_double)"

        if op in {"bitand", "bitor", "bitxor", "shl", "shr"} and len(args) == 2:
            if all(self._is_logical(value) for value in instr.args):
                logical = {
                    "bitand": ".and.",
                    "bitor": ".or.",
                    "bitxor": ".neqv.",
                }.get(op)
                if logical is None:
                    return None
                return f"({args[0]} {logical} {args[1]})"
            # Every nested bit expression uses one kind.  Keeping an authored
            # c_int32 operand unchanged while widening only a generic/real
            # neighbour lets an inlined ``ior(i32, i32)`` meet ``not(i64)`` in
            # an outer ``iand``, which Fortran correctly rejects.
            integer_args = [
                f"int({argument}, c_int64_t)" for argument in args
            ]
            expression = _BINARY[op].format(*integer_args)
            if str(instr.res.dtype or self.dtype) in _INTEGER_DTYPES:
                return expression
            return f"real({expression}, c_double)"

        # A string word is its typed signed 64-bit fnv1a identity.  It is not a
        # numerical value and is never carried through floating-point bits.
        if op == "string_token":
            token = int(instr.attributes["token"])
            return f"{token}_c_int64_t"

        if op == "StaticRef":
            return f"{int(instr.attributes['reference_handle'])}_c_int64_t"

        # A token comparison is a direct identity test on typed i64 values.
        if instr.attributes.get("string_compare") and op in (
            "equal", "not_equal", "Eq", "Ne"
        ) and len(args) == 2:
            comparator = "==" if op in ("equal", "Eq") else "/="
            return f"({args[0]} {comparator} {args[1]})"

        if op.casefold() in {"sin", "cos"} and len(args) == 1:
            from .bounded_constants import materialize_pi

            shift = (
                0.0 if op.casefold() == "sin"
                else float(materialize_pi("literal").value) * 0.5
            )
            if self.trig_solver == "lut":
                self.uses_sin_table = True
                return _table_sin_fortran(args[0], shift)
            return _series_sin_fortran(args[0], shift)

        if op == "Pi":
            # The same materialisation the other three lanes use, so one
            # constant with one declared error bound serves all of them
            # instead of each spelling its own literal.
            from .bounded_constants import materialize_pi

            materialization = materialize_pi(
                instr.attributes.get("constant_solver") or "literal",
                instr.attributes.get("requested_epsilon"),
            )
            if materialization.value is None:
                return None
            return _literal(float(materialization.value))

        if op in ("Const", "const"):
            if constant is None and "llvm_literal" in instr.attributes:
                return _literal(_llvm_literal(instr.attributes["llvm_literal"]))
            if constant is None and "values" in instr.attributes:
                # An array constant carries its elements under "values", not
                # the scalar "constant" key.  Reading only "constant" here
                # yielded None and reported "cannot express literal None",
                # which named the missing key rather than the real content.
                values = instr.attributes["values"]
                if not instr.res.shape and not isinstance(values, (list, tuple)):
                    # ``tensor_from_list`` is also the historical spelling
                    # used for captured scalar and word constants.  Those
                    # values are already scalar; iterating a float is an
                    # error and iterating a word would silently turn it into
                    # an array of character tokens.
                    return _literal(values)
                return _array_literal(
                    values,
                    instr.res.shape,
                    dtype=instr.res.dtype or self.dtype,
                )
            if constant is None and instr.attributes.get("value") is not None:
                # Control-flow scalars (loop bounds, strides) are recorded
                # under "value" rather than "constant".  This is checked after
                # "values" on purpose: an array constant carries both keys,
                # with a vestigial "value" of None, so testing it first would
                # discard the real elements.
                try:
                    return _literal(instr.attributes["value"])
                except FortranEmissionError as error:
                    raise FortranEmissionError(
                        f"{error}; function={self.function.name!r}; "
                        f"result_value_id={getattr(instr.res, 'id', None)!r}; "
                        f"attributes={instr.attributes!r}"
                    ) from error
            return _literal(constant)

        if op in ("Load", "load") and len(instr.args) == 1:
            producer = self._address_producers.get(instr.args[0].id)
            if producer is not None:
                collection, positions = producer
                subscripts = ", ".join(
                    (
                        f"{self._operand(position)} + 1"
                        if str(self._typed(position).dtype or "") in _INTEGER_DTYPES
                        else f"int({self._operand(position)}) + 1"
                    )
                    for position in positions
                )
                loaded = f"{self._operand(collection)}({subscripts})"
                if str(instr.res.dtype or "") == "opaque_ref" and str(
                    self._typed(collection).dtype or self.dtype
                ) not in _INTEGER_DTYPES:
                    return f"transfer({loaded}, 0_c_int64_t)"
                return loaded
            if int(instr.args[0].id) in self.array_base_ids:
                return f"{self._operand(instr.args[0])}(1)"

        if op in {"Cast", "CastLike", "cast_like"} and args:
            operand = args[0]
            source_rank = self.dynamic_array_ranks.get(
                int(instr.args[0].id), len(tuple(self._typed(instr.args[0]).shape))
            )
            result_rank = self.dynamic_array_ranks.get(
                int(instr.res.id), len(tuple(self._typed(instr.res).shape))
            )
            if source_rank > 0 and result_rank == 0:
                operand += "(" + ", ".join("1" for _ in range(source_rank)) + ")"
            return self._coerce_value_to_target(
                instr.res, instr.args[0], operand
            )

        structural = self._structural(instr, args, op)
        if structural is not None:
            return structural

        # Scalar operands recorded as attributes rather than SSA values.
        right = instr.attributes.get("right_scalar")
        left = instr.attributes.get("left_scalar")
        if right is not None and len(args) == 1:
            args = [args[0], _literal(right)]
        elif left is not None and len(args) == 1:
            args = [_literal(left), args[0]]

        if op in _REDUCTION and len(args) == 1:
            if op == "sum" and self._is_logical(instr.args[0]):
                # Summing a mask counts its true elements, which Fortran
                # states directly; SUM rejects a LOGICAL argument outright.
                return f"count({args[0]})"
            args = self._numeric(instr, args)
            return _REDUCTION[op].format(*args)
        if op in _BINARY and len(args) == 2:
            template = _BINARY[op]
            # Shape first, then type. Conforming subscripts an operand to
            # index away an extent-one dimension, and Fortran will not
            # subscript the result of an intrinsic -- so converting first
            # would produce real(x, c_double)(1), which is a syntax error.
            if op not in _SHAPE_CHANGING_BINARY:
                # Only elementwise operators require conforming shapes.
                # matmul's operands are meant to differ from its result.
                conformed = self._conform(instr, args)
                if conformed is None:
                    return None
                args = conformed
            if op not in _LOGICAL_BINARY:
                args = self._numeric(instr, args)
            if op in {"FloorDiv", "floordiv"} and str(
                instr.res.dtype or self.dtype
            ) in _INTEGER_DTYPES:
                return f"({args[0]} / {args[1]})"
            if op in {"Mod", "mod"} and str(
                instr.res.dtype or self.dtype
            ) in _INTEGER_DTYPES:
                return f"modulo(int({args[0]}, c_int64_t), int({args[1]}, c_int64_t))"
            if op in {
                "And", "Or", "Xor", "bitand", "bitor", "bitxor"
            }:
                # Nested bitwise expressions may inline a c_int32-producing
                # instruction beside an explicitly widened operand.  Fortran's
                # bit intrinsics require identical kind parameters.  State the
                # repository integer operation in one stable kind at every
                # nesting level; assignment converts to a narrower declared
                # result when that is the function contract.
                args = [f"int({argument}, c_int64_t)" for argument in args]
            if instr.attributes.get("reverse"):
                args = [args[1], args[0]]
            # A comparison landing in a numeric variable is converted at the
            # assignment, and one inlined into a numeric context is converted
            # by _numeric on its consumer. Converting here as well would
            # wrap what was already wrapped.
            return template.format(*args)
        if op in _UNARY and len(args) == 1:
            if op not in _LOGICAL_UNARY and op not in _LOGICAL_RESULT_UNARY:
                args = self._numeric(instr, args)
            return _UNARY[op].format(*args)
        if op in ("Select", "where") and len(args) == 3:
            conformed = self._conform(instr, args)
            if conformed is None:
                return None
            args = conformed
            mask = args[0]
            if not self._is_logical(instr.args[0]):
                mask_dtype = str(getattr(instr.args[0], "dtype", "")).casefold()
                zero = "0_c_int64_t" if mask_dtype.endswith("int64") else (
                    "0_c_int32_t" if mask_dtype.endswith(("int32", "int"))
                    else "0.0_c_double"
                )
                mask = f"({mask} /= {zero})"
            true_value = self._coerce_to_result(instr, instr.args[1], args[1])
            false_value = self._coerce_to_result(instr, instr.args[2], args[2])
            return f"merge({true_value}, {false_value}, {mask})"
        return None

    # -- statement emission ------------------------------------------------
    def _emit_block(self, block: BasicBlock, body: list[str]) -> None:
        body.append(f"    ! block {block.name}")
        # Only emit a statement label where something actually branches to it;
        # an unreferenced label is a compiler warning and pure noise.
        if block.name in self._branch_targets:
            body.append(f"{self._label(block.name)} continue")
        for index, instr in enumerate(block.instrs):
            if instr.op in ("Deploy", "Join"):
                # Deployment boundaries describe scheduling around the
                # numerical program.  A single native Fortran subroutine
                # executes that program serially, so the markers have no
                # runtime statement of their own, but retaining them as
                # comments keeps the structural boundary visible.
                body.append(f"    ! {instr.op} deployment boundary")
                continue
            if (
                instr.op in ("Call", "call")
                and not instr.attributes.get("tensor_operation")
                and instr.attributes.get("callee")
                == "turing_validation_error"
            ):
                # Validation calls live only in the failing CFG branch.  Use
                # Fortran's self-contained failure operation rather than an
                # unresolved external symbol that would make the DLL
                # impossible to link.
                error_code = int(instr.attributes.get("error_code", 1))
                body.append(f"    error stop {error_code}")
                continue
            if instr.op in ("Phi", "phi"):
                # Phi is realised by the predecessors, not here.
                continue
            if instr.op in ("Br", "br"):
                target = str(instr.attributes.get("target", ""))
                self._emit_phi_copies(block.name, target, body)
                body.append(f"    goto {self._label(target)}")
                continue
            if instr.op in ("CondBr", "condbr"):
                # Both spellings occur: the control lowering emits
                # true_target/false_target, hand-built SSA uses true/false.
                # An unrecognised name silently resolved to the first block,
                # so every branch went to the same wrong label.
                true_target = str(
                    instr.attributes.get("true")
                    or instr.attributes.get("true_target")
                    or ""
                )
                false_target = str(
                    instr.attributes.get("false")
                    or instr.attributes.get("false_target")
                    or ""
                )
                condition = self._operand(instr.args[0])
                body.append(f"    if ({condition}) then")
                self._emit_phi_copies(block.name, true_target, body, indent=6)
                body.append(f"      goto {self._label(true_target)}")
                body.append("    else")
                self._emit_phi_copies(block.name, false_target, body, indent=6)
                body.append(f"      goto {self._label(false_target)}")
                body.append("    end if")
                continue
            if instr.op in ("Ret", "ret", "Return", "return"):
                return_value = self.function.metadata.get("return_value")
                if return_value is not None and instr.args:
                    body.append(
                        f"    {_name(return_value)} = {self._operand(instr.args[0])}"
                    )
                body.append("    return")
                continue

            if instr.res is not None and instr.res.id in self._consumed:
                # Already emitted as part of a multi-instruction group.
                continue
            if instr.res is not None and instr.res.id in self._address_producers:
                # The address is folded into the subscript of the assignment
                # its Store becomes, so it never needs to exist as a value.
                continue

            group = self._region_call(block, index)
            if group is not None:
                body.extend(group)
                continue

            # Before _ssa_call deliberately: even when the kernel body was
            # imported into the module, the ELEMENTAL spelling is one
            # whole-array assignment the compiler can vectorise, where the
            # imported body is an explicit per-element loop.
            cast = self._cast_kernel_call(instr)
            if cast is not None:
                body.extend(cast)
                continue

            call = self._ssa_call(instr)
            if call is not None:
                body.extend(call)
                continue

            store = self._indexed_store(instr)
            if store is not None:
                body.extend(store)
                continue

            statements = self._statements(instr)
            if statements is not None:
                self._locals[instr.res.id] = self._typed(instr.res)
                body.extend(statements)
                continue

            expression = self._expression(instr)
            if expression is None:
                operation = str(
                    instr.attributes.get("tensor_operation") or instr.op
                )
                reason = "no Fortran intrinsic or expression is registered"
                operands = tuple(
                    (
                        int(source.id),
                        tuple(source.shape),
                        tuple(self._typed(source).shape),
                        str(self._typed(source).dtype),
                        int(source.id) in self.array_base_ids,
                    )
                    for source in instr.args
                )
                result = (
                    None
                    if instr.res is None
                    else (
                        int(instr.res.id),
                        tuple(instr.res.shape),
                        tuple(self._typed(instr.res).shape),
                        str(self._typed(instr.res).dtype),
                    )
                )
                attributes = {
                    str(key): value
                    for key, value in instr.attributes.items()
                    if isinstance(value, (str, int, float, bool, tuple, list, type(None)))
                }
                reason += (
                    f"; operands=(id, occurrence_shape, inferred_shape, dtype, "
                    f"dynamic_base){operands!r}; result={result!r}; "
                    f"attributes={attributes!r}"
                )
                self.shortfalls.append(
                    FortranShortfall(
                        operation,
                        block.name,
                        reason,
                    )
                )
                body.append(f"    ! UNSUPPORTED {instr.op}")
                continue
            if self._may_inline(instr, block):
                self._inlined[instr.res.id] = expression
                continue
            self._locals[instr.res.id] = self._typed(instr.res)
            if (
                self._instruction_is_logical(instr)
                and str(instr.res.dtype or self.dtype)
                not in _LOGICAL_DTYPES
            ):
                # A mask reaching a numeric variable. Fortran will not
                # convert LOGICAL on assignment the way it converts between
                # numeric kinds, so it is written out.
                expression = _UNARY["bool_to_float64"].format(expression)
            elif (
                not self._instruction_is_logical(instr)
                and str(instr.res.dtype or self.dtype) in _LOGICAL_DTYPES
            ):
                # And the reverse: a number recorded as bool, reaching a
                # variable declared LOGICAL. Non-zero is true, which is the
                # rule numpy applied when it produced the value.
                expression = f"(({expression}) /= 0)"
            body.append(f"    {_name(instr.res)} = {expression}")

    def _emit_phi_copies(
        self,
        source: str,
        target: str,
        body: list[str],
        *,
        indent: int = 4,
    ) -> None:
        pad = " " * indent
        for predecessor, result, incoming in self._phi_targets.get(target, ()):
            if predecessor != source:
                continue
            expression = self._coerce_value_to_target(
                result, incoming, _name(incoming)
            )
            body.append(f"{pad}{_name(result)} = {expression}")

    def _label(self, block_name: str) -> int:
        ordered = list(self.function.blocks)
        return 1000 + (ordered.index(block_name) if block_name in ordered else 0)

    def _collect_branch_targets(self) -> None:
        for block in self.function.blocks.values():
            for instr in block.instrs:
                if instr.op in ("Br", "br"):
                    self._branch_targets.add(
                        str(instr.attributes.get("target", ""))
                    )
                elif instr.op in ("CondBr", "condbr"):
                    # Must recognise the same spellings _emit_block does, or
                    # a branch is emitted to a label that is never placed.
                    self._branch_targets.add(
                        str(
                            instr.attributes.get("true")
                            or instr.attributes.get("true_target")
                            or ""
                        )
                    )
                    self._branch_targets.add(
                        str(
                            instr.attributes.get("false")
                            or instr.attributes.get("false_target")
                            or ""
                        )
                    )

    def _collect_phis(self) -> None:
        for block in self.function.blocks.values():
            for instr in block.instrs:
                if instr.op not in ("Phi", "phi"):
                    continue
                incoming = instr.attributes.get("incoming") or ()
                if not incoming:
                    blocks = tuple(
                        instr.attributes.get("incoming_blocks") or ()
                    )
                    if len(blocks) == len(instr.args):
                        incoming = tuple(zip(blocks, instr.args))
                self._locals[instr.res.id] = self._typed(instr.res)
                entries = self._phi_targets.setdefault(block.name, [])
                for predecessor, value in incoming:
                    entries.append((str(predecessor), instr.res, value))

    # -- assembly ----------------------------------------------------------
    def emit(self) -> FortranSubroutine:
        self._bound_ids = {a.id for a in self.function.args} | {
            value.id for value in self.outputs
        }
        self._collect_branch_targets()
        self._collect_use_sites()
        self._collect_phis()
        self._collect_address_producers()
        self._collect_producers()
        body: list[str] = []
        for block in self.function.blocks.values():
            self._emit_block(block, body)

        # A bind(C) procedure may not take assumed-shape (``x(:)``) dummies:
        # those need a descriptor the C caller has no way to build before
        # Fortran 2018 / TS 29113.  Arrays are therefore explicit-shape over
        # extents passed as leading arguments, which is what a C caller would
        # have to supply anyway and lets the compiler see the trip count.
        #
        # Every array keeps its own shape here -- one extent parameter per
        # distinct dimension size actually present, not one extent shared by
        # every array in the function.  A matmul's (rows, inner) x
        # (inner, cols) -> (rows, cols) genuinely has three different sizes;
        # forcing them through one shared extent either rejects it outright
        # or -- worse -- silently declares mismatched-size arrays under one
        # extent, which compiles cleanly and corrupts memory at runtime.
        all_values = (
            *(self._typed(value) for value in self.function.args),
            *(self._typed(value) for value in self.outputs),
            *self._locals.values(),
        )
        arrays_present = any(_is_array(value) for value in all_values)
        dim_extents = dimension_extents(all_values) if arrays_present else {}

        def dims(value: SSAValue) -> str:
            return ", ".join(dim_extents[int(size)] for size in value.shape)

        # A caller must expose every extent its emitted callees require, even
        # when that size exists only in a callee-local temporary.  Whole-field
        # reductions are the common case: the region has an extent-one scalar
        # result, while the control wrapper otherwise sees only the public
        # field extent.  The call already passes the callee's declared extent
        # list, so omitting it from this wrapper's own ABI produces an
        # undeclared ``extent_1`` identifier in otherwise valid Fortran.
        called_extent_names = {
            extent
            for block in self.function.blocks.values()
            for instr in block.instrs
            if instr.op in ("Call", "call")
            for extent in self.callee_extents.get(
                str(instr.attributes.get("callee") or ""), ()
            )
        }
        extent_names = sorted(
            set(dim_extents.values())
            | called_extent_names
            | set(self.dynamic_array_extents.values())
            | {
                extent
                for extents in self.dynamic_array_leading_extents.values()
                for extent in extents
            }
        )
        self.extent_names = tuple(extent_names)
        argument_ids = {argument.id for argument in self.function.args}
        output_ids = {value.id for value in self.outputs}
        assigned_ids = {
            int(instruction.res.id)
            for block in self.function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        arguments = list(extent_names)
        arguments.extend(_name(a) for a in self.function.args)
        arguments.extend(
            _name(value)
            for value in self.outputs
            if value.id not in argument_ids
        )

        declarations: list[str] = [
            f"    integer(c_int), intent(in), value :: {extent}"
            for extent in extent_names
        ]
        # A referenced table must also be declared. Rendering the body sets
        # this flag, and the body is rendered before this point.
        if self.uses_sin_table:
            declarations.append(sin_table_declaration())
        reference_argument_ids = {
            int(argument.id)
            for occurrence in self.function.args
            for argument in (self._typed(occurrence),)
            if (
                _is_array(argument)
                or int(argument.id) in self.array_base_ids
                or int(argument.id) in output_ids
                or int(argument.id) in assigned_ids
                or int(argument.id) in self.mutated_base_ids
                or (
                    int(argument.id) in self._collections
                    and self._collection_extent(
                        self._collections[int(argument.id)]
                    ) is not None
                )
            )
        }
        # A parameter used as an address base but with no compile-time shape is
        # a DYNAMIC array: a runtime-sized buffer indexed by a runtime value.
        # bind(C) forbids assumed-shape (needs a descriptor the C caller cannot
        # build), but assumed-size ``d(*)`` -- a bare base pointer the callee
        # indexes without knowing the extent -- is interoperable and is exactly
        # this case. So compile it as-is, no reduction to a fixed size and no
        # passed extent.
        for argument_occurrence in self.function.args:
            argument = self._typed(argument_occurrence)
            kind = _DTYPE_KIND.get(argument.dtype or self.dtype, "real(c_double)")
            if int(argument.id) in self.array_base_ids and not _is_array(argument):
                mutated = (
                    int(argument.id) in self.mutated_base_ids
                    or argument.id in output_ids
                )
                intent = "inout" if mutated else "in"
                leading_extents = self.dynamic_array_leading_extents.get(
                    int(argument.id), ()
                )
                dimensions = ", ".join(
                    leading_extents if leading_extents else ("*",)
                )
                declarations.append(
                    f"    {kind}, intent({intent}) :: "
                    f"{_name(argument)}({dimensions})"
                )
                continue
            if argument.id in output_ids:
                if _is_array(argument):
                    declarations.append(
                        f"    {kind}, intent(inout) :: {_name(argument)}({dims(argument)})"
                    )
                else:
                    declarations.append(
                        f"    {kind}, intent(inout) :: {_name(argument)}"
                    )
                continue
            if (
                int(argument.id) in assigned_ids
                or int(argument.id) in self.mutated_base_ids
            ):
                # A linked callee may publish directly into a caller value
                # that is also part of this wrapper's incoming frame.  That
                # is a genuine in/out ABI slot, even when it is not one of
                # the wrapper's final public outputs.
                if _is_array(argument):
                    declarations.append(
                        f"    {kind}, intent(inout) :: {_name(argument)}({dims(argument)})"
                    )
                else:
                    declarations.append(
                        f"    {kind}, intent(inout) :: {_name(argument)}"
                    )
                continue
            if argument.id in self._collections:
                # A collection accumulates one element per iteration and is
                # read back by the caller, so it is neither a scalar nor
                # intent(in) -- declaring it as the shapeless argument the IR
                # describes is what made indexing it unclassifiable.
                extent = self._collection_extent(self._collections[argument.id])
                if extent is not None:
                    declarations.append(
                        f"    {kind}, intent(inout) :: {_name(argument)}({extent})"
                    )
                    continue
            if (
                argument.id in self._mutated_arrays
                or int(argument.id) in self.mutated_base_ids
            ) and _is_array(argument):
                declarations.append(
                    f"    {kind}, intent(inout) :: {_name(argument)}({dims(argument)})"
                )
                continue
            if _is_array(argument):
                declarations.append(
                    f"    {kind}, intent(in) :: {_name(argument)}({dims(argument)})"
                )
            else:
                declarations.append(
                    f"    {kind}, intent(in), value :: {_name(argument)}"
                )
        for value in self.outputs:
            if value.id in argument_ids:
                continue
            kind = _DTYPE_KIND.get(value.dtype or self.dtype, "real(c_double)")
            dynamic_extents = self.dynamic_array_leading_extents.get(
                int(value.id), ()
            )
            if dynamic_extents:
                declarations.append(
                    f"    {kind}, intent(out) :: {_name(value)}"
                    f"({', '.join(dynamic_extents)})"
                )
            elif _is_array(value):
                declarations.append(
                    f"    {kind}, intent(out) :: {_name(value)}({dims(value)})"
                )
            else:
                declarations.append(
                    f"    {kind}, intent(out) :: {_name(value)}"
                )

        bound = {a.id for a in self.function.args} | {
            value.id for value in self.outputs
        }
        for value in self._locals.values():
            if value.id in bound:
                continue
            kind = _DTYPE_KIND.get(value.dtype or self.dtype, "real(c_double)")
            dynamic_extents = self.dynamic_array_leading_extents.get(
                int(value.id), ()
            )
            if dynamic_extents:
                declarations.append(
                    f"    {kind} :: {_name(value)}"
                    f"({', '.join(dynamic_extents)})"
                )
            elif _is_array(value):
                # An automatic array sized by its own extents: no allocate,
                # no heap.
                declarations.append(
                    f"    {kind} :: {_name(value)}({dims(value)})"
                )
            else:
                declarations.append(f"    {kind} :: {_name(value)}")

        declarations.extend(
            f"    integer(c_int) :: {index}" for index in self._loop_variables
        )

        name = self.function.name
        native_name = self.native_symbol
        lines = [
            f"  subroutine {native_name}({', '.join(arguments)}) bind(C, name=\"{name}\")",
            "    use, intrinsic :: iso_c_binding",
            "    use, intrinsic :: ieee_arithmetic",
            "    implicit none",
            *declarations,
            "",
            *body,
            f"  end subroutine {native_name}",
        ]
        return FortranSubroutine(
            name,
            "\n".join(lines),
            tuple(self.shortfalls),
            tuple(extent_names),
            tuple(sorted(self.dynamic_array_extents.items())),
            tuple(sorted(
                (int(value_id), tuple(dimensions))
                for value_id, dimensions in self.dynamic_array_leading_extents.items()
            )),
            tuple(sorted(reference_argument_ids)),
        )


def emit_function(
    function: Function,
    *,
    dtype: str = DEFAULT_DTYPE,
    outputs: Sequence[SSAValue] = (),
    callee_extents: Mapping[str, Sequence[str]] | None = None,
    callee_arity: Mapping[str, int] | None = None,
    callee_output_count: Mapping[str, int] | None = None,
    callee_outputs: Mapping[str, Sequence[SSAValue]] | None = None,
    callee_arguments: Mapping[str, Sequence[SSAValue]] | None = None,
    callee_array_arguments: Mapping[str, Sequence[int]] | None = None,
    callee_inout_pairs: Mapping[
        str, Sequence[tuple[int, int]]
    ] | None = None,
    trig_solver: str = "lut",
    array_base_ids: Sequence[int] = (),
    mutated_base_ids: Sequence[int] = (),
    dynamic_array_ranks: Mapping[int, int] | None = None,
    value_dtypes: Mapping[int, str] | None = None,
    tensor_table: SSATensorTable | None = None,
    native_symbol: str | None = None,
    callee_native_symbols: Mapping[str, str] | None = None,
    extent_namespace: str = "",
) -> FortranSubroutine:
    """Translate one SSA function into a bind(C) Fortran subroutine.

    ``outputs`` names the SSA values that leave the subroutine.  SSA itself
    records only arguments, so results would otherwise be emitted as dead
    locals; naming them promotes them to ``intent(out)`` parameters.

    ``callee_extents`` maps a called subroutine's name to the extent
    parameters it declares, so a call passes exactly the extents that
    subroutine expects rather than extents rederived at the call site.
    """

    from .machine_dialect_ssa import (
        format_machine_dialect_occurrences,
        module_machine_dialect_occurrences,
    )

    machine_residuals = module_machine_dialect_occurrences(
        {function.name: function}
    )
    if machine_residuals:
        raise FortranEmissionError(
            "Fortran emission requires legalized repository SSA; retained "
            "machine-state SSA remains: "
            + format_machine_dialect_occurrences(machine_residuals)
        )

    return _FunctionEmitter(
        function,
        dtype=dtype,
        trig_solver=trig_solver,
        outputs=outputs,
        callee_extents=callee_extents,
        callee_arity=callee_arity,
        callee_output_count=callee_output_count,
        callee_outputs=callee_outputs,
        callee_arguments=callee_arguments,
        callee_array_arguments=callee_array_arguments,
        callee_inout_pairs=callee_inout_pairs,
        array_base_ids=array_base_ids,
        mutated_base_ids=mutated_base_ids,
        dynamic_array_ranks=dynamic_array_ranks,
        value_dtypes=value_dtypes,
        tensor_table=tensor_table,
        native_symbol=native_symbol,
        callee_native_symbols=callee_native_symbols,
        extent_namespace=extent_namespace,
    ).emit()


@dataclass
class FortranModule:
    """A complete Fortran module wrapping one or more SSA functions."""

    name: str
    source: str
    subroutines: tuple[FortranSubroutine, ...] = ()
    # The calling contract for what was just generated. Emitted from the same
    # Function objects, so a caller binds argument order, element types and
    # by-value/by-reference from a record rather than by reading the source
    # and guessing. See compiler/compiled_program_api.py.
    api: Any = None

    @property
    def shortfalls(self) -> tuple[FortranShortfall, ...]:
        return tuple(s for sub in self.subroutines for s in sub.shortfalls)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.name}.f90"
        path.write_text(self.source, encoding="utf-8", newline="\n")
        if self.api is not None:
            # Beside the source, same stem: the artifact and its contract
            # travel together or the contract is worthless.
            self.api.write(path.with_suffix(".api.yaml"))
        return path


def emit_module(
    module: IRModule | Mapping[str, Function],
    *,
    name: str = "turing_ssa",
    dtype: str = DEFAULT_DTYPE,
    outputs: Mapping[str, Sequence[SSAValue]] | None = None,
    extra_roots: Sequence[str] = (),
    trig_solver: str = "lut",
) -> FortranModule:
    """Translate an SSA module into one Fortran module.

    ``outputs`` maps a function name to the SSA values it returns.
    ``extra_roots`` names additional functions to keep and export (each becomes
    its own ``bind(C)`` entry) even when nothing reachable from the ordinary
    roots calls them -- a library exports its whole surface, not just the
    functions the entry happens to reach.
    """

    if isinstance(module, IRModule):
        unresolved_calls = tuple(
            record
            for records in getattr(module, "call_table", {}).values()
            for record in records
            if record.resolution == "unresolved"
        )
        if unresolved_calls:
            details = "; ".join(
                f"{record.caller}@{record.callsite_id} -> "
                    f"{record.callee_symbol or record.callee_name} "
                    f"missing_frame={record.unresolved_frame_value_ids!r} "
                    f"bindings={record.frame_bindings!r} "
                    f"boundary={record.decomposition!r}"
                for record in unresolved_calls
            )
            raise FortranEmissionError(
                "repository SSA contains unresolved source call records; "
                "refusing to emit a DLL that omits authored execution: "
                + details
            )

    module_tensor_tables = (
        dict(getattr(module, "tensor_tables", {}))
        if isinstance(module, IRModule)
        else {}
    )
    module_sequence_tables = (
        dict(getattr(module, "sequence_tables", {}))
        if isinstance(module, IRModule)
        else {}
    )
    module_class_table = (
        getattr(module, "class_table", None)
        if isinstance(module, IRModule)
        else None
    )
    module_function_table = (
        getattr(module, "function_table", None)
        if isinstance(module, IRModule)
        else None
    )
    module_record_tables = (
        dict(getattr(module, "record_tables", {}))
        if isinstance(module, IRModule)
        else {}
    )
    functions = (
        module.functions if isinstance(module, IRModule) else dict(module)
    )
    # The decompiler reuses IRModule/Function as structural containers while
    # retaining exact machine-state transitions.  Container compatibility is
    # not repository-SSA legalization: reject every residual occurrence before
    # target emission can mistake a machine dialect for ordinary SSA.
    from .machine_dialect_ssa import (
        format_machine_dialect_occurrences,
        module_machine_dialect_occurrences,
    )
    machine_residuals = module_machine_dialect_occurrences(functions)
    if machine_residuals:
        raise FortranEmissionError(
            "Fortran emission requires legalized repository SSA; retained "
            "machine-state SSA remains: "
            + format_machine_dialect_occurrences(machine_residuals)
        )
    for function_name, table in module_sequence_tables.items():
        function = functions.get(function_name)
        if function is None:
            raise FortranEmissionError(
                f"sequence table names absent SSA function {function_name!r}"
            )
        available_value_ids = {int(value.id) for value in function.args}
        available_value_ids.update(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        for descriptor in table.sequences.values():
            pool = descriptor.child_table_pool
            if pool is None:
                continue
            pool_value_ids = {
                *map(int, pool.column_value_ids),
                int(pool.length_value_id),
                int(pool.capacity_value_id),
                int(pool.row_stride_value_id),
                *((int(pool.status_value_id),) if pool.status_value_id is not None else ()),
                *((int(pool.live_flags_value_id),) if pool.live_flags_value_id is not None else ()),
            }
            missing_values = pool_value_ids - available_value_ids
            if missing_values:
                raise FortranEmissionError(
                    f"sequence {descriptor.sequence_id} child-table pool names "
                    f"absent SSA values {sorted(missing_values)} in "
                    f"{function_name!r}"
                )
    for function_name, table in module_record_tables.items():
        function = functions.get(function_name)
        if function is None:
            raise FortranEmissionError(
                f"record table names absent SSA function {function_name!r}"
            )
        sequence_table = module_sequence_tables.get(function_name)
        sequence_ids = set(
            getattr(sequence_table, "descriptors", {})
            or getattr(sequence_table, "sequences", {})
        )
        record_ids = set(table.records)
        available_value_ids = {int(value.id) for value in function.args}
        available_value_ids.update(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        for descriptor in table.records.values():
            for field in descriptor.fields:
                missing_values = set(field.value_ids) - available_value_ids
                if missing_values:
                    raise FortranEmissionError(
                        f"record {descriptor.identity!r} field {field.name!r} "
                        "names absent SSA values "
                        f"{sorted(missing_values)} in {function_name!r}"
                    )
                if field.sequence_id is not None and field.sequence_id not in sequence_ids:
                    raise FortranEmissionError(
                        f"record {descriptor.identity!r} field {field.name!r} "
                        f"names absent sequence {field.sequence_id} in {function_name!r}"
                    )
                if field.record_id is not None and field.record_id not in record_ids:
                    raise FortranEmissionError(
                        f"record {descriptor.identity!r} field {field.name!r} "
                        f"names absent record {field.record_id} in {function_name!r}"
                    )
    named_outputs = dict(outputs or {})
    for function_name, function in functions.items():
        if function_name in named_outputs:
            continue
        return_value = function.metadata.get("return_value")
        if isinstance(return_value, SSAValue):
            named_outputs[function_name] = (return_value,)
            continue
        declared = tuple(function.metadata.get("named_outputs") or ())
        if not declared:
            continue
        values = {
            int(value.id): value
            for value in function.args
        }
        values.update({
            int(instruction.res.id): instruction.res
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        })
        resolved = tuple(
            values[int(value_id)]
            for _name, value_id in declared
            if int(value_id) in values
        )
        if resolved:
            named_outputs[function_name] = resolved
    named_outputs = {
        function_name: tuple({
            int(value.id): value for value in values
        }.values())
        for function_name, values in named_outputs.items()
    }
    roots = {
        str(name)
        for name in named_outputs
        if str(name) in functions
    }
    roots.update(
        str(name)
        for name, function in functions.items()
        if function.metadata.get("named_outputs")
        or function.metadata.get("control_ir")
    )
    roots.update(str(root) for root in extra_roots if str(root) in functions)
    if roots:
        reachable = set()
        pending = list(sorted(roots))
        while pending:
            function_name = pending.pop()
            if function_name in reachable or function_name not in functions:
                continue
            reachable.add(function_name)
            function = functions[function_name]
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in {"Call", "call"}:
                        continue
                    # A canonical tensor operation is emitted as a native
                    # Fortran expression.  Its C/LLVM helper name is retained
                    # for those targets, but is not a Fortran call edge.
                    if instruction.attributes.get("tensor_operation"):
                        continue
                    callee = instruction.attributes.get("callee")
                    if callee in functions and callee not in reachable:
                        pending.append(str(callee))
        # Keep ordinary wrappers that call a selected implementation.  A
        # module containing ``cycle -> advance`` with advance as the declared
        # output surface still needs cycle; dropping it loses the public
        # entry that owns the call.  Settle both directions so any sibling
        # callees required by such a wrapper are retained too.
        changed = True
        while changed:
            changed = False
            for function_name, function in functions.items():
                callees = {
                    str(instruction.attributes.get("callee"))
                    for block in function.blocks.values()
                    for instruction in block.instrs
                    if instruction.op in {"Call", "call"}
                    and not instruction.attributes.get("tensor_operation")
                    and instruction.attributes.get("callee") in functions
                }
                if function_name in reachable:
                    additions = callees - reachable
                elif callees & reachable:
                    additions = {function_name}
                else:
                    additions = set()
                if additions:
                    reachable.update(additions)
                    changed = True
        functions = {
            name: function
            for name, function in functions.items()
            if name in reachable
        }
    native_symbols = _fortran_symbol_table(functions)
    extent_namespaces = {
        function_name: hashlib.sha256(
            str(function_name).encode("utf-8")
        ).hexdigest()[:12]
        for function_name in functions
    }
    # Two passes: a subroutine that calls another must pass exactly the
    # extents that one declares, and those are only known once it has been
    # emitted. The first pass is discarded apart from its signatures.
    callee_arity = {
        function_name: len(function.args)
        for function_name, function in functions.items()
    }
    callee_output_count = {
        function_name: len(named_outputs.get(function_name, ()))
        for function_name in functions
    }
    # SSA value ids are function-local. Determine address bases from each
    # function's own operators and explicit ABI contracts; a same-numbered
    # value in a sibling region is not evidence that this value is an array.
    method_array_bases: dict[str, set[int]] = {}
    method_mutated_bases: dict[str, set[int]] = {}
    for function_name, function in functions.items():
        method = str(function_name)
        array_bases = method_array_bases.setdefault(method, set())
        mutated_bases = method_mutated_bases.setdefault(method, set())
        array_bases.update(
            int(argument.id)
            for argument in function.args
            if (
                str((argument.accounting or {}).get("program_abi_storage"))
                == "span"
                or int((argument.accounting or {}).get(
                    "program_abi_rank", 0
                )) > 0
                or int((argument.accounting or {}).get(
                    "ssa_call_rank", 0
                )) > 0
            )
        )
        sequence_arrays = set(map(
            int, function.metadata.get("sequence_array_argument_ids", ())
        ))
        array_bases.update(sequence_arrays)
        mutated_bases.update(sequence_arrays)
        address_to_base: dict[int, int] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op in ("GetElementPtr", "getelementptr")
                    and instruction.args
                    and instruction.attributes.get("aggregate_index") is None
                ):
                    array_bases.add(int(instruction.args[0].id))
                    if instruction.res is not None:
                        address_to_base[int(instruction.res.id)] = int(
                            instruction.args[0].id
                        )
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op in ("Store", "store") and len(instruction.args) == 2:
                    base = address_to_base.get(int(instruction.args[1].id))
                    if base is not None:
                        mutated_bases.add(base)
    # Rank follows actual call edges by argument position. Numeric ids remain
    # function-local: only the caller operand and the corresponding callee
    # parameter are unified. This reaches pass-through wrappers without
    # treating same-numbered values in sibling regions as aliases.
    function_value_ranks: dict[str, dict[int, int]] = {}
    for function_name, function in functions.items():
        ranks: dict[int, int] = {}
        for value in (
            *function.args,
            *(
                instruction.res
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            ),
        ):
            ranks[int(value.id)] = max(
                ranks.get(int(value.id), 0),
                len(tuple(value.shape or ())),
                int((value.accounting or {}).get("program_abi_rank", 0)),
                int((value.accounting or {}).get("ssa_call_rank", 0)),
            )
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op in {"GetElementPtr", "getelementptr"}
                    and instruction.args
                    and instruction.attributes.get("aggregate_index") is None
                ):
                    base_id = int(instruction.args[0].id)
                    ranks[base_id] = max(
                        ranks.get(base_id, 0), len(instruction.args) - 1
                    )
        for base_id in method_array_bases.get(str(function_name), set()):
            ranks[int(base_id)] = max(ranks.get(int(base_id), 0), 1)
        function_value_ranks[str(function_name)] = ranks
    changed_ranks = True

    def is_scalar_extraction(instruction: Instr, operation: str) -> bool:
        identity = str(
            instruction.attributes.get("extraction_identity") or ""
        )
        source_operator = str(
            instruction.attributes.get("source_operator") or ""
        ).casefold()
        return (
            operation in {"Cast", "cast"}
            and (
                identity in {"builtins.float", "builtins.int", "builtins.bool"}
                or source_operator in {"float", "int", "bool"}
            )
        )

    def unify_rank(
        left_ranks: dict[int, int],
        left: SSAValue,
        right_ranks: dict[int, int],
        right: SSAValue,
    ) -> bool:
        left_id = int(left.id)
        right_id = int(right.id)
        rank = max(
            left_ranks.get(left_id, 0), right_ranks.get(right_id, 0)
        )
        changed = False
        if left_ranks.get(left_id, 0) != rank:
            left_ranks[left_id] = rank
            changed = True
        if right_ranks.get(right_id, 0) != rank:
            right_ranks[right_id] = rank
            changed = True
        return changed

    while changed_ranks:
        changed_ranks = False
        for caller_name, caller in functions.items():
            caller_ranks = function_value_ranks[str(caller_name)]
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    operation = str(
                        instruction.attributes.get("tensor_operation")
                        or instruction.op
                    )
                    if (
                        is_scalar_extraction(instruction, operation)
                        and instruction.res is not None
                    ):
                        # Scalar Python constructors consume one value even
                        # when their source is represented by a singleton
                        # arena. Do not inherit the source arena's rank.
                        pass
                    elif (
                        operation in {"CastLike", "cast_like"}
                        and instruction.res is not None
                        and instruction.args
                    ):
                        result_id = int(instruction.res.id)
                        value_rank = max(
                            caller_ranks.get(result_id, 0),
                            caller_ranks.get(int(instruction.args[0].id), 0),
                        )
                        if caller_ranks.get(result_id, 0) != value_rank:
                            caller_ranks[result_id] = value_rank
                            changed_ranks = True
                    elif (
                        instruction.res is not None
                        and instruction.args
                        and (
                            operation in _BINARY
                            or operation in _UNARY
                            or operation in _SHAPE_ONLY
                            or operation in {"Cast", "cast", "where"}
                        )
                    ):
                        result_id = int(instruction.res.id)
                        result_rank = max(
                            caller_ranks.get(result_id, 0),
                            *(caller_ranks.get(int(value.id), 0)
                              for value in instruction.args),
                        )
                        if caller_ranks.get(result_id, 0) != result_rank:
                            caller_ranks[result_id] = result_rank
                            changed_ranks = True
                    if instruction.op not in {"Call", "call"}:
                        continue
                    callee_name = str(
                        instruction.attributes.get("callee") or ""
                    )
                    callee = functions.get(callee_name)
                    if callee is None:
                        continue
                    callee_ranks = function_value_ranks[callee_name]
                    for actual, formal in zip(instruction.args, callee.args):
                        if unify_rank(
                            caller_ranks, actual, callee_ranks, formal
                        ):
                            changed_ranks = True
                    callee_outputs = tuple(named_outputs.get(callee_name, ()))
                    if not callee_outputs:
                        continue
                    caller_outputs: dict[int, SSAValue] = {}
                    if (
                        instruction.res is not None
                        and instruction.attributes.get("result_convention")
                        == "ssa.aggregate"
                    ):
                        addresses = {
                            int(candidate.res.id): int(
                                candidate.attributes["aggregate_index"]
                            )
                            for candidate in block.instrs
                            if (
                                candidate.op in {"GetElementPtr", "getelementptr"}
                                and candidate.res is not None
                                and candidate.args
                                and candidate.args[0] is instruction.res
                                and candidate.attributes.get("aggregate_index")
                                is not None
                            )
                        }
                        caller_outputs = {
                            addresses[int(candidate.args[0].id)]: candidate.res
                            for candidate in block.instrs
                            if (
                                candidate.op in {"Load", "load"}
                                and candidate.res is not None
                                and candidate.args
                                and int(candidate.args[0].id) in addresses
                            )
                        }
                    elif instruction.res is not None and len(callee_outputs) == 1:
                        caller_outputs[0] = instruction.res
                    for output_index, callee_output in enumerate(callee_outputs):
                        caller_output = caller_outputs.get(output_index)
                        if caller_output is not None and unify_rank(
                            caller_ranks,
                            caller_output,
                            callee_ranks,
                            callee_output,
                        ):
                            changed_ranks = True
    scalar_control_ids: dict[str, set[int]] = {}
    for function_name, function in functions.items():
        named_scalar_parameters = {
            str(name)
            for name, receipt in dict(
                function.metadata.get("parameter_value_abi") or {}
            ).items()
            if int(receipt.get("rank", 0)) == 0
        }
        parameter_names = {
            int(value_id): str(name)
            for name, value_id in function.metadata.get(
                "parameter_names", ()
            )
        }
        explicit_scalars = {
            int(value.id)
            for value in function.args
            if (
                str((value.accounting or {}).get("program_abi_storage"))
                == "scalar"
                or parameter_names.get(int(value.id))
                in named_scalar_parameters
            )
        }
        explicit_scalars.update(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if (
                instruction.res is not None
                and is_scalar_extraction(
                    instruction,
                    str(
                        instruction.attributes.get("tensor_operation")
                        or instruction.op
                    ),
                )
            )
        )
        scalar_control_ids[function_name] = explicit_scalars | {
            int(instruction.args[0].id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"CondBr", "condbr"} and instruction.args
        }
    direct_scalar_control_ids = {
        function_name: set(value_ids)
        for function_name, value_ids in scalar_control_ids.items()
    }
    changed_scalars = True
    while changed_scalars:
        changed_scalars = False
        for function_name, function in functions.items():
            scalar_ids = scalar_control_ids[function_name]
            for block in function.blocks.values():
                for instruction in block.instrs:
                    operation = str(
                        instruction.attributes.get("tensor_operation")
                        or instruction.op
                    )
                    if (
                        operation in {"CastLike", "cast_like"}
                        and instruction.res is not None
                        and len(instruction.args) >= 2
                        and int(instruction.args[1].id) in scalar_ids
                    ):
                        for value in (
                            instruction.args[0], instruction.res
                        ):
                            if int(value.id) not in scalar_ids:
                                scalar_ids.add(int(value.id))
                                changed_scalars = True
                    if (
                        instruction.res is not None
                        and int(instruction.res.id) in scalar_ids
                        and (
                            operation in _BINARY
                            or operation in _UNARY
                            or operation in _SHAPE_ONLY
                            or instruction.op in {"Phi", "phi"}
                        )
                    ):
                        for argument in instruction.args:
                            if int(argument.id) not in scalar_ids:
                                scalar_ids.add(int(argument.id))
                                changed_scalars = True
                    if instruction.op not in {"Call", "call"}:
                        continue
                    callee_name = str(instruction.attributes.get("callee") or "")
                    callee = functions.get(callee_name)
                    if callee is None:
                        continue
                    callee_scalars = scalar_control_ids[callee_name]
                    for actual, formal in zip(instruction.args, callee.args):
                        actual_id, formal_id = int(actual.id), int(formal.id)
                        if actual_id in scalar_ids or formal_id in callee_scalars:
                            if actual_id not in scalar_ids:
                                scalar_ids.add(actual_id)
                                changed_scalars = True
                            if formal_id not in callee_scalars:
                                callee_scalars.add(formal_id)
                                changed_scalars = True
                        if (
                            actual_id in direct_scalar_control_ids[function_name]
                            or formal_id in direct_scalar_control_ids[callee_name]
                        ):
                            if actual_id not in direct_scalar_control_ids[function_name]:
                                direct_scalar_control_ids[function_name].add(actual_id)
                                changed_scalars = True
                            if formal_id not in direct_scalar_control_ids[callee_name]:
                                direct_scalar_control_ids[callee_name].add(formal_id)
                                changed_scalars = True
                    if instruction.res is None:
                        continue
                    addresses = {
                        int(candidate.res.id): int(candidate.attributes["aggregate_index"])
                        for candidate in block.instrs
                        if (
                            candidate.op in {"GetElementPtr", "getelementptr"}
                            and candidate.res is not None
                            and candidate.args
                            and candidate.args[0] is instruction.res
                            and candidate.attributes.get("aggregate_index") is not None
                        )
                    }
                    caller_outputs = {
                        addresses[int(candidate.args[0].id)]: candidate.res
                        for candidate in block.instrs
                        if (
                            candidate.op in {"Load", "load"}
                            and candidate.res is not None
                            and candidate.args
                            and int(candidate.args[0].id) in addresses
                        )
                    }
                    for output_index, callee_output in enumerate(
                        named_outputs.get(callee_name, ())
                    ):
                        caller_output = caller_outputs.get(output_index)
                        if caller_output is None:
                            continue
                        caller_id, callee_id = (
                            int(caller_output.id), int(callee_output.id)
                        )
                        if caller_id in scalar_ids or callee_id in callee_scalars:
                            if caller_id not in scalar_ids:
                                scalar_ids.add(caller_id)
                                changed_scalars = True
                            if callee_id not in callee_scalars:
                                callee_scalars.add(callee_id)
                                changed_scalars = True
                        if (
                            caller_id in direct_scalar_control_ids[function_name]
                            or callee_id in direct_scalar_control_ids[callee_name]
                        ):
                            if caller_id not in direct_scalar_control_ids[function_name]:
                                direct_scalar_control_ids[function_name].add(caller_id)
                                changed_scalars = True
                            if callee_id not in direct_scalar_control_ids[callee_name]:
                                direct_scalar_control_ids[callee_name].add(callee_id)
                                changed_scalars = True
    array_contract_ids = {
        function_name: set(map(int, method_array_bases.get(function_name, set())))
        for function_name in functions
    }
    changed_array_contracts = True
    while changed_array_contracts:
        changed_array_contracts = False
        for caller_name, caller in functions.items():
            caller_arrays = array_contract_ids[caller_name]
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in {"Call", "call"}:
                        continue
                    callee_name = str(instruction.attributes.get("callee") or "")
                    callee = functions.get(callee_name)
                    if callee is None:
                        continue
                    callee_arrays = array_contract_ids[callee_name]
                    for actual, formal in zip(instruction.args, callee.args):
                        actual_id, formal_id = int(actual.id), int(formal.id)
                        if actual_id in caller_arrays or formal_id in callee_arrays:
                            if actual_id not in caller_arrays:
                                caller_arrays.add(actual_id)
                                changed_array_contracts = True
                            if formal_id not in callee_arrays:
                                callee_arrays.add(formal_id)
                                changed_array_contracts = True
                    if instruction.res is None:
                        continue
                    addresses = {
                        int(candidate.res.id): int(candidate.attributes["aggregate_index"])
                        for candidate in block.instrs
                        if (
                            candidate.op in {"GetElementPtr", "getelementptr"}
                            and candidate.res is not None
                            and candidate.args
                            and candidate.args[0] is instruction.res
                            and candidate.attributes.get("aggregate_index") is not None
                        )
                    }
                    caller_outputs = {
                        addresses[int(candidate.args[0].id)]: candidate.res
                        for candidate in block.instrs
                        if (
                            candidate.op in {"Load", "load"}
                            and candidate.res is not None
                            and candidate.args
                            and int(candidate.args[0].id) in addresses
                        )
                    }
                    for output_index, callee_output in enumerate(
                        named_outputs.get(callee_name, ())
                    ):
                        caller_output = caller_outputs.get(output_index)
                        if caller_output is None:
                            continue
                        caller_id, callee_id = int(caller_output.id), int(callee_output.id)
                        if caller_id in caller_arrays or callee_id in callee_arrays:
                            if caller_id not in caller_arrays:
                                caller_arrays.add(caller_id)
                                changed_array_contracts = True
                            if callee_id not in callee_arrays:
                                callee_arrays.add(callee_id)
                                changed_array_contracts = True
    for function_name, scalar_ids in scalar_control_ids.items():
        for value_id in scalar_ids:
            if value_id not in array_contract_ids[function_name]:
                function_value_ranks[function_name][value_id] = 0
    for function_name, array_ids in array_contract_ids.items():
        protected_scalars = (
            direct_scalar_control_ids[function_name]
            - set(map(int, method_array_bases.get(function_name, set())))
            - set(map(int, array_ids))
        )
        for value_id in array_ids:
            if value_id in protected_scalars:
                continue
            function_value_ranks[function_name][value_id] = max(
                1, function_value_ranks[function_name].get(value_id, 0)
            )
    # Dtypes obey the same graph-wide signature constraints as ranks.  In
    # particular, aggregate loads frequently carry no dtype occurrence of
    # their own, while logical consumers provide stronger evidence than a
    # provisional numeric default inherited from Python scalar handling.
    function_value_dtypes: dict[str, dict[int, str]] = {}
    logical_dtype_ids: dict[str, set[int]] = {}
    for function_name, function in functions.items():
        dtypes: dict[int, str] = {}
        logical_ids: set[int] = set()
        values = [*function.args, *named_outputs.get(function_name, ())]
        values.extend(
            value
            for block in function.blocks.values()
            for instruction in block.instrs
            for value in (
                *instruction.args,
                *((instruction.res,) if instruction.res is not None else ()),
            )
        )
        for value in values:
            candidate = str(value.dtype or "")
            if candidate and candidate not in {"unknown", "aggregate", "ptr", "pointer", "void"}:
                dtypes.setdefault(int(value.id), candidate)
        for block in function.blocks.values():
            for instruction in block.instrs:
                operation = str(
                    instruction.attributes.get("tensor_operation")
                    or instruction.op
                )
                if (
                    operation in {"CastLike", "cast_like"}
                    and instruction.res is not None
                    and len(instruction.args) >= 2
                ):
                    reference_dtype = dtypes.get(
                        int(instruction.args[1].id)
                    )
                    if reference_dtype is not None:
                        dtypes[int(instruction.res.id)] = reference_dtype
                if operation in {
                    "LAnd", "LOr", "logical_and", "logical_or"
                }:
                    if instruction.res is not None:
                        logical_ids.add(int(instruction.res.id))
                elif operation in {"LNot", "Not", "logical_not"}:
                    logical_ids.update(int(value.id) for value in instruction.args)
                    if instruction.res is not None:
                        logical_ids.add(int(instruction.res.id))
                elif operation in _COMPARISON and instruction.res is not None:
                    logical_ids.add(int(instruction.res.id))
                if instruction.op in {"CondBr", "condbr"} and instruction.args:
                    logical_ids.add(int(instruction.args[0].id))
        for value_id in logical_ids:
            dtypes[value_id] = "bool"
        function_value_dtypes[str(function_name)] = dtypes
        logical_dtype_ids[str(function_name)] = logical_ids

    changed_dtypes = True
    while changed_dtypes:
        changed_dtypes = False
        for caller_name, caller in functions.items():
            caller_dtypes = function_value_dtypes[str(caller_name)]
            caller_logical = logical_dtype_ids[str(caller_name)]
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in {"Call", "call"}:
                        continue
                    callee_name = str(instruction.attributes.get("callee") or "")
                    callee = functions.get(callee_name)
                    if callee is None:
                        continue
                    callee_dtypes = function_value_dtypes[callee_name]
                    callee_logical = logical_dtype_ids[callee_name]
                    for actual, formal in zip(instruction.args, callee.args):
                        actual_id, formal_id = int(actual.id), int(formal.id)
                        if actual_id in caller_logical or formal_id in callee_logical:
                            settled = "bool"
                            caller_logical.add(actual_id)
                            callee_logical.add(formal_id)
                        else:
                            actual_dtype = caller_dtypes.get(actual_id)
                            formal_dtype = callee_dtypes.get(formal_id)
                            if actual_dtype is None and formal_dtype is not None:
                                caller_dtypes[actual_id] = formal_dtype
                                changed_dtypes = True
                            elif formal_dtype is None and actual_dtype is not None:
                                callee_dtypes[formal_id] = actual_dtype
                                changed_dtypes = True
                            continue
                        if settled and caller_dtypes.get(actual_id) != settled:
                            caller_dtypes[actual_id] = settled
                            changed_dtypes = True
                        if settled and callee_dtypes.get(formal_id) != settled:
                            callee_dtypes[formal_id] = settled
                            changed_dtypes = True
                    callee_output_values = tuple(named_outputs.get(callee_name, ()))
                    if not callee_output_values or instruction.res is None:
                        continue
                    addresses = {
                        int(candidate.res.id): int(candidate.attributes["aggregate_index"])
                        for candidate in block.instrs
                        if (
                            candidate.op in {"GetElementPtr", "getelementptr"}
                            and candidate.res is not None
                            and candidate.args
                            and candidate.args[0] is instruction.res
                            and candidate.attributes.get("aggregate_index") is not None
                        )
                    }
                    caller_outputs = {
                        addresses[int(candidate.args[0].id)]: candidate.res
                        for candidate in block.instrs
                        if (
                            candidate.op in {"Load", "load"}
                            and candidate.res is not None
                            and candidate.args
                            and int(candidate.args[0].id) in addresses
                        )
                    }
                    for output_index, callee_output in enumerate(callee_output_values):
                        caller_output = caller_outputs.get(output_index)
                        if caller_output is None:
                            continue
                        caller_id = int(caller_output.id)
                        callee_id = int(callee_output.id)
                        if caller_id in caller_logical or callee_id in callee_logical:
                            settled = "bool"
                            caller_logical.add(caller_id)
                            callee_logical.add(callee_id)
                        else:
                            caller_dtype = caller_dtypes.get(caller_id)
                            callee_dtype = callee_dtypes.get(callee_id)
                            if callee_dtype is not None and caller_dtype != callee_dtype:
                                caller_dtypes[caller_id] = callee_dtype
                                changed_dtypes = True
                            elif callee_dtype is None and caller_dtype is not None:
                                callee_dtypes[callee_id] = caller_dtype
                                changed_dtypes = True
                            continue
                        if settled and caller_dtypes.get(caller_id) != settled:
                            caller_dtypes[caller_id] = settled
                            changed_dtypes = True
                        if settled and callee_dtypes.get(callee_id) != settled:
                            callee_dtypes[callee_id] = settled
                            changed_dtypes = True
    for function_name, ranks in function_value_ranks.items():
        method_array_bases.setdefault(function_name, set()).update(
            value_id for value_id, rank in ranks.items() if rank > 0
        )
    function_mutated_values = {
        function_name: set(method_mutated_bases.get(function_name, set())) | {
            int(argument.id)
            for argument in function.args
            if (
                any(
                    int(argument.id) == int(output.id)
                    for output in named_outputs.get(function_name, ())
                )
                or any(
                    instruction.res is not None
                    and int(instruction.res.id) == int(argument.id)
                    for block in function.blocks.values()
                    for instruction in block.instrs
                )
            )
        }
        for function_name, function in functions.items()
    }
    changed_mutation = True
    while changed_mutation:
        changed_mutation = False
        for caller_name, caller in functions.items():
            caller_mutated = function_mutated_values[caller_name]
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in {"Call", "call"}:
                        continue
                    callee_name = str(
                        instruction.attributes.get("callee") or ""
                    )
                    callee = functions.get(callee_name)
                    if callee is None:
                        continue
                    callee_mutated = function_mutated_values[callee_name]
                    for actual, formal in zip(instruction.args, callee.args):
                        if (
                            int(formal.id) in callee_mutated
                            and int(actual.id) not in caller_mutated
                        ):
                            caller_mutated.add(int(actual.id))
                            changed_mutation = True
    method_mutated_bases = function_mutated_values
    callee_inout_pairs = {
        function_name: tuple(
            (input_index, output_index)
            for input_index, argument in enumerate(function.args)
            for output_index, output in enumerate(
                named_outputs.get(function_name, ())
            )
            if int(argument.id) == int(output.id)
        )
        for function_name, function in functions.items()
    }
    callee_array_arguments = {
        function_name: tuple(
            position
            for position, argument in enumerate(function.args)
            if int(argument.id) not in set(map(
                int,
                function.metadata.get("scalar_variant_argument_ids", ()),
            ))
            and (
                int(argument.id) in method_array_bases.get(
                    function_name, set()
                )
                or int((argument.accounting or {}).get(
                    "program_abi_rank", 0
                )) > 0
                or int((argument.accounting or {}).get(
                    "ssa_call_rank", 0
                )) > 0
                or function_value_ranks.get(function_name, {}).get(
                    int(argument.id), 0
                ) > 0
                or _is_array(argument)
            )
        )
        for function_name, function in functions.items()
    }
    # Extents are a finite set-union dataflow problem. Emit each function once
    # to discover only its local requirements, then close those sets over the
    # call graph. Re-emitting every function on every iteration is equivalent
    # but needlessly rebuilds enormous source strings for large programs.
    local_extents = {
            function_name: emit_function(
                function,
                dtype=dtype,
                trig_solver=trig_solver,
                outputs=named_outputs.get(function_name, ()),
                callee_extents={},
                callee_arity=callee_arity,
                callee_output_count=callee_output_count,
                callee_outputs=named_outputs,
                callee_arguments={
                    name: candidate.args for name, candidate in functions.items()
                },
                callee_array_arguments=callee_array_arguments,
                callee_inout_pairs=callee_inout_pairs,
                array_base_ids=(
                    method_array_bases.get(function_name, set())
                    - set(map(
                        int,
                        function.metadata.get(
                            "scalar_variant_argument_ids", ()
                        ),
                    ))
                ),
                mutated_base_ids=method_mutated_bases.get(
                    function_name, set()
                ),
                dynamic_array_ranks=function_value_ranks.get(
                    function_name, {}
                ),
                value_dtypes=function_value_dtypes.get(function_name, {}),
                tensor_table=module_tensor_tables.get(function_name),
                native_symbol=native_symbols[function_name],
                callee_native_symbols=native_symbols,
                extent_namespace=extent_namespaces[function_name],
            ).extent_names
            for function_name, function in functions.items()
    }
    direct_callees = {
        function_name: {
            str(instruction.attributes.get("callee") or "")
            for block in function.blocks.values()
            for instruction in block.instrs
            if (
                instruction.op in {"Call", "call"}
                and str(instruction.attributes.get("callee") or "") in functions
            )
        }
        for function_name, function in functions.items()
    }
    callee_extents: dict[str, tuple[str, ...]] = dict(local_extents)
    for _ in range(max(1, len(functions) + 1)):
        updated_extents = {
            function_name: tuple(sorted(
                set(local_extents[function_name]).union(*(
                    set(callee_extents[callee])
                    for callee in direct_callees[function_name]
                ))
            ))
            for function_name in functions
        }
        if updated_extents == callee_extents:
            break
        callee_extents = updated_extents
    else:  # pragma: no cover - finite extent-name unions must converge.
        raise RuntimeError("Fortran call ABI extents did not reach a fixed point")
    def emit_subroutines(
        signatures: Mapping[str, Sequence[str]],
    ) -> tuple[FortranSubroutine, ...]:
        return tuple(
            emit_function(
                function,
                trig_solver=trig_solver,
                dtype=dtype,
                outputs=named_outputs.get(function_name, ()),
                callee_extents=signatures,
                callee_arity=callee_arity,
                callee_output_count=callee_output_count,
                callee_outputs=named_outputs,
                callee_arguments={
                    key: candidate.args for key, candidate in functions.items()
                },
                callee_array_arguments=callee_array_arguments,
                callee_inout_pairs=callee_inout_pairs,
                array_base_ids=(
                    method_array_bases.get(function_name, set())
                    - set(map(
                        int,
                        function.metadata.get("scalar_variant_argument_ids", ()),
                    ))
                ),
                mutated_base_ids=method_mutated_bases.get(function_name, set()),
                dynamic_array_ranks=function_value_ranks.get(function_name, {}),
                value_dtypes=function_value_dtypes.get(function_name, {}),
                tensor_table=module_tensor_tables.get(function_name),
                native_symbol=native_symbols[function_name],
                callee_native_symbols=native_symbols,
                extent_namespace=extent_namespaces[function_name],
            )
            for function_name, function in functions.items()
        )

    for _ in range(max(1, len(functions) + 1)):
        subroutines = emit_subroutines(callee_extents)
        emitted_extents = {
            function_name: subroutine.extent_names
            for function_name, subroutine in zip(functions, subroutines)
        }
        if emitted_extents == callee_extents:
            break
        callee_extents = emitted_extents
    else:  # pragma: no cover - final signatures must stabilize.
        raise RuntimeError("Fortran emitted extents did not stabilize")
    emitted_dynamic_arrays = {
        function_name: dict(subroutine.dynamic_array_extents)
        for function_name, subroutine in zip(functions, subroutines)
    }
    emitted_dynamic_dimensions = {
        function_name: dict(subroutine.dynamic_array_dimensions)
        for function_name, subroutine in zip(functions, subroutines)
    }
    emitted_reference_arguments = {
        function_name: subroutine.reference_argument_ids
        for function_name, subroutine in zip(functions, subroutines)
    }
    lines = [
        f"module {name}",
        "  use, intrinsic :: iso_c_binding",
        "  use, intrinsic :: ieee_arithmetic",
        "  implicit none",
        "contains",
        "",
        *[sub.source for sub in subroutines],
        "",
        f"end module {name}",
    ]
    # Describe what was generated, from the same Function objects, so callers
    # stop rediscovering the signature by reading emitted source.
    from .compiled_program_api import CompiledProgramAPI, describe_fortran_function
    from .output_publication import publication_metadata

    def _kind(function_name: str) -> str:
        function = functions[function_name]
        if function.metadata.get("control_ir"):
            return "control"
        if function_name.startswith("numerical_region_"):
            return "region"
        if function_name.endswith("_control"):
            return "control"
        return "numerical"

    entry_points = []
    for function_name, function in functions.items():
        kind = _kind(function_name)
        note = None
        if kind == "control":
            note = (
                "call this one: a program whose control shell has a loop "
                "iterates here, and calling the numerical subroutine "
                "directly runs the loop body once"
            )
        source_names = {
            int(value_id): str(source_name)
            for source_name, value_id in function.metadata.get(
                "named_outputs", ()
            )
        }
        # These are declared function parameters, not aliases selected by
        # identity-table ordering. They authoritatively name ABI inputs.
        source_names.update({
            int(value_id): str(source_name)
            for source_name, value_id in function.metadata.get(
                "parameter_names", ()
            )
        })
        # Record-valued parameters are expanded into fields and carry their
        # own accounting names. Scalar source parameters can survive that
        # expansion without a ``parameter_names`` pair, while the extraction
        # contract still retains their ordered ``parameter_value_abi``. Use
        # that receipt only when the remaining source arguments match it
        # exactly; ambiguity must stay visible rather than guessed by id.
        scalar_abi_names = tuple(
            str(source_name)
            for source_name, receipt in dict(
                function.metadata.get("parameter_value_abi") or {}
            ).items()
            if str(receipt.get("function") or "") in {
                "",
                str(function_name),
                str(function_name).rsplit("__", 1)[-1],
            }
        )
        unnamed_source_args = tuple(
            value
            for value in function.args
            if (
                int(value.id) not in source_names
                and not (value.accounting or {}).get("program_abi_parameter")
                and not (value.accounting or {}).get("linked_call_frame_storage")
                and not (value.accounting or {}).get("returned_record_storage")
            )
        )
        if scalar_abi_names and len(unnamed_source_args) == len(scalar_abi_names):
            source_names.update(
                (int(value.id), source_name)
                for value, source_name in zip(
                    unnamed_source_args, scalar_abi_names
                )
            )
        entry_points.append(
            describe_fortran_function(
                function_name,
                function,
                # Use the final second-pass signature. A caller can acquire
                # transitive extent arguments from a callee even when its
                # own values do not have that shape; the discarded first
                # pass deliberately cannot see those call requirements.
                extent_names=emitted_extents.get(function_name, ()),
                outputs=named_outputs.get(function_name, ()),
                kind=kind,
                note=note,
                source_names=source_names,
                dynamic_array_extents=emitted_dynamic_arrays.get(
                    function_name, {}
                ),
                dynamic_array_dimensions=emitted_dynamic_dimensions.get(
                    function_name, {}
                ),
                array_argument_ids=(
                    int(function.args[position].id)
                    for position in callee_array_arguments.get(
                        function_name, ()
                    )
                ),
                reference_argument_ids=emitted_reference_arguments.get(
                    function_name, ()
                ),
            )
        )
    control_entries = [e.name for e in entry_points if e.kind == "control"]
    api = CompiledProgramAPI(
        module=name,
        language="fortran",
        entry=control_entries[0] if control_entries else None,
        entry_points=tuple(entry_points),
        metadata={
            "dtype": dtype,
            "fortran_internal_symbols": dict(native_symbols),
            **publication_metadata(functions),
            "tensor_table_schema": "turing.repository-ssa-tensor-table.v1",
            "tensor_tables": {
                function_name: [
                    descriptor.to_mapping()
                    for descriptor in table.tensors.values()
                ]
                for function_name, table in module_tensor_tables.items()
                if function_name in functions and table.tensors
            },
            # Sequence descriptors are the public memory contract for every
            # lowered list/set/dict/table.  Publishing them beside the tensor
            # table lets a non-Python caller allocate the exact resident
            # columns and mutable length/status cells the emitted SSA uses;
            # otherwise the generated signature exposes anonymous pointers
            # without their row layout, uniqueness, or capacity semantics.
            "sequence_table_schema": "turing.repository-ssa-sequence-table.v1",
            "sequence_tables": {
                function_name: [
                    {
                        **descriptor.to_mapping(),
                        "source_names": list(dict(
                            functions[function_name].metadata.get(
                                "sequence_value_names", ()
                            )
                        ).get(int(descriptor.sequence_id), ())),
                    }
                    for descriptor in table.sequences.values()
                ]
                for function_name, table in module_sequence_tables.items()
                if function_name in functions and table.sequences
            },
            "class_table_schema": "turing.repository-ssa-class-table.v1",
            "class_table": [
                {
                    "identity": record.identity,
                    "fields": [
                        {"name": field.name, "slot": int(field.slot)}
                        for field in record.fields
                    ],
                    "methods": [
                        {
                            "name": method.name,
                            "function_reference": int(
                                method.function_reference
                            ),
                            "function_name": method.function_name,
                        }
                        for method in record.methods
                    ],
                }
                for record in getattr(module_class_table, "classes", ())
            ],
            "function_table_schema": "turing.repository-ssa-function-table.v1",
            "function_table": [
                {
                    "reference": int(entry.reference.address),
                    "name": entry.name,
                    "qualified_name": entry.qualified_name,
                    "state": entry.state.value,
                    "recursive": bool(entry.recursive),
                    "parameter_contracts": [
                        {
                            "name": contract.name,
                            "transfer": contract.transfer.value,
                            "access": contract.access.value,
                            "storage": contract.storage.value,
                            "scope": contract.scope.value,
                        }
                        for contract in entry.parameter_contracts
                    ],
                }
                for entry in (module_function_table or ())
            ],
            "record_table_schema": "turing.repository-ssa-record-table.v1",
            "record_tables": {
                function_name: [
                    descriptor.to_mapping()
                    for descriptor in table.records.values()
                ]
                for function_name, table in module_record_tables.items()
                if function_name in functions and table.records
            },
        },
    )
    return FortranModule(name, "\n".join(lines) + "\n", subroutines, api=api)


# Toolchains that are commonly installed but left off PATH on Windows.  A
# compiler that exists is worth finding: without one the compile-and-run path
# silently disappears and emitted code goes unverified.
_KNOWN_COMPILER_PATHS = (
    r"C:\msys64\mingw64\bin\gfortran.exe",
    r"C:\MinGW\bin\gfortran.exe",
    r"C:\w64devkit\bin\gfortran.exe",
    r"C:\ProgramData\chocolatey\bin\gfortran.exe",
)


def fortran_compiler() -> str | None:
    """Return an available Fortran compiler, or ``None``.

    Emission never requires one.  This exists so callers can decide whether the
    compile-and-run path is available rather than discovering it by failure.
    ``TURING_FC`` overrides the search.
    """

    import os

    override = os.environ.get("TURING_FC")
    if override and Path(override).exists():
        return override
    for candidate in ("gfortran", "ifx", "ifort", "flang", "lfortran"):
        found = shutil.which(candidate)
        if found:
            return found
    for candidate in _KNOWN_COMPILER_PATHS:
        if Path(candidate).exists():
            return candidate
    return None


def compile_module(
    module: FortranModule,
    *,
    directory: str | Path | None = None,
    extra_flags: Sequence[str] | None = None,
    standalone: bool = True,
) -> Path:
    """Compile a generated module to a shared library.

    Raises ``FortranEmissionError`` when no compiler is present; callers that
    only need source should not call this.
    """

    compiler = fortran_compiler()
    if compiler is None:
        raise FortranEmissionError(
            "no Fortran compiler found; emission does not require one, "
            "but compile_module does"
        )
    import os
    import sys
    from .fortran_toolchain import (
        aggressive_fortran_flags,
        standalone_fortran_link_flags,
        standalone_runtime_shim_sources,
    )

    workdir = Path(
        directory or tempfile.mkdtemp(prefix="turing_fortran_")
    ).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    source = module.write(workdir)
    suffix = ".dll" if sys.platform == "win32" else ".so"
    library = workdir / f"{module.name}{suffix}"
    compile_flags = tuple(extra_flags or aggressive_fortran_flags(compiler))
    try:
        link_flags = (
            standalone_fortran_link_flags(compiler) if standalone else ()
        )
    except ValueError as error:
        raise FortranEmissionError(str(error)) from error
    shim_sources = standalone_runtime_shim_sources(
        compiler, workdir, standalone
    )
    if shim_sources and os.name == "nt":
        # GCC's LTO plugin publishes the shim's weak PE symbol as a different
        # symbol type from the archive reference.  Compile this small static
        # runtime link without LTO; ordinary optimized modules keep LTO.
        compile_flags = tuple(flag for flag in compile_flags if flag != "-flto")
        link_flags = tuple(flag for flag in link_flags if flag != "-flto")
    command = [
        compiler,
        "-shared",
        *(() if sys.platform == "win32" else ("-fPIC",)),
        *compile_flags,
        str(source),
        *shim_sources,
        "-o",
        str(library),
        *link_flags,
    ]

    # gfortran spawns f951, which loads libgmp/libmpfr from the toolchain's own
    # bin directory.  When the compiler is invoked by absolute path and that
    # directory is not on PATH, f951 fails to load and gfortran exits non-zero
    # with no diagnostic at all.  Put it on PATH for the child.
    environment = dict(os.environ)
    compiler_bin = str(Path(compiler).parent)
    environment["PATH"] = compiler_bin + os.pathsep + environment.get("PATH", "")

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(workdir),
        env=environment,
    )
    if completed.returncode != 0:
        raise FortranEmissionError(
            f"Fortran compilation failed:\n{completed.stderr}"
        )
    return library


__all__ = [
    "DEFAULT_DTYPE",
    "FortranEmissionError",
    "FortranModule",
    "FortranShortfall",
    "FortranSubroutine",
    "compile_module",
    "dimension_extents",
    "emit_function",
    "emit_module",
    "fortran_compiler",
    "supported_tensor_operations",
]
