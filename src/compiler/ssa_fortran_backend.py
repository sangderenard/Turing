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
import subprocess
import sys
import tempfile
import time
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
    # Repository/source ``>>`` follows Python signed-integer semantics.  A
    # zero-filling machine shift must arrive as a distinct legalized machine
    # operation; spelling the universal operator as SHIFTR makes negative
    # loop-carried values grow large and can make termination impossible.
    "Shr": "shifta({0}, {1})",
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
    "shr": "shifta({0}, {1})",
}

_UNARY: dict[str, str] = {
    "Neg": "(-{0})",
    "Abs": "abs({0})",
    "BitLength": "turing_python_bit_length(int({0}, c_int64_t))",
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
    "Exp": "exp({0})",
    "log": "log({0})",
    "Log": "log({0})",
    # The vehicle body uses tanh 21 times; the SSA spells it "Tanh" (the
    # lowercase "tanh" below is the recorded-tape vocabulary, a different
    # table), so without this entry the whole body was unemittable.
    "Tanh": "tanh({0})",
    "sin": "sin({0})",
    "cos": "cos({0})",
    "tan": "tan({0})",
    "asin": "asin({0})",
    "acos": "acos({0})",
    "atan": "atan({0})",
    "sinh": "sinh({0})",
    "cosh": "cosh({0})",
    "tanh": "tanh({0})",
    # Fortran has no sigmoid intrinsic. Written through tanh rather than as
    # 1/(1+exp(-x)) because tanh is elemental, saturates instead of
    # overflowing, and the identity is exact: sigmoid(x) = (1 + tanh(x/2))/2.
    "sigmoid": "(0.5_c_double * (1.0_c_double + tanh(0.5_c_double * ({0}))))",
    "asinh": "asinh({0})",
    "acosh": "acosh({0})",
    "atanh": "atanh({0})",
    # FLOOR/CEILING/NINT return INTEGER in Fortran, where the numpy
    # equivalents return a float. Keeping the recorded program's type means
    # converting back, which also stops these from poisoning every intrinsic
    # downstream that then sees mixed INTEGER and REAL operands. Fortran
    # assignment converts on the way into an integer variable, so this is
    # safe even when the result is declared integer.
    # FLOOR() returns a default INTEGER, which is four bytes: an argument
    # past about two billion overflows it and the result is nonsense
    # rather than an error. Range reduction produces exactly such
    # arguments -- a quarter-turn count near a trillion -- and measured,
    # this returned 2.98e+189 for sin(1e12) while the same kernel was
    # exact at 0.3. Naming a wider integer kind only moves the cliff to
    # nine quintillion; AINT stays in the reals, where a double is
    # already integral above 2**52 and the whole question dissolves. The
    # MERGE supplies the difference between truncation and flooring,
    # which is the one place they disagree: a negative with a fraction.
    "floor": (
        "(aint({0}) - merge(1.0_c_double, 0.0_c_double, "
        "{0} < aint({0})))"
    ),
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
#: Three-operand intrinsics. Kept out of ``_BINARY`` deliberately: five
#: other sites read membership there as "elementwise with two operands" --
#: conforming, broadcast decisions, the capability sets -- and every one of
#: them would be wrong about a ternary.
#:
#: ``ieee_fma`` is F2018, from the IEEE_ARITHMETIC module this lane already
#: opens for ieee_is_nan, and it is elemental, so it conforms over arrays
#: the same way an operator does. A compiler predating F2018 rejects it,
#: which is a real shortfall and not a reason to expand into a multiply and
#: an add: that rounds twice, and on a precision dual two roundings return
#: exactly zero.
_TERNARY: dict[str, str] = {"Fma": "ieee_fma({0}, {1}, {2})"}

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

#: ``Fma``/``fma`` belong here for the same reason the others do, and the
#: omission was measured: a precision section whose result dtype inference
#: had settled on an integer emitted
#: ``ieee_fma(t43, int(t55, c_int), -t915)``, and gfortran refused it --
#: "there is no specific function for the generic 'ieee_fma'" -- because
#: the generic has no integer form to resolve to. An FMA is a floating
#: primitive by definition; the C lane took the same SSA and computed it
#: correctly only because C promotes silently, so this was a Fortran-only
#: failure over an SSA sloppiness both lanes shared.
_REAL_OPERAND = frozenset(
    {"sign", "floor", "ceil", "round", "trunc", "sqrt", "exp", "log",
     "Fma", "fma"}
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
    scalar = frozenset(_BINARY) | frozenset(_UNARY) | frozenset(_TERNARY)
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
    # The RESOLVED dtype of every SSA formal, in argument order.  A caller
    # must coerce call operands to these -- the declaration is typed by the
    # callee's own local inference, which raw formal occurrences don't carry.
    argument_dtypes: tuple[str, ...] = ()
    # The RESOLVED dtype of every canonical output slot, in slot order.  A
    # caller's projection cell must match these at the call, bridging any
    # difference with an explicit conversion after the call returns.
    output_dtypes: tuple[str, ...] = ()

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
        return _literal(elements[0], dtype)
    constructor = (
        "[" + ", ".join(_literal(element, dtype) for element in elements)
        + "]"
    )
    if len(shape) <= 1:
        return constructor
    extents = ", ".join(str(int(size)) for size in shape)
    return f"reshape({constructor}, [{extents}])"



def sin_table_declaration() -> str:
    """The shared baked table as a Fortran parameter array."""

    from .fused_program_wasm_backend import lut_for

    values, _achieved, _lower, _upper, _periodic = lut_for("sin")
    items = [f"{value!r}" + "_c_double" for value in values]
    # Fortran free-form source has a processor-dependent line limit (132
    # columns in gfortran's default mode).  This LUT is deliberately large,
    # so it must be a continued array constructor rather than one giant line.
    width = 120
    lines = [
        f"    real(c_double), parameter :: turing_sin_table(0:{len(values) - 1}) = [ &"
    ]
    chunk: list[str] = []
    for item in items:
        candidate = ", ".join((*chunk, item))
        if chunk and len("      " + candidate + ", &") > width:
            lines.append("      " + ", ".join(chunk) + ", &")
            chunk = [item]
        else:
            chunk.append(item)
    lines.append("      " + ", ".join(chunk) + " ]")
    return "\n".join(lines)


def _table_sin_fortran(argument: str, shift: float) -> str:
    """sin(argument + shift) by interpolating the shared baked table."""

    from .fused_program_wasm_backend import lut_for
    from .bounded_constants import materialize_pi

    values, _achieved, lower, upper, periodic = lut_for("sin")
    intervals = len(values) - 1
    def literal(value: float) -> str:
        return f"{value!r}" + "_c_double"
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
        return f"{value!r}" + "_c_double"
    x = argument if shift == 0.0 else f"({argument} + {literal(shift)})"
    turns = f"nint({x} * {literal(1.0 / pi)})"
    r = f"({x} - {literal(pi)} * real({turns}, c_double))"
    horner = literal(coefficients[0])
    for coefficient in coefficients[1:]:
        horner = f"({horner} * ({r} * {r}) + {literal(coefficient)})"
    series = f"({horner} * {r})"
    return f"merge(-{series}, {series}, mod(abs({turns}), 2) == 1)"


def _literal(value: Any, dtype: str | None = None) -> str:
    """Spell one SSA constant as Fortran, in the type the SSA declares.

    ``dtype`` is not decoration. This used to render purely from the Python
    payload's own type, so a float64 constant carrying an integer payload --
    256 rather than 256.0, which is what a captured Python literal usually
    is -- came out as the Fortran INTEGER literal ``256``. Call-site
    coercion could not save it either: coercion compares the SSA dtypes,
    saw float64 against float64, and correctly did nothing. The declared
    type and the emitted token disagreed with nobody in a position to
    notice, and gfortran rejected the call with

        Type mismatch in argument 't419'; passed INTEGER(4) to REAL(8)

    So the declared type decides the spelling, and the payload only decides
    the value.
    """
    if (
        dtype
        and isinstance(value, int)
        and not isinstance(value, bool)
        and str(dtype) not in _INTEGER_DTYPES
        and str(_DTYPE_KIND.get(str(dtype), "")).startswith("real")
    ):
        value = float(value)
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and str(dtype) in {"int64", "i64", "opaque_ref"}
    ):
        return f"{value}_c_int64_t"
    return _literal_payload(value)


def _literal_payload(value: Any) -> str:
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
    from .ir_literals import decode_llvm_scalar_literal

    try:
        return decode_llvm_scalar_literal(value)
    except ValueError as error:
        raise FortranEmissionError(str(error)) from error


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
        callee_output_records: Mapping[str, Sequence[SSAValue]] | None = None,
        callee_arguments: Mapping[str, Sequence[SSAValue]] | None = None,
        callee_argument_dtypes: Mapping[str, Sequence[str]] | None = None,
        callee_output_dtypes: Mapping[str, Sequence[str]] | None = None,
        callee_array_arguments: Mapping[str, Sequence[int]] | None = None,
        callee_inout_pairs: Mapping[
            str, Sequence[tuple[int, int]]
        ] | None = None,
        trig_solver: str = "lut",
        array_base_ids: Sequence[int] = (),
        mutated_base_ids: Sequence[int] = (),
        dynamic_array_ranks: Mapping[int, int] | None = None,
        value_dtypes: Mapping[int, str] | None = None,
        value_shapes: Mapping[int, Sequence[int]] | None = None,
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
        # The UN-canonicalized declared return record per callee: one entry
        # per source return position, repeats intact.  ``aggregate_index``
        # positions at a call site index THIS record; the native slot list
        # (``callee_outputs``) is this record deduplicated by SSA id.
        self.callee_output_records = {
            str(name): tuple(values)
            for name, values in (callee_output_records or {}).items()
        }
        self.callee_arguments = {
            str(name): tuple(values)
            for name, values in (callee_arguments or {}).items()
        }
        self.callee_argument_dtypes = {
            str(name): tuple(str(dtype) for dtype in dtypes)
            for name, dtypes in (callee_argument_dtypes or {}).items()
        }
        self.callee_output_dtypes = {
            str(name): tuple(str(dtype) for dtype in dtypes)
            for name, dtypes in (callee_output_dtypes or {}).items()
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
        self._pointer_value_ids = {
            int(value.id)
            for block in self.function.blocks.values()
            for instruction in block.instrs
            for value in (*instruction.args, instruction.res)
            if value is not None
            and str(value.dtype or "").casefold() in {"ptr", "pointer"}
        }
        self._pointer_value_ids.update(
            int(value.id)
            for value in self.function.args
            if str(value.dtype or "").casefold() in {"ptr", "pointer"}
        )
        for value_id, value_shape in (value_shapes or {}).items():
            current = self._value_types.get(int(value_id))
            if current is None or tuple(current.shape):
                continue
            self._value_types[int(value_id)] = SSAValue(
                current.id,
                current.dtype,
                tuple(map(int, value_shape)),
                current.device,
                dict(current.accounting or {}),
            )
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
            and int(value_id) in self.array_base_ids
            and int(value_id) in self._value_types
            and int(value_id) not in self._pointer_value_ids
            and str(
                getattr(self._value_types.get(int(value_id)), "dtype", "")
            ).casefold() not in {"ssa.aggregate", "ptr", "pointer"}
        }
        for argument in self.function.args:
            argument_id = int(argument.id)
            if argument_id not in self.array_base_ids:
                continue
            if argument_id in self._pointer_value_ids:
                # Raw pointer formals are flat C-interoperable assumed-size
                # arrays. Their indexed span is governed by the explicit
                # scalar dimensions already present in the helper signature;
                # inventing a second extent ABI here makes internal helpers
                # leak synthetic storage parameters toward the root.
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
        self._propagate_phi_dynamic_ranks()
        self.shortfalls: list[FortranShortfall] = []
        # Dangling operands (consumed, never produced, not a formal) emit as
        # undeclared symbols gfortran rejects late and cryptically -- the
        # observed shape is an elided loop body whose carried value survived
        # in a Phi. Say it here, at the layer that knows, instead of letting
        # a completeness claim stand over a program that cannot compile.
        produced_ids = {int(argument.id) for argument in self.function.args}
        for block in self.function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is not None:
                    produced_ids.add(int(instruction.res.id))
        for block in self.function.blocks.values():
            for instruction in block.instrs:
                operands = list(instruction.args)
                operands.extend(
                    value
                    for _, value in (
                        instruction.attributes.get("incoming") or ()
                    )
                    if hasattr(value, "id")
                )
                for operand in operands:
                    if int(operand.id) not in produced_ids:
                        self.shortfalls.append(FortranShortfall(
                            str(instruction.op),
                            str(block.name),
                            f"operand {int(operand.id)} is consumed but "
                            "never produced (dangling SSA identity; the "
                            "observed cause is an elided loop body whose "
                            "carried value survived in a Phi)",
                        ))
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
        # Scalar cells binding a callee's declared-but-unconsumed output
        # slots at a call site: identifier -> Fortran kind.  Named per
        # callsite/position, so they can never collide with ``t{id}`` locals.
        self._discard_declarations: dict[str, str] = {}
        self._discard_array_declarations: dict[str, SSAValue] = {}
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

    def _propagate_phi_dynamic_ranks(self) -> None:
        """Carry dynamic-array rank through Phi chains, to a fixed point.

        A dynamic-extent array carries rank in ``dynamic_array_ranks`` with
        an EMPTY static shape, so the shape-tuple propagation in
        ``_infer_control_value_types`` never sees it. A phi over such a
        value is that array's loop-carried identity: give the phi the same
        rank and alias the SAME extent symbols (the latch copy requires
        equal extents anyway). Without this the phi declares scalar and
        gfortran rejects the latch copy with "incompatible ranks".
        """

        changed = True
        while changed:
            changed = False
            for block in self.function.blocks.values():
                for instr in block.instrs:
                    if instr.op not in ("Phi", "phi") or instr.res is None:
                        continue
                    result_id = int(instr.res.id)
                    if result_id in self.dynamic_array_ranks:
                        continue
                    if (
                        result_id in self._pointer_value_ids
                        or str(self._typed(instr.res).dtype or "").casefold()
                        == "ssa.aggregate"
                    ):
                        continue
                    candidates = list(instr.args)
                    candidates.extend(
                        value
                        for _, value in (
                            instr.attributes.get("incoming") or ()
                        )
                        if hasattr(value, "id")
                    )
                    dynamic = next(
                        (
                            value for value in candidates
                            if int(value.id) in self.dynamic_array_ranks
                        ),
                        None,
                    )
                    if dynamic is None:
                        continue
                    source_id = int(dynamic.id)
                    self.dynamic_array_ranks[result_id] = (
                        self.dynamic_array_ranks[source_id]
                    )
                    extents = self.dynamic_array_leading_extents.get(
                        source_id
                    )
                    if extents is not None:
                        self.dynamic_array_leading_extents[result_id] = extents
                    self.array_base_ids.add(result_id)
                    changed = True

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
        # The occurrence shape alone is not the authority: an empty-sequence
        # seed (``x = []``) records shape () while the DECLARED view carries
        # rank from the callee's sequence formal, so consult the typed view
        # and the array-base table too.
        if instr.op in ("Const", "const") and (
            tuple(result.shape)
            or tuple(self._typed(result).shape or ())
            or int(result.id) in self.array_base_ids
        ):
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
        producer = self._producers.get(value.id)
        if producer is not None and value.id in self._inlined:
            operation = (
                producer.attributes.get("tensor_operation") or producer.op
            )
            # An explicitly bitwise producer remains numeric even when the
            # source expression is consumed through Python truthiness.  The
            # control lowering can consequently record its surrounding value
            # as bool, but Fortran must still spell ``not(mask)`` as
            # ``mask == 0`` and ``logical_and(mask, flag)`` as
            # ``mask /= 0 .and. flag``.  Overloaded And/Or/Xor make the same
            # decision from their operands.
            if operation in {
                "bitand", "bitor", "bitxor", "shl", "shr",
                "And", "Or", "Xor",
            }:
                return self._instruction_is_logical(producer)
        # LLVM's i1 is intrinsically a logical value.  Check the recorded
        # scalar kind before examining an inlined producer: an imported
        # ``fcmp`` is represented as an ordinary Call and would otherwise be
        # misclassified merely because its expression was inlined.
        if str(getattr(value, "dtype", "") or "") in _LOGICAL_DTYPES:
            return True
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

    def _truth_zero(self, value: SSAValue) -> str:
        """Return a zero with the kind of ``value``'s emitted expression."""

        typed = self._typed(value)
        producer = self._producers.get(typed.id)
        if producer is not None and typed.id in self._inlined:
            operation = (
                producer.attributes.get("tensor_operation") or producer.op
            )
            if operation in {
                "bitand", "bitor", "bitxor", "shl", "shr",
                "And", "Or", "Xor",
            } and not self._instruction_is_logical(producer):
                # Bit expressions are deliberately normalized to c_int64_t
                # at every nesting level by _expression.
                return "0_c_int64_t"
        dtype = str(typed.dtype or self.dtype).casefold()
        if dtype.endswith("int64"):
            return "0_c_int64_t"
        if dtype.endswith(("int32", "int")):
            return "0_c_int32_t"
        return "0.0_c_double"

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
        if operation in {
            "bitand", "bitor", "bitxor", "And", "Or", "Xor",
        }:
            # These operators are overloaded in repository SSA.  Their
            # operands, not a context-refined result dtype, determine whether
            # the emitted expression is LOGICAL or an integer bit operation.
            return bool(instr.args) and all(
                self._is_logical(value) for value in instr.args
            )
        if operation in {"shl", "shr"}:
            return False
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
            # Mixed integer/real arithmetic PROMOTES. Trusting the result
            # dtype alone demotes instead, and dtype here is inferred
            # rather than declared, so one integer-typed operand is enough
            # to mistype the result and drag a real operand down with it.
            # MEASURED: SymPy canonicalises subtraction to ``Mul(-1, x)``
            # and types that -1 as an integer, so ``-1 * index`` inferred
            # an integer result and emitted ``int(index, c_int)`` -- which
            # both discards the fraction and overflows 2**31, wrong by
            # 5.1e+11 at an argument of 1e12 while every smaller argument
            # stayed exact. Promotion is what C and numpy already do here,
            # which is why this was a Fortran-only failure.
            if not target_real and any(
                str(getattr(value, "dtype", None) or self.dtype)
                not in _INTEGER_DTYPES
                for value in instr.args
            ):
                target_real = True
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

    def _resolved_formal(
        self, callee: str, position: int
    ) -> SSAValue | None:
        """The callee's formal at ``position``, typed as the callee DECLARES.

        A raw formal occurrence often carries no dtype; the callee's emitter
        resolves one locally (address-base rules, sequence machinery) and
        declares it.  Coercing against the raw occurrence then spells a
        conversion to the wrong native type, so prefer the resolved dtype
        published by the callee's own emission.
        """

        formals = self.callee_arguments.get(str(callee), ())
        formal = formals[position] if position < len(formals) else None
        dtypes = self.callee_argument_dtypes.get(str(callee), ())
        if position < len(dtypes) and dtypes[position]:
            if formal is not None:
                return SSAValue(
                    formal.id,
                    str(dtypes[position]),
                    tuple(formal.shape),
                    formal.device,
                    dict(formal.accounting or {}),
                )
        return formal

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
                # A bare comparison is default LOGICAL(4); the native dummy
                # is logical(c_bool) (LOGICAL(1)), and a VALUE dummy makes
                # the kind mismatch a hard gfortran error, not a warning.
                return f"logical({expression} /= {zero}, kind=c_bool)"
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

    def _callee_extent_arguments(
        self, callee: str, actual_arguments: Sequence[SSAValue],
    ) -> tuple[str, ...]:
        """Resolve a callee's dynamic extents from this call's actual views.

        Dynamic extent names are local to the callee's formal value ids. They
        are not global storage requirements and must not simply bubble to the
        program entry. When the linked call preserved an actual tensor view,
        pass its concrete dimension; when it remains dynamic, pass the
        caller's corresponding extent. Only genuinely unresolvable/local
        requirements remain identifiers for transitive propagation.
        """

        declared = tuple(self.callee_extents.get(str(callee), ()))
        formals = tuple(self.callee_arguments.get(str(callee), ()))
        actuals = tuple(actual_arguments)
        resolved: list[str] = []
        for extent in declared:
            extent = str(extent)
            replacement = None
            matches = []
            for position, formal in enumerate(formals[:len(actuals)]):
                formal_id = int(formal.id)
                match = re.search(
                    rf"_{formal_id}(?:_([1-9][0-9]*))?$", extent
                )
                if match is not None:
                    # ``..._212_1`` also superficially ends in formal id 1.
                    # The longest formal-id suffix is the unambiguous owner.
                    matches.append((len(str(formal_id)), position, match))
            for _specificity, position, match in sorted(
                matches, key=lambda candidate: candidate[0], reverse=True
            ):
                actual = actuals[position]
                axis = int(match.group(1) or 1)
                shape = tuple(self._typed(actual).shape or ())
                if axis <= len(shape) and int(shape[axis - 1]) > 0:
                    replacement = f"{int(shape[axis - 1])}_c_int"
                    break
                dimensions = tuple(
                    self.dynamic_array_leading_extents.get(
                        int(actual.id), ()
                    )
                )
                if axis <= len(dimensions):
                    replacement = str(dimensions[axis - 1])
                    break
                dynamic_extent = self.dynamic_array_extents.get(int(actual.id))
                if axis == 1 and dynamic_extent is not None:
                    replacement = str(dynamic_extent)
                    break
            resolved.append(replacement or extent)
        return tuple(resolved)

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
        # The callee publishes its complete declared output record; a caller
        # is free to consume any subset of those positions.  The native call
        # still binds every declared slot -- consumed positions bind the
        # caller's own projection values, unconsumed scalar slots bind a
        # per-callsite discard cell.  An unconsumed ARRAY slot has no discard
        # spelling here yet, so it stays a loud shortfall rather than a
        # silently mis-sized call.
        declared_record = tuple(self.callee_outputs.get(str(callee), ()))
        output_binding_values: list[SSAValue] = []
        output_binding_names: list[str] = []
        postlude: list[str] = []
        if declared_record:
            # The caller's ``aggregate_index`` positions index the callee's
            # declared return RECORD -- one entry per source return position,
            # repeats intact; the native slot list is that record
            # canonicalized by SSA id.  Translate position -> native slot
            # through the record; a repeated projection consumed twice
            # becomes one native argument plus an alias assignment after the
            # call.
            slot_index_by_id = {
                int(value.id): index
                for index, value in enumerate(declared_record)
            }
            raw_record = tuple(
                self.callee_output_records.get(str(callee), ())
            ) or declared_record
            position_to_slot: dict[int, int] = {}
            for position, record_value in enumerate(raw_record):
                slot = slot_index_by_id.get(int(record_value.id))
                if slot is None:
                    return None
                position_to_slot[position] = slot
            if any(position not in position_to_slot for position in outputs):
                return None
            slot_projections: dict[int, list[SSAValue]] = {}
            for position in sorted(outputs):
                slot_projections.setdefault(
                    position_to_slot[position], []
                ).append(outputs[position])
            callsite = int(instr.attributes.get("plan_callsite_id") or 0)
            slot_dtypes = self.callee_output_dtypes.get(str(callee), ())
            for slot, formal in enumerate(declared_record):
                projections = slot_projections.get(slot, ())
                if projections:
                    first = projections[0]
                    self._locals[first.id] = self._typed(first)
                    slot_dtype = (
                        slot_dtypes[slot] if slot < len(slot_dtypes) else None
                    )
                    first_dtype = str(self._typed(first).dtype or self.dtype)
                    if slot_dtype and slot_dtype != first_dtype:
                        kind = _DTYPE_KIND.get(slot_dtype, "real(c_double)")
                        bridge = f"bridge_c{callsite}_p{slot}"
                        self._discard_declarations[bridge] = kind
                        output_binding_values.append(
                            SSAValue(
                                first.id, slot_dtype, (), first.device,
                                dict(first.accounting or {}),
                            )
                        )
                        output_binding_names.append(bridge)
                        postlude.append(
                            f"    {_name(first)} = "
                            f"{self._coerce_call_operand(SSAValue(first.id, slot_dtype, (), first.device, {}), bridge, first)}"
                        )
                    else:
                        output_binding_values.append(first)
                        output_binding_names.append(_name(first))
                    for extra in projections[1:]:
                        if int(extra.id) == int(first.id):
                            continue
                        self._locals[extra.id] = self._typed(extra)
                        postlude.append(
                            f"    {_name(extra)} = {_name(first)}"
                        )
                    continue
                typed_formal = self._typed(formal)
                # The callee's DECLARED dtype for this slot, the same
                # authority the bridge above uses when the slot is consumed.
                #
                # Only the consumed path honoured it. A discarded slot was
                # typed from the raw formal instead, which carries the
                # caller's view and is often the wrong one -- so an output
                # the caller ignores was declared integer(c_int32_t) against
                # a callee formal of real(c_double), and gfortran refused
                # the call with "passed INTEGER(4) to REAL(8)". Whether a
                # caller happens to USE a result cannot change that
                # result's type.
                slot_dtype = (
                    slot_dtypes[slot] if slot < len(slot_dtypes) else None
                )
                kind = _DTYPE_KIND.get(
                    slot_dtype or typed_formal.dtype or self.dtype,
                    "real(c_double)",
                )
                scratch = f"discard_c{callsite}_p{slot}"
                if _is_array(typed_formal):
                    self._discard_array_declarations[scratch] = SSAValue(
                        typed_formal.id,
                        slot_dtype or typed_formal.dtype,
                        tuple(typed_formal.shape),
                        typed_formal.device,
                        dict(typed_formal.accounting or {}),
                    )
                else:
                    self._discard_declarations[scratch] = kind
                output_binding_values.append(typed_formal)
                output_binding_names.append(scratch)
        else:
            expected_outputs = int(self.callee_output_count.get(str(callee), 0))
            if expected_outputs and len(ordered_outputs) != expected_outputs:
                return None
            for value in ordered_outputs:
                self._locals[value.id] = self._typed(value)
            output_binding_values = list(ordered_outputs)
            output_binding_names = [_name(value) for value in ordered_outputs]
        for value in consumed:
            self._consumed.add(value.id)

        # The callee is emitted by this same module, so its argument order is
        # known: its own extents first, then feeds, then outputs. The extents
        # must be the ones it actually declares -- rederiving them from the
        # call site gives a different set whenever the callee has interior
        # arrays whose sizes never appear in its signature.
        declared = self.callee_extents.get(str(callee))
        if declared is not None:
            extents = list(self._callee_extent_arguments(callee, instr.args))
        else:
            call_values = [*instr.args, *output_binding_values]
            extents = sorted(dimension_extents(call_values).values())
        array_positions = self.callee_array_arguments.get(
            str(callee), frozenset()
        )
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
                    self._resolved_formal(str(callee), argument_index),
                )
            input_arguments.append(operand)
        output_arguments = list(output_binding_names)
        prelude = []
        aliased_output_indices = set()
        _MISSING_INOUT = object()
        for input_index, output_index in self.callee_inout_pairs.get(
            str(callee), ()
        ):
            if output_index >= len(output_arguments):
                return None
            output_argument = output_arguments[output_index]
            if input_index >= len(instr.args):
                # The call site under-supplies a carried inout feed: the
                # caller's Call lists only the genuine data feeds, while the
                # carried accumulators' current values never entered
                # feed_ids. The aggregate projection records exactly which
                # caller value seeds this slot (``source_value_id`` on its
                # Load), so seed the inout dummy from that local instead of
                # refusing the whole call.
                binding_value = output_binding_values[output_index]
                seed_id = (binding_value.accounting or {}).get(
                    "source_value_id"
                )
                if seed_id is None:
                    return None
                seed_value = self._locals.get(int(seed_id)) or next(
                    (
                        argument
                        for argument in self.function.args
                        if int(argument.id) == int(seed_id)
                    ),
                    None,
                )
                if seed_value is None:
                    return None
                source_value = self._typed(seed_value)
                target_value = self._typed(
                    output_binding_values[output_index]
                )
                def _declared_rank(value) -> int:
                    identity = int(value.id)
                    return max(
                        len(tuple(value.shape or ())),
                        int(self.dynamic_array_ranks.get(identity, 0)),
                        1 if identity in self.array_base_ids else 0,
                    )

                if _declared_rank(seed_value) != _declared_rank(
                    output_binding_values[output_index]
                ):
                    # The recorded seed id names a value of a different
                    # rank in the caller's frame (regions share the
                    # caller's value space, so an id can be a sequence
                    # array outside and a scalar inside; the typed view
                    # alone misses the declaration's dynamic extent).
                    # Seeding across that mismatch compiles something the
                    # author never wrote -- refuse loudly instead.
                    return None
                source_expression = _name(seed_value)
                while len(input_arguments) <= input_index:
                    input_arguments.append(_MISSING_INOUT)
            else:
                source_value = self._typed(instr.args[input_index])
                target_value = self._typed(output_binding_values[output_index])
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
        if any(
            argument is _MISSING_INOUT for argument in input_arguments
        ):
            # A synthesized inout slot left a gap the pair list never
            # filled; passing a placeholder would silently mis-bind the
            # native call, so refuse loudly instead.
            return None
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
        return [
            *prelude,
            f"    call {native_callee}({', '.join(arguments)})",
            *postlude,
        ]

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
                    self._resolved_formal(callee, position),
                )
            call_operands.append(operand)
        arguments = [
            *self._callee_extent_arguments(callee, instr.args),
            *call_operands,
        ]
        if output_argument is not None:
            position = int(output_argument)
            if (
                instr.res is None
                or position < 0
                or position >= len(instr.args)
            ):
                return None
            aliases_frame = bool(
                instr.attributes.get("result_aliases_frame", False)
            )
            if (
                int(instr.args[position].id) != int(instr.res.id)
                and not aliases_frame
            ):
                return None
            # Linked aggregate/sequence calls publish into an explicit caller
            # frame slot.  Their semantic Call result remains the authored
            # graph identity, while downstream storage consumers deliberately
            # use the frame argument.  Emitting the call is sufficient; no
            # fictitious assignment to the semantic aggregate is required.
            if aliases_frame and int(instr.res.id) not in {
                int(argument.id) for argument in self.function.args
            }:
                # A linked call may be the producer of a compiler-local
                # sequence frame.  In that case the aliased output operand is
                # intentionally absent from the caller's public arguments,
                # but it still needs a real automatic-array declaration in
                # the caller.  Treat the exact callee output operand as that
                # local; dynamic-rank propagation has already copied the
                # formal's array contract onto this identity.
                self._locals[instr.res.id] = self._typed(
                    instr.args[position]
                )
            elif not aliases_frame:
                self._locals[instr.res.id] = self._typed(instr.res)
        elif output_count == 1 and instr.res is not None:
            self._locals[instr.res.id] = self._typed(instr.res)
            arguments.append(_name(instr.res))
        elif output_count != 0 or instr.res is not None:
            return None
        native_callee = self.callee_native_symbols.get(str(callee), str(callee))
        return [f"    call {native_callee}({', '.join(arguments)})"]

    def _external_reference_call(self, instr: Instr) -> list[str] | None:
        """Emit one typed call to the C shell's external-reference thunk."""

        if not (
            instr.op in {"Call", "call"}
            and instr.attributes.get("external_reference")
            and instr.res is not None
        ):
            return None
        if any(_is_array(self._typed(argument)) for argument in instr.args):
            return None
        from .shell_external_references import external_reference_thunk_symbol

        symbol = external_reference_thunk_symbol(
            self.function.name,
            int(instr.attributes["external_callsite_id"]),
            str(instr.attributes["external_identity"]),
        )
        self._locals[int(instr.res.id)] = self._typed(instr.res)
        operands = [self._operand(argument) for argument in instr.args]
        return [
            f"    call {symbol}({', '.join((*operands, _name(instr.res)))})"
        ]

    def _external_reference_interfaces(self) -> list[str]:
        from .shell_external_references import external_reference_thunk_symbol

        interfaces = []
        seen = set()
        for block in self.function.blocks.values():
            for instr in block.instrs:
                if not (
                    instr.op in {"Call", "call"}
                    and instr.attributes.get("external_reference")
                    and instr.res is not None
                ):
                    continue
                symbol = external_reference_thunk_symbol(
                    self.function.name,
                    int(instr.attributes["external_callsite_id"]),
                    str(instr.attributes["external_identity"]),
                )
                if symbol in seen:
                    continue
                seen.add(symbol)
                values = (*instr.args, instr.res)
                if any(_is_array(self._typed(value)) for value in values):
                    continue
                names = tuple(f"a{index}" for index in range(len(values)))
                interfaces.extend((
                    "    interface",
                    f"      subroutine {symbol}({', '.join(names)}) "
                    f"bind(C, name=\"{symbol}\")",
                    "        use, intrinsic :: iso_c_binding",
                    *(
                        f"        {_DTYPE_KIND.get(str(self._typed(value).dtype or self.dtype), 'real(c_double)')} :: {name}"
                        for name, value in zip(names, values)
                    ),
                    f"      end subroutine {symbol}",
                    "    end interface",
                ))
        return interfaces

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

    def _constant_integer(self, value: SSAValue) -> int | None:
        for candidate_block in self.function.blocks.values():
            for candidate in candidate_block.instrs:
                if (
                    candidate.res is None
                    or int(candidate.res.id) != int(value.id)
                    or candidate.op not in {"Const", "const"}
                ):
                    continue
                held = candidate.attributes.get(
                    "constant", candidate.attributes.get("value")
                )
                if isinstance(held, (bool, int, float)):
                    return int(held)
                llvm_literal = candidate.attributes.get("llvm_literal")
                if isinstance(llvm_literal, str):
                    match = re.fullmatch(
                        r"i(?:1|8|16|32|64)\s+([-+]?\d+)",
                        llvm_literal.strip(),
                    )
                    if match is not None:
                        return int(match.group(1))
        return None

    def _pointer_array_tensor_kernel(
        self, block: BasicBlock, index: int,
    ) -> list[str] | None:
        """Legalize the complete pointer-table stack/cat group to Fortran."""

        pointer_array = block.instrs[index]
        if pointer_array.op != "PointerArray" or pointer_array.res is None:
            return None
        call = next((
            candidate
            for candidate in block.instrs[index + 1:]
            if candidate.op in {"Call", "call"}
            and candidate.args
            and int(candidate.args[0].id) == int(pointer_array.res.id)
            and str(candidate.attributes.get("callee") or "") in {
                "stack_double", "cat_double",
            }
        ), None)
        if call is None or call.res is None:
            return None
        callee = str(call.attributes.get("callee"))
        dim_position = 4 if callee == "stack_double" else 5
        if dim_position >= len(call.args):
            return None
        dim = self._constant_integer(call.args[dim_position])
        if dim is None:
            return None
        operation = "stack" if callee == "stack_double" else "concat"
        semantic = Instr(
            operation,
            list(pointer_array.args),
            call.res,
            attributes={"dim": int(dim)},
            source_span=call.source_span,
        )
        statements = self._statements(semantic)
        if statements is None:
            return None
        self._locals[int(call.res.id)] = self._typed(call.res)
        self._consumed.add(int(pointer_array.res.id))
        self._consumed.add(int(call.res.id))
        return statements

    def _memcpy_call(self, instr: Instr) -> list[str] | None:
        """Express a typed LLVM memcpy as a byte-count-bounded slice copy."""

        if not (
            instr.op in {"Call", "call"}
            and str(instr.attributes.get("callee") or "").startswith(
                "llvm.memcpy"
            )
            and len(instr.args) >= 3
        ):
            return None
        destination, source, byte_count = instr.args[:3]
        destination_type = str(self._typed(destination).dtype or self.dtype)
        source_type = str(self._typed(source).dtype or self.dtype)
        if destination_type != source_type:
            return None
        byte_width = {
            "bool": 1, "i1": 1,
            "int32": 4, "i32": 4, "float32": 4,
            "int64": 8, "i64": 8, "float64": 8, "double": 8,
        }.get(destination_type.casefold())
        if byte_width is None:
            return None
        destination_rank = max(
            len(tuple(self._typed(destination).shape or ())),
            int(self.dynamic_array_ranks.get(int(destination.id), 0)),
            1 if int(destination.id) in self.array_base_ids else 0,
        )
        source_rank = max(
            len(tuple(self._typed(source).shape or ())),
            int(self.dynamic_array_ranks.get(int(source.id), 0)),
            1 if int(source.id) in self.array_base_ids else 0,
        )
        if destination_rank == source_rank and destination_rank > 1:
            # Imported tensor helpers spell a whole-array copy as LLVM memcpy
            # over ``product(shape) * sizeof(element)``. Fortran already has
            # the semantic operation: conformable array assignment. Prove
            # this is that full-copy pattern from SSA dependencies before
            # discarding the byte-count carrier; a partial memcpy must not be
            # widened silently.
            count_value = byte_count
            producer = self._producers.get(int(count_value.id))
            while (
                producer is not None
                and producer.op in {"SExt", "ZExt", "Trunc", "BitCast"}
                and len(producer.args) == 1
            ):
                count_value = producer.args[0]
                producer = self._producers.get(int(count_value.id))
            if producer is not None and producer.op in {"Mul", "mul"}:
                width_operands = tuple(
                    value for value in producer.args
                    if self._constant_integer(value) == byte_width
                )
                if len(width_operands) == 1:
                    element_count = next(
                        value for value in producer.args
                        if value is not width_operands[0]
                    )
                    argument_names = tuple(
                        self.function.metadata.get("llvm_argument_names", ())
                    )
                    shape_ids = {
                        int(argument.id)
                        for position, argument in enumerate(self.function.args)
                        if position < len(argument_names)
                        and str(argument_names[position]) == "shape"
                    }
                    pending = [element_count]
                    visited: set[int] = set()
                    depends_on_shape = False
                    while pending:
                        value = pending.pop()
                        value_id = int(value.id)
                        if value_id in visited:
                            continue
                        visited.add(value_id)
                        if value_id in shape_ids:
                            depends_on_shape = True
                            break
                        dependency = self._producers.get(value_id)
                        if dependency is not None:
                            pending.extend(dependency.args)
                    if depends_on_shape:
                        return [
                            f"    {self._operand(destination)} = "
                            f"{self._operand(source)}"
                        ]
        if destination_rank != 1 or source_rank != 1:
            return None
        count = self._operand(byte_count)
        upper = f"int(({count}) / {byte_width}_c_int64_t, c_int64_t)"
        destination_name = self._operand(destination)
        source_name = self._operand(source)
        return [
            f"    {destination_name}(1:{upper}) = "
            f"{source_name}(1:{upper})"
        ]

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
                zero = self._truth_zero(value)
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
                kind = (
                    "c_int64_t"
                    if str(instr.res.dtype or self.dtype).endswith("64")
                    else "c_int"
                )
                return f"int({expression}, {kind})"
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

        bitwise_op = {
            "BitAnd": "bitand",
            "BitOr": "bitor",
            "BitXor": "bitxor",
        }.get(op, op)
        if bitwise_op in {
            "bitand", "bitor", "bitxor", "shl", "shr",
        } and len(args) == 2:
            if all(self._is_logical(value) for value in instr.args):
                logical = {
                    "bitand": ".and.",
                    "bitor": ".or.",
                    "bitxor": ".neqv.",
                }.get(bitwise_op)
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
            expression = _BINARY[bitwise_op].format(*integer_args)
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
                    return _literal(
                    values,
                    instr.res.dtype if instr.res is not None else None,
                )
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
                payload = instr.attributes["value"]
                if (
                    isinstance(payload, (list, tuple))
                    and not payload
                    and instr.res is not None
                    and (
                        int(instr.res.id) in self.array_base_ids
                        or tuple(self._typed(instr.res).shape or ())
                    )
                ):
                    # An authored empty-sequence seed (``x = []``) feeding a
                    # sequence arena.  The arena's emptiness is carried by its
                    # separate length cell (seeded 0 through the workspace
                    # chain); the arena content itself is a fresh buffer, so
                    # a whole-array zero fill states exactly that.  Only the
                    # EMPTY seed is spelled here -- a populated list literal
                    # still refuses rather than being guessed at.  The fill's
                    # kind follows the DECLARED type (the typed view), not the
                    # SSA occurrence dtype: the declaration is the authority
                    # the emitted call sites are checked against.
                    return _literal(
                        0,
                        self._typed(instr.res).dtype or self.dtype,
                    )
                try:
                    return _literal(
                        payload,
                        instr.res.dtype if instr.res is not None else None,
                    )
                except FortranEmissionError as error:
                    raise FortranEmissionError(
                        f"{error}; function={self.function.name!r}; "
                        f"result_value_id={getattr(instr.res, 'id', None)!r}; "
                        f"attributes={instr.attributes!r}"
                    ) from error
            return _literal(
                constant,
                instr.res.dtype if instr.res is not None else None,
            )

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
            if (
                str(instr.attributes.get("extraction_identity") or "")
                == "builtins.float"
                or str(instr.attributes.get("source_operator") or "") == "float"
            ):
                # Python float(...) is an authored conversion boundary, not
                # merely an assignment between coincidentally equal inferred
                # dtypes. Keep that conversion explicit after extracting a
                # scalar from resident array storage.
                return f"real({operand}, c_double)"
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
        if op in _TERNARY and len(args) == 3:
            return _TERNARY[op].format(*self._numeric(instr, args))
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
        if op in {"LNot", "logical_not"} and len(args) == 1:
            if self._is_logical(instr.args[0]):
                return f"(.not. {args[0]})"
            zero = self._truth_zero(instr.args[0])
            return f"({args[0]} == {zero})"
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
                zero = self._truth_zero(instr.args[0])
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
                if not self._is_logical(instr.args[0]):
                    zero = self._truth_zero(instr.args[0])
                    condition = f"({condition} /= {zero})"
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

            pointer_array = self._pointer_array_tensor_kernel(block, index)
            if pointer_array is not None:
                body.extend(pointer_array)
                continue

            memcpy = self._memcpy_call(instr)
            if memcpy is not None:
                body.extend(memcpy)
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

            external_call = self._external_reference_call(instr)
            if external_call is not None:
                body.extend(external_call)
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
            *self._discard_array_declarations.values(),
        )
        arrays_present = any(_is_array(value) for value in all_values)
        dim_extents = dimension_extents(all_values) if arrays_present else {}

        address_arities: dict[int, set[int]] = {}
        for collection, positions in self._address_producers.values():
            address_arities.setdefault(int(collection.id), set()).add(
                len(positions)
            )
        linear_array_ids = {
            value_id
            for value_id, arities in address_arities.items()
            if arities == {1}
        }

        def dims(value: SSAValue, *, dummy: bool = False) -> str:
            extents = tuple(dim_extents[int(size)] for size in value.shape)
            if int(value.id) in linear_array_ids:
                # Repository helpers frequently express tensor addressing as
                # one flat GEP even when storage analysis recovers the full
                # semantic shape.  The declaration must follow the actual SSA
                # address contract.  An assumed-size rank-one dummy accepts
                # the caller's contiguous tensor by sequence association;
                # locals retain the exact compiler-owned element count.
                return "*" if dummy else " * ".join(extents)
            return ", ".join(extents)

        def dynamic_dims(value_id: int, extents: Sequence[str], *, dummy: bool) -> str:
            if int(value_id) in linear_array_ids:
                return "*" if dummy else " * ".join(extents)
            return ", ".join(extents)

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
            for extent in self._callee_extent_arguments(
                str(instr.attributes.get("callee") or ""), instr.args
            )
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(extent))
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
        declarations.extend(self._external_reference_interfaces())
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
                dimensions = (
                    dynamic_dims(
                        int(argument.id), leading_extents, dummy=True
                    )
                    if leading_extents else "*"
                )
                declarations.append(
                    f"    {kind}, intent({intent}) :: "
                    f"{_name(argument)}({dimensions})"
                )
                continue
            if argument.id in output_ids:
                if _is_array(argument):
                    declarations.append(
                        f"    {kind}, intent(inout) :: {_name(argument)}({dims(argument, dummy=True)})"
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
                        f"    {kind}, intent(inout) :: {_name(argument)}({dims(argument, dummy=True)})"
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
                    f"    {kind}, intent(inout) :: {_name(argument)}({dims(argument, dummy=True)})"
                )
                continue
            if _is_array(argument):
                declarations.append(
                    f"    {kind}, intent(in) :: {_name(argument)}({dims(argument, dummy=True)})"
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
                    f"({dynamic_dims(int(value.id), dynamic_extents, dummy=True)})"
                )
            elif _is_array(value):
                declarations.append(
                    f"    {kind}, intent(out) :: {_name(value)}({dims(value, dummy=True)})"
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
                    f"({dynamic_dims(int(value.id), dynamic_extents, dummy=False)})"
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
            f"    {kind} :: {identifier}"
            for identifier, kind in self._discard_declarations.items()
        )
        declarations.extend(
            f"    {_DTYPE_KIND.get(value.dtype or self.dtype, 'real(c_double)')}"
            f" :: {identifier}({dims(value)})"
            for identifier, value in self._discard_array_declarations.items()
        )
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
            argument_dtypes=tuple(
                str(self._typed(argument).dtype or self.dtype)
                for argument in self.function.args
            ),
            output_dtypes=tuple(
                str(self._typed(value).dtype or self.dtype)
                for value in self.outputs
            ),
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
    callee_output_records: Mapping[str, Sequence[SSAValue]] | None = None,
    callee_arguments: Mapping[str, Sequence[SSAValue]] | None = None,
    callee_argument_dtypes: Mapping[str, Sequence[str]] | None = None,
    callee_output_dtypes: Mapping[str, Sequence[str]] | None = None,
    callee_array_arguments: Mapping[str, Sequence[int]] | None = None,
    callee_inout_pairs: Mapping[
        str, Sequence[tuple[int, int]]
    ] | None = None,
    trig_solver: str = "lut",
    array_base_ids: Sequence[int] = (),
    mutated_base_ids: Sequence[int] = (),
    dynamic_array_ranks: Mapping[int, int] | None = None,
    value_dtypes: Mapping[int, str] | None = None,
    value_shapes: Mapping[int, Sequence[int]] | None = None,
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
        callee_output_records=callee_output_records,
        callee_arguments=callee_arguments,
        callee_argument_dtypes=callee_argument_dtypes,
        callee_output_dtypes=callee_output_dtypes,
        callee_array_arguments=callee_array_arguments,
        callee_inout_pairs=callee_inout_pairs,
        array_base_ids=array_base_ids,
        mutated_base_ids=mutated_base_ids,
        dynamic_array_ranks=dynamic_array_ranks,
        value_dtypes=value_dtypes,
        value_shapes=value_shapes,
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
    # True when the SSA carried precision sections. The toolchain reads this
    # at COMPILE time to withdraw contraction (-ffp-contract=off): the
    # emitted expressions are parenthesised, but contraction is the one
    # rewrite Fortran leaves to the processor, and building a section
    # without withdrawing it would zero every residual silently.
    precision_sections: bool = False

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


#: Arithmetic whose result must be REAL as soon as any operand is.
_PROMOTING_ARITHMETIC = frozenset({
    "Add", "Sub", "Mul", "Div", "Fma", "Neg",
    "add", "sub", "mul", "div", "fma", "neg",
})


def _promote_mixed_arithmetic_results(module, dtype: str) -> int:
    """Retype a mixed integer/real result that was inferred integer.

    dtype on this path is INFERRED, not declared, and one integer-typed
    operand is enough to mistype an arithmetic result. Fortran then
    believes it, declares an integer temporary, and demotes the real
    operand to fill it -- so the value is truncated on the way in and
    again on the way out.

    MEASURED: SymPy canonicalises ``a - b`` to ``Add(a, Mul(-1, b))`` and
    types that ``-1`` as an integer, so ``-1 * index`` inferred an integer
    result. The emitted ``int(index, c_int)`` discarded the fraction and
    overflowed 2**31 -- wrong by 5.1e+11 at an argument of 1e12, while
    every argument small enough to survive the cast stayed exact, which is
    exactly the shape of failure that hides in a sample.

    Promotion is what C and numpy do with the same SSA, which is why both
    of those lanes were already correct here. Returns how many results
    were retyped.
    """

    functions = getattr(module, "functions", module) or {}
    repaired = 0
    for function in functions.values():
        for block in getattr(function, "blocks", {}).values():
            for instruction in getattr(block, "instrs", ()):
                if str(instruction.op) not in _PROMOTING_ARITHMETIC:
                    continue
                result = getattr(instruction, "res", None)
                if result is None:
                    continue
                current = str(getattr(result, "dtype", None) or dtype)
                if current not in _INTEGER_DTYPES:
                    continue
                promoted = next(
                    (
                        str(getattr(value, "dtype", None) or dtype)
                        for value in instruction.args
                        if str(getattr(value, "dtype", None) or dtype)
                        not in _INTEGER_DTYPES
                    ),
                    None,
                )
                if promoted is None:
                    continue
                result.dtype = promoted
                repaired += 1
    return repaired


def emit_module(
    module: IRModule | Mapping[str, Function],
    *,
    name: str = "turing_ssa",
    dtype: str = DEFAULT_DTYPE,
    outputs: Mapping[str, Sequence[SSAValue]] | None = None,
    extra_roots: Sequence[str] = (),
    trig_solver: str = "lut",
    progress: "Callable[[str], None] | None" = None,
) -> FortranModule:
    """Translate an SSA module into one Fortran module.

    ``outputs`` maps a function name to the SSA values it returns.
    ``extra_roots`` names additional functions to keep and export (each becomes
    its own ``bind(C)`` entry) even when nothing reachable from the ordinary
    roots calls them -- a library exports its whole surface, not just the
    functions the entry happens to reach.

    ``progress``, if given, is called with a short message at each major
    phase boundary. Without it, this function runs silently end to end --
    for a large whole-program module that can look identical to a hang
    whether it is fast or slow, so the default here prints to stderr
    (unbuffered) instead of staying silent; pass an explicit no-op to
    suppress it.
    """

    _emit_module_started = time.time()
    _promote_mixed_arithmetic_results(module, dtype)

    def _phase(message: str) -> None:
        elapsed = time.time() - _emit_module_started
        report = f"emit_module: {message} (+{elapsed:0.1f}s)"
        if progress is not None:
            progress(report)
        else:
            print(report, file=sys.stderr, flush=True)

    module_precision_sections = False
    if isinstance(module, IRModule):
        from .ir_identities import (
            PRECISION_PIPELINE_METADATA,
            precision_backend_shortfalls,
        )
        module_precision_sections = bool(
            (getattr(module, "metadata", {}) or {})
            .get(PRECISION_PIPELINE_METADATA, {})
            .get("section_contracts")
        )
        precision_shortfalls = precision_backend_shortfalls(module, "fortran")
        if precision_shortfalls:
            details = "; ".join(
                f"{item['function']} values={item['value_ids']!r} "
                f"missing={item['missing']!r}"
                for item in precision_shortfalls
            )
            raise FortranEmissionError(
                "Fortran cannot honour repository precision sections: "
                + details
            )
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
    host_linear_region_inlining = ()
    if isinstance(module, IRModule):
        from .ir_identities import inline_host_linear_source_regions
        functions, host_linear_region_inlining = (
            inline_host_linear_source_regions(functions)
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
    # A scheduled region publishes through the aggregate convention and
    # records nothing about it: no ``return_value``, no ``named_outputs``, no
    # ``Ret``. Its declared outputs were therefore empty while the CALL SITE
    # derived its actual arguments from the aggregate projections -- two
    # sources for one interface, one of them silent, so the callee declared
    # eleven dummies against a call passing twenty-eight and gfortran refused
    # the module.
    #
    # The projections are the authority available here: a planner region
    # shares its caller's value space, so the value a caller loads out of slot
    # k IS the value the region defined. Reading them back gives the callee
    # exactly the interface the call site is already going to use.
    aggregate_outputs: dict[str, dict[int, SSAValue]] = {}
    for function in functions.values():
        for block in function.blocks.values():
            for index, instruction in enumerate(block.instrs):
                if instruction.op not in ("Call", "call"):
                    continue
                if instruction.attributes.get(
                    "result_convention"
                ) != "ssa.aggregate" or instruction.res is None:
                    continue
                callee = str(instruction.attributes.get("callee") or "")
                if not callee:
                    continue
                slots = aggregate_outputs.setdefault(callee, {})
                addresses: dict[int, int] = {}
                for follower in block.instrs[index + 1:]:
                    if follower.res is None:
                        continue
                    if (
                        follower.op == "GetElementPtr" and follower.args
                        and int(follower.args[0].id) == int(instruction.res.id)
                    ):
                        position = follower.attributes.get("aggregate_index")
                        if position is not None:
                            addresses[int(follower.res.id)] = int(position)
                    elif (
                        follower.op == "Load" and follower.args
                        and int(follower.args[0].id) in addresses
                    ):
                        slots[addresses[int(follower.args[0].id)]] = (
                            follower.res
                        )

    explicitly_requested_roots = set(map(str, (outputs or {}).keys()))
    named_outputs = dict(outputs or {})
    for function_name, function in functions.items():
        return_value = function.metadata.get("return_value")
        if isinstance(return_value, SSAValue):
            # The function owns its return identity.  A caller-side aggregate
            # projection can carry a different local id for the same slot;
            # letting that derived catalogue override ``return_value`` makes
            # the callee assign an undeclared caller id and publishes the
            # wrong ABI symbol (binary_value and sum_double exposed this in a
            # whole-program native build).
            named_outputs[function_name] = (return_value,)
            continue
        if function_name in named_outputs:
            continue
        declared = tuple(function.metadata.get("named_outputs") or ())
        if not declared:
            projected = aggregate_outputs.get(str(function_name))
            if projected:
                named_outputs[function_name] = tuple(
                    projected[position] for position in sorted(projected)
                )
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
    # Keep the record as declared (position space, repeats intact) beside the
    # canonical native slot list: call sites translate ``aggregate_index``
    # positions through the record into slots.
    named_output_records = {
        function_name: tuple(values)
        for function_name, values in named_outputs.items()
    }
    named_outputs = {
        function_name: tuple({
            int(value.id): value for value in values
        }.values())
        for function_name, values in named_outputs.items()
    }
    # ``named_outputs`` is also the internal call-signature catalogue, so it
    # eventually contains every returning helper.  It is not an export list.
    # Reachability begins at the entries the caller explicitly requested;
    # otherwise imported tensor helpers whose calls were legalized to native
    # Fortran expressions become accidental roots (notably the pointer-table
    # stack implementation, which has no Fortran value-level equivalent).
    roots = {
        name for name in explicitly_requested_roots if name in functions
    }
    if not roots and outputs is None:
        roots.update(str(name) for name in named_outputs if str(name) in functions)
    roots.update(
        str(name)
        for name, function in functions.items()
        if function.metadata.get("control_ir")
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
    from .ssa_call_storage import call_array_argument_ids

    method_array_bases: dict[str, set[int]] = {
        function_name: set(value_ids)
        for function_name, value_ids in call_array_argument_ids(
            functions
        ).items()
    }
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
    _phase(f"starting rank propagation over {len(functions)} function(s)")
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

    # Several independent fixpoint passes below (rank, scalar-control,
    # array-contract, dtype, mutation propagation) each used to rescan an
    # entire block's instructions twice for every aggregate-returning Call
    # in it, on every pass of their own ``while changed_*`` loop --
    # O(calls x block size), repeated per pass. That was never exercised at
    # scale because a structurally unresolved loop bound (see the
    # record-field ``.shape[i]`` fix in glsl_deployment_strategy.py) kept
    # a real per-cell loop body -- and the calls in it -- from ever being
    # reached, so no block here ever grew large enough for the O(N^2) scan
    # to matter. Once real, it made this pass effectively hang. Block
    # structure never changes across any of these fixpoint passes, so
    # index each block's GetElementPtr/Load instructions once, by their
    # source operand, and reuse the same index for every call and every
    # pass of every one of these loops, instead of rescanning.
    _aggregate_output_index_cache: dict[int, tuple[dict, dict]] = {}

    def _aggregate_output_indices(block):
        cached = _aggregate_output_index_cache.get(id(block))
        if cached is not None:
            return cached
        gep_by_source: dict = {}
        loads_by_source_id: dict = {}
        for candidate in block.instrs:
            op = candidate.op
            if (
                op in {"GetElementPtr", "getelementptr"}
                and candidate.res is not None
                and candidate.args
                and candidate.attributes.get("aggregate_index") is not None
            ):
                gep_by_source.setdefault(
                    id(candidate.args[0]), []
                ).append(candidate)
            elif (
                op in {"Load", "load"}
                and candidate.res is not None
                and candidate.args
            ):
                loads_by_source_id.setdefault(
                    int(candidate.args[0].id), []
                ).append(candidate)
        result = (gep_by_source, loads_by_source_id)
        _aggregate_output_index_cache[id(block)] = result
        return result

    _rank_pass = 0
    while changed_ranks:
        _rank_pass += 1
        _phase(f"rank propagation pass {_rank_pass}")
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
                        gep_by_source, loads_by_source_id = (
                            _aggregate_output_indices(block)
                        )
                        addresses = {
                            int(candidate.res.id): int(
                                candidate.attributes["aggregate_index"]
                            )
                            for candidate in gep_by_source.get(
                                id(instruction.res), ()
                            )
                        }
                        for source_id, slot in addresses.items():
                            for candidate in loads_by_source_id.get(
                                source_id, ()
                            ):
                                caller_outputs[slot] = candidate.res
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
    _phase("rank propagation done, starting scalar-control propagation")
    changed_scalars = True
    _scalar_pass = 0
    while changed_scalars:
        _scalar_pass += 1
        _phase(f"scalar-control propagation pass {_scalar_pass}")
        changed_scalars = False
        for function_name, function in functions.items():
            scalar_ids = scalar_control_ids[function_name]
            for block in function.blocks.values():
                gep_by_source, loads_by_source_id = _aggregate_output_indices(
                    block
                )
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
                        for candidate in gep_by_source.get(id(instruction.res), ())
                    }
                    caller_outputs = {}
                    for source_id, slot in addresses.items():
                        for candidate in loads_by_source_id.get(source_id, ()):
                            caller_outputs[slot] = candidate.res
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
    _phase("scalar-control propagation done, starting array-contract propagation")
    changed_array_contracts = True
    _array_contract_pass = 0
    while changed_array_contracts:
        _array_contract_pass += 1
        _phase(f"array-contract propagation pass {_array_contract_pass}")
        changed_array_contracts = False
        for caller_name, caller in functions.items():
            caller_arrays = array_contract_ids[caller_name]
            for block in caller.blocks.values():
                gep_by_source, loads_by_source_id = _aggregate_output_indices(
                    block
                )
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
                        for candidate in gep_by_source.get(id(instruction.res), ())
                    }
                    caller_outputs = {}
                    for source_id, slot in addresses.items():
                        for candidate in loads_by_source_id.get(source_id, ()):
                            caller_outputs[slot] = candidate.res
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
                    if (
                        instruction.res is not None
                        and int(instruction.res.id)
                        not in array_contract_ids[str(function_name)]
                    ):
                        logical_ids.add(int(instruction.res.id))
                elif operation in {"LNot", "Not", "logical_not"}:
                    if (
                        instruction.res is not None
                        and int(instruction.res.id)
                        not in array_contract_ids[str(function_name)]
                    ):
                        logical_ids.add(int(instruction.res.id))
                elif (
                    operation in _COMPARISON
                    and instruction.res is not None
                    and int(instruction.res.id)
                    not in array_contract_ids[str(function_name)]
                ):
                    # Tensor comparison kernels in repository SSA use the
                    # same double-backed arena ABI as C and LLVM.  Only a
                    # scalar comparison becomes a Fortran LOGICAL value;
                    # promoting a tensor result to LOGICAL makes the imported
                    # numeric helper signature disagree with every caller.
                    logical_ids.add(int(instruction.res.id))
                # CondBr accepts ordinary scalar truthiness.  Its operand is
                # logical only when the producer/declared dtype says so; an
                # integer loop-carried value such as ``while value`` must stay
                # integer and be compared with zero at emission.
        for value_id in logical_ids:
            dtypes[value_id] = "bool"
        function_value_dtypes[str(function_name)] = dtypes
        logical_dtype_ids[str(function_name)] = logical_ids

    _phase("array-contract propagation done, starting dtype propagation")
    changed_dtypes = True
    _dtype_pass = 0
    while changed_dtypes:
        _dtype_pass += 1
        _phase(f"dtype propagation pass {_dtype_pass}")
        changed_dtypes = False
        for caller_name, caller in functions.items():
            caller_dtypes = function_value_dtypes[str(caller_name)]
            caller_logical = logical_dtype_ids[str(caller_name)]
            for block in caller.blocks.values():
                gep_by_source, loads_by_source_id = _aggregate_output_indices(
                    block
                )
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
                        # A value can be "logical" on one call boundary
                        # (e.g. it's the predicate of a comparison feeding
                        # THIS callee) while carrying an unrelated, already
                        # -settled, non-bool dtype from a DIFFERENT call
                        # boundary that happens to share the same SSA id.
                        # Unconditionally forcing "bool" onto that shared
                        # id, then having the other relationship's own pass
                        # sync it back to its real dtype, never converges --
                        # this is why dtype propagation ran thousands of
                        # passes instead of a handful. Only settle "bool"
                        # here when neither side already holds a different,
                        # concrete dtype; a genuine conflict is left
                        # unresolved for this pass rather than fought over.
                        actual_settled_conflict = caller_dtypes.get(
                            actual_id
                        ) not in (None, "bool")
                        formal_settled_conflict = callee_dtypes.get(
                            formal_id
                        ) not in (None, "bool")
                        if (
                            (actual_id in caller_logical or formal_id in callee_logical)
                            and not actual_settled_conflict
                            and not formal_settled_conflict
                        ):
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
                    # ``aggregate_index`` positions the source RETURN RECORD
                    # (repeats intact), not the canonicalized slot list -- the
                    # same distinction ``_region_call`` translates through at
                    # emission time.  Do the same translation here, or a
                    # repeated identity earlier in the record shifts every
                    # later position's dtype onto the wrong slot.
                    raw_record = tuple(
                        named_output_records.get(callee_name, ())
                    ) or callee_output_values
                    slot_index_by_id = {
                        int(value.id): index
                        for index, value in enumerate(callee_output_values)
                    }
                    position_to_slot = {
                        position: slot_index_by_id[int(record_value.id)]
                        for position, record_value in enumerate(raw_record)
                        if int(record_value.id) in slot_index_by_id
                    }
                    addresses = {
                        int(candidate.res.id): int(candidate.attributes["aggregate_index"])
                        for candidate in gep_by_source.get(id(instruction.res), ())
                    }
                    caller_outputs = {}
                    for source_id, slot in addresses.items():
                        if slot not in position_to_slot:
                            continue
                        for candidate in loads_by_source_id.get(source_id, ()):
                            caller_outputs[position_to_slot[slot]] = candidate.res
                    for output_index, callee_output in enumerate(callee_output_values):
                        caller_output = caller_outputs.get(output_index)
                        if caller_output is None:
                            continue
                        caller_id = int(caller_output.id)
                        callee_id = int(callee_output.id)
                        # Same conflict guard as the argument-side settling
                        # above: don't force "bool" onto a shared SSA id
                        # that already holds a different, concrete dtype
                        # from an unrelated call relationship.
                        caller_settled_conflict = caller_dtypes.get(
                            caller_id
                        ) not in (None, "bool")
                        callee_settled_conflict = callee_dtypes.get(
                            callee_id
                        ) not in (None, "bool")
                        if (
                            (caller_id in caller_logical or callee_id in callee_logical)
                            and not caller_settled_conflict
                            and not callee_settled_conflict
                        ):
                            settled = "bool"
                            caller_logical.add(caller_id)
                            callee_logical.add(callee_id)
                        else:
                            caller_dtype = caller_dtypes.get(caller_id)
                            callee_dtype = callee_dtypes.get(callee_id)
                            if callee_dtype is not None and caller_dtype is None:
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
        formal_ids = {
            int(argument.id) for argument in functions[function_name].args
        }
        method_array_bases.setdefault(function_name, set()).update(
            value_id
            for value_id, rank in ranks.items()
            if rank > 0 and value_id in formal_ids
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
    _phase("dtype propagation done, starting mutation propagation")
    changed_mutation = True
    _mutation_pass = 0
    while changed_mutation:
        _mutation_pass += 1
        _phase(f"mutation propagation pass {_mutation_pass}")
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
    _phase("mutation propagation done, generating Fortran source")
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
    from .ssa_storage_requirements import module_storage_requirements

    storage_requirements = module_storage_requirements(module)
    function_value_shapes = {
        function_name: {
            value_id: requirement.shape
            for value_id, requirement in storage_requirements.get(
                function_name, {}
            ).items()
            if requirement.shape
        }
        for function_name in functions
    }
    def extent_signature(
        function_name: str,
        signatures: Mapping[str, Sequence[str]],
    ) -> tuple[str, ...]:
        function = functions[function_name]
        return emit_function(
                function,
                dtype=dtype,
                trig_solver=trig_solver,
                outputs=named_outputs.get(function_name, ()),
                callee_extents=signatures,
                callee_arity=callee_arity,
                callee_output_count=callee_output_count,
                callee_outputs=named_outputs,
                callee_output_records=named_output_records,
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
                value_shapes=function_value_shapes.get(function_name, {}),
                tensor_table=module_tensor_tables.get(function_name),
                native_symbol=native_symbols[function_name],
                callee_native_symbols=native_symbols,
                extent_namespace=extent_namespaces[function_name],
            ).extent_names

    # Extents are a finite call-signature dataflow problem, but dynamic names
    # are callee-local formal dimensions, not global set members. Re-emit only
    # signatures so each call can substitute its actual tensor view before an
    # unresolved extent is propagated toward the public entry.
    local_extents = {
        function_name: extent_signature(function_name, {})
        for function_name in functions
    }
    callee_extents: dict[str, tuple[str, ...]] = dict(local_extents)
    for _ in range(max(1, len(functions) + 1)):
        updated_extents = {
            function_name: extent_signature(function_name, callee_extents)
            for function_name in functions
        }
        if updated_extents == callee_extents:
            break
        callee_extents = updated_extents
    else:  # pragma: no cover - finite call signatures must converge.
        raise RuntimeError("Fortran call ABI extents did not reach a fixed point")
    def emit_subroutines(
        signatures: Mapping[str, Sequence[str]],
        formal_dtypes: Mapping[str, Sequence[str]] | None = None,
        output_dtypes: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[FortranSubroutine, ...]:
        return tuple(
            emit_function(
                function,
                trig_solver=trig_solver,
                dtype=dtype,
                outputs=named_outputs.get(function_name, ()),
                callee_extents=signatures,
                callee_argument_dtypes=formal_dtypes,
                callee_output_dtypes=output_dtypes,
                callee_arity=callee_arity,
                callee_output_count=callee_output_count,
                callee_outputs=named_outputs,
                callee_output_records=named_output_records,
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
                value_shapes=function_value_shapes.get(function_name, {}),
                tensor_table=module_tensor_tables.get(function_name),
                native_symbol=native_symbols[function_name],
                callee_native_symbols=native_symbols,
                extent_namespace=extent_namespaces[function_name],
            )
            for function_name, function in functions.items()
        )

    callee_formal_dtypes: dict[str, tuple[str, ...]] = {}
    callee_result_dtypes: dict[str, tuple[str, ...]] = {}
    _signature_pass_limit = max(1, len(functions) + 2)
    for _signature_pass in range(_signature_pass_limit):
        _phase(
            f"emitting subroutine bodies, signature pass "
            f"{_signature_pass + 1}/{_signature_pass_limit}"
        )
        subroutines = emit_subroutines(
            callee_extents, callee_formal_dtypes, callee_result_dtypes
        )
        emitted_extents = {
            function_name: subroutine.extent_names
            for function_name, subroutine in zip(functions, subroutines)
        }
        emitted_formal_dtypes = {
            function_name: subroutine.argument_dtypes
            for function_name, subroutine in zip(functions, subroutines)
        }
        emitted_result_dtypes = {
            function_name: subroutine.output_dtypes
            for function_name, subroutine in zip(functions, subroutines)
        }
        if (
            emitted_extents == callee_extents
            and emitted_formal_dtypes == callee_formal_dtypes
            and emitted_result_dtypes == callee_result_dtypes
        ):
            break
        callee_extents = emitted_extents
        callee_formal_dtypes = emitted_formal_dtypes
        callee_result_dtypes = emitted_result_dtypes
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
        "  pure integer(c_int64_t) function turing_python_bit_length(value)",
        "    integer(c_int64_t), intent(in), value :: value",
        "    integer(c_int64_t) :: magnitude",
        "    if (value == 0_c_int64_t) then",
        "      turing_python_bit_length = 0_c_int64_t",
        "    else if (value == -huge(value) - 1_c_int64_t) then",
        "      turing_python_bit_length = bit_size(value)",
        "    else",
        "      magnitude = abs(value)",
        "      turing_python_bit_length = bit_size(magnitude) - leadz(magnitude)",
        "    end if",
        "  end function turing_python_bit_length",
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
        scalar_source_transforms = {
            int(value_id): (str(source_name), str(transform))
            for value_id, source_name, transform in function.metadata.get(
                "scalar_source_transforms", ()
            )
        }
        source_names.update({
            value_id: source_name
            for value_id, (source_name, _transform)
            in scalar_source_transforms.items()
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
                source_transforms={
                    value_id: transform
                    for value_id, (_source_name, transform)
                    in scalar_source_transforms.items()
                },
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
    sequence_output_surfaces: dict[str, list[dict[str, Any]]] = {}
    for function_name, table in module_sequence_tables.items():
        function = functions.get(function_name)
        if function is None:
            continue
        materializations = tuple(
            function.metadata.get("extraction_materializations", ())
        )
        surfaces = []
        for output_index, output in enumerate(named_outputs.get(function_name, ())):
            descriptor = next((
                candidate
                for candidate in table.sequences.values()
                if int(output.id) == int(candidate.sequence_id)
                or int(output.id) in set(map(int, candidate.column_value_ids))
            ), None)
            if descriptor is None:
                continue
            materialization = next((
                str(record.get("extraction_identity"))
                for record in materializations
                if int(record.get("source_sequence_id", -1))
                == int(descriptor.sequence_id)
                and str(record.get("lowering") or "").startswith("immutable")
            ), None)
            surfaces.append({
                "output_index": int(output_index),
                "sequence_id": int(descriptor.sequence_id),
                "materialization_identity": materialization,
            })
        if surfaces:
            sequence_output_surfaces[function_name] = surfaces
    sequence_runtime_bindings: dict[str, list[dict[str, Any]]] = {}
    sequence_source_transforms = {
        function_name: {
            int(sequence_id): (str(source_name), str(transform))
            for sequence_id, source_name, transform
            in function.metadata.get("sequence_source_transforms", ())
        }
        for function_name, function in functions.items()
    }
    entry_by_name = {entry.name: entry for entry in entry_points}
    for function_name, table in module_sequence_tables.items():
        described = entry_by_name.get(function_name)
        if described is None:
            continue
        parameter_names = {parameter.name for parameter in described.parameters}
        extent_parameters = tuple(
            parameter.name
            for parameter in described.parameters
            if parameter.role == "extent"
        )
        extent_runtime_policies = {
            extent: ("unit" if extent == "extent_1" else "capacity")
            for extent in extent_parameters
        }
        function = functions.get(function_name)
        authored_parameter_roots = {
            str(item if isinstance(item, str) else item[0])
            for item in (
                () if function is None
                else function.metadata.get("parameter_names", ())
            )
        }
        if function is not None:
            authored_parameter_roots.update(map(
                str,
                dict(function.metadata.get("parameter_record_abi") or {}),
            ))
            authored_parameter_roots.update(map(
                str,
                dict(function.metadata.get("parameter_value_abi") or {}),
            ))
        for parameter in described.parameters:
            if (
                not parameter.source_name
                or str(parameter.source_name).split(".", 1)[0]
                not in authored_parameter_roots
            ):
                continue
            parameter_extents = tuple(dict.fromkeys((
                *((parameter.extent,) if parameter.extent is not None else ()),
                *parameter.extents,
            )))
            for extent in parameter_extents:
                if extent in extent_runtime_policies:
                    extent_runtime_policies[extent] = (
                        f"source_length:{parameter.source_name}"
                    )
        bindings = []
        for descriptor in table.sequences.values():
            required_ids = (
                int(descriptor.length_address_id),
                int(descriptor.capacity_value_id),
                *((int(descriptor.status_address_id),)
                  if descriptor.status_address_id is not None else ()),
            )
            required_parameters = tuple(f"t{value_id}" for value_id in required_ids)
            if any(name not in parameter_names for name in required_parameters):
                continue
            status_values = next((
                dict(candidate.metadata.get("status_values") or {})
                for candidate in functions.values()
                if int(candidate.metadata.get(
                    "sequence_id",
                    candidate.metadata.get("destination_sequence_id", -1),
                )) == int(descriptor.sequence_id)
                and candidate.metadata.get("status_values")
            ), {})
            bindings.append({
                "sequence_id": int(descriptor.sequence_id),
                "column_parameters": [
                    f"t{int(value_id)}"
                    for value_id in descriptor.column_value_ids
                    if f"t{int(value_id)}" in parameter_names
                ],
                "local_column_value_ids": [
                    int(value_id)
                    for value_id in descriptor.column_value_ids
                    if f"t{int(value_id)}" not in parameter_names
                ],
                "length_parameter": f"t{int(descriptor.length_address_id)}",
                "capacity_parameter": f"t{int(descriptor.capacity_value_id)}",
                "status_parameter": (
                    f"t{int(descriptor.status_address_id)}"
                    if descriptor.status_address_id is not None else None
                ),
                # These values are now part of the published ABI contract;
                # consumers never need to reverse-engineer generated extent
                # spellings. Fixed one-cell length/status arenas use unit
                # extents and every remaining dynamic sequence extent uses
                # the caller-selected capacity.
                "extent_parameters": dict(extent_runtime_policies),
                "status_values": status_values,
            })
        if bindings:
            sequence_runtime_bindings[function_name] = bindings
    from .shell_external_references import external_reference_thunk_symbol

    external_reference_thunks = []
    for function_name, function in functions.items():
        resolved_dtypes = function_value_dtypes.get(function_name, {})
        for block in function.blocks.values():
            for instruction in block.instrs:
                if not (
                    instruction.op in {"Call", "call"}
                    and instruction.attributes.get("external_reference")
                    and instruction.res is not None
                ):
                    continue
                external_reference_thunks.append({
                    "symbol": external_reference_thunk_symbol(
                        function_name,
                        int(instruction.attributes["external_callsite_id"]),
                        str(instruction.attributes["external_identity"]),
                    ),
                    "function": str(function_name),
                    "callsite_id": int(
                        instruction.attributes["external_callsite_id"]
                    ),
                    "identity": str(
                        instruction.attributes["external_identity"]
                    ),
                    "argument_dtypes": [
                        str(
                            resolved_dtypes.get(int(argument.id))
                            or argument.dtype
                            or dtype
                        )
                        for argument in instruction.args
                    ],
                    "keyword_names": list(map(
                        str,
                        instruction.attributes.get("keyword_names") or (),
                    )),
                    "result_dtype": str(
                        resolved_dtypes.get(int(instruction.res.id))
                        or instruction.res.dtype
                        or dtype
                    ),
                    "shell_abi": str(instruction.attributes["shell_abi"]),
                    "native_abi": str(
                        instruction.attributes.get("native_abi") or ""
                    ),
                    "runtime_owner": str(
                        instruction.attributes.get("runtime_owner") or ""
                    ),
                    "shell_profiles": list(map(
                        str, instruction.attributes.get("shell_profiles") or (),
                    )),
                    "external_domain": str(
                        instruction.attributes["external_domain"]
                    ),
                })
    api = CompiledProgramAPI(
        module=name,
        language="fortran",
        entry=control_entries[0] if control_entries else None,
        entry_points=tuple(entry_points),
            metadata={
                "dtype": dtype,
                "fortran_internal_symbols": dict(native_symbols),
                "external_reference_thunk_schema": (
                    "turing.external-reference-c-thunk.v1"
                ),
                "external_reference_thunks": external_reference_thunks,
                **({
                    "host_linear_region_inlining": list(
                        host_linear_region_inlining
                    ),
                } if host_linear_region_inlining else {}),
                **({
                "shell_io": dict(getattr(module, "metadata", {})["shell_io"]),
            } if getattr(module, "metadata", {}).get("shell_io") else {}),
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
                        **({
                            "source_name": sequence_source_transforms[
                                function_name
                            ][int(descriptor.sequence_id)][0],
                            "source_names": [sequence_source_transforms[
                                function_name
                            ][int(descriptor.sequence_id)][0]],
                            "source_transform": sequence_source_transforms[
                                function_name
                            ][int(descriptor.sequence_id)][1],
                        } if int(descriptor.sequence_id) in (
                            sequence_source_transforms.get(function_name, {})
                        ) else {}),
                    }
                    for descriptor in table.sequences.values()
                ]
                for function_name, table in module_sequence_tables.items()
                if function_name in functions and table.sequences
            },
            # A sequence arena can be both an ABI input/inout and the authored
            # return value.  Ordinary scalar output accounting intentionally
            # omits that alias, so publish the return-to-descriptor correlation
            # explicitly rather than asking wrappers to infer it from argument
            # order or source spelling.
            "sequence_output_surface_schema": (
                "turing.repository-ssa-sequence-output-surfaces.v1"
            ),
            "sequence_output_surfaces": sequence_output_surfaces,
            "sequence_runtime_binding_schema": (
                "turing.repository-ssa-sequence-runtime-bindings.v1"
            ),
            "sequence_runtime_bindings": sequence_runtime_bindings,
            "validation_contract_schema": (
                "turing.repository-ssa-validation-contracts.v1"
            ),
            "validation_contracts": {
                function_name: list(
                    function.metadata.get("validation_contracts", ())
                )
                for function_name, function in functions.items()
                if function.metadata.get("validation_contracts")
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
    _phase("Fortran source generation done")
    return FortranModule(
        name, "\n".join(lines) + "\n", subroutines, api=api,
        precision_sections=module_precision_sections,
    )


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
    compile_flags = tuple(extra_flags or aggressive_fortran_flags(
        compiler,
        precision_sections=bool(getattr(module, "precision_sections", False)),
    ))
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


# ---------------------------------------------------------------------------
# Running a compiled Fortran core: the shared execution face.
#
# The packed pointer-array entry takes every published parameter in one
# call, and the API records name each formal by the SSA value id it came
# from -- so the same value-id-keyed feed dictionary every other lane
# consumes marshals this one too. Nothing here knows what the kernel
# computes; it reads the records.
# ---------------------------------------------------------------------------

import numpy as np


class FortranCoreExecution:
    """Allocated buffers and a bound packed entry, mirroring the LLVM face."""

    def __init__(self, buffers, entry, pointers, count):
        self.buffers = buffers
        self._entry = entry
        self._pointers = pointers
        self._count = count

    def run(self) -> "FortranCoreExecution":
        if not self._entry(self._pointers, self._count):
            raise RuntimeError("packed Fortran entry rejected its arguments")
        return self


class FortranCoreNative:
    """A compiled Fortran core behind the packed pointer-array ABI.

    The published API records name every parameter ``t<value_id>`` and name
    each dynamic extent with the value id of the array it measures, so the
    same value-id-keyed feed dictionary every other lane consumes marshals
    this one too -- nothing here knows the operator, only the records.
    """

    #: Loaded Fortran libraries, kept for the life of the process.
    #:
    #: A standalone-linked gfortran artifact carries its own runtime, and
    #: unloading that runtime at DLL_PROCESS_DETACH takes the process down
    #: -- observed as an access violation inside garbage collection the
    #: moment the last reference to a compiled core was dropped, long
    #: after the kernel itself had run correctly. Holding the handle here
    #: means a library is loaded once, never unloaded, and re-used by
    #: every later variant that resolves to the same path: the crash
    #: cannot happen, and repeated loads of the same artifact stop
    #: happening too.
    _LIBRARIES: dict = {}

    def __init__(self, library_path: Path, entry_record):
        import ctypes

        self._entry_record = entry_record
        symbol = f"{entry_record.symbol}__packed"
        resolved = str(Path(library_path).resolve())
        library = FortranCoreNative._LIBRARIES.get(resolved)
        if library is None:
            library = ctypes.CDLL(resolved)
            FortranCoreNative._LIBRARIES[resolved] = library
        self._library = library
        function = getattr(library, symbol)
        function.restype = ctypes.c_int
        function.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
        ]
        self._packed = function

    def prepare_execution(self, feeds) -> FortranCoreExecution:
        import ctypes
        import re

        parameters = tuple(self._entry_record.parameters)
        slots = []
        buffers: dict[int, Any] = {}
        for parameter in parameters:
            dtype = "int32" if str(parameter.c_type) == "int32_t" else "float64"
            if str(getattr(parameter, "role", "")) == "extent":
                # extent_dynamic_<hash>_<value_id>_<axis>: the runtime
                # length of the array that value id was fed with.
                matched = re.search(r"_(\d+)_(\d+)$", str(parameter.name))
                measured = np.asarray(feeds[int(matched.group(1))])
                held = np.asarray([measured.size], dtype=dtype)
            else:
                value_id = int(str(parameter.name).lstrip("t"))
                fed = feeds.get(value_id)
                held = np.ascontiguousarray(
                    np.atleast_1d(np.asarray(
                        0 if fed is None else fed
                    )), dtype=dtype,
                )
                buffers[value_id] = held
            slots.append(held)
        pointers = (ctypes.c_void_p * len(slots))(*(
            ctypes.c_void_p(int(held.ctypes.data)) for held in slots
        ))
        return FortranCoreExecution(
            buffers, self._packed, pointers, len(slots),
        )


