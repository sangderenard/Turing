"""Direct scalar repository-SSA to C emission.

This lane consumes ``Function`` instructions directly.  It does not construct
or consult a FusedProgram; unsupported SSA operations remain explicit
shortfalls.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from collections import Counter
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa import Function, IRModule
from .output_publication import (
    function_output_publications,
    publication_surface_plan,
)
from .ssa_aggregate_abi import analyze_aggregate_abi, is_storage_view


@dataclass(frozen=True, slots=True)
class CEmissionShortfall:
    operation: str
    reason: str


def summarize_c_shortfalls(
    shortfalls: Sequence[CEmissionShortfall],
) -> tuple[str, ...]:
    """Collapse causal C gaps separately from unavailable-operand fallout."""

    primary = Counter(
        (item.operation, item.reason)
        for item in shortfalls
        if item.operation != "operand"
    )
    unavailable = Counter(
        (
            item.reason.rpartition(" in ")[2]
            if " in " in item.reason else "unknown function"
        )
        for item in shortfalls
        if item.operation == "operand"
    )
    result = [
        f"{operation}: {reason}" + (f" ({count} occurrences)" if count > 1 else "")
        for (operation, reason), count in primary.items()
    ]
    # Name the first few unavailable operands per function: a fallout count
    # alone cannot be traced to the value whose emission failed.
    samples: dict[str, list[str]] = {}
    for item in shortfalls:
        if item.operation != "operand":
            continue
        function = (
            item.reason.rpartition(" in ")[2]
            if " in " in item.reason else "unknown function"
        )
        detail = item.reason.rpartition(" in ")[0] if " in " in item.reason else item.reason
        bucket = samples.setdefault(function, [])
        if len(bucket) < 6 and detail not in bucket:
            bucket.append(detail)
    result.extend(
        f"operand fallout: {count} unavailable use(s) in {function}"
        + (f" [{'; '.join(samples.get(function, ()))}]" if samples.get(function) else "")
        for function, count in unavailable.items()
    )
    return tuple(result)


@dataclass(slots=True)
class CFunctionArtifact:
    name: str
    source: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    shortfalls: tuple[CEmissionShortfall, ...]
    output_publications: tuple[Mapping[str, Any], ...] = ()
    output_surfaces: Mapping[str, Any] = field(default_factory=dict)
    #: True when any emitted instruction belonged to a precision section;
    #: gates the fast-math refusal below, same as the module artifact.
    precision_sections: bool = False
    library_path: Path | None = None
    _entry: Any = field(default=None, repr=False)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def compile(
        self, directory: str | Path, *, optimization: str = "O2",
    ) -> "CFunctionArtifact":
        if not self.complete:
            raise ValueError("C artifact has emission shortfalls")
        from .work_contract import active_contract

        flags = tuple(map(str, active_contract().compiler_flags))
        if self.precision_sections:
            hostile = tuple(
                flag for flag in flags if flag in _PRECISION_HOSTILE_FLAGS
            )
            if hostile:
                raise RuntimeError(
                    "refusing to build a precision section under "
                    f"{hostile!r}: these flags override FP_CONTRACT OFF, so "
                    "every residual would come back exactly zero. Build "
                    "under a contract without fast-math."
                )
        if optimization not in {"O0", "O1", "O2", "O3", "Os", "Oz"}:
            raise ValueError(f"unsupported C optimization level {optimization!r}")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        source_path = destination / f"{self.name}.c"
        library_path = destination / f"{self.name}.dll"
        source_path.write_text(self.source, encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable, "-m", "ziglang", "cc", "-shared",
                f"-{optimization}",
                *flags,
                "-o", str(library_path), str(source_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not library_path.is_file():
            raise RuntimeError(
                f"C compile failed ({completed.returncode}):\n"
                + completed.stderr[-2000:]
            )
        self.library_path = library_path
        return self

    def entry(self):
        if self.library_path is None:
            raise RuntimeError("C artifact was not compiled")
        if self._entry is None:
            function = getattr(ctypes.CDLL(str(self.library_path)), self.name)
            function.restype = None
            pointer = ctypes.POINTER(ctypes.c_double)
            function.argtypes = [pointer, pointer]
            self._entry = function
        return self._entry

    def run(self, inputs: Mapping[str, float] | Sequence[float]) -> tuple[float, ...]:
        values = (
            [float(inputs[name]) for name in self.input_names]
            if isinstance(inputs, Mapping)
            else [float(value) for value in inputs]
        )
        if len(values) != len(self.input_names):
            raise ValueError("C input count does not match emitted ABI")
        input_array = (ctypes.c_double * len(values))(*values)
        output_array = (ctypes.c_double * len(self.output_names))()
        self.entry()(input_array, output_array)
        return tuple(map(float, output_array))


#: min(8, cores-1) spirit, matching deployment_native_emission; a
#: calibration verdict can override at the product seam later.
_DEFAULT_POOL_WORKERS = 7

#: Extern surface of turing_pool.c, emitted only when a deployment outline
#: actually pools. Linking the pool stays an optimization: every emitted
#: deploy keeps its serial loop as the in-text fallback.
_POOL_DECLARATIONS = (
    "typedef void (*turing_span_fn)(void *, long, long);",
    "extern int turing_pool_start(int workers);",
    "extern int turing_pool_deploy_span(turing_span_fn fn, void *context, "
    "long item_count, long chunk_size);",
    "extern void turing_pool_effect_lock(void);",
    "extern void turing_pool_effect_unlock(void);",
)


_BINARY = {
    "Add": "+", "Sub": "-", "Mul": "*", "Div": "/",
}

_INTEGER_BINARY = {
    "And": "&", "Or": "|", "Xor": "^", "Shl": "<<",
    "AShr": ">>", "LShr": ">>",
    "BitAnd": "&", "BitOr": "|", "BitXor": "^", "Shr": ">>",
}

_LOGICAL_BINARY = {"LAnd": "&&", "LOr": "||"}

_C_MATH_INTRINSICS = {
    "pow": "pow",
    "llvm.floor.f64": "floor",
    "llvm.sqrt.f64": "sqrt",
    "llvm.fabs.f64": "fabs",
    "llvm.round.f64": "round",
    "llvm.trunc.f64": "trunc",
    "llvm.ceil.f64": "ceil",
    "exp": "exp", "log": "log", "tanh": "tanh",
    "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan",
    "sinh": "sinh", "cosh": "cosh",
    "asinh": "asinh", "acosh": "acosh", "atanh": "atanh",
}

#: Python's ``%`` and ``//`` are FLOORED -- the remainder carries the
#: divisor's sign -- while C's ``%`` and ``/`` truncate toward zero. The
#: difference is invisible until an operand goes negative, and then it is
#: an out-of-bounds read rather than a wrong number: a periodic wrap like
#: ``(row - 1) % height`` returns -1 instead of height-1 and addresses a
#: whole row BEFORE the buffer. The LLVM lane already spells the floored
#: form; these are the same correction in C, kept as named helpers because
#: the fix needs each operand twice and a macro would evaluate side
#: effects twice with it.
_FLOORED_HELPERS = (
    "static long long turing_imod(long long a, long long b) {",
    "    long long r = a % b;",
    "    return (r != 0 && ((r ^ b) < 0)) ? r + b : r;",
    "}",
    "static long long turing_ifloordiv(long long a, long long b) {",
    "    long long q = a / b, r = a % b;",
    "    return (r != 0 && ((r ^ b) < 0)) ? q - 1 : q;",
    "}",
    "static double turing_fmod(double a, double b) {",
    "    double r = fmod(a, b);",
    "    return (r != 0.0 && ((r < 0.0) != (b < 0.0))) ? r + b : r;",
    "}",
    "static double turing_ffloordiv(double a, double b) {",
    "    return floor(a / b);",
    "}",
)
#: (integer helper, double helper) per opcode.
_FLOORED = {
    "Mod": ("turing_imod", "turing_fmod"),
    "FloorDiv": ("turing_ifloordiv", "turing_ffloordiv"),
}

#: C is the one destination where both halves of the precision-section API
#: have a STANDARD answer rather than a compiler-specific one.
#:
#: `fma` is C99 <math.h> and the standard specifies it as a single rounding
#: of ``x * y + z``, so the obligation is met by the language rather than by
#: a flag that a different toolchain might spell differently or ignore.
#: There is deliberately no fallback: a target whose libm lacks `fma` must
#: shortfall, because substituting ``x * y + z`` computes a residual that is
#: identically zero -- a plausible number, silently wrong, which is worse
#: than refusing to emit.
_TERNARY = {"fma": "fma"}
_UNARY = {
    "Abs": "fabs", "Sqrt": "sqrt", "Neg": None,
    "Tanh": "tanh",
    # Range reduction is floor and nothing else, so a lane without it
    # cannot compile a transcendental outside its core's own interval.
    # C99 has all three in <math.h>.
    "Floor": "floor", "Ceil": "ceil", "Round": "nearbyint",
    "Trunc": "trunc", "Exp": "exp", "Log": "log", "Sin": "sin",
    "Cos": "cos", "Tan": "tan", "Asin": "asin", "Acos": "acos",
    "Atan": "atan", "Sinh": "sinh", "Cosh": "cosh",
    "Asinh": "asinh", "Acosh": "acosh", "Atanh": "atanh",
}
_UNARY_FOLDED = {
    key.casefold(): value for key, value in _UNARY.items()
    if value is not None
}


#: Operator spellings differ in case between the source vocabularies and the
#: repository SSA ("sin" vs "Sin"), so every table here is consulted through a
#: casefolded key. A capability that exists but is spelled differently is
#: indistinguishable from a missing one, which is how the Fortran lane's trig
#: sat unreachable.
_TRANSCENDENTAL = {"sin", "cos"}


def _table_sin_c(argument: str, shift: float, table: str, intervals: int,
                 lower: float, upper: float, periodic: bool) -> str:
    """sin(argument + shift) by interpolating the shared baked table."""

    shifted = argument if shift == 0.0 else f"({argument} + {shift.hex()})"
    span = upper - lower
    placed = (
        f"({shifted} - {span.hex()} * floor(({shifted} - {lower.hex()})"
        f" * {(1.0 / span).hex()}))"
        if periodic else shifted
    )
    scaled = f"(({placed} - {lower.hex()}) * {(intervals / span).hex()})"
    return (
        "({ double _t = " + scaled + "; "
        f"if (_t < 0.0) _t = 0.0; if (_t > {float(intervals).hex()}) "
        f"_t = {float(intervals).hex()}; "
        "long _i = (long)_t; "
        f"if (_i >= {intervals}) _i = {intervals - 1}; "
        "double _f = _t - (double)_i; "
        f"{table}[_i] + _f * ({table}[_i + 1] - {table}[_i]); }})"
    )


def _series_sin_c(argument: str, shift: float) -> str:
    """sin(argument + shift) as arithmetic, from the shared series."""

    from .bounded_constants import sin_series_terms

    coefficients, pi, _bound = sin_series_terms()
    shifted = argument if shift == 0.0 else f"({argument} + {shift.hex()})"
    reduced = (
        f"({shifted} - {pi.hex()} * nearbyint({shifted} * {(1.0/pi).hex()}))"
    )
    horner = coefficients[0].hex()
    for coefficient in coefficients[1:]:
        horner = f"({horner} * _r2 + {coefficient.hex()})"
    return (
        "({ double _r = " + reduced + "; double _r2 = _r * _r; "
        "double _s = " + horner + " * _r; "
        "(((long long)nearbyint(" + shifted + " * " + (1.0/pi).hex()
        + ")) & 1) ? -_s : _s; })"
    )


def emit_ssa_function_to_c(
    module: IRModule, function_name: str, *, entry_name: str | None = None,
    trig_solver: str = "lut",
) -> CFunctionArtifact:
    from .ir_identities import precision_backend_shortfalls
    from .work_contract import active_contract

    function: Function = module.functions[function_name]
    name = str(entry_name or function_name)
    inexact_spellings = active_contract().inexact_identities
    # The capability table says C meets both precision obligations; this
    # call is what keeps that claim checked rather than assumed, exactly as
    # the LLVM emitters do, and it fires before any emission happens.
    precision_refusals = tuple(
        CEmissionShortfall(
            "precision_section",
            "backend cannot honour precision obligations "
            + repr(item["missing"]),
        )
        for item in precision_backend_shortfalls(
            module, "c", (function_name,)
        )
    )
    if set(function.blocks) != {"entry"}:
        return CFunctionArtifact(
            name, "", (), (),
            (CEmissionShortfall("control", "direct scalar C requires one entry block"),),
        )
    input_names = tuple(function.metadata.get("argument_names", ()))
    if len(input_names) != len(function.args):
        input_names = tuple(f"arg{index}" for index in range(len(function.args)))
    output_names = tuple(function.metadata.get("output_names", ()))
    expressions = {int(value.id): f"in[{index}]" for index, value in enumerate(function.args)}
    constants: dict[int, float] = {}
    lines: list[str] = []
    outputs: tuple[int, ...] = ()
    shortfalls: list[CEmissionShortfall] = list(precision_refusals)
    emitted_tables: dict[str, str] = {}
    precision_present = any(
        instruction.attributes.get("precision_section")
        for block in function.blocks.values()
        for instruction in block.instrs
    )

    def expression(value_id: int) -> str | None:
        value = expressions.get(int(value_id))
        if value is None:
            shortfalls.append(CEmissionShortfall("operand", f"%t{value_id} is unavailable"))
        return value

    for instruction in function.blocks["entry"].instrs:
        op = str(instruction.op)
        if op == "Const" and instruction.res is not None:
            value = float(instruction.attributes.get("constant", instruction.attributes.get("value")))
            constants[int(instruction.res.id)] = value
            expressions[int(instruction.res.id)] = value.hex()
            continue
        if op == "Ret":
            outputs = tuple(int(value.id) for value in instruction.args)
            continue
        if op == "Pi" and instruction.res is not None:
            # One home for the constant across every lane: the same
            # materialisation, and the same declared error bound, the LLVM
            # backend uses. A local 3.14159... literal here would be a second
            # definition that could drift from it.
            from .bounded_constants import materialize_pi

            materialization = materialize_pi(
                instruction.attributes.get("constant_solver") or "literal",
                instruction.attributes.get("requested_epsilon"),
            )
            if materialization.value is None:
                shortfalls.append(CEmissionShortfall(
                    op, "pi materialisation was rejected",
                ))
                continue
            constants[int(instruction.res.id)] = float(materialization.value)
            expressions[int(instruction.res.id)] = float(
                materialization.value
            ).hex()
            continue
        if instruction.res is None:
            shortfalls.append(CEmissionShortfall(op, "instruction has no result"))
            continue
        args = [expression(value.id) for value in instruction.args]
        if any(value is None for value in args):
            continue
        result_id = int(instruction.res.id)
        rendered = None
        if op in {"Cast", "CastLike", "cast_like"} and len(args) >= 1:
            target = str(
                instruction.attributes.get("target_dtype")
                or instruction.res.dtype
                or "float64"
            ).casefold()
            if target in {"bool", "i1"}:
                rendered = f"(({args[0]}) != 0.0)"
            elif target in {"int", "int32", "i32"}:
                rendered = f"((double)((int)({args[0]})))"
            elif target in {"int64", "i64", "long"}:
                rendered = f"((double)((long long)({args[0]})))"
            else:
                rendered = args[0]
        elif op.casefold() in _TERNARY and len(args) == 3:
            rendered = (
                f"{_TERNARY[op.casefold()]}"
                f"({args[0]}, {args[1]}, {args[2]})"
            )
        elif op in _BINARY and len(args) == 2:
            rendered = f"({args[0]} {_BINARY[op]} {args[1]})"
        elif op in _FLOORED and len(args) == 2:
            # This lane is all-double, so only the floating spelling can
            # apply; the floored correction is the same one either way.
            rendered = f"{_FLOORED[op][1]}({args[0]}, {args[1]})"
        elif op in {"Max", "Min"} and len(args) == 2:
            rendered = f"f{op.lower()}({args[0]}, {args[1]})"
        elif op == "Neg" and len(args) == 1:
            rendered = f"(-{args[0]})"
        elif (
            op in {"Call", "call"}
            and len(args) == 1
            and str(instruction.attributes.get("callee") or "") in {
                "acos", "acosh", "asin", "asinh", "atan", "atanh",
                "cos", "cosh", "exp", "log", "sin", "sinh", "sqrt",
                "tan", "tanh",
            }
        ):
            rendered = (
                f"{instruction.attributes['callee']}({args[0]})"
            )
        elif op.casefold() in _TRANSCENDENTAL and len(args) == 1:
            from .bounded_constants import materialize_pi

            if str(trig_solver) not in {"lut", "continuous"}:
                shortfalls.append(CEmissionShortfall(
                    op, f"unknown trig solver {trig_solver!r}; expected 'lut' or 'continuous'",
                ))
                continue
            shift = (
                0.0 if op.casefold() == "sin"
                else float(materialize_pi("literal").value) * 0.5
            )
            if str(trig_solver) == "lut":
                from .fused_program_wasm_backend import lut_for

                values, _achieved, lower, upper, periodic = lut_for("sin")
                table_name = f"_turing_sin_table"
                if table_name not in emitted_tables:
                    emitted_tables[table_name] = (
                        "    static const double "
                        + table_name
                        + f"[{len(values)}] = {{"
                        + ", ".join(value.hex() for value in values)
                        + "};"
                    )
                rendered = _table_sin_c(
                    args[0], shift, table_name, len(values) - 1,
                    lower, upper, periodic,
                )
            else:
                rendered = _series_sin_c(args[0], shift)
        elif op.casefold() in _UNARY_FOLDED and len(args) == 1:
            rendered = f"{_UNARY_FOLDED[op.casefold()]}({args[0]})"
        elif op == "Pow" and len(args) == 2:
            exponent = constants.get(int(instruction.args[1].id))
            # Exact spellings only; the sqrt-family reductions change bits and
            # belong to ir_identities under the work contract, not to a
            # private table here. pow() is the faithful fallback.
            spellings = {
                2.0: f"({args[0]} * {args[0]})",
                -1.0: f"(1.0 / {args[0]})",
            }
            if inexact_spellings:
                spellings[-2.0] = f"(1.0 / ({args[0]} * {args[0]}))"
                spellings[0.5] = f"sqrt({args[0]})"
            rendered = spellings.get(exponent, f"pow({args[0]}, {args[1]})")
        if rendered is None:
            shortfalls.append(CEmissionShortfall(op, "no direct scalar C spelling"))
            continue
        expressions[result_id] = f"t{result_id}"
        lines.append(f"    const double t{result_id} = {rendered};")

    if not output_names:
        output_names = tuple(f"output{index}" for index in range(len(outputs)))
    if len(output_names) != len(outputs):
        shortfalls.append(CEmissionShortfall("Ret", "output names do not match return arity"))
    stores = []
    for index, value_id in enumerate(outputs):
        value = expression(value_id)
        if value is not None:
            stores.append(f"    out[{index}] = {value};")
    source = "\n".join((
        "#include <math.h>",
        "#include <limits.h>",
        "#include <stddef.h>",
        # SECTION_ISOLATION, in the language's own words rather than a
        # toolchain's. C already evaluates an expression as written -- it
        # grants no licence to reassociate -- so contraction is the only
        # rewrite the standard permits behind the author's back, and this
        # withdraws it. Every fused multiply-add still present is then one
        # this compiler emitted on purpose.
        #
        # Two caveats worth keeping visible. GCC does not implement this
        # pragma and wants -ffp-contract=off instead; clang, which is what
        # `zig cc` is, honours it. And -ffast-math overrides everything
        # here, so a precision section must never be built under it.
        "#pragma STDC FP_CONTRACT OFF",
        "#if defined(_WIN32)",
        "#define TURING_EXPORT __declspec(dllexport)",
        "#else",
        "#define TURING_EXPORT __attribute__((visibility(\"default\")))",
        "#endif",
        *_FLOORED_HELPERS,
        f"TURING_EXPORT void {name}(const double *in, double *out) {{",
        *emitted_tables.values(),
        *lines,
        *stores,
        "}",
        "",
    ))
    publications = function_output_publications(function)
    return CFunctionArtifact(
        name,
        source,
        input_names,
        output_names,
        tuple(shortfalls),
        publications,
        publication_surface_plan(publications, target="c"),
        precision_sections=precision_present,
    )


# ---------------------------------------------------------------------------
# Repository-call module emission: the batched lane.
#
# The scalar artifact above is one straight-line block behind a
# ``(const double *in, double *out)`` ABI. The kernels the precision
# benchmark builds are not that shape: a five-block counted loop calling a
# planned region once per element, with arrays addressed through
# GetElementPtr/Load/Store. This emitter takes that shape directly and
# presents the SAME public ABI the LLVM lane's artifacts use --
# ``void entry(void **buffers, long long *extents)`` with ``buffer_order``
# naming which SSA value each slot carries -- so a harness drives either
# lane with the same feed dictionary and reads outputs the same way.
# ---------------------------------------------------------------------------


#: Flags that would silently defeat `#pragma STDC FP_CONTRACT OFF` and the
#: written association order. A precision artifact refuses to build under
#: them rather than returning residuals that are identically zero -- the
#: failure would be a plausible number, which is worse than an error.
_PRECISION_HOSTILE_FLAGS = (
    "-ffast-math", "-Ofast", "-funsafe-math-optimizations",
    "-fassociative-math", "-ffp-contract=fast", "-ffp-contract=on",
)


def _scalar_c_type(dtype) -> str:
    name = str(dtype or "float64").casefold()
    if name == "ptr":
        return "double *"
    if name in {"bool", "i1"}:
        return "uint8_t"
    if name in {"int", "int32", "i32"}:
        return "int32_t"
    if name in {"int64", "i64", "long"}:
        return "int64_t"
    return "double"


def _unsigned_c_type(dtype) -> str:
    name = str(dtype or "int64").casefold()
    if name in {"bool", "i1"}:
        return "uint8_t"
    if name in {"int", "int32", "i32"}:
        return "uint32_t"
    return "uint64_t"


def _is_integer_dtype(dtype) -> bool:
    return _scalar_c_type(dtype) in {"uint8_t", "int32_t", "int64_t"}


def supported_scalar_operations() -> frozenset[str]:
    """Repository scalar opcodes the whole-module C lane spells directly."""

    return frozenset(
        set(_BINARY)
        | set(_INTEGER_BINARY)
        | set(_LOGICAL_BINARY)
        | set(_FLOORED)
        | set(_UNARY)
        | set(_C_COMPARISONS)
        | set(_TERNARY)
        | {
            "Cast", "CastLike", "cast_like", "Select", "Not", "LNot",
            "SExt", "ZExt", "Trunc", "UiToFp", "SiToFp", "FpToSi",
            "SIToFP", "FPToSI", "FpToUi", "ULt", "ULe", "UGt", "UGe", "Invert", "Max",
            "Min", "Pow", "Exp", "Log", "Sin", "Cos",
        }
    )


def supported_tensor_operations() -> frozenset[str]:
    """Tensor operations available through the shared repository SSA basis."""

    from .ssa_llvm_backend import supported_tensor_operations as llvm_tensor_ops

    # Both native lanes consume the same lowered repository helper closure.
    # Keeping this derived from that shared basis prevents target selection
    # from accidentally auditing the older AbstractTensor C runtime instead.
    return llvm_tensor_ops()


def _buffer_c_type(dtype) -> str:
    """The type a buffer SLOT holds, as opposed to arithmetic width.

    Feeds supply count buffers as int32, so the pointer type must match the
    memory, and the value widens on load rather than on the way in.
    """

    name = str(dtype or "float64").casefold()
    if name == "ptrptr_float64":
        return "double *"
    if name in {"bool", "i1"}:
        return "uint8_t"
    if name in {"int", "int32", "i32"}:
        return "int32_t"
    if name in {"int64", "i64", "long"}:
        return "int64_t"
    return "double"


def _value_buffer_c_type(value) -> str:
    """Return the declared C element type of one physical SSA buffer.

    ``SSAValue.dtype`` is the value's semantic dtype.  Tensor lowering may
    deliberately use a different storage dtype (for example, boolean masks
    carried through the repository's double-backed tensor helper ABI).  That
    distinction is already recorded by the lowering in ``physical_dtype``;
    buffer declarations must not silently throw it away.  Scalar declarations
    continue to use ``_scalar_c_type`` and therefore remain integer controls.
    """

    accounting = dict(getattr(value, "accounting", None) or {})
    return _buffer_c_type(accounting.get("physical_dtype", value.dtype))


def _pointer_value_depth(value) -> int:
    """Return pointer depth carried by the SSA value itself.

    This is deliberately separate from whether the value must be passed by
    address. A float64 tensor and a mutable float64 scalar both have pointer
    depth zero even though their C ABI parameters are pointers. Conversely,
    LLVM ``ptr`` and repository ``ptrptr_float64`` values carry one and two
    pointer levels as values. Conflating those facts turns a loaded pointer
    into the address of the local variable that holds it.
    """

    name = str(getattr(value, "dtype", None) or "").casefold()
    if name == "ptrptr_float64":
        return 2
    if name == "ptr":
        return 1
    return 0


def _publication_element_count(source_value, source_address: str) -> int:
    """Count the physical value published by one Ret operand.

    A local spelled ``&tN`` is one scalar even if a stale call-site view shape
    has been attached to the same SSA identity.
    """

    if source_address == f"&t{int(source_value.id)}":
        return 1
    shape = tuple(source_value.shape or ())
    return math.prod(map(int, shape)) if shape else 1


def _publication_source_value(output, output_index: int, returned_values):
    """Correlate a native output with its exact Ret value when available.

    Ret may include in/out formals that aggregate analysis intentionally omits
    from the separate output-parameter list. Pure positional pairing would
    then shift every later record field by one. Exact SSA identity wins;
    position remains the fallback for renamed loop-carried publications.
    """

    exact = next((
        value for value in returned_values
        if int(value.id) == int(output.id)
    ), None)
    if exact is not None:
        return exact
    return (
        returned_values[output_index]
        if output_index < len(returned_values)
        else output
    )


def _flatten_numeric_aggregate(value) -> tuple[bool | int | float, ...]:
    """Flatten one repository aggregate constant in row-major order."""

    if isinstance(value, (list, tuple)):
        return tuple(
            scalar
            for member in value
            for scalar in _flatten_numeric_aggregate(member)
        )
    if isinstance(value, (bool, int, float)):
        return (value,)
    raise TypeError(
        "C aggregate constants require bool/int/float elements; "
        f"received {type(value).__name__}"
    )


def _c_numeric_literal(value: bool | int | float, c_type: str) -> str:
    if c_type in {"uint8_t", "int32_t", "int64_t", "long long"}:
        return str(int(value))
    return float(value).hex()


def _numpy_dtype(dtype) -> str:
    name = str(dtype or "float64").casefold()
    if name in {"bool", "i1", "uint8", "u8"}:
        return "bool"
    if name in {"int", "int32", "i32"}:
        return "int32"
    if name in {"int64", "i64", "long"}:
        return "int64"
    return "float64"


def _dtype_for_c_storage(c_type: str) -> str:
    """Repository dtype spelling for an already-solved public C buffer."""

    if c_type == "uint8_t":
        return "bool"
    if c_type == "int32_t":
        return "int32"
    if c_type == "int64_t":
        return "int64"
    return "float64"


@dataclass(slots=True)
class CModuleExecution:
    """Allocated public buffers and the bound native entry, ready to run."""

    artifact: "CModuleArtifact"
    buffers: dict[int, Any]
    _pointers: Any = field(default=None, repr=False)
    _extents: Any = field(default=None, repr=False)

    def run(self) -> "CModuleExecution":
        self.artifact.entry()(self._pointers, self._extents)
        return self


@dataclass(frozen=True, slots=True)
class CStandaloneExecutable:
    """A C-module channel packaged with the native pointer-table host."""

    directory: Path
    executable_path: Path
    module_source_path: Path
    host_source_path: Path
    initial_state_path: Path
    final_outputs_path: Path
    entrypoint: str

    def run(self, *, frames: int = 1) -> subprocess.CompletedProcess[str]:
        if frames < 0:
            raise ValueError("native C shell frame count cannot be negative")
        return subprocess.run(
            [str(self.executable_path), str(frames)],
            cwd=str(self.directory),
            capture_output=True,
            text=True,
            check=True,
        )


@dataclass(slots=True)
class CModuleArtifact:
    """A whole repository-call module emitted as one C translation unit."""

    name: str
    source: str
    buffer_order: tuple[int, ...]
    buffer_dtypes: tuple[str, ...]
    shortfalls: tuple[CEmissionShortfall, ...]
    buffer_shapes: tuple[tuple[int, ...], ...] = ()
    extent_order: tuple[tuple[int, str, int | None], ...] = ()
    #: True when any emitted instruction belonged to a precision section.
    #: Carried on the artifact because the refusal it gates happens at
    #: COMPILE time, where the instructions are no longer visible.
    precision_sections: bool = False
    #: True when an outlined deployment lane emitted a native pool deploy;
    #: compile() then links turing_pool.c into the artifact. The emitted
    #: source always carries its own serial fallback, so a build without the
    #: pool is still correct -- just undeployed.
    pool_required: bool = False
    pooled_regions: tuple[tuple[str, int], ...] = ()
    library_path: Path | None = None
    _entry: Any = field(default=None, repr=False)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def compile(
        self, directory: str | Path, *, optimization: str = "O2",
    ) -> "CModuleArtifact":
        if not self.complete:
            raise ValueError(
                "C module artifact has emission shortfalls: "
                + "; ".join(
                    f"{item.operation}: {item.reason}"
                    for item in self.shortfalls[:8]
                )
            )
        from .work_contract import active_contract

        flags = tuple(map(str, active_contract().compiler_flags))
        if self.precision_sections:
            hostile = tuple(
                flag for flag in flags if flag in _PRECISION_HOSTILE_FLAGS
            )
            if hostile:
                raise RuntimeError(
                    "refusing to build a precision section under "
                    f"{hostile!r}: these flags override FP_CONTRACT OFF and "
                    "the written association order, so every residual would "
                    "come back exactly zero -- a plausible wrong answer "
                    "instead of a loud one. Build under a contract without "
                    "fast-math."
                )
        if optimization not in {"O0", "O1", "O2", "O3", "Os", "Oz"}:
            raise ValueError(f"unsupported C optimization level {optimization!r}")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        source_path = destination / f"{self.name}.c"
        library_path = destination / f"{self.name}.dll"
        source_path.write_text(self.source, encoding="utf-8")
        pool_sources: list[str] = []
        if self.pool_required:
            pool_home = (
                Path(__file__).resolve().parent.parent
                / "common" / "tensors" / "accelerator_backends" / "c_backend"
            )
            for pool_name in ("turing_pool.c", "turing_pool.h"):
                (destination / pool_name).write_text(
                    (pool_home / pool_name).read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            pool_sources.append(str(destination / "turing_pool.c"))
        command = [
            sys.executable, "-m", "ziglang", "cc", "-shared",
            f"-{optimization}",
            *flags, "-o", str(library_path), str(source_path), *pool_sources,
        ]
        # zig's bundled mingw-w64 CRT is built on demand into a shared
        # cache, and back-to-back invocations occasionally lose that race
        # ("error: sub-compilation of mingw-w64"). The failure is in the
        # toolchain's own bookkeeping, not in the source being compiled, so
        # one retry is honest; a second failure is reported as real.
        completed = None
        for _attempt in range(2):
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False,
            )
            if completed.returncode == 0 and library_path.is_file():
                self.library_path = library_path
                return self
            if "sub-compilation" not in (completed.stderr or ""):
                break
        raise RuntimeError(
            f"C compile failed ({completed.returncode}):\n"
            + completed.stderr[-2000:]
        )

    def entry(self):
        if self.library_path is None:
            raise RuntimeError("C module artifact was not compiled")
        if self._entry is None:
            function = getattr(ctypes.CDLL(str(self.library_path)), self.name)
            function.restype = None
            function.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_longlong),
            ]
            self._entry = function
        return self._entry

    def prepare_execution(
        self,
        feeds: Mapping[int, Any],
        *,
        shapes: Mapping[int, Sequence[int]] | None = None,
    ) -> CModuleExecution:
        """Allocate the public buffers from real feed values.

        Mirrors the LLVM lane's ``prepare_artifact_execution``: buffer sizes
        come from the feeds, not from declared shapes, because a region
        formal's declared shape is ``()`` whether it is a scalar or the base
        of a million-element array.
        """

        import numpy as np

        feed_values = {int(key): value for key, value in dict(feeds or {}).items()}
        shape_overrides = {
            int(key): tuple(map(int, value))
            for key, value in dict(shapes or {}).items()
        }
        buffers: dict[int, Any] = {}
        shapes = self.buffer_shapes or tuple(() for _ in self.buffer_order)
        for value_id, dtype, shape in zip(
            self.buffer_order, self.buffer_dtypes, shapes
        ):
            value_id = int(value_id)
            fed = feed_values.get(value_id)
            wanted = _numpy_dtype(dtype)
            if fed is None:
                runtime_shape = shape_overrides.get(value_id, tuple(shape or ()))
                if not runtime_shape and not shape:
                    runtime_shape = (1,)
                held = np.zeros(runtime_shape, dtype=wanted)
            else:
                held = np.ascontiguousarray(
                    np.atleast_1d(np.asarray(fed)), dtype=wanted
                )
                expected = shape_overrides.get(value_id)
                if expected is not None and tuple(held.shape) != expected:
                    raise ValueError(
                        f"feed {value_id} shape {held.shape!r} != {expected!r}"
                    )
            buffers[value_id] = held
        pointers = (ctypes.c_void_p * len(self.buffer_order))(*(
            ctypes.c_void_p(int(buffers[int(value_id)].ctypes.data))
            for value_id in self.buffer_order
        ))
        extent_values: list[int] = []
        for value_id, kind, axis in self.extent_order:
            held = buffers[int(value_id)]
            if kind in {"numel", "element_count"}:
                extent_values.append(int(held.size))
            elif kind == "rank":
                extent_values.append(int(held.ndim))
            elif kind in {"dim", "shape"} and axis is not None:
                extent_values.append(int(held.shape[int(axis)]))
            else:
                raise ValueError(
                    f"extent ({value_id}, {kind!r}, {axis!r}) cannot be measured"
                )
        extents = (ctypes.c_longlong * max(1, len(extent_values)))(
            *(extent_values or (0,))
        )
        return CModuleExecution(
            artifact=self,
            buffers=buffers,
            _pointers=pointers,
            _extents=extents,
        )

    def compile_standalone(
        self,
        directory: str | Path,
        feeds: Mapping[int, Any],
        *,
        optimization: str = "O2",
    ) -> CStandaloneExecutable:
        """Compile this complete C channel into a Python-free executable.

        The generated host owns one material-buffer table.  Its heap is the
        shell's existing per-program storage boundary, not compiler scheduling
        or an allocation performed by tensor operations.  The module receives
        exactly the same ``void **buffers`` ABI as :meth:`prepare_execution`.
        """

        if not self.complete:
            raise ValueError(
                "C module artifact has emission shortfalls: "
                + "; ".join(
                    f"{item.operation}: {item.reason}"
                    for item in self.shortfalls[:8]
                )
            )
        import numpy as np

        # Store absolute sibling paths in the returned artifact.  ``run``
        # deliberately changes cwd to this directory; retaining a relative
        # executable such as ``build/foo/foo.exe`` would otherwise apply that
        # prefix twice and make the native host miss its state file.
        destination = Path(directory).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        module_source_path = destination / f"{self.name}.c"
        host_source_path = destination / f"{self.name}_host.c"
        executable_path = destination / (
            f"{self.name}.exe" if sys.platform == "win32" else self.name
        )
        initial_state_path = destination / "initial-state.bin"
        final_outputs_path = destination / "final-outputs.bin"
        module_source_path.write_text(self.source, encoding="utf-8")

        shapes = self.buffer_shapes or tuple(
            () for _ in self.buffer_order
        )
        counts: list[int] = []
        item_sizes: list[int] = []
        runtime_shapes: dict[int, tuple[int, ...]] = {}
        with initial_state_path.open("wb") as stream:
            for value_id, dtype, shape in zip(
                self.buffer_order, self.buffer_dtypes, shapes
            ):
                wanted = np.dtype(_numpy_dtype(dtype))
                fed = feeds.get(int(value_id))
                if fed is None:
                    count = (
                        math.prod(tuple(map(int, shape))) if shape else 1
                    )
                    held = np.zeros((count,), dtype=wanted)
                    runtime_shapes[int(value_id)] = tuple(map(int, shape or ()))
                else:
                    # Public buffers are sized by the authored extraction
                    # receipt.  A function-local storage analysis can recover
                    # a larger internal view for the same SSA identity; that
                    # view governs compiler workspace, never the caller's
                    # material boundary.  This is the same rule used by
                    # prepare_execution and is why feeds are accepted here by
                    # value id rather than reconstructed from inferred shapes.
                    held = np.ascontiguousarray(
                        np.atleast_1d(np.asarray(fed)), dtype=wanted
                    ).reshape(-1)
                    count = int(held.size)
                    runtime_shapes[int(value_id)] = tuple(np.asarray(fed).shape)
                stream.write(held.tobytes(order="C"))
                counts.append(int(count))
                item_sizes.append(int(wanted.itemsize))

        count_literals = ", ".join(f"{count}ULL" for count in counts)
        size_literals = ", ".join(f"{size}ULL" for size in item_sizes)
        extent_values: list[int] = []
        for value_id, kind, axis in self.extent_order:
            runtime_shape = runtime_shapes[int(value_id)]
            if kind in {"numel", "element_count"}:
                extent_values.append(
                    math.prod(runtime_shape) if runtime_shape else 1
                )
            elif kind == "rank":
                extent_values.append(len(runtime_shape))
            elif kind in {"dim", "shape"} and axis is not None:
                extent_values.append(int(runtime_shape[int(axis)]))
            else:
                raise ValueError(
                    f"extent ({value_id}, {kind!r}, {axis!r}) cannot be measured"
                )
        extent_literals = ", ".join(map(str, extent_values or (0,)))
        host_source = "\n".join((
            "#include <errno.h>",
            "#include <stdint.h>",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <string.h>",
            "",
            f"void {self.name}(void **buffers, long long *extents);",
            f"enum {{ BUFFER_COUNT = {len(self.buffer_order)} }};",
            f"static const size_t element_counts[BUFFER_COUNT] = {{{count_literals}}};",
            f"static const size_t element_sizes[BUFFER_COUNT] = {{{size_literals}}};",
            "",
            "static int transfer_buffers(FILE *stream, void **buffers, int writing) {",
            "    for (size_t i = 0; i < BUFFER_COUNT; ++i) {",
            "        size_t count = element_counts[i];",
            "        size_t size = element_sizes[i];",
            "        size_t done = writing",
            "            ? fwrite(buffers[i], size, count, stream)",
            "            : fread(buffers[i], size, count, stream);",
            "        if (done != count) return 0;",
            "    }",
            "    return 1;",
            "}",
            "",
            "static int sibling_path(char *out, size_t capacity, const char *executable, const char *name) {",
            "    const char *forward = strrchr(executable, '/');",
            "    const char *backward = strrchr(executable, '\\\\');",
            "    const char *slash = forward;",
            "    if (!slash || (backward && backward > slash)) slash = backward;",
            "    size_t prefix = slash ? (size_t)(slash - executable + 1) : 0;",
            "    size_t suffix = strlen(name);",
            "    if (prefix + suffix + 1 > capacity) return 0;",
            "    if (prefix) memcpy(out, executable, prefix);",
            "    memcpy(out + prefix, name, suffix + 1);",
            "    return 1;",
            "}",
            "",
            "int main(int argc, char **argv) {",
            "    char *end = NULL;",
            "    unsigned long long frames = 1;",
            "    if (argc > 1) {",
            "        errno = 0;",
            "        frames = strtoull(argv[1], &end, 10);",
            "        if (errno || !end || *end) {",
            "            fputs(\"invalid frame count\\n\", stderr);",
            "            return 2;",
            "        }",
            "    }",
            "    void *buffers[BUFFER_COUNT] = {0};",
            "    for (size_t i = 0; i < BUFFER_COUNT; ++i) {",
            "        buffers[i] = calloc(element_counts[i], element_sizes[i]);",
            "        if (!buffers[i]) {",
            "            fputs(\"native material allocation failed\\n\", stderr);",
            "            return 3;",
            "        }",
            "    }",
            "    char input_path[4096], output_path[4096];",
            "    if (!sibling_path(input_path, sizeof(input_path), argv[0], \"initial-state.bin\")",
            "            || !sibling_path(output_path, sizeof(output_path), argv[0], \"final-outputs.bin\")) {",
            "        fputs(\"native artifact path is too long\\n\", stderr);",
            "        return 4;",
            "    }",
            "    FILE *input = fopen(input_path, \"rb\");",
            "    if (!input || !transfer_buffers(input, buffers, 0)) {",
            "        fputs(\"could not read complete initial-state.bin\\n\", stderr);",
            "        return 4;",
            "    }",
            "    fclose(input);",
            f"    long long extents[{max(1, len(extent_values))}] = "
            f"{{{extent_literals}}};",
            "    for (unsigned long long frame = 0; frame < frames; ++frame)",
            f"        {self.name}(buffers, extents);",
            "    FILE *output = fopen(output_path, \"wb\");",
            "    if (!output || !transfer_buffers(output, buffers, 1)) {",
            "        fputs(\"could not write complete final-outputs.bin\\n\", stderr);",
            "        return 5;",
            "    }",
            "    fclose(output);",
            "    for (size_t i = 0; i < BUFFER_COUNT; ++i) free(buffers[i]);",
            f"    printf(\"entry={self.name} frames=%llu buffers=%d\\n\", frames, BUFFER_COUNT);",
            "    return 0;",
            "}",
            "",
        ))
        host_source_path.write_text(host_source, encoding="utf-8")

        from .work_contract import active_contract

        flags = tuple(map(str, active_contract().compiler_flags))
        optimization = str(optimization).lstrip("-")
        if optimization not in {"O0", "O1", "O2", "O3", "Os", "Oz"}:
            raise ValueError(
                f"unsupported C optimization level {optimization!r}"
            )
        command = [
            sys.executable, "-m", "ziglang", "cc", f"-{optimization}",
            "-std=c11",
            *flags,
            "-o", str(executable_path),
            str(module_source_path), str(host_source_path),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0 or not executable_path.is_file():
            raise RuntimeError(
                f"standalone C compile failed ({completed.returncode}):\n"
                + (completed.stderr or completed.stdout)[-4000:]
            )
        return CStandaloneExecutable(
            directory=destination,
            executable_path=executable_path,
            module_source_path=module_source_path,
            host_source_path=host_source_path,
            initial_state_path=initial_state_path,
            final_outputs_path=final_outputs_path,
            entrypoint=self.name,
        )


def _module_call_closure(module: IRModule, root: str) -> tuple[str, ...]:
    """Every function reachable from ``root`` through Call attributes."""

    pending, seen = [str(root)], []
    while pending:
        name = pending.pop()
        if name in seen or name not in module.functions:
            continue
        seen.append(name)
        for block in module.functions[name].blocks.values():
            for instruction in block.instrs:
                if instruction.op in ("Call", "call"):
                    pending.append(
                        str(instruction.attributes.get("callee") or "")
                    )
    return tuple(seen)


def emit_ssa_module_to_c(
    module: IRModule,
    function_name: str,
    *,
    entry_name: str | None = None,
    watch: Sequence[int] = (),
) -> CModuleArtifact:
    """Emit ``function_name`` and its call closure as one C module.

    Control flow becomes labels and gotos -- C's goto is exactly the
    unstructured branch the SSA already speaks, so no loop reconstruction is
    needed or wanted. A Phi becomes one mutable local assigned on every
    predecessor edge before the jump, which is the textbook out-of-SSA
    translation and preserves evaluation order exactly.
    """

    from .ir_identities import precision_backend_shortfalls
    from .ssa_storage_requirements import module_storage_requirements

    name = str(entry_name or function_name)
    reachable = _module_call_closure(module, function_name)
    storage_requirements_by_function = module_storage_requirements(module)
    aggregate_abi = analyze_aggregate_abi(module, reachable)
    aggregate_calls = {
        id(record.call): record for record in aggregate_abi.calls
    }
    from .ssa_call_storage import call_array_argument_ids
    from .ssa_llvm_backend import _declared_span_rank

    call_array_ids = call_array_argument_ids(module.functions, reachable)

    # Array arguments cross the internal ABI by address. Their physical
    # element type is therefore one call-edge contract even when semantic
    # metadata differs (notably integer-valued gather indices stored in the
    # repository's double-backed tensor arena). Propagate only an explicitly
    # authored physical dtype; scalar call edges retain the normal conversion
    # path below.
    physical_parent: dict[tuple[str, int], tuple[str, int]] = {}

    def physical_find(key: tuple[str, int]) -> tuple[str, int]:
        physical_parent.setdefault(key, key)
        if physical_parent[key] != key:
            physical_parent[key] = physical_find(physical_parent[key])
        return physical_parent[key]

    def physical_union(left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root, right_root = physical_find(left), physical_find(right)
        if left_root != right_root:
            physical_parent[right_root] = left_root

    def authored_array_contract(value, owner: str) -> bool:
        """Whether a call operand is physically an array, not just indexed.

        Repository scalar cast helpers index their single input cell with a
        GEP.  ``call_array_argument_ids`` must conservatively notice that use,
        but it is not permission to make a bool caller cell share storage type
        with a double helper formal.  Shape/span metadata is authoritative;
        an explicit physical dtype also makes inferred array use meaningful
        for legacy rankless repository buffers.
        """

        accounting = dict(value.accounting or {})
        return bool(
            tuple(value.shape or ())
            or _declared_span_rank(value) > 0
            or accounting.get("program_abi_storage") == "span"
            or (
                int(value.id) in call_array_ids.get(owner, ())
                and accounting.get("physical_dtype") is not None
            )
        )

    for caller_name in reachable:
        for block in module.functions[caller_name].blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Call", "call"}:
                    continue
                callee_name = str(instruction.attributes.get("callee") or "")
                callee = module.functions.get(callee_name)
                if callee is None or callee_name not in reachable:
                    continue
                for actual, formal in zip(instruction.args, callee.args):
                    if (
                        authored_array_contract(actual, caller_name)
                        or authored_array_contract(formal, callee_name)
                    ):
                        physical_union(
                            (caller_name, int(actual.id)),
                            (callee_name, int(formal.id)),
                        )

    explicit_physical: dict[tuple[str, int], set[str]] = {}
    for owner in reachable:
        function = module.functions[owner]
        values = [*function.args]
        values.extend(
            instruction.res
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        for value in values:
            physical_dtype = dict(value.accounting or {}).get("physical_dtype")
            if physical_dtype is None:
                continue
            explicit_physical.setdefault(
                physical_find((owner, int(value.id))), set()
            ).add(_buffer_c_type(physical_dtype))
    native_outputs = {
        fn: tuple(aggregate_abi.outputs_by_callee.get(fn, ()))
        for fn in reachable
    }
    if watch:
        root_function = module.functions[function_name]
        root_values = {int(value.id): value for value in root_function.args}
        root_values.update({
            int(instruction.res.id): instruction.res
            for block in root_function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        })
        existing_output_ids = {
            int(value.id) for value in native_outputs[function_name]
        }
        missing = [int(value_id) for value_id in watch if int(value_id) not in root_values]
        if missing:
            raise ValueError(
                f"C watch value(s) are absent from root {function_name!r}: {missing!r}"
            )
        native_outputs[function_name] = (
            *native_outputs[function_name],
            *(
                root_values[int(value_id)]
                for value_id in watch
                if int(value_id) not in existing_output_ids
            ),
        )
    span_origin: dict[tuple[str, int], tuple[str, int]] = {}
    for caller_name in reachable:
        for block in module.functions[caller_name].blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Call", "call"}:
                    continue
                callee_name = str(instruction.attributes.get("callee") or "")
                callee = module.functions.get(callee_name)
                if callee is None:
                    continue
                for actual, formal in zip(instruction.args, callee.args):
                    key = (callee_name, int(formal.id))
                    candidate = (caller_name, int(actual.id))
                    existing = span_origin.get(key)
                    candidate_is_recursive = caller_name == callee_name
                    existing_is_recursive = (
                        existing is not None and existing[0] == callee_name
                    )
                    # A retry expressed as direct recursion passes each span
                    # formal back to itself.  That identity edge contains no
                    # storage-origin information and must never overwrite the
                    # external caller edge that reaches the root ProgramABI.
                    # Conversely, replace an earlier self-edge when an
                    # external callsite is discovered later.
                    if (
                        existing is None
                        or (existing_is_recursive and not candidate_is_recursive)
                    ):
                        span_origin[key] = candidate
    # Planned output slots cross the same physical call edge as formals: the
    # caller owns the destination storage the callee writes through.  Without
    # this edge a callee-internal value that IS a planned output (used before
    # any local definition, e.g. an in-place op slot) can never resolve its
    # extents to a public origin, and the emission dies with "no public
    # storage origin" even though the caller knows the exact runtime shape.
    for record in aggregate_abi.calls:
        for projection in record.projections:
            span_origin.setdefault(
                (record.callee, int(projection.output_id)),
                (record.caller, int(projection.value.id)),
            )

    # Ids the wrapper can actually measure at run time: root formals that
    # stay caller-owned buffers, plus the root's native outputs. An extent
    # slot registered against anything else could never be filled -- the
    # loud KeyError at execution-prepare time -- so origin resolution must
    # refuse instead of registering it.
    _root_record_parameter_ids = {
        int(value_id)
        for parameter_name, value_id in module.functions[function_name]
        .metadata.get("parameter_names", ())
        if str(parameter_name) in dict(
            module.functions[function_name].metadata.get(
                "parameter_record_abi"
            ) or {}
        )
    }

    def _root_public_ids() -> set[int]:
        from .ssa_storage_requirements import is_compiler_owned_storage

        public: set[int] = set()
        for formal in module.functions[function_name].args:
            accounting = dict(formal.accounting or {})
            if (
                int(formal.id) in _root_record_parameter_ids
                or str(formal.dtype or "").casefold() == "ssa.aggregate"
                or accounting.get("program_abi_storage") == "keyed"
                or is_compiler_owned_storage(formal)
                or accounting.get("split_from_unproven_alias") is not None
                or accounting.get("split_from_result_storage") is not None
            ):
                continue
            public.add(int(formal.id))
        for output in native_outputs.get(function_name, ()):
            accounting = dict(output.accounting or {})
            if (
                str(output.dtype or "").casefold() == "ssa.aggregate"
                or accounting.get("program_abi_storage") == "keyed"
                or int(output.id) in _root_record_parameter_ids
            ):
                # Structural values never become wrapper buffers, so an
                # extent slot keyed on one could not be filled at run time.
                continue
            public.add(int(output.id))
        return public

    root_public_ids = _root_public_ids()
    _root_value_by_id = {
        int(value.id): value
        for value in (
            *module.functions[function_name].args,
            *native_outputs.get(function_name, ()),
        )
    }

    def _trusted_declared_shape(value) -> tuple[int, ...] | None:
        """A root value's declared shape, when it IS the runtime shape.

        Rankless repository buffers are declared scalar but fed arrays;
        they always carry span/rank metadata, so a declared shape with no
        such metadata is the wrapper-enforced runtime truth.
        """

        accounting = dict(value.accounting or {})
        if (
            _declared_span_rank(value) > 0
            or int(accounting.get("program_abi_rank", 0) or 0) > 0
            or accounting.get("program_abi_storage") == "span"
        ):
            return None
        shape = tuple(value.shape or ())
        if all(isinstance(item, int) for item in shape):
            return shape
        return None
    #: value-shape transfer steps the origin walk may follow: op -> which
    #: operand carries the result's runtime shape.  These are exact by the
    #: operator contracts (elementwise repository helpers write the shape of
    #: their first tensor operand; cast_like takes the schema of the operand
    #: its attribute names).
    _SHAPE_TRANSFER_ARG0 = frozenset({
        "Cast", "cast", "unary_double", "binary_scalar_double",
        # A broadcast never shrinks its source; the consumer that demanded
        # the broadcast target combines shapes itself (see the
        # binary_double broadcast-combine below), so following the SOURCE
        # is exact.
        "broadcast_double",
    })

    def _broadcast_shapes(
        left: tuple[int, ...], right: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        rank = max(len(left), len(right))
        padded_left = (1,) * (rank - len(left)) + tuple(left)
        padded_right = (1,) * (rank - len(right)) + tuple(right)
        combined: list[int] = []
        for a, b in zip(padded_left, padded_right):
            if a == b or b == 1:
                combined.append(a)
            elif a == 1:
                combined.append(b)
            else:
                return None
        return tuple(combined)

    _owner_value_index: dict[str, dict[int, Any]] = {}

    def _owner_instruction(owner: str, value_id: int):
        index = _owner_value_index.get(owner)
        if index is None:
            index = {}
            owner_function = module.functions.get(owner)
            if owner_function is not None:
                for block in owner_function.blocks.values():
                    for instruction in block.instrs:
                        if instruction.res is not None:
                            index[int(instruction.res.id)] = instruction
            _owner_value_index[owner] = index
        return index.get(int(value_id))

    def resolve_span_origin(
        owner: str, value_id: int,
    ) -> tuple[str, Any] | None:
        """Walk a value to a measurable origin.

        Returns ``("public", root_value_id)`` when a wrapper-visible buffer
        carries the runtime shape, ``("static", shape)`` when a static
        tensor descriptor terminates the walk, or ``None`` (an honest
        refusal) when neither is provable.  Edges walked: callee formal ->
        caller actual, planned output -> caller destination, the linker's
        ``ssa_call_result_from`` identity hops, and exact shape-transfer
        operators (cast_like and elementwise repository helpers).
        """

        current = (str(owner), int(value_id))
        seen: set[tuple[str, int]] = set()
        while True:
            if current in seen:
                return None
            seen.add(current)
            owner_name, vid = current
            if owner_name == function_name and vid in root_public_ids:
                root_value = _root_value_by_id.get(vid)
                if root_value is not None:
                    declared = _trusted_declared_shape(root_value)
                    if declared is not None:
                        # The wrapper enforces this shape on the fed buffer,
                        # so the extents are compile-time constants and no
                        # runtime slot is needed at all.
                        return ("static", declared)
                return ("public", vid)
            table = getattr(module, "tensor_tables", {}).get(owner_name)
            descriptor = table.by_id(vid) if table is not None else None
            if (
                descriptor is not None
                and descriptor.metadata_state == "static"
                and tuple(descriptor.shape or ())
                and all(
                    isinstance(item, int) for item in descriptor.shape
                )
            ):
                return ("static", tuple(int(item) for item in descriptor.shape))
            # The LOCAL definition is the semantic producer and is analyzed
            # first; the caller-side edge is the fallback. Consulting the
            # edge first can close a two-node loop for in-place planned
            # outputs (projection edge to the caller, identity hop straight
            # back) before the producer's shape rule is ever reached.
            instruction = _owner_instruction(owner_name, vid)
            if instruction is None:
                step = span_origin.get(current)
                if step is not None:
                    current = step
                    continue
                return None
            accounting = dict(instruction.res.accounting or {})
            hop = accounting.get("ssa_call_result_from")
            if hop:
                current = (str(hop[0]), int(hop[1]))
                continue
            operation = str(instruction.op)
            attributes = instruction.attributes or {}
            callee = str(attributes.get("callee") or "")
            if operation == "cast_like" or callee == "cast_like":
                position = int(attributes.get("target_from_operand", 1))
                if position < len(instruction.args):
                    current = (
                        owner_name, int(instruction.args[position].id),
                    )
                    continue
                return None
            if (
                operation in _SHAPE_TRANSFER_ARG0
                or callee in _SHAPE_TRANSFER_ARG0
            ) and instruction.args:
                current = (owner_name, int(instruction.args[0].id))
                continue
            if (
                (operation == "binary_double" or callee == "binary_double")
                and len(instruction.args) >= 2
            ):
                # Elementwise binary: the result shape is the numpy
                # broadcast of both operand shapes -- exact even when one
                # operand was materialized by an explicit broadcast to the
                # (circular) destination shape.
                left = resolve_span_origin(
                    owner_name, int(instruction.args[0].id)
                )
                right = resolve_span_origin(
                    owner_name, int(instruction.args[1].id)
                )
                if left is None or right is None:
                    return None
                if left[0] == "static" and right[0] == "static":
                    combined = _broadcast_shapes(
                        tuple(left[1]), tuple(right[1])
                    )
                    if combined is not None:
                        return ("static", combined)
                    return None
                # Broadcasting against a scalar (or an all-ones shape) is
                # shape-identity, so a public origin on the other operand
                # carries the result's runtime shape exactly.
                for shaped, other in ((left, right), (right, left)):
                    if (
                        other[0] == "static"
                        and all(int(item) == 1 for item in other[1])
                    ):
                        return shaped
                return None
            # Local producer has no shape rule; fall back to the caller
            # edge when one exists.
            step = span_origin.get(current)
            if step is not None:
                current = step
                continue
            return None

    def public_span_value(owner: str, value_id: int) -> int | None:
        resolved = resolve_span_origin(owner, value_id)
        if resolved is None or resolved[0] != "public":
            return None
        return int(resolved[1])

    extent_order: list[tuple[int, str, int | None]] = []
    extent_slots: dict[tuple[int, str, int | None], int] = {}

    def extent_slot(
        owner: str, value_id: int, kind: str, axis: int | None,
    ) -> int | None:
        public_id = public_span_value(owner, value_id)
        if public_id is None:
            return None
        key = (public_id, str(kind), axis)
        if key not in extent_slots:
            extent_slots[key] = len(extent_order)
            extent_order.append(key)
        return extent_slots[key]
    shortfalls: list[CEmissionShortfall] = []
    shortfalls.extend(
        CEmissionShortfall(
            "precision_section",
            "backend cannot honour precision obligations "
            + repr(item["missing"]) + f" in {item['function']}",
        )
        for item in precision_backend_shortfalls(module, "c", reachable)
    )

    # Copy the working LLVM lane's extent ABI: only functions that consume
    # dynamic extents, and callers on paths to them, receive the hidden slot.
    from .hierarchical_plan import PREDICATE_OPERATIONS
    from .ssa_llvm_backend import _declared_span_rank, scalar_likeness

    extent_users: set[str] = {
        owner for owner in reachable
        for block in module.functions[owner].blocks.values()
        for instruction in block.instrs
        if instruction.op in {"GetElementPtr", "getelementptr"}
        and len(instruction.args) > 2
    }
    extent_users.update(
        owner for owner in reachable
        for block in module.functions[owner].blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
        and _declared_span_rank(instruction.res) > 0
        and scalar_likeness(str(instruction.op)) is not None
        and str(instruction.op) not in PREDICATE_OPERATIONS
    )

    def _extent_needs_runtime(owner: str, instruction) -> bool:
        """Whether an ``extent`` op will read the runtime extents array.

        Mirrors (as a superset of) the dynamic determination inside the
        emission arm: a non-static descriptor, a dynamic storage
        requirement, or a symbolic source shape all resolve through
        ``extents[slot]``, so the owning function must carry the hidden
        extents parameter. A false positive only adds an unused parameter.
        """

        source = instruction.args[0]
        table = getattr(module, "tensor_tables", {}).get(owner)
        descriptor = (
            table.by_id(int(source.id)) if table is not None else None
        )
        if descriptor is not None and descriptor.metadata_state != "static":
            return True
        requirement = storage_requirements_by_function.get(owner, {}).get(
            int(source.id)
        )
        if requirement is not None and requirement.dynamic:
            return True
        return any(
            not isinstance(item, int)
            for item in tuple(source.shape or ())
        )

    extent_users.update(
        owner for owner in reachable
        for block in module.functions[owner].blocks.values()
        for instruction in block.instrs
        if str(
            instruction.attributes.get("tensor_operation")
            or instruction.op
        ) == "extent"
        and instruction.args
        and _extent_needs_runtime(owner, instruction)
    )
    growing = True
    while growing:
        growing = False
        for owner in reachable:
            if owner in extent_users:
                continue
            calls_extent_user = any(
                instruction.op in {"Call", "call"}
                and str(instruction.attributes.get("callee") or "")
                in extent_users
                for block in module.functions[owner].blocks.values()
                for instruction in block.instrs
            )
            if calls_extent_user:
                extent_users.add(owner)
                growing = True
    # Match the working LLVM module lane: internal values cross function
    # boundaries as opaque storage addresses. Concrete C widths are selected
    # only by the operation that loads from or stores to an address.
    def solved_buffer_type(owner: str, value) -> str:
        """Map the working LLVM backend's authored storage-width contract."""

        physical = explicit_physical.get(
            physical_find((str(owner), int(value.id))), set()
        )
        if len(physical) == 1:
            return next(iter(physical))
        from .ssa_llvm_backend import _value_llvm_type

        llvm_type = _value_llvm_type(value)
        return {
            "double": "double", "i64": "int64_t", "i32": "int32_t",
            "i1": "uint8_t", "ptr": "void *",
        }.get(llvm_type, "double")

    # Internal repository functions publish through caller-owned output
    # storage, exactly like the LLVM module lane. Authored scalar-return C
    # kernels remain external call targets and keep their own signatures.
    function_return_types = {fn: "void" for fn in reachable}

    precision_present = False
    prototypes: list[str] = []
    definitions: list[str] = []
    # Outlined deployment lanes (deployment_outlining.py) become native span
    # deploys on the persistent turing_pool. Support text is collected here:
    # ctx typedefs and trampoline prototypes precede the function prototypes,
    # trampoline bodies follow the definitions, and the emitted parent keeps
    # its serial loop as the always-correct fallback.
    deployment_support: list[str] = []
    deployment_trampolines: list[str] = []
    pooled_regions: list[tuple[str, int]] = []
    effect_guard_used = [False]
    deployment_outlines = {
        key: record
        for key, record in (
            (module.metadata or {}).get("deployment_outlines", {}).items()
        )
    }

    for fn in reachable:
        function = module.functions[fn]
        function_return_type = function_return_types[fn]
        tensor_table = getattr(module, "tensor_tables", {}).get(fn)
        formal_ids = {int(value.id) for value in function.args}
        formal_values_by_id = {
            int(value.id): value for value in function.args
        }
        values_by_id = {int(value.id): value for value in function.args}
        values_by_id.update({
            int(instruction.res.id): instruction.res
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        })
        producers_by_id = {
            int(instruction.res.id): instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        frame_allocations: list[tuple[str, str, int]] = []

        def activation_array(element_type: str, count: int) -> str:
            """Spell LLVM's per-activation alloca as owned C storage."""

            name = f"frame_storage_{len(frame_allocations)}"
            frame_allocations.append((
                element_type, name, max(1, int(count)),
            ))
            return name

        token_value_ids = {
            int(value.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.attributes.get("string_compare")
            for value in instruction.args
        } | {
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if str(instruction.op) == "string_token"
            and instruction.res is not None
        }

        def buffer_type(value) -> str:
            if int(value.id) in token_value_ids:
                return "int64_t"
            return solved_buffer_type(fn, value)

        def carried_type(value) -> str:
            depth = _pointer_value_depth(value)
            if depth == 2:
                return "double **"
            if depth == 1:
                return "double *"
            return buffer_type(value)

        def storage_expression(value, address: str) -> str:
            if _pointer_value_depth(value) > 0:
                return f"(({carried_type(value)})({address}))"
            if _declared_span_rank(value) > 0 or tuple(value.shape or ()):
                return f"(({buffer_type(value)} *)({address}))"
            return f"(*(({buffer_type(value)} *)({address})))"

        # Match the working LLVM module lane: every semantic input and output
        # crosses an internal call boundary as an opaque storage pointer.
        parameters = [
            f"void *v{formal.id}" for formal in function.args
        ]
        output_destinations: dict[int, str] = {}
        for output in native_outputs[fn]:
            output_id = int(output.id)
            if output_id in formal_ids:
                output_destinations[output_id] = f"v{output_id}"
                continue
            destination = f"out{output_id}"
            output_destinations[output_id] = destination
            parameters.append(f"void *{destination}")
        if fn in extent_users:
            parameters.append("long long *extents")
        body: list[str] = []
        expressions: dict[int, str] = {
            int(formal.id): storage_expression(formal, f"v{formal.id}")
            for formal in function.args
        }
        addresses: dict[int, str] = {
            int(formal.id): f"v{formal.id}"
            for formal in function.args
        }
        address_buffer_types: dict[int, str] = {
            int(formal.id): _value_buffer_c_type(formal)
            for formal in function.args
        }
        # Element type belongs to the addressed storage, not to the value a
        # later Load happens to produce.  Fortran indexes the typed
        # collection and lets assignment convert to the result; LLVM carries
        # the same fact in span_address_types.  Retain it here so an i64
        # induction/result cannot turn an address into an i32 shape vector
        # into an eight-byte read (or write only half of a double slot).
        address_element_types: dict[int, str] = {}
        freshened_synthetic_ids = {
            int(original_id): int(freshened_id)
            for original_id, freshened_id in function.metadata.get(
                "freshened_synthetic_value_ids", ()
            )
        }
        expressions.update({
            int(output.id): storage_expression(
                output, output_destinations[int(output.id)]
            )
            for output in native_outputs[fn]
        })
        addresses.update({
            int(output.id): output_destinations[int(output.id)]
            for output in native_outputs[fn]
        })
        address_buffer_types.update({
            int(output.id): _value_buffer_c_type(output)
            for output in native_outputs[fn]
        })
        local_tensor_declarations: list[str] = []
        if tensor_table is not None:
            for descriptor in tensor_table.tensors.values():
                value_id = int(descriptor.data_value_id)
                if (
                    value_id in expressions
                    or descriptor.storage != "temporary"
                    or not descriptor.owns_allocation
                ):
                    continue
                declared_value = values_by_id.get(value_id)
                if declared_value is None:
                    # Tensor-table planning may retain an arena descriptor
                    # whose SSA value was eliminated. LLVM allocates values
                    # on demand, so an unreferenced descriptor owns nothing.
                    continue
                producer = producers_by_id.get(value_id)
                if (
                    producer is not None
                    and str(
                        producer.attributes.get("tensor_operation")
                        or producer.op
                    ) == "extent"
                ):
                    # Extent results are scalar metadata values.  They are
                    # emitted directly by the extent arm below and must not
                    # inherit a temporary tensor allocation merely because
                    # their producer retains the generic repository Call
                    # handler.
                    continue
                if descriptor.metadata_state != "static":
                    # Runtime metadata may stay dynamic while the storage
                    # ARENA is still statically bounded: the requirements
                    # solver aggregates every real call-site binding and is
                    # the authority on the byte obligation.  Semantic shape
                    # keeps flowing through the extents machinery; only the
                    # allocation uses the proven bound.
                    requirement = storage_requirements_by_function.get(
                        fn, {}
                    ).get(value_id)
                    if (
                        requirement is not None
                        and not requirement.dynamic
                        and requirement.element_count
                    ):
                        element_type = buffer_type(declared_value)
                        storage = activation_array(
                            element_type, int(requirement.element_count)
                        )
                        expressions[value_id] = storage
                        addresses[value_id] = storage
                        continue
                    shortfalls.append(CEmissionShortfall(
                        "temporary",
                        f"dynamic temporary %t{value_id} has no native "
                        f"activation-storage contract in {fn}",
                    ))
                    continue
                count = descriptor.static_element_count
                if count is None:
                    continue
                requirement = storage_requirements_by_function.get(
                    fn, {}
                ).get(value_id)
                if (
                    requirement is not None
                    and requirement.element_count is not None
                ):
                    count = max(count, requirement.element_count)
                element_type = buffer_type(declared_value)
                storage = activation_array(element_type, count)
                expressions[value_id] = storage
                addresses[value_id] = storage
        # Aggregate projections may be consumed in a merge/header block that
        # is emitted textually before the branch/body containing their Call.
        # Allocate each call-output slot in the function frame up front and
        # bind all projection aliases now; block traversal order must not
        # decide whether a valid SSA value exists.
        aggregate_projection_storage: dict[
            tuple[int, int], tuple[str, str]
        ] = {}
        for record in aggregate_abi.calls:
            if record.caller != fn or record.callee not in native_outputs:
                continue
            for slot, output in enumerate(native_outputs[record.callee]):
                projections = tuple(
                    projection for projection in record.projections
                    if int(projection.output_id) == int(output.id)
                )
                if not projections:
                    continue
                inout_source = next((
                    int(projection.value.id)
                    for projection in projections
                    if formal_values_by_id.get(
                        int(projection.value.id)
                    ) is projection.value
                ), None)
                if inout_source is None:
                    inout_source = next((
                    int(source_id)
                    for projection in projections
                    for source_id in ((projection.value.accounting or {}).get(
                        "source_value_id"
                    ),)
                    if (
                        (projection.value.accounting or {}).get(
                            "ssa_inout_write_version"
                        )
                        and source_id is not None
                        and int(source_id) in formal_ids
                    )
                    ), None)
                existing = None if inout_source is not None else next((
                    int(projection.value.id)
                    for projection in projections
                    if (
                        int(projection.value.id) in output_destinations
                        or (
                            int(projection.value.id) in addresses
                            and int(projection.value.id) not in formal_ids
                        )
                    )
                ), None)
                if inout_source is not None:
                    projected_address = addresses[inout_source]
                    projected_expression = expressions[inout_source]
                elif existing is not None:
                    projected_address = addresses[existing]
                    projected_expression = expressions.get(
                        existing,
                        projected_address
                        if tuple(output.shape or ())
                        else storage_expression(output, projected_address),
                    )
                else:
                    first_result_id = int(projections[0].value.id)
                    local_name = f"callout{first_result_id}_{slot}"
                    element_type = buffer_type(output)
                    shape = tuple(output.shape or ())
                    if shape:
                        count = math.prod(shape)
                        requirement = storage_requirements_by_function.get(
                            record.callee, {}
                        ).get(int(output.id))
                        if (
                            requirement is not None
                            and requirement.element_count is not None
                        ):
                            count = max(count, requirement.element_count)
                        storage = activation_array(element_type, count)
                        projected_expression = storage
                        projected_address = storage
                    else:
                        local_tensor_declarations.append(
                            f"    {element_type} {local_name};"
                        )
                        projected_expression = local_name
                        projected_address = f"&{local_name}"
                aggregate_projection_storage[
                    (id(record.call), int(output.id))
                ] = (projected_expression, projected_address)
                for projection in projections:
                    projection_id = int(projection.value.id)
                    # Planner/callee namespaces may reuse an integer id that
                    # is already one of this function's native output ports.
                    # That port's caller-owned destination is authoritative;
                    # a later aggregate record must not rebind it to another
                    # projection's slot merely because their integers match.
                    if (
                        projection_id in output_destinations
                        or projection_id in formal_ids
                    ):
                        continue
                    expressions[projection_id] = projected_expression
                    addresses[projection_id] = projected_address
                    address_buffer_types[projection_id] = buffer_type(output)
        integer_ids = {
            int(formal.id) for formal in function.args
            if _is_integer_dtype(formal.dtype)
        }
        phi_declarations: dict[int, str] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) in {"Phi", "phi"} and instruction.res is not None:
                    phi_declarations[int(instruction.res.id)] = buffer_type(
                        instruction.res
                    )
        # Every Phi declaration is function-scoped and every incoming edge may
        # reference another Phi in a loop/conditional cycle. Publish all names
        # before visiting any block so textual block order cannot manufacture
        # an unavailable operand.
        expressions.update({
            int(phi_id): f"t{phi_id}" for phi_id in phi_declarations
        })
        addresses.update({
            int(phi_id): f"&t{phi_id}" for phi_id in phi_declarations
        })
        address_buffer_types.update(phi_declarations)

        def is_integer(value) -> bool:
            return (
                int(value.id) in integer_ids
                or _is_integer_dtype(value.dtype)
            )

        def operand(value) -> str | None:
            held = expressions.get(int(value.id))
            if held is None:
                shortfalls.append(CEmissionShortfall(
                    "operand", f"%t{value.id} is unavailable in {fn}",
                ))
            return held

        def address_operand(value) -> str | None:
            held = addresses.get(int(value.id), expressions.get(int(value.id)))
            if held is None:
                shortfalls.append(CEmissionShortfall(
                    "operand", f"%t{value.id} has no address in {fn}",
                ))
            return held

        def scalar_operand(value) -> str | None:
            """Load one authored scalar from opaque storage when required."""

            held = operand(value)
            if held is None or _pointer_value_depth(value) > 0:
                return held
            home = addresses.get(int(value.id))
            pointer_view = (
                home is not None
                and (
                    held == home
                    or (
                        held.startswith("((")
                        and held.endswith("))")
                        and " *)(" in held
                    )
                )
            )
            if home is not None and (
                tuple(value.shape or ())
                or _declared_span_rank(value) > 0
                or pointer_view
            ):
                return f"*(({buffer_type(value)} *)({home}))"
            return held

        def phi_edge_assignments(source_block: str, target_block: str) -> list[str]:
            """The out-of-SSA copies owed on one CFG edge."""

            assignments = []
            target = function.blocks.get(target_block)
            if target is None:
                return assignments
            for instruction in target.instrs:
                if str(instruction.op) != "Phi" or instruction.res is None:
                    continue
                incoming = tuple(
                    instruction.attributes.get("incoming_blocks") or ()
                )
                for position, origin in enumerate(incoming):
                    if str(origin) == source_block and position < len(
                        instruction.args
                    ):
                        value = scalar_operand(instruction.args[position])
                        if value is not None:
                            assignments.append(
                                f"        t{instruction.res.id} = {value};"
                            )
            return assignments

        def output_publications(returned_values: Sequence = ()) -> list[str]:
            """Publish results by the same authority as the LLVM lane.

            The Ret operands are the physical values being returned.  A
            carried value can therefore have a different id from the
            declared output port.  Planned regions without Ret instructions
            fall back to the declared output ids, exactly as LLVM does after
            its aggregate-output analysis.
            """
            publications = []
            published_ids: set[int] = set()
            for output_index, output in enumerate(native_outputs[fn]):
                output_id = int(output.id)
                published_ids.add(output_id)
                destination = output_destinations.get(output_id)
                source_value = _publication_source_value(
                    output, output_index, returned_values,
                )
                source_id = freshened_synthetic_ids.get(
                    int(source_value.id), int(source_value.id)
                )
                source_value = values_by_id.get(source_id, source_value)
                source = addresses.get(source_id)
                if source is None:
                    source = addresses.get(output_id)
                if source is None or destination is None or source == destination:
                    continue

                # The callee's Ret operand is the physical publication
                # authority.  A call-site occurrence can carry a broader
                # (and, while contracts are being reconciled, occasionally
                # stale) view shape than the value the callee actually
                # returns.  Copying by the native output port's shape turns a
                # scalar reduction into an out-of-bounds read from its one
                # stack scalar.  Planned regions without Ret operands already
                # select ``output`` as source_value, preserving their declared
                # aggregate-copy behaviour.
                count = _publication_element_count(source_value, source)
                if count == 1:
                    carried = carried_type(output)
                    publications.append(
                        f"        *(({carried} *)({destination})) = "
                        f"*(({carried} *)({source}));"
                    )
                else:
                    publications.append(
                        f"        (void)memcpy({destination}, {source}, "
                        f"sizeof({buffer_type(output)}) * {count});"
                    )
            # Mutable scalar record fields are caller-owned in/out storage.
            # A planned reduction can deliberately reuse that formal's SSA id
            # for its result.  Scalar emission then changes ``addresses[id]``
            # from the incoming pointer to the produced stack scalar.  Such a
            # field need not also appear in aggregate_abi.outputs_by_callee,
            # so publish it explicitly instead of silently leaving the caller
            # unchanged.  Shaped fields are already mutated through their
            # resident pointer and require no copy here.
            for formal in function.args:
                formal_id = int(formal.id)
                accounting = dict(formal.accounting or {})
                if (
                    formal_id in published_ids
                    or not accounting.get("program_abi_field_written", False)
                    or (
                        tuple(formal.shape or ())
                        and accounting.get("program_abi_storage") != "scalar"
                    )
                ):
                    continue
                destination = f"v{formal_id}"
                source = addresses.get(
                    freshened_synthetic_ids.get(formal_id, formal_id)
                )
                if source is None or source == destination:
                    continue
                carried = carried_type(formal)
                publications.append(
                    f"        *(({carried} *)({destination})) = "
                    f"*(({carried} *)({source}));"
                )
            return publications

        aggregate_projection_instruction_ids = {
            id(instruction)
            for record in aggregate_abi.calls
            if record.caller == fn
            for projection in record.projections
            for instruction in (projection.address, projection.load)
        }
        aggregate_whole_call_ids = {
            id(record.call)
            for record in aggregate_abi.calls
            if record.caller == fn
            and record.call.res is not None
            and any(
                is_storage_view(argument, record.call.res)
                for block in function.blocks.values()
                for instruction in block.instrs
                if id(instruction) not in aggregate_projection_instruction_ids
                for argument in instruction.args
            )
        }

        block_names = list(function.blocks)
        effect_guarded_blocks = set(
            function.metadata.get("pool_effect_guarded_blocks") or ()
        )
        for block_name in block_names:
            block = function.blocks[block_name]
            body.append(f"    {_c_label(block_name)}: (void)0;")
            block_is_guarded = block_name in effect_guarded_blocks
            if block_is_guarded:
                # Order-insensitive shared effect (deployment_outlining.py):
                # the block body runs under the pool's effect lock so
                # concurrent lanes serialize the append without an order.
                effect_guard_used[0] = True
                body.append("        turing_pool_effect_lock();")
            for position, instruction in enumerate(block.instrs):
                if block_is_guarded and position == len(block.instrs) - 1:
                    body.append("        turing_pool_effect_unlock();")
                # Repository tensor intrinsics deliberately retain the generic
                # Call handler while carrying their canonical operation name
                # in ``tensor_operation``.  Dispatch on that semantic name,
                # as the Fortran/LLVM backends do, so scalar metadata calls
                # such as extent are not mistaken for unknown callees or
                # allocated as dynamic tensor temporaries.
                op = str(
                    instruction.attributes.get("tensor_operation")
                    or instruction.op
                )
                if id(instruction) in aggregate_projection_instruction_ids:
                    # The native call binds these abstract tuple projections
                    # directly to caller-owned output storage.
                    continue
                if instruction.attributes.get("precision_section"):
                    precision_present = True
                if op in {"Phi", "phi"}:
                    # Defined by edge assignments; the declaration is
                    # hoisted below so every predecessor can reach it.
                    expressions[int(instruction.res.id)] = f"t{instruction.res.id}"
                    if is_integer(instruction.res):
                        integer_ids.add(int(instruction.res.id))
                    continue
                if op in {"Const", "const"}:
                    if "llvm_literal" in instruction.attributes:
                        from .ir_literals import decode_llvm_scalar_literal

                        try:
                            held = decode_llvm_scalar_literal(
                                instruction.attributes["llvm_literal"]
                            )
                        except ValueError as error:
                            shortfalls.append(CEmissionShortfall(
                                op, f"{error} in {fn}",
                            ))
                            continue
                    elif "values" in instruction.attributes:
                        held = instruction.attributes["values"]
                    else:
                        held = instruction.attributes.get(
                            "constant", instruction.attributes.get("value")
                        )
                    if held is None:
                        shortfalls.append(CEmissionShortfall(
                            op,
                            f"invalid repository SSA Const %t"
                            f"{instruction.res.id} carries None in {fn}; "
                            "None must use the explicit NoneValue operation",
                        ))
                        continue
                    if isinstance(held, (list, tuple)):
                        try:
                            flattened = _flatten_numeric_aggregate(held)
                        except TypeError as error:
                            shortfalls.append(CEmissionShortfall(
                                op, f"{error} in {fn}",
                            ))
                            continue
                        result_id = int(instruction.res.id)
                        if (
                            len(flattened) == 1
                            and _declared_span_rank(instruction.res) == 0
                            and not tuple(instruction.res.shape or ())
                        ):
                            # Frontend index construction often preserves a
                            # one-limb tuple payload after shape specialization.
                            # Its USE is scalar (arithmetic into a GEP offset),
                            # so retain the value and discard only the obsolete
                            # aggregate wrapper.
                            held = flattened[0]
                        else:
                            element_type = buffer_type(instruction.res)
                            # ISO C has no zero-length arrays.  An empty authored
                            # aggregate has no legal element load, so one inert
                            # backing limb preserves every observable operation.
                            literals = tuple(
                                _c_numeric_literal(value, element_type)
                                for value in flattened
                            ) or (_c_numeric_literal(0, element_type),)
                            body.append(
                                f"        {element_type} t{result_id}[] = {{"
                                + ", ".join(literals) + "};"
                            )
                            expressions[result_id] = f"t{result_id}"
                            addresses[result_id] = f"t{result_id}"
                            continue
                    if is_integer(instruction.res) or isinstance(held, int):
                        expressions[int(instruction.res.id)] = str(int(held))
                        integer_ids.add(int(instruction.res.id))
                    else:
                        numeric = float(held)
                        expressions[int(instruction.res.id)] = (
                            "NAN" if math.isnan(numeric)
                            else "INFINITY" if numeric > 0 and math.isinf(numeric)
                            else "(-INFINITY)" if math.isinf(numeric)
                            else numeric.hex()
                        )
                    result_id = int(instruction.res.id)
                    if not tuple(instruction.res.shape or ()):
                        # Pointer propagation means a downstream native helper
                        # consumes this scalar by address.  A tensor-table
                        # scratch declaration for the same semantic id is not
                        # its value; passing that zero-initialized scratch loses
                        # the authored constant.  Materialize the scalar itself
                        # and make its address authoritative.
                        literal = expressions[result_id]
                        body.append(
                            f"        {buffer_type(instruction.res)} "
                            f"t{result_id} = {literal};"
                        )
                        expressions[result_id] = f"t{result_id}"
                        addresses[result_id] = f"&t{result_id}"
                    continue
                if op in {"Br", "br"}:
                    target = str(instruction.attributes.get("target"))
                    body.extend(phi_edge_assignments(block_name, target))
                    body.append(f"        goto {_c_label(target)};")
                    continue
                if op in {"CondBr", "condbr"}:
                    condition = scalar_operand(instruction.args[0])
                    on_true = str(instruction.attributes.get("true_target"))
                    on_false = str(instruction.attributes.get("false_target"))
                    if condition is None:
                        continue
                    body.append(f"        if ({condition}) {{")
                    body.extend(
                        "    " + line
                        for line in phi_edge_assignments(block_name, on_true)
                    )
                    body.append(f"            goto {_c_label(on_true)};")
                    body.append("        } else {")
                    body.extend(
                        "    " + line
                        for line in phi_edge_assignments(block_name, on_false)
                    )
                    body.append(f"            goto {_c_label(on_false)};")
                    body.append("        }")
                    continue
                if op in {"Ret", "ret", "Return", "return"}:
                    body.extend(output_publications(tuple(instruction.args)))
                    # Outputs live in caller-visible buffers already. Route
                    # every exit through the activation-storage cleanup.
                    body.append(f"        goto cleanup_{_c_symbol(fn)};")
                    continue
                if (
                    op in {"max", "min", "all", "any"}
                    and instruction.res is not None
                    and len(instruction.args) == 1
                    and not tuple(instruction.res.shape or ())
                ):
                    # Flat tensor reduction to a scalar (the dt controller's
                    # max-wave-speed / all-finite metrics). The SEMANTIC
                    # element count comes from the tensor descriptor, never
                    # the storage arena bound -- scanning the arena would
                    # read unrelated aliased slots.
                    source = instruction.args[0]
                    descriptor = (
                        tensor_table.by_id(int(source.id))
                        if tensor_table is not None else None
                    )
                    count = None
                    if (
                        descriptor is not None
                        and descriptor.metadata_state == "static"
                        and descriptor.static_element_count
                    ):
                        count = int(descriptor.static_element_count)
                    else:
                        source_shape = tuple(source.shape or ())
                        if source_shape and all(
                            isinstance(item, int) for item in source_shape
                        ):
                            count = math.prod(source_shape)
                        elif not source_shape:
                            count = 1
                    if count is None:
                        shortfalls.append(CEmissionShortfall(
                            op,
                            f"flat {op} reduction over %t{source.id} has no "
                            f"static semantic element count in {fn}",
                        ))
                        continue
                    result_id = int(instruction.res.id)
                    result_type = buffer_type(instruction.res)
                    if count == 1:
                        scalar = scalar_operand(source)
                        if scalar is None:
                            continue
                        rendered = (
                            scalar if op in {"max", "min"}
                            else f"((({scalar}) != 0.0) ? 1.0 : 0.0)"
                        )
                        body.append(
                            f"        {result_type} t{result_id} = "
                            f"{rendered};"
                        )
                    else:
                        source_address = address_operand(source)
                        if source_address is None:
                            continue
                        element_type = buffer_type(source)
                        if op in {"max", "min"}:
                            comparison = ">" if op == "max" else "<"
                            # NaN propagates, matching the numpy reference:
                            # once the accumulator is NaN no comparison can
                            # displace it, and a NaN element always enters.
                            step = (
                                f"if (turing_red_{result_id}_v {comparison} "
                                f"turing_red_{result_id} || "
                                f"turing_red_{result_id}_v != "
                                f"turing_red_{result_id}_v) "
                                f"turing_red_{result_id} = "
                                f"turing_red_{result_id}_v;"
                            )
                            initial = f"turing_red_{result_id}_src[0]"
                            start = 1
                        elif op == "all":
                            # NaN is truthy (NaN != 0), matching Python.
                            step = (
                                f"if (turing_red_{result_id}_v == 0.0) "
                                f"turing_red_{result_id} = 0.0;"
                            )
                            initial = "1.0"
                            start = 0
                        else:  # any
                            step = (
                                f"if (turing_red_{result_id}_v != 0.0) "
                                f"turing_red_{result_id} = 1.0;"
                            )
                            initial = "0.0"
                            start = 0
                        body.extend((
                            f"        {result_type} t{result_id};",
                            "        {",
                            f"        const {element_type} *turing_red_"
                            f"{result_id}_src = (const {element_type} *)"
                            f"({source_address});",
                            f"        {result_type} turing_red_{result_id} "
                            f"= {initial};",
                            f"        for (ptrdiff_t turing_red_{result_id}"
                            f"_i = {start}; turing_red_{result_id}_i < "
                            f"{count}; ++turing_red_{result_id}_i) {{",
                            f"            const {element_type} turing_red_"
                            f"{result_id}_v = turing_red_{result_id}_src["
                            f"turing_red_{result_id}_i];",
                            f"            {step}",
                            "        }",
                            f"        t{result_id} = turing_red_{result_id};",
                            "        }",
                        ))
                    expressions[result_id] = f"t{result_id}"
                    addresses[result_id] = f"&t{result_id}"
                    if _is_integer_dtype(instruction.res.dtype):
                        integer_ids.add(result_id)
                    continue
                if op == "extent" and instruction.res is not None and instruction.args:
                    source = instruction.args[0]
                    descriptor = (
                        tensor_table.by_id(int(source.id))
                        if tensor_table is not None else None
                    )
                    dynamic = bool(
                        descriptor is not None
                        and descriptor.metadata_state != "static"
                    ) or bool(
                        storage_requirements_by_function.get(fn, {}).get(
                            int(source.id)
                        )
                        and storage_requirements_by_function[fn][
                            int(source.id)
                        ].dynamic
                    ) or any(
                        not isinstance(item, int)
                        for item in tuple(source.shape or ())
                    )
                    resolved_static_shape = None
                    if dynamic:
                        # The origin walk can prove a runtime shape is in
                        # fact static (an identity-ledger chain terminating
                        # at a static descriptor). That downgrade is exact:
                        # the specialized program cannot present any other
                        # shape at this site.
                        resolved_origin = resolve_span_origin(
                            fn, int(source.id)
                        )
                        if (
                            resolved_origin is not None
                            and resolved_origin[0] == "static"
                        ):
                            resolved_static_shape = tuple(resolved_origin[1])
                            dynamic = False
                    if resolved_static_shape is not None:
                        shape = resolved_static_shape
                    elif descriptor is not None and not dynamic:
                        shape = tuple(map(int, descriptor.shape))
                    elif not dynamic:
                        try:
                            shape = tuple(map(int, source.shape or ()))
                        except (TypeError, ValueError):
                            shortfalls.append(CEmissionShortfall(
                                op, f"unresolved extent for %t{source.id} in {fn}",
                            ))
                            continue
                    kind = str(instruction.attributes.get("extent_kind") or "")
                    result_id = int(instruction.res.id)
                    if dynamic:
                        rank = (
                            len(tuple(source.shape or ()))
                            or int((source.accounting or {}).get(
                                "program_abi_rank", 0
                            ))
                            or (
                                int(instruction.res.shape[0])
                                if kind == "shape"
                                and tuple(instruction.res.shape or ())
                                and isinstance(instruction.res.shape[0], int)
                                else 0
                            )
                        )
                        if kind == "rank" and rank:
                            expressions[result_id] = str(rank)
                            integer_ids.add(result_id)
                            continue
                        if kind == "shape" and rank:
                            slots = tuple(
                                extent_slot(fn, int(source.id), "shape", axis)
                                for axis in range(rank)
                            )
                            if any(slot is None for slot in slots):
                                shortfalls.append(CEmissionShortfall(
                                    op, f"dynamic shape for %t{source.id} has no "
                                    f"public storage origin in {fn}",
                                ))
                                continue
                            values = ", ".join(
                                f"(int32_t)extents[{slot}]" for slot in slots
                            )
                            body.append(
                                f"        int32_t t{result_id}[] = "
                                f"{{{values}}};"
                            )
                            expressions[result_id] = f"t{result_id}"
                            addresses[result_id] = f"t{result_id}"
                            continue
                        if kind == "shape":
                            # A whole-shape request with unknown rank has no
                            # per-axis slots to register; a ('shape', None)
                            # row could never be measured by the wrapper.
                            shortfalls.append(CEmissionShortfall(
                                op,
                                f"dynamic shape for %t{source.id} has "
                                f"unknown rank in {fn}",
                            ))
                            continue
                        axis = (
                            int(instruction.attributes.get("axis", 0))
                            if kind == "dim" else None
                        )
                        slot = extent_slot(
                            fn, int(source.id),
                            "dim" if kind == "dim" else kind,
                            axis,
                        )
                        if slot is None:
                            shortfalls.append(CEmissionShortfall(
                                op, f"dynamic extent for %t{source.id} has no "
                                f"public storage origin in {fn}",
                            ))
                            continue
                        expressions[result_id] = f"extents[{slot}]"
                        integer_ids.add(result_id)
                        continue
                    if kind == "shape":
                        values = ", ".join(map(str, shape)) or "0"
                        body.append(
                            f"        int32_t t{result_id}[] = {{{values}}};"
                        )
                        expressions[result_id] = f"t{result_id}"
                        addresses[result_id] = f"t{result_id}"
                    else:
                        if kind in {"element_count", "numel"}:
                            value = 1
                            for extent in shape:
                                value *= extent
                        elif kind == "rank":
                            value = len(shape)
                        elif kind == "dim":
                            axis = int(instruction.attributes.get("axis", 0))
                            if not shape or not (-len(shape) <= axis < len(shape)):
                                shortfalls.append(CEmissionShortfall(
                                    op, f"axis {axis} is outside shape {shape!r} in {fn}",
                                ))
                                continue
                            value = shape[axis % len(shape)]
                        else:
                            shortfalls.append(CEmissionShortfall(
                                op, f"unknown extent kind {kind!r} in {fn}",
                            ))
                            continue
                        expressions[result_id] = str(int(value))
                        integer_ids.add(result_id)
                    continue
                if (
                    op in {"clone", "copy", "detach", "contiguous"}
                    and instruction.res is not None
                    and len(instruction.args) == 1
                ):
                    source_value = instruction.args[0]
                    result_id = int(instruction.res.id)
                    source = address_operand(source_value)
                    destination = addresses.get(
                        result_id, expressions.get(result_id)
                    )
                    result_shape = tuple(instruction.res.shape or ())
                    source_shape = tuple(source_value.shape or ())
                    count = None
                    if result_shape and all(
                        isinstance(item, int) for item in result_shape
                    ):
                        count = math.prod(result_shape)
                    elif source_shape and all(
                        isinstance(item, int) for item in source_shape
                    ):
                        count = math.prod(source_shape)
                    else:
                        requirement = storage_requirements_by_function.get(
                            fn, {}
                        ).get(result_id)
                        if (
                            requirement is not None
                            and not requirement.dynamic
                            and requirement.element_count is not None
                        ):
                            count = int(requirement.element_count)
                    if source is None:
                        continue
                    if count is None and not result_shape and not source_shape:
                        value = scalar_operand(source_value)
                        if value is None:
                            continue
                        result_type = _scalar_c_type(instruction.res.dtype)
                        body.append(
                            f"        {result_type} t{result_id} = "
                            f"({result_type})({value});"
                        )
                        expressions[result_id] = f"t{result_id}"
                        addresses[result_id] = f"&t{result_id}"
                        if _is_integer_dtype(instruction.res.dtype):
                            integer_ids.add(result_id)
                        continue
                    if destination is None or count is None:
                        shortfalls.append(CEmissionShortfall(
                            op,
                            f"native tensor copy %t{result_id} has no bounded "
                            f"destination in {fn}",
                        ))
                        continue
                    body.append(
                        f"        (void)memcpy({destination}, {source}, "
                        f"(size_t)({int(count)}) * "
                        f"sizeof({buffer_type(instruction.res)}));"
                    )
                    continue
                if op == "PointerArray" and instruction.res is not None:
                    args = [address_operand(value) for value in instruction.args]
                    if any(value is None for value in args):
                        continue
                    if any(
                        buffer_type(value) != "double"
                        for value in instruction.args
                    ):
                        shortfalls.append(CEmissionShortfall(
                            op,
                            "pointer arrays currently require float64 tensor "
                            f"buffers in {fn}",
                        ))
                        continue
                    result_id = int(instruction.res.id)
                    body.append(
                        f"        double *t{result_id}[] = {{"
                        + ", ".join(args) + "};"
                    )
                    expressions[result_id] = f"t{result_id}"
                    addresses[result_id] = f"t{result_id}"
                    continue
                if op in ("Call", "call"):
                    callee = str(instruction.attributes.get("callee") or "")
                    target = module.functions.get(callee)
                    rendered_intrinsic = _C_MATH_INTRINSICS.get(callee)
                    if rendered_intrinsic is not None:
                        args = [scalar_operand(value) for value in instruction.args]
                        if any(value is None for value in args):
                            continue
                        if instruction.res is None:
                            shortfalls.append(CEmissionShortfall(
                                op, f"math intrinsic {callee!r} has no result in {fn}",
                            ))
                            continue
                        result_id = int(instruction.res.id)
                        result_type = _scalar_c_type(instruction.res.dtype)
                        if _is_integer_dtype(instruction.res.dtype):
                            integer_ids.add(result_id)
                        body.append(
                            f"        "
                            f"{result_type} t{result_id} = "
                            f"{rendered_intrinsic}(" + ", ".join(args) + ");"
                        )
                        expressions[result_id] = f"t{result_id}"
                        addresses[result_id] = f"&t{result_id}"
                        continue
                    if callee in {"llvm.fcmp.ord", "llvm.fcmp.uno"}:
                        args = [scalar_operand(value) for value in instruction.args]
                        if any(value is None for value in args) or len(args) != 2:
                            continue
                        if instruction.res is None:
                            shortfalls.append(CEmissionShortfall(
                                op, f"floating comparison {callee!r} has no result in {fn}",
                            ))
                            continue
                        result_id = int(instruction.res.id)
                        relation = "&&" if callee.endswith("ord") else "||"
                        negate = "!" if callee.endswith("ord") else ""
                        body.append(
                            f"        "
                            f"long long t{result_id} = "
                            f"({negate}isnan({args[0]}) {relation} "
                            f"{negate}isnan({args[1]}));"
                        )
                        expressions[result_id] = f"t{result_id}"
                        addresses[result_id] = f"&t{result_id}"
                        integer_ids.add(result_id)
                        continue
                    if callee.startswith("llvm.memcpy") and len(
                        instruction.args
                    ) >= 3:
                        destination = address_operand(instruction.args[0])
                        source = address_operand(instruction.args[1])
                        count = scalar_operand(instruction.args[2])
                        if None not in {destination, source, count}:
                            body.append(
                                f"        (void)memcpy({destination}, {source}, "
                                f"(size_t)({count}));"
                            )
                        continue
                    if callee.startswith("llvm.memset") and len(
                        instruction.args
                    ) >= 3:
                        destination = address_operand(instruction.args[0])
                        value = scalar_operand(instruction.args[1])
                        count = scalar_operand(instruction.args[2])
                        if None not in {destination, value, count}:
                            body.append(
                                f"        (void)memset({destination}, "
                                f"(int)({value}), (size_t)({count}));"
                            )
                        continue
                    if target is None or callee not in reachable:
                        shortfalls.append(CEmissionShortfall(
                            op, f"call to unknown function {callee!r}",
                        ))
                        continue
                    if len(instruction.args) != len(target.args):
                        shortfalls.append(CEmissionShortfall(
                            op,
                            f"call to {callee!r} has {len(instruction.args)} "
                            f"SSA arguments but its contract declares "
                            f"{len(target.args)} in {fn}",
                        ))
                        continue
                    rendered_args = []
                    call_token = instruction.attributes.get(
                        "callsite_id",
                        instruction.res.id
                        if instruction.res is not None else id(instruction),
                    )
                    for argument_index, (actual, formal) in enumerate(zip(
                        instruction.args, target.args
                    )):
                        actual_has_array_contract = bool(
                            tuple(actual.shape or ())
                            or int((actual.accounting or {}).get(
                                "program_abi_rank", 0
                            ) or 0) > 0
                            or (actual.accounting or {}).get(
                                "program_abi_storage"
                            ) == "span"
                        )
                        formal_has_array_contract = bool(
                            tuple(formal.shape or ())
                            or int((formal.accounting or {}).get(
                                "program_abi_rank", 0
                            ) or 0) > 0
                            or (formal.accounting or {}).get(
                                "program_abi_storage"
                            ) == "span"
                        )
                        formal_is_callee_output = int(formal.id) in {
                            int(output.id)
                            for output in native_outputs.get(callee, ())
                        }
                        # The in-place convention: an actual that IS the
                        # call's own result (or one of this function's own
                        # output ports) is the destination the callee
                        # stores through -- a conversion temporary there is
                        # a too-small wrongly-typed cell and the callee's
                        # store shreds the stack.
                        actual_is_destination = (
                            (
                                instruction.res is not None
                                and int(actual.id) == int(instruction.res.id)
                            )
                            or int(actual.id) in {
                                int(output.id)
                                for output in native_outputs.get(fn, ())
                            }
                        )
                        if (
                            # A rankless scalar is sometimes indexed by a
                            # generated one-element cast helper.  GEP use makes
                            # call-array analysis conservatively mark that
                            # pointer as an array, but it does not create an
                            # authored span contract.  Keep scalar conversion
                            # at such call edges; forwarding the one-byte bool
                            # cell to a helper that loads a double is invalid.
                            #
                            # NEVER convert an operand bound to a callee
                            # OUTPUT slot: the callee stores its result dtype
                            # through that pointer, so a converted one-cell
                            # temporary (e.g. a uint8_t minted for a bool
                            # view) hands the callee a too-small, wrongly
                            # typed cell and the store shreds the stack.
                            not formal_is_callee_output
                            and not actual_is_destination
                            and not actual_has_array_contract
                            and not formal_has_array_contract
                            and _pointer_value_depth(actual) == 0
                            and _pointer_value_depth(formal) == 0
                        ):
                            # No storage is shared at a scalar call boundary:
                            # the callee receives a fresh one-cell temporary.
                            # Use each value's own authored storage dtype here,
                            # not the array-alias union solver.  A repository
                            # helper formal may also serve shaped callers, and
                            # that legitimate array component must not make a
                            # bool scalar caller masquerade as double.
                            actual_type = address_buffer_types.get(
                                int(actual.id), _value_buffer_c_type(actual)
                            )
                            formal_type = _value_buffer_c_type(formal)
                            if actual_type != formal_type:
                                scalar = scalar_operand(actual)
                                if scalar is None:
                                    rendered_args = None
                                    break
                                local_name = (
                                    f"callarg{call_token}_{argument_index}"
                                )
                                converted = (
                                    f"(({scalar}) != 0)"
                                    if formal_type == "uint8_t"
                                    else f"({formal_type})({scalar})"
                                )
                                body.append(
                                    f"        {formal_type} {local_name} = "
                                    f"{converted};"
                                )
                                rendered_args.append(f"&{local_name}")
                                continue
                        value = addresses.get(int(actual.id))
                        if value is None:
                            scalar = operand(actual)
                            if scalar is None:
                                rendered_args = None
                                break
                            local_name = f"callarg{call_token}_{argument_index}"
                            body.append(
                                f"        {buffer_type(formal)} "
                                f"{local_name} = {scalar};"
                            )
                            value = f"&{local_name}"
                        if value is None:
                            rendered_args = None
                            break
                        rendered_args.append(value)
                    if rendered_args is None:
                        continue
                    aggregate_record = aggregate_calls.get(id(instruction))
                    if aggregate_record is not None:
                        projected_addresses: dict[int, str] = {}
                        callee_formal_positions = {
                            int(formal.id): position
                            for position, formal in enumerate(target.args)
                        }
                        for slot, output in enumerate(native_outputs[callee]):
                            output_id = int(output.id)
                            projections = tuple(
                                projection
                                for projection in aggregate_record.projections
                                if projection.output_id == output_id
                            )
                            formal_position = callee_formal_positions.get(
                                output_id
                            )
                            prebound_storage = aggregate_projection_storage.get(
                                (id(instruction), output_id)
                            )
                            if formal_position is not None:
                                actual = instruction.args[formal_position]
                                projected_address = address_operand(actual)
                                if projected_address is None:
                                    rendered_args = None
                                    break
                                projected_expression = (
                                    projected_address
                                    if tuple(output.shape or ())
                                    else storage_expression(
                                        output, projected_address
                                    )
                                )
                                output_argument = None
                            else:
                                destination = output_destinations.get(output_id)
                            if formal_position is None and destination is not None:
                                output_argument = destination
                                projected_expression = (
                                    destination
                                    if tuple(output.shape or ())
                                    else storage_expression(output, destination)
                                )
                                projected_address = destination
                            elif formal_position is None and prebound_storage is not None:
                                projected_expression, projected_address = (
                                    prebound_storage
                                )
                                output_argument = projected_address
                            elif formal_position is None and projections and all(
                                int(projection.value.id) in addresses
                                for projection in projections
                            ):
                                first_result_id = int(projections[0].value.id)
                                output_argument = addresses[first_result_id]
                                projected_expression = expressions[first_result_id]
                                projected_address = addresses[first_result_id]
                            elif formal_position is None:
                                first_result_id = (
                                    int(projections[0].value.id)
                                    if projections else int(instruction.res.id)
                                )
                                existing_address = addresses.get(
                                    first_result_id
                                )
                                if existing_address is not None:
                                    # Same value id already has storage
                                    # from a producer on another control
                                    # path -- write the SAME local (see the
                                    # plain-call arm's rationale).
                                    projected_address = existing_address
                                    projected_expression = expressions.get(
                                        first_result_id, existing_address
                                    )
                                    output_argument = existing_address
                                else:
                                    local_name = (
                                        f"callout{first_result_id}_{slot}"
                                    )
                                    element_type = buffer_type(output)
                                    shape = tuple(output.shape or ())
                                    if shape:
                                        count = 1
                                        for extent in shape:
                                            count *= int(extent)
                                        storage = activation_array(
                                            element_type, count
                                        )
                                        output_argument = storage
                                        projected_expression = storage
                                        projected_address = storage
                                    else:
                                        local_tensor_declarations.append(
                                            f"    {element_type} "
                                            f"{local_name};"
                                        )
                                        output_argument = f"&{local_name}"
                                        projected_expression = local_name
                                        projected_address = f"&{local_name}"
                            if output_argument is not None:
                                rendered_args.append(output_argument)
                            for projection in projections:
                                expressions[int(projection.value.id)] = (
                                    projected_expression
                                )
                                addresses[int(projection.value.id)] = (
                                    projected_address
                                )
                                projected_addresses[
                                    int(projection.output_id)
                                ] = projected_address
                        if rendered_args is None:
                            continue
                        if id(instruction) in aggregate_whole_call_ids:
                            table_addresses = tuple(
                                projected_addresses.get(int(output_id))
                                for output_id in aggregate_record.output_ids
                            )
                            if any(
                                address is None for address in table_addresses
                            ):
                                shortfalls.append(CEmissionShortfall(
                                    op,
                                    f"aggregate result %t{instruction.res.id} "
                                    "escapes whole but its selected output "
                                    f"record is incomplete in {fn}",
                                ))
                                continue
                            result_id = int(instruction.res.id)
                            table_name = f"aggregate{result_id}"
                            body.append(
                                f"        double *{table_name}[] = {{"
                                + ", ".join(table_addresses) + "};"
                            )
                            expressions[result_id] = table_name
                            addresses[result_id] = table_name
                    elif (
                        native_outputs[callee]
                        and instruction.res is not None
                        and instruction.attributes.get("result_aliases_frame")
                    ):
                        # An identity/coercion call may return storage that is
                        # already carried by its input frame.  Once record
                        # materialization expands that return into several
                        # physical fields, every output must still be an exact
                        # callee formal; those formals are already present in
                        # rendered_args and no second caller result record or
                        # output argument exists.
                        formal_positions = {
                            int(formal.id): position
                            for position, formal in enumerate(target.args)
                        }
                        output_positions = tuple(
                            formal_positions.get(int(output.id))
                            for output in native_outputs[callee]
                        )
                        if any(
                            position is None
                            or position >= len(instruction.args)
                            or address_operand(
                                instruction.args[int(position)]
                            ) is None
                            for position in output_positions
                        ):
                            shortfalls.append(CEmissionShortfall(
                                op,
                                f"call to {callee!r} claims a frame-aliased "
                                "record result but one or more native outputs "
                                "are not exact addressable callee formals",
                            ))
                            continue
                        alias_position = int(
                            instruction.attributes.get("ssa_output_argument", -1)
                        )
                        if not 0 <= alias_position < len(instruction.args):
                            shortfalls.append(CEmissionShortfall(
                                op,
                                f"call to {callee!r} has an invalid aliased "
                                "result argument",
                            ))
                            continue
                        alias_actual = instruction.args[alias_position]
                        alias_address = address_operand(alias_actual)
                        result_id = int(instruction.res.id)
                        addresses[result_id] = alias_address
                        expressions[result_id] = (
                            alias_address
                            if tuple(instruction.res.shape or ())
                            else storage_expression(
                                instruction.res, alias_address
                            )
                        )
                    elif native_outputs[callee] and instruction.res is not None:
                        direct_output_ids = tuple(map(
                            int,
                            (instruction.res.accounting or {}).get(
                                "ssa_aggregate_outputs", (),
                            ),
                        ))
                        if not direct_output_ids and len(native_outputs[callee]) == 1:
                            direct_output_ids = (int(instruction.res.id),)
                        if len(direct_output_ids) != len(native_outputs[callee]):
                            shortfalls.append(CEmissionShortfall(
                                op,
                                f"call to {callee!r} returns "
                                f"{len(native_outputs[callee])} native outputs "
                                "without a matching caller result record",
                            ))
                            continue
                        for slot, (output, result_id) in enumerate(zip(
                            native_outputs[callee], direct_output_ids,
                        )):
                            formal_position = next((
                                position
                                for position, formal in enumerate(target.args)
                                if int(formal.id) == int(output.id)
                            ), None)
                            if formal_position is not None:
                                actual = instruction.args[formal_position]
                                result_address = address_operand(actual)
                                if result_address is None:
                                    rendered_args = None
                                    break
                                result_expression = (
                                    result_address
                                    if tuple(output.shape or ())
                                    else storage_expression(
                                        output, result_address
                                    )
                                )
                                output_argument = None
                            else:
                                destination = output_destinations.get(result_id)
                            if formal_position is None and destination is not None:
                                output_argument = destination
                                result_expression = (
                                    destination
                                    if tuple(output.shape or ())
                                    else storage_expression(output, destination)
                                )
                                result_address = destination
                            elif (
                                formal_position is None
                                and addresses.get(result_id) is not None
                            ):
                                # A previous call site on another control
                                # path already owns storage for this value
                                # id. Both producers MUST write the same
                                # local: minting a second one binds later
                                # textual readers to whichever producer was
                                # emitted last, and a path through the
                                # other producer then reads uninitialized
                                # memory.
                                result_address = addresses[result_id]
                                result_expression = expressions.get(
                                    result_id, result_address
                                )
                                output_argument = result_address
                            elif formal_position is None:
                                local_name = f"callout{result_id}_{slot}"
                                element_type = buffer_type(output)
                                shape = tuple(output.shape or ())
                                if shape:
                                    count = 1
                                    for extent in shape:
                                        count *= int(extent)
                                    storage = activation_array(
                                        element_type, count
                                    )
                                    output_argument = storage
                                    result_expression = storage
                                    result_address = storage
                                else:
                                    # Function-scope declaration: a branch-
                                    # local declaration cannot serve a
                                    # reader on the other branch.
                                    local_tensor_declarations.append(
                                        f"    {element_type} {local_name};"
                                    )
                                    output_argument = f"&{local_name}"
                                    result_expression = local_name
                                    result_address = f"&{local_name}"
                            if output_argument is not None:
                                rendered_args.append(output_argument)
                            expressions[result_id] = result_expression
                            addresses[result_id] = result_address
                        if rendered_args is None:
                            continue
                    if callee in extent_users:
                        rendered_args.append("extents")
                    invocation = (
                        f"{_c_symbol(callee)}("
                        + ", ".join(rendered_args) + ")"
                    )
                    callee_return_type = function_return_types[callee]
                    if callee_return_type != "void" and instruction.res is not None:
                        result_id = int(instruction.res.id)
                        body.append(
                            f"        const {callee_return_type} t{result_id} = "
                            f"{invocation};"
                        )
                        expressions[result_id] = f"t{result_id}"
                        if _is_integer_dtype(instruction.res.dtype):
                            integer_ids.add(result_id)
                    else:
                        body.append(f"        {invocation};")
                    continue
                if op in {"Deploy", "Join"} and instruction.res is None:
                    # The sequential C lane has already serialized the
                    # deployment frame. A Join is always a receipt; a Deploy
                    # whose lane was outlined becomes a native pool span
                    # deploy with the serial loop kept as the fallback.
                    if op != "Deploy":
                        continue
                    outline_record = deployment_outlines.get((
                        str(fn),
                        int(instruction.attributes.get("region_id", -1)),
                    ))
                    if outline_record is None:
                        continue
                    deploy_callee = module.functions.get(
                        outline_record.outline_name
                    )
                    bound_values = tuple(
                        values_by_id.get(value_id)
                        for value_id in (
                            outline_record.start_id,
                            outline_record.stop_id,
                            outline_record.step_id,
                        )
                    )
                    induction_value = values_by_id.get(
                        outline_record.induction_id
                    )
                    argument_addresses = {
                        value_id: addresses.get(value_id)
                        for value_id in outline_record.argument_ids
                        if value_id != outline_record.induction_id
                    }
                    if (
                        deploy_callee is None
                        or induction_value is None
                        or not _is_integer_dtype(induction_value.dtype)
                        or any(value is None for value in bound_values)
                        or any(
                            expressions.get(int(value.id)) is None
                            for value in bound_values
                        )
                        or any(
                            held is None
                            for held in argument_addresses.values()
                        )
                        or outline_record.exit_block not in function.blocks
                    ):
                        body.append(
                            "        /* deployment region "
                            f"{outline_record.region_id} stays serial: an "
                            "operand is unavailable at the deploy site */"
                        )
                        continue
                    start_value, stop_value, step_value = bound_values
                    uid = f"{_c_symbol(fn)}_r{outline_record.region_id}"
                    slot_count = max(1, len(argument_addresses))
                    deployment_support.extend((
                        f"typedef struct {{ long long start; long long step; "
                        f"void *a[{slot_count}]; long long *extents; }} "
                        f"turing_deploy_ctx_{uid};",
                        f"static void turing_deploy_span_{uid}"
                        "(void *raw, long start, long stop);",
                    ))
                    trampoline_args = []
                    slot = 0
                    for value_id in outline_record.argument_ids:
                        if value_id == outline_record.induction_id:
                            trampoline_args.append(
                                "(void *)&turing_induction"
                            )
                        else:
                            trampoline_args.append(f"ctx->a[{slot}]")
                            slot += 1
                    if outline_record.outline_name in extent_users:
                        trampoline_args.append("ctx->extents")
                    deployment_trampolines.append("\n".join((
                        f"static void turing_deploy_span_{uid}"
                        "(void *raw, long start, long stop) {",
                        f"    turing_deploy_ctx_{uid} *ctx = "
                        f"(turing_deploy_ctx_{uid} *)raw;",
                        "    for (long position = start; position < stop; "
                        "++position) {",
                        "        long long turing_induction = ctx->start "
                        "+ ctx->step * (long long)position;",
                        f"        {_c_symbol(outline_record.outline_name)}("
                        + ", ".join(trampoline_args) + ");",
                        "    }",
                        "}",
                    )))
                    slot_initializers = ", ".join(
                        argument_addresses[value_id]
                        for value_id in outline_record.argument_ids
                        if value_id != outline_record.induction_id
                    ) or "0"
                    extents_initializer = (
                        "extents"
                        if outline_record.outline_name in extent_users
                        else "0"
                    )
                    start_scalar = scalar_operand(start_value)
                    stop_scalar = scalar_operand(stop_value)
                    step_scalar = scalar_operand(step_value)
                    if None in (start_scalar, stop_scalar, step_scalar):
                        continue
                    body.extend((
                        "        {",
                        "        static int turing_pool_started = 0;",
                        "        if (!turing_pool_started) { "
                        f"turing_pool_start({_DEFAULT_POOL_WORKERS}); "
                        "turing_pool_started = 1; }",
                        f"        turing_deploy_ctx_{uid} turing_ctx = {{ "
                        f"(long long)({start_scalar}), "
                        f"(long long)({step_scalar}), "
                        f"{{ {slot_initializers} }}, "
                        f"{extents_initializer} }};",
                        "        long long turing_total = 0;",
                        "        if (turing_ctx.step > 0) turing_total = "
                        f"((long long)({stop_scalar}) - turing_ctx.start "
                        "+ turing_ctx.step - 1) / turing_ctx.step;",
                        "        if (turing_total > 1 && "
                        f"turing_pool_deploy_span(turing_deploy_span_{uid}, "
                        "&turing_ctx, (long)turing_total, 1) == 0) "
                        f"goto {_c_label(outline_record.exit_block)};",
                        "        }",
                    ))
                    pooled_regions.append(
                        (str(fn), int(outline_record.region_id))
                    )
                    continue
                if op == "string_token" and instruction.res is not None:
                    result_id = int(instruction.res.id)
                    expressions[result_id] = str(int(
                        instruction.attributes.get("token", 0)
                    ))
                    integer_ids.add(result_id)
                    continue
                if op == "NoneValue" and instruction.res is not None:
                    # Structural None is the native record/token zero sentinel,
                    # never a retained Python object.
                    result_id = int(instruction.res.id)
                    expressions[result_id] = "0"
                    integer_ids.add(result_id)
                    continue
                if instruction.res is None:
                    if op in {"Store", "store"} and len(instruction.args) >= 2:
                        stored_value = instruction.args[0]
                        value = scalar_operand(stored_value)
                        addressed_value = instruction.args[1]
                        where = address_operand(addressed_value)
                        if value is not None and where is not None:
                            stored_type = address_element_types.get(
                                int(addressed_value.id),
                                carried_type(stored_value),
                            )
                            body.append(
                                f"        *(({stored_type} *)({where})) = {value};"
                            )
                        continue
                    shortfalls.append(CEmissionShortfall(
                        op, "instruction has no result",
                    ))
                    continue
                if op in {"Tuple", "tuple"}:
                    # An aggregate constructor has no physical value: its
                    # projections are aliased to member values and a starred
                    # call argument is expanded into members.  A surviving
                    # constructor is dead; emit nothing for it.  Any real use
                    # of it faults loudly as an unavailable operand.
                    continue
                args = [scalar_operand(value) for value in instruction.args]
                if any(value is None for value in args):
                    continue
                result_id = int(instruction.res.id)
                declared = None
                if op in {"GetElementPtr", "getelementptr"} and len(args) >= 2:
                    base = instruction.args[0]
                    pointer_depth = _pointer_value_depth(base)
                    if pointer_depth >= 2:
                        # ptrptr_float64 indexes a table whose elements are
                        # double pointers. The resulting address is therefore
                        # double ** and Load removes exactly one level.
                        element = "double *"
                    elif pointer_depth == 1:
                        # A ptr value indexes its double payload.
                        element = "double"
                    else:
                        element = buffer_type(base)
                    base_expression = address_operand(base)
                    if base_expression is None:
                        continue
                    indices = args[1:]
                    offset = indices[0]
                    if len(indices) > 1:
                        requirement = storage_requirements_by_function.get(
                            fn, {}
                        ).get(int(base.id))
                        static_shape = tuple(
                            requirement.shape if requirement is not None else ()
                        )
                        if static_shape and len(static_shape) >= len(indices):
                            strides = [
                                math.prod(static_shape[axis + 1:])
                                for axis in range(len(indices))
                            ]
                            offset = " + ".join(
                                f"(({index}) * {int(stride)})"
                                for index, stride in zip(indices, strides)
                            )
                        else:
                            terms = []
                            unresolved = False
                            for axis, index in enumerate(indices):
                                trailing = []
                                for trailing_axis in range(
                                    axis + 1, len(indices)
                                ):
                                    slot = extent_slot(
                                        fn, int(base.id), "dim", trailing_axis
                                    )
                                    if slot is None:
                                        unresolved = True
                                        break
                                    trailing.append(f"extents[{slot}]")
                                if unresolved:
                                    break
                                stride = " * ".join(trailing) or "1"
                                terms.append(f"(({index}) * ({stride}))")
                            if unresolved:
                                shortfalls.append(CEmissionShortfall(
                                    op,
                                    f"multi-axis address of %t{base.id} has no "
                                    f"public extent origin in {fn}",
                                ))
                                continue
                            offset = " + ".join(terms)
                    declared = (
                        f"{element} *t{result_id} = "
                        f"(({element} *)({base_expression})) "
                        f"+ (ptrdiff_t)({offset});"
                    )
                    address_element_types[result_id] = element
                elif op in {"Load", "load"} and len(args) == 1:
                    addressed_value = instruction.args[0]
                    source_address = address_operand(addressed_value)
                    if source_address is None:
                        continue
                    loaded_type = carried_type(instruction.res)
                    storage_type = address_element_types.get(
                        int(addressed_value.id), loaded_type,
                    )
                    qualifier = ""
                    if (
                        storage_type.rstrip().endswith("*")
                        and not tuple(instruction.res.shape or ())
                        and _declared_span_rank(instruction.res) == 0
                        and _pointer_value_depth(instruction.res) == 0
                    ):
                        # A GEP into a pointer table addresses one pointer
                        # slot. Loading a scalar projection therefore has two
                        # distinct dereferences: table slot -> scalar storage,
                        # then scalar storage -> value. Casting the first
                        # pointer directly to double is invalid C and, where a
                        # compiler accepts an extension, reads pointer bits as
                        # the validator metric.
                        declared = (
                            f"{qualifier}{loaded_type} t{result_id} = "
                            f"*(({loaded_type} *)(*(({storage_type} *)"
                            f"({source_address}))));"
                        )
                    else:
                        declared = (
                            f"{qualifier}{loaded_type} "
                            f"t{result_id} = ({loaded_type})"
                            f"(*(({storage_type} *)({source_address})));"
                        )
                    if loaded_type in {"int32_t", "int64_t"}:
                        integer_ids.add(result_id)
                elif op in {"Cast", "CastLike"} and len(args) >= 1:
                    result_type = _scalar_c_type(instruction.res.dtype)
                    if _is_integer_dtype(instruction.res.dtype):
                        integer_ids.add(result_id)
                    declared = (
                        f"const {result_type} t{result_id} = "
                        f"({result_type})({args[0]});"
                    )
                elif op == "SExt" and len(args) == 1:
                    result_type = _scalar_c_type(instruction.res.dtype)
                    source_type = _scalar_c_type(instruction.args[0].dtype)
                    integer_ids.add(result_id)
                    declared = (
                        f"const {result_type} t{result_id} = "
                        f"({result_type})(({source_type})({args[0]}));"
                    )
                elif op == "ZExt" and len(args) == 1:
                    result_type = _scalar_c_type(instruction.res.dtype)
                    source_type = _unsigned_c_type(instruction.args[0].dtype)
                    integer_ids.add(result_id)
                    declared = (
                        f"const {result_type} t{result_id} = "
                        f"({result_type})(({source_type})({args[0]}));"
                    )
                elif op == "Trunc" and len(args) == 1 and _is_integer_dtype(
                    instruction.res.dtype
                ):
                    result_type = _scalar_c_type(instruction.res.dtype)
                    unsigned_type = _unsigned_c_type(instruction.res.dtype)
                    integer_ids.add(result_id)
                    declared = (
                        f"const {result_type} t{result_id} = "
                        f"({result_type})(({unsigned_type})({args[0]}));"
                    )
                elif op in {
                    "UiToFp", "SiToFp", "SIToFP", "FpToSi", "FPToSI",
                    "FpToUi",
                } and len(args) == 1:
                    result_type = _scalar_c_type(instruction.res.dtype)
                    if _is_integer_dtype(instruction.res.dtype):
                        integer_ids.add(result_id)
                    source = args[0]
                    if op == "UiToFp":
                        source = (
                            f"({_unsigned_c_type(instruction.args[0].dtype)})"
                            f"({source})"
                        )
                    if op == "FpToUi":
                        result_type = _unsigned_c_type(instruction.res.dtype)
                    declared = (
                        f"const {result_type} t{result_id} = "
                        f"({result_type})({source});"
                    )
                elif op == "Select" and len(args) == 3:
                    result_type = _scalar_c_type(instruction.res.dtype)
                    if _is_integer_dtype(instruction.res.dtype):
                        integer_ids.add(result_id)
                    declared = (
                        f"const {result_type} t{result_id} = "
                        f"({args[0]} ? {args[1]} : {args[2]});"
                    )
                elif (
                    op.casefold() == "indexed"
                    and len(args) == 1
                    and not tuple(instruction.args[0].shape or ())
                    and not tuple(instruction.res.shape or ())
                ):
                    # Aggregate ABI lowering can select a tuple/record leaf
                    # before a planned numerical region is materialized.  In
                    # that form the former projection has one scalar operand:
                    # there is no remaining array base or index to address.
                    # Preserve the selected value with the result's declared
                    # type instead of reporting a tensor-gather shortfall.
                    result_type = _scalar_c_type(instruction.res.dtype)
                    if _is_integer_dtype(instruction.res.dtype):
                        integer_ids.add(result_id)
                    declared = (
                        f"const {result_type} t{result_id} = "
                        f"({result_type})({args[0]});"
                    )
                elif op == "cast_like" and len(args) == 2:
                    result_type = _scalar_c_type(instruction.res.dtype)
                    if _is_integer_dtype(instruction.res.dtype):
                        integer_ids.add(result_id)
                    declared = (
                        f"const {result_type} t{result_id} = "
                        f"({result_type})({args[0]});"
                    )
                elif (
                    op.casefold() in {"max", "min"}
                    and len(args) == 1
                    and tuple(instruction.args[0].shape or ())
                    and not tuple(instruction.res.shape or ())
                ):
                    element_count = math.prod(instruction.args[0].shape)
                    result_type = _scalar_c_type(instruction.res.dtype)
                    source = address_operand(instruction.args[0])
                    if source is None:
                        continue
                    initial = (
                        "LLONG_MIN"
                        if _is_integer_dtype(instruction.res.dtype) and op.casefold() == "max"
                        else "LLONG_MAX"
                        if _is_integer_dtype(instruction.res.dtype)
                        else "(-INFINITY)"
                        if op.casefold() == "max"
                        else "INFINITY"
                    )
                    comparison = ">" if op.casefold() == "max" else "<"
                    body.append(
                        f"        {result_type} t{result_id} = {initial};"
                    )
                    body.append(
                        f"        for (ptrdiff_t r{result_id} = 0; "
                        f"r{result_id} < {int(element_count)}; ++r{result_id}) "
                        f"if ((({buffer_type(instruction.args[0])} *)"
                        f"({source}))[r{result_id}] {comparison} t{result_id}) "
                        f"t{result_id} = (({buffer_type(instruction.args[0])} *)"
                        f"({source}))[r{result_id}];"
                    )
                    expressions[result_id] = f"t{result_id}"
                    addresses[result_id] = f"&t{result_id}"
                    if _is_integer_dtype(instruction.res.dtype):
                        integer_ids.add(result_id)
                    continue
                elif op.casefold() in {"max", "min"} and len(args) == 2:
                    integral = all(map(is_integer, instruction.args))
                    result_type = _scalar_c_type(instruction.res.dtype)
                    comparison = ">" if op.casefold() == "max" else "<"
                    if integral:
                        integer_ids.add(result_id)
                    declared = (
                        f"const {result_type} t{result_id} = "
                        f"(({args[0]}) {comparison} ({args[1]}) "
                        f"? ({args[0]}) : ({args[1]}));"
                    )
                elif op == "Pow" and len(args) == 2:
                    # Constant exponents approved by the active work contract
                    # have already been rewritten by
                    # reduce_constant_exponent_pow. A Pow that survives that
                    # shared pass is the generic operation; libm pow is C's
                    # faithful spelling, not a backend-private identity.
                    declared = (
                        f"const double t{result_id} = "
                        f"pow({args[0]}, {args[1]});"
                    )
                elif op.casefold() in _TERNARY and len(args) == 3:
                    declared = (
                        f"const double t{result_id} = "
                        f"fma({args[0]}, {args[1]}, {args[2]});"
                    )
                elif op in _BINARY and len(args) == 2:
                    integral = all(map(is_integer, instruction.args))
                    kind = _scalar_c_type(instruction.res.dtype)
                    if integral:
                        integer_ids.add(result_id)
                    expression = f"({args[0]} {_BINARY[op]} {args[1]})"
                    if integral and op in {"Add", "Sub", "Mul"}:
                        unsigned = _unsigned_c_type(instruction.res.dtype)
                        expression = (
                            f"({kind})(({unsigned})({args[0]}) "
                            f"{_BINARY[op]} ({unsigned})({args[1]}))"
                        )
                    declared = (
                        f"const {kind} t{result_id} = "
                        f"{expression};"
                    )
                elif op in _INTEGER_BINARY and len(args) == 2:
                    integer_ids.add(result_id)
                    kind = _scalar_c_type(instruction.res.dtype)
                    left, right = args
                    if op in {"LShr", "Shr"}:
                        left = f"({_unsigned_c_type(instruction.args[0].dtype)})({left})"
                    declared = (
                        f"const {kind} t{result_id} = "
                        f"({kind})({left} {_INTEGER_BINARY[op]} {right});"
                    )
                elif op in _LOGICAL_BINARY and len(args) == 2:
                    integer_ids.add(result_id)
                    declared = (
                        f"const {_scalar_c_type(instruction.res.dtype)} t{result_id} = "
                        f"(({args[0]}) {_LOGICAL_BINARY[op]} ({args[1]}));"
                    )
                elif op in {"LNot", "Not"} and len(args) == 1:
                    integer_ids.add(result_id)
                    declared = (
                        f"const {_scalar_c_type(instruction.res.dtype)} t{result_id} = (!({args[0]}));"
                    )
                elif op == "Invert" and len(args) == 1:
                    integer_ids.add(result_id)
                    declared = (
                        f"const {_scalar_c_type(instruction.res.dtype)} "
                        f"t{result_id} = (~({args[0]}));"
                    )
                elif op in _FLOORED and len(args) == 2:
                    integral = all(map(is_integer, instruction.args))
                    helper = _FLOORED[op][0 if integral else 1]
                    kind = _scalar_c_type(instruction.res.dtype)
                    if integral:
                        integer_ids.add(result_id)
                    declared = (
                        f"const {kind} t{result_id} = "
                        f"{helper}({args[0]}, {args[1]});"
                    )
                elif op in _C_UNSIGNED_COMPARISONS and len(args) == 2:
                    integer_ids.add(result_id)
                    left_type = _unsigned_c_type(instruction.args[0].dtype)
                    right_type = _unsigned_c_type(instruction.args[1].dtype)
                    declared = (
                        f"const {_scalar_c_type(instruction.res.dtype)} t{result_id} = "
                        f"(({left_type})({args[0]}) "
                        f"{_C_UNSIGNED_COMPARISONS[op]} "
                        f"({right_type})({args[1]}));"
                    )
                elif op in _C_COMPARISONS and len(args) == 2:
                    integer_ids.add(result_id)
                    declared = (
                        f"const {_scalar_c_type(instruction.res.dtype)} t{result_id} = "
                        f"({args[0]} {_C_COMPARISONS[op]} {args[1]});"
                    )
                elif op == "Neg" and len(args) == 1:
                    kind = (
                        _scalar_c_type(instruction.res.dtype)
                    )
                    declared = f"const {kind} t{result_id} = (-{args[0]});"
                elif op.casefold() in _UNARY_FOLDED and len(args) == 1:
                    declared = (
                        f"const double t{result_id} = "
                        f"{_UNARY_FOLDED[op.casefold()]}({args[0]});"
                    )
                if declared is None:
                    shortfalls.append(CEmissionShortfall(
                        op, f"no module-lane C spelling in {fn}",
                    ))
                    continue
                expressions[result_id] = f"t{result_id}"
                if op in {"GetElementPtr", "getelementptr"}:
                    addresses[result_id] = f"t{result_id}"
                elif _pointer_value_depth(instruction.res) > 0:
                    # This produced value is itself an address. GEP and call
                    # consumers need that address, not the address of the
                    # local pointer variable which happens to hold it.
                    addresses[result_id] = f"t{result_id}"
                elif not tuple(instruction.res.shape or ()):
                    # A scalar producer forwarded to a pointer formal must use
                    # the produced scalar's address, never a tensor-table tmp
                    # that merely shares its semantic id.
                    if declared.startswith("const "):
                        declared = declared[len("const "):]
                    addresses[result_id] = f"&t{result_id}"
                body.append("        " + declared)

        # Planner regions commonly have no Ret instruction: their declared
        # output record is the terminator contract.  Publish scalar results
        # at the lexical end as well; shaped results were written directly
        # into their caller-owned buffers throughout the body.
        if not any(
            instruction.op in {"Ret", "ret", "Return", "return"}
            for block in function.blocks.values()
            for instruction in block.instrs
        ):
            body.extend(output_publications())

        declarations = [
            f"    {kind} t{phi_id};"
            for phi_id, kind in sorted(phi_declarations.items())
        ]
        declarations.extend(local_tensor_declarations)
        declarations.extend(
            f"    {element_type} *{storage_name} = NULL;"
            for element_type, storage_name, _count in frame_allocations
        )
        allocation_setup = [
            line
            for element_type, storage_name, count in frame_allocations
            for line in (
                f"    {storage_name} = calloc({count}, "
                f"sizeof(*{storage_name}));",
                f"    if ({storage_name} == NULL) "
                f"goto cleanup_{_c_symbol(fn)};",
            )
        ]
        cleanup = [
            f"cleanup_{_c_symbol(fn)}:",
            *(
                f"    free({storage_name});"
                for _element_type, storage_name, _count
                in reversed(frame_allocations)
            ),
            "    return;",
        ]
        prototypes.append(
            f"static {function_return_type} {_c_symbol(fn)}("
            + ", ".join(parameters) + ");"
        )
        definitions.append("\n".join((
            f"static {function_return_type} {_c_symbol(fn)}("
            + ", ".join(parameters) + ") {",
            *declarations,
            *allocation_setup,
            *body,
            *cleanup,
            "}",
        )))

    # -- the public wrapper: same buffer ABI as the LLVM lane ---------------
    root = module.functions[function_name]
    from .ssa_storage_requirements import (
        function_storage_requirements,
        is_compiler_owned_storage,
    )

    storage_requirements = function_storage_requirements(module, function_name)
    buffer_order: list[int] = []
    buffer_shapes: list[tuple[int, ...]] = []
    buffer_dtypes: list[str] = []
    entry_lines: list[str] = []
    root_allocations: list[tuple[str, str, int]] = []

    def activation_array(element_type: str, count: int) -> str:
        """Allocate wrapper-owned storage with LLVM alloca lifetime."""

        name = f"root_storage_{len(root_allocations)}"
        root_allocations.append((element_type, name, max(1, int(count))))
        return name
    rendered_actuals: list[str] = []
    owned_storage_names: dict[int, str] = {}
    record_parameter_ids = {
        int(value_id)
        for parameter_name, value_id in root.metadata.get(
            "parameter_names", ()
        )
        if str(parameter_name) in dict(
            root.metadata.get("parameter_record_abi") or {}
        )
    }

    def is_structural_abi_value(value) -> bool:
        """Whether *value* is a descriptor rather than physical storage."""

        accounting = dict(value.accounting or {})
        return (
            int(value.id) in record_parameter_ids
            or str(value.dtype or "").casefold() == "ssa.aggregate"
            or accounting.get("program_abi_storage") == "keyed"
        )

    def is_private_root_storage(value) -> bool:
        accounting = dict(value.accounting or {})
        return (
            is_compiler_owned_storage(value)
            or is_structural_abi_value(value)
            or accounting.get("split_from_unproven_alias") is not None
            or accounting.get("split_from_result_storage") is not None
        )

    def rendered_root_value(value_id: int) -> str | None:
        if value_id in owned_storage_names:
            return owned_storage_names[value_id]
        if value_id in buffer_order:
            return f"b{buffer_order.index(value_id)}"
        return None

    for slot, formal in enumerate(root.args):
        value_id = int(formal.id)
        requirement = storage_requirements.get(value_id)
        if is_private_root_storage(formal):
            if value_id not in owned_storage_names:
                owned = f"frame_{value_id}"
                owned_storage_names[value_id] = owned
                held = solved_buffer_type(function_name, formal)
                split_source = (formal.accounting or {}).get(
                    "split_from_unproven_alias"
                )
                authored_shape = tuple(formal.shape or ())
                element_count = (
                    math.prod(authored_shape)
                    if split_source is not None and authored_shape
                    else requirement.element_count
                    if requirement is not None
                    and requirement.element_count is not None
                    else 1
                )
                if value_id in record_parameter_ids:
                    # A record parameter's arena is its cell table: one
                    # 8-byte cell per declared field, initialized by the
                    # relocation prologue below. A storage requirement of a
                    # single handle cell would make those writes a heap
                    # overrun.
                    record_descriptor = (
                        getattr(module, "record_tables", {})
                        .get(function_name)
                    )
                    if record_descriptor is not None:
                        record_entry = record_descriptor.records.get(value_id)
                        if record_entry is not None:
                            element_count = max(
                                int(element_count),
                                len(record_entry.fields),
                            )
                owned = activation_array(held, element_count)
                owned_storage_names[value_id] = owned
                copied_from = (
                    rendered_root_value(int(split_source))
                    if split_source is not None else None
                )
                if copied_from is None:
                    allocation_count = next(
                        count for _kind, name, count in root_allocations
                        if name == owned
                    )
                    entry_lines.append(
                        f"    memset({owned}, 0, sizeof(*{owned}) "
                        f"* {allocation_count});"
                    )
                else:
                    allocation_count = next(
                        count for _kind, name, count in root_allocations
                        if name == owned
                    )
                    entry_lines.append(
                        f"    memcpy({owned}, {copied_from}, "
                        f"sizeof(*{owned}) * {allocation_count});"
                    )
            rendered_actuals.append(owned_storage_names[value_id])
            continue
        if value_id in buffer_order:
            rendered_actuals.append(f"b{buffer_order.index(value_id)}")
            continue
        buffer_order.append(value_id)
        buffer_shapes.append(tuple(
            requirement.shape if requirement is not None else formal.shape or ()
        ))
        held = solved_buffer_type(function_name, formal)
        buffer_dtypes.append(_dtype_for_c_storage(held))
        index = len(buffer_order) - 1
        entry_lines.append(
            f"    {held} *b{index} = "
            f"({held} *)buffers[{index}];"
        )
        rendered_actuals.append(f"b{index}")
    root_formal_ids = {int(formal.id) for formal in root.args}
    for output in native_outputs[function_name]:
        output_id = int(output.id)
        if is_structural_abi_value(output) and output_id not in set(map(int, watch)):
            owned = f"frame_output_{output_id}"
            element_type = solved_buffer_type(function_name, output)
            shape = tuple(output.shape or ())
            if shape:
                owned = activation_array(element_type, math.prod(shape))
                rendered_actuals.append(owned)
            else:
                entry_lines.append(f"    {element_type} {owned} = 0;")
                rendered_actuals.append(f"&{owned}")
            continue
        if output_id in root_formal_ids:
            continue
        if output_id in buffer_order:
            rendered_actuals.append(f"b{buffer_order.index(output_id)}")
            continue
        buffer_order.append(output_id)
        buffer_shapes.append(tuple(
            storage_requirements.get(output_id).shape
            if output_id in storage_requirements else output.shape or ()
        ))
        element_type = solved_buffer_type(function_name, output)
        buffer_dtypes.append(_dtype_for_c_storage(element_type))
        index = len(buffer_order) - 1
        entry_lines.append(
            f"    {element_type} *b{index} = "
            f"({element_type} *)buffers[{index}];"
        )
        rendered_actuals.append(f"b{index}")
    # -- record-parameter relocation prologue -------------------------------
    # Record parameters are runtime cell arenas (the program mutates scalar
    # cells in place), but their private storage starts calloc-zeroed, so
    # every record-navigated read returned 0.0 -- dead inputs, verified on
    # the vehicle program (hub 0 m vs 5 m: bit-identical physics). Before
    # the entry call, initialize each parameter record's cells from the
    # live bindings: SPAN/SEQUENCE cells receive the field storage POINTER,
    # SCALAR cells copy the bound value. Cells are 8-byte slots in field
    # declaration order, matching the emitted GEP offsets.
    record_table = getattr(module, "record_tables", {}).get(function_name)
    if record_table is not None:
        from ..transmogrifier.ssa import SSARecordFieldStorage

        for formal in root.args:
            record_id = int(formal.id)
            descriptor = record_table.records.get(record_id)
            if descriptor is None or record_id not in record_parameter_ids:
                continue
            arena = rendered_root_value(record_id)
            if arena is None:
                continue
            for slot_index, field in enumerate(descriptor.fields):
                value_ids = tuple(field.value_ids or ())
                if not value_ids:
                    continue
                bound = rendered_root_value(int(value_ids[0]))
                if bound is None:
                    continue
                if field.storage == SSARecordFieldStorage.SCALAR:
                    entry_lines.append(
                        f"    ((double *){arena})[{slot_index}] = "
                        f"((double *)({bound}))[0];"
                        f"  /* {field.name} */"
                    )
                else:
                    entry_lines.append(
                        f"    ((void **){arena})[{slot_index}] = "
                        f"(void *)({bound});"
                        f"  /* {field.name} */"
                    )
    entry_lines.append(
        f"    {_c_symbol(function_name)}("
        + ", ".join((
            *rendered_actuals,
            *(("extents",) if function_name in extent_users else ()),
        )) + ");"
    )
    entry_lines = [
        *(
            f"    {element_type} *{storage_name} = NULL;"
            for element_type, storage_name, _count in root_allocations
        ),
        *(
            line
            for _element_type, storage_name, count in root_allocations
            for line in (
                f"    {storage_name} = calloc({count}, "
                f"sizeof(*{storage_name}));",
                f"    if ({storage_name} == NULL) "
                f"goto cleanup_{_c_symbol(name)};",
            )
        ),
        *entry_lines,
        f"cleanup_{_c_symbol(name)}:",
        *(
            f"    free({storage_name});"
            for _element_type, storage_name, _count
            in reversed(root_allocations)
        ),
    ]

    source = "\n".join((
        "#include <math.h>",
        "#include <limits.h>",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        # Same rationale as the scalar lane: C grants no reassociation
        # licence, so withdrawing contraction is the whole of
        # SECTION_ISOLATION, and every fma() still present is deliberate.
        "#pragma STDC FP_CONTRACT OFF",
        "#if defined(_WIN32)",
        "#define TURING_EXPORT __declspec(dllexport)",
        "#else",
        "#define TURING_EXPORT __attribute__((visibility(\"default\")))",
        "#endif",
        "",
        *_FLOORED_HELPERS,
        "",
        *(
            _POOL_DECLARATIONS
            if pooled_regions or effect_guard_used[0] else ()
        ),
        *deployment_support,
        "",
        *prototypes,
        "",
        *definitions,
        "",
        *deployment_trampolines,
        "",
        f"TURING_EXPORT void {name}(void **buffers, long long *extents) {{",
        *entry_lines,
        "}",
        "",
    ))
    return CModuleArtifact(
        name=name,
        source=source,
        buffer_order=tuple(buffer_order),
        buffer_dtypes=tuple(buffer_dtypes),
        shortfalls=tuple(shortfalls),
        buffer_shapes=tuple(buffer_shapes),
        extent_order=tuple(extent_order),
        precision_sections=precision_present,
        pool_required=bool(pooled_regions) or effect_guard_used[0],
        pooled_regions=tuple(pooled_regions),
    )


_C_COMPARISONS = {
    "Lt": "<", "Le": "<=", "Gt": ">", "Ge": ">=", "Eq": "==", "Ne": "!=",
}

_C_UNSIGNED_COMPARISONS = {
    "ULt": "<", "ULe": "<=", "UGt": ">", "UGe": ">=",
}


def _c_symbol(name: str) -> str:
    # The "impl_" prefix keeps every internal static distinct from the
    # exported entry, whose spelling the harness owns.
    return "impl_" + "".join(
        ch if ch.isalnum() or ch == "_" else "_" for ch in str(name)
    )


def _c_label(name: str) -> str:
    return "L_" + _c_symbol(name)


def emit_ssa_to_c(
    module: IRModule,
    function_name: str,
    *,
    entry_name: str | None = None,
    watch: Sequence[int] = (),
) -> CModuleArtifact:
    """Canonical C backend entry: preserve the complete repository module.

    ``emit_ssa_function_to_c`` remains as the legacy scalar ABI adapter for
    existing callers that explicitly need ``double* in/out``. New compiler
    products must use this module-preserving route.
    """

    return emit_ssa_module_to_c(
        module, function_name, entry_name=entry_name, watch=watch,
    )


__all__ = [
    "CEmissionShortfall",
    "CFunctionArtifact",
    "CModuleArtifact",
    "CModuleExecution",
    "emit_ssa_to_c",
    "emit_ssa_function_to_c",
    "emit_ssa_module_to_c",
    "supported_scalar_operations",
    "supported_tensor_operations",
]
