"""Direct scalar repository-SSA to C emission.

This lane consumes ``Function`` instructions directly.  It does not construct
or consult a FusedProgram; unsupported SSA operations remain explicit
shortfalls.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa import Function, IRModule
from .output_publication import (
    function_output_publications,
    publication_surface_plan,
)


@dataclass(frozen=True, slots=True)
class CEmissionShortfall:
    operation: str
    reason: str


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

    def compile(self, directory: str | Path) -> "CFunctionArtifact":
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
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        source_path = destination / f"{self.name}.c"
        library_path = destination / f"{self.name}.dll"
        source_path.write_text(self.source, encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
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


_BINARY = {
    "Add": "+", "Sub": "-", "Mul": "*", "Div": "/",
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
    # Range reduction is floor and nothing else, so a lane without it
    # cannot compile a transcendental outside its core's own interval.
    # C99 has all three in <math.h>.
    "Floor": "floor", "Ceil": "ceil", "Round": "nearbyint",
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
        elif op in {"Max", "Min"} and len(args) == 2:
            rendered = f"f{op.lower()}({args[0]}, {args[1]})"
        elif op == "Neg" and len(args) == 1:
            rendered = f"(-{args[0]})"
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
    if name in {"int", "int32", "i32", "int64", "i64", "long", "bool", "i1"}:
        # One integer width for every SSA integer: indices multiply against
        # limb strides, so the narrow declared width of a COUNT buffer must
        # not become the arithmetic width.
        return "long long"
    return "double"


def _buffer_c_type(dtype) -> str:
    """The type a buffer SLOT holds, as opposed to arithmetic width.

    Feeds supply count buffers as int32, so the pointer type must match the
    memory, and the value widens on load rather than on the way in.
    """

    name = str(dtype or "float64").casefold()
    if name in {"int", "int32", "i32", "bool", "i1"}:
        return "int32_t"
    if name in {"int64", "i64", "long"}:
        return "int64_t"
    return "double"


def _numpy_dtype(dtype) -> str:
    name = str(dtype or "float64").casefold()
    if name in {"int", "int32", "i32", "bool", "i1"}:
        return "int32"
    if name in {"int64", "i64", "long"}:
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


@dataclass(slots=True)
class CModuleArtifact:
    """A whole repository-call module emitted as one C translation unit."""

    name: str
    source: str
    buffer_order: tuple[int, ...]
    buffer_dtypes: tuple[str, ...]
    shortfalls: tuple[CEmissionShortfall, ...]
    #: True when any emitted instruction belonged to a precision section.
    #: Carried on the artifact because the refusal it gates happens at
    #: COMPILE time, where the instructions are no longer visible.
    precision_sections: bool = False
    library_path: Path | None = None
    _entry: Any = field(default=None, repr=False)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def compile(self, directory: str | Path) -> "CModuleArtifact":
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
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        source_path = destination / f"{self.name}.c"
        library_path = destination / f"{self.name}.dll"
        source_path.write_text(self.source, encoding="utf-8")
        command = [
            sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
            *flags, "-o", str(library_path), str(source_path),
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

    def prepare_execution(self, feeds: Mapping[int, Any]) -> CModuleExecution:
        """Allocate the public buffers from real feed values.

        Mirrors the LLVM lane's ``prepare_artifact_execution``: buffer sizes
        come from the feeds, not from declared shapes, because a region
        formal's declared shape is ``()`` whether it is a scalar or the base
        of a million-element array.
        """

        import numpy as np

        buffers: dict[int, Any] = {}
        for value_id, dtype in zip(self.buffer_order, self.buffer_dtypes):
            fed = feeds.get(int(value_id))
            wanted = _numpy_dtype(dtype)
            if fed is None:
                held = np.zeros(1, dtype=wanted)
            else:
                held = np.ascontiguousarray(
                    np.atleast_1d(np.asarray(fed)), dtype=wanted
                )
            buffers[int(value_id)] = held
        pointers = (ctypes.c_void_p * len(self.buffer_order))(*(
            ctypes.c_void_p(int(buffers[int(value_id)].ctypes.data))
            for value_id in self.buffer_order
        ))
        extents = (ctypes.c_longlong * 1)(0)
        return CModuleExecution(
            artifact=self,
            buffers=buffers,
            _pointers=pointers,
            _extents=extents,
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
    module: IRModule, function_name: str, *, entry_name: str | None = None,
) -> CModuleArtifact:
    """Emit ``function_name`` and its call closure as one C module.

    Control flow becomes labels and gotos -- C's goto is exactly the
    unstructured branch the SSA already speaks, so no loop reconstruction is
    needed or wanted. A Phi becomes one mutable local assigned on every
    predecessor edge before the jump, which is the textbook out-of-SSA
    translation and preserves evaluation order exactly.
    """

    from .ir_identities import precision_backend_shortfalls

    name = str(entry_name or function_name)
    reachable = _module_call_closure(module, function_name)
    shortfalls: list[CEmissionShortfall] = []
    shortfalls.extend(
        CEmissionShortfall(
            "precision_section",
            "backend cannot honour precision obligations "
            + repr(item["missing"]) + f" in {item['function']}",
        )
        for item in precision_backend_shortfalls(module, "c", reachable)
    )

    # -- classify every formal as pointer-like or scalar, to a fixed point --
    #
    # A formal is a POINTER when something addresses through it (it is a
    # GetElementPtr base or a Store destination) or when a call passes it
    # into a pointer-classified position; otherwise it is a SCALAR passed by
    # value. The two uses are contradictory in C, so a formal claimed both
    # ways is a shortfall rather than a guess.
    pointer_formals: dict[str, set[int]] = {fn: set() for fn in reachable}
    for fn in reachable:
        function = module.functions[fn]
        for block in function.blocks.values():
            for instruction in block.instrs:
                op = str(instruction.op)
                if op == "GetElementPtr" and instruction.args:
                    pointer_formals[fn].add(int(instruction.args[0].id))
                elif op == "Store" and len(instruction.args) >= 2:
                    pointer_formals[fn].add(int(instruction.args[1].id))
    changed = True
    while changed:
        changed = False
        for fn in reachable:
            function = module.functions[fn]
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in ("Call", "call"):
                        continue
                    callee = str(instruction.attributes.get("callee") or "")
                    target = module.functions.get(callee)
                    if target is None or callee not in pointer_formals:
                        continue
                    for actual, formal in zip(instruction.args, target.args):
                        if (
                            int(formal.id) in pointer_formals[callee]
                            and int(actual.id) not in pointer_formals[fn]
                        ):
                            pointer_formals[fn].add(int(actual.id))
                            changed = True

    precision_present = False
    prototypes: list[str] = []
    definitions: list[str] = []

    for fn in reachable:
        function = module.functions[fn]
        formal_ids = {int(value.id) for value in function.args}
        pointers = pointer_formals[fn] & formal_ids
        # Formal declarations: pointers keep their buffer element type,
        # scalars arrive already widened by the caller.
        parameters = []
        for index, formal in enumerate(function.args):
            if int(formal.id) in pointers:
                parameters.append(
                    f"{_buffer_c_type(formal.dtype)} *restrict v{formal.id}"
                )
            else:
                parameters.append(f"{_scalar_c_type(formal.dtype)} v{formal.id}")
        body: list[str] = []
        # value id -> C expression. Pointer formals are usable only as GEP
        # bases; putting them here anyway lets the base lookup stay uniform.
        expressions: dict[int, str] = {
            int(formal.id): f"v{formal.id}" for formal in function.args
        }
        integer_ids = {
            int(formal.id) for formal in function.args
            if _scalar_c_type(formal.dtype) == "long long"
            and int(formal.id) not in pointers
        }
        phi_declarations: dict[int, str] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) == "Phi" and instruction.res is not None:
                    phi_declarations[int(instruction.res.id)] = _scalar_c_type(
                        instruction.res.dtype
                    )

        def is_integer(value) -> bool:
            return (
                int(value.id) in integer_ids
                or _scalar_c_type(value.dtype) == "long long"
            )

        def operand(value) -> str | None:
            held = expressions.get(int(value.id))
            if held is None:
                shortfalls.append(CEmissionShortfall(
                    "operand", f"%t{value.id} is unavailable in {fn}",
                ))
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
                        value = operand(instruction.args[position])
                        if value is not None:
                            assignments.append(
                                f"        t{instruction.res.id} = {value};"
                            )
            return assignments

        block_names = list(function.blocks)
        for block_name in block_names:
            block = function.blocks[block_name]
            body.append(f"    {_c_label(block_name)}: (void)0;")
            for instruction in block.instrs:
                op = str(instruction.op)
                if instruction.attributes.get("precision_section"):
                    precision_present = True
                if op == "Phi":
                    # Defined by edge assignments; the declaration is
                    # hoisted below so every predecessor can reach it.
                    expressions[int(instruction.res.id)] = f"t{instruction.res.id}"
                    if is_integer(instruction.res):
                        integer_ids.add(int(instruction.res.id))
                    continue
                if op == "Const":
                    held = instruction.attributes.get(
                        "constant", instruction.attributes.get("value")
                    )
                    if is_integer(instruction.res) or isinstance(held, int):
                        expressions[int(instruction.res.id)] = str(int(held))
                        integer_ids.add(int(instruction.res.id))
                    else:
                        expressions[int(instruction.res.id)] = float(held).hex()
                    continue
                if op == "Br":
                    target = str(instruction.attributes.get("target"))
                    body.extend(phi_edge_assignments(block_name, target))
                    body.append(f"        goto {_c_label(target)};")
                    continue
                if op == "CondBr":
                    condition = operand(instruction.args[0])
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
                if op == "Ret":
                    # Outputs live in caller-visible buffers already; the
                    # returned ids exist for the harness, not the callee.
                    body.append("        return;")
                    continue
                if op in ("Call", "call"):
                    callee = str(instruction.attributes.get("callee") or "")
                    target = module.functions.get(callee)
                    if target is None or callee not in pointer_formals:
                        shortfalls.append(CEmissionShortfall(
                            op, f"call to unknown function {callee!r}",
                        ))
                        continue
                    rendered_args = []
                    for actual, formal in zip(instruction.args, target.args):
                        value = operand(actual)
                        if value is None:
                            rendered_args = None
                            break
                        if (
                            int(formal.id) in pointer_formals[callee]
                            and int(actual.id) not in pointer_formals[fn]
                            and int(actual.id) not in formal_ids
                        ):
                            shortfalls.append(CEmissionShortfall(
                                op,
                                f"computed value %t{actual.id} passed to "
                                f"pointer formal of {callee!r}",
                            ))
                            rendered_args = None
                            break
                        rendered_args.append(value)
                    if rendered_args is None:
                        continue
                    body.append(
                        f"        {_c_symbol(callee)}("
                        + ", ".join(rendered_args) + ");"
                    )
                    continue
                if instruction.res is None:
                    if op == "Store" and len(instruction.args) >= 2:
                        value = operand(instruction.args[0])
                        where = operand(instruction.args[1])
                        if value is not None and where is not None:
                            body.append(f"        *({where}) = {value};")
                        continue
                    shortfalls.append(CEmissionShortfall(
                        op, "instruction has no result",
                    ))
                    continue
                args = [operand(value) for value in instruction.args]
                if any(value is None for value in args):
                    continue
                result_id = int(instruction.res.id)
                declared = None
                if op == "GetElementPtr" and len(args) >= 2:
                    base = instruction.args[0]
                    element = _buffer_c_type(base.dtype)
                    declared = (
                        f"{element} *restrict t{result_id} = "
                        f"{args[0]} + (ptrdiff_t)({args[1]});"
                    )
                elif op == "Load" and len(args) == 1:
                    declared = (
                        f"const {_scalar_c_type(instruction.res.dtype)} "
                        f"t{result_id} = *({args[0]});"
                    )
                elif op.casefold() in _TERNARY and len(args) == 3:
                    declared = (
                        f"const double t{result_id} = "
                        f"fma({args[0]}, {args[1]}, {args[2]});"
                    )
                elif op in _BINARY and len(args) == 2:
                    integral = all(map(is_integer, instruction.args))
                    kind = "long long" if integral else "double"
                    if integral:
                        integer_ids.add(result_id)
                    declared = (
                        f"const {kind} t{result_id} = "
                        f"({args[0]} {_BINARY[op]} {args[1]});"
                    )
                elif op in _C_COMPARISONS and len(args) == 2:
                    integer_ids.add(result_id)
                    declared = (
                        f"const long long t{result_id} = "
                        f"({args[0]} {_C_COMPARISONS[op]} {args[1]});"
                    )
                elif op == "Neg" and len(args) == 1:
                    kind = (
                        "long long" if is_integer(instruction.args[0])
                        else "double"
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
                body.append("        " + declared)

        declarations = [
            f"    {kind} t{phi_id};"
            for phi_id, kind in sorted(phi_declarations.items())
        ]
        prototypes.append(
            f"static void {_c_symbol(fn)}(" + ", ".join(parameters) + ");"
        )
        definitions.append("\n".join((
            f"static void {_c_symbol(fn)}(" + ", ".join(parameters) + ") {",
            *declarations,
            *body,
            "}",
        )))

    # -- the public wrapper: same buffer ABI as the LLVM lane ---------------
    root = module.functions[function_name]
    root_pointers = pointer_formals[function_name]
    buffer_order: list[int] = []
    buffer_dtypes: list[str] = []
    entry_lines: list[str] = []
    rendered_actuals: list[str] = []
    for slot, formal in enumerate(root.args):
        value_id = int(formal.id)
        if value_id in buffer_order:
            rendered_actuals.append(f"b{buffer_order.index(value_id)}")
            continue
        buffer_order.append(value_id)
        buffer_dtypes.append(str(formal.dtype or "float64"))
        held = _buffer_c_type(formal.dtype)
        index = len(buffer_order) - 1
        if value_id in root_pointers:
            entry_lines.append(
                f"    {held} *restrict b{index} = "
                f"({held} *)buffers[{index}];"
            )
            rendered_actuals.append(f"b{index}")
        else:
            entry_lines.append(
                f"    const {_scalar_c_type(formal.dtype)} b{index} = "
                f"({_scalar_c_type(formal.dtype)})*({held} *)buffers[{index}];"
            )
            rendered_actuals.append(f"b{index}")
    entry_lines.append(
        f"    {_c_symbol(function_name)}(" + ", ".join(rendered_actuals) + ");"
    )

    source = "\n".join((
        "#include <math.h>",
        "#include <stddef.h>",
        "#include <stdint.h>",
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
        *prototypes,
        "",
        *definitions,
        "",
        f"TURING_EXPORT void {name}(void **buffers, long long *extents) {{",
        "    (void)extents;",
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
        precision_sections=precision_present,
    )


_C_COMPARISONS = {
    "Lt": "<", "Le": "<=", "Gt": ">", "Ge": ">=", "Eq": "==", "Ne": "!=",
}


def _c_symbol(name: str) -> str:
    # The "impl_" prefix keeps every internal static distinct from the
    # exported entry, whose spelling the harness owns.
    return "impl_" + "".join(
        ch if ch.isalnum() or ch == "_" else "_" for ch in str(name)
    )


def _c_label(name: str) -> str:
    return "L_" + _c_symbol(name)


__all__ = [
    "CEmissionShortfall",
    "CFunctionArtifact",
    "CModuleArtifact",
    "CModuleExecution",
    "emit_ssa_function_to_c",
    "emit_ssa_module_to_c",
]
