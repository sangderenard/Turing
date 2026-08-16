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
    library_path: Path | None = None
    _entry: Any = field(default=None, repr=False)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def compile(self, directory: str | Path) -> "CFunctionArtifact":
        if not self.complete:
            raise ValueError("C artifact has emission shortfalls")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        source_path = destination / f"{self.name}.c"
        library_path = destination / f"{self.name}.dll"
        source_path.write_text(self.source, encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
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
_UNARY = {"Abs": "fabs", "Sqrt": "sqrt", "Neg": None}
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
    function: Function = module.functions[function_name]
    name = str(entry_name or function_name)
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
    shortfalls: list[CEmissionShortfall] = []
    emitted_tables: dict[str, str] = {}

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
            rendered = {
                2.0: f"({args[0]} * {args[0]})",
                -1.0: f"(1.0 / {args[0]})",
                -2.0: f"(1.0 / ({args[0]} * {args[0]}))",
                0.5: f"sqrt({args[0]})",
            }.get(exponent, f"pow({args[0]}, {args[1]})")
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
    )


__all__ = ["CEmissionShortfall", "CFunctionArtifact", "emit_ssa_function_to_c"]
