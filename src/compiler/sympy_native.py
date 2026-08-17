"""Take SymPy in, hand back a native callable.

One entry point for the whole route: authored mathematics -> repository
SSA -> LLVM -> a loaded DLL -> a Python callable that runs the native
code. The caller supplies equations (or a file, or an AST that realises
into equations) and receives something it can call, without touching a
lowering, an emitter or a linker.

Why this exists as its own thing
--------------------------------
Anything the tracer claims about a spectrum is only as good as the
arithmetic that produced it, and there are two ways to get that
arithmetic: write it twice -- once in SymPy for the proof and once in
float for the run -- or write it once and compile it. The first is how
two implementations drift while both look right. This is the second.

So a reference that must be exact -- colour matching functions, a
convergence bound, a solver step -- is authored once as SymPy, checked
symbolically, and then run natively as the same expression rather than a
hand transcription of it.

Every stage is composed from what already exists:
``compile_sympy_equations`` for the lowering, ``emit_ssa_function_to_llvm``
and ``compile_artifact`` for the backend, ``prepare_artifact_execution``
for the ABI. Nothing here re-implements any of them, so a fix in one is a
fix here.

    equations = [sympy.Eq(y, sympy.sin(x) ** 2 + sympy.cos(x) ** 2)]
    native = compile_native(equations, name="pythagorean")
    native(x=0.7)          # -> {"y": 1.0}
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import sympy


@dataclass(frozen=True, slots=True)
class NativeExpression:
    """A compiled SymPy program, callable with named inputs."""

    name: str
    equations: tuple[Any, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    directory: Path
    _compilation: Any
    _artifact: Any
    _native: Any

    def __call__(self, **values: float) -> dict[str, float]:
        """Run the native code. Inputs are bound BY NAME, never by position.

        Positional binding across a compiled boundary is how this tree has
        repeatedly paired the wrong value with the right-looking slot, and
        an equation's free symbols have no inherent order to rely on.
        """
        from .ssa_llvm_backend import prepare_artifact_execution
        import numpy as np

        missing = [name for name in self.inputs if name not in values]
        if missing:
            raise TypeError(
                f"{self.name}: no value supplied for {', '.join(missing)}"
            )
        unexpected = [name for name in values if name not in self.inputs]
        if unexpected:
            raise TypeError(
                f"{self.name}: not an input of this program: "
                f"{', '.join(unexpected)}"
            )
        feed = {
            int(self._compilation.input_ids[name]):
                np.asarray([float(values[name])], dtype=np.float64)
            for name in self.inputs
        }
        execution = prepare_artifact_execution(self._native, feed)
        execution.run()
        produced: dict[str, float] = {}
        for name in self.outputs:
            buffer = execution.buffers.get(
                int(self._compilation.output_ids[name])
            )
            if buffer is None:
                continue
            produced[name] = float(
                np.asarray(buffer, dtype=float).reshape(-1)[0]
            )
        return produced

    def check(self, tolerance: float = 1e-9, **values: float) -> dict:
        """Run natively and symbolically on the same inputs, and compare.

        The point of authoring once is that the two CAN be compared. This
        is the comparison, and it belongs next to the compiler rather than
        in whatever happens to call it, so that "the native version agrees
        with the mathematics" is a thing that gets asserted rather than
        assumed.
        """
        produced = self(**values)
        substitutions = {
            symbol: sympy.Float(values[str(symbol)])
            for equation in self.equations
            for symbol in equation.rhs.free_symbols
            if str(symbol) in values
        }
        report = {}
        for equation in self.equations:
            name = str(equation.lhs)
            if name not in produced:
                continue
            expected = float(equation.rhs.subs(substitutions).evalf())
            gap = abs(expected - produced[name])
            report[name] = {
                "symbolic": expected,
                "native": produced[name],
                "gap": gap,
                "agrees": gap <= tolerance,
            }
        return report


def realize(source: Any, *, symbol: str = "EQUATIONS") -> tuple[Any, ...]:
    """Turn a file, a source string, an AST or a sequence into equations.

    A module is executed and its ``EQUATIONS`` read, because equations are
    most naturally written as ordinary SymPy code rather than encoded in a
    data format that would then need its own parser and its own bugs.
    """
    if isinstance(source, (list, tuple)) and all(
        isinstance(item, sympy.Equality) for item in source
    ):
        return tuple(source)
    if isinstance(source, ast.AST):
        text = ast.unparse(source)
    elif isinstance(source, Path) or (
        isinstance(source, str) and source.endswith(".py")
        and Path(source).is_file()
    ):
        text = Path(source).read_text(encoding="utf-8")
    elif isinstance(source, str):
        text = source
    else:
        raise TypeError(f"cannot realize equations from {type(source).__name__}")

    namespace: dict[str, Any] = {"sympy": sympy}
    exec(compile(text, "<sympy-native>", "exec"), namespace)
    found = namespace.get(symbol)
    if found is None:
        raise ValueError(
            f"the source defines no {symbol!r}; it must name the equations"
        )
    return tuple(found)


def compile_native(
    source: Any,
    *,
    name: str = "sympy_native",
    directory: Path | str | None = None,
    publications: Sequence[Any] = (),
) -> NativeExpression:
    """Authored SymPy to a loaded native callable, in one call."""
    from .symbolic_equation_compiler import compile_sympy_equations
    from .ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

    equations = realize(source)
    compilation = compile_sympy_equations(
        equations, name=name, publications=tuple(publications),
    )
    artifact = emit_ssa_function_to_llvm(compilation.module, name)
    if artifact.shortfalls:
        raise RuntimeError(
            f"{name}: emission reported shortfalls rather than emitting "
            f"partially: {artifact.shortfalls[:4]}"
        )
    target = Path(directory) if directory is not None else (
        Path("build") / f"sympy_native_{name}"
    )
    target.mkdir(parents=True, exist_ok=True)
    native = compile_artifact(artifact, directory=target)
    return NativeExpression(
        name=name,
        equations=tuple(equations),
        inputs=tuple(compilation.input_ids),
        outputs=tuple(compilation.output_ids),
        directory=target,
        _compilation=compilation,
        _artifact=artifact,
        _native=native,
    )


__all__ = ["NativeExpression", "realize", "compile_native"]
