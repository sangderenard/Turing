"""Run a lowered program as fully instrumented Python.

The materializer made lowered SSA *readable* -- this makes it *observable*.
``compile_python_shell`` takes an ``IRModule``, materializes every function
back to Python, and rewrites each body so that every SSA value assignment is
recorded as it happens. Running the shell yields the result AND the complete
value history: every ``tN`` in every function, in execution order, once per
loop iteration.

Why this is a different instrument from the round trip alone. Comparing
outputs says *whether* two lanes disagree; comparing value histories says
*where* -- the first SSA id whose value diverges names the instruction, which
is the routing question every hard defect here reduces to. The emitted local
names ARE the SSA value ids (``t7`` is value 7), so a shell trace is directly
addressable against the IR, against ``SSAReferenceEvaluator``, or against a
native artifact's public buffers, with no correlation table in between.

This is also the honest version of "add a print to see the value". The
decision tree forbids source-level probes because they shift value ids and
have twice produced wrong conclusions; the shell records AFTER lowering, so
the program being observed is bit-for-bit the program that was compiled, and
observation cannot perturb it.

Limits, so a clean trace is not over-read:

* Coverage is the materializer's: single-block functions and the five-block
  counted loop. Functions it cannot render are reported in ``skipped`` and are
  absent from the trace -- absence is not evidence.
* The trace records committed assignments. An expression that faults mid-
  statement leaves no record of its partial work.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable

from .ssa_python_materializer import materialize_ir_module, to_source


@dataclass(frozen=True)
class TraceEntry:
    """One committed SSA value assignment, in execution order."""

    function: str
    name: str
    value: Any

    @property
    def value_id(self) -> int | None:
        """The SSA value id, when the local is a materializer-minted ``tN``."""

        if self.name.startswith("t") and self.name[1:].isdigit():
            return int(self.name[1:])
        return None


@dataclass
class ShellRun:
    """The result of one instrumented execution."""

    result: Any
    trace: tuple[TraceEntry, ...]

    def values_for(self, function: str) -> tuple[TraceEntry, ...]:
        return tuple(entry for entry in self.trace if entry.function == function)

    def last_value(self, function: str, value_id: int) -> Any:
        """The final value a given SSA id held, or a refusal -- never a default."""

        for entry in reversed(self.trace):
            if entry.function == function and entry.value_id == value_id:
                return entry.value
        raise KeyError(
            f"value {value_id} of {function!r} was never assigned in this run; "
            "that is an observation you did not make, not a zero"
        )

    def first_divergence(self, other: "ShellRun") -> tuple[int, TraceEntry, TraceEntry] | None:
        """The first position where two traces disagree, or None.

        This is the routing instrument: run the same shell twice with inputs
        that SHOULD agree (or one shell against another lane's history) and
        the first divergent entry names the instruction where they part.
        """

        for index, (mine, theirs) in enumerate(zip(self.trace, other.trace)):
            if (mine.function, mine.name) != (theirs.function, theirs.name):
                return index, mine, theirs
            if mine.value != theirs.value:
                return index, mine, theirs
        if len(self.trace) != len(other.trace):
            index = min(len(self.trace), len(other.trace))
            longer = self.trace if len(self.trace) > index else other.trace
            return index, longer[index], longer[index]
        return None


class _Recorder(ast.NodeTransformer):
    """After every simple assignment, record the committed value.

    Only ``name = expression`` forms are touched -- that is the entire shape
    the materializer emits -- and the record call reads the name AFTER the
    assignment commits, so the recorded value is the one the program actually
    carries forward.
    """

    def __init__(self) -> None:
        self._function: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self._function.append(node.name)
        node.body = self._instrument(node.body)
        self.generic_visit(node)
        self._function.pop()
        return node

    def _instrument(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for statement in statements:
            if isinstance(statement, (ast.While, ast.If)):
                statement.body = self._instrument(statement.body)
                statement.orelse = self._instrument(statement.orelse)
            out.append(statement)
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                target = statement.targets[0].id
                out.append(ast.Expr(value=ast.Call(
                    func=ast.Name(id="__record__", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=self._function[-1]),
                        ast.Constant(value=target),
                        ast.Name(id=target, ctx=ast.Load()),
                    ],
                    keywords=[],
                )))
        return out


@dataclass
class PythonShellProgram:
    """A compiled, instrumented, repeatedly runnable Python program.

    ``bindings`` are Python callables hosted alongside the compiled functions.
    This is the shell's second role and the reason it exists beyond
    diagnostics: the compiler can RETAIN parts of a program as Python rather
    than lowering them, and a retained call site arrives in the SSA as a Call
    whose callee is not among the module's functions. The shell is where those
    two worlds execute together -- compiled functions and retained callables
    in one namespace, every compiled value still recorded.

    A binding may not shadow a compiled function: silently preferring one over
    the other is exactly the plausible-wrong-answer shape this tree keeps
    paying for, so the collision refuses at compile time.
    """

    source: str
    skipped: dict[str, str]
    bindings: dict[str, Callable[..., Any]]
    _code: Any
    _entry: str

    def run(self, **arguments: Any) -> ShellRun:
        trace: list[TraceEntry] = []

        def record(function: str, name: str, value: Any) -> None:
            trace.append(TraceEntry(function, name, value))

        namespace: dict[str, Any] = {"__record__": record, **self.bindings}
        exec(self._code, namespace)
        result = namespace[self._entry](**arguments)
        return ShellRun(result=result, trace=tuple(trace))

    def unresolved_callees(self) -> tuple[str, ...]:
        """Call targets that neither the module nor the bindings supply.

        Reported rather than discovered at runtime as a NameError deep in a
        trace: an unresolved callee is a hole in the program, and the shell
        should be able to say so before anything runs.
        """

        import re

        emitted = set(re.findall(r"(?m)^def (\w+)\(", self.source))
        called = set(re.findall(r"(\w+)\(", self.source))
        keywords = {"__record__", "range", "abs", "max", "min", "round",
                    "float", "int", "bool", "not"}
        return tuple(sorted(
            name for name in called
            if name not in emitted
            and name not in self.bindings
            and name not in keywords
            and not name.startswith("math")
        ))

    def write(self, path: Any) -> None:
        """Write the shell as a standalone executable Python file.

        The file is the instrumented source plus a ``__record__`` that
        appends to a module-level trace -- a real .py, runnable and
        importable, which is the fully realized form: the compiled program
        living AS Python. Bindings are declared as import-me placeholders at
        the top, because a callable cannot be serialized honestly.
        """

        from pathlib import Path

        header = ['"""Instrumented shell emitted by python_shell. Runnable as-is."""',
                  "TRACE = []",
                  "def __record__(function, name, value):",
                  "    TRACE.append((function, name, value))",
                  ""]
        for name in sorted(self.bindings):
            header.append(f"# binding required at import time: {name}")
        newline = chr(10)
        Path(path).write_text(
            newline.join(header) + newline * 2 + self.source + newline,
            encoding="utf-8",
        )


def compile_python_shell(
    module: Any,
    entry: str,
    *,
    bindings: dict[str, Callable[..., Any]] | None = None,
) -> PythonShellProgram:
    """Materialize ``module`` and instrument every value assignment.

    ``entry`` is the emitted function name to expose (the materializer keeps
    the IR's own names, e.g. ``lp__train``). Raises if the entry is among the
    functions that could not be materialized, because a shell whose entry is
    silently absent would fail later with a bare NameError.

    ``bindings`` supplies Python callables for retained call sites -- names
    the SSA calls but the module does not define. A binding that collides
    with a compiled function refuses here.
    """

    emitted, skipped = materialize_ir_module(module)
    if entry in skipped:
        raise ValueError(
            f"entry {entry!r} could not be materialized: {skipped[entry]}"
        )
    names = {node.name for node in emitted.body if isinstance(node, ast.FunctionDef)}
    if entry not in names:
        raise ValueError(
            f"entry {entry!r} is not among the emitted functions {sorted(names)}"
        )

    supplied = dict(bindings or {})
    collisions = sorted(names & set(supplied))
    if collisions:
        raise ValueError(
            f"bindings {collisions} collide with compiled functions; the "
            "shell will not silently prefer one over the other"
        )

    instrumented = _Recorder().visit(emitted)
    ast.fix_missing_locations(instrumented)
    return PythonShellProgram(
        source=to_source(instrumented),
        skipped=dict(skipped),
        bindings=supplied,
        _code=compile(instrumented, "<python-shell>", "exec"),
        _entry=entry,
    )


__all__ = [
    "PythonShellProgram",
    "ShellRun",
    "TraceEntry",
    "compile_python_shell",
]
