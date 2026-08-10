"""Bootstrap the AOT compiler from Python straight up to the dual IR, and run
that dual IR in the Fortran shell.

This is the start-to-finish, backend-neutral route -- NOT the JIT/tape path and
NOT a whole-program bake. A Python program (given by file path *or* by its
source contents) is:

1. resolved -- the driver checks whether the argument is a file it can open or
   the program text itself, and reads it accordingly;
2. compiled with the whole-program **no-bake** route
   (``compile_ast_aot(precompile_only=True, bake_mode="whole_program")``), which
   keeps every mutable parameter symbolic and stops at the highest checkpoint
   that carries **no backend-specific reduction** -- the dual IR
   (``FusedProgram`` + ``ControlProgram`` + ``map_ir``), arranged as a
   :class:`DualIRShell`;
3. released -- that dual-IR shell is written to the shell archive as the
   deliverable checkpoint (``save_shell``), so the neutral program exists as an
   artifact independent of any backend;
4. run in the Fortran shell -- the *same* compilation is handed to
   ``compile_ast_fortran_c_shell`` (which lowers the already-produced dual IR to
   SSA and then Fortran via ``ssa_fortran_backend``, builds the C shell, and
   returns an executable). The Fortran shell is the runner of the neutral
   checkpoint; it is not a second compile and not a JIT.

The entrypoint is either named explicitly or auto-detected (a lone top-level
function, or one named ``main``). Its parameters stay parameters: they are
declared ``mutable_parameters`` and left unfed, so the capture produces a
program rather than a baked instance.
"""
from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class ResolvedProgram:
    """A Python program ready to compile: its source, chosen entrypoint, and
    where the source came from (a path, or inline contents)."""

    source: str
    entrypoint: str
    origin: str  # "file:<path>" or "contents"
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class BootstrapResult:
    """Everything the bootstrap produced: the neutral dual-IR checkpoint (and
    the file it was released to) plus, when run, the Fortran-shell executable."""

    resolved: ResolvedProgram
    compilation: Any            # AOTCompilation
    shell: Any                  # DualIRShell -- the highest neutral checkpoint
    checkpoint_path: Path       # where the dual IR was released
    executable: Any | None      # FortranCShellExecutable, when run=True
    run_result: Any | None      # subprocess.CompletedProcess, when executed


# --- input resolution ---------------------------------------------------------
def _looks_like_path(argument: str) -> bool:
    """A single-line string naming a file that exists is a path; anything with a
    newline or a ``def``/``import`` is program text. The file check is the
    decisive one, so ``foo.py`` that exists wins even though it is also valid to
    read as (degenerate) contents."""

    if "\n" in argument or "\r" in argument:
        return False
    try:
        return os.path.isfile(argument)
    except (OSError, ValueError):
        return False


def _top_level_functions(module: ast.Module) -> list[ast.FunctionDef]:
    return [
        node for node in module.body if isinstance(node, ast.FunctionDef)
    ]


def _choose_entrypoint(
    module: ast.Module, entrypoint: str | None, origin: str
) -> ast.FunctionDef:
    functions = _top_level_functions(module)
    if entrypoint is not None:
        for function in functions:
            if function.name == entrypoint:
                return function
        available = ", ".join(function.name for function in functions) or "none"
        raise ValueError(
            f"entrypoint {entrypoint!r} not found in {origin} "
            f"(top-level functions: {available})"
        )
    if not functions:
        raise ValueError(f"{origin} defines no top-level function to compile")
    if len(functions) == 1:
        return functions[0]
    named_main = [f for f in functions if f.name == "main"]
    if len(named_main) == 1:
        return named_main[0]
    names = ", ".join(function.name for function in functions)
    raise ValueError(
        f"{origin} defines several top-level functions ({names}); name one "
        "explicitly with entrypoint="
    )


def _parameter_names(function: ast.FunctionDef) -> tuple[str, ...]:
    args = function.args
    ordered = [*args.posonlyargs, *args.args]
    if args.vararg is not None:
        ordered.append(args.vararg)
    ordered.extend(args.kwonlyargs)
    if args.kwarg is not None:
        ordered.append(args.kwarg)
    return tuple(argument.arg for argument in ordered)


def resolve_program(
    program: str, entrypoint: str | None = None
) -> ResolvedProgram:
    """Resolve ``program`` (a file path *or* Python source contents) plus an
    optional entrypoint into a :class:`ResolvedProgram`. Detects which form the
    argument is and reads the source accordingly."""

    if _looks_like_path(program):
        path = Path(program)
        source = path.read_text(encoding="utf-8")
        origin = f"file:{path}"
    else:
        source = program
        origin = "contents"
    module = ast.parse(source, filename=origin)
    function = _choose_entrypoint(module, entrypoint, origin)
    return ResolvedProgram(
        source=source,
        entrypoint=function.name,
        origin=origin,
        parameters=_parameter_names(function),
    )


# --- compile to the neutral dual IR (whole-program, no bake) -------------------
def compile_to_dual_ir(
    resolved: ResolvedProgram,
    *,
    feeds: Mapping[str, Any] | None = None,
    mutable_parameters: Sequence[str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Any, Any]:
    """Compile a resolved program up to the dual IR and arrange it as a
    :class:`DualIRShell`. Returns ``(compilation, shell)``.

    The whole-program no-bake route: ``precompile_only=True`` keeps parameters
    symbolic and ``bake_mode="whole_program"`` withholds the final
    backend-specific fused reduction, so the compilation stops at the neutral
    dual IR -- the highest checkpoint we release.
    """

    from ..common.tensors.accelerator_backends.aot_compile import compile_ast_aot
    from ..common.tensors.accelerator_backends.dual_ir_shell import (
        compose_dual_ir_shell,
    )

    # Parameters stay parameters: everything the entrypoint takes is a mutable
    # parameter left unfed, so we capture a program rather than a baked instance.
    parameters = (
        tuple(mutable_parameters)
        if mutable_parameters is not None
        else resolved.parameters
    )
    compilation = compile_ast_aot(
        resolved.source,
        resolved.entrypoint,
        dict(feeds or {}),
        backend="c",
        precompile_only=True,
        bake_mode="whole_program",
        mutable_parameters=parameters,
        progress=progress,
    )
    shell = compose_dual_ir_shell(compilation)
    return compilation, shell


def release_checkpoint(shell: Any, *, key: str | None = None) -> Path:
    """Release the dual-IR shell -- the highest backend-neutral checkpoint -- to
    the shell archive and return the file written."""

    from ..common.tensors.accelerator_backends.shell_archive import save_shell

    return save_shell(shell, key=key)


# --- run the neutral checkpoint in the Fortran shell --------------------------
def run_in_fortran_shell(
    resolved: ResolvedProgram,
    compilation: Any,
    directory: str | Path,
    *,
    feeds: Mapping[str, Any] | None = None,
    mutable_parameters: Sequence[str] | None = None,
    frames: int = 1,
    files: Mapping[str, str | Path] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Any, Any]:
    """Run the already-produced dual IR in the Fortran shell. Hands the same
    ``compilation`` to ``compile_ast_fortran_c_shell`` (no recompile), builds
    the executable, and runs it. Returns ``(executable, completed_process)``."""

    from .fortran_c_shell import compile_ast_fortran_c_shell

    parameters = (
        tuple(mutable_parameters)
        if mutable_parameters is not None
        else resolved.parameters
    )
    executable = compile_ast_fortran_c_shell(
        resolved.source,
        resolved.entrypoint,
        dict(feeds or {}),
        directory,
        name=resolved.entrypoint,
        progress=progress,
        mutable_parameters=parameters,
        compilation=compilation,
    )
    completed = executable.run(frames=frames, files=dict(files or {}))
    return executable, completed


# --- the whole route ----------------------------------------------------------
def bootstrap(
    program: str,
    entrypoint: str | None = None,
    *,
    feeds: Mapping[str, Any] | None = None,
    mutable_parameters: Sequence[str] | None = None,
    directory: str | Path | None = None,
    run: bool = True,
    frames: int = 1,
    files: Mapping[str, str | Path] | None = None,
    progress: Callable[[str], None] | None = None,
) -> BootstrapResult:
    """Python (path or contents) -> dual IR -> released checkpoint -> Fortran
    shell, start to finish.

    Set ``run=False`` to stop after releasing the neutral checkpoint (compile
    the program to dual IR without building/executing the Fortran shell).
    """

    def _report(message: str) -> None:
        if progress is not None:
            progress(message)

    resolved = resolve_program(program, entrypoint)
    _report(
        f"resolved {resolved.origin}; entrypoint {resolved.entrypoint}"
        f"({', '.join(resolved.parameters) or ''})"
    )

    _report("compiling to dual IR (whole-program, no bake)")
    compilation, shell = compile_to_dual_ir(
        resolved,
        feeds=feeds,
        mutable_parameters=mutable_parameters,
        progress=progress,
    )

    checkpoint_path = release_checkpoint(shell, key=resolved.entrypoint)
    _report(f"released dual-IR checkpoint -> {checkpoint_path}")

    executable = None
    run_result = None
    if run:
        work = Path(directory) if directory is not None else (
            checkpoint_path.parent / f"{resolved.entrypoint}-fortran-shell"
        )
        work.mkdir(parents=True, exist_ok=True)
        _report(f"running dual IR in the Fortran shell ({work})")
        executable, run_result = run_in_fortran_shell(
            resolved,
            compilation,
            work,
            feeds=feeds,
            mutable_parameters=mutable_parameters,
            frames=frames,
            files=files,
            progress=progress,
        )
        _report("Fortran shell run complete")

    return BootstrapResult(
        resolved=resolved,
        compilation=compilation,
        shell=shell,
        checkpoint_path=checkpoint_path,
        executable=executable,
        run_result=run_result,
    )


def _parse_feeds(items: Sequence[str] | None) -> dict[str, Any]:
    """Parse ``name=<python-literal>`` CLI feed items into a feeds dict."""

    feeds: dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"feed {item!r} must be name=<python-literal>")
        name, _, literal = item.partition("=")
        feeds[name.strip()] = ast.literal_eval(literal)
    return feeds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a Python program (file path or source) to the neutral "
            "dual IR and run it in the Fortran shell."
        )
    )
    parser.add_argument(
        "program",
        help="a Python file path, or the program source contents",
    )
    parser.add_argument(
        "-e", "--entrypoint", default=None,
        help="entrypoint function name (auto-detected if omitted)",
    )
    parser.add_argument(
        "-f", "--feed", action="append", metavar="NAME=LITERAL", default=[],
        help="a feed as name=<python-literal> (repeatable)",
    )
    parser.add_argument(
        "-p", "--parameter", action="append", metavar="NAME", default=None,
        help="explicit mutable parameter name (repeatable); defaults to the "
        "entrypoint's parameters",
    )
    parser.add_argument(
        "-d", "--directory", default=None,
        help="work directory for the Fortran shell build",
    )
    parser.add_argument(
        "--no-run", action="store_true",
        help="stop after releasing the dual-IR checkpoint (do not run Fortran)",
    )
    parser.add_argument(
        "--frames", type=int, default=1, help="frames to run in the shell",
    )
    arguments = parser.parse_args(argv)

    result = bootstrap(
        arguments.program,
        arguments.entrypoint,
        feeds=_parse_feeds(arguments.feed),
        mutable_parameters=arguments.parameter,
        directory=arguments.directory,
        run=not arguments.no_run,
        frames=arguments.frames,
        progress=lambda message: print(f"[bootstrap] {message}"),
    )

    print(f"entrypoint     : {result.resolved.entrypoint}")
    print(f"origin         : {result.resolved.origin}")
    print(f"dual IR release: {result.checkpoint_path}")
    if result.run_result is not None:
        print("--- Fortran shell stdout ---")
        print(result.run_result.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
