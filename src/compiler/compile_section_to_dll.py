"""Compile a whole module file as standalone source into a DLL. No time cap.

Streams every compiler phase to stdout. The module's own source dictates the
types -- no fragment extraction, no symbolic-parameter stripping.

Usage (run from the turing repo root):

    python -m src.compiler.compile_section_to_dll <module_path> <entrypoint> <outdir>
"""
from __future__ import annotations

import ast
import os
import sys
import time


def _class_method_names(source: str, class_name: str) -> list[str]:
    """Every method a class defines, in source order -- constructors
    (``__init__``/``__new__``) and dunders included; a class compiles as the
    union of all of them, none privileged."""

    module = ast.parse(source)
    classes = [
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(
            f"expected exactly one class {class_name!r}; found {len(classes)}"
        )
    return [
        node.name
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _is_class(source: str, name: str) -> bool:
    module = ast.parse(source)
    return any(
        isinstance(node, ast.ClassDef) and node.name == name
        for node in module.body
    )


def compile_class_to_dll(source, class_name, outdir, *, progress=None):
    """Compile a whole class to a DLL with no privileged entry function.

    The class is the unit: the dependency closure is seeded from EVERY method
    (constructors included), so the whole object is retained and compiled as one
    general dependency. Object state flows through get/set-attribute field slots,
    which is what keeps it safe. Each method already emits ``bind(C, name=...)``,
    so all export cleanly from one shared library.
    """

    from src.compiler.fortran_c_shell import compile_ast_fortran_c_shell

    methods = _class_method_names(source, class_name)
    if not methods:
        raise ValueError(f"class {class_name!r} defines no methods to compile")
    qualified = [f"{class_name}.{name}" for name in methods]
    # Nominal entry drives naming/ABI only; the closure is seeded from ALL
    # methods below. Prefer a constructor so it is never dropped.
    nominal = (
        f"{class_name}.__init__" if "__init__" in methods else qualified[0]
    )
    seeds = tuple(name for name in qualified if name != nominal)
    return compile_ast_fortran_c_shell(
        source, nominal, {}, outdir, name=class_name,
        library=True, dependency_seeds=seeds, progress=progress,
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(__doc__)
        return 2
    module_path, entrypoint, outdir = argv
    source = open(module_path, "r", encoding="utf-8").read()
    from src.compiler.fortran_c_shell import compile_ast_fortran_c_shell

    def progress(message: str) -> None:
        print(message, flush=True)

    started = time.time()
    try:
        if _is_class(source, entrypoint):
            # A class has no entry function: seed the closure from all methods.
            print(f"(compiling class {entrypoint!r} -- seeding from all methods)",
                  flush=True)
            handle = compile_class_to_dll(
                source, entrypoint, outdir, progress=progress,
            )
        else:
            handle = compile_ast_fortran_c_shell(
                source, entrypoint, {}, outdir, name=entrypoint,
                library=True, progress=progress,
            )
    except Exception as error:  # noqa: BLE001
        print(
            f"\nERROR after {round(time.time() - started, 1)}s: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return 1
    path = str(handle.executable_path)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    print(
        f"\nOK in {round(time.time() - started, 1)}s -> {path} ({size} bytes)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
