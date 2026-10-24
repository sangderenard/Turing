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
import traceback


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


def _module_package(module_path: str) -> str | None:
    """The package a module file's relative imports resolve against: its
    directory as a dotted path from the working directory (the repo root),
    or None for a file outside it.  ``src/common/tensors/abstraction.py``
    is ``src.common.tensors``, so ``from . import autograd`` resolves."""

    relative = os.path.relpath(os.path.abspath(module_path), os.getcwd())
    parts = os.path.dirname(relative).split(os.sep)
    if not relative or relative.startswith("..") or not all(
        part.isidentifier() for part in parts
    ):
        return None
    return ".".join(parts)


def _is_class(source: str, name: str) -> bool:
    module = ast.parse(source)
    return any(
        isinstance(node, ast.ClassDef) and node.name == name
        for node in module.body
    )


PROGRAM_EXTRACTION_CONTRACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "extraction_contracts", "program_extraction.yaml",
)


def compile_class_to_dll(
    source, class_name, outdir, *, progress=None, python_package=None,
    extraction_contract=PROGRAM_EXTRACTION_CONTRACT,
):
    """Compile a whole class to one shared library with no privileged entry.

    DEPRECATED, and redirected.  The class is still the unit: the closure is
    seeded from EVERY method (constructors included), and the methods share
    one ``self`` record, so one library exports them all.  That lowering is
    now the canonical repository-SSA compiler (``lower_ast_source_to_ssa``),
    then Fortran emission and the shared-library link; the handle is the same
    ``FortranCShellExecutable``.

    Program: ``__init__(self, c, n): self.n = 0; if c: self.n = n``.  Wrong
    record: this adapter called the legacy ``compile_ast_fortran_c_shell``,
    which derives no record ABI for the receiver, so every field cell was
    float64 and both stores ran at entry, outside the ``if`` (``get()``
    returned 0 for c true).  The canonical lowering stores each field in its
    arm, in a typed column (int64 for ``n``).
    """

    import warnings

    from src.compiler.compiler_entrypoints import CANONICAL_SOURCE_COMPILER
    from src.compiler.fortran_c_shell import (
        compile_fortran_module_c_shell, lower_ast_source_to_ssa,
    )
    from src.compiler.ssa_fortran_backend import emit_module

    with warnings.catch_warnings():
        # Loud: shown every time, whatever the caller's warning filters.
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            f"compile_class_to_dll is deprecated and is redirected to "
            f"{CANONICAL_SOURCE_COMPILER} (every method of {class_name!r} "
            "seeded, one shared self record), then Fortran emission and a "
            "shared-library link. Call that route directly.",
            DeprecationWarning,
            stacklevel=2,
        )

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
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, nominal, name=class_name, dependency_seeds=seeds,
        python_package=python_package,
        extraction_contract=extraction_contract,
        progress=progress,
    )
    fortran_module = emit_module(module, progress=progress)
    return compile_fortran_module_c_shell(
        fortran_module, {}, outdir, name=class_name, library=True,
    )


def _public_surface_classes(module) -> list:
    """The public classes a module defines itself (not imported): the public
    surface to export. A leading underscore marks a name private."""

    import inspect

    return [
        obj
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and inspect.isclass(obj)
        and getattr(obj, "__module__", None) == module.__name__
    ]


def export_public_surface(module_name: str, outdir, *, progress=None):
    """Ingest a module by name and export its whole public surface as a DLL.

    The public classes the module defines are ``retain``ed, which pulls in each
    class's FULL method surface -- body methods and methods bound onto the class
    from other modules (``_attach_external_methods``) -- and fills those regions
    out. ``library=True`` then emits every one as a linkable export. This is the
    foundational-library pattern: a module's public API compiled to a shared
    object other compiled programs link against as externals.
    """

    from src.compiler.compiler_entrypoints import warn_legacy_source_compiler

    warn_legacy_source_compiler("export_public_surface")

    import ast
    import importlib
    import inspect

    from src.compiler.fortran_c_shell import compile_ast_fortran_c_shell

    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    classes = _public_surface_classes(module)
    if not classes:
        raise ValueError(f"{module_name!r} defines no public classes to export")

    # Nominal entrypoint (naming/ABI only): a method of the first public class,
    # resolved by its qualified name. retain fills out every retained class.
    tree = ast.parse(source)
    first = classes[0].__name__
    first_methods = _class_method_names(source, first)
    if not first_methods:
        raise ValueError(f"public class {first!r} defines no methods")
    nominal = (
        f"{first}.__init__" if "__init__" in first_methods
        else f"{first}.{first_methods[0]}"
    )
    if progress is not None:
        progress(
            f"exporting public surface of {module_name}: "
            f"{[c.__name__ for c in classes]}"
        )
    return compile_ast_fortran_c_shell(
        source, nominal, {}, outdir,
        name=module_name.rsplit(".", 1)[-1],
        library=True, retain=tuple(classes), progress=progress,
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(__doc__)
        return 2
    module_path, entrypoint, outdir = argv
    source = open(module_path, "r", encoding="utf-8").read()
    python_package = _module_package(module_path)
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
                python_package=python_package,
            )
        else:
            handle = compile_ast_fortran_c_shell(
                source, entrypoint, {}, outdir, name=entrypoint,
                library=True, progress=progress,
                runtime_closure_only=True, python_package=python_package,
            )
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
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
