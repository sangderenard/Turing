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

    from src.compiler.compiler_entrypoints import warn_legacy_source_compiler
    from src.compiler.fortran_c_shell import compile_ast_fortran_c_shell

    warn_legacy_source_compiler("compile_class_to_dll")

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
        runtime_closure_only=True,
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
                runtime_closure_only=True,
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
