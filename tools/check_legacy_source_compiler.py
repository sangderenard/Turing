"""Flag calls to the deprecated two-step source compiler in a target file.

``compile_ast_aot`` + ``lower_precompile_and_control_to_ssa`` is the
deprecated pattern that :mod:`src.compiler.compiler_entrypoints` exists to
flag for migration (see ``warn_legacy_source_compiler``).  The single
sanctioned whole-program source compiler is
``src.compiler.fortran_c_shell.lower_ast_source_to_ssa``.

This walks a target Python file's AST, resolves import aliases, and reports
every call site that invokes a legacy entry point directly -- by name,
regardless of how it was imported or aliased.  Exit status 2 if any are
found, 0 if clean.

Usage:
    python tools/check_legacy_source_compiler.py path/to/target.py [more.py ...]
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CANONICAL_SOURCE_COMPILER = "src.compiler.fortran_c_shell.lower_ast_source_to_ssa"

LEGACY_NAMES = {
    "compile_ast_aot",
    "lower_precompile_and_control_to_ssa",
}


def _local_names_for(tree: ast.Module) -> dict[str, str]:
    """Map local names bound in ``tree`` back to the legacy name they alias."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in LEGACY_NAMES:
                    bound[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                tail = alias.name.rsplit(".", 1)[-1]
                if tail in LEGACY_NAMES:
                    bound[alias.asname or alias.name] = tail
    return bound


def _call_target_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return ``(line, legacy_name)`` for every legacy call site in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    local_names = _local_names_for(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _call_target_name(node.func)
        if called is None:
            continue
        legacy_name = local_names.get(called, called if called in LEGACY_NAMES else None)
        if legacy_name is not None:
            hits.append((node.lineno, legacy_name))
    return sorted(hits)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    exit_code = 0
    for arg in argv:
        path = Path(arg)
        if not path.is_file():
            print(f"SKIP {path}: not a file")
            exit_code = 1
            continue
        hits = scan_file(path)
        if not hits:
            print(f"CLEAN {path}")
            continue
        exit_code = 2
        for line, legacy_name in hits:
            print(f"LEGACY {path}:{line} calls {legacy_name}(); "
                  f"use {CANONICAL_SOURCE_COMPILER} instead")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
