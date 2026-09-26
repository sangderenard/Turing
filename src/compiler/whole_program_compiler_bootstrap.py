"""Build the source compiler as one linked native program.

This is deliberately not the historical compiler-bootstrap creep.  There are
no generations, independently installed Python callables, or native products
fed back into the next compile.  The canonical source compiler pursues one
fixed-signature root into one ProcessGraph, lowers the complete reachable
program to one repository-SSA module, and the C backend emits that root and
its complete ``Call`` closure into one DLL.

The planning and lowering stages are separate because the compiler's resolved
compilation-unit plan is useful evidence in its own right.  Planning does not
claim a native result.  A native result exists only after C emission reports no
shortfalls and the single shared library has been linked.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


BOOTSTRAP_SCHEMA = "turing.whole-program-compiler-bootstrap.v1"
COMPILER_PROGRAM_SOURCE = """\
def compile_program(source, entrypoint, extraction_contract):
    return source_compiler_impl(
        source,
        entrypoint,
        extraction_contract=extraction_contract,
    )
"""


@dataclass(frozen=True)
class WholeProgramCompilerProduct:
    """The single-image bootstrap result and its audit receipt."""

    module: Any
    outputs: Mapping[str, Any]
    exports: tuple[str, ...]
    root: str
    artifact: Any
    library: Path
    manifest: Path


def _source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@contextmanager
def _without_incremental_bootstrap(directory: Path) -> Iterator[None]:
    """Keep the canonical compiler from activating historical products.

    ``lower_ast_source_to_ssa`` checks the receipt-gated registry on every
    invocation.  Pointing that check at an absent file is the supported way to
    obtain the authored compiler.  Restoring both variables matters in shared
    test processes and interactive sessions.
    """

    from .compiler_bootstrap_runtime import (
        COMPILER_BOOTSTRAP_PRODUCTS_ENV,
        COMPILER_BOOTSTRAP_REGISTRY_ENV,
    )

    previous = {
        name: os.environ.get(name)
        for name in (
            COMPILER_BOOTSTRAP_PRODUCTS_ENV,
            COMPILER_BOOTSTRAP_REGISTRY_ENV,
        )
    }
    os.environ[COMPILER_BOOTSTRAP_PRODUCTS_ENV] = ""
    os.environ[COMPILER_BOOTSTRAP_REGISTRY_ENV] = str(
        directory / "no-incremental-bootstrap-registry.json"
    )
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _compiler_bindings(
    source_compiler_impl: Callable[..., Any] | None,
) -> dict[str, Any]:
    if source_compiler_impl is None:
        # Compile the implementation behind the public identity-book wrapper.
        # The outer invocation still enters exclusively through the canonical
        # public lower_ast_source_to_ssa entry.  Giving the compiled program a
        # fixed signature avoids turning *args/**kwargs transport into part of
        # the native ABI while preserving the compiler implementation itself.
        from .fortran_c_shell import _lower_ast_source_to_ssa_impl

        source_compiler_impl = _lower_ast_source_to_ssa_impl
    return {"source_compiler_impl": source_compiler_impl}


def plan_whole_program_compiler(
    directory: str | Path,
    *,
    source_compiler_impl: Callable[..., Any] | None = None,
    extraction_contract: str | Path = (
        "extraction_contracts/program_extraction.yaml"
    ),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Resolve the compiler program and return its exact compilation-unit plan."""

    from .fortran_c_shell import lower_ast_source_to_ssa

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plans: list[dict[str, Any]] = []
    with _without_incremental_bootstrap(destination):
        lower_ast_source_to_ssa(
            COMPILER_PROGRAM_SOURCE,
            "compile_program",
            python_bindings=_compiler_bindings(source_compiler_impl),
            extraction_contract=extraction_contract,
            name="whole_program_compiler",
            compilation_unit_plan_sink=lambda plan: plans.append(dict(plan)),
            stop_after_compilation_unit_plan=True,
            progress=progress,
        )
    if len(plans) != 1:
        raise RuntimeError(
            "compiler planning did not publish exactly one compilation-unit plan"
        )
    receipt = {
        "schema": BOOTSTRAP_SCHEMA,
        "stage": "planned",
        "strategy": "single-root-single-module-single-library",
        "incremental_bootstrap_products": [],
        "program_source_sha256": _source_sha256(COMPILER_PROGRAM_SOURCE),
        "compilation_unit_plan": plans[0],
    }
    path = destination / "plan.json"
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def _root_export(exports: tuple[str, ...]) -> str:
    matches = tuple(
        name for name in map(str, exports)
        if name == "whole_program_compiler__compile_program"
        or name.endswith("__compile_program")
    )
    if len(matches) != 1:
        raise RuntimeError(
            "whole compiler image must expose exactly one compile_program root; "
            f"exports={exports!r}"
        )
    return matches[0]


def build_whole_program_compiler(
    directory: str | Path,
    *,
    source_compiler_impl: Callable[..., Any] | None = None,
    extraction_contract: str | Path = (
        "extraction_contracts/program_extraction.yaml"
    ),
    optimization: str = "O0",
    progress: Callable[[str], None] | None = None,
) -> WholeProgramCompilerProduct:
    """Lower and link the complete reachable compiler program into one DLL."""

    from .fortran_c_shell import lower_ast_source_to_ssa
    from .ssa_c_backend import emit_ssa_to_c

    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with _without_incremental_bootstrap(destination):
        module, outputs, exports_value = lower_ast_source_to_ssa(
            COMPILER_PROGRAM_SOURCE,
            "compile_program",
            python_bindings=_compiler_bindings(source_compiler_impl),
            extraction_contract=extraction_contract,
            name="whole_program_compiler",
            progress=progress,
        )
    exports = tuple(map(str, exports_value))
    root = _root_export(exports)
    artifact = emit_ssa_to_c(
        module,
        root,
        entry_name="turing_compile_program",
    )
    if not artifact.complete:
        frontier = {
            "schema": BOOTSTRAP_SCHEMA,
            "stage": "c-emission-frontier",
            "strategy": "single-root-single-module-single-library",
            "incremental_bootstrap_products": [],
            "root": root,
            "ssa_function_count": len(module.functions),
            "shortfalls": [
                {
                    "operation": str(item.operation),
                    "reason": str(item.reason),
                }
                for item in artifact.shortfalls
            ],
        }
        path = destination / "frontier.json"
        path.write_text(
            json.dumps(frontier, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        raise RuntimeError(
            "whole compiler reached C emission with shortfalls; "
            f"see {path}"
        )
    artifact.compile(destination, optimization=optimization)
    if artifact.library_path is None or not artifact.library_path.is_file():
        raise RuntimeError("C backend returned without a linked compiler DLL")
    calls = tuple(sorted(map(str, module.functions)))
    manifest_value = {
        "schema": BOOTSTRAP_SCHEMA,
        "stage": "linked",
        "strategy": "single-root-single-module-single-library",
        "incremental_bootstrap_products": [],
        "program_source_sha256": _source_sha256(COMPILER_PROGRAM_SOURCE),
        "root": root,
        "exports": list(exports),
        "ssa_function_count": len(calls),
        "ssa_functions": list(calls),
        "library": artifact.library_path.name,
        "library_sha256": hashlib.sha256(
            artifact.library_path.read_bytes()
        ).hexdigest(),
        "optimization": optimization,
        "compilation_unit_plan": module.metadata.get(
            "compilation_unit_plan", {}
        ),
    }
    manifest = destination / "manifest.json"
    manifest.write_text(
        json.dumps(manifest_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return WholeProgramCompilerProduct(
        module=module,
        outputs=outputs,
        exports=exports,
        root=root,
        artifact=artifact,
        library=artifact.library_path,
        manifest=manifest,
    )


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "COMPILER_PROGRAM_SOURCE",
    "WholeProgramCompilerProduct",
    "build_whole_program_compiler",
    "plan_whole_program_compiler",
]
