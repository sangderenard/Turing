"""Whole-class compile trial: a standing source of frontier refusals.

A class is compiled two ways, and both results are reported side by side:

1. **Class route** (``compile_class_to_dll``): every method seeds one shared
   library, with the program extraction contract and the module's package.
   The trial captures the whole-object SSA module the route hands to its
   backend and checks it against the class's own method list.  A method with
   no function in that module is a FALSE ACCEPTANCE candidate: the library
   would ship without it and nothing refused it by name.
2. **Per-method catalogue** (``tools.compile_project_catalogue``, optional):
   each method compiled as its own unit, so every method reports its own
   first refusal instead of the class stopping at the first one.

Usage (from the repo root)::

    python -u -m tools.trial_whole_class_compile \
        --source src/compiler/ssa_reference_evaluator.py \
        --class SSAReferenceEvaluator --output out/ssa_interpreter_class \
        [--catalogue] [--jobs 2]

It writes ``summary.json`` in the output directory and prints the same
facts.  The trial never decides a refusal is acceptable; it only lists them.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def class_methods(source: str, class_name: str) -> list[str]:
    """The methods a class defines, in source order, dunders included."""

    from src.compiler.compile_section_to_dll import _class_method_names

    return _class_method_names(source, class_name)


def method_coverage(
    class_name: str, methods: list[str], module: object,
) -> dict[str, str]:
    """``present`` / ``missing`` per method, or ``unreached`` when the route
    refused before it produced a whole-object module.

    Presence is read from each SSA function's ``source_qualified_name``
    record, which the class surface writes for every method it lowers.
    """

    if module is None:
        return {method: "unreached" for method in methods}
    lowered = {
        str(function.metadata.get("source_qualified_name"))
        for function in getattr(module, "functions", {}).values()
        if function.metadata.get("source_qualified_name") is not None
    }
    return {
        method: (
            "present" if f"{class_name}.{method}" in lowered else "missing"
        )
        for method in methods
    }


def run_class_route(source_path: Path, class_name: str, output: Path) -> dict:
    """Compile through the class route and capture the whole-object module."""

    import src.compiler.fortran_c_shell as shell
    from src.compiler.compile_section_to_dll import (
        _module_package, compile_class_to_dll,
    )

    source = source_path.read_text(encoding="utf-8")
    captured: dict[str, object] = {}
    lower = shell.lower_ast_source_to_ssa

    def capture(*args, **kwargs):
        result = lower(*args, **kwargs)
        module = result[0]
        # The units the compiler planned for this object: a method planned
        # here but absent from the module was dropped after planning, not
        # left unplanned.
        plan = (getattr(module, "metadata", {}) or {}).get(
            "compilation_unit_plan"
        ) or {}
        prefix = f"{class_name}."
        captured["planned"] = sorted({
            str(name)[len(prefix):]
            for unit in plan.get("units", ())
            for name in unit.get("qualified_names", ())
            if str(name).startswith(prefix)
        })
        captured["module"] = module
        captured["exports"] = tuple(result[2])
        return result

    shell.lower_ast_source_to_ssa = capture
    started = time.monotonic()
    refusal = None
    library = None
    try:
        handle = compile_class_to_dll(
            source, class_name, str(output / "class_route"),
            progress=lambda _message: None,
            python_package=_module_package(str(source_path)),
        )
        library = str(getattr(handle, "executable_path", "") or "")
    except Exception as error:  # noqa: BLE001 -- the refusal IS the product
        frames = traceback.extract_tb(error.__traceback__)
        refusal = {
            "type": type(error).__name__,
            "message": str(error),
            "raised_at": [
                f"{Path(frame.filename).name}:{frame.lineno} {frame.name}"
                for frame in frames[-4:]
            ],
        }
    finally:
        shell.lower_ast_source_to_ssa = lower
    module = captured.get("module")
    functions = (
        None if module is None else sorted(getattr(module, "functions", {}))
    )
    methods = class_methods(source, class_name)
    coverage = method_coverage(class_name, methods, module)
    return {
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "library": library,
        "refusal": refusal,
        "planned_methods": captured.get("planned"),
        "surface_functions": functions,
        "exports": list(captured.get("exports", ())),
        "method_coverage": coverage,
        "false_acceptance_candidates": sorted(
            method for method, state in coverage.items() if state == "missing"
        ),
    }


def run_catalogue(
    source_path: Path, class_name: str, output: Path, jobs: int,
) -> dict:
    """Compile each method as its own unit and read every unit's refusal."""

    source = source_path.read_text(encoding="utf-8")
    catalogue = output / "catalogue"
    command = [
        sys.executable, "-u", "-m", "tools.compile_project_catalogue",
        "--source", str(source_path), "--output", str(catalogue),
        "--emit-native", "--jobs", str(jobs),
    ]
    for method in class_methods(source, class_name):
        command += ["--entry", f"{class_name}.{method}"]
    started = time.monotonic()
    with open(output / "catalogue.log", "w", encoding="utf-8") as log:
        returncode = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    manifest_path = catalogue / "manifest.json"
    if not manifest_path.exists():
        return {"returncode": returncode, "units": [], "manifest": None}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = []
    for unit in manifest.get("units", ()):
        stage = unit.get("stage")
        compiler = stage.get("compiler", {}) if isinstance(stage, dict) else {}
        row = {
            "qualified_name": unit.get("qualified_name"),
            "status": unit.get("status"),
            "native_status": unit.get("native_status"),
            "stage": compiler.get("stage_key"),
            "error_type": unit.get("error_type"),
            "error": unit.get("error"),
        }
        if unit.get("status") == "partial":
            row["incomplete"] = {
                key: unit.get(key)
                for key in (
                    "repository_ssa_complete", "missing_lexical_captures",
                    "missing_used_parameters",
                    "unmaterialized_extraction_boundaries",
                    "unresolved_program_abi_references",
                    "unresolved_required_source_values", "control_shortfalls",
                )
                if unit.get(key)
            }
        units.append(row)
    return {
        "returncode": returncode,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "extraction_contract": manifest.get("extraction_contract"),
        "manifest": str(manifest_path),
        "units": units,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalogue", action="store_true",
                        help="also compile every method as its own unit")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": str(args.source),
        "class": args.class_name,
        "class_route": run_class_route(args.source, args.class_name, args.output),
    }
    if args.catalogue:
        summary["catalogue"] = run_catalogue(
            args.source, args.class_name, args.output, args.jobs,
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )

    route = summary["class_route"]
    print(f"class route ({route['elapsed_seconds']}s), planned: "
          f"{route['planned_methods']}")
    for method, state in route["method_coverage"].items():
        print(f"  {state:9s} {method}")
    if route["refusal"] is not None:
        print(f"  REFUSED {route['refusal']['type']}: "
              f"{route['refusal']['message'][:400]}")
        print(f"    at {' <- '.join(reversed(route['refusal']['raised_at']))}")
    else:
        print(f"  library {route['library']}")
    for method in route["false_acceptance_candidates"]:
        print(f"  FALSE-ACCEPTANCE candidate: {method} has no surface function")
    for unit in summary.get("catalogue", {}).get("units", ()):
        print(f"unit {unit['status']:8s} {unit['qualified_name']} "
              f"[{unit['stage']}] {unit['error_type'] or ''}: "
              f"{str(unit['error'] or unit.get('incomplete') or '')[:300]}")
    print(f"summary -> {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
