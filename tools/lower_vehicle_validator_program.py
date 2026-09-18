"""Probe whole-program ingestion of the actual Python dually validator.

This delegates to the existing runner without copying its scientific loop or
substituting a smaller graph. It does not execute the validator. A successful
plan is only an ingestion diagnostic, not a native build or parity result.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.project_compilation_product import compile_project_call
    from src.compiler.work_contract import set_active_contract

    set_active_contract("develop")
    policy = ExtractionContract(
        ROOT / "extraction_contracts" / "program_extraction.yaml"
    ).with_execution_file(
        ROOT / "extraction_contracts" / "vehicle_full_native_execution.yaml"
    )
    started = time.perf_counter()
    receipt = {
        "schema": "turing.whole-program-lowering-attempt.v1",
        "status": "running",
        "authored_entrypoint": "tools.run_vehicle_native_assembly._run_dually_python_profile",
        "compiler_entrypoint": "src.compiler.project_compilation_product.compile_project_call",
        "source_path": str(ROOT / "tools" / "run_vehicle_native_assembly.py"),
        "extraction_policy": str(ROOT / "extraction_contracts" / "program_extraction.yaml"),
        "execution_overlay": str(ROOT / "extraction_contracts" / "vehicle_full_native_execution.yaml"),
        "plan_only": args.plan_only,
        "execution": policy.execution.receipt(),
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")

    try:
        product = compile_project_call(
            ROOT / "tools" / "run_vehicle_native_assembly.py",
            "_run_dually_python_profile", args.output,
            extraction_contract=policy,
            progress=lambda message: print(message, flush=True),
            plan_only=args.plan_only,
        )
        complete = bool(product.get("repository_ssa_complete"))
        receipt.update(
            status="planned" if args.plan_only else "lowered" if complete else "incomplete",
            product=product,
        )
        return_code = 0 if args.plan_only or complete else 1
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
        boundaries = getattr(error, "boundaries", ())
        if boundaries:
            receipt["subdivision_boundaries"] = list(boundaries)
            receipt["frontier_kind"] = "compilation-subdivision-required"
        traceback.print_exc()
        # Preserve the innermost failing graph for inspection without another
        # whole-program ingestion. This observes exception frames only.
        frame = error.__traceback__
        failed_graph = None
        while frame is not None:
            frame_locals = frame.tb_frame.f_locals
            candidate = frame_locals.get("graph")
            # Precompile walks selected shells using ``target``. The outer
            # source catalogue may also be in scope; the exception's loop
            # boundaries belong to the selected target, not that catalogue.
            if frame.tb_frame.f_code.co_name == "prepare_graph_precompile":
                candidate = getattr(frame_locals.get("target"), "process_graph", candidate)
            if getattr(candidate, "G", None) is not None:
                failed_graph = candidate
            frame = frame.tb_next
        if failed_graph is not None:
            import networkx as nx
            from src.compiler.project_compilation_product import _dump_resolved_process_graph
            try:
                with (args.output / "failed-process-graph.pkl").open("wb") as stream:
                    _dump_resolved_process_graph(failed_graph, stream)
                dependencies = failed_graph.G.copy()
                dependencies.add_edges_from(
                    (int(parent), int(node))
                    for node, data in failed_graph.G.nodes(data=True)
                    for parent, _role in data.get("parents") or ()
                    if int(parent) in dependencies and int(parent) != int(node)
                )
                try:
                    cycle = nx.find_cycle(dependencies)
                except nx.NetworkXNoCycle:
                    cycle = []
                cycle_nodes = sorted({node for edge in cycle for node in edge[:2]})
                receipt["failed_graph"] = {
                    "function": failed_graph.G.graph.get("function_name"),
                    "cycle": cycle,
                    "levels_cover_nodes": set(failed_graph.levels) == set(failed_graph.G),
                    "has_recursion_table": bool(failed_graph.G.graph.get("recursion_table")),
                    "cycle_sources": {
                        str(node): ast.unparse(expression)
                        for node in cycle_nodes
                        for expression in (failed_graph.G.nodes[node].get("expr_obj"),)
                        if isinstance(expression, ast.AST)
                    },
                }
            except Exception as diagnostic_error:
                receipt["graph_diagnostic_error"] = repr(diagnostic_error)
        return_code = 1
    receipt["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
