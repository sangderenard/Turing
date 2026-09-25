from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import audit_ancestry_retained_loop_graph as audit
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.hierarchical_plan import PlanCall, PlanClosure


base = audit._base_records()
policy = ExtractionContract(audit.CONTRACTS / "program_extraction.yaml").with_program_abi({
    "records": {
        "Metrics": base["records"]["Metrics"],
        "Targets": base["records"]["Targets"],
        "STController": base["records"]["STController"],
        "BalloonTireManagedState": base["records"]["BalloonTireManagedState"],
    },
    "bindings": [
        {"function": "*", "parameter": "metrics", "record": "Metrics"},
        {"function": "*", "parameter": "targets", "record": "Targets"},
        {"function": "*", "parameter": "ctrl", "record": "STController"},
        {"function": "*", "parameter": "state", "record": "BalloonTireManagedState"},
    ],
    "values": [],
})
captured = []
lower_ast_source_to_ssa(
    audit._source(), "root", name="cycle_id_map",
    extraction_contract=policy,
    resolved_process_graph_sink=captured.append,
    stop_after_compilation_unit_plan=True,
)
root = captured[0]
wanted = {19, 47, 49, 51, 59, 138, 145, 266, 283, 293, 294, 299,
          377, 380, 381, 382, 383, 384, 385, 392, 395, 396, 398, 399,
          400, 402, 420, 421, 422, 424, 519, 520, 523, 524, 526, 528,
          530, 543, 545, 547, 570}
for entry in root.function_table:
    graph = getattr(entry, "graph", None)
    if getattr(graph, "G", None) is None:
        continue
    if graph.G.graph.get("function_name") != "step_with_dt_control_used":
        continue
    print(f"ENTRY {type(entry).__name__} fields={tuple(vars(entry)) if hasattr(entry, '__dict__') else ()}")
    for field_name, field_value in vars(entry).items():
        if isinstance(field_value, PlanClosure):
            pending = [field_value]
            while pending:
                closure = pending.pop()
                for item in closure.items:
                    if isinstance(item, PlanCall):
                        print(
                            f"PLANCALL field={field_name} id={item.callsite_id} "
                            f"callee={item.callee.name} args={item.argument_value_ids} "
                            f"results={item.result_value_ids}"
                        )
                        pending.append(item.callee)
                    elif isinstance(item, PlanClosure):
                        pending.append(item)
    print(f"FUNCTION {graph.G.graph.get('function_name')} nodes={graph.G.number_of_nodes()}")
    for node_id in sorted(wanted):
        data = graph.G.nodes.get(node_id)
        if data is None:
            print(f"{node_id}: MISSING")
            continue
        expression = data.get("expr_obj")
        try:
            source = ast.unparse(expression) if isinstance(expression, ast.AST) else repr(expression)
        except Exception:
            source = repr(expression)
        print(
            f"{node_id}: type={data.get('type')!r} op={data.get('op')!r} "
            f"line={getattr(expression, 'lineno', None)!r} source={source!r} "
            f"parents={data.get('parents')!r} attrs={data.get('attributes')!r}"
        )
