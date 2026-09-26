from pathlib import Path
import ast
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from llvm_dt_system import dt_system_contract
from src.common.tensors import AbstractTensor
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

source = (
    "from src.common.dt_system.dt_scaler import coerce_metrics\n"
    "def root(metrics):\n"
    "    metrics = coerce_metrics(metrics)\n"
    "    return metrics.pub_tau.shape[0]\n"
)
captured = []
lower_ast_source_to_ssa(
    source, "root", name="inspect_coerce",
    extraction_contract=dt_system_contract("root", (), 1, 2),
    python_bindings={"AbstractTensor": AbstractTensor},
    stop_after_compilation_unit_plan=True,
    resolved_process_graph_sink=captured.append,
)
for entry in captured[0].function_table:
    graph = entry.graph.G
    if graph.graph.get("function_name") not in {"root", "coerce_metrics"}:
        continue
    print("FUNCTION", graph.graph.get("function_name"))
    print("OUTPUTS", graph.graph.get("function_outputs"), graph.graph.get("return_slot_values"))
    for node_id, data in sorted(graph.nodes(data=True)):
        expression = data.get("expr_obj")
        try:
            rendered = ast.unparse(expression) if isinstance(expression, ast.AST) else repr(expression)
        except Exception:
            rendered = repr(expression)
        if "coerce_metrics" in rendered or "pub_tau" in rendered or "shape" in rendered or data.get("type") == "PlanCall":
            print(node_id, data)
