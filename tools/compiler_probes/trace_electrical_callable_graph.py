from pathlib import Path
import ast
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler import glsl_deployment_strategy as deployment

root = Path(__file__).resolve().parents[3]
turing = root / "turing"
source_path = root / "spectral-analyzer" / "electrical_dt_engine.py"
sys.path.insert(0, str(source_path.parent))
contract = ExtractionContract(
    turing / "extraction_contracts" / "program_extraction.yaml"
).with_sources([
    ("electrical_dt_engine", source_path),
    ("electrical_tensor_network", root / "spectral-analyzer" / "electrical_tensor_network.py"),
    ("engine_toy.dc_power", turing / "engine_toy" / "dc_power.py"),
])

original = deployment.ProcessGraphGLSLDeployment.prepare_graph_precompile


def traced(self, *args, **kwargs):
    for shell in deployment._walk_planned_shells(
        self,
        include_function_registry=bool(getattr(
            self, "prepare_complete_catalogue", False,
        )),
    ):
        graph = shell.process_graph.G
        if graph.graph.get("function_name") == "__init__":
            print(
                "INIT_SHELL",
                graph.graph.get("method_owner"),
                len(graph.nodes),
                flush=True,
            )
        if graph.graph.get("function_name") != "advance":
            continue
        print("ADVANCE_METADATA", {
            key: value for key, value in graph.graph.items()
            if key in {
                "method_owner", "program_abi", "parameter_record_abi",
                "class_field_mapping_contracts", "identity_table",
            }
        }, flush=True)
        for node_id, data in graph.nodes(data=True):
            expression = data.get("expr_obj")
            source = ast.unparse(expression) if isinstance(expression, ast.AST) else ""
            if (
                "advance" in source or "port_laws" in source
                or (data.get("attributes") or {}).get("binding_name") == "law"
            ):
                print("ADVANCE_NODE", node_id, data.get("type"), source,
                      data.get("parents"), data.get("attributes"), flush=True)
    return original(self, *args, **kwargs)


deployment.ProcessGraphGLSLDeployment.prepare_graph_precompile = traced
try:
    lower_ast_source_to_ssa(
        source_path.read_text(encoding="utf-8"),
        "ComplexElectricalEngine.step",
        name="complex_electrical_engine_trace",
        extraction_contract=contract,
        runtime_closure_only=True,
    )
finally:
    deployment.ProcessGraphGLSLDeployment.prepare_graph_precompile = original
