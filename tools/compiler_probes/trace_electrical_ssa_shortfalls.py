from pathlib import Path
import ast
import contextlib
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler import fortran_c_shell as shell_module
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
OUT = sys.__stdout__
original_fields = shell_module._field_slot_ops
original_overlay = deployment._overlay_control_or_require_subdivision
seen_fields = set()
seen_overlay = False


def source(graph, node_id):
    if int(node_id) not in graph.G:
        return None
    expression = graph.G.nodes[int(node_id)].get("expr_obj")
    return ast.unparse(expression) if isinstance(expression, ast.AST) else None


def traced_fields(graph_obj, *args, **kwargs):
    result = original_fields(graph_obj, *args, **kwargs)
    function = graph_obj.graph.get("function_name")
    owner = graph_obj.graph.get("method_owner")
    signature = (owner, function, tuple(result[8]), tuple(result[6]))
    if function in {"set_thermal_temperatures", "__init__"} and signature not in seen_fields:
        seen_fields.add(signature)
        print("FIELDS", owner, function, {
            "declarations": result[8],
            "initializations": result[6],
            "identities": graph_obj.graph.get("identity_table"),
        }, file=OUT, flush=True)
        for node_id, data in graph_obj.nodes(data=True):
            value_id = int(data.get("value_id", node_id))
            if (function == "set_thermal_temperatures" and value_id == 3) or (
                function == "__init__" and value_id in {438, 440, 447}
            ):
                print("FIELD_NODE", owner, function, node_id, {
                    "value_id": value_id,
                    "type": data.get("type"),
                    "source": source(type("G", (), {"G": graph_obj})(), node_id),
                    "parents": data.get("parents"),
                    "attributes": data.get("attributes"),
                    "span": data.get("source_span"),
                }, file=OUT, flush=True)
    return result


def traced_overlay(graph, runtime_regions, reductions, loop_controls,
                   conditional_controls, nesting, *, region_dependencies=()):
    global seen_overlay
    if (not seen_overlay and graph.G.graph.get("method_owner") == "ComplexTensorCircuit"
            and graph.G.graph.get("function_name") == "__init__"):
        seen_overlay = True
        for index, (reduction, control) in enumerate(zip(
            reductions, loop_controls, strict=True
        )):
            if not (
                any(440 in pair or 447 in pair for pair in control.value_aliases)
                or any(
                    mutation.effect_node_id == 438
                    for mutation in getattr(control.root, "sequence_mutations", ())
                )
                or int(reduction.loop_node_id) in {213, 390, 391, 438}
            ):
                continue
            print("CONTROL", index, {
                "loop_node": int(reduction.loop_node_id),
                "source": source(graph, reduction.loop_node_id),
                "regions": tuple(control.region_indices),
                "value_aliases": tuple(control.value_aliases),
                "root": repr(control.root),
            }, file=OUT, flush=True)
    return original_overlay(
        graph, runtime_regions, reductions, loop_controls,
        conditional_controls, nesting,
        region_dependencies=region_dependencies,
    )


shell_module._field_slot_ops = traced_fields
deployment._overlay_control_or_require_subdivision = traced_overlay
try:
    with open(os.devnull, "w") as quiet, \
            contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        try:
            lower_ast_source_to_ssa(
                source_path.read_text(encoding="utf-8"),
                "ComplexElectricalEngine.step",
                name="complex_electrical_engine_shortfall_trace",
                extraction_contract=contract,
                runtime_closure_only=True,
            )
        except Exception as error:
            print("FINAL", type(error).__name__, str(error), file=OUT, flush=True)
finally:
    shell_module._field_slot_ops = original_fields
    deployment._overlay_control_or_require_subdivision = original_overlay
