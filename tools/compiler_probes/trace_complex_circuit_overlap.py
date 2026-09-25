from pathlib import Path
import ast
import contextlib
import inspect
import os
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


original = deployment._overlay_control_or_require_subdivision
OUT = sys.__stdout__


def source(graph, node_id):
    data = graph.G.nodes[int(node_id)]
    expression = data.get("expr_obj")
    return ast.unparse(expression) if isinstance(expression, ast.AST) else None


def traced(graph, runtime_regions, reductions, loop_controls,
           conditional_controls, nesting, *, region_dependencies=()):
    owner = graph.G.graph.get("method_owner")
    function = graph.G.graph.get("function_name")
    if owner == "ComplexTensorCircuit" and function == "__init__":
        caller = inspect.currentframe().f_back
        target = caller.f_locals.get("target")
        print("CIRCUIT_OVERLAY", {
            "runtime_regions": tuple(runtime_regions),
            "nesting": {int(k): tuple(v) for k, v in nesting.items()},
            "region_dependencies": tuple(
                edge for edge in region_dependencies
                if 25 in edge or 26 in edge or 5 in edge
            ),
        }, flush=True, file=OUT)
        for index, (reduction, control) in enumerate(zip(
            reductions, loop_controls, strict=True
        )):
            if index not in {11, 19}:
                continue
            print("LOOP", index, {
                "node": int(reduction.loop_node_id),
                "source": source(graph, reduction.loop_node_id),
                "regions": tuple(control.region_indices),
                "blockers": tuple(reduction.blockers),
                "root": repr(control.root),
            }, flush=True, file=OUT)
        if target is not None:
            for plan in target.loop_plans:
                if int(plan.loop.node_id) not in {384, 407}:
                    continue
                print("PLAN", int(plan.loop.node_id), {
                    "source": source(graph, plan.loop.node_id),
                    "body_nodes": tuple(map(int, plan.loop.body_nodes)),
                    "condition_nodes": tuple(map(int, plan.loop.condition_nodes)),
                    "iterable_node": plan.loop.iterable_node,
                    "target_bindings": plan.loop.target_bindings,
                    "iteration_outputs": plan.loop.iteration_outputs,
                    "state_effects": plan.loop.state_effects,
                }, flush=True, file=OUT)
            for region_index, subgraph in enumerate(target.dispatch_subgraphs):
                if region_index not in {5, 25, 26}:
                    continue
                members = []
                for node_id, data in subgraph.G.nodes(data=True):
                    members.append({
                        "node": int(node_id),
                        "type": data.get("type"),
                        "source": source(graph, node_id),
                        "parents": tuple(data.get("parents") or ()),
                        "value_id": data.get("value_id"),
                        "span": data.get("source_span"),
                        "attributes": data.get("attributes"),
                    })
                print("REGION", region_index, members, flush=True, file=OUT)
    return original(
        graph, runtime_regions, reductions, loop_controls,
        conditional_controls, nesting,
        region_dependencies=region_dependencies,
    )


deployment._overlay_control_or_require_subdivision = traced
try:
    with open(os.devnull, "w") as quiet, \
            contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        try:
            lower_ast_source_to_ssa(
                source_path.read_text(encoding="utf-8"),
                "ComplexElectricalEngine.step",
                name="complex_electrical_engine_overlap_trace",
                extraction_contract=contract,
                runtime_closure_only=True,
            )
        except ValueError as error:
            print("EXPECTED", str(error).split("; function=")[0], file=OUT)
finally:
    deployment._overlay_control_or_require_subdivision = original
