from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import llvm_dt_system as lds
from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.loop_composer import LoopBackendCapabilities, LoopComposer


piece = lds.LLVMPiece.load(
    ROOT / "artifacts/llvm_pieces/voxel_air_step/b64/voxel_air_step.piece"
)
bindings = lds.bind_pieces([piece])
source = inspect.getsource(lds) + "\n\n" + lds.generated_source([piece])
captured = []
lower_ast_source_to_ssa(
    source,
    "dt_system_over",
    python_bindings={"AbstractTensor": AbstractTensor, **bindings},
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    runtime_closure_only=True,
    name="llvm_dt_system_plan_inspection",
    extraction_contract=lds.dt_system_contract(
        "dt_system_over", lds.column_names_of([piece]), piece.batch, 1
    ),
    resolved_process_graph_sink=captured.append,
    stop_after_compilation_unit_plan=True,
)

for entry in captured[0].function_table:
    graph = getattr(entry, "graph", None)
    if getattr(graph, "G", None) is None:
        continue
    name = str(graph.G.graph.get("function_name", ""))
    if ("step_with_dt_control_used" not in name and "run_superstep" not in name
            and "_apply_energy_sidechain" not in name
            and "advance_pieces" not in name):
        continue
    print(f"FUNCTION {name} qualified={getattr(entry, 'qualified_name', None)}")
    print(f"SPECIALIZATIONS {graph.G.graph.get('planner_specializations')!r}")
    if "step_with_dt_control_used" in name:
        composer = LoopComposer(LoopBackendCapabilities(backend="c", native_while=True))
        for loop_id, loop_data in graph.G.nodes(data=True):
            if isinstance(loop_data.get("expr_obj"), ast.While):
                description = composer.describe(graph, int(loop_id))
                print(f"LOOP {loop_id} return_controls={description.return_controls!r}")
    for node_id, data in sorted(graph.G.nodes(data=True)):
        if not ((250 <= int(node_id) <= 405) or (40 <= int(node_id) <= 60)):
            continue
        expression = data.get("expr_obj")
        try:
            rendered = ast.unparse(expression) if isinstance(expression, ast.AST) else repr(expression)
        except Exception:
            rendered = repr(expression)
        print(
            f"NODE {node_id} type={data.get('type')!r} op={data.get('op')!r} "
            f"line={getattr(expression, 'lineno', None)!r} source={rendered!r} "
            f"parents={data.get('parents')!r} attrs={data.get('attributes')!r}"
        )


