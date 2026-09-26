from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


SOURCE = """
import torch

def solve_two(matrix: torch.Tensor, rhs: torch.Tensor):
    return torch.linalg.solve(matrix, rhs)
"""

root = Path(__file__).resolve().parents[2]
contract = ExtractionContract(
    root / "extraction_contracts" / "program_extraction.yaml"
).with_program_abi({
    "records": {},
    "bindings": [],
    "values": [
        {
            "function": "solve_two", "parameter": "matrix",
            "storage": "span", "dtype": "float64", "rank": 2,
            "shape": [2, 2], "python_type": "AbstractTensor",
        },
        {
            "function": "solve_two", "parameter": "rhs",
            "storage": "span", "dtype": "float64", "rank": 1,
            "shape": [2], "python_type": "AbstractTensor",
        },
    ],
})

graphs = []
module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE,
    "solve_two",
    name="abstract_solve_shape_trace",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    resolved_process_graph_sink=graphs.append,
    progress=lambda _message: None,
)
qualified = "abstract_solve_shape_trace__solve_two"
function = module.functions[qualified]
print("PARAMETERS", function.metadata.get("parameter_names"))
print("ROOT_ARGS", [
    (value.id, value.dtype, value.shape, value.accounting)
    for value in function.args if value.id in {0, 1, 2}
])
print("OUTPUTS", [
    (value.id, value.dtype, value.shape, value.accounting)
    for value in outputs[qualified]
])
print("ROOT_METADATA", {
    key: value for key, value in function.metadata.items()
    if "shape" in key or "parameter" in key or "output" in key
})
for deployment in graphs:
    for entry in deployment.function_table:
        function_graph = getattr(getattr(entry, "graph", None), "G", None)
        if function_graph is None or entry.name not in {
            "_lu_decompose_inplace", "_masked_pivot_rows",
            "_forward_substitute", "_back_substitute",
        }:
            continue
        print("FUNCTION_GRAPH", entry.name)
        for node_id, data in function_graph.nodes(data=True):
            operation = str(data.get("op") or data.get("type") or "").casefold()
            if int(data.get("value_id", node_id)) not in {
                0, 1, 106, 142, 188, 190, 192, 193, 199, 202,
            } and not (
                entry.name in {"_forward_substitute", "_back_substitute"}
                and operation in {
                    "indexed", "mul", "mult", "sum", "unsqueeze", "reshape",
                }
            ):
                continue
            print("  NODE", node_id, {
                key: value for key, value in data.items()
                if key not in {"expr_obj", "children"}
            })
for block_name, block in function.blocks.items():
    for instruction in block.instrs:
        if instruction.attributes.get("callee") == "turing_validation_error":
            print(
                "ROOT_INSTRUCTION", block_name, instruction.op,
                [(arg.id, arg.dtype, arg.shape, arg.accounting) for arg in instruction.args[:4]],
                None if instruction.res is None else (
                    instruction.res.id, instruction.res.dtype,
                    instruction.res.shape, instruction.res.accounting,
                ),
                instruction.attributes,
            )
solve_qualified = "abstract_solve_shape_trace__solve__specialized_c6467d804af4"
solve = module.functions[solve_qualified]
print("SOLVE_ARGS", [
    (value.id, value.dtype, value.shape, value.accounting)
    for value in solve.args if value.id in {0, 1}
])
for block_name, block in solve.blocks.items():
    for instruction in block.instrs:
        if (
            instruction.attributes.get("callee") == "turing_validation_error"
            or (
                instruction.op == "Call"
                and instruction.attributes.get("region_index") in {0, 1, 8}
            )
        ):
            print(
                "SOLVE_INSTRUCTION", block_name, instruction.op,
                [(arg.id, arg.dtype, arg.shape, arg.accounting) for arg in instruction.args],
                None if instruction.res is None else (
                    instruction.res.id, instruction.res.dtype,
                    instruction.res.shape, instruction.res.accounting,
                ),
                instruction.attributes,
            )
for name, child in module.functions.items():
    if name in {
        solve_qualified + "__planned_region_0",
        solve_qualified + "__planned_region_1",
        solve_qualified + "__planned_region_8",
    }:
        print("REGION", name, [
            (value.id, value.dtype, value.shape, value.accounting)
            for value in child.args
        ])
        for block_name, block in child.blocks.items():
            for instruction in block.instrs:
                print(
                    "  ", block_name, instruction.op,
                    [(arg.id, arg.dtype, arg.shape) for arg in instruction.args],
                    None if instruction.res is None else (
                        instruction.res.id, instruction.res.dtype,
                        instruction.res.shape,
                    ),
                    instruction.attributes,
                )
for graph in graphs:
    selected_graphs = [
        (entry.name, entry.graph)
        for entry in graph.function_table
        if entry.name == "solve" and getattr(entry.graph, "G", None) is not None
    ]
    for graph_name, function_graph in selected_graphs:
      for node, attrs in function_graph.G.nodes(data=True):
        if attrs.get("type") not in {"get_shape", "dim"}:
            continue
        clean = {
            key: value for key, value in attrs.items()
            if key not in {"expr_obj", "parents", "children"}
        }
        print("GRAPH_ACCESSOR", graph_name, node, clean)
        for parent, role in attrs.get("parents", ()):
            parent_attrs = function_graph.G.nodes[parent]
            print("  PARENT", parent, role, {
                key: value for key, value in parent_attrs.items()
                if key not in {"expr_obj", "parents", "children"}
            })
        for child, role in attrs.get("children", ()):
            child_attrs = function_graph.G.nodes[child]
            print("  CHILD", child, role, {
                key: value for key, value in child_attrs.items()
                if key not in {"expr_obj", "parents", "children"}
            })
