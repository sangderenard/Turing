from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.common.tensors.linalg import solve as abstract_solve
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm


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

resolved = []
module, _outputs, _exports = lower_ast_source_to_ssa(
    SOURCE,
    "solve_two",
    name="abstract_solve_probe",
    extraction_contract=contract,
    tensor_code_references={"solve": abstract_solve},
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    resolved_process_graph_sink=resolved.append,
    progress=lambda _message: None,
)

print("FUNCTIONS", tuple(module.functions))
for function_name, function in module.functions.items():
    print("SSA_FUNCTION", function_name, tuple((value.id, value.shape, value.dtype) for value in function.args))
    for block_name, block in function.blocks.items():
        for instruction in block.instrs:
            print("SSA", block_name, instruction.op, tuple(value.id for value in instruction.args), None if instruction.res is None else instruction.res.id, instruction.attributes)
for entry in resolved[0].function_table:
    graph = getattr(getattr(entry, "graph", None), "G", None)
    if graph is None:
        continue
    print("GRAPH_FUNCTION", entry.name)
    for node_id, data in graph.nodes(data=True):
        attributes = data.get("attributes") or {}
        if (
            attributes.get("tensor_candidate") == "solve"
            or attributes.get("tensor") == "solve"
            or attributes.get("callee_ref") is not None
        ):
            print(
                "NODE", node_id, data.get("type"), data.get("op"),
                attributes,
            )

root_name = "abstract_solve_probe__solve_two"
artifact = emit_ssa_function_to_llvm(
    module, root_name, entry_name="abstract_solve_probe_solve_two",
)
print("LLVM_SHORTFALLS", artifact.shortfalls)
print("LLVM_IR", artifact.llvm_ir)
