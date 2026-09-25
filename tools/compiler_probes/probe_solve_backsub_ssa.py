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
    "records": {}, "bindings": [],
    "values": [
        {"function": "solve_two", "parameter": "matrix", "storage": "span",
         "dtype": "float64", "rank": 2, "shape": [2, 2],
         "python_type": "AbstractTensor"},
        {"function": "solve_two", "parameter": "rhs", "storage": "span",
         "dtype": "float64", "rank": 1, "shape": [2],
         "python_type": "AbstractTensor"},
    ],
})

module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE, "solve_two", name="solve_backsub",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda _message: None,
)
for name, function in module.functions.items():
    if "_back_substitute" not in name:
        continue
    print("=" * 20, name, flush=True)
    print("ARGS", [(v.id, v.dtype, v.shape) for v in function.args], flush=True)
    for block_name, block in function.blocks.items():
        print(" BLOCK", block_name, flush=True)
        for instruction in block.instrs:
            print("   ", instruction, flush=True)
