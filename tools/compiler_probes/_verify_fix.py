import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

SOURCE = """
import torch

def solve_two(matrix: torch.Tensor, rhs: torch.Tensor):
    return torch.linalg.solve(matrix, rhs)
"""
MATRIX = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
RHS = np.asarray([1.0, 2.0], dtype=np.float64)
root = Path.cwd()
contract = ExtractionContract(root / "extraction_contracts" / "program_extraction.yaml").with_program_abi({
    "records": {}, "bindings": [],
    "values": [
        {"function": "solve_two", "parameter": "matrix", "storage": "span", "dtype": "float64", "rank": 2, "shape": list(MATRIX.shape), "python_type": "AbstractTensor"},
        {"function": "solve_two", "parameter": "rhs", "storage": "span", "dtype": "float64", "rank": 1, "shape": list(RHS.shape), "python_type": "AbstractTensor"},
    ],
})
module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE, "solve_two", name="abstract_solve_numeric",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda m: None,
)

for name in module.functions:
    if "_masked_pivot_rows__specialized_5171c8115542" in name and "planned_region" not in name:
        fn = module.functions[name]
        print(f"=== {name} ===")
        print("FORMALS[:3]:", [(int(a.id), a.dtype, a.shape, dict(a.accounting or {})) for a in fn.args[:3]])
