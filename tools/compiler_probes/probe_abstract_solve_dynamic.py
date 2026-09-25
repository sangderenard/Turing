from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm


SOURCE = """
import torch

def solve_dynamic(matrix: torch.Tensor, rhs: torch.Tensor):
    return torch.linalg.solve(matrix, rhs)
"""

root = Path(__file__).resolve().parents[2]
contract = ExtractionContract(
    root / "extraction_contracts" / "program_extraction.yaml"
)

module, _outputs, _exports = lower_ast_source_to_ssa(
    SOURCE,
    "solve_dynamic",
    name="abstract_solve_dynamic",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda _message: None,
)

for function_name, function in module.functions.items():
    if any(fragment in function_name for fragment in (
        "linalg__solve", "forward_substitute", "back_substitute",
        "reshape_dispatch", "solve_dynamic",
    )):
        print("FUNCTION", function_name)
        for value in function.args:
            print("ARG", value.id, value.dtype, value.shape, value.accounting)

artifact = emit_ssa_function_to_llvm(
    module,
    "abstract_solve_dynamic__solve_dynamic",
    entry_name="abstract_solve_dynamic_solve_dynamic",
)
print("LLVM_SHORTFALLS", artifact.shortfalls)
