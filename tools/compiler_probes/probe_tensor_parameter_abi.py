from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.extraction_contract import ExtractionContract

root_path = Path(__file__).resolve().parents[2]

module, _outputs, _exports = lower_ast_source_to_ssa(
    "import torch\n\n"
    "def root(left: torch.Tensor, right: torch.Tensor):\n"
    "    return left + right\n",
    "root",
    name="tensor_parameter_abi",
    extraction_contract=ExtractionContract(
        root_path / "extraction_contracts" / "program_extraction.yaml"
    ),
)
root = module.functions["tensor_parameter_abi__root"]
for value in root.args[:2]:
    print(value.id, value.dtype, value.shape, value.accounting)
