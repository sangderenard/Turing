"""The smallest program that still gets it wrong.

    index.unsqueeze(-1) == index.unsqueeze(-2)

is (2,1) against (1,2), which broadcasts to (2,2).  Compiled, it writes two
elements and leaves the other two as whatever the allocation held -- the
denormals in ``identity_seed: [1.0, 1.0, 6.95e-310, 6.95e-310]``.  Eagerly it
is the identity matrix.

Lowers to SSA only, so it answers in seconds, and asks the concordance what
shape each step settled to rather than keeping a private tally.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    authored_function_name, identity_book, shape_store_report,
)

SOURCE = """
from src.common.tensors.linalg import _axis_index

def stage(matrix):
    index = _axis_index(matrix, 2)
    return (index.unsqueeze(-1) == index.unsqueeze(-2)).to_dtype(
        matrix.get_dtype()
    )
"""

MATRIX = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
root = Path(__file__).resolve().parents[2]

# Eager truth first: whatever the compiler does, this is the answer.
from src.common.tensors.abstraction import AbstractTensor

eager_index = AbstractTensor.arange(2).to_dtype("float64")
eager = (
    eager_index.unsqueeze(-1) == eager_index.unsqueeze(-2)
).to_dtype("float64")
print("EAGER:", np.asarray(eager.tolist()).reshape(-1), flush=True)

contract = ExtractionContract(
    root / "extraction_contracts" / "program_extraction.yaml"
).with_program_abi({
    "records": {}, "bindings": [],
    "values": [{
        "function": "stage", "parameter": "matrix", "storage": "span",
        "dtype": "float64", "rank": 2, "shape": [2, 2],
        "python_type": "AbstractTensor",
    }],
})

module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE, "stage", name="seed",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda _m: None,
)

book = identity_book(module)
print(shape_store_report(book), flush=True)

for name, function in module.functions.items():
    short = name.split("__")[-1]
    if authored_function_name(name) not in {"stage", "_axis_index"}:
        continue
    print(f"== {short}", flush=True)
    for block_name, block in function.blocks.items():
        for index, instruction in enumerate(block.instrs):
            result = instruction.res
            print(
                f"  {block_name}#{index} {instruction.op:14s}"
                f" res={getattr(result, 'id', None)}"
                f" shape={getattr(result, 'shape', None)}"
                f" args={[(int(getattr(a, 'id', -1)), tuple(getattr(a, 'shape', None) or ())) for a in instruction.args]}"
                f" callee={instruction.attributes.get('callee')}"
                f" feed_shapes={instruction.attributes.get('feed_shapes')}",
                flush=True,
            )

print("PROVEN SHAPES:", flush=True)
page = book.page("proven_shape")
for row in sorted(page.rows(), key=str):
    print(f"  {row}: {page.latest(row)}", flush=True)
