"""The two lines that emit nothing.

    hot.cumsum(dim=-1)                  -> MATCH   [1. 1.]
    (hot == 1).to_dtype(dtype)          -> MATCH   [1. 0.]
    hot.cumsum(dim=-1).to_dtype(dtype)  -> the output buffer is never written

cumsum works.  to_dtype works.  to_dtype applied to a cumsum result emits no
store at all, and everything above it in the solver -- _first_occurrence,
_pivot_mask, _masked_pivot_rows, lu_U -- reads the buffer it left
uninitialized.  Lowers to SSA only, so it answers in seconds.
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
from src.compiler.identity_concordance import identity_book

WORKS = """
from src.common.tensors.linalg import _one_hot_axis

def stage(matrix):
    hot = _one_hot_axis(matrix, 2, 0)
    return (hot == 1).to_dtype(matrix.get_dtype())
"""

BROKEN = """
from src.common.tensors.linalg import _one_hot_axis

def stage(matrix):
    hot = _one_hot_axis(matrix, 2, 0)
    return hot.cumsum(dim=-1).to_dtype(matrix.get_dtype())
"""

root = Path(__file__).resolve().parents[2]


def lower(label, source):
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
        source, "stage", name=f"cc_{label}",
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _m: None,
    )
    print(f"########## {label}", flush=True)
    qualified = f"cc_{label}__stage"
    published = [int(v.id) for v in outputs[qualified]]
    print("published:", published, flush=True)
    for name, function in module.functions.items():
        if "_one_hot_axis" in name or "_axis_index" in name:
            continue
        print(f"== {name.split('__', 1)[-1]}"
              f" formals={[int(a.id) for a in getattr(function, 'args', ())]}",
              flush=True)
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                result = instruction.res
                print(
                    f"  {block_name}#{index} {instruction.op:14s}"
                    f" res={getattr(result, 'id', None)}"
                    f" shape={tuple(getattr(result, 'shape', None) or ())}"
                    f" args={[(int(getattr(a, 'id', -1)), tuple(getattr(a, 'shape', None) or ())) for a in instruction.args]}"
                    f" callee={instruction.attributes.get('callee')}"
                    f" out={instruction.attributes.get('output_ids')}",
                    flush=True,
                )
    book = identity_book(module)
    page = book.page("proven_shape")
    print("proven:", {
        str(row): page.latest(row) for row in sorted(page.rows(), key=str)
    }, flush=True)


EQ = """
from src.common.tensors.linalg import _one_hot_axis

def stage(matrix):
    hot = _one_hot_axis(matrix, 2, 0)
    return (hot.cumsum(dim=-1) == 1).to_dtype(matrix.get_dtype())
"""

lower("eq", EQ)
lower("works", WORKS)
lower("broken", BROKEN)
