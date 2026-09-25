from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)


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

module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE,
    "solve_two",
    name="abstract_solve_execute",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda _message: None,
)
qualified = "abstract_solve_execute__solve_two"
function = module.functions[qualified]
parameters = dict(function.metadata["parameter_names"])
artifact = emit_ssa_function_to_llvm(
    module, qualified, entry_name="abstract_solve_execute_solve_two",
)
print("LLVM_SHORTFALLS", artifact.shortfalls)
print("PARAMETERS", parameters)
print("OUTPUTS", [(value.id, value.shape, value.dtype) for value in outputs[qualified]])
for shortfall in artifact.shortfalls:
    failed = module.functions[shortfall.function]
    print("SHORTFALL_FUNCTION", shortfall.function)
    for block_name, block in failed.blocks.items():
        for instruction in block.instrs:
            print("  ", block_name, instruction)
(root / "build" / "abstract_solve_execute.ll").write_text(
    artifact.llvm_ir, encoding="utf-8",
)
for function_name, ssa_function in module.functions.items():
    selected = []
    for block_name, block in ssa_function.blocks.items():
        for instruction in block.instrs:
            ids = {
                *(int(value.id) for value in instruction.args),
                *((int(instruction.res.id),) if instruction.res is not None else ()),
            }
            if 193 in ids or 2305843010213695298 in ids:
                selected.append((block_name, instruction))
    if selected:
        print("SSA_IDENTITY", function_name)
        print("FUNCTION_ALIASES", ssa_function.metadata.get("value_aliases"))
        print("FUNCTION_ARGS", [
            (value.id, value.dtype, value.shape, value.accounting)
            for value in ssa_function.args
        ])
        for block_name, instruction in selected:
            print(block_name, instruction)
book = module.metadata["identity_book"]
for page_name, page in sorted(book.pages.items()):
    selected = []
    for row in page.rows():
        if any(str(item).endswith(
            "___lu_decompose_inplace__specialized_60b2b0696fb1"
        ) for item in (row if isinstance(row, tuple) else (row,))):
            history = page.history(row)
            if any(
                int(value) in {124, 185, 193}
                for _column, fact in history
                for value in (
                    fact.values() if isinstance(fact, dict)
                    else fact if isinstance(fact, (tuple, list))
                    else (fact,)
                )
                if isinstance(value, int)
            ) or any(
                isinstance(item, int) and item in {124, 185, 193}
                for item in (row if isinstance(row, tuple) else (row,))
            ):
                selected.append((row, history))
    if selected:
        print("CONCORDANCE", page_name, selected)
if artifact.shortfalls:
    raise SystemExit(1)

native = compile_artifact(artifact, directory=root / "build" / "abstract_solve_execute")
matrix = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
rhs = np.asarray([1.0, 2.0], dtype=np.float64)
feeds = {
    parameters["matrix"]: matrix.copy(),
    parameters["rhs"]: rhs.copy(),
}
for formal in function.args:
    value_id = int(formal.id)
    if value_id in feeds:
        continue
    count = max(1, int(np.prod(tuple(formal.shape or (1,)))))
    feeds[value_id] = np.zeros(count, dtype=np.float64)
execution = prepare_artifact_execution(native, feeds).run()
expected = np.linalg.solve(matrix, rhs)
produced = {
    int(value.id): np.asarray(execution.buffers[int(value.id)]).copy()
    for value in outputs[qualified]
    if int(value.id) in execution.buffers
}
print("EXPECTED", expected)
print("PRODUCED", produced)
if not any(
    values.size >= expected.size
    and np.allclose(values.reshape(-1)[:expected.size], expected)
    for values in produced.values()
):
    raise AssertionError("compiled AbstractTensor solve did not publish NumPy solution")
