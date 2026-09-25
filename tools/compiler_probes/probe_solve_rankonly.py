"""Compile `torch.linalg.solve` on a 2x2 contract and check it against NumPy.

Same entry and same discipline as `tests/test_native_shaped_view_lowering.py`,
on the LLVM lane instead of the C lane: `lower_ast_source_to_ssa` for the
program, `-O0` for the build, and the executed result compared against the
eager reference.  A shape or identity mistake in this path does not raise --
it produces a complete, compiling program that reads the wrong elements -- so
value comparison is the only honest gate.

Outputs are filled with NaN before the run, as that harness does.  A buffer
the program never writes then reads as NaN instead of as 0.0, which is a
plausible number and has already cost a wrong diagnosis once.
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
    concordance_report,
    identity_book,
    render_row,
)
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

MATRIX = np.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=np.float64)
RHS = np.asarray([1.0, 2.0], dtype=np.float64)

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
            "python_type": "AbstractTensor",
        },
        {
            "function": "solve_two", "parameter": "rhs",
            "storage": "span", "dtype": "float64", "rank": 1,
            "python_type": "AbstractTensor",
        },
    ],
})

module, outputs, _exports = lower_ast_source_to_ssa(
    SOURCE,
    "solve_two",
    name="abstract_solve_rankonly",
    extraction_contract=contract,
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda message: print("PROGRESS", message, flush=True),
)

# The concordance is the backbone diagnostic; read it before anything else.
print(concordance_report(module, limit=8), flush=True)
book = identity_book(module)
for page_name, page in sorted(book.pages.items()):
    oscillating = page.oscillating_rows()
    if oscillating:
        print(f"OSCILLATING {page_name}: {len(oscillating)} row(s)", flush=True)
        for row, runs in list(oscillating.items())[:5]:
            print("   ", render_row(row),
                  [fact for _start, _end, fact in runs][:6], flush=True)
    if page_name in {"value_shape", "call_edge"}:
        print(f"PAGE {page_name}: {len(page.rows())} row(s)", flush=True)

# What the three shortfalls actually need: does each helper's formal have a
# call edge at all, and did a shape ever land on it?
shapes = book.page("value_shape")
edges = book.page("call_edge")
# Every value the descriptor query answered with a bare scalar while its
# operation is one that produces a tensor.  A wrong shape anywhere in the
# program appears here with the operation that produced it.
descriptors = book.page("proven_shape")
SCALAR_PRODUCERS = {
    "", "const", "constant", "input", "getattr", "dim", "ndim", "ndims",
    "numel", "len", "int", "float", "range", "add", "sub", "mul", "div",
    "truediv", "floordiv", "mod", "lt", "le", "gt", "ge", "eq", "ne",
}
print("SUSPECT SCALARS (tensor-producing operation, no extents):", flush=True)
for row in sorted(descriptors.rows(), key=str):
    if str(row[0]) == "_lu_decompose_inplace" and int(row[1]) in (
        3, 11, 15, 20, 22, 101, 107
    ):
        print("    PROVEN", row, descriptors.latest(row), flush=True)
for name in ("formal_literal", "formal_shape"):
    page = book.page(name)
    print(f"    {name}: {len(page.rows())} row(s)", flush=True)
    for row in sorted(page.rows(), key=str):
        print("      ", row, page.latest(row), flush=True)
conflicting = [
    row for row in descriptors.rows()
    if (descriptors.latest(row) or ("",))[0] == "conflicting"
]
print(f"   proven={len(descriptors.rows())} conflicting={len(conflicting)}",
      flush=True)
for row in sorted(conflicting, key=str)[:10]:
    print("    CONFLICT", render_row(row),
          [f for _c, f in descriptors.history(row)][:4], flush=True)

incoming = {}
for row in edges.rows():
    incoming.setdefault((row[0], row[1]), []).append((row[2], row[3]))
for row in sorted(set(shapes.rows()) | set(incoming)):
    history = [fact for _column, fact in shapes.history(row)]
    settled = history[-1] if history else None
    # Say so loudly when a value that callers feed never settled a shape.
    unsettled = settled is None or settled[0] == "polymorphic"
    print("   ", "UNSETTLED" if unsettled else "settled  ",
          render_row(row),
          "from", incoming.get(row, []),
          "|", history, flush=True)

qualified = "abstract_solve_rankonly__solve_two"
function = module.functions[qualified]
artifact = emit_ssa_function_to_llvm(
    module, qualified, entry_name="abstract_solve_rankonly_solve_two",
)
print("LLVM_SHORTFALLS", artifact.shortfalls, flush=True)
if artifact.shortfalls:
    raise SystemExit(1)
(root / "build" / "abstract_solve_rankonly.ll").write_text(
    artifact.llvm_ir, encoding="utf-8",
)

native = compile_artifact(
    artifact, directory=root / "build" / "abstract_solve_rankonly",
    optimization="O0",
)

expected = np.linalg.solve(MATRIX, RHS)
parameters = dict(function.metadata["parameter_names"])
published = [int(value.id) for value in outputs[qualified]]
feeds = {parameters["matrix"]: MATRIX.copy(), parameters["rhs"]: RHS.copy()}
for value_id in published:
    feeds[value_id] = np.full(np.shape(expected), np.nan)

execution = prepare_artifact_execution(native, feeds).run()
produced = [
    np.asarray(execution.buffers[value_id]).reshape(np.shape(expected))
    for value_id in published
    if value_id in execution.buffers
]
print("EXPECTED", expected, flush=True)
print("PRODUCED", produced, flush=True)
if not any(np.allclose(value, expected) for value in produced):
    unwritten = [value for value in produced if np.all(np.isnan(value))]
    raise AssertionError(
        "compiled solve did not publish the NumPy solution"
        + (" (output never written)" if unwritten else "")
    )
print("NUMERIC_MATCH", flush=True)
