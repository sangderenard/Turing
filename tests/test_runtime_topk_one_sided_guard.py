"""A control-owned lower k bound uses the runtime input axis as capacity."""

from pathlib import Path

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    GUARDED_CALL_ACTUAL_RANGE, SOURCE_SPAN, TENSOR_RUNTIME_TOPK_SELECTION,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, EDGE_PAGE, identity_book,
)
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_runtime_topk_uses_exact_one_sided_callsite_guard(tmp_path, monkeypatch):
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({"records": {}, "bindings": [], "values": [
        {"function": "root", "parameter": "values", "storage": "span",
         "dtype": "int64", "rank": 1,
         "python_type": "src.common.tensors.abstraction.AbstractTensor"},
        {"function": "root", "parameter": "k", "storage": "scalar",
         "dtype": "int64", "rank": 0, "python_type": "builtins.int"},
    ]})
    module, _, exports = lower_ast_source_to_ssa(
        "def root(values: AbstractTensor, k: int):\n"
        "    if k > 1:\n"
        "        return AbstractTensor.topk(values, k=k, dim=0).indices.sum().item()\n"
        "    return 0\n",
        "root", name="runtime_topk_one_sided_guard",
        python_bindings={"AbstractTensor": AbstractTensor},
        extraction_contract=contract,
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_TOPK_SELECTION)
    rows = tuple(page.rows())
    assert len(rows) == 2
    for row in rows:
        fact = page.latest(row)
        guard = fact.nonnegative_cell
        assert guard.page == GUARDED_CALL_ACTUAL_RANGE
        assert book.page(GUARDED_CALL_ACTUAL_RANGE).cell(
            guard.row, guard.column) == (2, None)
    assert CorrelationTable.build(module).findings(module) == []

    guard = page.latest(rows[0]).nonnegative_cell
    bounds_page = book.page(GUARDED_CALL_ACTUAL_RANGE)
    original = bounds_page.cell(guard.row, guard.column)
    for forged in ((None, None), (None, 3), (-1, None), (2, 1)):
        bounds_page.cells[(guard.row, guard.column)] = forged
        assert any(f.kind in {"guarded-call-actual-range",
                              "tensor-runtime-topk-selection-disagreement"}
                   for f in CorrelationTable.build(module).findings(module))
    bounds_page.cells[(guard.row, guard.column)] = original

    guard_spans = {source for source, _ in book.edges_into(guard)
                   if source.page == SOURCE_SPAN}
    assert guard_spans
    edges = book.page(EDGE_PAGE)
    original_rows = edges.scope_rows
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original_rows(scope)
        if not (scope == guard.key and
                edge[1] in {source.key for source in guard_spans})
    ))
    assert any(f.kind in {"guarded-call-actual-range",
                          "intrinsic-guarded-actual-origin"}
               for f in CorrelationTable.build(module).findings(module))
    monkeypatch.setattr(edges, "scope_rows", original_rows)

    artifact = emit_ssa_module_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "runtime_topk_guard")
    root = module.functions[exports[0]]
    ports = dict(root.metadata["parameter_names"])
    result_id = next(int(value.id) for block in root.blocks.values()
                     for instruction in block.instrs if instruction.op == "Ret"
                     for value in instruction.args)
    cases = (
        (np.array([4, 1, 3], dtype=np.int64), 2, 2),
        (np.array([4, 1, 3], dtype=np.int64), 4, 3),
        (np.array([4, 1, 3], dtype=np.int64), (1 << 63) - 1, 3),
        (np.empty(0, dtype=np.int64), 2, 0),
        (np.array([4, 1, 3], dtype=np.int64), 1, 0),
    )
    for values, k, expected in cases:
        buffers = artifact.prepare_execution({
            int(ports["values"]): values,
            int(ports["k"]): np.array([k], dtype=np.int64),
        }).run().buffers
        assert int(np.asarray(buffers[result_id]).item()) == expected
