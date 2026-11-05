"""A broadcast's pruned static count retains its exact typed literal."""

from pathlib import Path

import pytest
import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    SSA_CONSTANT_MATERIALIZATION, SSA_VALUE, TENSOR_RUNTIME_BROADCAST_VALIDITY,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, EDGE_PAGE, identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


@pytest.fixture(scope="module")
def broadcast_product():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1] / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "bits", "storage": "span",
                    "dtype": "int64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    module, _, exports = lower_ast_source_to_ssa(
        "def root(bits: AbstractTensor):\n"
        "    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))\n"
        "    shaped = bits.reshape(-1, 8).to_dtype('float64')\n"
        "    return (shaped * weights.unsqueeze(0)).sum(dim=1).to_dtype('int64').sum().item()\n",
        "root", name="dead_tensor_geometry_audit",
        extraction_contract=contract, python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    row, = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY).rows()
    fact = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY).latest(row)
    function = module.functions[row[1]]
    count_id = fact[10]
    assert not any(i.res is not None and i.res.id == count_id
                   for block in function.blocks.values() for i in block.instrs)
    page = book.page(SSA_CONSTANT_MATERIALIZATION)
    receipt_row, = (r for r in page.scope_rows(row[0])
                   if r[1] == function.name and r[3] == count_id)
    receipt = book.latest_ref(SSA_CONSTANT_MATERIALIZATION, receipt_row)
    assert page.latest(receipt_row) == ("Const", "int32", "8")
    assert CorrelationTable.build(module).findings(module) == []
    return module, book, row, receipt, exports[0]


def test_pruned_broadcast_count_keeps_typed_literal(broadcast_product):
    module, _, _, _, _ = broadcast_product
    assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module) == []


@pytest.mark.parametrize("dtype,literal", (("int32", "9"), ("int64", "8"),
                                           ("int32", str(1 << 40))))
def test_pruned_count_rejects_forged_literal(broadcast_product, dtype, literal):
    module, book, _, ref, _ = broadcast_product
    page = book.page(SSA_CONSTANT_MATERIALIZATION)
    original = page.cell(ref.row, ref.column)
    page.cells[(ref.row, ref.column)] = ("Const", dtype, literal)
    try:
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
        if dtype != "int32" or int(literal) >= (1 << 31):
            assert CorrelationTable._constant_materialization_findings(module)
    finally:
        page.cells[(ref.row, ref.column)] = original


def test_pruned_count_requires_exact_value_cause(broadcast_product, monkeypatch):
    module, book, row, ref, _ = broadcast_product
    value = book.latest_ref(SSA_VALUE, (row[0], ref.row[3]))
    edges = book.page(EDGE_PAGE)
    original = edges.scope_rows
    monkeypatch.setattr(edges, "scope_rows", lambda scope: tuple(
        edge for edge in original(scope)
        if not (scope == ref.key and edge[1] == value.key)
    ))
    assert CorrelationTable._constant_materialization_findings(module)
    assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)


def test_pruned_count_requires_literal_receipt(broadcast_product, monkeypatch):
    module, book, _, ref, _ = broadcast_product
    page = book.page(SSA_CONSTANT_MATERIALIZATION)
    original = page.scope_rows
    monkeypatch.setattr(page, "scope_rows", lambda scope: tuple(
        row for row in original(scope) if row != ref.row
    ))
    assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)


def test_weighted_reshape_broadcast_native_matches_byte_weights(broadcast_product, tmp_path):
    module, _, _, _, entrypoint = broadcast_product
    artifact = emit_ssa_module_to_c(module, entrypoint)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "weighted_reshape_broadcast")
    root = module.functions[entrypoint]
    source_id = dict(root.metadata["parameter_names"])["bits"]
    output_id, = (value.id for block in root.blocks.values()
                  for instruction in block.instrs if instruction.op == "Ret"
                  for value in instruction.args)
    weights = np.asarray((128, 64, 32, 16, 8, 4, 2, 1), dtype=np.int64)
    for size in (0, 8, 16, 2048):
        bits = (np.arange(size, dtype=np.int64) // 3) % 2
        expected = (bits.reshape(-1, 8) * weights).sum(dtype=np.int64)
        actual = np.asarray(artifact.prepare_execution({source_id: bits}).run().buffers[output_id])
        assert actual.dtype == np.dtype("int64")
        assert actual.item() == int(expected)
