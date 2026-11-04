"""A region Call keeps metadata bindings for every computed tensor input."""

from dataclasses import replace
from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler import tensor_ssa_lowering
from src.compiler.concordance_declarations import (
    TENSOR_METADATA_CALL_BINDING, TENSOR_METADATA_RESULT_BINDING,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book


def test_two_runtime_views_keep_both_call_metadata_bindings(monkeypatch):
    source = """def root(left: AbstractTensor, right: AbstractTensor) -> AbstractTensor:
    a = left.unsqueeze(1)
    b = right.unsqueeze(0)
    return a + b
"""
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": name, "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"}
            for name in ("left", "right")
        ],
        "records": {}, "bindings": [],
    })
    publish = tensor_ssa_lowering.publish_computed_region_result_metadata
    observed = []

    class AtPublication(Exception):
        pass

    def capture(module, **options):
        changed = publish(module, **options)
        book = identity_book(module)
        rows = tuple(book.page(TENSOR_METADATA_CALL_BINDING).rows())
        if rows:
            observed.append(module)
            raise AtPublication
        return changed

    monkeypatch.setattr(
        tensor_ssa_lowering, "publish_computed_region_result_metadata", capture,
    )
    try:
        lower_ast_source_to_ssa(
            source, "root", name="two_view_metadata_audit",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except AtPublication:
        pass
    assert observed, "the real source must reach the metadata publisher"
    module = observed[-1]
    book = identity_book(module)
    page = book.page(TENSOR_METADATA_CALL_BINDING)
    row, = tuple(page.rows())
    ref = book.latest_ref(TENSOR_METADATA_CALL_BINDING, row)
    fact = page.cell(ref.row, ref.column)
    assert len(fact[0]) == len(fact[1]) == 6
    predecessors = tuple(
        cause for cause, _ in book.edges_into(ref)
        if cause.page == TENSOR_METADATA_RESULT_BINDING
    )
    assert len(predecessors) == 2
    assert not CorrelationTable._tensor_metadata_call_binding_findings(module)

    # A last-writer-only page used to keep the second view's three values
    # while the live Call still carried both views. The audit must reject it.
    page.cells[(ref.row, ref.column)] = (fact[0][3:], fact[1][3:])
    try:
        assert CorrelationTable._tensor_metadata_call_binding_findings(module)
    finally:
        page.cells[(ref.row, ref.column)] = fact

    callee = module.functions[row[4]]
    tensor_formal = callee.args[0]
    original_accounting = dict(tensor_formal.accounting)
    tensor_formal.accounting = {
        **original_accounting,
        "tensor_shape_value_id": fact[1][3],
    }
    try:
        assert CorrelationTable._tensor_metadata_call_binding_findings(module)
    finally:
        tensor_formal.accounting = original_accounting

    table = module.tensor_tables[callee.name]
    descriptor = table.by_id(int(tensor_formal.id))
    table.tensors[int(tensor_formal.id)] = replace(
        descriptor, shape_value_id=fact[1][3],
    )
    try:
        assert CorrelationTable._tensor_metadata_call_binding_findings(module)
    finally:
        table.tensors[int(tensor_formal.id)] = descriptor
