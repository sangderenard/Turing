"""Rank-one constructor captures require an exact tuple-member region feed."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    NUMERICAL_CONSTRUCTOR_DIMENSION, REGION_SIGNATURE,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, EDGE_PAGE


def test_dynamic_constructor_dimension_has_exact_member_and_region_cause():
    source = """def root(like: AbstractTensor, count: int) -> AbstractTensor:
    return AbstractTensor.zeros((count,), cls=type(like))
"""
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({
        "values": [
            {"function": "root", "parameter": "like", "storage": "span",
             "dtype": "int64", "rank": 1,
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
            {"function": "root", "parameter": "count", "storage": "scalar",
             "dtype": "int64", "rank": 0, "shape": [],
             "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", name="dimension_member_audit",
        extraction_contract=contract,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    page = book.page(NUMERICAL_CONSTRUCTOR_DIMENSION)
    rows = tuple(page.rows())
    assert len(rows) == 1
    assert CorrelationTable._numerical_constructor_dimension_findings(
        module,
    ) == []
    row = rows[0]
    receipt = book.latest_ref(NUMERICAL_CONSTRUCTOR_DIMENSION, row)
    member, layout, operands = page.cell(receipt.row, receipt.column)
    assert member != row[2]

    # A coincident container ID cannot replace the one proved member.
    key = (receipt.row, receipt.column)
    page.cells[key] = (row[2], layout, operands)
    try:
        assert CorrelationTable._numerical_constructor_dimension_findings(
            module,
        )
    finally:
        page.cells[key] = (member, layout, operands)

    signatures = tuple(
        ref
        for signature_row in book.page(REGION_SIGNATURE).rows()
        for ref in (book.latest_ref(REGION_SIGNATURE, signature_row),)
        if receipt in {source for source, _ in book.edges_into(ref)}
    )
    assert len(signatures) == 1
    signature = signatures[0]
    assert member.row[1] in book.page(REGION_SIGNATURE).cell(
        signature.row, signature.column,
    ).feeds
    edge_page = book.page(EDGE_PAGE)
    edge_row = next(
        (signature.key, source.key, stage.name)
        for source, stage in book.edges_into(signature)
        if source == receipt
    )
    edge_fact = edge_page.cells.pop((edge_row, 0))
    edge_stamp = edge_page.stamps.pop((edge_row, 0))
    edge_page.row_columns.pop(edge_row)
    edge_page.scopes[signature.key].pop(edge_row)
    try:
        assert CorrelationTable._numerical_constructor_dimension_findings(
            module,
        )
    finally:
        edge_page.cells[(edge_row, 0)] = edge_fact
        edge_page.stamps[(edge_row, 0)] = edge_stamp
        edge_page.row_columns[edge_row] = [0]
        edge_page.scopes[signature.key][edge_row] = None
