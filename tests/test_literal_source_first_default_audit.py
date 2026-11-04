"""The selected constructor default can precede its executable Const."""

from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import NUMERICAL_CONSTRUCTOR_DEFAULT
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    ConcordanceRefusal, CorrelationTable, identity_book,
)


def test_source_first_literal_default_keeps_exact_upstream_proof():
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    return (bits.reshape(-1, 8) * weights.unsqueeze(0)).sum(dim=1)
"""
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi({
        "values": [{"function": "root", "parameter": "bits",
                    "storage": "span", "dtype": "int64", "rank": 1,
                    "python_type": "src.common.tensors.abstraction.AbstractTensor"}],
        "records": {}, "bindings": [],
    })
    try:
        module, _, _ = lower_ast_source_to_ssa(
            source, "root", name="source_first_literal_probe",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except ConcordanceRefusal as refusal:
        # The broadcast's separate physical conversion may still refuse.
        # Audit the retained module's exact selected-default proof here.
        retained = refusal.rejected_source_result
        module = retained[0] if isinstance(retained, tuple) else retained
        assert module is not None

    book = identity_book(module)
    page = book.page(NUMERICAL_CONSTRUCTOR_DEFAULT)
    row = next(row for row in page.rows()
               if page.latest(row)[0] == "ensure_tensor")
    receipt = book.latest_ref(NUMERICAL_CONSTRUCTOR_DEFAULT, row)
    result_ssa = page.cell(receipt.row, receipt.column)[2]
    state = next(
        parent for parent, _ in book.edges_into(result_ssa)
        if parent.page.name == "shape_transformation_state"
    )
    transformation = next(
        parent for parent, _ in book.edges_into(state)
        if parent.page.name == "shape_transformation_concordance"
    )
    assert not CorrelationTable._numerical_constructor_default_findings(module)

    transform_page = book.page(transformation.page)
    original = transform_page.cell(transformation.row, transformation.column)
    transform_page.cells[(transformation.row, transformation.column)] = (
        original[0], ((8,), "int64", 1, "static", None),
    )
    try:
        assert any(
            finding.kind == "numerical-constructor-default"
            for finding in CorrelationTable._numerical_constructor_default_findings(
                module,
            )
        )
    finally:
        transform_page.cells[(transformation.row, transformation.column)] = original
