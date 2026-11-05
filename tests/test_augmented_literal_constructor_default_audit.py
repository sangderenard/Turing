"""A retained broadcast helper keeps the selected literal default's source."""

from dataclasses import replace
from pathlib import Path

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    NUMERICAL_CONSTRUCTOR_DEFAULT, VALUE_TYPE_ABI,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    ConcordanceRefusal, CorrelationTable, identity_book,
)


def test_augmented_reference_retains_exact_literal_default_abi_source():
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [{
            "function": "root", "parameter": "bits", "storage": "span",
            "dtype": "int64", "rank": 1,
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        }],
        "records": {}, "bindings": [],
    })
    source = """def root(bits: AbstractTensor):
    weights = bits.ensure_tensor((128, 64, 32, 16, 8, 4, 2, 1))
    shaped = bits.reshape(-1, 8).to_dtype('float64')
    return (shaped * weights.unsqueeze(0)).sum(dim=1).to_dtype('int64').sum().item()
"""
    try:
        module, _, _ = lower_ast_source_to_ssa(
            source, "root", name="augmented_literal_default_audit",
            extraction_contract=contract,
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _: None, progress_bars=False,
        )
    except ConcordanceRefusal as error:
        module = error.rejected_source_result[0]

    book = identity_book(module)
    page = book.page(NUMERICAL_CONSTRUCTOR_DEFAULT)
    rows = tuple(row for row in page.rows()
                 if page.latest(row)[0] == "ensure_tensor")
    assert len(rows) == 1
    receipt = page.latest(rows[0])
    result_ssa = receipt[2]
    assert not CorrelationTable._numerical_constructor_default_findings(module)

    selected = module.metadata["tensor_ssa_reference"]
    assert "+sha256:" in selected.source_identity
    module.metadata["tensor_ssa_reference"] = replace(
        selected, source_identity=selected.source_identity + "-forged",
    )
    try:
        assert CorrelationTable._numerical_constructor_default_findings(module)
    finally:
        module.metadata["tensor_ssa_reference"] = selected

    abi_refs = tuple(
        parent for parent, _ in book.edges_into(result_ssa)
        if parent.page == VALUE_TYPE_ABI and parent.row == rows[0][0].row
    )
    assert len(abi_refs) == 1
    abi_ref = abi_refs[0]
    abi_page = book.page(VALUE_TYPE_ABI)
    abi_fact = abi_page.cell(abi_ref.row, abi_ref.column)
    assert len(abi_fact.physical_alternatives) == 1
    arm = abi_fact.physical_alternatives[0]
    shape_states = tuple(cell for cell in arm.source_cells
                         if cell.page.name == "shape_transformation_state")
    assert len(shape_states) == 1
    abi_page.cells[(abi_ref.row, abi_ref.column)] = replace(
        abi_fact, physical_alternatives=(replace(
            arm, source_cells=tuple(cell for cell in arm.source_cells
                                    if cell != shape_states[0]),
        ),),
    )
    try:
        assert any(
            finding.kind == "numerical-constructor-default"
            for finding in CorrelationTable._numerical_constructor_default_findings(
                module,
            )
        )
    finally:
        abi_page.cells[(abi_ref.row, abi_ref.column)] = abi_fact
