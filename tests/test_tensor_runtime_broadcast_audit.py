"""A mixed runtime broadcast retains its authored helper and guard proof."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler import tensor_ssa_lowering
from src.compiler.concordance_declarations import TENSOR_RUNTIME_BROADCAST_VALIDITY
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, IMPORTED_CELL, identity_book,
)


@pytest.fixture(scope="module")
def broadcast_module():
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
        if tuple(identity_book(module).page(
            TENSOR_RUNTIME_BROADCAST_VALIDITY
        ).rows()):
            observed.append(module)
            raise AtPublication
        return changed

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            tensor_ssa_lowering,
            "publish_computed_region_result_metadata", capture,
        )
        with pytest.raises(AtPublication):
            lower_ast_source_to_ssa(
                source, "root", name="broadcast_validity_audit",
                extraction_contract=contract,
                python_bindings={"AbstractTensor": AbstractTensor},
                tensor_ssa_reference=c_backend_repository_ssa_reference(),
                progress=lambda _: None, progress_bars=False,
            )
    module, = observed
    return module


def test_mixed_broadcast_has_complete_guarded_helper_proof(broadcast_module):
    module = broadcast_module
    book = identity_book(module)
    row, = tuple(book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY).rows())
    assert not CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
    assert not CorrelationTable._tensor_elementwise_extent_binding_findings(module)
    assert module.functions[row[1]].blocks["entry.broadcast_fault"].instrs[-1].op == "Trap"


@pytest.mark.parametrize("field,replacement", [
    (10, -1),  # wrong right input element-count identity
    (15, -1),  # wrong helper status identity
    (16, "unproved_helper"),
])
def test_mixed_broadcast_audit_rejects_forged_identity(
    broadcast_module, field, replacement,
):
    module = broadcast_module
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY)
    row, = tuple(page.rows())
    ref = book.latest_ref(TENSOR_RUNTIME_BROADCAST_VALIDITY, row)
    original = page.cell(ref.row, ref.column)
    changed = list(original)
    changed[field] = replacement
    page.cells[(ref.row, ref.column)] = tuple(changed)
    try:
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
    finally:
        page.cells[(ref.row, ref.column)] = original


def test_mixed_broadcast_audit_rejects_unchecked_fault(broadcast_module):
    module = broadcast_module
    book = identity_book(module)
    row, = tuple(book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY).rows())
    fault = module.functions[row[1]].blocks["entry.broadcast_fault"]
    original = list(fault.instrs)
    fault.instrs[:] = original[:1]
    try:
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
    finally:
        fault.instrs[:] = original


def test_mixed_broadcast_audit_rejects_foreign_helper_proof(broadcast_module):
    module = broadcast_module
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY)
    row, = tuple(page.rows())
    ref = book.latest_ref(TENSOR_RUNTIME_BROADCAST_VALIDITY, row)
    imported, = tuple(
        cause for cause, _ in book.edges_into(ref)
        if cause.page == IMPORTED_CELL
    )[:1]
    imports = book.page(IMPORTED_CELL)
    original = imports.cell(imported.row, imported.column)
    imports.cells[(imported.row, imported.column)] = replace(
        original, owner=object(),
    )
    try:
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
    finally:
        imports.cells[(imported.row, imported.column)] = original


def test_mixed_broadcast_audit_rejects_unchecked_count(broadcast_module):
    module = broadcast_module
    book = identity_book(module)
    page = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY)
    row, = tuple(page.rows())
    fact = page.latest(row)
    function = module.functions[row[1]]
    count = next(
        instruction for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None and instruction.res.id == fact[14]
    )
    original = dict(count.attributes)
    count.attributes = {**original, "overflow": "wrap"}
    try:
        assert CorrelationTable._tensor_runtime_broadcast_validity_findings(module)
    finally:
        count.attributes = original
