"""An ABI-backed span owns a C call edge even if resident labels are stale."""

import sys
from pathlib import Path

from src.compiler.concordance_declarations import (
    CELL_SET, SEQUENCE_CONTRACT_CONCORDANCE, VALUE_TYPE_ABI,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.source_evidence_resolver import (
    PhysicalLayoutEvidence, UnresolvedSourceDemand, require_physical_layout,
)
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_record_return_state import function_scope_of


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    'def helper(xs: list[float]) -> float:\n'
    '    return xs[0]\n'
    'def root(xs: list[float]) -> float:\n'
    '    return helper(xs)\n'
)


def test_exact_span_abi_controls_the_earliest_c_array_call_decision():
    contract = ExtractionContract(
        ROOT / 'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml'
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        SOURCE, 'root', name='span_call_evidence',
        extraction_contract=contract, progress=lambda _message: None,
    )
    book = module.metadata['identity_book']
    root = module.functions[exports[0]]
    scope = function_scope_of(root)
    before = tuple(book.clock)
    span = require_physical_layout(scope, root.args[0].id, book=book)
    assert isinstance(span, PhysicalLayoutEvidence)
    assert (span.physical_dtype, span.storage, span.rank, span.shape) == (
        'float64', 'span', 1, ())
    assert any(cell.page == VALUE_TYPE_ABI for cell in span.proof_cells)
    assert span.control_scope == scope

    # The generated extent has a descriptor mint and exact sequence-contract
    # revision, even though it has no VALUE_TYPE_ABI arm of its own.
    extent = require_physical_layout(scope, root.args[1].id, book=book)
    assert isinstance(extent, PhysicalLayoutEvidence)
    assert (extent.physical_dtype, extent.storage, extent.rank, extent.shape) == (
        'int64', 'span', 1, (1,))
    assert any(cell.page == CELL_SET for cell in extent.proof_cells)
    assert any(cell.page == SEQUENCE_CONTRACT_CONCORDANCE
               for cell in extent.proof_cells)
    assert tuple(book.clock) == before

    original_mint_of = book.mint_of
    try:
        book.mint_of = lambda cell: (None if cell == extent.proof_cells[0]
                                     else original_mint_of(cell))
        missing = require_physical_layout(scope, root.args[1].id, book=book)
    finally:
        book.mint_of = original_mint_of
    assert isinstance(missing, UnresolvedSourceDemand)
    assert missing.missing == 'missing_unique_span_abi'

    root.args[0].accounting.update({
        'program_abi_storage': 'scalar',
        'program_abi_rank': 0,
        'physical_dtype': None,
    })
    root.args[1].shape = ()
    root.args[1].accounting.update({
        'program_abi_storage': 'scalar',
        'program_abi_rank': 0,
        'physical_dtype': None,
    })
    decisions = []

    def trace(frame, event, result):
        if (event == 'return'
                and frame.f_code.co_name == 'authored_array_contract'
                and frame.f_locals.get('owner') == exports[0]
                and getattr(frame.f_locals.get('value'), 'id', None)
                in {root.args[0].id, root.args[1].id}):
            decisions.append((frame.f_locals['value'].id, result))
        return trace

    sys.setprofile(trace)
    try:
        artifact = emit_ssa_to_c(module, exports[0])
    finally:
        sys.setprofile(None)
    assert decisions == [(root.args[0].id, True), (root.args[1].id, True)]
    assert artifact.complete, artifact.shortfalls
    assert artifact.buffer_dtypes[0] == 'float64'
