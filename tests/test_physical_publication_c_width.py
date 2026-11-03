"""C scalar widths follow posted physical SSA facts, including control indices."""

from pathlib import Path

from src.compiler.concordance_declarations import (
    FRAME_LINK, REVOKE_PHYSICAL, SSA_CONSTANT_MATERIALIZATION,
    SSA_VALUE, VALUE_TYPE_ABI, rebuilt_ssa_value_fact,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import Derived, Mode
from src.compiler.source_evidence_resolver import (
    PhysicalLayoutEvidence, require_physical_layout,
)
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_record_return_state import function_scope_of


ROOT = Path(__file__).resolve().parents[1]


def test_control_scalar_publications_and_c_width_use_their_exact_cells():
    contract = ExtractionContract(
        ROOT / 'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml'
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        'def root(x: int) -> int:\n    return x + 1\n',
        'root', name='physical_publication_c_width',
        extraction_contract=contract, progress=lambda _message: None,
    )
    book = module.metadata['identity_book']
    root = module.functions[exports[0]]
    region = next(function for name, function in module.functions.items()
                  if name.startswith(exports[0] + '__planned_region_'))
    scope = function_scope_of(region)
    region_literal = require_physical_layout(scope, 1, book=book)
    assert isinstance(region_literal, PhysicalLayoutEvidence)
    assert region_literal.physical_dtype == 'int64'
    assert any(cell.page == VALUE_TYPE_ABI
               for cell in region_literal.proof_cells)

    root_scope = function_scope_of(root)
    minted = next(
        instruction.res for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == 'Const' and instruction.res is not None
        and instruction.attributes.get('value') == 0
        and int(instruction.res.id) >= 1 << 61
    )
    control_index = require_physical_layout(root_scope, minted.id, book=book)
    assert isinstance(control_index, PhysicalLayoutEvidence)
    assert control_index.physical_dtype == 'int64'
    assert any(cell.page == SSA_CONSTANT_MATERIALIZATION
               for cell in control_index.proof_cells)
    assert book.latest_ref(SSA_VALUE, (root_scope, minted.id)) == (
        control_index.subject_cell)

    # The resident accounting is a view. A stale view must not change the
    # root public C width established by the posted formal's SSA_VALUE.
    root.args[0].accounting['physical_dtype'] = 'float64'
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    assert artifact.buffer_dtypes == ('int64', 'int64')

    # A missing scalar physical row cannot borrow the stale accounting view.
    formal_row = (root_scope, int(root.args[0].id))
    prior = book.latest_ref(SSA_VALUE, formal_row)
    prior_fact = book.page(SSA_VALUE).cell(prior.row, prior.column)
    book.post(
        SSA_VALUE, formal_row,
        rebuilt_ssa_value_fact(
            rebuilt_ssa_value_fact(
                prior_fact, prior_fact.dtype, prior_fact.shape,
                prior_fact.origin, physical=REVOKE_PHYSICAL),
            prior_fact.dtype, prior_fact.shape, prior_fact.origin,
            physical={"storage": "scalar", "rank": 0}),
        stage=FRAME_LINK, provenance=Derived((prior,)), mode=Mode.REVISE,
    )
    refused = emit_ssa_to_c(module, exports[0])
    assert not refused.complete
    assert any('scalar-physical-layout-unproven' in item.reason
               for item in refused.shortfalls)
