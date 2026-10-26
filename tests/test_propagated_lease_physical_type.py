"""A propagated frame lease is born from its callee's posted physical row."""
from pathlib import Path

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.concordance_declarations import SSA_VALUE
from src.compiler.ssa_record_return_state import ssa_value_identity_cell


def test_propagated_status_lease_posts_physical_layout_at_birth():
    root = Path(__file__).resolve().parents[1]
    module, _, exports = lower_ast_source_to_ssa(
        (root / 'tests/record_view_corpus/c631_record_sequence_status.py').read_text(),
        'root', name='propagated_status_physical',
        extraction_contract=root / 'extraction_contracts/program_extraction.yaml',
        progress=lambda _: None, progress_bars=False,
    )
    function = module.functions[exports[0]]
    leases = [argument for argument in function.args
              if 'propagated_formal_id' in argument.accounting
              and argument.accounting.get('physical_dtype') == 'int']
    assert leases
    book = module.metadata['identity_book']
    for lease in leases:
        cell = ssa_value_identity_cell(function, lease.id, book=book)
        history = book.page(SSA_VALUE).history(cell.row)
        assert history
        assert history[0][1].physical_dtype == 'int'
        assert history[-1][1].physical_dtype == lease.accounting['physical_dtype']
    assert CorrelationTable.build(module).findings(module) == []
