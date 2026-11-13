"""Equal reset literals retain the exact authored clear and resident identity."""
import pytest
from pathlib import Path
from tests.test_owned_record_return_source import SOURCE
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.concordance_declarations import (
    SSA_VALUE, CONTROL_CONST, CELL_SET, CONTROL_BLOCK, CANONICAL_VALUE)
from src.compiler.ssa_record_return_state import function_scope_of

def _ssa_history(book, cell):
    pending, seen = [cell], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(p for p, _ in book.edges_into(current)
                       if p.page == SSA_VALUE and p.row == cell.row)
    return seen


def _lower(source):
    module, _, exports = lower_ast_source_to_ssa(
        source, 'root', extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    return module, exports[0]



@pytest.mark.parametrize('explicit_clears', [False, True])
def test_sequence_clear_zero_retains_distinct_effect_and_length(explicit_clears):
    source = SOURCE
    if explicit_clears:
        source = source.replace('out.append(-7)', 'out.append(-7)\n            out.clear()\n            out.clear()')
    module, name = _lower(source)
    function = module.functions[name]
    book = module.metadata['identity_book']
    resets = []
    for block in function.blocks.values():
        for op in block.instrs:
            if op.attributes.get('binding') != 'ssa_sequence_clear':
                continue
            zero, length = op.args
            zref = book.latest_ref(SSA_VALUE, (function_scope_of(function), zero.id))
            minted = [book.mint_of(c) for c in _ssa_history(book, zref)
                      if book.mint_of(c) is not None]
            assert len(minted) == 1 and minted[0][0] == CONTROL_CONST
            bundle, = minted[0][1]
            assert bundle.page == CELL_SET
            causes = {p for p, _ in book.edges_into(bundle)}
            assert set(book.page(bundle.page).cell(bundle.row,bundle.column)) == {p.key for p in causes}
            controls = [p for p in causes if p.page == CONTROL_BLOCK]
            effects = [p for p in causes if p.page == CANONICAL_VALUE
                       and p.row[-1] == op.attributes['source_effect_node_id']]
            assert len(controls) == len(effects) == 1
            lref = book.latest_ref(SSA_VALUE, (function_scope_of(function), length.id))
            assert _ssa_history(book,lref).intersection(causes)
            fact = book.page(controls[0].page).cell(controls[0].row,controls[0].column)
            matches = [m for m in fact.sequence_mutations if m[3] == effects[0]]
            assert len(matches) == 1
            assert matches[0][1:3] == ('clear', ())
            assert matches[0][0].row[-1] == op.attributes['sequence_id']
            resets.append((zero.id,length.id,effects[0],controls[0]))
    assert len(resets) == (5 if explicit_clears else 3)
    assert len({r[0] for r in resets}) == len(resets)
    assert len({r[2] for r in resets}) == len(resets)
    if explicit_clears:
        assert max(sum(r[1] == x[1] for x in resets) for r in resets) == 3
