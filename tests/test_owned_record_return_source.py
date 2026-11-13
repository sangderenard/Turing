"""Exact constructor return evidence is distinct from a transfer permission."""
from pathlib import Path
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_storage_requirements import public_owned_sequence_returns

SOURCE = """from dataclasses import dataclass
@dataclass
class Result:
    items: list[int]

def root(n: int, choose: bool, fallback: bool) -> Result:
    if choose:
        out = []
        for i in range(n):
            out.append(i)
    else:
        if fallback:
            out = []
            for i in range(n):
                out.append(i)
        else:
            out = []
            out.append(-7)
            for i in range(n):
                out.append(-i)
    return Result(items=out)
"""


def test_keyword_constructor_source_is_exact_but_not_a_transfer_grant():
    from src.compiler.source_owned_record_returns import owned_record_return_sources
    module, _, exports = lower_ast_source_to_ssa(
        SOURCE, 'root', extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    name = exports[0]
    from src.compiler.identity_concordance import CorrelationTable
    assert CorrelationTable.build(module).findings(module) == []
    evidence = owned_record_return_sources(module, name)
    assert len(evidence) == 1
    proof = evidence[0]
    assert proof.field_name == 'items'
    assert public_owned_sequence_returns(module, name) == ()
    table = module.sequence_tables[name]
    assert table.sequences[proof.sequence_id].heap_owner_value_id == proof.owner_id
    book = table.book
    page = book.page(proof.constructor_operands_cell.page)
    key = (proof.constructor_operands_cell.row, proof.constructor_operands_cell.column)
    original = page.cells[key]
    # Missing/misnamed constructor evidence cannot be reconstructed from a
    # matching record field name or the pointer's apparent shape.
    page.cells[key] = tuple((cell, 'kw:other' if role == 'kw:items' else role)
                            for cell, role in original)
    try:
        assert owned_record_return_sources(module, name) == ()
    finally:
        page.cells[key] = original
    assert owned_record_return_sources(module, name) == evidence

    # A return slot naming the sequence itself is not a return of the
    # authored record constructor, even though that field value is related.
    slot_page = book.page(proof.return_slot_cell.page)
    slot_key = (proof.return_slot_cell.row, proof.return_slot_cell.column)
    slot_original = slot_page.cells[slot_key]
    slot_page.cells[slot_key] = proof.source_cell
    try:
        assert owned_record_return_sources(module, name) == ()
    finally:
        slot_page.cells[slot_key] = slot_original
    # Neither the callee object nor another keyword is the field's source.
    callee = next(cell for cell, role in original if role == 'callee')
    page.cells[key] = tuple((callee if role == 'kw:items' else cell, role)
                           for cell, role in original)
    try:
        assert owned_record_return_sources(module, name) == ()
    finally:
        page.cells[key] = original
