"""Exact constructor return evidence is distinct from a transfer permission."""
from pathlib import Path
from src.compiler.extraction_contract import ExtractionContract
from tests.owned_record_test_support import lower_before_owned_record_boundary as lower_ast_source_to_ssa
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


def test_captured_source_proof_survives_only_ret_projection_not_identity_changes():
    from dataclasses import replace
    from src.compiler.source_owned_record_returns import (
        owned_record_return_sources, validate_owned_record_return_source,
    )
    module, _, exports = lower_ast_source_to_ssa(
        SOURCE, 'root', extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    name = exports[0]
    proof, = owned_record_return_sources(module, name)
    valid = lambda p: validate_owned_record_return_source(module, name, p)
    assert valid(proof)
    assert not validate_owned_record_return_source(module, 'missing', proof)
    for bad in (
        replace(proof, field_name='other'),
        replace(proof, record_id=proof.record_id+1),
        replace(proof, sequence_id=proof.sequence_id+1),
        replace(proof, owner_id=proof.owner_id+1),
        replace(proof, source_cell=proof.constructor_operands_cell),
    ):
        assert not valid(bad)
    # Simulate only the later boundary rewriting Ret. Captured source lineage
    # remains true; the ordinary reader still refuses to call it a live return.
    rets = [op for block in module.functions[name].blocks.values()
            for op in block.instrs if op.op == 'Ret']
    saved = [list(op.args) for op in rets]
    try:
        for op in rets:
            op.args = []
        assert valid(proof)
        assert owned_record_return_sources(module, name) == ()
        assert public_owned_sequence_returns(module, name) == ()
    finally:
        for op, args in zip(rets, saved):
            op.args = args
    book = module.sequence_tables[name].book
    page = book.page(proof.constructor_operands_cell.page)
    key = (proof.constructor_operands_cell.row, proof.constructor_operands_cell.column)
    original = page.cells[key]
    try:
        page.cells[key] = tuple((cell, 'kw:other' if role == 'kw:items' else role)
                                for cell, role in original)
        assert not valid(proof)
    finally:
        page.cells[key] = original
    assert valid(proof)
