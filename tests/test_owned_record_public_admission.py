"""Public ports require the complete source-specific record ownership proof."""
from dataclasses import replace
from tests.test_owned_record_transfer_plan import _lower, SOURCE
from src.compiler.source_owned_record_transfer import materialize_owned_record_transfers
from src.compiler.source_owned_record_projection_validation import (
    publish_record_owned_public_returns, validated_record_public_owned_return,
)
from src.compiler.ssa_storage_requirements import public_owned_sequence_returns
from src.compiler.identity_concordance import CorrelationTable


def test_explicit_record_admission_joins_exact_source_descriptor_and_owner():
    module,name=_lower(SOURCE)
    projected=materialize_owned_record_transfers(module,name)
    assert len(projected)==1
    assert public_owned_sequence_returns(module,name)==()
    admitted=publish_record_owned_public_returns(module,name)
    assert len(admitted)==1
    cell=admitted[0]
    proof=validated_record_public_owned_return(module,cell)
    assert proof is not None
    ports=public_owned_sequence_returns(module,name)
    assert len(ports)==1 and ports[0].data_id==projected[0].row[1]
    assert CorrelationTable.build(module).findings(module)==[]
    assert publish_record_owned_public_returns(module,name)==admitted
    book=module.metadata['identity_book']; page=book.page(cell.page); key=(cell.row,cell.column)
    original=page.cells[key]
    try:
        page.cells[key]=replace(original,source_cell=proof[1].source_cell)
        assert validated_record_public_owned_return(module,cell) is None
        assert public_owned_sequence_returns(module,name)==()
        assert any(f.kind=='public-owned-sequence-return' for f in CorrelationTable.build(module).findings(module))
    finally:page.cells[key]=original
    assert public_owned_sequence_returns(module,name)==ports


def test_wrong_element_has_no_projection_or_public_ports():
    bad=SOURCE.replace('out.append(i)', "out.append('x')").replace('out.append(-i)',"out.append('y')").replace('out.append(-7)',"out.append('seed')")
    module,name=_lower(bad)
    assert materialize_owned_record_transfers(module,name)==()
    assert publish_record_owned_public_returns(module,name)==()
    assert public_owned_sequence_returns(module,name)==()
