"""Fresh public descriptor cells preserve the source's recorded physical layout."""
from dataclasses import replace
from tests.test_owned_record_transfer_plan import _lower, SOURCE
from src.compiler.source_owned_record_transfer import materialize_owned_record_transfers
from src.compiler.source_owned_record_projection_validation import validated_record_owned_return_structure


def test_fresh_member_physical_fields_cannot_diverge_from_source():
    module,name=_lower(SOURCE)
    cell,=materialize_owned_record_transfers(module,name)
    book=module.metadata['identity_book']
    fact=book.page(cell.page).cell(cell.row,cell.column)
    check=lambda:validated_record_owned_return_structure(module,name,cell.row[1])
    assert check() == (cell,fact)
    for _,public in fact.member_pairs:
        page=book.page(public.page); key=(public.row,public.column); original=page.cells[key]
        changes={'physical_dtype':'float32' if original.physical_dtype!='float32' else 'int64',
                 'storage':'scalar' if original.storage!='scalar' else 'span',
                 'rank':99,'arena':not original.arena,'device':'unexpected-device'}
        for field,value in changes.items():
            try:
                page.cells[key]=replace(original,**{field:value})
                assert check() is None, (public,field)
            finally:
                page.cells[key]=original
    assert check() == (cell,fact)
