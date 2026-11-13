"""Writer-use exclusion is restricted to a structurally proven move suffix."""
from tests.test_owned_record_transfer_plan import _lower, SOURCE
from src.compiler.source_owned_record_transfer import materialize_owned_record_transfers
from src.compiler.source_owned_record_projection_validation import validated_record_owned_return_suffix


def test_exact_suffix_excludes_no_earlier_native_operation():
    module, name = _lower(SOURCE)
    refs = materialize_owned_record_transfers(module, name)
    assert len(refs) == 1
    public_id = refs[0].row[1]
    suffix = validated_record_owned_return_suffix(module, name, public_id)
    assert suffix is not None
    label, start, operations = suffix
    block = module.functions[name].blocks[label]
    assert start > 0
    assert operations == tuple(block.instrs[start:])
    assert operations[-1].op == 'Ret'
    assert all(op is not block.instrs[start-1] for op in operations)
    assert validated_record_owned_return_suffix(module, name, -1) is None
    # A changed payload invalidates exclusions even if operation shape remains.
    zero = block.instrs[-3]
    old_attributes = zero.attributes
    try:
        zero.attributes = {'value': 1}
        assert validated_record_owned_return_suffix(module, name, public_id) is None
    finally:
        zero.attributes = old_attributes
    assert validated_record_owned_return_suffix(module, name, public_id) == suffix


def test_snapshot_cannot_reuse_source_or_public_member_identity():
    from dataclasses import replace
    module, name = _lower(SOURCE)
    refs = materialize_owned_record_transfers(module, name)
    assert len(refs) == 1
    cell = refs[0]
    book = module.metadata['identity_book']
    page = book.page(cell.page)
    key = (cell.row, cell.column)
    original = page.cells[key]
    for reused in original.member_pairs[0]:
        try:
            page.cells[key] = replace(original,
                snapshot_cells=(reused, *original.snapshot_cells[1:]))
            assert validated_record_owned_return_suffix(module, name, cell.row[1]) is None
        finally:
            page.cells[key] = original


def test_fresh_slots_cannot_be_used_before_transfer_or_have_foreign_mints(monkeypatch):
    from src.transmogrifier.ssa import Instr
    from src.compiler.concordance_declarations import CONTROL_CONST
    module,name=_lower(SOURCE)
    refs=materialize_owned_record_transfers(module,name)
    assert len(refs)==1
    cell=refs[0]; book=module.metadata['identity_book']
    fact=book.page(cell.page).cell(cell.row,cell.column)
    function=module.functions[name]
    check=lambda:validated_record_owned_return_suffix(module,name,cell.row[1])
    assert check() is not None
    args={arg.id:arg for arg in function.args}
    entry=next(iter(function.blocks.values()))
    for public_ref in (fact.member_pairs[0][1],fact.member_pairs[-1][1]):
        before=list(entry.instrs)
        try:
            entry.instrs.insert(len(entry.instrs)-1,
                Instr('Call',[args[public_ref.row[-1]]],None,attributes={'callee':'unregistered'}))
            assert check() is None
        finally:entry.instrs[:]=before
    owner=args[fact.member_pairs[-1][1].row[-1]]
    function.args.append(owner)
    try:assert check() is None
    finally:function.args.pop()
    original_mint=book.mint_of
    for ref in (fact.member_pairs[0][1],fact.snapshot_cells[0],fact.null_cell):
        def foreign_mint(candidate):
            original=original_mint(candidate)
            return (CONTROL_CONST,original[1]) if candidate==ref else original
        with monkeypatch.context() as patch:
            patch.setattr(book,'mint_of',foreign_mint)
            assert check() is None
    assert check() is not None
