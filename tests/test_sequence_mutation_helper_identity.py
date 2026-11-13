"""Generated helper identity is retained at each native mutation invocation."""
from copy import copy
from types import SimpleNamespace
from tests.test_owned_record_transfer_plan import _lower, SOURCE
from src.compiler.precompile_to_ssa import _ControlSSABuilder
from src.compiler.ssa_record_return_state import function_scope_of
from src.compiler.concordance_declarations import SSA_BLOCK, SSA_VALUE, CELL_SET, FORMAL_ACTUAL_OCCURRENCE


def test_mutation_status_retains_registered_helper_block_and_formal_inventory():
    module, name = _lower(SOURCE)
    book = module.metadata['identity_book']
    function = module.functions[name]
    helpers = {}
    for label, block in function.blocks.items():
        for index, op in enumerate(block.instrs):
            if op.attributes.get('ssa_sequence_operation') != 'append':
                continue
            helper = module.functions[op.attributes['callee']]
            helpers[helper.name] = helper
            value = book.latest_ref(SSA_VALUE,(function_scope_of(function),op.res.id))
            bundle = book.mint_of(value)[1][0]
            assert bundle.page == CELL_SET
            causes = {p for p,_ in book.edges_into(bundle)}
            expected_blocks = {book.latest_ref(SSA_BLOCK,(helper.name,helper.name,label))
                               for label in helper.blocks}
            assert None not in expected_blocks
            assert {cell for cell in causes if cell.page == SSA_BLOCK} == expected_blocks
            # Later registration can remint a shared helper's row formal.
            # The final live positional census, not the old status mint,
            # establishes current formal/actual identities.
            assert op.attributes['callee_input_ids'] == tuple(v.id for v in helper.args)
            for position, (formal, actual) in enumerate(zip(helper.args,op.args)):
                row = (helper.name,formal.id,name,label,index,position,function_scope_of(function))
                cell = book.latest_ref(FORMAL_ACTUAL_OCCURRENCE,row)
                actual_cell = book.latest_ref(SSA_VALUE,(function_scope_of(function),actual.id))
                assert cell is not None
                assert book.page(cell.page).cell(cell.row,cell.column) == actual.id
                assert actual_cell in {p for p,_ in book.edges_into(cell)}
            # Changing only the Call target cannot change the retained identity.
            other = next(f for f in module.functions.values()
                         if f.name != helper.name and f.metadata.get('ssa_sequence_operation') == 'append')
            assert not {book.latest_ref(SSA_BLOCK,(other.name,other.name,label))
                        for label in other.blocks} <= causes

    helper = next(iter(helpers.values()))
    builder = SimpleNamespace(sequence_helper_functions=helpers, _book=lambda:book)
    read = lambda h: _ControlSSABuilder._sequence_helper_call_sources(builder,h)
    assert read(helper)
    # Same string name does not register a different function object.
    assert read(copy(helper)) == ()
    original = builder.sequence_helper_functions
    builder.sequence_helper_functions = {}
    assert read(helper) == ()
    builder.sequence_helper_functions = original
    label = next(iter(helper.blocks))
    cell = book.latest_ref(SSA_BLOCK,(helper.name,helper.name,label))
    page=book.page(SSA_BLOCK);key=(cell.row,cell.column);old=page.cells[key]
    try:
        page.cells[key] = None
        assert read(helper) == ()
    finally:
        page.cells[key] = old
    helper.blocks['unregistered'] = helper.blocks[label]
    try:
        assert read(helper) == ()
    finally:
        del helper.blocks['unregistered']
    assert read(helper)
