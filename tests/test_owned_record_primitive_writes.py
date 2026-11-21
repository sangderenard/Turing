"""Declared primitive fields never bless unproven or differently typed writes."""
from dataclasses import replace
from tests.test_owned_record_primitive_elements import SOURCE, lower
from src.compiler.source_owned_record_elements import owned_record_primitive_element
from src.compiler.source_owned_record_writes import owned_record_primitive_writes
from src.compiler.concordance_declarations import FORMAL_ACTUAL_OCCURRENCE


def test_repeated_values_require_exact_invocation_and_source_argument():
    module, name, source = lower(SOURCE.replace('ITEM','int').replace('VALUE','7'))
    element = owned_record_primitive_element(module,name,source)
    check = lambda: owned_record_primitive_writes(module,name,source,element)
    expected = check()
    assert expected is not None
    calls = [(block,index,op) for block in module.functions[name].blocks.values()
             for index,op in enumerate(block.instrs)
             if op.attributes.get('ssa_sequence_operation') == 'append']
    first, second = [(b,i,op) for b,i,op in calls
        if sum(other.attributes.get('callee') == op.attributes.get('callee')
               for _,_,other in calls) == 2]
    block,index,op = first
    _,_,other = second
    original_attrs = dict(op.attributes)
    try:
        # Same helper and equal literal values still have different effects.
        op.attributes['source_effect_node_id'] = other.attributes['source_effect_node_id']
        assert check() is None
        op.attributes.clear(); op.attributes.update(original_attrs)
        block.instrs[index] = replace(op,args=[*op.args[:-1],other.args[-1]])
        assert check() is None
        op.attributes['source_effect_node_id'] = other.attributes['source_effect_node_id']
        assert check() is None
        op.attributes.clear(); op.attributes.update(original_attrs)
        block.instrs[index] = replace(op,op='Store')
        assert check() is None
    finally:
        block.instrs[index] = op
        op.attributes.clear(); op.attributes.update(original_attrs)
    assert check() == expected
    # Keep selector geometry and existing call positions intact, so the new
    # closed member-use census, not a stale selector receipt, rejects these.
    target_block,_,target_call = next((b,i,c) for b,i,c in calls if b.name.startswith('loop_body'))
    leaf_id = target_call.attributes['sequence_id']
    descriptor = module.sequence_tables[name].sequences[leaf_id]
    values = {v.id:v for b in module.functions[name].blocks.values() for ins in b.instrs
              for v in (*ins.args,*((ins.res,) if ins.res is not None else ()))}
    from src.transmogrifier.ssa import Instr
    from src.compiler.source_sequence_selection import local_selection_lifetimes
    from src.compiler.source_owned_record_returns import validate_owned_record_return_source
    for member_id in (descriptor.heap_owner_value_id, descriptor.capacity_value_id,
                      descriptor.length_address_id, descriptor.status_address_id):
        for operation in ('Call','Load'):
            extra = Instr(operation,[values[member_id]],None,attributes={'callee':'unknown_owner_consumer'})
            target_block.instrs.insert(len(target_block.instrs)-1,extra)
            try:
                assert local_selection_lifetimes(module,name,source.sequence_id) is not None
                assert validate_owned_record_return_source(module,name,source)
                assert owned_record_primitive_element(module,name,source) == element
                assert check() is None
            finally:
                target_block.instrs.remove(extra)
    original_attrs = dict(op.attributes)
    try:
        op.attributes['callee'] = next(c.attributes['callee'] for _,_,c in calls if c.attributes['callee'] != original_attrs['callee'])
        assert check() is None
    finally:
        op.attributes.clear(); op.attributes.update(original_attrs)
    assert check() == expected
    book = module.metadata['identity_book']
    page = book.page(FORMAL_ACTUAL_OCCURRENCE)
    receipt = next(c for c in expected if c.page == FORMAL_ACTUAL_OCCURRENCE)
    key = (receipt.row,receipt.column)
    original = page.cells[key]
    try:
        page.cells[key] = original + 1
        assert check() is None
    finally:
        page.cells[key] = original
    assert check() == expected
    from src.compiler.concordance_declarations import SSA_CONSTANT_HOIST, SSA_CONSTANT_MATERIALIZATION
    for page_kind in (SSA_CONSTANT_HOIST, SSA_CONSTANT_MATERIALIZATION):
        receipt = next(c for c in expected if c.page == page_kind)
        page = book.page(page_kind); key = (receipt.row,receipt.column)
        fact = page.cells[key]
        try:
            page.cells[key] = (fact[0],'unproven_move') if page_kind == SSA_CONSTANT_HOIST else (fact[0],fact[1],'1')
            assert check() is None
        finally:
            page.cells[key] = fact
    assert check() == expected


def test_declared_int_rejects_actual_string_writers():
    module,name,source = lower(SOURCE.replace('ITEM','int').replace('VALUE',"'same'"))
    element = owned_record_primitive_element(module,name,source)
    assert element is not None
    assert owned_record_primitive_writes(module,name,source,element) is None

def test_authored_loop_values_and_unary_values_have_exact_writer_proof():
    from tests.test_owned_record_return_source import SOURCE as authored
    module,name,source = lower(authored)
    element = owned_record_primitive_element(module,name,source)
    assert owned_record_primitive_writes(module,name,source,element) is not None

import pytest

@pytest.mark.parametrize('replacement', ['out = [7]', 'out = []\n        out.extend([7])'])
def test_unproved_initialization_or_bulk_writer_refuses(replacement):
    text = SOURCE.replace('ITEM','int').replace('VALUE','7').replace('out = []',replacement,1)
    module,name,source = lower(text)
    element = owned_record_primitive_element(module,name,source)
    assert owned_record_primitive_writes(module,name,source,element) is None

def test_reset_status_and_return_cannot_escape_original_native_placement():
    from src.compiler.source_sequence_selection import local_selection_lifetimes
    from src.compiler.source_owned_record_returns import validate_owned_record_return_source
    module,name,source = lower(SOURCE.replace('ITEM','int').replace('VALUE','7'))
    element = owned_record_primitive_element(module,name,source)
    check = lambda: owned_record_primitive_writes(module,name,source,element)
    expected = check(); assert expected is not None
    def prerequisites():
        assert local_selection_lifetimes(module,name,source.sequence_id) is not None
        assert validate_owned_record_return_source(module,name,source)
        assert owned_record_primitive_element(module,name,source) == element
    function = module.functions[name]
    reset_block = next(b for b in function.blocks.values()
                      if any(op.attributes.get('binding') == 'ssa_sequence_clear' for op in b.instrs)
                      and any(op.op == 'Call' for op in b.instrs))
    reset = next(op for op in reset_block.instrs if op.attributes.get('binding') == 'ssa_sequence_clear')
    before = list(reset_block.instrs)
    try:
        reset_block.instrs.insert(len(reset_block.instrs)-1,replace(reset))
        prerequisites(); assert check() is None
        reset_block.instrs[:] = before
        reset_block.instrs.remove(reset)
        reset_block.instrs.insert(len(reset_block.instrs)-1,reset)
        prerequisites(); assert check() is None
    finally:
        reset_block.instrs[:] = before
    assert check() == expected
    status_block = next(b for b in function.blocks.values() if b.name.startswith('loop_body'))
    status = next(op for op in status_block.instrs if op.attributes.get('binding') == 'ssa_sequence_status' and op.op == 'Store')
    before = list(status_block.instrs)
    try:
        status_block.instrs.insert(len(status_block.instrs)-1,replace(status))
        prerequisites(); assert check() is None
        status_block.instrs[:] = before
        status_block.instrs.remove(status)
        status_block.instrs.insert(1,status)
        prerequisites(); assert check() is None
    finally:
        status_block.instrs[:] = before
    assert check() == expected
    # Even restating selector geometry cannot authorize an extra owner/header
    # return. Source/element/lifetime premises still hold for this tamper.
    selection = local_selection_lifetimes(module,name,source.sequence_id).selection
    book = module.sequence_tables[name].book
    cell = selection.geometry_cell
    block = function.blocks[cell.row[-1]]
    ret = block.instrs[-1]
    page = book.page(cell.page); key = (cell.row,cell.column)
    saved = page.cells[key]
    values = {v.id:v for b in function.blocks.values() for op in b.instrs
              for v in (*op.args,*((op.res,) if op.res is not None else ()))}
    descriptor = module.sequence_tables[name].sequences[source.sequence_id]
    try:
        for extra in (descriptor.heap_owner_value_id,descriptor.capacity_value_id,
                      descriptor.length_address_id,descriptor.status_address_id):
            block.instrs[-1] = replace(ret,args=[*ret.args,values[extra]])
            page.cells[key] = (tuple((op.op,tuple(v.id for v in op.args),None if op.res is None else op.res.id,
                                     tuple(op.attributes.get('incoming_blocks') or ())) for op in block.instrs),tuple(block.successors))
            prerequisites(); assert check() is None
    finally:
        block.instrs[-1] = ret
        page.cells[key] = saved
    assert check() == expected


def test_mandatory_status_sinks_and_terminal_initializers_cannot_be_deleted():
    from src.compiler.source_sequence_selection import local_selection_lifetimes
    from src.compiler.source_owned_record_returns import validate_owned_record_return_source
    module,name,source = lower(SOURCE.replace('ITEM','int').replace('VALUE','7'))
    element = owned_record_primitive_element(module,name,source)
    check = lambda: owned_record_primitive_writes(module,name,source,element)
    expected = check(); assert expected is not None
    function = module.functions[name]
    candidates = [(block,index,op) for block in function.blocks.values()
                  for index,op in enumerate(block.instrs)
                  if op.op == 'Store' and op.attributes.get('binding') in
                  ('ssa_sequence_status','ssa_local_sequence_initialize')]
    assert any(op.attributes['binding'] == 'ssa_sequence_status' for _,_,op in candidates)
    assert any(op.attributes['binding'] == 'ssa_local_sequence_initialize' for _,_,op in candidates)
    for block,index,op in candidates:
        before = list(block.instrs)
        try:
            del block.instrs[index]
            assert local_selection_lifetimes(module,name,source.sequence_id) is not None
            assert validate_owned_record_return_source(module,name,source)
            assert owned_record_primitive_element(module,name,source) == element
            assert check() is None
        finally:
            block.instrs[:] = before
        assert check() == expected
