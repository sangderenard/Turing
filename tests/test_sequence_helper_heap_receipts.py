import pytest

from tests.test_owned_record_current_declaration import lower_source
from src.compiler.concordance_declarations import (
    SEQUENCE_HELPER_HEAP_OPERATION, SSA_CONSTANT_MATERIALIZATION,
)
from src.compiler.source_sequence_helper_heap import sequence_helper_heap_operation_sources


@pytest.fixture(scope='module')
def registered():
    return lower_source()[0]


def test_registered_heap_and_constant_receipts_are_exact(registered):
    module = registered
    book = module.metadata['identity_book']
    rows = tuple(book.page(SEQUENCE_HELPER_HEAP_OPERATION).rows())
    assert rows
    kinds = set()
    for scope, name, label, index in rows:
        assert scope == name
        proof = sequence_helper_heap_operation_sources(module, name, label, index)
        assert proof is not None
        operation = module.functions[name].blocks[label].instrs[index]
        kinds.add(operation.op)
    assert kinds == {'Alloca', 'Call'}
    helpers = {r[1] for r in rows}
    for name in helpers:
        for label, block in module.functions[name].blocks.items():
            for operation in block.instrs:
                if operation.op != 'Const':
                    continue
                row = (name, name, label, operation.res.id)
                fact = book.page(SSA_CONSTANT_MATERIALIZATION).latest(row)
                if fact is None:
                    from src.compiler.concordance_declarations import SSA_CONSTANT_HOIST
                    old = [r for r in book.page(SSA_CONSTANT_MATERIALIZATION).rows()
                           if r[:2] == (name, name) and r[-1] == operation.res.id]
                    assert len(old) == 1
                    moves = [r for r in book.page(SSA_CONSTANT_HOIST).rows()
                             if r[0] == (name, name) and r[1] == operation.res.id
                             and r[2] == old[0][2] and r[4] == label]
                    assert len(moves) == 1
                    fact = book.page(SSA_CONSTANT_MATERIALIZATION).latest(old[0])
                assert fact == ('Const', operation.res.dtype, repr(operation.attributes.get('value')))


def test_heap_receipts_reject_changed_operation_or_missing_authority(registered, monkeypatch):
    module = registered
    book = module.metadata['identity_book']
    page = book.page(SEQUENCE_HELPER_HEAP_OPERATION)
    rows = tuple(page.rows())
    for row in rows:
        _, name, label, index = row
        operation = module.functions[name].blocks[label].instrs[index]
        check = lambda: sequence_helper_heap_operation_sources(module, name, label, index)
        assert check() is not None
        original_op, attrs, args, result = operation.op, operation.attributes, operation.args, operation.res
        variations = [{'native_heap_abi': 'other'}, {'callee': 'unregistered_copy'}] if original_op == 'Call' else [
            {'allocation_kind': 'other'}, {'element_dtype': 'float64'},
            {'zero_initialize': not attrs.get('zero_initialize', False)},
        ]
        try:
            for replacement in variations:
                operation.attributes = {**attrs, **replacement}
                assert check() is None
            operation.attributes = attrs
            operation.op = 'Load'
            assert check() is None
            operation.op = original_op
            operation.args = list(reversed(args))
            assert check() is None
            operation.args = args
            operation.res = args[0] if result is None else None
            assert check() is None
        finally:
            operation.op, operation.attributes, operation.args, operation.res = original_op, attrs, args, result
        cell = book.latest_ref(SEQUENCE_HELPER_HEAP_OPERATION, row)
        fact = page.cell(cell.row, cell.column)
        original_edges = book.edges_into
        with monkeypatch.context() as patch:
            patch.setattr(book, 'edges_into', lambda c:
                          tuple((p, s) for p, s in original_edges(c) if p != fact.control_cell)
                          if c == cell else original_edges(c))
            assert check() is None
        assert check() is not None


def test_heap_registration_receipt_does_not_allow_accounting_substitution(registered):
    module = registered
    book = module.metadata['identity_book']
    for _, name, label, index in book.page(SEQUENCE_HELPER_HEAP_OPERATION).rows():
        operation = module.functions[name].blocks[label].instrs[index]
        carriers = operation.args[:2] if operation.op == 'Call' else (operation.res,)
        for carrier in carriers:
            accounting = carrier.accounting
            try:
                carrier.accounting = {**accounting, 'heap_owner_value_id': -1}
                assert sequence_helper_heap_operation_sources(module, name, label, index) is None
            finally:
                carrier.accounting = accounting


def test_registered_literal_requires_exact_kept_width(registered, monkeypatch):
    from dataclasses import replace
    from src.compiler.concordance_declarations import SCALAR_INTEGER_WIDTH, IntegerWidthDecision
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.source_sequence_helper_heap import registered_helper_integer_materialization
    module, book = registered, registered.metadata['identity_book']
    rows = [r for r in book.page(SSA_CONSTANT_MATERIALIZATION).rows()
            if 'append_child_copy' in r[1]
            and book.page(SSA_CONSTANT_MATERIALIZATION).latest(r)[1] == 'int']
    assert rows
    row = rows[0]
    material = book.latest_ref(SSA_CONSTANT_MATERIALIZATION, row)
    width = book.latest_ref(SCALAR_INTEGER_WIDTH, (row[0], row[-1]))
    assert registered_helper_integer_materialization(module, row, material)
    latest, edges = book.latest_ref, book.edges_into
    with monkeypatch.context() as patch:
        patch.setattr(book, 'latest_ref', lambda page, key: None
                      if page == SCALAR_INTEGER_WIDTH and key == width.row else latest(page, key))
        assert not registered_helper_integer_materialization(module, row, material)
        assert CorrelationTable._constant_materialization_findings(module)
    foreign = next(c for r in rows[1:] for c in (book.latest_ref(SCALAR_INTEGER_WIDTH, (r[0], r[-1])),)
                   if c is not None and c != width)
    with monkeypatch.context() as patch:
        patch.setattr(book, 'edges_into', lambda c: edges(foreign) if c == width else edges(c))
        assert not registered_helper_integer_materialization(module, row, material)
    page = book.page(SCALAR_INTEGER_WIDTH)
    read = page.cell
    fact = read(width.row, width.column)
    for wrong in (replace(fact, decision=IntegerWidthDecision.WIDENED),
                  replace(fact, source_dtype='int64'), replace(fact, dtype='int64')):
        with monkeypatch.context() as patch:
            patch.setattr(page, 'cell', lambda row, col, wrong=wrong:
                          wrong if (row, col) == (width.row, width.column) else read(row, col))
            assert not registered_helper_integer_materialization(module, row, material)
    assert registered_helper_integer_materialization(module, row, material)


def test_helper_factory_preserves_bool_and_native_integer_width():
    from src.compiler.ir_sequence_tables import _Builder
    from src.transmogrifier.ssa import BasicBlock
    from src.transmogrifier.dtype_layout import resolve
    builder = _Builder(1)
    block = BasicBlock('entry')
    for literal in (False, True, 0, 1):
        value = builder.const(block, literal, dtype='bool')
        assert value.dtype == 'bool'
        assert type(block.instrs[-1].attributes['value']) is bool
        assert block.instrs[-1].attributes['value'] is bool(literal)
    integer = builder.const(block, 7)
    assert integer.dtype == 'int' and resolve(integer.dtype).name == 'int32'
    for dtype in ('int8', 'int32', 'int64', 'uint64'):
        assert builder.const(block, 7, dtype=dtype).dtype == dtype
    builder.const(block, 2, dtype='bool')
    assert type(block.instrs[-1].attributes['value']) is int  # unsupported is not silently truthified


def test_heap_formal_requires_exact_pointer_owner_call_chain(registered, monkeypatch):
    from src.compiler.concordance_declarations import SSA_VALUE
    module, book = registered, registered.metadata['identity_book']
    page = book.page(SEQUENCE_HELPER_HEAP_OPERATION)
    cases = []
    for row in page.rows():
        fact = page.latest(row)
        for old in fact.argument_cells:
            current = book.latest_ref(SSA_VALUE, old.row)
            if old != current:
                cases.append((row, old, current))
    assert cases
    row, old, current = cases[0]
    _, name, label, index = row
    check = lambda: sequence_helper_heap_operation_sources(module, name, label, index)
    assert check() is not None
    edges = book.edges_into
    with monkeypatch.context() as patch:
        patch.setattr(book, 'edges_into', lambda c: tuple((p,s) for p,s in edges(c) if p != old)
                      if c == current else edges(c))
        assert check() is None
    # Wrong live occurrence or a borrowed payload in place of the owned
    # pointer cell must not become a valid imported-table allocation root.
    callsites = [(b, i, op) for f in module.functions.values() for b in f.blocks.values()
                 for i, op in enumerate(b.instrs) if op.op == 'Call' and op.attributes.get('callee') == name]
    assert callsites
    block, index_in_block, call = callsites[0]
    args = call.args
    try:
        call.args = [args[-1], *args[1:]]
        assert check() is None
        call.args = args
        block.instrs.insert(index_in_block, block.instrs[index_in_block])
        assert check() is None
    finally:
        call.args = args
        if block.instrs[index_in_block:index_in_block+2] == [call, call]:
            block.instrs.pop(index_in_block)
    assert check() is not None


def test_historical_literal_is_not_live_heap_permission(registered):
    from src.compiler.source_sequence_helper_heap import registered_helper_integer_materialization
    module, book = registered, registered.metadata['identity_book']
    heap_row = next(r for r in book.page(SEQUENCE_HELPER_HEAP_OPERATION).rows())
    name = heap_row[1]
    row = next(r for r in book.page(SSA_CONSTANT_MATERIALIZATION).rows()
               if r[1] == name and book.page(SSA_CONSTANT_MATERIALIZATION).latest(r)[1] == 'int')
    material = book.latest_ref(SSA_CONSTANT_MATERIALIZATION, row)
    assert registered_helper_integer_materialization(module, row, material)
    function = module.functions.pop(name)
    try:
        assert registered_helper_integer_materialization(module, row, material)
        assert sequence_helper_heap_operation_sources(module, *heap_row[1:]) is None
    finally:
        module.functions[name] = function
    definition = next((block, i, op) for block in function.blocks.values()
                      for i, op in enumerate(block.instrs) if op.res is not None and op.res.id == row[-1])
    block, index, operation = definition
    assert any(v.id == row[-1] for b in function.blocks.values() for op in b.instrs for v in op.args)
    block.instrs.pop(index)
    try:
        assert not registered_helper_integer_materialization(module, row, material)  # orphan use is not retirement
    finally:
        block.instrs.insert(index, operation)
    attributes = operation.attributes
    try:
        operation.attributes = {**attributes, 'value': attributes['value'] + 1}
        assert not registered_helper_integer_materialization(module, row, material)
    finally:
        operation.attributes = attributes
    assert registered_helper_integer_materialization(module, row, material)


def test_borrowed_payload_cannot_be_an_owned_pointer_root(registered):
    from src.compiler.source_sequence_helper_heap import _registered_pointer_root_sources
    from src.compiler.source_owned_record_returns import owned_record_return_sources
    module = registered
    roots = [(name, s) for name in module.functions for s in owned_record_return_sources(module, name)]
    assert len(roots) == 1
    name, source = roots[0]
    call = next(op for b in module.functions[name].blocks.values() for op in b.instrs
                if op.op == 'Call' and op.attributes.get('ssa_sequence_operation') == 'append_child_copy')
    assert _registered_pointer_root_sources(module, name, source.sequence_id, source.owner_id, 'int64')
    assert _registered_pointer_root_sources(module, name, call.args[-3].id, source.owner_id, 'int64') is None
    assert _registered_pointer_root_sources(module, name, source.sequence_id, call.args[-3].id, 'int64') is None
