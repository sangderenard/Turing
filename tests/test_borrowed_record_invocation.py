from pathlib import Path
from dataclasses import replace
from contextlib import contextmanager

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.concordance_declarations import CANONICAL_VALUE, SSA_VALUE
from src.compiler.ssa_record_return_state import function_scope_of
from src.compiler.source_borrowed_record_invocation import borrowed_record_string_invocation_sources
from tests.owned_record_test_support import lower_before_owned_record_boundary


SOURCE = '''from dataclasses import dataclass
@dataclass
class Node:
    key: str
    other: str
def root(nodes: list[Node]) -> int:
    out: list[str] = []
    for node in nodes:
        out.append(node.key)
        out.append(node.key)
        out.append(node.other)
    return len(out)
'''


def test_final_invocation_revalidates_exact_builder_views(monkeypatch):
    import src.compiler.source_borrowed_record_spans as spans
    original = spans.borrowed_record_string_view
    captured = {}

    def inspect(builder, value_id):
        proof = original(builder, value_id)
        assert proof
        source = builder._book().latest_ref(CANONICAL_VALUE, (builder.lexical_read_scope, value_id))
        captured[source] = proof
        return proof

    monkeypatch.setattr(spans, 'borrowed_record_string_view', inspect)
    module, _, exports = lower_before_owned_record_boundary(
        SOURCE, 'root', extraction_contract=ExtractionContract(
            Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    name = exports[0]
    function = module.functions[name]
    book = module.sequence_tables[name].book
    scope = function_scope_of(function)
    calls = [op for block in function.blocks.values() for op in block.instrs
             if op.op == 'Call' and op.attributes.get('ssa_sequence_operation') == 'append_child_copy']
    assert len(calls) == 3 and len(captured) == 3
    admitted = []
    for call in calls:
        result = book.latest_ref(SSA_VALUE, (scope, call.res.id))
        proven = [(source, borrowed_record_string_invocation_sources(module, name, source, result))
                  for source in captured]
        matches = [(source, proof) for source, proof in proven if proof]
        assert len(matches) == 1
        source, proof = matches[0]
        admitted.append((call, source, result, proof))
        assert set(captured[source]) <= set(proof)
        assert borrowed_record_string_invocation_sources(module, name, result, source) is None


    @contextmanager
    def changed_fact(cell, replacement):
        page = book.page(cell.page)
        key = (cell.row, cell.column)
        old = page.cells[key]
        page.cells[key] = replacement
        try:
            yield
        finally:
            page.cells[key] = old

    def fact(cell):
        return book.page(cell.page).cell(cell.row, cell.column)

    from src.compiler.concordance_declarations import (
        ALIAS_APPLICATION, PLANNING_VALUE, SEQUENCE_DESCRIPTOR,
        CONTROL_VALUE_BINDING, BindingKind, BORROWED_RECORD_SPAN_ROLE,
        CELL_SET, FORMAL_ACTUAL_OCCURRENCE,
    )
    # Both live and retired recipes remain invocation-specific, including
    # the current physical call position rather than a historical index.
    for call, source, result, proof in admitted:
        def read():
            return borrowed_record_string_invocation_sources(module, name, source, result)
        original_args, original_callee = list(call.args), call.attributes['callee']
        try:
            call.args = original_args[:-1]
            assert read() is None, 'truncated call'
            call.args = original_args
            call.attributes['callee'] = 'unregistered_helper'
            assert read() is None, 'unregistered callee'
        finally:
            call.args, call.attributes['callee'] = original_args, original_callee
        block = next(b for b in function.blocks.values() if call in b.instrs)
        offset = block.instrs.index(call)
        try:
            block.instrs[offset - 1], block.instrs[offset] = block.instrs[offset], block.instrs[offset - 1]
            assert read() is None, 'moved current call'
        finally:
            block.instrs[offset - 1], block.instrs[offset] = block.instrs[offset], block.instrs[offset - 1]
        role = next(c for c in proof if c.page == BORROWED_RECORD_SPAN_ROLE)
        with changed_fact(role, replace(fact(role), parent_cell=source)):
            assert read() is None, 'foreign parent role'
        logical = next(c for c in captured[source] if c.page == CONTROL_VALUE_BINDING
                       and c.row[-1] == source.row[-1])
        with changed_fact(logical, replace(fact(logical), kind=BindingKind.RESTORED)):
            assert read() is None, 'substituted captured logical binding'
        row_binding = next(c for c in captured[source] if c.page == CONTROL_VALUE_BINDING
                           and c.row[-1] != source.row[-1])
        with changed_fact(row_binding, replace(fact(row_binding), ssa_value=fact(logical).ssa_value)):
            assert read() is None, 'wrong induction identity'
        latest_row = book.latest_ref(CONTROL_VALUE_BINDING, row_binding.row)
        with changed_fact(latest_row, replace(fact(latest_row), ssa_value=fact(logical).ssa_value)):
            assert read() is None, 'wrong restored pre-loop payload'
        for occurrence in (c for c in proof if c.page == FORMAL_ACTUAL_OCCURRENCE):
            with changed_fact(occurrence, -1):
                assert read() is None, 'missing current occurrence identity'
        assert read() == proof

    call, source, result, proof = admitted[1]
    def repeated():
        return borrowed_record_string_invocation_sources(module, name, source, result)
    aliases = [c for c in proof if c.page == ALIAS_APPLICATION]
    plans = [c for c in proof if c.page == PLANNING_VALUE]
    assert len(aliases) == len(plans) == 3
    for cell in aliases:
        old, resident, selected, reason, label = fact(cell)
        with changed_fact(cell, (old, resident, selected, 'unproved_alias', label)):
            assert repeated() is None, 'one missing alias arm'
    for cell in plans:
        with changed_fact(cell, -1):
            assert repeated() is None, 'one missing planning equivalence'
    old_descriptor = next(c for c in proof if c.page == SEQUENCE_DESCRIPTOR
                          and c.row[-1] == source.row[-1] and fact(c) is not None)
    removal = book.latest_ref(SEQUENCE_DESCRIPTOR, old_descriptor.row)
    assert fact(removal) is None and removal != old_descriptor
    with changed_fact(removal, fact(old_descriptor)):
        assert repeated() is None, 'no current retirement'
    with changed_fact(old_descriptor, replace(fact(old_descriptor), capacity_value_id=fact(old_descriptor).length_address_id)):
        assert repeated() is None, 'altered retired descriptor member'
    # Remove immutable old recipe evidence, not just the descriptor label.
    old_column = book.latest_ref(SSA_VALUE, (scope, fact(old_descriptor).column_value_ids[0]))
    old_mint = book.mint_of(old_column)
    bundle, = old_mint[1]
    assert bundle.page == CELL_SET
    with changed_fact(bundle, ()):
        assert repeated() is None, 'missing old ADDRESS mint operands'
    normalized = next(c for c in proof if c.page == SSA_VALUE and c.row[0] == scope
                      and c.column > 0 and any(stage.name == 'sequence_helper_value_types'
                                              for _, stage in book.edges_into(c)))
    formal = next(c for c, _ in book.edges_into(normalized)
                  if c.page == SSA_VALUE and c.row[0] != scope)
    with changed_fact(formal, replace(fact(formal), shape=())):
        assert repeated() is None, 'scalar helper formal cannot normalize pointer view'
    with changed_fact(normalized, replace(fact(normalized), arena=not fact(normalized).arena)):
        assert repeated() is None, 'changed physical role in SSA revision'
    assert repeated() == proof

    for receipt in (*aliases, *plans, removal):
        incoming = tuple(book.edges_into(receipt))
        assert incoming
        with monkeypatch.context() as patch:
            original_edges = book.edges_into
            patch.setattr(book, 'edges_into', lambda c, receipt=receipt, incoming=incoming:
                          incoming[1:] if c == receipt else original_edges(c))
            assert repeated() is None, 'missing exact alias/plan/retirement cause'
        assert repeated() == proof
    with monkeypatch.context() as patch:
        original_edges = book.edges_into
        patch.setattr(book, 'edges_into', lambda c:
                      tuple(e for e in original_edges(c) if e[0] != formal)
                      if c == normalized else original_edges(c))
        assert repeated() is None, 'missing exact helper formal revision cause'
    assert repeated() == proof

    from copy import copy
    foreign_book = copy(book)
    assert foreign_book is not book
    with monkeypatch.context() as patch:
        patch.setitem(module.metadata, 'identity_book', foreign_book)
        assert repeated() is None, 'cannot combine table and loop proofs from different books'
    original_view = fact(old_descriptor)
    with changed_fact(old_descriptor, replace(original_view,
            column_value_ids=(original_view.length_address_id,),
            length_address_id=original_view.column_value_ids[0])):
        assert repeated() is None, 'swapped historical descriptor roles'
    assert repeated() == proof

    # A call cannot pick one convenient binding from multiple captured
    # revisions, even if all cell-set facts and edges agree about the set.
    withdrawn = book.latest_ref(CONTROL_VALUE_BINDING, (scope, int(source.row[-1])))
    status_mint = book.mint_of(result)
    status_bundle, = status_mint[1]
    incoming = tuple(book.edges_into(status_bundle))
    assert withdrawn not in {p for p, _ in incoming}
    with changed_fact(status_bundle, (*fact(status_bundle), withdrawn.key)):
        with monkeypatch.context() as patch:
            original_edges = book.edges_into
            patch.setattr(book, 'edges_into', lambda c:
                          (*incoming, (withdrawn, incoming[0][1]))
                          if c == status_bundle else original_edges(c))
            assert repeated() is None, 'duplicate captured logical revisions'
    withdrawal = fact(withdrawn)
    with changed_fact(withdrawn, type(withdrawal)(withdrawal.reason, read=())):
        assert repeated() is None, 'withdrawal must retain exact loop/latch read proof'
    assert repeated() == proof
