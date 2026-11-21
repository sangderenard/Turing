"""Native call receipts preserve effects even when helper and literals repeat."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from tests.test_owned_record_transfer_plan import _lower, SOURCE
from src.compiler.concordance_declarations import (
    SSA_VALUE, CELL_SET, REGION_CALL_RESULT, CONTROL_BLOCK,
    CANONICAL_VALUE, FORMAL_ACTUAL_OCCURRENCE)
from src.compiler.ssa_record_return_state import function_scope_of
from src.compiler.precompile_to_ssa import _ControlSSABuilder


@pytest.mark.parametrize('equal_literals', [False, True])
def test_append_result_mint_names_exact_effect_and_call_occurrence(equal_literals):
    source = SOURCE
    if equal_literals:
        source = source.replace('out.append(i)', 'out.append(7)').replace(
            'out.append(-7)', 'out.append(7)').replace('out.append(-i)', 'out.append(7)')
    module, name = _lower(source)
    function = module.functions[name]
    book = module.metadata['identity_book']
    calls = []
    for label, block in function.blocks.items():
        for index, op in enumerate(block.instrs):
            if op.attributes.get('ssa_sequence_operation') != 'append':
                continue
            result = book.latest_ref(SSA_VALUE, (function_scope_of(function), op.res.id))
            transform, operands = book.mint_of(result)
            assert transform == REGION_CALL_RESULT and len(operands) == 1
            bundle = operands[0]
            assert bundle.page == CELL_SET
            causes = {p for p, _ in book.edges_into(bundle)}
            controls = [p for p in causes if p.page == CONTROL_BLOCK]
            effects = [p for p in causes if p.page == CANONICAL_VALUE
                       and p.row[-1] == op.attributes['source_effect_node_id']]
            assert len(controls) == len(effects) == 1
            fact = book.page(controls[0].page).cell(controls[0].row, controls[0].column)
            matching = [m for m in fact.sequence_mutations if m[3] == effects[0]]
            assert len(matching) == 1
            callee = module.functions[op.attributes['callee']]
            for position, (formal, actual) in enumerate(zip(callee.args, op.args)):
                row = (callee.name, formal.id, name, label, index, position, function_scope_of(function))
                occurrence = book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row)
                assert occurrence is not None
                assert book.page(occurrence.page).cell(occurrence.row, occurrence.column) == actual.id
                actual_cell = book.latest_ref(SSA_VALUE, (function_scope_of(function), actual.id))
                assert actual_cell in causes
                assert actual_cell in {p for p, _ in book.edges_into(occurrence)}
            calls.append((callee.name, effects[0], controls[0], matching[0]))
    assert len(calls) == 4
    shared = [item for item in calls if sum(c[0] == item[0] for c in calls) > 1]
    assert len(shared) == 2 and shared[0][1] != shared[1][1]

    # Missing, duplicated or wrong active control provides no added authority.
    _, effect, control, mutation_tuple = shared[0]
    target, operation, args, _, policy, kind, publication = mutation_tuple
    canonical = {p.row[-1]: p for p in (target, effect, *args)}
    mutation = SimpleNamespace(sequence_value_id=target.row[-1], operator=operation,
        argument_value_ids=tuple(p.row[-1] for p in args), effect_node_id=effect.row[-1],
        policy=policy, argument_kind=kind, object_contract_cell=publication)
    builder = SimpleNamespace(_control_cells=[control], _book=lambda: book,
                              _canonical_cell=lambda value: canonical.get(value))
    read = lambda: _ControlSSABuilder._sequence_mutation_call_sources(builder, mutation)
    assert read() == (effect, control)
    builder._control_cells = []
    assert read() == ()
    builder._control_cells = [shared[1][2]]
    assert read() == ()
    builder._control_cells = [control]
    page = book.page(control.page); key = (control.row, control.column)
    old = page.cells[key]
    try:
        page.cells[key] = replace(old, sequence_mutations=(*old.sequence_mutations, mutation_tuple))
        assert read() == ()
    finally:
        page.cells[key] = old
    canonical.pop(effect.row[-1])
    assert read() == ()
