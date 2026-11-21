"""Native literal materialization preserves occurrence identity, not value equality."""
from tests.test_owned_record_primitive_elements import SOURCE, lower
from src.compiler.concordance_declarations import CONTROL_BLOCK
from src.compiler.ssa_record_return_state import ssa_value_identity_cell


def _actual_argument(book, actual, source):
    # Follow only exact same-value SSA revisions and CONTROL_CONST sources.
    # Equal payloads are intentionally not considered.
    from src.compiler.concordance_declarations import CANONICAL_VALUE, CELL_SET, SSA_VALUE, CONTROL_CONST
    from src.compiler.source_sequence_lifetime import _same_origin
    pending, seen, matched = [actual], set(), False
    while pending:
        cell = pending.pop()
        if cell in seen:
            continue
        seen.add(cell)
        if cell.page == CANONICAL_VALUE:
            if not _same_origin(book, cell, source):
                return None
            matched = True
        elif cell.page == CELL_SET:
            parents = tuple(p for p, _ in book.edges_into(cell))
            if set(book.page(cell.page).cell(cell.row,cell.column)) != {p.key for p in parents}:
                return None
            pending.extend(p for p in parents if p.page == CANONICAL_VALUE)
        elif cell.page == SSA_VALUE:
            mint = book.mint_of(cell)
            if mint is not None:
                if mint[0] != CONTROL_CONST:
                    return None
                pending.extend(mint[1])
            else:
                pending.extend(p for p, _ in book.edges_into(cell)
                    if p.page == CANONICAL_VALUE or (p.page == SSA_VALUE and p.row == cell.row))
        else:
            return None
    return seen if matched else None


def test_repeated_append_literals_keep_distinct_source_occurrences():
    module, name, _ = lower(SOURCE.replace('ITEM', 'int').replace('VALUE', '7'))
    book = module.sequence_tables[name].book
    function = module.functions[name]
    arguments = {}
    for row in book.page(CONTROL_BLOCK).rows():
        cell = book.latest_ref(CONTROL_BLOCK, row)
        fact = book.page(cell.page).cell(cell.row, cell.column)
        for target, operation, args, effect, policy, kind, publication in fact.sequence_mutations:
            if operation == 'append':
                arguments[int(effect.row[-1])] = args[0]
    occurrences = []
    for block in function.blocks.values():
        for op in block.instrs:
            if op.op == 'Call' and op.attributes.get('ssa_sequence_operation') == 'append':
                argument = arguments[op.attributes['source_effect_node_id']]
                actual = ssa_value_identity_cell(function, op.args[-1].id, book=book)
                assert _actual_argument(book, actual, argument) is not None
                occurrences.append((actual, argument))
    assert len(occurrences) == 4
    assert len({arg for _, arg in occurrences}) == 4
    for actual, argument in occurrences:
        for _, other in occurrences:
            if other != argument:
                assert _actual_argument(book, actual, other) is None
