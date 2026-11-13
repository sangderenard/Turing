"""Read exact constructor-return sequence sources without authorizing ownership transfer.

This is an inspection/proof boundary, not an output ABI: selected owner
escape and cleanup remain obligations of the caller. No ledger is mutated.
"""
from dataclasses import dataclass
from .identity_concordance import Ref


@dataclass(frozen=True)
class OwnedRecordReturnSource:
    record_id: int
    field_name: str
    sequence_id: int
    return_slot_cell: Ref
    constructor_operands_cell: Ref
    source_cell: Ref
    descriptor_cell: Ref
    record_cell: Ref
    owner_id: int


def owned_record_return_sources(module, function_name):
    """Join explicit keyword constructor fields to exact returned resident sequences.

    Positional constructors are intentionally not inferred from field order.
    A source proof here does not establish an owner move or borrowed lifetime.
    """
    from .concordance_declarations import (
        RETURN_SITE_SLOT, PROCESS_GRAPH_OPERANDS, RECORD_DESCRIPTOR,
        CANONICAL_VALUE,
    )
    from .concorded_literals import origin_cell
    from .identity_concordance import Ref
    from .ssa_storage_requirements import _sequence_descriptor_identity
    function = module.functions.get(str(function_name))
    sequences = (getattr(module, 'sequence_tables', {}) or {}).get(str(function_name))
    records = (getattr(module, 'record_tables', {}) or {}).get(str(function_name))
    if function is None or sequences is None or records is None:
        return ()
    book = sequences.book
    returned = {int(v.id) for block in function.blocks.values()
                for op in block.instrs if op.op == 'Ret'
                and not op.attributes.get('fault_exit') for v in op.args}
    slots = book.page(RETURN_SITE_SLOT)
    result = []
    for record_id, record in records.records.items():
        record_cell = book.latest_ref(RECORD_DESCRIPTOR, (records.owner, int(record_id)))
        if record_cell is None:
            continue
        for field in record.fields:
            ids = tuple(map(int, field.value_ids))
            sequence_id = field.sequence_id if field.sequence_id is not None else (ids[0] if len(ids) == 1 else None)
            if sequence_id is None or int(sequence_id) not in returned:
                continue
            joined = _sequence_descriptor_identity(sequences, int(sequence_id), ingestion_origin=False)
            if joined is None:
                continue
            descriptor, descriptor_cell, _contract, source = joined
            if descriptor.heap_owner_value_id is None or len(descriptor.column_value_ids) != 1:
                continue
            matches = []
            for row in slots.rows():
                slot_cell = book.latest_ref(RETURN_SITE_SLOT, row)
                constructor = slots.cell(slot_cell.row, slot_cell.column)
                if not isinstance(constructor, Ref) or constructor.page != CANONICAL_VALUE:
                    continue
                if int(constructor.row[-1]) != int(record_id):
                    continue
                operands_cell = book.latest_ref(PROCESS_GRAPH_OPERANDS, constructor.row)
                if operands_cell is None:
                    continue
                if constructor not in {parent for parent, _stage in book.edges_into(operands_cell)}:
                    continue
                # The record descriptor must descend from this very authored
                # constructor, not merely reuse a numeric id from another scope.
                pending, seen, constructor_proven = [record_cell], set(), False
                while pending:
                    current = pending.pop()
                    if current in seen:
                        continue
                    seen.add(current)
                    if current.page == CANONICAL_VALUE:
                        try:
                            if origin_cell(book, current) == origin_cell(book, constructor):
                                constructor_proven = True
                                break
                        except ValueError:
                            pass
                    pending.extend(parent for parent, _stage in book.edges_into(current))
                if not constructor_proven:
                    continue
                operands = book.page(PROCESS_GRAPH_OPERANDS).cell(operands_cell.row, operands_cell.column)
                candidates = [cell for cell, role in operands if role == 'kw:' + field.name]
                if len(candidates) != 1:
                    continue
                try:
                    same = origin_cell(book, candidates[0]) == origin_cell(book, source)
                except ValueError:
                    continue
                if not same:
                    continue
                matches.append(OwnedRecordReturnSource(
                    int(record_id), field.name, int(sequence_id), slot_cell,
                    operands_cell, candidates[0], descriptor_cell, record_cell,
                    int(descriptor.heap_owner_value_id),
                ))
            if len(matches) == 1:
                result.append(matches[0])
    return tuple(result)
