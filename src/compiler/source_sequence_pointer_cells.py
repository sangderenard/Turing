"""Native pointer-cell extents, independent of dynamic sequence payload extents."""
from dataclasses import dataclass
from .concordance_declarations import (
    CANONICAL_VALUE, SEQUENCE_DESCRIPTOR, SEQUENCE_CONTRACT_CONCORDANCE, SSA_VALUE,
)
from .source_sequence_lifetime import _fact, _mint_links
from .ssa_record_return_state import function_scope_of
from ..transmogrifier.ssa import SSASequenceCapacityPolicy


@dataclass(frozen=True)
class SequencePointerCell:
    value_id: int
    proof_cells: tuple
    element_count: int = 1


def sequence_pointer_cell(module, function_name, value_id):
    """Prove one native pointer slot; do not prove payload capacity or ownership."""
    function = module.functions.get(str(function_name))
    table = getattr(module, "sequence_tables", {}).get(str(function_name))
    if function is None or table is None:
        return None
    value_id = int(value_id)
    formal = next((v for v in function.args if int(v.id) == value_id), None)
    if formal is None or str(formal.dtype) != "ptr" or tuple(formal.shape or ()):
        return None
    candidates = [d for d in table.sequences.values() if value_id in d.column_value_ids]
    if len(candidates) != 1:
        return None
    descriptor = candidates[0]
    if descriptor.capacity_policy is not SSASequenceCapacityPolicy.DYNAMIC or descriptor.heap_owner_value_id is None:
        return None
    book, scope = table.book, function_scope_of(function)
    declared = book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner, int(descriptor.sequence_id)))
    if declared is None or _fact(book, declared) != descriptor:
        return None
    contracts = tuple(c for c, _ in book.edges_into(declared) if c.page == SEQUENCE_CONTRACT_CONCORDANCE)
    if len(contracts) != 1 or contracts[0].row != (str(function_name), int(descriptor.sequence_id)):
        return None
    contract = contracts[0]
    source = tuple(c for c, _ in book.edges_into(contract) if c.page == CANONICAL_VALUE)
    if len(source) != 1 or int(source[0].row[1]) != int(descriptor.sequence_id):
        return None
    def member(i):
        return book.latest_ref(SSA_VALUE, (scope, int(i)))
    column, data = member(value_id), member(descriptor.column_value_ids[0])
    owner = member(descriptor.heap_owner_value_id)
    length, capacity = member(descriptor.length_address_id), member(descriptor.capacity_value_id)
    if any(c is None for c in (column, data, owner, length, capacity)):
        return None
    native, owner_fact = _fact(book,column), _fact(book,owner)
    if (native.dtype != "ptr" or tuple(native.shape or ())
            or owner_fact.dtype != "ptr" or tuple(owner_fact.shape or ()) != (1,)):
        return None
    # These are members minted for THIS declared arena, not an unrelated
    # pointer or an incoming user span which happens to have one element.
    if any(not _mint_links(book,c,contract,data) for c in (owner,length,capacity)):
        return None
    if value_id != int(descriptor.column_value_ids[0]) and not _mint_links(book,column,contract,data):
        return None
    return SequencePointerCell(value_id, tuple(dict.fromkeys((
        declared,contract,*source,column,data,owner,length,capacity))))
