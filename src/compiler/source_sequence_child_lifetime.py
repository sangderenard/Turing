"""Exact local child-pool storage membership; no selection or escape grant."""
from .concordance_declarations import (
    CELL_SET, DESCRIPTOR_CELL, SEQUENCE_DESCRIPTOR, SEQUENCE_MEMBER, SSA_VALUE,
    SequenceLifetimeKind,
)
from .source_sequence_lifetime import _fact, _mint_links, validated_sequence_lifetime


def validated_sequence_child_storage(module, function_name, sequence_id):
    """Return complete local child-member proof, or None when unproved.

    The owner comes from the existing lifetime admission; child membership
    comes from the descriptor's typed role ledger and exact descriptor mints,
    never heap accounting or the fact that two values happen to be pointers.
    """
    try:
        return _validate_child_storage(module, str(function_name), int(sequence_id))
    except (ValueError, KeyError, AttributeError, TypeError):
        return None


def _validate_child_storage(module, name, sid):
    from .ssa_record_return_state import function_scope_of
    from ..transmogrifier.dtype_layout import layout_for
    from ..transmogrifier.ssa import SSASequenceCapacityPolicy
    table = module.sequence_tables[name]
    function = module.functions[name]
    book, scope = table.book, function_scope_of(function)
    lifetime = validated_sequence_lifetime(module, name, sid)
    if lifetime is None:
        return None
    lifetime_cell, fact = lifetime
    descriptor = table.sequences[sid]
    pool = descriptor.child_table_pool
    if (fact.kind is not SequenceLifetimeKind.LOCAL_NEW or pool is None
            or descriptor.capacity_policy is not SSASequenceCapacityPolicy.DYNAMIC
            or fact.owner_cell is None or not pool.writable
            or descriptor.key_columns or descriptor.live_flags_value_id is not None
            or pool.key_columns or pool.live_flags_value_id is not None
            or any(x is not None for x in (pool.shape_value_id,pool.rank_value_id,pool.shape_stride_value_id))
            or not 0 <= pool.handle_column < len(descriptor.column_value_ids)
            or len(pool.column_dtypes) != len(pool.column_value_ids)):
        return None
    descriptor_cell = book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner,sid))
    causes = {p for p,_ in book.edges_into(descriptor_cell)}
    owner = book.latest_ref(SSA_VALUE,(scope,descriptor.heap_owner_value_id))
    owner_role = book.latest_ref(SEQUENCE_MEMBER,(table.owner,descriptor.heap_owner_value_id))
    if (owner is None or owner != fact.owner_cell or owner not in causes or owner_role is None
            or tuple(_fact(book,owner_role)) != ((sid,('heap_owner_value_id',)),)
            or descriptor_cell not in {p for p,_ in book.edges_into(owner_role)}):
        return None
    members = [(value_id, ('child_pool','column',index), dtype, True)
               for index,(value_id,dtype) in enumerate(zip(pool.column_value_ids,pool.column_dtypes))]
    members.extend(((pool.length_value_id,('child_pool','length_value_id'),'int64',True),
                    (pool.capacity_value_id,('child_pool','capacity_value_id'),'int64',False),
                    (pool.row_stride_value_id,('child_pool','row_stride_value_id'),'int64',False)))
    if pool.status_value_id is not None:
        members.append((pool.status_value_id,('child_pool','status_value_id'),'int',True))
    ids = [value_id for value_id,_,_,_ in members]
    outer = {*descriptor.column_value_ids,descriptor.length_address_id,descriptor.capacity_value_id,
             descriptor.status_address_id,descriptor.heap_owner_value_id}
    if len(set(ids)) != len(ids) or outer.intersection(ids):
        return None
    proof = {lifetime_cell,descriptor_cell,fact.owner_cell,fact.contract_cell,owner,owner_role}
    for value_id,role,dtype,arena in members:
        cell = book.latest_ref(SSA_VALUE,(scope,value_id))
        role_cell = book.latest_ref(SEQUENCE_MEMBER,(table.owner,value_id))
        if (cell is None or cell not in causes or role_cell is None
                or tuple(_fact(book,role_cell)) != ((sid,role),)
                or descriptor_cell not in {p for p,_ in book.edges_into(role_cell)}
                or not _mint_links(book,cell,fact.contract_cell,fact.column_cells[0])):
            return None
        physical = _fact(book,cell)
        layout = layout_for(dtype)
        if arena:
            if (physical.dtype != 'ptr' or tuple(physical.shape or ())
                    or physical.physical_dtype != layout.name
                    or physical.storage != 'span' or physical.rank != 1 or not physical.arena
                    or physical.device is not None):
                return None
        elif (physical.dtype != 'int64' or tuple(physical.shape or ())
              or physical.physical_dtype != 'int64' or physical.arena
              or physical.storage != 'scalar' or physical.rank != 0
              or physical.device is not None):
            return None
        # Retain the exact same-row history through the birth, not a new
        # storage identity inferred from a matching native type.
        pending,seen,births = [cell],set(),[]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            mint = book.mint_of(current)
            if mint is not None:
                births.append(mint)
            else:
                pending.extend(p for p,_ in book.edges_into(current)
                               if p.page == SSA_VALUE and p.row == cell.row)
        if len(births) != 1 or births[0][0] != DESCRIPTOR_CELL:
            return None
        bundle, = births[0][1]
        if bundle.page != CELL_SET:
            return None
        proof.update((cell,role_cell,bundle,*seen,*(p for p,_ in book.edges_into(bundle))))
    return tuple(sorted(proof,key=lambda c:repr(c.key)))
