"""Exact address-Phi lineage; structural evidence is not ownership permission."""
from dataclasses import dataclass
from .identity_concordance import Ref


@dataclass(frozen=True)
class SequenceSelection:
    sequence_id: int
    descriptor_cell: Ref
    member_ids: tuple[int, ...]
    geometry_cell: Ref | None = None
    predecessors: tuple[str, ...] = ()
    arms: tuple['SequenceSelection', ...] = ()


def sequence_selection(module, function_name, sequence_id):
    """Return a concorded selection tree, with unclassified terminal descriptors.

    A leaf is NOT a claim of local ownership. Its lifetime admission is a
    separate obligation, including when an owner cell happens to be present.
    """
    from .concordance_declarations import (SSA_BLOCK_GEOMETRY, SEQUENCE_DESCRIPTOR,
                                           SEQUENCE_CONTRACT_CONCORDANCE, CANONICAL_VALUE)

    function = module.functions.get(str(function_name))
    table = (getattr(module, 'sequence_tables', {}) or {}).get(str(function_name))
    if function is None or table is None:
        return None
    book = table.book

    def members(d):
        return (*d.column_value_ids, d.length_address_id, d.capacity_value_id,
                *((d.status_address_id,) if d.status_address_id is not None else ()),
                d.heap_owner_value_id)

    def layout(d):
        return (len(d.column_value_ids), tuple(d.column_dtypes), tuple(d.column_shapes),
                d.capacity_policy, d.writable, d.status_address_id is not None)

    def join(sid, seen):
        if sid in seen:
            return None
        cell = book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner, sid))
        descriptor = table.sequences.get(sid)
        if cell is None or descriptor is None:
            return None
        contracts = [p for p, _ in book.edges_into(cell)
                     if p.page == SEQUENCE_CONTRACT_CONCORDANCE]
        if len(contracts) != 1 or int(contracts[0].row[-1]) != sid:
            return None
        # Current declaration identifies the resident. Historical mutation
        # operations are causes too, not alternative identities of its arena.
        sources = [p for p, _ in book.edges_into(contracts[0])
                   if p.page == CANONICAL_VALUE]
        if len(sources) != 1 or int(sources[0].row[-1]) != sid:
            return None
        if (descriptor.heap_owner_value_id is None or descriptor.child_table_pool is not None
                or descriptor.live_flags_value_id is not None or descriptor.key_columns):
            return None
        ids = tuple(map(int, members(descriptor)))
        producers = {}
        for label, block in function.blocks.items():
            for op in block.instrs:
                if op.res is not None and int(op.res.id) in ids:
                    if int(op.res.id) in producers:
                        return None
                    producers[int(op.res.id)] = (label, block, op)
        selected = [producers.get(i) for i in ids]
        phis = [p for p in selected if p is not None and p[2].op == 'Phi']
        if not phis:
            return SequenceSelection(sid, cell, ids)
        if len(phis) != len(ids) or len({p[0] for p in phis}) != 1:
            return None
        label, block, _ = phis[0]
        incoming = tuple(phis[0][2].attributes.get('incoming_blocks') or ())
        if len(incoming) != 2 or len(set(incoming)) != 2:
            return None
        if any(tuple(p[2].attributes.get('incoming_blocks') or ()) != incoming
               or len(p[2].args) != len(incoming) for p in phis):
            return None
        # Require the exact published block geometry, not just mutable IR.
        fact = (tuple((str(op.op), tuple(int(v.id) for v in op.args),
                       None if op.res is None else int(op.res.id),
                       tuple(op.attributes.get('incoming_blocks') or ()))
                      for op in block.instrs), tuple(block.successors))
        geometry = []
        page = book.page(SSA_BLOCK_GEOMETRY)
        for row in page.rows():
            if tuple(row)[-2:] == (str(function.name), str(label)):
                ref = book.latest_ref(SSA_BLOCK_GEOMETRY, row)
                if page.cell(ref.row, ref.column) == fact:
                    geometry.append(ref)
        if len(geometry) != 1:
            return None
        arms = []
        for index in range(len(incoming)):
            vector = tuple(int(p[2].args[index].id) for p in phis)
            candidates = [int(k) for k, d in table.sequences.items()
                          if tuple(members(d)) == vector and layout(d) == layout(descriptor)]
            if len(candidates) != 1:
                return None
            arm = join(candidates[0], seen | {sid})
            if arm is None:
                return None
            arms.append(arm)
        return SequenceSelection(sid, cell, ids, geometry[0], incoming, tuple(arms))

    return join(int(sequence_id), set())


@dataclass(frozen=True)
class LocalSelectionLifetimes:
    selection: SequenceSelection
    terminal_cells: tuple[Ref, ...]
    original_owner_ids: tuple[int, ...]


def local_selection_lifetimes(module, function_name, sequence_id):
    """Prove selected terminals are local-owned, without granting an escape.

    The owner IDs identify original owner cells, not a new allocation for
    the selector. A caller exporting multiple fields must additionally
    refuse intersecting owner sets until shared exports have their own ABI.
    """
    from .concordance_declarations import SequenceLifetimeKind
    from .source_sequence_lifetime import validated_sequence_lifetime
    tree = sequence_selection(module, function_name, sequence_id)
    if tree is None:
        return None
    table = module.sequence_tables[str(function_name)]
    cells, owners = [], []
    pending = [tree]
    while pending:
        node = pending.pop()
        if node.arms:
            pending.extend(reversed(node.arms))
            continue
        try:
            joined = validated_sequence_lifetime(module, function_name, node.sequence_id)
            if joined is None:
                return None
            cell, fact = joined
            terminal_cell = cell
            visited = set()
            while fact.kind == SequenceLifetimeKind.ADOPTED:
                if cell in visited or fact.parent_cell is None:
                    return None
                visited.add(cell)
                cell = fact.parent_cell
                fact = table.book.page(cell.page).cell(cell.row, cell.column)
            if fact.kind != SequenceLifetimeKind.LOCAL_NEW or fact.owner_cell is None:
                return None
            owner = int(fact.owner_cell.row[-1])
            if owner != node.member_ids[-1]:
                return None
        except (ValueError, KeyError, AttributeError, TypeError):
            return None
        cells.append(terminal_cell)
        owners.append(owner)
    return LocalSelectionLifetimes(tree, tuple(cells), tuple(dict.fromkeys(owners)))
