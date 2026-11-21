"""Exact descriptor lifetime births, separate from any public owner transfer."""
from .concordance_declarations import (
    CANONICAL_VALUE, CELL_SET, CONTROL_BLOCK, DESCRIPTOR_CELL, LOOP_RESULT_PORT,
    NAME_BINDING, NATIVE_OBJECT_SOURCE, PARAMETER_ABI_KIND,
    PARAMETER_ABI_DECLARATION, PARAMETER_ABI_UNDECLARED, ParameterAbiKind,
    SEQUENCE_STORAGE_LIFETIME, SequenceStorageLifetime, SequenceLifetimeKind,
    SOURCE_SEQUENCE_CONSTRUCTOR_SNAPSHOT, SSA_VALUE, STORAGE_ROOT_STAGE,
    SEQUENCE_CONTRACT_CONCORDANCE, SEQUENCE_DESCRIPTOR,
)
from .identity_concordance import Derived, Mode, Unresolved


def _fact(book, cell):
    if cell is None:
        raise ValueError('missing lifetime evidence')
    return book.page(cell.page).cell(cell.row, cell.column)


def _causes(fact):
    return tuple(dict.fromkeys(cell for cell in (
        fact.source_cell, fact.contract_cell, *fact.column_cells,
        fact.length_cell, fact.capacity_cell, fact.owner_cell,
        *fact.birth_cells, fact.parent_cell) if cell is not None))


def _ast_kind(book, source):
    fact = _fact(book, source)
    if source.page != CANONICAL_VALUE or not isinstance(fact, tuple):
        return None
    try:
        return fact[fact.index('field:node.ast') + 1]
    except (ValueError, IndexError):
        return None


def _same_origin(book, a, b):
    from .concorded_literals import origin_cell
    if a == b:
        return True
    try:
        return origin_cell(book, a) == origin_cell(book, b)
    except ValueError:
        return False


def _local_birth(book, source):
    if _ast_kind(book, source) == 'value:List':
        return (source,)
    for row in book.page(SOURCE_SEQUENCE_CONSTRUCTOR_SNAPSHOT).rows():
        cell = book.latest_ref(SOURCE_SEQUENCE_CONSTRUCTOR_SNAPSHOT, row)
        fact = _fact(book, cell)
        if _same_origin(book, fact.destination_cell, source):
            return (cell,)
    mint = book.mint_of(source)
    if not mint or mint[0] != LOOP_RESULT_PORT or len(mint[1]) != 1:
        return ()
    loop = mint[1][0]
    fact = _fact(book, source)
    if not isinstance(fact, tuple) or fact[:1] != ('LoopResult',):
        return ()
    matches = []
    for row in book.page(CONTROL_BLOCK).rows():
        control_cell = book.latest_ref(CONTROL_BLOCK, row)
        for mutation in getattr(_fact(book, control_cell), 'sequence_mutations', ()):
            target, operation, arguments, effect, policy, kind, publication = mutation
            if (target != source or effect != loop or operation != 'append'
                    or publication is None or publication.page != NATIVE_OBJECT_SOURCE
                    or publication.row[-1] != 'publication'):
                continue
            materializer = book.latest_ref(CANONICAL_VALUE, publication.row[:2])
            if materializer is None:
                continue
            # The compiler's retained list comprehension has an explicit
            # materializer, not a loop-carried alias of an incoming list.
            if _ast_kind(book, materializer) not in ('value:ListComp', 'value:GeneratorExp'):
                continue
            matches.append((control_cell, publication, materializer, loop))
    return matches[0] if len(matches) == 1 else ()


def _mint_links(book, member, contract, data):
    """Follow only revisions of this member to its exact descriptor mint."""
    pending, seen, births = [member], set(), []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        mint = book.mint_of(current)
        if mint is not None:
            births.append(mint)
        else:
            pending.extend(parent for parent, _ in book.edges_into(current)
                           if parent.page == member.page and parent.row == member.row)
    if len(births) != 1 or births[0][0] != DESCRIPTOR_CELL:
        return False
    operands = births[0][1]
    if len(operands) != 1 or operands[0].page != CELL_SET:
        return False
    bundle = operands[0]
    parents = {parent for parent, _ in book.edges_into(bundle)}
    if {parent.key for parent in parents} != set(_fact(book, bundle)):
        return False
    return (any(parent.page == contract.page and parent.row == contract.row for parent in parents)
            and any(parent.page == data.page and parent.row == data.row for parent in parents))


def _validate(book, cell, active=None):
    active = set() if active is None else active
    if cell in active:
        raise ValueError('cyclic lifetime inheritance')
    active.add(cell)
    fact = _fact(book, cell)
    if not isinstance(fact, SequenceStorageLifetime):
        raise ValueError('wrong lifetime fact')
    if cell.page != SEQUENCE_STORAGE_LIFETIME:
        raise ValueError('wrong lifetime page')
    if not fact.column_cells or fact.source_cell.page != CANONICAL_VALUE:
        raise ValueError('lifetime lacks source or arena')
    scope, sequence_id = cell.row
    members = (*fact.column_cells, fact.length_cell, fact.capacity_cell,
               *((fact.owner_cell,) if fact.owner_cell is not None else ()))
    if any(member.page != SSA_VALUE or member.row[0] != scope for member in members):
        raise ValueError('cross-scope lifetime member')
    if len({member.row for member in members}) != len(members):
        raise ValueError('aliased lifetime descriptor members')
    if set(_causes(fact)) != {parent for parent,_ in book.edges_into(cell)}:
        raise ValueError('missing lifetime proof causes')
    for cause in _causes(fact):
        _fact(book, cause)
    if fact.contract_cell.page != SEQUENCE_CONTRACT_CONCORDANCE or fact.contract_cell.row[1] != sequence_id:
        raise ValueError('lifetime has wrong descriptor contract')
    declared = tuple(parent for parent,_ in book.edges_into(fact.contract_cell)
                     if parent.page == CANONICAL_VALUE)
    if len(declared) != 1 or not _same_origin(book, declared[0],fact.source_cell):
        raise ValueError('lifetime source differs from exact contract declaration')
    for member in (fact.length_cell,fact.capacity_cell,
                   *((fact.owner_cell,) if fact.owner_cell is not None and fact.kind is not SequenceLifetimeKind.ADOPTED else ())):
        if not _mint_links(book,member,fact.contract_cell,fact.column_cells[0]):
            raise ValueError('lifetime member mint is not this descriptor arena/contract')
    if fact.kind is SequenceLifetimeKind.LOCAL_NEW:
        if fact.owner_cell is None or fact.parent_cell is not None:
            raise ValueError('local birth lacks unique owner')
        owner = _fact(book, fact.owner_cell)
        mint = book.mint_of(fact.owner_cell)
        if owner.dtype != 'ptr' or tuple(owner.shape or ()) != (1,) or not mint or mint[0] != DESCRIPTOR_CELL:
            raise ValueError('local owner lacks descriptor birth')
        if not fact.birth_cells or fact.birth_cells != _local_birth(book, fact.source_cell):
            raise ValueError('local lifetime has no affirmative fresh source birth')
    elif fact.kind is SequenceLifetimeKind.BORROWED:
        if len(fact.birth_cells) != 2 or fact.parent_cell is not None:
            raise ValueError('borrowed lifetime lacks signature proof')
        kind, binding = fact.birth_cells
        if (kind.page != PARAMETER_ABI_KIND or binding.page != NAME_BINDING
                or binding.row != (*kind.row, 0)):
            raise ValueError('borrowed kind does not identify signature binding')
        kind_fact = _fact(book, kind)
        kind_mint = book.mint_of(kind)
        declared = (isinstance(kind_fact, ParameterAbiKind)
                    and kind_mint == (PARAMETER_ABI_DECLARATION, (binding,)))
        undeclared = (isinstance(kind_fact, Unresolved)
                      and kind_fact.reason == PARAMETER_ABI_UNDECLARED
                      and kind_fact.read == (binding,)
                      and binding in {parent for parent, _ in book.edges_into(kind)})
        if not (declared or undeclared):
            raise ValueError('borrowed kind does not identify signature binding')
        bound = _fact(book,binding)
        canonical = book.latest_ref(CANONICAL_VALUE, (binding.row[0], int(bound.value_id)))
        if canonical is None or not _same_origin(book, canonical, fact.source_cell):
            raise ValueError('borrowed source does not match parameter')
    elif fact.kind is SequenceLifetimeKind.ADOPTED:
        if fact.parent_cell is None or fact.birth_cells:
            raise ValueError('adopted lifetime lacks original owner receipt')
        parent = _validate(book, fact.parent_cell, active)
        if (fact.owner_cell is None or parent.owner_cell is None
                or fact.owner_cell.row != parent.owner_cell.row
                or fact.column_cells[0].row != parent.column_cells[0].row):
            raise ValueError('adoption changed arena or owner')
    else:
        raise ValueError('unsupported lifetime kind')
    active.remove(cell)
    return fact


def publish_sequence_lifetime(builder, descriptor, *, parameter_kind=None, adopted_owner=None):
    """Publish only affirmative supported births, after borrowed ABI admission."""
    book, scope = builder._book(), builder._scope()
    row = (scope, int(descriptor.sequence_id))
    if book.latest_ref(SEQUENCE_STORAGE_LIFETIME, row) is not None:
        return book.latest_ref(SEQUENCE_STORAGE_LIFETIME, row)
    source = builder._canonical_cell(int(descriptor.sequence_id))
    contract = builder._sequence_contract_cell(int(descriptor.sequence_id))
    if source is None or contract is None:
        return None
    def member(value_id):
        return book.latest_ref(SSA_VALUE, (scope, int(value_id)))
    columns = tuple(member(value_id) for value_id in descriptor.column_value_ids)
    length, capacity = member(descriptor.length_address_id), member(descriptor.capacity_value_id)
    owner = None if descriptor.heap_owner_value_id is None else member(descriptor.heap_owner_value_id)
    if any(cell is None for cell in (*columns,length,capacity)):
        return None
    parent = None
    if parameter_kind is None:
        parameter_name = next((name for name in builder.parameter_names
            if builder.value_name_histories.get(name, ())
            and int(builder.value_name_histories[name][0]) == int(descriptor.sequence_id)), None)
        if parameter_name is not None and builder.lexical_read_scope is not None:
            parameter_kind = book.latest_ref(PARAMETER_ABI_KIND,
                (tuple(builder.lexical_read_scope), str(parameter_name)))
    if parameter_kind is not None:
        binding = book.latest_ref(NAME_BINDING, (*parameter_kind.row,0))
        if binding is None:
            return None
        kind, births = SequenceLifetimeKind.BORROWED, (parameter_kind,binding)
    elif adopted_owner is not None or (owner is not None and not _mint_links(book, owner, contract, columns[0])):
        candidates = []
        for other_row in book.page(SEQUENCE_STORAGE_LIFETIME).rows():
            other = book.latest_ref(SEQUENCE_STORAGE_LIFETIME,other_row)
            fact = _validate(book,other)
            if (owner is not None and fact.owner_cell is not None
                    and owner.row == fact.owner_cell.row
                    and columns[0].row == fact.column_cells[0].row):
                candidates.append(other)
        if len(candidates) != 1:
            return None
        kind, births, parent = SequenceLifetimeKind.ADOPTED, (), candidates[0]
    else:
        if owner is None:
            return None
        births = _local_birth(book,source)
        if not births:
            return None
        kind = SequenceLifetimeKind.LOCAL_NEW
    fact = SequenceStorageLifetime(kind,source,contract,columns,length,capacity,owner,births,parent)
    return book.post(SEQUENCE_STORAGE_LIFETIME,row,fact,stage=STORAGE_ROOT_STAGE,
                     provenance=Derived(_causes(fact)),mode=Mode.CONCORD)


def validated_sequence_lifetime(module, function_name, sequence_id):
    """Join immutable birth members to the current descriptor, without escape."""
    from .ssa_record_return_state import function_scope_of
    table = getattr(module,'sequence_tables',{}).get(str(function_name))
    function = module.functions.get(str(function_name))
    if table is None or function is None:
        return None
    book = table.book
    cell = book.latest_ref(SEQUENCE_STORAGE_LIFETIME,(function_scope_of(function),int(sequence_id)))
    descriptor = table.sequences.get(int(sequence_id))
    if cell is None or descriptor is None:
        return None
    fact = _validate(book,cell)
    descriptor_cell = book.latest_ref(SEQUENCE_DESCRIPTOR,(table.owner,int(sequence_id)))
    if descriptor_cell is None or _fact(book,descriptor_cell) != descriptor:
        raise ValueError('unposted current lifetime descriptor')
    contracts = tuple(parent for parent,_ in book.edges_into(descriptor_cell)
                      if parent.page == SEQUENCE_CONTRACT_CONCORDANCE)
    if len(contracts) != 1 or contracts[0].row != fact.contract_cell.row:
        raise ValueError('current descriptor has different lifetime contract')
    if (tuple(c.row[1] for c in fact.column_cells) != tuple(descriptor.column_value_ids)
            or fact.length_cell.row[1] != descriptor.length_address_id
            or fact.capacity_cell.row[1] != descriptor.capacity_value_id
            or (None if fact.owner_cell is None else fact.owner_cell.row[1]) != descriptor.heap_owner_value_id):
        raise ValueError('lifetime and current descriptor disagree')
    return cell,fact


def sequence_lifetime_findings(module):
    from .identity_concordance import Finding
    book = dict(getattr(module,'metadata',{}) or {}).get('identity_book')
    if book is None:
        return ()
    findings=[]
    for row in book.page(SEQUENCE_STORAGE_LIFETIME).rows():
        try:
            _validate(book,book.latest_ref(SEQUENCE_STORAGE_LIFETIME,row))
            from .ssa_record_return_state import function_scope_of
            tables = getattr(module, 'sequence_tables', {})
            matches = [name for name, function in module.functions.items()
                       if function_scope_of(function) == row[0]
                       and name in tables and int(row[1]) in tables[name].sequences]
            if len(matches) != 1:
                raise ValueError('lifetime has no unique current descriptor owner')
            validated_sequence_lifetime(module, matches[0], row[1])
        except (ValueError,KeyError,AttributeError,TypeError) as error:
            findings.append(Finding('sequence-lifetime-unproven',str(row[0]),int(row[1]),str(error)))
    return tuple(findings)
