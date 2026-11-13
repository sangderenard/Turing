"""Closed append-writer proof for primitive owned record sequences.

An annotation is not evidence about a stored value. This reader checks the
actual source arguments and their native uses; unsupported mutations refuse.
"""
from .concordance_declarations import (
    ADDRESS, CANONICAL_VALUE, CELL_SET, CONTROL_BLOCK, CONTROL_BLOCK_PLACEMENT,
    CONTROL_CONST, CONTROL_EXPRESSION, FORMAL_ACTUAL_OCCURRENCE,
    RECORD_OWNED_RETURN_PROJECTION, REGION_CALL_RESULT, PROCESS_GRAPH_OPERANDS, SEQUENCE_CONTRACT_CONCORDANCE,
    SOURCE_VALUE_CLASS, SSA_VALUE, SSA_CONSTANT_MATERIALIZATION, SSA_CONSTANT_HOIST, SSA_BLOCK, SSABlockKind,
)


def _fact(book, cell):
    return book.page(cell.page).cell(cell.row, cell.column)


def _actual_argument(book, actual, source):
    """Exact identity or native constant materialization, not numerical equality."""
    from .source_sequence_lifetime import _same_origin
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
            continue
        if cell.page == CELL_SET:
            members = tuple(p for p, _ in book.edges_into(cell))
            if set(_fact(book, cell)) != {p.key for p in members}:
                return None
            pending.extend(p for p in members if p.page == CANONICAL_VALUE)
            continue
        if cell.page != SSA_VALUE:
            return None
        mint = book.mint_of(cell)
        if mint is not None:
            if mint[0] != CONTROL_CONST:
                return None
            pending.extend(mint[1])
        else:
            causes = tuple(p for p, _ in book.edges_into(cell)
                           if p.page == CANONICAL_VALUE or (p.page == SSA_VALUE and p.row == cell.row))
            if not causes:
                return None
            pending.extend(causes)
    return tuple(seen) if matched else None


def _ssa_history(book, cell):
    pending, seen = [cell], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(p for p, _ in book.edges_into(current)
                       if p.page == SSA_VALUE and p.row == cell.row)
    return seen


def _call_occurrence_sources(book, result, actuals, control, effect, callee):
    history = _ssa_history(book, result)
    minted = [(cell, book.mint_of(cell)) for cell in history if book.mint_of(cell) is not None]
    if len(minted) != 1:
        return None
    cell, mint = minted[0]
    if mint[0] != REGION_CALL_RESULT or len(mint[1]) != 1:
        return None
    bundle = mint[1][0]
    if bundle.page != CELL_SET:
        return None
    members = {p for p, _ in book.edges_into(bundle)}
    if set(_fact(book, bundle)) != {p.key for p in members} or not {control, effect} <= members:
        return None
    blocks = {book.latest_ref(SSA_BLOCK,(callee.name,callee.name,str(label))) for label in callee.blocks}
    if not blocks or None in blocks or any(_fact(book,c) != SSABlockKind.SEQUENCE_HELPER for c in blocks):
        return None
    if {c for c in members if c.page == SSA_BLOCK} != blocks:
        return None
    refs = set(history) | {bundle, control, effect} | blocks
    for actual in actuals:
        chain = _ssa_history(book, actual)
        if not chain.intersection(members):
            return None
        refs.update(chain)
    return refs


def _zero_receipt(book, function, definitions, value_id):
    from .ssa_record_return_state import ssa_value_identity_cell, function_scope_of
    entries = definitions.get(int(value_id), ())
    if len(entries) != 1:
        return None
    label, op = entries[0]
    if op.op != 'Const' or op.args or type(op.attributes.get('value')) is not int or op.attributes['value'] != 0:
        return None
    scope = function_scope_of(function)
    actual = ssa_value_identity_cell(function,value_id,book=book)
    if actual is None:
        return None
    history = _ssa_history(book,actual)
    candidates = []
    for row in book.page(SSA_CONSTANT_MATERIALIZATION).rows():
        if row[:2] != (scope,function.name) or int(row[-1]) != int(value_id):
            continue
        cell = book.latest_ref(SSA_CONSTANT_MATERIALIZATION,row)
        if _fact(book,cell) != ('Const',str(op.res.dtype),'0'):
            continue
        if not history.intersection(p for p,_ in book.edges_into(cell)):
            continue
        refs = {cell,actual}
        if row[2] != label:
            moves = []
            for move_row in book.page(SSA_CONSTANT_HOIST).rows():
                if (move_row[0] != (scope,function.name) or int(move_row[1]) != int(value_id)
                        or move_row[2] != row[2] or move_row[4] != label):
                    continue
                move = book.latest_ref(SSA_CONSTANT_HOIST,move_row)
                fact = _fact(book,move)
                causes = {p for p,_ in book.edges_into(move)}
                if (len(fact) != 2 or fact[1] != 'operand_free_immutable_definition'
                        or not history.intersection(causes)):
                    continue
                geometry = move_row[5]
                instructions = _fact(book,geometry)[0]
                if not 0 <= int(move_row[3]) < len(instructions):
                    continue
                if instructions[int(move_row[3])] != ('Const',(),int(value_id),()):
                    continue
                bundles = [p for p in causes if p.page == CELL_SET]
                if len(bundles) != 1:
                    continue
                bundle = bundles[0]
                parents = {p for p,_ in book.edges_into(bundle)}
                if geometry not in parents or set(_fact(book,bundle)) != parents:
                    continue
                moves.append({move,geometry,bundle,*causes})
            if len(moves) != 1:
                continue
            refs.update(moves[0])
        candidates.append(refs)
    return tuple(candidates[0]) if len(candidates) == 1 else None


def _mint_causes(book, actual, transform):
    mints = [book.mint_of(c) for c in _ssa_history(book,actual) if book.mint_of(c) is not None]
    if len(mints) != 1 or mints[0][0] != transform or len(mints[0][1]) != 1:
        return None
    bundle = mints[0][1][0]
    if bundle.page != CELL_SET:
        return {bundle}, {bundle}
    causes = {p for p,_ in book.edges_into(bundle)}
    if set(_fact(book,bundle)) != {p.key for p in causes}:
        return None
    return causes, {bundle,*causes}


def _clear_placement(book,function,label,block,zero_id,length_id,zero_refs,definitions):
    # A current geometry restatement alone cannot authorize a reset. The
    # immutable pre-hoist snapshot records its original adjacent Const/Store.
    moves = [c for c in zero_refs if c.page == SSA_CONSTANT_HOIST and int(c.row[1]) == zero_id]
    material = [c for c in zero_refs if c.page == SSA_CONSTANT_MATERIALIZATION]
    if len(moves) != 1 or len(material) != 1 or material[0].row[2] != label:
        return None
    move = moves[0]
    geometry = move.row[5]
    old, successors = _fact(book,geometry)
    index = int(move.row[3])
    if index+1 >= len(old) or old[index+1] != ('Store',(zero_id,length_id),None,()):
        return None
    removed, proof = set(), {move,geometry}
    for row in book.page(SSA_CONSTANT_HOIST).rows():
        if row[5] != geometry:
            continue
        ref = book.latest_ref(SSA_CONSTANT_HOIST,row)
        fact = _fact(book,ref)
        position = int(row[3])
        if (len(fact) != 2 or fact[1] != 'operand_free_immutable_definition'
                or not 0 <= position < len(old)
                or old[position] != ('Const',(),int(row[1]),())):
            return None
        defs = definitions.get(int(row[1]),())
        if len(defs) != 1 or defs[0][0] != row[4] or defs[0][1].op != 'Const' or defs[0][1].args:
            return None
        removed.add(position); proof.add(ref)
    current = tuple((op.op,tuple(int(v.id) for v in op.args),
                     None if op.res is None else int(op.res.id),
                     tuple(op.attributes.get('incoming_blocks') or ())) for op in block.instrs)
    if current != tuple(op for i,op in enumerate(old) if i not in removed) or tuple(block.successors) != successors:
        return None
    return proof


def owned_record_primitive_writes(module, function_name, source, element_proof):
    """Return exact writer evidence or None; no ownership move is authorized."""
    from .source_owned_record_elements import owned_record_primitive_element
    from .source_sequence_selection import local_selection_lifetimes
    from .source_sequence_lifetime import _same_origin, _ast_kind
    from .source_bytes_concat_class import _class_matches_operand
    from .ssa_record_return_state import ssa_value_identity_cell
    name = str(function_name)
    if owned_record_primitive_element(module, name, source) != element_proof:
        return None
    if element_proof is None:
        return None
    lifetimes = local_selection_lifetimes(module, name, source.sequence_id)
    if lifetimes is None:
        return None
    function, table = module.functions[name], module.sequence_tables[name]
    book = table.book
    proof = set(lifetimes.terminal_cells)
    nodes, pending = {}, [lifetimes.selection]
    while pending:
        node = pending.pop()
        nodes[node.sequence_id] = node
        pending.extend(node.arms)
    identities, descriptors = {}, {}
    for sid, node in nodes.items():
        contracts = tuple(p for p, _ in book.edges_into(node.descriptor_cell)
                          if p.page == SEQUENCE_CONTRACT_CONCORDANCE)
        if len(contracts) != 1:
            return None
        sources = tuple(p for p, _ in book.edges_into(contracts[0]) if p.page == CANONICAL_VALUE)
        if len(sources) != 1:
            return None
        identities[sid] = sources[0]
        descriptors[sid] = table.sequences[sid]
        proof.update((node.descriptor_cell, contracts[0], sources[0]))
        if not node.arms:
            # Copy/comprehension births need their own complete writer proof.
            if _ast_kind(book, sources[0]) != 'value:List':
                return None
            operands = book.latest_ref(PROCESS_GRAPH_OPERANDS, sources[0].row)
            if operands is None or tuple(_fact(book, operands)):
                return None
            proof.add(operands)
    scopes = {cell.row[0] for cell in identities.values()}
    mutations, clears = {}, {}
    for row in book.page(CONTROL_BLOCK).rows():
        if row[0] not in scopes:
            continue
        cell = book.latest_ref(CONTROL_BLOCK, row)
        placed = book.latest_ref(CONTROL_BLOCK_PLACEMENT, (row[0], cell))
        if placed is None:
            continue
        for mutation in getattr(_fact(book, cell), 'sequence_mutations', ()):
            target, operation, arguments, effect, policy, kind, publication = mutation
            residents = [sid for sid, origin in identities.items() if _same_origin(book, target, origin)]
            if not residents:
                continue
            if len(residents) != 1 or policy != 'duplicates' or kind != 'value' or publication is not None:
                return None
            if not {target, effect, *arguments} <= {p for p, _ in book.edges_into(cell)}:
                return None
            proof.update((cell, placed, target, effect, *arguments))
            if operation == 'clear' and not arguments:
                key = (residents[0],int(effect.row[-1]))
                if key in clears:
                    return None
                clears[key] = (cell,effect)
                continue
            if operation != 'append' or len(arguments) != 1:
                return None
            classes = []
            for class_row in book.page(SOURCE_VALUE_CLASS).rows():
                class_cell = book.latest_ref(SOURCE_VALUE_CLASS, class_row)
                if _class_matches_operand(book, class_cell, arguments[0]):
                    classes.append(class_cell)
            if not classes or any(_fact(book, c)[0] != element_proof.semantic_class for c in classes):
                return None
            proof.update(classes)
            key = (residents[0], int(effect.row[-1]))
            if key in mutations:
                return None
            mutations[key] = (cell, arguments[0], effect)
    data_ids = {int(d.column_value_ids[0]) for d in descriptors.values()}
    members = {}
    for sid, descriptor in descriptors.items():
        for kind, value_id in (('data',descriptor.column_value_ids[0]),
                ('length',descriptor.length_address_id), ('capacity',descriptor.capacity_value_id),
                ('owner',descriptor.heap_owner_value_id), ('status',descriptor.status_address_id)):
            if value_id is not None:
                if int(value_id) in members and members[int(value_id)] != (sid,kind):
                    return None
                members[int(value_id)] = (sid,kind)
    definitions = {}
    for label, block in function.blocks.items():
        for op in block.instrs:
            if op.res is not None:
                definitions.setdefault(int(op.res.id),[]).append((str(label),op))
    # The only permitted derived addresses are existing zero-offset metadata
    # cells. Every consumer of those addresses remains in the use census.
    addresses = {}
    for value_id, entries in definitions.items():
        if len(entries) != 1:
            continue
        _, op = entries[0]
        if op.op != 'GetElementPtr' or len(op.args) != 2 or int(op.args[0].id) not in members:
            continue
        sid,kind = members[int(op.args[0].id)]
        if kind not in ('length','status'):
            continue
        zero = _zero_receipt(book,function,definitions,op.args[1].id)
        result = ssa_value_identity_cell(function,value_id,book=book)
        base = ssa_value_identity_cell(function,op.args[0].id,book=book)
        if zero is None or result is None or base is None:
            continue
        mints = [book.mint_of(c) for c in _ssa_history(book,result) if book.mint_of(c) is not None]
        if len(mints) != 1 or mints[0][0] != ADDRESS or len(mints[0][1]) != 1:
            continue
        bundle = mints[0][1][0]
        if bundle.page == CELL_SET:
            causes = {p for p,_ in book.edges_into(bundle)}
            if set(_fact(book,bundle)) != {p.key for p in causes}:
                continue
        else:
            causes = {bundle}
        if not _ssa_history(book,base).intersection(causes):
            continue
        addresses[value_id] = (sid,kind)
        proof.update((*zero,result,base,bundle,*causes))
    tracked = set(members) | set(addresses)
    statuses = {}
    seen_initializations, seen_clears = set(), set()
    seen_calls, seen_status_sinks = set(), set()
    # A verified transfer suffix is not a new element writer. No other loads,
    # stores, calls or aliases are exempted from the closed native-use census.
    from .source_owned_record_projection_validation import validated_record_owned_return_suffix
    suffixes = {}
    for row in book.page(RECORD_OWNED_RETURN_PROJECTION).rows():
        if row[0] != name:
            continue
        cell = book.latest_ref(RECORD_OWNED_RETURN_PROJECTION, row)
        if _fact(book, cell).source_sequence_id != source.sequence_id:
            continue
        suffix = validated_record_owned_return_suffix(module, name, row[1])
        if suffix is None:
            return None
        label, start, operations = suffix
        for offset, operation in enumerate(operations):
            suffixes[(label, start + offset)] = operation
    for label, block in function.blocks.items():
        for index, op in enumerate(block.instrs):
            if suffixes.get((label, index)) is op:
                continue
            if not tracked.intersection(int(v.id) for v in op.args):
                continue
            if op.op == 'Phi' and op.res is not None:
                selected = [(node,position) for node in nodes.values() if node.arms
                            for position,value_id in enumerate(node.member_ids) if value_id == int(op.res.id)]
                if len(selected) == 1 and tuple(int(v.id) for v in op.args) == tuple(a.member_ids[selected[0][1]] for a in selected[0][0].arms):
                    continue
                return None
            if op.op == 'Ret' and not op.attributes.get('fault_exit'):
                if tuple(int(v.id) for v in op.args if int(v.id) in tracked) == (source.sequence_id,):
                    continue
                return None
            if op.op == 'GetElementPtr' and op.res is not None and int(op.res.id) in addresses:
                continue
            if op.op == 'Store' and len(op.args) == 2:
                destination = int(op.args[1].id)
                member = addresses.get(destination, members.get(destination))
                if member is not None and member[1] == 'length':
                    zero = _zero_receipt(book,function,definitions,op.args[0].id)
                    zero_cell = ssa_value_identity_cell(function,op.args[0].id,book=book)
                    causes = None if zero_cell is None else _mint_causes(book,zero_cell,CONTROL_CONST)
                    length = descriptors[member[0]].length_address_id
                    length_cell = ssa_value_identity_cell(function,length,book=book)
                    if zero is None or causes is None or length_cell is None or not _ssa_history(book,length_cell).intersection(causes[0]):
                        return None
                    if destination in addresses:
                        if (str(label) != str(next(iter(function.blocks))) or member[0] in seen_initializations
                                or index == 0 or block.instrs[index-1].op != 'GetElementPtr'
                                or block.instrs[index-1].res.id != destination):
                            return None
                        resident_ids = {v for v,m in members.items() if m[0] == member[0]}
                        if any(resident_ids.intersection(int(v.id) for v in previous.args)
                               for previous in block.instrs[:index-1]):
                            return None
                        seen_initializations.add(member[0])
                    else:
                        candidates = [(key,c,e) for key,(c,e) in clears.items()
                                      if key[0] == member[0] and {c,e} <= causes[0]]
                        if len(candidates) != 1 or candidates[0][0] in seen_clears:
                            return None
                        placement = _clear_placement(book,function,str(label),block,int(op.args[0].id),
                                                     int(length),zero,definitions)
                        if placement is None:
                            return None
                        proof.update(placement)
                        seen_clears.add(candidates[0][0])
                    proof.update((*zero,*causes[1]))
                    continue
                if member is not None and member[1] == 'status':
                    call = statuses.get(int(op.args[0].id))
                    expected_index = index - (2 if destination in addresses else 1)
                    if (call == (member[0],str(label),expected_index)
                            and int(op.args[0].id) not in seen_status_sinks):
                        if destination not in addresses or (index > 0 and block.instrs[index-1].op == 'GetElementPtr'
                                and block.instrs[index-1].res.id == destination):
                            seen_status_sinks.add(int(op.args[0].id))
                            continue
                return None
            attrs = op.attributes or {}
            key = (attrs.get('sequence_id'), attrs.get('source_effect_node_id'))
            if op.op != 'Call' or attrs.get('ssa_sequence_operation') != 'append' or key not in mutations or key in seen_calls:
                return None
            descriptor = descriptors[key[0]]
            storage = (*descriptor.column_value_ids, descriptor.length_address_id,
                       descriptor.capacity_value_id, descriptor.heap_owner_value_id)
            if tuple(int(v.id) for v in op.args[:-1]) != storage:
                return None
            callee = module.functions.get(attrs.get('callee'))
            if callee is None or len(callee.args) != len(op.args):
                return None
            control, argument, effect = mutations[key]
            actual_cells = []
            scope = str(function.metadata.get('tensor_shape_concordance_scope') or function.name)
            for position, (formal_value, actual_value) in enumerate(zip(callee.args, op.args)):
                row = (callee.name, int(formal_value.id), name, str(label), index, position, scope)
                occurrence = book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row)
                actual_cell = ssa_value_identity_cell(function, actual_value.id, book=book)
                if (occurrence is None or actual_cell is None
                        or _fact(book, occurrence) != int(actual_value.id)
                        or actual_cell not in {p for p, _ in book.edges_into(occurrence)}):
                    return None
                proof.update((occurrence, actual_cell))
                actual_cells.append(actual_cell)
            result = None if op.res is None else ssa_value_identity_cell(function, op.res.id, book=book)
            if result is None or effect is None:
                return None
            invocation = _call_occurrence_sources(book, result, actual_cells, control, effect, callee)
            if invocation is None:
                return None
            proof.update(invocation)
            formal = ssa_value_identity_cell(callee, callee.args[-1].id, book=book)
            if formal is None:
                return None
            formal_history = _ssa_history(book,formal)
            formal_mints = [book.mint_of(c) for c in formal_history if book.mint_of(c) is not None]
            if len(formal_mints) != 1:
                return None
            formal_mint = formal_mints[0]
            proof.update(formal_history)
            permitted_owners = {v[0] for k, v in mutations.items() if k[0] == key[0]}
            if (formal_mint is None or formal_mint[0] != CONTROL_EXPRESSION
                    or len(formal_mint[1]) != 1 or formal_mint[1][0] not in permitted_owners):
                return None
            proof.add(formal_mint[1][0])
            actual = ssa_value_identity_cell(function, op.args[-1].id, book=book)
            if actual is None:
                return None
            actual_proof = _actual_argument(book, actual, argument)
            if actual_proof is None:
                from .source_loop_induction_identity import loop_induction_sources
                actual_proof = loop_induction_sources(module,name,actual,argument,str(label))
            if actual_proof is None:
                return None
            proof.update((formal, actual, *actual_proof))
            statuses[int(op.res.id)] = (key[0],str(label),index)
            seen_calls.add(key)
    expected_initializations = {sid for sid, node in nodes.items() if not node.arms}
    if (seen_calls != set(mutations) or seen_clears != set(clears)
            or seen_status_sinks != set(statuses)
            or seen_initializations != expected_initializations):
        return None
    return tuple(sorted(proof, key=repr))
