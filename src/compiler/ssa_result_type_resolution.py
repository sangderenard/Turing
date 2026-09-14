"""Monotone aggregate result-type propagation with incumbent tie breaking."""
from .transformation_priority import TransformationLedger, TransformationRule


def _type(value):
    return (value.dtype, tuple(value.shape), value.device)


def _storage_type(value):
    # Logical Boolean/integer tensors can deliberately use double-valued
    # repository buffers. Absence, however, never becomes a numeric payload.
    dtype = value.dtype if value.dtype == 'none' else (value.accounting or {}).get('physical_dtype') or value.dtype
    return (dtype, tuple(value.shape), value.device)


def equivalent_physical_layout(left, right):
    """Contiguous views can differ in shape while sharing the same buffer ABI."""
    if left[0] != right[0] or (left[2] is not None and right[2] is not None and left[2] != right[2]):
        return False
    if left[1] == right[1]:
        return True
    from math import prod
    from operator import index
    try:
        shapes = [tuple(index(dimension) for dimension in shape) for shape in (left[1], right[1])]
    except (TypeError, ValueError):
        return False
    return all(dimension >= 0 for shape in shapes for dimension in shape) and prod(shapes[0]) == prod(shapes[1])


def settle_call_result_types(functions, emit_outputs, function_values):
    ledger = TransformationLedger((
        TransformationRule('provisional', 1),
        TransformationRule('callee_output', 2),
        TransformationRule('physical_storage', 3),
    ), {'provisional': ('callee_output', 'physical_storage'),
        'callee_output': ('physical_storage',)})
    for function in functions.values():
        for value in function_values(function).values():
            accounting = value.accounting or {}
            physical = (accounting.get('physical_dtype') or accounting.get('program_abi_storage')) and value.dtype not in {None, '', 'unknown'}
            ledger.propose((function.name, int(value.id)),
                'physical_storage' if physical else 'provisional',
                ('initial', _type(value)), after=_type(value))

    def bindings():
        for caller in functions.values():
            values = function_values(caller)
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if (instruction.op not in {'Call', 'call'}
                            or instruction.attributes.get('result_convention') != 'ssa.aggregate'):
                        continue
                    callee = functions.get(str(instruction.attributes.get('callee', '')))
                    ids = tuple(map(int, instruction.attributes.get('output_ids', ())))
                    if callee is None or not ids:
                        continue
                    outputs = tuple(emit_outputs(callee.name, callee))
                    positions = tuple(map(
                        int,
                        instruction.attributes.get("output_positions", ()),
                    ))
                    selected = tuple(map(int, instruction.attributes.get('callee_output_ids', ())))
                    if positions:
                        # The linker correlates caller aggregate members with
                        # the callee's physical Ret ABI by position.  That is
                        # stronger than numeric identity: caller/callee SSA
                        # ids are function-local, and a callee output may be
                        # collision-freshened after its source-level result
                        # binding was recorded.  Looking up the old semantic
                        # ids then skips type settlement entirely (a matmul
                        # weight gradient whose id matched a helper-local
                        # integer constant was exposed as a scalar).  Preserve
                        # the exact positional contract the linker proved.
                        if (
                            len(positions) != len(ids)
                            or any(
                                position < 0 or position >= len(outputs)
                                for position in positions
                            )
                        ):
                            continue
                        outputs = tuple(outputs[position] for position in positions)
                    elif selected:
                        by_id = {int(value.id): value for value in outputs}
                        if len(selected) != len(ids) or any(value_id not in by_id for value_id in selected):
                            continue
                        outputs = tuple(by_id[value_id] for value_id in selected)
                    elif len(ids) != len(outputs):
                        continue
                    for value_id, source in zip(ids, outputs):
                        if value_id in values:
                            yield caller, values[value_id], callee, source

    exact_values = set()
    changed = True
    rounds = 0
    while changed:
        changed = False
        rounds += 1
        for caller, target, callee, source in bindings():
            if source.dtype in {None, '', 'unknown'} and not source.shape and source.device is None:
                continue
            identity = (caller.name, int(target.id))
            source_priority = ledger.incumbent_priority((callee.name, int(source.id))) or 1
            rule = 'physical_storage' if source_priority == 3 else 'callee_output'
            previous_priority = ledger.incumbent_priority(identity)
            accepted = ledger.propose(identity, rule,
                (callee.name, int(source.id), _type(source)), before=_type(target), after=_type(source))
            winner = ledger.incumbent_target(identity)
            if winner != _type(target):
                target.dtype, target.shape, target.device = winner
                changed = True
            if ledger.incumbent_priority(identity) != previous_priority:
                changed = True
            if accepted:
                target.accounting = {**dict(target.accounting or {}),
                    'ssa_call_result_from': (callee.name, int(source.id))}
                physical_dtype = (source.accounting or {}).get('physical_dtype')
                if physical_dtype and previous_priority != 3:
                    target.accounting['physical_dtype'] = physical_dtype
                    target.accounting['physical_dtype_provenance'] = (
                        'callee_output', callee.name, int(source.id))
            elif winner == _type(source):
                target.accounting = dict(target.accounting or {})
                target.accounting.setdefault('ssa_call_result_from', (callee.name, int(source.id)))
            # A rejected physical incumbent need not have a callee origin.
            # Only accepted/equivalent result bindings authorize downstream
            # consumers to propagate that origin.
            if (accepted or winner == _type(source)) and (target.accounting or {}).get('ssa_call_result_from'):
                exact_values.add(id(target))

    conflicts = tuple(dict(caller=caller.name, value_id=int(target.id),
        retained=_type(target), retained_storage=_storage_type(target),
        callee=callee.name, callee_value_id=int(source.id),
        proposed=_type(source), proposed_storage=_storage_type(source))
        for caller, target, callee, source in bindings()
        if source.dtype not in {None, '', 'unknown'} and not equivalent_physical_layout(_storage_type(source), _storage_type(target)))
    return exact_values, ledger.events, conflicts, rounds
