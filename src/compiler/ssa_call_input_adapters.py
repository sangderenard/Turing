"""Adapt logical values to already compiled physical buffer contracts."""
from __future__ import annotations

from itertools import chain
from math import prod
import time

from .monotonic_ids import GLOBAL_MONOTONIC_IDS
from .identity_concordance import current_identity_book
from ..transmogrifier.ssa import Instr, SSAValue


_NUMERIC_DTYPES = {'bool', 'int32', 'int64', 'float32', 'float64'}


def _concord_exact_region_feed_dtypes(functions, changes=None) -> int:
    """Restore exact region-feed views after result projection rewiring.

    A planned-region call records the dtype of every feed at the moment the
    region is cut. Product wiring can later replace that occurrence with a
    projection carrying the producer's *logical* result dtype. Those facts
    may differ when the producer uses a wider physical buffer, but the call
    view and its callee formal must retain the exact feed contract.

    This is a view correction, not a value conversion: the SSA id and buffer
    identity stay unchanged. A real physical mismatch is left for the
    adapter below, which inserts a Cast. Both sides consume one concordance
    page so another call cannot silently reinterpret the identity.
    """
    page = current_identity_book().page('exact_region_feed_dtype')
    changed = 0
    for caller in functions.values():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {'Call', 'call'}:
                    continue
                feed_dtypes = tuple(
                    str(dtype or '')
                    for dtype in instruction.attributes.get('feed_dtypes', ())
                )
                feed_ids = tuple(map(
                    int, instruction.attributes.get('feed_ids', ())
                ))
                callee = functions.get(str(
                    instruction.attributes.get('callee', '')
                ))
                if (
                    callee is None
                    or not callee.metadata.get('source_region_integral')
                    or len(feed_dtypes) != len(instruction.args)
                    or len(feed_ids) != len(instruction.args)
                    or len(instruction.args) != len(callee.args)
                ):
                    continue
                for position, (actual, formal, feed_id, exact_dtype) in enumerate(zip(
                    instruction.args, callee.args, feed_ids, feed_dtypes,
                )):
                    if exact_dtype not in _NUMERIC_DTYPES:
                        continue
                    facts = (
                        (
                            ('feed', str(caller.name), int(feed_id)),
                            (exact_dtype,),
                        ),
                        (
                            ('formal', str(callee.name), int(formal.id)),
                            (exact_dtype,),
                        ),
                    )
                    for row, proposed in facts:
                        incumbent = page.latest(row)
                        if incumbent is not None and tuple(incumbent) != proposed:
                            raise ValueError(
                                'exact region feed dtype concordance '
                                f'disagreement for {row!r}: '
                                f'recorded={incumbent!r}, proposed={proposed!r}'
                            )
                        if incumbent is None:
                            page.set(row, 0, proposed)

                    actual_accounting = dict(actual.accounting or {})
                    formal_accounting = dict(formal.accounting or {})
                    actual_physical = actual_accounting.get('physical_dtype')
                    formal_physical = formal_accounting.get('physical_dtype')

                    # A different explicit representation needs an
                    # elementwise adapter. Keep the source intact and only
                    # install the exact target contract on the formal.
                    if actual_physical in _NUMERIC_DTYPES and actual_physical != exact_dtype:
                        pass
                    elif actual.dtype != exact_dtype:
                        if changes is not None:
                            changes.append((
                                'actual', str(caller.name), str(callee.name),
                                int(feed_id), int(actual.id), str(actual.dtype),
                                exact_dtype, actual_physical,
                            ))
                        instruction.args[position] = SSAValue(
                            int(actual.id),
                            dtype=exact_dtype,
                            shape=tuple(actual.shape or ()),
                            device=actual.device,
                            accounting={
                                **actual_accounting,
                                'exact_region_feed_dtype': exact_dtype,
                                'exact_region_feed_source': (
                                    str(caller.name), str(callee.name),
                                    int(feed_id), int(position),
                                ),
                            },
                        )
                        actual = instruction.args[position]
                        changed += 1

                    if formal_physical in _NUMERIC_DTYPES and formal_physical != exact_dtype:
                        continue
                    if (
                        formal.dtype != exact_dtype
                        or formal_accounting.get('exact_region_feed_dtype')
                        != exact_dtype
                    ):
                        if changes is not None:
                            changes.append((
                                'formal', str(caller.name), str(callee.name),
                                int(feed_id), int(formal.id), str(formal.dtype),
                                exact_dtype, formal_physical,
                            ))
                        formal.dtype = exact_dtype
                        formal.accounting = {
                            **formal_accounting,
                            'ssa_call_dtype': exact_dtype,
                            'exact_region_feed_dtype': exact_dtype,
                            'exact_region_feed_source': (
                                str(caller.name), int(actual.id), int(position),
                            ),
                        }
                        changed += 1
    return changed


def _call_writes_argument(instruction, position: int) -> bool:
    """Read the call's exact physical output contract when it has one.

    ``output_ids`` records semantic publications in the caller's value
    namespace.  Repository tensor calls additionally carry
    ``ssa_output_argument``: the one argument the imported implementation
    physically writes.  Treating every semantic publication as a writable
    operand suppresses required read-only ABI conversions (a Boolean region
    capture feeding the double-backed ``where_double`` condition was the
    concrete failure).  The positional repository receipt is authoritative;
    older/source calls without one retain the identity-based convention.
    """
    output_position = instruction.attributes.get('ssa_output_argument')
    if output_position is not None:
        return int(output_position) == int(position)
    if int(position) >= len(instruction.args):
        return False
    argument_id = int(instruction.args[int(position)].id)
    return argument_id in set(map(
        int, instruction.attributes.get('output_ids', ()),
    ))


def _intern_unshadowed_formal_uses(functions):
    """Bind same-ID operands to their incumbent formal descriptor.

    Region planning can create lightweight capture occurrences before whole-
    program type settlement.  Once collision freshening has guaranteed that
    no instruction result shadows a formal ID, those occurrences name the
    formal's exact storage and must share its final physical descriptor.
    """
    count = 0
    for function in functions.values():
        formals = {int(value.id): value for value in function.args}
        produced = {
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        rebound_ids = set()
        for block in function.blocks.values():
            for instruction in block.instrs:
                rebound = []
                for argument in instruction.args:
                    value_id = int(argument.id)
                    formal = formals.get(value_id)
                    if (
                        formal is None
                        or value_id in produced
                        or argument is formal
                    ):
                        rebound.append(argument)
                        continue
                    # One storage identity may carry several shaped views.
                    # An occurrence that declares itself an exact view of
                    # this same storage (``b.reshape((-1, 1, 2))``) owns its
                    # extents: interning it to the formal silently restored
                    # ``b``'s own shape, so the broadcast conformed the wrong
                    # axes.  Adopt the formal's storage contract and keep the
                    # authored view shape.
                    declared_view = (argument.accounting or {}).get(
                        'ssa_storage_view'
                    )
                    exact_feed_dtype = (argument.accounting or {}).get(
                        'exact_region_feed_dtype'
                    )
                    if (
                        exact_feed_dtype
                        and str(argument.dtype) == str(exact_feed_dtype)
                    ):
                        # This occurrence is the concordance-owned typed view
                        # of the same storage at a region boundary. Replacing
                        # it with the storage formal would discard that view
                        # and recreate the identical change every round.
                        rebound.append(argument)
                        continue
                    view_shape = tuple(
                        (declared_view or {}).get('view_shape') or ()
                    )
                    if (
                        view_shape
                        and view_shape == tuple(argument.shape or ())
                        and tuple(formal.shape or ())
                        and prod(view_shape) == prod(tuple(formal.shape))
                    ):
                        rebound.append(SSAValue(
                            int(formal.id),
                            dtype=formal.dtype,
                            shape=view_shape,
                            device=formal.device,
                            accounting={
                                **dict(formal.accounting or {}),
                                'ssa_storage_view': dict(declared_view),
                            },
                        ))
                        continue
                    # Identity is already proven by the integer ID and lack
                    # of a shadowing producer.  The declared formal is the
                    # incumbent descriptor; do not merge stale occurrence
                    # accounting into it, especially frame-storage labels
                    # left from an earlier result projection.
                    rebound.append(formal)
                    rebound_ids.add(value_id)
                    count += 1
                instruction.args = rebound
        if rebound_ids:
            function.metadata['canonicalized_formal_use_ids'] = tuple(sorted(
                set(function.metadata.get('canonicalized_formal_use_ids', ()))
                | rebound_ids
            ))
    return count


def _infer_pointer_storage(functions):
    """Recover typed LLVM load/store contracts without changing pointer types."""
    canonical = {'double': 'float64', 'float': 'float32', 'i64': 'int64', 'i32': 'int32', 'i1': 'bool'}
    for function in functions.values():
        instructions = [instruction for block in function.blocks.values() for instruction in block.instrs]
        for formal in function.args:
            if formal.dtype != 'ptr' or (formal.accounting or {}).get('physical_dtype'):
                continue
            aliases = {int(formal.id)}
            changed = True
            while changed:
                changed = False
                for instruction in instructions:
                    if (instruction.op in {'GEP', 'GetElementPtr', 'getelementptr', 'PtrCast'}
                            and instruction.res is not None and instruction.args
                            and int(instruction.args[0].id) in aliases
                            and int(instruction.res.id) not in aliases):
                        aliases.add(int(instruction.res.id))
                        changed = True
            types = set()
            for instruction in instructions:
                dtype = None
                if instruction.op == 'Load' and instruction.args and int(instruction.args[0].id) in aliases and instruction.res is not None:
                    dtype = instruction.res.dtype
                elif instruction.op == 'Store' and len(instruction.args) >= 2 and int(instruction.args[1].id) in aliases:
                    dtype = instruction.args[0].dtype
                if dtype is not None:
                    types.add(canonical.get(dtype, dtype))
            if len(types) == 1 and 'ptr' not in types:
                formal.accounting = {**dict(formal.accounting or {}),
                    'physical_dtype': next(iter(types)), 'physical_dtype_provenance': 'typed_memory_operations'}


def physical_call_input_conflicts(functions):
    """Collect the whole frontier before inference mutates any value types."""
    conflicts = []
    for caller in functions.values():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                callee = functions.get(str(instruction.attributes.get('callee', '')))
                if (instruction.op not in {'Call', 'call'} or callee is None
                        or len(instruction.args) != len(callee.args)):
                    continue
                for actual, formal in zip(instruction.args, callee.args):
                    # A keyed mapping's own occurrence is a descriptor that
                    # names its length/keys/values slots, not physical storage
                    # (the backends treat it as structural); its dtype is the
                    # sequence convention's column-0 placeholder, never a
                    # buffer representation to reconcile at a call.
                    if any((value.accounting or {}).get('program_abi_storage') == 'keyed'
                           for value in (actual, formal)):
                        continue
                    # A typed pointer constrains its element representation;
                    # "ptr" itself is not a competing numerical dtype.
                    actual_type = (actual.accounting or {}).get('physical_dtype') or actual.dtype
                    formal_type = (formal.accounting or {}).get('physical_dtype') or formal.dtype
                    if callee.metadata.get('source_region_integral') and not formal.shape and formal.dtype != 'ptr':
                        formal_type = formal.dtype
                    if (actual_type in {None, '', 'unknown', 'ptr'} or formal_type in {None, '', 'unknown', 'ptr'}
                            or actual_type == formal_type):
                        continue
                    if all((value.accounting or {}).get('physical_dtype')
                           or (value.accounting or {}).get('program_abi_storage')
                           for value in (actual, formal)):
                        conflicts.append((caller.name, int(actual.id), actual_type,
                                          callee.name, int(formal.id), formal_type))
    return tuple(conflicts)


def _read_only_feed(
    function, value_id, functions=None, active=frozenset(), diagnostics=None,
):
    if diagnostics is not None:
        diagnostics['queries'] += 1
        diagnostics['max_depth'] = max(
            diagnostics['max_depth'], len(active) + 1,
        )
        now = time.monotonic()
        if now >= diagnostics['next_report']:
            diagnostics['report'](
                'physical-call adaptation round '
                f"{diagnostics['round']} read-only walk: "
                f"queries={diagnostics['queries']} "
                f"instruction_scans={diagnostics['instruction_scans']} "
                f"cycle_edges={diagnostics['cycle_edges']} "
                f"max_depth={diagnostics['max_depth']} "
                f"current={function.name}:{int(value_id)}"
            )
            diagnostics['next_report'] = now + diagnostics['interval']
    identity = (function.name, int(value_id))
    if identity in active:
        if diagnostics is not None:
            diagnostics['cycle_edges'] += 1
        return True  # This path adds no new operation; other exits are still inspected.
    active = active | {identity}
    functions = functions or {}
    aliases = {int(value_id)}
    instructions = [instruction for block in function.blocks.values() for instruction in block.instrs]
    if diagnostics is not None:
        diagnostics['instruction_scans'] += len(instructions)
    changed = True
    while changed:
        changed = False
        for instruction in instructions:
            if (instruction.op in {'GEP', 'GetElementPtr', 'getelementptr', 'PtrCast', 'View'} and instruction.res is not None
                    and any(int(value.id) in aliases for value in instruction.args)
                    and int(instruction.res.id) not in aliases):
                aliases.add(int(instruction.res.id))
                changed = True
    for instruction in instructions:
        if instruction.res is not None and int(instruction.res.id) == int(value_id):
            return False
        uses = [index for index, value in enumerate(instruction.args) if int(value.id) in aliases]
        if not uses:
            continue
        if instruction.op in {'Store', 'store'}:
            return False
        if instruction.op in {'Call', 'call'}:
            if any(
                _call_writes_argument(instruction, index)
                for index in uses
            ):
                return False
            callee = functions.get(str(instruction.attributes.get('callee', '')))
            if callee is not None and len(instruction.args) == len(callee.args):
                if any(not _read_only_feed(
                    callee, callee.args[index].id, functions, active,
                    diagnostics,
                ) for index in uses):
                    return False
                continue
            output = instruction.attributes.get('ssa_output_argument')
            if output is None or int(output) in uses:
                return False
    return True


def _adapt_physical_call_inputs_round(
    functions, *, progress=None, round_index=0,
) -> int:
    """Perform one ordered physical-call adaptation round.

    Regions precede physical record layout discovery. Tensor backends may also
    represent logical Boolean values in double buffers. Convert at these value
    boundaries rather than changing the shared caller storage representation.
    """
    report = progress or (lambda _message: None)
    started = time.monotonic()
    report(
        f'physical-call adaptation round {round_index} start: '
        f'functions={len(functions)}'
    )
    # Formal interning must precede the exact feed view.  The reverse order
    # let interning immediately replace three freshly concorded region-feed
    # occurrences with their physical-storage formal on every round.  The
    # round reported three changes but ended in the state in which it began.
    _intern_unshadowed_formal_uses(functions)
    report(
        f'physical-call adaptation round {round_index}: formal uses interned; '
        f'elapsed={time.monotonic() - started:.1f}s'
    )
    region_changes = []
    count = _concord_exact_region_feed_dtypes(functions, region_changes)
    report(
        f'physical-call adaptation round {round_index}: exact region feeds '
        f'settled; changes={count}; changed_rows={region_changes[:8]!r}; '
        f'elapsed={time.monotonic() - started:.1f}s'
    )
    _infer_pointer_storage(functions)
    report(
        f'physical-call adaptation round {round_index}: pointer storage '
        f'inferred; elapsed={time.monotonic() - started:.1f}s'
    )
    diagnostics = {
        'report': report,
        'round': int(round_index),
        'interval': 5.0,
        'next_report': time.monotonic() + 5.0,
        'queries': 0,
        'instruction_scans': 0,
        'cycle_edges': 0,
        'max_depth': 0,
        'call_operands': 0,
    }
    for caller in functions.values():
        caller_formals = {int(value.id): value for value in caller.args}
        for block_name, block in caller.blocks.items():
            rewritten = []
            for instruction in block.instrs:
                callee_name = str(instruction.attributes.get('callee', ''))
                callee = functions.get(callee_name)
                if (
                    instruction.op in {'Call', 'call'}
                    and callee_name == 'broadcast_double'
                    and instruction.args
                ):
                    source = instruction.args[0]
                    source_dtype = (
                        (source.accounting or {}).get('physical_dtype')
                        or source.dtype
                    )
                    if (
                        not tuple(source.shape or ())
                        and str(source_dtype).casefold() in {
                            'bool', 'i1', 'int', 'int32', 'int64', 'long',
                            'float32',
                        }
                    ):
                        # Opaque pointers do not carry the authored kernel's
                        # pointee type. `broadcast_double` position zero is a
                        # double buffer, so a scalar integer actual requires
                        # a value conversion after whole-program type
                        # settlement; passing its pointer directly is byte
                        # reinterpretation (integer 1 becomes 5e-324).
                        converted = SSAValue(
                            GLOBAL_MONOTONIC_IDS.mint(),
                            dtype='float64', shape=(), device=source.device,
                            accounting={
                                'physical_dtype': 'float64',
                                'kernel_input_conversion': (
                                    caller.name, int(source.id),
                                    'broadcast_double', 0,
                                ),
                            },
                        )
                        rewritten.append(Instr(
                            'Cast', [source], converted,
                            attributes={
                                'source_dtype': str(source_dtype),
                                'target_dtype': 'float64',
                                'concordant_kernel_input_conversion': True,
                            },
                        ))
                        instruction.args[0] = converted
                        conversion = {
                            'source_id': int(source.id),
                            'converted_id': int(converted.id),
                            'source_dtype': str(source_dtype),
                            'target_dtype': 'float64',
                            'callee': 'broadcast_double',
                            'operand_position': 0,
                            'block': str(block_name),
                            'consumer_id': (
                                None if instruction.res is None
                                else int(instruction.res.id)
                            ),
                        }
                        caller.metadata['kernel_input_conversions'] = (
                            *tuple(caller.metadata.get(
                                'kernel_input_conversions', (),
                            )),
                            conversion,
                        )
                        page = current_identity_book().page(
                            'kernel_input_conversion'
                        )
                        row = (
                            str(caller.name), str(block_name),
                            conversion['consumer_id'], 0,
                        )
                        history = page.history(row)
                        column = history[-1][0] + 1 if history else 0
                        page.set(row, column, (
                            int(source.id), int(converted.id),
                            str(source_dtype), 'float64',
                            'broadcast_double',
                        ))
                        count += 1
                if (instruction.op in {'Call', 'call'} and callee is not None
                        and len(instruction.args) == len(callee.args)):
                    for index, (actual, formal) in enumerate(zip(instruction.args, callee.args)):
                        diagnostics['call_operands'] += 1
                        accounting = {**dict(actual.accounting or {}),
                            **dict((caller_formals.get(int(actual.id), actual).accounting or {}))}
                        scalar_region = bool(callee.metadata.get('source_region_integral') and not formal.shape and formal.dtype != 'ptr')
                        source_dtype = accounting.get('physical_dtype') or actual.dtype
                        target_dtype = formal.dtype if scalar_region else (formal.accounting or {}).get('physical_dtype')
                        numeric = {'bool', 'int32', 'int64', 'float32', 'float64'}
                        # A generated scalar (most importantly a loop-carried
                        # induction value) may feed an authored tensor helper
                        # whose scalar formal is consumed in the tensor's
                        # numerical dtype.  The call edge is then a VALUE
                        # conversion, not permission to reinterpret the
                        # integer's bytes as a double.  Physical/program ABI
                        # storage remains immutable and follows the stricter
                        # region rule below.
                        logical_scalar_conversion = bool(
                            not actual.shape
                            and not formal.shape
                            and formal.dtype != 'ptr'
                            and not accounting.get('program_abi_storage')
                            and not accounting.get('physical_dtype')
                            # A sequence arena (a record's list column) is
                            # storage passed by reference, not a scalar value.
                            and not (
                                accounting.get('sequence_arena')
                                and current_identity_book().page(
                                    'call_input_storage_concordance'
                                ).concord(
                                    (str(caller.name), int(actual.id)),
                                    'sequence_arena_by_reference',
                                ) == 'sequence_arena_by_reference'
                            )
                            and source_dtype in numeric
                            and formal.dtype in numeric
                            and source_dtype != formal.dtype
                            and not _call_writes_argument(instruction, index)
                            and _read_only_feed(
                                callee, formal.id, functions,
                                diagnostics=diagnostics,
                            )
                        )
                        physical_conversion = not (
                            not (accounting.get('program_abi_storage') or accounting.get('physical_dtype')
                                 or int(actual.id) in caller_formals)
                            or source_dtype == target_dtype
                            or (not scalar_region and formal.dtype != 'ptr' and actual.dtype != formal.dtype)
                            or (scalar_region and actual.shape)
                            or (not actual.shape and (accounting.get('program_abi_storage') == 'span'
                                or int(accounting.get('program_abi_rank') or 0) > 0))
                            or _call_writes_argument(instruction, index)
                            or not _read_only_feed(
                                callee, formal.id, functions,
                                diagnostics=diagnostics,
                            )
                        )
                        if not logical_scalar_conversion and not physical_conversion:
                            continue
                        # Only numerical scalar representation changes belong
                        # here. Records, references and string tokens are not
                        # interchangeable numerical values.
                        if logical_scalar_conversion:
                            target_dtype = formal.dtype
                        if source_dtype not in numeric or target_dtype not in numeric:
                            continue
                        # Shape is retained; a span conversion is elementwise.
                        converted = SSAValue(GLOBAL_MONOTONIC_IDS.mint(), dtype=target_dtype, shape=actual.shape, device=actual.device,
                            accounting={'physical_dtype': target_dtype,
                                'call_input_conversion': (caller.name, int(actual.id), callee.name, int(formal.id)),
                                'call_input_conversion_kind': (
                                    'read_only_scalar_numeric'
                                    if logical_scalar_conversion
                                    else 'physical_region'
                                )})
                        conversion_attributes = {
                            'target_dtype': target_dtype,
                            'source_dtype': source_dtype,
                        }
                        if logical_scalar_conversion:
                            conversion_attributes['concordant_call_input_conversion'] = True
                        else:
                            conversion_attributes['physical_region_input_conversion'] = True
                        rewritten.append(Instr(
                            'Cast', [actual], converted,
                            attributes=conversion_attributes,
                        ))
                        instruction.args[index] = converted
                        formal.accounting = {**dict(formal.accounting or {}),
                            'source_physical_dtype': source_dtype,
                            'physical_dtype': target_dtype}
                        conversion = {
                            'caller': str(caller.name),
                            'actual_id': int(actual.id),
                            'callee': str(callee.name),
                            'formal_id': int(formal.id),
                            'converted_id': int(converted.id),
                            'source_dtype': str(source_dtype),
                            'target_dtype': str(target_dtype),
                            'kind': (
                                'read_only_scalar_numeric'
                                if logical_scalar_conversion
                                else 'physical_region'
                            ),
                        }
                        caller.metadata['call_input_conversions'] = (
                            *tuple(caller.metadata.get('call_input_conversions', ())),
                            conversion,
                        )
                        page = current_identity_book().page('call_input_conversion')
                        row = (
                            str(caller.name), int(actual.id),
                            str(callee.name), int(formal.id),
                        )
                        history = page.history(row)
                        column = history[-1][0] + 1 if history else 0
                        page.set(row, column, (
                            int(converted.id), str(source_dtype),
                            str(target_dtype), conversion['kind'],
                        ))
                        count += 1
                publication = None
                output_position = instruction.attributes.get('ssa_output_argument')
                if (instruction.op in {'Call', 'call'} and callee is not None
                        and output_position is not None and instruction.res is not None
                        and 0 <= int(output_position) < min(len(instruction.args), len(callee.args))):
                    position = int(output_position)
                    original = instruction.args[position]
                    formal = callee.args[position]
                    target_dtype = (original.accounting or {}).get('physical_dtype') or original.dtype
                    backend_dtype = (formal.accounting or {}).get('physical_dtype') or formal.dtype
                    numeric = {'bool', 'int32', 'int64', 'float32', 'float64'}
                    if (target_dtype in numeric and backend_dtype in numeric and target_dtype != backend_dtype
                            and int(instruction.res.id) == int(original.id)
                            and not (not original.shape and (original.accounting or {}).get('program_abi_storage') == 'span')
                            and int(original.id) not in {int(value.id) for value in caller.args}):
                        temporary = SSAValue(GLOBAL_MONOTONIC_IDS.mint(), dtype=backend_dtype, shape=original.shape, device=original.device,
                            accounting={'physical_dtype': backend_dtype,
                                'call_output_conversion': (callee.name, int(formal.id), caller.name, int(original.id))})
                        instruction.args[position] = temporary
                        instruction.res = temporary
                        if 'output_ids' in instruction.attributes:
                            instruction.attributes['output_ids'] = tuple(
                                temporary.id if int(value_id) == int(original.id) else value_id
                                for value_id in instruction.attributes['output_ids'])
                        publication = Instr('Cast', [temporary], original,
                            attributes={'target_dtype': target_dtype, 'source_dtype': backend_dtype,
                                'physical_call_output_conversion': True})
                        count += 1
                rewritten.append(instruction)
                if publication is not None:
                    rewritten.append(publication)
            block.instrs = rewritten
    report(
        f'physical-call adaptation round {round_index} complete: '
        f'changes={count} call_operands={diagnostics["call_operands"]} '
        f'read_only_queries={diagnostics["queries"]} '
        f'instruction_scans={diagnostics["instruction_scans"]} '
        f'cycle_edges={diagnostics["cycle_edges"]} '
        f'max_depth={diagnostics["max_depth"]} '
        f'elapsed={time.monotonic() - started:.1f}s'
    )
    return count


def _adaptation_state_signature(functions):
    """Return semantic call-boundary state, independent of temporary IDs.

    Conversion passes mint fresh SSA IDs. A syntactic snapshot would regard
    ``cast(cast(x))`` as novel forever even when it repeats the same semantic
    boundary contract. Collapse every compiler conversion result back to its
    source storage identity and compare only formals plus ordered call edges.
    """

    conversion_sources = {}
    for function in functions.values():
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.res is None
                    or not instruction.args
                    or instruction.op not in {'Cast', 'cast'}
                ):
                    continue
                if any(instruction.attributes.get(key) for key in (
                    'concordant_kernel_input_conversion',
                    'concordant_call_input_conversion',
                    'physical_region_input_conversion',
                    'physical_call_output_conversion',
                )):
                    conversion_sources[int(instruction.res.id)] = int(
                        instruction.args[0].id
                    )

    def storage_identity(value_id):
        current = int(value_id)
        visited = set()
        while current in conversion_sources and current not in visited:
            visited.add(current)
            current = int(conversion_sources[current])
        return current

    def descriptor(value):
        accounting = dict(value.accounting or {})
        return (
            storage_identity(value.id), str(value.dtype),
            tuple(value.shape or ()),
            accounting.get('physical_dtype'),
            accounting.get('exact_region_feed_dtype'),
            accounting.get('program_abi_storage'),
        )

    return tuple(
        (
            str(function_name),
            tuple(descriptor(value) for value in function.args),
            tuple(
                (
                    str(block_name),
                    None if instruction.res is None
                    else descriptor(instruction.res),
                    tuple(descriptor(value) for value in instruction.args),
                    str(instruction.attributes.get('callee', '')),
                    instruction.attributes.get('ssa_output_argument'),
                )
                for block_name, block in function.blocks.items()
                for instruction in block.instrs
                if instruction.op in {'Call', 'call'}
            ),
        )
        for function_name, function in sorted(functions.items())
    )


def adapt_physical_call_inputs(functions, *, progress=None) -> int:
    """Converge physical call-edge contracts and record every adaptation.

    A later caller can settle a region formal after that region's own calls
    were visited in the same ordered scan.  One pass therefore is not a
    whole-program result: the newly settled storage identity must flow back
    through the region on a subsequent round.  Converge monotonically, with a
    structural bound, and publish the round ledger on the shared concordance.
    """
    call_operand_count = sum(
        len(instruction.args)
        for function in functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {'Call', 'call'}
    )
    # A full scan advances every already-settled call edge. In the adverse
    # function order an acyclic dependency needs at most one round per
    # function. Cycles are rejected by semantic-state recurrence below.
    round_bound = max(1, len(functions) + 1)
    total = 0
    row = ('whole-program', tuple(sorted(map(str, functions))))
    page = current_identity_book().page(
        'physical_call_input_adaptation_fixed_point'
    )
    report = progress or (lambda _message: None)
    report(
        'physical-call adaptation fixed point start: '
        f'functions={len(functions)} call_operands={call_operand_count} '
        f'round_bound={round_bound}'
    )
    state = _adaptation_state_signature(functions)
    seen_states = {state: -1}
    for round_index in range(round_bound):
        changed = _adapt_physical_call_inputs_round(
            functions, progress=progress, round_index=round_index,
        )
        page.set(row, round_index, int(changed))
        total += changed
        if changed == 0:
            report(
                'physical-call adaptation fixed point converged: '
                f'rounds={round_index + 1} total_changes={total}'
            )
            return total
        next_state = _adaptation_state_signature(functions)
        previous_round = seen_states.get(next_state)
        if previous_round is not None:
            raise ValueError(
                'physical call input adaptation repeated a completed state: '
                f'round {round_index} returned to round {previous_round}; '
                f'last round reported {changed} change(s) over '
                f'{call_operand_count} call operands'
            )
        seen_states[next_state] = round_index
        state = next_state
    raise ValueError(
        'physical call input adaptation did not converge after '
        f'{round_bound} rounds over {call_operand_count} call operands'
    )
