"""Adapt logical values to already compiled physical buffer contracts."""
from __future__ import annotations

from itertools import chain
from math import prod
import time

from .identity_concordance import current_identity_book, Derived, Mode
from .concordance_declarations import (
    CONTROL_CAST, PHYSICAL_CALL_INPUT_ADAPTATION, CALL_INPUT_CONVERSION,
    KERNEL_INPUT_CONVERSION,
)
from .ssa_record_return_state import mint_ssa_value, ssa_value_identity_cell, identity_cells
from ..transmogrifier.ssa import Instr, SSAValue


_NUMERIC_DTYPES = {'bool', 'int32', 'int64', 'float32', 'float64'}


def call_input_rank_updates(callee, actual, formal, *, by_value_kernel_formal=False):
    """Rank updates for one already-bound call edge; the caller posts changes."""
    # N4 advance_pieces region111 converts scalar int64 zero to double for
    # index_assign_double's %values pointer. A shared repository kernel's
    # buffer formal may also receive spans from other calls; its shape is
    # not a shape equality with this caller's scalar. Copying rank1 onto
    # that Cast invented a span with no extent origin. The declared kernel
    # ABI owns pointer/value transport, while each actual owns its axes.
    if (callee.metadata or {}).get('llvm_argument_names'):
        return ()
    def rank(value):
        return max(
            len(tuple(value.shape or ())),
            int((value.accounting or {}).get('program_abi_rank', 0) or 0),
            int((value.accounting or {}).get('ssa_call_rank', 0) or 0),
        )
    actual_rank, formal_rank = rank(actual), rank(formal)
    call_rank = 0 if by_value_kernel_formal else max(actual_rank, formal_rank)
    return tuple((value, call_rank) for value, own_rank in (
        (actual, actual_rank), (formal, formal_rank),
    ) if own_rank != call_rank and call_rank != 0)


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
    from .concordance_declarations import (
        EXACT_REGION_FEED_DTYPE, PHYSICAL_CALL_INPUT_ADAPTATION,
    )
    from .identity_concordance import ConcordanceRefusal, Derived, Mode
    from .ssa_record_return_state import ssa_value_identity_cell

    book = current_identity_book()
    page = book.page('exact_region_feed_dtype')
    changed = 0
    from .ssa_record_return_state import accepted_ssa_component
    for caller in functions.values():
        if accepted_ssa_component(caller):
            continue
        # A call occurrence is not the storage authority. In the compiler's
        # Kahn-frontier micro-test, a loop-result Phi produced int64, while
        # the earlier region cut expected float64. Retyping the same-ID call
        # occurrence made the native store read integer bytes as a double.
        # Resolve the exact producer/formal before creating any typed view;
        # a representation mismatch must reach the existing Cast adapter.
        residents = {int(value.id): value for value in caller.args}
        residents.update(
            (int(item.res.id), item.res)
            for owner_block in caller.blocks.values()
            for item in owner_block.instrs if item.res is not None
        )
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
                    # Page ``exact_region_feed_dtype`` row ``(function, value
                    # id)``: the exact dtype the value is fed (or received) as.
                    # DERIVED from the value's ``ssa_value`` cell in its own
                    # function -- the caller's feed value, the callee's formal.
                    facts = (
                        (
                            (str(caller.name), int(feed_id)),
                            (exact_dtype,), caller, int(actual.id),
                        ),
                        (
                            (str(callee.name), int(formal.id)),
                            (exact_dtype,), callee, int(formal.id),
                        ),
                    )
                    for row, proposed, owner, owner_value_id in facts:
                        incumbent = page.latest(row)
                        if incumbent is not None and tuple(incumbent) != proposed:
                            raise ValueError(
                                'exact region feed dtype concordance '
                                f'disagreement for {row!r}: '
                                f'recorded={incumbent!r}, proposed={proposed!r}'
                            )
                        if incumbent is None:
                            value_cell = ssa_value_identity_cell(
                                owner, owner_value_id, book=book,
                            )
                            if value_cell is None:
                                raise ConcordanceRefusal(
                                    f"exact region feed {row!r} has no published SSA identity"
                                )
                            else:
                                book.post(
                                    EXACT_REGION_FEED_DTYPE, row, proposed,
                                    stage=PHYSICAL_CALL_INPUT_ADAPTATION,
                                    provenance=Derived(tuple(dict.fromkeys(cell for cell in (
                                        value_cell,
                                        ssa_value_identity_cell(caller, int(actual.id), book=book),
                                        ssa_value_identity_cell(caller, instruction.res.id, book=book)
                                        if instruction.res is not None else None,
                                    ) if cell is not None))),
                                    mode=Mode.CONCORD,
                                )

                    actual_accounting = dict(actual.accounting or {})
                    formal_accounting = dict(formal.accounting or {})
                    actual_physical = actual_accounting.get('physical_dtype')
                    resident = residents.get(int(actual.id))
                    if resident is not None:
                        resident_type = (resident.accounting or {}).get('physical_dtype') or resident.dtype
                        if resident_type in _NUMERIC_DTYPES:
                            actual_physical = resident_type
                            if resident_type != exact_dtype:
                                instruction.args[position] = resident
                                actual = resident
                                actual_accounting = dict(resident.accounting or {})
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


def _infer_pointer_storage(functions, native_storage=None):
    """Recover typed LLVM load/store contracts without changing pointer types."""
    if native_storage is not None:
        for function in functions.values():
            for formal in function.args:
                if formal.dtype != 'ptr' or (formal.accounting or {}).get('physical_dtype'):
                    continue
                dtype = native_storage.get((function.name, int(formal.id)))
                if dtype is not None:
                    formal.accounting = {**dict(formal.accounting or {}),
                        'physical_dtype': dtype, 'physical_dtype_provenance': 'typed_memory_operations'}
        return
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


def _read_only_module_inputs(functions):
    """Encode existing SSA identities/operand occurrences, never infer effects.

    Opcode/base/Store roles and accepted physical call contracts are the SAME
    ones read by _read_only_feed. The native program owns all transitive alias
    closure, recursive-call closure and final effect-table writes.
    """
    identities = {}
    for function in functions.values():
        for value in chain(function.args, (value for block in function.blocks.values()
                for instruction in block.instrs for value in (*instruction.args, instruction.res)
                if value is not None)):
            identities.setdefault((function.name, int(value.id)), len(identities))
    blocked = [0] * len(identities)
    kinds, results, offsets, operands, formals, writes = [], [], [0], [], [], []
    active, memory_addresses, memory_dtypes = [], [], []
    canonical = {'double': 'float64', 'float': 'float32', 'i64': 'int64', 'i32': 'int32', 'i1': 'bool'}
    dtype_ids = {'ptr': 1}
    for function in functions.values():
        interface = function.metadata.get('native_interface') or function.metadata.get('linked_piece_interface')
        if interface:
            for (owner, value_id), position in identities.items():
                if owner == function.name:
                    blocked[position] = int(not interface['read_only_arguments'].get(value_id, False))
        for block in function.blocks.values():
            for instruction in block.instrs:
                # View belongs to read-only address closure, NOT the narrower
                # existing pointer-storage closure. Keep these laws distinct.
                kinds.append(2 if instruction.op == 'View' else int(
                    instruction.op in {'GEP', 'GetElementPtr', 'getelementptr', 'PtrCast'}))
                active.append(int(not interface))
                results.append(-1 if instruction.res is None else identities[(function.name, int(instruction.res.id))])
                memory_address, memory_dtype = -1, None
                if instruction.op == 'Load' and instruction.args and instruction.res is not None:
                    memory_address = identities[(function.name, int(instruction.args[0].id))]
                    memory_dtype = instruction.res.dtype
                elif instruction.op == 'Store' and len(instruction.args) >= 2:
                    memory_address = identities[(function.name, int(instruction.args[1].id))]
                    memory_dtype = instruction.args[0].dtype
                memory_addresses.append(memory_address)
                if memory_dtype is None:
                    memory_dtypes.append(0)
                else:
                    dtype = canonical.get(memory_dtype, memory_dtype)
                    memory_dtypes.append(dtype_ids.setdefault(dtype, len(dtype_ids) + 1))
                callee = functions.get(str(instruction.attributes.get('callee', '')))
                matched = callee is not None and len(instruction.args) == len(callee.args)
                for index, argument in enumerate(instruction.args):
                    operands.append(identities[(function.name, int(argument.id))])
                    formal, write = -1, False
                    if instruction.op in {'Store', 'store'}:
                        write = index != 0
                    elif instruction.op in {'Call', 'call'}:
                        write = _call_writes_argument(instruction, index)
                        if not write and matched:
                            formal = identities[(callee.name, int(callee.args[index].id))]
                        elif not write:
                            output = instruction.attributes.get('ssa_output_argument')
                            write = output is None or int(output) == index
                    formals.append(formal if not interface else -1)
                    writes.append(int(write and not interface))
                offsets.append(len(operands))
    return tuple(identities), dict(n=len(identities), instruction_count=len(kinds), use_count=len(operands),
        blocked=blocked, kinds=kinds, results=results, argument_offsets=offsets, operands=operands,
        call_formals=formals, writes=writes, active=active,
        memory_addresses=memory_addresses, memory_dtypes=memory_dtypes,
        _dtype_names={code: dtype for dtype, code in dtype_ids.items()})


def _read_only_feed(
    function, value_id, functions=None, active=frozenset(), diagnostics=None,
):
    interface = function.metadata.get('native_interface') or function.metadata.get('linked_piece_interface')
    if interface:
        # Empty declaration blocks do not prove purity. Use the accepted
        # callee's exact per-formal effect summary, not a body walk.
        return interface['read_only_arguments'].get(int(value_id), False)
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
                    and instruction.args and int(instruction.args[0].id) in aliases
                    and int(instruction.res.id) not in aliases):
                # ``heap[position] = target`` writes heap, not position.
                # Treating the GEP's index as an address alias marked this
                # scalar feed writable, blocked its numeric Cast, and made
                # the float-view region read int64 position 1 as ~5e-324.
                # The address/view aliases only its base (operand zero).
                aliases.add(int(instruction.res.id))
                changed = True
    for instruction in instructions:
        if instruction.res is not None and int(instruction.res.id) == int(value_id):
            return False
        uses = [index for index, value in enumerate(instruction.args) if int(value.id) in aliases]
        if not uses:
            continue
        if instruction.op in {'Store', 'store'}:
            # Store reads operand zero and writes through operand one.
            # ``status[0] = size`` in the compiler's graph-frontier kernel
            # needs an integer->float region-feed Cast for ``size``. Marking
            # its VALUE operand writable refused that conversion and left
            # the callee interpreting integer bytes as double. Only a path
            # into the destination is a write to this formal's storage.
            if any(position != 0 for position in uses):
                return False
            continue
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
    from .native_compiler_accelerators import READ_ONLY_MODULE, compiler_accelerators, load_storage_effects_accelerator
    effect_provider = compiler_accelerators.resolve(READ_ONLY_MODULE)
    if effect_provider.tier != 'native':
        installed = load_storage_effects_accelerator()
        if installed is not None:
            effect_provider = installed
    native_effects, native_storage = (effect_provider.run(functions, include_storage_types=True)
        if effect_provider.tier == 'native' else (None, None))
    _infer_pointer_storage(functions, native_storage)
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
    if native_effects is not None:
        report(f'physical-call adaptation round {round_index}: native whole-module effects; '
               f'values={len(native_effects)}; provider={effect_provider.name}')

    def read_only(function, value_id):
        if native_effects is not None:
            return native_effects[(function.name, int(value_id))]
        return _read_only_feed(function, value_id, functions, diagnostics=diagnostics)

    for caller in functions.values():
        from .ssa_record_return_state import accepted_ssa_component
        if accepted_ssa_component(caller):
            continue
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
                            mint_ssa_value(caller, CONTROL_CAST, (source,),
                                dtype='float64', stage=PHYSICAL_CALL_INPUT_ADAPTATION),
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
                        current_identity_book().post(KERNEL_INPUT_CONVERSION, row, (
                            int(source.id), int(converted.id),
                            str(source_dtype), 'float64',
                            'broadcast_double',
                        ), stage=PHYSICAL_CALL_INPUT_ADAPTATION,
                            provenance=Derived(identity_cells(caller, source, converted)),
                            mode=Mode.REVISE)
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
                        logical_pointer_scalar_conversion = bool(
                            not actual.shape
                            and not accounting.get('program_abi_storage')
                            and not accounting.get('physical_dtype')
                            and int(actual.id) not in caller_formals
                            and int(accounting.get('program_abi_rank') or 0) == 0
                            and int(accounting.get('ssa_call_rank') or 0) == 0
                            and accounting.get('program_abi_storage') not in {'span', 'table', 'keyed'}
                            and not accounting.get('sequence_arena')
                            and 'sequence_row_shape' not in accounting
                            and formal.dtype == 'ptr'
                            and source_dtype in numeric
                            and target_dtype in numeric
                            and source_dtype != target_dtype
                            and not _call_writes_argument(instruction, index)
                            and read_only(callee, formal.id)
                        )
                        # Authored `array[2] = False` produces a local bool
                        # Const, not a caller formal or an explicitly physical
                        # slot. The pointer contract still requires a VALUE
                        # conversion to its inferred double element type;
                        # without it C/LLVM read eight bytes from a bool slot.
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
                            and read_only(callee, formal.id)
                        )
                        # A float32 SPAN is stored as float32 (its own storage
                        # class), so an actual of that kind has physical storage
                        # a double-backed formal cannot read in place.
                        float32_span = bool(
                            tuple(actual.shape or ())
                            and str(source_dtype).casefold() == 'float32'
                        )
                        physical_conversion = not (
                            not (accounting.get('program_abi_storage') or accounting.get('physical_dtype')
                                 or int(actual.id) in caller_formals or float32_span)
                            or source_dtype == target_dtype
                            or (not scalar_region and formal.dtype != 'ptr' and actual.dtype != formal.dtype)
                            or (scalar_region and actual.shape)
                            or (not actual.shape and (accounting.get('program_abi_storage') == 'span'
                                or int(accounting.get('program_abi_rank') or 0) > 0))
                            or _call_writes_argument(instruction, index)
                            or not read_only(callee, formal.id)
                        )
                        if not (logical_scalar_conversion or logical_pointer_scalar_conversion or physical_conversion):
                            continue
                        # Only numerical scalar representation changes belong
                        # here. Records, references and string tokens are not
                        # interchangeable numerical values.
                        if logical_scalar_conversion:
                            target_dtype = formal.dtype
                        if source_dtype not in numeric or target_dtype not in numeric:
                            continue
                        # Shape is retained; a span conversion is elementwise.
                        # These six energy-helper conversions used to mint
                        # outside the book. The source value and the exact
                        # callee formal are the causes of this ABI conversion.
                        converted = SSAValue(mint_ssa_value(caller, CONTROL_CAST,
                            (actual, ssa_value_identity_cell(callee, formal.id)),
                            dtype=target_dtype, shape=actual.shape,
                            stage=PHYSICAL_CALL_INPUT_ADAPTATION), dtype=target_dtype, shape=actual.shape, device=actual.device,
                            accounting={'physical_dtype': target_dtype,
                                'call_input_conversion': (caller.name, int(actual.id), callee.name, int(formal.id)),
                                'call_input_conversion_kind': (
                                    'read_only_scalar_numeric'
                                    if logical_scalar_conversion or logical_pointer_scalar_conversion
                                    else 'physical_region'
                                )})
                        conversion_attributes = {
                            'target_dtype': target_dtype,
                            'source_dtype': source_dtype,
                        }
                        if logical_scalar_conversion or logical_pointer_scalar_conversion:
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
                                if logical_scalar_conversion or logical_pointer_scalar_conversion
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
                        current_identity_book().post(CALL_INPUT_CONVERSION, row, (
                            int(converted.id), str(source_dtype),
                            str(target_dtype), conversion['kind'],
                        ), stage=PHYSICAL_CALL_INPUT_ADAPTATION,
                            provenance=Derived(identity_cells(caller, actual, converted,
                                ssa_value_identity_cell(callee, formal.id))),
                            mode=Mode.REVISE)
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
                        temporary = SSAValue(mint_ssa_value(caller, CONTROL_CAST,
                            (original, ssa_value_identity_cell(callee, formal.id)),
                            dtype=backend_dtype, shape=original.shape,
                            stage=PHYSICAL_CALL_INPUT_ADAPTATION), dtype=backend_dtype, shape=original.shape, device=original.device,
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
    from .concordance_declarations import (
        EXACT_FORMAL_DTYPE, EXACT_REGION_FEED_DTYPE,
        PHYSICAL_CALL_INPUT_ADAPTATION_ROUND,
    )
    from .identity_concordance import ConcordanceRefusal, Derived, Mode
    from .ssa_record_return_state import accepted_ssa_component, ssa_block_identity_cell
    calls = [(caller, block, instruction)
             for caller in functions.values() for block in caller.blocks.values()
             for instruction in block.instrs if instruction.op in {'Call', 'call'}
             and not accepted_ssa_component(caller)]
    if not calls:
        return 0
    book = current_identity_book()
    round_scope = book.mint_scope(
        ('physical-call-input-adaptation', tuple(sorted(map(str, functions)))),
        PHYSICAL_CALL_INPUT_ADAPTATION,
    )
    total = 0

    def round_sources(previous_edges=()):
        sources = []
        for caller, block, instruction in calls:
            sources.append(ssa_block_identity_cell(caller, block.name, book=book))
            callee = functions.get(str(instruction.attributes.get('callee', '')))
            for owner, values in ((caller, instruction.args),
                                  (callee, () if callee is None else callee.args)):
                if owner is None:
                    continue
                for value in values:
                    sources.append(ssa_value_identity_cell(owner, int(value.id), book=book))
                    for declaration in (EXACT_REGION_FEED_DTYPE, EXACT_FORMAL_DTYPE):
                        sources.append(book.latest_ref(declaration, (str(owner.name), int(value.id))))
            if callee is not None:
                for value, formal in zip(instruction.args, callee.args):
                    sources.append(book.latest_ref(CALL_INPUT_CONVERSION,
                        (str(caller.name), int(value.id), str(callee.name), int(formal.id))))
        sources.extend(book.latest_ref(CALL_INPUT_CONVERSION, edge) for edge in previous_edges)
        return tuple(dict.fromkeys(cell for cell in sources if cell is not None))
    report = progress or (lambda _message: None)
    report(
        'physical-call adaptation fixed point start: '
        f'functions={len(functions)} call_operands={call_operand_count} '
        f'round_bound={round_bound}'
    )
    state = _adaptation_state_signature(functions)
    seen_states = {state: -1}
    for round_index in range(round_bound):
        sources = round_sources()
        if not sources:
            raise ConcordanceRefusal("physical call-input adaptation has no recorded call inputs")
        previous_edges = tuple(
            (str(caller.name), int(value.id), str(callee.name), int(formal.id))
            for caller, _block, instruction in calls
            if (callee := functions.get(str(instruction.attributes.get('callee', '')))) is not None
            for value, formal in zip(instruction.args, callee.args)
        )
        changed = _adapt_physical_call_inputs_round(
            functions, progress=progress, round_index=round_index,
        )
        book.post(PHYSICAL_CALL_INPUT_ADAPTATION_ROUND, (round_scope, round_index),
                  int(changed), stage=PHYSICAL_CALL_INPUT_ADAPTATION,
                  provenance=Derived(tuple(dict.fromkeys((*sources, *round_sources(previous_edges))))),
                  mode=Mode.CONCORD)
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
