"""Adapt logical values to already compiled physical buffer contracts."""
from __future__ import annotations

from itertools import chain
from math import prod

from .monotonic_ids import GLOBAL_MONOTONIC_IDS
from .identity_concordance import current_identity_book
from ..transmogrifier.ssa import Instr, SSAValue


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


def _read_only_feed(function, value_id, functions=None, active=frozenset()):
    identity = (function.name, int(value_id))
    if identity in active:
        return True  # This path adds no new operation; other exits are still inspected.
    active = active | {identity}
    functions = functions or {}
    aliases = {int(value_id)}
    instructions = [instruction for block in function.blocks.values() for instruction in block.instrs]
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
            if aliases.intersection(map(int, instruction.attributes.get('output_ids', ()))):
                return False
            callee = functions.get(str(instruction.attributes.get('callee', '')))
            if callee is not None and len(instruction.args) == len(callee.args):
                if any(not _read_only_feed(callee, callee.args[index].id, functions, active) for index in uses):
                    return False
                continue
            output = instruction.attributes.get('ssa_output_argument')
            if output is None or int(output) in uses:
                return False
    return True


def adapt_physical_call_inputs(functions) -> int:
    """Keep storage typed and bridge read-only inputs and fresh helper results.

    Regions precede physical record layout discovery. Tensor backends may also
    represent logical Boolean values in double buffers. Convert at these value
    boundaries rather than changing the shared caller storage representation.
    """
    _intern_unshadowed_formal_uses(functions)
    _infer_pointer_storage(functions)
    count = 0
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
                    outputs = set(map(int, instruction.attributes.get('output_ids', ())))
                    for index, (actual, formal) in enumerate(zip(instruction.args, callee.args)):
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
                            and source_dtype in numeric
                            and formal.dtype in numeric
                            and source_dtype != formal.dtype
                            and int(actual.id) not in outputs
                            and _read_only_feed(callee, formal.id, functions)
                        )
                        physical_conversion = not (
                            not (accounting.get('program_abi_storage') or accounting.get('physical_dtype')
                                 or int(actual.id) in caller_formals)
                            or source_dtype == target_dtype
                            or (not scalar_region and formal.dtype != 'ptr' and actual.dtype != formal.dtype)
                            or (scalar_region and actual.shape)
                            or (not actual.shape and (accounting.get('program_abi_storage') == 'span'
                                or int(accounting.get('program_abi_rank') or 0) > 0))
                            or int(actual.id) in outputs
                            or not _read_only_feed(callee, formal.id, functions)
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
    return count
