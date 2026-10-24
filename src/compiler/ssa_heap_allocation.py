"""Explicit owned heap storage using the existing Alloca/Call vocabulary.

An owner is one caller-owned native pointer cell, initially null. Allocations
link into that owner and survive the allocating function's return. Release is
explicit; the owner must outlive every arena allocated through it. This states
memory ownership, not a Python object representation or a callback ABI.
"""
from __future__ import annotations

from ..transmogrifier.dtype_layout import layout_for
from ..transmogrifier.ssa import Instr, SSAValue
from ..transmogrifier.ssa_registry import Handler


HEAP_RELEASE_SYMBOL = "turing_heap_release"
HEAP_COPY_SYMBOL = "llvm.memmove.p0.p0.i64"


def project_native_heap_call_contexts(functions) -> None:
    """Publish pointer-cell contracts through exact source-linked call edges.

    The source context micro passed ptr cells through its native constructor,
    but the caller kept only raw ``ptr`` (historically a typed address). Its
    public ABI therefore said double and marshalling copied arena addresses
    away. The native formal already declares the cell and owner: transport
    that contract by callee_input_ids, never infer it from a pointer spelling.
    """
    from collections import deque
    seeds = [name for name, function in functions.items() if any(
        (value.accounting or {}).get('heap_owner')
        or (value.accounting or {}).get('heap_owner_value_id') is not None for value in function.args)]
    if not seeds:
        return  # Ordinary compiler modules do not pay another instruction scan.
    incoming = {}
    inventories = {}
    for function in functions.values():
        occurrences = {}
        for value in function.args:
            occurrences.setdefault(value.id, []).append(value)
        for block in function.blocks.values():
            for instruction in block.instrs:
                for value in (*instruction.args, *((instruction.res,) if instruction.res is not None else ())):
                    occurrences.setdefault(value.id, []).append(value)
        inventories[function.name] = occurrences
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op != 'Call' or not instruction.attributes.get('source_linked'):
                    continue
                callee = functions.get(instruction.attributes.get('callee'))
                if callee is None:
                    continue
                incoming.setdefault(callee.name, []).append((function, instruction))
    # A monotonic reverse-call worklist transports the same context through
    # forwarding layers and recursive SCCs without depending on function order.
    pending = deque(seeds)
    queued = set(pending)
    while pending:
        callee = functions[pending.popleft()]
        queued.remove(callee.name)
        for function, instruction in incoming.get(callee.name, ()):
            ids = tuple(instruction.attributes.get('callee_input_ids', ()))
            if ids != tuple(value.id for value in callee.args) or len(ids) != len(instruction.args):
                raise ValueError('native heap context projection requires exact live callee input identities')
            bindings = dict(zip(ids, instruction.args))
            changed = False
            for formal in callee.args:
                accounting = formal.accounting or {}
                owner_id = accounting.get('heap_owner_value_id')
                if not accounting.get('heap_owner') and owner_id is None:
                    continue
                actual = bindings[formal.id]
                if actual.dtype != 'ptr':
                    raise ValueError('native heap context requires a pointer-cell input, not an opaque handle')
                if owner_id is not None and int(owner_id) not in bindings:
                    raise ValueError('native heap context is missing its exact owner binding')
                published = {'heap_owner': True} if accounting.get('heap_owner') else {
                    'heap_element_dtype': accounting['heap_element_dtype'],
                    'heap_owner_value_id': int(bindings[int(owner_id)].id),
                }
                for occurrence in inventories[function.name][actual.id]:
                    current = dict(occurrence.accounting or {})
                    for key, value in published.items():
                        if key in current and current[key] != value:
                            raise ValueError(f'native heap context projection conflicts with incumbent ownership: {function.name} {actual.id} {key}={current[key]!r} vs {value!r} from {callee.name}')
                    if all(current.get(key) == value for key, value in published.items()):
                        continue
                    occurrence.accounting = {**current, **published,
                        'native_heap_context_from': (callee.name, formal.id,
                            tuple((key, bindings[key].id) for key in ids))}
                    changed = True
            if changed and function.name not in queued:
                pending.append(function.name)
                queued.add(function.name)


def validate_heap_context_projection(module, function_name, value) -> None:
    receipt = (value.accounting or {}).get('native_heap_context_from')
    if receipt is None:
        return
    callee_name, formal_id, pairs = receipt
    callee = module.functions[callee_name]
    formals = {formal.id: formal for formal in callee.args}
    bindings = dict(pairs)
    formal = formals[formal_id]
    accounting = formal.accounting or {}
    if bindings.get(formal_id) != value.id or value.dtype != 'ptr':
        raise ValueError('native heap context projection has a stale storage identity')
    keys = {'heap_owner': True} if accounting.get('heap_owner') else {
        'heap_element_dtype': accounting['heap_element_dtype'],
        'heap_owner_value_id': bindings[accounting['heap_owner_value_id']],
    }
    if any((value.accounting or {}).get(key) != expected for key, expected in keys.items()):
        raise ValueError('native heap context projection disagrees with its live owner/layout')
    if not any(instruction.op == 'Call' and instruction.attributes.get('source_linked')
        and instruction.attributes.get('callee') == callee_name
        and tuple(zip(instruction.attributes.get('callee_input_ids', ()),
                      (argument.id for argument in instruction.args))) == pairs
        for block in module.functions[function_name].blocks.values() for instruction in block.instrs):
        raise ValueError('native heap context projection has no matching live call edge')


def require_heap_owner(owner: SSAValue) -> None:
    if (str(owner.dtype) != "ptr" or tuple(owner.shape or ()) != (1,)
            or not (owner.accounting or {}).get("heap_owner")):
        raise ValueError("heap owner must be a declared mutable ptr cell of shape (1,)")


def validate_heap_allocation(instruction: Instr):
    """Read the one allocation contract without editing it during emission."""
    if (instruction.op != Handler.Alloca.value
            or instruction.attributes.get("allocation_kind") != "owned_heap"
            or len(instruction.args) != 2):
        raise ValueError("Alloca requires an explicit owned-heap allocation contract")
    count, owner = instruction.args
    require_heap_owner(owner)
    count_layout = layout_for(count.dtype)
    if count_layout.kind not in {"int", "uint"} or count_layout.byte_size > 8 or tuple(count.shape or ()):
        raise ValueError("heap count must be an integer scalar")
    layout = layout_for(instruction.attributes["element_dtype"])
    if layout.llvm_type == "ptr" or layout.alignment > 8:
        raise ValueError("heap element layout must be fixed-width and aligned at most 8")
    result = instruction.res
    if result is None or str(result.dtype) != "ptr" or tuple(result.shape or ()):
        raise ValueError("heap result must be a scalar native address, not an opaque handle")
    if (result.accounting or {}).get("heap_element_dtype") != layout.name:
        raise ValueError("Alloca result lacks its declared element-layout accounting")
    if (result.accounting or {}).get("heap_owner_value_id") != int(owner.id):
        raise ValueError("Alloca result and instruction disagree about allocation owner")
    return count, owner, layout


def heap_allocate(count: SSAValue, owner: SSAValue, result: SSAValue,
                  *, element_dtype: str, zero_initialize: bool = False) -> Instr:
    """Declare runtime-sized allocation with caller-controlled lifetime."""
    layout = layout_for(element_dtype)  # Strict: unknown never defaults to double.
    require_heap_owner(owner)
    count_layout = layout_for(count.dtype)
    if count_layout.kind not in {"int", "uint"} or count_layout.byte_size > 8 or tuple(count.shape or ()):
        raise ValueError("heap count must be an integer scalar")
    if layout.llvm_type == "ptr" or layout.alignment > 8:
        raise ValueError("heap element layout must be fixed-width and aligned at most 8")
    if str(result.dtype) != "ptr" or tuple(result.shape or ()):
        raise ValueError("heap result must be a scalar native address, not an opaque handle")
    incumbent = (result.accounting or {}).get("heap_element_dtype")
    if incumbent is not None and layout_for(incumbent) != layout:
        raise ValueError("heap result and allocation disagree about element layout")
    incumbent_owner = (result.accounting or {}).get("heap_owner_value_id")
    if incumbent_owner is not None and int(incumbent_owner) != int(owner.id):
        raise ValueError("heap result and allocation disagree about allocation owner")
    result.accounting = {
        **dict(result.accounting or {}), "heap_element_dtype": layout.name,
        "heap_owner_value_id": int(owner.id),
    }
    instruction = Instr(Handler.Alloca.value, [count, owner], result, attributes={
        "allocation_kind": "owned_heap", "element_dtype": layout.name,
        "zero_initialize": bool(zero_initialize),
    })
    validate_heap_allocation(instruction)
    return instruction


def heap_release(owner: SSAValue) -> Instr:
    """Release every allocation of this owner and reset its cell to null."""
    require_heap_owner(owner)
    return Instr(Handler.Call.value, [owner], None, attributes={
        "callee": HEAP_RELEASE_SYMBOL, "native_heap_abi": "release_owner",
    })


def validate_heap_copy(instruction: Instr):
    """Read a typed element transfer between existing native arena cells."""
    if (instruction.op != Handler.Call.value or instruction.res is not None
            or instruction.attributes.get("callee") != HEAP_COPY_SYMBOL
            or instruction.attributes.get("native_heap_abi") != "copy_elements"
            or len(instruction.args) != 3):
        raise ValueError("heap copy requires destination, source, count and no result")
    destination, source, count = instruction.args
    count_layout = layout_for(count.dtype)
    if count_layout.kind not in {"int", "uint"} or count_layout.byte_size > 8 or tuple(count.shape or ()):
        raise ValueError("heap copy count must be an integer scalar")
    layouts = []
    for value in (destination, source):
        accounting = value.accounting or {}
        if str(value.dtype) != "ptr" or tuple(value.shape or ()):
            raise ValueError("heap copy requires native arena pointer cells, not opaque handles")
        owner_id = accounting.get("heap_owner_value_id")
        if not isinstance(owner_id, int) or isinstance(owner_id, bool) or owner_id < 0:
            raise ValueError("heap copy arena lacks its allocation owner identity")
        layout = layout_for(accounting.get("heap_element_dtype"))
        if layout.llvm_type == "ptr" or layout.alignment > 8:
            raise ValueError("heap copy requires fixed-width element storage")
        layouts.append(layout)
    if layouts[0] != layouts[1]:
        raise ValueError("heap copy source and destination element layouts disagree")
    return destination, source, count, layouts[0]


def heap_copy(count: SSAValue, source: SSAValue, destination: SSAValue) -> Instr:
    """Copy elements byte-exactly; the caller proves both live extents.

    Counts are elements, never guessed bytes. This is the existing native
    memory-transfer ABI, not object graph copying or handle dereferencing.
    Memmove also keeps an explicitly aliased transfer well-defined.
    """
    instruction = Instr(Handler.Call.value, [destination, source, count], None,
        attributes={"callee": HEAP_COPY_SYMBOL, "native_heap_abi": "copy_elements"})
    validate_heap_copy(instruction)
    return instruction
