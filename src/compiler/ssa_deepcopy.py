"""Generic, iterative, reference-preserving deep copy as repository SSA.

One routine, driven entirely by data (a uniform ``TypeDescriptor`` layout),
handles any record/reference graph the compiler encounters -- not a
per-class generated routine, not a Python-side loop of independent
GetAttr/SetAttr calls. Built directly from repository SSA
(``Instr``/``BasicBlock``/``Function``), not raw LLVM IR text, so it can be
validated with this compiler's own SSA tooling.

Memory layout (all fields are pointer-width words; a target's own pointer
size applies, this module only counts words):

    TypeDescriptor:
        [0] size          -- byte size of one instance of this type
        [1] offset_count  -- number of pointer-valued slots in it
        [2] offsets       -- ptr to an i64[offset_count] array of byte
                             offsets, each naming one pointer-valued slot
        [3] child_descs   -- ptr to a ptr[offset_count] array; entry k is
                             the TypeDescriptor of whatever offsets[k] points
                             at

    WorkItem (one bounded stack slot):
        [0] src        -- source pointer to duplicate
        [1] desc        -- src's TypeDescriptor
        [2] patch_at    -- address to receive the new pointer once known;
                            0/null marks the root item, whose result becomes
                            the routine's return value instead of a patch

    SeenEntry (one bounded lookup-table slot):
        [0] original    -- a source pointer already duplicated
        [1] duplicate   -- its new pointer

The seen-table is what makes this correct for shared references and
cycles: a source pointer reachable two different ways (or reachable from
itself) is duplicated exactly once, and every reference to it patches to
the same new pointer, instead of silently duplicating it again.

This is UNVERIFIED: written directly against the repository SSA
constructors (no test yet exercises it). It is one nested pair of
iterative loops (the worklist stack; the offset scan) plus a bounded
linear search (the seen-table) -- no recursion anywhere, matching the
explicit "iterative, not recursive" requirement this was built against.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..transmogrifier.ssa import (
    BasicBlock, Function, Instr, SSAValue,
    SSARecordFieldStorage, SSARecordTable,
)
from ..transmogrifier.ssa_registry import Handler

WORD_BYTES = 8
MAX_STACK_DEPTH = 4096
MAX_SEEN_ENTRIES = 4096
DEEPCOPY_FUNCTION_NAME = "turing_deepcopy"


def build_deepcopy_function(
    *, max_stack_depth: int = MAX_STACK_DEPTH,
    max_seen_entries: int = MAX_SEEN_ENTRIES,
) -> Function:
    """Construct the generic ``turing_deepcopy(root, root_desc) -> ptr``.

    Every value id and block name is minted fresh inside this function --
    it is self-contained and can be linked into any module the same way
    ``ssa_features.py``'s feature modules are, by name.
    """

    next_id = [1]

    def value(dtype: str = "ptr") -> SSAValue:
        result = SSAValue(next_id[0], dtype=dtype)
        next_id[0] += 1
        return result

    def const(dtype: str, literal) -> SSAValue:
        result = value(dtype)
        result.accounting["constant"] = literal
        return result

    blocks: dict[str, BasicBlock] = {}

    def block(name: str) -> BasicBlock:
        created = BasicBlock(name=name, instrs=[])
        blocks[name] = created
        return created

    def emit(target: BasicBlock, op: str, args: list[SSAValue], res: SSAValue | None = None, **attributes) -> SSAValue | None:
        target.instrs.append(Instr(op, args, res, attributes=dict(attributes)))
        return res

    def br(target: BasicBlock, destination: str) -> None:
        emit(target, Handler.Br.value, [], attributes={"target": destination})
        target.successors.append(destination)

    def condbr(target: BasicBlock, condition: SSAValue, true_name: str, false_name: str) -> None:
        emit(
            target, Handler.CondBr.value, [condition], None,
            true_target=true_name, false_target=false_name,
        )
        target.successors.extend((true_name, false_name))

    def phi(target: BasicBlock, dtype: str, incoming: list[tuple[SSAValue, str]]) -> SSAValue:
        result = value(dtype)
        emit(
            target, Handler.Phi.value,
            [source for source, _predecessor in incoming], result,
            incoming=[(source.id, predecessor) for source, predecessor in incoming],
        )
        return result

    root = SSAValue(next_id[0], dtype="ptr"); next_id[0] += 1
    root_desc = SSAValue(next_id[0], dtype="ptr"); next_id[0] += 1

    # ---- entry: allocate the two bounded tables, seed the worklist with
    # the root item (patch_at = null marks it as the root). ----
    entry = block("entry")
    work_stack = emit(
        entry, Handler.Alloca.value, [], value("ptr"),
        element_bytes=3 * WORD_BYTES, count=max_stack_depth,
    )
    seen_table = emit(
        entry, Handler.Alloca.value, [], value("ptr"),
        element_bytes=2 * WORD_BYTES, count=max_seen_entries,
    )

    def slot_ptr(target: BasicBlock, base: SSAValue, index: SSAValue, stride_words: int, field_index: int) -> SSAValue:
        stride = const("int64", stride_words * WORD_BYTES)
        byte_index = value("int64")
        emit(target, Handler.Mul.value, [index, stride], byte_index)
        offset = const("int64", field_index * WORD_BYTES)
        total = value("int64")
        emit(target, Handler.Add.value, [byte_index, offset], total)
        return emit(target, Handler.GetElementPtr.value, [base, total], value("ptr"))

    zero_idx = const("int64", 0)
    root_src_slot = slot_ptr(entry, work_stack, zero_idx, 3, 0)
    emit(entry, Handler.Store.value, [root_src_slot, root], None)
    root_desc_slot = slot_ptr(entry, work_stack, zero_idx, 3, 1)
    emit(entry, Handler.Store.value, [root_desc_slot, root_desc], None)
    root_patch_slot = slot_ptr(entry, work_stack, zero_idx, 3, 2)
    null_ptr = const("ptr", 0)
    emit(entry, Handler.Store.value, [root_patch_slot, null_ptr], None)
    sp_initial = const("int64", 1)
    seen_count_initial = const("int64", 0)
    root_result_initial = const("ptr", 0)
    br(entry, "loop_header")

    # ---- loop_header: any work left? ----
    loop_header = block("loop_header")
    sp = phi(loop_header, "int64", [(sp_initial, "entry")])
    seen_count = phi(loop_header, "int64", [(seen_count_initial, "entry")])
    root_result = phi(loop_header, "ptr", [(root_result_initial, "entry")])
    zero_i64 = const("int64", 0)
    has_work = value("bool")
    emit(loop_header, Handler.Ne.value, [sp, zero_i64], has_work)
    condbr(loop_header, has_work, "pop", "done")

    done = block("done")
    emit(done, Handler.Ret.value, [root_result], None)

    # ---- pop: read the top work item. ----
    pop = block("pop")
    one_i64 = const("int64", 1)
    sp_after_pop = value("int64")
    emit(pop, Handler.Sub.value, [sp, one_i64], sp_after_pop)
    popped_src = emit(pop, Handler.Load.value, [slot_ptr(pop, work_stack, sp_after_pop, 3, 0)], value("ptr"))
    popped_desc = emit(pop, Handler.Load.value, [slot_ptr(pop, work_stack, sp_after_pop, 3, 1)], value("ptr"))
    popped_patch_at = emit(pop, Handler.Load.value, [slot_ptr(pop, work_stack, sp_after_pop, 3, 2)], value("ptr"))
    search_j_initial = const("int64", 0)
    br(pop, "search_header")

    # ---- search_header/body/next: linear scan of the seen-table for
    # ``popped_src``, so an already-duplicated pointer is never copied
    # twice -- this is what preserves shared references and cycles. ----
    search_header = block("search_header")
    search_j = phi(search_header, "int64", [(search_j_initial, "pop")])
    search_continue = value("bool")
    emit(search_header, Handler.Lt.value, [search_j, seen_count], search_continue)
    condbr(search_header, search_continue, "search_body", "search_miss")

    search_body = block("search_body")
    candidate = emit(search_body, Handler.Load.value, [slot_ptr(search_body, seen_table, search_j, 2, 0)], value("ptr"))
    is_match = value("bool")
    emit(search_body, Handler.Eq.value, [candidate, popped_src], is_match)
    condbr(search_body, is_match, "search_hit", "search_next")

    search_next = block("search_next")
    search_j_next = value("int64")
    emit(search_next, Handler.Add.value, [search_j, one_i64], search_j_next)
    blocks["search_header"].instrs[0].attributes["incoming"].append((search_j_next.id, "search_next"))
    br(search_next, "search_header")

    search_hit = block("search_hit")
    hit_new_ptr = emit(search_hit, Handler.Load.value, [slot_ptr(search_hit, seen_table, search_j, 2, 1)], value("ptr"))
    br(search_hit, "patch")

    # ---- search_miss: genuinely new -- allocate, byte-copy, record it,
    # then scan its pointer-valued slots. ----
    search_miss = block("search_miss")
    copy_size = emit(search_miss, Handler.Load.value, [popped_desc], value("int64"))
    new_ptr = emit(search_miss, Handler.Call.value, [copy_size], value("ptr"), callee="malloc")
    emit(search_miss, Handler.Call.value, [new_ptr, popped_src, copy_size], None, callee="memcpy")
    seen_original_slot = slot_ptr(search_miss, seen_table, seen_count, 2, 0)
    emit(search_miss, Handler.Store.value, [seen_original_slot, popped_src], None)
    seen_new_slot = slot_ptr(search_miss, seen_table, seen_count, 2, 1)
    emit(search_miss, Handler.Store.value, [seen_new_slot, new_ptr], None)
    seen_count_recorded = value("int64")
    emit(search_miss, Handler.Add.value, [seen_count, one_i64], seen_count_recorded)
    desc_offset_count_ptr = slot_ptr(search_miss, popped_desc, zero_idx, 1, 1)
    offset_count = emit(search_miss, Handler.Load.value, [desc_offset_count_ptr], value("int64"))
    desc_offsets_ptr_slot = slot_ptr(search_miss, popped_desc, zero_idx, 1, 2)
    offsets_base = emit(search_miss, Handler.Load.value, [desc_offsets_ptr_slot], value("ptr"))
    desc_child_descs_slot = slot_ptr(search_miss, popped_desc, zero_idx, 1, 3)
    child_descs_base = emit(search_miss, Handler.Load.value, [desc_child_descs_slot], value("ptr"))
    scan_i_initial = const("int64", 0)
    br(search_miss, "scan_header")

    # ---- scan_header/body/push/next: for each pointer-valued slot in the
    # freshly copied block, read it; a non-null slot enqueues its target
    # (as its own work item, patch_at pointing back at this slot in the
    # new copy) instead of dereferencing it here -- the recursion this
    # replaces becomes another iteration of the same outer loop. ----
    scan_header = block("scan_header")
    scan_i = phi(scan_header, "int64", [(scan_i_initial, "search_miss")])
    sp_grow = phi(scan_header, "int64", [(sp_after_pop, "search_miss")])
    scan_continue = value("bool")
    emit(scan_header, Handler.Lt.value, [scan_i, offset_count], scan_continue)
    condbr(scan_header, scan_continue, "scan_body", "scan_done")

    scan_body = block("scan_body")
    offset_word = value("int64")
    emit(scan_body, Handler.GetElementPtr.value, [offsets_base, scan_i], offset_word, element_bytes=WORD_BYTES)
    offset = emit(scan_body, Handler.Load.value, [offset_word], value("int64"))
    child_slot = emit(scan_body, Handler.GetElementPtr.value, [popped_src, offset], value("ptr"))
    child_src = emit(scan_body, Handler.Load.value, [child_slot], value("ptr"))
    child_is_null = value("bool")
    emit(scan_body, Handler.Eq.value, [child_src, null_ptr], child_is_null)
    condbr(scan_body, child_is_null, "scan_continue_block", "scan_push")

    scan_push = block("scan_push")
    child_desc_word = value("ptr")
    emit(scan_push, Handler.GetElementPtr.value, [child_descs_base, scan_i], child_desc_word, element_bytes=WORD_BYTES)
    child_desc = emit(scan_push, Handler.Load.value, [child_desc_word], value("ptr"))
    patch_target = emit(scan_push, Handler.GetElementPtr.value, [new_ptr, offset], value("ptr"))
    push_src_slot = slot_ptr(scan_push, work_stack, sp_grow, 3, 0)
    emit(scan_push, Handler.Store.value, [push_src_slot, child_src], None)
    push_desc_slot = slot_ptr(scan_push, work_stack, sp_grow, 3, 1)
    emit(scan_push, Handler.Store.value, [push_desc_slot, child_desc], None)
    push_patch_slot = slot_ptr(scan_push, work_stack, sp_grow, 3, 2)
    emit(scan_push, Handler.Store.value, [push_patch_slot, patch_target], None)
    sp_grow_pushed = value("int64")
    emit(scan_push, Handler.Add.value, [sp_grow, one_i64], sp_grow_pushed)
    br(scan_push, "scan_continue_block")

    scan_continue_block = block("scan_continue_block")
    sp_grow_out = phi(
        scan_continue_block, "int64",
        [(sp_grow, "scan_body"), (sp_grow_pushed, "scan_push")],
    )
    scan_i_next = value("int64")
    emit(scan_continue_block, Handler.Add.value, [scan_i, one_i64], scan_i_next)
    blocks["scan_header"].instrs[0].attributes["incoming"].append((scan_i_next.id, "scan_continue_block"))
    blocks["scan_header"].instrs[1].attributes["incoming"].append((sp_grow_out.id, "scan_continue_block"))
    br(scan_continue_block, "scan_header")

    scan_done = block("scan_done")
    br(scan_done, "patch")

    # ---- patch: merge the "already seen" and "freshly copied" paths,
    # write the new pointer into whoever referenced this source (unless
    # this was the root item, which instead becomes the return value). ----
    patch = block("patch")
    patched_new_ptr = phi(patch, "ptr", [(hit_new_ptr, "search_hit"), (new_ptr, "scan_done")])
    patched_sp = phi(patch, "int64", [(sp_after_pop, "search_hit"), (sp_grow_out, "scan_done")])
    patched_seen_count = phi(
        patch, "int64",
        [(seen_count, "search_hit"), (seen_count_recorded, "scan_done")],
    )
    patched_patch_at = phi(
        patch, "ptr",
        [(popped_patch_at, "search_hit"), (popped_patch_at, "scan_done")],
    )
    is_root_item = value("bool")
    emit(patch, Handler.Eq.value, [patched_patch_at, null_ptr], is_root_item)
    condbr(patch, is_root_item, "mark_root", "do_patch")

    do_patch = block("do_patch")
    emit(do_patch, Handler.Store.value, [patched_patch_at, patched_new_ptr], None)
    br(do_patch, "loop_tail")

    mark_root = block("mark_root")
    br(mark_root, "loop_tail")

    loop_tail = block("loop_tail")
    root_result_next = phi(
        loop_tail, "ptr",
        [(root_result, "do_patch"), (patched_new_ptr, "mark_root")],
    )
    blocks["loop_header"].instrs[0].attributes["incoming"].append((patched_sp.id, "loop_tail"))
    blocks["loop_header"].instrs[1].attributes["incoming"].append((patched_seen_count.id, "loop_tail"))
    blocks["loop_header"].instrs[2].attributes["incoming"].append((root_result_next.id, "loop_tail"))
    br(loop_tail, "loop_header")

    return Function(
        name=DEEPCOPY_FUNCTION_NAME,
        args=[root, root_desc],
        blocks=blocks,
        metadata={"return_value": SSAValue(0, dtype="ptr")},
    )


_POINTER_FIELD_STORAGE = (
    SSARecordFieldStorage.RECORD, SSARecordFieldStorage.REFERENCE,
)


@dataclass(frozen=True)
class TypeDescriptorEntry:
    """One compile-time-derived ``TypeDescriptor``, ready to be laid out as
    constant data: ``size`` in bytes, and one ``(byte_offset, child_index)``
    pair per pointer-valued field -- ``child_index`` is this entry's own
    index into the same table, not a second lookup structure.
    """

    record_id: int
    size: int
    pointer_fields: tuple[tuple[int, int], ...]


def build_type_descriptor_table(
    table: "SSARecordTable", root_record_id: int,
) -> tuple[TypeDescriptorEntry, ...]:
    """Enumerate every distinct record type reachable from one root record.

    Iterative (explicit worklist, no recursion), same shape as the runtime
    engine's own traversal: a record id already indexed is never re-walked,
    so a type that is reachable more than once, or a directly or
    indirectly self-referential type, still produces exactly one entry.

    A field's byte offset is its own ``offset`` when the compiler already
    assigned one; otherwise this assigns one word per field, in field
    order, rather than inventing a second layout convention. ``size`` is
    one word past the last field's own offset -- the smallest layout
    consistent with the offsets actually used, not a guess.
    """

    order: list[int] = []
    index_of: dict[int, int] = {}
    worklist = [int(root_record_id)]
    queued = {int(root_record_id)}
    while worklist:
        current_id = worklist.pop()
        if current_id in index_of:
            continue
        index_of[current_id] = len(order)
        order.append(current_id)
        descriptor = table.records.get(current_id)
        if descriptor is None:
            continue
        for record_field in descriptor.fields:
            if (
                record_field.storage in _POINTER_FIELD_STORAGE
                and record_field.record_id is not None
                and int(record_field.record_id) not in queued
            ):
                queued.add(int(record_field.record_id))
                worklist.append(int(record_field.record_id))

    entries: list[TypeDescriptorEntry] = []
    for current_id in order:
        descriptor = table.records.get(current_id)
        if descriptor is None:
            entries.append(TypeDescriptorEntry(current_id, WORD_BYTES, ()))
            continue
        pointer_fields: list[tuple[int, int]] = []
        highest_offset_word = 0
        for field_index, record_field in enumerate(descriptor.fields):
            offset_words = (
                record_field.offset if record_field.offset is not None
                else field_index
            )
            highest_offset_word = max(highest_offset_word, offset_words)
            if (
                record_field.storage in _POINTER_FIELD_STORAGE
                and record_field.record_id is not None
            ):
                child_index = index_of[int(record_field.record_id)]
                pointer_fields.append((
                    offset_words * WORD_BYTES, child_index,
                ))
        entries.append(TypeDescriptorEntry(
            current_id,
            (highest_offset_word + 1) * WORD_BYTES,
            tuple(pointer_fields),
        ))
    return tuple(entries)


__all__ = [
    "DEEPCOPY_FUNCTION_NAME",
    "MAX_SEEN_ENTRIES",
    "MAX_STACK_DEPTH",
    "WORD_BYTES",
    "TypeDescriptorEntry",
    "build_deepcopy_function",
    "build_type_descriptor_table",
]
