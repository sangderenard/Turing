"""Expand canonical tensor instructions into ordinary repository SSA.

The code reference owns implementations; this module only supplies the ABI
recipe which turns a canonical tensor call into calls to those implementations.
No tensor executor, backend object, or late runtime dispatch survives this pass.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from typing import Any, Iterable, Mapping

from ..common.tensors.abstraction_methods.indexing import normalize_basic_index
from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import c_tensor_opcode
from ..transmogrifier.dtype_layout import dtype_byte_size
from ..transmogrifier.ssa import (
    IRModule,
    Instr,
    SSAValue,
    SSATensorDescriptor,
    SSATensorTable,
)
from ..transmogrifier.ssa_registry import Handler
from ..transmogrifier.tensor_ssa_reference import SSATensorCodeReference
from .identity_concordance import identity_book
from .ssa_aggregate_abi import (
    _constant_integer,
    legalize_aggregate_adapters,
    legalize_aggregate_output_views,
)
from .monotonic_ids import GLOBAL_MONOTONIC_IDS


def _accounting_view(owner: Any, value: Any, module: Any, labels=None) -> dict:
    """``value``'s accounting with its physical labels read from the row
    ``owner`` posted for it (none when ``owner`` is unknown or has no row)."""

    from .frame_identity_book import (
        _frame_accounting_from_posted, _frame_physical_fields,
    )

    accounting = {**dict(value.accounting or {}), **dict(labels or {})}
    if owner is None:
        return accounting
    return _frame_accounting_from_posted(
        accounting,
        _frame_physical_fields(owner, int(value.id), book=identity_book(module)),
    )


def _post_double_backed_boolean(owner: Any, value: Any, module: Any,
                                sources: tuple = ()) -> None:
    """The repository stores a Boolean tensor in double-backed spans.

    Program: tensor comparison masks and Boolean reductions in
    ``lower_tensor_calls_to_repository_ssa``.  Wrong record: the float64
    physical dtype of a Boolean tensor was stamped on accounting alone, so its
    ``ssa_value`` row kept no physical dtype.  Rule: the stamp is a REVISE of
    the value's row (DERIVED from its previous cell and ``sources``), and the
    accounting is a view of it."""

    if owner is None:
        return
    from .concordance_declarations import TENSOR_SSA_LOWERING
    from .ssa_record_return_state import revise_ssa_value

    revise_ssa_value(
        owner, int(value.id), dtype=value.dtype,
        shape=tuple(value.shape or ()), stage=TENSOR_SSA_LOWERING,
        sources=tuple(cell for cell in sources if cell is not None),
        book=identity_book(module), physical={"physical_dtype": "float64"},
    )
    value.accounting = _accounting_view(owner, value, module)


@dataclass(frozen=True, order=True)
class TensorSSALoweringShortfall:
    function: str
    block: str
    operation: str
    reason: str


def _own_dynamic_tensor_output(
    module: IRModule, function, book, tensor_table, fresh, register_tensor,
    result: SSAValue, count: SSAValue, element_dtype: str,
    prefix: list[Instr],
) -> SSAValue:
    """Bind one logical dynamic result to native heap storage."""
    from .ssa_heap_allocation import heap_allocate
    from .concordance_declarations import (
        TENSOR_OWNED_HEAP_BINDING,
        TENSOR_HEAP_OWNER_CALL_BINDING,
        TENSOR_SSA_LOWERING as _HEAP_TENSOR_STAGE,
    )
    from .identity_concordance import Derived, Mode
    from .ssa_record_return_state import (
        function_scope_of,
        ssa_value_identity_cell,
    )
    incumbent = book.latest_ref(TENSOR_OWNED_HEAP_BINDING, (
        function_scope_of(function), function.name, int(result.id),
    ))
    if incumbent is not None:
        carrier_id, owner_id, count_id, dtype = book.page(
            TENSOR_OWNED_HEAP_BINDING,
        ).cell(incumbent.row, incumbent.column)
        allocations = tuple(
            instruction for block in function.blocks.values()
            for instruction in (*block.instrs, *prefix)
            if instruction.res is not None and int(instruction.res.id) == carrier_id
        )
        descriptor = tensor_table.by_id(int(result.id))
        if (count_id != int(count.id) or dtype != str(element_dtype)
                or len(allocations) != 1 or allocations[0].op != "Alloca"
                or allocations[0].attributes.get("allocation_kind") != "owned_heap"
                or tuple(int(value.id) for value in allocations[0].args) != (count_id, owner_id)
                or descriptor is None or descriptor.data_value_id != carrier_id
                or descriptor.heap_owner_value_id != owner_id):
            raise ValueError("owned dynamic tensor disagrees with its incumbent allocation")
        # Checked scalar arithmetic stays raw SSA and is visited on
        # whole-program retries. Its one heap identity is already
        # booked; revisiting the operation must not allocate it again.
        return allocations[0].res
    owner = fresh(
        dtype="ptr", shape=(1,), physical={
            "physical_dtype": "ptr", "storage": "span",
            "rank": 1,
        },
    )
    owner.accounting = {
        **dict(owner.accounting or {}),
        "heap_owner": True,
    }
    function.args.append(owner)
    from .precompile_to_ssa import _mint_ssa_id
    from .concordance_declarations import (
        TENSOR_LOWERING_VALUE,
    )
    region_calls = tuple(
        (caller, region_call)
        for caller in module.functions.values()
        for region_block in caller.blocks.values()
        for region_call in region_block.instrs
        if region_call.op == "Call"
        and region_call.attributes.get("callee")
        == function.name
        and region_call.res is not None
    )
    if not region_calls:
        raise ValueError(
            "owned dynamic tensor has no exact caller"
        )
    owner_call_rows = []
    for caller, region_call in region_calls:
        call_cell = ssa_value_identity_cell(
            caller, int(region_call.res.id), book=book,
        )
        tensor_cell = ssa_value_identity_cell(
            function, int(result.id), book=book,
        )
        formal_cell = ssa_value_identity_cell(
            function, int(owner.id), book=book,
        )
        if any(cell is None for cell in (
            call_cell, tensor_cell, formal_cell,
        )):
            raise ValueError(
                "owned dynamic tensor call lacks exact SSA cells"
            )
        caller_scope = function_scope_of(caller)
        actual = SSAValue(
            _mint_ssa_id(
                book, caller_scope,
                TENSOR_LOWERING_VALUE,
                (call_cell, tensor_cell, formal_cell),
                dtype="ptr", shape=(1,),
                stage=_HEAP_TENSOR_STAGE,
                physical={
                    "physical_dtype": "ptr",
                    "storage": "span", "rank": 1,
                },
            ),
            dtype="ptr", shape=(1,),
            accounting={
                "physical_dtype": "ptr",
                "heap_owner": True,
                "compiler_frame_storage": caller.name,
            },
        )
        caller.args.append(actual)
        region_call.args.append(actual)
        # weighted_native_probe's private reduction output adds an
        # owner after the region cut. Keep its exact feed slot with
        # the Call actual; computed metadata later inserts ahead of
        # that slot. Otherwise an int32 extent is read as the ptr
        # owner and late feed settlement retypes the owner.
        for key, appended in (
            ("feed_ids", int(actual.id)),
            ("feed_shapes", tuple(actual.shape or ())),
            ("feed_dtypes", str(actual.dtype)),
        ):
            prior = tuple(region_call.attributes.get(key, ()))
            if len(prior) != len(region_call.args) - 1:
                raise ValueError(
                    "owned tensor has no complete exact region feed"
                )
            region_call.attributes[key] = (*prior, appended)
        input_ids = region_call.attributes.get(
            "callee_input_ids"
        )
        if input_ids is None:
            # The planned region's original Call has
            # positional formals but no explicit id
            # list. The appended owner makes the
            # existing positions an exact published
            # binding; every formal is named here.
            input_ids = tuple(
                int(formal.id)
                for formal in function.args[:-1]
            )
        region_call.attributes[
            "callee_input_ids"
        ] = (*tuple(map(int, input_ids)),
             int(owner.id))
        actual_cell = ssa_value_identity_cell(
            caller, int(actual.id), book=book,
        )
        if actual_cell is None:
            raise ValueError(
                "owned dynamic tensor caller has no owner cell"
            )
        owner_call_rows.append((
            (caller_scope, caller.name,
             int(region_call.res.id), function.name,
             int(result.id)),
            (int(result.id), int(actual.id),
             int(owner.id)),
            (call_cell, tensor_cell,
             actual_cell, formal_cell),
        ))
    carrier = fresh(dtype="ptr", physical={
        "physical_dtype": "ptr", "storage": "scalar",
        "rank": 0,
    })
    allocation = heap_allocate(
        count, owner, carrier,
        element_dtype=element_dtype,
    )
    descriptor = register_tensor(
        result, storage="temporary",
    )
    tensor_table.tensors[int(result.id)] = (
        dataclasses.replace(
            descriptor,
            data_value_id=int(carrier.id),
            heap_owner_value_id=int(owner.id),
            dtype=str(element_dtype),
        )
    )
    page_sources = tuple(
        cell for cell in (
            ssa_value_identity_cell(
                function, int(value.id), book=book,
            )
            for value in (
                result, carrier, owner, count,
            )
        ) if cell is not None
    )
    if len(page_sources) != 4:
        raise ValueError(
            "owned dynamic tensor lacks exact SSA cells"
        )
    book.post(
        TENSOR_OWNED_HEAP_BINDING,
        (function_scope_of(function), function.name,
         int(result.id)),
        (int(carrier.id), int(owner.id),
         int(count.id), str(element_dtype)),
        stage=_HEAP_TENSOR_STAGE,
        provenance=Derived(page_sources),
        mode=Mode.CONCORD,
    )
    producer_cell = book.latest_ref(
        TENSOR_OWNED_HEAP_BINDING,
        (function_scope_of(function), function.name,
         int(result.id)),
    )
    for row, fact, causes in owner_call_rows:
        book.post(
            TENSOR_HEAP_OWNER_CALL_BINDING,
            row, fact,
            stage=_HEAP_TENSOR_STAGE,
            provenance=Derived((
                *causes, producer_cell,
            )),
            mode=Mode.CONCORD,
        )
    prefix.append(allocation)
    return carrier

def wire_repository_ssa_region_products(module: IRModule) -> bool:
    """Connect region outputs to later region feeds inside each function.

    Region planning records semantic source identities in ``output_ids`` and
    ``feed_ids``.  Source-call linking can temporarily materialize a
    caller-owned frame argument when a producer region has not yet been
    inserted.  Once the complete repository module exists, the declared
    identities provide an exact, deterministic replacement: use the
    producer's projected SSA value and schedule its call/projection cluster
    before the consumer.
    """

    changed = False
    for function in module.functions.values():
        for block in function.blocks.values():
            instructions = block.instrs
            producer_by_source: dict[int, tuple[Instr, SSAValue, list[Instr]]] = {}
            for index, instruction in enumerate(tuple(instructions)):
                if (
                    instruction.op not in {"Call", "call"}
                    or instruction.res is None
                    or not tuple(instruction.attributes.get("output_ids", ()))
                ):
                    continue
                cluster = [instruction]
                cursor = index + 1
                while cursor < len(instructions) and instructions[cursor].op in {
                    "Const", "GetElementPtr", "getelementptr", "Load", "load",
                }:
                    cluster.append(instructions[cursor])
                    cursor += 1
                for projected in cluster:
                    if (
                        projected.op in {"Load", "load"}
                        and projected.res is not None
                        and projected.attributes.get("source_output_id") is not None
                    ):
                        producer_by_source[int(
                            projected.attributes["source_output_id"]
                        )] = (instruction, projected.res, cluster)

            for consumer in tuple(instructions):
                if consumer.op not in {"Call", "call"}:
                    continue
                feeds = tuple(map(
                    int, consumer.attributes.get("feed_ids", ())
                ))
                if not feeds:
                    continue
                for position, feed_id in enumerate(feeds):
                    producer = producer_by_source.get(feed_id)
                    if producer is None or producer[0] is consumer:
                        continue
                    producer_call, produced_value, cluster = producer
                    if position >= len(consumer.args):
                        continue
                    previous = consumer.args[position]
                    # A repeated seed-producing region can occur inside a
                    # loop body even though the live value for this semantic
                    # feed is the header Phi.  Control lowering marks that
                    # exact case.  Do not let whole-module product wiring
                    # replace the Phi with the repeated seed projection and
                    # reset the carried value on every iteration.
                    if (
                        (previous.accounting or {}).get(
                            "ssa_loop_carried_feed"
                        ) == int(feed_id)
                    ):
                        continue
                    if int(consumer.args[position].id) != int(produced_value.id):
                        declared_shapes = tuple(
                            consumer.attributes.get("feed_shapes", ())
                        )
                        declared_dtypes = tuple(
                            consumer.attributes.get("feed_dtypes", ())
                        )
                        view_shape = (
                            tuple(declared_shapes[position])
                            if position < len(declared_shapes)
                            else tuple(previous.shape or produced_value.shape)
                        )
                        view_dtype = (
                            str(declared_dtypes[position])
                            if position < len(declared_dtypes)
                            and str(declared_dtypes[position])
                            else previous.dtype or produced_value.dtype
                        )
                        consumer.args[position] = SSAValue(
                            int(produced_value.id),
                            dtype=view_dtype,
                            shape=view_shape,
                            device=previous.device or produced_value.device,
                            accounting={
                                **dict(produced_value.accounting or {}),
                                **dict(previous.accounting or {}),
                                "ssa_storage_alias": int(produced_value.id),
                                "ssa_region_feed": (int(feed_id), int(position)),
                            },
                        )
                        changed = True
                    producer_index = instructions.index(producer_call)
                    consumer_index = instructions.index(consumer)
                    if producer_index > consumer_index:
                        for member in cluster:
                            instructions.remove(member)
                        consumer_index = instructions.index(consumer)
                        instructions[consumer_index:consumer_index] = cluster
                        changed = True
    return changed


def _bind_source_return_owned_producers(module: IRModule, book) -> bool:
    """Give a booked source-returned dynamic kernel output its existing heap owner."""

    from .concordance_declarations import (
        CALL_RECORD, TENSOR_LOWERING_VALUE, TENSOR_OWNED_HEAP_BINDING,
        TENSOR_SSA_LOWERING,
    )
    from .precompile_to_ssa import _mint_ssa_id
    from .ssa_call_input_adapters import _infer_pointer_storage
    from .ssa_record_return_state import (
        function_scope_of, posted_ssa_value_fact, revise_ssa_value,
        ssa_value_identity_cell,
    )

    if getattr(module, "call_table", None) is None:
        return False
    changed = False
    for outer in tuple(module.functions.values()):
        source_calls = tuple(
            instruction for block in outer.blocks.values()
            for instruction in block.instrs
            if instruction.op == "Call"
            and instruction.attributes.get("source_linked")
            and instruction.res is not None
        )
        for source_call in source_calls:
            source_name = str(source_call.attributes.get("callee") or "")
            source = module.functions.get(source_name)
            record_ref = book.latest_ref(
                CALL_RECORD, (module.call_table.owner, outer.name),
            )
            if source is None or record_ref is None:
                continue
            records = book.page(CALL_RECORD).cell(
                record_ref.row, record_ref.column,
            )
            matched = tuple(
                record for record in records
                if record.callee_symbol == source_name
                and record.callsite_id == source_call.attributes.get(
                    "plan_callsite_id"
                )
            )
            if len(matched) != 1 or len(matched[0].result_bindings) != 1:
                continue
            source_id, caller_id = map(int, matched[0].result_bindings[0])
            if any(
                ret.op == "Ret"
                and any(int(value.id) == caller_id for value in ret.args)
                for block in outer.blocks.values() for ret in block.instrs
            ):
                continue
            source_returns = tuple(
                ret for block in source.blocks.values()
                for ret in block.instrs if ret.op == "Ret"
                and any(int(value.id) == source_id for value in ret.args)
            )
            if len(source_returns) != 1:
                continue
            region_calls = tuple(
                call for block in source.blocks.values()
                for call in block.instrs
                if call.op == "Call" and call.res is not None
                and source_id in tuple(map(
                    int, call.attributes.get("output_ids", ()),
                ))
                and str(call.attributes.get("callee") or "")
                    in module.tensor_tables
            )
            if len(region_calls) != 1:
                continue
            region_call = region_calls[0]
            producer_name = str(region_call.attributes["callee"])
            producer = module.functions.get(producer_name)
            table = module.tensor_tables.get(producer_name)
            descriptor = None if table is None else table.by_id(source_id)
            if producer is None or descriptor is None:
                continue
            owned_row = (function_scope_of(producer), producer_name, source_id)
            if book.latest_ref(TENSOR_OWNED_HEAP_BINDING, owned_row) is not None:
                continue
            if (descriptor.metadata_state != "dynamic"
                    or descriptor.element_count_value_id is None
                    or int(descriptor.data_value_id) != source_id):
                continue
            producer_calls = tuple(
                (block, call) for block in producer.blocks.values()
                for call in block.instrs
                if call.op == "Call" and call.res is not None
                and int(call.res.id) == source_id
            )
            if len(producer_calls) != 1:
                continue
            producer_block, native_call = producer_calls[0]
            native_name = str(native_call.attributes.get("callee") or "")
            selected = module.functions.get(native_name)
            position = None if selected is None else selected.metadata.get(
                "ssa_output_argument"
            )
            if (selected is None or not isinstance(position, int)
                    or position < 0 or position >= len(selected.args)
                    or position >= len(native_call.args)
                    or int(native_call.args[position].id) != source_id):
                continue
            _infer_pointer_storage({native_name: selected})
            formal = selected.args[position]
            formal_fact = posted_ssa_value_fact(
                selected, int(formal.id), book=book,
            )
            result_cell = ssa_value_identity_cell(producer, source_id, book=book)
            formal_cell = ssa_value_identity_cell(
                selected, int(formal.id), book=book,
            )
            region_cell = ssa_value_identity_cell(
                source, int(region_call.res.id), book=book,
            )
            count = next((value for value in producer.args
                          if int(value.id) == int(
                              descriptor.element_count_value_id)), None)
            if (formal_fact is None or formal_fact.physical_dtype is None
                    or result_cell is None or formal_cell is None
                    or region_cell is None or count is None):
                continue
            result_fact = posted_ssa_value_fact(
                producer, source_id, book=book,
            )
            if (result_fact is None or result_fact.storage != "span"
                    or type(result_fact.rank) is not int
                    or result_fact.rank <= 0
                    or result_fact.physical_dtype not in (
                        None, formal_fact.physical_dtype,
                    )):
                raise ValueError(
                    "source-owned kernel result disagrees with selected output storage"
                )
            # make(count).to_dtype(float64) acquires its private output
            # after the original cast visit. The selected cast output formal
            # proves float64, but result4's SSA_VALUE still has no physical
            # dtype. Post that exact kernel edge before the owned carrier and
            # later source/caller Call transport read the logical result.
            result_cell = revise_ssa_value(
                producer, source_id, dtype=result_fact.dtype,
                shape=result_fact.shape,
                stage=TENSOR_SSA_LOWERING, sources=(formal_cell,), book=book,
                physical={"physical_dtype": formal_fact.physical_dtype},
            )
            native_call.res.accounting = _accounting_view(
                producer, native_call.res, module,
            )
            causes = (result_cell, formal_cell, region_cell, record_ref)
            scope = function_scope_of(producer)

            def fresh(*, dtype="unknown", shape=(), physical=None):
                value_id = _mint_ssa_id(
                    book, scope, TENSOR_LOWERING_VALUE, causes,
                    dtype=dtype, shape=tuple(shape or ()),
                    stage=TENSOR_SSA_LOWERING, physical=physical,
                )
                return SSAValue(value_id, dtype=dtype, shape=tuple(shape or ()))

            prefix: list[Instr] = []
            carrier = _own_dynamic_tensor_output(
                module, producer, book, table, fresh,
                lambda value, storage: table.by_id(int(value.id)),
                native_call.res, count, str(formal_fact.physical_dtype),
                prefix,
            )
            native_call.args[position] = carrier
            insert_at = producer_block.instrs.index(native_call)
            producer_block.instrs[insert_at:insert_at] = prefix
            changed = True
    return changed


def _publish_source_owned_return_bindings(module: IRModule, book) -> bool:
    """Carry one allocated arena through an exact ordinary source result."""

    from .concordance_declarations import (
        CALL_RECORD, TENSOR_HEAP_OWNER_CALL_BINDING,
        TENSOR_LOWERING_VALUE, TENSOR_METADATA_RESULT_BINDING,
        TENSOR_OWNED_HEAP_BINDING, TENSOR_SOURCE_OWNED_RETURN,
        TENSOR_SSA_LOWERING,
    )
    from .identity_concordance import Derived, Mode
    from .precompile_to_ssa import _mint_ssa_id
    from .ssa_record_return_state import (
        function_scope_of, metadata_result_caller_value,
        ssa_value_identity_cell,
    )

    if getattr(module, "call_table", None) is None:
        return False
    changed = False
    for outer in module.functions.values():
        outer_scope = function_scope_of(outer)
        for block in outer.blocks.values():
            for source_call in tuple(block.instrs):
                if (source_call.op != "Call" or source_call.res is None
                        or not source_call.attributes.get("source_linked")
                        or source_call.attributes.get("result_convention")
                            != "ssa.aggregate"):
                    continue
                source_name = str(source_call.attributes.get("callee") or "")
                source = module.functions.get(source_name)
                record_ref = book.latest_ref(
                    CALL_RECORD, (module.call_table.owner, outer.name),
                )
                if source is None or record_ref is None:
                    continue
                records = book.page(CALL_RECORD).cell(
                    record_ref.row, record_ref.column,
                )
                bound = tuple(
                    record for record in records
                    if record.callee_symbol == source_name
                    and record.callsite_id == source_call.attributes.get(
                        "plan_callsite_id"
                    )
                )
                if len(bound) != 1 or len(bound[0].result_bindings) != 1:
                    continue
                source_id, caller_id = map(int, bound[0].result_bindings[0])
                row = (outer_scope, outer.name, int(source_call.res.id),
                       source_name, source_id)
                if book.latest_ref(TENSOR_SOURCE_OWNED_RETURN, row) is not None:
                    continue
                source_scope = function_scope_of(source)
                outer_result_ref = book.latest_ref(
                    TENSOR_METADATA_RESULT_BINDING,
                    (outer_scope, int(source_call.res.id),
                     source_name, source_id),
                )
                if outer_result_ref is None:
                    continue
                projected = metadata_result_caller_value(
                    module, outer, source_call, outer_result_ref, book=book,
                )
                if projected is None or projected[0] != caller_id:
                    continue
                region_rows = tuple(
                    candidate for candidate in book.page(
                        TENSOR_HEAP_OWNER_CALL_BINDING
                    ).rows()
                    if candidate[0] == source_scope
                    and candidate[1] == source_name
                    and int(candidate[4]) == source_id
                )
                if len(region_rows) != 1:
                    continue
                owner_row = region_rows[0]
                owner_ref = book.latest_ref(
                    TENSOR_HEAP_OWNER_CALL_BINDING, owner_row,
                )
                owner_fact = book.page(TENSOR_HEAP_OWNER_CALL_BINDING).cell(
                    owner_ref.row, owner_ref.column,
                )
                source_owner = int(owner_fact[1])
                producer_name = str(owner_row[3])
                producer = module.functions.get(producer_name)
                producer_table = module.tensor_tables.get(producer_name)
                producer_descriptor = (None if producer_table is None else
                                       producer_table.by_id(source_id))
                if producer is None or producer_descriptor is None:
                    continue
                owned_ref = book.latest_ref(TENSOR_OWNED_HEAP_BINDING, (
                    function_scope_of(producer), producer_name, source_id,
                ))
                if owned_ref is None:
                    continue
                owned = book.page(TENSOR_OWNED_HEAP_BINDING).cell(
                    owned_ref.row, owned_ref.column,
                )
                if (int(owned[0]) != int(producer_descriptor.data_value_id)
                        or int(owned[1]) != int(owner_fact[2])):
                    raise ValueError("source return lost its owned producer")
                region_calls = tuple(
                    call for source_block in source.blocks.values()
                    for call in source_block.instrs
                    if call.op == "Call" and call.res is not None
                    and int(call.res.id) == int(owner_row[2])
                    and call.attributes.get("callee") == producer_name
                )
                if len(region_calls) != 1:
                    continue
                region_call = region_calls[0]
                source_result_ref = book.latest_ref(
                    TENSOR_METADATA_RESULT_BINDING,
                    (source_scope, int(region_call.res.id),
                     producer_name, source_id),
                )
                source_table = module.tensor_tables.get(source_name)
                source_descriptor = (None if source_table is None else
                                     source_table.by_id(source_id))
                if (source_result_ref is None or source_descriptor is None
                        or source_descriptor.metadata_state != "dynamic"
                        or source_descriptor.element_count_value_id is None
                        or tuple(map(int, region_call.attributes.get(
                            "callee_output_ids", ())))[:1]
                            != (int(owned[0]),)):
                    continue
                formal_ids = tuple(map(int, source_call.attributes.get(
                    "callee_input_ids", ())))
                if len(formal_ids) != len(source_call.args):
                    continue
                owner_actuals = tuple(
                    actual for formal_id, actual in zip(
                        formal_ids, source_call.args, strict=True,
                    ) if formal_id == source_owner
                )
                if len(owner_actuals) != 1:
                    continue
                caller_owner = int(owner_actuals[0].id)
                if not any(
                    int(formal_id) == source_owner
                    and kind in {"caller_storage", "caller_value", "caller_alias"}
                    and type(actual_id) is int
                    and int(actual_id) == caller_owner
                    for formal_id, kind, actual_id in bound[0].frame_bindings
                ):
                    # The late frame binder owns this relation. A live Call
                    # operand without its CALL_RECORD is not an owner proof.
                    continue
                if any(
                    ret.op == "Ret"
                    and any(int(value.id) == caller_id for value in ret.args)
                    for caller_block in outer.blocks.values()
                    for ret in caller_block.instrs
                ):
                    raise ValueError("owned source tensor escapes its caller")

                def projection(function, call, data_id):
                    matches = tuple(
                        (projection_block, index, instruction)
                        for projection_block in function.blocks.values()
                        for index, instruction in enumerate(
                            projection_block.instrs
                        )
                        if instruction.op == "Load"
                        and instruction.res is not None
                        and int(instruction.res.id) == data_id
                        and instruction.attributes.get("aggregate_index") == 0
                        and instruction.attributes.get("source_output_id")
                            == data_id
                        and instruction.args
                        and any(
                            gep.op == "GetElementPtr"
                            and gep.res is instruction.args[0]
                            and gep.args and gep.args[0] is call.res
                            for gep in projection_block.instrs
                        )
                    )
                    return matches[0] if len(matches) == 1 else None

                source_projection = projection(source, region_call, source_id)
                caller_projection = projection(outer, source_call, caller_id)
                if source_projection is None or caller_projection is None:
                    continue
                source_cells = tuple(
                    ssa_value_identity_cell(source, int(value.id), book=book)
                    for value in (source_projection[2].res,
                                  source_projection[2].args[0],
                                  source.args[formal_ids.index(source_owner)])
                )
                caller_cells = tuple(
                    ssa_value_identity_cell(outer, int(value.id), book=book)
                    for value in (caller_projection[2].res,
                                  caller_projection[2].args[0],
                                  owner_actuals[0])
                )
                causes = tuple(dict.fromkeys((
                    owned_ref, owner_ref, source_result_ref,
                    outer_result_ref, record_ref,
                    *projected[1], *source_cells, *caller_cells,
                )))
                if any(cell is None for cell in causes):
                    raise ValueError("owned source return lacks exact SSA causes")

                def loaded_carrier(function, source_projection, sources):
                    projection_block, index, instruction = source_projection
                    ptr_id = _mint_ssa_id(
                        book, function_scope_of(function),
                        TENSOR_LOWERING_VALUE, sources,
                        dtype="ptr", shape=(), stage=TENSOR_SSA_LOWERING,
                        physical={"physical_dtype": "ptr",
                                  "storage": "scalar", "rank": 0},
                    )
                    carrier = SSAValue(ptr_id, dtype="ptr", shape=())
                    projection_block.instrs.insert(index + 1, Instr(
                        "Load", [instruction.args[0]], carrier,
                        attributes={"aggregate_index": 0,
                                    "source_output_id": int(instruction.res.id),
                                    "source_owned_carrier": True},
                    ))
                    return carrier

                source_carrier = loaded_carrier(
                    source, source_projection, causes,
                )
                caller_carrier = loaded_carrier(
                    outer, caller_projection, (
                        *causes,
                        ssa_value_identity_cell(
                            source, int(source_carrier.id), book=book,
                        ),
                    ),
                )
                source_table.tensors[source_id] = dataclasses.replace(
                    source_descriptor,
                    data_value_id=int(source_carrier.id),
                    heap_owner_value_id=source_owner,
                )
                caller_metadata = book.page(
                    TENSOR_METADATA_RESULT_BINDING
                ).cell(outer_result_ref.row, outer_result_ref.column)[2]
                caller_table = module.tensor_tables.setdefault(
                    outer.name, SSATensorTable(),
                )
                incumbent = caller_table.by_id(caller_id)
                if incumbent is not None:
                    raise ValueError("owned source caller has conflicting descriptor")
                caller_table.register(dataclasses.replace(
                    source_descriptor, tensor_id=caller_id,
                    data_value_id=int(caller_carrier.id),
                    heap_owner_value_id=caller_owner,
                    shape_value_id=int(caller_metadata[0]),
                    rank_value_id=int(caller_metadata[1]),
                    element_count_value_id=int(caller_metadata[2]),
                    storage="temporary",
                ))
                carrier_causes = tuple(
                    ssa_value_identity_cell(function, int(value.id), book=book)
                    for function, value in (
                        (source, source_carrier), (outer, caller_carrier),
                    )
                )
                transport_causes = tuple(
                    ssa_value_identity_cell(function, value_id, book=book)
                    for function, value_id in (
                        (source, int(source_descriptor.element_count_value_id)),
                        (outer, int(caller_metadata[2])),
                        (outer, int(source_call.res.id)),
                    )
                )
                if any(cell is None for cell in transport_causes):
                    raise ValueError("owned source return lacks count or Call proof")
                book.post(TENSOR_SOURCE_OWNED_RETURN, row, (
                    int(source_carrier.id), source_owner,
                    int(caller_carrier.id), caller_owner,
                    caller_id, int(source_descriptor.element_count_value_id),
                    str(owned[3]),
                ), stage=TENSOR_SSA_LOWERING,
                    provenance=Derived((*causes, *carrier_causes,
                                        *transport_causes)),
                    mode=Mode.CONCORD)
                changed = True
    return changed


# In _ascending_order, region4's aggregate fold changed tensor22's returned
# shape/rank/count from 4054/4057/4060 to 4192/4195/4198. Its source Ret
# used the new IDs, while the source descriptor and ordinary caller still
# described the old ones. Carry only that exact booked fold through the
# source result, owned count, and dependent metadata Call proof.
def _refresh_source_return_passthrough_metadata(module: IRModule, book) -> bool:
    """Follow an exact aggregate metadata fold through a source return."""

    from .concordance_declarations import (
        AGGREGATE_PASSTHROUGH_REBINDING, CALL_RECORD, SSA_VALUE,
        TENSOR_METADATA_CALL_BINDING, TENSOR_METADATA_RESULT_BINDING,
        TENSOR_RUNTIME_BROADCAST_VALIDITY,
        TENSOR_SOURCE_OWNED_RETURN,
        TENSOR_SSA_LOWERING,
    )
    from .identity_concordance import Derived, Mode, Ref
    from .ssa_record_return_state import function_scope_of, ssa_value_identity_cell

    changed = False
    result_page = book.page(TENSOR_METADATA_RESULT_BINDING)
    return_page = book.page(TENSOR_SOURCE_OWNED_RETURN)
    fields = ("shape_value_id", "rank_value_id", "element_count_value_id")
    accounting_keys = (
        "tensor_shape_value_id", "tensor_rank_value_id",
        "tensor_element_count_value_id",
    )
    for source in module.functions.values():
        source_ids = tuple(map(int, (source.metadata or {}).get(
            "source_output_value_ids", (),
        )))
        if len(source_ids) != 1 or (source.metadata or {}).get(
            "source_region_integral"
        ):
            continue
        source_id = source_ids[0]
        table = module.tensor_tables.get(source.name)
        descriptor = None if table is None else table.by_id(source_id)
        returns = tuple(instruction for block in source.blocks.values()
                        for instruction in block.instrs if instruction.op == "Ret")
        if descriptor is None or descriptor.metadata_state != "dynamic" or len(returns) != 1:
            continue
        ret_ids = tuple(int(value.id) for value in returns[0].args)
        if len(ret_ids) != 4 or ret_ids[0] != source_id:
            continue
        current_ids = ret_ids[1:]
        old_ids = tuple(getattr(descriptor, field) for field in fields)
        if old_ids == current_ids:
            continue
        scope = function_scope_of(source)
        upstream = tuple(
            row for row in result_page.rows()
            if row[0] == scope and int(row[3]) == source_id
            and tuple(result_page.latest(row)[2]) == current_ids
        )
        if len(upstream) != 1:
            raise ValueError("source Ret metadata has no unique computed producer")
        upstream_ref = book.latest_ref(TENSOR_METADATA_RESULT_BINDING, upstream[0])
        if upstream_ref.column < 1:
            raise ValueError("source metadata changed without aggregate revision")
        predecessor = Ref(TENSOR_METADATA_RESULT_BINDING,
                          upstream_ref.row, upstream_ref.column - 1)
        upstream_causes = {
            parent for parent, _stage in book.edges_into(upstream_ref)
        }
        rebinding_refs = tuple(
            parent for parent in upstream_causes
            if parent.page == AGGREGATE_PASSTHROUGH_REBINDING
        )
        semantic = returns[0].args[0]
        if (tuple(result_page.cell(predecessor.row, predecessor.column)[2])
                != old_ids or predecessor not in upstream_causes
                or len(rebinding_refs) != 3
                or {int(ref.row[1]) for ref in rebinding_refs} != set(old_ids)
                or tuple((semantic.accounting or {}).get(key)
                         for key in accounting_keys) != current_ids
                or any(ssa_value_identity_cell(source, value_id, book=book)
                       is None for value_id in current_ids)):
            raise ValueError("source metadata fold lacks exact Ret ancestry")
        table.tensors[source_id] = dataclasses.replace(
            descriptor,
            **dict(zip(fields, current_ids, strict=True)),
        )
        changed = True
        for caller in module.functions.values():
            record_ref = (book.latest_ref(
                CALL_RECORD, (module.call_table.owner, caller.name),
            ) if getattr(module, "call_table", None) is not None else None)
            if record_ref is None:
                continue
            records = book.page(CALL_RECORD).cell(record_ref.row,
                                                   record_ref.column)
            for block in caller.blocks.values():
                for call in block.instrs:
                    if (call.op != "Call" or call.res is None
                            or not call.attributes.get("source_linked")
                            or call.attributes.get("callee") != source.name):
                        continue
                    bound = tuple(record for record in records
                                  if record.callee_symbol == source.name
                                  and record.callsite_id == call.attributes.get(
                                      "plan_callsite_id")
                                  and len(record.result_bindings) == 1
                                  and int(record.result_bindings[0][0]) == source_id)
                    if len(bound) != 1:
                        raise ValueError("source metadata fold lost its Call record")
                    caller_id = int(bound[0].result_bindings[0][1])
                    row = (function_scope_of(caller), int(call.res.id),
                           source.name, source_id)
                    prior = book.latest_ref(TENSOR_METADATA_RESULT_BINDING, row)
                    if prior is None:
                        continue
                    previous = result_page.cell(prior.row, prior.column)
                    caller_ids = tuple(map(int, previous[2]))
                    if (tuple(previous[1]) != old_ids
                            or tuple(map(int, call.attributes.get(
                                "callee_output_ids", ()))) != (source_id, *old_ids)
                            or tuple(map(int, call.attributes.get(
                                "output_ids", ()))) != (caller_id, *caller_ids)):
                        raise ValueError("source Call has conflicting old metadata projection")
                    proof = tuple(dict.fromkeys((
                        prior, upstream_ref, record_ref,
                        *rebinding_refs,
                        ssa_value_identity_cell(caller, int(call.res.id), book=book),
                        ssa_value_identity_cell(source, source_id, book=book),
                        ssa_value_identity_cell(caller, caller_id, book=book),
                        *(ssa_value_identity_cell(source, value_id, book=book)
                          for value_id in current_ids),
                        *(ssa_value_identity_cell(caller, value_id, book=book)
                          for value_id in caller_ids),
                    )))
                    if any(cell is None for cell in proof):
                        raise ValueError("source metadata fold lacks exact SSA cells")
                    book.post(
                        TENSOR_METADATA_RESULT_BINDING, row,
                        (source_id, current_ids, caller_ids),
                        stage=TENSOR_SSA_LOWERING,
                        provenance=Derived(proof), mode=Mode.REVISE,
                    )
                    call.attributes["callee_output_ids"] = (source_id, *current_ids)
                    latest_result = book.latest_ref(TENSOR_METADATA_RESULT_BINDING,
                                                    row)
                    binding_page = book.page(TENSOR_METADATA_CALL_BINDING)
                    for binding_row in binding_page.rows():
                        if binding_row[0] != function_scope_of(caller):
                            continue
                        binding_ref = book.latest_ref(
                            TENSOR_METADATA_CALL_BINDING, binding_row,
                        )
                        binding_fact = binding_page.cell(
                            binding_ref.row, binding_ref.column,
                        )
                        if prior not in {
                            parent for parent, _stage in book.edges_into(
                                binding_ref)
                        }:
                            continue
                        if tuple(binding_fact[0]) != caller_ids:
                            raise ValueError(
                                "source metadata consumer has conflicting actuals"
                            )
                        book.post(
                            TENSOR_METADATA_CALL_BINDING, binding_row,
                            binding_fact, stage=TENSOR_SSA_LOWERING,
                            provenance=Derived((binding_ref, latest_result)),
                            mode=Mode.CONCORD,
                        )
                        # _ascending_order's root broadcast lives in a
                        # planned consumer, not in the source Call's own
                        # function. The existing metadata Call receipt and
                        # live actual/formal pairs identify that consumer's
                        # exact tensor input and three geometry formals.
                        # Its older broadcast receipt cited the source
                        # result before this aggregate fold. Re-cite the
                        # changed source result, current Call record and
                        # three rebindings without changing the law/fact.
                        target = module.functions.get(str(binding_row[4]))
                        source_block = caller.blocks.get(str(binding_row[2]))
                        linked = () if source_block is None else tuple(
                            instruction for instruction in source_block.instrs
                            if instruction.op == "Call"
                            and instruction.res is not None
                            and int(instruction.res.id) == int(binding_row[3])
                            and instruction.attributes.get("callee") == binding_row[4]
                        )
                        if (target is None or len(linked) != 1
                                or len(linked[0].args) != len(target.args)):
                            continue
                        if (tuple(map(int, linked[0].attributes.get(
                                "callee_input_ids", ()))) != tuple(
                                int(formal.id) for formal in target.args)
                                or tuple(map(int, linked[0].attributes.get(
                                    "feed_ids", ()))) != tuple(
                                    int(actual.id) for actual in linked[0].args)):
                            continue
                        pairs = tuple((int(actual.id), int(formal.id))
                                      for actual, formal in zip(
                                          linked[0].args, target.args, strict=True,
                                      ))
                        data_formals = tuple(formal_id for actual_id, formal_id
                                             in pairs if actual_id == caller_id)
                        metadata_pairs = tuple(zip(
                            *binding_fact, strict=True,
                        ))
                        if (len(data_formals) != 1
                                or any(pair not in pairs
                                       for pair in metadata_pairs)
                                or tuple(binding_fact[0]) != caller_ids):
                            continue
                        tensor_formal = data_formals[0]
                        formal_ids = tuple(binding_fact[1])
                        validity_page = book.page(TENSOR_RUNTIME_BROADCAST_VALIDITY)
                        for validity_row in tuple(validity_page.rows()):
                            if validity_row[:2] != (
                                function_scope_of(target), target.name,
                            ):
                                continue
                            validity_ref = book.latest_ref(
                                TENSOR_RUNTIME_BROADCAST_VALIDITY, validity_row,
                            )
                            validity_fact = validity_page.cell(
                                validity_ref.row, validity_ref.column,
                            )
                            if (not isinstance(validity_fact, tuple)
                                    or len(validity_fact) != 19
                                    or tuple(validity_fact[1:3]).count(
                                        tensor_formal) != 1):
                                continue
                            geometry = (tuple(validity_fact[5:8])
                                        if int(validity_fact[3]) == tensor_formal
                                        else tuple(validity_fact[8:11])
                                        if int(validity_fact[4]) == tensor_formal
                                        else None)
                            if geometry != formal_ids or prior not in {
                                parent for parent, _stage in book.edges_into(
                                    validity_ref)
                            }:
                                continue
                            book.post(
                                TENSOR_RUNTIME_BROADCAST_VALIDITY, validity_row,
                                validity_fact, stage=TENSOR_SSA_LOWERING,
                                provenance=Derived((latest_result, record_ref,
                                                    *rebinding_refs,
                                                    binding_ref)),
                                mode=Mode.CONCORD,
                            )
                    return_row = (function_scope_of(caller), caller.name,
                                  int(call.res.id), source.name, source_id)
                    return_ref = book.latest_ref(TENSOR_SOURCE_OWNED_RETURN,
                                                 return_row)
                    if return_ref is None:
                        continue
                    return_fact = return_page.cell(return_ref.row,
                                                   return_ref.column)
                    if int(return_fact[5]) != int(old_ids[2]):
                        raise ValueError("source owned count conflicts with aggregate fold")
                    return_sources = tuple(dict.fromkeys((
                        return_ref, latest_result, upstream_ref, record_ref,
                        *rebinding_refs,
                        *(parent for parent, _stage in book.edges_into(return_ref)),
                        ssa_value_identity_cell(source, current_ids[2], book=book),
                    )))
                    book.post(
                        TENSOR_SOURCE_OWNED_RETURN, return_row,
                        (*return_fact[:5], current_ids[2], return_fact[6]),
                        stage=TENSOR_SSA_LOWERING,
                        provenance=Derived(return_sources), mode=Mode.REVISE,
                    )
    return changed


def publish_computed_region_result_metadata(
    module: IRModule, *, finalize_private_owners: bool = True,
) -> bool:
    """Return a region's computed shape/rank/count through its aggregate Call.

    The callee-local metadata IDs cannot be copied into a caller result's
    accounting. They become ordinary returned SSA values, projected from the
    existing aggregate result and correlated on one concordance page.
    """

    from .concordance_declarations import (
        ADDRESS, LOAD, SSA_VALUE, TENSOR_LOWERING_VALUE,
        TENSOR_METADATA_RESULT_BINDING, TENSOR_METADATA_CALL_BINDING,
        TENSOR_RUNTIME_RESHAPE_VALIDITY,
        TENSOR_RUNTIME_VIEW_GEOMETRY,
        TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY,
        TENSOR_SSA_LOWERING,
    )
    from .identity_concordance import Derived, Mode
    from .precompile_to_ssa import _mint_ssa_id
    from .ssa_record_return_state import (
        function_scope_of, metadata_result_caller_value,
        ssa_value_identity_cell,
    )

    book = identity_book(module)
    changed = _bind_source_return_owned_producers(module, book)
    changed |= _refresh_source_return_passthrough_metadata(module, book)
    pending_input_bindings = []
    keys = (
        "tensor_shape_value_id", "tensor_rank_value_id",
        "tensor_element_count_value_id",
    )

    def booked_metadata_replay(
        binding, result_row, downstream, target, metadata_actuals, local_ids,
        target_tensor_id,
    ) -> bool:
        """Read an earlier exact Call binding when its tensor descriptor is pending.

        Runtime reshape's consumer can already have all three metadata formals
        while its tensor formal has no descriptor. The booked RESULT predecessor
        and live Call ports, rather than an empty descriptor triple, prove replay.
        """
        if binding is None:
            return False
        result_ref = book.latest_ref(TENSOR_METADATA_RESULT_BINDING, result_row)
        if result_ref is None or result_ref not in {
            cause for cause, _stage in book.edges_into(binding)
        }:
            return False
        bound_actuals, bound_formals = book.page(
            TENSOR_METADATA_CALL_BINDING
        ).cell(binding.row, binding.column)
        if (len(bound_actuals) != len(bound_formals)
                or len(downstream.args) != len(target.args)):
            return False
        booked_pairs = tuple(zip(
            map(int, bound_actuals), map(int, bound_formals), strict=True,
        ))
        live_pairs = tuple(
            (int(actual.id), int(formal.id))
            for actual, formal in zip(
                downstream.args, target.args, strict=True,
            )
        )
        actual_ids = tuple(int(value.id) for value in metadata_actuals)
        if len(actual_ids) != len(keys):
            return False
        if not local_ids:
            tensor_formals = tuple(
                formal for formal in target.args
                if int(formal.id) == target_tensor_id
            )
            if len(tensor_formals) != 1:
                return False
            formal_accounting = tensor_formals[0].accounting or {}
            local_ids = tuple(formal_accounting.get(key) for key in keys)
            if any(type(value_id) is not int for value_id in local_ids):
                return False
            for kind, value_id in zip(
                ("shape", "rank", "element_count"), local_ids, strict=True,
            ):
                extent_formals = tuple(
                    formal for formal in target.args
                    if int(formal.id) == value_id
                    and (formal.accounting or {}).get("tensor_extent_for")
                        == target_tensor_id
                    and (formal.accounting or {}).get("tensor_extent_kind")
                        == kind
                )
                if len(extent_formals) != 1 or ssa_value_identity_cell(
                    target, value_id, book=book,
                ) is None:
                    return False
        if local_ids:
            if (len(local_ids) != len(keys)
                    or any(value_id is None for value_id in local_ids)):
                return False
        # One shape/rank/count actual can feed several tensor formals. Its
        # exact target ports, not uniqueness of the caller value across the
        # whole signature, identify this tensor's geometry binding.
        selected = tuple(zip(actual_ids, map(int, local_ids), strict=True))
        if any(pair not in booked_pairs or pair not in live_pairs for pair in selected):
            return False
        # In compact_codewords region6 -> region8, aggregate passthrough
        # replaces metadata Call actuals 6687/6690/6693 with 6449/6452/6455
        # while feed_ids retains the canonical source identities. The revised
        # Call receipt and live actual/formal ports prove the metadata pair;
        # feed_ids is not a second storage-actual binding.
        callee_input_ids = downstream.attributes.get("callee_input_ids")
        if callee_input_ids is not None and tuple(map(int, callee_input_ids)) != tuple(
            int(value.id) for value in target.args
        ):
            return False
        return True

    # A source function can return a computed region value. Publish region
    # outputs first, then read their projected metadata at the source Ret.
    # The caller's existing geometry fixed point revisits source functions
    # whose upstream Call has not yet acquired that result binding.
    function_items = tuple(module.functions.items())
    for callee_name, callee in (
        *(item for item in function_items
          if (item[1].metadata or {}).get("source_region_integral")),
        *(item for item in function_items
          if not (item[1].metadata or {}).get("source_region_integral")),
    ):
        table = module.tensor_tables.get(callee_name)
        returns = [
            instruction for block in callee.blocks.values()
            for instruction in block.instrs if instruction.op == "Ret"
        ]
        if len(returns) != 1:
            continue
        returned = returns[0]
        source_outputs = tuple(map(int, (callee.metadata or {}).get(
            "source_output_value_ids", (),
        )))
        if (source_outputs and len(returned.args) == 1
                and (table is None or table.by_id(int(returned.args[0].id)) is None)):
            from .ssa_record_return_state import posted_ssa_value_fact
            source_output = returned.args[0]
            source_id = int(source_output.id)
            source_metadata = tuple((source_output.accounting or {}).get(key)
                                    for key in keys)
            source_scope = function_scope_of(callee)
            upstream = tuple(
                row for row in book.page(TENSOR_METADATA_RESULT_BINDING).rows()
                if row[0] == source_scope and row[3] == source_id
                and book.page(TENSOR_METADATA_RESULT_BINDING).latest(row)[2]
                    == source_metadata
            )
            physical = posted_ssa_value_fact(callee, source_id, book=book)
            if (source_outputs == (source_id,)
                    and len(upstream) == 1
                    and all(type(value_id) is int for value_id in source_metadata)
                    and physical is not None and physical.storage == "span"
                    and isinstance(physical.rank, int) and physical.rank > 0
                    and all(ssa_value_identity_cell(callee, value_id, book=book)
                            is not None for value_id in source_metadata)):
                table = module.tensor_tables.setdefault(
                    callee_name, SSATensorTable(),
                )
                table.register(SSATensorDescriptor(
                    tensor_id=source_id, data_value_id=source_id,
                    dtype=str(physical.dtype), shape=tuple(source_output.shape or ()),
                    shape_value_id=source_metadata[0],
                    rank_value_id=source_metadata[1],
                    element_count_value_id=source_metadata[2],
                    storage="output", metadata_state="dynamic",
                    arena_id=source_id, allocation_owner=source_id,
                ))
        if table is None:
            continue
        original_outputs = tuple(returned.args)
        region_outputs = tuple(map(int, (
            (callee.metadata or {}).get("source_region_integral") or {}
        ).get("output_value_ids", ()) or source_outputs))
        # A late tensor slice can publish this region's metadata before a
        # sibling consumer is part of that partial module. On the full
        # module the Ret already has metadata values, and the booked result
        # binding is the authority for recognizing that exact suffix. Keep
        # the original semantic outputs so the same downstream forwarding
        # writer can run when the sibling arrives.
        if region_outputs and len(original_outputs) > len(region_outputs):
            booked_metadata = tuple(dict.fromkeys(
                int(value_id)
                for row in book.page(TENSOR_METADATA_RESULT_BINDING).rows()
                if row[2] == callee_name and int(row[3]) in region_outputs
                for value_id in book.page(
                    TENSOR_METADATA_RESULT_BINDING
                ).latest(row)[1]
            ))
            if tuple(int(value.id) for value in original_outputs[
                len(region_outputs):
            ]) == booked_metadata:
                original_outputs = original_outputs[:len(region_outputs)]
        semantic_outputs = (
            region_outputs if len(region_outputs) == len(original_outputs)
            else tuple(int(value.id) for value in original_outputs)
        )
        computed = {}
        view_proofs = {}
        for output, semantic_id in zip(original_outputs, semantic_outputs):
            descriptor = table.by_id(int(semantic_id))
            if int(semantic_id) != int(output.id):
                # A runtime reshape returns its source's data carrier. Its
                # semantic output is the distinct view whose shape vector
                # was validated above, not the storage owner's geometry.
                view_row = (function_scope_of(callee), callee.name,
                            int(semantic_id))
                proof = (
                    book.latest_ref(TENSOR_RUNTIME_RESHAPE_VALIDITY, view_row)
                    or book.latest_ref(TENSOR_RUNTIME_VIEW_GEOMETRY, view_row)
                    or book.latest_ref(
                        TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY, view_row,
                    )
                )
                source_id = (
                    int(book.page(TENSOR_RUNTIME_VIEW_GEOMETRY).cell(
                        proof.row, proof.column,
                    )[1])
                    if proof is not None
                    and proof.page == TENSOR_RUNTIME_VIEW_GEOMETRY
                    else int(book.page(
                        TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY,
                    ).cell(proof.row, proof.column)[0])
                    if proof is not None
                    and proof.page == TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY
                    else int(output.id)
                )
                source_descriptor = table.by_id(source_id)
                if (
                    proof is None or descriptor is None
                    or descriptor.alias_of != source_id
                    or descriptor.data_value_id != int(output.id)
                    or (
                        source_id != int(output.id)
                        and (
                            source_descriptor is None
                            or source_descriptor.data_value_id != int(output.id)
                        )
                    )
                ):
                    continue
                view_proofs[int(semantic_id)] = proof
            if descriptor is None or descriptor.metadata_state != "dynamic":
                continue
            ids = (
                descriptor.shape_value_id, descriptor.rank_value_id,
                descriptor.element_count_value_id,
            )
            if any(item is None for item in ids):
                continue
            computed[int(semantic_id)] = tuple(map(int, ids))
        if not computed:
            continue
        callee_values = {
            **{int(value.id): value for value in callee.args},
            **{
            int(value.id): value
            for block in callee.blocks.values()
            for instruction in block.instrs
            for value in (*instruction.args, instruction.res)
            if value is not None
            },
        }
        metadata_ids = tuple(dict.fromkeys(
            value_id for ids in computed.values() for value_id in ids
        ))
        if any(value_id not in callee_values for value_id in metadata_ids):
            continue
        if source_outputs and not (callee.metadata or {}).get("source_region_integral"):
            from .concordance_declarations import CALL_RECORD
            for caller in module.functions.values():
                records = tuple(module.call_table.get(caller.name, ()))
                record_ref = book.latest_ref(CALL_RECORD, (
                    module.call_table.owner, caller.name,
                ))
                for block in caller.blocks.values():
                    for call in tuple(block.instrs):
                        if (call.op != "Call" or call.res is None
                                or call.attributes.get("callee") != callee_name
                                or not call.attributes.get("source_linked")
                                or call.attributes.get("result_convention") == "ssa.aggregate"):
                            continue
                        caller_id = int(call.res.id)
                        matching = tuple(record for record in records
                                         if record.callee_symbol == callee_name
                                         and record.callsite_id == call.attributes.get(
                                             "plan_callsite_id")
                                         and record.result_bindings == ((
                                             int(source_outputs[0]), caller_id,
                                         ),))
                        original_cell = ssa_value_identity_cell(
                            caller, caller_id, book=book,
                        )
                        upstream_row = next((row for row in book.page(
                            TENSOR_METADATA_RESULT_BINDING
                        ).rows() if row[0] == function_scope_of(callee)
                            and row[3] == int(source_outputs[0])
                            and book.page(TENSOR_METADATA_RESULT_BINDING).latest(
                                row)[2] == computed[int(source_outputs[0])]), None)
                        upstream_ref = (None if upstream_row is None else
                                        book.latest_ref(TENSOR_METADATA_RESULT_BINDING,
                                                        upstream_row))
                        if not matching:
                            # A frame-aliased or otherwise non-materialized
                            # source return has no direct result binding to
                            # project through this aggregate output contract.
                            continue
                        if (len(matching) != 1 or record_ref is None
                                or original_cell is None or upstream_ref is None
                                or matching[0] not in book.page(CALL_RECORD).cell(
                                    record_ref.row, record_ref.column)):
                            raise ValueError(
                                "source tensor return has no exact linked result binding"
                            )
                        caller_scope = function_scope_of(caller)
                        causes = (record_ref, original_cell, upstream_ref)
                        aggregate = SSAValue(_mint_ssa_id(
                            book, caller_scope, TENSOR_LOWERING_VALUE, causes,
                            dtype="ssa.aggregate", shape=(),
                            stage=TENSOR_SSA_LOWERING,
                        ), dtype="ssa.aggregate", accounting={
                            "ssa_aggregate_outputs": (caller_id,),
                        })
                        index = SSAValue(_mint_ssa_id(
                            book, caller_scope, TENSOR_LOWERING_VALUE, causes,
                            dtype="int64", shape=(),
                            stage=TENSOR_SSA_LOWERING,
                            physical={"physical_dtype": "int64",
                                      "storage": "scalar", "rank": 0},
                        ), dtype="int64")
                        address = SSAValue(_mint_ssa_id(
                            book, caller_scope, ADDRESS, causes,
                            dtype="ptr", shape=(),
                            stage=TENSOR_SSA_LOWERING,
                        ), dtype="ptr")
                        result = call.res
                        call.res = aggregate
                        call.attributes = {
                            **dict(call.attributes),
                            "result_convention": "ssa.aggregate",
                            "output_ids": (caller_id,),
                            "callee_output_ids": (int(source_outputs[0]),),
                            "output_positions": (0,),
                            "computed_source_metadata_outputs": True,
                        }
                        position = block.instrs.index(call) + 1
                        block.instrs[position:position] = [
                            Instr("Const", [], index,
                                  attributes={"constant": 0}),
                            Instr("GetElementPtr", [aggregate, index], address,
                                  attributes={"aggregate_index": 0,
                                              "source_output_id": caller_id}),
                            Instr("Load", [address], result,
                                  attributes={"aggregate_index": 0,
                                              "source_output_id": caller_id}),
                        ]
                        changed = True
        calls = [
            (caller, block, call)
            for caller in module.functions.values()
            for block in caller.blocks.values()
            for call in block.instrs
            if call.op == "Call"
            and call.attributes.get("callee") == callee_name
            and call.res is not None
            and call.attributes.get("result_convention") == "ssa.aggregate"
        ]
        if not calls:
            continue
        prior_projections = {}
        for caller, _block, call in calls:
            output_ids = tuple(map(int, call.attributes.get("output_ids", ())))
            expected_outputs = semantic_outputs
            if call.attributes.get("computed_source_metadata_outputs"):
                bound = tuple(record for record in module.call_table.get(
                    caller.name, (),
                ) if record.callee_symbol == callee_name
                    and record.callsite_id == call.attributes.get("plan_callsite_id")
                    and tuple(int(callee_id) for callee_id, _caller_id
                              in record.result_bindings) == semantic_outputs)
                if len(bound) != 1:
                    raise ValueError("source metadata lost its exact Call result binding")
                expected_outputs = tuple(int(caller_id)
                                         for _callee_id, caller_id
                                         in bound[0].result_bindings)
            facts = {}
            for output_id, callee_ids in computed.items():
                row = (function_scope_of(caller), int(call.res.id),
                       str(callee_name), int(output_id))
                prior = book.page(TENSOR_METADATA_RESULT_BINDING).latest(row)
                if prior is None or tuple(prior[1]) != tuple(callee_ids):
                    break
                facts[output_id] = prior
            if output_ids == expected_outputs:
                if not facts:
                    prior_projections[id(call)] = None
                    continue
                # An aggregate output view may fold a returned metadata
                # slot onto the identical caller input formal. The booked
                # result row then names those input IDs, while the live
                # Call keeps only its semantic output. Reuse that exact
                # binding; minting new Loads would conflict with the row.
                caller_actual_ids = tuple(int(value.id) for value in call.args)
                callee_formal_ids = tuple(int(value.id) for value in callee.args)
                if (
                    len(facts) != len(computed)
                    or any(
                        int(caller_id) not in caller_actual_ids
                        or int(callee_id) not in callee_formal_ids
                        for fact in facts.values()
                        for callee_id, caller_id in zip(
                            fact[1], fact[2], strict=True,
                        )
                    )
                ):
                    prior_projections = {}
                    break
                prior_projections[id(call)] = facts
                continue
            projected_ids = tuple(dict.fromkeys(
                int(value_id) for fact in facts.values()
                for value_id in fact[2]
            ))
            if (
                len(facts) != len(computed)
                or not call.attributes.get("computed_tensor_metadata_outputs")
                or output_ids != (*expected_outputs, *projected_ids)
            ):
                prior_projections = {}
                break
            prior_projections[id(call)] = facts
        if len(prior_projections) != len(calls):
            continue
        if len(returned.args) == len(original_outputs):
            returned.args.extend(callee_values[value_id]
                                 for value_id in metadata_ids)
        elif tuple(int(value.id) for value in returned.args[
            len(original_outputs):
        ]) != metadata_ids:
            raise ValueError("computed region Ret has unbooked metadata outputs")
        callee_scope = function_scope_of(callee)
        for caller, block, call in calls:
            caller_scope = function_scope_of(caller)
            call_cell = ssa_value_identity_cell(
                caller, int(call.res.id), book=book,
            )
            if call_cell is None:
                raise ValueError("computed region result has no aggregate call cell")
            projected = {}
            replay = prior_projections[id(call)] is not None
            live_values = {
                int(value.id): value
                for block_values in caller.blocks.values()
                for instruction in block_values.instrs
                for value in (*instruction.args,
                              *((instruction.res,)
                                if instruction.res is not None else ()))
            } if replay else {}
            insert_at = block.instrs.index(call) + 1
            for position, callee_id in enumerate(
                metadata_ids, start=len(original_outputs)
            ):
                if replay:
                    booked = tuple(
                        int(caller_id)
                        for fact in prior_projections[id(call)].values()
                        for source_id, caller_id in zip(
                            fact[1], fact[2], strict=True,
                        ) if int(source_id) == int(callee_id)
                    )
                    if (len(set(booked)) != 1
                            or booked[0] not in live_values
                            or ssa_value_identity_cell(
                                caller, booked[0], book=book,
                            ) is None):
                        raise ValueError(
                            "computed region result lost its booked projected value"
                        )
                    projected[callee_id] = live_values[booked[0]]
                    continue
                source = callee_values[callee_id]
                source_cell = ssa_value_identity_cell(
                    callee, callee_id, book=book,
                )
                if source_cell is None:
                    raise ValueError("computed region metadata has no callee cell")
                source_fact = book.page(SSA_VALUE).cell(
                    source_cell.row, source_cell.column,
                )
                if source_fact.physical_dtype is None:
                    raise ValueError("computed region metadata has no physical dtype")
                causes = (call_cell, source_cell)

                def mint(dtype, shape, transform, physical=None):
                    return SSAValue(_mint_ssa_id(
                        book, caller_scope, transform, causes,
                        dtype=dtype, shape=shape,
                        stage=TENSOR_SSA_LOWERING, physical=physical,
                    ), dtype=dtype, shape=shape)

                index = mint("int64", (), TENSOR_LOWERING_VALUE,
                             {"physical_dtype": "int64",
                              "storage": "scalar", "rank": 0})
                address = mint("ptr", (), ADDRESS)
                result = mint(
                    str(source.dtype), tuple(source.shape or ()), LOAD,
                    {"physical_dtype": source_fact.physical_dtype,
                     "storage": "span" if source.shape else "scalar",
                     "rank": len(tuple(source.shape or ()))},
                )
                attributes = {
                    "region_index": call.attributes.get("region_index"),
                    "aggregate_index": int(position),
                    "source_output_id": int(result.id),
                    "computed_tensor_metadata": True,
                }
                block.instrs[insert_at:insert_at] = [
                    Instr("Const", [], index,
                          attributes={"constant": int(position)}),
                    Instr("GetElementPtr", [call.res, index], address,
                          attributes=attributes),
                    Instr("Load", [address], result,
                          attributes=attributes),
                ]
                insert_at += 3
                projected[callee_id] = result
            caller_ids = tuple(int(projected[item].id) for item in metadata_ids)
            if not replay:
                call.attributes = {
                    **dict(call.attributes),
                    "output_ids": (*tuple(map(int, call.attributes["output_ids"])),
                                   *caller_ids),
                    "callee_output_ids": (
                        *tuple(int(value.id) for value in original_outputs),
                        *metadata_ids,
                    ),
                    "output_positions": tuple(range(
                        len(original_outputs) + len(metadata_ids)
                    )),
                    "computed_tensor_metadata_outputs": True,
                }
                call.res.accounting = {
                    **dict(call.res.accounting or {}),
                    "ssa_aggregate_outputs": call.attributes["output_ids"],
                }
            for output_id, callee_ids in computed.items():
                caller_data_id = int(call.attributes["output_ids"][
                    semantic_outputs.index(output_id)
                ])
                source_call_proofs = ()
                if call.attributes.get("computed_source_metadata_outputs"):
                    from .concordance_declarations import CALL_RECORD
                    record_ref = book.latest_ref(CALL_RECORD, (
                        module.call_table.owner, caller.name,
                    ))
                    upstream_rows = tuple(
                        upstream_row for upstream_row in book.page(
                            TENSOR_METADATA_RESULT_BINDING
                        ).rows() if upstream_row[0] == callee_scope
                        and upstream_row[3] == output_id
                        and book.page(TENSOR_METADATA_RESULT_BINDING).latest(
                            upstream_row)[2] == callee_ids
                    )
                    if record_ref is None or len(upstream_rows) != 1:
                        raise ValueError(
                            "source tensor metadata lost its linked return proof"
                        )
                    source_call_proofs = (
                        record_ref,
                        book.latest_ref(TENSOR_METADATA_RESULT_BINDING,
                                        upstream_rows[0]),
                        ssa_value_identity_cell(caller, caller_data_id, book=book),
                    )
                caller_metadata = tuple(int(projected[item].id)
                                        for item in callee_ids)
                row = (caller_scope, int(call.res.id),
                       str(callee_name), int(output_id))
                sources = tuple(dict.fromkeys((
                    call_cell,
                    ssa_value_identity_cell(callee, output_id, book=book),
                    view_proofs.get(output_id),
                    *(ssa_value_identity_cell(callee, item, book=book)
                      for item in callee_ids),
                    *(ssa_value_identity_cell(caller, item, book=book)
                      for item in caller_metadata),
                    *source_call_proofs,
                )))
                if replay:
                    same_fact = (
                        int(output_id), callee_ids, caller_metadata,
                    )
                    if book.page(TENSOR_METADATA_RESULT_BINDING).latest(row) != same_fact:
                        raise ValueError(
                            "computed region result changed its booked projection"
                        )
                    # The float64 source-owned cast posts its selected
                    # output physical dtype after this metadata tuple was
                    # first projected. Its source4 and caller1 SSA_VALUE
                    # cells advance, but the tuple does not. Concord those
                    # new exact cells on the incumbent result row so its
                    # direct proof describes the current semantic values.
                    prior_result = book.latest_ref(
                        TENSOR_METADATA_RESULT_BINDING, row,
                    )
                    prior_causes = {
                        parent for parent, _stage in book.edges_into(prior_result)
                    }
                    new_sources = tuple(
                        cell for cell in sources
                        if cell is not None and cell.page == SSA_VALUE
                        and cell not in prior_causes
                    )
                    if new_sources:
                        book.post(
                            TENSOR_METADATA_RESULT_BINDING, row, same_fact,
                            stage=TENSOR_SSA_LOWERING,
                            provenance=Derived(new_sources), mode=Mode.CONCORD,
                        )
                else:
                    book.post(
                        TENSOR_METADATA_RESULT_BINDING, row,
                        (int(output_id), callee_ids, caller_metadata),
                        stage=TENSOR_SSA_LOWERING,
                        provenance=Derived(tuple(cell for cell in sources
                                                 if cell is not None)),
                        mode=Mode.CONCORD,
                    )
                projected_data = metadata_result_caller_value(
                    module, caller, call,
                    book.latest_ref(TENSOR_METADATA_RESULT_BINDING, row),
                    book=book,
                )
                if projected_data is None or projected_data[0] != caller_data_id:
                    raise ValueError(
                        "computed metadata lost its exact caller data identity"
                    )
                if call.attributes.get("computed_source_metadata_outputs"):
                    source_projection = tuple(
                        instruction for instruction in block.instrs
                        if instruction.op == "Load"
                        and instruction.res is not None
                        and int(instruction.res.id) == caller_data_id
                        and instruction.attributes.get("aggregate_index")
                            == semantic_outputs.index(output_id)
                    )
                    if len(source_projection) != 1:
                        raise ValueError(
                            "source tensor metadata lost its exact result projection"
                        )
                    source_projection[0].res.accounting = {
                        **dict(source_projection[0].res.accounting or {}),
                        "tensor_metadata_state": "dynamic",
                        **dict(zip(keys, caller_metadata, strict=True)),
                    }
                # A later region consuming this computed tensor receives
                # its metadata by the existing actual/formal call ABI. The
                # output projection above is the first point at which those
                # caller-owned SSA values exist, so input metadata forwarding
                # must follow it, in exact instruction order.
                from .concordance_declarations import TENSOR_METADATA_CALL_BINDING
                for downstream_block in caller.blocks.values():
                    for downstream in tuple(downstream_block.instrs):
                        if downstream.op != "Call" or downstream is call:
                            continue
                        downstream_name = downstream.attributes.get("callee")
                        target = module.functions.get(downstream_name)
                        if target is None or "__planned_region_" not in downstream_name:
                            continue
                        feeds = tuple(map(int, downstream.attributes.get("feed_ids", ())))
                        positions = tuple(index for index, actual in enumerate(downstream.args)
                                          if int(actual.id) == caller_data_id)
                        if len(positions) != 1:
                            continue
                        position = positions[0]
                        if (position >= len(downstream.args)
                                or position >= len(target.args)
                                or position >= len(feeds)
                                or int(target.args[position].id) != feeds[position]):
                            continue
                        target_tensor_id = int(target.args[position].id)
                        alias_cell = None
                        if target_tensor_id != caller_data_id:
                            from .concordance_declarations import AGGREGATE_PASSTHROUGH_REBINDING
                            alias_cell = book.latest_ref(AGGREGATE_PASSTHROUGH_REBINDING, (
                                function_scope_of(caller), target_tensor_id,
                            ))
                            if (alias_cell is None or book.page(alias_cell.page).cell(
                                    alias_cell.row, alias_cell.column)[0] != caller_data_id):
                                continue
                        if downstream_block is block and (
                            downstream_block.instrs.index(downstream)
                            <= downstream_block.instrs.index(call)
                        ):
                            continue
                        metadata_actuals = tuple(projected[item] for item in callee_ids)
                        target_table = module.tensor_tables.get(target.name)
                        local_descriptor = (
                            None if target_table is None
                            else target_table.by_id(target_tensor_id)
                        )
                        local_ids = (
                            () if local_descriptor is None
                            else (local_descriptor.shape_value_id,
                                  local_descriptor.rank_value_id,
                                  local_descriptor.element_count_value_id)
                        )
                        if downstream.res is None:
                            # ensure_tensor's raw-wrapper region returns its
                            # input. Aggregate legalization has already retired
                            # that Call's result after forwarding its metadata.
                            # Replay the exact live binding; a void Call has no
                            # new aggregate identity from which to mint another.
                            from .concordance_declarations import DEAD_STRUCTURAL_RETIREMENT
                            from .identity_concordance import Ref
                            retired = downstream.attributes.get("aggregate_retired_result")
                            binding = (book.latest_ref(TENSOR_METADATA_CALL_BINDING, (
                                function_scope_of(caller), caller.name,
                                downstream_block.name, retired.row[1], target.name,
                            )) if isinstance(retired, Ref)
                                and retired.page == DEAD_STRUCTURAL_RETIREMENT
                                and retired.row[0] == function_scope_of(caller)
                                and book.latest_ref(retired.page, retired.row) == retired
                                else None)
                            if booked_metadata_replay(
                                binding, row, downstream, target, metadata_actuals,
                                local_ids, target_tensor_id,
                            ):
                                continue
                            raise ValueError(
                                "void region call has no exact live tensor metadata binding"
                            )
                        existing_row = (
                            function_scope_of(caller), str(caller.name),
                            str(downstream_block.name),
                            int(downstream.res.id), str(target.name),
                        )
                        existing_binding = book.latest_ref(
                            TENSOR_METADATA_CALL_BINDING, existing_row,
                        )
                        if booked_metadata_replay(
                            existing_binding, row, downstream, target,
                            metadata_actuals, local_ids, target_tensor_id,
                        ):
                            # This exact consumer was forwarded in an earlier
                            # partial-module publication. Its local metadata
                            # formals already have a booked Call identity.
                            continue
                        promoted_extents = ()
                        if (local_descriptor is not None
                                and local_descriptor.metadata_state == "dynamic"
                                and all(value_id is not None for value_id in local_ids)
                                and tuple(map(int, local_ids)) != tuple(
                                    int(value.id) for value in metadata_actuals
                                )):
                            # Program: private arange -> cast -> sum. The cast
                            # region already defined exact extent(%6) values
                            # 4002/4003/4004 before the upstream Call bound
                            # caller metadata 4010/4013/4016. Both triples
                            # name one tensor geometry. Promote the local
                            # definitions to incoming formals at this Call
                            # binding, then retire their duplicate queries.
                            definitions = tuple(
                                (extent_block, extent)
                                for extent_block in target.blocks.values()
                                for extent in extent_block.instrs
                                if extent.res is not None
                                and int(extent.res.id) in local_ids
                            )
                            if (
                                len(set(map(int, local_ids))) != len(keys)
                                or len(definitions) != len(keys)
                                or any(
                                    sum(int(extent.res.id) == int(value_id)
                                        for _block, extent in definitions) != 1
                                    for value_id in local_ids
                                )
                                or any(
                                    extent.op != "extent"
                                    or tuple(int(value.id) for value in extent.args)
                                       != (target_tensor_id,)
                                    or extent.attributes.get("extent_kind") != kind
                                    or extent.attributes.get("tensor_operation") != "extent"
                                    or ssa_value_identity_cell(
                                        target, int(value_id), book=book,
                                    ) is None
                                    for kind, value_id in zip(
                                        ("shape", "rank", "element_count"), local_ids,
                                        strict=True,
                                    )
                                    for _block, extent in definitions
                                    if int(extent.res.id) == int(value_id)
                                )
                            ):
                                raise ValueError(
                                    "computed tensor metadata has no exact local extent promotion"
                                )
                            promoted_extents = definitions
                        formal_ids = (
                            tuple(map(int, local_ids)) if promoted_extents
                            else tuple(int(value.id) for value in metadata_actuals)
                        )
                        fresh_formals = tuple(
                            SSAValue(formal_id, dtype=actual.dtype,
                                     shape=tuple(actual.shape or ()),
                                     accounting={
                                         "tensor_extent_for": target_tensor_id,
                                         "tensor_extent_kind": key.removeprefix(
                                             "tensor_"
                                         ).removesuffix("_value_id"),
                                     })
                            for key, actual, formal_id in zip(
                                keys, metadata_actuals, formal_ids, strict=True,
                            )
                            if formal_id not in {
                                int(formal.id) for formal in target.args
                            }
                        )
                        if not fresh_formals:
                            if local_descriptor is None:
                                raise ValueError(
                                    "computed tensor metadata has no exact live Call binding"
                                )
                            continue
                        from .concordance_declarations import (
                            TENSOR_HEAP_OWNER_CALL_BINDING,
                        )
                        owner_rows = tuple(
                            owner_row for owner_row in book.page(
                                TENSOR_HEAP_OWNER_CALL_BINDING
                            ).rows()
                            if owner_row[:4] == (
                                function_scope_of(caller), caller.name,
                                int(downstream.res.id), target.name,
                            )
                        )
                        owner_facts = tuple(
                            book.page(TENSOR_HEAP_OWNER_CALL_BINDING).latest(
                                owner_row
                            ) for owner_row in owner_rows
                        )
                        owner_pairs = tuple(
                            (int(fact[1]), int(fact[2]))
                            for fact in owner_facts
                        )
                        # One aggregate producer may own both values and
                        # indices. The booked owner pairs must be precisely
                        # the live aligned call/formal tail before metadata
                        # can be inserted ahead of those owner ports.
                        owner_tail_size = len(owner_pairs)
                        if owner_tail_size:
                            live_tail = tuple(zip(
                                (int(value.id) for value in downstream.args[-owner_tail_size:]),
                                (int(value.id) for value in target.args[-owner_tail_size:]),
                                strict=True,
                            ))
                            if (len(set(owner_pairs)) != owner_tail_size
                                    or len(set(live_tail)) != owner_tail_size
                                    or set(live_tail) != set(owner_pairs)):
                                raise ValueError(
                                    "computed metadata found owner outside its exact call tail"
                                )
                        if owner_tail_size:
                            target.args[-owner_tail_size:-owner_tail_size] = fresh_formals
                            downstream.args[-owner_tail_size:-owner_tail_size] = metadata_actuals
                        else:
                            target.args.extend(fresh_formals)
                            downstream.args.extend(metadata_actuals)
                        for extent_block, extent in promoted_extents:
                            extent_block.instrs.remove(extent)
                        declared_inputs = downstream.attributes.get(
                            "callee_input_ids"
                        )
                        if declared_inputs is None:
                            original_count = (
                                len(downstream.args)
                                - len(metadata_actuals)
                            )
                            initial_actuals = downstream.args[:original_count]
                            initial_formals = target.args[:original_count]
                            if (
                                len(feeds) == original_count
                                and len(initial_formals) == original_count
                                and all(
                                    int(feed_id) == int(actual.id)
                                    == int(formal.id)
                                    for feed_id, actual, formal in zip(
                                        feeds, initial_actuals,
                                        initial_formals,
                                    )
                                )
                            ):
                                # The source region's exact feed IDs name
                                # both original endpoints. Publish that
                                # binding with the new metadata formals so
                                # later lifetime readers never infer it from
                                # an unqualified positional coincidence.
                                declared_inputs = tuple(
                                    int(formal.id)
                                    for formal in initial_formals
                                )
                        if declared_inputs is not None:
                            # The owner-bearing region Call already names
                            # its exact physical input formals. Computed
                            # metadata extends that same live Call, so its
                            # positional binding must extend with the new
                            # metadata formals in the same transaction.
                            if len(declared_inputs) != (
                                len(downstream.args)
                                - len(metadata_actuals)
                            ):
                                raise ValueError(
                                    "computed tensor metadata found stale "
                                    "callee input binding"
                                )
                            old = tuple(map(int, declared_inputs))
                            new = tuple(int(formal.id)
                                        for formal in fresh_formals)
                            downstream.attributes[
                                "callee_input_ids"
                            ] = (
                                (*old[:-owner_tail_size], *new,
                                 *old[-owner_tail_size:])
                                if owner_tail_size else (*old, *new)
                            )
                        metadata_feed_ids = tuple(
                            int(value.id) for value in metadata_actuals
                        )
                        metadata_feed_shapes = tuple(
                            tuple(value.shape or ())
                            for value in metadata_actuals
                        )
                        metadata_feed_dtypes = tuple(
                            str(value.dtype) for value in metadata_actuals
                        )
                        old_feed_shapes = tuple(downstream.attributes.get(
                            "feed_shapes", ()))
                        old_feed_dtypes = tuple(downstream.attributes.get(
                            "feed_dtypes", ()))
                        if (len(feeds) != len(old_feed_shapes)
                                or len(feeds) != len(old_feed_dtypes)):
                            raise ValueError(
                                "computed metadata found incomplete region feeds"
                            )
                        downstream.attributes = {
                            **dict(downstream.attributes),
                            "feed_ids": (
                                (*feeds[:-owner_tail_size], *metadata_feed_ids,
                                 *feeds[-owner_tail_size:])
                                if owner_tail_size else (*feeds, *metadata_feed_ids)
                            ),
                            "feed_shapes": (
                                (*old_feed_shapes[:-owner_tail_size], *metadata_feed_shapes,
                                 *old_feed_shapes[-owner_tail_size:])
                                if owner_tail_size else (*old_feed_shapes, *metadata_feed_shapes)
                            ),
                            "feed_dtypes": (
                                (*old_feed_dtypes[:-owner_tail_size], *metadata_feed_dtypes,
                                 *old_feed_dtypes[-owner_tail_size:])
                                if owner_tail_size else (*old_feed_dtypes, *metadata_feed_dtypes)
                            ),
                        }
                        integral = dict(target.metadata.get("source_region_integral", {}))
                        if integral:
                            integral["capture_value_ids"] = tuple(dict.fromkeys((
                                *map(int, integral.get("capture_value_ids", ())),
                                *(int(value.id) for value in fresh_formals),
                            )))
                            target.metadata["source_region_integral"] = integral
                        pending_input_bindings.append((
                            caller, downstream_block, downstream, target,
                            metadata_actuals, fresh_formals, row, target_tensor_id, alias_cell,
                        ))
                        changed = True
            changed = changed or not replay
    # A later computed result inserts its aggregate GEP/Loads before an
    # earlier region Call.  Its instruction index is mutable across later
    # passes, whereas its booked aggregate result ID is the exact Call
    # identity. Program: arange -> cast -> sum. Rows keyed 47/60 by position
    # later named GEPs; the result identity stays on the Call.
    posted_binding_rows = set()
    for (caller, downstream_block, downstream, target, metadata_actuals,
         fresh_formals, result_row, target_tensor_id, alias_cell) in pending_input_bindings:
        binding_row = (
            function_scope_of(caller), str(caller.name),
            str(downstream_block.name),
            int(downstream.res.id), str(target.name),
        )
        if binding_row not in posted_binding_rows:
            # Two distinct computed views may feed one later planned Call.
            # The Call has one identity and one complete argument list, so
            # its receipt must contain both metadata triples and cite both
            # producer result bindings. Reposting only the last triple made
            # the page disagree with the live six-argument Call while the
            # prior view's metadata had already been inserted.
            same_call = tuple(
                entry for entry in pending_input_bindings
                if entry[2] is downstream and entry[3] is target
            )
            requested_pairs = tuple(
                (int(actual.id), int(formal.id))
                for entry in same_call
                for actual, formal in zip(entry[4], entry[5], strict=True)
            )
            if len(set(requested_pairs)) != len(requested_pairs):
                raise ValueError("computed tensor metadata repeats a Call binding")
            live_pairs = tuple(
                (int(actual.id), int(formal.id))
                for actual, formal in zip(
                    downstream.args, target.args, strict=True,
                )
                if (int(actual.id), int(formal.id)) in requested_pairs
            )
            if len(live_pairs) != len(requested_pairs):
                raise ValueError("computed tensor metadata lost a live Call binding")
            result_refs = tuple(
                book.latest_ref(TENSOR_METADATA_RESULT_BINDING, entry[6])
                for entry in same_call
            )
            if any(ref is None for ref in result_refs):
                raise ValueError("computed tensor metadata lost a producer binding")
            binding_sources = tuple(dict.fromkeys(
                item for item in (
                    *result_refs,
                    *(entry[8] for entry in same_call if entry[8] is not None),
                    ssa_value_identity_cell(
                        caller, downstream.res.id, book=book,
                    ),
                    *(ssa_value_identity_cell(caller, actual_id, book=book)
                      for actual_id, _formal_id in live_pairs),
                    *(ssa_value_identity_cell(target, formal_id, book=book)
                      for _actual_id, formal_id in live_pairs),
                ) if item is not None
            ))
            book.post(
                TENSOR_METADATA_CALL_BINDING, binding_row,
                (tuple(actual_id for actual_id, _ in live_pairs),
                 tuple(formal_id for _, formal_id in live_pairs)),
                stage=TENSOR_SSA_LOWERING,
                provenance=Derived(binding_sources), mode=Mode.REVISE,
            )
            posted_binding_rows.add(binding_row)
        # Program: root(bits).unsqueeze(1) + 1. The downstream region's
        # live tensor formal 3 received shape/rank/count formals here, but
        # its accounting still named none of them. The incumbent formal
        # pruner then dropped all three before the deferred Add could read
        # them. The exact call binding is the authority for this local
        # view: project its paired metadata formals onto every occurrence
        # of the one tensor formal that the upstream result feeds.
        bound = book.latest_ref(TENSOR_METADATA_CALL_BINDING, binding_row)
        if bound is None:
            raise ValueError("computed tensor metadata binding was not posted")
        tensor_id = target_tensor_id
        formal_matches = tuple(
            formal for formal in target.args if int(formal.id) == tensor_id
        )
        if len(formal_matches) != 1:
            raise ValueError("computed tensor metadata has no unique live tensor formal")
        metadata_ids = tuple(int(formal.id) for formal in fresh_formals)
        if len(metadata_ids) != len(keys):
            raise ValueError("computed tensor metadata has incomplete formal geometry")
        occurrences = (
            *formal_matches,
            *(value for block in target.blocks.values()
              for instruction in block.instrs
              for value in (*instruction.args,
                            *((instruction.res,) if instruction.res is not None else ()))
              if int(value.id) == tensor_id),
        )
        for occurrence in occurrences:
            prior = dict(occurrence.accounting or {})
            if any(prior.get(key) not in {None, metadata_id}
                   for key, metadata_id in zip(keys, metadata_ids, strict=True)):
                raise ValueError("computed tensor metadata conflicts with existing formal geometry")
            occurrence.accounting = {
                **prior,
                "tensor_metadata_state": "dynamic",
                **dict(zip(keys, metadata_ids, strict=True)),
            }
        target_table = module.tensor_tables.get(target.name)
        existing = None if target_table is None else target_table.by_id(tensor_id)
        if existing is not None:
            if existing.metadata_state not in {"unresolved", "dynamic"}:
                raise ValueError("computed tensor metadata conflicts with static descriptor")
            if any(prior not in {None, metadata_id} for prior, metadata_id in zip(
                (existing.shape_value_id, existing.rank_value_id,
                 existing.element_count_value_id), metadata_ids, strict=True,
            )):
                raise ValueError(
                    "computed tensor metadata conflicts with existing descriptor "
                    f"{target.name} %{tensor_id}: "
                    f"{(existing.shape_value_id, existing.rank_value_id, existing.element_count_value_id)} "
                    f"vs {metadata_ids}"
                )
            target_table.tensors[tensor_id] = dataclasses.replace(
                existing, metadata_state="dynamic", byte_size=None,
                shape_value_id=metadata_ids[0],
                rank_value_id=metadata_ids[1],
                element_count_value_id=metadata_ids[2],
            )
    # A later invocation may receive the same booked Call after source type
    # settlement. Program: reshape -> scalar-first Sub -> private sum. Its
    # region2 formal7 had the exact metadata binding before its SSA_VALUE
    # established span/rank2; no fresh binding is pending on the later pass.
    # Read the live Call and its existing receipt to register that input
    # descriptor as soon as the physical declaration is complete.
    from .ssa_record_return_state import posted_ssa_value_fact
    for binding_row in tuple(book.page(TENSOR_METADATA_CALL_BINDING).rows()):
        target = module.functions.get(str(binding_row[4]))
        caller = module.functions.get(str(binding_row[1]))
        if target is None or caller is None:
            continue
        bound = book.latest_ref(TENSOR_METADATA_CALL_BINDING, binding_row)
        actual_ids, formal_ids = book.page(TENSOR_METADATA_CALL_BINDING).cell(
            bound.row, bound.column,
        )
        block = caller.blocks.get(str(binding_row[2]))
        matching = () if block is None else tuple(
            instruction for instruction in block.instrs
            if instruction.op == "Call" and instruction.res is not None
            and int(instruction.res.id) == int(binding_row[3])
            and instruction.attributes.get("callee") == target.name
        )
        if len(matching) != 1 or len(matching[0].args) != len(target.args):
            continue
        live_pairs = tuple((int(actual.id), int(formal.id))
                           for actual, formal in zip(
                               matching[0].args, target.args, strict=True,
                           ))
        if any((int(actual_id), int(formal_id)) not in live_pairs
               for actual_id, formal_id in zip(
                   actual_ids, formal_ids, strict=True,
               )):
            continue
        for formal in target.args:
            tensor_id = int(formal.id)
            accounting = formal.accounting or {}
            metadata_ids = tuple(accounting.get(key) for key in keys)
            if (accounting.get("tensor_metadata_state") != "dynamic"
                    or any(type(value_id) is not int for value_id in metadata_ids)
                    or any(value_id not in formal_ids for value_id in metadata_ids)):
                continue
            target_table = module.tensor_tables.get(target.name)
            if target_table is not None and target_table.by_id(tensor_id) is not None:
                continue
            formal_fact = posted_ssa_value_fact(target, tensor_id, book=book)
            if (formal_fact is None or formal_fact.storage != "span"
                    or not isinstance(formal_fact.rank, int)
                    or formal_fact.rank <= 0):
                continue
            metadata_formals = tuple(next((member for member in target.args
                                          if int(member.id) == value_id), None)
                                     for value_id in metadata_ids)
            if (any(member is None for member in metadata_formals)
                    or any((member.accounting or {}).get("tensor_extent_for") != tensor_id
                           or (member.accounting or {}).get("tensor_extent_kind") != kind
                           or ssa_value_identity_cell(target, value_id, book=book)
                              is None
                           for kind, value_id, member in zip(
                               ("shape", "rank", "element_count"),
                               metadata_ids, metadata_formals, strict=True,
                           ))
                    or not any(
                        cause.page == TENSOR_METADATA_RESULT_BINDING
                        and cause.row[3] == tensor_id
                        for cause, _stage in book.edges_into(bound)
                    )):
                raise ValueError(
                    "computed tensor input has no exact metadata formal proof"
                )
            target_table = module.tensor_tables.setdefault(
                target.name, SSATensorTable(),
            )
            target_table.register(SSATensorDescriptor(
                tensor_id=tensor_id, data_value_id=tensor_id,
                dtype=str(formal_fact.dtype),
                shape=tuple(formal.shape or ()),
                shape_value_id=metadata_ids[0],
                rank_value_id=metadata_ids[1],
                element_count_value_id=metadata_ids[2],
                storage="input", metadata_state="dynamic",
                arena_id=tensor_id, allocation_owner=tensor_id,
                writable=False,
            ))
            changed = True
    # A local source section can publish a reshape's metadata only after
    # visiting its still-deferred consumer.  For reshape + private arange,
    # attempting release here previously rejected that consumer before the
    # existing whole-program tensor retry could lower it.  Section linking
    # publishes metadata now; the complete module runs this same strict owner
    # proof after that retry. No release receipt is invented while deferred.
    changed |= _publish_source_owned_return_bindings(module, book)
    if not finalize_private_owners:
        return changed

    # Program: private arange -> cast -> sum returns a scalar. The two heap
    # owners are caller frame cells, and neither tensor escapes. Their
    # allocation outlives every region consumer and is released by the
    # existing heap instruction immediately before this root's Ret. The
    # A later sole Ret is also safe when the producer block dominates it and
    # every use stays in the producer block. The control proof comes from the
    # exact booked SSA edges; an unbooked successor remains a refusal.
    from .concordance_declarations import (
        SSA_CONTROL_EDGE,
        TENSOR_HEAP_OWNER_CALL_BINDING,
        TENSOR_HEAP_RELEASE_BINDING,
        TENSOR_HEAP_CONSUMER_NONESCAPE,
        TENSOR_SOURCE_OWNED_RETURN,
    )
    from .ssa_heap_allocation import heap_release
    from .ssa_record_return_state import ssa_block_identity_cell
    from .identity_concordance import CorrelationTable
    from .identity_concordance import Ref
    from .concordance_declarations import (
        AGGREGATE_PASSTHROUGH_REBINDING, DEAD_STRUCTURAL_RETIREMENT,
    )
    source_returns = tuple(
        (row, book.latest_ref(TENSOR_SOURCE_OWNED_RETURN, row),
         book.page(TENSOR_SOURCE_OWNED_RETURN).latest(row))
        for row in book.page(TENSOR_SOURCE_OWNED_RETURN).rows()
    )
    owner_contexts = []
    for owner_row in tuple(book.page(TENSOR_HEAP_OWNER_CALL_BINDING).rows()):
        owner_fact = book.page(TENSOR_HEAP_OWNER_CALL_BINDING).latest(owner_row)
        forwarded = tuple(
            source_ref for source_row, source_ref, source_fact in source_returns
            if (source_row[3], int(source_row[4]), int(source_fact[1]))
            == (str(owner_row[1]), int(owner_row[4]), int(owner_fact[1]))
        )
        if forwarded:
            # This owner is a source function's formal. The exact returned
            # carrier forwards its cell to each calling frame; releasing it
            # here would invalidate the caller's borrowed result.
            continue
        owner_contexts.append((
            owner_row,
            book.latest_ref(TENSOR_HEAP_OWNER_CALL_BINDING, owner_row),
            owner_fact,
        ))
    for source_row, source_ref, source_fact in source_returns:
        owner_contexts.append((
            (source_row[0], source_row[1], source_row[2],
             source_row[3], source_fact[4]),
            source_ref,
            (source_fact[4], source_fact[3], source_fact[1]),
        ))
    for owner_row, owner_origin, owner_fact in owner_contexts:
        caller = module.functions.get(str(owner_row[1]))
        if caller is None:
            raise ValueError("owned tensor release lost its exact caller")
        tensor_id, owner_actual_id, _owner_formal_id = map(int, owner_fact)
        release_row = (
            function_scope_of(caller), caller.name, owner_actual_id,
        )
        if book.latest_ref(TENSOR_HEAP_RELEASE_BINDING, release_row) is not None:
            continue
        owner_actuals = tuple(
            value for value in caller.args
            if int(value.id) == owner_actual_id
        )
        returns = tuple(
            (block, instruction)
            for block in caller.blocks.values()
            for instruction in block.instrs
            if instruction.op == "Ret"
        )
        producer = tuple(
            (block, instruction)
            for block in caller.blocks.values()
            for instruction in block.instrs
            if instruction.op == "Call" and instruction.res is not None
            and int(instruction.res.id) == int(owner_row[2])
            and instruction.attributes.get("callee") == owner_row[3]
        )
        if (
            len(owner_actuals) != 1
            or len(returns) != 1
            or len(producer) != 1
            or any(int(value.id) in {tensor_id, owner_actual_id}
                   for value in returns[0][1].args)
        ):
            raise ValueError(
                "private owned tensor has no single-block nonescaping lifetime"
            )
        block = returns[0][0]
        producer_block = producer[0][0]
        producer_position = producer_block.instrs.index(producer[0][1])
        return_position = block.instrs.index(returns[0][1])
        uses = tuple(
            (other_block, other_block.instrs.index(instruction))
            for other_block in caller.blocks.values()
            for instruction in other_block.instrs
            if instruction is not producer[0][1]
            and any(int(value.id) == tensor_id
                    for value in instruction.args)
        )
        control_cells = ()
        # A guarded Huffman selector allocates and consumes private spans in
        # its taken block, then joins an early-return path. That producer
        # cannot dominate the sole Ret. Its lifetime nevertheless ends before
        # leaving its own block: all uses and their exact nonescape proofs
        # below remain in that block. Release before its control terminator.
        if (producer_block is not block
                and producer_block.name not in CorrelationTable._dominators(
                    caller).get(block.name, ())
                and all(other is producer_block for other, _position in uses)
                and producer_block.instrs
                and producer_block.instrs[-1].op in {"Br", "CondBr"}):
            block = producer_block
            return_position = len(block.instrs) - 1
        if producer_block is not block:
            dominators = CorrelationTable._dominators(caller)
            if producer_block.name not in dominators.get(block.name, ()):
                raise ValueError(
                    "private owned tensor producer does not dominate its return"
                )
            scope = function_scope_of(caller)
            reachable = set(dominators)
            live_edges = {
                (name, successor)
                for name in reachable
                for successor in caller.blocks[name].successors
                if successor in reachable
            }
            booked_edges = {}
            for edge_row in book.page(SSA_CONTROL_EDGE).rows():
                if edge_row[:2] != (scope, caller.name):
                    continue
                before, after = edge_row[2:4]
                edge = (str(before.row[2]), str(after.row[2]))
                if edge not in live_edges:
                    continue
                if (
                    before != ssa_block_identity_cell(
                        caller, edge[0], book=book,
                    )
                    or after != ssa_block_identity_cell(
                        caller, edge[1], book=book,
                    )
                ):
                    continue
                edge_cell = book.latest_ref(SSA_CONTROL_EDGE, edge_row)
                if edge_cell is not None and book.page(SSA_CONTROL_EDGE).cell(
                    edge_cell.row, edge_cell.column,
                ) is True:
                    booked_edges.setdefault(edge, set()).add(edge_cell)
            if any(len(booked_edges.get(edge, ())) != 1
                   for edge in live_edges):
                raise ValueError(
                    "private owned tensor return has unbooked SSA control"
                )
            control_cells = tuple(
                next(iter(booked_edges[edge])) for edge in sorted(live_edges)
            )
        if (
            (producer_block is block and producer_position >= return_position)
            or any(other_block is not producer_block
                   or position <= producer_position
                   for other_block, position in uses)
        ):
            raise ValueError(
                "private owned tensor has no ordered final consumer"
            )
        nonescape_cells = []
        for other_block, position in uses:
            consumer = other_block.instrs[position]
            target = module.functions.get(str(
                consumer.attributes.get("callee") or ""
            )) if consumer.op == "Call" else None
            if target is None or "__planned_region_" not in target.name:
                raise ValueError(
                    "private owned tensor has an unproved escaping use"
                )
            declared = consumer.attributes.get("callee_input_ids")
            if declared is None:
                raise ValueError(
                    "private owned tensor consumer has no exact call binding"
                )
            declared_inputs = tuple(map(int, declared))
            if len(declared_inputs) != len(consumer.args):
                raise ValueError(
                    "private owned tensor consumer lacks exact formal binding"
                )
            consumed_formals = {
                formal_id for formal_id, actual in zip(
                    declared_inputs, consumer.args
                ) if int(actual.id) == tensor_id
            }
            if not consumed_formals:
                raise ValueError(
                    "private owned tensor use lost its formal identity"
                )
            target_table = module.tensor_tables.get(target.name)
            target_returns = tuple(
                returned for target_block in target.blocks.values()
                for returned in target_block.instrs
                if returned.op == "Ret"
            )
            if target_table is None or len(target_returns) != 1:
                raise ValueError(
                    "private owned tensor consumer has no return layout"
                )
            # `_ascending_order`: ensure_tensor's planned wrapper returns
            # borrowed indices19 to the same owning caller. Aggregate
            # legalization retired that Call to void and rebound view20 to
            # the caller's raw19. The later cast consumes raw19 and returns
            # distinct storage22. This exact retired/rebound identity can
            # remain live until that cast; it does not transfer the owner.
            reborrow = None
            if consumer.res is None:
                retired = consumer.attributes.get("aggregate_retired_result")
                passthroughs = tuple(consumer.attributes.get(
                    "aggregate_output_passthrough_bindings", (),
                ))
                if (not isinstance(retired, Ref)
                        or retired.page != DEAD_STRUCTURAL_RETIREMENT
                        or book.latest_ref(retired.page, retired.row) != retired
                        or book.page(retired.page).cell(retired.row, retired.column)
                            != ("Call", "None")
                        or consumer.attributes.get("result_convention") != "void"
                        or tuple(consumer.attributes.get("output_ids", ()))
                        or tuple(consumer.attributes.get("callee_output_ids", ()))
                        or len(passthroughs) != 1
                        or len(passthroughs[0]) != 6):
                    raise ValueError(
                        "private owned tensor void consumer lacks exact retirement"
                    )
                retirement_causes = {
                    source for source, _stage in book.edges_into(retired)
                }
                prior_calls = tuple(source for source in retirement_causes
                                    if source.page == SSA_VALUE
                                    and source.row[0] == function_scope_of(caller))
                rebindings = tuple(source for source in retirement_causes
                                   if source.page == AGGREGATE_PASSTHROUGH_REBINDING)
                if len(prior_calls) != 1 or len(rebindings) != 1:
                    raise ValueError(
                        "private owned tensor retirement lost its prior Call or rebinding"
                    )
                prior_call, rebinding = prior_calls[0], rebindings[0]
                alias_id = int(rebinding.row[1])
                rebinding_causes = {
                    source for source, _stage in book.edges_into(rebinding)
                }
                binding_fact = book.page(AGGREGATE_PASSTHROUGH_REBINDING).cell(
                    rebinding.row, rebinding.column,
                )
                if (retirement_causes != {prior_call, rebinding}
                        or not all(any(
                            source.page == SSA_VALUE
                            and source.row == (function_scope_of(caller), value_id)
                            for source in rebinding_causes
                        ) for value_id in (alias_id, tensor_id))
                        or book.latest_ref(prior_call.page, prior_call.row) != prior_call
                        or book.latest_ref(rebinding.page, rebinding.row) != rebinding
                        or rebinding.row != (function_scope_of(caller), alias_id)
                        or binding_fact != (tensor_id, target.name, alias_id)
                        or passthroughs[0][4] != "exact_callee_output_formal"
                        or any(int(value.id) == alias_id
                               for caller_block in caller.blocks.values()
                               for live in caller_block.instrs
                               for value in live.args)
                        or any(int(value.id) in {tensor_id, alias_id, owner_actual_id}
                               for value in returns[0][1].args)):
                    raise ValueError(
                        "private owned tensor reborrow escapes its owning caller"
                    )
                reborrow = (prior_call, rebinding, retired)
            metadata_result_id = (
                int(consumer.res.id) if consumer.res is not None
                else int(reborrow[0].row[1]) if reborrow is not None else None
            )
            metadata_rows = tuple(
                metadata_row for metadata_row in book.page(
                    TENSOR_METADATA_RESULT_BINDING
                ).rows()
                if metadata_result_id is not None
                and metadata_row[:3] == (
                    function_scope_of(caller), metadata_result_id, target.name,
                )
            )
            metadata_ids = {
                int(value_id)
                for metadata_row in metadata_rows
                for value_id in book.page(
                    TENSOR_METADATA_RESULT_BINDING
                ).latest(metadata_row)[1]
            }
            if reborrow is not None:
                data_outputs = tuple(
                    int(output.id) for output in target_returns[0].args
                    if int(output.id) not in metadata_ids
                )
                if (len(data_outputs) != 1
                        or data_outputs[0] not in consumed_formals
                        or passthroughs[0][:4] != (
                            0, alias_id, data_outputs[0], tensor_id,
                        )):
                    raise ValueError(
                        "private owned tensor reborrow lacks exact data output"
                    )
                reborrow = (*reborrow, data_outputs[0])
            output_layouts = []
            output_cells = []
            for output in target_returns[0].args:
                if int(output.id) in metadata_ids:
                    continue
                descriptor = target_table.by_id(int(output.id))
                if descriptor is None:
                    # Program: private (left.unsqueeze(1) +
                    # right.unsqueeze(0)).sum().item(). Ret10 is an int64
                    # scalar, not a tensor descriptor. Its exact posted
                    # scalar storage cannot carry the borrowed span; keep
                    # that same output identity in the nonescape receipt.
                    from ..transmogrifier.dtype_layout import resolve
                    from .ssa_record_return_state import posted_ssa_value_fact

                    scalar = posted_ssa_value_fact(
                        target, int(output.id), book=book,
                    )
                    layout = resolve(
                        None if scalar is None else scalar.physical_dtype
                    )
                    cell = ssa_value_identity_cell(
                        target, int(output.id), book=book,
                    )
                    if (scalar is None or scalar.storage != "scalar"
                            or scalar.rank != 0 or layout is None
                            or layout.kind not in {"int", "uint", "float", "bool"}
                            or int(output.id) in consumed_formals or cell is None):
                        raise ValueError(
                            "private owned tensor consumer may return its storage"
                        )
                    output_layouts.append((int(output.id), int(output.id), None))
                    output_cells.append(cell)
                    continue
                if (
                    int(descriptor.data_value_id) in consumed_formals
                    or descriptor.alias_of in consumed_formals
                ) and not (
                    reborrow is not None
                    and int(output.id) == reborrow[3]
                    and int(descriptor.data_value_id) == reborrow[3]
                    and descriptor.alias_of is None
                ):
                    raise ValueError(
                        "private owned tensor consumer may return its storage"
                    )
                output_layouts.append((
                    int(output.id), int(descriptor.data_value_id),
                    None if descriptor.alias_of is None
                    else int(descriptor.alias_of),
                ))
                for value_id in (
                    int(output.id), int(descriptor.data_value_id),
                    *((int(descriptor.alias_of),)
                      if descriptor.alias_of is not None else ()),
                ):
                    cell = ssa_value_identity_cell(
                        target, value_id, book=book,
                    )
                    if cell is None:
                        raise ValueError(
                            "private tensor consumer layout lacks an SSA cell"
                        )
                    output_cells.append(cell)
            if consumer.res is None and reborrow is None:
                raise ValueError(
                    "private owned tensor consumer lacks a Call result"
                )
            consumer_cell = ssa_value_identity_cell(
                caller, (int(consumer.res.id) if consumer.res is not None
                         else int(reborrow[0].row[1])), book=book,
            )
            tensor_cell = ssa_value_identity_cell(
                caller, tensor_id, book=book,
            )
            if consumer_cell is None or tensor_cell is None:
                raise ValueError(
                    "private owned tensor consumer lacks exact SSA cells"
                )
            metadata_cells = tuple(
                book.latest_ref(TENSOR_METADATA_RESULT_BINDING, row)
                for row in metadata_rows
            )
            nonescape_row = (
                function_scope_of(caller), caller.name,
                int(consumer.res.id) if consumer.res is not None
                else int(reborrow[0].row[1]), owner_actual_id,
            )
            book.post(
                TENSOR_HEAP_CONSUMER_NONESCAPE, nonescape_row,
                (target.name, tensor_id,
                 tuple(sorted(consumed_formals)), tuple(output_layouts)),
                stage=TENSOR_SSA_LOWERING,
                provenance=Derived(tuple(dict.fromkeys((
                    owner_origin,
                    consumer_cell, tensor_cell, *output_cells,
                    *metadata_cells,
                    *((reborrow[1], reborrow[2])
                      if reborrow is not None else ()),
                )))),
                mode=Mode.CONCORD,
            )
            nonescape_cells.append(book.latest_ref(
                TENSOR_HEAP_CONSUMER_NONESCAPE, nonescape_row,
            ))
        owner_actual = owner_actuals[0]
        block.instrs.insert(return_position, heap_release(owner_actual))
        release_sources = (
            owner_origin,
            ssa_value_identity_cell(caller, int(owner_row[2]), book=book),
            ssa_value_identity_cell(caller, tensor_id, book=book),
            ssa_value_identity_cell(caller, owner_actual_id, book=book),
            ssa_block_identity_cell(caller, block.name, book=book),
            *((ssa_block_identity_cell(caller, producer_block.name, book=book),)
              if producer_block is not block else ()),
            *control_cells,
            *nonescape_cells,
        )
        if any(cell is None for cell in release_sources):
            raise ValueError("private owned tensor release has no exact cells")
        book.post(
            TENSOR_HEAP_RELEASE_BINDING, release_row,
            (int(owner_row[2]), tensor_id, block.name),
            stage=TENSOR_SSA_LOWERING,
            provenance=Derived(release_sources), mode=Mode.CONCORD,
        )
    return changed


def settle_shape_only_repository_returns(module: IRModule) -> bool:
    """Restore identity returns erased from pure shape-only regions.

    The numeric isolator can place a terminal ``reshape`` whose source and
    target descriptors are already identical on the boundary of a planned
    region.  Structural SSA then quite correctly removes the data-moving
    operation, but older region assembly also removes the return itself.  The
    caller still records the exact one-input/one-output region contract.  Use
    that contract to spell the remaining operation as an identity return so
    the caller-owned output buffer is populated instead of left uninitialized.

    This is deliberately limited to pure regions and an unambiguous unary
    call edge with an identical tensor descriptor.  Numeric regions, scalar
    transforms, broadcasts, and shape-changing views are not inferred here.
    """

    calls_by_callee: dict[str, list[tuple[Any, Instr]]] = {}
    for caller in module.functions.values():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Call", "call"}:
                    continue
                callee = str(instruction.attributes.get("callee") or "")
                if callee in module.functions:
                    calls_by_callee.setdefault(callee, []).append((
                        caller, instruction,
                    ))

    changed = False
    harmless = {"Const", "StaticRef", "extent", "getattr", "GetAttr"}
    return_ops = {Handler.Ret.value, "ret", "Return", "return"}
    for name, function in module.functions.items():
        instructions = [
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
        ]
        if any(instruction.op in return_ops for instruction in instructions):
            continue
        if any(instruction.op not in harmless for instruction in instructions):
            continue
        if len(function.args) != 1:
            continue
        callsites = calls_by_callee.get(name, ())
        if not callsites:
            continue
        source = function.args[0]
        source_shape = tuple(source.shape or ())
        source_dtype = source.dtype
        exact = True
        for caller, call in callsites:
            output_ids = tuple(map(
                int, call.attributes.get("output_ids", ())
            ))
            if len(call.args) != 1 or len(output_ids) != 1:
                exact = False
                break
            projected = next((
                instruction.res
                for block in caller.blocks.values()
                for instruction in block.instrs
                if instruction.op in {"Load", "load"}
                and instruction.res is not None
                and int(instruction.attributes.get(
                    "source_output_id", -1
                )) == output_ids[0]
            ), None)
            if projected is None:
                exact = False
                break
            if tuple(projected.shape or ()) != source_shape:
                exact = False
                break
            if (
                source_dtype is not None
                and projected.dtype is not None
                and source_dtype != projected.dtype
            ):
                exact = False
                break
        if not exact:
            continue
        last_block = next(reversed(tuple(function.blocks.values())), None)
        if last_block is None:
            continue
        last_block.instrs.append(Instr(Handler.Ret.value, [source], None))
        function.metadata["source_output_value_ids"] = (int(source.id),)
        changed = True
    return changed


_VIEW_OPERATIONS = frozenset({
    "reshape", "view", "flatten", "unsqueeze", "squeeze", "contiguous",
    "detach",
})
_CAST_OPERATIONS = {
    # Value-precision casts under the double-typed working representation.
    # The reference semantics is the numpy backend's ``_cast_`` map:
    # float -> float32 values, double -> float64 values (a copying identity
    # here), int/long -> truncated integer values, bool -> nonzero.  ``double``
    # previously shared the narrowing kernel and silently truncated every
    # mantissa to single precision.
    "float": "cast_double_to_float_values",
    "double": "cast_double_to_double_values",
    "bool": "cast_double_to_bool_values",
    "to_dtype": None,
    "astype": None,
    "to": None,
    "long": "cast_double_to_int_values",
    "int": "cast_double_to_int_values",
    "long_cast": "cast_double_to_int_values",
    # Frontend tensor construction is normalization.  Existing tensor spans
    # retain their storage; resident sequences use the sequence descriptor to
    # expose their one arena as a shaped view when dtype already agrees.
    "tensor": None,
}
_REDUCTION_CODES = {"sum": 0, "prod": 1, "min": 2, "max": 3, "any": 4, "all": 5}
_SHAPED_SSA_OPERATIONS = {
    "Indexed": "basic_index", "indexed": "basic_index",
    "IndexedStore": "basic_index_store",
    "index_set": "basic_index_store",
    "MatMul": "matmul", "matmul": "matmul",
    "Add": "add", "add": "add",
    "Sub": "sub", "sub": "sub",
    "Mul": "mul", "Mult": "mul", "mul": "mul",
    "Div": "truediv", "truediv": "truediv",
    # Comparisons are catalogued elementwise kernels too.  Without these
    # spellings a tensor comparison fell through to the scalar emitter, which
    # computed element zero and left the rest of the buffer as the allocation
    # found it -- the same failure the mod/floordiv note below describes.  A
    # scalar comparison is unaffected: the gate still requires an operand
    # with real extents, so loop conditions stay scalar.
    "Eq": "equal", "eq": "equal", "equal": "equal",
    "Ne": "not_equal", "ne": "not_equal", "not_equal": "not_equal",
    "Lt": "less", "lt": "less", "less": "less",
    "Le": "less_equal", "le": "less_equal", "less_equal": "less_equal",
    "Gt": "greater", "gt": "greater", "greater": "greater",
    "Ge": "greater_equal", "ge": "greater_equal",
    "greater_equal": "greater_equal",
    # Tensor ``%`` and ``//`` are catalogued binary kernels (CT_OP_MOD and
    # CT_OP_FLOORDIV).  Without these spellings they fell through to the
    # scalar instruction emitter, which computed element zero only and
    # never filled the result buffer (the periodic terrain's ``uv`` and
    # ``cell_index`` were silently zero).
    "Mod": "mod", "mod": "mod",
    "FloorDiv": "floordiv", "floordiv": "floordiv",
    "Pow": "pow", "pow": "pow",
    "sum": "sum", "mean": "mean", "prod": "prod",
    "min": "min", "max": "max", "any": "any", "all": "all",
    # SymPy's equation graph uses class-style opcodes for elementwise
    # maximum/minimum.  Tensor methods named ``max``/``min`` remain reductions;
    # these spellings are the binary symbolic operators.
    "Max": "maximum", "Min": "minimum",
    "matmul": "matmul", "transpose": "transpose",
    "broadcast_to": "broadcast_to", "expand": "expand",
    "unfold2d": "unfold2d", "fold2d": "fold2d",
}


#: The tensor-contract accounting keys a call edge transports
#: (``enrich``) and therefore the keys the call-metadata fixed point
#: snapshots.  Program: ``reindex``'s ``row_columns.setdefault(row,
#: []).append(column)`` (test_authored_tuple_loop_row_remains_integer_table_
#: key_handle).  Wrong record: the edge carried ``sequence_row_shape`` onto
#: the sequence helpers' formals in round 1, but the snapshot listed the
#: other six keys only, so round 1 equalled round 0 while a pass reported a
#: change and the fixed point refused as a cycle.  Rule: one key list for
#: what the edge writes and what the snapshot reads.
_TENSOR_CONTRACT_KEYS = (
    "program_abi_storage",
    "program_abi_rank",
    "tensor_metadata_state",
    "tensor_shape_value_id",
    "tensor_rank_value_id",
    "tensor_element_count_value_id",
    "sequence_row_shape",
)


def _int64_lane_checked(instruction: Any, operation: str, result: Any) -> bool:
    """An int64 tensor ``add``/``sub``/``mul`` whose lanes must not wrap
    (``python_int_lane_overflow_checked``); ``binary_int64`` cannot check."""

    from .ssa_llvm_backend import python_int_lane_overflow_checked

    return (
        str(operation) in {"add", "sub", "mul"}
        and str(result.dtype or "").casefold() in {"int64", "i64", "long"}
        and python_int_lane_overflow_checked(instruction, operation=operation)
    )


def _int64_binary_kernel(
    operation: str, result: Any, operands: Any, *, owner: Any, book: Any,
) -> bool:
    """Select the integer kernel only for proven integer-backed operands.

    ``compact_codewords`` casts a double-backed mask to logical int64. Its
    cast result is still stored as float64 by ``cast_double_to_int_values``;
    selecting ``binary_int64`` from the semantic dtype alone reinterpreted
    those bytes as integers. The scoped SSA_VALUE is the storage authority.
    """

    from .ssa_record_return_state import posted_ssa_value_fact

    def integer_storage(operand: Any) -> bool:
        physical = posted_ssa_value_fact(owner, int(operand.id), book=book)
        if physical is None:
            return False
        if physical.rank is not None and physical.rank > 0:
            return physical.storage == "span" and physical.physical_dtype == "int64"
        return (
            physical.rank == 0 and physical.storage == "scalar"
            and physical.physical_dtype in {"int32", "int64"}
        )

    return (
        str(operation) in {"add", "sub", "mul", "mod"}
        and str(result.dtype or "").casefold() in {"int64", "i64", "long"}
        and all(
            str(operand.dtype or "").casefold()
            in {"int", "int32", "i32", "int64", "i64", "long"}
            and integer_storage(operand)
            for operand in operands
        )
    )


def _int64_lane_stays_elementwise(
    instruction: Any, operation: str, result: Any, operands: Any,
    *, owner: Any, book: Any,
) -> bool:
    """Whether an int64 tensor op stays the instruction itself.

    Each backend's elementwise span loop then spells it on the integer path.
    Program: ``-tokens`` / ``abs(tokens)`` on an int64 tensor.  Wrong record:
    the only repository unary kernel is ``unary_double``; the lanes were cast
    to float64 (2**53 + 1 came back 2**53) or, after an int64 producer, had
    their bits negated as a double's sign.  And ``tokens * 1103515245``
    through ``binary_int64`` wrapped every lane leaving int64.  Rule: int64
    ``neg``/``abs`` always stay; int64 ``add``/``sub``/``mul`` stay when
    ``python_int_lane_overflow_checked`` asks for per-lane checks and no
    operand needs a broadcast kernel (each is the result's shape or rank 0).
    """

    from .ssa_record_return_state import posted_ssa_value_fact

    operands = list(operands or ())
    def integer_storage(value: Any) -> bool:
        physical = posted_ssa_value_fact(owner, int(value.id), book=book)
        return bool(
            physical is not None
            and physical.physical_dtype == "int64"
            and (physical.rank == 0 and physical.storage == "scalar"
                 or physical.rank is not None and physical.rank > 0
                 and physical.storage == "span")
        )
    operation = str(operation)
    int64 = {"int64", "i64", "long"}
    if (
        operation in {"neg", "abs"}
        and len(operands) == 1
        and str(result.dtype or "").casefold() in int64
        and str(operands[0].dtype or "").casefold() in int64
        and integer_storage(operands[0])
    ):
        return True
    if not _int64_lane_checked(instruction, operation, result):
        return False
    shape = tuple(result.shape or ())
    return bool(shape) and all(
        str(value.dtype or "").casefold() in int64 | {"int", "int32", "i32"}
        and integer_storage(value)
        and (
            tuple(value.shape or ()) == shape
            or (
                not tuple(value.shape or ())
                and int((value.accounting or {}).get(
                    "program_abi_rank", 0
                ) or 0) == 0
            )
        )
        for value in (getattr(instruction, "args", None) or operands)
    )


def _elementwise_result_physical(
    owner: Any, operands: Any, *, book: Any,
) -> dict[str, Any]:
    """The storage and rank of an elementwise kernel result, read from its
    operands' posted ``ssa_value`` rows.

    Program: ``_energy_time_limit``'s ``power > 0.0`` (value 53) compares
    the rank-1 span ``power`` (value 28, posted ``storage='span', rank=1``)
    with a rank-0 literal through ``binary_scalar_double``, whose output is a
    ``double *`` buffer.  Wrong record: the result's row carried only
    ``physical_dtype='float64'``; its rank and storage were never posted, so
    the region ``where(mask, power, ones)`` received the mask as a
    shapeless scalar ``bool`` formal while the caller held a double buffer
    ("incompatible physical call inputs").  Rule: an elementwise result
    broadcasts its operands, so its rank is the largest operand rank and its
    storage is a span exactly when that rank is positive.  Derived only when
    every operand's row records its rank; otherwise nothing is posted and the
    fields stay absent rather than guessed.
    """

    from .ssa_record_return_state import posted_ssa_value_fact

    ranks = []
    for operand in operands:
        fact = posted_ssa_value_fact(owner, int(operand.id), book=book)
        if fact is None or fact.rank is None:
            return {}
        ranks.append(int(fact.rank))
    if not ranks:
        return {}
    rank = max(ranks)
    return {"storage": "span" if rank > 0 else "scalar", "rank": rank}


def _settle_unary_result_metadata(source: SSAValue, result: SSAValue) -> None:
    """Apply the unary kernel's shape law without erasing result semantics."""

    result.shape = tuple(source.shape or ())
    if result.dtype in {None, "unknown"} and source.dtype is not None:
        result.dtype = source.dtype


def _records_same_identity_shape_transport(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    declared_view: Any,
    target_id: int | None = None,
) -> bool:
    """Whether a canonical-id occurrence states that id's own shape.

    Repository lowering represents a storage view with an ``SSAValue`` whose
    integer id is the allocation owner's id while ``ssa_storage_view`` keeps
    the view extents.  The authored reshape still has its own semantic SSA id
    and its real ``owner -> result`` shape edge is published from the graph
    tensor descriptor.  Publishing the collapsed storage occurrence as an
    additional ``owner -> owner`` edge would make two legitimate views revise
    one identity row (for example ``(4, 41)`` and ``(164,)``).
    """

    # Program: a passthrough call rebinds a formal's storage with a
    # ``tensor_descriptor`` view occurrence (storage id 1, view (8,1,1))
    # beside the storage's own occurrences (8,).  Wrong record: the view
    # occurrence was read as the owner edge ``source -> 1`` whenever its
    # shape happened to equal the source's, so one row key carried two
    # target states and REVISE refused it.  Rule: an occurrence that
    # declares ``ssa_storage_view`` of this very storage is a view
    # identity keyed (storage_value_id, view_shape); it never publishes
    # into the owner edge, whatever its shape.  The view's own edge is
    # published from the descriptor-derived view producer.
    if (
        isinstance(declared_view, Mapping)
        and target_id is not None
        and declared_view.get("storage_value_id") is not None
        and int(declared_view["storage_value_id"]) == int(target_id)
    ):
        return False
    declared_view_shape = tuple(
        declared_view.get("view_shape") or ()
    ) if isinstance(declared_view, Mapping) else ()
    return not (
        declared_view_shape
        and tuple(target_shape) == declared_view_shape
        and tuple(target_shape) != tuple(source_shape)
    )


def _shape_polymorphic_function(function_name: str) -> bool:
    """Whether call-edge settlement proved multiple shapes for this helper.

    Read from the book: the ABI settlement posts every proof on
    ``value_shape_polymorphism`` (row (authored function, value), DERIVED
    from the disagreeing ``copy_value_shape`` statements, cross-copy and
    cross-callsite alike).  No label on the ``value_shape`` ledger is read
    (lane A, 2026-10-03)."""

    from .concordance_declarations import VALUE_SHAPE_POLYMORPHISM
    from .identity_concordance import (
        authored_function_name,
        current_identity_book,
    )

    authored_name = authored_function_name(function_name)
    page = current_identity_book().pages.get(VALUE_SHAPE_POLYMORPHISM.name)
    if page is None:
        return False
    return any(
        authored_function_name(row[0]) == authored_name
        for row in page.rows()
    )


def _settle_operand_shapes(
    function: Any, values: Any, descriptor_of: Any = None,
) -> None:
    """Read each operand's shape from the concordance before dispatching.

    An SSAValue's shape is a field somebody filled in at construction, and
    nothing re-read it, so every lowering guard was a race against when that
    annotation happened to be taken -- ``b.unsqueeze(-1)`` proven ``(2, 1)``
    still arrived at its consumer carrying ``(2,)``.  The resolver holds the
    fact for the identity; taking it here makes the guard depend on what is
    known rather than on what was cached.
    """

    from .identity_concordance import proven_shape_of, shape_scope_of
    # The shapes proven for the COPY this function was lowered from; a
    # sibling specialization's proofs live in its own scope.
    scope = shape_scope_of(function)
    shape_polymorphic = _shape_polymorphic_function(function.name)
    for value in values or ():
        # A shaped view shares its storage owner's value id and nothing else.
        # The resolver is keyed by that id, so asking it here answers with the
        # OWNER's shape and restamps every view with it: ``index.unsqueeze(-1)``
        # and ``index.unsqueeze(-2)`` both became plain ``index``, the
        # comparison between them went elementwise instead of broadcasting,
        # and half the (2, 2) result was never written.  The view already
        # states its own extents, and ``ssa_storage_view`` is the declaration
        # that it speaks for itself.
        if (value.accounting or {}).get("ssa_storage_view"):
            continue
        settled = proven_shape_of(scope, int(value.id))
        if settled and tuple(value.shape or ()) != settled:
            # The call-edge concordance has already proved that this helper
            # has no single formal shape.  A nonempty shape on its exact
            # specialized SSA occurrence is therefore more specific than an
            # authored-row proof left by a peer specialization.  Preserve
            # that local fact; the shape.ssa page records what this consumer
            # actually reads.  Missing extents still receive the shared fact.
            if shape_polymorphic and tuple(value.shape or ()):
                continue
            value.shape = settled
        if tuple(value.shape or ()) or descriptor_of is None:
            continue
        # Region extraction can create the consumer SSAValue occurrence
        # before call-edge settlement publishes the exact tensor descriptor
        # for that same source-integral identity.  ``proven_shape`` may then
        # have no row in the consumer region even though the repository tensor
        # table has one unanimous, allocation-owning contract.  Re-read that
        # descriptor before classifying a call operand as scalar.  This is the
        # same identity lookup used by region output/call settlement; shaped
        # storage views above remain authoritative for themselves.
        descriptor = descriptor_of(value)
        descriptor_shape = tuple(
            getattr(descriptor, "shape", ()) or ()
        ) if descriptor is not None else ()
        if descriptor_shape:
            value.shape = descriptor_shape


def _post_scalar_kernel_operand(
    module: IRModule, function: Any, row: tuple, fact: tuple, values: Any,
) -> None:
    """``scalar_kernel_operand_concordance`` CONCORD, DERIVED(the
    ``ssa_value`` cells of the result and the scalar operand);
    ``Unsourced`` when neither has a cell."""

    from .concordance_declarations import (
        SCALAR_KERNEL_OPERAND, TENSOR_SSA_LOWERING,
    )
    from .identity_concordance import Derived, Mode
    from .ssa_record_return_state import ssa_value_identity_cell

    book = identity_book(module)
    cells = tuple(dict.fromkeys(
        cell for cell in (
            ssa_value_identity_cell(function, int(value.id), book=book)
            for value in values
        )
        if cell is not None
    ))
    if not cells:
        raise ValueError(
            f"{SCALAR_KERNEL_OPERAND.name}: no source cell for {row!r} -- "
            "neither the result nor the scalar operand has an ssa_value cell"
        )
    book.post(
        SCALAR_KERNEL_OPERAND, row, fact, stage=TENSOR_SSA_LOWERING,
        provenance=Derived(cells), mode=Mode.CONCORD,
    )


def _concord_scalar_kernel_operand(
    module: IRModule, function: Any, row: tuple, fact: Any, values: Any,
) -> Any:
    """``IdentityPage.concord`` for ``scalar_kernel_operand_concordance``,
    written through the book: the first statement owns the row (posted by
    ``_post_scalar_kernel_operand``), a repeat records its edge, a different
    proposal is a disagreement (``ValueError``).  Returns the committed
    fact, which callers use as the decision."""

    from .concordance_declarations import SCALAR_KERNEL_OPERAND

    stored = identity_book(module).pages.get(SCALAR_KERNEL_OPERAND.name)
    incumbent = None if stored is None else stored.latest(row)
    if incumbent is not None and incumbent != fact:
        raise ValueError(
            f"{SCALAR_KERNEL_OPERAND.name} disagreement for {row!r}: "
            f"recorded={incumbent!r}, proposed={fact!r}"
        )
    _post_scalar_kernel_operand(module, function, row, fact, values)
    return fact


def _scalar_elementwise_execution(module, function, instruction, fact, *, book):
    """Return direct execution proof cells for one scalar/span extent receipt.

    The original raw binary can remain executable (checked int64 lane), or
    tensor lowering can replace it with the selected scalar kernel.  A full
    binary_double Call does not qualify: its second pointer formal cannot
    receive the booked scalar operand.
    """
    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import c_tensor_opcode
    from .concordance_declarations import KERNEL_INPUT_CONVERSION, SCALAR_KERNEL_OPERAND, SSA_VALUE, SSAValueFact
    from .ssa_record_return_state import function_scope_of, posted_ssa_value_fact

    operation, source_id, scalar_id, _shape, _rank, count_id = fact
    signature = c_tensor_opcode(operation)
    if signature is None or signature[0] != "binary" or instruction.res is None:
        return None
    if instruction.op != "Call":
        if (c_tensor_opcode(_SHAPED_SSA_OPERATIONS.get(
                instruction.op, instruction.op)) == signature
                and tuple(int(arg.id) for arg in instruction.args)
                in ((source_id, scalar_id), (scalar_id, source_id))):
            return ()
        return None
    callee = str(instruction.attributes.get("callee") or "")
    physical = posted_ssa_value_fact(function, int(instruction.res.id), book=book)
    kernel_dtype = {"binary_scalar_double": "float64",
                    "binary_scalar_int64": "int64"}.get(callee)
    selected = module.functions.get(callee)
    args = instruction.args
    scalar_physical = (
        posted_ssa_value_fact(function, int(args[1].id), book=book)
        if len(args) == 6 else None
    )
    table = module.tensor_tables.get(function.name)
    result_descriptor = None if table is None else table.by_id(int(instruction.res.id))
    if (kernel_dtype is None or physical is None
            or physical.physical_dtype != kernel_dtype
            or scalar_physical is None
            or scalar_physical.physical_dtype != kernel_dtype
            or scalar_physical.storage != "scalar"
            or scalar_physical.rank != 0
            or selected is None or len(selected.args) != 6
            or selected.metadata.get("ssa_output_argument") != 2
            or len(args) != 6
            or instruction.attributes.get("ssa_output_argument") != 2
            or int(args[0].id) != source_id
            or int(args[3].id) != count_id
            or result_descriptor is None
            or type(result_descriptor.data_value_id) is not int
            or int(result_descriptor.data_value_id) != int(args[2].id)):
        return None
    scope = function_scope_of(function)
    position_ref = book.latest_ref(SCALAR_KERNEL_OPERAND, (
        str(function.name), int(instruction.res.id),
    ))
    position_fact = (None if position_ref is None else book.page(
        SCALAR_KERNEL_OPERAND).cell(position_ref.row, position_ref.column))
    if (not isinstance(position_fact, tuple) or len(position_fact) != 2
            or position_fact[0] != "scalar_operand"
            or position_fact[1] not in (0, 1)):
        return None
    position = position_fact[1]
    position_causes = {cause for cause, _stage in book.edges_into(position_ref)}
    # Check the exact SSA_VALUE cells cited when this Call was selected.
    # Later revisions can change a latest_ref without changing the direct
    # cause of this already published scalar-operand receipt.
    def cited_value(owner, value_id):
        row = (function_scope_of(owner), int(value_id))
        return any(
            cause.page == SSA_VALUE and cause.row == row
            and isinstance(book.page(SSA_VALUE).cell(cause.row, cause.column), SSAValueFact)
            for cause in position_causes
        )

    if (any(not cited_value(function, value.id) for value in (*args, instruction.res))
            or any(not cited_value(selected, selected.args[index].id)
                   for index in (0, 1, 2))):
        return None
    definitions = tuple(
        (block.name, candidate) for block in function.blocks.values()
        for candidate in block.instrs if candidate.res is not None
    )

    def unique_const(value_id):
        matches = tuple(candidate for _block, candidate in definitions
                        if int(candidate.res.id) == int(value_id)
                        and candidate.op == "Const")
        return matches[0].attributes.get("constant") if len(matches) == 1 else None

    if (unique_const(args[4].id) != signature[1]
            or unique_const(args[5].id) != (1 if position == 0 else 0)):
        return None
    if int(args[1].id) == scalar_id:
        return (position_ref,)
    conversion = book.latest_ref(KERNEL_INPUT_CONVERSION, (
        scope, next((block.name for block in function.blocks.values()
                     if any(candidate is instruction for candidate in block.instrs)), ""),
        int(instruction.res.id), position,
    ))
    conversion_fact = (None if conversion is None else book.page(
        KERNEL_INPUT_CONVERSION).cell(conversion.row, conversion.column))
    casts = tuple(candidate for _block, candidate in definitions
                  if int(candidate.res.id) == int(args[1].id)
                  and candidate.op == "Cast")
    if (not isinstance(conversion_fact, tuple) or len(conversion_fact) != 5
            or conversion_fact[0] != scalar_id
            or conversion_fact[1] != int(args[1].id)
            or conversion_fact[3:] != (kernel_dtype, callee)
            or len(casts) != 1 or len(casts[0].args) != 1
            or int(casts[0].args[0].id) != scalar_id
            or casts[0].attributes.get("target_dtype") != kernel_dtype):
        return None
    return (position_ref, conversion)


def _elementwise_extent_execution(module, function, instruction, fact, *, book):
    """Read the live execution of the existing elementwise geometry receipt."""
    if len(fact) == 6:
        return _scalar_elementwise_execution(module, function, instruction, fact, book=book)
    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import c_tensor_opcode
    from .ssa_record_return_state import posted_ssa_value_fact, ssa_value_identity_cell

    operation, source_id, _shape, _rank, count_id = fact
    signature = c_tensor_opcode(operation)
    if signature is None or signature[0] != "unary" or instruction.res is None:
        return None
    source = posted_ssa_value_fact(function, source_id, book=book)
    result = posted_ssa_value_fact(function, int(instruction.res.id), book=book)
    if (source is None or result is None or source.storage != "span"
            or result.storage != "span" or source.rank != result.rank):
        return None
    if instruction.op != "Call":
        return () if (
            source.physical_dtype == result.physical_dtype == "int64"
            and c_tensor_opcode(_SHAPED_SSA_OPERATIONS.get(instruction.op, instruction.op)) == signature
            and tuple(int(value.id) for value in instruction.args) == (source_id,)
        ) else None
    args = instruction.args
    selected = module.functions.get("unary_double")
    table = module.tensor_tables.get(function.name)
    descriptor = None if table is None else table.by_id(int(instruction.res.id))
    if (instruction.attributes.get("callee") != "unary_double"
            or instruction.attributes.get("ssa_output_argument") != 1
            or len(args) != 4 or int(args[0].id) != source_id
            or int(args[2].id) != count_id
            or descriptor is None or int(args[1].id) != descriptor.data_value_id
            or source.physical_dtype != "float64" or result.physical_dtype != "float64"
            or selected is None or len(selected.args) != 4
            or selected.metadata.get("ssa_output_argument") != 1):
        return None
    constants = tuple(candidate for block in function.blocks.values() for candidate in block.instrs
                      if candidate.op == "Const" and candidate.res is not None
                      and int(candidate.res.id) == int(args[3].id))
    if len(constants) != 1 or constants[0].attributes.get("constant") != signature[1]:
        return None
    cells = tuple(ssa_value_identity_cell(owner, int(value.id), book=book)
                  for owner, values in ((function, (*args, instruction.res)),
                                        (selected, tuple(selected.args)))
                  for value in values)
    return tuple(dict.fromkeys(cells)) if all(cell is not None for cell in cells) else None


def _dynamic_tensor_geometry_evidence(module, function, value_id, *, pending=(), _ancestors=()):
    """Read exact local extents or the existing transported metadata binding.

    _ascending_order combines a local scaled input with computed ordinals.
    Requiring both operands' geometry to be returned by an earlier region
    excludes the local span. Its actual extent operations are equally exact
    SSA evidence; no source name, display shape or default extent is needed.
    """
    from .concordance_declarations import (
        BindingKind, CONTROL_VALUE_BINDING, ControlBinding,
        TENSOR_ELEMENTWISE_EXTENT_BINDING,
        TENSOR_METADATA_CALL_BINDING, TENSOR_METADATA_RESULT_BINDING,
        TENSOR_RUNTIME_BROADCAST_VALIDITY,
    )
    from .ssa_record_return_state import (
        function_scope_of, metadata_result_caller_value,
        posted_ssa_value_fact, ssa_value_identity_cell,
    )

    book = identity_book(module)
    if value_id in _ancestors:
        return None
    table = module.tensor_tables.get(function.name)
    descriptor = None if table is None else table.by_id(int(value_id))
    physical = posted_ssa_value_fact(function, int(value_id), book=book)
    if (descriptor is None or descriptor.metadata_state != "dynamic"
            or physical is None or physical.storage != "span"
            or not isinstance(physical.rank, int) or physical.rank <= 0):
        return None
    ids = tuple(getattr(descriptor, field) for field in (
        "shape_value_id", "rank_value_id", "element_count_value_id",
    ))
    if any(type(identity) is not int for identity in ids):
        return None
    values = {int(value.id): value for value in function.args}
    definitions = {}
    for block in function.blocks.values():
        instructions = (*block.instrs, *pending) if len(function.blocks) == 1 else block.instrs
        for index, instruction in enumerate(instructions):
            if instruction.res is not None:
                identity = int(instruction.res.id)
                definitions.setdefault(identity, []).append((block.name, index, instruction))
                values[identity] = instruction.res
    if any(identity not in values for identity in ids):
        return None
    metadata = tuple(values[identity] for identity in ids)
    cells = tuple(ssa_value_identity_cell(function, identity, book=book)
                  for identity in (int(value_id), *ids))
    if any(cell is None for cell in cells):
        return None
    for index, cell in enumerate(cells[1:]):
        fact = book.page(cell.page).cell(cell.row, cell.column)
        if (fact.physical_dtype != "int32"
                or fact.storage != ("span" if index == 0 else "scalar")
                or fact.rank != (1 if index == 0 else 0)):
            return None
    kinds = ("shape", "rank", "element_count")
    broadcast = book.latest_ref(TENSOR_RUNTIME_BROADCAST_VALIDITY, (
        function_scope_of(function), function.name, value_id,
    ))
    if broadcast is not None:
        fact = book.page(broadcast.page).cell(broadcast.row, broadcast.column)
        producers = definitions.get(int(value_id), ())
        if (not isinstance(fact, tuple) or len(fact) != 19
                or tuple(fact[12:15]) != ids or len(producers) != 1):
            return None
        from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import c_tensor_opcode
        instruction = producers[0][2]
        signature = c_tensor_opcode(fact[0])
        raw = (instruction.op != "Call" and physical.physical_dtype == "int64"
               and c_tensor_opcode(_SHAPED_SSA_OPERATIONS.get(instruction.op, instruction.op)) == signature)
        native = (instruction.op == "Call" and physical.physical_dtype == "float64"
                  and instruction.attributes.get("callee") == "binary_double"
                  and instruction.attributes.get("ssa_output_argument") == 2
                  and len(instruction.args) == 5 and int(instruction.args[3].id) == ids[2]
                  and int(instruction.args[2].id) == descriptor.data_value_id)
        if signature is None or signature[0] != "binary" or not (raw or native):
            return None
        return metadata, (*cells, broadcast)
    inherited = book.latest_ref(TENSOR_ELEMENTWISE_EXTENT_BINDING, (
        function_scope_of(function), function.name, value_id,
    ))
    if inherited is not None:
        fact = book.page(inherited.page).cell(inherited.row, inherited.column)
        source_definitions = definitions.get(int(value_id), ())
        if (not isinstance(fact, tuple) or len(fact) not in (5, 6)
                or not isinstance(fact[0], str)
                or any(type(identity) is not int for identity in fact[1:])
                or tuple(fact[-3:]) != ids or len(source_definitions) != 1):
            return None
        instruction = source_definitions[0][2]
        scalar = posted_ssa_value_fact(function, fact[2], book=book) if len(fact) == 6 else None
        execution = _elementwise_extent_execution(
            module, function, instruction, fact, book=book,
        )
        if (execution is None or len(fact) == 6 and (scalar is None
                or scalar.storage != "scalar" or scalar.rank != 0)):
            return None
        evidence = _dynamic_tensor_geometry_evidence(
            module, function, fact[1], pending=pending,
            _ancestors=(*_ancestors, value_id),
        )
        causes = {cause for cause, _ in book.edges_into(inherited)}
        if (evidence is not None and tuple(int(value.id) for value in evidence[0]) == ids
                and set(evidence[1]) <= causes
                and set(execution) <= causes):
            return metadata, (*cells, inherited, *evidence[1], *execution)
        return None
    local = tuple(definitions.get(identity, ()) for identity in ids)
    if all(len(matches) == 1 for matches in local):
        for kind, matches in zip(kinds, local, strict=True):
            instruction = matches[0][2]
            if (str(instruction.attributes.get("tensor_operation") or instruction.op) != "extent"
                    or instruction.attributes.get("extent_kind") != kind
                    or len(instruction.args) != 1 or int(instruction.args[0].id) != value_id):
                return None
        source_definitions = definitions.get(int(value_id), ())
        if len(source_definitions) == 1:
            source_block, source_index, _instruction = source_definitions[0]
            if not all(matches[0][0] == source_block and source_index < matches[0][1]
                       for matches in local):
                return None
            return metadata, cells
        if not any(int(value.id) == value_id for value in function.args):
            return None
        binding = book.latest_ref(CONTROL_VALUE_BINDING, (function_scope_of(function), value_id))
        fact = None if binding is None else book.page(binding.page).cell(binding.row, binding.column)
        if (isinstance(fact, ControlBinding)
                and fact.kind in {BindingKind.PARAMETER_SEED, BindingKind.FIELD_INCUMBENT}
                and fact.ssa_value.row == (function_scope_of(function), value_id)):
            return metadata, (*cells, binding)
        return None
    if not all(any(int(formal.id) == identity for formal in function.args) for identity in ids):
        return None
    if any((value.accounting or {}).get("tensor_extent_for") != value_id
           or (value.accounting or {}).get("tensor_extent_kind") != kind
           for value, kind in zip(metadata, kinds, strict=True)):
        return None
    proofs = []
    for row in book.page(TENSOR_METADATA_CALL_BINDING).rows():
        if row[4] != function.name:
            continue
        binding = book.latest_ref(TENSOR_METADATA_CALL_BINDING, row)
        pairs = tuple(zip(*book.page(binding.page).cell(binding.row, binding.column), strict=True))
        caller = module.functions.get(str(row[1]))
        if caller is None or row[0] != function_scope_of(caller):
            continue
        block = None if caller is None else caller.blocks.get(str(row[2]))
        calls = () if block is None else tuple(instruction for instruction in block.instrs
            if instruction.op == "Call" and instruction.res is not None
            and int(instruction.res.id) == row[3]
            and instruction.attributes.get("callee") == function.name)
        if len(calls) != 1 or len(calls[0].args) != len(function.args):
            continue
        live_pairs = tuple((int(actual.id), int(formal.id)) for actual, formal in zip(
            calls[0].args, function.args, strict=True,
        ))
        if any(pair not in live_pairs for pair in pairs):
            continue
        predecessors = tuple(cause for cause, _stage in book.edges_into(binding)
                             if cause.page == TENSOR_METADATA_RESULT_BINDING
                             and cause.row[0] == row[0])
        projected_producers = []
        for predecessor in predecessors:
            producer_calls = tuple(
                instruction for producer_block in caller.blocks.values()
                for instruction in producer_block.instrs
                if instruction.op == "Call" and instruction.res is not None
                and int(instruction.res.id) == predecessor.row[1]
                and instruction.attributes.get("callee") == predecessor.row[2]
            )
            if len(producer_calls) != 1:
                continue
            projection = metadata_result_caller_value(
                module, caller, producer_calls[0], predecessor, book=book,
            )
            if projection is not None and sum(
                actual_id == projection[0] and formal_id == value_id
                for actual_id, formal_id in live_pairs
            ) == 1:
                projected_producers.append((predecessor, projection[1]))
        if len(projected_producers) != 1:
            continue
        predecessor, projection_cells = projected_producers[0]
        projected = book.page(predecessor.page).cell(
            predecessor.row, predecessor.column,
        )[2]
        if all(pairs.count((actual, formal)) == 1
               for actual, formal in zip(projected, ids, strict=True)):
            proofs.extend((binding, predecessor, *projection_cells))
    return (metadata, tuple(dict.fromkeys((*cells, *proofs)))) if proofs else None


def _post_reduction_domain(
    module: IRModule, function: Any, row: tuple, fact: tuple, values: Any,
    extra_cells: tuple[Any, ...] = (),
) -> None:
    """Post the all-axis or tagged runtime-axis reduction domain.

    CONCORD, DERIVED from the scoped ``ssa_value`` cells of the result,
    source, and each runtime geometry value supplied by the caller.
    """

    from .concordance_declarations import (
        TENSOR_REDUCTION_DOMAIN, TENSOR_SSA_LOWERING,
    )
    from .identity_concordance import Derived, Mode
    from .ssa_record_return_state import ssa_value_identity_cell

    book = identity_book(module)
    cells = tuple(dict.fromkeys((
        *extra_cells,
        *(
        cell for cell in (
            ssa_value_identity_cell(function, int(value.id), book=book)
            for value in values
        )
        if cell is not None
        ),
    )))
    if not cells:
        raise ValueError(
            f"{TENSOR_REDUCTION_DOMAIN.name}: no source cell for {row!r} -- "
            "neither the result nor the reduced source has an ssa_value cell"
        )
    book.post(
        TENSOR_REDUCTION_DOMAIN, row, fact, stage=TENSOR_SSA_LOWERING,
        provenance=Derived(cells), mode=Mode.CONCORD,
    )


def _post_call_edge_fact(
    module: IRModule, page: Any, row: tuple, fact: Any, cells: Any,
) -> None:
    """Post ``fact`` on declared ``page`` REVISE, DERIVED from ``cells``;
    ``Unsourced`` (listed by the audit) when no cell exists or the fact
    changed over the same unchanged cells.  A fact the row already holds is
    not restated."""

    from .concordance_declarations import TENSOR_SSA_LOWERING
    from .identity_concordance import Derived, Mode

    book = identity_book(module)
    stored = book.pages.get(page.name)
    if stored is not None and stored.latest(row) == fact:
        return
    sources = tuple(dict.fromkeys(cell for cell in cells if cell is not None))
    if not sources:
        raise ValueError(
            f"{page.name}: no source cell for {row!r} -- the call edge names "
            "no cell"
        )
    book.post(
        page, row, fact, stage=TENSOR_SSA_LOWERING,
        provenance=Derived(sources), mode=Mode.REVISE,
    )


def _publish_exact_ssa_call_shapes(module: IRModule) -> dict[
    tuple[str, int], list[tuple[tuple[int, ...], str]]
]:
    """Publish current exact actual/formal contracts after metadata settles."""

    from .concordance_declarations import (
        SSA_CALL_SHAPE, SSA_CALL_SHAPE_EVIDENCE,
    )
    from .identity_concordance import record_proven_shape, shape_scope_of
    from .ssa_record_return_state import ssa_value_identity_cell

    contracts: dict[
        tuple[str, int], list[tuple[tuple[int, ...], str]]
    ] = {}
    native_result_shapes_by_owner: dict[
        str, dict[int, tuple[tuple[int, ...], str]]
    ] = {}
    for caller_name, caller in module.functions.items():
        region_integral = dict(
            (caller.metadata or {}).get("source_region_integral") or {}
        )
        owner_name = str(region_integral.get("owner") or caller_name)
        native_result_shapes = native_result_shapes_by_owner.setdefault(
            owner_name, {}
        )
        for caller_block in caller.blocks.values():
            for producer in caller_block.instrs:
                native_contract = tuple(
                    producer.attributes.get("native_result_contract", ())
                )
                if producer.op not in {"Call", "call"}:
                    continue

                def retain_native_result_shape(
                    output_id: int, contract_row: tuple,
                ) -> None:
                    if (
                        len(contract_row) < 3
                        or not tuple(contract_row[2] or ())
                    ):
                        return
                    if not contract_row[1]:
                        raise ValueError(
                            "native call result contract for owner "
                            f"{owner_name!r} value {int(output_id)} carries "
                            f"shape {tuple(contract_row[2])} but no dtype"
                        )
                    proposed = (
                        tuple(contract_row[2]),
                        str(contract_row[1]),
                    )
                    incumbent = native_result_shapes.get(int(output_id))
                    if incumbent is not None and incumbent != proposed:
                        raise ValueError(
                            "native call result shape disagreement for "
                            f"owner {owner_name!r} value {int(output_id)}: "
                            f"recorded={incumbent!r}, proposed={proposed!r}"
                        )
                    native_result_shapes[int(output_id)] = proposed

                if producer.res is not None and len(native_contract) == 1:
                    retain_native_result_shape(
                        int(producer.res.id), native_contract[0],
                    )

                output_ids = tuple(map(
                    int, producer.attributes.get("output_ids", ())
                ))
                output_positions = tuple(map(
                    int, producer.attributes.get(
                        "output_positions", range(len(output_ids)),
                    )
                ))
                if len(output_ids) != len(output_positions):
                    continue
                for output_id, output_position in zip(
                    output_ids, output_positions, strict=True,
                ):
                    if 0 <= output_position < len(native_contract):
                        retain_native_result_shape(
                            output_id, native_contract[output_position],
                        )

    for caller_name, caller in module.functions.items():
        region_integral = dict(
            (caller.metadata or {}).get("source_region_integral") or {}
        )
        owner_name = str(region_integral.get("owner") or caller_name)
        native_result_shapes = native_result_shapes_by_owner.get(
            owner_name, {}
        )
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if instruction.op not in {"Call", "call"}:
                    continue
                callee_name = str(instruction.attributes.get("callee") or "")
                callee = module.functions.get(callee_name)
                if callee is None or len(instruction.args) != len(callee.args):
                    continue
                for actual, formal in zip(instruction.args, callee.args):
                    native_result = native_result_shapes.get(int(actual.id))
                    descriptor = getattr(module, "tensor_tables", {}).get(
                        str(caller_name), SSATensorTable()
                    ).by_id(int(actual.id))
                    explicit_view = (actual.accounting or {}).get(
                        "ssa_storage_view"
                    )
                    # compact_codewords: the Mul feeding sum has a display
                    # shape (1, 8), but its producer descriptor has runtime
                    # shape/rank/count. An SSA occurrence alone cannot state
                    # full extents for the next region's formal. Wait for the
                    # existing computed-metadata Call binding to carry them.
                    if (native_result is None and not explicit_view
                            and (descriptor is None
                                 or descriptor.metadata_state != "static")):
                        continue
                    extents = tuple(
                        native_result[0]
                        if native_result is not None and not explicit_view
                        else actual.shape or ()
                        if explicit_view
                        else (
                            tuple(descriptor.shape or ())
                            if descriptor is not None
                            and descriptor.metadata_state == "static"
                            and tuple(descriptor.shape or ())
                            else tuple(actual.shape or ())
                        )
                    )
                    if not extents:
                        continue
                    fact_dtype = (
                        (None if native_result is None else native_result[1])
                        or actual.dtype
                        or (None if descriptor is None else descriptor.dtype)
                    )
                    if not fact_dtype:
                        raise ValueError(
                            f"callsite {caller_name!r} -> {callee_name!r}: "
                            f"actual %t{int(actual.id)} for formal "
                            f"%t{int(formal.id)} has extents {extents} but "
                            "no recorded dtype (native result, actual and "
                            "descriptor all lack one)"
                        )
                    fact = (extents, str(fact_dtype))
                    row = (
                        callee_name, int(formal.id),
                        str(caller_name), int(actual.id),
                    )
                    edge_cells = (
                        ssa_value_identity_cell(
                            caller, int(actual.id), book=identity_book(module),
                        ),
                        ssa_value_identity_cell(
                            callee, int(formal.id), book=identity_book(module),
                        ),
                    )
                    _post_call_edge_fact(module, SSA_CALL_SHAPE, row, fact, edge_cells)
                    evidence_page = identity_book(module).page(
                        "ssa_call_shape_evidence"
                    )
                    evidence = (
                        "native_result_contract"
                        if native_result is not None and not explicit_view
                        else "explicit_storage_view"
                        if explicit_view
                        else "resident_descriptor"
                        if descriptor is not None
                        and descriptor.metadata_state == "static"
                        and tuple(descriptor.shape or ())
                        else "ssa_occurrence"
                    )
                    _post_call_edge_fact(
                        module, SSA_CALL_SHAPE_EVIDENCE, row, evidence,
                        edge_cells,
                    )
                    # The formal's proven extents ARE this call edge's fact:
                    # the proof derives from the ``ssa_call_shape`` cell just
                    # posted and the actual's / formal's ``ssa_value`` cells.
                    record_proven_shape(
                        shape_scope_of(callee), int(formal.id), extents,
                        actual.dtype,
                        sources=(
                            identity_book(module).latest_ref(
                                SSA_CALL_SHAPE, row,
                            ),
                            *edge_cells,
                        ),
                    )
                    contracts.setdefault(
                        (callee_name, int(formal.id)), []
                    ).append(fact)
    return contracts


def _settle_exact_ssa_call_shapes(
    module: IRModule,
    contracts: Mapping[
        tuple[str, int], list[tuple[tuple[int, ...], str]]
    ],
) -> None:
    """Apply only unanimous exact call-edge facts, retaining their receipt."""

    for (callee_name, formal_id), candidates in contracts.items():
        exact = set(candidates)
        if len(exact) != 1:
            continue
        extents, dtype = next(iter(exact))
        callee = module.functions[callee_name]
        occurrences = [*callee.args]
        for block in callee.blocks.values():
            for instruction in block.instrs:
                occurrences.extend(instruction.args)
                if instruction.res is not None:
                    occurrences.append(instruction.res)
        for occurrence in occurrences:
            if (
                int(occurrence.id) != int(formal_id)
                or (occurrence.accounting or {}).get("ssa_storage_view")
            ):
                continue
            occurrence.shape = tuple(extents)
            if occurrence.dtype is None:
                occurrence.dtype = dtype
            occurrence.accounting = {
                **dict(occurrence.accounting or {}),
                "shape_settlement_provenance": (
                    "unanimous_exact_ssa_call_edges",
                    callee_name,
                    int(formal_id),
                ),
                "shape_settlement_tie_policy": "exact-call-edge",
            }


def _record_ssa_shape(function: Any, value: Any) -> None:
    """What the lowering actually reads, under the shape scope of the copy
    ``function`` was lowered from -- the key every other shape store uses, so
    a value whose stores disagree is one row."""

    from .concordance_declarations import SHAPE_SSA_STORE, TENSOR_SSA_LOWERING
    from .identity_concordance import (
        Derived, Mode, current_identity_book, shape_scope_of,
    )
    from .ssa_record_return_state import ssa_value_identity_cell

    extents = tuple(int(e) for e in (value.shape or ()))
    if not extents:
        return
    book = current_identity_book()
    cell = ssa_value_identity_cell(function, int(value.id), book=book)
    if cell is None:
        raise ValueError(
            f"{SHAPE_SSA_STORE.name}: no ssa_value cell for value "
            f"{int(value.id)} in {function.name!r}; the shape is not posted "
            "from a fact the book holds"
        )
    book.post(
        SHAPE_SSA_STORE, (shape_scope_of(function), int(value.id)), extents,
        stage=TENSOR_SSA_LOWERING, provenance=Derived((cell,)),
        mode=Mode.REVISE,
    )


def _used_value_ids(module: IRModule) -> set[int]:
    return {
        int(value.id)
        for function in module.functions.values()
        for value in function.args
    } | {
        int(value.id)
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        for value in (*instruction.args, instruction.res)
        if value is not None
    }


def _constant_payload(instruction: Instr) -> Any:
    if instruction.op != Handler.Const.value:
        return None
    attributes = instruction.attributes
    if attributes.get("values") is not None:
        # The whole-object compiler spells a scalar constant with the same
        # "values" key a vector constant uses; a scalar payload stays scalar.
        payload = attributes["values"]
        sequence = _as_sequence(payload)
        return sequence if sequence is not None else payload
    for key in ("constant", "value", "data"):
        if key in attributes and attributes[key] is not None:
            return attributes[key]
    return None


def _as_sequence(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return None


def _known_count(value: SSAValue, *, physical=None) -> int | None:
    shape = tuple(value.shape)
    if shape:
        if any(int(extent) < 0 for extent in shape):
            return None
        if (physical is not None and physical.rank is not None
                and physical.rank != len(shape)):
            return None
        return prod(int(extent) for extent in shape)
    if physical is not None:
        # _code_lengths' octet indices are int64/span/rank1 with display
        # shape (). Their count is runtime geometry, not one scalar.
        return 1 if physical.storage == "scalar" and physical.rank == 0 else None
    # A typed rank-zero value is a real scalar. An untyped rank-zero value is
    # the whole-object compiler's current spelling for an extent not yet known.
    return 1 if value.dtype is not None else None


def _booked_count(function: Function, value: SSAValue, book) -> int | None:
    """Read a static count only from this function's posted physical view."""
    from .ssa_record_return_state import posted_ssa_value_fact

    physical = posted_ssa_value_fact(function, int(value.id), book=book)
    if physical is None and not tuple(value.shape or ()):
        return None
    return _known_count(value, physical=physical)


def _attribute(attributes: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in attributes and attributes[name] is not None:
            return attributes[name]
    return None


def settle_static_repository_view_shapes(module: IRModule) -> bool:
    """Resolve literal reshape/view extents before region ABI propagation.

    View lowering aliases storage and therefore normally computes its result
    descriptor while rewriting the instruction.  A later numerical region may
    consume that view, however, so the descriptor has to cross the call edge
    before kernel selection starts.  This pass evaluates only the already-SSA
    literal shape operand; it does not execute Python or infer data values.
    """

    changed = False
    for function in module.functions.values():
        constants = {
            int(instruction.res.id): payload
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
            and (payload := _constant_payload(instruction)) is not None
        }
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is None or len(instruction.args) < 2:
                    continue
                operation = str(
                    instruction.attributes.get("tensor_operation")
                    or instruction.attributes.get("tensor")
                    or instruction.attributes.get("tensor_candidate")
                    or instruction.op
                ).casefold()
                if operation not in {"reshape", "view"}:
                    continue
                requested = _as_sequence(constants.get(
                    int(instruction.args[1].id)
                ))
                if requested is None:
                    continue
                resolved = [int(extent) for extent in requested]
                inferred = [
                    index for index, extent in enumerate(resolved)
                    if extent == -1
                ]
                if len(inferred) > 1 or any(
                    extent <= 0 and extent != -1 for extent in resolved
                ):
                    continue
                source_count = _booked_count(
                    function, instruction.args[0], identity_book(module),
                )
                known_count = prod(
                    extent for extent in resolved if extent != -1
                )
                if inferred:
                    if (
                        source_count is None or not known_count
                        or source_count % known_count
                    ):
                        continue
                    resolved[inferred[0]] = source_count // known_count
                result_shape = tuple(resolved)
                if tuple(instruction.res.shape) != result_shape:
                    instruction.res.shape = result_shape
                    changed = True
    return changed


def settle_canonical_value_metadata(module: IRModule) -> bool:
    """Settle one canonical value id from its exact or unanimous writer."""

    values: dict[tuple[str, int], list[SSAValue]] = {}
    for function in module.functions.values():
        region_receipt = function.metadata.get("source_region_integral") or {}
        owner = str(region_receipt.get("owner") or function.name)
        for value in function.args:
            values.setdefault((owner, int(value.id)), []).append(value)
        for block in function.blocks.values():
            for instruction in block.instrs:
                for value in (*instruction.args, instruction.res):
                    if value is not None:
                        values.setdefault(
                            (owner, int(value.id)), []
                        ).append(value)
    changed = False
    excluded = {"ssa.aggregate", "ptr", "pointer", "ptrptr_float64"}
    for identity, occurrences in values.items():
        numerical = [
            value for value in occurrences
            if str(value.dtype or "") not in excluded
            # The generated VJP's bw_matmul reshape has storage id 81 but
            # view shape (2, 2).  Its view is not a writer of storage 81's
            # shape, including while that storage's shape is unresolved.
            and not (value.accounting or {}).get("ssa_storage_view")
        ]
        exact_writer_shapes = {
            tuple(value.shape)
            for value in numerical
            if tuple(value.shape)
            and tuple((value.accounting or {}).get(
                "shape_settlement_provenance", ()
            ) or ())[:1] == ("shape_preserving_ssa_edge",)
        }
        shapes = {
            tuple(value.shape) for value in numerical if tuple(value.shape)
        }
        if len(exact_writer_shapes) == 1:
            shape = next(iter(exact_writer_shapes))
            provenance = "canonical_identity_exact_writer"
        elif len(shapes) == 1:
            shape = next(iter(shapes))
            provenance = "canonical_identity_unanimous"
        else:
            continue
        dtype = next((
            value.dtype for value in numerical
            if tuple(value.shape) == shape and value.dtype is not None
        ), None)
        for value in numerical:
            if (
                tuple(value.shape or ()) != shape
                and not (value.accounting or {}).get("ssa_storage_view")
            ):
                value.shape = shape
                if value.dtype is None and dtype is not None:
                    value.dtype = dtype
                value.accounting = {
                    **dict(value.accounting or {}),
                    "shape_settlement_provenance": (
                        provenance,
                        identity[0],
                        identity[1],
                    ),
                    "shape_settlement_tie_policy": "incumbent",
                }
                changed = True
    return changed


def settle_repository_ssa_static_extent_operands(module: IRModule) -> bool:
    """Restamp explicit kernel extents after whole-module ABI settlement.

    Tensor lowering materializes shape/rank operands as ordinary constants.
    A producer's exact allocation contract can settle later, when all region
    descriptors coexist.  Keep those already-lowered operands synchronized so
    backends never observe an obsolete view shape.
    """

    changed = False
    for function_name, function in module.functions.items():
        table = getattr(module, "tensor_tables", {}).get(function_name)
        if table is None:
            continue
        definitions: dict[int, list[Instr]] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op == "Const" and instruction.res is not None:
                    definitions.setdefault(
                        int(instruction.res.id), []
                    ).append(instruction)

        def stamp(value: SSAValue, payload: Any, *, vector: bool) -> None:
            nonlocal changed
            for definition in definitions.get(int(value.id), ()):
                attributes = dict(definition.attributes)
                if vector:
                    wanted = tuple(map(int, payload))
                    if tuple(attributes.get("values") or ()) == wanted:
                        continue
                    attributes["values"] = wanted
                    attributes["constant"] = None
                else:
                    wanted = int(payload)
                    if attributes.get("constant") == wanted:
                        continue
                    attributes["constant"] = wanted
                definition.attributes = attributes
                changed = True

        def exact_feed_count(value: SSAValue) -> int | None:
            """Count the actual crossing this region's recorded call edge.

            A scalar actual can acquire the destination span's shape on its
            callee-local occurrence.  That view is useful to the indexed-store
            kernel but does not enlarge the actual's storage.  Conversely, a
            linked tensor result may settle from rank zero to its true span
            only after this kernel was first lowered.  The exact feed receipt
            distinguishes those cases without choosing a local incumbent.
            """

            receipt = dict(value.accounting or {}).get(
                "exact_region_feed_source"
            )
            if isinstance(receipt, (tuple, list)) and len(receipt) == 3:
                caller_name, actual_id, argument_index = receipt
                caller = module.functions.get(str(caller_name))
                counts: set[int] = set()
                if caller is not None:
                    for caller_block in caller.blocks.values():
                        for call_instruction in caller_block.instrs:
                            if (
                                call_instruction.op not in {"Call", "call"}
                                or call_instruction.attributes.get("callee")
                                != function_name
                            ):
                                continue
                            position = int(argument_index)
                            if position >= len(call_instruction.args):
                                continue
                            actual = call_instruction.args[position]
                            if int(actual.id) != int(actual_id):
                                continue
                            count = _booked_count(
                                caller, actual, identity_book(module),
                            )
                            if count is not None:
                                counts.add(count)
                if len(counts) == 1:
                    return next(iter(counts))
            return _booked_count(
                function, value, identity_book(module),
            )

        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op != "Call":
                    continue
                callee = instruction.attributes.get("callee")
                if callee == "broadcast_double" and len(instruction.args) >= 6:
                    source_shape = tuple(instruction.args[0].shape or ())
                    output_shape = tuple(instruction.args[1].shape or ())
                    # Exact SSA occurrences, not canonical integer ids, own
                    # view shape. Consulting the table first would replace
                    # legitimate reshape/broadcast views with their allocation
                    # owner's shape.
                    if source_shape:
                        stamp(instruction.args[2], source_shape, vector=True)
                        stamp(
                            instruction.args[3], len(source_shape), vector=False
                        )
                    if output_shape:
                        stamp(instruction.args[4], output_shape, vector=True)
                        stamp(
                            instruction.args[5], len(output_shape), vector=False
                        )
                    continue
                if callee not in {"index_assign_double", "index_set_double"}:
                    continue
                inplace = callee == "index_assign_double"
                minimum_args = 7 if inplace else 8
                if len(instruction.args) < minimum_args:
                    continue
                source_index = 0
                shape_index = 1 if inplace else 2
                rank_index = 2 if inplace else 3
                value_index = 5 if inplace else 6
                count_index = 6 if inplace else 7
                source_shape = tuple(
                    instruction.args[source_index].shape or ()
                )
                if source_shape:
                    stamp(
                        instruction.args[shape_index], source_shape, vector=True
                    )
                    stamp(
                        instruction.args[rank_index],
                        len(source_shape),
                        vector=False,
                    )
                value_count = exact_feed_count(
                    instruction.args[value_index]
                )
                if value_count is not None:
                    stamp(
                        instruction.args[count_index],
                        value_count,
                        vector=False,
                    )
    return changed


def settle_shape_preserving_value_metadata(module: IRModule) -> bool:
    """Carry settled extents through exact shape-preserving SSA edges.

    Control SSA is built before source-call frames finish linking.  A loop
    seed can consequently be rank-unknown when its Phi/Cast is created and
    become shaped only when the linked callee output is installed later.
    These operators are equality edges for extents: Cast changes dtype, and
    Phi selects one of its incoming values. A Phi refines tensor geometry
    only when every arm has the same proven static tensor extent and its
    incoming control edge is booked. Sequence carrier cell shapes are not
    their logical lengths. Disagreement stays unresolved.

    This walks the SSA graph directly.  It creates no parallel identity map;
    the values on the instructions and the shared proven-shape page remain
    the facts consumed by subsequent passes.
    """

    from .bounded_fixed_point import BoundedFixedPoint, bound_for

    # Each changing round refines at least one still-unshaped result, so the
    # number of instruction results bounds the rounds.
    guard = BoundedFixedPoint(
        "shape-preserving-settlement",
        bound_for(
            "settlement",
            functions=len(module.functions),
            nodes=sum(
                len(block.instrs)
                for function in module.functions.values()
                for block in function.blocks.values()
            ),
        ),
        scope=("whole-program",),
    )
    changed_any = False
    changed = True
    while changed:
        changed = False
        for function_name, function in module.functions.items():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    result = instruction.res
                    if result is None:
                        continue
                    operation = str(instruction.op).casefold()
                    descriptor_operation = None
                    if operation in {"call"}:
                        descriptor_operation = str(
                            instruction.attributes.get("callee") or ""
                        ).casefold()
                    if descriptor_operation == "binary_double":
                        sources = tuple(instruction.args[:2])
                        shaped = tuple(
                            tuple(source.shape or ())
                            for source in sources
                            if tuple(source.shape or ())
                        )
                        if not shaped:
                            continue
                        rank = max(map(len, shaped))
                        padded = tuple(
                            (1,) * (rank - len(shape)) + shape
                            for shape in shaped
                        )
                        broadcast = []
                        for extents in zip(*padded):
                            nonunit = {int(extent) for extent in extents
                                       if int(extent) != 1}
                            if len(nonunit) > 1:
                                broadcast = []
                                break
                            broadcast.append(
                                next(iter(nonunit)) if nonunit else 1
                            )
                        if not broadcast:
                            continue
                        shapes = {tuple(broadcast)}
                    elif descriptor_operation in {
                        "binary_scalar_double", "unary_double",
                    }:
                        sources = tuple(instruction.args[:1])
                        shapes = {
                            tuple(source.shape)
                            for source in sources
                            if tuple(source.shape or ())
                        }
                    elif operation in {"cast", "castlike"}:
                        sources = tuple(instruction.args[:1])
                        shapes = {
                            tuple(source.shape)
                            for source in sources
                            if tuple(source.shape or ())
                        }
                    elif operation == "phi":
                        sources = tuple(instruction.args)
                        # encode_octets' return Phi joins two byte-sequence
                        # arenas.  One arm's ptr-cell shape (1,) is not the
                        # logical extent of the other arena.  Only a closed
                        # set of static tensor descriptors can establish a
                        # tensor shape across a Phi; sequence return storage
                        # is owned by its sequence merge protocol.
                        sequence_table = getattr(
                            module, "sequence_tables", {}
                        ).get(function_name)
                        if sequence_table is not None and any(
                            sequence_table.by_id(int(source.id)) is not None
                            for source in sources
                        ):
                            continue
                        tensor_table = getattr(
                            module, "tensor_tables", {}
                        ).get(function_name)
                        descriptors = tuple(
                            None if tensor_table is None else tensor_table.by_id(
                                int(source.id)
                            ) for source in sources
                        )
                        if not sources or any(
                            descriptor is None
                            or descriptor.metadata_state != "static"
                            or not tuple(source.shape or ())
                            or tuple(descriptor.shape or ()) != tuple(source.shape)
                            for source, descriptor in zip(
                                sources, descriptors, strict=True,
                            )
                        ):
                            continue
                        arm_dtypes = {
                            str(descriptor.dtype) for descriptor in descriptors
                        }
                        if len(arm_dtypes) != 1 or str(result.dtype) not in arm_dtypes:
                            continue
                        shapes = {
                            tuple(source.shape)
                            for source in sources
                            if tuple(source.shape or ())
                        }
                        if len(shapes) > 1:
                            initial_id = instruction.attributes.get(
                                "initial_value_id"
                            )
                            initial_shapes = {
                                tuple(source.shape)
                                for source in sources
                                if initial_id is not None
                                and int(source.id) == int(initial_id)
                                and tuple(source.shape or ())
                            }
                            if len(initial_shapes) == 1:
                                # A loop-carried Phi's authored initial edge
                                # anchors the recurrence.  Its update can only
                                # acquire a shape after the Phi itself has one;
                                # requiring unanimity first leaves the cycle
                                # stuck on a provisional update extent.
                                shapes = initial_shapes
                    else:
                        continue
                    if len(shapes) != 1:
                        continue
                    shape = next(iter(shapes))
                    if tuple(result.shape or ()) == shape:
                        continue
                    from .ssa_record_return_state import (
                        posted_ssa_value_fact, publish_ssa_value_type,
                        ssa_block_identity_cell, ssa_value_identity_cell,
                    )
                    from .concordance_declarations import TENSOR_SSA_LOWERING
                    book = identity_book(module)
                    source_cells = tuple(
                        ssa_value_identity_cell(
                            function, int(source.id), book=book,
                        ) for source in sources
                    )
                    if not source_cells or any(cell is None for cell in source_cells):
                        raise ValueError(
                            "shape-preserving SSA edge has no exact operand cells"
                        )
                    if operation == "phi":
                        # A Phi is control-owned: its operands alone do not
                        # prove which arm reaches the result.  Each incoming
                        # block must have its exact SSA control cell.
                        incoming = tuple(instruction.attributes.get(
                            "incoming_blocks", (),
                        ))
                        if len(incoming) != len(sources):
                            raise ValueError(
                                "tensor Phi shape has no complete incoming blocks"
                            )
                        block_cells = tuple(
                            ssa_block_identity_cell(function, name, book=book)
                            for name in incoming
                        )
                        if any(cell is None for cell in block_cells):
                            raise ValueError(
                                "tensor Phi shape has no exact control cells"
                            )
                        from .identity_concordance import (
                            PROVEN_SHAPE_PAGE, shape_scope_of,
                        )
                        proof_cells = tuple(
                            book.latest_ref(PROVEN_SHAPE_PAGE, (
                                shape_scope_of(function), int(source.id),
                            )) for source in sources
                        )
                        if any(cell is None for cell in proof_cells):
                            continue
                        if any(
                            (proof := book.page(PROVEN_SHAPE_PAGE).cell(
                                cell.row, cell.column,
                            ))[0] != "proven"
                            or tuple(proof[1]) != tuple(source.shape)
                            or str(proof[2]) != str(source.dtype)
                            for source, cell in zip(
                                sources, proof_cells, strict=True,
                            )
                        ):
                            continue
                        if any(
                            (fact := posted_ssa_value_fact(
                                function, int(source.id), book=book,
                            )) is None
                            or fact.shape != tuple(source.shape)
                            for source in sources
                        ):
                            continue
                        source_cells = (
                            *source_cells, *proof_cells, *block_cells,
                        )
                    publish_ssa_value_type(
                        function, result, dtype=result.dtype,
                        shape=shape, sources=source_cells,
                        stage=TENSOR_SSA_LOWERING, book=book,
                    )
                    result.accounting = {
                        **dict(result.accounting or {}),
                        "shape_settlement_provenance": (
                            "shape_preserving_ssa_edge",
                            str(function_name),
                            str(instruction.op),
                        ),
                        "shape_settlement_tie_policy": "unanimous",
                    }
                    from .identity_concordance import (
                        record_proven_shape,
                        shape_scope_of,
                    )

                    record_proven_shape(
                        shape_scope_of(function), int(result.id), shape,
                        result.dtype or next((
                            source.dtype for source in sources
                            if source.dtype is not None
                        ), None),
                        None, sources=source_cells,
                    )
                    table = getattr(module, "tensor_tables", {}).get(
                        function_name
                    )
                    descriptor = (
                        table.by_id(int(result.id))
                        if table is not None else None
                    )
                    if descriptor is not None and tuple(
                        descriptor.shape or ()
                    ) != shape:
                        stride = 1
                        reverse_strides = []
                        for extent in reversed(shape):
                            reverse_strides.append(stride)
                            stride *= int(extent)
                        dtype_bytes = dtype_byte_size(
                            str(result.dtype or descriptor.dtype)
                        )
                        table.tensors[int(result.id)] = dataclasses.replace(
                            descriptor,
                            dtype=str(result.dtype or descriptor.dtype),
                            shape=shape,
                            strides=tuple(reversed(reverse_strides)),
                            byte_size=prod(shape) * dtype_bytes,
                            metadata_state="static",
                        )
                    changed = True
                    changed_any = True
        if settle_canonical_value_metadata(module):
            changed = True
            changed_any = True
        guard.round(changed)
    return changed_any


def _booked_static_view_rebinding(
    book, function, value_id: int, source_owner, source_id: int,
    view_shape: tuple[int, ...],
) -> bool:
    """Whether an aggregate fold binds this exact view to the new storage."""
    from .concordance_declarations import (
        AGGREGATE_PASSTHROUGH_REBINDING, SSA_VALUE,
    )
    from .ssa_record_return_state import (
        function_scope_of, ssa_value_identity_cell,
    )

    ref = book.latest_ref(
        AGGREGATE_PASSTHROUGH_REBINDING,
        (function_scope_of(function), int(value_id)),
    )
    if ref is None:
        return False
    fact = book.page(AGGREGATE_PASSTHROUGH_REBINDING).cell(
        ref.row, ref.column,
    )
    if (len(fact) != 4 or int(fact[0]) != int(source_id)
            or int(fact[2]) != int(value_id)
            or tuple(fact[3]) != tuple(view_shape)):
        return False
    source_cell = ssa_value_identity_cell(
        source_owner, int(source_id), book=book,
    )
    target_cell = ssa_value_identity_cell(
        function, int(value_id), book=book,
    )
    if source_cell is None or target_cell is None:
        return False
    causes = {parent for parent, _stage in book.edges_into(ref)}
    return all(any(
        parent.page == SSA_VALUE and parent.row == cell.row
        for parent in causes
    ) for cell in (source_cell, target_cell))


def _matching_elementwise_metadata_cause(
    book: Any, function: Any, value_id: int,
    accounting: Mapping[str, Any],
) -> Any:
    """Read this result's booked runtime extent, never another value's."""

    from .concordance_declarations import TENSOR_ELEMENTWISE_EXTENT_BINDING
    from .ssa_record_return_state import function_scope_of

    if accounting.get("tensor_metadata_state") != "dynamic":
        return None
    row = (function_scope_of(function), function.name, int(value_id))
    cell = book.latest_ref(TENSOR_ELEMENTWISE_EXTENT_BINDING, row)
    if cell is None:
        return None
    fact = book.page(TENSOR_ELEMENTWISE_EXTENT_BINDING).cell(
        cell.row, cell.column,
    )
    metadata = tuple(accounting.get(key) for key in (
        "tensor_shape_value_id", "tensor_rank_value_id",
        "tensor_element_count_value_id",
    ))
    return (
        cell if isinstance(fact, tuple) and len(fact) in (5, 6)
        and all(type(value_id) is int for value_id in metadata)
        and tuple(fact[-3:]) == metadata else None
    )


def propagate_repository_ssa_call_metadata(
    module: IRModule, *, authoritative_returns: bool = False,
) -> bool:
    """Settle tensor dtype/shape facts across repository-SSA call edges.

    Planned regions return through explicit aggregate projections while source
    functions use ordinary ``Ret`` values.  This pass correlates both forms,
    including an aggregate forwarded into a projection-only adapter.  It only
    copies already-proven SSA metadata; it executes no source code and invents
    no extents.
    """

    # A composed module may be inspected after the planning context ended.
    # Its attached book still owns local rows, and accepted components retain
    # separate owner books.  Resume the module transaction before any shape
    # post or imported accepted-cell receipt; otherwise those writes land in
    # an unrelated ambient book and source cells appear absent.
    from .identity_concordance import (
        begin_identity_book, current_identity_book, end_identity_book,
    )

    module_book = identity_book(module)
    if current_identity_book() is not module_book:
        _book, token = begin_identity_book(module_book)
        try:
            return propagate_repository_ssa_call_metadata(
                module, authoritative_returns=authoritative_returns,
            )
        finally:
            end_identity_book(token)

    changed_any = False

    # The SSA instruction topology is immutable throughout this metadata
    # fixed point.  Only ``SSAValue.shape``, ``dtype``, and ``accounting`` are
    # enriched.  The old implementation nevertheless rebuilt whole-function
    # value, constant, return, and aggregate-projection indices for every
    # call edge on every iteration.  A training motion has many calls into
    # the same planned regions, so that made propagation a repeated
    # all-module scan rather than a metadata fixed point.
    #
    # Cache references to the live SSAValue objects, not copies of their
    # metadata.  Enrichment therefore remains immediately visible through
    # every cached index and the convergence semantics are unchanged.
    value_cache: dict[int, tuple[SSAValue, ...]] = {}
    values_by_id_cache: dict[int, dict[int, tuple[SSAValue, ...]]] = {}
    constant_result_ids_cache: dict[int, frozenset[int]] = {}
    constant_indices_cache: dict[int, dict[int, int]] = {}
    returned_cache: dict[int, tuple[SSAValue, ...]] = {}
    projection_cache: dict[tuple[int, int], dict[int, SSAValue]] = {}
    function_names_by_id: dict[int, str] = {
        id(fn): name for name, fn in module.functions.items()
    }
    # A round-boundary snapshot (fixed_point_state) can only ever show a
    # difference when the whole round's net effect differs from the last one
    # seen.  Two authoritative writers disagreeing about one value WITHIN a
    # single round -- write A, then write B that reverts it -- nets to zero
    # by the time the round-end snapshot is taken, even though `changed` was
    # honestly set twice.  An IdentityPage sees it anyway: row = the exact
    # (function, value id, field) written, column = the round it was written
    # in, so a row that revisits an earlier fact is a genuine round-trip,
    # found by reading straight across a row instead of diffing round
    # boundaries.  Taken from the module's shared IdentityBook, not a
    # standalone page, so a value's argument-binding history (written by
    # fortran_c_shell.py) and its shape-enrichment history (written here)
    # sit on the same book and can be read against each other by anything
    # that shares the same row-key convention.
    enrichment_book = identity_book(module)
    mutation_page = enrichment_book.page("tensor_shape_enrichment")
    # Two writes can land in the SAME round (the fill, then the revert that
    # produced tonight's paradox both did) -- round alone is not a unique
    # label, so pair it with a monotonic step so every individual write is
    # its own revision instead of the second silently overwriting the
    # first.  The round is the primary label; it rides along in the fact.
    import itertools
    mutation_step = itertools.count()

    def _log_enrichment(
        function: Any, function_name: Any, value_id: int, field: str,
        old: Any, new: Any, authoritative: bool, source_id: int,
        fixed_point_round: int, source_owner: Any = None,
    ) -> Any:
        """One enrichment write, posted DERIVED from the value's and the
        source value's ``ssa_value`` cells in ``function``'s scope."""

        from .concordance_declarations import (
            TENSOR_SHAPE_ENRICHMENT, TENSOR_SSA_LOWERING,
        )
        from .identity_concordance import Derived, Mode
        from .ssa_record_return_state import ssa_value_identity_cell
        from .frame_identity_book import _frame_value_cell

        row = (str(function_name), int(value_id), str(field))
        incumbent_cell = enrichment_book.latest_ref(
            TENSOR_SHAPE_ENRICHMENT, row,
        )
        incumbent = (
            None if incumbent_cell is None
            else mutation_page.latest(row)
        )
        transition = (old, new, bool(authoritative), int(source_id))
        if (
            isinstance(incumbent, tuple) and len(incumbent) == 6
            and tuple(incumbent[2:]) == transition
        ):
            # The same transition from the same cause, repeated by a
            # fixed-point round that changed nothing: no new fact.
            return incumbent_cell
        # Each write derives from the value's and the source's cells and
        # from the row's previous write, which is what a later write
        # revises (a round-trip is a revision of the write it undoes).
        cells = tuple(dict.fromkeys(
            cell for cell in (
                ssa_value_identity_cell(
                    function, int(value_id), book=enrichment_book,
                ),
                _frame_value_cell(
                    function if source_owner is None else source_owner,
                    int(source_id),
                ),
                incumbent_cell,
            )
            if cell is not None
        ))
        return enrichment_book.post(
            TENSOR_SHAPE_ENRICHMENT, row,
            (int(fixed_point_round), next(mutation_step), *transition),
            stage=TENSOR_SSA_LOWERING, provenance=Derived(cells),
            mode=Mode.REVISE,
        )

    def values(function) -> tuple[SSAValue, ...]:
        key = id(function)
        cached = value_cache.get(key)
        if cached is not None:
            return cached
        discovered = [
            *function.args,
            *(
                value
                for block in function.blocks.values()
                for instruction in block.instrs
                for value in (*instruction.args, instruction.res)
                if value is not None
            ),
        ]
        # One SSAValue is normally referenced by several instructions.  Keep
        # each live object once while retaining distinct objects that share a
        # canonical integer value id and all need the same metadata.
        cached = tuple({id(value): value for value in discovered}.values())
        value_cache[key] = cached
        return cached

    def values_by_id(function) -> dict[int, tuple[SSAValue, ...]]:
        key = id(function)
        cached = values_by_id_cache.get(key)
        if cached is not None:
            return cached
        grouped: dict[int, list[SSAValue]] = {}
        for value in values(function):
            grouped.setdefault(int(value.id), []).append(value)
        cached = {
            value_id: tuple(group)
            for value_id, group in grouped.items()
        }
        values_by_id_cache[key] = cached
        return cached

    def constant_result_ids(function) -> frozenset[int]:
        key = id(function)
        cached = constant_result_ids_cache.get(key)
        if cached is not None:
            return cached
        cached = frozenset(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op == "Const" and instruction.res is not None
        )
        constant_result_ids_cache[key] = cached
        return cached

    def enrich(
        function, value_id: int, source: SSAValue, *, authoritative: bool = False,
        source_function: Any = None,
        source_owner: Any = None,
        tensor_contract_override: Mapping[str, Any] | None = None,
        contract_cause: Any = None,
        stage: str = "repository_ssa_enrichment",
    ) -> bool:
        # ``contract_cause`` is the booked row ``tensor_contract_override``
        # was read from (a TENSOR_METADATA_RESULT_BINDING for a call result).
        # The override can turn the target's metadata dynamic; that row is
        # then the exact cause of the transport edge's revision and is cited.
        # An accepted SSA component is an immutable transaction.  Its caller
        # may read its exact rows through the retained owner book, but cannot
        # revise its values or descriptors in the ambient compilation book.
        from .ssa_record_return_state import accepted_ssa_component
        if accepted_ssa_component(function):
            return False
        changed = False
        source_accounting = dict(source.accounting or {})
        source_shape = tuple(source.shape or ())
        source_dtype = source.dtype
        source_aggregate = tuple(
            source_accounting.get("ssa_aggregate_outputs", ())
        )
        source_physical_dtype = source_accounting.get("physical_dtype")
        # These fields are one indivisible dynamic tensor contract.  Shape,
        # rank, and element-count are ordinary SSA identities minted by the
        # caller that owns the public span; a specialized region formal must
        # retain those exact identities so its tensor descriptor and every
        # backend can follow the call edge back to that span.  Previously this
        # fixed point propagated only the display shape/dtype, leaving the
        # formal ``unresolved`` even though the actual was fully accounted.
        tensor_contract_keys = _TENSOR_CONTRACT_KEYS
        source_tensor_contract = {
            key: source_accounting[key]
            for key in tensor_contract_keys
            if source_accounting.get(key) is not None
        }
        if tensor_contract_override is not None:
            source_tensor_contract.update(tensor_contract_override)
        # All occurrences in one owner scope with this canonical value id
        # are one SSA identity.  A dynamic descriptor on any occurrence is
        # therefore the contract of the others too, even when the immediate
        # transport source is a plain repository result with no tensor
        # accounting of its own.  Leaving an unannotated sibling to the
        # shape state's ``static`` default made successive occurrences post
        # dynamic and static target states to the same transformation row.
        # Carry only concordant sibling fields; disagreement remains visible
        # to the ordinary settlement instead of being guessed away here.
        sibling_accounting = tuple(
            dict(occurrence.accounting or {})
            for occurrence in occurrences_of(int(value_id), function)
            if (occurrence.accounting or {}).get(
                "tensor_metadata_state"
            ) == "dynamic"
        )
        if (
            sibling_accounting
            and "tensor_metadata_state" not in source_tensor_contract
        ):
            source_tensor_contract["tensor_metadata_state"] = "dynamic"
            for key in tensor_contract_keys:
                if key in source_tensor_contract:
                    continue
                candidates = {
                    accounting[key]
                    for accounting in sibling_accounting
                    if accounting.get(key) is not None
                }
                if len(candidates) == 1:
                    source_tensor_contract[key] = candidates.pop()
        if source_owner is not None and source_owner is not function:
            # The unsqueeze region's local extent values 3961/3962/3960
            # were copied backwards onto root input 0 even though none is
            # a value of root. A cross-frame metadata ID is transferable
            # only through the exact mapped result binding supplied above,
            # or when this target's own descriptor already owns that same
            # local SSA identity. The source's numeric ID alone is no edge.
            target_table = module.tensor_tables.get(function.name)
            target_descriptor = (
                None if target_table is None else
                target_table.by_id(int(value_id))
            )
            from .ssa_record_return_state import ssa_value_identity_cell
            for key, field in (
                ("tensor_shape_value_id", "shape_value_id"),
                ("tensor_rank_value_id", "rank_value_id"),
                ("tensor_element_count_value_id",
                 "element_count_value_id"),
            ):
                candidate = source_tensor_contract.get(key)
                if candidate is None or (
                    tensor_contract_override is not None
                    and key in tensor_contract_override
                ):
                    continue
                if (
                    target_descriptor is not None
                    and getattr(target_descriptor, field) == int(candidate)
                    and int(candidate) in values_by_id(function)
                    and ssa_value_identity_cell(
                        function, int(candidate), book=identity_book(module),
                    ) is not None
                ):
                    continue
                source_tensor_contract.pop(key, None)
        # Distinct SSAValue objects may carry one canonical integer id.  A
        # repository transport sometimes selects an unresolved occurrence as
        # ``source`` even though another occurrence already owns the exact
        # dtype.  That resolved sibling is not a competing inference: it is
        # the same SSA identity.  Use its concordant dtype for this whole
        # enrichment pass so a later occurrence cannot revise a proven
        # float64 edge back to ``unknown``.
        if source_dtype in {None, "unknown"}:
            sibling_dtypes = {
                str(occurrence.dtype)
                for occurrence in semantic_occurrences_of(int(value_id), function)
                if occurrence.dtype not in {None, "unknown"}
            }
            if len(sibling_dtypes) == 1:
                source_dtype = sibling_dtypes.pop()
        fixed_constant = int(value_id) in constant_result_ids(function)
        shape_transformation_records = []
        # Program: a computed arange result captured by a later planned region.
        # The later region references its producer's SSAValue object as an
        # operand.  Its metadata actuals belong to the later region's formal,
        # while writing those IDs onto the producer object corrupts the
        # producer's own accounting.  A local metadata target must be defined
        # here as a formal or instruction result; an operand reference alone
        # does not transfer ownership across the call edge.
        local_occurrences = {
            *(id(item) for item in function.args),
            *(
                id(instruction.res)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            ),
        }
        for value in occurrences_of(int(value_id), function):
            before = (tuple(value.shape or ()), value.dtype)
            enrichment_cells = []
            # An occurrence that declares itself an exact view of this same
            # storage owns its shape.  Restamping it with the allocation
            # owner's shape silently rewrote ``b.reshape((-1, 1, 2))`` back
            # to ``b``'s own extents, so the broadcast kernel conformed the
            # wrong axes and every consumer read misaligned elements.
            declared_view = (value.accounting or {}).get("ssa_storage_view")
            declared_view_shape = tuple(
                declared_view.get("view_shape") or ()
            ) if isinstance(declared_view, Mapping) else ()
            from .concordance_declarations import TENSOR_RUNTIME_VIEW_GEOMETRY
            from .ssa_record_return_state import (
                function_scope_of, posted_ssa_value_fact,
                ssa_value_identity_cell,
            )
            runtime_view_cell = identity_book(module).latest_ref(
                TENSOR_RUNTIME_VIEW_GEOMETRY,
                (function_scope_of(function), function.name, int(value.id)),
            )
            runtime_view_fact = (
                identity_book(module).page(TENSOR_RUNTIME_VIEW_GEOMETRY).cell(
                    runtime_view_cell.row, runtime_view_cell.column,
                ) if runtime_view_cell is not None else None
            )
            retained_runtime_view = (
                runtime_view_fact is not None
                and runtime_view_fact[0] == "unsqueeze"
            )
            retained_view = (
                retained_runtime_view
                or (isinstance(declared_view, Mapping)
                    and bool(declared_view_shape)
                    and bool(source_shape)
                    and prod(declared_view_shape) == prod(source_shape))
            )
            # compact_codewords: the authored static byte-weight vector is
            # unsqueezed to (1, 8). The root carrier has rank 1, while the
            # planned-region view has an existing rank-2 SSA_VALUE. Transport
            # of the carrier's physical rank must not overwrite that exact
            # view declaration. A matching shape spelling alone is not proof.
            view_book = identity_book(module)
            view_fact = posted_ssa_value_fact(
                function, int(value.id), book=view_book,
            )
            # canonical_codes_from_lengths: the aggregate fold rebinds the
            # rank-2 unsqueeze view 64 from its earlier storage 6 to actual 0.
            # The old occurrence still names 6, while the booked rebinding
            # names 0 and preserves the view's full shape.  Only that exact
            # source/target row and its SSA causes authorize retaining rank 2.
            _exact_rebound_view = _booked_static_view_rebinding(
                view_book, function, int(value.id),
                function if source_owner is None else source_owner,
                int(source.id), declared_view_shape,
            )
            static_view_rank_proved = (
                retained_view and not retained_runtime_view
                and isinstance(declared_view, Mapping)
                and (declared_view.get("storage_value_id") == int(source.id)
                     or _exact_rebound_view)
                and view_fact is not None
                and view_fact.storage == "span"
                and view_fact.rank == len(declared_view_shape)
                and tuple(view_fact.shape) == declared_view_shape
                and ssa_value_identity_cell(
                    function if source_owner is None else source_owner,
                    int(source.id), book=view_book,
                ) is not None
            )
            retained_axis_view = retained_runtime_view or static_view_rank_proved
            local_shape_writer = tuple(
                (value.accounting or {}).get(
                    "shape_settlement_provenance", ()
                ) or ()
            )[:1] in {
                ("shape_preserving_ssa_edge",),
                ("canonical_identity_exact_writer",),
            }
            def _log(field: str, old: Any, new: Any) -> None:
                cell = _log_enrichment(
                    function, function_names_by_id.get(id(function), "?"),
                    int(value.id), field, old, new, authoritative,
                    int(source.id), fixed_point_round, source_owner,
                )
                if cell is not None:
                    enrichment_cells.append(cell)

            if retained_view and tuple(value.shape or ()) != declared_view_shape:
                _log("shape", tuple(value.shape or ()), declared_view_shape)
                value.shape = declared_view_shape
                changed = True

            if (
                not fixed_constant
                and authoritative
                and not retained_view
                and not local_shape_writer
                and tuple(value.shape or ()) != source_shape
            ):
                _log("shape", tuple(value.shape or ()), source_shape)
                value.shape = source_shape
                changed = True
            elif (
                not fixed_constant
                and source_shape
                and not tuple(value.shape or ())
            ):
                _log("shape", tuple(value.shape or ()), source_shape)
                value.shape = source_shape
                changed = True
            if (
                source_dtype not in {None, "unknown"}
                and (value.dtype in {None, "unknown"} or authoritative)
                and value.dtype != source_dtype
            ):
                # ``"unknown"`` is the state's spelling of an absent dtype
                # (``shape_transformation_state``); an occurrence left at
                # it beside a resolved one posted the same shape edge with
                # another target state on every pass.
                _log("dtype", value.dtype, source_dtype)
                value.dtype = source_dtype
                changed = True
            if (
                source_physical_dtype is not None
                and (value.accounting or {}).get("physical_dtype") is None
            ):
                _log("physical_dtype", None, str(source_physical_dtype))
                # Rule: post first; accounting is a view.  The physical dtype
                # carried across this transport edge is revised onto the
                # target value's ``ssa_value`` row, DERIVED from the source
                # value's cell (the edge's own source cell) and the target's
                # previous cell; the accounting is then set FROM that fact.
                from .ssa_record_return_state import (
                    revise_ssa_value as _revise_physical,
                )
                from .frame_identity_book import _frame_value_cell
                _physical_book = identity_book(module)
                _revise_physical(
                    function, int(value.id), dtype=value.dtype,
                    shape=tuple(value.shape or ()), stage=stage,
                    sources=(_frame_value_cell(
                        function if source_owner is None else source_owner,
                        int(source.id),
                    ),),
                    book=_physical_book,
                    physical={"physical_dtype": str(source_physical_dtype)},
                )
                value.accounting = _accounting_view(function, value, module)
                changed = True
            if (
                source_aggregate
                and not tuple(
                    (value.accounting or {}).get(
                        "ssa_aggregate_outputs", ()
                    )
                )
            ):
                _log("ssa_aggregate_outputs", (), source_aggregate)
                value.accounting = {
                    **dict(value.accounting or {}),
                    "ssa_aggregate_outputs": source_aggregate,
                }
                changed = True
            # A computed return's mapped metadata IDs are materialized in the
            # caller by aggregate Loads. Source-owner sibling regions share
            # its semantic value ID, but those caller-local metadata values
            # do not exist in the sibling function until an exact input call
            # adds them as formals. Keep the output binding within its caller.
            for key, source_contract_value in source_tensor_contract.items():
                if retained_axis_view and key in {
                    "program_abi_rank", "tensor_shape_value_id",
                    "tensor_rank_value_id", "tensor_element_count_value_id",
                }:
                    continue
                if (
                    key in {
                        "tensor_shape_value_id", "tensor_rank_value_id",
                        "tensor_element_count_value_id",
                    }
                    and id(value) not in local_occurrences
                ):
                    continue
                old_contract_value = (value.accounting or {}).get(key)
                if old_contract_value is not None and not authoritative:
                    continue
                if old_contract_value == source_contract_value:
                    continue
                _log(key, old_contract_value, source_contract_value)
                value.accounting = {
                    **dict(value.accounting or {}),
                    key: source_contract_value,
                }
                changed = True
            # Program: any call whose actual's span contract is transported
            # onto a callee formal or load (history_projection: kernel formals
            # and region Loads carrying ``span`` of rank 3).  Wrong record:
            # the contract reached ``accounting`` but the value's
            # ``ssa_value`` row kept storage/rank at their defaults, and the
            # physical dtype post above carried only ``physical_dtype``.
            # Rule: the edge posts what it carried -- the source row's posted
            # storage/rank/arena, and the contract's storage/rank for the keys
            # this edge delivered (accounting equal to the source's) -- onto
            # the target row, DERIVED from the source's cell; accounting is
            # then the view.  A partial post merges over the recorded fact.
            from .concordance_declarations import (
                SSA_VALUE_PHYSICAL_STORAGES as _PHYSICAL_STORAGES,
            )
            from .ssa_record_return_state import (
                posted_ssa_value_fact as _posted_edge_fact,
                revise_ssa_value as _revise_edge_physical,
            )
            from .frame_identity_book import (
                _frame_owner_book, _frame_value_cell, _frame_cite,
            )
            _edge_book = identity_book(module)
            _edge_source_owner = (
                function if source_owner is None else source_owner
            )
            _edge_source_book, _edge_source_digest = _frame_owner_book(
                _edge_source_owner
            )
            _edge_source_fact = _posted_edge_fact(
                _edge_source_owner, int(source.id),
                book=_edge_source_book,
            )
            _edge_physical: dict[str, Any] = {}
            if _edge_source_fact is not None:
                # Weighted float multiplication: binary_double Loads188/189
                # have no accounting cache, but binary_value's exact scalar
                # edge proves float64. Copy that source cell's physical
                # dtype with its storage/rank; the cache is only a view.
                if _edge_source_fact.physical_dtype is not None:
                    _edge_physical["physical_dtype"] = (
                        _edge_source_fact.physical_dtype
                    )
                if _edge_source_fact.storage is not None:
                    _edge_physical["storage"] = _edge_source_fact.storage
                if _edge_source_fact.rank is not None:
                    _edge_physical["rank"] = _edge_source_fact.rank
                if _edge_source_fact.arena:
                    _edge_physical["arena"] = True
            if retained_axis_view:
                # The exact view receipt has already posted rank N+1 for
                # this logical value; the source's rank N describes its
                # storage, not the view's axes.
                _edge_physical.pop("rank", None)
            for _contract_key, _physical_field in (
                ("program_abi_storage", "storage"), ("program_abi_rank", "rank"),
            ):
                _delivered = source_tensor_contract.get(_contract_key)
                if (
                    _delivered is not None
                    and not (retained_axis_view
                             and _physical_field == "rank")
                    and _physical_field not in _edge_physical
                    and (value.accounting or {}).get(_contract_key) == _delivered
                    and (
                        _physical_field != "storage"
                        or _delivered in _PHYSICAL_STORAGES
                    )
                ):
                    _edge_physical[_physical_field] = (
                        int(_delivered) if _physical_field == "rank"
                        else _delivered
                    )
            if not authoritative and _edge_physical:
                # Program: the view probe's two stores into ``m`` call the
                # one shared ``index_assign_double`` kernel; its ``values``
                # formal (582) is fed a scalar by one store and a span by
                # the other.  Wrong record: every caller's edge REVISED the
                # shared formal's row, which alternated span/scalar each
                # round while accounting kept the first fill
                # (``ssa-publication-type-disagreement``).  Rule: a
                # non-authoritative edge is fill-only on the row exactly as
                # on accounting (``old_contract_value is not None and not
                # authoritative``): it posts only the fields the target row
                # does not record.
                _edge_target_fact = _posted_edge_fact(
                    function, int(value.id), book=_edge_book,
                )
                if _edge_target_fact is not None:
                    _edge_physical = {
                        field: item for field, item in _edge_physical.items()
                        if (
                            getattr(_edge_target_fact, field) is None
                            if field != "arena"
                            else not _edge_target_fact.arena
                        )
                    }
            _edge_layout_target = _posted_edge_fact(
                function, int(value.id), book=_edge_book,
            ) if _edge_physical else None
            if _edge_physical and _edge_layout_target is not None:
                # A physical transport may target an occurrence of storage0
                # viewed as flatten(2,3)->(6,). Its physical fields belong to
                # storage0; its displayed axes do not revise storage0's
                # semantic shape. Retain that row while filling layout.
                _revise_edge_physical(
                    function, int(value.id), dtype=_edge_layout_target.dtype,
                    shape=_edge_layout_target.shape, stage=stage,
                    sources=(_frame_value_cell(
                        _edge_source_owner, int(source.id),
                    ),),
                    book=_edge_book, physical=_edge_physical,
                )
            from .identity_concordance import (
                PROVEN_SHAPE_PAGE,
                SHAPE_STATE_PAGE,
                record_shape_transformation,
                shape_scope_of,
            )
            from .ssa_record_return_state import (
                revise_ssa_value,
            )

            book = identity_book(module)
            source_value_cell = _frame_value_cell(
                _edge_source_owner, int(source.id),
            )
            target_value_cell = _frame_value_cell(
                function, int(value.id),
            )
            if before != (tuple(value.shape or ()), value.dtype):
                # The transport changed this value's dtype or shape: that
                # is a revision of its ``ssa_value`` row, DERIVED from the
                # row's previous cell and the source value's cell.  The
                # shape edge below then stands on a cell that changed,
                # which is the cause of its revision.
                target_value_cell = revise_ssa_value(
                    function, int(value.id), dtype=value.dtype,
                    shape=tuple(value.shape or ()), stage=stage,
                    sources=(source_value_cell,), book=book,
                ) or target_value_cell

            # ``source`` can itself be one of this canonical id's
            # occurrences and may have been enriched above.  Publish from
            # its live contract, not the snapshot taken on entry.
            live_source_accounting = {
                **dict(source.accounting or {}),
                # When the transport source is a helper-function occurrence
                # of this same canonical id, the owner scope's concordant
                # dynamic contract still governs the shared identity.
                **source_tensor_contract,
            }
            live_source_shape = tuple(source.shape or ())
            live_source_dtype = (
                source_dtype
                if source_dtype not in {None, "unknown"}
                else source.dtype
            )
            source_rank = int(live_source_accounting.get(
                "program_abi_rank", len(live_source_shape),
            ) or len(live_source_shape))
            target_shape = tuple(value.shape or ())
            target_accounting = dict(value.accounting or {})
            target_rank = int(target_accounting.get(
                "program_abi_rank", len(target_shape),
            ) or len(target_shape))
            # Program: arange -> shape-preserving cast -> sum.  The sum
            # region's input formal has a dynamic tensor descriptor and its
            # SSA_VALUE row posts span rank 1, but one region-feed occurrence
            # carries no accounting rank.  Reading that occurrence as rank 0
            # made one shape edge alternate rank 1/0 without a changed cause.
            # Only a dynamic descriptor of this exact function/value permits
            # the posted rank to settle its geometry here; an arbitrary
            # kernel pointer formal does not define tensor rank.
            target_descriptor = (
                getattr(module, "tensor_tables", {}).get(function.name)
            )
            target_descriptor = (
                target_descriptor.by_id(int(value.id))
                if target_descriptor is not None else None
            )
            if (
                target_descriptor is not None
                and target_descriptor.metadata_state == "dynamic"
                and target_descriptor.shape_value_id is not None
                and target_descriptor.rank_value_id is not None
            ):
                from .frame_identity_book import _frame_physical_fields
                physical_fields = _frame_physical_fields(
                    function, int(value.id)
                )
                posted_rank = (
                    physical_fields.get("rank")
                    if isinstance(physical_fields, dict) else None
                )
                if posted_rank is not None:
                    target_rank = int(posted_rank)
            # A lowered storage alias can deliberately retain a different
            # view shape while carrying the allocation owner's integer id.
            # Its semantic reshape edge was already published under the
            # reshape result id; do not collapse that view into a conflicting
            # self-edge on the storage id here.
            if not _records_same_identity_shape_transport(
                live_source_shape, target_shape, declared_view,
                int(value.id),
            ):
                continue
            _shape_source_owner = (
                source_function or source_owner or function
            )
            if (
                int(source.id) == int(value.id)
                and shape_scope_of(_shape_source_owner)
                == shape_scope_of(function)
            ):
                # expand_to's planned region and enclosing function both
                # name value 2 in one shape scope.  Transporting its updated
                # SSA row is a view refresh, not a transformation from the
                # value to itself; a self-edge would invent a second shape
                # writer with no distinct source cell.
                continue
            # Only this value's exact booked extent triple can cause a
            # static-to-dynamic metadata edge. A receipt for another result
            # or another set of metadata IDs cannot revise its shape row.
            elementwise_geometry_cause = _matching_elementwise_metadata_cause(
                book, function, int(value.id), target_accounting,
            )
            shape_transformation_records.append((
                (
                    shape_scope_of(_shape_source_owner),
                    int(source.id),
                    shape_scope_of(function),
                    int(value.id),
                ),
                {
                    "stage": stage,
                    "operation": "ssa_metadata_transport",
                    "source_state": {
                    "shape": live_source_shape,
                    "dtype": str(live_source_dtype or "unknown"),
                    "rank": source_rank,
                    "metadata_state": str(live_source_accounting.get(
                        "tensor_metadata_state", "static",
                    )),
                    **(
                        {"sequence_row_shape": tuple(live_source_accounting[
                            "sequence_row_shape"
                        ])}
                        if live_source_accounting.get(
                            "sequence_row_shape"
                        ) is not None else {}
                    ),
                    },
                    "target_state": {
                    "shape": target_shape,
                    "dtype": str(value.dtype or "unknown"),
                    "rank": target_rank,
                    "metadata_state": str(target_accounting.get(
                        "tensor_metadata_state", "static",
                    )),
                    **(
                        {"sequence_row_shape": tuple(target_accounting[
                            "sequence_row_shape"
                        ])}
                        if target_accounting.get(
                            "sequence_row_shape"
                        ) is not None else {}
                    ),
                    },
                    "role": "ssa_value",
                # The shape was read off the source value: the edge derives
                # from that value's ``ssa_value`` cell in its own function.
                    "source_cells": tuple(
                    cell for cell in (
                        source_value_cell,
                        # The target state was read off the target value:
                        # its ``ssa_value`` cell (revised above when the
                        # transport changed it) is a source of this edge.
                        target_value_cell,
                        elementwise_geometry_cause,
                        # The result binding whose caller-side metadata ids
                        # the override carried: when the transported contract
                        # turns dynamic through it, it is that change's cause.
                        contract_cause,
                        # The state the source was last resolved to: when
                        # that changed, this edge's revision has its cause.
                        _frame_cite(
                            _edge_source_book,
                            _edge_source_book.latest_ref(
                                SHAPE_STATE_PAGE, (
                                    shape_scope_of(_shape_source_owner),
                                    int(source.id),
                                ),
                            ),
                            _edge_source_digest,
                        ),
                        # A later shape proof can refine an occurrence whose
                        # ``ssa_value`` identity has not itself been revised
                        # yet.  That proof is the exact cause of transporting
                        # the newly visible extent.
                        identity_book(module).latest_ref(
                            PROVEN_SHAPE_PAGE, (
                                shape_scope_of(function), int(value.id),
                            ),
                        ),
                        # Accounting-only enrichment (notably a dynamic to
                        # static tensor metadata transition) changes the
                        # shape state without changing the SSA value's
                        # display shape or dtype, so ``revise_ssa_value`` has
                        # no row to revise.  The exact enrichment write is
                        # the cause of this edge revision.
                        *enrichment_cells,
                    )
                    if cell is not None
                ),
                },
            ))
        # One canonical id can have several SSAValue occurrences.  Settle
        # all of them before publishing their common shape edge: otherwise
        # an already-settled occurrence can revise the row before the later
        # occurrence that actually changes has produced the causal receipt.
        from .identity_concordance import (
            SHAPE_EDGE_PAGE, shape_transformation_state,
        )
        for arguments, keywords in shape_transformation_records:
            source_scope, source_id, target_scope, target_id = arguments
            if (
                source_scope != target_scope
                and keywords["operation"] == "ssa_metadata_transport"
            ):
                # compress_octets' sequence-31 append_slice helper forwards
                # one canonical storage id in both directions.  The forward
                # edge may have just revised ptr to int64 from an exact
                # metadata receipt, while the reverse edge's SSA and source
                # shape cells still predate its prior column.  Cite that
                # *new* matching forward edge as the reverse write's cause.
                stage_name = str(keywords["stage"])
                role_name = str(keywords["role"])
                reverse_row = (
                    target_scope, target_id, source_scope, source_id,
                    stage_name, "ssa_metadata_transport", role_name,
                )
                forward_row = (
                    source_scope, source_id, target_scope, target_id,
                    stage_name, "ssa_metadata_transport", role_name,
                )
                incumbent = book.latest_ref(SHAPE_EDGE_PAGE, reverse_row)
                forward = book.latest_ref(SHAPE_EDGE_PAGE, forward_row)
                if (
                    incumbent is not None
                    and forward is not None
                    and book.stamp_of(forward) > book.stamp_of(incumbent)
                    and book.page(SHAPE_EDGE_PAGE).cell(
                        forward.row, forward.column,
                    ) == (
                        shape_transformation_state(keywords["target_state"]),
                        shape_transformation_state(keywords["source_state"]),
                    )
                ):
                    keywords = {
                        **keywords,
                        "source_cells": tuple(dict.fromkeys((
                            *keywords["source_cells"], forward,
                        ))),
                    }
            record_shape_transformation(*arguments, **keywords)
        return changed

    def settle_specialized_formal_descriptor(
        callee_name: str, formal: SSAValue, source: SSAValue,
        *, source_owner: Any = None,
    ) -> bool:
        """Keep a region input descriptor aligned with its exact call ABI."""

        from .ssa_record_return_state import accepted_ssa_component
        callee = module.functions.get(callee_name)
        if callee is not None and accepted_ssa_component(callee):
            return False

        shape = tuple(map(int, source.shape or ()))
        if not shape:
            return False
        table = getattr(module, "tensor_tables", {}).get(callee_name)
        descriptor = (
            table.by_id(int(formal.id)) if table is not None else None
        )
        if (
            descriptor is None
            or descriptor.storage != "input"
            or tuple(descriptor.shape) == shape
        ):
            return False
        stride = 1
        reversed_strides = []
        for extent in reversed(shape):
            reversed_strides.append(stride)
            stride *= int(extent)
        dtype_bytes = dtype_byte_size(str(source.dtype or descriptor.dtype))
        _log_enrichment(
            module.functions.get(callee_name), callee_name, int(formal.id),
            "descriptor.shape", tuple(descriptor.shape), shape, True,
            int(source.id), fixed_point_round, source_owner,
        )
        table.tensors[int(formal.id)] = dataclasses.replace(
            descriptor,
            dtype=str(source.dtype or descriptor.dtype),
            shape=shape,
            strides=tuple(reversed(reversed_strides)),
            byte_size=prod(shape) * dtype_bytes,
            metadata_state="static",
        )
        return True

    def returned(function) -> tuple[SSAValue, ...]:
        key = id(function)
        cached = returned_cache.get(key)
        if cached is not None:
            return cached
        cached = next((
            tuple(instruction.args)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        ), ())
        returned_cache[key] = cached
        return cached

    def constant_indices(function) -> dict[int, int]:
        key = id(function)
        cached = constant_indices_cache.get(key)
        if cached is not None:
            return cached
        result = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op != "Const" or instruction.res is None:
                    continue
                payload = _constant_integer(instruction)
                if payload is not None:
                    result[int(instruction.res.id)] = payload
        constant_indices_cache[key] = result
        return result

    def projections(function, aggregate_id: int) -> dict[int, SSAValue]:
        key = (id(function), int(aggregate_id))
        cached = projection_cache.get(key)
        if cached is not None:
            return cached
        constants = constant_indices(function)
        addresses: dict[int, int] = {}
        result: dict[int, SSAValue] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op in {"GetElementPtr", "getelementptr"}
                    and instruction.res is not None
                    and instruction.args
                    and int(instruction.args[0].id) == int(aggregate_id)
                ):
                    position = instruction.attributes.get("aggregate_index")
                    if position is None and len(instruction.args) > 1:
                        position = constants.get(int(instruction.args[1].id))
                    if position is not None:
                        addresses[int(instruction.res.id)] = int(position)
                elif (
                    instruction.op in {"Load", "load"}
                    and instruction.res is not None
                    and instruction.args
                    and int(instruction.args[0].id) in addresses
                ):
                    result[addresses[int(instruction.args[0].id)]] = instruction.res
        projection_cache[key] = result
        return result

    def fixed_point_state() -> tuple[Any, ...]:
        """Snapshot exactly the facts this fixed point may change."""

        def aggregate_item_state(item: Any) -> tuple[Any, ...]:
            if hasattr(item, "id"):
                return (
                    "value", int(item.id), item.dtype,
                    tuple(item.shape or ()),
                )
            if isinstance(item, int):
                return ("value_id", int(item))
            return ("opaque", type(item).__name__, repr(item))

        value_state = tuple(
            (
                str(function_name), occurrence, int(value.id), value.dtype,
                tuple(value.shape or ()),
                (value.accounting or {}).get("physical_dtype"),
                tuple(
                    aggregate_item_state(item)
                    for item in (value.accounting or {}).get(
                        "ssa_aggregate_outputs", ()
                    )
                ),
                tuple(
                    (key, (value.accounting or {}).get(key))
                    for key in _TENSOR_CONTRACT_KEYS
                ),
            )
            for function_name, function in module.functions.items()
            for occurrence, value in enumerate(values(function))
        )
        descriptor_state = tuple(
            (
                str(function_name), int(tensor_id), descriptor.dtype,
                tuple(descriptor.shape), tuple(descriptor.strides),
                (
                    None if descriptor.byte_size is None
                    else int(descriptor.byte_size)
                ),
                str(descriptor.metadata_state),
            )
            for function_name, table in getattr(
                module, "tensor_tables", {}
            ).items()
            for tensor_id, descriptor in sorted(table.tensors.items())
        )
        return value_state, descriptor_state

    # Source regions share their owner's value space, exactly as
    # settle_canonical_value_metadata records above. Equal local integers in
    # unrelated functions are not identities: energy helper value 68 (bool)
    # used to retype binary_value's unrelated value 68 (double), poisoning
    # subsequent native channel arithmetic. Cross-owner propagation follows
    # the actual/formal call edges below, never a module-wide integer match.
    scopes = {
        id(function): str((function.metadata.get("source_region_integral") or {}).get(
            "owner") or function.name)
        for function in module.functions.values()
    }
    scoped_values: dict[tuple[str, int], dict[int, SSAValue]] = {}
    reference_counts: dict[tuple[str, int], set[str]] = {}
    # The blocks in which each owner references the value: a formal is
    # referenced by the function's entry block, an operand or result by the
    # block of its instruction.  Each is a witness of the reference.
    reference_blocks: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for owner_name, owner_function in module.functions.items():
        for value in values(owner_function):
            key = (scopes[id(owner_function)], int(value.id))
            scoped_values.setdefault(key, {})[id(value)] = value
            reference_counts.setdefault(key, set()).add(owner_name)
        entry_label = next(iter(owner_function.blocks), None)
        for formal in owner_function.args:
            if entry_label is not None:
                reference_blocks.setdefault(
                    (scopes[id(owner_function)], int(formal.id)), set(),
                ).add((str(owner_name), str(entry_label)))
        for label, block in owner_function.blocks.items():
            for instruction in block.instrs:
                for value in (*instruction.args, instruction.res):
                    if value is None:
                        continue
                    reference_blocks.setdefault(
                        (scopes[id(owner_function)], int(value.id)), set(),
                    ).add((str(owner_name), str(label)))
    from .concordance_declarations import CROSS_FUNCTION_REFERENCES
    from .ssa_record_return_state import (
        function_scope_of, ssa_block_identity_cell, ssa_value_identity_cell,
    )
    from .frame_identity_book import _frame_owner_book

    def reference_witness(owner_name, *, value_id=None, label=None):
        # _split composes the accepted _read_header regions. Their shared
        # value 5 and block witnesses remain in the dependency's own book;
        # looking in this unit's book lost every cause of the reference row.
        # Transport those exact cells through the existing imported-proof
        # edge, without copying foreign value rows into the caller's scope.
        function = module.functions[owner_name]
        destination = identity_book(module)
        owner, digest = (
            _frame_owner_book(function)
            if function.metadata.get("accepted_ssa_component")
            else (destination, None)
        )
        cell = (
            ssa_value_identity_cell(function, value_id, book=owner)
            if value_id is not None
            else ssa_block_identity_cell(function, label, book=owner)
        )
        if cell is not None and owner is not destination:
            cell = destination.import_cell(owner, cell, digest)
        return cell

    for key, owners in reference_counts.items():
        if len(owners) > 1:
            # DERIVED from the value's cell in each owner and from the
            # ``ssa_block`` cell of every block that references it: the
            # owner set is what those blocks say, so a block or a whole
            # function leaving the module changes the cells this fact
            # stands on, which is the cause of its revision.  (Regions share
            # their owner's value scope, so the value cells alone are one
            # cell for every owner.)
            _post_call_edge_fact(
                module, CROSS_FUNCTION_REFERENCES, key, tuple(sorted(owners)),
                (
                    *(
                        reference_witness(owner, value_id=int(key[1]))
                        for owner in sorted(owners)
                    ),
                    *(
                        reference_witness(owner, label=label)
                        for owner, label in sorted(reference_blocks.get(key, ()))
                        if owner in module.functions
                    ),
                ),
            )

    def occurrences_of(value_id: int, function: Any) -> tuple[SSAValue, ...]:
        """Occurrences in this exact SSA function's value space.

        In `reshape(-1, 8) * weights.unsqueeze(0)`, root and planned
        region 3 both use canonical value 20, but their metadata SSA ids
        belong to different frames. The owner scope is valid for semantic
        value identity; it cannot authorize copying root's caller metadata
        ids onto the region-local Mul result. Explicit Call bindings below
        carry those ids across frames.
        """
        return values_by_id(function).get(int(value_id), ())

    def semantic_occurrences_of(value_id: int, function: Any) -> tuple[SSAValue, ...]:
        """Source-owner siblings may settle one semantic dtype."""
        return tuple(scoped_values.get(
            (scopes[id(function)], int(value_id)), {},
        ).values())

    changed = True
    settle_exact_formals = True
    # Authoritative return metadata is a settling phase, not a permanent
    # overwrite mode.  Leaving it enabled for every fixed-point round lets
    # distinct SSA aliases with the same canonical id overwrite one another
    # on every scan, so ``changed`` can remain true forever even though no new
    # fact enters the module.  Apply the exact return contract once, then let
    # the ordinary fill-only propagation carry those settled facts outward.
    settle_exact_returns = bool(authoritative_returns)
    initial_state = fixed_point_state()
    seen_states = {initial_state: 0}
    history = [initial_state]
    state_receipts = [{
        "round": 0,
        "digest": sha256(repr(initial_state).encode("utf-8")).hexdigest(),
    }]
    fixed_point_round = 0
    while changed:
        fixed_point_round += 1
        changed = False
        for function_name, function in module.functions.items():
            # Elementwise/reduction results inherit a shaped numerical input;
            # exact result extents already present elsewhere always win.
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.res is None:
                        continue
                    operation = str(
                        instruction.attributes.get("tensor_candidate")
                        or instruction.attributes.get("tensor_operation")
                        or instruction.op
                    )
                    opcode = c_tensor_opcode(operation)
                    if operation not in {
                        "Div", "Sub", "Add", "Mul",
                        "div", "truediv", "sub", "add", "mul",
                        "broadcast_to", "expand",
                    } and not (opcode is not None and opcode[0] == "unary"):
                        continue
                    shaped = next(
                        (value for value in instruction.args if tuple(value.shape or ())),
                        None,
                    )
                    if shaped is not None:
                        changed |= enrich(function, int(instruction.res.id), shaped)

            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in {"Call", "call"}:
                        continue
                    callee_name = str(instruction.attributes.get("callee") or "")
                    callee = module.functions.get(callee_name)
                    if callee is None:
                        continue
                    if instruction.attributes.get("ssa_output_argument") is not None:
                        # Authored repository kernels are generic definitions
                        # shared by every tensor shape. Their concrete call
                        # operands are already complete; propagating one
                        # invocation's shapes through the shared formals and
                        # back into another invocation corrupts metadata.
                        continue
                    specialized_call = (
                        "__planned_region_" in callee_name
                    )
                    feed_ids = tuple(map(
                        int, instruction.attributes.get("feed_ids", ())
                    ))
                    for position, (actual, formal) in enumerate(zip(
                        instruction.args, callee.args
                    )):
                        semantic_actual = actual
                        if (
                            position < len(feed_ids)
                            and not tuple(actual.shape or ())
                        ):
                            candidates = values_by_id(function).get(
                                feed_ids[position], ()
                            )
                            numerical_candidates = tuple(
                                candidate for candidate in candidates
                                if str(candidate.dtype or "") not in {
                                    "ssa.aggregate", "ptr", "pointer",
                                    "ptrptr_float64",
                                }
                            )
                            shaped_candidates = tuple(
                                candidate for candidate in numerical_candidates
                                if tuple(candidate.shape or ())
                            )
                            shaped_contracts = {
                                (
                                    tuple(candidate.shape),
                                    str(candidate.dtype or ""),
                                )
                                for candidate in shaped_candidates
                            }
                            # Reshapes are storage aliases and deliberately
                            # permit one semantic feed id to have several
                            # shaped views.  An empty materialized operand may
                            # inherit only a unanimous contract; choosing the
                            # first view would make source order decide ABI.
                            if len(shaped_contracts) == 1:
                                semantic_actual = shaped_candidates[0]
                            elif not shaped_contracts:
                                semantic_actual = next(
                                    iter(numerical_candidates), actual,
                                )
                            # The call operand may be a caller-owned storage
                            # formal while feed_ids names the canonical source
                            # value.  They are one dependency edge: carry the
                            # Program-ABI/source descriptor to both endpoints.
                            if (
                                tuple(semantic_actual.shape or ())
                                and not tuple(actual.shape or ())
                            ):
                                _log_enrichment(
                                    function, function_name, int(actual.id),
                                    "shape", tuple(actual.shape or ()),
                                    tuple(semantic_actual.shape), False,
                                    int(semantic_actual.id), fixed_point_round,
                                )
                                actual.shape = tuple(semantic_actual.shape)
                                changed = True
                            if (
                                semantic_actual.dtype is not None
                                and actual.dtype is None
                            ):
                                _log_enrichment(
                                    function, function_name, int(actual.id),
                                    "dtype", actual.dtype,
                                    semantic_actual.dtype, False,
                                    int(semantic_actual.id), fixed_point_round,
                                )
                                actual.dtype = semantic_actual.dtype
                                changed = True
                        # The caller's actual value is the concrete contract
                        # for this specialized formal, including the
                        # distinction between a scalar ``()`` and a tensor.
                        # Reverse propagation remains fill-only so a stale
                        # placeholder cannot reshape the actual argument.
                        # A repository kernel is the imported, complete
                        # LLVM body shared by every caller; its formals are
                        # its own ABI, not any one caller's actual.  Every
                        # caller writing its storage onto the one formal row
                        # made ``index_assign_double``'s ``values`` formal
                        # alternate span/scalar.  The body stays its input;
                        # the call edge is settled on the caller's side.
                        if not (callee.metadata or {}).get(
                            "llvm_repository_kernel"
                        ):
                            changed |= enrich(
                                callee, int(formal.id), semantic_actual,
                                authoritative=(
                                    specialized_call and settle_exact_formals
                                ),
                                source_owner=function,
                            )
                        if specialized_call:
                            changed |= settle_specialized_formal_descriptor(
                                callee_name, formal, semantic_actual,
                                source_owner=function,
                            )
                        # A shared in-place kernel (it declares an output
                        # argument; the call carries none) owns only that
                        # argument's storage.  Its other formals are filled
                        # by every caller, so their shape is no fact about
                        # this caller's actual.  ``m[2] = m[2] * 0.5;
                        # m[:2] -= t``: ``index_assign_double``'s ``values``
                        # formal took (2,) from the second store and was
                        # written back onto the first store's scalar, which
                        # then stored element 0 of a 2-span (0.0).
                        kernel_output = (callee.metadata or {}).get(
                            "ssa_output_argument"
                        )
                        if (
                            specialized_call
                            or kernel_output is None
                            or int(kernel_output) == position
                        ):
                            changed |= enrich(
                                function, int(actual.id), formal,
                                source_owner=callee,
                            )

                    callee_returns = returned(callee)
                    declared = tuple(map(
                        int, instruction.attributes.get("output_ids", ())
                    ))
                    declared_positions = tuple(map(
                        int,
                        instruction.attributes.get("output_positions", ()),
                    ))
                    if declared and instruction.res is not None:
                        caller_outputs = projections(function, int(instruction.res.id))
                        callee_values = {
                            value_id: candidates[0]
                            for value_id, candidates in values_by_id(callee).items()
                        }
                        callee_declared = tuple(map(int,
                            instruction.attributes.get("callee_output_ids", declared)
                            if instruction.attributes.get(
                                "computed_tensor_metadata_outputs"
                            ) else declared
                        ))
                        for position, output_id in enumerate(declared):
                            caller_value = caller_outputs.get(position)
                            callee_position = (
                                declared_positions[position]
                                if position < len(declared_positions)
                                else None
                            )
                            positional_callee_value = (
                                callee_returns[callee_position]
                                if (
                                    callee_position is not None
                                    and 0 <= callee_position < len(callee_returns)
                                )
                                else None
                            )
                            callee_value_id = (
                                int(positional_callee_value.id)
                                if positional_callee_value is not None
                                else int(output_id)
                            )
                            tensor_table = getattr(
                                module, "tensor_tables", {}
                            ).get(callee_name)
                            descriptor = (
                                tensor_table.by_id(callee_value_id)
                                if tensor_table is not None else None
                            )
                            material_descriptor = (
                                descriptor
                                if descriptor is not None
                                and bool(tuple(descriptor.shape))
                                and bool(descriptor.owns_allocation)
                                and int(descriptor.data_value_id)
                                == int(output_id)
                                else None
                            )
                            view_descriptor = (
                                descriptor
                                if descriptor is not None
                                and descriptor.storage == "view"
                                and bool(tuple(descriptor.shape))
                                and int(descriptor.tensor_id)
                                == int(callee_value_id)
                                else None
                            )
                            if view_descriptor is not None:
                                # A planned region may return the same storage
                                # more than once through different shaped
                                # views. Ret therefore contains the resident
                                # id repeatedly and cannot carry the semantic
                                # extents by id alone. The tensor table is the
                                # exact concordance from the authored output
                                # identity to that resident storage; retain
                                # its view shape instead of replacing it with
                                # the allocation owner's shape.
                                callee_value = SSAValue(
                                    callee_value_id,
                                    dtype=str(view_descriptor.dtype),
                                    shape=tuple(view_descriptor.shape),
                                    accounting={
                                        "ssa_storage_view": {
                                            "storage_value_id": int(
                                                view_descriptor.data_value_id
                                            ),
                                            "view_shape": tuple(
                                                view_descriptor.shape
                                            ),
                                            "operation": "tensor_descriptor",
                                        },
                                    },
                                )
                            elif material_descriptor is not None:
                                # The tensor table is the allocation contract
                                # for a lowered region output.  One canonical
                                # source id can also name earlier singleton
                                # views; choosing the first SSA occurrence
                                # stamped those view extents onto a fully
                                # materialized output at the next call edge.
                                callee_value = SSAValue(
                                    callee_value_id,
                                    dtype=str(material_descriptor.dtype),
                                    shape=tuple(material_descriptor.shape),
                                )
                            elif positional_callee_value is not None:
                                # Source-linked aggregate ids belong to the
                                # caller's SSA namespace.  The linker records
                                # the exact callee Ret position separately;
                                # use that physical ABI instead of looking up
                                # the caller id in the callee, where the same
                                # integer may name an unrelated local constant.
                                callee_value = positional_callee_value
                            else:
                                callee_value = callee_values.get(
                                    callee_declared[position]
                                    if position < len(callee_declared)
                                    else output_id
                                )
                                if callee_value is None and descriptor is not None:
                                    callee_value = callee_values.get(
                                        int(descriptor.data_value_id)
                                    )
                            if caller_value is None or callee_value is None:
                                continue
                            result_override = None
                            from .concordance_declarations import (
                                TENSOR_METADATA_RESULT_BINDING,
                            )
                            binding_row = (
                                function_scope_of(function),
                                int(instruction.res.id),
                                str(callee_name),
                                int(output_id),
                            )
                            binding = enrichment_book.page(
                                TENSOR_METADATA_RESULT_BINDING
                            ).latest(binding_row)
                            binding_cause = (
                                enrichment_book.latest_ref(
                                    TENSOR_METADATA_RESULT_BINDING, binding_row,
                                ) if binding is not None else None
                            )
                            if binding is not None:
                                result_override = dict(zip((
                                    "tensor_shape_value_id",
                                    "tensor_rank_value_id",
                                    "tensor_element_count_value_id",
                                ), binding[2]))
                            changed |= enrich(
                                function, int(caller_value.id), callee_value,
                                authoritative=(
                                    settle_exact_returns
                                    or view_descriptor is not None
                                    or material_descriptor is not None
                                ),
                                source_owner=callee,
                                tensor_contract_override=result_override,
                                contract_cause=binding_cause,
                            )
                            if not authoritative_returns and binding is None:
                                changed |= enrich(
                                    callee, int(callee_value.id), caller_value,
                                    source_owner=function,
                                )
                    elif len(callee_returns) == 1 and instruction.res is not None:
                        changed |= enrich(
                            function, int(instruction.res.id), callee_returns[0],
                            authoritative=settle_exact_returns,
                            source_owner=callee,
                        )
                        if not authoritative_returns:
                            changed |= enrich(
                                callee, int(callee_returns[0].id),
                                instruction.res,
                                source_owner=function,
                            )
                    elif len(callee_returns) > 1 and instruction.res is not None:
                        # An aggregate may be forwarded into a small adapter
                        # whose sole job is to project its members.
                        for consumer_block in function.blocks.values():
                            for consumer in consumer_block.instrs:
                                if consumer.op not in {"Call", "call"}:
                                    continue
                                adapter = module.functions.get(str(
                                    consumer.attributes.get("callee") or ""
                                ))
                                if adapter is None:
                                    continue
                                for position, actual in enumerate(consumer.args):
                                    if int(actual.id) != int(instruction.res.id):
                                        continue
                                    formal = adapter.args[position]
                                    adapter_outputs = projections(adapter, int(formal.id))
                                    for output_index, source in enumerate(callee_returns):
                                        projected = adapter_outputs.get(output_index)
                                        if projected is not None:
                                            changed |= enrich(
                                                adapter, int(projected.id), source,
                                                authoritative=settle_exact_returns,
                                                source_owner=callee,
                                            )
        # Region returns and their owner's Phi values are one constraint
        # cycle.  Settle local shape-preserving kernel/Phi writers inside the
        # same fixed point, after this round's exact call edges, so a
        # provisional return shape cannot be copied outward and then fight
        # its own binary/broadcast writer in a separate outer pass.
        changed |= settle_shape_preserving_value_metadata(module)
        settle_exact_formals = False
        settle_exact_returns = False
        state = fixed_point_state()
        history.append(state)
        state_receipts.append({
            "round": fixed_point_round,
            "digest": sha256(repr(state).encode("utf-8")).hexdigest(),
        })
        if changed and state in seen_states:
            first_round = seen_states[state]
            module.metadata["call_metadata_fixed_point_receipts"] = tuple(
                state_receipts
            )
            # The digest alone says THAT the module is cycling, not WHICH
            # value is disputed or by whom.  Diff every consecutive pair of
            # recorded rounds between the first and repeated occurrence to
            # name the exact (function, value id) whose fact flips back and
            # forth -- a receipt the next run can read directly instead of
            # re-deriving this by re-running the whole lowering.
            def _keyed(round_state: Any) -> dict[Any, Any]:
                # Key by (function, occurrence) -- the Python-object identity
                # fixed_point_state() actually iterates -- never by value id
                # alone.  Two distinct SSA aliases can share one canonical id
                # and legitimately disagree; collapsing them onto one id-keyed
                # entry silently keeps only whichever occurrence iterated
                # last, hiding exactly the disagreement this diagnostic
                # exists to show.  The value id is still visible inside each
                # entry's own tuple for grouping/reporting.
                value_state, descriptor_state = round_state
                keyed: dict[Any, Any] = {
                    (entry[0], "value", entry[1]): entry for entry in value_state
                }
                keyed.update({
                    (entry[0], "descriptor", entry[1]): entry
                    for entry in descriptor_state
                })
                return keyed

            oscillating: dict[Any, list[tuple[int, Any]]] = {}
            for round_index in range(first_round, fixed_point_round):
                before = _keyed(history[round_index])
                after = _keyed(history[round_index + 1])
                for key in before.keys() | after.keys():
                    before_fact, after_fact = before.get(key), after.get(key)
                    if before_fact != after_fact:
                        oscillating.setdefault(
                            key, [(round_index, before_fact)]
                        ).append((round_index + 1, after_fact))
            module.metadata["call_metadata_fixed_point_oscillation"] = {
                str(key): tuple(trail) for key, trail in oscillating.items()
            }
            trail_report = "; ".join(
                f"{key}: " + " -> ".join(f"r{r}={fact}" for r, fact in trail)
                for key, trail in list(oscillating.items())[:5]
            )
            # A round-boundary snapshot can only differ when a round's NET
            # effect differs from one already seen -- it is structurally
            # blind to two writers disagreeing about one value WITHIN the
            # same round (write A, then a later write reverts it to what
            # round 0 already had).  The oscillation trail above is real but
            # can legitimately be empty for exactly that reason.  The
            # mutation page records every individual write as its own cell,
            # so it can still show that in-round round-trip: a row (the
            # exact (function, value id, field) written) whose fact history
            # revisits an earlier value is exactly this shape of defect,
            # confirmed against the actual product build 2026-09-17/18
            # (`step_1__restore__specialized_*`: filled from a shared
            # template formal, then authoritatively reverted using a stale
            # copy of itself as the "source of truth").
            # Fact: (round, step, old, new, authoritative, source id).
            repeated_writes = mutation_page.oscillating_rows(
                key=lambda fact: fact[3], old_key=lambda fact: fact[2],
            )
            module.metadata["call_metadata_fixed_point_repeated_writes"] = {
                str(target): entries for target, entries in repeated_writes.items()
            }
            repeated_report = "; ".join(
                f"{target}: " + " -> ".join(
                    f"round{fact[0]}.{fact[1]} {fact[2]}->{fact[3]} "
                    f"(authoritative={fact[4]}, source={fact[5]})"
                    for _column, fact in mutation_page.history(target)
                )
                for target in list(repeated_writes)[:5]
            )
            # The other half of why this book is shared: fortran_c_shell.py
            # already recorded, on the SAME book, who decided each of these
            # values' caller bindings and from which callsite -- before this
            # pass ever touched their shape.  Pull it onto the same row key
            # (function, value id) so the report shows not just THAT a value
            # round-tripped but WHOSE two binding decisions disagreed about it.
            from .identity_concordance import argument_binding_history

            binding_page = identity_book(module).pages.get("argument_binding")
            binding_report = "; ".join(
                f"{target[:2]}: " + " -> ".join(
                    f"callsite{callsite} {fact[0]}={fact[1]}"
                    if isinstance(fact, tuple) and len(fact) == 2
                    else f"callsite{callsite} {fact!r}"
                    for callsite, fact in argument_binding_history(
                        binding_page, target[0], target[1],
                    )
                )
                for target in list(repeated_writes)[:5]
            ) if binding_page is not None else ""
            raise RuntimeError(
                "repository SSA call metadata fixed point repeated state "
                f"from round {first_round} at round {fixed_point_round} "
                "while a pass still reported change. "
                f"Disputed fact(s) (round-boundary diff): {trail_report}. "
                f"Repeated writes within the disputed round(s): {repeated_report}. "
                f"Argument-binding history for the same values: {binding_report}"
            )
        seen_states.setdefault(state, fixed_point_round)
        changed_any |= changed
    module.metadata["call_metadata_fixed_point_rounds"] = fixed_point_round
    module.metadata["call_metadata_fixed_point_receipts"] = tuple(
        state_receipts
    )
    return changed_any


def settle_repository_ssa_shape_metadata(module: IRModule) -> bool:
    """Converge local shape laws with exact repository call-edge metadata.

    Either side can expose the next fact needed by the other: a local
    elementwise result may acquire extents only after its operands settle,
    while a downstream planned-region formal can acquire those extents only
    across its exact call edge.  Both passes are monotone on unresolved
    metadata, and the call-edge pass retains its disagreement and oscillation
    checks, so alternate them until a complete round reports no change.
    """

    from .bounded_fixed_point import BoundedFixedPoint, bound_for

    # Both passes only enrich SSAValue shape/dtype/accounting on an immutable
    # topology, so a changing round enriches at least one value: the number
    # of SSA values (plus call operands) bounds the rounds.
    guard = BoundedFixedPoint(
        "repository-shape-settlement",
        bound_for(
            "settlement",
            functions=len(module.functions),
            nodes=sum(
                len(instruction.args) + 1
                for function in module.functions.values()
                for block in function.blocks.values()
                for instruction in block.instrs
            ),
        ),
        scope=("whole-program",),
    )
    changed_any = False
    while True:
        # Settle the local writer first.  Exact call-return propagation is
        # authoritative only after the callee's own shape-preserving kernel
        # and Phi edges have stated their current result; applying the return
        # first can copy a stale pre-broadcast extent into every region and
        # then fight the local writer on the next round.
        changed = settle_shape_preserving_value_metadata(module)
        changed |= propagate_repository_ssa_call_metadata(module)
        changed_any |= changed
        guard.round(changed)
        if not changed:
            return changed_any


def _post_scalar_kernel_input_conversion(
    book: Any, function: Any, block_name: str, consumer_id: int,
    operand_position: int, source: SSAValue, converted: SSAValue,
    callee: str = "binary_scalar_double",
) -> None:
    """One ``kernel_input_conversion`` row for a promoted input of
    ``callee``. Scalar promotions and the counted scatter values span use
    the same source/converted identity row: (function scope, block,
    consumer result, operand position) -> (source id, converted id, source
    dtype, target dtype, callee), REVISE, DERIVED from the source's
    ``ssa_value`` cell and the converted value's (minted by ``fresh``);
    a row already stating this fact is not restated."""

    from .concordance_declarations import (
        KERNEL_INPUT_CONVERSION, TENSOR_SSA_LOWERING,
    )
    from .identity_concordance import Derived, Mode
    from .ssa_record_return_state import (
        function_scope_of, ssa_value_identity_cell,
    )

    cell = ssa_value_identity_cell(function, source.id, book=book)
    row = (
        function_scope_of(function), str(block_name), int(consumer_id),
        int(operand_position),
    )
    fact = (
        int(source.id), int(converted.id), str(source.dtype), "float64",
        str(callee),
    )
    if cell is None:
        raise ValueError(
            f"{KERNEL_INPUT_CONVERSION.name}: no source cell for {row!r} -- "
            "the source value has no ssa_value cell on the book"
        )
    stored = book.pages.get(KERNEL_INPUT_CONVERSION.name)
    if stored is not None and stored.latest(row) == fact:
        return
    book.post(
        KERNEL_INPUT_CONVERSION, row, fact, stage=TENSOR_SSA_LOWERING,
        provenance=Derived(tuple(dict.fromkeys(
            found for found in (
                cell,
                ssa_value_identity_cell(function, converted.id, book=book),
            )
            if found is not None
        ))),
        mode=Mode.REVISE,
    )


def lower_tensor_calls_to_repository_ssa(
    module: IRModule,
    reference: SSATensorCodeReference,
    *, finalize_private_owners: bool = True,
) -> tuple[TensorSSALoweringShortfall, ...]:
    """Replace tensor calls with fully explicit calls into ``reference``.

    Static shapes are converted to ordinary integer/vector constants. Metadata
    views become SSA aliases. If a runtime extent is genuinely absent, the call
    remains visible and a precise shortfall is returned instead of compiling it
    as a one-element tensor.
    """

    # A keyed Tensor child is physically selected in the control function and
    # numerically consumed in a planned region.  The lookup commits its span
    # identities to the shared concordance after the region was first built;
    # transfer that exact contract onto every occurrence before tensor
    # recognition.  This is identity transport, not shape inference.
    book = identity_book(module)
    shape_page = book.pages.get("tensor_shape_concordance")
    precision_region_page = book.pages.get(
        "source_precision_region_concordance"
    )

    def is_concorded_precision_operator(instruction: Instr) -> bool:
        """Whether this operation belongs to an indivisible Precision region.

        Precision arithmetic is expanded by ``apply_precision_pipeline`` after
        repository tensor lowering. Treating one of its shaped operations as
        an ordinary tensor opcode here severs the region: for example,
        ``precision_sqrt`` becomes ``unary_double`` and can no longer consume
        or produce the region's limb representation. The source concordance,
        rather than an opcode allow-list, owns the exception.
        """

        if (
            not instruction.attributes.get("python_precision_operator")
            or instruction.res is None
            or precision_region_page is None
        ):
            return False
        raw_row = instruction.attributes.get("source_precision_region")
        if not isinstance(raw_row, (tuple, list)) or len(raw_row) != 2:
            return False
        row = (str(raw_row[0]), int(raw_row[1]))
        fact = precision_region_page.latest(row)
        if not isinstance(fact, tuple) or len(fact) != 4:
            return False
        members = tuple(map(int, fact[0]))
        return int(instruction.res.id) in members

    # Seed formals from the call graph so the metadata fixed point can lower
    # operations whose callee-local occurrences began empty.  These are
    # explicitly provisional concordance entries: source-linked result
    # contracts can become more precise during the fixed point below, and the
    # same edge rows are republished afterwards before lowering consumes them.
    provisional_call_shapes = _publish_exact_ssa_call_shapes(module)
    _settle_exact_ssa_call_shapes(module, provisional_call_shapes)

    if shape_page is not None:
        contract_keys = (
            "program_abi_storage",
            "program_abi_rank",
            "tensor_metadata_state",
            "tensor_shape_value_id",
            "tensor_rank_value_id",
            "tensor_element_count_value_id",
        )
        for function_name, function in module.functions.items():
            owner_name = str(function.metadata.get(
                "tensor_shape_concordance_scope"
            ) or str(function_name).split("__planned_region_", 1)[0])
            occurrences = [*function.args]
            for block in function.blocks.values():
                for instruction in block.instrs:
                    occurrences.extend(instruction.args)
                    if instruction.res is not None:
                        occurrences.append(instruction.res)
            for value in occurrences:
                fact = shape_page.latest((owner_name, int(value.id)))
                if not isinstance(fact, Mapping):
                    continue
                transferred = {
                    key: fact[key]
                    for key in contract_keys
                    if fact.get(key) is not None
                }
                if transferred:
                    value.accounting = {
                        **dict(value.accounting or {}),
                        **transferred,
                    }
                    # Program: a keyed Tensor child selected in the control
                    # function and consumed in a planned region (history_
                    # projection ``tick``: span rank 3 Loads).  Wrong record:
                    # the shape concordance row's storage/rank reached
                    # ``accounting`` only; the value's ``ssa_value`` row kept
                    # the defaults.  Rule: the contract read from that row is
                    # posted on the occurrence's row, DERIVED from the
                    # concordance cell it was read from; accounting is then
                    # the view.
                    from .concordance_declarations import (
                        SSA_VALUE_PHYSICAL_STORAGES as _TRANSFER_STORAGES,
                        TENSOR_SHAPE_CONCORDANCE as _TRANSFER_PAGE,
                        TENSOR_SSA_LOWERING as _TRANSFER_STAGE,
                    )
                    from .ssa_record_return_state import (
                        revise_ssa_value as _revise_transfer,
                    )
                    _transfer_physical = {}
                    if transferred.get("program_abi_storage") in _TRANSFER_STORAGES:
                        _transfer_physical["storage"] = transferred[
                            "program_abi_storage"
                        ]
                    if transferred.get("program_abi_rank") is not None:
                        _transfer_physical["rank"] = int(
                            transferred["program_abi_rank"]
                        )
                    if _transfer_physical:
                        _revise_transfer(
                            function, int(value.id), dtype=value.dtype,
                            shape=tuple(value.shape or ()),
                            stage=_TRANSFER_STAGE,
                            sources=(book.latest_ref(
                                _TRANSFER_PAGE, (owner_name, int(value.id)),
                            ),),
                            book=book, physical=_transfer_physical,
                        )

        # The metadata identities are produced beside the selected child span
        # in the control function.  Make them ordinary call operands/formals
        # at the same boundary.  Keeping an id only in accounting would name a
        # value the planned function cannot read; the concordance contract is
        # complete only when its shape/rank/count values cross the call too.
        # ``Tensor.real`` of a non-complex tensor is an identity view: its
        # result is the source's dtype and tensor contract.  Settle that on
        # every occurrence of the result in the source function's regions now,
        # so a consumer region's formal receives the same extents below.
        for view_function_name, view_function in module.functions.items():
            for view_block in view_function.blocks.values():
                for view_instr in view_block.instrs:
                    if (
                        str(view_instr.op).casefold() != "real"
                        or view_instr.res is None
                        or len(view_instr.args) != 1
                    ):
                        continue
                    view_source = view_instr.args[0]
                    view_root = str(view_function_name).split(
                        "__planned_region_"
                    )[0]
                    # The source's proven dtype and tensor contract live on
                    # its producing occurrence in the source function.
                    family = tuple(
                        member
                        for sibling_name, sibling in module.functions.items()
                        if str(sibling_name).split("__planned_region_")[0]
                        == view_root
                        for member in (
                            *sibling.args,
                            *(
                                m for block in sibling.blocks.values()
                                for instr in block.instrs
                                for m in (*instr.args, instr.res)
                                if m is not None
                            ),
                        )
                        if int(member.id) == int(view_source.id)
                    )
                    view_dtypes = {
                        str(member.dtype) for member in family
                        if member.dtype not in {None, "unknown"}
                    }
                    if len(view_dtypes) != 1:
                        continue
                    view_dtype = next(iter(view_dtypes))
                    if view_dtype.casefold().startswith(
                        ("complex", "c64", "c128")
                    ):
                        continue
                    view_contract = {}
                    for member in family:
                        for key in _TENSOR_CONTRACT_KEYS:
                            if key in (member.accounting or {}):
                                view_contract.setdefault(
                                    key, member.accounting[key]
                                )
                    for sibling_name, sibling in module.functions.items():
                        if str(sibling_name).split("__planned_region_")[0] != view_root:
                            continue
                        for value in (
                            *sibling.args,
                            *(
                                member
                                for block in sibling.blocks.values()
                                for instr in block.instrs
                                for member in (*instr.args, instr.res)
                                if member is not None
                            ),
                        ):
                            if int(value.id) != int(view_instr.res.id):
                                continue
                            if value.dtype in {None, "unknown"}:
                                value.dtype = view_dtype
                            value.accounting = {
                                **view_contract,
                                **dict(value.accounting or {}),
                            }
        function_values: dict[str, dict[int, SSAValue]] = {}
        for function_name, function in module.functions.items():
            occurrences = [*function.args]
            for block in function.blocks.values():
                for instruction in block.instrs:
                    occurrences.extend(instruction.args)
                    if instruction.res is not None:
                        occurrences.append(instruction.res)
            function_values[str(function_name)] = {
                int(value.id): value for value in occurrences
            }
        metadata_keys = (
            "tensor_shape_value_id",
            "tensor_rank_value_id",
            "tensor_element_count_value_id",
        )
        for callee_name, callee in module.functions.items():
            if "__planned_region_" not in str(callee_name):
                continue
            original_formals = tuple(callee.args)
            metadata_formals: list[SSAValue] = []
            for formal in original_formals:
                accounting = dict(formal.accounting or {})
                if accounting.get("program_abi_storage") != "span":
                    continue
                for key in metadata_keys:
                    value_id = accounting.get(key)
                    if value_id is None or any(
                        int(existing.id) == int(value_id)
                        for existing in (*callee.args, *metadata_formals)
                    ):
                        continue
                    metadata_formals.append(SSAValue(
                        int(value_id), dtype="int32",
                        accounting={
                            "tensor_extent_for": int(formal.id),
                            "tensor_extent_kind": key.removeprefix(
                                "tensor_"
                            ).removesuffix("_value_id"),
                        },
                    ))
            # A metadata identity may have been minted inside this planned
            # region while an earlier lowering attempt was still resolving
            # its source sequence.  Such an identity is local computation,
            # not a call-frame value.  Promote it to the ABI only when every
            # exact incoming call already owns the same identity; otherwise
            # the operation below reconstructs metadata from the concorded
            # live sequence length and row layout inside the region.
            incoming_caller_values = tuple(
                function_values[str(caller_name)]
                for caller_name, caller in module.functions.items()
                if any(
                    instruction.op in {"Call", "call"}
                    and instruction.attributes.get("callee") == callee_name
                    for block in caller.blocks.values()
                    for instruction in block.instrs
                )
            )
            if incoming_caller_values:
                metadata_formals = [
                    formal for formal in metadata_formals
                    if all(
                        int(formal.id) in caller_values
                        for caller_values in incoming_caller_values
                    )
                ]
            if not metadata_formals:
                continue
            callee.args.extend(metadata_formals)
            integral = dict(callee.metadata.get(
                "source_region_integral", {}
            ))
            if integral:
                integral["capture_value_ids"] = tuple(dict.fromkeys((
                    *map(int, integral.get("capture_value_ids", ())),
                    *(int(value.id) for value in metadata_formals),
                )))
                callee.metadata["source_region_integral"] = integral
            from .concordance_declarations import (
                SSA_VALUE, TENSOR_METADATA_CALL_BINDING,
                TENSOR_SHAPE_CONCORDANCE, TENSOR_SSA_LOWERING,
            )
            from .identity_concordance import Derived, Mode
            from .ssa_record_return_state import (
                function_scope_of, ssa_value_identity_cell,
            )
            book = identity_book(module)
            for caller_name, caller in module.functions.items():
                caller_values = function_values[str(caller_name)]
                for block_name, block in caller.blocks.items():
                    for instruction_index, instruction in enumerate(
                        block.instrs
                    ):
                        if (
                            instruction.op not in {"Call", "call"}
                            or instruction.attributes.get("callee")
                            != callee_name
                        ):
                            continue
                        actuals = []
                        for formal in metadata_formals:
                            actual = caller_values.get(int(formal.id))
                            if actual is None:
                                raise ValueError(
                                    "tensor shape concordance names unavailable "
                                    f"caller value %{formal.id} for "
                                    f"{caller_name} -> {callee_name}"
                                )
                            actuals.append(actual)
                        instruction.args.extend(actuals)
                        attributes = dict(instruction.attributes)
                        attributes["feed_ids"] = tuple(dict.fromkeys((
                            *map(int, attributes.get("feed_ids", ())),
                            *(int(value.id) for value in actuals),
                        )))
                        attributes["feed_shapes"] = tuple((
                            *attributes.get("feed_shapes", ()),
                            *(tuple(value.shape or ()) for value in actuals),
                        ))
                        attributes["feed_dtypes"] = tuple((
                            *attributes.get("feed_dtypes", ()),
                            *(str(value.dtype or "int32") for value in actuals),
                        ))
                        instruction.attributes = attributes
                        caller_scope = function_scope_of(caller)
                        callee_scope = function_scope_of(callee)
                        row = (
                            caller_scope, str(caller_name), str(block_name),
                            int(instruction.res.id), str(callee_name),
                        )
                        sources = tuple(dict.fromkeys(
                            cell for cell in (
                                ssa_value_identity_cell(
                                    caller, int(instruction.res.id), book=book,
                                ),
                                *(
                                    ssa_value_identity_cell(
                                        caller, int(actual.id), book=book,
                                    )
                                    for actual in actuals
                                ),
                                *(
                                    ssa_value_identity_cell(
                                        callee, int(formal.id), book=book,
                                    )
                                    for formal in metadata_formals
                                ),
                                *(
                                    book.latest_ref(
                                        TENSOR_SHAPE_CONCORDANCE,
                                        (str(caller_name), int(actual.id)),
                                    )
                                    for actual in actuals
                                ),
                                book.latest_ref(
                                    TENSOR_METADATA_CALL_BINDING, row,
                                ),
                            )
                            if cell is not None
                        ))
                        if not sources:
                            raise ValueError(
                                "tensor metadata call binding has no "
                                f"concordance cells for {caller_name} -> "
                                f"{callee_name}"
                            )
                        book.post(
                            TENSOR_METADATA_CALL_BINDING,
                            row,
                            (
                                tuple(int(value.id) for value in actuals),
                                tuple(
                                    int(value.id)
                                    for value in metadata_formals
                                ),
                            ),
                            stage=TENSOR_SSA_LOWERING,
                            provenance=Derived(sources),
                            mode=Mode.REVISE,
                        )

    # Direct source compilation and autograd both consume the same repository
    # SSA.  Region calls must therefore settle their exact actual/formal and
    # return metadata here, at the common tensor boundary, rather than relying
    # on one caller to remember a private prelude.  All three operations are
    # idempotent; autograd's historical explicit calls remain harmless.
    settle_shape_only_repository_returns(module)
    # Resolve facts owned by instructions/identity before the first call-edge
    # pass.  Specialized-formal settlement is exact on its first round; if it
    # runs first, an empty downstream placeholder can overwrite a literal
    # reshape result before that result has stated its extents.
    settle_static_repository_view_shapes(module)
    settle_canonical_value_metadata(module)
    settle_shape_preserving_value_metadata(module)
    wire_repository_ssa_region_products(module)
    settle_repository_ssa_shape_metadata(module)

    # Publish call-edge shapes only after source-linked results and canonical
    # metadata have reached their fixed point.  Recording them earlier stores
    # the placeholder a call happened to carry while its real result contract
    # was still in flight.  The callee-qualified concordance rows retain every
    # transition and avoid collapsing a planned-region formal into the
    # authored helper's (potentially polymorphic) value-id namespace.
    exact_call_shapes = _publish_exact_ssa_call_shapes(module)
    _settle_exact_ssa_call_shapes(module, exact_call_shapes)

    linked_roots: set[str] = set()
    shortfalls: list[TensorSSALoweringShortfall] = []

    def link_repository_function(name: str, function: Any) -> None:
        """Run the incumbent repository identity writer on one selected leaf."""
        from ..common.tensors.accelerator_backends.llvm_repository_ssa import (
            post_repository_kernel_identity,
        )

        existing = module.functions.get(name)
        if existing is not None and existing is not function:
            raise ValueError(f"repository SSA function collision for {name!r}")
        module.functions[name] = function
        post_repository_kernel_identity(function, identity_book(module))
        reference_table = getattr(reference.module, "tensor_tables", {}).get(name)
        if reference_table is not None:
            existing_table = module.tensor_tables.get(name)
            if existing_table is not None and existing_table != reference_table:
                raise ValueError(f"repository SSA tensor-table collision for {name!r}")
            module.tensor_tables[name] = reference_table

    #: ``(function, instruction)`` being lowered: every ``fresh`` value is
    #: made for that tensor operation, inside that function.
    lowering_at: list[Any] = [None, None]

    def recorded_dtype(value: SSAValue, role: str) -> str:
        if not value.dtype:
            owner, source = lowering_at
            raise ValueError(
                f"tensor lowering of {None if source is None else source.op!r}"
                f": {role} %t{int(value.id)} has no recorded dtype; the "
                "pipeline never settled its type"
            )
        return str(value.dtype)

    def fresh(*, shape=(), dtype: str | None, physical=None) -> SSAValue:
        if not dtype:
            raise ValueError(
                "tensor lowering minted a value with no dtype for an "
                f"operation lowered at {lowering_at[1]!r}"
            )
        # Minted through the book (plan 100, 4.1 item 2): a NOVEL
        # ``ssa_value`` row under the function's scope, from the cell of the
        # tensor operation being lowered (its result's, else its first
        # argument's that has one), else the function root.  The id comes
        # from the same monotonic source the bare mint used, so the emitted
        # text is unchanged; what changes is that a kernel's opcode / shape
        # constant in a planned region now has an identity cell.
        from .concordance_declarations import (
            TENSOR_LOWERING_VALUE, TENSOR_SSA_LOWERING,
        )
        from .precompile_to_ssa import _function_root_cell, _mint_ssa_id
        from .ssa_record_return_state import (
            function_scope_of, ssa_value_identity_cell,
        )

        owner, source = lowering_at
        scope = function_scope_of(owner)
        cell = None
        if source is not None:
            for value in (
                *((source.res,) if source.res is not None else ()),
                *source.args,
            ):
                cell = ssa_value_identity_cell(owner, value.id, book=book)
                if cell is not None:
                    break
        if cell is None:
            cell = _function_root_cell(book, scope)
        return SSAValue(
            _mint_ssa_id(
                book, scope, TENSOR_LOWERING_VALUE, (cell,),
                dtype=dtype, shape=tuple(shape), stage=TENSOR_SSA_LOWERING,
                physical=physical,
            ),
            dtype=dtype,
            shape=tuple(shape),
            accounting=(
                {"physical_dtype": physical["physical_dtype"]}
                if physical and physical.get("physical_dtype") else {}
            ),
        )

    def constant(value: Any, dtype: str, *, physical=None) -> tuple[SSAValue, Instr]:
        result = fresh(dtype=dtype, physical=physical)
        record_geometry_literal(result, value)
        return result, Instr(Handler.Const.value, [], result, attributes={"constant": value})

    def int_vector(values: Iterable[int], *, physical=None) -> tuple[SSAValue, Instr]:
        sequence = tuple(int(value) for value in values)
        result = fresh(shape=(len(sequence),), dtype="int32", physical=physical)
        record_geometry_literal(result, sequence)
        return result, Instr(
            Handler.Const.value, [], result,
            attributes={"values": sequence, "constant": None},
        )

    def record_geometry_literal(value: SSAValue, literal: Any) -> None:
        """Retain explicitly typed geometry even when its Const is pruned."""
        from .identity_concordance import Derived, Mode
        from .concordance_declarations import (
            SSA_CONSTANT_MATERIALIZATION, TENSOR_SSA_LOWERING,
        )
        from .ssa_record_return_state import (
            function_scope_of, posted_ssa_value_fact, ssa_value_identity_cell,
        )

        owner, _source = lowering_at
        fact = posted_ssa_value_fact(owner, value.id, book=book)
        if (fact is None or fact.dtype != "int32"
                or fact.physical_dtype != "int32"
                or (fact.storage, fact.rank) not in {
                    ("scalar", 0), ("span", 1),
                }):
            return
        # weighted_cast_source_probe's broadcast count8 is used by its
        # concordance geometry but not its native Call, so DCE removes the
        # Const. The literal and explicit int32 domain belong to its minted
        # identity; preserving them here serves every typed geometry reader.
        cell = ssa_value_identity_cell(owner, value.id, book=book)
        if cell is None:
            raise ValueError("typed geometry literal lacks its SSA identity")
        book.post(
            SSA_CONSTANT_MATERIALIZATION,
            (function_scope_of(owner), owner.name, str(block_name), int(value.id)),
            ("Const", "int32", repr(literal)), stage=TENSOR_SSA_LOWERING,
            provenance=Derived((cell,)), mode=Mode.CONCORD,
        )

    def call(
        callee: str,
        arguments: list[SSAValue],
        result: SSAValue,
        source: Instr,
        *,
        output_argument: int | None = None,
    ) -> Instr:
        # The imported C/LLVM tensor kernels use double-backed buffers even
        # when a value's semantic dtype is Boolean (comparisons and masks are
        # represented as 0.0/1.0). Preserve that physical ABI on every shaped
        # Boolean occurrence crossing one of these calls. Without it, native C
        # allocates one byte per element and ``broadcast_double``/``where``
        # read or write eight, corrupting the activation heap.
        # Rule: post first; accounting is a view.  Each shaped Boolean's row
        # is revised with the double-backed physical dtype, DERIVED from the
        # linked kernel's formal cell at that position (the output argument's
        # for the result) and the value's own previous cell; the accounting
        # is then set FROM that fact (a value with no row keeps the literal).
        from .concordance_declarations import TENSOR_SSA_LOWERING as _KERNEL_STAGE
        from .ssa_record_return_state import (
            posted_ssa_value_fact as _posted_kernel_fact,
            revise_ssa_value as _revise_kernel_value,
            ssa_value_identity_cell as _kernel_identity_cell,
        )
        kernel = module.functions.get(callee)
        if kernel is None:
            kernel = reference.module.functions.get(callee)
            if kernel is not None:
                link_repository_function(callee, kernel)
        if kernel is not None:
            from .ssa_call_input_adapters import _infer_pointer_storage
            _infer_pointer_storage({callee: kernel})
        kernel_owner = lowering_at[0]
        kernel_book = identity_book(module)
        int64_buffer_positions = {
            "binary_int64": frozenset({0, 1, 2}),
            "binary_scalar_int64": frozenset({0, 2}),
            "broadcast_int64": frozenset({0, 1}),
            "create_arange_int64": frozenset({3}),
            "cast_double_to_int64_values": frozenset({1}),
        }.get(callee, frozenset())
        for position, value in enumerate((*arguments, result)):
            typed_kernel_physical = (
                "int64" if position in int64_buffer_positions else None
            )
            if (
                (tuple(value.shape or ()) or (
                    typed_kernel_physical is not None
                    and (value.accounting or {}).get("tensor_metadata_state")
                    == "dynamic"
                ))
                and (
                    typed_kernel_physical is not None
                    or str(value.dtype or "").casefold() in {"bool", "i1"}
                )
            ):
                formal_position = (
                    position if position < len(arguments) else output_argument
                )
                formal = (
                    kernel.args[formal_position]
                    if kernel is not None and formal_position is not None
                    and 0 <= formal_position < len(kernel.args) else None
                )
                formal_cell = (
                    None if formal is None else _kernel_identity_cell(
                        kernel, int(formal.id), book=kernel_book,
                    )
                )
                formal_fact = (
                    None if formal is None else _posted_kernel_fact(
                        kernel, int(formal.id), book=kernel_book,
                    )
                )
                expected_physical = typed_kernel_physical or "float64"
                if (formal_cell is None or formal_fact is None
                        or formal_fact.physical_dtype != expected_physical):
                    raise ValueError(
                        "selected tensor kernel lacks exact typed formal proof"
                    )
                current = (
                    None if kernel_owner is None else _posted_kernel_fact(
                        kernel_owner, int(value.id), book=kernel_book,
                    )
                )
                # Kernel input formals state what bytes they require; they
                # cannot turn a caller's existing float64 cast buffer into
                # int64 storage. A missing input proof remains missing for
                # the adapter/physical-call gate to resolve or refuse.
                is_output = position == output_argument or position == len(arguments)
                if (not is_output and current is not None
                        and current.physical_dtype is not None
                        and current.physical_dtype != expected_physical):
                    raise ValueError(
                        "selected tensor kernel input physical dtype conflicts with source"
                    )
                if kernel_owner is not None and is_output:
                    _revise_kernel_value(
                        kernel_owner, int(value.id), dtype=value.dtype,
                        shape=tuple(value.shape or ()), stage=_KERNEL_STAGE,
                        sources=(formal_cell,),
                        book=kernel_book,
                        physical={"physical_dtype": expected_physical},
                    )
                value.accounting = _accounting_view(kernel_owner, value, module)
        linked_roots.add(callee)
        attributes = {
            key: value for key, value in source.attributes.items()
            if key not in {
                "tensor_operation", "tensor", "tensor_candidate",
                "callee", "lowered_from"
            }
        }
        attributes["callee"] = callee
        if output_argument is not None:
            attributes["ssa_output_argument"] = output_argument
        return Instr(
            Handler.Call.value, arguments, result, attributes=attributes,
            source_span=source.source_span,
        )

    for function_name, function in tuple(module.functions.items()):
        if not hasattr(module, "tensor_tables"):
            module.tensor_tables = {}
        tensor_table = module.tensor_tables.setdefault(
            function_name, SSATensorTable()
        )
        shape_polymorphic_function = _shape_polymorphic_function(
            function_name
        )
        # A function's own formals are the same split-identity hazard as
        # any operand: a planned region carved out of this function, or a
        # caller-side linking pass, may have already proven this identity's
        # true shape onto a DIFFERENT SSAValue object sharing the same id
        # under this function's authored name. Settle every formal from
        # the concordance before anything below reads its .shape to decide
        # a static count -- the exact read that baked a wrong element
        # count from a formal whose true rank was already proven
        # elsewhere, just not onto this particular object.
        numeric_scope = (function.metadata or {}).get("source_numeric_scope")
        source_owner = str(
            ((function.metadata or {}).get("source_region_integral") or {}).get(
                "owner"
            )
            or numeric_scope
            or function_name
        )
        sequence_table = (
            getattr(module, "sequence_tables", {}).get(function_name)
            or getattr(module, "sequence_tables", {}).get(source_owner)
        )

        def resident_sequence(value: SSAValue):
            """The exact sequence descriptor naming this arena identity."""

            if sequence_table is None:
                return None
            accounting = dict(value.accounting or {})
            storage_view = accounting.get("ssa_storage_view") or {}
            from .identity_concordance import current_identity_book

            concordance = current_identity_book().page(
                "control_value_concordance"
            )
            candidate_ids = {
                int(value.id),
                int(concordance.resolve_alias(source_owner, int(value.id))),
                *(
                    (int(accounting["ssa_storage_alias"]),)
                    if accounting.get("ssa_storage_alias") is not None else ()
                ),
                *(
                    (int(storage_view["storage_value_id"]),)
                    if storage_view.get("storage_value_id") is not None else ()
                ),
            }
            descriptors = {
                descriptor.sequence_id: descriptor
                for candidate_id in candidate_ids
                for descriptor in (sequence_table.by_id(candidate_id),)
                if descriptor is not None
            }
            for candidate_id in candidate_ids:
                descriptors.update({
                    int(sequence_id): sequence_table.by_id(int(sequence_id))
                    for sequence_id, role
                    in sequence_table.member_claims(candidate_id)
                    if tuple(role) == ("column", 0)
                    and sequence_table.by_id(int(sequence_id)) is not None
                })
            if len(descriptors) != 1:
                return None
            return next(iter(descriptors.values()))

        def owner_tensor_descriptor(value: SSAValue | int):
            """A unanimous descriptor published by this source integral.

            Planned numerical regions are separate repository functions but
            their receipt gives them one source owner and their value ids are
            identities in that owner's concordance.  A producer region may
            therefore publish the tensor view before a later consumer region
            sees the same id as a formal.  Do not compare unrelated local ids:
            only tables whose function receipt names this exact owner enter.
            """

            candidates = []
            for table_name, table in getattr(
                module, "tensor_tables", {}
            ).items():
                table_function = module.functions.get(str(table_name))
                if table_function is None:
                    continue
                receipt = (
                    table_function.metadata or {}
                ).get("source_region_integral") or {}
                table_owner = str(
                    receipt.get("owner") or table_function.name
                )
                if table_owner != source_owner:
                    continue
                descriptor = table.by_id(int(value.id if isinstance(value, SSAValue) else value))
                if descriptor is not None:
                    candidates.append(descriptor)
            contracts = {
                (
                    int(descriptor.data_value_id),
                    str(descriptor.dtype),
                    tuple(descriptor.shape),
                    tuple(descriptor.strides),
                    str(descriptor.metadata_state),
                )
                for descriptor in candidates
            }
            if len(contracts) != 1:
                return None
            return candidates[0]

        def sequence_outer_extent(sequence) -> int | None:
            """Return the exact static producer extent for one sequence."""

            owner_function = module.functions.get(source_owner)
            if owner_function is None:
                return None
            authored_bounds = {
                int(sequence_id): int(extent)
                for sequence_id, extent in (
                    owner_function.metadata or {}
                ).get("sequence_static_capacity_bounds", ())
            }
            if int(sequence.sequence_id) in authored_bounds:
                return authored_bounds[int(sequence.sequence_id)]
            source_ids = {
                int(source_id)
                for sequence_id, source_id in (
                    owner_function.metadata or {}
                ).get("sequence_capacity_sources", ())
                if int(sequence_id) == int(sequence.sequence_id)
            }
            if len(source_ids) != 1:
                return None
            source_id = next(iter(source_ids))
            extents: set[int] = set()
            for candidate_function in module.functions.values():
                receipt = (
                    candidate_function.metadata or {}
                ).get("source_region_integral") or {}
                candidate_owner = str(
                    receipt.get("owner") or candidate_function.name
                )
                if candidate_owner != source_owner:
                    continue
                values = list(candidate_function.args)
                for candidate_block in candidate_function.blocks.values():
                    for candidate_instruction in candidate_block.instrs:
                        values.extend(candidate_instruction.args)
                        if candidate_instruction.res is not None:
                            values.append(candidate_instruction.res)
                extents.update(
                    int(value.shape[0])
                    for value in values
                    if int(value.id) == source_id and tuple(value.shape or ())
                )
            for table_name, table in getattr(
                module, "tensor_tables", {}
            ).items():
                candidate_function = module.functions.get(str(table_name))
                if candidate_function is None:
                    continue
                receipt = (
                    candidate_function.metadata or {}
                ).get("source_region_integral") or {}
                candidate_owner = str(
                    receipt.get("owner") or candidate_function.name
                )
                if candidate_owner != source_owner:
                    continue
                descriptor = table.by_id(source_id)
                if descriptor is not None and tuple(descriptor.shape):
                    extents.add(int(descriptor.shape[0]))
            if len(extents) != 1:
                return None
            return next(iter(extents))
        _settle_operand_shapes(
            function, function.args, owner_tensor_descriptor,
        )
        # In the generated bw_tanh VJP, planned-region formal %3 is the
        # leading data operand of tanh and has neither extents nor a scalar
        # receipt during section lowering.  Treating that absence as rank
        # zero freezes tanh, square and subtraction at one element, although
        # its exact call feed later proves (2, 2).
        #
        # Defer only when that unresolved formal is a tensor operation's
        # leading data operand (or one of the three data operands of where or
        # scatter).  Constructor shape/dtype/fill formals and operation
        # metadata are scalars by contract; deferring the whole region for
        # those structural formals prevents the same pass from settling its
        # real tensor operands and leaves downstream shaped arithmetic scalar.
        if (function.metadata or {}).get("source_region_integral"):
            scalar_formals = set(map(int, (
                (function.metadata or {}).get("scalar_variant_argument_ids")
                or ()
            )))
            # In compact_codewords, symbol_count * max_bits shares a region
            # with tensor arithmetic after arange.  Its formal symbol_count
            # has an exact scalar SSA_VALUE receipt but empty accounting;
            # ignoring that receipt deferred the whole tensor region.  Read
            # the function-scoped physical fact rather than guessing from an
            # empty shape or requiring a second accounting copy of it.
            from .ssa_record_return_state import posted_ssa_value_fact

            scalar_formals.update(
                int(formal.id) for formal in function.args
                if (fact := posted_ssa_value_fact(
                    function, int(formal.id), book=book,
                )) is not None
                and fact.physical_dtype is not None
                and fact.storage == "scalar" and fact.rank == 0
            )
            unresolved_formals = {
                int(formal.id)
                for formal in function.args
                if (
                not tuple(formal.shape or ())
                and (formal.accounting or {}).get("tensor_extent_for") is None
                and int(formal.id) not in scalar_formals
                and (formal.accounting or {}).get("program_abi_storage")
                is None
                and (formal.accounting or {}).get("exact_region_feed_source")
                is None
                )
            }
            unresolved_data_formals = set()
            constructor_operations = {
                "arange", "empty", "fill", "full", "ones", "zeros",
            }
            for block in function.blocks.values():
                for instruction in block.instrs:
                    operation = (
                        instruction.attributes.get("tensor_operation")
                        or instruction.attributes.get("tensor")
                        or instruction.attributes.get("tensor_candidate")
                        or _SHAPED_SSA_OPERATIONS.get(instruction.op)
                    )
                    if operation is None or str(operation) in constructor_operations:
                        continue
                    data_positions = (
                        range(min(3, len(instruction.args)))
                        if str(operation) in {"scatter", "where"}
                        else range(min(1, len(instruction.args)))
                    )
                    unresolved_data_formals.update(
                        int(instruction.args[position].id)
                        for position in data_positions
                        if int(instruction.args[position].id)
                        in unresolved_formals
                    )
            if unresolved_data_formals:
                continue
        function_argument_ids = {int(value.id) for value in function.args}
        unresolved_argument_ids = {
            int(value.id)
            for value in function.args
            if value.dtype is None and not tuple(value.shape)
        }

        def shape_unknown(value: SSAValue) -> bool:
            """True only when a value's shape is genuinely not known.

            An EMPTY shape with a known dtype is a rank-0 scalar, a fully
            static shape (``alpha = (microstep + 1.0) / microstep_count``).
            Treating every empty shape as "unknown" routed such scalars
            through runtime broadcasting with minted extent queries, and the
            dynamic state then cascaded into every elementwise result after
            them, none of which had a public extent origin at emission.
            """

            # compact_codewords' mixed Mul keeps the display shape (1, 8)
            # while its booked descriptor carries runtime row geometry.
            # That descriptor controls reduction inputs and output extent;
            # the display tuple cannot turn it into a fixed one-row tensor.
            existing = tensor_table.by_id(int(value.id))
            if existing is not None and existing.metadata_state != "static":
                return True
            if tuple(value.shape or ()):
                return False
            from .ssa_record_return_state import posted_ssa_value_fact
            physical = posted_ssa_value_fact(
                function, int(value.id), book=book,
            )
            if (physical is not None and physical.storage == "span"
                    and isinstance(physical.rank, int)
                    and physical.rank > 0):
                # _code_lengths' public octet selector has display (), but
                # its exact SSA row says span/rank1. It has no scalar count.
                return True
            if value.dtype is None or int(value.id) in unresolved_argument_ids:
                return True
            return False

        def booked_element_count(value: SSAValue) -> int | None:
            descriptor = tensor_table.by_id(int(value.id))
            if descriptor is not None and descriptor.metadata_state != "static":
                return None
            return _booked_count(function, value, book)

        def scatter_derived_index_count(
            indices: SSAValue, values: SSAValue,
            pending: list[Instr],
        ) -> SSAValue | None:
            """Read a values count proved to be the selector's count."""
            from .concordance_declarations import (
                TENSOR_ELEMENTWISE_EXTENT_BINDING,
                TENSOR_METADATA_RESULT_BINDING,
            )
            from .ssa_record_return_state import (
                function_scope_of, posted_ssa_value_fact,
                ssa_value_identity_cell,
            )

            selector_fact = posted_ssa_value_fact(
                function, int(indices.id), book=book,
            )
            values_fact = posted_ssa_value_fact(
                function, int(values.id), book=book,
            )
            if (selector_fact is None or selector_fact.storage != "span"
                    or selector_fact.rank != 1
                    or values_fact is None or values_fact.storage != "span"
                    or values_fact.rank != 1):
                return None
            projected = _dynamic_tensor_geometry_evidence(
                module, function, int(values.id), pending=pending,
            )
            if projected is None:
                return None
            result_refs = tuple(ref for ref in projected[1]
                if ref.page == TENSOR_METADATA_RESULT_BINDING
                and ref.row[0] == function_scope_of(function)
                and int(ref.row[3]) == int(values.id))
            if len(result_refs) != 1:
                return None
            result_ref = result_refs[0]
            producer = module.functions.get(str(result_ref.row[2]))
            target_source_cell = ssa_value_identity_cell(
                function, int(indices.id), book=book,
            )
            producer_source_cell = None if producer is None else ssa_value_identity_cell(
                producer, int(indices.id), book=book,
            )
            if (producer is None or target_source_cell is None
                    or target_source_cell != producer_source_cell):
                return None
            index_geometry = _dynamic_tensor_geometry_evidence(
                module, producer, int(indices.id),
            )
            values_geometry = _dynamic_tensor_geometry_evidence(
                module, producer, int(values.id),
            )
            result_fact = book.page(TENSOR_METADATA_RESULT_BINDING).cell(
                result_ref.row, result_ref.column,
            )
            if (
                index_geometry is None or values_geometry is None
                or tuple(value.id for value in index_geometry[0])
                    != tuple(value.id for value in values_geometry[0])
                or tuple(value.id for value in values_geometry[0])
                    != tuple(result_fact[1])
                or tuple(value.id for value in projected[0])
                    != tuple(result_fact[2])
                or not set(index_geometry[1]) <= set(values_geometry[1])
                or not any(ref.page == TENSOR_ELEMENTWISE_EXTENT_BINDING
                           for ref in values_geometry[1])
            ):
                return None
            return projected[0][2]

        def strides(shape: tuple[int, ...]) -> tuple[int, ...]:
            stride = 1
            result = []
            for extent in reversed(shape):
                result.append(stride)
                stride *= int(extent)
            return tuple(reversed(result))

        def register_tensor(
            value: SSAValue,
            *,
            storage: str,
            metadata_state: str | None = None,
            alias_of: int | None = None,
            data_value_id: int | None = None,
            extent_ids_from: SSATensorDescriptor | None = None,
            extent_values: tuple[SSAValue, SSAValue, SSAValue] | None = None,
            alias_proof: Any = None,
        ) -> SSATensorDescriptor:
            tensor_id = int(value.id)
            existing = tensor_table.by_id(tensor_id)
            if existing is not None:
                from .concordance_declarations import SOURCE_TENSOR_PREFIX_SLICE
                from .identity_concordance import Ref
                if (
                    isinstance(alias_proof, Ref)
                    and alias_proof.page == SOURCE_TENSOR_PREFIX_SLICE
                ):
                    _slice_fact = book.page(SOURCE_TENSOR_PREFIX_SLICE).cell(
                        alias_proof.row, alias_proof.column,
                    )
                    _source_owner = (
                        tensor_table.by_id(int(alias_of))
                        if alias_of is not None else None
                    )
                    if not (
                        book.latest_ref(
                            SOURCE_TENSOR_PREFIX_SLICE,
                            (alias_proof.row[0], tensor_id),
                        ) == alias_proof
                        and isinstance(_slice_fact, tuple)
                        and len(_slice_fact) == 3
                        and _slice_fact[0] == alias_of
                        and _source_owner is not None
                        and _source_owner.data_value_id == data_value_id
                        and existing.storage == "output"
                        and existing.data_value_id == tensor_id
                        and existing.alias_of is None
                        and existing.heap_owner_value_id is None
                        and existing.dtype == _source_owner.dtype
                        and existing.element_count_value_id is None
                    ):
                        raise ValueError(
                            "runtime prefix slice conflicts with incumbent tensor storage"
                        )
                    existing = dataclasses.replace(
                        existing, storage="view",
                        alias_of=_source_owner.tensor_id,
                        data_value_id=_source_owner.data_value_id,
                        arena_id=_source_owner.arena_id,
                        allocation_owner=_source_owner.allocation_owner,
                        owns_allocation=False,
                        element_offset=_source_owner.element_offset,
                        byte_offset=_source_owner.byte_offset,
                    )
                    tensor_table.tensors[tensor_id] = existing
                elif alias_proof is not None:
                    # _ascending_order's ensure_tensor requested view20 of
                    # raw19 after a placeholder output20 already existed.
                    # Returning that placeholder left data20 independently
                    # owned although NATIVE_TENSOR_WRAP proved data19. The
                    # exact wrapping receipt refines this same identity;
                    # it cannot replace an allocated carrier or other alias.
                    from .concordance_declarations import (
                        NATIVE_TENSOR_WRAP, TENSOR_OWNED_HEAP_BINDING,
                    )
                    from .identity_concordance import Ref
                    from .ssa_record_return_state import function_scope_of

                    scope = function_scope_of(function)
                    owner = tensor_table.by_id(alias_of) if alias_of is not None else None
                    wrapping = (book.page(NATIVE_TENSOR_WRAP).cell(
                        alias_proof.row, alias_proof.column,
                    ) if isinstance(alias_proof, Ref)
                        and alias_proof.page == NATIVE_TENSOR_WRAP else None)
                    if (owner is None or storage != "view"
                            or not isinstance(wrapping, tuple) or len(wrapping) != 5
                            or book.latest_ref(NATIVE_TENSOR_WRAP, (scope, tensor_id)) != alias_proof
                            or wrapping[1].row != (scope, int(alias_of))
                            or wrapping[3].row != (scope, tensor_id)
                            or data_value_id != owner.data_value_id
                            or existing.data_value_id not in {tensor_id, owner.data_value_id}
                            or existing.alias_of not in {None, alias_of}
                            or existing.heap_owner_value_id is not None
                            or book.latest_ref(TENSOR_OWNED_HEAP_BINDING,
                                               (scope, function.name, tensor_id)) is not None
                            or existing.arena_id not in {None, tensor_id, owner.arena_id}
                            or existing.allocation_owner not in {
                                None, tensor_id, owner.allocation_owner,
                            }
                            or any(
                                emitted.op == "Alloca" and emitted.res is not None
                                and int(emitted.res.id) == existing.data_value_id
                                for emitted_block in function.blocks.values()
                                for emitted in emitted_block.instrs
                            )
                            or existing.dtype != owner.dtype):
                        raise ValueError("native wrapper alias conflicts with incumbent storage")
                    existing = dataclasses.replace(
                        existing, data_value_id=owner.data_value_id,
                        storage="view", alias_of=owner.tensor_id,
                        arena_id=owner.arena_id, allocation_owner=owner.allocation_owner,
                        owns_allocation=False, element_offset=owner.element_offset,
                        byte_offset=owner.byte_offset,
                    )
                    tensor_table.tensors[tensor_id] = existing
                # _ascending_order's Neg17 was registered before Add16's
                # metadata arrived. A later shape-preserving request supplied
                # all three exact source16 extents, but this incumbent return
                # kept Neg17 unresolved. Settle the same descriptor from the
                # writer's explicit geometry; keep its carrier/owner/alias.
                if existing.metadata_state == "unresolved" and metadata_state == "dynamic":
                    requested = (
                        tuple(int(member.id) for member in extent_values)
                        if extent_values is not None else
                        tuple(getattr(extent_ids_from, field) for field in (
                            "shape_value_id", "rank_value_id", "element_count_value_id",
                        )) if extent_ids_from is not None
                        and extent_ids_from.metadata_state == "dynamic" else ()
                    )
                    from .ssa_record_return_state import ssa_value_identity_cell
                    if (len(requested) == 3
                            and all(type(identity) is int for identity in requested)
                            and all(ssa_value_identity_cell(function, identity, book=book) is not None
                                    for identity in requested)):
                        existing = dataclasses.replace(
                            existing, metadata_state="dynamic", byte_size=None,
                            shape_value_id=requested[0], rank_value_id=requested[1],
                            element_count_value_id=requested[2],
                        )
                        tensor_table.tensors[tensor_id] = existing
                return existing
            shape = tuple(map(int, value.shape))
            value_accounting = dict(value.accounting or {})
            declared_rank = int(
                value_accounting.get("program_abi_rank", 0) or 0
            )
            declared_dynamic = (
                str(value_accounting.get("tensor_metadata_state") or "")
                == "dynamic"
                or declared_rank > len(shape)
            )
            # The source-integral producer of compact_codewords' Mul already
            # owns dynamic extents when the later sum region first sees this
            # formal. Its (1, 8) display shape is not a fixed allocation.
            # Keep the formal unresolved until computed metadata crosses the
            # booked Call edge; the static byte-weight view remains static.
            incoming = (
                owner_tensor_descriptor(value)
                if storage == "input" and tensor_id in function_argument_ids
                else None
            )
            if incoming is not None and incoming.metadata_state == "dynamic":
                declared_dynamic = True
            planned_formal = (
                storage == "input"
                and tensor_id in function_argument_ids
                and bool((function.metadata or {}).get("source_region_integral"))
            )
            # compact_codewords' Mul result reaches the following sum region
            # with display shape (1, 8) before its producer has published a
            # tensor descriptor. That occurrence proves rank, not row count.
            # A planned-region formal starts unresolved until a booked static
            # call shape, an owner descriptor, or an exact static storage view
            # proves all extents. The byte-weight unsqueeze keeps its static
            # (1, 8) geometry through the latter proof.
            full_shape_proved = not planned_formal
            if planned_formal:
                full_shape_proved = bool(
                    incoming is not None
                    and incoming.metadata_state == "static"
                    and tuple(incoming.shape) == shape
                )
                if not full_shape_proved:
                    # Static ABI inputs (2,1)+(3,) and flatten(2,3)
                    # already declare every extent before an owner tensor
                    # table exists. Read the exact formal's ABI cause; a
                    # computed value's display shape is not such a cause.
                    from .concordance_declarations import (
                        SSA_VALUE as _INPUT_SSA_VALUE,
                        VALUE_TYPE_ABI as _INPUT_VALUE_TYPE_ABI,
                    )
                    from .ssa_record_return_state import ssa_value_identity_cell
                    shape_book = identity_book(module)
                    input_cell = ssa_value_identity_cell(
                        function, tensor_id, book=shape_book,
                    )
                    pending = [input_cell] if input_cell is not None else []
                    visited = set()
                    declared_abis = []
                    while pending:
                        cell = pending.pop()
                        if cell in visited:
                            continue
                        visited.add(cell)
                        for parent, _stage in shape_book.edges_into(cell):
                            if parent.page == _INPUT_VALUE_TYPE_ABI:
                                declared_abis.append(shape_book.page(
                                    _INPUT_VALUE_TYPE_ABI,
                                ).cell(parent.row, parent.column))
                            elif (parent.page == _INPUT_SSA_VALUE
                                  and parent.row == input_cell.row):
                                pending.append(parent)
                    # tensor_octets_to_bytes' `int(count)` consumes a
                    # planned formal whose optional ABI has None/no-storage
                    # and int/scalar arms. Both have a static rank-zero
                    # geometry, and the shared payload's scoped SSA fact
                    # proves scalar storage. Rejecting the None arm here
                    # invented dynamic span extents and routed the scalar
                    # result through a pointer cast. This proves geometry
                    # only; the union tag and branch retain presence truth.
                    scalar_slot = (
                        shape == () and input_cell is not None
                        and (slot_fact := shape_book.page(
                            _INPUT_SSA_VALUE,
                        ).cell(input_cell.row, input_cell.column)).storage
                        == "scalar"
                        and slot_fact.rank == 0
                    )
                    full_shape_proved = bool(declared_abis) and all(
                        bool(abi.physical_alternatives) and all(
                            (
                                arm.storage == "span"
                                and tuple(arm.shape) == shape
                                and arm.rank == len(shape)
                            ) or (
                                scalar_slot
                                and arm.storage in {"scalar", "none"}
                                and tuple(arm.shape) == ()
                                and arm.rank == 0
                                and arm.metadata_state == "static"
                            )
                            for arm in abi.physical_alternatives
                        ) for abi in declared_abis
                    )
                if not full_shape_proved:
                    shape_book = identity_book(module)
                    full_shape_proved = any(
                        row[0] == function_name
                        and int(row[1]) == tensor_id
                        and tuple(shape_book.page("ssa_call_shape").latest(row)[0])
                        == shape
                        for row in shape_book.page("ssa_call_shape").rows()
                    )
                storage_view = value_accounting.get("ssa_storage_view")
                if not full_shape_proved and isinstance(storage_view, Mapping):
                    source_id = storage_view.get("storage_value_id")
                    source_descriptor = (
                        owner_tensor_descriptor(int(source_id))
                        if isinstance(source_id, int) else None
                    )
                    if (
                        storage_view.get("operation") == "unsqueeze"
                        and tuple(storage_view.get("view_shape") or ()) == shape
                        and source_descriptor is not None
                        and source_descriptor.metadata_state == "static"
                        and len(shape) == len(source_descriptor.shape) + 1
                        and any(
                            shape[:axis] + shape[axis + 1:]
                            == tuple(source_descriptor.shape)
                            for axis, extent in enumerate(shape)
                            if extent == 1
                        )
                    ):
                        from .ssa_record_return_state import posted_ssa_value_fact
                        view_fact = posted_ssa_value_fact(
                            function, tensor_id, book=identity_book(module),
                        )
                        full_shape_proved = bool(
                            view_fact is not None
                            and tuple(view_fact.shape) == shape
                            and view_fact.rank == len(shape)
                        )
            state = metadata_state or (
                "dynamic"
                if declared_dynamic
                else "unresolved"
                if tensor_id in unresolved_argument_ids or not full_shape_proved
                else "static"
            )
            owner = tensor_table.by_id(alias_of) if alias_of is not None else None
            if (
                "physical_dtype" not in (value.accounting or {})
                and not value.dtype
            ):
                raise ValueError(
                    f"tensor descriptor for %t{int(value.id)} has neither a "
                    "recorded dtype nor a physical dtype"
                )
            physical_dtype = str(
                (value.accounting or {}).get("physical_dtype", value.dtype)
            ).lower()
            dtype_bytes = dtype_byte_size(physical_dtype)
            element_count = prod(shape) if shape else 1
            # A dynamic descriptor requires all three extent values; a result
            # inheriting a dynamic source's state inherits its extents too
            # (elementwise results share the source's runtime metadata).
            extent_ids: dict[str, int | None] = {
                "shape_value_id": value_accounting.get(
                    "tensor_shape_value_id"
                ),
                "rank_value_id": value_accounting.get(
                    "tensor_rank_value_id"
                ),
                "element_count_value_id": value_accounting.get(
                    "tensor_element_count_value_id"
                ),
            }
            if extent_values is not None:
                extent_ids = dict(zip(
                    ("shape_value_id", "rank_value_id", "element_count_value_id"),
                    (int(member.id) for member in extent_values), strict=True,
                ))
            if state == "dynamic":
                if all(value_id is not None for value_id in extent_ids.values()):
                    extent_ids = {
                        key: int(value_id)
                        for key, value_id in extent_ids.items()
                    }
                elif (
                    extent_ids_from is None
                    or extent_ids_from.shape_value_id is None
                ):
                    state = "unresolved"
                else:
                    extent_ids = {
                        "shape_value_id": extent_ids_from.shape_value_id,
                        "rank_value_id": extent_ids_from.rank_value_id,
                        "element_count_value_id": extent_ids_from.element_count_value_id,
                    }
            return tensor_table.register(SSATensorDescriptor(
                tensor_id=tensor_id,
                data_value_id=int(
                    tensor_id if data_value_id is None else data_value_id
                ),
                dtype=str(value.dtype),
                shape=shape,
                strides=strides(shape),
                storage=storage,
                metadata_state=state,
                **extent_ids,
                arena_id=owner.arena_id if owner is not None else tensor_id,
                allocation_owner=(
                    owner.allocation_owner if owner is not None else tensor_id
                ),
                owns_allocation=owner is None,
                element_offset=(owner.element_offset if owner is not None else 0),
                byte_offset=(owner.byte_offset if owner is not None else 0),
                byte_size=(
                    element_count * dtype_bytes if state == "static" else None
                ),
                alias_of=alias_of,
                writable=storage != "input",
            ))

        def own_dynamic_tensor(
            result: SSAValue, count: SSAValue, element_dtype: str,
            prefix: list[Instr],
        ) -> SSAValue:
            return _own_dynamic_tensor_output(
                module, function, book, tensor_table, fresh, register_tensor,
                result, count, element_dtype, prefix,
            )

        def has_private_region_result(result: SSAValue) -> bool:
            """Whether each exact region output is consumed before caller Ret.

            A directly returned tensor needs the public owned-result ABI;
            this private heap path cannot take that lifetime. Keep its
            existing dynamic-storage refusal until the public result contract
            is explicitly connected.
            """

            callers = tuple(
                (caller, call)
                for caller in module.functions.values()
                for block in caller.blocks.values()
                for call in block.instrs
                if call.op == "Call"
                and call.attributes.get("callee") == function.name
                and int(result.id) in tuple(map(
                    int, call.attributes.get("output_ids", ()),
                ))
            )
            if not callers:
                return False
            return all(
                not any(
                    int(value.id) == int(result.id)
                    for block in caller.blocks.values()
                    for instruction in block.instrs
                    if instruction.op == "Ret"
                    for value in instruction.args
                )
                for caller, _call in callers
            )

        def private_dynamic_result(result: SSAValue) -> bool:
            return not any(
                instruction.op == "Ret" and any(int(value.id) == int(result.id)
                                                for value in instruction.args)
                for block in function.blocks.values() for instruction in block.instrs
            ) or has_private_region_result(result)

        def post_unary_geometry(operation, source, result, evidence, execution_cells=()):
            from .concordance_declarations import TENSOR_ELEMENTWISE_EXTENT_BINDING, TENSOR_SSA_LOWERING
            from .identity_concordance import Derived, Mode
            from .ssa_record_return_state import function_scope_of, ssa_value_identity_cell
            metadata, proof = evidence
            cells = tuple(dict.fromkeys((
                *proof, *execution_cells,
                *(ssa_value_identity_cell(function, int(value.id), book=book)
                  for value in (source, result, *metadata)),
            )))
            if any(cell is None for cell in cells):
                raise ValueError("unary geometry lost its exact SSA evidence")
            book.post(
                TENSOR_ELEMENTWISE_EXTENT_BINDING,
                (function_scope_of(function), function.name, int(result.id)),
                (str(operation), int(source.id), *(int(value.id) for value in metadata)),
                stage=TENSOR_SSA_LOWERING, provenance=Derived(cells), mode=Mode.CONCORD,
            )

        constants: dict[int, Any] = {}
        constant_definitions: dict[int, Instr] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is not None:
                    payload = _constant_payload(instruction)
                    if payload is not None:
                        constants[int(instruction.res.id)] = payload
                        constant_definitions[int(instruction.res.id)] = instruction

        # Runtime extents: when a static shape is absent (the whole-program
        # capture keeps parameters symbolic), extent metadata becomes ordinary
        # SSA via the ``extent`` tensor operation -- the same contract the
        # descriptor was designed for ("shape/stride information is either
        # static (the tuples) or itself ordinary SSA (the optional value
        # ids)") and the Fortran emitter already recognises. Each backend
        # lowers ``extent`` its own way; nothing here is float- or
        # backend-specific.
        dynamic_extents: dict[int, dict[str, SSAValue]] = {}


        def ensure_dynamic(
            prefix: list[Instr], tensor: SSAValue
        ) -> dict[str, SSAValue]:
            """Cross ``tensor`` to dynamic metadata, whole.

            The descriptor's own validation states the contract: a dynamic
            tensor carries shape, rank, AND element-count as ordinary SSA,
            never a subset. All three ``extent`` instructions are minted on
            the first request and reused after.
            """

            tensor_id = int(tensor.id)
            cached = dynamic_extents.get(tensor_id)
            if cached is not None:
                return cached
            existing = tensor_table.by_id(tensor_id)
            if existing is not None and existing.metadata_state == "dynamic":
                existing_ids = (
                    existing.shape_value_id, existing.rank_value_id,
                    existing.element_count_value_id,
                )
                if all(value_id is not None for value_id in existing_ids):
                    # Program: private arange -> cast -> sum. Region2 input8
                    # already received the exact metadata formals through
                    # TENSOR_METADATA_CALL_BINDING. Reminting extent(%8)
                    # overwrote its descriptor with 4043/4044/4045 while
                    # its formal/accounting still named 4019/4022/4025.
                    # The descriptor's complete SSA identities are the
                    # existing extent values for this reader.
                    available = {
                        int(value.id): value for value in function.args
                    }
                    for extent_block in function.blocks.values():
                        for extent_instruction in extent_block.instrs:
                            for value in (*extent_instruction.args,
                                          *((extent_instruction.res,)
                                            if extent_instruction.res is not None
                                            else ())):
                                available.setdefault(int(value.id), value)
                    from .ssa_record_return_state import ssa_value_identity_cell
                    if any(
                        int(value_id) not in available
                        or ssa_value_identity_cell(
                            function, int(value_id), book=book,
                        ) is None
                        for value_id in existing_ids
                    ):
                        raise ValueError(
                            "dynamic tensor descriptor has no exact local extent values"
                        )
                    settled = dict(zip(
                        ("shape", "rank", "element_count"),
                        (available[int(value_id)] for value_id in existing_ids),
                        strict=True,
                    ))
                    dynamic_extents[tensor_id] = settled
                    return settled
            minted: dict[str, SSAValue] = {}
            from .frame_identity_book import _frame_physical_fields
            # A dynamic reduction may have no physical row for its result
            # yet. The frame reader returns KEEP_PHYSICAL in that case;
            # only a posted mapping can supply its rank.
            physical_fields = _frame_physical_fields(function, tensor_id)
            posted_rank = (
                physical_fields.get("rank")
                if isinstance(physical_fields, dict) else None
            )
            shape_width = (
                int(posted_rank) if posted_rank is not None
                else len(tuple(tensor.shape or ()))
            )
            for kind in ("shape", "rank", "element_count"):
                # Shape metadata is an integer vector with one slot per
                # tensor axis.  Treating it as a scalar under-allocates the
                # native frame and lets extent emission overwrite the
                # adjacent rank/count values.  Rank and element count remain
                # ordinary scalar metadata.
                value = fresh(
                    shape=(shape_width,)
                    if kind == "shape" else (),
                    dtype="int32",
                    physical={
                        "physical_dtype": "int32",
                        "storage": "span" if kind == "shape" else "scalar",
                        "rank": 1 if kind == "shape" else 0,
                    },
                )
                prefix.append(Instr(
                    "extent", [tensor], value,
                    attributes={
                        "tensor_operation": "extent", "extent_kind": kind,
                    },
                ))
                minted[kind] = value
            dynamic_extents[tensor_id] = minted
            existing = tensor_table.by_id(tensor_id)
            if existing is not None:
                tensor_table.tensors[tensor_id] = dataclasses.replace(
                    existing,
                    metadata_state="dynamic",
                    byte_size=None,
                    shape_value_id=int(minted["shape"].id),
                    rank_value_id=int(minted["rank"].id),
                    element_count_value_id=int(minted["element_count"].id),
                )
            return minted

        per_dim_extents: dict[tuple[int, int], SSAValue] = {}

        def dim_extent(
            prefix: list[Instr], tensor: SSAValue, axis: int
        ) -> SSAValue:
            key = (int(tensor.id), int(axis))
            cached = per_dim_extents.get(key)
            if cached is not None:
                return cached
            # The private rank-two gather multiplies the selector count by
            # this trailing-axis extent. Without its scalar physical row,
            # the next tensor pass rewrites that scalar Mul as binary_double
            # and asks for extent(product) before defining the product.
            # An extent of one axis is a scalar int32 by the existing
            # tensor-metadata contract.
            value = fresh(dtype="int32", physical={
                "physical_dtype": "int32", "storage": "scalar", "rank": 0,
            })
            prefix.append(Instr(
                "extent", [tensor], value,
                attributes={
                    "tensor_operation": "extent", "extent_kind": "dim",
                    "axis": int(axis),
                },
            ))
            per_dim_extents[key] = value
            return value

        aliases: dict[int, SSAValue] = {}
        pending_runtime_view_guards = []
        pending_runtime_broadcast_guards = []
        mixed_runtime_geometry: dict[int, SSAValue] = {}

        def resolve(value: SSAValue) -> SSAValue:
            seen: set[int] = set()
            while int(value.id) in aliases and int(value.id) not in seen:
                seen.add(int(value.id))
                value = aliases[int(value.id)]
            return value

        for block_name, block in function.blocks.items():
            rewritten: list[Instr] = []
            for original in block.instrs:
                lowering_at[:] = [function, original]
                instruction = dataclasses.replace(
                    original, args=[resolve(argument) for argument in original.args]
                )
                if instruction.op == "ensure_tensor" and len(instruction.args) == 2:
                    from .native_tensor_family import wrap_family

                    receiver, raw = instruction.args
                    wrapping = wrap_family(function, receiver, raw, book=book)
                    if wrapping is not None and instruction.res is not None:
                        family, receiver_proof = wrapping
                        from .concordance_declarations import (
                            NATIVE_TENSOR_WRAP, TENSOR_SSA_LOWERING,
                        )
                        from .identity_concordance import Derived, Mode
                        from .ssa_record_return_state import (
                            function_scope_of, posted_ssa_value_fact,
                            revise_ssa_value, ssa_value_identity_cell,
                        )

                        # Huffman's topk indices are raw CTensor, not a list
                        # literal. The selected backend/source family proves
                        # ensure_tensor's result.data = tensor branch. Keep
                        # that exact storage and physical dtype; the receiver's
                        # dtype and constructor default do not describe it.
                        raw_fact = posted_ssa_value_fact(function, raw.id, book=book)
                        if (raw_fact is None or raw_fact.storage != "span"
                                or raw_fact.physical_dtype is None):
                            raise ValueError("raw native tensor wrapper lacks physical storage")
                        result = instruction.res
                        result.dtype = raw.dtype
                        result.shape = tuple(raw.shape or ())
                        result.accounting = {
                            **dict(raw.accounting or {}),
                            "physical_dtype": raw_fact.physical_dtype,
                            "program_abi_storage": "span",
                            "program_abi_rank": raw_fact.rank,
                        }
                        revised = revise_ssa_value(
                            function, result.id, dtype=result.dtype,
                            shape=tuple(result.shape), stage=TENSOR_SSA_LOWERING,
                            sources=(family, ssa_value_identity_cell(
                                function, raw.id, book=book)), book=book,
                            physical={"physical_dtype": raw_fact.physical_dtype,
                                      "storage": "span", "rank": raw_fact.rank},
                        )
                        raw_cell = ssa_value_identity_cell(function, raw.id, book=book)
                        receiver_cell = ssa_value_identity_cell(function, receiver.id, book=book)
                        result_cell = ssa_value_identity_cell(function, result.id, book=book)
                        book.post(
                            NATIVE_TENSOR_WRAP,
                            (function_scope_of(function), int(result.id)),
                            (family, raw_cell, receiver_cell, result_cell, receiver_proof),
                            stage=TENSOR_SSA_LOWERING,
                            provenance=Derived((family, raw_cell, receiver_cell, result_cell,
                                                receiver_proof)),
                            mode=Mode.CONCORD,
                        )
                        owner = register_tensor(raw, storage=(
                            "input" if int(raw.id) in function_argument_ids else "temporary"
                        ))
                        register_tensor(
                            result, storage="view", alias_of=owner.tensor_id,
                            data_value_id=owner.data_value_id,
                            metadata_state=owner.metadata_state,
                            extent_ids_from=owner,
                            alias_proof=book.latest_ref(
                                NATIVE_TENSOR_WRAP,
                                (function_scope_of(function), int(result.id)),
                            ),
                        )
                        aliases[int(result.id)] = raw
                        continue
                    from .numerical_constructor_defaults import (
                        literal_constructor_default,
                    )
                    from .concordance_declarations import TENSOR_SSA_LOWERING
                    from .ssa_record_return_state import (
                        revise_ssa_value as revise_literal_ssa_value,
                    )

                    source = constant_definitions.get(int(instruction.args[1].id))
                    materialized = (
                        literal_constructor_default(
                            function, instruction, source, reference, book=book,
                        ) if source is not None else None
                    )
                    if materialized is not None:
                        payload, dtype, receipt = materialized
                        result = instruction.res
                        if (result is None or len(tuple(result.shape)) != 1
                                or len(payload) != int(result.shape[0])
                                or any(type(item) not in {bool, int, float}
                                       for item in payload)):
                            raise ValueError(
                                "concorded tensor literal has no flat numeric "
                                "result shape"
                            )
                        result.dtype = dtype
                        result.accounting = {
                            **dict(result.accounting or {}),
                            "physical_dtype": dtype,
                            "program_abi_storage": "span",
                            "program_abi_rank": 1,
                        }
                        revise_literal_ssa_value(
                            function, result.id, dtype=dtype,
                            shape=tuple(result.shape),
                            stage=TENSOR_SSA_LOWERING,
                            sources=(receipt,), book=book,
                            physical={"physical_dtype": dtype,
                                      "storage": "span", "rank": 1},
                        )
                        # The literal is executable tensor storage, not just
                        # a Const spelling.  Without its static descriptor the
                        # later unsqueeze loses the fixed eight-element owner,
                        # leaving compact_codewords' mixed Mul unresolved.
                        descriptor = register_tensor(
                            result, storage="temporary", metadata_state="static",
                        )
                        if (descriptor.dtype != dtype
                                or tuple(descriptor.shape) != tuple(result.shape)):
                            raise ValueError(
                                "selected literal tensor descriptor disagrees "
                                "with its physical default"
                            )
                        constants[int(result.id)] = payload
                        rewritten.append(Instr(
                            "Const", [], result,
                            attributes={"value": payload,
                                        "constructor_default": receipt},
                            source_span=instruction.source_span,
                        ))
                        continue
                # A shaped operation inside a concorded Precision region is
                # not independently lowerable. Preserve it for the compiler's
                # limb pipeline, which expands the entire region later.
                if is_concorded_precision_operator(instruction):
                    rewritten.append(instruction)
                    continue
                if (
                    instruction.op in {"Call", "call"}
                    and instruction.attributes.get("call_role_set") == "blas"
                    and instruction.attributes.get("call_role") == "gemm"
                    and instruction.attributes.get("callee") == "blas.gemm"
                ):
                    # AbstractTensor submitted a finite BLAS role call.  Its
                    # semantic identity remains attached while the universal
                    # tensor lowering supplies the ordinary executable
                    # fallback. A backend identity may subsequently replace
                    # that fallback by qualified location.
                    attributes = dict(instruction.attributes)
                    attributes.pop("callee", None)
                    instruction = dataclasses.replace(
                        instruction,
                        op="matmul",
                        attributes=attributes,
                    )
                if (
                    instruction.op in {"Call", "call"}
                    and instruction.attributes.get("callee") is not None
                ):
                    raw_axes = instruction.attributes.get(
                        "basic_index_axes"
                    )
                    source_shape = tuple(map(
                        int,
                        instruction.attributes.get(
                            "basic_index_source_shape", ()
                        ),
                    ))
                    axes = (
                        None if raw_axes is None else tuple(
                            (tuple(map(int, indices)), bool(drop_axis))
                            for indices, drop_axis in raw_axes
                        )
                    )
                    if (
                        instruction.attributes.get("callee")
                        == "index_select_double"
                        and instruction.res is not None
                        and instruction.args
                        and axes is not None
                        and len(axes) == len(source_shape)
                        and all(
                            drop_axis and len(indices) == 1
                            for indices, drop_axis in axes
                        )
                    ):
                        # A region may already have selected the repository
                        # index kernel before whole-module tensor lowering.
                        # Legalize the same dropped-axis scalar contract here
                        # rather than preserving a one-element tensor call.
                        linear_index = 0
                        for axis, (indices, _drop_axis) in enumerate(axes):
                            linear_index = (
                                linear_index * source_shape[axis]
                                + int(indices[0])
                            )
                        index_value, index_definition = constant(
                            linear_index, "int64"
                        )
                        address = fresh(dtype="ptr")
                        instruction.res.shape = ()
                        rewritten.extend((
                            index_definition,
                            Instr(
                                "GetElementPtr",
                                [instruction.args[0], index_value],
                                address,
                                attributes={
                                    "basic_index_linearized": True,
                                },
                                source_span=instruction.source_span,
                            ),
                            Instr(
                                "Load", [address], instruction.res,
                                attributes={
                                    "basic_index_linearized": True,
                                },
                                source_span=instruction.source_span,
                            ),
                        ))
                        continue
                    # AOT numerical-region capture may already have selected
                    # the universal repository fallback before this pass sees
                    # the call.  Keep the AbstractTensor semantic identity on
                    # that fallback so a destination can still replace it by
                    # its qualified GEMM intrinsic.  Without this, authored
                    # ``left @ right``/``left.matmul(right)`` reaches here as
                    # an otherwise-correct ``matmul_double`` call, but the
                    # WebGPU identity has nothing left to recognize.
                    tensor_spelling = (
                        instruction.attributes.get("tensor_operation")
                        or instruction.attributes.get("tensor_candidate")
                    )
                    if (
                        str(tensor_spelling) == "matmul"
                        and instruction.attributes.get("callee")
                        == "matmul_double"
                    ):
                        attributes = dict(instruction.attributes)
                        attributes.setdefault(
                            "backend_intrinsic_candidate",
                            {
                                "semantic_identity": (
                                    "src.common.tensors.abstraction."
                                    "AbstractTensor.matmul"
                                ),
                                "lowering_namespace": "abstract_tensor",
                                "ingested_fallback": False,
                            },
                        )
                        attributes.setdefault(
                            "backend_intrinsic_family", "blas.gemm"
                        )
                        instruction = dataclasses.replace(
                            instruction, attributes=attributes
                        )
                    rewritten.append(instruction)
                    continue
                # Whether this opcode is a TENSOR operation is judged below
                # from its operands' extents.  Those are fields somebody filled
                # in when the value was created, and a consumer holds its own
                # SSAValue carrying the same id, so an operand whose shape was
                # proven after that moment still reads as rank 0 here -- and
                # the opcode falls through to the scalar emitter, which writes
                # element zero and leaves the rest of the buffer untouched.
                # Take the settled fact for the identity before deciding.
                _settle_operand_shapes(
                    function, instruction.args, owner_tensor_descriptor,
                )
                if instruction.res is not None:
                    _settle_operand_shapes(
                        function, (instruction.res,), owner_tensor_descriptor,
                    )
                # The runtime reshape's checked quotient and remainder are
                # ordinary rank-zero SSA arithmetic.  A later tensor pass
                # otherwise lifts their Mod/FloorDiv into tensor kernels and
                # asks for an extent of the result before defining it.  The
                # posted physical SSA cells, including the result's, decide
                # this scalar case; an unknown or span operand stays on the
                # ordinary tensor decision path below.
                from .ssa_record_return_state import posted_ssa_value_fact

                scalar_facts = tuple(
                    posted_ssa_value_fact(function, int(value.id), book=book)
                    for value in (*instruction.args, instruction.res)
                    if value is not None
                )
                if (
                    instruction.op in _SHAPED_SSA_OPERATIONS
                    or c_tensor_opcode(str(instruction.op)) is not None
                ) and scalar_facts and all(
                    fact is not None
                    and fact.storage == "scalar" and fact.rank == 0
                    for fact in scalar_facts
                ):
                    rewritten.append(instruction)
                    continue
                candidate_only = bool(
                    instruction.attributes.get("tensor_candidate") is not None
                    and instruction.attributes.get("tensor_operation") is None
                    and instruction.attributes.get("tensor") is None
                )
                operation_value = (
                    instruction.attributes.get("tensor_operation")
                    or instruction.attributes.get("tensor")
                    or instruction.attributes.get("tensor_candidate")
                    or (
                        # Program: `_header(*values)` widens an integer before
                        # `Shr`/`BitAnd`.  The Cast records the source opcode
                        # for provenance, but is still a native conversion;
                        # treating `source_operator='shr'` as the Cast's tensor
                        # operation stamped float64 and demanded an extent.
                        # Only an authored cast operation selects a tensor
                        # conversion here.  The following scalar opcode keeps
                        # its own rank and physical proof.
                        instruction.attributes.get("source_operator")
                        if instruction.op in {"Cast", "cast"}
                        and str(instruction.attributes.get(
                            "source_operator") or ""
                        ).casefold() in _CAST_OPERATIONS
                        else None
                    )
                )
                # ProcessGraph spells Python arithmetic as ordinary SSA
                # opcodes rather than attaching tensor metadata. Once shape
                # propagation proves the operands are tensors, route those
                # opcodes through the authored row-major tensor source.
                if (
                    operation_value is None
                    and (
                        instruction.op in _SHAPED_SSA_OPERATIONS
                        or c_tensor_opcode(str(instruction.op)) is not None
                    )
                    and instruction.res is not None
                    and (
                        tuple(instruction.res.shape)
                        or any(tuple(argument.shape) for argument in instruction.args)
                        # A retained Python slice is already proof that this
                        # is semantic tensor indexing, even when its source
                        # extent arrives later than the local SSA value.  It
                        # must reach basic-index normalization rather than the
                        # scalar address pass.
                        or (
                            instruction.op in {"Indexed", "IndexedStore"}
                            and any(
                                isinstance(constants.get(int(selector.id)), slice)
                                for selector in (
                                    instruction.args[1:]
                                    if instruction.op == "Indexed"
                                    else instruction.args[1:-1]
                                )
                            )
                        )
                        # Dynamic tensor spans deliberately have no invented
                        # static extents.  Their tensor identity is carried by
                        # the shape concordance as ordinary SSA accounting;
                        # accept that contract here instead of falling through
                        # to generic pointer indexing merely because ``shape``
                        # is empty.
                        or any(
                            int((argument.accounting or {}).get(
                                "program_abi_rank", 0
                            ) or 0) > 0
                            and str((argument.accounting or {}).get(
                                "program_abi_storage", ""
                            )) == "span"
                            for argument in instruction.args
                        )
                        # Symbolic shapes prove nothing either way; an operand
                        # already registered as a tensor descriptor (a kernel
                        # call's result) is proof enough of tensorhood.
                        or any(
                            tensor_table.by_id(int(argument.id)) is not None
                            for argument in instruction.args
                        )
                    )
                ):
                    operation_value = _SHAPED_SSA_OPERATIONS.get(
                        instruction.op, str(instruction.op)
                    )
                if operation_value is None or instruction.res is None:
                    rewritten.append(instruction)
                    continue
                operation = str(operation_value)
                result = instruction.res
                args = list(instruction.args)
                if result.dtype == "ssa.aggregate":
                    # The source layout has already selected these ordered
                    # result leaves. Use the passed repository reference's
                    # canonical operation entrypoint and named output formals;
                    # the pair has no element dtype of its own.
                    from .concordance_declarations import SOURCE_AGGREGATE_LAYOUT

                    layout_proof = tuple(
                        cell for cell in instruction.attributes.get(
                            "intrinsic_named_result_proof", (),
                        ) if cell.page == SOURCE_AGGREGATE_LAYOUT
                    )
                    operation_reference = reference.operation(operation)
                    entrypoints = (
                        () if operation_reference is None
                        else tuple(operation_reference.entrypoints)
                    )
                    roles = tuple(map(str, instruction.arg_roles))
                    outputs = tuple(map(int, instruction.attributes.get(
                        "output_ids", (),
                    )))
                    if (len(layout_proof) != 1 or len(entrypoints) != 1
                            or len(args) != len(roles)
                            or set(roles) != {"arg:0", "kw:k", "kw:dim"}
                            or len(outputs) != 2):
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name, block_name, operation,
                            "named aggregate call lacks exact layout, input roles or repository entrypoint",
                        ))
                        rewritten.append(instruction)
                        continue
                    layout = book.page(SOURCE_AGGREGATE_LAYOUT).cell(
                        layout_proof[0].row, layout_proof[0].column,
                    )
                    if tuple(int(member.value_id) for member in layout.members) != outputs:
                        raise ValueError("named aggregate call output identities disagree with source layout")
                    kernel_name = str(entrypoints[0])
                    kernel = reference.module.functions[kernel_name]
                    from ..common.tensors.accelerator_backends.llvm_repository_ssa import (
                        post_repository_kernel_identity,
                    )
                    from .concordance_declarations import (
                        CALLSITE_RETURN_MEMBER, CANONICAL_VALUE,
                        GUARDED_CALL_ACTUAL_RANGE, PROCESS_GRAPH_OPERANDS,
                        SSA_VALUE, SOURCE_TENSOR_SHAPE_DIMENSION,
                        TENSOR_OUTPUT_CAPACITY_BOUND, TENSOR_RUNTIME_TOPK_SELECTION,
                        TensorOutputCapacityBound, TensorRuntimeTopkSelection,
                        TENSOR_SSA_LOWERING,
                    )
                    from .concorded_literals import (
                        MISSING as MISSING_LITERAL, literal_from_identity,
                    )
                    from .ssa_record_return_state import (
                        function_scope_of, revise_ssa_value,
                        ssa_value_identity_cell,
                    )

                    post_repository_kernel_identity(kernel, book)
                    from .ssa_call_input_adapters import _infer_pointer_storage
                    _infer_pointer_storage({str(kernel.name): kernel})
                    named_outputs = dict(kernel.metadata.get("named_outputs") or ())
                    if (tuple(named_outputs) != ("values", "indices")
                            or len(kernel.args) != 7):
                        raise ValueError("named aggregate repository output contract is incomplete")
                    by_role = dict(zip(roles, args, strict=True))
                    source = by_role["arg:0"]
                    if str(source.dtype) != "int64":
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name, block_name, operation,
                            "repository named aggregate entrypoint has no matching input dtype",
                        ))
                        rewritten.append(instruction)
                        continue
                    # The source's two named leaf descriptors are the
                    # accepted output extents. Read their exact member cells
                    # and require the authored static axis/k to agree before
                    # binding a native count. In root(values[3], k=4), the
                    # old Call sent 4 while its allocated output had shape
                    # (4,); CBackend.topk_ selects min(3, 4) = 3.
                    member_receipts = tuple(
                        next((cell for cell in member.source_cells
                              if cell.page == CALLSITE_RETURN_MEMBER), None)
                        for member in layout.members
                    )
                    if any(cell is None for cell in member_receipts):
                        raise ValueError("named topk leaf has no descriptor receipt")
                    member_shapes = tuple(
                        tuple(book.page(CALLSITE_RETURN_MEMBER).cell(
                            cell.row, cell.column,
                        )[0]) for cell in member_receipts
                    )
                    static_shape = tuple(source.shape or ())
                    operand_cell = book.latest_ref(
                        PROCESS_GRAPH_OPERANDS, layout_proof[0].row,
                    )
                    source_operands = (
                        () if operand_cell is None else
                        book.page(PROCESS_GRAPH_OPERANDS).cell(
                            operand_cell.row, operand_cell.column,
                        )
                    )
                    def literal_actual(role):
                        matches = tuple(parent for parent, name in source_operands
                                        if name == role)
                        actual = by_role[role]
                        if (len(matches) != 1
                                or matches[0].page != CANONICAL_VALUE
                                or matches[0].row[-1] != int(actual.id)):
                            return None
                        value, _literal_cells = literal_from_identity(
                            book, matches[0],
                        )
                        if value is MISSING_LITERAL or type(value) is not int:
                            return None
                        local = constants.get(int(actual.id))
                        if local is not None and local != value:
                            raise ValueError("topk SSA constant disagrees with source literal")
                        return value
                    k_literal = literal_actual("kw:k")
                    dim_literal = literal_actual("kw:dim")
                    selected_count = None
                    selected_axis = None
                    dynamic_guard = None
                    runtime_dimension_proof = None
                    if (static_shape and all(member_shapes)
                            and (k_literal is None or dim_literal is None)):
                        raise ValueError(
                            "named topk static extents lack exact literal selection proof"
                        )
                    if (static_shape and isinstance(k_literal, int)
                            and not isinstance(k_literal, bool)
                            and isinstance(dim_literal, int)
                            and not isinstance(dim_literal, bool)):
                        if (k_literal < 0 or not -len(static_shape) <= dim_literal
                                < len(static_shape)):
                            raise ValueError("named topk has no admitted static k or axis")
                        selected_axis = dim_literal % len(static_shape)
                        selected_count = min(k_literal, static_shape[selected_axis])
                        expected_shape = list(static_shape)
                        expected_shape[selected_axis] = selected_count
                        if member_shapes != (tuple(expected_shape),) * 2:
                            raise ValueError(
                                "named topk source extents disagree with selected count"
                            )
                    elif static_shape and k_literal is None and isinstance(dim_literal, int):
                        if (not -len(static_shape) <= dim_literal < len(static_shape)
                                or member_shapes != ((), ())):
                            raise ValueError("dynamic named topk has no exact axis or leaf layout")
                        selected_axis = dim_literal % len(static_shape)
                        call_identity = book.latest_ref(CANONICAL_VALUE, layout_proof[0].row)
                        k_parents = tuple(parent for parent, role in source_operands
                                          if role == "kw:k" and parent.page == CANONICAL_VALUE)
                        def fork_ancestors(cell):
                            pending, seen = [cell], set()
                            while pending:
                                current = pending.pop()
                                if current in seen or current is None:
                                    continue
                                seen.add(current)
                                pending.extend(source for source, stage in book.edges_into(current)
                                               if source.page == CANONICAL_VALUE
                                               and stage.name == "read_scope_fork")
                            return seen
                        if call_identity is not None and len(k_parents) == 1:
                            call_ancestors = fork_ancestors(call_identity)
                            actual_ancestors = fork_ancestors(k_parents[0])
                            guards = []
                            for guard_row in book.page(GUARDED_CALL_ACTUAL_RANGE).rows():
                                guard_cell = book.latest_ref(GUARDED_CALL_ACTUAL_RANGE, guard_row)
                                causes = {source for source, _stage in book.edges_into(guard_cell)}
                                if causes & call_ancestors and causes & actual_ancestors:
                                    guards.append(guard_cell)
                            intervals = {
                                tuple(bounds)
                                for guard in guards
                                if isinstance((bounds := book.page(
                                    GUARDED_CALL_ACTUAL_RANGE).cell(
                                        guard.row, guard.column)), tuple)
                                and len(bounds) == 2
                                and all(bound is None or type(bound) is int
                                        for bound in bounds)
                            }
                            lows = [bounds[0] for bounds in intervals if type(bounds[0]) is int]
                            highs = [bounds[1] for bounds in intervals if type(bounds[1]) is int]
                            if lows and highs:
                                low, high = max(lows), min(highs)
                                if (low >= 0 and high >= low
                                        and static_shape[selected_axis] <= (1 << 31) - 1):
                                    dynamic_guard = tuple(guards)
                        if not dynamic_guard:
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "dynamic named topk lacks exact nonnegative guarded k domain",
                            ))
                            rewritten.append(instruction)
                            continue
                    elif not static_shape and member_shapes == ((), ()):
                        # _ascending_order reads k from weights.shape[0]. Its
                        # source dimension receipt proves nonnegative k, while
                        # the current input descriptor supplies the axis at
                        # runtime. An empty display shape proves no capacity.
                        from .ssa_record_return_state import posted_ssa_value_fact
                        source_fact = posted_ssa_value_fact(
                            function, int(source.id), book=book,
                        )
                        if (source_fact is None or source_fact.storage != "span"
                                or not isinstance(source_fact.rank, int)
                                or source_fact.rank <= 0
                                or not isinstance(dim_literal, int)
                                or not -source_fact.rank <= dim_literal < source_fact.rank):
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "runtime named topk lacks exact input rank or literal axis",
                            ))
                            rewritten.append(instruction)
                            continue
                        selected_axis = dim_literal % source_fact.rank
                        k_cell = ssa_value_identity_cell(
                            function, int(by_role["kw:k"].id), book=book,
                        )
                        k_origins = {
                            cause.row
                            for cause, _ in book.edges_into(k_cell)
                            if cause.page == CANONICAL_VALUE
                        } if k_cell is not None else set()
                        matching_dimensions = tuple(
                            cell for row in book.page(
                                SOURCE_TENSOR_SHAPE_DIMENSION
                            ).rows()
                            if (cell := book.latest_ref(
                                SOURCE_TENSOR_SHAPE_DIMENSION, row
                            )) is not None
                            and row in k_origins
                            and tuple(book.page(
                                SOURCE_TENSOR_SHAPE_DIMENSION
                            ).cell(cell.row, cell.column))[3] >= 0
                        )
                        if len(set(matching_dimensions)) == 1:
                            runtime_dimension_proof = matching_dimensions[0]
                        if runtime_dimension_proof is None and k_cell is not None:
                            # LZ `grams` is read under `if grams > 1`. Its
                            # callsite guard proves a lower bound, while the
                            # runtime input axis supplies the upper capacity.
                            # Match the exact call and k forks cited by the
                            # existing guarded-actual receipt.
                            def fork_ancestors(cell):
                                pending, seen = [cell], set()
                                while pending:
                                    current = pending.pop()
                                    if current is None or current in seen:
                                        continue
                                    seen.add(current)
                                    pending.extend(source for source, stage in book.edges_into(current)
                                                   if source.page == CANONICAL_VALUE
                                                   and stage.name == "read_scope_fork")
                                return seen
                            call_identity = book.latest_ref(CANONICAL_VALUE, layout_proof[0].row)
                            k_parents = tuple(parent for parent, role in source_operands
                                              if role == "kw:k" and parent.page == CANONICAL_VALUE)
                            if call_identity is not None and len(k_parents) == 1:
                                call_ancestors = fork_ancestors(call_identity)
                                actual_ancestors = fork_ancestors(k_parents[0])
                                guards = tuple(
                                    guard for row in book.page(GUARDED_CALL_ACTUAL_RANGE).rows()
                                    if (guard := book.latest_ref(GUARDED_CALL_ACTUAL_RANGE, row)) is not None
                                    and isinstance((bounds := book.page(GUARDED_CALL_ACTUAL_RANGE).cell(
                                        guard.row, guard.column)), tuple)
                                    and len(bounds) == 2
                                    and type(bounds[0]) is int
                                    and bounds[0] >= 0
                                    and (bounds[1] is None or
                                         type(bounds[1]) is int and bounds[1] >= bounds[0])
                                    and (causes := {source for source, _ in book.edges_into(guard)})
                                        & call_ancestors
                                    and causes & actual_ancestors
                                )
                                if (guards and len({tuple(book.page(
                                        GUARDED_CALL_ACTUAL_RANGE).cell(
                                            guard.row, guard.column))
                                        for guard in guards}) == 1):
                                    runtime_dimension_proof = guards[0]
                        if runtime_dimension_proof is None:
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "runtime named topk k lacks exact nonnegative source dimension",
                            ))
                            rewritten.append(instruction)
                            continue
                    if runtime_dimension_proof is not None:
                        input_descriptor = tensor_table.by_id(int(source.id))
                        has_metadata = (
                            input_descriptor is not None
                            and input_descriptor.metadata_state == "dynamic"
                            and all(value_id is not None for value_id in (
                                input_descriptor.shape_value_id,
                                input_descriptor.rank_value_id,
                                input_descriptor.element_count_value_id,
                            ))
                        )
                        if not has_metadata:
                            from .concordance_declarations import (
                                BindingKind, CONTROL_VALUE_BINDING, ControlBinding,
                            )
                            binding_ref = book.latest_ref(
                                CONTROL_VALUE_BINDING,
                                (function_scope_of(function), int(source.id)),
                            )
                            binding = (None if binding_ref is None else
                                       book.page(CONTROL_VALUE_BINDING).cell(
                                           binding_ref.row, binding_ref.column))
                            declared_input = (
                                int(source.id) in function_argument_ids
                                and isinstance(binding, ControlBinding)
                                and binding.kind in {
                                    BindingKind.PARAMETER_SEED,
                                    BindingKind.FIELD_INCUMBENT,
                                }
                                and binding.ssa_value.row == (
                                    function_scope_of(function), int(source.id),
                                )
                            )
                            if declared_input and (
                                input_descriptor is None
                                or input_descriptor.metadata_state == "unresolved"
                            ):
                                register_tensor(source, storage="input")
                            else:
                                shortfalls.append(TensorSSALoweringShortfall(
                                    function_name, block_name, operation,
                                    "runtime named topk awaits exact transported input metadata",
                                ))
                                rewritten.append(instruction)
                                continue
                    prefix = []
                    dynamic = ensure_dynamic(prefix, source)
                    def prove_native_actual(actual, formal, *, storage, rank, extra_sources=()):
                        formal_cell = book.latest_ref(
                            SSA_VALUE, (function_scope_of(kernel), int(formal.id)),
                        )
                        if formal_cell is None:
                            raise ValueError("repository formal lacks its ingested identity")
                        formal_fact = book.page(SSA_VALUE).cell(
                            formal_cell.row, formal_cell.column,
                        )
                        if not getattr(formal_fact, "dtype", None):
                            raise ValueError("repository formal has no declared element dtype")
                        proof_sources = (layout_proof[0], formal_cell, *extra_sources)
                        native_dtype = str(formal_fact.dtype)
                        revised = revise_ssa_value(
                            function, int(actual.id), dtype=native_dtype,
                            shape=tuple(actual.shape or ()),
                            stage=TENSOR_SSA_LOWERING,
                            sources=proof_sources, book=book,
                            physical={
                                "physical_dtype": native_dtype,
                                "storage": storage, "rank": rank,
                            },
                        )
                        if revised is None:
                            raise ValueError("native actual lacks its SSA identity")
                        actual.dtype = native_dtype
                        actual.accounting = {
                            **dict(actual.accounting or {}),
                            "physical_dtype": native_dtype,
                            "program_abi_storage": storage,
                            "program_abi_rank": rank,
                        }
                    def i32_actual(operand):
                        if str(operand.dtype) == "int32":
                            return operand
                        converted = fresh(dtype="int32")
                        prefix.append(Instr(
                            "Cast", [operand], converted,
                            attributes={"target_dtype": "int32"},
                        ))
                        return converted
                    native_k = by_role["kw:k"]
                    native_dim = by_role["kw:dim"]
                    if selected_axis is not None and static_shape:
                        # Keep the selected count as ordinary SSA, so the
                        # native actual computes the same bounded count that
                        # the two booked output leaves state. The bound and
                        # k are the original exact SSA operands.
                        axis_bound, axis_definition = constant(
                            static_shape[selected_axis], str(native_k.dtype),
                        )
                        bounded_k = fresh(dtype=str(native_k.dtype))
                        prefix.extend((axis_definition, Instr(
                            "Min", [native_k, axis_bound], bounded_k,
                            attributes={"source_operator": "topk"},
                        )))
                        native_k = bounded_k
                        if dim_literal != selected_axis:
                            native_dim, dim_definition = constant(
                                selected_axis, str(native_dim.dtype),
                            )
                            prefix.append(dim_definition)
                    runtime_selection = None
                    if runtime_dimension_proof is not None:
                        from .ssa_call_input_adapters import _numeric_conversion_attributes
                        scalar32 = {"physical_dtype": "int32", "storage": "scalar", "rank": 0}
                        scalar64 = {"physical_dtype": "int64", "storage": "scalar", "rank": 0}
                        axis_index, axis_index_def = constant(
                            selected_axis, "int64", physical=scalar64,
                        )
                        axis_address = fresh(dtype="ptr")
                        axis_extent = fresh(dtype="int32", physical=scalar32)
                        axis_extent64 = fresh(dtype="int64", physical=scalar64)
                        bounded_k = fresh(dtype="int64", physical=scalar64)
                        selected32 = fresh(dtype="int32", physical=scalar32)
                        prefix.extend((
                            axis_index_def,
                            Instr(Handler.GetElementPtr.value,
                                  [dynamic["shape"], axis_index], axis_address),
                            Instr(Handler.Load.value, [axis_address], axis_extent),
                            Instr(Handler.Cast.value, [axis_extent], axis_extent64,
                                  attributes={**_numeric_conversion_attributes(
                                      "int32", "int64"),
                                      "source_dtype": "int32", "target_dtype": "int64"}),
                            Instr("Min", [native_k, axis_extent64], bounded_k,
                                  attributes={"source_operator": "topk"}),
                            Instr(Handler.Cast.value, [bounded_k], selected32,
                                  attributes={**_numeric_conversion_attributes(
                                      "int64", "int32"),
                                      "source_dtype": "int64", "target_dtype": "int32"}),
                        ))
                        native_k = selected32
                        native_dim, native_dim_def = constant(
                            selected_axis, "int32", physical=scalar32,
                        )
                        prefix.append(native_dim_def)
                        runtime_selection = (
                            axis_extent, bounded_k, selected32,
                        )
                    actuals = [
                        source, dynamic["shape"], dynamic["rank"],
                        i32_actual(native_dim),
                        i32_actual(native_k),
                    ]
                    if runtime_selection is not None:
                        input_rank = int(source_fact.rank)
                        shape_value, shape_definition = int_vector(
                            (0,) * input_rank, physical={
                                "physical_dtype": "int32", "storage": "span", "rank": 1,
                            },
                        )
                        prefix.append(shape_definition)
                        other_extents = []
                        for axis in range(input_rank):
                            index, index_def = constant(axis, "int64", physical=scalar64)
                            input_address = fresh(dtype="ptr")
                            output_address = fresh(dtype="ptr")
                            prefix.extend((
                                index_def,
                                Instr(Handler.GetElementPtr.value,
                                      [shape_value, index], output_address),
                            ))
                            if axis == selected_axis:
                                extent_value = runtime_selection[2]
                            else:
                                prefix.append(Instr(
                                    Handler.GetElementPtr.value,
                                    [dynamic["shape"], index], input_address,
                                ))
                                extent_value = fresh(dtype="int32", physical=scalar32)
                                prefix.append(Instr(
                                    Handler.Load.value, [input_address], extent_value,
                                ))
                                other_extents.append(extent_value)
                            prefix.append(Instr(
                                Handler.Store.value,
                                [extent_value, output_address], None,
                            ))
                        rank_value, rank_definition = constant(
                            input_rank, "int32", physical=scalar32,
                        )
                        prefix.append(rank_definition)
                        element_count = runtime_selection[2]
                        if other_extents:
                            # Same checked count program as runtime broadcast:
                            # seed zero when any axis is empty, then narrow
                            # after each wide product. A later zero axis must
                            # not spuriously overflow an earlier product.
                            dimensions = (runtime_selection[2], *other_extents)
                            zero32, zero_def = constant(0, "int32", physical=scalar32)
                            one32, one_def = constant(1, "int32", physical=scalar32)
                            prefix.extend((zero_def, one_def))
                            empty = None
                            for dimension in dimensions:
                                is_zero = fresh(dtype="bool", physical={
                                    "physical_dtype": "bool", "storage": "scalar", "rank": 0,
                                })
                                prefix.append(Instr("Eq", [dimension, zero32], is_zero))
                                if empty is None:
                                    empty = is_zero
                                else:
                                    either = fresh(dtype="bool", physical={
                                        "physical_dtype": "bool", "storage": "scalar", "rank": 0,
                                    })
                                    prefix.append(Instr("LOr", [empty, is_zero], either))
                                    empty = either
                            element_count = fresh(dtype="int32", physical=scalar32)
                            prefix.append(Instr(
                                "Select", [empty, zero32, one32], element_count,
                            ))
                            for dimension in dimensions:
                                widened_count = fresh(dtype="int64", physical=scalar64)
                                widened_dimension = fresh(dtype="int64", physical=scalar64)
                                product = fresh(dtype="int64", physical=scalar64)
                                narrowed = fresh(dtype="int32", physical=scalar32)
                                prefix.extend((
                                    Instr("Cast", [element_count], widened_count,
                                          attributes={"source_dtype": "int32",
                                                      "target_dtype": "int64"}),
                                    Instr("Cast", [dimension], widened_dimension,
                                          attributes={"source_dtype": "int32",
                                                      "target_dtype": "int64"}),
                                    Instr("Mul", [widened_count, widened_dimension], product),
                                    Instr("Cast", [product], narrowed, attributes={
                                        **_numeric_conversion_attributes(
                                            "int64", "int32"),
                                        "source_dtype": "int64", "target_dtype": "int32",
                                    }),
                                ))
                                element_count = narrowed
                        for name, member_id in zip(named_outputs, outputs, strict=True):
                            output_formal = book.latest_ref(
                                SSA_VALUE,
                                (function_scope_of(kernel), int(named_outputs[name])),
                            )
                            if output_formal is None or str(book.page(SSA_VALUE).cell(
                                output_formal.row, output_formal.column,
                            ).dtype) != "int64":
                                raise ValueError("runtime topk output has no int64 formal")
                            tensor_table.register(SSATensorDescriptor(
                                tensor_id=member_id, data_value_id=member_id,
                                dtype="int64", shape=(), strides=(),
                                storage="temporary", metadata_state="dynamic",
                                shape_value_id=int(shape_value.id),
                                rank_value_id=int(rank_value.id),
                                element_count_value_id=int(element_count.id),
                                arena_id=member_id, allocation_owner=member_id,
                                owns_allocation=True, byte_size=None,
                            ))
                            member_fact_cell = ssa_value_identity_cell(
                                function, member_id, book=book,
                            )
                            member_fact = book.page(SSA_VALUE).cell(
                                member_fact_cell.row, member_fact_cell.column,
                            )
                            member_value = SSAValue(
                                member_id, dtype=member_fact.dtype, shape=(),
                                accounting={
                                    "physical_dtype": member_fact.physical_dtype,
                                    "program_abi_storage": "span",
                                    "program_abi_rank": input_rank,
                                },
                            )
                            own_dynamic_tensor(
                                member_value, element_count, "int64", prefix,
                            )
                    if dynamic_guard:
                        # The guarded source interval proves k nonnegative;
                        # the same Min result supplies both the native k and
                        # the logical output shape/count.  Capacity remains
                        # the input tensor's static axis extent.
                        shape_value, shape_definition = int_vector(
                            static_shape, physical={
                                "physical_dtype": "int32", "storage": "span", "rank": 1,
                            },
                        )
                        axis_value, axis_definition = constant(selected_axis, "int64")
                        shape_address = fresh(dtype="ptr")
                        rank_value, rank_definition = constant(
                            len(static_shape), "int32", physical={
                                "physical_dtype": "int32", "storage": "scalar", "rank": 0,
                            },
                        )
                        prefix.extend((
                            shape_definition, axis_definition,
                            Instr(Handler.GetElementPtr.value,
                                  [shape_value, axis_value], shape_address),
                            Instr(Handler.Store.value,
                                  [actuals[4], shape_address], None),
                            rank_definition,
                        ))
                        other_extent = prod(extent for axis, extent in enumerate(static_shape)
                                            if axis != selected_axis)
                        if other_extent == 1:
                            element_count = actuals[4]
                        else:
                            width, width_definition = constant(other_extent, "int32")
                            element_count = fresh(dtype="int32", physical={
                                "physical_dtype": "int32", "storage": "scalar", "rank": 0,
                            })
                            prefix.extend((width_definition, Instr(
                                Handler.Mul.value, [actuals[4], width], element_count,
                            )))
                        for name, member_id in zip(named_outputs, outputs, strict=True):
                            output_formal = book.latest_ref(
                                SSA_VALUE,
                                (function_scope_of(kernel), int(named_outputs[name])),
                            )
                            if output_formal is None:
                                raise ValueError("named topk output formal lacks its repository identity")
                            output_dtype = str(book.page(SSA_VALUE).cell(
                                output_formal.row, output_formal.column,
                            ).dtype)
                            if output_dtype != "int64":
                                raise ValueError("named topk output formal disagrees with int64 source layout")
                            tensor_table.register(SSATensorDescriptor(
                                tensor_id=member_id, data_value_id=member_id,
                                dtype=output_dtype, shape=(), strides=(),
                                storage="temporary", metadata_state="dynamic",
                                shape_value_id=int(shape_value.id),
                                rank_value_id=int(rank_value.id),
                                element_count_value_id=int(element_count.id),
                                arena_id=member_id, allocation_owner=member_id,
                                owns_allocation=True, byte_size=None,
                            ))
                    for position, actual in enumerate(actuals[1:], 1):
                        prove_native_actual(
                            actual, kernel.args[position],
                            storage="span" if position == 1 else "scalar",
                            rank=1 if position == 1 else 0,
                            extra_sources=dynamic_guard if position == 4 and dynamic_guard else (),
                        )
                    if dynamic_guard:
                        from .identity_concordance import Derived, Mode
                        scope = function_scope_of(function)
                        def held(value):
                            cell = ssa_value_identity_cell(function, int(value.id), book=book)
                            if cell is None:
                                raise ValueError("dynamic topk capacity lost an exact SSA value")
                            return cell
                        input_cell = held(source)
                        selection_input_cell = held(by_role["kw:k"])
                        bound_cell = held(axis_bound)
                        selection_cell = held(bounded_k)
                        shape_cell = held(shape_value)
                        rank_cell = held(rank_value)
                        count_cell = held(element_count)
                        for member_id in outputs:
                            output_cell = ssa_value_identity_cell(
                                function, member_id, book=book,
                            )
                            if output_cell is None:
                                raise ValueError("dynamic topk output lacks its SSA identity")
                            bound = TensorOutputCapacityBound(
                                input_cell, selection_input_cell, bound_cell,
                                selection_cell, output_cell, shape_cell,
                                rank_cell, count_cell, layout_proof[0],
                                tuple(dynamic_guard), int(selected_axis),
                                tuple(static_shape),
                            )
                            book.post(
                                TENSOR_OUTPUT_CAPACITY_BOUND, (scope, member_id),
                                bound, stage=TENSOR_SSA_LOWERING,
                                provenance=Derived(tuple(dict.fromkeys((
                                    input_cell, selection_input_cell, bound_cell,
                                    selection_cell, output_cell, shape_cell,
                                    rank_cell, count_cell, layout_proof[0],
                                    *dynamic_guard,
                                )))), mode=Mode.CONCORD,
                            )
                    if runtime_selection is not None:
                        from .identity_concordance import Derived, Mode
                        scope = function_scope_of(function)
                        def exact(value):
                            cell = ssa_value_identity_cell(
                                function, int(value.id), book=book,
                            )
                            if cell is None:
                                raise ValueError("runtime topk selection lost an SSA cell")
                            return cell
                        for member_id in outputs:
                            output_cell = ssa_value_identity_cell(
                                function, member_id, book=book,
                            )
                            if output_cell is None:
                                raise ValueError("runtime topk leaf has no SSA cell")
                            selection = TensorRuntimeTopkSelection(
                                exact(source), exact(dynamic["shape"]),
                                exact(dynamic["rank"]),
                                exact(dynamic["element_count"]),
                                exact(by_role["kw:k"]),
                                exact(runtime_selection[0]),
                                exact(runtime_selection[1]), output_cell,
                                exact(shape_value), exact(rank_value),
                                exact(element_count), layout_proof[0],
                                runtime_dimension_proof, selected_axis,
                            )
                            book.post(
                                TENSOR_RUNTIME_TOPK_SELECTION,
                                (scope, function.name, member_id), selection,
                                stage=TENSOR_SSA_LOWERING,
                                provenance=Derived(tuple(dict.fromkeys((
                                    selection.input_cell,
                                    selection.input_shape_cell,
                                    selection.input_rank_cell,
                                    selection.input_count_cell,
                                    selection.selection_input_cell,
                                    selection.axis_extent_cell,
                                    selection.selection_cell,
                                    selection.output_cell,
                                    selection.shape_cell,
                                    selection.rank_cell,
                                    selection.count_cell,
                                    selection.layout_cell,
                                    selection.nonnegative_cell,
                                )))), mode=Mode.CONCORD,
                            )
                    attributes = {
                        key: value for key, value in instruction.attributes.items()
                        if key not in {"tensor_candidate", "tensor_operation", "tensor"}
                    }
                    attributes.update({
                        "callee": kernel_name,
                        "result_convention": "ssa.aggregate",
                        "output_ids": outputs,
                        "output_positions": (0, 1),
                        "callee_input_ids": tuple(int(arg.id) for arg in kernel.args[:5]),
                        "callee_output_ids": (
                            int(named_outputs["values"]),
                            int(named_outputs["indices"]),
                        ),
                    })
                    from .native_tensor_family import post_result_family
                    for slot, member in enumerate(member_receipts):
                        native_output = ssa_value_identity_cell(
                            kernel, int(tuple(named_outputs.values())[slot]), book=book,
                        )
                        native_output_fact = book.page(SSA_VALUE).cell(
                            native_output.row, native_output.column,
                        )
                        if native_output_fact.physical_dtype is None:
                            raise ValueError("native named output lacks typed storage proof")
                        from .frame_identity_book import _frame_physical_fields
                        member_physical = _frame_physical_fields(
                            function, outputs[slot], book=book,
                        )
                        if (not isinstance(member_physical, dict)
                                or member_physical.get("storage") != "span"
                                or not isinstance(member_physical.get("rank"), int)):
                            raise ValueError("native named output lacks its authored member geometry")
                        # _ascending_order's raw indices19 already have rank1
                        # in their member record while their extent is runtime.
                        # The kernel formal supplies the element dtype, not a
                        # logical rank. Replacing that rank with len(()) made
                        # indices19 and ensure_tensor20 rank0, contradicting
                        # their exact source ABI. Retain the member geometry.
                        revise_ssa_value(
                            function, outputs[slot], dtype=native_output_fact.dtype,
                            shape=member_shapes[slot], stage=TENSOR_SSA_LOWERING,
                            sources=(native_output, member, layout_proof[0]), book=book,
                            physical={
                                **member_physical,
                                "physical_dtype": native_output_fact.physical_dtype,
                            },
                        )
                        post_result_family(
                            reference, operation, function, kernel, source,
                            layout_proof[0], member, slot, book=book,
                        )
                        if not dynamic_guard and runtime_selection is None:
                            # The selected literal k/axis proved these extents
                            # above. Publish the output owner here, before a
                            # later region receives the leaf as a formal.
                            # Otherwise it has only a display shape and
                            # invents runtime extents for topk's fixed result.
                            member_id = outputs[slot]
                            member_cell = ssa_value_identity_cell(
                                function, member_id, book=book,
                            )
                            member_fact = book.page(SSA_VALUE).cell(
                                member_cell.row, member_cell.column,
                            )
                            register_tensor(SSAValue(
                                member_id, dtype=member_fact.dtype,
                                shape=member_shapes[slot],
                                accounting={
                                    "physical_dtype": member_fact.physical_dtype,
                                    "program_abi_storage": "span",
                                    "program_abi_rank": len(member_shapes[slot]),
                                },
                            ), storage="temporary", metadata_state="static")
                    linked_roots.add(kernel_name)
                    rewritten.extend(prefix)
                    rewritten.append(Instr(
                        "Call", actuals, result, attributes=attributes,
                        source_span=instruction.source_span,
                    ))
                    continue
                raw_basic_axes = instruction.attributes.get(
                    "basic_index_axes"
                )
                basic_source_shape = tuple(map(
                    int,
                    instruction.attributes.get(
                        "basic_index_source_shape", ()
                    ),
                ))
                basic_axes = (
                    None if raw_basic_axes is None else tuple(
                        (tuple(map(int, indices)), bool(drop_axis))
                        for indices, drop_axis in raw_basic_axes
                    )
                )
                if (
                    operation in {"slice", "basic_index"}
                    and args
                    and basic_axes is not None
                    and len(basic_axes) == len(basic_source_shape)
                    and all(
                        drop_axis and len(indices) == 1
                        for indices, drop_axis in basic_axes
                    )
                ):
                    # ``slice`` and ``basic_index`` are two planner spellings
                    # of the same normalized index contract.  Decide scalar
                    # semantics from the authoritative dropped-axis metadata
                    # before either spelling selects a tensor kernel.
                    linear_index = 0
                    for axis, (indices, _drop_axis) in enumerate(basic_axes):
                        linear_index = (
                            linear_index * basic_source_shape[axis]
                            + int(indices[0])
                        )
                    index_value, index_definition = constant(
                        linear_index, "int64"
                    )
                    address = fresh(dtype="ptr")
                    result.shape = ()
                    rewritten.extend((
                        index_definition,
                        Instr(
                            "GetElementPtr", [args[0], index_value], address,
                            attributes={"basic_index_linearized": True},
                            source_span=instruction.source_span,
                        ),
                        Instr(
                            "Load", [address], result,
                            attributes={"basic_index_linearized": True},
                            source_span=instruction.source_span,
                        ),
                    ))
                    continue
                if (
                    len(args) == 2
                    and operation in {"min", "max"}
                    and _attribute(
                        instruction.attributes, "axis", "dim"
                    ) is None
                    and all(booked_element_count(argument) == 1 for argument in args)
                    and booked_element_count(result) == 1
                ):
                    # The planned graph can retain a count-one tensor shape on
                    # a scalar reduction result. Python's two-argument
                    # built-ins still select between two scalar values; they
                    # do not require a dynamic tensor-extent kernel.
                    result.shape = ()
                    rewritten.append(dataclasses.replace(
                        instruction,
                        op={"min": "Min", "max": "Max"}[operation],
                        attributes={
                            key: value
                            for key, value in instruction.attributes.items()
                            if key not in {
                                "tensor_operation", "tensor",
                                "tensor_candidate",
                            }
                        },
                    ))
                    continue
                # Python's two-argument built-ins are the scalar spelling of
                # the same elementwise operations exposed by AbstractTensor.
                # ProcessGraph retains the legal source names ``min``/``max``;
                # normalize them here, after call arity is known, so they use
                # the existing broadcast/scalar repository kernels.  Do not
                # rewrite one-argument iterable reductions.
                if (
                    len(args) == 2
                    and operation in {"min", "max"}
                    and _attribute(
                        instruction.attributes, "axis", "dim"
                    ) is None
                ):
                    operation = {
                        "min": "minimum",
                        "max": "maximum",
                    }[operation]
                prefix: list[Instr] = []
                emitted: list[Instr] = []

                if operation in {"basic_index", "basic_index_store"}:
                    _prefix_ref = instruction.attributes.get(
                        "source_tensor_prefix_slice_cell"
                    )
                    if operation == "basic_index" and _prefix_ref is not None:
                        from .concordance_declarations import (
                            SOURCE_TENSOR_PREFIX_SLICE,
                            TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY,
                            TENSOR_SSA_LOWERING,
                        )
                        from .identity_concordance import Derived, Mode, Ref
                        from .ssa_call_input_adapters import (
                            _numeric_conversion_attributes,
                        )
                        from .ssa_record_return_state import (
                            function_scope_of, posted_ssa_value_fact,
                            revise_ssa_value, ssa_value_identity_cell,
                        )

                        _recipe = (
                            book.page(SOURCE_TENSOR_PREFIX_SLICE).cell(
                                _prefix_ref.row, _prefix_ref.column,
                            ) if isinstance(_prefix_ref, Ref)
                            and _prefix_ref.page == SOURCE_TENSOR_PREFIX_SLICE
                            else None
                        )
                        if not (
                            len(args) == 2 and result is not None
                            and isinstance(_recipe, tuple)
                            and len(_recipe) == 3
                            and _recipe[0] == int(args[0].id)
                            and _recipe[2] == int(args[1].id)
                            and _prefix_ref.row[1] == int(result.id)
                            and book.latest_ref(
                                SOURCE_TENSOR_PREFIX_SLICE,
                                _prefix_ref.row,
                            ) == _prefix_ref
                        ):
                            raise ValueError(
                                "runtime prefix slice has no exact authored recipe"
                            )
                        _source, _upper = args
                        _source_cell = ssa_value_identity_cell(
                            function, int(_source.id), book=book,
                        )
                        _upper_cell = ssa_value_identity_cell(
                            function, int(_upper.id), book=book,
                        )
                        _source_fact = posted_ssa_value_fact(
                            function, int(_source.id), book=book,
                        )
                        _upper_fact = posted_ssa_value_fact(
                            function, int(_upper.id), book=book,
                        )
                        _result_fact = posted_ssa_value_fact(
                            function, int(result.id), book=book,
                        )
                        if not (
                            _source_cell is not None
                            and _upper_cell is not None
                            and _source_fact is not None
                            and _source_fact.storage == "span"
                            and _source_fact.physical_dtype in {
                                "int64", "float64",
                            }
                            and _source_fact.rank is not None
                            and _source_fact.rank >= 1
                            and _upper_fact is not None
                            and _upper_fact.physical_dtype == "int64"
                            and _upper_fact.storage == "scalar"
                            and _upper_fact.rank == 0
                            and _result_fact is not None
                            and _result_fact.dtype == _source_fact.dtype
                            and _result_fact.rank == _source_fact.rank
                            and _result_fact.physical_dtype is None
                        ):
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "runtime prefix slice lacks typed source, "
                                "scalar upper, or rank-preserving result proof",
                            ))
                            rewritten.append(instruction)
                            continue
                        _source_descriptor = register_tensor(
                            _source,
                            storage=("input" if int(_source.id)
                                     in function_argument_ids else "temporary"),
                        )
                        _source_metadata = ensure_dynamic(prefix, _source)
                        _rank = int(_source_fact.rank)
                        _scalar32 = {
                            "physical_dtype": "int32", "storage": "scalar",
                            "rank": 0,
                        }
                        _scalar64 = {
                            "physical_dtype": "int64", "storage": "scalar",
                            "rank": 0,
                        }
                        _axis_extent = dim_extent(prefix, _source, 0)
                        _extent64 = fresh(dtype="int64", physical=_scalar64)
                        _zero64, _zero_def = constant(
                            0, "int64", physical=_scalar64,
                        )
                        _negative = fresh(dtype="bool", physical={
                            "physical_dtype": "bool", "storage": "scalar",
                            "rank": 0,
                        })
                        _negative_offset = fresh(
                            dtype="int64", physical=_scalar64,
                        )
                        _relative = fresh(dtype="int64", physical=_scalar64)
                        _normalized = fresh(dtype="int64", physical=_scalar64)
                        _nonnegative = fresh(dtype="int64", physical=_scalar64)
                        _clipped64 = fresh(dtype="int64", physical=_scalar64)
                        _clipped = fresh(dtype="int32", physical=_scalar32)
                        prefix.extend((
                            _zero_def,
                            Instr("Cast", [_axis_extent], _extent64,
                                  attributes={
                                      **_numeric_conversion_attributes(
                                          "int32", "int64"),
                                      "source_dtype": "int32",
                                      "target_dtype": "int64",
                                      "binding": "prefix-slice-axis-extent",
                                  }),
                            Instr("Lt", [_upper, _zero64], _negative),
                            # INT64_MAX is a valid positive bound. Select zero
                            # on that arm before Add, so even its discarded
                            # relative value has defined int64 arithmetic.
                            Instr("Select", [_negative, _upper, _zero64],
                                  _negative_offset),
                            Instr("Add", [_negative_offset, _extent64],
                                  _relative),
                            Instr("Select", [_negative, _relative, _upper],
                                  _normalized),
                            Instr("Max", [_normalized, _zero64],
                                  _nonnegative),
                            Instr("Min", [_nonnegative, _extent64],
                                  _clipped64),
                            Instr("Cast", [_clipped64], _clipped,
                                  attributes={
                                      **_numeric_conversion_attributes(
                                          "int64", "int32"),
                                      "source_dtype": "int64",
                                      "target_dtype": "int32",
                                      "binding": "prefix-slice-clipped-count",
                                  }),
                        ))
                        _shape, _shape_def = int_vector(
                            (0,) * _rank, physical={
                                "physical_dtype": "int32", "storage": "span",
                                "rank": 1,
                            },
                        )
                        prefix.append(_shape_def)
                        _dimensions = []
                        _geometry_values = [
                            _source_metadata["shape"],
                            _source_metadata["rank"],
                            _source_metadata["element_count"],
                            _axis_extent, _negative_offset, _clipped64, _shape,
                        ]
                        for _axis in range(_rank):
                            _extent = (
                                _clipped if _axis == 0
                                else dim_extent(prefix, _source, _axis)
                            )
                            _index, _index_def = constant(
                                _axis, "int64", physical=_scalar64,
                            )
                            _address = fresh(dtype="ptr")
                            prefix.extend((
                                _index_def,
                                Instr("GetElementPtr", [_shape, _index],
                                      _address,
                                      attributes={"binding": "prefix-slice-shape"}),
                                Instr("Store", [_extent, _address], None,
                                      attributes={"binding": "prefix-slice-shape"}),
                            ))
                            _geometry_values.extend((_extent, _index, _address))
                            _dimensions.append(_extent)
                        # A later empty dimension must zero the entire view before
                        # any earlier wide product is narrowed to an int32 count.
                        # Use the same checked count program as runtime topk.
                        _zero32, _zero32_def = constant(
                            0, "int32", physical=_scalar32,
                        )
                        _one32, _one32_def = constant(
                            1, "int32", physical=_scalar32,
                        )
                        prefix.extend((_zero32_def, _one32_def))
                        _geometry_values.extend((_zero32, _one32))
                        _empty = None
                        for _dimension in _dimensions:
                            _is_zero = fresh(dtype="bool", physical={
                                "physical_dtype": "bool", "storage": "scalar",
                                "rank": 0,
                            })
                            prefix.append(Instr(
                                "Eq", [_dimension, _zero32], _is_zero,
                            ))
                            _geometry_values.append(_is_zero)
                            if _empty is None:
                                _empty = _is_zero
                            else:
                                _either = fresh(dtype="bool", physical={
                                    "physical_dtype": "bool", "storage": "scalar",
                                    "rank": 0,
                                })
                                prefix.append(Instr(
                                    "LOr", [_empty, _is_zero], _either,
                                ))
                                _empty = _either
                                _geometry_values.append(_either)
                        _count = fresh(dtype="int32", physical=_scalar32)
                        prefix.append(Instr(
                            "Select", [_empty, _zero32, _one32], _count,
                        ))
                        _geometry_values.append(_count)
                        for _dimension in _dimensions:
                            _widened_count = fresh(
                                dtype="int64", physical=_scalar64,
                            )
                            _widened_dimension = fresh(
                                dtype="int64", physical=_scalar64,
                            )
                            _product = fresh(dtype="int64", physical=_scalar64)
                            _narrowed = fresh(dtype="int32", physical=_scalar32)
                            prefix.extend((
                                Instr("Cast", [_count], _widened_count,
                                      attributes={"source_dtype": "int32",
                                                  "target_dtype": "int64"}),
                                Instr("Cast", [_dimension], _widened_dimension,
                                      attributes={"source_dtype": "int32",
                                                  "target_dtype": "int64"}),
                                Instr("Mul", [_widened_count,
                                              _widened_dimension], _product),
                                Instr("Cast", [_product], _narrowed,
                                      attributes={
                                          **_numeric_conversion_attributes(
                                              "int64", "int32"),
                                          "source_dtype": "int64",
                                          "target_dtype": "int32",
                                      }),
                            ))
                            _geometry_values.extend((
                                _widened_count, _widened_dimension,
                                _product, _narrowed,
                            ))
                            _count = _narrowed
                        _result_rank, _rank_def = constant(
                            _rank, "int32", physical=_scalar32,
                        )
                        prefix.append(_rank_def)
                        _result_cell = revise_ssa_value(
                            function, int(result.id),
                            dtype=_result_fact.dtype,
                            shape=tuple(_result_fact.shape),
                            stage=TENSOR_SSA_LOWERING,
                            sources=(_prefix_ref, _source_cell, _upper_cell),
                            book=book,
                            physical={
                                "physical_dtype": _source_fact.physical_dtype,
                                "storage": "span", "rank": _rank,
                            },
                        )
                        result.accounting = {
                            **dict(result.accounting or {}),
                            "physical_dtype": _source_fact.physical_dtype,
                            "program_abi_storage": "span",
                            "program_abi_rank": _rank,
                            "tensor_metadata_state": "dynamic",
                            "tensor_shape_value_id": int(_shape.id),
                            "tensor_rank_value_id": int(_result_rank.id),
                            "tensor_element_count_value_id": int(_count.id),
                            "ssa_storage_view": {
                                "storage_value_id": int(_source.id),
                                "view_shape": (),
                                "operation": "prefix_slice",
                            },
                        }
                        register_tensor(
                            result, storage="view", metadata_state="dynamic",
                            alias_of=_source_descriptor.tensor_id,
                            data_value_id=_source_descriptor.data_value_id,
                            extent_values=(_shape, _result_rank, _count),
                            alias_proof=_prefix_ref,
                        )
                        _geometry_cells = tuple(
                            ssa_value_identity_cell(
                                function, int(value.id), book=book,
                            ) for value in (
                                *_geometry_values, _result_rank, _count,
                            )
                        )
                        if _result_cell is None or any(
                            cell is None for cell in _geometry_cells
                        ):
                            raise ValueError(
                                "runtime prefix slice metadata lacks SSA identity"
                            )
                        book.post(
                            TENSOR_RUNTIME_PREFIX_SLICE_GEOMETRY,
                            (function_scope_of(function), function.name,
                             int(result.id)),
                            (int(_source.id), int(_recipe[1]), int(_upper.id),
                             int(_source_metadata["shape"].id),
                             int(_source_metadata["rank"].id),
                             int(_source_metadata["element_count"].id),
                             int(_axis_extent.id), int(_clipped64.id),
                             int(_shape.id), int(_result_rank.id),
                             int(_count.id)),
                            stage=TENSOR_SSA_LOWERING,
                            provenance=Derived(tuple(dict.fromkeys((
                                _prefix_ref, _source_cell, _upper_cell,
                                _result_cell, *_geometry_cells,
                            )))),
                            mode=Mode.CONCORD,
                        )
                        aliases[int(result.id)] = SSAValue(
                            _source.id, dtype=result.dtype or _source.dtype,
                            shape=tuple(result.shape or ()),
                            device=result.device or _source.device,
                            accounting=dict(result.accounting),
                        )
                        rewritten.extend(prefix)
                        continue
                    raw_axes = instruction.attributes.get(
                        "basic_index_axes"
                    )
                    if raw_axes is None and args:
                        source = args[0]
                        authored_indices = (
                            args[1:]
                            if operation == "basic_index"
                            else args[1:-1]
                        )
                        decoded_indices = tuple(
                            constants.get(int(value.id), value)
                            for value in authored_indices
                        )
                        source_shape = tuple(map(
                            int,
                            instruction.attributes.get(
                                "basic_index_source_shape",
                                tuple(source.shape or ()),
                            ),
                        ))
                        # Program: Huffman's weights[selected] and the
                        # bounded int64 span/indices example. The selector
                        # is itself a tensor, so scalar GEP/Load consumed
                        # only indices[0] and left the selected span empty.
                        # The physical SSA rows distinguish this case from
                        # a scalar selector. The selected repository kernel
                        # keeps int64 values and int64 indices in their
                        # declared storage; no float round trip or int32
                        # reinterpretation is valid here.
                        if (
                            operation == "basic_index"
                            and len(authored_indices) == 1
                            and not isinstance(decoded_indices[0], (
                                int, slice,
                            ))
                            and decoded_indices[0] is not Ellipsis
                        ):
                            selector = authored_indices[0]
                            selector_fact = posted_ssa_value_fact(
                                function, int(selector.id), book=book,
                            )
                            source_fact = posted_ssa_value_fact(
                                function, int(source.id), book=book,
                            )
                            result_fact = posted_ssa_value_fact(
                                function, int(result.id), book=book,
                            )
                            if (
                                selector_fact is not None
                                and selector_fact.storage == "span"
                                and selector_fact.rank == 1
                                and selector_fact.physical_dtype == "int64"
                                and tuple(selector_fact.shape) == ()
                                and source_fact is not None
                                and source_fact.storage == "span"
                                and source_fact.physical_dtype
                                in {"int64", "float64"}
                                and source_fact.rank is not None
                                and int(source_fact.rank) >= 1
                                and result_fact is not None
                                and result_fact.storage == "span"
                                and result_fact.rank == source_fact.rank
                                and result_fact.dtype == source_fact.dtype
                                and result_fact.physical_dtype is None
                                and tuple(result.shape) == ()
                            ):
                                # A runtime Indexed result is born before its
                                # selected kernel, so its physical element
                                # field can be unset. The source's exact SSA
                                # span and the selected output formal agree
                                # on the copied element representation. Post
                                # that derivation before owned allocation.
                                kernel_name = (
                                    "index_select_int64_indices_int64"
                                    if source_fact.physical_dtype == "int64"
                                    else "index_select_double_indices_int64"
                                )
                                selected = reference.module.functions.get(
                                    kernel_name
                                )
                                if (
                                    selected is None
                                    or selected.metadata.get(
                                        "ssa_output_argument"
                                    ) != 1
                                    or len(selected.args) <= 1
                                    or tuple(selected.metadata.get(
                                        "named_outputs", ()
                                    )) != (("out", int(selected.args[1].id)),)
                                ):
                                    raise ValueError(
                                        "runtime Indexed selected output formal disagrees"
                                    )
                                link_repository_function(kernel_name, selected)
                                from .ssa_call_input_adapters import (
                                    _infer_pointer_storage,
                                )
                                _infer_pointer_storage({kernel_name: selected})
                                from .ssa_record_return_state import (
                                    ssa_value_identity_cell,
                                    revise_ssa_value as _revise_indexed_result,
                                )
                                from .concordance_declarations import (
                                    TENSOR_SSA_LOWERING as _INDEXED_STAGE,
                                )
                                source_cell = ssa_value_identity_cell(
                                    function, int(source.id), book=book,
                                )
                                output_cell = ssa_value_identity_cell(
                                    selected, int(selected.args[1].id),
                                    book=book,
                                )
                                output_fact = posted_ssa_value_fact(
                                    selected, int(selected.args[1].id),
                                    book=book,
                                )
                                if (
                                    source_cell is None or output_cell is None
                                    or output_fact is None
                                    or output_fact.physical_dtype
                                    != source_fact.physical_dtype
                                ):
                                    raise ValueError(
                                        "runtime Indexed lacks matching source and kernel output proof"
                                    )
                                result_cell = _revise_indexed_result(
                                    function, int(result.id),
                                    dtype=result.dtype,
                                    shape=tuple(result.shape or ()),
                                    stage=_INDEXED_STAGE,
                                    sources=(source_cell, output_cell),
                                    book=book,
                                    physical={
                                        "physical_dtype":
                                        source_fact.physical_dtype,
                                        "storage": "span",
                                        "rank": int(source_fact.rank),
                                    },
                                )
                                if result_cell is None:
                                    raise ValueError(
                                        "runtime Indexed has no result identity"
                                    )
                                result.accounting = _accounting_view(
                                    function, result, module,
                                )
                                result_fact = posted_ssa_value_fact(
                                    function, int(result.id), book=book,
                                )
                            if (
                                selector_fact is not None
                                and selector_fact.storage == "span"
                                and selector_fact.rank == 1
                                and selector_fact.physical_dtype == "int64"
                                and source_fact is not None
                                and source_fact.storage == "span"
                                and source_fact.rank == len(source_shape)
                                and source_fact.physical_dtype
                                in {"int64", "float64"}
                                and result_fact is not None
                                and result_fact.storage == "span"
                                and result_fact.physical_dtype
                                == source_fact.physical_dtype
                                and tuple(selector_fact.shape)
                                == tuple(selector.shape)
                                and len(selector.shape) == 1
                                and source_shape
                                and tuple(result.shape)
                                == (int(selector.shape[0]), *source_shape[1:])
                            ):
                                kernel = (
                                    "index_select_int64_indices_int64"
                                    if source_fact.physical_dtype == "int64"
                                    else "index_select_double_indices_int64"
                                )
                                register_tensor(source, storage=(
                                    "input" if int(source.id)
                                    in function_argument_ids else "temporary"
                                ))
                                register_tensor(result, storage="temporary")
                                shape_value, shape_def = int_vector(source_shape)
                                rank_value, rank_def = constant(
                                    len(source_shape), "int32",
                                )
                                axis_value, axis_def = constant(0, "int32")
                                count_value, count_def = constant(
                                    int(selector.shape[0]), "int32",
                                )
                                rewritten.extend((
                                    shape_def, rank_def, axis_def, count_def,
                                    call(
                                        kernel, [
                                            source, result, shape_value,
                                            rank_value, axis_value,
                                            selector, count_value,
                                        ], result, instruction,
                                        output_argument=1,
                                    ),
                                ))
                                continue
                            if (
                                selector_fact is not None
                                and selector_fact.storage == "span"
                                and selector_fact.rank == 1
                                and selector_fact.physical_dtype == "int64"
                                and tuple(selector_fact.shape) == ()
                                and source_fact is not None
                                and source_fact.storage == "span"
                                and source_fact.rank == 1
                                and source_fact.physical_dtype
                                in {"int64", "float64"}
                                and len(source_shape) == 1
                                and result_fact is not None
                                and result_fact.storage == "span"
                                and result_fact.rank == 1
                                and result_fact.physical_dtype
                                == source_fact.physical_dtype
                                and tuple(result.shape) == ()
                            ):
                                # The selector's runtime element count is
                                # the exact rank-one output extent. Keep its
                                # shape/rank/count SSA identities through the
                                # ordinary metadata publisher and give a
                                # private result its incumbent owned arena.
                                register_tensor(selector, storage=(
                                    "input" if int(selector.id)
                                    in function_argument_ids else "temporary"
                                ))
                                extents = ensure_dynamic(prefix, selector)
                                result.accounting = {
                                    **dict(result.accounting or {}),
                                    "tensor_metadata_state": "dynamic",
                                    "tensor_shape_value_id": int(
                                        extents["shape"].id
                                    ),
                                    "tensor_rank_value_id": int(
                                        extents["rank"].id
                                    ),
                                    "tensor_element_count_value_id": int(
                                        extents["element_count"].id
                                    ),
                                }
                                carrier = (
                                    own_dynamic_tensor(
                                        result, extents["element_count"],
                                        str(source_fact.physical_dtype), prefix,
                                    )
                                    if has_private_region_result(result)
                                    else result
                                )
                                if carrier is result:
                                    register_tensor(result, storage="temporary")
                                shape_value, shape_def = int_vector(source_shape)
                                rank_value, rank_def = constant(1, "int32")
                                axis_value, axis_def = constant(0, "int32")
                                prefix.extend((
                                    shape_def, rank_def, axis_def,
                                ))
                                kernel = (
                                    "index_select_int64_indices_int64"
                                    if source_fact.physical_dtype == "int64"
                                    else "index_select_double_indices_int64"
                                )
                                rewritten.extend((
                                    *prefix,
                                    call(
                                        kernel, [
                                            source, carrier, shape_value,
                                            rank_value, axis_value,
                                            selector, extents["element_count"],
                                        ], result, instruction,
                                        output_argument=1,
                                    ),
                                ))
                                continue
                            if (
                                selector_fact is not None
                                and selector_fact.storage == "span"
                                and selector_fact.rank == 1
                                and selector_fact.physical_dtype == "int64"
                                and tuple(selector_fact.shape) == ()
                                and source_fact is not None
                                and source_fact.storage == "span"
                                and source_fact.rank is not None
                                and int(source_fact.rank) >= 2
                                and source_fact.physical_dtype
                                in {"int64", "float64"}
                                and result_fact is not None
                                and result_fact.storage == "span"
                                and result_fact.rank == source_fact.rank
                                and result_fact.physical_dtype
                                == source_fact.physical_dtype
                                and tuple(result.shape) == ()
                            ):
                                # Every trailing source dimension is copied
                                # for each selected row. The output's first
                                # extent is the selector's runtime count;
                                # its remaining extents are the source's
                                # booked axes. Huffman's membership table is
                                # the rank-two instance of this same law.
                                source_descriptor = register_tensor(source, storage=(
                                    "input" if int(source.id)
                                    in function_argument_ids else "temporary"
                                ))
                                register_tensor(selector, storage=(
                                    "input" if int(selector.id)
                                    in function_argument_ids else "temporary"
                                ))
                                source_rank = int(source_fact.rank)
                                if (
                                    source_descriptor.metadata_state == "static"
                                    and len(source_shape) == source_rank
                                ):
                                    source_shape_value, source_shape_def = (
                                        int_vector(source_shape, physical={
                                            "physical_dtype": "int32",
                                            "storage": "span", "rank": 1,
                                        })
                                    )
                                    source_rank_value, source_rank_def = constant(
                                        source_rank, "int32", physical={
                                            "physical_dtype": "int32",
                                            "storage": "scalar", "rank": 0,
                                        },
                                    )
                                    prefix.extend((
                                        source_shape_def, source_rank_def,
                                    ))
                                    source_axes = []
                                    for axis in range(1, source_rank):
                                        extent, definition = constant(
                                            source_shape[axis], "int32",
                                        )
                                        prefix.append(definition)
                                        source_axes.append(extent)
                                else:
                                    source_extents = ensure_dynamic(prefix, source)
                                    source_shape_value = source_extents["shape"]
                                    source_rank_value = source_extents["rank"]
                                    source_axes = [
                                        dim_extent(prefix, source, axis)
                                        for axis in range(1, source_rank)
                                    ]
                                index_extents = ensure_dynamic(prefix, selector)
                                output_shape, output_shape_def = int_vector(
                                    (0,) * source_rank, physical={
                                        "physical_dtype": "int32",
                                        "storage": "span", "rank": 1,
                                    },
                                )
                                output_rank, output_rank_def = constant(
                                    source_rank, "int32", physical={
                                        "physical_dtype": "int32",
                                        "storage": "scalar", "rank": 0,
                                    },
                                )
                                prefix.extend((output_shape_def, output_rank_def))
                                output_count = index_extents["element_count"]
                                for axis, extent in enumerate((
                                    index_extents["element_count"],
                                    *source_axes,
                                )):
                                    axis_value, axis_def = constant(axis, "int64")
                                    axis_address = fresh(dtype="ptr")
                                    prefix.extend((
                                        axis_def,
                                        Instr(
                                            Handler.GetElementPtr.value,
                                            [output_shape, axis_value],
                                            axis_address,
                                        ),
                                        Instr(
                                            Handler.Store.value,
                                            [extent, axis_address], None,
                                        ),
                                    ))
                                    if axis:
                                        multiplied = fresh(
                                            dtype="int32", physical={
                                                "physical_dtype": "int32",
                                                "storage": "scalar",
                                                "rank": 0,
                                            },
                                        )
                                        prefix.append(Instr(
                                            Handler.Mul.value,
                                            [output_count, extent], multiplied,
                                        ))
                                        output_count = multiplied
                                result.accounting = {
                                    **dict(result.accounting or {}),
                                    "tensor_metadata_state": "dynamic",
                                    "tensor_shape_value_id": int(
                                        output_shape.id
                                    ),
                                    "tensor_rank_value_id": int(
                                        output_rank.id
                                    ),
                                    "tensor_element_count_value_id": int(
                                        output_count.id
                                    ),
                                }
                                carrier = (
                                    own_dynamic_tensor(
                                        result, output_count,
                                        str(source_fact.physical_dtype), prefix,
                                    )
                                    if has_private_region_result(result)
                                    else result
                                )
                                if carrier is result:
                                    register_tensor(result, storage="temporary")
                                axis_value, axis_def = constant(0, "int32")
                                prefix.append(axis_def)
                                kernel = (
                                    "index_select_int64_indices_int64"
                                    if source_fact.physical_dtype == "int64"
                                    else "index_select_double_indices_int64"
                                )
                                rewritten.extend((
                                    *prefix,
                                    call(
                                        kernel, [
                                            source, carrier,
                                            source_shape_value,
                                            source_rank_value, axis_value,
                                            selector,
                                            index_extents["element_count"],
                                        ], result, instruction,
                                        output_argument=1,
                                    ),
                                ))
                                continue
                        source_sequence = resident_sequence(source)
                        resident_row_shape = tuple(
                            (source_sequence.column_shapes or ((),))[0]
                        ) if (
                            source_sequence is not None
                            and len(source_sequence.column_value_ids) == 1
                        ) else ()
                        accounted_row_shape = tuple(map(
                            int,
                            (source.accounting or {}).get(
                                "sequence_row_shape", (),
                            ),
                        ))
                        proven_row_shape = (
                            accounted_row_shape or resident_row_shape
                        )
                        declared_source_rank = int(
                            (source.accounting or {}).get(
                                "program_abi_rank", 0
                            ) or 0
                        )
                        if not declared_source_rank and proven_row_shape:
                            # A resident sequence is a dynamic leading axis
                            # followed by its proven element extents.  That
                            # rank is already carried by the physical arena's
                            # concorded row layout; it is not a ProgramABI
                            # boundary and must not require a second rank map.
                            declared_source_rank = 1 + len(proven_row_shape)
                        if declared_source_rank > len(source_shape):
                            row_shape = (
                                proven_row_shape
                                if proven_row_shape else None
                            )
                            if (
                                row_shape == ()
                                and source_sequence is not None
                                and declared_source_rank
                                == 1 + len(source_shape)
                            ):
                                row_shape = source_shape
                            outer_extent = (
                                sequence_outer_extent(source_sequence)
                                if source_sequence is not None else None
                            )
                            if (
                                row_shape is not None
                                and outer_extent is not None
                                and declared_source_rank
                                == 1 + len(row_shape)
                            ):
                                source_shape = (outer_extent, *row_shape)
                                source.shape = source_shape
                            else:
                                published = owner_tensor_descriptor(source)
                            if (
                                source_shape == tuple(source.shape or ())
                                and len(source_shape) == declared_source_rank
                            ):
                                pass
                            elif (
                                published is not None
                                and len(tuple(published.shape))
                                == declared_source_rank
                            ):
                                source_shape = tuple(published.shape)
                                source.shape = source_shape
                                source.accounting = {
                                    **dict(source.accounting or {}),
                                    "ssa_storage_view": {
                                        "storage_value_id": int(
                                            published.data_value_id
                                        ),
                                        "view_shape": source_shape,
                                        "operation": (
                                            "source_integral_tensor_view"
                                        ),
                                    },
                                }
                        if not source_shape:
                            source_accounting = dict(source.accounting or {})
                            fixed_length = source_accounting.get(
                                "program_abi_fixed_length"
                            )
                            if (
                                fixed_length is not None
                                and int(source_accounting.get(
                                    "program_abi_rank", 0
                                ) or 0) == 1
                            ):
                                source_shape = (int(fixed_length),)
                        if (
                            source_shape
                            and (
                                declared_source_rank <= 0
                                or len(source_shape) == declared_source_rank
                            )
                            and authored_indices
                            and all(
                                isinstance(value, (int, slice))
                                or value is Ellipsis
                                for value in decoded_indices
                            )
                        ):
                            try:
                                normalized_axes, selected_shape = (
                                    normalize_basic_index(
                                        tuple(decoded_indices), source_shape
                                    )
                                )
                            except IndexError as exc:
                                source_definitions = tuple(
                                    {
                                        "block": defining_block,
                                        "op": defining.op,
                                        "args": tuple(
                                            int(value.id)
                                            for value in defining.args
                                        ),
                                        "attributes": dict(
                                            defining.attributes or {}
                                        ),
                                    }
                                    for defining_block, defining_body
                                    in function.blocks.items()
                                    for defining in defining_body.instrs
                                    if defining.res is not None
                                    and int(defining.res.id) == int(source.id)
                                )
                                raise ValueError(
                                    "repository tensor index contract failed: "
                                    f"function={function_name!r} "
                                    f"block={block_name!r} "
                                    f"operation={operation!r} "
                                    f"result=%{int(result.id)} "
                                    f"source=%{int(source.id)} "
                                    f"source_shape={source_shape!r} "
                                    f"selectors={decoded_indices!r} "
                                    f"source_accounting="
                                    f"{dict(source.accounting or {})!r} "
                                    f"source_definitions="
                                    f"{source_definitions!r} "
                                    f"source_owner={source_owner!r} "
                                    f"sequence_table_owners="
                                    f"{tuple(getattr(module, 'sequence_tables', {}))!r} "
                                    f"owner_sequences="
                                    f"{tuple((int(item.sequence_id), tuple(item.column_value_ids), tuple(item.column_shapes)) for item in (sequence_table.sequences.values() if sequence_table is not None else ()))!r} "
                                    f"capacity_sources="
                                    f"{tuple((module.functions.get(source_owner).metadata or {}).get('sequence_capacity_sources', ())) if module.functions.get(source_owner) is not None else ()!r}"
                                ) from exc
                            raw_axes = tuple(
                                (tuple(map(int, indices)), bool(drop_axis))
                                for indices, drop_axis in normalized_axes
                            )
                            result.shape = tuple(map(int, selected_shape))
                            instruction = dataclasses.replace(
                                instruction,
                                attributes={
                                    **instruction.attributes,
                                    "basic_index_axes": raw_axes,
                                    "basic_index_source_shape": source_shape,
                                },
                            )
                    if (
                        operation == "basic_index"
                        and raw_axes is None
                        and args
                    ):
                        source = args[0]
                        source_accounting = dict(source.accounting or {})
                        source_rank = int(source_accounting.get(
                            "program_abi_rank", 0
                        ) or 0)
                        source_sequence = resident_sequence(source)
                        resident_row_shape = tuple(
                            (source_sequence.column_shapes or ((),))[0]
                        ) if (
                            source_sequence is not None
                            and len(source_sequence.column_value_ids) == 1
                        ) else ()
                        proven_row_shape = tuple(map(
                            int,
                            source_accounting.get(
                                "sequence_row_shape", (),
                            ),
                        )) or resident_row_shape
                        if not source_rank and proven_row_shape:
                            source_rank = 1 + len(proven_row_shape)
                        authored_indices = args[1:]
                        decoded_indices = tuple(
                            constants.get(int(value.id), value)
                            for value in authored_indices
                        )
                        selected_axes = tuple(
                            axis for axis, value in enumerate(decoded_indices)
                            if not (
                                isinstance(value, slice)
                                and value.start is None
                                and value.stop is None
                                and value.step is None
                            )
                        )
                        # A trailing scalar selection such as ``x[:, i]`` is
                        # the dynamic counterpart of the existing static
                        # basic-index path.  Its runtime shape/rank/count are
                        # the concorded span operands above; submit the same
                        # repository index-select law instead of flattening a
                        # Python slice token into an address.
                        if (
                            source_rank > 1
                            and len(authored_indices) == source_rank
                            and selected_axes == (source_rank - 1,)
                        ):
                            shape_id = source_accounting.get(
                                "tensor_shape_value_id"
                            )
                            rank_id = source_accounting.get(
                                "tensor_rank_value_id"
                            )
                            values_by_id = {
                                int(value.id): value
                                for value in function.args
                            }
                            if (
                                (
                                    shape_id is None
                                    or rank_id is None
                                    or int(shape_id) not in values_by_id
                                    or int(rank_id) not in values_by_id
                                )
                                and (
                                    source_sequence is None
                                    or len(source_sequence.column_value_ids)
                                    == 1
                                )
                            ):
                                sequence_id = int(source_accounting.get(
                                    "sequence_id",
                                    -1 if source_sequence is None else
                                    source_sequence.sequence_id,
                                ))
                                length_value = next((
                                    value for value in function.args
                                    if int((value.accounting or {}).get(
                                        "sequence_length_for", -1
                                    )) == sequence_id
                                ), None)
                                row_shape = proven_row_shape
                                if (
                                    not row_shape
                                    and source_rank
                                    == 1 + len(tuple(source.shape or ()))
                                ):
                                    row_shape = tuple(source.shape)
                                if (
                                    length_value is not None
                                    and source_rank == 2
                                    and len(row_shape) == 1
                                ):
                                    live_length_i64 = fresh(dtype="int64")
                                    live_length = fresh(dtype="int32")
                                    shape_value, shape_definition = int_vector(
                                        (0, prod(row_shape))
                                    )
                                    rank_value, rank_definition = constant(
                                        source_rank, "int32"
                                    )
                                    zero, zero_definition = constant(0, "int64")
                                    row_width, row_width_definition = constant(
                                        prod(row_shape), "int32"
                                    )
                                    first_address = fresh(dtype="ptr")
                                    source_element_count = fresh(dtype="int32")
                                    prefix.extend((
                                        shape_definition,
                                        Instr(
                                            Handler.Load.value,
                                            [length_value], live_length_i64,
                                            attributes={
                                                "binding": "sequence-tensor-length",
                                            },
                                        ),
                                        Instr(
                                            Handler.Cast.value,
                                            [live_length_i64], live_length,
                                            attributes={
                                                "source_dtype": "int64",
                                                "target_dtype": "int32",
                                                "binding": "sequence-tensor-length",
                                            },
                                        ),
                                        zero_definition,
                                        Instr(
                                            Handler.GetElementPtr.value,
                                            [shape_value, zero], first_address,
                                        ),
                                        Instr(
                                            Handler.Store.value,
                                            [live_length, first_address], None,
                                        ),
                                        row_width_definition,
                                        rank_definition,
                                        Instr(
                                            Handler.Mul.value,
                                            [live_length, row_width],
                                            source_element_count,
                                        ),
                                    ))
                                    shape_id = int(shape_value.id)
                                    rank_id = int(rank_value.id)
                                    values_by_id[shape_id] = shape_value
                                    values_by_id[rank_id] = rank_value
                                    source.accounting = {
                                        **source_accounting,
                                        "tensor_metadata_state": "dynamic",
                                        "tensor_shape_value_id": shape_id,
                                        "tensor_rank_value_id": rank_id,
                                        "tensor_element_count_value_id": int(
                                            source_element_count.id
                                        ),
                                        "sequence_id": int(
                                            sequence_id
                                        ),
                                        "sequence_row_shape": row_shape,
                                    }
                                    source_accounting = dict(source.accounting)
                            if shape_id is None or rank_id is None:
                                shortfalls.append(TensorSSALoweringShortfall(
                                    function_name,
                                    block_name,
                                    operation,
                                    "dynamic basic index has no concorded "
                                    "shape/rank operands",
                                ))
                                rewritten.append(instruction)
                                continue
                            shape_value = values_by_id.get(int(shape_id))
                            rank_value = values_by_id.get(int(rank_id))
                            if shape_value is None or rank_value is None:
                                shortfalls.append(TensorSSALoweringShortfall(
                                    function_name,
                                    block_name,
                                    operation,
                                    "dynamic basic index shape/rank identities "
                                    "are not region formals",
                                ))
                                rewritten.append(instruction)
                                continue
                            source_descriptor = register_tensor(
                                source,
                                storage=(
                                    "input"
                                    if int(source.id) in function_argument_ids
                                    else "temporary"
                                ),
                            )
                            dynamic_index = authored_indices[-1]
                            decoded_index = decoded_indices[-1]
                            if isinstance(decoded_index, int):
                                index_vector, index_definition = int_vector((
                                    decoded_index,
                                ))
                                index_definition.attributes["binding"] = (
                                    "dynamic-index-vector"
                                )
                                index_prefix = (index_definition,)
                            else:
                                index_vector, index_definition = int_vector((0,))
                                index_definition.attributes["binding"] = (
                                    "dynamic-index-vector"
                                )
                                index_zero, index_zero_definition = constant(
                                    0, "int64"
                                )
                                index_address = fresh(dtype="ptr")
                                index_prefix = (
                                    index_definition,
                                    index_zero_definition,
                                    Instr(
                                        Handler.GetElementPtr.value,
                                        [index_vector, index_zero],
                                        index_address,
                                        attributes={
                                            "binding": "dynamic-index-vector",
                                        },
                                    ),
                                    Instr(
                                        Handler.Store.value,
                                        [dynamic_index, index_address],
                                        None,
                                        attributes={
                                            "binding": "dynamic-index-vector",
                                        },
                                    ),
                                )
                            output_rank, output_rank_def = constant(
                                source_rank - 1, "int32"
                            )
                            output_count = fresh(dtype="int32")
                            axis_value, axis_def = constant(
                                source_rank - 1, "int32"
                            )
                            selection_count, selection_count_def = constant(
                                1, "int32"
                            )
                            # Rule: post first; accounting is a view.  The
                            # axis-dropping view's span storage and rank are
                            # a REVISE of the result's row, DERIVED from the
                            # source tensor's cell.
                            if lowering_at[0] is not None:
                                from .concordance_declarations import (
                                    TENSOR_SSA_LOWERING as _INDEX_STAGE,
                                )
                                from .ssa_record_return_state import (
                                    revise_ssa_value as _revise_index_view,
                                    ssa_value_identity_cell as _index_cell,
                                )
                                _revise_index_view(
                                    lowering_at[0], int(result.id),
                                    dtype=result.dtype,
                                    shape=tuple(result.shape or ()),
                                    stage=_INDEX_STAGE,
                                    sources=tuple(
                                        cell for cell in (_index_cell(
                                            lowering_at[0], int(source.id),
                                            book=identity_book(module),
                                        ),) if cell is not None
                                    ),
                                    book=identity_book(module),
                                    physical={
                                        "storage": "span",
                                        "rank": int(source_rank - 1),
                                    },
                                )
                            result.accounting = {
                                **_accounting_view(
                                    lowering_at[0], result, module,
                                ),
                                "tensor_metadata_state": "dynamic",
                                # Dropping the final axis leaves the leading
                                # source extents contiguous at this pointer.
                                "tensor_shape_value_id": int(shape_value.id),
                                "tensor_rank_value_id": int(output_rank.id),
                                "tensor_element_count_value_id": int(
                                    output_count.id
                                ),
                            }
                            register_tensor(result, storage="temporary")
                            slice_ids = {
                                int(value.id)
                                for value, decoded in zip(
                                    authored_indices, decoded_indices
                                )
                                if isinstance(decoded, slice)
                            }
                            if slice_ids:
                                rewritten = [
                                    prior for prior in rewritten
                                    if not (
                                        prior.op == Handler.Const.value
                                        and prior.res is not None
                                        and int(prior.res.id) in slice_ids
                                    )
                                ]
                            prefix.extend((
                                *index_prefix,
                                output_rank_def,
                                Instr(
                                    Handler.Load.value,
                                    [shape_value],
                                    output_count,
                                    attributes={
                                        "binding": "dynamic-index-output-count",
                                    },
                                ),
                                axis_def,
                                selection_count_def,
                            ))
                            emitted.append(call(
                                "index_select_double",
                                [
                                    source, result, shape_value, rank_value,
                                    axis_value, index_vector,
                                    selection_count,
                                ],
                                result,
                                instruction,
                                output_argument=1,
                            ))
                            rewritten.extend((*prefix, *emitted))
                            continue
                    if raw_axes is not None and args:
                        axes = tuple(
                            (tuple(map(int, indices)), bool(drop_axis))
                            for indices, drop_axis in raw_axes
                        )
                        source = args[0]
                        source_shape = tuple(map(
                            int,
                            instruction.attributes.get(
                                "basic_index_source_shape",
                                tuple(source.shape or ()),
                            ),
                        ))
                        if source_shape:
                            result.shape = tuple(
                                (
                                    len(axes[axis][0])
                                    if axis < len(axes) else source_shape[axis]
                                )
                                for axis in range(len(source_shape))
                                if axis >= len(axes) or not axes[axis][1]
                            )
                        source_descriptor = register_tensor(
                            source,
                            storage=(
                                "input"
                                if int(source.id) in function_argument_ids
                                else "temporary"
                            ),
                        )
                        if operation == "basic_index_store" and len(args) >= 2:
                            value = args[-1]
                            register_tensor(
                                value,
                                storage=(
                                    "input"
                                    if int(value.id) in function_argument_ids
                                    else "temporary"
                                ),
                            )
                            offsets = [0]
                            flattened_indices = []
                            for indices, _drop_axis in axes:
                                flattened_indices.extend(indices)
                                offsets.append(len(flattened_indices))
                            shape_value, shape_def = int_vector(source_shape)
                            offsets_value, offsets_def = int_vector(offsets)
                            indices_value, indices_def = int_vector(
                                flattened_indices
                            )
                            rank_value, rank_def = constant(
                                len(source_shape), "int32"
                            )
                            known_value_count = booked_element_count(value)
                            if known_value_count is None:
                                shortfalls.append(TensorSSALoweringShortfall(
                                    function.name,
                                    block.name,
                                    operation,
                                    "index assignment value extent is unknown",
                                ))
                                rewritten.append(instruction)
                                continue
                            value_count, value_count_def = constant(
                                known_value_count,
                                "int32",
                            )
                            result.shape = source_shape
                            inplace_store = str(instruction.op) == "IndexedStore"
                            if inplace_store:
                                register_tensor(
                                    result,
                                    storage="view",
                                    alias_of=int(source_descriptor.tensor_id),
                                    data_value_id=int(
                                        source_descriptor.data_value_id
                                    ),
                                )
                                aliases[int(result.id)] = source
                            else:
                                register_tensor(result, storage="temporary")
                            prefix.extend((
                                shape_def, rank_def, offsets_def, indices_def,
                                value_count_def,
                            ))
                            emitted.append(call(
                                (
                                    "index_assign_double"
                                    if inplace_store else "index_set_double"
                                ),
                                (
                                    [
                                        source, shape_value, rank_value,
                                        offsets_value, indices_value, value,
                                        value_count,
                                    ]
                                    if inplace_store else [
                                        source, result, shape_value, rank_value,
                                        offsets_value, indices_value, value,
                                        value_count,
                                    ]
                                ),
                                result,
                                instruction,
                                output_argument=(None if inplace_store else 1),
                            ))
                            rewritten.extend((*prefix, *emitted))
                            continue
                        if operation == "basic_index":
                            scalar_selection = (
                                len(axes) == len(source_shape)
                                and all(
                                    bool(drop_axis) and len(indices) == 1
                                    for indices, drop_axis in axes
                                )
                            )
                            if scalar_selection:
                                # Every integer-indexed axis is removed by
                                # Python indexing.  The result is one scalar,
                                # not a dynamic one-element tensor requiring
                                # index_select/broadcast extent metadata.
                                linear_index = 0
                                for axis, (indices, _drop_axis) in enumerate(
                                    axes
                                ):
                                    linear_index = (
                                        linear_index * source_shape[axis]
                                        + int(indices[0])
                                    )
                                index_value, index_definition = constant(
                                    linear_index, "int64"
                                )
                                address = fresh(dtype="ptr")
                                result.shape = ()
                                rewritten.extend((
                                    index_definition,
                                    Instr(
                                        "GetElementPtr",
                                        [source, index_value],
                                        address,
                                        attributes={
                                            "basic_index_linearized": True,
                                        },
                                        source_span=instruction.source_span,
                                    ),
                                    Instr(
                                        "Load", [address], result,
                                        attributes={
                                            "basic_index_linearized": True,
                                        },
                                        source_span=instruction.source_span,
                                    ),
                                ))
                                continue
                            changed_axes = tuple(
                                (axis, indices)
                                for axis, (indices, _drop_axis) in enumerate(axes)
                                if tuple(indices) != tuple(range(source_shape[axis]))
                            )
                            if not changed_axes:
                                register_tensor(
                                    result,
                                    storage="view",
                                    alias_of=source_descriptor.tensor_id,
                                    data_value_id=source_descriptor.data_value_id,
                                )
                                aliases[int(result.id)] = SSAValue(
                                    source.id,
                                    dtype=result.dtype or source.dtype,
                                    shape=tuple(result.shape),
                                    device=result.device or source.device,
                                    accounting=dict(source.accounting),
                                )
                                continue
                            current = source
                            current_shape = list(source_shape)
                            for selection_index, (axis, indices) in enumerate(
                                reversed(changed_axes)
                            ):
                                last = selection_index == len(changed_axes) - 1
                                destination_shape = list(current_shape)
                                destination_shape[axis] = len(indices)
                                destination = (
                                    result if last else fresh(
                                        shape=tuple(destination_shape),
                                        dtype=result.dtype or source.dtype,
                                    )
                                )
                                register_tensor(destination, storage="temporary")
                                shape_value, shape_def = int_vector(current_shape)
                                indices_value, indices_def = int_vector(indices)
                                rank_value, rank_def = constant(
                                    len(current_shape), "int32"
                                )
                                axis_value, axis_def = constant(axis, "int32")
                                count_value, count_def = constant(
                                    len(indices), "int32"
                                )
                                prefix.extend((
                                    shape_def, rank_def, axis_def,
                                    indices_def, count_def,
                                ))
                                emitted.append(call(
                                    "index_select_double",
                                    [
                                        current, destination, shape_value,
                                        rank_value, axis_value, indices_value,
                                        count_value,
                                    ],
                                    destination, instruction,
                                    output_argument=1,
                                ))
                                current = destination
                                current_shape = destination_shape
                            rewritten.extend((*prefix, *emitted))
                            continue

                # Shape-only operations do not exist at runtime. Preserve the
                # result's shape/dtype annotation while aliasing its storage.
                if operation in _VIEW_OPERATIONS and args:
                    source = args[0]
                    explicit_reshape_shape = False
                    runtime_unsqueeze = False
                    runtime_flatten = False
                    complete_dynamic = False
                    if operation in {"unsqueeze", "squeeze"}:
                        # These CHANGE the axes, and the annotation they were
                        # given may predate the moment the operand's shape was
                        # proven.  The axis is right here, so state the result
                        # from the operand rather than trusting whatever the
                        # annotation held: ``b.unsqueeze(-1)`` carrying ``b``'s
                        # own shape made the matmul that consumes it read a
                        # column as a vector.
                        extents = tuple(int(e) for e in (source.shape or ()))
                        axis = _attribute(instruction.attributes, "dim", "axis")
                        if axis is None and len(args) > 1:
                            axis = next((
                                int(literal)
                                for argument in args[1:]
                                for literal in (
                                    constants.get(int(argument.id)),
                                )
                                if isinstance(literal, int)
                                and not isinstance(literal, bool)
                            ), None)
                        if extents and axis is not None:
                            axis = int(axis)
                            if operation == "unsqueeze":
                                position = (
                                    axis if axis >= 0
                                    else axis + len(extents) + 1
                                )
                                if 0 <= position <= len(extents):
                                    result.shape = (
                                        extents[:position] + (1,)
                                        + extents[position:]
                                    )
                            else:
                                position = (
                                    axis if axis >= 0 else axis + len(extents)
                                )
                                if (
                                    0 <= position < len(extents)
                                    and extents[position] == 1
                                ):
                                    result.shape = (
                                        extents[:position]
                                        + extents[position + 1:]
                                    )
                        if operation == "unsqueeze" and not extents:
                            from .concordance_declarations import (
                                CANONICAL_VALUE, TENSOR_RUNTIME_VIEW_GEOMETRY,
                                TENSOR_SSA_LOWERING,
                            )
                            from .concorded_literals import (
                                MISSING as MISSING_LITERAL,
                                literal_from_identity,
                            )
                            from .identity_concordance import Derived, Mode
                            from .ssa_record_return_state import (
                                function_scope_of, posted_ssa_value_fact,
                                revise_ssa_value, ssa_value_identity_cell,
                            )

                            source_cell = ssa_value_identity_cell(
                                function, int(source.id), book=book,
                            )
                            source_fact = posted_ssa_value_fact(
                                function, int(source.id), book=book,
                            )
                            axis_cell = (
                                ssa_value_identity_cell(
                                    function, int(args[1].id), book=book,
                                ) if len(args) == 2 else None
                            )
                            axis_origins = (
                                tuple(parent for parent, _stage in
                                      book.edges_into(axis_cell)
                                      if parent.page == CANONICAL_VALUE)
                                if axis_cell is not None else ()
                            )
                            axis_literal, literal_cells = (
                                literal_from_identity(book, axis_origins[0])
                                if len(axis_origins) == 1 else
                                (MISSING_LITERAL, ())
                            )
                            source_rank = (
                                None if source_fact is None else
                                source_fact.rank
                            )
                            local_axis = constants.get(int(args[1].id)) if len(args) == 2 else None
                            if (
                                source_cell is None
                                or source_fact is None
                                or source_fact.physical_dtype is None
                                or source_fact.storage != "span"
                                or source_rank is None or source_rank < 1
                                or type(axis_literal) is not int
                                or local_axis != axis_literal
                            ):
                                shortfalls.append(TensorSSALoweringShortfall(
                                    function_name, block_name, operation,
                                    "runtime unsqueeze lacks exact source rank, "
                                    "physical span or literal axis proof",
                                ))
                                rewritten.append(instruction)
                                continue
                            normalized_axis = (
                                axis_literal if axis_literal >= 0
                                else axis_literal + source_rank + 1
                            )
                            if not 0 <= normalized_axis <= source_rank:
                                shortfalls.append(TensorSSALoweringShortfall(
                                    function_name, block_name, operation,
                                    "runtime unsqueeze axis is outside the "
                                    "proven source rank",
                                ))
                                rewritten.append(instruction)
                                continue
                            scalar32 = {
                                "physical_dtype": "int32", "storage": "scalar",
                                "rank": 0,
                            }
                            register_tensor(
                                source,
                                storage=(
                                    "input" if int(source.id)
                                    in function_argument_ids else "temporary"
                                ),
                            )
                            source_metadata = ensure_dynamic(rewritten, source)
                            shape_value, shape_definition = int_vector(
                                (0,) * (source_rank + 1), physical={
                                    "physical_dtype": "int32", "storage": "span",
                                    "rank": 1,
                                },
                            )
                            rank_value, rank_definition = constant(
                                source_rank + 1, "int32", physical=scalar32,
                            )
                            unit, unit_definition = constant(
                                1, "int32", physical=scalar32,
                            )
                            rewritten.extend((shape_definition, rank_definition,
                                              unit_definition))
                            shape_program_values = [unit]
                            for output_axis in range(source_rank + 1):
                                dimension = unit
                                if output_axis != normalized_axis:
                                    input_axis = (
                                        output_axis if output_axis < normalized_axis
                                        else output_axis - 1
                                    )
                                    source_index, source_index_def = constant(
                                        input_axis, "int64",
                                    )
                                    source_address = fresh(dtype="ptr")
                                    dimension = fresh(
                                        dtype="int32", physical=scalar32,
                                    )
                                    rewritten.extend((
                                        source_index_def,
                                        Instr("GetElementPtr", [
                                            source_metadata["shape"], source_index,
                                        ], source_address),
                                        Instr("Load", [source_address], dimension),
                                    ))
                                    shape_program_values.extend((
                                        source_index, source_address, dimension,
                                    ))
                                result_index, result_index_def = constant(
                                    output_axis, "int64",
                                )
                                result_address = fresh(dtype="ptr")
                                rewritten.extend((
                                    result_index_def,
                                    Instr("GetElementPtr", [
                                        shape_value, result_index,
                                    ], result_address),
                                    Instr("Store", [dimension, result_address],
                                          None),
                                ))
                                shape_program_values.extend((
                                    result_index, result_address,
                                ))
                            result_cell = revise_ssa_value(
                                function, int(result.id), dtype=result.dtype,
                                shape=tuple(result.shape or ()),
                                stage=TENSOR_SSA_LOWERING,
                                sources=(source_cell, axis_cell,
                                         axis_origins[0], *literal_cells),
                                book=book, physical={
                                    "physical_dtype": source_fact.physical_dtype,
                                    "storage": "span", "rank": source_rank + 1,
                                },
                            )
                            if result_cell is None:
                                raise ValueError(
                                    "runtime unsqueeze result has no SSA identity"
                                )
                            result.accounting = {
                                **dict(result.accounting or {}),
                                "physical_dtype": source_fact.physical_dtype,
                                "program_abi_storage": "span",
                                "program_abi_rank": source_rank + 1,
                                "tensor_metadata_state": "dynamic",
                                "tensor_shape_value_id": int(shape_value.id),
                                "tensor_rank_value_id": int(rank_value.id),
                                "tensor_element_count_value_id": int(
                                    source_metadata["element_count"].id
                                ),
                                "ssa_storage_view": {
                                    "storage_value_id": int(source.id),
                                    "view_shape": (),
                                    "operation": "unsqueeze",
                                },
                            }
                            geometry_values = (
                                source_metadata["shape"],
                                source_metadata["rank"],
                                source_metadata["element_count"],
                                shape_value, rank_value,
                                *shape_program_values,
                            )
                            geometry_cells = tuple(
                                ssa_value_identity_cell(
                                    function, int(value.id), book=book,
                                ) for value in geometry_values
                            )
                            if any(cell is None for cell in geometry_cells):
                                raise ValueError(
                                    "runtime unsqueeze metadata lacks SSA identity"
                                )
                            book.post(
                                TENSOR_RUNTIME_VIEW_GEOMETRY,
                                (function_scope_of(function), function.name,
                                 int(result.id)),
                                ("unsqueeze", int(source.id), int(args[1].id),
                                 int(normalized_axis),
                                 int(source_metadata["shape"].id),
                                 int(source_metadata["rank"].id),
                                 int(source_metadata["element_count"].id),
                                 int(shape_value.id), int(rank_value.id)),
                                stage=TENSOR_SSA_LOWERING,
                                provenance=Derived(tuple(dict.fromkeys((
                                    source_cell, axis_cell, axis_origins[0],
                                    *literal_cells, result_cell,
                                    *geometry_cells,
                                )))), mode=Mode.CONCORD,
                            )
                            runtime_unsqueeze = True
                    if operation == "flatten" and not tuple(source.shape or ()):
                        from .concordance_declarations import (
                            TENSOR_RUNTIME_VIEW_GEOMETRY,
                            TENSOR_SSA_LOWERING,
                        )
                        from .identity_concordance import Derived, Mode
                        from .ssa_record_return_state import (
                            function_scope_of, posted_ssa_value_fact,
                            revise_ssa_value, ssa_value_identity_cell,
                        )

                        source_cell = ssa_value_identity_cell(
                            function, int(source.id), book=book,
                        )
                        source_fact = posted_ssa_value_fact(
                            function, int(source.id), book=book,
                        )
                        if (
                            source_cell is None or source_fact is None
                            or source_fact.physical_dtype is None
                            or source_fact.storage != "span"
                            or type(source_fact.rank) is not int
                            or source_fact.rank < 1
                            or len(args) != 1
                        ):
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "runtime flatten lacks exact source rank, "
                                "physical span or sole receiver proof",
                            ))
                            rewritten.append(instruction)
                            continue
                        register_tensor(
                            source,
                            storage=(
                                "input" if int(source.id)
                                in function_argument_ids else "temporary"
                            ),
                        )
                        source_metadata = ensure_dynamic(rewritten, source)
                        scalar32 = {
                            "physical_dtype": "int32", "storage": "scalar",
                            "rank": 0,
                        }
                        shape_value, shape_definition = int_vector(
                            (0,), physical={
                                "physical_dtype": "int32", "storage": "span",
                                "rank": 1,
                            },
                        )
                        rank_value, rank_definition = constant(
                            1, "int32", physical=scalar32,
                        )
                        shape_index, index_definition = constant(0, "int64")
                        shape_address = fresh(dtype="ptr")
                        rewritten.extend((
                            shape_definition, rank_definition,
                            index_definition,
                            Instr("GetElementPtr", [shape_value, shape_index],
                                  shape_address),
                            Instr("Store", [source_metadata["element_count"],
                                            shape_address], None),
                        ))
                        result_cell = revise_ssa_value(
                            function, int(result.id), dtype=result.dtype,
                            shape=tuple(result.shape or ()),
                            stage=TENSOR_SSA_LOWERING,
                            sources=(source_cell,), book=book,
                            physical={
                                "physical_dtype": source_fact.physical_dtype,
                                "storage": "span", "rank": 1,
                            },
                        )
                        if result_cell is None:
                            raise ValueError(
                                "runtime flatten result has no SSA identity"
                            )
                        result.accounting = {
                            **dict(result.accounting or {}),
                            "physical_dtype": source_fact.physical_dtype,
                            "program_abi_storage": "span",
                            "program_abi_rank": 1,
                            "tensor_metadata_state": "dynamic",
                            "tensor_shape_value_id": int(shape_value.id),
                            "tensor_rank_value_id": int(rank_value.id),
                            "tensor_element_count_value_id": int(
                                source_metadata["element_count"].id
                            ),
                            "ssa_storage_view": {
                                "storage_value_id": int(source.id),
                                "view_shape": (),
                                "operation": "flatten",
                            },
                        }
                        geometry_values = (
                            source_metadata["shape"],
                            source_metadata["rank"],
                            source_metadata["element_count"],
                            shape_value, rank_value, shape_index,
                            shape_address,
                        )
                        geometry_cells = tuple(
                            ssa_value_identity_cell(
                                function, int(value.id), book=book,
                            ) for value in geometry_values
                        )
                        if any(cell is None for cell in geometry_cells):
                            raise ValueError(
                                "runtime flatten metadata lacks SSA identity"
                            )
                        book.post(
                            TENSOR_RUNTIME_VIEW_GEOMETRY,
                            (function_scope_of(function), function.name,
                             int(result.id)),
                            ("flatten", int(source.id),
                             int(source_metadata["shape"].id),
                             int(source_metadata["rank"].id),
                             int(source_metadata["element_count"].id),
                             int(shape_value.id), int(rank_value.id)),
                            stage=TENSOR_SSA_LOWERING,
                            provenance=Derived(tuple(dict.fromkeys((
                                source_cell, result_cell, *geometry_cells,
                            )))), mode=Mode.CONCORD,
                        )
                        runtime_flatten = True
                    if operation in {"reshape", "view"} and len(args) > 1:
                        requested = _as_sequence(constants.get(int(args[1].id)))
                        if requested is not None:
                            resolved = [int(extent) for extent in requested]
                            inferred = [
                                index for index, extent in enumerate(resolved)
                                if extent == -1
                            ]
                            if len(inferred) <= 1 and all(
                                extent > 0 or extent == -1
                                for extent in resolved
                            ):
                                source_count = booked_element_count(source)
                                known_count = prod(
                                    extent for extent in resolved
                                    if extent != -1
                                )
                                if (
                                    inferred and source_count is not None
                                    and known_count
                                    and source_count % known_count == 0
                                ):
                                    resolved[inferred[0]] = (
                                        source_count // known_count
                                    )
                                if -1 not in resolved:
                                    result.shape = tuple(resolved)
                                    explicit_reshape_shape = True
                    # compact_codewords reshapes `bits` with (-1, max_bits),
                    # where max_bits is an SSA formal.  The former rewrite
                    # erased that view while its target descriptor still had
                    # no shape, rank, or count; the later region then read the
                    # returned storage as a scalar.  A runtime view remains
                    # an operation until its own geometry is on the ledger.
                    if operation in {"reshape", "view"}:
                        result_metadata = dict(result.accounting or {})
                        complete_dynamic = (
                            result_metadata.get("tensor_metadata_state")
                            == "dynamic"
                            and all(
                                result_metadata.get(key) is not None
                                for key in (
                                    "tensor_shape_value_id",
                                    "tensor_rank_value_id",
                                    "tensor_element_count_value_id",
                                )
                            )
                        )
                        runtime_inferred_axis = (
                            len(args) == 3
                            and constants.get(int(args[1].id)) == -1
                            and str(args[2].dtype or "").lower()
                            in {"int", "int32", "i32", "int64", "i64", "long"}
                            and tuple(args[2].shape or ()) == ()
                            and str((source.accounting or {}).get(
                                "program_abi_storage") or "") == "span"
                            and int((source.accounting or {}).get(
                                "program_abi_rank", 0) or 0) > 0
                            and len(function.blocks) == 1
                            and not pending_runtime_view_guards
                        )
                        if (
                            runtime_inferred_axis
                            and not explicit_reshape_shape
                            and not tuple(result.shape or ())
                            and not complete_dynamic
                        ):
                            from .native_faults import RESHAPE_INVALID_SHAPE

                            scalar32 = {
                                "physical_dtype": "int32", "storage": "scalar",
                                "rank": 0,
                            }
                            shape_value, shape_definition = int_vector(
                                (0, 0), physical={
                                    "physical_dtype": "int32", "storage": "span",
                                    "rank": 1,
                                },
                            )
                            source_count = ensure_dynamic(
                                rewritten, source,
                            )["element_count"]
                            width32 = fresh(dtype="int32", physical=scalar32)
                            one, one_definition = constant(
                                1, "int32", physical=scalar32,
                            )
                            zero, zero_definition = constant(
                                0, "int32", physical=scalar32,
                            )
                            safe_width = fresh(dtype="int32", physical=scalar32)
                            remainder = fresh(dtype="int32", physical=scalar32)
                            quotient = fresh(dtype="int32", physical=scalar32)
                            bool_scalar = {
                                "physical_dtype": "bool", "storage": "scalar",
                                "rank": 0,
                            }
                            invalid_width = fresh(
                                dtype="bool", physical=bool_scalar,
                            )
                            invalid_remainder = fresh(
                                dtype="bool", physical=bool_scalar,
                            )
                            invalid = fresh(dtype="bool", physical=bool_scalar)
                            rewritten.extend((
                                Instr("Cast", [args[2]], width32, attributes={
                                    "source_dtype": str(args[2].dtype),
                                    "target_dtype": "int32",
                                    "conversion_kind": "integer_conversion",
                                    "overflow": "trap",
                                    "binding": "tensor-reshape-width",
                                }),
                                one_definition, zero_definition,
                                Instr("Max", [width32, one], safe_width),
                                Instr("Mod", [source_count, safe_width], remainder),
                                Instr("FloorDiv", [source_count, safe_width], quotient),
                                Instr("Le", [width32, zero], invalid_width),
                                Instr("Ne", [remainder, zero], invalid_remainder),
                                Instr("Or", [invalid_width, invalid_remainder], invalid),
                                shape_definition,
                            ))
                            for position, dimension in enumerate((quotient, width32)):
                                index, index_definition = constant(
                                    position, "int64",
                                )
                                address = fresh(dtype="ptr")
                                rewritten.extend((
                                    index_definition,
                                    Instr("GetElementPtr", [shape_value, index],
                                          address, attributes={
                                              "binding": "tensor-reshape-shape",
                                          }),
                                    Instr("Store", [dimension, address], None,
                                          attributes={
                                              "binding": "tensor-reshape-shape",
                                          }),
                                ))
                            rank, rank_definition = constant(
                                2, "int32", physical=scalar32,
                            )
                            rewritten.append(rank_definition)
                            result.accounting = {
                                **result_metadata,
                                "program_abi_storage": "span",
                                "program_abi_rank": 2,
                                "tensor_metadata_state": "dynamic",
                                "tensor_shape_value_id": int(shape_value.id),
                                "tensor_rank_value_id": int(rank.id),
                                "tensor_element_count_value_id": int(
                                    source_count.id
                                ),
                            }
                            guard = Instr("CondBr", [invalid], None,
                                          attributes={
                                              "runtime_reshape_error":
                                              RESHAPE_INVALID_SHAPE,
                                          })
                            rewritten.append(guard)
                            pending_runtime_view_guards.append((
                                block, guard, source, args[2], width32, result,
                                source_count, safe_width, remainder, quotient,
                                shape_value, rank, invalid,
                            ))
                            complete_dynamic = True
                        if (
                            not explicit_reshape_shape
                            and not tuple(result.shape or ())
                            and not complete_dynamic
                        ):
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "runtime view target has no exact shape, "
                                "rank and element-count proof",
                            ))
                            rewritten.append(instruction)
                            continue
                    source_descriptor = register_tensor(
                        source,
                        storage=(
                            "input"
                            if int(source.id) in function_argument_ids
                            else "temporary"
                        ),
                    )
                    register_tensor(
                        result,
                        storage="view",
                        metadata_state=(
                            "dynamic" if complete_dynamic or runtime_unsqueeze
                            else source_descriptor.metadata_state
                        ) if operation in {"reshape", "view"}
                        else (
                            "dynamic" if runtime_unsqueeze or runtime_flatten
                            else source_descriptor.metadata_state
                        ),
                        alias_of=source_descriptor.tensor_id,
                        data_value_id=source_descriptor.data_value_id,
                    )
                    if os.environ.get("TURING_DEBUG_VIEW_ALIAS"):
                        print(
                            f"DEBUG-VIEW-ALIAS {function_name} {operation} "
                            f"result=%{result.id} shape={tuple(result.shape or ())!r} "
                            f"source=%{source.id} shape={tuple(source.shape or ())!r} "
                            f"args={[int(a.id) for a in args]!r}",
                            file=sys.stderr, flush=True,
                        )
                    # One storage identity may carry several shaped views.
                    # Mark this occurrence as a view of that storage so
                    # whole-module ABI propagation keeps the authored view
                    # instead of restamping it with the allocation owner's
                    # shape; the element count proves they describe the same
                    # bytes.
                    view_shape = tuple(result.shape) or tuple(source.shape)
                    view_accounting = dict(
                        result.accounting if runtime_unsqueeze or runtime_flatten
                        else source.accounting or {}
                    )
                    if runtime_unsqueeze or runtime_flatten or (view_shape and (
                        (
                            tuple(source.shape or ())
                            and tuple(view_shape) != tuple(source.shape or ())
                        )
                        # The generated VJP's bw_matmul reshapes an as-yet
                        # unshaped call result (storage id 81) to the exact
                        # requested (2, 2).  Like unsqueeze, that result is
                        # a view even before the storage shape settles.
                        or (
                            not tuple(source.shape or ())
                            and (
                                str(operation) == "unsqueeze"
                                or explicit_reshape_shape
                            )
                        )
                    )):
                        view_accounting["ssa_storage_view"] = {
                            "storage_value_id": int(source.id),
                            "view_shape": tuple(int(e) for e in view_shape),
                            "operation": str(operation),
                        }
                    aliases[int(result.id)] = SSAValue(
                        source.id,
                        dtype=result.dtype or source.dtype,
                        shape=view_shape,
                        device=result.device or source.device,
                        accounting=view_accounting,
                    )
                    continue

                # Constants after the first data operand are structural call
                # operands (axis, shape, dtype token, keepdim, ...).
                # Operand zero is the tensor value even when constant folding
                # proved its payload.  Only later constant operands are call
                # structure (shape, axis, dtype, keepdim, ...).  Dropping a
                # constant first operand made calls such as
                # ``broadcast_to(1.0, (m, n))`` appear to have no source.
                _settle_operand_shapes(
                    function, args, owner_tensor_descriptor,
                )
                if instruction.res is not None:
                    _settle_operand_shapes(
                        function, (instruction.res,), owner_tensor_descriptor,
                    )
                opcode_contract = c_tensor_opcode(operation)
                data_positions = (
                    frozenset({0, 1, 2})
                    if operation == "where"
                    else frozenset({0, 1})
                    if opcode_contract is not None
                    and opcode_contract[0] == "binary"
                    else frozenset({0})
                )
                data_args = [
                    argument for position, argument in enumerate(args)
                    if position in data_positions
                    or int(argument.id) not in constants
                ]
                metadata = [
                    constants[int(argument.id)]
                    for position, argument in enumerate(args)
                    if position not in data_positions
                    and int(argument.id) in constants
                ]
                source = data_args[0] if data_args else (args[0] if args else None)
                if opcode_contract is not None and opcode_contract[0] == "binary":
                    from .ssa_record_return_state import (
                        posted_ssa_value_fact, ssa_value_identity_cell,
                    )

                    span_operands = tuple(
                        operand for operand in data_args
                        if (fact := posted_ssa_value_fact(
                            function, int(operand.id), book=book,
                        )) is not None
                        and fact.storage == "span"
                        and fact.rank is not None and fact.rank > 0
                    )
                    if len(span_operands) == 1:
                        # Operand order remains in data_args for subtraction
                        # and other noncommutative kernels. This selects only
                        # the extent-bearing operand as the shape source.
                        source = span_operands[0]
                unsettled_span_operands = tuple(
                    operand for operand in data_args
                    if not tuple(operand.shape or ())
                    and (operand.accounting or {}).get(
                        "program_abi_storage"
                    ) == "span"
                )
                if operation == "where" and unsettled_span_operands:
                    # A linked multi-result call publishes each result's
                    # extents only after whole-program call linking has paired
                    # its output_ids/output_positions with the native result
                    # contract.  Until then an empty-shaped span is not a
                    # scalar.  Retain the semantic operation so the existing
                    # late tensor slice can consume that exact call-edge fact;
                    # lowering it now would irreversibly turn the span into a
                    # by-value fill_double operand.
                    rewritten.append(instruction)
                    continue
                constructor_count: SSAValue | None = None
                constructor_operations = {
                    "fill", "full", "zeros", "empty", "ones",
                }
                if (
                    operation in constructor_operations
                    and args
                    and not tuple(result.shape or ())
                    and int((result.accounting or {}).get(
                        "program_abi_rank", 0
                    ) or 0) == 1
                ):
                    # A scalar constructor shape is runtime data, not the
                    # tensor being filled.  Publish the dependent rank-one
                    # shape as ordinary SSA before descriptor registration;
                    # storage-capacity analysis remains free to prove a
                    # static arena bound through the exact caller edge.
                    # compact_codewords' dynamic zeros result carries this
                    # constructor shape, rank and count across a planned
                    # region Call. The int32 Cast and Const definitions are
                    # their physical declarations; record those on the
                    # minted SSA cells before the aggregate return projects
                    # them into the caller.
                    scalar32 = {
                        "physical_dtype": "int32", "storage": "scalar",
                        "rank": 0,
                    }
                    constructor_count = fresh(
                        dtype="int32", physical=scalar32,
                    )
                    shape_value, shape_definition = int_vector(
                        (0,), physical={
                            "physical_dtype": "int32", "storage": "span",
                            "rank": 1,
                        },
                    )
                    rank_value, rank_definition = constant(
                        1, "int32", physical=scalar32,
                    )
                    zero, zero_definition = constant(0, "int64")
                    shape_address = fresh(dtype="ptr")
                    prefix.extend((
                        Instr(
                            Handler.Cast.value,
                            [args[0]], constructor_count,
                            attributes={
                                "source_dtype": str(args[0].dtype or "int64"),
                                "target_dtype": "int32",
                                "binding": "tensor-constructor-count",
                            },
                        ),
                        shape_definition,
                        zero_definition,
                        Instr(
                            Handler.GetElementPtr.value,
                            [shape_value, zero], shape_address,
                            attributes={
                                "binding": "tensor-constructor-shape",
                            },
                        ),
                        Instr(
                            Handler.Store.value,
                            [constructor_count, shape_address], None,
                            attributes={
                                "binding": "tensor-constructor-shape",
                            },
                        ),
                        rank_definition,
                    ))
                    result.accounting = {
                        **dict(result.accounting or {}),
                        "tensor_metadata_state": "dynamic",
                        "tensor_shape_value_id": int(shape_value.id),
                        "tensor_rank_value_id": int(rank_value.id),
                        "tensor_element_count_value_id": int(
                            constructor_count.id
                        ),
                    }
                    # Constructors have no data-source tensor.  Their first
                    # operand describes the result shape.
                    source = None
                # The broadcast instruction owns its destination extents.
                # Resolve them before registering the allocation descriptor;
                # otherwise a provisional singleton shape becomes the ABI
                # even though the kernel later writes the full broadcast.
                broadcast_shape = None
                if operation in {"broadcast_to", "expand"}:
                    requested_shape = _attribute(
                        instruction.attributes, "shape", "size", "sizes"
                    )
                    if requested_shape is None:
                        requested_shape = next((
                            item for item in metadata
                            if _as_sequence(item) is not None
                        ), None)
                    broadcast_shape = _as_sequence(requested_shape)
                    if broadcast_shape is None and result.shape:
                        broadcast_shape = tuple(result.shape)
                    if broadcast_shape is not None:
                        broadcast_shape = tuple(map(int, broadcast_shape))
                        result.shape = broadcast_shape
                # An earlier instruction in this same rewrite can establish
                # the physical descriptor for a value whose later occurrence
                # was captured before shape settlement.  Use that resident
                # descriptor only to fill a missing occurrence shape.  An
                # already shaped occurrence remains the incumbent, including
                # an equal-priority view spelling.  The exception is a helper
                # the call-edge concordance proved shape-polymorphic: there,
                # the descriptor produced earlier in THIS specialized
                # function outranks a stale occurrence stamped from another
                # specialization's authored-row proof.
                for operand in data_args:
                    incumbent = tensor_table.by_id(int(operand.id))
                    if (
                        incumbent is None
                        or incumbent.metadata_state != "static"
                        or not tuple(incumbent.shape or ())
                    ):
                        continue
                    operand_shape = tuple(operand.shape or ())
                    incumbent_shape = tuple(incumbent.shape or ())
                    if operand_shape and not (
                        shape_polymorphic_function
                        and operand_shape != incumbent_shape
                        and not (operand.accounting or {}).get(
                            "ssa_storage_view"
                        )
                    ):
                        continue
                    operand.shape = tuple(incumbent.shape)
                    if operand.dtype is None and incumbent.dtype:
                        operand.dtype = incumbent.dtype
                    operand.accounting = {
                        **dict(operand.accounting or {}),
                        "shape_settlement_provenance": (
                            "resident_descriptor_incumbent",
                            function_name,
                            int(operand.id),
                        ),
                        "shape_settlement_tie_policy": "incumbent",
                    }
                tensor_opcode = c_tensor_opcode(operation)
                if (
                    tensor_opcode is not None
                    and data_args
                    and all(
                        not tuple(argument.shape or ())
                        and int((argument.accounting or {}).get(
                            "program_abi_rank", 0
                        ) or 0) == 0
                        for argument in data_args
                    )
                ):
                    # Exact scalar operands override provisional count-one
                    # shapes propagated before dropped-axis indexing was
                    # legalized.  Otherwise ordinary scalar arithmetic is
                    # sent through dynamic broadcast kernels solely because
                    # stale planner metadata still says ``(1,)``.
                    result.shape = ()
                if (
                    source is not None
                    and operation in {"sum", "mean"}
                    and str(result.dtype or "unknown") == "unknown"
                ):
                    # Tensor semantics: summing or averaging a floating
                    # tensor keeps its floating dtype.  (Integer/bool sums
                    # promote and are left to their own evidence.)
                    if str(source.dtype or "").casefold() in {
                        "float64", "float32", "float16", "f64", "f32", "f16",
                    }:
                        result.dtype = source.dtype
                if (
                    source is not None
                    and tuple(source.shape or ())
                    and (operation in _REDUCTION_CODES or operation == "mean")
                ):
                    reduction_axis = _attribute(
                        instruction.attributes, "axis", "dim"
                    )
                    if reduction_axis is None:
                        reduction_axis = next((
                            item for item in metadata
                            if isinstance(item, (int, float))
                            or _as_sequence(item) is not None
                        ), None)
                    reduction_keepdim = bool(
                        _attribute(instruction.attributes, "keepdim")
                        or False
                    )
                    if reduction_axis is None and operation in {"sum", "mean"}:
                        result.shape = ()
                    elif reduction_axis is not None:
                        raw_axes = _as_sequence(reduction_axis)
                        axes = (
                            tuple(raw_axes)
                            if raw_axes is not None
                            else (reduction_axis,)
                        )
                        rank = len(source.shape)
                        normalized = {
                            int(axis) % rank for axis in axes
                        }
                        result.shape = tuple(
                            1 if reduction_keepdim and index in normalized else extent
                            for index, extent in enumerate(source.shape)
                            if reduction_keepdim or index not in normalized
                        )
                sequence_metadata = tuple(
                    tuple(map(int, sequence))
                    for item in metadata
                    if (sequence := _as_sequence(item)) is not None
                    and all(isinstance(member, (int, float)) for member in sequence)
                )
                if (
                    operation == "unfold2d"
                    and source is not None
                    and len(tuple(source.shape or ())) == 4
                ):
                    n, channels, height, width = map(int, source.shape)
                    kernel = _as_sequence(_attribute(
                        instruction.attributes, "kernel_size"
                    ))
                    stride = _as_sequence(_attribute(
                        instruction.attributes, "stride"
                    )) or (1, 1)
                    padding = _as_sequence(_attribute(
                        instruction.attributes, "padding"
                    )) or (0, 0)
                    dilation = _as_sequence(_attribute(
                        instruction.attributes, "dilation"
                    )) or (1, 1)
                    if kernel is not None and all(
                        len(pair) == 2
                        for pair in (kernel, stride, padding, dilation)
                    ):
                        kh, kw = map(int, kernel)
                        sh, sw = map(int, stride)
                        ph, pw = map(int, padding)
                        dh, dw = map(int, dilation)
                        output_h = (
                            height + 2 * ph - dh * (kh - 1) - 1
                        ) // sh + 1
                        output_w = (
                            width + 2 * pw - dw * (kw - 1) - 1
                        ) // sw + 1
                        result.shape = (
                            n, channels * kh * kw, output_h * output_w,
                        )
                elif operation == "fold2d":
                    output_size = _as_sequence(_attribute(
                        instruction.attributes, "output_size"
                    ))
                    if output_size is None and sequence_metadata:
                        output_size = sequence_metadata[0]
                    if output_size is not None and len(output_size) == 4:
                        result.shape = tuple(map(int, output_size))
                elif operation == "arange":
                    count_value = instruction.attributes.get("arange_count")
                    start_value = instruction.attributes.get(
                        "arange_start", 0.0,
                    )
                    step_value = instruction.attributes.get(
                        "arange_step", 1.0,
                    )
                    if count_value is not None and int(count_value) > 0:
                        arange_dtype = str(
                            _attribute(instruction.attributes, "dtype")
                            or recorded_dtype(result, "arange result")
                        ).casefold()
                        integer_arange = arange_dtype in {
                            "int64", "i64", "long",
                        }
                        start, start_def = constant(
                            int(start_value) if integer_arange else float(start_value),
                            "int64" if integer_arange else "float64",
                        )
                        step, step_def = constant(
                            int(step_value) if integer_arange else float(step_value),
                            "int64" if integer_arange else "float64",
                        )
                        count, count_def = constant(
                            int(count_value), "int32",
                        )
                        result.shape = (int(count_value),)
                        if integer_arange:
                            result.dtype = "int64"
                        prefix.extend((start_def, step_def, count_def))
                        emitted.append(call(
                            (
                                "create_arange_int64"
                                if integer_arange else "create_arange"
                            ),
                            [start, step, count, result],
                            result,
                            instruction,
                            output_argument=3,
                        ))
                    elif (
                        count_value is None
                        and len(args) == 1
                        and str(args[0].dtype) == "int64"
                        and str(result.dtype) == "int64"
                    ):
                        # Program: canonical_codes_from_lengths constructs
                        # arange(bit_count) with a scalar int64 bound.  The
                        # source's omitted end=None and step=1 are exact
                        # SOURCE_CALL_PARAMETER_DEFAULT facts.  The result's
                        # runtime count is max(0, bound), narrowed under the
                        # existing numeric overflow contract before either
                        # native allocation or create_arange_int64 sees it.
                        from .concordance_declarations import (
                            SOURCE_CALL_PARAMETER_DEFAULT,
                            SOURCE_PARAMETER_DEFAULT,
                            SpecializationFact,
                            SpecializationSource,
                        )
                        defaults = instruction.attributes.get(
                            "source_parameter_default_cells"
                        ) or {}
                        end_ref = defaults.get("end")

                        def _arange_default(parameter):
                            ref = defaults.get(parameter)
                            if (
                                ref is None
                                or ref.page != SOURCE_CALL_PARAMETER_DEFAULT
                                or ref.row[1] != parameter
                                or end_ref is None
                                or ref.row[0] != end_ref.row[0]
                            ):
                                return False, None
                            fact = book.page(SOURCE_CALL_PARAMETER_DEFAULT).cell(
                                ref.row, ref.column,
                            )
                            if not (
                                isinstance(fact, tuple)
                                and len(fact) == 2
                                and fact[0].page == SOURCE_PARAMETER_DEFAULT
                                and fact[1] == instruction.attributes.get(
                                    "source_signature_cell"
                                )
                            ):
                                return False, None
                            value = book.page(SOURCE_PARAMETER_DEFAULT).cell(
                                fact[0].row, fact[0].column,
                            )
                            if (
                                not isinstance(value, SpecializationFact)
                                or value.source is not SpecializationSource.DEFAULT
                            ):
                                return False, None
                            return True, value.value

                        has_end, default_end = _arange_default("end")
                        has_step, default_step = _arange_default("step")
                        has_dtype, default_dtype = _arange_default("dtype")
                        if (
                            has_end and default_end is None
                            and has_step and default_step == 1
                            and has_dtype and default_dtype is None
                        ):
                            from .ssa_call_input_adapters import (
                                _numeric_conversion_attributes,
                            )
                            scalar64 = {"physical_dtype": "int64",
                                        "storage": "scalar", "rank": 0}
                            scalar32 = {"physical_dtype": "int32",
                                        "storage": "scalar", "rank": 0}
                            zero64, zero64_def = constant(0, "int64", physical=scalar64)
                            one64, one64_def = constant(1, "int64", physical=scalar64)
                            max32, max32_def = constant(
                                (1 << 31) - 1, "int64", physical=scalar64,
                            )
                            zero32, zero32_def = constant(0, "int32", physical=scalar32)
                            nonnegative = fresh(dtype="int64", physical=scalar64)
                            narrowed = fresh(dtype="int32", physical=scalar32)
                            exceeds32 = fresh(dtype="bool", physical={
                                "physical_dtype": "bool", "storage": "scalar",
                                "rank": 0,
                            })
                            count32 = fresh(dtype="int32", physical=scalar32)
                            shape_value, shape_def = int_vector((0,), physical={
                                "physical_dtype": "int32", "storage": "span",
                                "rank": 1,
                            })
                            shape_address = fresh(dtype="ptr")
                            rank, rank_def = constant(1, "int32", physical=scalar32)
                            prefix.extend((
                                zero64_def, one64_def, max32_def, zero32_def,
                                Instr("Max",
                                      [args[0], zero64], nonnegative,
                                      attributes={"binding": "arange-count-clamp"}),
                                Instr(Handler.Cast.value,
                                      [nonnegative], narrowed,
                                      attributes={
                                          **_numeric_conversion_attributes(
                                              "int64", "int32"
                                          ),
                                          "source_dtype": "int64",
                                          "target_dtype": "int32",
                                          "binding": "arange-count-narrowing",
                                      }),
                                Instr(Handler.Gt.value,
                                      [nonnegative, max32], exceeds32,
                                      attributes={"binding": "arange-count-bound"}),
                                Instr(Handler.Select.value,
                                      [exceeds32, zero32, narrowed], count32,
                                      attributes={"binding": "arange-safe-count"}),
                                shape_def,
                                Instr(Handler.GetElementPtr.value,
                                      [shape_value, zero64], shape_address,
                                      attributes={"binding": "arange-shape"}),
                                Instr(Handler.Store.value,
                                      [count32, shape_address], None,
                                      attributes={"binding": "arange-shape"}),
                                rank_def,
                            ))
                            result.accounting = {
                                **dict(result.accounting or {}),
                                "tensor_metadata_state": "dynamic",
                                "tensor_shape_value_id": int(shape_value.id),
                                "tensor_rank_value_id": int(rank.id),
                                "tensor_element_count_value_id": int(count32.id),
                            }
                            # Exact safe count is produced in this region;
                            # its native arena belongs to the caller's
                            # explicit heap owner cell.
                            if has_private_region_result(result):
                                carrier = own_dynamic_tensor(
                                    result, count32, "int64", prefix,
                                )
                            else:
                                # Public tensor return keeps the incumbent
                                # logical descriptor and its metadata output
                                # projection. Its native dynamic storage
                                # still requires a separate public capacity
                                # contract and remains guarded downstream.
                                register_tensor(
                                    result, storage="temporary",
                                )
                                carrier = result
                            emitted.append(call(
                                "create_arange_int64",
                                [zero64, one64, count32, carrier],
                                result, instruction, output_argument=3,
                            ))
                elif operation == "gather" and len(data_args) >= 2:
                    indices = data_args[1]
                    if not tuple(source.shape or ()):
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name,
                            block_name,
                            operation,
                            (
                                "gather source has no declared rank; "
                                f"source=%{source.id}, indices=%{indices.id}, "
                                "argument_shapes="
                                f"{tuple(tuple(arg.shape or ()) for arg in data_args)!r}"
                            ),
                        ))
                        rewritten.append(instruction)
                        continue
                    raw_dim = _attribute(
                        instruction.attributes, "axis", "dim"
                    )
                    if raw_dim is None:
                        raw_dim = next((
                            item for item in metadata
                            if isinstance(item, (int, float))
                        ), 0)
                    axis = int(raw_dim) % len(source.shape)
                    known_index_count = booked_element_count(indices)
                    if known_index_count is None:
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name, block_name, operation,
                            "gather index count has no exact static geometry",
                        ))
                        rewritten.append(instruction)
                        continue
                    source_shape, source_shape_def = int_vector(source.shape)
                    rank, rank_def = constant(len(source.shape), "int32")
                    axis_value, axis_def = constant(axis, "int32")
                    index_count, index_count_def = constant(
                        known_index_count, "int32"
                    )
                    prefix.extend((
                        source_shape_def, rank_def, axis_def,
                        index_count_def,
                    ))
                    emitted.append(call(
                        "gather_values_double",
                        [
                            source, result, source_shape, rank, axis_value,
                            indices, index_count,
                        ],
                        result, instruction, output_argument=1,
                    ))
                elif operation == "scatter" and len(data_args) >= 3:
                    indices = data_args[1]
                    values = data_args[2]
                    reduction = _attribute(instruction.attributes, "reduce")
                    # Graphs captured before scatter recorded its reduction
                    # policy carry no attribute.  The AbstractTensor API's
                    # declared default is sum, so absence has that one exact
                    # meaning.  Every declared policy selects its own wrapper
                    # around the shared repository reduction kernel.
                    reduction = "sum" if reduction is None else str(reduction)
                    if reduction not in {"sum", "mean", "max", "min"}:
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name,
                            block_name,
                            operation,
                            (
                                f"scatter reduce={reduction!r} is not one of "
                                "('sum', 'mean', 'max', 'min')"
                            ),
                        ))
                        rewritten.append(instruction)
                        continue
                    source_shape = tuple(source.shape or ())
                    index_count = booked_element_count(indices)
                    runtime_index_count = (
                        scatter_derived_index_count(indices, values, prefix)
                        if index_count is None and len(source_shape) == 1
                        else None
                    )
                    if not source_shape or (
                        index_count is None and runtime_index_count is None
                    ):
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name,
                            block_name,
                            operation,
                            (
                                "scatter-add needs a declared destination "
                                "shape and index extent; "
                                f"source=%{source.id}, indices=%{indices.id}, "
                                "argument_shapes="
                                f"{tuple(tuple(arg.shape or ()) for arg in data_args)!r}"
                            ),
                        ))
                        rewritten.append(instruction)
                        continue
                    raw_dim = _attribute(
                        instruction.attributes, "axis", "dim"
                    )
                    if raw_dim is None:
                        raw_dim = next((
                            item for item in metadata
                            if isinstance(item, (int, float))
                        ), 0)
                    axis = int(raw_dim) % len(source_shape)
                    expected_values_shape = list(source_shape)
                    if index_count is not None:
                        expected_values_shape[axis] = int(index_count)
                    if (
                        runtime_index_count is not None
                        and (axis != 0 or tuple(values.shape or ()))
                    ) or (
                        index_count is not None
                        and tuple(values.shape or ())
                            != tuple(expected_values_shape)
                    ):
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name,
                            block_name,
                            operation,
                            (
                                "scatter-add values must replace the indexed "
                                f"axis with the index extent; values=%{values.id} "
                                f"has {tuple(values.shape or ())!r}, expected "
                                f"{tuple(expected_values_shape)!r}"
                            ),
                        ))
                        rewritten.append(instruction)
                        continue
                    result.shape = source_shape
                    shape_value, shape_def = int_vector(source_shape)
                    rank, rank_def = constant(len(source_shape), "int32")
                    axis_value, axis_def = constant(axis, "int32")
                    if runtime_index_count is None:
                        index_count_value, index_count_def = constant(
                            int(index_count), "int32"
                        )
                        prefix.extend((
                            shape_def, rank_def, axis_def, index_count_def,
                        ))
                    else:
                        index_count_value = runtime_index_count
                        prefix.extend((shape_def, rank_def, axis_def))
                    from .ssa_record_return_state import (
                        posted_ssa_value_fact, revise_ssa_value,
                        ssa_value_identity_cell,
                    )
                    values_fact = posted_ssa_value_fact(
                        function, int(values.id), book=book,
                    )
                    if (values_fact is not None
                            and values_fact.physical_dtype == "int64"
                            and values_fact.storage == "span"
                            and values_fact.rank == 1):
                        # _code_lengths scatters octets + 1 with a runtime
                        # selector count. A scalar call-input Cast converted
                        # only the first limb and let scatter_reduce_double
                        # read past it. Convert the exact counted values span
                        # before the selected scatter Call instead.
                        from .ssa_call_input_adapters import _infer_pointer_storage
                        from .concordance_declarations import TENSOR_SSA_LOWERING
                        selected_cast = reference.module.functions.get(
                            "cast_int64_to_double_values"
                        )
                        if (selected_cast is None
                                or selected_cast.metadata.get(
                                    "ssa_output_argument") != 1
                                or len(selected_cast.args) != 3):
                            raise ValueError(
                                "scatter values promotion lacks selected counted cast"
                            )
                        link_repository_function(
                            selected_cast.name, selected_cast,
                        )
                        _infer_pointer_storage({
                            selected_cast.name: selected_cast,
                        })
                        formal_cells = tuple(ssa_value_identity_cell(
                            selected_cast, int(formal.id), book=book,
                        ) for formal in selected_cast.args[:2])
                        formal_facts = tuple(posted_ssa_value_fact(
                            selected_cast, int(formal.id), book=book,
                        ) for formal in selected_cast.args[:2])
                        origins = (selected_cast.metadata.get(
                            "llvm_repository_kernel"
                        ) or (None, {}))[1]
                        if (any(cell is None for cell in formal_cells)
                                or any(fact is None for fact in formal_facts)
                                or tuple(fact.physical_dtype
                                         for fact in formal_facts)
                                != ("int64", "float64")
                                or any(
                                    (origin := origins.get(int(formal.id))) is None
                                    or origin[1] != "argument"
                                    or not str(origin[2]).startswith("ptr ")
                                    for formal in selected_cast.args[:2]
                                )):
                            raise ValueError(
                                "scatter values promotion lacks typed span formals"
                            )
                        values_descriptor = tensor_table.by_id(int(values.id))
                        dynamic_values = (
                            values_descriptor is not None
                            and values_descriptor.metadata_state == "dynamic"
                        )
                        if dynamic_values:
                            values_geometry = _dynamic_tensor_geometry_evidence(
                                module, function, int(values.id), pending=prefix,
                            )
                            valid_count = (
                                values_geometry is not None
                                and int(values_geometry[0][2].id)
                                == int(index_count_value.id)
                            )
                        else:
                            valid_count = (
                                values_descriptor is not None
                                and values_descriptor.metadata_state == "static"
                                and tuple(values_fact.shape)
                                == tuple(values_descriptor.shape)
                                and values_descriptor.static_element_count
                                == int(index_count)
                            )
                        if not valid_count:
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "scatter values promotion lacks exact span count",
                            ))
                            rewritten.extend(prefix)
                            rewritten.append(instruction)
                            continue
                        converted_values = fresh(
                            shape=tuple(values.shape), dtype="float64",
                        )
                        source_cell = ssa_value_identity_cell(
                            function, int(values.id), book=book,
                        )
                        if source_cell is None:
                            raise ValueError(
                                "scatter values promotion lost source SSA cell"
                            )
                        revise_ssa_value(
                            function, int(converted_values.id),
                            dtype="float64", shape=tuple(values.shape),
                            stage=TENSOR_SSA_LOWERING,
                            sources=(source_cell, *formal_cells), book=book,
                            physical={
                                "physical_dtype": "float64",
                                "storage": "span", "rank": 1,
                            },
                        )
                        converted_values.accounting = {
                            **dict(converted_values.accounting or {}),
                            "physical_dtype": "float64",
                            "program_abi_storage": "span",
                            "program_abi_rank": 1,
                        }
                        if dynamic_values:
                            converted_values.accounting.update({
                                "tensor_metadata_state": "dynamic",
                                "tensor_shape_value_id": int(values_geometry[0][0].id),
                                "tensor_rank_value_id": int(values_geometry[0][1].id),
                                "tensor_element_count_value_id": int(index_count_value.id),
                            })
                        register_tensor(
                            converted_values, storage="temporary",
                            metadata_state=("dynamic" if dynamic_values else "static"),
                            extent_ids_from=(values_descriptor
                                             if dynamic_values else None),
                        )
                        carrier = own_dynamic_tensor(
                            converted_values, index_count_value,
                            "float64", prefix,
                        )
                        prefix.append(call(
                            selected_cast.name,
                            [values, carrier, index_count_value],
                            converted_values, instruction,
                            output_argument=1,
                        ))
                        _post_scalar_kernel_input_conversion(
                            book, function, str(block_name), int(result.id),
                            2, values, converted_values,
                            callee=f"scatter_{reduction}_double",
                        )
                        values = converted_values
                    elif (values_fact is None
                          or values_fact.physical_dtype != "float64"
                          or values_fact.storage != "span"
                          or values_fact.rank != 1):
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name, block_name, operation,
                            "scatter values lack selected physical span",
                        ))
                        rewritten.extend(prefix)
                        rewritten.append(instruction)
                        continue
                    emitted.append(call(
                        f"scatter_{reduction}_double",
                        [
                            source, result, shape_value, rank, axis_value,
                            indices, index_count_value, values,
                        ],
                        result, instruction, output_argument=1,
                    ))
                opcode_contract = (
                    None if emitted else c_tensor_opcode(operation)
                )
                if (
                    source is not None
                    and opcode_contract is not None
                ):
                    # A checked int64 binary stays a raw SSA Add/Sub/Mul at
                    # dispatch, but its output still needs the operand-proved
                    # int64 span/rank fact before the extent binding cites
                    # this result. Excluding checked lanes here left
                    # root(bits).unsqueeze(1) + 1 result5 without physical
                    # dtype even though the source span and scalar were
                    # already posted on this book.
                    if str(result.dtype or "").casefold() not in {
                        "float32", "float16", "f32", "f16",
                    }:
                        # A narrower float result is NOT stamped double: its
                        # declared dtype is the value's storage, and the call
                        # adapters convert at the kernel boundary (a double
                        # temporary, then ``physical_call_output_conversion``
                        # back to the declared width).  Stamping it double
                        # made a float32 value live in, and be returned as,
                        # float64.
                        # Program: history_identity's planned_region_6 emits
                        # a ``binary_scalar_double`` result (value 62).  The
                        # ABI stamp below lived on accounting alone, so the
                        # value's ``ssa_value`` row stayed unposted and the
                        # caller's Load of it proposed physical None against
                        # an initial float64 (TransformationConflict).
                        # Wrong record: an ABI declared on accounting, never
                        # posted.  Rule: post first, accounting is a view;
                        # the result's row is revised with the dispatch
                        # kernel's double output ABI, DERIVED from its
                        # previous cell and each operand's row.
                        from .concordance_declarations import (
                            TENSOR_SSA_LOWERING as _OPCODE_STAGE,
                        )
                        from .ssa_record_return_state import (
                            posted_ssa_value_fact as _posted_opcode_fact,
                            revise_ssa_value as _revise_opcode_value,
                            ssa_value_identity_cell as _opcode_identity_cell,
                        )
                        _opcode_owner = lowering_at[0]
                        _opcode_book = identity_book(module)
                        if _opcode_owner is not None:
                            _revise_opcode_value(
                                _opcode_owner, int(result.id),
                                dtype=result.dtype,
                                shape=tuple(result.shape or ()),
                                stage=_OPCODE_STAGE,
                                sources=tuple(
                                    _opcode_identity_cell(
                                        _opcode_owner, int(operand.id),
                                        book=_opcode_book,
                                    )
                                    for operand in data_args
                                ),
                                book=_opcode_book,
                                physical={
                                    "physical_dtype": (
                                        "int64" if _int64_lane_stays_elementwise(
                                            instruction, operation, result, data_args,
                                            owner=_opcode_owner, book=_opcode_book,
                                        ) or opcode_contract[0] == "binary" and _int64_binary_kernel(
                                            operation, result, data_args,
                                            owner=_opcode_owner, book=_opcode_book,
                                        ) else "float64"
                                    ),
                                    **_elementwise_result_physical(
                                        _opcode_owner, data_args,
                                        book=_opcode_book,
                                    ),
                                },
                            )
                        # Both authored C dispatch kernels take ``double *``
                        # outputs; a Boolean mask keeps that physical ABI.
                        # Accounting is the view of the posted row.
                        result.accounting = _accounting_view(
                            _opcode_owner, result, module,
                        )
                    if opcode_contract[0] == "binary":
                        shaped_operands = [
                            tuple(operand.shape or ())
                            for operand in data_args
                            if tuple(operand.shape or ())
                        ]
                        if shaped_operands:
                            import numpy as _np

                            try:
                                result.shape = tuple(_np.broadcast_shapes(
                                    *shaped_operands
                                ))
                            except ValueError:
                                shortfalls.append(TensorSSALoweringShortfall(
                                    function_name,
                                    block_name,
                                    operation,
                                    (
                                        "tensor operands have incompatible "
                                        f"shapes {tuple(shaped_operands)!r}; "
                                        f"result=%{result.id}"
                                    ),
                                ))
                                rewritten.append(instruction)
                                continue
                        elif len(data_args) == 1:
                            result.shape = tuple(source.shape or ())
                if (
                    source is not None
                    and opcode_contract is not None
                    and opcode_contract[0] == "unary"
                ):
                    # ``unary_double`` is elementwise, so its output shape is
                    # identical to its data source.  Its semantic dtype is
                    # not necessarily identical: ``isfinite``/``isnan``/
                    # ``isinf`` and ``logical_not`` produce Boolean values
                    # in the repository's double-backed Boolean ABI.  Region
                    # 19 of ``balloon_tire_managed_advance`` exposed the old
                    # unconditional copy by changing its exact ``isfinite``
                    # result from bool to float64 before call-edge settlement.
                    _settle_unary_result_metadata(source, result)
                if operation == "where" and len(data_args) == 3:
                    # Settle the broadcast contract before registering the
                    # result descriptor.  Registration owns allocation size;
                    # changing only SSAValue.shape later would leave a
                    # one-element arena behind even if the kernel count were
                    # subsequently corrected.
                    shaped_operands = tuple(
                        tuple(operand.shape or ())
                        for operand in data_args
                        if tuple(operand.shape or ())
                    )
                    if shaped_operands:
                        import numpy as _np

                        try:
                            result.shape = tuple(_np.broadcast_shapes(
                                *shaped_operands
                            ))
                        except ValueError:
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name,
                                block_name,
                                operation,
                                "where operands have incompatible shapes "
                                f"{shaped_operands!r}; result=%{result.id}",
                            ))
                            rewritten.append(instruction)
                            continue
                # Registration fixes the physical allocation width, so the
                # repository's double-backed Boolean ABI must be stamped
                # before any descriptor is created.  Doing this only while
                # constructing the eventual Call is too late: byte_size has
                # already been frozen as one byte per semantic bool.
                for value in (*data_args, result):
                    if (
                        tuple(value.shape or ())
                        and str(value.dtype or "").casefold() in {"bool", "i1"}
                    ):
                        _post_double_backed_boolean(
                            lowering_at[0], value, module,
                        )
                if operation == "real" and source is not None:
                    # ``Tensor.real`` of a non-complex tensor is the tensor
                    # itself: same dtype, same storage, an identity view.  A
                    # complex source's component dtype has no lowering in the
                    # repository reference (no ``real`` entry), so it is a
                    # named refusal, never a default.
                    source_dtype = str(source.dtype or "").casefold()
                    if not source_dtype or source_dtype in {"unknown", "none"} \
                            or source_dtype.startswith(("complex", "c64", "c128")):
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name, block_name, operation,
                            "real part of a tensor whose dtype is "
                            f"{source.dtype!r} (complex or unproven) has no "
                            f"lowering; source=%{source.id}",
                        ))
                        rewritten.append(instruction)
                        continue
                    result.dtype = source.dtype
                if operation in {"tensor", "real", "to_dtype"} and source is not None:
                    sequence = resident_sequence(source)
                    row_shape: tuple[int, ...] | None = None
                    if sequence is not None:
                        if len(sequence.column_value_ids) != 1:
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name,
                                block_name,
                                operation,
                                "tensor materialization requires one resident "
                                "sequence value column",
                            ))
                            rewritten.append(instruction)
                            continue
                        row_shape = tuple(
                            (sequence.column_shapes or ((),))[0]
                        )
                        column_dtype = str(
                            (sequence.column_dtypes or (source.dtype,))[0]
                            or source.dtype or "unknown"
                        )
                        if column_dtype not in {"", "None", "unknown"}:
                            source.dtype = column_dtype
                        declared_rank = int(
                            (result.accounting or {}).get(
                                "program_abi_rank", 0
                            ) or 0
                        )
                        if (
                            not row_shape
                            and declared_rank
                            == 1 + len(tuple(result.shape or ()))
                        ):
                            row_shape = tuple(result.shape)
                        if (
                            declared_rank == 1 + len(row_shape)
                            and tuple(result.shape or ()) == row_shape
                        ):
                            outer_extent = sequence_outer_extent(sequence)
                            if outer_extent is not None:
                                result.shape = (outer_extent, *row_shape)
                    requested_dtype = str(
                        _attribute(instruction.attributes, "dtype", "target_dtype")
                        if operation == "to_dtype"
                        else result.dtype or source.dtype or ""
                    ).lower()
                    # _octets' list(data) has one concorded int64 column and
                    # a runtime length but no static outer extent. Its
                    # tensor(...).to_dtype('int64') keeps that exact storage;
                    # selecting cast_double_to_int_values by target alone
                    # binds int64 buffers to float64 kernel formals.
                    dynamic_sequence_view = (
                        sequence is not None
                        and not tuple(result.shape or ())
                        and declared_rank == 1 + len(row_shape)
                        and sequence.length_address_id is not None
                        and str((sequence.column_dtypes or (None,))[0])
                        == requested_dtype
                    )
                    shape_preserving = (
                        tuple(result.shape or ()) == tuple(source.shape or ())
                        if sequence is None
                        else (
                            bool(tuple(result.shape or ()))
                            and tuple(result.shape or ())[1:] == row_shape
                        ) or dynamic_sequence_view
                    )
                    if (
                        shape_preserving
                        and (operation != "to_dtype" or sequence is not None)
                        and requested_dtype == str(source.dtype or "")
                    ):
                        if sequence is not None:
                            # The sequence column and tensor data pointer are
                            # the same SSA identity.  Once the graph proves
                            # the outer extent, publish the complete logical
                            # view on that identity before its tensor
                            # descriptor freezes strides and byte width.
                            source.shape = tuple(result.shape)
                            source.accounting = {
                                **dict(source.accounting or {}),
                                "sequence_id": int(sequence.sequence_id),
                                "sequence_row_shape": row_shape,
                            }
                        source_descriptor = register_tensor(
                            source,
                            storage=(
                                "input"
                                if int(source.id) in function_argument_ids
                                else "temporary"
                            ),
                        )
                        result.accounting = {
                            **dict(result.accounting or {}),
                            "ssa_storage_view": {
                                "storage_value_id": int(source.id),
                                "view_shape": tuple(result.shape),
                                "operation": (
                                    "tensor_sequence_view"
                                    if sequence is not None
                                    else "tensor_identity_view"
                                ),
                            },
                            **(
                                {
                                    "sequence_id": int(sequence.sequence_id),
                                    "sequence_row_shape": row_shape,
                                }
                                if sequence is not None else {}
                            ),
                        }
                        register_tensor(
                            result,
                            storage="view",
                            alias_of=int(source_descriptor.tensor_id),
                            data_value_id=int(source_descriptor.data_value_id),
                        )
                        aliases[int(result.id)] = SSAValue(
                            int(source.id),
                            dtype=result.dtype or source.dtype,
                            shape=tuple(result.shape),
                            device=result.device or source.device,
                            accounting={
                                **dict(source.accounting or {}),
                                **dict(result.accounting or {}),
                            },
                        )
                        if operation == "real":
                            # The authored property stays an instruction
                            # (consumers read its source span); its result
                            # is the typed identity view registered above.
                            rewritten.append(instruction)
                        continue
                source_descriptor = None
                from .concordance_declarations import TENSOR_RUNTIME_BROADCAST_VALIDITY
                from .ssa_record_return_state import function_scope_of
                completed_broadcast = book.latest_ref(TENSOR_RUNTIME_BROADCAST_VALIDITY, (
                    function_scope_of(function), function.name, int(result.id),
                ))
                if completed_broadcast is not None:
                    # A raw checked Add remains executable after its broadcast
                    # program is complete. Later geometry arrival for another
                    # local value must not emit that program a second time.
                    rewritten.append(instruction)
                    continue
                if source is not None:
                    source_descriptor = register_tensor(
                        source,
                        storage=(
                            "input"
                            if int(source.id) in function_argument_ids
                            else "temporary"
                        ),
                    )
                for operand in data_args[1:]:
                    register_tensor(
                        operand,
                        storage=(
                            "input"
                            if int(operand.id) in function_argument_ids
                            else "temporary"
                        ),
                    )
                from .ssa_record_return_state import function_scope_of, posted_ssa_value_fact
                scalar_geometry = None
                if (opcode_contract is not None
                        and (opcode_contract[0] == "binary" and len(data_args) == 2
                             or opcode_contract[0] == "unary" and len(data_args) == 1)):
                    facts = tuple(posted_ssa_value_fact(function, int(value.id), book=book)
                                  for value in data_args)
                    from .concordance_declarations import BindingKind, CONTROL_VALUE_BINDING, ControlBinding
                    for operand, fact in zip(data_args, facts, strict=True):
                        descriptor = tensor_table.by_id(int(operand.id))
                        binding_ref = book.latest_ref(CONTROL_VALUE_BINDING, (
                            function_scope_of(function), int(operand.id),
                        ))
                        binding = (None if binding_ref is None else book.page(
                            binding_ref.page).cell(binding_ref.row, binding_ref.column))
                        # Only the actual declared input/field owns local
                        # extent queries. REGION_RESULT and CALL_RESULT formals
                        # await their producer's transported metadata instead.
                        if (descriptor is not None
                                and descriptor.metadata_state == "unresolved"
                                and fact is not None and fact.storage == "span"
                                and type(fact.rank) is int and fact.rank > 0
                                and int(operand.id) in function_argument_ids
                                and isinstance(binding, ControlBinding)
                                and binding.kind in {BindingKind.PARAMETER_SEED, BindingKind.FIELD_INCUMBENT}
                                and binding.ssa_value.row == (function_scope_of(function), int(operand.id))):
                            ensure_dynamic(prefix, operand)
                    span_positions = [index for index, fact in enumerate(facts)
                                      if fact is not None and fact.storage == "span"
                                      and type(fact.rank) is int and fact.rank > 0]
                    if len(data_args) == 2 and len(span_positions) == 1:
                        span_position = span_positions[0]
                        span = data_args[span_position]
                        scalar = data_args[1 - span_position]
                        scalar_fact = facts[1 - span_position]
                        descriptor = tensor_table.by_id(int(span.id))
                        if scalar_fact is not None and scalar_fact.storage == "scalar" and scalar_fact.rank == 0:
                            # data * multiplier keeps data's geometry. Read a
                            # declared input before computing its result, never
                            # extent(result) after leaving storage unresolved.
                            evidence = _dynamic_tensor_geometry_evidence(
                                module, function, int(span.id), pending=prefix,
                            )
                            if evidence is not None:
                                scalar_geometry = (span, scalar, evidence)
                                _post_scalar_kernel_operand(
                                    module, function,
                                    (str(function.name), int(result.id)),
                                    ("scalar_operand", int(1 - span_position)),
                                    (result, scalar),
                                )
                                from .concordance_declarations import (
                                    SCALAR_KERNEL_OPERAND,
                                    TENSOR_ELEMENTWISE_EXTENT_BINDING,
                                    TENSOR_SSA_LOWERING,
                                )
                                from .identity_concordance import Derived, Mode
                                from .ssa_record_return_state import ssa_value_identity_cell
                                metadata, geometry_causes = evidence
                                causes = tuple(dict.fromkeys((
                                    *geometry_causes,
                                    book.latest_ref(SCALAR_KERNEL_OPERAND, (
                                        str(function.name), int(result.id),
                                    )),
                                    *(ssa_value_identity_cell(
                                        function, int(value.id), book=book,
                                    ) for value in (span, scalar, result, *metadata)),
                                )))
                                if all(cell is not None for cell in causes):
                                    # Program: derive(x) returns x+1 beside
                                    # x.to_dtype(int64). The Add5 result gets
                                    # runtime metadata after its graph shape
                                    # edge was posted static(4). Publish the
                                    # existing elementwise extent receipt at
                                    # this first accounting write; it is the
                                    # changed source of the later dynamic
                                    # metadata transport edge.
                                    book.post(
                                        TENSOR_ELEMENTWISE_EXTENT_BINDING,
                                        (function_scope_of(function),
                                         function.name, int(result.id)),
                                        (str(operation), int(span.id),
                                         int(scalar.id),
                                         *(int(value.id) for value in metadata)),
                                        stage=TENSOR_SSA_LOWERING,
                                        provenance=Derived(causes),
                                        mode=Mode.CONCORD,
                                    )
                                    result.accounting = {
                                        **dict(result.accounting or {}),
                                        "tensor_metadata_state": "dynamic",
                                        **dict(zip((
                                            "tensor_shape_value_id",
                                            "tensor_rank_value_id",
                                            "tensor_element_count_value_id",
                                        ), (int(value.id) for value in metadata),
                                            strict=True)),
                                    }
                                    dynamic_extents[int(result.id)] = dict(zip(
                                        ("shape", "rank", "element_count"),
                                        metadata, strict=True,
                                    ))
                    source_descriptor = tensor_table.by_id(int(source.id)) if source is not None else None
                if (
                    ((opcode_contract is not None
                      and opcode_contract[0] == "binary"
                      and len(data_args) == 2)
                     or (operation == "sum" and len(data_args) == 1))
                    and any(
                        (descriptor := tensor_table.by_id(int(operand.id))) is not None
                        and descriptor.metadata_state == "unresolved"
                        and (fact := posted_ssa_value_fact(
                            function, int(operand.id), book=book,
                        )) is not None and fact.storage == "span"
                        and type(fact.rank) is int and fact.rank > 0
                        for operand in data_args
                    )
                ):
                    # In root(bits).unsqueeze(1) + 1 and the weighted
                    # compact_codewords Mul, a positive-rank source first
                    # arrives before computed metadata crosses the Call.
                    # The static other operand may give result a display
                    # shape such as (1, 8), but it does not prove the dynamic
                    # axis. Minting extent(result) here reads before its
                    # definition. Leave the authored operation for the
                    # existing whole-program retry and exact Call binding.
                    rewritten.extend(prefix)
                    rewritten.append(instruction)
                    continue
                if (
                    opcode_contract is not None
                    and opcode_contract[0] == "binary"
                    and len(data_args) == 2
                    and any(
                        (tensor_table.by_id(int(value.id)) is not None
                         and tensor_table.by_id(int(value.id)).metadata_state
                         == "dynamic")
                        for value in data_args
                    )
                ):
                    from .ssa_record_return_state import (
                        posted_ssa_value_fact, ssa_value_identity_cell,
                    )

                    operand_facts = tuple(
                        posted_ssa_value_fact(function, int(value.id), book=book)
                        for value in data_args
                    )
                    mixed_ranks = tuple(
                        int(fact.rank) if fact is not None
                        and fact.storage == "span" and fact.rank is not None
                        else None
                        for fact in operand_facts
                    )
                    if all(type(rank) is int and rank > 0 for rank in mixed_ranks):
                        broadcast_rank = max(mixed_ranks)
                        wide_position = mixed_ranks.index(broadcast_rank)
                        wide = data_args[wide_position]
                        narrow = data_args[1 - wide_position]
                        narrow_rank = mixed_ranks[1 - wide_position]
                        wide_descriptor = tensor_table.by_id(int(wide.id))
                        narrow_descriptor = tensor_table.by_id(int(narrow.id))
                        descriptors = (wide_descriptor, narrow_descriptor)
                        wide_evidence = (
                            _dynamic_tensor_geometry_evidence(module, function, int(wide.id), pending=prefix)
                            if wide_descriptor is not None and wide_descriptor.metadata_state == "dynamic"
                            else None
                        )
                        narrow_evidence = (
                            _dynamic_tensor_geometry_evidence(module, function, int(narrow.id), pending=prefix)
                            if narrow_descriptor is not None and narrow_descriptor.metadata_state == "dynamic"
                            else None
                        )
                        wide_metadata = () if wide_evidence is None else wide_evidence[0]
                        narrow_metadata = () if narrow_evidence is None else narrow_evidence[0]
                        geometry_cells = tuple(dict.fromkeys(
                            cell for evidence in (wide_evidence, narrow_evidence)
                            if evidence is not None for cell in evidence[1]
                        ))
                        static_operands = tuple(
                            (value, descriptor, fact)
                            for value, descriptor, fact in (
                                (wide, wide_descriptor,
                                 operand_facts[data_args.index(wide)]),
                                (narrow, narrow_descriptor,
                                 operand_facts[data_args.index(narrow)]),
                            )
                            if descriptor is not None
                            and descriptor.metadata_state == "static"
                        )
                        static_proved = all(
                            fact is not None and fact.storage == "span"
                            and fact.rank == len(descriptor.shape)
                            and tuple(fact.shape) == tuple(descriptor.shape)
                            and descriptor.static_element_count is not None
                            and all(0 <= int(axis) <= 0x7fffffff
                                    for axis in descriptor.shape)
                            and 0 <= int(descriptor.static_element_count) <= 0x7fffffff
                            for _value, descriptor, fact in static_operands
                        )
                        # Seven mask casts in one root place Add37 after
                        # Add36's booked broadcast pass arm. The second Add
                        # has exact geometry, but the old whole-function
                        # single-block test refused that terminal pass block.
                        # Its sole incoming booked validation edge and Ret
                        # make it the same safe continuation for another
                        # guard; the existing split below posts the next
                        # SSA_BLOCK and SSA_CONTROL_EDGE receipts.
                        _booked_terminal_pass = False
                        if len(function.blocks) > 1:
                            from .concordance_declarations import (
                                SSA_BLOCK, SSA_CONTROL_EDGE, SSABlockKind,
                            )
                            from .ssa_record_return_state import function_scope_of
                            _scope = function_scope_of(function)
                            _block_ref = book.latest_ref(
                                SSA_BLOCK, (_scope, function.name, block.name),
                            )
                            _incoming = tuple(
                                edge for edge in book.page(SSA_CONTROL_EDGE).rows()
                                if len(edge) == 5 and edge[0:2] == (
                                    _scope, function.name,
                                ) and edge[3] == _block_ref
                                and edge[4] == "orelse"
                                and book.page(SSA_CONTROL_EDGE).latest(edge) is True
                            )
                            _booked_terminal_pass = (
                                _block_ref is not None
                                and book.page(SSA_BLOCK).cell(
                                    _block_ref.row, _block_ref.column,
                                ) == SSABlockKind.VALIDATION_PASS
                                and len(_incoming) == 1
                                and not block.successors
                                and bool(block.instrs)
                                and block.instrs[-1].op == "Ret"
                                and (parent := function.blocks.get(
                                    _incoming[0][2].row[2],
                                )) is not None
                                and parent.instrs
                                and parent.instrs[-1].op == "CondBr"
                                and parent.instrs[-1].attributes.get(
                                    "false_target",
                                ) == block.name
                            )
                        if not (
                            all(descriptor is not None
                                    and descriptor.metadata_state
                                    in {"dynamic", "static"}
                                    for descriptor in descriptors)
                            and len(static_operands) <= 1
                            and static_proved
                            and (len(wide_metadata) == 3
                                 if wide_descriptor.metadata_state == "dynamic"
                                 else len(wide_metadata) == 0)
                            and (len(narrow_metadata) == 3
                                 if narrow_descriptor.metadata_state == "dynamic"
                                 else len(narrow_metadata) == 0)
                            and (len(function.blocks) == 1
                                 or _booked_terminal_pass)
                            and not pending_runtime_view_guards
                            and not pending_runtime_broadcast_guards
                        ):
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "mixed binary has no exact dynamic/static metadata "
                                "binding and terminal guard topology",
                            ))
                            rewritten.extend(prefix)
                            rewritten.append(instruction)
                            continue
                        from .authored_broadcast_reference import (
                            authored_broadcast_call_actuals,
                            with_authored_broadcast_extents,
                        )
                        from .native_faults import BROADCAST_INCOMPATIBLE_SHAPE

                        reference = with_authored_broadcast_extents(
                            reference, module=module,
                        )
                        scalar32 = {
                            "physical_dtype": "int32", "storage": "scalar", "rank": 0,
                        }
                        vector32 = {
                            "physical_dtype": "int32", "storage": "span", "rank": 1,
                        }
                        axis_indices = []
                        for axis in range(broadcast_rank):
                            index, index_def = constant(axis, "int64")
                            axis_indices.append(index)
                            prefix.append(index_def)
                        output_shape, output_def = int_vector(
                            (0,) * broadcast_rank, physical=vector32,
                        )
                        rank, rank_def = constant(broadcast_rank, "int32", physical=scalar32)
                        start, start_def = constant(0, "int32", physical=scalar32)
                        prefix.extend((
                            output_def, rank_def,
                            start_def,
                        ))
                        # The authored byte-weight tensor in compact_codewords
                        # has a fixed (1, 8) descriptor while the other Mul
                        # operand carries Call-bound runtime extents. Its
                        # fixed shape is a local SSA program, with the
                        # original scoped SSA_VALUE and descriptor as causes;
                        # no second metadata Call predecessor exists.
                        static_cells = []
                        for value, descriptor, _fact in static_operands:
                            static_shape, shape_def = int_vector(
                                descriptor.shape, physical=vector32,
                            )
                            static_rank, rank_defn = constant(
                                len(descriptor.shape), "int32",
                                physical=scalar32,
                            )
                            static_count, count_defn = constant(
                                int(descriptor.static_element_count), "int32",
                                physical=scalar32,
                            )
                            prefix.extend((shape_def, rank_defn, count_defn))
                            if int(value.id) == int(wide.id):
                                wide_metadata = (
                                    static_shape, static_rank, static_count,
                                )
                            else:
                                narrow_metadata = (
                                    static_shape, static_rank, static_count,
                                )
                            # The later repository conformance call reads
                            # this very geometry through ensure_dynamic.
                            # Reusing the booked static program avoids
                            # reminting extent(weight) and turning a fixed
                            # descriptor into a second, unproved dynamic one.
                            dynamic_extents[int(value.id)] = {
                                "shape": static_shape,
                                "rank": static_rank,
                                "element_count": static_count,
                            }
                            static_cell = ssa_value_identity_cell(
                                function, int(value.id), book=book,
                            )
                            if static_cell is None:
                                raise ValueError(
                                    "mixed static broadcast lost its scoped value proof"
                                )
                            static_cells.append(static_cell)
                        aligned = narrow_metadata[0]
                        alignment_values = []
                        padding = broadcast_rank - narrow_rank
                        if padding:
                            aligned, aligned_def = int_vector(
                                (1,) * padding + (0,) * narrow_rank, physical=vector32,
                            )
                            prefix.append(aligned_def)
                            for axis in range(narrow_rank):
                                narrow_address = fresh(dtype="ptr")
                                narrow_dim = fresh(dtype="int32", physical=scalar32)
                                aligned_address = fresh(dtype="ptr")
                                prefix.extend((
                                    Instr("GetElementPtr", [narrow_metadata[0], axis_indices[axis]],
                                          narrow_address),
                                    Instr("Load", [narrow_address], narrow_dim),
                                    Instr("GetElementPtr", [aligned, axis_indices[padding + axis]],
                                          aligned_address),
                                    Instr("Store", [narrow_dim, aligned_address], None),
                                ))
                                alignment_values.extend((
                                    narrow_address, narrow_dim, aligned_address,
                                ))
                        helper, actuals, imported_cells = authored_broadcast_call_actuals(
                            reference, {"rank": rank, "start": start,
                                        "left": wide_metadata[0], "right": aligned,
                                        "output": output_shape},
                        )
                        status = fresh(dtype="int64", physical={
                            "physical_dtype": "int64", "storage": "scalar", "rank": 0,
                        })
                        helper_call = call(helper.name, list(actuals), status,
                                           instruction)
                        helper_call.attributes["callee_input_ids"] = tuple(
                            int(formal.id) for formal in helper.args
                        )
                        prefix.append(helper_call)
                        zero, zero_def = constant(0, "int64")
                        invalid = fresh(dtype="bool", physical={
                            "physical_dtype": "bool", "storage": "scalar", "rank": 0,
                        })
                        guard = Instr("CondBr", [invalid], None, attributes={
                            "runtime_broadcast_error": BROADCAST_INCOMPATIBLE_SHAPE,
                        })
                        prefix.extend((zero_def, Instr("Eq", [status, zero], invalid),
                                       guard))
                        dimensions = []
                        for index in axis_indices:
                            address = fresh(dtype="ptr")
                            dimension = fresh(dtype="int32", physical=scalar32)
                            prefix.extend((
                                Instr("GetElementPtr", [output_shape, index], address),
                                Instr("Load", [address], dimension),
                            ))
                            dimensions.append(dimension)
                        # Geometry is int32. Check each product before the next
                        # multiplication, so arbitrary rank cannot overflow an
                        # int64 intermediate. An empty axis seeds zero before
                        # the chain; large dimensions on another axis cannot
                        # spuriously reject an empty tensor.
                        zero32, zero32_def = constant(0, "int32", physical=scalar32)
                        one32, one32_def = constant(1, "int32", physical=scalar32)
                        prefix.extend((zero32_def, one32_def))
                        count_values = [zero32, one32]
                        empty = None
                        for dimension in dimensions:
                            is_zero = fresh(dtype="bool", physical={
                                "physical_dtype": "bool", "storage": "scalar", "rank": 0,
                            })
                            prefix.append(Instr("Eq", [dimension, zero32], is_zero))
                            count_values.append(is_zero)
                            if empty is None:
                                empty = is_zero
                            else:
                                combined = fresh(dtype="bool", physical={
                                    "physical_dtype": "bool", "storage": "scalar", "rank": 0,
                                })
                                prefix.append(Instr("LOr", [empty, is_zero], combined))
                                count_values.append(combined)
                                empty = combined
                        count = fresh(dtype="int32", physical=scalar32)
                        prefix.append(Instr("Select", [empty, zero32, one32], count))
                        count_values.append(count)
                        for dimension in dimensions:
                            widened = [fresh(dtype="int64", physical={
                                "physical_dtype": "int64", "storage": "scalar", "rank": 0,
                            }) for _ in range(2)]
                            prefix.extend(Instr("Cast", [value], widened_value,
                                                attributes={"source_dtype": "int32",
                                                            "target_dtype": "int64"})
                                          for value, widened_value in zip((count, dimension), widened))
                            product = fresh(dtype="int64", physical={
                                "physical_dtype": "int64", "storage": "scalar", "rank": 0,
                            })
                            count = fresh(dtype="int32", physical=scalar32)
                            prefix.extend((
                                Instr("Mul", widened, product),
                                Instr("Cast", [product], count, attributes={
                                    "source_dtype": "int64", "target_dtype": "int32",
                                    "conversion_kind": "integer_conversion",
                                    "overflow": "trap", "binding": "broadcast-count-narrowing",
                                }),
                            ))
                            count_values.extend((*widened, product, count))
                        result.accounting = {
                            **dict(result.accounting or {}),
                            "tensor_metadata_state": "dynamic",
                            "tensor_shape_value_id": int(output_shape.id),
                            "tensor_rank_value_id": int(rank.id),
                            "tensor_element_count_value_id": int(count.id),
                        }
                        register_tensor(
                            result, storage="temporary", metadata_state="dynamic",
                            extent_values=(output_shape, rank, count),
                        )
                        for operand, values in ((wide, wide_metadata),
                                                (narrow, narrow_metadata)):
                            dynamic_extents[int(operand.id)] = dict(zip(
                                ("shape", "rank", "element_count"), values,
                            ))
                        dynamic_extents[int(result.id)] = {
                            "shape": output_shape, "rank": rank,
                            "element_count": count,
                        }
                        mixed_runtime_geometry[int(result.id)] = count
                        pending_runtime_broadcast_guards.append((
                            block, guard, instruction, tuple(data_args),
                            wide, narrow, wide_metadata, narrow_metadata,
                            aligned, output_shape, rank, count, status,
                            invalid, helper, imported_cells,
                            (*geometry_cells, *static_cells),
                            (*axis_indices, *alignment_values,
                             *dimensions, *count_values),
                        ))
                from .ssa_record_return_state import posted_ssa_value_fact
                result_physical = posted_ssa_value_fact(
                    function, int(result.id), book=book,
                )
                scalar_result = (
                    result_physical is not None
                    and result_physical.storage == "scalar"
                    and result_physical.rank == 0
                )
                # Program: dynamic arange -> cast -> sum().item(). The sum
                # result10 already has a scalar/rank0 SSA_VALUE. Copying its
                # input8 dynamic span descriptor attached shape/rank/count
                # to a scalar and made the Call publish false tensor outputs.
                # The result's own physical cell decides its descriptor.
                result_descriptor = register_tensor(
                    result,
                    storage="temporary",
                    metadata_state=(
                        "static" if scalar_result else "dynamic" if scalar_geometry is not None
                        else source_descriptor.metadata_state
                        if source_descriptor is not None
                        and not tuple(result.shape)
                        else None
                    ),
                    extent_ids_from=(None if scalar_result else source_descriptor),
                    extent_values=(None if scalar_geometry is None else scalar_geometry[2][0]),
                )
                unary_geometry = (
                    _dynamic_tensor_geometry_evidence(module, function, int(source.id), pending=prefix)
                    if opcode_contract is not None and opcode_contract[0] == "unary"
                    and len(data_args) == 1 and source_descriptor is not None
                    and source_descriptor.metadata_state == "dynamic"
                    else None
                )
                # Program: a dynamic arange span is captured by the next
                # planned region as a typed formal with display shape ().
                # _known_count sees that spelling as one scalar, but its
                # tensor descriptor is explicitly unresolved.  The descriptor
                # owns the extent decision; only a settled static shape may
                # supply a compile-time kernel count.
                source_count = (
                    booked_element_count(source)
                    if source is not None and not shape_unknown(source)
                    else None
                )
                result_count = (
                    None if int(result.id) in mixed_runtime_geometry
                    else booked_element_count(result) if not shape_unknown(result)
                    else None
                )

                def need_count(
                    of: SSAValue | None, count: int | None
                ) -> SSAValue | None:
                    if count is not None:
                        value, definition = constant(count, "int32")
                        prefix.append(definition)
                        return value
                    if of is None:
                        return None
                    if of is result and result_descriptor.metadata_state == "dynamic":
                        mixed_count = mixed_runtime_geometry.get(int(result.id))
                        if mixed_count is not None:
                            return mixed_count
                        if scalar_geometry is not None:
                            from .concordance_declarations import TENSOR_ELEMENTWISE_EXTENT_BINDING
                            span, scalar, evidence = scalar_geometry
                            metadata, _geometry_causes = evidence
                            if book.latest_ref(
                                TENSOR_ELEMENTWISE_EXTENT_BINDING,
                                (function_scope_of(function), function.name,
                                 int(result.id)),
                            ) is not None:
                                return metadata[2]
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name, block_name, operation,
                            "binary result runtime extent has no exact operand geometry binding",
                        ))
                        return None
                    # Extent unknown statically: mint it as ordinary SSA.
                    return ensure_dynamic(prefix, of)["element_count"]

                # Derived source definition: cbrt(x) = sign(x)*pow(abs(x),1/3).
                # It is deliberately expressed through the finite primitive
                # basis instead of becoming a new runtime intrinsic.
                if emitted:
                    pass
                elif operation in {
                    "fill", "full", "full_like", "zeros", "zeros_like",
                    "empty", "empty_like", "ones", "ones_like",
                }:
                    like_operation = operation.endswith("_like")
                    if like_operation and source is not None:
                        result.shape = tuple(source.shape or ())
                        if source.dtype is not None:
                            result.dtype = source.dtype
                    fill_value: float | None
                    if operation in {"ones", "ones_like"}:
                        fill_value = 1.0
                    elif operation in {
                        "zeros", "zeros_like", "empty", "empty_like",
                    }:
                        fill_value = 0.0
                    else:
                        explicit = _attribute(
                            instruction.attributes, "fill_value", "value"
                        )
                        if explicit is None:
                            explicit = next((
                                item for item in metadata
                                if isinstance(item, (int, float))
                                and not isinstance(item, bool)
                            ), None)
                        fill_value = (
                            None if explicit is None else float(explicit)
                        )
                    count = (
                        constructor_count
                        if constructor_count is not None
                        else need_count(result, booked_element_count(result))
                    )
                    fill_operand = None
                    if fill_value is not None:
                        fill_operand, scalar_def = constant(
                            fill_value, "float64"
                        )
                        prefix.append(scalar_def)
                    elif operation in {"fill", "full"} and len(args) > 1:
                        fill_operand = args[1]
                    if fill_operand is not None and count is not None:
                        emitted.append(call(
                            "fill_double", [result, fill_operand, count], result,
                            instruction, output_argument=0,
                        ))
                elif operation == "cbrt" and source is not None:
                    count = need_count(source, source_count)
                    cbrt_dtype = recorded_dtype(source, "cbrt operand")
                    abs_result = fresh(shape=source.shape, dtype=cbrt_dtype)
                    sign_result = fresh(shape=source.shape, dtype=cbrt_dtype)
                    power_result = fresh(shape=source.shape, dtype=cbrt_dtype)
                    unary_abs, abs_def = constant(c_tensor_opcode("abs")[1], "int32")
                    binary_pow, pow_def = constant(c_tensor_opcode("pow")[1], "int32")
                    third, third_def = constant(1.0 / 3.0, "float64")
                    reverse, reverse_def = constant(0, "int32")
                    binary_mul, mul_def = constant(c_tensor_opcode("mul")[1], "int32")
                    prefix.extend((abs_def, pow_def, third_def, reverse_def, mul_def))
                    emitted.extend((
                        call("unary_double", [source, abs_result, count, unary_abs], abs_result, instruction, output_argument=1),
                        call("sign_double", [source, sign_result, count], sign_result, instruction, output_argument=1),
                        call("binary_scalar_double", [abs_result, third, power_result, count, binary_pow, reverse], power_result, instruction, output_argument=2),
                        call("binary_double", [sign_result, power_result, result, count, binary_mul], result, instruction, output_argument=2),
                    ))

                elif operation in _CAST_OPERATIONS and source is not None:
                    # A value cast changes dtype, never extents.  The result's
                    # annotation was taken when it was created, which may have
                    # been before the operand's own shape settled; an empty
                    # one then makes the element loop below write nothing.
                    # The fact belongs to the identity, because the consumer
                    # holds a different SSAValue with the same id.
                    if not tuple(result.shape or ()) and tuple(
                        source.shape or ()
                    ):
                        result.shape = tuple(source.shape)
                        from .identity_concordance import (
                            record_proven_shape,
                            shape_scope_of,
                        )

                        record_proven_shape(
                            shape_scope_of(function), int(result.id),
                            tuple(source.shape),
                            result.dtype or source.dtype,
                        )
                    callee = _CAST_OPERATIONS[operation]
                    dtype_hint = str(
                        _attribute(instruction.attributes, "dtype", "target_dtype")
                        or (metadata[-1] if metadata else result.dtype or "")
                    ).lower()
                    if callee is None:
                        # Dispatch on the requested dtype exactly.  The old
                        # two-way split sent every non-integer request --
                        # including an explicit ``to_dtype("float64")`` --
                        # through the single-precision narrowing kernel, and
                        # sent ``bool`` through integer truncation (2.7 -> 2
                        # where the reference says nonzero -> 1).
                        if any(
                            token in dtype_hint
                            for token in ("bool", "logical")
                        ):
                            callee = "cast_double_to_bool_values"
                        elif any(
                            token in dtype_hint for token in ("int", "long")
                        ):
                            callee = "cast_double_to_int_values"
                        elif any(
                            token in dtype_hint
                            for token in ("float64", "double")
                        ):
                            callee = "cast_double_to_double_values"
                        else:
                            callee = "cast_double_to_float_values"
                    if (
                        callee == "cast_double_to_double_values"
                        and str(source.dtype or "").casefold()
                        in {"int64", "i64", "long"}
                    ):
                        callee = "cast_int64_to_double_values"
                    from .ssa_record_return_state import (
                        posted_ssa_value_fact as _copy_source_fact,
                    )
                    copy_source_fact = _copy_source_fact(function, source.id, book=book)
                    if (
                        operation == "to_dtype"
                        and dtype_hint in {"int64", "long"}
                        and (
                            (source_descriptor is not None
                             and (source_descriptor.metadata_state == "dynamic"
                                  or bool(tuple(source_descriptor.shape))))
                            or (copy_source_fact is not None
                                and (copy_source_fact.storage == "span"
                                     or (type(copy_source_fact.rank) is int
                                         and copy_source_fact.rank > 0)))
                        )
                        and (
                            copy_source_fact is None
                            or copy_source_fact.storage != "span"
                            or type(copy_source_fact.rank) is not int
                            or copy_source_fact.rank < 1
                            or copy_source_fact.physical_dtype not in {
                                "float64", "int64",
                            }
                            or (copy_source_fact.physical_dtype == "int64"
                                and copy_source_fact.dtype != "int64")
                        )
                    ):
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name, block_name, operation,
                            "int64 cast has no proved physical input span",
                        ))
                        rewritten.append(instruction)
                        continue
                    # The axis-sum and inverse-mask probes cast a float64
                    # span to logical int64 and then reduce it. The old
                    # cast_double_to_int_values writes doubles, so the next
                    # sum_double loses integer cancellation above 2**53.
                    # Select genuine int64 storage only from the authored
                    # target and this source's posted float64 span fact.
                    genuine_integer_cast = (
                        operation == "to_dtype"
                        and dtype_hint in {"int64", "long"}
                        and copy_source_fact is not None
                        and copy_source_fact.physical_dtype == "float64"
                        and copy_source_fact.storage == "span"
                        and type(copy_source_fact.rank) is int
                        and copy_source_fact.rank > 0
                    )
                    if genuine_integer_cast:
                        callee = "cast_double_to_int64_values"
                        selected_cast = reference.module.functions.get(callee)
                        if (
                            selected_cast is None
                            or selected_cast.metadata.get("ssa_output_argument") != 1
                            or len(selected_cast.args) != 3
                            or tuple(selected_cast.metadata.get("named_outputs", ()))
                            != (("out", int(selected_cast.args[1].id)),)
                        ):
                            raise ValueError(
                                "int64 cast lacks selected repository output"
                            )
                        from .ssa_call_input_adapters import _infer_pointer_storage
                        from .ssa_record_return_state import (
                            posted_ssa_value_fact as _typed_cast_fact,
                            ssa_value_identity_cell as _typed_cast_cell,
                        )
                        link_repository_function(callee, selected_cast)
                        _infer_pointer_storage({callee: selected_cast})
                        typed_formals = selected_cast.args[:2]
                        typed_facts = tuple(_typed_cast_fact(
                            selected_cast, int(formal.id), book=book,
                        ) for formal in typed_formals)
                        typed_cells = tuple(_typed_cast_cell(
                            selected_cast, int(formal.id), book=book,
                        ) for formal in typed_formals)
                        imported_origins = (
                            selected_cast.metadata.get("llvm_repository_kernel")
                            or (None, {})
                        )[1]
                        pointer_origins = tuple(imported_origins.get(
                            int(formal.id),
                        ) for formal in typed_formals)
                        if (
                            any(fact is None or cell is None for fact, cell
                                in zip(typed_facts, typed_cells, strict=True))
                            or any(origin is None or origin[1] != "argument"
                                   or not str(origin[2]).startswith("ptr ")
                                   for origin in pointer_origins)
                            or typed_facts[0].physical_dtype != "float64"
                            or typed_facts[1].physical_dtype != "int64"
                        ):
                            raise ValueError(
                                "int64 cast lacks typed source or output formal proof"
                            )
                    exact_integer_copy = (
                        operation == "to_dtype" and dtype_hint in {"int64", "long"}
                        and str(source.dtype) == "int64"
                        and copy_source_fact is not None
                        and copy_source_fact.physical_dtype == "int64"
                        and copy_source_fact.storage == "span"
                        and copy_source_fact.rank is not None
                        and copy_source_fact.rank > 0
                    )
                    if exact_integer_copy:
                        # topk's native indices already occupy int64 lanes.
                        # Their explicit to_dtype(int64) is a fresh copy, not
                        # integer truncation through double (which rounds
                        # lanes above 2**53). Equal input/output geometry in
                        # the existing typed broadcast kernel is that copy.
                        callee = "broadcast_int64"
                        from .concordance_declarations import TENSOR_SSA_LOWERING as _COPY_STAGE
                        from .ssa_record_return_state import (
                            revise_ssa_value as _revise_copy,
                            ssa_value_identity_cell as _copy_cell,
                        )
                        _revise_copy(
                            function, result.id, dtype=result.dtype,
                            shape=tuple(result.shape or ()),
                            stage=_COPY_STAGE,
                            sources=(_copy_cell(function, source.id, book=book),),
                            book=book, physical={"physical_dtype": "int64",
                                                "storage": "span",
                                                "rank": copy_source_fact.rank},
                        )
                        result.accounting = _accounting_view(function, result, module)
                    if not tuple(source.shape or ()) and int(
                        (source.accounting or {}).get("program_abi_rank", 0)
                        or 0
                    ) == 0 and not shape_unknown(source) and (
                        _concord_scalar_kernel_operand(
                            module, function,
                            (str(function_name), int(result.id)),
                            ("scalar_cast",), (result, source),
                        ) == ("scalar_cast",)
                    ):
                        # An exact scalar conversion (``float(flag)`` on a
                        # scalar record field) is ordinary scalar SSA, as a
                        # scalar reduction is above.  Sending it through the
                        # elementwise kernel invented a span of unknown
                        # length, which the call-type pass then pushed onto
                        # the operand and the emitter rightly refused.
                        result.shape = ()
                        emitted.append(Instr(
                            "Cast", [source], result,
                            attributes={
                                "lowered_from": operation,
                                "scalar_cast_identity": True,
                            },
                            source_span=instruction.source_span,
                        ))
                    else:
                        count = need_count(source, source_count)
                        if count is not None:
                            # Program: dynamic arange -> to_dtype(float64) ->
                            # sum.  The cast kernel writes one output element
                            # per source element, and its count is the exact
                            # source extent produced above.  Registration ran
                            # before need_count made that extent available;
                            # leaving result8 unresolved prevented its
                            # metadata from crossing the next region call.
                            # The existing dynamic descriptor is the source
                            # of the output's geometry; only dtype changes.
                            live_source = tensor_table.by_id(int(source.id))
                            live_result = tensor_table.by_id(int(result.id))
                            if (
                                live_source is not None
                                and live_result is not None
                                and live_source.metadata_state == "dynamic"
                                and all(
                                    getattr(live_source, key) is not None
                                    for key in (
                                        "shape_value_id", "rank_value_id",
                                        "element_count_value_id",
                                    )
                                )
                            ):
                                tensor_table.tensors[int(result.id)] = (
                                    dataclasses.replace(
                                        live_result,
                                        metadata_state="dynamic",
                                        shape_value_id=live_source.shape_value_id,
                                        rank_value_id=live_source.rank_value_id,
                                        element_count_value_id=(
                                            live_source.element_count_value_id
                                        ),
                                        byte_size=None,
                                    )
                                )
                            output_storage = result
                            live_result = tensor_table.by_id(int(result.id))
                            if (
                                live_result is not None
                                and live_result.metadata_state == "dynamic"
                                and live_result.element_count_value_id is not None
                                and int(live_result.element_count_value_id)
                                == int(count.id)
                                and has_private_region_result(result)
                            ):
                                # A dynamic value cast preserves the source
                                # extent but produces new element storage.
                                # weighted_native_probe's to_dtype(int64)
                                # chooses cast_double_to_int_values, whose
                                # authored kernel stores double after integer
                                # truncation. The output formal's typed Store
                                # witness, not the semantic int64 result,
                                # decides the owned element representation.
                                # Infer that existing formal proof before
                                # allocating, then carry it to the exact
                                # logical result on this Call output edge.
                                from .ssa_call_input_adapters import (
                                    _infer_pointer_storage,
                                )
                                from .ssa_record_return_state import (
                                    posted_ssa_value_fact as _cast_formal_fact,
                                    ssa_value_identity_cell as _cast_formal_cell,
                                    revise_ssa_value as _revise_cast_output,
                                )
                                from .concordance_declarations import (
                                    TENSOR_SSA_LOWERING as _CAST_STAGE,
                                )
                                selected_kernel = reference.module.functions.get(callee)
                                selected_output_position = (
                                    None if selected_kernel is None else
                                    selected_kernel.metadata.get(
                                        "ssa_output_argument"
                                    )
                                )
                                if (selected_kernel is None
                                        or selected_output_position != 1
                                        or len(selected_kernel.args) <= 1):
                                    raise ValueError(
                                        "dynamic cast has no selected output formal"
                                    )
                                selected_outputs = tuple(
                                    selected_kernel.metadata.get(
                                        "named_outputs", ()
                                    )
                                )
                                if (len(selected_outputs) != 1
                                        or int(selected_outputs[0][1])
                                        != int(selected_kernel.args[1].id)):
                                    raise ValueError(
                                        "dynamic cast output name disagrees with selected formal"
                                    )
                                link_repository_function(callee, selected_kernel)
                                _infer_pointer_storage({callee: selected_kernel})
                                output_formal = selected_kernel.args[1]
                                input_formal = selected_kernel.args[0]
                                input_fact = _cast_formal_fact(
                                    selected_kernel, int(input_formal.id), book=book,
                                )
                                input_cell = _cast_formal_cell(
                                    selected_kernel, int(input_formal.id), book=book,
                                )
                                output_fact = _cast_formal_fact(
                                    selected_kernel, int(output_formal.id),
                                    book=book,
                                )
                                output_cell = _cast_formal_cell(
                                    selected_kernel, int(output_formal.id),
                                    book=book,
                                )
                                output_physical = (
                                    None if output_fact is None else
                                    output_fact.physical_dtype
                                )
                                source_cell = _cast_formal_cell(
                                    function, int(source.id), book=book,
                                )
                                live_source_fact = _cast_formal_fact(
                                    function, int(source.id), book=book,
                                )
                                imported_origins = (
                                    selected_kernel.metadata.get(
                                        "llvm_repository_kernel"
                                    ) or (None, {})
                                )[1]
                                typed_origins = tuple(
                                    imported_origins.get(int(formal.id))
                                    for formal in (input_formal, output_formal)
                                )
                                if (
                                    input_cell is None or input_fact is None
                                    or output_cell is None or output_physical is None
                                    or source_cell is None or live_source_fact is None
                                    or live_source_fact.storage != "span"
                                    or type(live_source_fact.rank) is not int
                                    or live_source_fact.rank < 1
                                    or any(origin is None
                                           or origin[1] != "argument"
                                           or not str(origin[2]).startswith("ptr ")
                                           for origin in typed_origins)
                                    or input_fact.physical_dtype !=
                                    live_source_fact.physical_dtype
                                    or (callee == "cast_double_to_int64_values"
                                        and output_physical != "int64")
                                ):
                                    raise ValueError(
                                        "dynamic cast lacks typed output or source span proof"
                                    )
                                # In compact_codewords, the selected cast wrote
                                # a double-backed logical int64 result, but its
                                # SSA_VALUE retained storage/rank=None. The
                                # following scalar-first subtraction then could
                                # not recognize that result as a span and lost
                                # its extent binding. The source's posted span
                                # rank and this selected output formal prove the
                                # cast result's physical layout and element type.
                                _revise_cast_output(
                                    function, int(result.id), dtype=result.dtype,
                                    shape=tuple(result.shape or ()),
                                    stage=_CAST_STAGE,
                                    sources=(source_cell, input_cell, output_cell), book=book,
                                    physical={
                                        "physical_dtype": output_physical,
                                        "storage": "span",
                                        "rank": live_source_fact.rank,
                                    },
                                )
                                result.accounting = _accounting_view(
                                    function, result, module,
                                )
                                output_storage = own_dynamic_tensor(
                                    result, count, str(output_physical), prefix,
                                )
                            elif (
                                live_result is not None
                                and live_result.metadata_state == "dynamic"
                                and (
                                    live_result.element_count_value_id is None
                                    or int(live_result.element_count_value_id)
                                    != int(count.id)
                                )
                            ):
                                shortfalls.append(TensorSSALoweringShortfall(
                                    function_name, block_name, operation,
                                    "dynamic cast output count disagrees with "
                                    "its tensor descriptor",
                                ))
                                rewritten.append(instruction)
                                continue
                            emitted.append(call(
                                callee, (
                                    [source, output_storage,
                                     *(ensure_dynamic(prefix, source)[kind]
                                       for kind in ("shape", "rank")),
                                     *(ensure_dynamic(prefix, source)[kind]
                                       for kind in ("shape", "rank"))]
                                    if exact_integer_copy else
                                    [source, output_storage, count]
                                ), result,
                                instruction, output_argument=1,
                            ))

                elif (
                    operation in {"transpose", "swapaxes", "permute"}
                    and source is not None
                    and not source.shape
                ):
                    # Symbolic source: a swap of axes 0 and 1 states rank two
                    # (the same contract matmul_double already carries), so
                    # the axes vector is static while shape and rank ride as
                    # runtime extents. Any other permutation of a symbolic
                    # tensor stays a refusal until ranks are inferred.
                    dims = [int(item) for item in metadata if isinstance(item, (int, float))]
                    dim0 = _attribute(instruction.attributes, "dim0", "axis1")
                    dim1 = _attribute(instruction.attributes, "dim1", "axis2")
                    if dim0 is not None and dim1 is not None:
                        dims = [int(dim0), int(dim1)]
                    if set(dims) == {0, 1}:
                        extents = ensure_dynamic(prefix, source)
                        ensure_dynamic(prefix, result)
                        axes_value, axes_def = int_vector((1, 0))
                        ndim, ndim_def = constant(2, "int32")
                        prefix.extend((axes_def, ndim_def))
                        emitted.append(call(
                            "transpose_double",
                            [source, result, extents["shape"], axes_value, ndim],
                            result, instruction, output_argument=1,
                        ))

                elif operation in {"transpose", "swapaxes", "permute"} and source is not None and source.shape:
                    rank = len(source.shape)
                    axes = _attribute(instruction.attributes, "dims", "axes", "permutation")
                    if axes is None and operation == "permute":
                        axes = next((_as_sequence(item) for item in metadata if _as_sequence(item) is not None), None)
                    if axes is None and operation in {"transpose", "swapaxes"}:
                        dims = [int(item) for item in metadata if isinstance(item, (int, float))]
                        dim0 = _attribute(instruction.attributes, "dim0", "axis1")
                        dim1 = _attribute(instruction.attributes, "dim1", "axis2")
                        if dim0 is not None and dim1 is not None:
                            dims = [int(dim0), int(dim1)]
                        if len(dims) >= 2:
                            axes_list = list(range(rank))
                            a, b = dims[-2] % rank, dims[-1] % rank
                            axes_list[a], axes_list[b] = axes_list[b], axes_list[a]
                            axes = tuple(axes_list)
                    axes_seq = _as_sequence(axes)
                    if axes_seq is not None and len(axes_seq) == rank:
                        shape_value, shape_def = int_vector(source.shape)
                        axes_value, axes_def = int_vector(int(axis) % rank for axis in axes_seq)
                        ndim, ndim_def = constant(rank, "int32")
                        prefix.extend((shape_def, axes_def, ndim_def))
                        emitted.append(call(
                            "transpose_double", [source, result, shape_value, axes_value, ndim],
                            result, instruction, output_argument=1,
                        ))

                elif (
                    operation in _REDUCTION_CODES or operation == "mean"
                ) and source is not None:
                    axis = _attribute(instruction.attributes, "axis", "dim")
                    if axis is None:
                        axis = next((
                            item for item in metadata
                            if isinstance(item, (int, float))
                            or (
                                _as_sequence(item) is not None
                                and all(
                                    isinstance(member, (int, float))
                                    for member in _as_sequence(item) or ()
                                )
                            )
                        ), None)
                    axis_sequence = _as_sequence(axis)
                    if axis_sequence is not None and len(axis_sequence) == 1:
                        axis = axis_sequence[0]
                    if (
                        axis is None
                        and not shape_unknown(source)
                        and not tuple(source.shape or ())
                    ):
                        # Rank zero is already the reduction of its sole
                        # element.  This is especially important for Python
                        # ``any``/``all`` lowered from generator predicates:
                        # concordance proves the carried predicate scalar,
                        # so asking for tensor extents here would discard a
                        # valid descriptor merely because ``()`` is falsey.
                        result.shape = ()
                        emitted.append(Instr(
                            (
                                "clone"
                                if str(source.dtype) == str(result.dtype)
                                else "Cast"
                            ),
                            [source], result,
                            attributes={
                                "lowered_from": operation,
                                "scalar_reduction_identity": True,
                            },
                            source_span=instruction.source_span,
                        ))
                    elif axis is None and operation in {"sum", "mean"}:
                        count = need_count(source, source_count)
                        if count is not None and operation == "sum":
                            sum_physical = posted_ssa_value_fact(
                                function, int(source.id), book=book,
                            )
                            # A gathered int64 span remains int64 through
                            # AbstractTensor.sum. The old sum_double choice
                            # would read integer limbs as doubles; converting
                            # them would lose values above 2**53. Select
                            # the authored typed reduction only from the
                            # source's physical span and result's int64 type.
                            sum_kernel = (
                                "sum_int64"
                                if sum_physical is not None
                                and sum_physical.storage == "span"
                                and sum_physical.rank is not None
                                and int(sum_physical.rank) > 0
                                and sum_physical.physical_dtype == "int64"
                                and sum_physical.dtype == "int64"
                                and str(result.dtype) == "int64"
                                else "sum_double"
                            )
                            emitted.append(call(
                                sum_kernel, [source, count], result,
                                instruction,
                            ))
                        elif count is not None:
                            # mean = flat sum scaled by the element count; the
                            # division is ordinary scalar SSA, not a kernel.
                            total = fresh(dtype=recorded_dtype(result, "mean result"))
                            emitted.append(call("sum_double", [source, count], total, instruction))
                            emitted.append(Instr(
                                "Div", [total, count], result,
                                attributes={"lowered_from": "mean"},
                                source_span=instruction.source_span,
                            ))
                    elif (
                        axis is None and operation in {"all", "any"}
                        and shape_unknown(source)
                    ):
                        # A public span formal can define its own runtime
                        # extents. A computed region formal must wait for the
                        # producer's metadata Call binding instead.
                        from .concordance_declarations import (
                            BindingKind, CONTROL_VALUE_BINDING, ControlBinding,
                        )
                        source_binding_ref = book.latest_ref(
                            CONTROL_VALUE_BINDING,
                            (function_scope_of(function), int(source.id)),
                        )
                        source_binding = (
                            None if source_binding_ref is None else
                            book.page(CONTROL_VALUE_BINDING).cell(
                                source_binding_ref.row,
                                source_binding_ref.column,
                            )
                        )
                        source_descriptor = tensor_table.by_id(int(source.id))
                        if (
                            source_descriptor is not None
                            and source_descriptor.metadata_state == "unresolved"
                            and int(source.id) in function_argument_ids
                            and isinstance(source_binding, ControlBinding)
                            and source_binding.kind in {
                                BindingKind.PARAMETER_SEED,
                                BindingKind.FIELD_INCUMBENT,
                            }
                            and source_binding.ssa_value.row == (
                                function_scope_of(function), int(source.id),
                            )
                        ):
                            ensure_dynamic(prefix, source)
                        geometry = _dynamic_tensor_geometry_evidence(
                            module, function, int(source.id), pending=prefix,
                        )
                        source_fact = posted_ssa_value_fact(
                            function, int(source.id), book=book,
                        )
                        if (geometry is not None and source_fact is not None
                                and source_fact.physical_dtype == "float64"
                                and source_fact.storage == "span"
                                and type(source_fact.rank) is int
                                and source_fact.rank > 0):
                            source_metadata, geometry_cells = geometry
                            source_count_value = source_metadata[2]
                            flat_shape, flat_shape_def = int_vector(
                                (0,), physical={
                                    "physical_dtype": "int32",
                                    "storage": "span", "rank": 1,
                                },
                            )
                            zero, zero_def = constant(0, "int64")
                            shape_slot = fresh(dtype="ptr")
                            flat_rank, rank_def = constant(
                                1, "int32", physical={
                                    "physical_dtype": "int32",
                                    "storage": "scalar", "rank": 0,
                                },
                            )
                            flat_axis, axis_def = constant(
                                0, "int32", physical={
                                    "physical_dtype": "int32",
                                    "storage": "scalar", "rank": 0,
                                },
                            )
                            code, code_def = constant(
                                _REDUCTION_CODES[operation], "int32",
                                physical={
                                    "physical_dtype": "int32",
                                    "storage": "scalar", "rank": 0,
                                },
                            )
                            prefix.extend((
                                flat_shape_def, zero_def,
                                Instr("GetElementPtr", [flat_shape, zero], shape_slot),
                                Instr("Store", [source_count_value, shape_slot], None),
                                rank_def, axis_def, code_def,
                            ))
                            selected = reference.module.functions.get("reduce_dim_double")
                            if (selected is None
                                    or selected.metadata.get("ssa_output_argument") != 1
                                    or len(selected.args) < 2):
                                raise ValueError(
                                    "runtime Boolean reduction lacks selected output port"
                                )
                            from .ssa_call_input_adapters import _infer_pointer_storage
                            from .ssa_record_return_state import ssa_value_identity_cell
                            link_repository_function("reduce_dim_double", selected)
                            _infer_pointer_storage({"reduce_dim_double": selected})
                            typed_formals = selected.args[:2]
                            typed_facts = tuple(posted_ssa_value_fact(
                                selected, int(formal.id), book=book,
                            ) for formal in typed_formals)
                            typed_cells = tuple(ssa_value_identity_cell(
                                selected, int(formal.id), book=book,
                            ) for formal in typed_formals)
                            origins = (selected.metadata.get(
                                "llvm_repository_kernel"
                            ) or (None, {}))[1]
                            if (
                                any(fact is None or fact.physical_dtype != "float64"
                                    for fact in typed_facts)
                                or any(cell is None for cell in typed_cells)
                                or any(
                                    (origin := origins.get(int(formal.id))) is None
                                    or origin[1] != "argument"
                                    or not str(origin[2]).startswith("ptr ")
                                    for formal in typed_formals
                                )
                            ):
                                raise ValueError(
                                    "runtime Boolean reduction lacks typed kernel formals"
                                )
                            result.shape = ()
                            _post_double_backed_boolean(
                                lowering_at[0], result, module, typed_cells,
                            )
                            domain_fact = (
                                "all-axis-runtime", int(source.id), operation,
                                *(int(value.id) for value in source_metadata),
                                int(flat_shape.id), int(flat_rank.id),
                                int(flat_axis.id), int(code.id),
                            )
                            _post_reduction_domain(
                                module, function,
                                (str(function_name), int(result.id)),
                                domain_fact,
                                (source, result, *source_metadata, flat_shape,
                                 flat_rank, flat_axis, code, zero, shape_slot),
                                extra_cells=(*geometry_cells, *typed_cells),
                            )
                            reduction_instruction = dataclasses.replace(
                                instruction,
                                attributes={
                                    **dict(instruction.attributes or {}),
                                    "tensor_reduction_domain_concordance": domain_fact,
                                },
                            )
                            emitted.append(call(
                                "reduce_dim_double",
                                [source, result, flat_shape, flat_rank,
                                 flat_axis, code], result, reduction_instruction,
                                output_argument=1,
                            ))
                    elif (
                        axis is None
                        and operation in _REDUCTION_CODES
                        and source_count is not None
                    ):
                        # The repository reduction kernel is axis-based.  The
                        # eager C backend already implements an all-axis
                        # reduction by viewing the same storage as one flat
                        # dimension; publish that exact view here instead of
                        # leaving max/any/all/prod/min without call operands.
                        # This changes only the reduction domain, never the
                        # source tensor's storage identity or descriptor.
                        flat_shape, flat_shape_def = int_vector((
                            int(source_count),
                        ))
                        flat_rank, flat_rank_def = constant(1, "int32")
                        flat_axis, flat_axis_def = constant(0, "int32")
                        code, code_def = constant(
                            _REDUCTION_CODES[operation], "int32"
                        )
                        domain_fact = (
                            int(source.id), str(operation),
                            tuple(map(int, source.shape)),
                            int(source_count),
                        )
                        domain_page = identity_book(module).page(
                            "tensor_reduction_domain_concordance"
                        )
                        domain_row = (str(function_name), int(result.id))
                        incumbent = domain_page.latest(domain_row)
                        if incumbent is not None and incumbent != domain_fact:
                            raise ValueError(
                                "tensor reduction domain concordance "
                                f"disagreement for {domain_row!r}: "
                                f"{incumbent!r} != {domain_fact!r}"
                            )
                        if incumbent is None:
                            _post_reduction_domain(
                                module, function, domain_row, domain_fact,
                                (result, source),
                            )
                        # A full reduction's result is a scalar: the shape
                        # change is a REVISE of its row, as the domain row
                        # just posted says (no in-place-only reshape).
                        result.shape = ()
                        if lowering_at[0] is not None:
                            from .concordance_declarations import (
                                TENSOR_SSA_LOWERING as _REDUCTION_STAGE,
                            )
                            from .ssa_record_return_state import (
                                revise_ssa_value as _revise_reduction,
                            )
                            _revise_reduction(
                                lowering_at[0], int(result.id),
                                dtype=result.dtype, shape=(),
                                stage=_REDUCTION_STAGE,
                                book=identity_book(module),
                            )
                        if str(result.dtype or "").casefold() in {"bool", "i1"}:
                            _post_double_backed_boolean(
                                lowering_at[0], result, module,
                            )
                        prefix.extend((
                            flat_shape_def, flat_rank_def, flat_axis_def,
                            code_def,
                        ))
                        reduction_instruction = dataclasses.replace(
                            instruction,
                            attributes={
                                **dict(instruction.attributes or {}),
                                "tensor_reduction_domain_concordance": (
                                    domain_fact
                                ),
                            },
                        )
                        emitted.append(call(
                            "reduce_dim_double",
                            [
                                source, result, flat_shape, flat_rank,
                                flat_axis, code,
                            ],
                            result, reduction_instruction,
                            output_argument=1,
                        ))
                    elif (
                        axis is not None
                        and shape_unknown(source)
                        and operation == "mean"
                    ):
                        # A repository sequence has a runtime leading extent
                        # and a statically known row shape.  Reducing axis zero
                        # is therefore an ordinary mean_dim over its one
                        # physical arena, with the sequence extent carried by
                        # the descriptor's live shape vector.
                        extents = ensure_dynamic(prefix, source)
                        dim, dim_def = constant(int(axis), "int32")
                        prefix.append(dim_def)
                        emitted.append(call(
                            "mean_dim",
                            [
                                source, result, extents["shape"],
                                extents["rank"], dim,
                            ],
                            result, instruction, output_argument=1,
                        ))
                    elif axis is not None and shape_unknown(source) and operation in _REDUCTION_CODES:
                        # compact_codewords: reduce_dim_double reads a dynamic
                        # (N, 8) product through axis 1. Its result has N
                        # elements, not the displayed one-element shape.
                        # Project the surviving source dimensions before the
                        # kernel writes its result; extent(result) here would
                        # read the output before it was defined.
                        from .ssa_record_return_state import (
                            posted_ssa_value_fact, revise_ssa_value,
                        )
                        from .concordance_declarations import (
                            TENSOR_SSA_LOWERING as _REDUCTION_STAGE,
                        )
                        source_fact = posted_ssa_value_fact(
                            function, int(source.id), book=book,
                        )
                        # Run66's loop-carried int64 rank-two source %138 had
                        # runtime geometry but no physical element type.  A
                        # dynamic shape does not select a typed reduction
                        # kernel.  Keep the authored sum for the Call closure
                        # until its exact source contract is published; no
                        # output descriptor or private heap may be claimed yet.
                        if (
                            source_fact is None
                            or source_fact.physical_dtype is None
                            or source_fact.storage != "span"
                        ):
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "dynamic reduction lacks a typed source span",
                            ))
                            rewritten.append(instruction)
                            continue
                        # A real int64 axis sum needs an int64 input and
                        # output kernel. Using reduce_dim_double here loses
                        # cancellation above 2**53 before the next sum.
                        if (operation == "sum"
                                and source_fact.physical_dtype == "int64"
                                and str(result.dtype) == "int64"):
                            axis_kernel = "sum_dim_int64"
                            kernel_dtype = "int64"
                        elif (source_fact.physical_dtype == "float64"
                              and not (operation == "sum"
                                       and str(result.dtype) == "int64")):
                            axis_kernel = "reduce_dim_double"
                            kernel_dtype = "float64"
                        else:
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "dynamic reduction lacks selected typed kernel",
                            ))
                            rewritten.append(instruction)
                            continue
                        source_rank = None if source_fact is None else source_fact.rank
                        raw_axis = int(axis)
                        normalized_axis = (
                            raw_axis if raw_axis >= 0
                            else raw_axis + source_rank
                            if source_rank is not None else -1
                        )
                        if (
                            source_rank is None or source_rank < 1
                            or not 0 <= normalized_axis < source_rank
                        ):
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name, block_name, operation,
                                "dynamic reduction lacks a proven rank or valid axis",
                            ))
                            rewritten.append(instruction)
                            continue
                        extents = ensure_dynamic(prefix, source)
                        keepdim = bool(_attribute(
                            instruction.attributes, "keepdim",
                        ) or False)
                        output_rank = (
                            source_rank if keepdim else source_rank - 1
                        )
                        scalar32 = {
                            "physical_dtype": "int32", "storage": "scalar",
                            "rank": 0,
                        }
                        geometry_values: list[SSAValue] = []
                        output_shape, output_shape_def = int_vector(
                            (0,) * output_rank, physical={
                                "physical_dtype": "int32", "storage": "span",
                                "rank": 1,
                            },
                        )
                        output_rank_value, output_rank_def = constant(
                            output_rank, "int32", physical=scalar32,
                        )
                        prefix.extend((output_shape_def, output_rank_def))
                        output_dims: list[SSAValue] = []
                        for source_axis in range(source_rank):
                            if source_axis == normalized_axis and not keepdim:
                                continue
                            if source_axis == normalized_axis:
                                dimension, definition = constant(
                                    1, "int32", physical=scalar32,
                                )
                                prefix.append(definition)
                                geometry_values.append(dimension)
                            else:
                                source_index, source_index_def = constant(
                                    source_axis, "int64",
                                )
                                source_address = fresh(dtype="ptr")
                                dimension = fresh(
                                    dtype="int32", physical=scalar32,
                                )
                                prefix.extend((
                                    source_index_def,
                                    Instr("GetElementPtr", [
                                        extents["shape"], source_index,
                                    ], source_address),
                                    Instr("Load", [source_address], dimension),
                                ))
                                geometry_values.extend((
                                    source_index, source_address, dimension,
                                ))
                            output_axis = len(output_dims)
                            output_index, output_index_def = constant(
                                output_axis, "int64",
                            )
                            output_address = fresh(dtype="ptr")
                            prefix.extend((
                                output_index_def,
                                Instr("GetElementPtr", [
                                    output_shape, output_index,
                                ], output_address),
                                Instr("Store", [dimension, output_address], None),
                            ))
                            geometry_values.extend((
                                output_index, output_address,
                            ))
                            output_dims.append(dimension)
                        if output_dims:
                            output_count = output_dims[0]
                            for dimension in output_dims[1:]:
                                product = fresh(
                                    dtype="int32", physical=scalar32,
                                )
                                prefix.append(Instr(
                                    "Mul", [output_count, dimension], product,
                                ))
                                geometry_values.append(product)
                                output_count = product
                        else:
                            output_count, output_count_def = constant(
                                1, "int32", physical=scalar32,
                            )
                            prefix.append(output_count_def)
                            geometry_values.append(output_count)
                        if output_rank:
                            result.accounting = {
                                **dict(result.accounting or {}),
                                "tensor_metadata_state": "dynamic",
                                "tensor_shape_value_id": int(output_shape.id),
                                "tensor_rank_value_id": int(output_rank_value.id),
                                "tensor_element_count_value_id": int(output_count.id),
                            }
                            revised = revise_ssa_value(
                                function, int(result.id),
                                dtype=result.dtype, shape=tuple(result.shape),
                                stage=_REDUCTION_STAGE, book=book,
                                physical={"rank": output_rank},
                            )
                            incumbent = tensor_table.by_id(int(result.id))
                            if incumbent is not None:
                                tensor_table.tensors[int(result.id)] = (
                                    dataclasses.replace(
                                        incumbent,
                                        metadata_state="dynamic",
                                        shape_value_id=int(output_shape.id),
                                        rank_value_id=int(output_rank_value.id),
                                        element_count_value_id=int(output_count.id),
                                        byte_size=None,
                                    )
                                )
                            else:
                                register_tensor(
                                    result, storage="temporary",
                                    metadata_state="dynamic",
                                )
                        domain_fact = (
                            "axis-runtime", int(source.id), str(operation),
                            int(normalized_axis), bool(keepdim),
                            int(extents["shape"].id), int(extents["rank"].id),
                            int(extents["element_count"].id),
                            int(output_shape.id), int(output_rank_value.id),
                            int(output_count.id),
                        )
                        _post_reduction_domain(
                            module, function,
                            (str(function_name), int(result.id)), domain_fact,
                            (source, result, *extents.values(), output_shape,
                             output_rank_value, output_count,
                             *geometry_values),
                        )
                        dim, dim_def = constant(normalized_axis, "int32")
                        prefix.append(dim_def)
                        if axis_kernel == "reduce_dim_double":
                            code, code_def = constant(
                                _REDUCTION_CODES[operation], "int32",
                            )
                            prefix.append(code_def)
                        reduction_output = result
                        if output_rank and has_private_region_result(result):
                            # weighted_native_probe sum27 has a runtime axis-zero
                            # result count. The authored float64 sum and the
                            # reduce_dim_double output formal agree; its
                            # private result uses the established owned tensor
                            # carrier with that exact posted count.
                            reduction_output = own_dynamic_tensor(
                                result, output_count, kernel_dtype, prefix,
                            )
                            # native_weighted_cast_tail's axis sum produces
                            # logical %27 in this region. Its rank-only SSA
                            # revision left storage/dtype unknown, although
                            # the selected reduce_dim_double output formal
                            # and owned carrier already proved a float64
                            # span. Publish that physical result here, before
                            # the following region's to_dtype reads %27.
                            from .concordance_declarations import (
                                TENSOR_OWNED_HEAP_BINDING,
                                TENSOR_REDUCTION_DOMAIN,
                            )
                            from .ssa_call_input_adapters import (
                                _infer_pointer_storage,
                            )
                            from .ssa_record_return_state import (
                                function_scope_of, ssa_value_identity_cell,
                            )
                            selected_reduction = reference.module.functions.get(
                                axis_kernel
                            )
                            if (
                                selected_reduction is None
                                or len(selected_reduction.args)
                                != (5 if axis_kernel == "sum_dim_int64" else 6)
                                or selected_reduction.metadata.get(
                                    "ssa_output_argument"
                                ) != 1
                                or selected_reduction.metadata.get(
                                    "named_outputs"
                                ) != (("out", int(
                                    selected_reduction.args[1].id
                                )),)
                            ):
                                raise ValueError(
                                    "runtime reduction lacks selected output formal"
                                )
                            link_repository_function(
                                axis_kernel, selected_reduction,
                            )
                            _infer_pointer_storage({
                                axis_kernel: selected_reduction,
                            })
                            input_formal, output_formal = (
                                selected_reduction.args[:2]
                            )
                            formal_facts = tuple(posted_ssa_value_fact(
                                selected_reduction, int(formal.id), book=book,
                            ) for formal in (input_formal, output_formal))
                            formal_cells = tuple(ssa_value_identity_cell(
                                selected_reduction, int(formal.id), book=book,
                            ) for formal in (input_formal, output_formal))
                            origins = (
                                selected_reduction.metadata.get(
                                    "llvm_repository_kernel"
                                ) or (None, {})
                            )[1]
                            owned_ref = book.latest_ref(
                                TENSOR_OWNED_HEAP_BINDING,
                                (function_scope_of(function), function.name,
                                 int(result.id)),
                            )
                            domain_ref = book.latest_ref(
                                TENSOR_REDUCTION_DOMAIN,
                                (function.name, int(result.id)),
                            )
                            result_fact = posted_ssa_value_fact(
                                function, int(result.id), book=book,
                            )
                            descriptor = tensor_table.by_id(int(result.id))
                            owned_fact = (
                                None if owned_ref is None else
                                book.page(TENSOR_OWNED_HEAP_BINDING).cell(
                                    owned_ref.row, owned_ref.column,
                                )
                            )
                            if (
                                any(fact is None or cell is None
                                    for fact, cell in zip(
                                        formal_facts, formal_cells, strict=True,
                                    ))
                                or any(fact.physical_dtype != kernel_dtype
                                       for fact in formal_facts)
                                or any(
                                    (origin := origins.get(int(formal.id)))
                                    is None or origin[1] != "argument"
                                    or not str(origin[2]).startswith("ptr ")
                                    for formal in (input_formal, output_formal)
                                )
                                or source_fact is None
                                or source_fact.physical_dtype != kernel_dtype
                                or source_fact.storage != "span"
                                or result_fact is None
                                or result_fact.rank != output_rank
                                or result_fact.physical_dtype not in (
                                    None, kernel_dtype,
                                )
                                or result_fact.storage not in (None, "span")
                                or descriptor is None
                                or descriptor.data_value_id != int(
                                    reduction_output.id
                                )
                                or owned_fact != (
                                    int(reduction_output.id),
                                    descriptor.heap_owner_value_id,
                                    int(output_count.id), kernel_dtype,
                                )
                                or domain_ref is None
                            ):
                                raise ValueError(
                                    "runtime reduction output lacks typed owned span proof"
                                )
                            source_cell = ssa_value_identity_cell(
                                function, int(source.id), book=book,
                            )
                            geometry_cells = tuple(ssa_value_identity_cell(
                                function, int(value.id), book=book,
                            ) for value in (
                                *extents.values(), output_shape,
                                output_rank_value, output_count,
                            ))
                            if source_cell is None or any(
                                cell is None for cell in geometry_cells
                            ):
                                raise ValueError(
                                    "runtime reduction output lacks geometry cells"
                                )
                            revised = revise_ssa_value(
                                function, int(result.id), dtype=result.dtype,
                                shape=tuple(result.shape or ()),
                                stage=_REDUCTION_STAGE, book=book,
                                sources=(
                                    source_cell, *formal_cells,
                                    *geometry_cells, domain_ref, owned_ref,
                                ),
                                physical={
                                    "physical_dtype": kernel_dtype,
                                    "storage": "span", "rank": output_rank,
                                },
                            )
                            if revised is None:
                                raise ValueError(
                                    "runtime reduction output has no SSA identity"
                                )
                        elif axis_kernel == "sum_dim_int64" and not output_rank:
                            # int64_axis_edge_scalar's sum(dim=0) returns a
                            # scalar. Its posted int64/scalar fact is already
                            # physical, so revise_ssa_value would be a no-op.
                            # Cite the selected pointer formals and the exact
                            # source/domain geometry on that same fact; a
                            # direct scalar output needs no heap owner.
                            from .concordance_declarations import (
                                SSA_CONSTANT_MATERIALIZATION, SSA_VALUE,
                                TENSOR_REDUCTION_DOMAIN,
                            )
                            from .identity_concordance import Derived, Mode
                            from .ssa_call_input_adapters import (
                                _infer_pointer_storage,
                            )
                            from .ssa_record_return_state import (
                                function_scope_of, ssa_value_identity_cell,
                            )
                            selected_reduction = reference.module.functions.get(
                                axis_kernel
                            )
                            if (
                                selected_reduction is None
                                or len(selected_reduction.args) != 5
                                or selected_reduction.metadata.get(
                                    "ssa_output_argument"
                                ) != 1
                                or selected_reduction.metadata.get(
                                    "named_outputs"
                                ) != (("out", int(
                                    selected_reduction.args[1].id
                                )),)
                            ):
                                raise ValueError(
                                    "runtime scalar reduction lacks selected output formal"
                                )
                            link_repository_function(
                                axis_kernel, selected_reduction,
                            )
                            _infer_pointer_storage({
                                axis_kernel: selected_reduction,
                            })
                            formal_cells = tuple(ssa_value_identity_cell(
                                selected_reduction, int(formal.id), book=book,
                            ) for formal in selected_reduction.args[:2])
                            formal_facts = tuple(posted_ssa_value_fact(
                                selected_reduction, int(formal.id), book=book,
                            ) for formal in selected_reduction.args[:2])
                            origins = (
                                selected_reduction.metadata.get(
                                    "llvm_repository_kernel"
                                ) or (None, {})
                            )[1]
                            result_fact = posted_ssa_value_fact(
                                function, int(result.id), book=book,
                            )
                            descriptor = tensor_table.by_id(int(result.id))
                            domain_ref = book.latest_ref(
                                TENSOR_REDUCTION_DOMAIN,
                                (function.name, int(result.id)),
                            )
                            source_cell = ssa_value_identity_cell(
                                function, int(source.id), book=book,
                            )
                            geometry_cells = tuple(ssa_value_identity_cell(
                                function, int(value.id), book=book,
                            ) for value in (
                                *extents.values(), output_shape,
                                output_rank_value, output_count,
                            ))
                            if (
                                source_cell is None or domain_ref is None
                                or any(cell is None for cell in (
                                    *formal_cells, *geometry_cells,
                                ))
                                or any(fact is None or
                                       fact.physical_dtype != "int64"
                                       for fact in formal_facts)
                                or any(
                                    (origin := origins.get(int(formal.id)))
                                    is None or origin[1] != "argument"
                                    or not str(origin[2]).startswith("ptr ")
                                    for formal in selected_reduction.args[:2]
                                )
                                or result_fact is None
                                or (result_fact.dtype,
                                    result_fact.physical_dtype,
                                    result_fact.storage, result_fact.rank)
                                != ("int64", "int64", "scalar", 0)
                                or descriptor is None
                                or descriptor.dtype != "int64"
                                or descriptor.data_value_id != int(result.id)
                                or descriptor.metadata_state != "static"
                            ):
                                raise ValueError(
                                    "runtime scalar reduction lacks typed output proof"
                                )
                            # The scalar result's output shape (), rank 0, and
                            # count 1 are explicit int32 helper constants. The
                            # later pure-instruction prune may remove all three
                            # Const definitions because the kernel only reads
                            # the source geometry. Keep their exact typed
                            # literal identities on the incumbent materialization
                            # page, caused by this reduction domain and its
                            # selected int64 pointer formals.
                            scalar_geometry_receipts = []
                            for geometry_value, definition, literal in (
                                (output_shape, output_shape_def, ()),
                                (output_rank_value, output_rank_def, 0),
                                (output_count, output_count_def, 1),
                            ):
                                geometry_cell = ssa_value_identity_cell(
                                    function, int(geometry_value.id), book=book,
                                )
                                if (
                                    definition.op != "Const"
                                    or int(definition.res.id) != int(geometry_value.id)
                                    or geometry_cell is None
                                    or definition.attributes.get(
                                        "values" if isinstance(literal, tuple)
                                        else "constant"
                                    ) != literal
                                ):
                                    raise ValueError(
                                        "runtime scalar reduction lacks exact "
                                        "typed geometry literal"
                                    )
                                scalar_geometry_receipts.append(book.post(
                                    SSA_CONSTANT_MATERIALIZATION,
                                    (function_scope_of(function), function.name,
                                     block_name, int(geometry_value.id)),
                                    ("Const", "int32", repr(literal)),
                                    stage=_REDUCTION_STAGE,
                                    provenance=Derived((
                                        geometry_cell, domain_ref, *formal_cells,
                                    )),
                                    mode=Mode.CONCORD,
                                ))
                            book.post(
                                SSA_VALUE,
                                (function_scope_of(function), int(result.id)),
                                result_fact,
                                stage=_REDUCTION_STAGE,
                                provenance=Derived((
                                    source_cell, *formal_cells,
                                    *geometry_cells, domain_ref,
                                    *scalar_geometry_receipts,
                                )),
                                mode=Mode.CONCORD,
                            )
                        emitted.append(call(
                            axis_kernel,
                            [source, reduction_output, extents["shape"],
                             extents["rank"], dim]
                            + ([code] if axis_kernel == "reduce_dim_double" else []),
                            result, instruction, output_argument=1,
                        ))
                    elif axis is not None and source.shape and operation == "mean":
                        rank = len(source.shape)
                        axis_value = int(axis) % rank
                        shape_value, shape_def = int_vector(source.shape)
                        ndim, ndim_def = constant(rank, "int32")
                        dim, dim_def = constant(axis_value, "int32")
                        prefix.extend((shape_def, ndim_def, dim_def))
                        emitted.append(call(
                            "mean_dim",
                            [source, result, shape_value, ndim, dim],
                            result, instruction, output_argument=1,
                        ))
                    elif axis is not None and source.shape and operation in _REDUCTION_CODES:
                        rank = len(source.shape)
                        axis_value = int(axis) % rank
                        shape_value, shape_def = int_vector(source.shape)
                        ndim, ndim_def = constant(rank, "int32")
                        dim, dim_def = constant(axis_value, "int32")
                        code, code_def = constant(_REDUCTION_CODES[operation], "int32")
                        prefix.extend((shape_def, ndim_def, dim_def, code_def))
                        emitted.append(call(
                            "reduce_dim_double", [source, result, shape_value, ndim, dim, code],
                            result, instruction, output_argument=1,
                        ))
                    elif (
                        axis is not None
                        and source_count is not None
                        and result_count is not None
                        and source_count == result_count
                    ):
                        # The reduced axis has extent one: that extent IS the
                        # ratio of the operand's element count to the result's.
                        # Summing, multiplying or selecting over a single
                        # element returns that element, so every reduction code
                        # agrees with the repository's shape-only alias and no
                        # reduction kernel is required.  Region planning
                        # produces exactly this case -- a specialized slice of
                        # length one arrives as one element with no axis left
                        # to spell -- and the counts still refuse a genuine
                        # multi-element axis, whose ratio is greater than one.
                        emitted.append(Instr(
                            "clone", [source], result,
                            attributes={"lowered_from": operation},
                            source_span=instruction.source_span,
                        ))

                elif operation == "cumsum" and source is not None and source.shape:
                    # A prefix scan does not change shape.  The kernel's shape
                    # vector is built from the operand below, but the RESULT
                    # kept whatever annotation it arrived with -- for an
                    # intermediate that is ``()``, which the next operation
                    # reads as a rank-0 scalar and whose element loop then
                    # writes nothing.  ``hot.cumsum(dim=-1).to_dtype(...)``
                    # left its output buffer untouched for exactly this
                    # reason, and every value above it in the pivot search
                    # read the uninitialized result.  Returning the cumsum
                    # directly hid it, because the published contract stamped
                    # the shape back on.
                    from .ssa_record_return_state import posted_ssa_value_fact
                    scan_source_fact = posted_ssa_value_fact(
                        function, int(source.id), book=book,
                    )
                    scan_kernel_name = {
                        "float64": "cumsum_dim_double",
                        "int64": "cumsum_dim_int64",
                    }.get(
                        None if scan_source_fact is None else
                        scan_source_fact.physical_dtype
                    )
                    if scan_kernel_name is None:
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name, block_name, operation,
                            "static cumsum lacks exact typed span input",
                        ))
                        rewritten.append(instruction)
                        continue
                    # The static scan has the same typed-buffer contract as
                    # the runtime scan.  A known shape does not make an
                    # int64 span safe to pass to cumsum_dim_double.
                    from .concordance_declarations import TENSOR_SSA_LOWERING
                    from .ssa_call_input_adapters import _infer_pointer_storage
                    from .ssa_record_return_state import (
                        revise_ssa_value, ssa_value_identity_cell,
                    )
                    scan_kernel = reference.module.functions.get(scan_kernel_name)
                    if (
                        scan_kernel is None or len(scan_kernel.args) != 5
                        or scan_kernel.metadata.get(
                            "ssa_output_argument"
                        ) != 1
                        or len(tuple(scan_kernel.metadata.get(
                            "named_outputs", ()
                        ))) != 1
                        or int(tuple(scan_kernel.metadata[
                            "named_outputs"
                        ])[0][1]) != int(scan_kernel.args[1].id)
                    ):
                        raise ValueError(
                            "static cumsum lacks selected typed kernel"
                        )
                    link_repository_function(scan_kernel_name, scan_kernel)
                    _infer_pointer_storage({scan_kernel_name: scan_kernel})
                    scan_formals = scan_kernel.args[:2]
                    scan_formal_facts = tuple(posted_ssa_value_fact(
                        scan_kernel, int(formal.id), book=book,
                    ) for formal in scan_formals)
                    scan_formal_cells = tuple(ssa_value_identity_cell(
                        scan_kernel, int(formal.id), book=book,
                    ) for formal in scan_formals)
                    scan_origins = (
                        scan_kernel.metadata.get(
                            "llvm_repository_kernel") or (None, {})
                    )[1]
                    if (
                        any(fact is None or cell is None for fact, cell in zip(
                            scan_formal_facts, scan_formal_cells, strict=True,
                        ))
                        or any(
                            (origin := scan_origins.get(int(formal.id))) is None
                            or origin[1] != "argument"
                            or not str(origin[2]).startswith("ptr ")
                            for formal in scan_formals
                        )
                        or any(fact.physical_dtype !=
                               scan_source_fact.physical_dtype
                               for fact in scan_formal_facts)
                    ):
                        raise ValueError(
                            "static cumsum disagrees with typed kernel formals"
                        )
                    scan_result_fact = posted_ssa_value_fact(
                        function, int(result.id), book=book,
                    )
                    if (
                        scan_result_fact is None
                        or scan_result_fact.physical_dtype not in (
                            None, scan_formal_facts[1].physical_dtype,
                        )
                    ):
                        raise ValueError(
                            "static cumsum result has conflicting physical type"
                        )
                    if scan_result_fact.physical_dtype is None:
                        scan_source_cell = ssa_value_identity_cell(
                            function, int(source.id), book=book,
                        )
                        if scan_source_cell is None or revise_ssa_value(
                            function, int(result.id), dtype=result.dtype,
                            shape=tuple(result.shape or ()),
                            stage=TENSOR_SSA_LOWERING,
                            sources=(scan_source_cell, *scan_formal_cells),
                            book=book,
                            physical={"physical_dtype":
                                      scan_formal_facts[1].physical_dtype},
                        ) is None:
                            raise ValueError(
                                "static cumsum result lacks typed physical proof"
                            )
                    if not tuple(result.shape or ()):
                        result.shape = tuple(source.shape)
                        result.dtype = result.dtype or source.dtype
                    # Setting the field only fixes THIS object.  The consumer
                    # holds a different SSAValue carrying the same id, so the
                    # fact belongs to the identity, not to the occurrence --
                    # record it where every occurrence can read it.
                    from .identity_concordance import (
                        record_proven_shape,
                        shape_scope_of,
                    )

                    record_proven_shape(
                        shape_scope_of(function), int(result.id),
                        tuple(source.shape),
                        result.dtype or source.dtype,
                    )
                    axis = _attribute(instruction.attributes, "axis", "dim")
                    if axis is None:
                        axis = next((item for item in metadata if isinstance(item, (int, float))), 0)
                    rank = len(source.shape)
                    shape_value, shape_def = int_vector(source.shape)
                    ndim, ndim_def = constant(rank, "int32")
                    dim, dim_def = constant(int(axis) % rank, "int32")
                    prefix.extend((shape_def, ndim_def, dim_def))
                    emitted.append(call(
                        scan_kernel_name, [source, result, shape_value, ndim, dim],
                        result, instruction, output_argument=1,
                    ))

                elif operation == "cumsum" and source is not None:
                    # A runtime-sized scan preserves the source geometry. The
                    # source's booked extent triple, rather than the empty
                    # display shape of result3 in dynamic_cumsum_probe, owns
                    # the element count and the private output allocation.
                    from .concordance_declarations import (
                        BindingKind, CANONICAL_VALUE, CONTROL_VALUE_BINDING,
                        ControlBinding, TENSOR_OWNED_HEAP_BINDING,
                        TENSOR_RUNTIME_SCAN_GEOMETRY, TENSOR_SSA_LOWERING,
                    )
                    from .concorded_literals import (
                        MISSING as MISSING_LITERAL, literal_from_identity,
                    )
                    from .identity_concordance import Derived, Mode
                    from .ssa_record_return_state import (
                        function_scope_of, posted_ssa_value_fact,
                        ssa_value_identity_cell,
                    )

                    scope = function_scope_of(function)
                    source_fact = posted_ssa_value_fact(
                        function, int(source.id), book=book,
                    )
                    result_fact = posted_ssa_value_fact(
                        function, int(result.id), book=book,
                    )
                    axis_value = args[1] if len(args) == 2 else None
                    axis_cell = (
                        None if axis_value is None else
                        ssa_value_identity_cell(
                            function, int(axis_value.id), book=book,
                        )
                    )
                    axis_origins = (
                        () if axis_cell is None else tuple(
                            parent for parent, _stage in book.edges_into(axis_cell)
                            if parent.page == CANONICAL_VALUE
                        )
                    )
                    axis_literal, literal_cells = (
                        literal_from_identity(book, axis_origins[0])
                        if len(axis_origins) == 1 else (MISSING_LITERAL, ())
                    )
                    local_axis = (
                        None if axis_value is None else
                        constants.get(int(axis_value.id))
                    )
                    attributed_axis = _attribute(
                        instruction.attributes, "axis", "dim",
                    )
                    source_rank = None if source_fact is None else source_fact.rank
                    normalized_axis = (
                        axis_literal if type(axis_literal) is int
                        and axis_literal >= 0 else
                        axis_literal + source_rank
                        if type(axis_literal) is int and source_rank is not None
                        else None
                    )
                    binding_ref = book.latest_ref(
                        CONTROL_VALUE_BINDING, (scope, int(source.id)),
                    )
                    binding = (
                        None if binding_ref is None else
                        book.page(CONTROL_VALUE_BINDING).cell(
                            binding_ref.row, binding_ref.column,
                        )
                    )
                    booked_ids = (
                        () if source_descriptor is None else tuple(
                            getattr(source_descriptor, field) for field in (
                                "shape_value_id", "rank_value_id",
                                "element_count_value_id",
                            )
                        )
                    )
                    can_mint_input = (
                        int(source.id) in function_argument_ids
                        and isinstance(binding, ControlBinding)
                        and binding.kind in {
                            BindingKind.PARAMETER_SEED,
                            BindingKind.FIELD_INCUMBENT,
                        }
                        and binding.ssa_value.row == (scope, int(source.id))
                    )
                    if (
                        source_fact is None or result_fact is None
                        or source_fact.physical_dtype not in {
                            "float64", "int64",
                        }
                        or result_fact.physical_dtype not in {
                            None, source_fact.physical_dtype,
                        }
                        or result_fact.dtype != source_fact.dtype
                        or source_fact.storage != "span"
                        or result_fact.storage != "span"
                        or type(source_rank) is not int or source_rank < 1
                        or result_fact.rank != source_rank
                        or type(axis_literal) is not int
                        or (local_axis is not None
                            and local_axis != axis_literal)
                        or (attributed_axis is not None
                            and attributed_axis != axis_literal)
                        or normalized_axis is None
                        or not 0 <= normalized_axis < source_rank
                        or not private_dynamic_result(result)
                        or (not can_mint_input and not (
                            source_descriptor is not None
                            and source_descriptor.metadata_state == "dynamic"
                            and len(booked_ids) == 3
                            and all(type(identity) is int
                                    for identity in booked_ids)
                        ))
                    ):
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name, block_name, operation,
                            "runtime cumsum lacks exact physical span, axis, "
                            "input geometry or private output proof",
                        ))
                        rewritten.append(instruction)
                        continue
                    extents = ensure_dynamic(prefix, source)
                    evidence = _dynamic_tensor_geometry_evidence(
                        module, function, int(source.id), pending=prefix,
                    )
                    if evidence is None or tuple(
                        int(value.id) for value in evidence[0]
                    ) != tuple(int(extents[kind].id) for kind in (
                        "shape", "rank", "element_count",
                    )):
                        shortfalls.append(TensorSSALoweringShortfall(
                            function_name, block_name, operation,
                            "runtime cumsum source has no exact extent binding",
                        ))
                        rewritten.append(instruction)
                        continue
                    from .ssa_call_input_adapters import (
                        _infer_pointer_storage,
                    )
                    selected_kernel_name = (
                        "cumsum_dim_int64"
                        if source_fact.physical_dtype == "int64"
                        else "cumsum_dim_double"
                    )
                    selected_kernel = reference.module.functions.get(
                        selected_kernel_name
                    )
                    if (
                        selected_kernel is None
                        or len(selected_kernel.args) != 5
                        or selected_kernel.metadata.get(
                            "ssa_output_argument"
                        ) != 1
                        or len(tuple(selected_kernel.metadata.get(
                            "named_outputs", ()
                        ))) != 1
                        or int(tuple(selected_kernel.metadata[
                            "named_outputs"
                        ])[0][1]) != int(selected_kernel.args[1].id)
                    ):
                        raise ValueError(
                            "runtime cumsum lacks selected kernel output port"
                        )
                    link_repository_function(
                        selected_kernel_name, selected_kernel,
                    )
                    _infer_pointer_storage({
                        selected_kernel_name: selected_kernel,
                    })
                    selected_formals = (
                        selected_kernel.args[0], selected_kernel.args[1],
                    )
                    typed_facts = tuple(posted_ssa_value_fact(
                        selected_kernel, int(formal.id), book=book,
                    ) for formal in selected_formals)
                    typed_cells = tuple(ssa_value_identity_cell(
                        selected_kernel, int(formal.id), book=book,
                    ) for formal in selected_formals)
                    imported_origins = (
                        selected_kernel.metadata.get(
                            "llvm_repository_kernel") or (None, {})
                    )[1]
                    pointer_origins = tuple(
                        imported_origins.get(int(formal.id))
                        for formal in selected_formals
                    )
                    if (
                        any(fact is None or cell is None
                            for fact, cell in zip(
                                typed_facts, typed_cells, strict=True,
                            ))
                        or any(
                            origin is None or origin[1] != "argument"
                            or not str(origin[2]).startswith("ptr ")
                            for origin in pointer_origins
                        )
                        or typed_facts[0].physical_dtype !=
                        typed_facts[1].physical_dtype
                        or typed_facts[0].physical_dtype !=
                        source_fact.physical_dtype
                        or result_fact.physical_dtype not in (
                            None, typed_facts[1].physical_dtype,
                        )
                    ):
                        raise ValueError(
                            "runtime cumsum physical span disagrees with selected kernel"
                        )
                    # compact_codewords' lengths.cumsum(dim=0) is a real
                    # int64 scan.  Its logical result can enter this region
                    # before physical dtype settlement; the selected typed
                    # output formal and the source's posted span prove that
                    # physical result, including the empty-input case.
                    if result_fact.physical_dtype is None:
                        from .ssa_record_return_state import revise_ssa_value
                        source_cell = ssa_value_identity_cell(
                            function, int(source.id), book=book,
                        )
                        if source_cell is None:
                            raise ValueError(
                                "runtime cumsum source lacks physical SSA proof"
                            )
                        revised = revise_ssa_value(
                            function, int(result.id), dtype=result.dtype,
                            shape=tuple(result.shape or ()),
                            stage=TENSOR_SSA_LOWERING,
                            sources=(source_cell, *typed_cells), book=book,
                            physical={
                                "physical_dtype": typed_facts[1].physical_dtype,
                            },
                        )
                        if revised is None:
                            raise ValueError(
                                "runtime cumsum result lacks physical SSA proof"
                            )
                    result.accounting = {
                        **dict(result.accounting or {}),
                        "tensor_metadata_state": "dynamic",
                        "tensor_shape_value_id": int(extents["shape"].id),
                        "tensor_rank_value_id": int(extents["rank"].id),
                        "tensor_element_count_value_id": int(
                            extents["element_count"].id
                        ),
                    }
                    register_tensor(
                        result, storage="temporary", metadata_state="dynamic",
                        extent_values=tuple(extents[kind] for kind in (
                            "shape", "rank", "element_count",
                        )),
                    )
                    dynamic_extents[int(result.id)] = dict(extents)
                    carrier = own_dynamic_tensor(
                        result, extents["element_count"],
                        str(typed_facts[1].physical_dtype), prefix,
                    )
                    dim, dim_def = constant(normalized_axis, "int32")
                    prefix.append(dim_def)
                    emitted.append(call(
                        selected_kernel_name, [
                            source, carrier, extents["shape"],
                            extents["rank"], dim,
                        ], result, instruction, output_argument=1,
                    ))
                    owner_ref = book.latest_ref(
                        TENSOR_OWNED_HEAP_BINDING,
                        (scope, function.name, int(result.id)),
                    )
                    scan_values = (
                        source, result, axis_value, *evidence[0], carrier, dim,
                    )
                    scan_cells = tuple(
                        ssa_value_identity_cell(
                            function, int(value.id), book=book,
                        ) for value in scan_values
                    )
                    causes = tuple(dict.fromkeys((
                        *scan_cells, *evidence[1], *literal_cells,
                        axis_origins[0], owner_ref, *typed_cells,
                    )))
                    if any(cell is None for cell in causes):
                        raise ValueError(
                            "runtime cumsum lost exact scan evidence"
                        )
                    book.post(
                        TENSOR_RUNTIME_SCAN_GEOMETRY,
                        (scope, function.name, int(result.id)),
                        (
                            "cumsum", int(source.id), int(axis_value.id),
                            normalized_axis, *(int(value.id)
                                               for value in evidence[0]),
                            int(carrier.id), selected_kernel_name,
                        ),
                        stage=TENSOR_SSA_LOWERING,
                        provenance=Derived(causes), mode=Mode.CONCORD,
                    )

                elif operation == "stack" and data_args:
                    operand_shape = tuple(source.shape or ())
                    statically_known = (
                        bool(operand_shape)
                        or all(
                            int(operand.id) not in unresolved_argument_ids
                            for operand in data_args
                        )
                    )
                    if statically_known and all(
                        tuple(operand.shape or ()) == operand_shape
                        for operand in data_args
                    ):
                        rank = len(operand_shape)
                        raw_dim = _attribute(instruction.attributes, "axis", "dim")
                        if raw_dim is None:
                            raw_dim = next((
                                item for item in metadata
                                if isinstance(item, (int, float))
                            ), 0)
                        dim_index = int(raw_dim) % (rank + 1)
                        result.shape = (
                            *operand_shape[:dim_index], len(data_args),
                            *operand_shape[dim_index:],
                        )
                        pointers = fresh(
                            shape=(len(data_args),), dtype="ptrptr_float64"
                        )
                        shape_value, shape_def = int_vector(operand_shape)
                        count_value, count_def = constant(
                            len(data_args), "int32"
                        )
                        rank_value, rank_def = constant(rank, "int32")
                        dim_value, dim_def = constant(dim_index, "int32")
                        prefix.extend((
                            Instr("PointerArray", data_args, pointers),
                            shape_def, count_def, rank_def, dim_def,
                        ))
                        emitted.append(call(
                            "stack_double",
                            [
                                pointers, count_value, shape_value,
                                rank_value, dim_value, result,
                            ],
                            result, instruction, output_argument=5,
                        ))

                elif operation in {"cat", "concat", "concatenate"} and data_args:
                    rank = len(tuple(source.shape or ()))
                    raw_dim = _attribute(instruction.attributes, "axis", "dim")
                    if raw_dim is None:
                        raw_dim = next((
                            item for item in metadata
                            if isinstance(item, (int, float))
                        ), 0)
                    dim_index = int(raw_dim) % rank if rank else 0
                    shapes = tuple(tuple(value.shape or ()) for value in data_args)
                    compatible = bool(rank) and all(
                        len(shape) == rank
                        and all(
                            axis == dim_index or shape[axis] == source.shape[axis]
                            for axis in range(rank)
                        )
                        for shape in shapes
                    )
                    if compatible:
                        result_shape = list(map(int, source.shape))
                        result_shape[dim_index] = sum(
                            int(shape[dim_index]) for shape in shapes
                        )
                        result.shape = tuple(result_shape)
                        pointers = fresh(
                            shape=(len(data_args),), dtype="ptrptr_float64"
                        )
                        sizes, sizes_def = int_vector(
                            shape[dim_index] for shape in shapes
                        )
                        shape_value, shape_def = int_vector(source.shape)
                        count_value, count_def = constant(
                            len(data_args), "int32"
                        )
                        rank_value, rank_def = constant(rank, "int32")
                        dim_value, dim_def = constant(dim_index, "int32")
                        prefix.extend((
                            Instr("PointerArray", data_args, pointers),
                            sizes_def, shape_def, count_def, rank_def, dim_def,
                        ))
                        emitted.append(call(
                            "cat_double",
                            [
                                pointers, sizes, count_value, shape_value,
                                rank_value, dim_value, result,
                            ],
                            result, instruction, output_argument=6,
                        ))

                elif operation == "where" and len(data_args) == 3:
                    count = need_count(result, result_count)
                    conformed = []
                    for operand in data_args:
                        if tuple(operand.shape or ()) == tuple(result.shape or ()):
                            # ``where_double`` reads DOUBLE buffers.  A
                            # shape-matched scalar literal of another dtype
                            # is respelled as its float64 constant, the same
                            # rule the broadcast branch below applies.  In
                            # BACKWARD_RULES["maximum"] (``where(y>x, 1,
                            # ...)``, scalar operands) the int64 literal 1
                            # crossed as raw bytes and read 4.9e-324, so
                            # d Max(0,u)*y / du at y=2 came out 1e-323.
                            payload = constants.get(int(operand.id))
                            if (
                                not tuple(operand.shape or ())
                                and isinstance(payload, (int, float))
                                and not isinstance(payload, bool)
                                and str(operand.dtype or "").casefold()
                                != "float64"
                            ):
                                respelled, respelled_def = constant(
                                    float(payload), "float64"
                                )
                                prefix.append(respelled_def)
                                operand = respelled
                            conformed.append(operand)
                            continue
                        temporary = fresh(
                            dtype=recorded_dtype(result, "where result"),
                            shape=tuple(result.shape or ()),
                        )
                        register_tensor(temporary, storage="temporary")
                        if not tuple(operand.shape or ()):
                            scalar = operand
                            payload = constants.get(int(operand.id))
                            if (
                                isinstance(payload, (int, float))
                                and not isinstance(payload, bool)
                                and str(operand.dtype or "").casefold()
                                != "float64"
                            ):
                                scalar, scalar_def = constant(
                                    float(payload), "float64"
                                )
                                prefix.append(scalar_def)
                            emitted.append(call(
                                "fill_double", [temporary, scalar, count],
                                temporary, instruction, output_argument=0,
                            ))
                        elif operand.shape and result.shape:
                            source_shape, source_shape_def = int_vector(operand.shape)
                            source_rank, source_rank_def = constant(
                                len(operand.shape), "int32"
                            )
                            output_shape, output_shape_def = int_vector(result.shape)
                            output_rank, output_rank_def = constant(
                                len(result.shape), "int32"
                            )
                            prefix.extend((
                                source_shape_def, source_rank_def,
                                output_shape_def, output_rank_def,
                            ))
                            emitted.append(call(
                                "broadcast_double",
                                [
                                    operand, temporary, source_shape, source_rank,
                                    output_shape, output_rank,
                                ],
                                temporary, instruction, output_argument=1,
                            ))
                        else:
                            shortfalls.append(TensorSSALoweringShortfall(
                                function_name,
                                block_name,
                                operation,
                                "where operand cannot be conformed without "
                                f"a result shape; operand=%{operand.id} "
                                f"result=%{result.id}",
                            ))
                            emitted.clear()
                            break
                        conformed.append(temporary)
                    if len(conformed) != 3:
                        rewritten.append(instruction)
                        continue
                    emitted.append(call(
                        "where_double", [*conformed, result, count], result,
                        instruction, output_argument=3,
                    ))

                elif operation in {"broadcast_to", "expand"} and source is not None:
                    output_shape = broadcast_shape
                    if (
                        output_shape is not None
                        and not tuple(source.shape or ())
                        and int((source.accounting or {}).get(
                            "program_abi_rank", 0
                        ) or 0) == 0
                        and not shape_unknown(source)
                        and _concord_scalar_kernel_operand(
                            module, function,
                            (str(function_name), int(result.id)),
                            ("scalar_fill",), (result, source),
                        ) == ("scalar_fill",)
                    ):
                        # Broadcasting an exact scalar is a fill: the value
                        # crosses by value.  Passing it to the array kernel as
                        # a rank-zero buffer made its caller's scalar a span
                        # of unknown length at the call boundary.
                        result.shape = tuple(map(int, output_shape))
                        count = need_count(result, booked_element_count(result))
                        if count is not None:
                            emitted.append(call(
                                "fill_double", [result, source, count],
                                result, instruction, output_argument=0,
                            ))
                    elif output_shape is not None and (
                        source.shape or source.dtype is not None
                    ):
                        output_shape = tuple(map(int, output_shape))
                        result.shape = output_shape
                        input_shape, input_shape_def = int_vector(source.shape)
                        input_rank, input_rank_def = constant(
                            len(source.shape), "int32"
                        )
                        result_shape, result_shape_def = int_vector(output_shape)
                        result_rank, result_rank_def = constant(
                            len(output_shape), "int32"
                        )
                        prefix.extend((
                            input_shape_def,
                            input_rank_def,
                            result_shape_def,
                            result_rank_def,
                        ))
                        emitted.append(call(
                            "broadcast_double",
                            [
                                source, result, input_shape, input_rank,
                                result_shape, result_rank,
                            ],
                            result, instruction, output_argument=1,
                        ))
                    elif output_shape is not None and shape_unknown(source):
                        output_shape = tuple(map(int, output_shape))
                        result.shape = output_shape
                        source_extents = ensure_dynamic(prefix, source)
                        result_shape, result_shape_def = int_vector(output_shape)
                        result_rank, result_rank_def = constant(
                            len(output_shape), "int32"
                        )
                        prefix.extend((result_shape_def, result_rank_def))
                        emitted.append(call(
                            "broadcast_double",
                            [
                                source, result,
                                source_extents["shape"],
                                source_extents["rank"],
                                result_shape, result_rank,
                            ],
                            result, instruction, output_argument=1,
                        ))

                elif operation == "unfold2d" and source is not None:
                    kernel = _as_sequence(_attribute(
                        instruction.attributes, "kernel_size"
                    ))
                    stride = _as_sequence(_attribute(
                        instruction.attributes, "stride"
                    )) or (1, 1)
                    padding = _as_sequence(_attribute(
                        instruction.attributes, "padding"
                    )) or (0, 0)
                    dilation = _as_sequence(_attribute(
                        instruction.attributes, "dilation"
                    )) or (1, 1)
                    if (
                        len(tuple(source.shape or ())) == 4
                        and kernel is not None
                        and all(
                            len(pair) == 2
                            for pair in (kernel, stride, padding, dilation)
                        )
                    ):
                        dimensions = (
                            *map(int, source.shape), *map(int, kernel),
                            *map(int, stride), *map(int, padding),
                            *map(int, dilation),
                        )
                        dimension_values = []
                        for extent in dimensions:
                            value, definition = constant(extent, "int32")
                            dimension_values.append(value)
                            prefix.append(definition)
                        emitted.append(call(
                            "unfold2d_double",
                            [source, result, *dimension_values],
                            result, instruction, output_argument=1,
                        ))

                elif operation == "fold2d" and source is not None:
                    sequences = list(sequence_metadata)
                    output_size = _as_sequence(_attribute(
                        instruction.attributes, "output_size"
                    ))
                    kernel = _as_sequence(_attribute(
                        instruction.attributes, "kernel_size"
                    ))
                    stride = _as_sequence(_attribute(
                        instruction.attributes, "stride"
                    ))
                    padding = _as_sequence(_attribute(
                        instruction.attributes, "padding"
                    ))
                    dilation = _as_sequence(_attribute(
                        instruction.attributes, "dilation"
                    ))
                    if output_size is None and sequences:
                        output_size = sequences.pop(0)
                    resolved = []
                    for explicit, default in (
                        (kernel, None), (stride, (1, 1)),
                        (padding, (0, 0)), (dilation, (1, 1)),
                    ):
                        if explicit is not None:
                            resolved.append(tuple(explicit))
                        elif sequences:
                            resolved.append(tuple(sequences.pop(0)))
                        else:
                            resolved.append(default)
                    kernel, stride, padding, dilation = resolved
                    if (
                        output_size is not None
                        and len(output_size) == 4
                        and kernel is not None
                        and all(
                            pair is not None and len(pair) == 2
                            for pair in (kernel, stride, padding, dilation)
                        )
                    ):
                        dimensions = (
                            *map(int, output_size), *map(int, kernel),
                            *map(int, stride), *map(int, padding),
                            *map(int, dilation),
                        )
                        dimension_values = []
                        for extent in dimensions:
                            value, definition = constant(extent, "int32")
                            dimension_values.append(value)
                            prefix.append(definition)
                        emitted.append(call(
                            "fold2d_double",
                            [source, result, *dimension_values],
                            result, instruction, output_argument=1,
                        ))

                elif operation == "sign" and source is not None:
                    count = need_count(source, source_count)
                    emitted.append(call("sign_double", [source, result, count], result, instruction, output_argument=1))

                elif operation == "matmul" and len(data_args) == 2:
                    left, right = data_args
                    _record_ssa_shape(function, left)
                    _record_ssa_shape(function, right)
                    if (
                        len(left.shape) == len(right.shape) == 1
                        and left.shape[0] == right.shape[0]
                    ):
                        m, n, p = 1, int(left.shape[0]), 1
                        result.shape = ()
                        dimensions = []
                        for extent in (m, n, p):
                            value, definition = constant(extent, "int32")
                            dimensions.append(value)
                            prefix.append(definition)
                        emitted.append(call(
                            "matmul_double",
                            [left, right, result, *dimensions], result,
                            instruction, output_argument=2,
                        ))
                    elif (
                        len(left.shape) == 2
                        and len(right.shape) == 1
                        and left.shape[1] == right.shape[0]
                    ):
                        # NumPy/Python ``matrix @ vector`` is the existing
                        # rank-two kernel with a one-column right operand;
                        # the singleton result axis is semantic only and is
                        # squeezed from the public result. No vector copy or
                        # second physical object is needed.
                        m, n, p = (
                            int(left.shape[0]), int(left.shape[1]), 1,
                        )
                        result.shape = (m,)
                        dimensions = []
                        for extent in (m, n, p):
                            value, definition = constant(extent, "int32")
                            dimensions.append(value)
                            prefix.append(definition)
                        emitted.append(call(
                            "matmul_double",
                            [left, right, result, *dimensions], result,
                            instruction, output_argument=2,
                        ))
                    elif (
                        len(left.shape) == 1
                        and len(right.shape) == 2
                        and left.shape[0] == right.shape[0]
                    ):
                        m, n, p = (
                            1, int(left.shape[0]), int(right.shape[1]),
                        )
                        result.shape = (p,)
                        dimensions = []
                        for extent in (m, n, p):
                            value, definition = constant(extent, "int32")
                            dimensions.append(value)
                            prefix.append(definition)
                        emitted.append(call(
                            "matmul_double",
                            [left, right, result, *dimensions], result,
                            instruction, output_argument=2,
                        ))
                    elif (
                        len(left.shape) == len(right.shape) == 2
                        and left.shape[1] == right.shape[0]
                    ):
                        dimensions = []
                        for extent in (left.shape[0], left.shape[1], right.shape[1]):
                            value, definition = constant(int(extent), "int32")
                            dimensions.append(value)
                            prefix.append(definition)
                        emitted.append(call(
                            "matmul_double", [left, right, result, *dimensions], result,
                            instruction, output_argument=2,
                        ))
                    elif (
                        len(left.shape) >= 2
                        and len(right.shape) >= 2
                        and left.shape[-1] == right.shape[-2]
                    ):
                        left_batch = tuple(map(int, left.shape[:-2]))
                        right_batch = tuple(map(int, right.shape[:-2]))
                        batch_rank = max(len(left_batch), len(right_batch))
                        left_aligned = (1,) * (batch_rank - len(left_batch)) + left_batch
                        right_aligned = (1,) * (batch_rank - len(right_batch)) + right_batch
                        batch_shape: list[int] = []
                        compatible = True
                        for left_extent, right_extent in zip(
                            left_aligned, right_aligned
                        ):
                            if left_extent == right_extent or left_extent == 1:
                                batch_shape.append(right_extent)
                            elif right_extent == 1:
                                batch_shape.append(left_extent)
                            else:
                                compatible = False
                                break
                        if compatible:
                            m = int(left.shape[-2])
                            n = int(left.shape[-1])
                            p = int(right.shape[-1])
                            result.shape = (*batch_shape, m, p)

                            # The authored C kernel accepts one element offset
                            # per broadcasted batch. Build that finite routing
                            # table here from the static tensor descriptors;
                            # matrix values remain ordinary repository tensors.
                            left_matrix_size = m * n
                            right_matrix_size = n * p
                            left_offsets: list[int] = []
                            right_offsets: list[int] = []
                            batch_count = prod(batch_shape)
                            for flat_index in range(batch_count):
                                remaining = flat_index
                                coordinates = [0] * batch_rank
                                for axis in range(batch_rank - 1, -1, -1):
                                    extent = batch_shape[axis]
                                    coordinates[axis] = remaining % extent
                                    remaining //= extent

                                left_batch_index = 0
                                right_batch_index = 0
                                for coordinate, left_extent, right_extent in zip(
                                    coordinates, left_aligned, right_aligned
                                ):
                                    left_batch_index = (
                                        left_batch_index * left_extent
                                        + (0 if left_extent == 1 else coordinate)
                                    )
                                    right_batch_index = (
                                        right_batch_index * right_extent
                                        + (0 if right_extent == 1 else coordinate)
                                    )
                                left_offsets.append(
                                    left_batch_index * left_matrix_size
                                )
                                right_offsets.append(
                                    right_batch_index * right_matrix_size
                                )

                            left_offsets_value, left_offsets_def = int_vector(
                                left_offsets
                            )
                            right_offsets_value, right_offsets_def = int_vector(
                                right_offsets
                            )
                            prefix.extend((left_offsets_def, right_offsets_def))
                            dimensions = []
                            for extent in (batch_count, m, n, p):
                                value, definition = constant(extent, "int32")
                                dimensions.append(value)
                                prefix.append(definition)
                            emitted.append(call(
                                "batched_matmul_indexed_double",
                                [
                                    left, right, result,
                                    left_offsets_value, right_offsets_value,
                                    *dimensions,
                                ],
                                result, instruction, output_argument=2,
                            ))
                    elif shape_unknown(left) and shape_unknown(right):
                        # Symbolic operands: matmul_double's contract is rank
                        # two, so the three dims become runtime extents.
                        ensure_dynamic(prefix, left)
                        ensure_dynamic(prefix, right)
                        ensure_dynamic(prefix, result)
                        dimensions = [
                            dim_extent(prefix, left, 0),
                            dim_extent(prefix, left, 1),
                            dim_extent(prefix, right, 1),
                        ]
                        emitted.append(call(
                            "matmul_double", [left, right, result, *dimensions], result,
                            instruction, output_argument=2,
                        ))

                elif _int64_lane_stays_elementwise(
                    instruction, operation, result, data_args,
                    owner=function, book=book,
                ):
                    # Program: ``-tokens`` / ``abs(tokens)`` on an int64
                    # tensor.  Wrong record: the only repository unary kernel
                    # is ``unary_double``, so the lanes were cast to float64
                    # (2**53 + 1 came back 2**53, the result buffer float64)
                    # or, after an int64 producer, their bits were negated as
                    # a double's sign.  Rule: int64 Neg/Abs stay the
                    # instruction itself, which each backend's elementwise
                    # span loop spells on the integer path, wrapping at
                    # INT64_MIN exactly as the NumPy reference does.
                    if unary_geometry is not None:
                        post_unary_geometry(operation, source, result, unary_geometry)
                        if private_dynamic_result(result):
                            own_dynamic_tensor(result, unary_geometry[0][2], "int64", prefix)
                    rewritten.extend(prefix)
                    rewritten.append(instruction)
                    continue

                else:
                    opcode = c_tensor_opcode(operation)
                    if opcode is not None and source is not None:
                        kind, opcode_value = opcode
                        int64_binary = (
                            kind == "binary"
                            and _int64_binary_kernel(
                                operation, result, data_args,
                                owner=function, book=book,
                            )
                        )
                        binary_kernel = (
                            "binary_int64" if int64_binary
                            else "binary_double"
                        )
                        scalar_kernel = (
                            "binary_scalar_int64" if int64_binary
                            else "binary_scalar_double"
                        )
                        broadcast_kernel = (
                            "broadcast_int64" if int64_binary
                            else "broadcast_double"
                        )
                        kernel_dtype = "int64" if int64_binary else "float64"
                        count_value = source_count if kind == "unary" else result_count
                        count = need_count(source if kind == "unary" else result, count_value)
                        binary_output = result
                        if (
                            kind == "binary" and count is not None
                            and (int(result.id) in mixed_runtime_geometry or scalar_geometry is not None)
                            and private_dynamic_result(result)
                        ):
                            binary_output = own_dynamic_tensor(
                                result, count, kernel_dtype, prefix,
                            )
                        opcode_ssa, opcode_instruction = constant(opcode_value, "int32")
                        if count is not None:
                            prefix.append(opcode_instruction)
                            if kind == "unary":
                                unary_output = (
                                    own_dynamic_tensor(result, count, "float64", prefix)
                                    if unary_geometry is not None and private_dynamic_result(result)
                                    else result
                                )
                                emitted.append(call(
                                    "unary_double", [source, unary_output, count, opcode_ssa], result,
                                    instruction, output_argument=1,
                                ))
                                if unary_geometry is not None:
                                    selected = reference.module.functions["unary_double"]
                                    link_repository_function("unary_double", selected)
                                    # unary_double's pointer formals acquire
                                    # their exact pointee types from its Loads
                                    # and Stores. Publish that incumbent ABI
                                    # before this receipt cites the formals;
                                    # late inference otherwise revises them
                                    # after the receipt was closed.
                                    from .ssa_call_input_adapters import _infer_pointer_storage
                                    _infer_pointer_storage({selected.name: selected})
                                    from .ssa_record_return_state import ssa_value_identity_cell
                                    post_unary_geometry(operation, source, result, unary_geometry,
                                        tuple(ssa_value_identity_cell(owner, int(value.id), book=book)
                                              for owner, values in ((function, tuple(emitted[-1].args)),
                                                                    (selected, tuple(selected.args)))
                                              for value in values))
                            elif kind == "binary" and len(data_args) == 2 and scalar_geometry is not None and int64_binary and _int64_lane_checked(instruction, operation, result):
                                emitted.append(instruction)
                            elif kind == "binary" and len(data_args) == 2 and scalar_geometry is not None:
                                # A declared runtime span and scalar may both
                                # spell shape=(). The booked physical ranks,
                                # not that display spelling, select the
                                # existing scalar kernel. Passing the scalar
                                # to binary_double's pointer formal read an
                                # unallocated span in input_scalar_float_geometry.
                                array, scalar, _evidence = scalar_geometry
                                scalar_position = next(
                                    position for position, operand in enumerate(data_args)
                                    if int(operand.id) == int(scalar.id)
                                )
                                if not int64_binary and str(scalar.dtype).casefold() != "float64":
                                    promoted = fresh(shape=(), dtype="float64")
                                    prefix.append(Instr(
                                        "Cast", [scalar], promoted,
                                        attributes={
                                            "source_dtype": str(scalar.dtype),
                                            "target_dtype": "float64",
                                            "scalar_kernel_operand": True,
                                        },
                                    ))
                                    _post_scalar_kernel_input_conversion(
                                        book, function, str(block_name),
                                        int(result.id), int(scalar_position),
                                        scalar, promoted,
                                    )
                                    scalar = promoted
                                reverse, reverse_def = constant(
                                    1 if scalar_position == 0 else 0, "int32"
                                )
                                prefix.append(reverse_def)
                                selected = reference.module.functions.get(
                                    scalar_kernel
                                )
                                if selected is None:
                                    raise ValueError(
                                        "selected scalar kernel has no repository SSA definition"
                                    )
                                link_repository_function(scalar_kernel, selected)
                                emitted.append(call(
                                    scalar_kernel,
                                    [array, scalar, binary_output, count,
                                     opcode_ssa, reverse],
                                    result, instruction, output_argument=2,
                                ))
                                if int(scalar.id) != int(data_args[scalar_position].id):
                                    from .concordance_declarations import (
                                        KERNEL_INPUT_CONVERSION,
                                        TENSOR_SSA_LOWERING as _SCALAR_STAGE,
                                    )
                                    from .ssa_record_return_state import (
                                        posted_ssa_value_fact,
                                        revise_ssa_value,
                                        ssa_value_identity_cell,
                                    )
                                    source_scalar = data_args[scalar_position]
                                    source_scalar_fact = posted_ssa_value_fact(
                                        function, int(source_scalar.id), book=book,
                                    )
                                    selected_scalar = selected.args[1]
                                    selected_scalar_fact = posted_ssa_value_fact(
                                        selected, int(selected_scalar.id), book=book,
                                    )
                                    conversion_row = (
                                        function_scope_of(function),
                                        str(block_name), int(result.id),
                                        int(scalar_position),
                                    )
                                    conversion_cell = book.latest_ref(
                                        KERNEL_INPUT_CONVERSION,
                                        conversion_row,
                                    )
                                    source_scalar_cell = ssa_value_identity_cell(
                                        function, int(source_scalar.id), book=book,
                                    )
                                    selected_scalar_cell = ssa_value_identity_cell(
                                        selected, int(selected_scalar.id), book=book,
                                    )
                                    if (
                                        source_scalar_fact is None
                                        or source_scalar_fact.storage != "scalar"
                                        or source_scalar_fact.rank != 0
                                        or selected_scalar_fact is None
                                        or selected_scalar_fact.physical_dtype
                                        != kernel_dtype
                                        or any(cell is None for cell in (
                                            conversion_cell, source_scalar_cell,
                                            selected_scalar_cell,
                                        ))
                                    ):
                                        raise ValueError(
                                            "scalar kernel conversion lacks exact physical proof"
                                        )
                                    revise_ssa_value(
                                        function, int(scalar.id),
                                        dtype=scalar.dtype,
                                        shape=tuple(scalar.shape or ()),
                                        stage=_SCALAR_STAGE,
                                        sources=(source_scalar_cell,
                                                 selected_scalar_cell,
                                                 conversion_cell),
                                        book=book,
                                        physical={
                                            "physical_dtype": kernel_dtype,
                                            "storage": "scalar", "rank": 0,
                                        },
                                    )
                                # The scalar-operand receipt predates kernel
                                # selection. Concord the same fact from the
                                # exact opcode/reverse constants and selected
                                # native input/output formals as well; the
                                # extent receipt already cites this page.
                                from .concordance_declarations import SCALAR_KERNEL_OPERAND, TENSOR_SSA_LOWERING
                                from .identity_concordance import Derived, Mode
                                from .ssa_record_return_state import ssa_value_identity_cell
                                scalar_row = (str(function.name), int(result.id))
                                scalar_fact = ("scalar_operand", int(scalar_position))
                                call_cells = tuple(dict.fromkeys(
                                    ssa_value_identity_cell(function, int(value.id), book=book)
                                    for value in (array, scalar, binary_output,
                                                  count, opcode_ssa, reverse, result)
                                ))
                                formal_cells = tuple(
                                    ssa_value_identity_cell(
                                        selected, int(selected.args[position].id), book=book,
                                    ) for position in (0, 1, 2)
                                )
                                if any(cell is None for cell in (*call_cells, *formal_cells)):
                                    raise ValueError("selected scalar kernel lacks exact SSA cells")
                                book.post(
                                    SCALAR_KERNEL_OPERAND, scalar_row, scalar_fact,
                                    stage=TENSOR_SSA_LOWERING,
                                    provenance=Derived(tuple(dict.fromkeys(
                                        (*call_cells, *formal_cells)))),
                                    mode=Mode.CONCORD,
                                )
                                # The extent binding was posted when the
                                # source count became available, before the
                                # selected scalar Call and any scalar input
                                # conversion existed. Concord that same
                                # geometry now from the actual Call operands,
                                # selected formals, scalar role and conversion
                                # receipt. The audit reads this exact execution
                                # rather than accepting an early geometry
                                # claim without its executable witness.
                                from .concordance_declarations import (
                                    KERNEL_INPUT_CONVERSION,
                                    TENSOR_ELEMENTWISE_EXTENT_BINDING,
                                )
                                extent_row = (
                                    function_scope_of(function),
                                    function.name, int(result.id),
                                )
                                extent_ref = book.latest_ref(
                                    TENSOR_ELEMENTWISE_EXTENT_BINDING,
                                    extent_row,
                                )
                                scalar_ref = book.latest_ref(
                                    SCALAR_KERNEL_OPERAND, scalar_row,
                                )
                                conversion_ref = book.latest_ref(
                                    KERNEL_INPUT_CONVERSION,
                                    (function_scope_of(function),
                                     str(block_name), int(result.id),
                                     int(scalar_position)),
                                )
                                if extent_ref is not None:
                                    old_causes = tuple(
                                        cause for cause, _stage
                                        in book.edges_into(extent_ref)
                                    )
                                    book.post(
                                        TENSOR_ELEMENTWISE_EXTENT_BINDING,
                                        extent_row,
                                        book.page(
                                            TENSOR_ELEMENTWISE_EXTENT_BINDING
                                        ).cell(extent_row, extent_ref.column),
                                        stage=TENSOR_SSA_LOWERING,
                                        provenance=Derived(tuple(dict.fromkeys((
                                            extent_ref, *old_causes, scalar_ref,
                                            *((conversion_ref,)
                                              if conversion_ref is not None
                                              else ()),
                                            *call_cells, *formal_cells,
                                        )))),
                                        mode=Mode.CONCORD,
                                    )
                            elif kind == "binary" and len(data_args) == 2:
                                left_constant = constants.get(
                                    int(data_args[0].id)
                                )
                                if (
                                    int(data_args[0].id) in constants
                                    and not tuple(data_args[0].shape)
                                    and tuple(data_args[1].shape)
                                    and isinstance(left_constant, (int, float))
                                    and not isinstance(left_constant, bool)
                                ):
                                    # AST arithmetic retains a literal in
                                    # operand zero (``1 + tensor``).  The
                                    # repository tensor ABI is double-based;
                                    # broadcasting the original i32 storage
                                    # through ``broadcast_double`` reads four
                                    # uninitialised bytes and corrupts the
                                    # result.  Materialize the numerical
                                    # scalar at the ABI dtype and use the
                                    # existing scalar kernel, exactly as the
                                    # right-literal path below already does.
                                    scalar, scalar_def = constant(
                                        (
                                            int(left_constant) if int64_binary
                                            else float(left_constant)
                                        ),
                                        kernel_dtype,
                                    )
                                    reverse, reverse_def = constant(1, "int32")
                                    prefix.extend((scalar_def, reverse_def))
                                    emitted.append(call(
                                        scalar_kernel,
                                        [
                                            data_args[1], scalar, binary_output,
                                            count, opcode_ssa, reverse,
                                        ],
                                        result,
                                        instruction,
                                        output_argument=2,
                                    ))
                                elif any(
                                    not tuple(operand.shape or ())
                                    and int((operand.accounting or {}).get(
                                        "program_abi_rank", 0
                                    ) or 0) == 0
                                    and not shape_unknown(operand)
                                    for operand in data_args
                                ) and any(
                                    tuple(operand.shape or ())
                                    == tuple(result.shape or ())
                                    and tuple(result.shape or ())
                                    for operand in data_args
                                ):
                                    # One exact scalar operand against an
                                    # array (``x > limit``): the scalar crosses
                                    # by value into the scalar kernel, exactly
                                    # as a literal does above.  Broadcasting
                                    # it through ``broadcast_double`` passed a
                                    # rank-zero buffer and made the caller's
                                    # scalar a span of unknown length.
                                    scalar_position = next(
                                        position
                                        for position, operand in enumerate(data_args)
                                        if not tuple(operand.shape or ())
                                        and not shape_unknown(operand)
                                    )
                                    scalar = data_args[scalar_position]
                                    _post_scalar_kernel_operand(
                                        module, function,
                                        (str(function_name), int(result.id)),
                                        ("scalar_operand", int(scalar_position)),
                                        (result, scalar),
                                    )
                                    array = data_args[1 - scalar_position]
                                    if (
                                        not int64_binary
                                        and str(scalar.dtype or "").casefold()
                                        != "float64"
                                    ):
                                        promoted = fresh(shape=(), dtype="float64")
                                        prefix.append(Instr(
                                            "Cast", [scalar], promoted,
                                            attributes={
                                                "source_dtype": str(scalar.dtype),
                                                "target_dtype": "float64",
                                                "scalar_kernel_operand": True,
                                            },
                                        ))
                                        # The promotion is a conversion at THIS
                                        # kernel's input (Python keeps an int
                                        # an int; the comparison promotes it):
                                        # a ``kernel_input_conversion`` row
                                        # DERIVED from the scalar's
                                        # ``ssa_value`` cell.
                                        _post_scalar_kernel_input_conversion(
                                            book, function, str(block_name),
                                            int(result.id),
                                            int(scalar_position), scalar,
                                            promoted,
                                        )
                                        scalar = promoted
                                    reverse, reverse_def = constant(
                                        1 if scalar_position == 0 else 0, "int32"
                                    )
                                    prefix.append(reverse_def)
                                    emitted.append(call(
                                        scalar_kernel,
                                        [array, scalar, binary_output, count, opcode_ssa, reverse],
                                        result,
                                        instruction,
                                        output_argument=2,
                                    ))
                                else:
                                    conformed_args = []
                                    for operand_position, operand in enumerate(data_args):
                                        operand_physical = posted_ssa_value_fact(
                                            function, int(operand.id), book=book,
                                        )
                                        operand_descriptor = tensor_table.by_id(
                                            int(operand.id)
                                        )
                                        if (
                                            tuple(operand.shape) == tuple(result.shape)
                                            and int(result.id)
                                            not in mixed_runtime_geometry
                                            and (
                                                int64_binary
                                                or operand_descriptor is not None
                                                and operand_descriptor.metadata_state
                                                == "static"
                                                or operand_physical is not None
                                                and operand_physical.physical_dtype
                                                == kernel_dtype
                                            )
                                        ):
                                            conformed_args.append(operand)
                                            continue
                                        broadcast_source = operand
                                        if (
                                            not int64_binary
                                            and operand_physical is not None
                                            and operand_physical.storage == "span"
                                            and operand_physical.rank is not None
                                            and operand_physical.rank > 0
                                            and operand_physical.physical_dtype == "int64"
                                            and operand_descriptor is not None
                                            and operand_descriptor.metadata_state
                                            == "dynamic"
                                        ):
                                            # The no-explicit-cast weighted
                                            # Mul has source16 int64/span/rank2
                                            # with a runtime shape spelled ().
                                            # A scalar Cast converted one limb
                                            # and left broadcast_double reading
                                            # unproved storage. Use the selected
                                            # counted span kernel and its typed
                                            # output formal for every limb.
                                            from .ssa_call_input_adapters import (
                                                _infer_pointer_storage,
                                            )
                                            from .ssa_record_return_state import (
                                                revise_ssa_value as _revise_span_conversion,
                                                ssa_value_identity_cell as _span_cell,
                                            )
                                            from .concordance_declarations import (
                                                TENSOR_SSA_LOWERING as _SPAN_CAST_STAGE,
                                            )
                                            selected_cast = reference.module.functions.get(
                                                "cast_int64_to_double_values"
                                            )
                                            if (selected_cast is None
                                                    or selected_cast.metadata.get(
                                                        "ssa_output_argument") != 1
                                                    or len(selected_cast.args) != 3):
                                                raise ValueError(
                                                    "span promotion lacks selected "
                                                    "counted cast output"
                                                )
                                            link_repository_function(
                                                selected_cast.name, selected_cast,
                                            )
                                            _infer_pointer_storage({
                                                selected_cast.name: selected_cast,
                                            })
                                            formal_cell = _span_cell(
                                                selected_cast,
                                                int(selected_cast.args[1].id),
                                                book=book,
                                            )
                                            formal_fact = posted_ssa_value_fact(
                                                selected_cast,
                                                int(selected_cast.args[1].id),
                                                book=book,
                                            )
                                            if (formal_cell is None
                                                    or formal_fact is None
                                                    or formal_fact.physical_dtype
                                                    != "float64"):
                                                raise ValueError(
                                                    "span promotion lacks typed "
                                                    "selected output formal"
                                                )
                                            source_descriptor = operand_descriptor
                                            if (source_descriptor is None
                                                    or source_descriptor.metadata_state
                                                    != "dynamic"
                                                    or any(value is None for value in (
                                                        source_descriptor.shape_value_id,
                                                        source_descriptor.rank_value_id,
                                                        source_descriptor.element_count_value_id,
                                                    ))):
                                                raise ValueError(
                                                    "span promotion lacks source "
                                                    "runtime descriptor"
                                                )
                                            source_extents = ensure_dynamic(
                                                prefix, operand,
                                            )
                                            source_count = source_extents[
                                                "element_count"
                                            ]
                                            broadcast_source = fresh(
                                                shape=operand.shape,
                                                dtype="float64",
                                            )
                                            _revise_span_conversion(
                                                function, int(broadcast_source.id),
                                                dtype="float64",
                                                shape=tuple(operand.shape),
                                                stage=_SPAN_CAST_STAGE,
                                                sources=(formal_cell,
                                                         _span_cell(
                                                             function,
                                                             int(operand.id),
                                                             book=book,
                                                         )),
                                                book=book,
                                                physical={
                                                    "physical_dtype": "float64",
                                                    "storage": "span",
                                                    "rank": int(
                                                        operand_physical.rank
                                                    ),
                                                },
                                            )
                                            broadcast_source.accounting = {
                                                **dict(
                                                    broadcast_source.accounting
                                                    or {}
                                                ),
                                                "physical_dtype": "float64",
                                                "program_abi_storage": "span",
                                                "program_abi_rank": int(
                                                    operand_physical.rank
                                                ),
                                                "tensor_metadata_state": "dynamic",
                                                "tensor_shape_value_id": int(
                                                    source_descriptor.shape_value_id
                                                ),
                                                "tensor_rank_value_id": int(
                                                    source_descriptor.rank_value_id
                                                ),
                                                "tensor_element_count_value_id": int(
                                                    source_descriptor.element_count_value_id
                                                ),
                                            }
                                            register_tensor(
                                                broadcast_source,
                                                storage="temporary",
                                                metadata_state="dynamic",
                                                extent_ids_from=source_descriptor,
                                            )
                                            span_storage = own_dynamic_tensor(
                                                broadcast_source, source_count,
                                                "float64", prefix,
                                            )
                                            prefix.append(call(
                                                selected_cast.name,
                                                [operand, span_storage,
                                                 source_count],
                                                broadcast_source, instruction,
                                                output_argument=1,
                                            ))
                                            _post_scalar_kernel_input_conversion(
                                                book, function, str(block_name),
                                                int(result.id),
                                                int(operand_position), operand,
                                                broadcast_source,
                                                callee="broadcast_double",
                                            )
                                        if (
                                            not int64_binary
                                            and operand_physical is not None
                                            and operand_physical.storage == "scalar"
                                            and operand_physical.rank == 0
                                            and str(operand.dtype or "").casefold()
                                            in {
                                                "bool", "i1", "int", "int32",
                                                "int64", "long", "float32",
                                            }
                                        ):
                                            # `broadcast_double` consumes a
                                            # DOUBLE buffer.  A scalar loop
                                            # index is a numerical value, not
                                            # a pointer whose integer bytes
                                            # may be reinterpreted as a
                                            # double.  Materialize the exact
                                            # promotion at this repository
                                            # boundary and publish that
                                            # transformation to the shared
                                            # concordance.
                                            broadcast_source = fresh(
                                                shape=(), dtype="float64",
                                            )
                                            broadcast_source.accounting = {
                                                **dict(
                                                    broadcast_source.accounting
                                                    or {}
                                                ),
                                                "kernel_input_conversion": (
                                                    function_name,
                                                    int(operand.id),
                                                    "broadcast_double",
                                                    int(operand_position),
                                                ),
                                            }
                                            prefix.append(Instr(
                                                "Cast", [operand],
                                                broadcast_source,
                                                attributes={
                                                    "source_dtype": str(
                                                        operand.dtype
                                                    ),
                                                    "target_dtype": "float64",
                                                    "concordant_kernel_input_conversion": True,
                                                },
                                            ))
                                            conversion = {
                                                "source_id": int(operand.id),
                                                "converted_id": int(
                                                    broadcast_source.id
                                                ),
                                                "source_dtype": str(
                                                    operand.dtype
                                                ),
                                                "target_dtype": "float64",
                                                "callee": "broadcast_double",
                                                "operand_position": int(
                                                    operand_position
                                                ),
                                                "block": str(block_name),
                                                "consumer_id": int(result.id),
                                            }
                                            function.metadata[
                                                "kernel_input_conversions"
                                            ] = (
                                                *tuple(function.metadata.get(
                                                    "kernel_input_conversions",
                                                    (),
                                                )),
                                                conversion,
                                            )
                                            # The conversion at THIS
                                            # kernel's input, DERIVED from
                                            # the operand's ``ssa_value``
                                            # cell and the converted
                                            # value's.
                                            _post_scalar_kernel_input_conversion(
                                                identity_book(module),
                                                function, str(block_name),
                                                int(result.id),
                                                int(operand_position),
                                                operand, broadcast_source,
                                                callee="broadcast_double",
                                            )
                                        if (
                                            shape_unknown(operand)
                                            or shape_unknown(result)
                                            or int(result.id)
                                            in mixed_runtime_geometry
                                        ):
                                            # Shapes unknown statically: conforming
                                            # is a runtime decision, so route the
                                            # operand through broadcast_double with
                                            # extent metadata -- never a silent
                                            # mismatched elementwise call.
                                            operand_extents = ensure_dynamic(prefix, operand)
                                            # The result is defined by the call
                                            # being lowered, so ``extent(result)``
                                            # here would read it before it
                                            # exists (refused: use-not-
                                            # dominated).  An elementwise result
                                            # has its operands' broadcast shape:
                                            # when another operand already has
                                            # the result's exact static shape,
                                            # its extents ARE the result's, and
                                            # they are defined before this call.
                                            result_extents = dynamic_extents.get(int(result.id))
                                            if result_extents is None and not shape_unknown(result):
                                                peer = next((
                                                    candidate for candidate in data_args
                                                    if candidate is not operand
                                                    and not shape_unknown(candidate)
                                                    and tuple(candidate.shape or ()) == tuple(result.shape or ())
                                                ), None)
                                                if peer is not None:
                                                    result_extents = ensure_dynamic(prefix, peer)
                                                    dynamic_extents[int(result.id)] = result_extents
                                            if result_extents is None:
                                                result_extents = ensure_dynamic(prefix, result)
                                            # A conformance buffer has the
                                            # result's proven span/rank and the
                                            # selected repository kernel's
                                            # element type. The older mixed-only
                                            # gate left this Call output with
                                            # no physical storage even when the
                                            # result's scoped SSA_VALUE already
                                            # proved it (compact_codewords Mul98).
                                            conformed_result_fact = posted_ssa_value_fact(
                                                function, int(result.id), book=book,
                                            )
                                            conformed_physical = (
                                                {"physical_dtype": kernel_dtype,
                                                 "storage": "span",
                                                 "rank": int(conformed_result_fact.rank)}
                                                if conformed_result_fact is not None
                                                and conformed_result_fact.storage == "span"
                                                and conformed_result_fact.rank is not None
                                                and conformed_result_fact.rank > 0
                                                else None
                                            )
                                            broadcasted = fresh(
                                                shape=result.shape,
                                                dtype=kernel_dtype,
                                                physical=conformed_physical,
                                            )
                                            if int(result.id) in mixed_runtime_geometry:
                                                broadcasted.accounting = {
                                                    **dict(broadcasted.accounting or {}),
                                                    "tensor_metadata_state": "dynamic",
                                                    "tensor_shape_value_id": int(
                                                        result_descriptor.shape_value_id
                                                    ),
                                                    "tensor_rank_value_id": int(
                                                        result_descriptor.rank_value_id
                                                    ),
                                                    "tensor_element_count_value_id": int(
                                                        result_descriptor.element_count_value_id
                                                    ),
                                                }
                                            broadcast_output = (
                                                own_dynamic_tensor(
                                                    broadcasted, count,
                                                    kernel_dtype, prefix,
                                                )
                                                if int(result.id)
                                                in mixed_runtime_geometry
                                                else broadcasted
                                            )
                                            emitted.append(call(
                                                broadcast_kernel,
                                                [
                                                    broadcast_source,
                                                    broadcast_output,
                                                    operand_extents["shape"],
                                                    operand_extents["rank"],
                                                    result_extents["shape"],
                                                    result_extents["rank"],
                                                ],
                                                broadcasted,
                                                instruction,
                                                output_argument=1,
                                            ))
                                            register_tensor(
                                                broadcasted, storage="temporary",
                                                metadata_state="dynamic",
                                                extent_ids_from=tensor_table.by_id(int(result.id)),
                                            )
                                            conformed_args.append(broadcasted)
                                            continue
                                        broadcasted = fresh(
                                            shape=result.shape,
                                            dtype=kernel_dtype,
                                        )
                                        input_shape, input_shape_def = int_vector(
                                            operand.shape
                                        )
                                        input_rank, input_rank_def = constant(
                                            len(operand.shape), "int32"
                                        )
                                        output_shape, output_shape_def = int_vector(
                                            result.shape
                                        )
                                        output_rank, output_rank_def = constant(
                                            len(result.shape), "int32"
                                        )
                                        prefix.extend((
                                            input_shape_def,
                                            input_rank_def,
                                            output_shape_def,
                                            output_rank_def,
                                        ))
                                        emitted.append(call(
                                            broadcast_kernel,
                                            [
                                                broadcast_source,
                                                broadcasted,
                                                input_shape,
                                                input_rank,
                                                output_shape,
                                                output_rank,
                                            ],
                                            broadcasted,
                                            instruction,
                                            output_argument=1,
                                        ))
                                        register_tensor(
                                            broadcasted, storage="temporary"
                                        )
                                        conformed_args.append(broadcasted)
                                    if int64_binary and _int64_lane_checked(
                                        instruction, operation, result,
                                    ):
                                        # The operands are conformed by the
                                        # int64 broadcast kernel; the
                                        # arithmetic stays the instruction, so
                                        # the backend's span loop checks each
                                        # lane (``binary_int64`` would wrap).
                                        emitted.append(dataclasses.replace(
                                            instruction,
                                            args=list(conformed_args),
                                        ))
                                    else:
                                        emitted.append(call(
                                            binary_kernel, [conformed_args[0], conformed_args[1], binary_output, count, opcode_ssa],
                                            result, instruction, output_argument=2,
                                        ))
                            elif kind == "binary" and len(data_args) == 1:
                                scalar_key = next((key for key in ("right_scalar", "left_scalar") if key in instruction.attributes), None)
                                scalar_payload = instruction.attributes.get(scalar_key) if scalar_key else (metadata[-1] if metadata else None)
                                if scalar_payload is not None and not isinstance(scalar_payload, (tuple, list)):
                                    scalar, scalar_def = constant(
                                        (
                                            int(scalar_payload) if int64_binary
                                            else float(scalar_payload)
                                        ),
                                        kernel_dtype,
                                    )
                                    reverse, reverse_def = constant(int(scalar_key == "left_scalar"), "int32")
                                    prefix.extend((scalar_def, reverse_def))
                                    emitted.append(call(
                                        scalar_kernel, [source, scalar, result, count, opcode_ssa, reverse],
                                        result, instruction, output_argument=2,
                                    ))

                if emitted:
                    rewritten.extend(prefix)
                    rewritten.extend(emitted)
                    continue

                recognized = (
                    operation in _VIEW_OPERATIONS
                    or operation in _CAST_OPERATIONS
                    or operation == "cbrt"
                    or reference.operation(operation) is not None
                )
                if recognized and not candidate_only:
                    reason = (
                        "tensor extent/shape or structural operands are not available in SSA"
                        if source is not None and booked_element_count(source) is None
                        else "referenced SSA exists but its concrete call operands cannot be derived from this instruction"
                    )
                    reason += (
                        "; argument_shapes="
                        f"{tuple(tuple(argument.shape or ()) for argument in args)!r}, "
                        f"result=%{result.id} shape={tuple(result.shape or ())!r}"
                    )
                    shortfalls.append(TensorSSALoweringShortfall(
                        function_name, block_name, operation, reason,
                    ))
                rewritten.append(instruction)
            block.instrs[:] = rewritten

        for (
            guarded_block, guard, source, width, width32, result,
            source_count, safe_width, remainder, quotient,
            shape_value, rank, invalid,
        ) in pending_runtime_view_guards:
            from ..transmogrifier.ssa import BasicBlock
            from .concordance_declarations import (
                SSA_BLOCK, SSA_CONTROL_EDGE, SSABlockKind,
                TENSOR_RUNTIME_RESHAPE_VALIDITY, TENSOR_SSA_LOWERING,
            )
            from .identity_concordance import Derived, Mode
            from .ssa_record_return_state import (
                function_scope_of, ssa_value_identity_cell,
            )

            position = guarded_block.instrs.index(guard)
            continuation = guarded_block.instrs[position + 1:]
            old_successors = list(guarded_block.successors)
            fault_name = guarded_block.name + ".reshape_fault"
            pass_name = guarded_block.name + ".reshape_checked"
            if (
                fault_name in function.blocks or pass_name in function.blocks
                or not continuation or continuation[-1].op != "Ret"
            ):
                raise ValueError(
                    "runtime reshape has no single-block validation continuation"
                )
            guard.attributes = {
                **dict(guard.attributes),
                "true_target": fault_name,
                "false_target": pass_name,
            }
            guarded_block.instrs[:] = guarded_block.instrs[:position + 1]
            guarded_block.successors = [fault_name, pass_name]
            fault = BasicBlock(fault_name, [
                Instr("Call", [], None, attributes={
                    "callee": "turing_validation_error",
                    "error_code": int(guard.attributes["runtime_reshape_error"]),
                    "failure": "reshape invalid shape",
                }),
                Instr("Trap", [], None, attributes={"fault_exit": True}),
            ], [])
            passing = BasicBlock(pass_name, continuation, old_successors)
            function.blocks[fault_name] = fault
            function.blocks[pass_name] = passing
            scope = function_scope_of(function)
            causes = tuple(dict.fromkeys(
                cell for value in (
                    source, width, width32, result, source_count,
                    safe_width, remainder, quotient, shape_value,
                    rank, invalid,
                )
                if (cell := ssa_value_identity_cell(
                    function, int(value.id), book=book,
                )) is not None
            ))
            for name, kind in (
                (fault_name, SSABlockKind.VALIDATION_FAIL),
                (pass_name, SSABlockKind.VALIDATION_PASS),
            ):
                row = (scope, function.name, name)
                book.post(
                    SSA_BLOCK, row, kind, stage=TENSOR_SSA_LOWERING,
                    provenance=Derived(causes), mode=Mode.CONCORD,
                )
            before = book.latest_ref(
                SSA_BLOCK, (scope, function.name, guarded_block.name),
            )
            fail_cell = book.latest_ref(
                SSA_BLOCK, (scope, function.name, fault_name),
            )
            pass_cell = book.latest_ref(
                SSA_BLOCK, (scope, function.name, pass_name),
            )
            if before is None:
                raise ValueError("runtime reshape lost its source block cell")
            for first, second, arm in (
                (before, fail_cell, "body"),
                (before, pass_cell, "orelse"),
            ):
                book.post(
                    SSA_CONTROL_EDGE,
                    (scope, function.name, first, second, arm), True,
                    stage=TENSOR_SSA_LOWERING,
                    provenance=Derived((first, second, *causes)),
                    mode=Mode.CONCORD,
                )
            edge_cells = tuple(
                book.latest_ref(SSA_CONTROL_EDGE,
                                (scope, function.name, first, second, arm))
                for first, second, arm in (
                    (before, fail_cell, "body"),
                    (before, pass_cell, "orelse"),
                )
            )
            reshape_row = (scope, function.name, int(result.id))
            book.post(
                TENSOR_RUNTIME_RESHAPE_VALIDITY, reshape_row,
                (int(source.id), int(width.id), int(width32.id),
                 int(source_count.id), int(safe_width.id),
                 int(remainder.id), int(quotient.id),
                 int(shape_value.id), int(rank.id), int(invalid.id),
                 fault_name, pass_name),
                stage=TENSOR_SSA_LOWERING,
                provenance=Derived((*causes, before, fail_cell, pass_cell,
                                    *edge_cells)),
                mode=Mode.CONCORD,
            )
            # weighted_cast_source_probe's checked reshape %16 aliases the
            # int64 ProgramABI span %0. Its rank-two shape was proved here,
            # but SSA_VALUE %16 retained physical_dtype=None, so the next
            # typed span cast refused. The checked view receipt and source's
            # posted physical span prove the same element storage for %16.
            from .concordance_declarations import SSA_VALUE
            from .ssa_record_return_state import (
                posted_ssa_value_fact, revise_ssa_value,
            )
            source_fact = posted_ssa_value_fact(
                function, int(source.id), book=book,
            )
            result_fact = posted_ssa_value_fact(
                function, int(result.id), book=book,
            )
            descriptor = tensor_table.by_id(int(result.id))
            view_ref = book.latest_ref(
                TENSOR_RUNTIME_RESHAPE_VALIDITY, reshape_row,
            )
            if source_fact is not None and source_fact.physical_dtype is not None:
                source_cell = ssa_value_identity_cell(
                    function, int(source.id), book=book,
                )
                geometry_cells = tuple(ssa_value_identity_cell(
                    function, int(value.id), book=book,
                ) for value in (source_count, shape_value, rank))
                if (
                    source_fact.storage != "span"
                    or source_fact.rank is None or source_fact.rank < 1
                    or result_fact is None
                    or result_fact.storage not in (None, "span")
                    or result_fact.rank not in (None, 2)
                    or result_fact.physical_dtype not in (
                        None, source_fact.physical_dtype,
                    )
                    or str(result.dtype) != str(source.dtype)
                    or descriptor is None
                    or descriptor.storage != "view"
                    or descriptor.alias_of != int(source.id)
                    or descriptor.data_value_id != int(source.id)
                    or descriptor.metadata_state != "dynamic"
                    or (descriptor.shape_value_id,
                        descriptor.rank_value_id,
                        descriptor.element_count_value_id)
                    != (int(shape_value.id), int(rank.id),
                        int(source_count.id))
                    or source_cell is None or view_ref is None
                    or any(cell is None for cell in geometry_cells)
                ):
                    raise ValueError(
                        "runtime reshape physical view lacks its exact source"
                    )
                view_causes = (
                    source_cell, *geometry_cells, view_ref,
                )
                revised = revise_ssa_value(
                    function, int(result.id), dtype=result.dtype,
                    shape=tuple(result.shape or ()),
                    stage=TENSOR_SSA_LOWERING,
                    sources=view_causes, book=book,
                    physical={
                        "physical_dtype": source_fact.physical_dtype,
                        "storage": "span", "rank": 2,
                    },
                )
                if revised is None:
                    raise ValueError(
                        "runtime reshape physical view lacks result identity"
                    )
                # If the source had already given the view this same
                # physical fact, attach the checked receipt without a fake
                # unchanged revision or an SSA_VALUE self edge.
                book.post(
                    SSA_VALUE, (scope, int(result.id)),
                    posted_ssa_value_fact(function, int(result.id), book=book),
                    stage=TENSOR_SSA_LOWERING,
                    provenance=Derived(view_causes), mode=Mode.CONCORD,
                )

        for (
            guarded_block, guard, source_instruction, ordered_operands,
            wide, narrow, wide_metadata, narrow_metadata, aligned,
            output_shape, rank, count, status, invalid, helper,
            imported_cells, binding_cells, program_values,
        ) in pending_runtime_broadcast_guards:
            if guard not in guarded_block.instrs:
                # The original binary remains visible with a named shortfall.
                continue
            from ..transmogrifier.ssa import BasicBlock
            from .concordance_declarations import (
                SSA_BLOCK, SSA_CONTROL_EDGE, SSABlockKind,
                TENSOR_RUNTIME_BROADCAST_VALIDITY, TENSOR_SSA_LOWERING,
            )
            from .identity_concordance import Derived, Mode
            from .ssa_record_return_state import (
                function_scope_of, ssa_value_identity_cell,
            )

            position = guarded_block.instrs.index(guard)
            continuation = guarded_block.instrs[position + 1:]
            old_successors = list(guarded_block.successors)
            fault_name = guarded_block.name + ".broadcast_fault"
            pass_name = guarded_block.name + ".broadcast_checked"
            if (
                fault_name in function.blocks or pass_name in function.blocks
                or not continuation or continuation[-1].op != "Ret"
                or old_successors
            ):
                raise ValueError(
                    "runtime broadcast has no single-block validation continuation"
                )
            guard.attributes = {
                **dict(guard.attributes),
                "true_target": fault_name,
                "false_target": pass_name,
            }
            guarded_block.instrs[:] = guarded_block.instrs[:position + 1]
            guarded_block.successors = [fault_name, pass_name]
            fault = BasicBlock(fault_name, [
                Instr("Call", [], None, attributes={
                    "callee": "turing_validation_error",
                    "error_code": int(guard.attributes["runtime_broadcast_error"]),
                    "failure": "incompatible broadcast shapes",
                }),
                Instr("Trap", [], None, attributes={"fault_exit": True}),
            ], [])
            passing = BasicBlock(pass_name, continuation, old_successors)
            function.blocks[fault_name] = fault
            function.blocks[pass_name] = passing
            scope = function_scope_of(function)
            values = (
                *ordered_operands, wide, narrow,
                *wide_metadata, *narrow_metadata,
                aligned, output_shape, rank, count, status, invalid,
                *program_values,
                source_instruction.res,
            )
            value_cells = tuple(
                ssa_value_identity_cell(
                    function, int(value.id), book=book,
                ) for value in values if value is not None
            )
            if (any(cell is None for cell in value_cells)
                    or any(cell is None for cell in imported_cells)):
                raise ValueError("runtime broadcast lost its exact SSA cause")
            causes = tuple(dict.fromkeys((
                *value_cells, *binding_cells, *imported_cells,
            )))
            for name, kind in (
                (fault_name, SSABlockKind.VALIDATION_FAIL),
                (pass_name, SSABlockKind.VALIDATION_PASS),
            ):
                book.post(
                    SSA_BLOCK, (scope, function.name, name), kind,
                    stage=TENSOR_SSA_LOWERING,
                    provenance=Derived(causes), mode=Mode.CONCORD,
                )
            before = book.latest_ref(
                SSA_BLOCK, (scope, function.name, guarded_block.name),
            )
            fail_cell = book.latest_ref(
                SSA_BLOCK, (scope, function.name, fault_name),
            )
            pass_cell = book.latest_ref(
                SSA_BLOCK, (scope, function.name, pass_name),
            )
            if before is None:
                raise ValueError("runtime broadcast lost its source block cell")
            edge_cells = []
            for first, second, arm in (
                (before, fail_cell, "body"),
                (before, pass_cell, "orelse"),
            ):
                book.post(
                    SSA_CONTROL_EDGE,
                    (scope, function.name, first, second, arm), True,
                    stage=TENSOR_SSA_LOWERING,
                    provenance=Derived((first, second, *causes)),
                    mode=Mode.CONCORD,
                )
                edge_cells.append(book.latest_ref(
                    SSA_CONTROL_EDGE,
                    (scope, function.name, first, second, arm),
                ))
            book.post(
                TENSOR_RUNTIME_BROADCAST_VALIDITY,
                (scope, function.name, int(source_instruction.res.id)),
                (
                    str(_SHAPED_SSA_OPERATIONS.get(
                        source_instruction.op, source_instruction.op,
                    )),
                    int(ordered_operands[0].id), int(ordered_operands[1].id),
                    int(wide.id), int(narrow.id),
                    *(int(value.id) for value in wide_metadata),
                    *(int(value.id) for value in narrow_metadata),
                    int(aligned.id), int(output_shape.id), int(rank.id),
                    int(count.id), int(status.id), str(helper.name),
                    fault_name, pass_name,
                ),
                stage=TENSOR_SSA_LOWERING,
                provenance=Derived((*causes, before, fail_cell, pass_cell,
                                    *edge_cells)),
                mode=Mode.CONCORD,
            )

        # Slice objects are compile-time selector syntax, never runtime scalar
        # constants.  Once every consumer has normalized away, discard their
        # now-dead definitions so a scalar backend cannot be asked to emit
        # ``float(slice(...))``.  Shared selectors remain until their final
        # semantic consumer is lowered.
        used_value_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        for block in function.blocks.values():
            block.instrs[:] = [
                instruction
                for instruction in block.instrs
                if not (
                    instruction.op in {"Const", "const"}
                    and instruction.res is not None
                    and isinstance(
                        constants.get(int(instruction.res.id)), slice
                    )
                    and int(instruction.res.id) not in used_value_ids
                )
            ]

        if aliases:
            for block in function.blocks.values():
                for index, instruction in enumerate(block.instrs):
                    replacement_args = [resolve(argument) for argument in instruction.args]
                    if replacement_args != instruction.args:
                        block.instrs[index] = dataclasses.replace(instruction, args=replacement_args)

        returned_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {Handler.Ret.value, "ret", "Return", "return"}
            for argument in instruction.args
        }
        for tensor_id in returned_ids:
            descriptor = tensor_table.by_id(tensor_id)
            if descriptor is not None:
                tensor_table.tensors[tensor_id] = dataclasses.replace(
                    descriptor, storage="output", writable=True
                )

    for name, function in reference.dependency_closure(*sorted(linked_roots)).items():
        # A repository kernel enters the module here, and the metadata
        # settlement below reads and writes about its values at once: its
        # definition root and per-value ``ssa_value`` rows are posted on
        # this book as it is linked (the same writer ``ir_indexing`` runs
        # before its own decisions; posting twice reads the existing rows).
        link_repository_function(name, function)
    publish_computed_region_result_metadata(
        module, finalize_private_owners=finalize_private_owners,
    )
    if legalize_aggregate_adapters(module):
        settle_canonical_value_metadata(module)
        settle_shape_preserving_value_metadata(module)
    # Tensor descriptors are created while rewriting functions above.  A
    # producer's final allocation shape can therefore become authoritative
    # only after the whole module has been visited, regardless of whether an
    # aggregate adapter happened to be removed on this invocation.
    settle_repository_ssa_shape_metadata(module)
    settle_repository_ssa_static_extent_operands(module)
    legalize_aggregate_output_views(module)
    return tuple(shortfalls)


def complete_tensor_call_closure(
    module: IRModule,
    reference: SSATensorCodeReference,
    *,
    finalize_private_owners: bool = False,
    initial_all: bool = False,
) -> tuple[TensorSSALoweringShortfall, ...]:
    """Settle the incumbent affected Call closure before reporting shortfalls.

    A computed region can first register unresolved tensor geometry, then
    publish its shape/rank/count through a later Call metadata edge. Revisit
    only functions whose exact descriptor identities changed and their live
    callers. This is the same closure used by whole-program late lowering;
    local precompile must finish it before shortfalls become terminal.
    """

    from .ssa_record_return_state import posted_ssa_value_fact
    from .authored_broadcast_reference import (
        retained_authored_broadcast_reference,
    )

    reference = retained_authored_broadcast_reference(reference, module)
    module.metadata["tensor_ssa_reference"] = reference
    propagate_repository_ssa_call_metadata(module)

    def raw_binary_span_ready(function: Function, instruction: Instr) -> bool:
        operation = _SHAPED_SSA_OPERATIONS.get(str(instruction.op))
        signature = None if operation is None else c_tensor_opcode(operation)
        if signature is None or signature[0] != "binary":
            return False
        # In make(count) -> root, Add3 first reaches this closure before
        # source Call metadata gives its input1 an extent. The input's posted
        # physical dtype is still absent, while Add3's own SSA_VALUE already
        # proves a float64 span. Keep this exact raw operation in the affected
        # Call closure so the later input descriptor revision revisits it.
        return any(
            (fact := posted_ssa_value_fact(
                function, int(value.id), book=identity_book(module),
            )) is not None
            and fact.physical_dtype is not None
            and fact.storage == "span"
            and fact.rank is not None and fact.rank > 0
            for value in (*instruction.args,
                          *((instruction.res,) if instruction.res is not None else ()))
        )

    affected = {
        str(name)
        for name, function in module.functions.items()
        if any(
            instruction.attributes.get("recovered_structural_input_view")
            or str(instruction.op) in {"Indexed", "IndexedStore"}
            or any(instruction.attributes.get(key) is not None
                   for key in ("tensor_operation", "tensor", "tensor_candidate"))
            or raw_binary_span_ready(function, instruction)
            for block in function.blocks.values()
            for instruction in block.instrs
        )
    }

    def close_callers(names: set[str]) -> set[str]:
        result = set(names)
        changed = True
        while changed:
            changed = False
            for caller_name, caller in module.functions.items():
                if str(caller_name) in result:
                    continue
                if any(
                    str(instruction.attributes.get("callee") or "") in result
                    for block in caller.blocks.values()
                    for instruction in block.instrs
                    if str(instruction.op).casefold() == "call"
                ):
                    result.add(str(caller_name))
                    changed = True
        return result

    def include_metadata_producers(names: set[str]) -> set[str]:
        """Keep the exact producer visible when a consumer retries metadata.

        In ``1 - numeric_valid``, the consumer region's descriptor becomes
        dynamic after the cast region publishes its result. Its next narrowed
        slice had the caller and consumer, but omitted the cast producer.
        The existing metadata reader therefore could not verify the owned
        output projection cited by the consumer's Call binding. Follow that
        binding's direct RESULT cause to include its producer in the same
        incumbent Call closure; no extent or data identity is reconstructed.
        """
        from .concordance_declarations import (
            TENSOR_METADATA_CALL_BINDING, TENSOR_METADATA_RESULT_BINDING,
        )
        from .ssa_record_return_state import (
            function_scope_of, metadata_result_caller_value,
        )

        book = identity_book(module)
        result = set(names)
        changed = True
        while changed:
            changed = False
            for row in book.page(TENSOR_METADATA_CALL_BINDING).rows():
                if row[4] not in result:
                    continue
                caller = module.functions.get(str(row[1]))
                if caller is None or row[0] != function_scope_of(caller):
                    continue
                block = caller.blocks.get(str(row[2]))
                live_calls = () if block is None else tuple(
                    instruction for instruction in block.instrs
                    if instruction.op == "Call" and instruction.res is not None
                    and int(instruction.res.id) == int(row[3])
                    and instruction.attributes.get("callee") == row[4]
                )
                binding = book.latest_ref(TENSOR_METADATA_CALL_BINDING, row)
                if len(live_calls) != 1 or binding is None:
                    continue
                predecessors = tuple(
                    cause for cause, _stage in book.edges_into(binding)
                    if cause.page == TENSOR_METADATA_RESULT_BINDING
                    and cause.row[0] == row[0]
                )
                for predecessor in predecessors:
                    producer_calls = tuple(
                        instruction for producer_block in caller.blocks.values()
                        for instruction in producer_block.instrs
                        if instruction.op == "Call" and instruction.res is not None
                        and int(instruction.res.id) == int(predecessor.row[1])
                        and instruction.attributes.get("callee")
                        == predecessor.row[2]
                    )
                    if (len(producer_calls) != 1
                            or metadata_result_caller_value(
                                module, caller, producer_calls[0],
                                predecessor, book=book,
                            ) is None):
                        continue
                    producer_name = str(predecessor.row[2])
                    for name in (str(row[1]), producer_name):
                        if name in module.functions and name not in result:
                            result.add(name)
                            changed = True
        return result

    affected = close_callers(affected)

    def value_geometry() -> dict[tuple[str, int], tuple | None]:
        geometry = {}
        for name in sorted(affected):
            if name not in module.functions:
                continue
            function = module.functions[name]
            table = (getattr(module, "tensor_tables", {}) or {}).get(name)
            value_ids = {
                int(formal.id) for formal in function.args
                if not (formal.accounting or {}).get("heap_owner")
                and (formal.accounting or {}).get("tensor_extent_for") is None
            }
            if table is not None:
                value_ids.update(
                    int(identity) for identity, descriptor in table.tensors.items()
                    if descriptor.metadata_state == "dynamic"
                )
            for identity in sorted(value_ids):
                descriptor = None if table is None else table.by_id(identity)
                geometry[(name, identity)] = (
                    None if descriptor is None else (
                        descriptor.metadata_state, descriptor.shape,
                        descriptor.shape_value_id, descriptor.rank_value_id,
                        descriptor.element_count_value_id,
                    )
                )
        return geometry

    # Local precompile originally linked every function on its first pass.
    # Keep that coverage; only changed geometry and its exact callers are
    # revisited after the initial pass. Whole-program late work starts with
    # the already selected tensor slice.
    pending = include_metadata_producers(
        set(module.functions) if initial_all else set(affected)
    )
    shortfalls_by_function: dict[str, list[TensorSSALoweringShortfall]] = {}
    first_pass = True
    while first_pass or pending:
        first_pass = False
        prior_geometry = value_geometry()
        # make([start])'s int64 column is already in the owner's sequence
        # table. Constructing a new IRModule here dropped that registry, so
        # the cast region could not read the existing sequence view and
        # refused its physical span. This is a view of the same compilation:
        # narrow functions/tensors while retaining every other registry by
        # identity, rather than rebuilding an incomplete compilation model.
        slice_module = dataclasses.replace(
            module,
            functions={name: function for name, function in module.functions.items()
                       if name in pending},
            metadata={
                **module.metadata,
                "identity_book": identity_book(module),
                "tensor_ssa_reference": reference,
            },
            tensor_tables={
                name: module.tensor_tables[name]
                for name in pending
                if name in (getattr(module, "tensor_tables", {}) or {})
            },
        )
        current = lower_tensor_calls_to_repository_ssa(
            slice_module, reference, finalize_private_owners=False,
        )
        # Program: a mixed broadcast in one narrowed Call slice imports an
        # authored helper with its accepted owner book.  The next slice used
        # the original reference and compiled a second helper under the same
        # function name; the live function and imported formal cells then had
        # different artifact digests.  The slice's recorded augmentation is
        # the one reference for every subsequent slice and the parent module.
        reference = retained_authored_broadcast_reference(
            reference, slice_module,
        )
        module.metadata["tensor_ssa_reference"] = reference
        for name in pending:
            shortfalls_by_function.pop(name, None)
        for item in current:
            shortfalls_by_function.setdefault(item.function, []).append(item)
        for name, function in slice_module.functions.items():
            resident = module.functions.get(name)
            if (
                resident is not None and resident is not function
                and (
                    resident.metadata.get("accepted_ssa_component")
                    or function.metadata.get("accepted_ssa_component")
                )
            ):
                raise ValueError(
                    f"accepted repository SSA function has two owners: {name}"
                )
            module.functions.setdefault(name, function)
        for name, table in slice_module.tensor_tables.items():
            module.tensor_tables.setdefault(name, table)
        publish_computed_region_result_metadata(
            module, finalize_private_owners=False,
        )
        propagate_repository_ssa_call_metadata(module)
        settle_repository_ssa_static_extent_operands(module)
        changed_names = {
            name for (name, value_id), geometry in value_geometry().items()
            if geometry != prior_geometry.get((name, value_id))
        }
        pending = include_metadata_producers(
            close_callers(changed_names) & affected
        )
    if finalize_private_owners:
        publish_computed_region_result_metadata(module)
    return tuple(
        item for name in sorted(shortfalls_by_function)
        for item in shortfalls_by_function[name]
    )


__all__ = [
    "TensorSSALoweringShortfall",
    "lower_tensor_calls_to_repository_ssa",
    "legalize_aggregate_adapters",
    "legalize_aggregate_output_views",
    "propagate_repository_ssa_call_metadata",
    "settle_shape_only_repository_returns",
    "wire_repository_ssa_region_products",
]
