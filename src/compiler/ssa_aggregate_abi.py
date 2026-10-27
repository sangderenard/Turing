"""Backend-neutral analysis of planned-region aggregate call records.

The planner represents a multiple-output call as one abstract aggregate value
followed by ``GetElementPtr``/``Load`` projections.  Native backends do not
need to materialize that abstract tuple: they need the declared output record
and the concrete SSA value bound to each record position.  This module keeps
that identity-sensitive analysis in one place so C, LLVM, and Fortran can use
the same contract.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..transmogrifier.ssa import Function, Instr, IRModule, SSAValue


@dataclass(frozen=True, slots=True)
class AggregateProjection:
    position: int
    output_id: int
    address: Instr
    load: Instr

    @property
    def value(self) -> SSAValue:
        assert self.load.res is not None
        return self.load.res


@dataclass(frozen=True, slots=True)
class AggregateCallRecord:
    caller: str
    callee: str
    call: Instr
    output_ids: tuple[int, ...]
    projections: tuple[AggregateProjection, ...]

    def projection_for_output(self, output_id: int) -> AggregateProjection | None:
        return next(
            (item for item in self.projections if item.output_id == output_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class AggregateABIAnalysis:
    calls: tuple[AggregateCallRecord, ...]
    outputs_by_callee: Mapping[str, tuple[SSAValue, ...]]

    def call_record(self, instruction: Instr) -> AggregateCallRecord | None:
        return next((item for item in self.calls if item.call is instruction), None)


def is_storage_view(value: SSAValue, storage: SSAValue, *, function=None, book=None) -> bool:
    """Whether ``value`` is an explicitly correlated view of ``storage``.

    Ordered call views are distinct objects so they may carry per-position
    shapes.  Numeric equality alone is unsafe because linker/planner numbering
    domains can collide; the ``ssa_storage_alias`` receipt is authoritative.
    """

    if value is storage or (
        (value.accounting or {}).get("ssa_storage_alias") == int(storage.id)
    ):
        return True
    if function is None:
        return False
    from .ssa_record_return_state import ssa_value_identity_cell
    from .structural_concordance import function_value_structure
    from ..transmogrifier.dtype_layout import resolve

    value_cell = ssa_value_identity_cell(function, value.id, book=book)
    storage_cell = ssa_value_identity_cell(function, storage.id, book=book)
    if value_cell is None or value_cell != storage_cell:
        return False
    # Distinct Python occurrences may be views of one posted SSA identity.
    # A declared reshape remains a distinct view, even of the same storage.
    if tuple(value.shape or ()) != tuple(storage.shape or ()):
        return False
    left, right = resolve(value.dtype), resolve(storage.dtype)
    if left is None or right is None or left.name != right.name:
        return False
    for occurrence in (value, storage):
        declared = (occurrence.accounting or {}).get("ssa_storage_view")
        if declared and tuple(declared["view_shape"]) != tuple(storage.shape or ()):
            return False
    structure, _cell = function_value_structure(function, value.id, book=book)
    for occurrence in (value, storage):
        accounting = occurrence.accounting or {}
        for label, authoritative in (
            ("physical_dtype", structure.physical_dtype),
            ("program_abi_storage", structure.declared_storage or structure.storage),
            ("program_abi_rank", structure.rank),
        ):
            if label in accounting and accounting[label] != authoritative:
                return False
    return structure.dtype == left.name and tuple(structure.shape) == tuple(value.shape or ())


def _function_values(function: Function) -> dict[int, SSAValue]:
    values = {int(value.id): value for value in function.args}
    if function.metadata.get("native_interface") or function.metadata.get("linked_piece_interface"):
        values.update((int(value.id), value) for value in
                      function.metadata["declared_native_outputs"])
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.res is None:
                continue
            value_id = int(instruction.res.id)
            existing = values.get(value_id)
            if existing is None or (
                not tuple(existing.shape or ())
                and tuple(instruction.res.shape or ())
            ):
                values[value_id] = instruction.res
    return values


def analyze_aggregate_abi(
    module: IRModule, function_names: Iterable[str] | None = None,
) -> AggregateABIAnalysis:
    """Resolve planned aggregate records without changing the IR.

    ``output_ids`` is the authored return record and may repeat identities.
    Native output parameters are canonicalized by identity, while projections
    retain their record positions.  Type/shape information is taken from the
    callee when available and otherwise from the caller projection carrying
    that exact declared identity.
    """

    selected = tuple(
        str(name) for name in (
            function_names if function_names is not None else module.functions
        )
        if str(name) in module.functions
    )
    values = {
        name: _function_values(module.functions[name]) for name in selected
    }
    calls: list[AggregateCallRecord] = []
    output_candidates: dict[str, dict[int, SSAValue]] = {}
    output_order: dict[str, list[int]] = {}
    returned_layouts: dict[str, tuple[SSAValue, ...]] = {}

    # Ordinary source functions state their native output ABI directly with
    # Ret.  Planned regions may omit Ret and publish the same information via
    # aggregate call records below.  Seed both through one identity table so
    # every backend sees the same caller-owned output contract.
    for function_name in selected:
        returned = next((
            tuple(instruction.args)
            for block in module.functions[function_name].blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        ), tuple(module.functions[function_name].metadata.get("declared_native_outputs", ())))
        if len(returned) == 1:
            aggregate_ids = tuple(map(
                int,
                (returned[0].accounting or {}).get(
                    "ssa_aggregate_outputs", (),
                ),
            ))
            if aggregate_ids and all(
                value_id in values[function_name]
                for value_id in aggregate_ids
            ):
                returned = tuple(
                    values[function_name][value_id]
                    for value_id in aggregate_ids
                )
        returned_layouts[function_name] = tuple(returned)
        order = output_order.setdefault(function_name, [])
        candidates = output_candidates.setdefault(function_name, {})
        for value in returned:
            value_id = int(value.id)
            if value_id not in order:
                order.append(value_id)
            candidates[value_id] = value

    for caller_name in selected:
        function = module.functions[caller_name]
        instructions = [
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
        ]
        for call in instructions:
            if (
                call.op not in {"Call", "call"}
                or call.res is None
                or call.attributes.get("result_convention") != "ssa.aggregate"
            ):
                continue
            callee = str(call.attributes.get("callee") or "")
            caller_output_ids = tuple(map(
                int, call.attributes.get("output_ids", ())
            ))
            output_positions = tuple(map(
                int,
                call.attributes.get("output_positions", range(
                    len(caller_output_ids)
                )),
            ))
            callee_output_ids = tuple(map(
                int,
                call.attributes.get("callee_output_ids", caller_output_ids),
            ))
            if not callee or not caller_output_ids:
                continue
            # ``callee_output_ids`` / ``output_positions`` are derived
            # correlations of the caller's output list.  A stale derivation
            # of a different length must not silently drop the call from the
            # aggregate ABI: with no record the call is never emitted, its
            # projections have no storage, and the failure surfaces only as
            # untraceable operand fallout (the contact graph's 20).  Fall
            # back to the positional identity, which is exact for a region.
            if len(output_positions) != len(caller_output_ids):
                output_positions = tuple(range(len(caller_output_ids)))
            if len(callee_output_ids) != len(caller_output_ids):
                callee_output_ids = caller_output_ids
            selected_by_position = {
                position: (caller_id, callee_id)
                for position, caller_id, callee_id in zip(
                    output_positions, caller_output_ids, callee_output_ids
                )
            }
            # Call records may omit outputs already carried by an in/out
            # formal while retaining the original sparse output_positions.
            # Reconstruct the complete positional aggregate from the callee's
            # Ret layout.  Compressing the sparse list shifts every later
            # field left (for example Metrics.div_inf into max_vel) and loses
            # repeated identities such as max_vel/max_flux entirely.
            for position, returned_value in enumerate(
                returned_layouts.get(callee, ())
            ):
                selected_by_position.setdefault(
                    int(position),
                    (int(returned_value.id), int(returned_value.id)),
                )
            complete_callee_output_ids = tuple(
                int(selected_by_position[position][1])
                for position in sorted(selected_by_position)
            )
            addresses: dict[int, tuple[int, Instr]] = {}
            projections: list[AggregateProjection] = []
            for follower in instructions:
                if (
                    follower.op in {"GetElementPtr", "getelementptr"}
                    and follower.res is not None
                    and follower.args
                    and int(follower.args[0].id) == int(call.res.id)
                ):
                    position = follower.attributes.get("aggregate_index")
                    if position is not None:
                        addresses[int(follower.res.id)] = (int(position), follower)
                elif (
                    follower.op in {"Load", "load"}
                    and follower.res is not None
                    and follower.args
                    and int(follower.args[0].id) in addresses
                ):
                    position, address = addresses[int(follower.args[0].id)]
                    selected = selected_by_position.get(position)
                    if selected is not None:
                        _caller_id, callee_id = selected
                        projections.append(AggregateProjection(
                            position, callee_id, address, follower,
                        ))
            record = AggregateCallRecord(
                caller_name, callee, call, complete_callee_output_ids,
                tuple(projections),
            )
            calls.append(record)
            order = output_order.setdefault(callee, [])
            candidates = output_candidates.setdefault(callee, {})
            callee_values = values.get(callee, {})
            for output_id in callee_output_ids:
                if output_id not in order:
                    order.append(output_id)
                candidate = callee_values.get(output_id)
                projection = record.projection_for_output(output_id)
                if candidate is None and projection is not None:
                    projected = projection.value
                    candidate = SSAValue(
                        output_id,
                        dtype=projected.dtype,
                        shape=tuple(projected.shape or ()),
                        device=projected.device,
                        accounting=dict(projected.accounting or {}),
                    )
                if candidate is None:
                    candidate = values[caller_name].get(output_id)
                if candidate is not None:
                    existing = candidates.get(output_id)
                    if existing is None or (
                        not tuple(existing.shape or ())
                        and tuple(candidate.shape or ())
                    ):
                        candidates[output_id] = candidate

    outputs_by_callee = {
        callee: tuple(
            output_candidates[callee][output_id]
            for output_id in order
            if output_id in output_candidates[callee]
        )
        for callee, order in output_order.items()
    }
    return AggregateABIAnalysis(tuple(calls), outputs_by_callee)


def _constant_integer(instruction: Instr) -> int | None:
    if instruction.op not in {"Const", "const"}:
        return None
    for key in ("constant", "value", "data"):
        value = instruction.attributes.get(key)
        if isinstance(value, (bool, int)):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _aggregate_projections(
    function: Function, aggregate: SSAValue,
) -> dict[int, AggregateProjection]:
    """Return projections of one exact aggregate occurrence.

    Integer SSA ids are not globally unique across linked planner domains.
    The aggregate object (or an explicit ``ssa_storage_alias`` view of it) is
    therefore the root of the walk; numeric equality alone is never enough.
    """

    instructions = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
    ]
    constants = {
        id(instruction.res): value
        for instruction in instructions
        if instruction.res is not None
        and (value := _constant_integer(instruction)) is not None
    }
    addresses: dict[int, tuple[int, Instr]] = {}
    projections: dict[int, AggregateProjection] = {}
    for instruction in instructions:
        if (
            instruction.op in {"GetElementPtr", "getelementptr"}
            and instruction.res is not None
            and instruction.args
            and (
                instruction.args[0] is aggregate
                or is_storage_view(instruction.args[0], aggregate)
            )
        ):
            position = instruction.attributes.get("aggregate_index")
            if position is None and len(instruction.args) > 1:
                position = constants.get(id(instruction.args[1]))
            if position is not None:
                addresses[id(instruction.res)] = (int(position), instruction)
        elif (
            instruction.op in {"Load", "load"}
            and instruction.res is not None
            and instruction.args
            and id(instruction.args[0]) in addresses
        ):
            position, address = addresses[id(instruction.args[0])]
            projections[position] = AggregateProjection(
                position, int(instruction.res.id), address, instruction,
            )
    return projections


def _post_passthrough_rebinding(
    function: Function, source: SSAValue, replacement: SSAValue,
    via: tuple[str, int], *, sources=(),
) -> int:
    """Post ``aggregate_passthrough_rebinding`` for one folded projection and
    return the replacement id THE ROW states.

    DERIVED(the original's and the replacement's ``ssa_value`` cells; a fold
    that can name neither is written raw, which the book tags ``Unsourced``).
    The row's fourth field is the declared view shape the replacement carries
    (absent when the output is the storage itself).
    """

    from .concordance_declarations import (
        AGGREGATE_ABI_LEGALIZATION, AGGREGATE_PASSTHROUGH_REBINDING,
    )
    from .identity_concordance import Mode, current_identity_book
    from .precompile_to_ssa import _post_derived_or_raw
    from .ssa_record_return_state import function_scope_of, identity_cells

    book = current_identity_book()
    row = (function_scope_of(function), int(source.id))
    declared_view = (replacement.accounting or {}).get("ssa_storage_view")
    _post_derived_or_raw(
        book, AGGREGATE_PASSTHROUGH_REBINDING, row,
        (int(replacement.id), str(via[0]), int(via[1]), *(
            (tuple(int(extent) for extent in declared_view["view_shape"]),)
            if declared_view else ()
        )),
        (*identity_cells(function, source, replacement), *sources),
        stage=AGGREGATE_ABI_LEGALIZATION, mode=Mode.REVISE,
    )
    # Folding advance's returned publication spans retired their projection
    # slots, but return-field Phi aliases still pointed at those slots. The
    # operand rewrite below and the residency ledger are the same proven fold,
    # not independent records. Advance dependent aliases through this exact
    # passthrough row before removing the source definition.
    aliases = dict(function.metadata.get("value_aliases", {}) or {})
    if int(source.id) in aliases or int(source.id) in set(map(int, aliases.values())):
        from .fortran_c_shell import _publish_concordant_function_aliases
        _publish_concordant_function_aliases(
            function, {int(source.id): int(replacement.id)},
            provenance="exact_aggregate_passthrough_rebinding",
            sources={int(source.id): (
                book.latest_ref(AGGREGATE_PASSTHROUGH_REBINDING, row),
                *identity_cells(function, source, replacement),
            )},
        )
    return int(book.page(AGGREGATE_PASSTHROUGH_REBINDING).latest(row)[0])


def _replace_exact_uses(
    function: Function, source: SSAValue, replacement: SSAValue,
    *, via: tuple[str, int], sources=(),
) -> None:
    """Rebind every use of ``source`` to ``replacement`` (the fold removes
    ``source``'s definition), as one row on the book.

    A use is an operand or an id-carrying reference: a Phi's
    ``initial_value_id`` names the value that stood before the merge, and a
    Phi whose initial is the folded projection must name what the row says
    now stands there.  Left alone it names an id with no definition in the
    function (N=2 orbital dt system: the return-merge field Phis of
    ``pub_limits`` / ``pub_limits_present`` of ``step_with_dt_control_used``).
    """

    rebound_id = _post_passthrough_rebinding(function, source, replacement, via, sources=sources)
    for block in function.blocks.values():
        for instruction in block.instrs:
            instruction.args[:] = [
                replacement
                if is_storage_view(argument, source, function=function)
                else argument
                for argument in instruction.args
            ]
            attributes = instruction.attributes
            if (
                attributes
                and attributes.get("initial_value_id") is not None
                and int(attributes["initial_value_id"]) == int(source.id)
            ):
                attributes["initial_value_id"] = rebound_id


def _remove_instructions(function: Function, removed: set[int]) -> None:
    for block in function.blocks.values():
        block.instrs[:] = [
            instruction for instruction in block.instrs
            if id(instruction) not in removed
        ]


def _remove_dead_projection_constants(
    function: Function, candidates: Iterable[SSAValue],
) -> None:
    candidate_ids = {id(value) for value in candidates}
    if not candidate_ids:
        return
    used = {
        id(argument)
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
                and id(instruction.res) in candidate_ids
                and id(instruction.res) not in used
            )
        ]


def legalize_aggregate_adapters(module: IRModule) -> bool:
    """Expand projection adapters onto their producer's real tensor values.

    A linked multi-output call is a structural aggregate, not flat numerical
    storage.  Region planning can place a small consumer after such a call
    whose formal is the aggregate and whose first instructions merely project
    its members.  Native backends must see those projected tensor addresses as
    ordinary call operands.  Pass-through adapter results are rebound in the
    caller, while genuinely computed results retain the aggregate output ABI.

    The transformation is deliberately shared and identity-sensitive so C,
    LLVM, and Fortran consume the same legalized call record.
    """

    changed = False
    # Snapshot because a projection-only adapter can become unreachable and be
    # removed after its sole call disappears.
    for caller in tuple(module.functions.values()):
        calls = [
            instruction
            for block in caller.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Call", "call"}
        ]
        producers = [
            instruction for instruction in calls
            if instruction.res is not None
            and instruction.attributes.get("result_convention") == "ssa.aggregate"
        ]
        for call in tuple(calls):
            callee_name = str(call.attributes.get("callee") or "")
            adapter = module.functions.get(callee_name)
            if adapter is None or call not in calls:
                continue
            for formal_position, (actual, formal) in enumerate(tuple(zip(
                call.args, adapter.args,
            ))):
                formal_projections = _aggregate_projections(adapter, formal)
                if not formal_projections:
                    continue
                candidates = [
                    producer for producer in producers
                    if producer is not call
                    and producer.res is not None
                    and is_storage_view(actual, producer.res)
                ]
                linked_from = (actual.accounting or {}).get(
                    "ssa_linked_storage_from"
                )
                if linked_from is not None:
                    exact = [
                        producer for producer in candidates
                        if int(producer.attributes.get(
                            "plan_callsite_id", -1
                        )) == int(linked_from)
                    ]
                    if exact:
                        candidates = exact
                if len(candidates) != 1:
                    continue
                producer = candidates[0]
                assert producer.res is not None
                producer_projections = _aggregate_projections(
                    caller, producer.res,
                )
                if not all(
                    position in producer_projections
                    for position in formal_projections
                ):
                    continue

                expanded_formals: list[SSAValue] = []
                expanded_actuals: list[SSAValue] = []
                source_by_formal_id: dict[int, SSAValue] = {}
                removed_adapter: set[int] = set()
                adapter_constant_candidates: list[SSAValue] = []
                for position, projection in formal_projections.items():
                    projected = producer_projections[position].value
                    formal_value = projection.value
                    actual_view = SSAValue(
                        int(projected.id),
                        dtype=formal_value.dtype or projected.dtype,
                        shape=tuple(formal_value.shape or projected.shape or ()),
                        device=formal_value.device or projected.device,
                        accounting={
                            **dict(projected.accounting or {}),
                            **dict(formal_value.accounting or {}),
                            "ssa_storage_alias": int(projected.id),
                            "ssa_aggregate_projection": (
                                int(producer.res.id), int(position),
                            ),
                        },
                    )
                    expanded_formals.append(formal_value)
                    expanded_actuals.append(actual_view)
                    source_by_formal_id[int(formal_value.id)] = projected
                    removed_adapter.update((
                        id(projection.address), id(projection.load),
                    ))
                    if len(projection.address.args) > 1:
                        adapter_constant_candidates.append(
                            projection.address.args[1]
                        )

                adapter.args[formal_position:formal_position + 1] = (
                    expanded_formals
                )
                call.args[formal_position:formal_position + 1] = expanded_actuals
                _remove_instructions(adapter, removed_adapter)
                _remove_dead_projection_constants(
                    adapter, adapter_constant_candidates,
                )

                output_ids = tuple(map(
                    int, call.attributes.get("output_ids", ())
                ))
                output_positions = tuple(map(
                    int,
                    call.attributes.get(
                        "output_positions", range(len(output_ids))
                    ),
                ))
                caller_projections = _aggregate_projections(
                    caller, call.res,
                ) if call.res is not None else {}
                tensor_table = module.tensor_tables.get(callee_name)

                def source_for_output(output_id: int) -> SSAValue | None:
                    pending = [int(output_id)]
                    seen: set[int] = set()
                    while pending:
                        value_id = pending.pop()
                        if value_id in seen:
                            continue
                        seen.add(value_id)
                        source = source_by_formal_id.get(value_id)
                        if source is not None:
                            return source
                        descriptor = (
                            tensor_table.by_id(value_id)
                            if tensor_table is not None else None
                        )
                        if descriptor is not None:
                            pending.append(int(descriptor.data_value_id))
                            if descriptor.alias_of is not None:
                                pending.append(int(descriptor.alias_of))
                    return None

                retained_indices: list[int] = []
                removed_caller: set[int] = set()
                caller_constant_candidates: list[SSAValue] = []
                for index, (output_id, output_position) in enumerate(zip(
                    output_ids, output_positions,
                )):
                    source = source_for_output(output_id)
                    projection = caller_projections.get(output_position)
                    if source is None:
                        retained_indices.append(index)
                        continue
                    # An unobserved pass-through output is dead for the same
                    # reason as an observed one: the adapter computes no new
                    # storage for it.  Only the observed form needs caller-use
                    # rebinding and projection removal.
                    if projection is None:
                        continue
                    replacement = SSAValue(
                        int(source.id),
                        dtype=projection.value.dtype or source.dtype,
                        shape=tuple(projection.value.shape or source.shape or ()),
                        device=projection.value.device or source.device,
                        accounting={
                            **dict(source.accounting or {}),
                            **dict(projection.value.accounting or {}),
                            "ssa_storage_alias": int(source.id),
                            "ssa_aggregate_passthrough": (
                                callee_name, int(output_id),
                            ),
                        },
                    )
                    _replace_exact_uses(
                        caller, projection.value, replacement,
                        via=(callee_name, int(output_id)),
                    )
                    removed_caller.update((
                        id(projection.address), id(projection.load),
                    ))
                    if len(projection.address.args) > 1:
                        caller_constant_candidates.append(
                            projection.address.args[1]
                        )

                _remove_instructions(caller, removed_caller)
                _remove_dead_projection_constants(
                    caller, caller_constant_candidates,
                )

                # Remove expanded formals which served only pass-through
                # publication.  The computed adapter keeps exactly the tensor
                # projections it really reads.
                used_formals = {
                    id(argument)
                    for block in adapter.blocks.values()
                    for instruction in block.instrs
                    for argument in instruction.args
                }
                keep_expanded = [
                    index for index, value in enumerate(expanded_formals)
                    if id(value) in used_formals
                ]
                start = formal_position
                adapter.args[start:start + len(expanded_formals)] = [
                    expanded_formals[index] for index in keep_expanded
                ]
                call.args[start:start + len(expanded_actuals)] = [
                    expanded_actuals[index] for index in keep_expanded
                ]

                for key in (
                    "output_ids", "output_positions", "output_slots",
                    "callee_output_ids",
                ):
                    values = tuple(call.attributes.get(key, ()))
                    if key == "output_positions" and not values:
                        values = output_positions
                    if values and len(values) == len(output_ids):
                        call.attributes[key] = tuple(
                            values[index] for index in retained_indices
                        )
                call.attributes["aggregate_adapter_legalized"] = True

                if not retained_indices:
                    meaningful = [
                        instruction
                        for block in adapter.blocks.values()
                        for instruction in block.instrs
                        if instruction.op not in {"Const", "const"}
                    ]
                    if not meaningful:
                        _remove_instructions(caller, {id(call)})
                changed = True

    if changed:
        called = {
            str(instruction.attributes.get("callee") or "")
            for function in module.functions.values()
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Call", "call"}
        }
        for name in tuple(module.functions):
            function = module.functions[name]
            if (
                name not in called
                and "__planned_region_" in name
                and not any(
                    instruction.op not in {"Const", "const"}
                    for block in function.blocks.values()
                    for instruction in block.instrs
                )
            ):
                module.functions.pop(name, None)
                module.tensor_tables.pop(name, None)
    return changed


def legalize_aggregate_output_views(module: IRModule) -> bool:
    """Name aggregate outputs by their resident tensor storage identity.

    Planned regions may publish several reshape/view identities backed by one
    computed tensor.  The caller keeps those semantic projection identities
    and shapes, while the callee output ABI must name the one value it really
    computes.  ``callee_output_ids`` is the existing shared correlation for
    precisely that distinction and naturally retains repeated aliases.
    """

    changed = False
    from .concordance_declarations import (
        CALL_RECORD, ARGUMENT_BINDING, RESULT_STORAGE_BINDING,
        SEQUENCE_DESCRIPTOR, AGGREGATE_ABI_LEGALIZATION,
    )
    from .identity_concordance import current_identity_book

    book = current_identity_book()

    def returned_formal_cells(function, call, callee_id, caller_id, actual):
        """The exact callsite's incumbent formal/result/storage relation."""
        site = call.attributes.get("plan_callsite_id")
        if site is None:
            return ()
        callee = str(call.attributes.get("callee") or "")
        page = book.page(CALL_RECORD)
        cells = []
        for row in page.rows():
            if len(row) != 2 or row[1] != function.name:
                continue
            for record in page.latest(row) or ():
                if (record.caller != function.name or record.callee_symbol != callee
                        or int(record.callsite_id) != int(site)):
                    continue
                bindings = [binding for binding in record.frame_bindings
                            if int(binding[0]) == int(callee_id)]
                if (len(bindings) != 1
                        or bindings[0][1] not in {"caller_value", "caller_alias", "caller_storage"}
                        or int(bindings[0][2]) != int(actual.id)
                        or (int(callee_id), int(caller_id)) not in record.result_bindings):
                    return ()
                cells.append(book.latest_ref(CALL_RECORD, row))
        if not cells:
            return ()
        # A compiler-owned actual may have an Unresolved argument binding
        # whose dependency is its exact result-storage allocation. Keep both
        # existing cells; no name or signature-position proxy replaces them.
        cells.extend((
            book.latest_ref(ARGUMENT_BINDING, (callee, function.name, int(callee_id), int(site))),
            book.latest_ref(RESULT_STORAGE_BINDING, (function.name, int(site), int(callee_id), 0)),
        ))
        return tuple(dict.fromkeys(cell for cell in cells if cell is not None))

    def transport_returned_sequence(function, call, callee_id, caller_id, cells):
        """Project the whole returned frame through its existing call record.

        A sequence's column alias alone is not its storage identity. Length,
        capacity, status, live flags, owner and nested pool travel together,
        using the same incumbent remapper as record-return materialization.
        """
        nonlocal changed
        from .ssa_record_return_state import map_sequence_descriptor, map_child_pool
        from .frame_identity_book import _frame_register_sequence, _frame_table_cell
        from ..transmogrifier.ssa import sequence_member_roles

        callee_name = str(call.attributes.get("callee") or "")
        source_table = module.sequence_tables.get(callee_name)
        target_table = module.sequence_tables.get(function.name)
        source = None if source_table is None else source_table.by_id(int(callee_id))
        if source is None:
            return True
        if target_table is None or not cells:
            return False
        incumbent = target_table.by_id(int(caller_id))
        if incumbent is None:
            return False
        frame = {}
        for cell in cells:
            if cell.page != CALL_RECORD:
                continue
            for record in book.page(CALL_RECORD).cell(cell.row, cell.column) or ():
                if (record.caller != function.name or record.callee_symbol != callee_name
                        or int(record.callsite_id) != int(call.attributes["plan_callsite_id"])):
                    continue
                for formal, kind, actual in record.frame_bindings:
                    if kind in {"caller_value", "caller_alias", "caller_storage"}:
                        old = frame.setdefault(int(formal), int(actual))
                        if old != int(actual):
                            return False
        # Incumbent result-storage rows also name members materialized from
        # authored constants rather than passed as native formals.
        result_page = book.page(RESULT_STORAGE_BINDING)
        for row in result_page.rows():
            if row[:2] != (function.name, int(call.attributes["plan_callsite_id"])):
                continue
            ref = result_page.latest(row)
            if not hasattr(ref, "page") or ref.page.name != "ssa_value":
                continue
            old = frame.setdefault(int(row[2]), int(ref.row[1]))
            if old != int(ref.row[1]):
                return False
        if not set(sequence_member_roles(source)).issubset(frame):
            return False
        mapped = map_sequence_descriptor(
            source, frame, sequence_id=int(caller_id),
            child_table_pool=map_child_pool(source.child_table_pool, frame),
        )
        descriptor_cell = _frame_table_cell(
            source_table, source.sequence_id, SEQUENCE_DESCRIPTOR,
            function=module.functions.get(callee_name),
        )
        if descriptor_cell is None:
            return False
        proof = (*cells, descriptor_cell)
        # Match physical members by their declared role, never by field names
        # or allocation positions. The semantic handle keeps its caller ID.
        old_roles = {role: value for value, claims in sequence_member_roles(incumbent).items()
                     for _sequence, role in claims if role != ("handle",)}
        new_roles = {role: value for value, claims in sequence_member_roles(mapped).items()
                     for _sequence, role in claims if role != ("handle",)}
        if not set(old_roles).issubset(new_roles):
            return False
        resident = _function_values(function)
        for role, old_id in old_roles.items():
            new_id = new_roles[role]
            if old_id == new_id or old_id == caller_id:
                continue
            original, replacement = resident.get(old_id), resident.get(new_id)
            if original is None or replacement is None:
                return False
        if incumbent != mapped:
            _frame_register_sequence(
                function, target_table, mapped, *proof,
                stage=AGGREGATE_ABI_LEGALIZATION, revise=True,
            )
            changed = True
        for role, old_id in old_roles.items():
            new_id = new_roles[role]
            if old_id != new_id and old_id != caller_id:
                _replace_exact_uses(
                    function, resident[old_id], resident[new_id],
                    via=(callee_name, old_id), sources=proof,
                )
        return True
    for function in module.functions.values():
        from .ssa_record_return_state import accepted_ssa_component
        if accepted_ssa_component(function):
            # An accepted body and its descriptor proof remain on their
            # verified owner book. Only the enclosing caller's views link.
            continue
        for block in function.blocks.values():
            for call in block.instrs:
                if (
                    call.op not in {"Call", "call"}
                    or call.res is None
                    or call.attributes.get("result_convention") != "ssa.aggregate"
                ):
                    continue
                caller_ids = tuple(map(
                    int, call.attributes.get("output_ids", ())
                ))
                if not caller_ids:
                    continue
                debug_callsite = os.environ.get(
                    "TURING_DEBUG_AGGREGATE_CALLSITE"
                )
                debug_this_call = bool(
                    debug_callsite
                    and int(call.attributes.get("plan_callsite_id", -1))
                    == int(debug_callsite)
                )
                if debug_this_call:
                    print(
                        "DEBUG-AGGREGATE-BEFORE "
                        f"function={function.name!r} "
                        f"attributes={dict(call.attributes)!r}",
                        file=sys.stderr,
                    )
                callee_name = str(call.attributes.get("callee") or "")
                table = module.tensor_tables.get(callee_name)
                # Native output ports may already have folded their column
                # projections. The complete semantic return relation remains
                # the call record; project its whole sequence frames before
                # materializing the remaining native-output view below.
                native_inputs = dict(zip(
                    map(int, call.attributes.get("callee_input_ids", ())),
                    call.args,
                ))
                for row in book.page(CALL_RECORD).rows():
                    if len(row) != 2 or row[1] != function.name:
                        continue
                    for record in book.page(CALL_RECORD).latest(row) or ():
                        if (record.caller != function.name or record.callee_symbol != callee_name
                                or int(record.callsite_id) != int(call.attributes.get("plan_callsite_id", -1))):
                            continue
                        for callee_id, caller_id in record.result_bindings:
                            actual = native_inputs.get(int(callee_id))
                            if actual is not None:
                                cells = returned_formal_cells(function, call, callee_id, caller_id, actual)
                                transport_returned_sequence(function, call, callee_id, caller_id, cells)
                authored = tuple(map(
                    int,
                    call.attributes.get("callee_output_ids", caller_ids),
                ))
                if len(authored) != len(caller_ids):
                    # A derived callee list of the wrong length is stale;
                    # the caller list is the authority.  Recompute rather
                    # than leaving the two in disagreement.
                    authored = caller_ids
                canonical: list[int] = []
                output_descriptors: list[Any] = []
                for output_id in authored:
                    current = int(output_id)
                    output_descriptors.append(None if table is None else table.by_id(current))
                    seen: set[int] = set()
                    while current not in seen:
                        seen.add(current)
                        descriptor = None if table is None else table.by_id(current)
                        if descriptor is None:
                            break
                        storage = int(descriptor.data_value_id)
                        if storage == current:
                            break
                        current = storage
                    canonical.append(current)
                settled = tuple(canonical)
                target = module.functions.get(callee_name)
                formal_positions = {
                    int(formal.id): position
                    for position, formal in enumerate(target.args)
                } if target is not None else {}
                output_positions = tuple(map(
                    int,
                    call.attributes.get(
                        "output_positions", range(len(caller_ids))
                    ),
                ))
                projections = _aggregate_projections(
                    function, call.res,
                )
                retained: list[int] = []
                passthrough_bindings = []
                removed: set[int] = set()
                constant_candidates: list[SSAValue] = []
                for index, (caller_id, callee_id, output_position) in enumerate(
                    zip(caller_ids, settled, output_positions)
                ):
                    formal_position = formal_positions.get(callee_id)
                    if (
                        formal_position is None
                        or formal_position >= len(call.args)
                    ):
                        retained.append(index)
                        continue
                    actual = call.args[formal_position]
                    projection = projections.get(output_position)
                    if projection is None:
                        # A tensor descriptor can identify an output with an
                        # input formal even when the caller consumes that
                        # output directly (not through an aggregate
                        # projection).  There is no adapter to remove in that
                        # case, and dropping the output would also drop its
                        # producer from the call ABI.  Prune only projections
                        # whose uses are concretely rebound below.
                        retained.append(index)
                        continue
                    binding_cells = returned_formal_cells(
                        function, call, callee_id, caller_id, actual,
                    )
                    # Posted occurrences can share the projection's identity
                    # only across this exact recorded caller/callee binding.
                    # Raw standalone SSA retains the incumbent object path.
                    allow_posted_views = bool(binding_cells)
                    has_direct_consumer = any(
                        int(argument.id) == int(caller_id)
                        and argument is not projection.value
                        and not is_storage_view(
                            argument, projection.value,
                            function=function if allow_posted_views else None,
                        )
                        for consumer_block in function.blocks.values()
                        for consumer in consumer_block.instrs
                        if consumer is not projection.address
                        and consumer is not projection.load
                        for argument in consumer.args
                    )
                    if has_direct_consumer:
                        # Some linked callers consume a physical output
                        # directly while another path projects the enclosing
                        # aggregate at the same position.  The projection can
                        # be folded only if doing so does not erase that
                        # independent call result from the ABI.
                        retained.append(index)
                        continue
                    if not transport_returned_sequence(
                        function, call, callee_id, caller_id, binding_cells,
                    ):
                        retained.append(index)
                        continue
                    # Program: ``valid.reshape((-1, 1, 1))`` planned as a call
                    # whose output is a view of the formal ``valid`` and
                    # consumed by a later region.  Wrong record: the fold
                    # rebound the output to the actual's bare id, dropping the
                    # callee descriptor's declared view, so call-metadata
                    # propagation restamped the occurrence to the storage's
                    # own extents and recorded ``proven_shape conflicting``
                    # against the consumer's ``(8, 1, 1)`` formal.  Rule: a
                    # declared view of a storage is part of the value's
                    # identity and travels with every rebinding of it; the
                    # declaration is derived from the descriptor this pass
                    # read (as tensor_ssa_lowering declares a descriptor
                    # view) and the rebinding row states the view shape.
                    view_accounting: dict[str, Any] = {}
                    output_descriptor = output_descriptors[index]
                    storage_descriptor = None if table is None else table.by_id(int(callee_id))
                    if (
                        output_descriptor is not None
                        and storage_descriptor is not None
                        and (
                            output_descriptor.storage == "view"
                            or output_descriptor.alias_of is not None
                        )
                        and tuple(output_descriptor.shape or ())
                        and tuple(output_descriptor.shape)
                        != tuple(storage_descriptor.shape or ())
                    ):
                        view_accounting["ssa_storage_view"] = {
                            "storage_value_id": int(actual.id),
                            "view_shape": tuple(
                                int(extent)
                                for extent in output_descriptor.shape
                            ),
                            "operation": "tensor_descriptor",
                        }
                    replacement = SSAValue(
                        int(actual.id),
                        dtype=projection.value.dtype or actual.dtype,
                        shape=tuple(
                            projection.value.shape or actual.shape or ()
                        ),
                        device=projection.value.device or actual.device,
                        accounting={
                            **dict(actual.accounting or {}),
                            **dict(projection.value.accounting or {}),
                            "ssa_storage_alias": int(actual.id),
                            "ssa_aggregate_passthrough": (
                                callee_name, int(caller_id),
                            ),
                            **view_accounting,
                        },
                    )
                    _replace_exact_uses(
                        function, projection.value, replacement,
                        via=(callee_name, int(caller_id)),
                        sources=binding_cells,
                    )
                    passthrough_bindings.append((
                        int(output_position), int(caller_id),
                        int(callee_id), int(actual.id),
                        "exact_callee_output_formal",
                        "incumbent_on_equal_priority",
                    ))
                    removed.update((
                        id(projection.address), id(projection.load),
                    ))
                    if len(projection.address.args) > 1:
                        constant_candidates.append(
                            projection.address.args[1]
                        )
                    changed = True
                if len(retained) != len(caller_ids):
                    _remove_instructions(function, removed)
                    _remove_dead_projection_constants(
                        function, constant_candidates,
                    )
                    for key, values in (
                        ("output_ids", caller_ids),
                        ("output_positions", output_positions),
                        ("callee_output_ids", settled),
                    ):
                        call.attributes[key] = tuple(
                            values[index] for index in retained
                        )
                    for key in ("output_slots",):
                        values = tuple(call.attributes.get(key, ()))
                        if len(values) == len(caller_ids):
                            call.attributes[key] = tuple(
                                values[index] for index in retained
                            )
                    if not retained and not any(
                        argument is call.res
                        for consumer_block in function.blocks.values()
                        for consumer in consumer_block.instrs
                        for argument in consumer.args
                    ):
                        # Every semantic result is now the recorded actual
                        # storage. The empty abstract tuple has no consumer;
                        # retaining it invents an output parameter on the
                        # native call after its callee's outputs were folded.
                        from .ir_identities import _post_dead_structural_retirement
                        from .concordance_declarations import AGGREGATE_PASSTHROUGH_REBINDING
                        from .ssa_record_return_state import function_scope_of

                        _post_dead_structural_retirement(
                            function, call, sources=tuple(
                                cell for binding in passthrough_bindings
                                for cell in (book.latest_ref(
                                    AGGREGATE_PASSTHROUGH_REBINDING,
                                    (function_scope_of(function), int(binding[1])),
                                ),) if cell is not None
                            ),
                        )
                        call.res = None
                        call.attributes["result_convention"] = "void"
                elif settled != authored:
                    call.attributes["callee_output_ids"] = settled
                    changed = True
                if settled != authored or len(retained) != len(caller_ids):
                    call.attributes["aggregate_output_views_legalized"] = True
                if passthrough_bindings:
                    call.attributes[
                        "aggregate_output_passthrough_bindings"
                    ] = tuple(passthrough_bindings)
                if debug_this_call:
                    print(
                        "DEBUG-AGGREGATE-AFTER "
                        f"function={function.name!r} retained={retained!r} "
                        f"attributes={dict(call.attributes)!r}",
                        file=sys.stderr,
                    )
    return changed


__all__ = [
    "AggregateABIAnalysis", "AggregateCallRecord", "AggregateProjection",
    "analyze_aggregate_abi", "is_storage_view", "legalize_aggregate_adapters",
    "legalize_aggregate_output_views",
]
