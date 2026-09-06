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
from typing import Iterable, Mapping

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


def is_storage_view(value: SSAValue, storage: SSAValue) -> bool:
    """Whether ``value`` is an explicitly correlated view of ``storage``.

    Ordered call views are distinct objects so they may carry per-position
    shapes.  Numeric equality alone is unsafe because linker/planner numbering
    domains can collide; the ``ssa_storage_alias`` receipt is authoritative.
    """

    return value is storage or (
        (value.accounting or {}).get("ssa_storage_alias") == int(storage.id)
    )


def _function_values(function: Function) -> dict[int, SSAValue]:
    values = {int(value.id): value for value in function.args}
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
        ), ())
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
        if isinstance(value, (bool, int, float)):
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


def _replace_exact_uses(
    function: Function, source: SSAValue, replacement: SSAValue,
) -> None:
    for block in function.blocks.values():
        for instruction in block.instrs:
            instruction.args[:] = [
                replacement
                if argument is source or is_storage_view(argument, source)
                else argument
                for argument in instruction.args
            ]


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
                    _replace_exact_uses(caller, projection.value, replacement)
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
    for function in module.functions.values():
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
                if table is None:
                    continue
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
                for output_id in authored:
                    current = int(output_id)
                    seen: set[int] = set()
                    while current not in seen:
                        seen.add(current)
                        descriptor = table.by_id(current)
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
                    has_direct_consumer = any(
                        int(argument.id) == int(caller_id)
                        and argument is not projection.value
                        and not is_storage_view(argument, projection.value)
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
                        },
                    )
                    _replace_exact_uses(
                        function, projection.value, replacement,
                    )
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
                elif settled != authored:
                    call.attributes["callee_output_ids"] = settled
                    changed = True
                if settled != authored or len(retained) != len(caller_ids):
                    call.attributes["aggregate_output_views_legalized"] = True
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
