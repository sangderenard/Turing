"""Explicit native representation for optional scalar function results."""

from __future__ import annotations

from typing import Any

from .monotonic_ids import GLOBAL_MONOTONIC_IDS
from ..transmogrifier.ssa import Instr, SSAValue


_RETURN_OPS = {"Ret", "ret", "Return", "return"}
_PHI_OPS = {"Phi", "phi"}
_NONE_OPS = {"NoneValue"}
_EQ_OPS = {"Eq", "eq", "Equal", "equal"}
_NE_OPS = {"Ne", "ne", "NotEqual", "not_equal"}


def _functions(module: Any) -> dict[str, Any]:
    functions = getattr(module, "functions", module)
    return dict(functions or {})

def _insert_before_terminator(block: Any, instructions: list[Instr]) -> None:
    index = len(block.instrs)
    if index and block.instrs[-1].op in {
        "Br", "br", "Branch", "branch", "CondBr", "condbr", *_RETURN_OPS,
    }:
        index -= 1
    block.instrs[index:index] = instructions


def lower_optional_scalar_returns(module: Any) -> tuple[dict[str, Any], ...]:
    """Split every proven scalar-or-None return into payload and presence.

    The semantic result identity remains the payload id. Its physical return
    layout becomes ``(payload, presence)``. Linked callers receive the pair as
    an ordinary aggregate and exact identity comparisons with ``None`` read the
    Boolean presence result. Unsupported or ambiguous shapes remain unchanged,
    so the existing optional-merge checker continues to reject them.
    """

    functions = _functions(module)
    contracts: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []

    for function_name, function in functions.items():
        producers = {
            int(instruction.res.id): instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        presence_by_payload: dict[int, SSAValue] = {}
        # Linked record fields have an explicit physical presence member.
        # Read the function-local descriptor, not copied accounting IDs from
        # the callee's namespace, when lowering tests of a returned field.
        record_table = (getattr(module, "record_tables", {}) or {}).get(function_name)
        if record_table is not None:
            values = {int(value.id): value for value in function.args}
            values.update({value_id: instruction.res for value_id, instruction in producers.items()})
            candidates: dict[int, set[int]] = {}
            for record in record_table.records.values():
                fields = {field.name: field for field in record.fields}
                for name, field in fields.items():
                    presence_field = fields.get(f"{name}.__present")
                    if (len(field.value_ids) == 1 and presence_field is not None
                            and len(presence_field.value_ids) == 1):
                        candidates.setdefault(int(field.value_ids[0]), set()).add(
                            int(presence_field.value_ids[0])
                        )
            for payload_id, presence_ids in candidates.items():
                if len(presence_ids) != 1:
                    continue
                presence_id = next(iter(presence_ids))
                if presence_id in values and values[presence_id].dtype == "bool":
                    presence_by_payload[payload_id] = values[presence_id]
        lowered_one = True
        while lowered_one:
            lowered_one = False
            for _block_name, block in function.blocks.items():
                for instruction in tuple(block.instrs):
                    if (
                        instruction.op not in _PHI_OPS
                        or instruction.res is None
                        or int(instruction.res.id) in presence_by_payload
                        or str(instruction.attributes.get("binding") or "")
                        in {"optional_value_presence", "optional_return_presence"}
                    ):
                        continue
                    absent = tuple(
                        str(value.dtype or "").casefold() == "none"
                        or (
                            int(value.id) in producers
                            and producers[int(value.id)].op in _NONE_OPS
                        )
                        for value in instruction.args
                    )
                    if not any(absent) or all(absent):
                        continue
                    payloads = tuple(
                        value for value, is_absent in zip(
                            instruction.args, absent, strict=True,
                        ) if not is_absent
                    )
                    signatures = {
                        (str(value.dtype or ""), tuple(value.shape or ()), value.device)
                        for value in payloads
                    }
                    # Scalar optionals have one physical payload shape. Span
                    # and record optionals also require ownership and remain
                    # visible to the existing hard gate.
                    if len(signatures) != 1 or any(value.shape for value in payloads):
                        continue
                    dtype, shape, device = next(iter(signatures))
                    if not dtype or dtype.casefold() in {"none", "unknown"}:
                        continue
                    incoming_blocks = tuple(instruction.attributes.get(
                        "incoming_blocks", (),
                    ))
                    if len(incoming_blocks) != len(instruction.args):
                        continue

                    payload_args: list[SSAValue] = []
                    presence_args: list[SSAValue] = []
                    for incoming, value, is_absent in zip(
                        incoming_blocks, instruction.args, absent, strict=True,
                    ):
                        predecessor = function.blocks.get(str(incoming))
                        if predecessor is None:
                            break
                        edge_instructions: list[Instr] = []
                        if is_absent:
                            payload = SSAValue(
                                GLOBAL_MONOTONIC_IDS.mint(),
                                dtype=dtype,
                                shape=shape,
                                device=device,
                                accounting={"optional_inactive_payload": True},
                            )
                            edge_instructions.append(Instr(
                                "Const", [], payload,
                                attributes={
                                    "value": (
                                        False if dtype.casefold() == "bool" else 0
                                    ),
                                    "optional_inactive_payload": True,
                                },
                            ))
                        else:
                            payload = value
                        present = presence_by_payload.get(int(value.id))
                        if present is None:
                            present = SSAValue(
                                GLOBAL_MONOTONIC_IDS.mint(),
                                dtype="bool",
                                accounting={"ssa_optional_presence_edge": True},
                            )
                            edge_instructions.append(Instr(
                                "Const", [], present,
                                attributes={
                                    "value": not is_absent,
                                    "ssa_optional_presence_edge": True,
                                },
                            ))
                        _insert_before_terminator(
                            predecessor, edge_instructions
                        )
                        payload_args.append(payload)
                        presence_args.append(present)
                    else:
                        is_return = str(instruction.attributes.get(
                            "binding"
                        ) or "") == "return_merge"
                        presence = SSAValue(
                            GLOBAL_MONOTONIC_IDS.mint(),
                            dtype="bool",
                            accounting={
                                "ssa_optional_presence": True,
                                "ssa_optional_payload_id": int(instruction.res.id),
                            },
                        )
                        instruction.args = payload_args
                        instruction.res.dtype = dtype
                        instruction.res.shape = shape
                        instruction.res.device = device
                        instruction.res.accounting = {
                            **dict(instruction.res.accounting or {}),
                            "ssa_optional_payload": True,
                            "ssa_optional_presence_id": int(presence.id),
                        }
                        presence_phi = Instr(
                            "Phi", presence_args, presence,
                            attributes={
                                "incoming_blocks": incoming_blocks,
                                "binding": (
                                    "optional_return_presence" if is_return
                                    else "optional_value_presence"
                                ),
                                "payload_value_id": int(instruction.res.id),
                            },
                        )
                        live_index = block.instrs.index(instruction)
                        block.instrs.insert(live_index + 1, presence_phi)
                        presence_by_payload[int(instruction.res.id)] = presence
                        receipt = {
                            "function": str(function_name),
                            "payload_value_id": int(instruction.res.id),
                            "presence_value_id": int(presence.id),
                            "dtype": dtype,
                            "shape": shape,
                            "priority": (
                                "exact_mixed_return_phi" if is_return
                                else "exact_mixed_scalar_phi"
                            ),
                            "tie_policy": "incumbent",
                        }
                        receipts.append(receipt)
                        if is_return:
                            for candidate_block in function.blocks.values():
                                for terminator in candidate_block.instrs:
                                    if terminator.op not in _RETURN_OPS:
                                        continue
                                    expanded = []
                                    for value in terminator.args:
                                        if (
                                            value is instruction.res
                                            or int(value.id)
                                            == int(instruction.res.id)
                                        ):
                                            expanded.extend((
                                                instruction.res, presence,
                                            ))
                                        else:
                                            expanded.append(value)
                                    terminator.args = expanded
                            layouts = dict(function.metadata.get(
                                "aggregate_return_layouts", (),
                            ))
                            layouts[int(instruction.res.id)] = (
                                int(instruction.res.id), int(presence.id),
                            )
                            function.metadata[
                                "aggregate_return_layouts"
                            ] = tuple(layouts.items())
                            contracts[str(function_name)] = receipt
                        lowered_one = True
                        break
                if lowered_one:
                    break

        # Identity tests of a local optional Phi consume its local presence
        # value just like tests of an optional Call result do below.
        for block in function.blocks.values():
            for consumer in block.instrs:
                if consumer.op not in (_EQ_OPS | _NE_OPS):
                    continue
                payload = next((
                    value for value in consumer.args
                    if int(value.id) in presence_by_payload
                ), None)
                if payload is None:
                    continue
                other = tuple(
                    value for value in consumer.args
                    if value is not payload
                )
                if len(other) != 1:
                    continue
                none_producer = producers.get(int(other[0].id))
                if none_producer is None or none_producer.op not in _NONE_OPS:
                    continue
                presence = presence_by_payload[int(payload.id)]
                consumer.args = [presence]
                consumer.op = "LNot" if consumer.op in _EQ_OPS else "Cast"
                consumer.attributes.update({
                    "target_dtype": "bool",
                    "optional_presence_test": True,
                    "payload_value_id": int(payload.id),
                    "presence_value_id": int(presence.id),
                    "priority": "exact_optional_phi",
                    "tie_policy": "incumbent",
                })

    for caller_name, caller in functions.items():
        producers = {
            int(instruction.res.id): instruction
            for block in caller.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        for block in caller.blocks.values():
            index = 0
            while index < len(block.instrs):
                call = block.instrs[index]
                contract = contracts.get(str(call.attributes.get("callee") or ""))
                if (
                    call.op not in {"Call", "call"}
                    or call.res is None
                    or contract is None
                ):
                    index += 1
                    continue
                payload = call.res
                payload.dtype = str(contract["dtype"])
                payload.shape = tuple(contract["shape"])
                payload.accounting = {
                    **dict(payload.accounting or {}),
                    "ssa_optional_payload": True,
                }
                presence = SSAValue(
                    GLOBAL_MONOTONIC_IDS.mint(),
                    dtype="bool",
                    accounting={
                        "ssa_optional_presence": True,
                        "ssa_optional_payload_id": int(payload.id),
                    },
                )
                aggregate = SSAValue(
                    GLOBAL_MONOTONIC_IDS.mint(),
                    dtype="ssa.aggregate",
                    shape=(2,),
                    accounting={
                        "ssa_aggregate_outputs": (payload, presence),
                        "ssa_optional_result": True,
                    },
                )
                projection: list[Instr] = []
                for position, output in enumerate((payload, presence)):
                    position_value = SSAValue(
                        GLOBAL_MONOTONIC_IDS.mint(),
                        dtype="int",
                    )
                    address = SSAValue(
                        GLOBAL_MONOTONIC_IDS.mint(),
                        dtype="ptr",
                    )
                    projection.extend((
                        Instr("Const", [], position_value,
                              attributes={"value": position}),
                        Instr("GetElementPtr", [aggregate, position_value], address,
                              attributes={"aggregate_index": position}),
                        Instr("Load", [address], output, attributes={
                            "aggregate_index": position,
                            "source_output_id": int(output.id),
                            "ssa_optional_result": True,
                        }),
                    ))
                call.res = aggregate
                call.attributes.update({
                    "result_convention": "ssa.aggregate",
                    "output_ids": (int(payload.id), int(presence.id)),
                    "callee_output_ids": (
                        int(contract["payload_value_id"]),
                        int(contract["presence_value_id"]),
                    ),
                    "native_result_contract": (
                        (int(contract["payload_value_id"]),
                         str(contract["dtype"]), tuple(contract["shape"])),
                        (int(contract["presence_value_id"]), "bool", ()),
                    ),
                    "ssa_optional_result": True,
                })
                block.instrs[index + 1:index + 1] = projection
                index += 1 + len(projection)

                for consumer_block in caller.blocks.values():
                    for consumer in consumer_block.instrs:
                        if consumer.op not in (_EQ_OPS | _NE_OPS):
                            continue
                        if not any(int(value.id) == int(payload.id)
                                   for value in consumer.args):
                            continue
                        other = tuple(
                            value for value in consumer.args
                            if int(value.id) != int(payload.id)
                        )
                        if len(other) != 1:
                            continue
                        none_producer = producers.get(int(other[0].id))
                        if none_producer is None or none_producer.op not in _NONE_OPS:
                            continue
                        consumer.args = [presence]
                        consumer.op = (
                            "LNot" if consumer.op in _EQ_OPS else "Cast"
                        )
                        consumer.attributes.update({
                            "target_dtype": "bool",
                            "optional_presence_test": True,
                            "payload_value_id": int(payload.id),
                            "presence_value_id": int(presence.id),
                            "priority": "exact_optional_call_result",
                            "tie_policy": "incumbent",
                        })
                receipts.append({
                    **contract,
                    "caller": str(caller_name),
                    "callsite_id": call.attributes.get("plan_callsite_id"),
                    "caller_payload_value_id": int(payload.id),
                    "caller_presence_value_id": int(presence.id),
                    "aggregate_value_id": int(aggregate.id),
                })
                index += 1

    if receipts:
        metadata = getattr(module, "metadata", None)
        if metadata is None:
            module.metadata = {}
            metadata = module.metadata
        metadata["optional_scalar_return_receipts"] = tuple(receipts)
    return tuple(receipts)


__all__ = ["lower_optional_scalar_returns"]
