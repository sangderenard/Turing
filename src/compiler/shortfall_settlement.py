"""Shortfall settlement: settle non-live structural shortfalls, rebind recorded scalar identities, reconcile region-local view shortfalls (relocated from fortran_c_shell)"""

from __future__ import annotations
from typing import (
    Any,
    Iterable,
    Mapping,
)


def _retained_parameter_identity(
    parameter_name: str,
    identity_history: Iterable[int],
    retained_parameters: Iterable[tuple[str, int]],
) -> int | None:
    """Select the physical formal retained for one authored parameter.

    Source identity history includes structural placeholders as well as the
    later physical projection that survives repository-SSA lowering.  The
    function's ``parameter_names`` receipt identifies that retained formal;
    using the first historical id can bind a call to a discarded aggregate
    wrapper and leave the real scalar formal backed by zeroed scratch storage.
    """

    retained = tuple(
        int(value_id)
        for name, value_id in retained_parameters
        if str(name) == str(parameter_name)
    )
    if len(set(retained)) == 1:
        return retained[0]
    history = tuple(map(int, identity_history))
    return history[0] if history else None


def _settle_nonlive_structural_shortfalls(
    function: Any,
    shortfalls: Iterable[Any],
    *,
    authoritative_output_ids: Iterable[int] = (),
    required_source_ids: Iterable[int] = (),
) -> tuple[Any, ...]:
    """Discard failed recursive probes that are absent from the final SSA.

    Structural recovery explores source dependencies speculatively. A nested
    call handle can be unavailable as one Python object even though its linked
    physical fields satisfy the enclosing output, and an evaporated collection
    can disappear before its ``sorted``/``tuple`` wrappers are needed. Those
    failed probes are not defects once no formal, instruction operand, required
    source, or authoritative output names the identity. Direct live failures
    remain hard findings.
    """
    live_ids = {
        *(int(value.id) for value in function.args),
        *(int(argument.id)
          for block in function.blocks.values()
          for instruction in block.instrs
          for argument in instruction.args),
        *map(int, authoritative_output_ids),
        *map(int, required_source_ids),
    }
    retained = []
    settled = []
    for finding in shortfalls:
        if (
            isinstance(finding, tuple)
            and len(finding) == 3
            and isinstance(finding[0], int)
            and int(finding[0]) not in live_ids
        ):
            settled.append({
                "value_id": int(finding[0]),
                "operation": str(finding[1]),
                "reason": str(finding[2]),
                "resolution": "absent_from_final_ssa",
                "priority": "exact_live_value_surface",
                "tie_policy": "incumbent",
            })
            continue
        retained.append(finding)
    if settled:
        prior = tuple(function.metadata.get(
            "settled_nonlive_structural_shortfalls", ()
        ))
        function.metadata["settled_nonlive_structural_shortfalls"] = (
            *prior,
            *(receipt for receipt in settled if receipt not in prior),
        )
    return tuple(retained)


def _rebind_recorded_scalar_identities(
    function: Any, source_graph: Any | None = None,
) -> tuple[dict[str, Any], ...]:
    """Apply proven scalar ``item()`` identities to every stale SSA object.

    Structural recovery can retire one provisional formal while planned-region
    calls still hold distinct ``SSAValue`` objects carrying the same source
    id.  ``control_identity_receipts`` is the proof that the rank-zero item is
    exactly its operand.  Rebind only an unambiguous receipt to one canonical
    formal or definition; conflicting receipts and duplicate definitions keep
    the incumbent.
    """
    receipts_by_source: dict[int, set[int]] = {}
    for receipt in function.metadata.get("control_identity_receipts", ()):
        if len(receipt) != 3 or str(receipt[2]) != "scalar_item_identity":
            continue
        source_id, replacement_id = map(int, receipt[:2])
        if source_id != replacement_id:
            receipts_by_source.setdefault(source_id, set()).add(replacement_id)

    candidates: dict[int, list[Any]] = {}
    for value in function.args:
        candidates.setdefault(int(value.id), []).append(value)
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.res is not None:
                bucket = candidates.setdefault(int(instruction.res.id), [])
                if all(instruction.res is not value for value in bucket):
                    bucket.append(instruction.res)

    canonical: dict[int, Any] = {}
    for source_id, replacement_ids in receipts_by_source.items():
        if len(replacement_ids) != 1:
            continue
        replacement_id = next(iter(replacement_ids))
        choices = candidates.get(replacement_id, ())
        if len(choices) == 1:
            canonical[source_id] = choices[0]

    changes = []
    for block_name, block in function.blocks.items():
        for instruction_index, instruction in enumerate(block.instrs):
            replaced_positions = []
            rewritten = []
            for position, argument in enumerate(instruction.args):
                replacement = canonical.get(int(argument.id))
                if replacement is None or replacement is argument:
                    rewritten.append(argument)
                    continue
                rewritten.append(replacement)
                replaced_positions.append((
                    position, int(argument.id), int(replacement.id)
                ))
            if not replaced_positions:
                continue
            instruction.args = rewritten
            if "feed_ids" in instruction.attributes:
                instruction.attributes["feed_ids"] = tuple(
                    int(argument.id) for argument in rewritten
                )
            for position, source_id, replacement_id in replaced_positions:
                changes.append({
                    "block": str(block_name),
                    "instruction_index": int(instruction_index),
                    "operand_index": int(position),
                    "value_id": source_id,
                    "replacement_value_id": replacement_id,
                    "priority": "exact_structural_identity",
                    "tie_policy": "incumbent",
                })
    if changes:
        prior = tuple(function.metadata.get(
            "structural_identity_rebindings", ()
        ))
        function.metadata["structural_identity_rebindings"] = (
            *prior,
            *(change for change in changes if change not in prior),
        )
        rebound_targets = {
            int(change["replacement_value_id"]) for change in changes
        }
        existing_names = tuple(function.metadata.get("parameter_names", ()))
        names_by_id = {int(value_id): str(name) for name, value_id in existing_names}
        ids_by_name = {str(name): int(value_id) for name, value_id in existing_names}
        recovered_names = []
        for replacement_id in sorted(rebound_targets):
            replacement = canonical.get(next((
                source_id for source_id, value in canonical.items()
                if int(value.id) == replacement_id
            ), -1))
            if replacement not in function.args or source_graph is None:
                continue
            matching_nodes = tuple(
                data
                for node_id, data in source_graph.nodes(data=True)
                if int(data.get("value_id", node_id)) == replacement_id
            )
            if len(matching_nodes) != 1:
                continue
            attributes = matching_nodes[0].get("attributes") or {}
            if str(attributes.get("binding_kind")) != "parameter":
                continue
            name = str(attributes.get("binding_name") or "")
            if not name:
                continue
            if replacement_id in names_by_id or name in ids_by_name:
                continue
            existing_names = (*existing_names, (name, replacement_id))
            names_by_id[replacement_id] = name
            ids_by_name[name] = replacement_id
            recovered_names.append({
                "name": name,
                "value_id": replacement_id,
                "priority": "exact_source_parameter_identity",
                "tie_policy": "incumbent",
            })
        if recovered_names:
            function.metadata["parameter_names"] = existing_names
            prior_names = tuple(function.metadata.get(
                "recovered_structural_parameter_names", ()
            ))
            function.metadata["recovered_structural_parameter_names"] = (
                *prior_names,
                *(entry for entry in recovered_names if entry not in prior_names),
            )
    return tuple(changes)


def _reconcile_region_local_view_shortfalls(
    symbol: str,
    graph: Any,
    function: Any,
    all_functions: Mapping[str, Any],
    all_tensor_tables: Mapping[str, Any],
    structural_shortfalls: Iterable[Any],
    semantic_output_ids: Iterable[int],
) -> tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...]]:
    """Close view findings already proved inside an owned numerical region.

    Region lowering deliberately erases a reshape instruction when its result
    is only a different static view of the same resident allocation.  The
    region tensor table is the durable proof of that transformation: it names
    the source identity, storage identity, allocation owner, byte extent and
    target shape.  A later wrapper-only structural scan cannot see the erased
    instruction and used to report the identity as missing.

    Consume only exact ``reshape``/``view`` ``operator`` findings for values
    which are internal to the wrapper and have one complete owned-region
    receipt.  Public semantic outputs and wrapper-visible values still require
    an actual wrapper definition.  Equal candidate receipts do not replace an
    incumbent: ambiguity leaves the finding open.
    """

    semantic_outputs = {int(value_id) for value_id in semantic_output_ids}
    wrapper_value_ids = {int(value.id) for value in function.args}
    wrapper_value_ids.update(
        int(instruction.res.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    )
    region_prefix = f"{symbol}__planned_region_"
    remaining: list[Any] = []
    receipts: list[tuple[Any, ...]] = []

    for finding in structural_shortfalls:
        if not (
            isinstance(finding, tuple)
            and len(finding) == 3
            and isinstance(finding[0], int)
            and str(finding[1]).casefold() in {"reshape", "view"}
            and str(finding[2]) == "operator"
        ):
            remaining.append(finding)
            continue
        value_id = int(finding[0])
        if value_id in semantic_outputs or value_id in wrapper_value_ids:
            remaining.append(finding)
            continue
        node = graph.nodes.get(value_id, {})
        operation = str(
            node.get("op") or node.get("type") or ""
        ).casefold()
        if operation not in {"reshape", "view"}:
            remaining.append(finding)
            continue

        candidates: list[tuple[Any, ...]] = []
        for region_symbol, tensor_table in all_tensor_tables.items():
            region_symbol = str(region_symbol)
            if not region_symbol.startswith(region_prefix):
                continue
            descriptor = tensor_table.by_id(value_id)
            if descriptor is None or descriptor.storage != "view":
                continue
            storage_id = int(descriptor.data_value_id)
            if (
                descriptor.owns_allocation
                or descriptor.alias_of is None
                or int(descriptor.alias_of) != storage_id
                or descriptor.allocation_owner is None
            ):
                continue
            storage = tensor_table.by_id(storage_id)
            region = all_functions.get(region_symbol)
            if storage is None or region is None:
                continue
            if int(storage.data_value_id) != storage_id:
                continue
            if descriptor.dtype != storage.dtype:
                continue
            if (
                descriptor.byte_size is not None
                and storage.byte_size is not None
                and int(descriptor.byte_size) != int(storage.byte_size)
            ):
                continue
            region_value_ids = {int(value.id) for value in region.args}
            region_value_ids.update(
                int(instruction.res.id)
                for block in region.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            )
            if storage_id not in region_value_ids:
                continue
            candidates.append((
                value_id,
                operation,
                region_symbol,
                storage_id,
                tuple(map(int, descriptor.shape)),
            ))

        # One proof owns the identity. Multiple regions bearing the same
        # integer are not interchangeable scopes, so a tie keeps the open
        # finding as the incumbent state.
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            remaining.append(finding)
            continue
        receipts.append(candidates[0])

    return tuple(remaining), tuple(dict.fromkeys(receipts))


def _intern_writable_region_outputs(function: Any, tensor_table: Any = None) -> None:
    """Keep declared writable output storage attached to its SSA producer."""
    region_integral = dict(
        (function.metadata or {}).get("source_region_integral") or {}
    )
    declared_region_inouts = (
        set(map(int, region_integral.get("capture_value_ids", ())))
        & set(map(int, region_integral.get("output_value_ids", ())))
    )
    tensor_outputs = {
        int(descriptor.data_value_id)
        for descriptor in (() if tensor_table is None else tensor_table.tensors.values())
        if descriptor.storage == "output" and descriptor.writable
    }
    writable_formals = {
        int(argument.id): argument
        for argument in function.args
        if (
            int(argument.id) in declared_region_inouts
            or (
                bool((argument.accounting or {}).get("program_abi_mutable"))
                and (
                    int(argument.id) in tensor_outputs
                    or bool((argument.accounting or {}).get(
                        "program_abi_field_written"
                    ))
                )
                and (argument.accounting or {}).get("program_abi_storage")
            )
        )
    }
    if not writable_formals:
        return
    replacements: dict[int, Any] = {}
    for block in function.blocks.values():
        for instruction in block.instrs:
            result = instruction.res
            if result is None:
                continue
            formal = writable_formals.get(int(result.id))
            if formal is None or result is formal:
                continue
            replacements[id(result)] = formal
            if formal.dtype in {None, "unknown"} and result.dtype:
                formal.dtype = result.dtype
            if not tuple(formal.shape or ()) and tuple(result.shape or ()):
                formal.shape = tuple(result.shape)
            formal.accounting = {
                **dict(result.accounting or {}),
                **dict(formal.accounting or {}),
            }
            instruction.res = formal
    if not replacements:
        return
    for block in function.blocks.values():
        for instruction in block.instrs:
            instruction.args = [
                replacements.get(id(argument), argument)
                for argument in instruction.args
            ]
    function.metadata["interned_program_abi_output_ids"] = tuple(
        sorted({int(value.id) for value in replacements.values()})
    )
