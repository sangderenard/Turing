"""Resident sequence materialisation: loop-carried storage aliases, literal and append mutations, capacity bounds (relocated from fortran_c_shell)."""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import replace
from typing import Any, Iterable, Mapping


def _loop_carried_storage_aliases(graph_obj) -> dict[int, int]:
    """Graph-derived storage aliases for loop-carried in-place mutation.

    A loop that mutates an array through ``IndexedStore`` leaves a
    LOOPRESULT node as the array's post-loop identity. That identity is
    the SAME STORAGE as the array it mutated -- in-place is the point --
    but the planner schedules later consumers against the loopresult id,
    and a lowering that cannot link it materializes a fresh, unconnected
    FORMAL: the second of two sequential loops over one array then reads
    the original buffer and the first loop's stores silently vanish (the
    sequential same-array store defect, pinned in
    ``test_compiled_linalg.py``).

    The chase is deliberately narrow, one level per node, composed by the
    builder's existing alias-chain walk (``external_value``):

    * a ``loopresult``/``loopexit`` node aliases its preferred-role parent
      ONLY when that parent is itself an ``IndexedStore`` version or
      another loop identity -- scalar carried results (a ``total``) are
      genuine values with carried-port machinery of their own and are
      never touched;
    * an ``IndexedStore`` node aliases its ``base`` -- the store versions
      resident memory, it does not mint storage (the same rule
      ``ir_indexing`` applies inside a function, extended to the graph
      so it holds ACROSS loops).
    """

    aliases: dict[int, int] = {}
    chainable = {"indexedstore", "loopresult", "loopexit"}

    def node_kind(node_id) -> str:
        data = graph_obj.nodes.get(node_id) or {}
        return str(data.get("op") or data.get("type") or "").casefold()

    def value_id_of(node_id) -> int:
        data = graph_obj.nodes.get(node_id) or {}
        return int(data.get("value_id", node_id))

    for node_id, data in graph_obj.nodes(data=True):
        kind = str(data.get("op") or data.get("type") or "").casefold()
        parents = tuple(data.get("parents") or ())
        if kind == "indexedstore":
            base = next(
                (parent for parent, role in parents
                 if str(role) == "base" and parent in graph_obj),
                None,
            )
            if base is not None:
                aliases[value_id_of(node_id)] = value_id_of(base)
            continue
        if kind in {"loopresult", "loopexit"}:
            for preferred_role in (
                "updated", "value", "body", "initial", "orelse"
            ):
                parent = next(
                    (parent for parent, role in parents
                     if str(role) == preferred_role
                     and parent in graph_obj),
                    None,
                )
                if parent is None:
                    continue
                if node_kind(parent) in chainable:
                    aliases[value_id_of(node_id)] = value_id_of(parent)
                break
    def root(value):
        seen = set()
        while value in aliases and value not in seen:
            seen.add(value)
            value = aliases[value]
        return value

    changed = True
    while changed:
        changed = False
        for node_id, data in graph_obj.nodes(data=True):
            if node_kind(node_id) != "phi":
                continue
            parents = [root(value_id_of(parent)) for parent, role in data.get("parents") or ()
                       if role in {"body", "orelse"} and parent in graph_obj]
            if len(parents) != 2 or parents[0] != parents[1]:
                continue
            resident = parents[0]
            if (graph_obj.nodes.get(resident, {}).get("attributes") or {}).get("aggregate_kind") != "dict":
                continue
            value = value_id_of(node_id)
            if value != resident and aliases.get(value) != resident:
                aliases[value] = resident
                changed = True
    aliases = {source: root(target) for source, target in aliases.items()}
    # Never allow a cycle to reach the builder's chase (it guards with a
    # seen-set, but a self-alias is meaningless regardless).
    return {
        source: target for source, target in aliases.items()
        if source != target
    }


def _promote_conditional_sequence_aliases(
    control: Any,
    graph_obj: Any,
    *,
    call_result_kinds: Mapping[int, str] | None = None,
    structural_aliases: Mapping[int, tuple[int, str]] | None = None,
):
    """Turn branch-carried aggregate versions into resident arena carries.

    Ordinary conditional planning records every authored assignment in the
    scalar-shaped ``carried_aliases`` tuple.  Once source/call analysis proves
    those values are sequences, retain one initial arena, copy the selected
    branch value into it, and correlate the post-branch SSA spelling with that
    arena.  This is an IR correction, not a receipt-only relabeling.
    """

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock, WhileBlock,
    )

    sequence_kinds = {"list", "bytes", "bytearray"}
    resident_by_value: dict[int, int] = {}
    kind_by_value: dict[int, str] = {}
    for node_id, data in graph_obj.nodes(data=True):
        attributes = data.get("attributes") or {}
        kind = str(attributes.get("aggregate_kind") or "")
        if kind not in sequence_kinds:
            continue
        value_id = int(data.get("value_id", node_id))
        resident_by_value[value_id] = value_id
        kind_by_value[value_id] = kind
    for value_id, kind in dict(call_result_kinds or {}).items():
        if str(kind) in sequence_kinds:
            resident_by_value[int(value_id)] = int(value_id)
            kind_by_value[int(value_id)] = str(kind)
    for value_id, (resident_id, kind) in dict(
        structural_aliases or {}
    ).items():
        if str(kind) not in sequence_kinds:
            continue
        resident_by_value[int(value_id)] = int(resident_id)
        kind_by_value[int(value_id)] = str(kind)
        kind_by_value.setdefault(int(resident_id), str(kind))

    # In-place stores and their Phi/loop-result spellings are storage
    # versions, not scalar values.  Resolve them to their original arena before
    # inspecting conditional carries, otherwise the branch becomes an opaque
    # scalar Phi and every later row projection leaks into the public ABI.
    storage_aliases = _loop_carried_storage_aliases(graph_obj)
    changed = True
    while changed:
        changed = False
        for alias_id, source_id in storage_aliases.items():
            resident_id = resident_by_value.get(int(source_id))
            kind = kind_by_value.get(int(source_id))
            if resident_id is None or kind not in sequence_kinds:
                continue
            if resident_by_value.get(int(alias_id)) != int(resident_id):
                resident_by_value[int(alias_id)] = int(resident_id)
                kind_by_value[int(alias_id)] = str(kind)
                changed = True
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("op") or data.get("type") or "").casefold() not in {
                "phi", "loopresult", "loopexit",
            }:
                continue
            value_id = int(data.get("value_id", node_id))
            resolved_parents = {
                (
                    int(resident_by_value[parent_value]),
                    str(kind_by_value[parent_value]),
                )
                for parent, role in data.get("parents") or ()
                if str(role) not in {"control", "test"}
                and parent in graph_obj
                for parent_value in (
                    int(graph_obj.nodes[parent].get("value_id", parent)),
                )
                if parent_value in resident_by_value
                and parent_value in kind_by_value
            }
            if len(resolved_parents) != 1:
                continue
            resident_id, kind = next(iter(resolved_parents))
            if resident_by_value.get(value_id) != resident_id:
                resident_by_value[value_id] = resident_id
                kind_by_value[value_id] = kind
                changed = True

    promoted_aliases: dict[int, tuple[int, str]] = {}
    destination_ids: set[int] = set()

    def promote(block):
        if isinstance(block, ConditionalBlock):
            scalar = []
            # ``carried_field_cells`` is index-aligned with
            # ``carried_aliases`` (the control builder zips them).  A
            # promoted alias leaves ``carried_aliases``, so its field cells
            # leave with it, in lockstep (as the retained-values projection
            # in control_source does).  Dropping only the alias shifted every
            # later alias onto its predecessor's cells: ``rejected = False``
            # after ``m.unresolved_report = list(lines)`` was lowered as the
            # field merge of ``unresolved_report`` (dt_controller
            # step_with_dt_control_used; ConcordanceRefusal on
            # ssa_field_version in ``_carried_field_arm``).
            scalar_field_cells = []
            field_cells = tuple(block.carried_field_cells) + (None,) * (
                len(block.carried_aliases) - len(block.carried_field_cells)
            )
            sequences = list(block.carried_sequence_aliases)
            for carried, cells in zip(block.carried_aliases, field_cells):
                true_id, false_id, initial_id, merged_id = map(int, carried)
                initial_resident = resident_by_value.get(initial_id)
                initial_kind = kind_by_value.get(initial_id)
                true_resident = resident_by_value.get(true_id)
                false_resident = resident_by_value.get(false_id)
                true_kind = kind_by_value.get(true_id)
                false_kind = kind_by_value.get(false_id)
                if (
                    initial_resident is None
                    or initial_kind not in sequence_kinds
                    or true_resident is None
                    or false_resident is None
                    or true_kind != initial_kind
                    or false_kind != initial_kind
                ):
                    scalar.append(carried)
                    scalar_field_cells.append(cells)
                    continue
                destination = int(initial_resident)
                sequences.append((
                    int(true_resident), int(false_resident),
                    destination, merged_id,
                ))
                destination_ids.add(destination)
                resident_by_value[merged_id] = destination
                kind_by_value[merged_id] = initial_kind
                promoted_aliases[merged_id] = (destination, initial_kind)
            return replace(
                block,
                body=promote(block.body),
                orelse=(
                    None if block.orelse is None else promote(block.orelse)
                ),
                carried_aliases=tuple(scalar),
                carried_field_cells=tuple(scalar_field_cells),
                carried_sequence_aliases=tuple(sequences),
            )
        if isinstance(block, SequenceBlock):
            return replace(block, blocks=tuple(promote(x) for x in block.blocks))
        if isinstance(block, LoopBlock):
            promoted_body = promote(block.body)
            scalar_carries = []
            sequence_updates: set[tuple[int, int]] = set()
            for updated_id, initial_id in block.carried_aliases:
                updated_id = int(updated_id)
                initial_id = int(initial_id)
                updated_resident = resident_by_value.get(updated_id)
                initial_resident = resident_by_value.get(initial_id)
                if (
                    updated_resident is None
                    or initial_resident is None
                    or updated_resident != initial_resident
                    or kind_by_value.get(updated_id) not in sequence_kinds
                    or kind_by_value.get(initial_id) not in sequence_kinds
                ):
                    scalar_carries.append((updated_id, initial_id))
                    continue
                sequence_updates.add((updated_id, initial_id))
                kind = str(kind_by_value[initial_id])
                promoted_aliases[updated_id] = (
                    int(initial_resident), kind,
                )
                destination_ids.add(int(initial_resident))
            scalar_ports = []
            for port_id, initial_id, updated_id in block.result_ports:
                key = (int(updated_id), int(initial_id))
                if key not in sequence_updates:
                    scalar_ports.append((
                        int(port_id), int(initial_id), int(updated_id)
                    ))
                    continue
                resident_id = int(resident_by_value[int(initial_id)])
                kind = str(kind_by_value[int(initial_id)])
                resident_by_value[int(port_id)] = resident_id
                kind_by_value[int(port_id)] = kind
                promoted_aliases[int(port_id)] = (resident_id, kind)
            return replace(
                block,
                body=promoted_body,
                carried_aliases=tuple(scalar_carries),
                result_ports=tuple(scalar_ports),
            )
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=promote(block.condition),
                body=promote(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=promote(block.callee))
        return block

    promoted = replace(control, root=promote(control.root))
    return promoted, promoted_aliases, tuple(sorted(destination_ids))


def _constant_struct_pack_materializations(graph_obj: Any):
    """Fold fully source-static ``struct.pack`` calls to byte-sequence facts."""

    import struct

    materializations = []
    for node_id, data in sorted(graph_obj.nodes(data=True)):
        attributes = data.get("attributes") or {}
        if attributes.get("extraction_identity") != "_struct.pack":
            continue
        arguments = tuple(
            int(parent)
            for parent, role in sorted(
                data.get("parents") or (),
                key=lambda item: int(str(item[1]).split(":", 1)[1])
                if str(item[1]).startswith("arg:") else 1 << 30,
            )
            if str(role).startswith("arg:") and int(parent) in graph_obj
        )
        values = []
        for argument in arguments:
            argument_data = graph_obj.nodes[argument]
            argument_attributes = argument_data.get("attributes") or {}
            if "value" in argument_attributes:
                values.append(argument_attributes["value"])
            elif argument_data.get("constant") is not None:
                values.append(argument_data["constant"])
            else:
                break
        if len(values) != len(arguments) or not values:
            continue
        try:
            payload = struct.pack(*values)
        except (struct.error, TypeError, ValueError):
            continue
        value_id = int(data.get("value_id", node_id))
        updated = dict(attributes)
        updated.update({
            "producer_kind": "constant_sequence",
            "aggregate_kind": "bytes",
            "sequence_key_columns": (),
            "sequence_column_count": 1,
            "sequence_writable": False,
            "constant_bytes": bytes(payload),
        })
        data["attributes"] = updated
        materializations.append((
            value_id,
            bytes(payload),
            int(node_id),
            "_struct.pack",
        ))
    return tuple(materializations)


def _constant_byte_literal_materializations(graph_obj: Any):
    """Publish authored byte literals through the resident sequence ABI."""

    materializations = []
    for node_id, data in sorted(graph_obj.nodes(data=True)):
        attributes = data.get("attributes") or {}
        literal = attributes.get("value", data.get("constant"))
        if not isinstance(literal, bytes):
            continue
        value_id = int(data.get("value_id", node_id))
        updated = dict(attributes)
        updated.update({
            "producer_kind": "constant_sequence",
            "aggregate_kind": "bytes",
            "sequence_key_columns": (),
            "sequence_column_count": 1,
            "sequence_writable": False,
            "constant_bytes": bytes(literal),
        })
        data["attributes"] = updated
        materializations.append((
            value_id, bytes(literal), int(node_id), None,
        ))
    return tuple(materializations)


def _source_sequence_snapshots(graph_obj: Any):
    """Tuple conversion captures contents and length in separate storage."""
    from .control_source import ControlSequenceMutation
    mutations = []
    for node_id, data in graph_obj.nodes(data=True):
        attrs = data.get("attributes") or {}
        if (attrs.get("aggregate_kind") != "tuple"
                or attrs.get("producer_kind") != "aggregate_materialization"):
            continue
        sources = [int(parent) for parent, role in data.get("parents") or ()
                   if str(role) == "arg:0"]
        if len(sources) != 1:
            continue
        source = sources[0]
        if (graph_obj.nodes.get(source, {}).get("attributes") or {}).get("aggregate_kind") not in {"list", "tuple"}:
            continue
        mutations.append(ControlSequenceMutation(
            int(node_id), "replace", (source,), int(node_id), policy="duplicates",
        ))
    return tuple(mutations)


def _source_mapping_mutations(graph_obj: Any):
    """Retain copies and writes to graph-owned dictionary materializers."""
    from .control_source import ControlSequenceMutation

    aliases = _loop_carried_storage_aliases(graph_obj)
    materializers = {}
    for node_id, data in graph_obj.nodes(data=True):
        attrs = data.get("attributes") or {}
        if (attrs.get("aggregate_kind") == "dict"
                and attrs.get("producer_kind") == "aggregate_materialization"
                and not attrs.get("binding_name")):
            materializers[int(node_id)] = data
    mutations = []
    for node_id, data in materializers.items():
        arguments = [int(parent) for parent, role in data.get("parents") or ()
                     if str(role).startswith("arg:")]
        if len(arguments) != 1:
            continue
        source = arguments[0]
        source_data = graph_obj.nodes.get(source, {})
        expression = source_data.get("expr_obj")
        if isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.Or):
            source = next((int(parent) for parent, role in source_data.get("parents") or ()
                           if role == "value:0"), source)
        source = int(aliases.get(source, source))
        if (graph_obj.nodes.get(source, {}).get("attributes") or {}).get("aggregate_kind") != "dict":
            continue
        mutations.append(ControlSequenceMutation(
            int(data.get("value_id", node_id)), "replace", (source,),
            int(data.get("value_id", node_id)), policy="unique",
        ))
    for node_id, data in graph_obj.nodes(data=True):
        if str(data.get("type") or data.get("op")).lower() != "indexedstore":
            continue
        roles = {str(role): int(parent) for parent, role in data.get("parents") or ()}
        base = roles.get("base")
        base = aliases.get(base, base)
        if base not in materializers or "index" not in roles or "value" not in roles:
            continue
        mutations.append(ControlSequenceMutation(
            int(graph_obj.nodes[base].get("value_id", base)), "update",
            (roles["index"], roles["value"]), int(data.get("value_id", node_id)),
            policy="unique", argument_kind="mapping_items",
        ))
    return tuple(mutations)


def _identity_return_aliases(graph_obj: Any, function_table: Any) -> dict[int, int]:
    """Prove caller aliases using callee return slots that return a formal."""
    if function_table is None:
        return {}
    result = {}
    for call_id, data in graph_obj.nodes(data=True):
        reference = (data.get("attributes") or {}).get("callee_ref")
        if reference is None:
            continue
        child = getattr(function_table.entry(reference), "graph", None)
        child = getattr(child, "G", None)
        if child is None:
            continue
        sites = tuple((child.graph.get("return_slot_values") or {}).values())
        if not sites or len({len(site) for site in sites}) != 1:
            continue
        parameters = tuple(child.graph.get("function_parameters") or ())
        actuals = {str(role): int(parent) for parent, role in data.get("parents") or ()}
        for index, slots in enumerate(zip(*sites)):
            if len(set(slots)) != 1 or slots[0] is None:
                continue
            formal = child.nodes.get(int(slots[0]), {})
            attrs = formal.get("attributes") or {}
            name = attrs.get("binding_name")
            if attrs.get("binding_kind") != "parameter" or name not in parameters:
                continue
            actual = actuals.get(f"arg:{parameters.index(name)}")
            if actual is None:
                continue
            if len(sites[0]) == 1:
                result[int(call_id)] = actual
            for node_id, projection in graph_obj.nodes(data=True):
                if str(projection.get("type") or projection.get("op") or "").casefold() in {"indexedstore", "subscriptstore", "setitem"}:
                    continue
                roles = {str(role): int(parent) for parent, role in projection.get("parents") or ()}
                if roles.get("base") != int(call_id):
                    continue
                key = (graph_obj.nodes.get(roles.get("index"), {}).get("attributes") or {}).get("value")
                if key == index:
                    result[int(node_id)] = actual
    return result


def _static_mapping_capacity_bounds(
    graph_obj: Any,
    *,
    nonmutating_call_ids: Iterable[int] = (),
) -> dict[int, int]:
    """Bound local keyed arenas by their complete constant-key universe.

    Repeated writes, including writes in loops, cannot increase this bound.
    Copies inherit the source universe. Dynamic keys and external sources
    deliberately have no bound here; they need a runtime capacity contract.
    """
    aliases = _loop_carried_storage_aliases(graph_obj)
    nonmutating_calls = set(map(int, nonmutating_call_ids))
    keys, copies, unknown = {}, {}, set()
    for node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        attributes = data.get("attributes") or {}
        if (attributes.get("aggregate_kind") == "dict"
                and attributes.get("materialization_kind") == "unrolled_loop"):
            universe = keys.setdefault(int(node_id), set())
            for row_id in attributes.get("materialized_value_ids", ()):
                row = graph_obj.nodes.get(int(row_id), {})
                leaves = tuple(int(parent) for parent, role in row.get("parents", ()) if role == "elts")
                key_data = graph_obj.nodes.get(leaves[0], {}) if leaves else {}
                try:
                    if "value" in (key_data.get("attributes") or {}):
                        key = key_data["attributes"]["value"]
                    else:
                        key = ast.literal_eval(key_data.get("expr_obj"))
                    universe.add(key)
                except (ValueError, TypeError):
                    unknown.add(int(node_id))
        if isinstance(expression, ast.Dict):
            try:
                keys[int(node_id)] = {ast.literal_eval(key) for key in expression.keys}
            except (ValueError, TypeError):
                unknown.add(int(node_id))
    for mutation in _source_mapping_mutations(graph_obj):
        destination = int(mutation.sequence_value_id)
        if mutation.operator == "replace":
            copies[destination] = int(mutation.argument_value_ids[0])
            keys.setdefault(destination, set())
        elif mutation.operator == "update":
            for key_id in mutation.argument_value_ids[::2]:
                data = graph_obj.nodes.get(int(key_id), {})
                try:
                    key = ast.literal_eval(data.get("expr_obj"))
                    keys.setdefault(destination, set()).add(key)
                except (ValueError, TypeError):
                    unknown.add(destination)
    for _node_id, data in graph_obj.nodes(data=True):
        roles = {str(role): int(parent) for parent, role in data.get("parents") or ()}
        operation = str(data.get("type") or data.get("op") or "").casefold()
        if operation == "indexedstore":
            destination = int(aliases.get(roles.get("base"), roles.get("base", -1)))
            if destination in keys:
                try:
                    key = ast.literal_eval(graph_obj.nodes[roles["index"]].get("expr_obj"))
                    keys[destination].add(key)
                except (KeyError, ValueError, TypeError):
                    unknown.add(destination)
        elif (operation in {"call", "plancall", "update", "setdefault"}
              and int(_node_id) not in nonmutating_calls
              and int(_node_id) not in copies):
            for parent in roles.values():
                destination = int(aliases.get(parent, parent))
                if destination in keys:
                    unknown.add(destination)
    changed = True
    while changed:
        changed = False
        for destination, source in copies.items():
            if source not in keys or source in unknown:
                if destination not in unknown:
                    unknown.add(destination)
                    changed = True
            else:
                previous = len(keys[destination])
                keys[destination].update(keys[source])
                changed |= len(keys[destination]) != previous
    return {node_id: len(universe) for node_id, universe in keys.items()
            if node_id not in unknown}


def _static_sequence_capacity_bounds(
    graph_obj: Any,
    *,
    nonmutating_call_ids: Iterable[int] = (),
) -> dict[int, int]:
    bounds = _static_mapping_capacity_bounds(
        graph_obj,
        nonmutating_call_ids=nonmutating_call_ids,
    )
    local_lists = {int(node_id): len(data["expr_obj"].elts)
                   for node_id, data in graph_obj.nodes(data=True)
                   if isinstance(data.get("expr_obj"), ast.List)}
    unknown = set()
    loops = [data["expr_obj"] for _, data in graph_obj.nodes(data=True)
             if isinstance(data.get("expr_obj"), (ast.For, ast.While))]
    for mutation in _sequence_append_call_mutations(graph_obj):
        sequence_id = int(mutation.sequence_value_id)
        if sequence_id not in local_lists or mutation.operator == "clear":
            continue
        expression = graph_obj.nodes.get(int(mutation.effect_node_id), {}).get("expr_obj")
        line = getattr(expression, "lineno", None)
        if (mutation.operator != "append" or line is None
                or any(loop.lineno <= line <= loop.end_lineno for loop in loops)):
            unknown.add(sequence_id)
        else:
            local_lists[sequence_id] += 1
    for _node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Call):
            continue
        identity = (data.get("attributes") or {}).get("extraction_identity")
        if identity in {"builtins.tuple", "builtins.len", "builtins.bool"}:
            continue
        for parent, role in data.get("parents") or ():
            if str(role).startswith("arg:") and int(parent) in local_lists:
                unknown.add(int(parent))
    bounds.update({key: count for key, count in local_lists.items() if key not in unknown})
    for mutation in _source_sequence_snapshots(graph_obj):
        source = int(mutation.argument_value_ids[0])
        if source in bounds:
            bounds[int(mutation.sequence_value_id)] = bounds[source]
    return bounds


def _linked_sequence_propagation_kind(
    mapped_arguments: Iterable[Any],
) -> str | None:
    """Return the exact ABI provenance that lets sequence storage escape."""

    accounting = tuple(
        dict(argument.accounting or {}) for argument in mapped_arguments
    )
    if any(item.get("program_abi_parameter") for item in accounting):
        return "program_abi_parameter"
    returned_record_identities = {
        (
            item.get("program_abi_record"),
            item.get("program_abi_keyed_owner")
            or str(item.get("program_abi_field")).split(".", 1)[0],
            item.get("returned_record_storage"),
        )
        for item in accounting
        if item.get("program_abi_record") is not None
        and item.get("program_abi_field") is not None
        and item.get("returned_record_storage") is not None
    }
    if accounting and len(returned_record_identities) == 1:
        return "exact_returned_record_storage"
    return None


def _sequence_append_call_mutations(graph_obj: Any):
    """Recover authored resident ``append/add/clear`` calls as lexical effects."""

    from .control_source import ControlExpression, ControlSequenceMutation
    from .string_table import string_token

    def retained_argument_expressions(
        record: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        expression = record.get("expression")
        if not isinstance(expression, ast.Call) or len(expression.args) != 1:
            return ()
        argument = expression.args[0]
        if not isinstance(argument, ast.Constant) or not isinstance(
            argument.value, (str, bool, int, float)
        ):
            return ()
        literal = (
            string_token(argument.value)
            if isinstance(argument.value, str)
            else argument.value
        )
        # Structural folding may remove the literal's graph node after this
        # source-effect snapshot. Keep its value on the mutation itself so
        # control lowering emits a local constant rather than inventing an
        # unexplained function formal for the vanished node id.
        return (ControlExpression("const", literal=literal),)

    sequence_kinds = {"list", "set", "bytes", "bytearray"}
    mutations = [
        ControlSequenceMutation(
            sequence_value_id=int(record["sequence_value_id"]),
            operator=str(record["operator"]),
            argument_value_ids=tuple(map(
                int, record["argument_value_ids"]
            )),
            effect_node_id=int(node_id),
            policy=record.get("policy"),
            argument_expressions=retained_argument_expressions(record),
        )
        for node_id, record in sorted(
            (
                graph_obj.graph.get("source_sequence_mutation_records")
                or {}
            ).items(),
            key=lambda item: int(item[0]),
        )
    ]
    mutations.extend(_source_mapping_mutations(graph_obj))
    mutations.extend(_source_sequence_snapshots(graph_obj))
    recorded_effect_ids = {
        int(mutation.effect_node_id) for mutation in mutations
    }
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr in {"append", "add", "clear"}
        ):
            continue
        if int(node_id) in recorded_effect_ids:
            continue
        parents = tuple(data.get("parents") or ())
        destination_node = next((
            int(parent) for parent, role in parents
            if str(role) == "operand" and int(parent) in graph_obj
        ), None)
        arguments = tuple(
            int(parent) for parent, role in parents
            if str(role).startswith("arg:") and int(parent) in graph_obj
        )
        arity = 0 if expression.func.attr == "clear" else 1
        if destination_node is None or len(arguments) != arity:
            continue
        destination = graph_obj.nodes[destination_node]
        attributes = destination.get("attributes") or {}
        if attributes.get("aggregate_kind") not in sequence_kinds:
            continue
        mutations.append(ControlSequenceMutation(
            sequence_value_id=int(destination.get(
                "value_id", destination_node
            )),
            operator=str(expression.func.attr),
            argument_value_ids=tuple(int(graph_obj.nodes[argument].get(
                "value_id", argument
            )) for argument in arguments),
            effect_node_id=int(data.get("value_id", node_id)),
            policy=(
                "unique" if attributes.get("aggregate_kind") == "set"
                else "duplicates"
            ),
        ))
    nodes_by_value = {
        int(data.get("value_id", node_id)): data
        for node_id, data in graph_obj.nodes(data=True)
    }
    expanded_mutations = []
    for mutation in mutations:
        arguments: list[int] = []
        for value_id in mutation.argument_value_ids:
            attributes = (
                nodes_by_value.get(int(value_id), {}).get("attributes") or {}
            )
            leaves = tuple(map(
                int, attributes.get("aggregate_leaf_value_ids", ())
            ))
            if attributes.get("aggregate_kind") == "tuple" and leaves:
                arguments.extend(leaves)
            else:
                arguments.append(int(value_id))
        expanded_mutations.append(replace(
            mutation,
            argument_value_ids=tuple(arguments),
            argument_kind=(
                "row" if len(arguments) > 1 and not mutation.argument_kind.startswith("mapping_")
                else mutation.argument_kind
            ),
        ))
    return tuple(expanded_mutations)


def _joined_byte_sequence_ids(
    graph_obj: Any,
    *,
    call_result_kinds: Mapping[int, str],
    declared_sequence_ids: Iterable[int],
    structural_byte_sequence_ids: Iterable[int] = (),
    transformed_call_source_ids: Iterable[int] = (),
) -> tuple[int, ...]:
    """Find resident ``list[bytes]`` identities needing dual ABI views.

    The authored outer list owns an element count, while its eventual empty
    byte join owns a flattened byte stream. Parameters already express these
    as ``row_count``/``join_bytes`` source transforms; this recovers the same
    representation for locals without inspecting Python values.
    """

    declared = set(map(int, declared_sequence_ids))
    node_by_value = {
        int(data.get("value_id", node_id)): data
        for node_id, data in graph_obj.nodes(data=True)
    }
    byte_values = {
        int(value_id)
        for value_id, data in node_by_value.items()
        if str((data.get("attributes") or {}).get("aggregate_kind"))
        in {"bytes", "bytearray"}
    }
    byte_values.update(
        int(value_id)
        for value_id, kind in call_result_kinds.items()
        if str(kind) in {"bytes", "bytearray"}
    )
    byte_values.update(map(int, structural_byte_sequence_ids))
    seeds: set[int] = set(map(int, transformed_call_source_ids))
    for value_id, data in node_by_value.items():
        attributes = data.get("attributes") or {}
        if attributes.get("aggregate_kind") != "list":
            continue
        leaves = tuple(map(
            int, attributes.get("aggregate_leaf_value_ids", ())
        ))
        if not leaves:
            leaves = tuple(
                int(graph_obj.nodes[parent].get("value_id", parent))
                for parent, role in (data.get("parents") or ())
                if str(role).startswith("elts") and parent in graph_obj
            )
        if leaves and all(int(leaf) in byte_values for leaf in leaves):
            seeds.add(int(value_id))
    append_mutations = _sequence_append_call_mutations(graph_obj)
    for mutation in append_mutations:
        if (
            mutation.operator == "append"
            and len(mutation.argument_value_ids) == 1
            and int(mutation.argument_value_ids[0]) in byte_values
        ):
            seeds.add(int(mutation.sequence_value_id))

    if os.environ.get("TURING_DEBUG_JOINED_SEQUENCE"):
        print(
            "DEBUGJOINED "
            f"fn={graph_obj.graph.get('function_name')} "
            f"bytes={tuple(sorted(byte_values))!r} "
            f"seeds={tuple(sorted(seeds))!r} "
            f"mutations={tuple((int(item.sequence_value_id), tuple(map(int, item.argument_value_ids))) for item in append_mutations)!r} "
            f"transformed={tuple((int(value_id), str((node_by_value.get(int(value_id), {}).get('attributes') or {}).get('binding_name')), ast.dump(node_by_value.get(int(value_id), {}).get('expr_obj'), include_attributes=False) if isinstance(node_by_value.get(int(value_id), {}).get('expr_obj'), ast.AST) else None) for value_id in sorted(map(int, transformed_call_source_ids)))!r} "
            f"identities={tuple((str(name), tuple(map(int, history))) for name, history in (graph_obj.graph.get('identity_table') or {}).items() if str(name) in {'types', 'entries'})!r} "
            f"declared={tuple(sorted(declared))!r}",
            file=sys.stderr,
        )

    mutation_destinations = {
        int(item.sequence_value_id) for item in append_mutations
    }
    joined = {
        int(value_id)
        for value_id in seeds
        if (
            int(value_id) in declared
            or int(value_id) in mutation_destinations
            or (
                node_by_value.get(int(value_id), {}).get("attributes") or {}
            ).get("aggregate_kind") == "list"
            or isinstance(
                (
                    (graph_obj.nodes.get(int(value_id), {}).get("attributes")
                     or {}).get(
                        "value",
                        graph_obj.nodes.get(int(value_id), {}).get("constant"),
                    )
                ),
                list,
            )
        )
    }
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        identities = set(map(int, history))
        if identities & seeds:
            joined.update(identities & declared)
    return tuple(sorted(joined))


def _joined_list_literal_mutations(
    graph_obj: Any, joined_sequence_ids: Iterable[int]
) -> tuple[Any, ...]:
    """Materialize each authored list element into its resident dual view."""

    from .control_source import ControlSequenceMutation

    joined = set(map(int, joined_sequence_ids))
    mutations = []
    emitted: set[tuple[int, int, int]] = set()

    def append_mutation(
        sequence_id: int, effect_node_id: int, value_id: int,
    ) -> None:
        key = (int(sequence_id), int(effect_node_id), int(value_id))
        if key in emitted:
            return
        emitted.add(key)
        mutations.append(ControlSequenceMutation(
            sequence_value_id=int(sequence_id),
            operator="append",
            argument_value_ids=(int(value_id),),
            effect_node_id=int(effect_node_id),
            policy="duplicates",
            argument_kind="joined_literal_element",
        ))

    nodes = tuple(graph_obj.nodes(data=True))

    def expression_node(expression: ast.AST) -> tuple[int, int] | None:
        exact = [
            (int(node_id), int(data.get("value_id", node_id)))
            for node_id, data in nodes
            if data.get("expr_obj") is expression
        ]
        if len(exact) == 1:
            return exact[0]
        position = (
            int(getattr(expression, "lineno", -1) or -1),
            int(getattr(expression, "col_offset", -1) or -1),
            int(getattr(expression, "end_lineno", -1) or -1),
            int(getattr(expression, "end_col_offset", -1) or -1),
            type(expression),
        )
        positioned = [
            (int(node_id), int(data.get("value_id", node_id)))
            for node_id, data in nodes
            for candidate in (data.get("expr_obj"),)
            if isinstance(candidate, ast.AST)
            and (
                int(getattr(candidate, "lineno", -2) or -2),
                int(getattr(candidate, "col_offset", -2) or -2),
                int(getattr(candidate, "end_lineno", -2) or -2),
                int(getattr(candidate, "end_col_offset", -2) or -2),
                type(candidate),
            ) == position
        ]
        return positioned[0] if len(positioned) == 1 else None

    for node_id, data in sorted(
        nodes, key=lambda item: int(item[0])
    ):
        sequence_id = int(data.get("value_id", node_id))
        if sequence_id not in joined:
            continue
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.List):
            continue
        elements = sorted(
            [
                (
                int(parent), str(role),
                int(graph_obj.nodes[parent].get("value_id", parent)),
                )
                for parent, role in (data.get("parents") or ())
                if str(role).startswith("elts") and parent in graph_obj
            ],
            key=lambda item: (
                int(item[1].split(":", 1)[1])
                if ":" in item[1] and item[1].split(":", 1)[1].isdigit()
                else item[0]
            ),
        )
        for parent, _role, value_id in elements:
            append_mutation(sequence_id, parent, value_id)

    # The source realizer may specialize an inline dynamic list into an empty
    # structural resident while retaining the authored list only on the Call
    # that consumes it (``_vector([uleb(index)])``). Its elements are still
    # ordinary ProcessGraph nodes. Correlate those exact AST objects/source
    # spans back to their deterministic value identities and initialize the
    # resident before the consuming call.
    for _call_node_id, data in sorted(nodes, key=lambda item: int(item[0])):
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Call) or not expression.args:
            continue
        literal = expression.args[0]
        if not isinstance(literal, ast.List):
            continue
        sequence_ids = tuple(dict.fromkeys(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (data.get("parents") or ())
            if str(role) == "arg:0" and parent in graph_obj
        ))
        if len(sequence_ids) != 1 or sequence_ids[0] not in joined:
            continue
        for element in literal.elts:
            correlated = expression_node(element)
            if correlated is None:
                continue
            effect_node_id, value_id = correlated
            append_mutation(sequence_ids[0], effect_node_id, value_id)
    return tuple(mutations)


def _dict_literal_mutations(graph_obj: Any) -> tuple[Any, ...]:
    """Materialize authored dynamic dict rows at the literal's source site.

    Compile-time mappings use ``literal_table=`` initialization, but a dict
    whose values are computed SSA results cannot be initialized in the entry
    prelude.  Its AST already provides an exact ordered key/value relation;
    retain each row as an ordinary unique-sequence insertion at its key's
    lexical position inside the literal. Data scheduling separately ensures
    that the corresponding value producer completes first. String keys become
    the repository's content-addressed token, the same representation used by
    dynamic keyed lookups.
    """

    from .control_source import ControlExpression, ControlSequenceMutation
    from .string_table import string_token

    nodes = tuple(graph_obj.nodes(data=True))

    def expression_value(expression: ast.AST) -> tuple[int, int] | None:
        exact = tuple(
            (int(node_id), int(data.get("value_id", node_id)))
            for node_id, data in nodes
            if data.get("expr_obj") is expression
        )
        if len(exact) == 1:
            return exact[0]
        position = (
            int(getattr(expression, "lineno", -1) or -1),
            int(getattr(expression, "col_offset", -1) or -1),
            int(getattr(expression, "end_lineno", -1) or -1),
            int(getattr(expression, "end_col_offset", -1) or -1),
            type(expression),
        )
        positioned = tuple(
            (int(node_id), int(data.get("value_id", node_id)))
            for node_id, data in nodes
            for candidate in (data.get("expr_obj"),)
            if isinstance(candidate, ast.AST)
            and (
                int(getattr(candidate, "lineno", -2) or -2),
                int(getattr(candidate, "col_offset", -2) or -2),
                int(getattr(candidate, "end_lineno", -2) or -2),
                int(getattr(candidate, "end_col_offset", -2) or -2),
                type(candidate),
            ) == position
        )
        return positioned[0] if len(positioned) == 1 else None

    mutations = []
    for node_id, data in sorted(nodes, key=lambda item: int(item[0])):
        expression = data.get("expr_obj")
        attributes = data.get("attributes") or {}
        if (attributes.get("aggregate_kind") == "dict"
                and attributes.get("materialization_kind") == "unrolled_loop"):
            sequence_id = int(data.get("value_id", node_id))
            mutations.append(ControlSequenceMutation(
                sequence_value_id=sequence_id, operator="clear",
                argument_value_ids=(), effect_node_id=int(node_id), policy="unique",
            ))
            for row_id in attributes.get("materialized_value_ids", ()):
                row = graph_obj.nodes.get(int(row_id), {})
                leaves = tuple(int(parent) for parent, role in row.get("parents", ())
                               if role == "elts")
                if len(leaves) != 2:
                    raise ValueError(f"Dictionary row {row_id} must publish one key and value")
                mutations.append(ControlSequenceMutation(
                    sequence_value_id=sequence_id, operator="update",
                    argument_value_ids=tuple(int(graph_obj.nodes[parent].get("value_id", parent)) for parent in leaves),
                    effect_node_id=int(row_id), policy="unique",
                    argument_kind="mapping_items",
                ))
            continue
        if not (
            isinstance(expression, ast.Dict)
            and attributes.get("aggregate_kind") == "dict"
            and len(expression.keys) == len(expression.values)
        ):
            continue
        sequence_id = int(data.get("value_id", node_id))
        key_parents = tuple(
            (int(parent), int(graph_obj.nodes[parent].get("value_id", parent)))
            for parent, role in data.get("parents") or ()
            if str(role).startswith("keys") and parent in graph_obj
        )
        # Each execution of a literal constructs a fresh mapping. Keep one
        # lexical owner even for constant rows and for literals inside loops.
        mutations.append(ControlSequenceMutation(
            sequence_value_id=sequence_id, operator="clear",
            argument_value_ids=(), effect_node_id=int(node_id), policy="unique",
        ))
        value_parents = tuple(
            (int(parent), int(graph_obj.nodes[parent].get("value_id", parent)))
            for parent, role in data.get("parents") or ()
            if str(role).startswith("values") and parent in graph_obj
        )
        for index, (key_expression, value_expression) in enumerate(zip(
            expression.keys, expression.values
        )):
            if key_expression is None:
                continue
            key = (
                key_parents[index]
                if len(key_parents) == len(expression.keys)
                else expression_value(key_expression)
            )
            value = (
                value_parents[index]
                if len(value_parents) == len(expression.values)
                else expression_value(value_expression)
            )
            if key is None or value is None:
                continue
            key_node_id, key_value_id = key
            _value_node_id, value_value_id = value
            key_literal = (
                key_expression.value
                if isinstance(key_expression, ast.Constant)
                else None
            )
            key_expression_ir = None
            if isinstance(key_literal, str):
                key_expression_ir = ControlExpression(
                    "const", value_id=key_value_id,
                    literal=string_token(key_literal),
                )
            elif isinstance(key_literal, (bool, int, float)):
                key_expression_ir = ControlExpression(
                    "const", value_id=key_value_id, literal=key_literal,
                )
            mutations.append(ControlSequenceMutation(
                sequence_value_id=sequence_id,
                operator="update",
                argument_value_ids=(key_value_id, value_value_id),
                # The stored value can name an SSA producer authored much
                # earlier than the literal. Its producer position is not the
                # store's effect position: using it moved the first row ahead
                # of the literal's clear. The key is an exact AST child of
                # this literal and gives every row a stable authored order.
                effect_node_id=key_node_id,
                policy="unique",
                argument_kind="mapping_items",
                argument_expressions=(key_expression_ir, None),
            ))
    return tuple(mutations)
