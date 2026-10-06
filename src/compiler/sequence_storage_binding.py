"""Sequence storage binding: authored source ids, linked parameter aliases, member binding (relocated from fortran_c_shell)."""

from __future__ import annotations

import ast
from typing import Any, Collection, Iterable, Mapping


def _authored_source_sequence_ids(
    graph_obj: Any,
    sequence_declarations: Iterable[tuple[int, str, int, bool]],
) -> tuple[int, ...]:
    """Identify resident arenas whose initial contents come from the caller."""

    declared = {
        int(sequence_id) for sequence_id, *_rest in sequence_declarations
    }
    if not declared:
        return ()
    identity = graph_obj.graph.get("identity_table") or {}
    source_ids = {
        int(value_id)
        for parameter_name in graph_obj.graph.get("function_parameters") or ()
        for value_id in identity.get(str(parameter_name), ())
        if int(value_id) in declared
    }
    self_fields = {
        str(field_name)
        for field_name, receipt in dict(
            dict(
                (graph_obj.graph.get("parameter_record_abi") or {}).get(
                    "self"
                ) or {}
            ).get("fields") or {}
        ).items()
        if str(receipt.get("storage") or "") == "span"
    }
    for node_id, data in graph_obj.nodes(data=True):
        value_id = int(data.get("value_id", node_id))
        if value_id not in declared:
            continue
        attributes = data.get("attributes") or {}
        if attributes.get("program_abi_sequence_row_identity") is not None:
            source_ids.add(value_id)
            continue
        if str(attributes.get("binding_kind") or "") in {
            "parameter", "closure", "external",
        }:
            source_ids.add(value_id)
            continue
        operation = str(data.get("op") or data.get("type") or "").casefold()
        if (
            operation == "getattr"
            and str(attributes.get("attribute") or "") in self_fields
        ):
            source_ids.add(value_id)
    return tuple(sorted(source_ids))


def _linked_authored_parameter_aliases(
    caller: Any,
    callee: Any,
    caller_graph: Any,
    callee_graph: Any,
    argument_bindings: Any,
    caller_record_table: Any = None,
    callee_record_table: Any = None,
) -> dict[str, str]:
    """Map a linked callee formal onto an outer authored formal exactly.

    Method record fields retain their local spelling (usually ``self``).
    When the exact PlanCall binding says an authored caller parameter such as
    ``body`` supplies that receiver, the public ABI must expose
    ``body.field`` rather than ``self.field``.  Only deterministic formal
    identities participate; later same-spelling SSA versions are not aliases.
    """

    def identities(
        function: Any, graph: Any, record_table: Any,
    ) -> dict[int, str]:
        metadata = dict(getattr(function, "metadata", {}) or {})
        found = {
            int(value_id): str(name)
            for name, value_id in metadata.get("parameter_names", ())
        }
        graph_metadata = (
            dict(getattr(graph, "graph", {}) or {})
            if graph is not None else {}
        )
        identity_table = dict(graph_metadata.get("identity_table") or {})
        parameter_roots = {
            *map(str, dict(metadata.get("parameter_record_abi") or {})),
            *map(str, dict(metadata.get("parameter_value_abi") or {})),
            *map(str, graph_metadata.get("function_parameters", ()) or ()),
        }
        for name in sorted(parameter_roots):
            history = tuple(identity_table.get(name, ()))
            if history:
                found.setdefault(int(history[0]), name)
        # Method shells can remove the shapeless aggregate formal after all
        # fields are projected. The record descriptor still owns its exact
        # deterministic aggregate identity even when the final shell no
        # longer carries the ProcessGraph identity catalogue.
        records = dict(getattr(record_table, "records", {}) or {})
        for name, receipt in dict(
            metadata.get("parameter_record_abi") or {}
        ).items():
            identity = str(dict(receipt or {}).get("identity") or "")
            candidates = [
                int(record_id)
                for record_id, descriptor in records.items()
                if str(getattr(descriptor, "identity", "")) == identity
            ]
            if len(candidates) == 1:
                found.setdefault(candidates[0], str(name))
        return found

    caller_names = identities(caller, caller_graph, caller_record_table)
    callee_names = identities(callee, callee_graph, callee_record_table)
    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for caller_id, callee_id in argument_bindings:
        caller_name = caller_names.get(int(caller_id))
        callee_name = callee_names.get(int(callee_id))
        if caller_name is None or callee_name is None:
            continue
        previous = aliases.setdefault(callee_name, caller_name)
        if previous != caller_name:
            ambiguous.add(callee_name)
    for name in ambiguous:
        aliases.pop(name, None)
    return aliases


def _bind_sequence_storage_members(
    storage_bindings: dict[int, int],
    callee_sequence: Any,
    caller_sequence: Any,
    *,
    provisional_targets: Collection[int] = (),
) -> bool:
    """Bind every physical member of one exact sequence argument."""

    if (
        callee_sequence is None
        or caller_sequence is None
        or len(callee_sequence.column_value_ids)
        != len(caller_sequence.column_value_ids)
    ):
        return False
    provisional_targets = set(map(int, provisional_targets))

    def bind(callee_id: int, caller_id: int) -> None:
        callee_id = int(callee_id)
        caller_id = int(caller_id)
        incumbent = storage_bindings.get(callee_id)
        if incumbent is None or int(incumbent) in provisional_targets:
            storage_bindings[callee_id] = caller_id

    # The descriptor handle is part of the structural frame contract too.
    # Record propagation consults it when rebuilding a nested sequence field;
    # binding only the columns and counters leaves an otherwise exact record
    # with two different sequence identities at the caller boundary.
    bind(int(callee_sequence.sequence_id), int(caller_sequence.sequence_id))
    for callee_id, caller_id in zip(
        map(int, callee_sequence.column_value_ids),
        map(int, caller_sequence.column_value_ids),
    ):
        bind(callee_id, caller_id)
    bind(
        int(callee_sequence.length_address_id),
        int(caller_sequence.length_address_id),
    )
    bind(
        int(callee_sequence.capacity_value_id),
        int(caller_sequence.capacity_value_id),
    )
    for attribute in ("status_address_id", "live_flags_value_id"):
        callee_member = getattr(callee_sequence, attribute, None)
        caller_member = getattr(caller_sequence, attribute, None)
        if callee_member is not None and caller_member is not None:
            bind(int(callee_member), int(caller_member))
    return True


def _sequence_length_values(
    graph_obj: Any,
    sequence_declarations: Iterable[tuple[int, str, int, bool]],
    aliases: Mapping[int, int] | Iterable[tuple[int, int]] = (),
) -> dict[int, int]:
    """Map authored ``len(sequence)`` results to resident descriptors."""

    declared = {
        int(sequence_id)
        for sequence_id, _policy, _columns, _writable
        in sequence_declarations
    }
    resident = {value_id: value_id for value_id in declared}
    resident.update({
        int(alias): int(source)
        for alias, source in dict(aliases).items()
    })
    changed = True
    while changed:
        changed = False
        for alias, source in tuple(resident.items()):
            target = resident.get(int(source))
            if target is not None and resident.get(int(alias)) != target:
                resident[int(alias)] = int(target)
                changed = True
    values = {}
    for node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id == "len"
        ):
            continue
        arguments = tuple(
            int(parent)
            for parent, role in data.get("parents") or ()
            if str(role).startswith("arg:") and parent in graph_obj
        )
        if len(arguments) != 1:
            continue
        argument_id = int(graph_obj.nodes[arguments[0]].get(
            "value_id", arguments[0]
        ))
        sequence_id = resident.get(argument_id)
        if sequence_id in declared:
            values[int(data.get("value_id", node_id))] = int(sequence_id)
    return values
