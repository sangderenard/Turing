"""Source control retention: retained control records, sequence-mutation records, control expressions, record-field state keys and the missing-phi initial-identity repair."""

from __future__ import annotations

import ast
from typing import Any, Iterable, Mapping

from .control_source import ControlExpression
from .source_node_facts import (
    _ast_source_location,
    _ast_source_signature,
)


def _retain_source_control_records(graph_obj: Any) -> None:
    """Snapshot authored controls before structural folding removes nodes."""

    records = dict(graph_obj.graph.get("source_control_records") or {})
    for node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        if not isinstance(
            expression, (ast.If, ast.IfExp, ast.Try, ast.BoolOp),
        ):
            continue
        predicate_id = next((
            int(parent) for parent, role in (data.get("parents") or ())
            if str(role) == "test"
        ), None)
        records[int(node_id)] = {
            "expression": expression,
            "predicate_id": predicate_id,
        }
    if records:
        graph_obj.graph["source_control_records"] = records


def _retain_source_sequence_mutation_records(graph_obj: Any) -> None:
    """Snapshot authored append/add effects before structural reduction."""

    records = dict(
        graph_obj.graph.get("source_sequence_mutation_records") or {}
    )
    for node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr in {"append", "add"}
        ):
            continue
        parents = tuple(data.get("parents") or ())
        destination = next((
            int(parent) for parent, role in parents
            if str(role) == "operand" and int(parent) in graph_obj
        ), None)
        arguments = tuple(
            int(parent) for parent, role in parents
            if str(role).startswith("arg:") and int(parent) in graph_obj
        )
        if destination is None or len(arguments) != 1:
            continue
        destination_data = graph_obj.nodes[destination]
        if (destination_data.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in {"list", "set", "bytes", "bytearray"}:
            continue
        records[int(node_id)] = {
            "expression": expression,
            "sequence_value_id": int(destination_data.get(
                "value_id", destination
            )),
            "argument_value_ids": (
                int(graph_obj.nodes[arguments[0]].get(
                    "value_id", arguments[0]
                )),
            ),
            "operator": str(expression.func.attr),
            "policy": (
                "unique" if expression.func.attr == "add"
                else "duplicates"
            ),
        }
    controls = dict(graph_obj.graph.get("source_control_records") or {})
    effect_signatures = {
        int(effect_id): _ast_source_signature(record["expression"])
        for effect_id, record in records.items()
        if isinstance(record.get("expression"), ast.AST)
    }
    effect_locations = {signature[:-1] for signature in effect_signatures.values()}
    memberships_by_signature = {}
    for control_id, control_record in controls.items():
        conditional = control_record.get("expression")
        if not isinstance(conditional, ast.If):
            continue
        for arm, statements in (("body", conditional.body), ("orelse", conditional.orelse)):
            for statement in statements:
                for member in ast.walk(statement):
                    if _ast_source_location(member) in effect_locations:
                        memberships_by_signature.setdefault(_ast_source_signature(member), []).append(
                            (int(control_id), arm))
    for effect_id, record in records.items():
        expression = record.get("expression")
        if not isinstance(expression, ast.AST):
            continue
        effect_signature = effect_signatures[int(effect_id)]
        memberships = memberships_by_signature.get(effect_signature, ())
        record["branch_memberships"] = tuple(dict.fromkeys(memberships))
    if records:
        graph_obj.graph["source_sequence_mutation_records"] = records


def _source_control_records(graph_obj: Any) -> dict[int, dict[str, Any]]:
    """Return retained plus still-live authored control records."""

    _retain_source_control_records(graph_obj)
    return {
        int(node_id): dict(record)
        for node_id, record in (
            graph_obj.graph.get("source_control_records") or {}
        ).items()
    }


def _source_control_expression(
    graph_obj: Any, node_id: int,
) -> ast.AST | None:
    record = _source_control_records(graph_obj).get(int(node_id))
    expression = None if record is None else record.get("expression")
    return expression if isinstance(expression, ast.AST) else None


def _retained_control_value_id(
    graph_obj: Any,
    source_value_id: int | None,
    expression: ast.AST | None,
) -> int | None:
    """Correlate a pre-fold control value with its retained graph identity."""

    if source_value_id is not None and int(source_value_id) in graph_obj:
        return int(source_value_id)
    if expression is not None:
        signature = _ast_source_signature(expression)
        matches = tuple(sorted(
            int(node_id)
            for node_id, data in graph_obj.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.AST)
            and _ast_source_signature(data["expr_obj"]) == signature
        ))
        if matches:
            return matches[0]
    if source_value_id is not None:
        source_value_id = int(source_value_id)
        for history in (graph_obj.graph.get("identity_table") or {}).values():
            ordered = tuple(map(int, history))
            if source_value_id in ordered:
                retained = tuple(value for value in ordered if value in graph_obj)
                if retained:
                    return retained[-1]
        matches = tuple(sorted(
            int(node_id)
            for node_id, data in graph_obj.nodes(data=True)
            if int(data.get("value_id", node_id)) == source_value_id
        ))
        if matches:
            return matches[0]
    return None


def _optional_presence_control_expression(
    graph_obj: Any,
    expression: ast.AST | None,
) -> ControlExpression | None:
    """Recover an erased ``optional_record_field is [not] None`` test.

    ProgramABI splits an optional scalar field into a typed payload and an
    explicit Boolean presence slot. Structural folding can remove the Python
    ``None`` comparison after the effective local name has become the field
    payload. The payload's existence is not proof that the value is present:
    absent values deliberately retain a typed filler. Recover only the exact
    one-name, one-field, one-presence-slot relation and leave every ambiguous
    case to the incumbent unresolved behavior.
    """

    if not (
        isinstance(expression, ast.Compare)
        and len(expression.ops) == 1
        and isinstance(expression.ops[0], (ast.Is, ast.IsNot))
        and len(expression.comparators) == 1
        and isinstance(expression.comparators[0], ast.Constant)
        and expression.comparators[0].value is None
        and isinstance(expression.left, (ast.Name, ast.Attribute))
    ):
        return None
    histories = graph_obj.graph.get("identity_table") or {}
    if isinstance(expression.left, ast.Name):
        payload_ids = tuple(dict.fromkeys(
            int(value_id)
            for value_id in histories.get(expression.left.id, ())
            if int(value_id) in graph_obj
        ))
    else:
        signature = _ast_source_signature(expression.left)
        payload_ids = tuple(dict.fromkeys(
            int(node_id)
            for node_id, data in graph_obj.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.Attribute)
            and _ast_source_signature(data["expr_obj"]) == signature
        ))
    # A spelling's identity history can legitimately contain its default
    # formal, a selected IfExp arm, and several loads of the same record field.
    # That is not semantic ambiguity. Resolve every resident GetAttr through
    # its exact receiver and require the candidates to agree on ONE physical
    # optional-presence relation. This is stricter than choosing an incumbent
    # value id and broad enough for ``x = self.x if x is None else x``.
    relations = []
    for candidate_id in payload_ids:
        payload = graph_obj.nodes[candidate_id]
        if str(
            payload.get("type") or payload.get("op") or ""
        ).casefold() != "getattr":
            continue
        candidate_field = (payload.get("attributes") or {}).get("attribute")
        receiver_ids = tuple(dict.fromkeys(
            int(parent)
            for parent, role in payload.get("parents") or ()
            if str(role) in {"value", "object", "base", "receiver"}
            and int(parent) in graph_obj
        ))
        if candidate_field is None or len(receiver_ids) != 1:
            continue
        receiver_id = receiver_ids[0]
        visited: set[int] = set()
        while receiver_id in graph_obj and receiver_id not in visited:
            visited.add(receiver_id)
            receiver = graph_obj.nodes[receiver_id]
            forwarded = (receiver.get("attributes") or {}).get(
                "structural_identity_actual_value_id"
            )
            if forwarded is None or int(forwarded) not in graph_obj:
                break
            receiver_id = int(forwarded)
        receiver = graph_obj.nodes[receiver_id]
        candidate_parameter = (receiver.get("attributes") or {}).get(
            "binding_name"
        )
        if candidate_parameter is None:
            continue
        presence = tuple(
            (int(node_id), attributes)
            for node_id, data in graph_obj.nodes(data=True)
            for attributes in (data.get("attributes") or {},)
            if attributes.get("program_abi_optional_presence")
            and str(attributes.get("program_abi_parameter"))
            == str(candidate_parameter)
            and str(attributes.get("program_abi_field"))
            == str(candidate_field)
        )
        if len(presence) != 1:
            continue
        candidate_presence_id, candidate_attributes = presence[0]
        relations.append((
            int(candidate_id), str(candidate_parameter), str(candidate_field),
            int(candidate_presence_id), candidate_attributes,
        ))
    semantic_relations = {
        (parameter, field, presence_id)
        for _payload_id, parameter, field, presence_id, _attributes in relations
    }
    if len(semantic_relations) != 1:
        return None
    parameter, field, presence_id = next(iter(semantic_relations))
    payload_id, _parameter, _field, _presence_id, attributes = next(
        relation for relation in reversed(relations)
        if relation[1:4] == (parameter, field, presence_id)
    )
    present_when = bool(attributes.get(
        "program_abi_optional_present_when", True,
    ))
    is_not_none = isinstance(expression.ops[0], ast.IsNot)
    direct = present_when == is_not_none
    record = dict(
        graph_obj.graph.get("parameter_record_abi") or {}
    ).get(str(parameter), {})
    predicate = ControlExpression(
        "optional_presence",
        value_id=presence_id,
        program_abi_parameter=str(parameter),
        program_abi_field=str(field),
        program_abi_record=(
            None
            if record.get("identity") is None
            else str(record["identity"])
        ),
    )
    if not direct:
        predicate = ControlExpression("not", (predicate,), value_id=presence_id)
    receipt = {
        "name": (
            str(expression.left.id)
            if isinstance(expression.left, ast.Name)
            else ast.unparse(expression.left)
        ),
        "payload_value_id": int(payload_id),
        "presence_value_id": int(presence_id),
        "parameter": str(parameter),
        "field": str(field),
        "comparison": "is_not_none" if is_not_none else "is_none",
        "predicate": "presence" if direct else "not_presence",
        "priority": "exact_program_abi_optional_presence",
        "tie_policy": "incumbent",
    }
    receipts = list(graph_obj.graph.get(
        "optional_presence_control_receipts", ()
    ))
    if receipt not in receipts:
        receipts.append(receipt)
        graph_obj.graph["optional_presence_control_receipts"] = tuple(receipts)
    return predicate


def _record_field_state_keys(
    graph: Any,
    node_by_value: Mapping[int, int],
    value_ids: Iterable[int],
) -> frozenset[tuple[int, str]]:
    """Return exact receiver/field identities represented by graph values."""

    keys: set[tuple[int, str]] = set()
    for value_id in value_ids:
        node_id = node_by_value.get(int(value_id))
        if node_id is None or node_id not in graph:
            continue
        data = graph.nodes[node_id]
        attributes = data.get("attributes") or {}
        state = attributes.get("record_field_state")
        if (
            isinstance(state, (tuple, list))
            and len(state) == 2
            and state[0] is not None
            and state[1] is not None
        ):
            keys.add((int(state[0]), str(state[1])))
        operation = str(data.get("type") or data.get("op") or "").casefold()
        attribute = attributes.get("attribute")
        if operation not in {"getattr", "setattr"} or attribute is None:
            continue
        receiver_roles = (
            {"object", "base", "receiver"}
            if operation == "setattr"
            else {"value", "object", "base", "receiver"}
        )
        receiver = next((
            int(parent)
            for parent, role in data.get("parents") or ()
            if str(role) in receiver_roles
        ), None)
        if receiver is not None:
            keys.add((receiver, str(attribute)))
    return frozenset(keys)


def _repair_missing_phi_initial_identities(graph_obj):
    """Recover a removed conditional incumbent from exact dataflow identity.

    Reduction may remove an intermediate spelling while a surviving Phi still
    carries that spelling in ``initial_value_id``. Treating the absent integer
    as an external value fabricates a function formal. Recovery is legal only
    when one resident value in the Phi binding's identity history is the unique
    nearest common dataflow ancestor of both authored arms. A resident initial
    is the incumbent and is never challenged; equal nearest candidates remain
    unresolved.
    """
    import networkx as nx

    node_by_value = {
        int(data.get("value_id", node_id)): node_id
        for node_id, data in graph_obj.nodes(data=True)
    }
    dataflow = nx.DiGraph()
    dataflow.add_nodes_from(graph_obj.nodes)
    for node_id, data in graph_obj.nodes(data=True):
        for parent, _role in data.get("parents") or ():
            if parent in graph_obj:
                dataflow.add_edge(parent, node_id)
    identity_table = graph_obj.graph.get("identity_table") or {}
    receipts = []
    for phi_node_id, phi_data in graph_obj.nodes(data=True):
        if str(phi_data.get("type") or phi_data.get("op") or "").casefold() != "phi":
            continue
        attributes = phi_data.get("attributes") or {}
        initial_value_id = attributes.get("initial_value_id")
        binding_name = attributes.get("binding_name")
        if (
            initial_value_id is None
            or int(initial_value_id) in node_by_value
            or binding_name is None
        ):
            continue
        parents = {
            str(role): parent
            for parent, role in phi_data.get("parents") or ()
            if parent in graph_obj
        }
        if not {"body", "orelse"}.issubset(parents):
            continue
        body_node = parents["body"]
        orelse_node = parents["orelse"]
        common = (
            nx.ancestors(dataflow, body_node) | {body_node}
        ) & (
            nx.ancestors(dataflow, orelse_node) | {orelse_node}
        )
        candidates = []
        for ordinal, value_id in enumerate(
            identity_table.get(str(binding_name), ())
        ):
            value_id = int(value_id)
            candidate = node_by_value.get(value_id)
            if candidate not in common or candidate == phi_node_id:
                continue
            try:
                distance = (
                    nx.shortest_path_length(dataflow, candidate, body_node)
                    + nx.shortest_path_length(
                        dataflow, candidate, orelse_node
                    )
                )
            except nx.NetworkXNoPath:
                continue
            candidates.append((int(distance), int(ordinal), value_id))
        if not candidates:
            continue
        nearest_distance = min(item[0] for item in candidates)
        nearest = [
            item for item in candidates if item[0] == nearest_distance
        ]
        nearest_value_ids = tuple(dict.fromkeys(
            item[2] for item in nearest
        ))
        if len(nearest_value_ids) != 1:
            unresolved = {
                "phi_value_id": int(phi_data.get(
                    "value_id", phi_node_id
                )),
                "binding_name": str(binding_name),
                "missing_initial_value_id": int(initial_value_id),
                "candidate_value_ids": nearest_value_ids,
                "reason": "equal_priority_candidates",
                "tie_policy": "incumbent",
            }
            ledger = graph_obj.graph.setdefault(
                "unresolved_phi_initial_identities", []
            )
            if unresolved not in ledger:
                ledger.append(unresolved)
            continue
        replacement = nearest_value_ids[0]
        attributes["initial_value_id"] = int(replacement)
        attributes["recovered_initial_value_id"] = int(initial_value_id)
        receipts.append({
            "phi_value_id": int(phi_data.get("value_id", phi_node_id)),
            "binding_name": str(binding_name),
            "missing_initial_value_id": int(initial_value_id),
            "resident_initial_value_id": int(replacement),
            "dataflow_distance": int(nearest_distance),
            "priority": "exact_common_dataflow_ancestor",
            "tie_policy": "incumbent",
        })
    if receipts:
        graph_obj.graph["phi_initial_identity_repairs"] = tuple((
            *graph_obj.graph.get("phi_initial_identity_repairs", ()),
            *receipts,
        ))
    return tuple(receipts)
