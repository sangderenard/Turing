"""Master correlation table over one lowered SSA module.

A side product of compilation: neither source, nor IR, nor artifact.  The
lowering keeps many partial records of what a value *is* -- ``parameter_names``,
``storage_formals``, ``program_abi_*`` accounting, sequence descriptors,
record tables, ``value_aliases``, ``callee_input_ids`` -- each written by a
different pass and each correlated through its own key (a name, a signature
position, a source coordinate, an id).  Every fault traced on 2026-09-17 was
two of those records disagreeing about one value while nothing compared them.

This module collects every such claim into one table, one row per
``(function, value_id)``, and reports the places where the claims disagree.
It writes nothing back.  The table is the thing a future single identity
authority would own; until then it is the audit that says where the
authority is missing.

Findings (each is one concrete disagreement, with the two claims):

``multiple-definition``
    one value id defined by more than one instruction (in/out ABI fields
    that are deliberately redefined by every producer are exempt).
``unaccounted-formal``
    a formal no record names: not a parameter, not ABI storage, not leased
    frame storage, not a closure/member formal.
``conflicting-storage-claims``
    a formal claimed as two different ABI fields, or as an ABI field and
    as leased frame storage at once.
``descriptor-member-unknown``
    a sequence descriptor names a member id the function neither receives
    nor defines.
``descriptor-member-shared``
    one storage id plays a role in two descriptors of one function.
``helper-operand-outside-descriptor``
    a keyed lookup helper call reads storage no descriptor of the function
    names (the rewrite bound the call but not the descriptor, or vice
    versa).
``duplicate-storage-across-call``
    a callee formal identified as an ABI field or a sequence member is fed
    freshly leased caller storage although the caller already owns a value
    with that same identity.
``use-not-dominated``
    an instruction reads a value whose definition does not dominate the
    read (Phi operands exempt; they arrive over predecessor edges).
``alias-target-missing``
    a recorded value alias points at neither a physical definition nor an
    owned structural record descriptor.
``alias-not-concorded``
    a durable function-local alias receipt is absent from, or disagrees with,
    the shared planning identity page.
``source-field-identity-disagreement``
    repeated source-stage reads assign different class identities to one
    authored object field.
``callable-identity-disagreement``
    one exact source value is assigned different function-table addresses as
    it moves from first-class function syntax through a callable record field.
``source-parameter-identity-disagreement``
    one discovered static parameter identity changes between source stages.
``source-precision-boundary-disagreement``
    one authored Precision boundary changes kind, operand, or limb width
    between source reduction and structural output recovery.
``source-precision-operator-disagreement``
    one authored Precision operator changes operation, receiver, class, or
    limb width between source stages.
``operator-result-type-disagreement``
    one planned operation assigns more than one result dtype to the same
    region value identity.
``planning-alias-transition-disagreement``
    a recorded planning refinement does not begin at the resident established
    by the preceding refinement for that exact value.
``layout-type-unknown``
    a value's accounting names a struct/union type (``ssa_layout_kind`` /
    ``ssa_layout_identity``) that no live row of the module's struct or
    union table declares.
``layout-member-unknown``
    a live struct/union row embeds a nested row id no live row holds.
``layout-redeclaration``
    one type identity was declared twice with different layouts; the
    supersession edge on ``layout_supersession`` is reported with the
    fields that changed.
``layout-derivation-invalidated``
    a row laid out from a since-redeclared row has not itself been
    re-declared (its ``layout_state`` is still ``invalidated``).
``unsourced-fact``
    a resolved cell on a registered page with neither an inbound edge on
    ``concordance_edge`` nor a mint edge on ``concordance_mint``.  While the
    book's latch is OPEN every write through a raw page primitive is tagged
    on ``concordance_unsourced`` and listed here by page and stage: that
    list is the migration worklist for ``IdentityBook.post``.
``unsourced-identity``
    a ``MINTED`` value id among a function's values that no mint edge
    accounts for (it was minted directly, not through a ``Novel`` post).

The two ``unsourced-*`` kinds are reported on their own line of
``concordance_report`` and are not counted in its first line, which is the
pass/fail gate of ``tools/audit_identity_concordance.py``.
"""

from __future__ import annotations

import ast
import contextvars
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from collections.abc import MutableMapping
from enum import Enum, IntEnum
from typing import Any, Iterable, Mapping

from .id_space import MINTED, group_by_prefix, has_flag, label as id_label
from .monotonic_ids import GLOBAL_MONOTONIC_IDS
from ..transmogrifier.ssa import SSAValue


def _edge_count(graph_obj: Any) -> int:
    """Edge count for the publication receipt.  ``number_of_edges()`` walks a
    degree view in Python (O(nodes) per call, hot on 200k-node graphs); a
    plain directed graph's adjacency sums at C speed."""

    adjacency = getattr(graph_obj, "_adj", None)
    if adjacency is None or graph_obj.is_multigraph() or not graph_obj.is_directed():
        return int(graph_obj.number_of_edges())
    return sum(map(len, adjacency.values()))


def publish_program_abi_graph_identities(
    graph_obj: Any, *, force: bool = False,
) -> None:
    """Publish exact record/table identities along source graph edges.

    ProgramABI names the root record.  GetAttr, table indexing, and loop
    target bindings are the authoritative transformation spine from that
    root to nested records.  Publishing the result on the graph value lets
    every later consumer ask the same concordance instead of reconstructing
    a field identity from a name or a local cache.
    """

    program_abi_source = graph_obj.graph.get("program_abi")
    parameter_records_source = graph_obj.graph.get("parameter_record_abi")
    identities_source = graph_obj.graph.get("identity_table")
    program_abi = program_abi_source or {}
    parameter_records = parameter_records_source or {}
    identities = identities_source or {}
    receipt = (
        len(graph_obj),
        _edge_count(graph_obj),
        id(program_abi_source),
        id(parameter_records_source),
        id(identities_source),
    )
    if not force and graph_obj.graph.get(
        "program_abi_identity_publication_receipt"
    ) == receipt:
        return

    repository_records = dict(program_abi.get("records") or {})

    def schema_named(identity: object):
        matches = tuple(
            (str(name), record)
            for name, record in repository_records.items()
            if str(name) == str(identity)
            or str(record.get("identity") or "") == str(identity)
        )
        return matches[0] if len(matches) == 1 else None

    # One pass indexes the nodes by value id; the old form rescanned every
    # node for each (record, value id) pair -- records x ids x nodes, which
    # dominated a 838-column dt system.
    nodes_by_value: dict[int, list] = {}
    if parameter_records:
        for node_id, data in graph_obj.nodes(data=True):
            nodes_by_value.setdefault(
                int(data.get("value_id", node_id)), [],
            ).append(data)
    for parameter_name, record in dict(
        parameter_records
    ).items():
        record_identity = str(record.get("identity") or "")
        for value_id in identities.get(str(parameter_name), ()):
            for data in nodes_by_value.get(int(value_id), ()):
                data.setdefault("attributes", {})[
                    "program_abi_record_identity"
                ] = record_identity

    from .bounded_fixed_point import BoundedFixedPoint, bound_for

    # Identities only propagate along existing edges: a changing round moves
    # at least one identity one edge further, so nodes + edges bound it.
    guard = BoundedFixedPoint(
        "program-abi-identity-publication",
        bound_for(
            "identity-publication", nodes=len(graph_obj),
            callsites=int(graph_obj.number_of_edges()),
        ),
        scope=(str(graph_obj.graph.get("function_name") or ""),),
    )
    # Reverse insertion order used to propagate one identity per WHOLE-GRAPH
    # scan: N linked record fields cost N * N node evaluations.  These are
    # the same transfer rules in the same insertion order and round order;
    # only values whose inputs changed need another evaluation.  Parent
    # records, not a possibly unreconstructed NetworkX adjacency, are the
    # dependency authority.  Loop target writes invalidate that target's
    # consumers too.  All rules read only the three propagated identity
    # attributes and immutable source/schema facts during this publication.
    node_order = tuple(graph_obj)
    order_index = {node: index for index, node in enumerate(node_order)}
    consumers = {node: set() for node in node_order}
    for node, node_data in graph_obj.nodes(data=True):
        for parent, _role in node_data.get("parents") or ():
            if parent in consumers:
                consumers[parent].add(node)
    pending = set(node_order)
    changed = True
    while changed:
        changed = False
        changed_nodes = set()
        for node_id in sorted(pending, key=order_index.__getitem__):
            data = graph_obj.nodes[node_id]
            attributes = data.setdefault("attributes", {})
            operation = str(
                data.get("op") or data.get("type") or ""
            ).casefold()
            if operation == "call":
                # A schema-declared constructor is the exact record identity
                # of its result.  Publish that identity on the constructor
                # edge itself so a resident collection receiving the result
                # can retain its record-row ABI without inferring anything
                # from the collection binding's spelling.
                constructor_candidates = tuple(filter(None, (
                    attributes.get("class_ref"),
                    attributes.get("static_python_reference"),
                )))
                constructor_schemas = tuple({
                    str(match[0]): match
                    for candidate in constructor_candidates
                    for match in (schema_named(candidate),)
                    if match is not None
                }.values())
                if len(constructor_schemas) == 1:
                    record_name, record = constructor_schemas[0]
                    record_identity = str(
                        record.get("identity") or record_name
                    )
                    incumbent = attributes.get(
                        "program_abi_record_identity"
                    )
                    if incumbent not in {None, record_identity}:
                        raise ValueError(
                            "record constructor identity disagreement: "
                            f"recorded={incumbent!r}, "
                            f"declared={record_identity!r}"
                        )
                    if incumbent is None:
                        attributes[
                            "program_abi_record_identity"
                        ] = record_identity
                        changed_nodes.add(node_id)
                        changed = True
            if (
                isinstance(data.get("expr_obj"), ast.Call)
                and isinstance(data["expr_obj"].func, ast.Attribute)
                and data["expr_obj"].func.attr in {"append", "add"}
            ):
                # The authored mutation edge transfers its one record
                # operand's exact schema to the destination row.  This is the
                # same ProgramABI identity already published on the
                # constructor/projection, not a type guess from ``append`` or
                # the destination name.
                parents = tuple(
                    (parent, str(role))
                    for parent, role in data.get("parents") or ()
                    if parent in graph_obj
                )
                destination = next((
                    graph_obj.nodes[parent]
                    for parent, role in parents if role == "operand"
                ), None)
                argument_identities = {
                    (graph_obj.nodes[parent].get("attributes") or {}).get(
                        "program_abi_record_identity"
                    )
                    for parent, role in parents
                    if role.startswith("arg:")
                    and (graph_obj.nodes[parent].get("attributes") or {}).get(
                        "program_abi_record_identity"
                    ) is not None
                }
                destination_attributes = (
                    None if destination is None else
                    destination.setdefault("attributes", {})
                )
                if (
                    destination_attributes is not None
                    and destination_attributes.get("aggregate_kind")
                    in {"list", "set"}
                    and len(argument_identities) == 1
                ):
                    row_identity = str(next(iter(argument_identities)))
                    incumbent = destination_attributes.get(
                        "program_abi_sequence_row_identity"
                    )
                    if incumbent not in {None, row_identity}:
                        raise ValueError(
                            "record sequence row identity disagreement: "
                            f"recorded={incumbent!r}, "
                            f"appended={row_identity!r}"
                        )
                    if incumbent is None:
                        destination_attributes[
                            "program_abi_sequence_row_identity"
                        ] = row_identity
                        destination_id = next(
                            parent for parent, role in parents
                            if role == "operand"
                        )
                        changed_nodes.add(destination_id)
                        changed = True
            # A typed local mapping is not a ProgramABI root parameter, but
            # its reducer receipt names the same repository value record.
            # Seed that exact keyed-value identity before projecting through
            # values()/comprehension edges; otherwise the retained Segment
            # declaration stops at the local mapping node.
            mapping_record = attributes.get("mapping_value_record")
            mapping_target = (
                None if mapping_record is None
                else schema_named(mapping_record)
            )
            if mapping_target is not None:
                mapping_identity = str(
                    mapping_target[1].get("identity")
                    or mapping_target[0]
                )
                if attributes.get(
                    "program_abi_indexed_value_identity"
                ) != mapping_identity:
                    attributes[
                        "program_abi_indexed_value_identity"
                    ] = mapping_identity
                    changed_nodes.add(node_id)
                    changed = True
            parents = tuple(
                (parent, str(role))
                for parent, role in data.get("parents") or ()
                if parent in graph_obj
            )
            expression = data.get("expr_obj")
            if isinstance(expression, ast.Call):
                # Resolved calls are respelled to their semantic operation
                # (``values``, ``tuple``, ...), so their graph operation is
                # intentionally not the frontend word ``Call``.  Preserve
                # identities using the exact authored call and parent edges.
                if (
                    isinstance(expression.func, ast.Attribute)
                    and expression.func.attr == "values"
                ):
                    incoming = {
                        identity
                        for parent, role in parents
                        if role in {
                            "operand", "value", "receiver", "callee",
                            "func", "function", "callable",
                        }
                        for identity in ((
                            graph_obj.nodes[parent].get("attributes") or {}
                        ).get("program_abi_sequence_row_identity"), (
                            graph_obj.nodes[parent].get("attributes") or {}
                        ).get("program_abi_indexed_value_identity"))
                        if identity is not None
                    }
                    if len(incoming) == 1:
                        row_identity = next(iter(incoming))
                        if attributes.get(
                            "program_abi_sequence_row_identity"
                        ) != row_identity:
                            attributes[
                                "program_abi_sequence_row_identity"
                            ] = row_identity
                            changed_nodes.add(node_id)
                            changed = True
                if (
                    isinstance(expression.func, ast.Name)
                    and expression.func.id in {"list", "tuple", "set"}
                ):
                    incoming = {
                        (graph_obj.nodes[parent].get("attributes") or {}).get(
                            "program_abi_sequence_row_identity"
                        )
                        for parent, role in parents
                        if str(role).startswith("arg:")
                        and (graph_obj.nodes[parent].get("attributes") or {}).get(
                            "program_abi_sequence_row_identity"
                        ) is not None
                    }
                    if len(incoming) == 1:
                        row_identity = next(iter(incoming))
                        if attributes.get(
                            "program_abi_sequence_row_identity"
                        ) != row_identity:
                            attributes[
                                "program_abi_sequence_row_identity"
                            ] = row_identity
                            changed_nodes.add(node_id)
                            changed = True
            if operation == "getattr":
                owner = next((
                    graph_obj.nodes[parent]
                    for parent, role in parents
                    if role in {"value", "object", "base", "receiver"}
                ), None)
                owner_identity = (
                    None if owner is None else
                    (owner.get("attributes") or {}).get(
                        "program_abi_record_identity"
                    )
                )
                schema_match = (
                    None if owner_identity is None
                    else schema_named(owner_identity)
                )
                field_name = str(attributes.get("attribute") or "")
                field = (
                    None if schema_match is None else
                    dict(schema_match[1].get("fields") or {}).get(field_name)
                )
                if isinstance(field, Mapping):
                    storage = str(field.get("storage") or "")
                    declared_shape = field.get("shape")
                    if (
                        declared_shape is None
                        and storage == "span"
                        and field.get("fixed_length") is not None
                    ):
                        declared_shape = (int(field["fixed_length"]),)
                    if storage == "span" and declared_shape is not None:
                        declared_shape = tuple(map(int, declared_shape))
                        declared_state = {
                            "shape": declared_shape,
                            "dtype": str(field.get("dtype") or "float64"),
                            "rank": int(field.get(
                                "rank", len(declared_shape),
                            )),
                            "metadata_state": "static",
                        }
                        # The declared span shape was read off the record's
                        # field declaration (``class_field_declaration``
                        # cell) and the edge carries it onto the getattr
                        # node whose owner holds the record: it derives from
                        # that declaration cell and the owner node's
                        # identity cell.
                        from .frame_identity_book import (
                            _record_field_declaration_cell,
                        )
                        from ..common.tensors.topological_reducer import (
                            _posted_node_identity_cell,
                        )

                        abi_source_cells = tuple(dict.fromkeys(
                            cell for cell in (
                                _record_field_declaration_cell(
                                    schema_match[1], field_name,
                                    owner_identity,
                                ),
                                _posted_node_identity_cell(
                                    graph_obj, int(next(
                                        parent for parent, role in parents
                                        if role in {
                                            "value", "object", "base",
                                            "receiver",
                                        }
                                    )),
                                ),
                            ) if cell is not None
                        ))
                        record_shape_transformation(
                            f"ProgramABI:{owner_identity}",
                            (str(owner_identity), field_name),
                            shape_scope_of(graph_obj),
                            int(data.get("value_id", node_id)),
                            stage="program_abi_graph_publication",
                            operation="getattr",
                            source_state=declared_state,
                            target_state=declared_state,
                            role=field_name,
                            source_cells=abi_source_cells,
                        )
                    target_name = (
                        field.get("record") if storage == "record"
                        else field.get("row_record") if storage == "table"
                        else field.get("value_record") if storage == "keyed"
                        else None
                    )
                    target = (
                        None if target_name is None
                        else schema_named(target_name)
                    )
                    key = (
                        "program_abi_sequence_row_identity"
                        if storage == "table"
                        else "program_abi_indexed_value_identity"
                        if storage == "keyed"
                        else "program_abi_record_identity"
                    )
                    if target is not None:
                        identity = str(
                            target[1].get("identity") or target[0]
                        )
                        if attributes.get(key) != identity:
                            attributes[key] = identity
                            changed_nodes.add(node_id)
                            changed = True
                # ``mapping.values()`` is an identity-preserving view over
                # the declared value side of that exact keyed field.  Keep
                # the row identity on the accessor edge so the subsequent
                # Call and retained comprehension loop do not have to
                # rediscover it from a binding name or a Python container.
                if (
                    field_name == "values"
                    and owner is not None
                    and (
                        owner.get("attributes") or {}
                    ).get("program_abi_indexed_value_identity") is not None
                ):
                    row_identity = (owner.get("attributes") or {})[
                        "program_abi_indexed_value_identity"
                    ]
                    if attributes.get(
                        "program_abi_sequence_row_identity"
                    ) != row_identity:
                        attributes[
                            "program_abi_sequence_row_identity"
                        ] = row_identity
                        changed_nodes.add(node_id)
                        changed = True
            elif operation in {"indexed", "load"}:
                base = next((
                    graph_obj.nodes[parent]
                    for parent, role in parents if role == "base"
                ), None)
                row_identity = (
                    None if base is None else
                    (
                        (base.get("attributes") or {}).get(
                            "program_abi_sequence_row_identity"
                        )
                        or (base.get("attributes") or {}).get(
                            "program_abi_indexed_value_identity"
                        )
                    )
                )
                if row_identity is not None and attributes.get(
                    "program_abi_record_identity"
                ) != row_identity:
                    attributes["program_abi_record_identity"] = row_identity
                    changed_nodes.add(node_id)
                    changed = True
            elif operation in {"for", "comprehension"}:
                iterable = next((
                    graph_obj.nodes[parent]
                    for parent, role in parents if role in {"iterable", "iter"}
                ), None)
                row_identity = (
                    None if iterable is None else
                    (iterable.get("attributes") or {}).get(
                        "program_abi_sequence_row_identity"
                    )
                )
                if row_identity is not None:
                    target_ids = set(map(int, dict(attributes.get(
                        "loop_target_bindings", {}
                    )).values()))
                    for target_node in target_ids:
                        if target_node not in graph_obj:
                            continue
                        target_data = graph_obj.nodes[target_node]
                        target_attributes = target_data.setdefault(
                            "attributes", {}
                        )
                        if target_attributes.get(
                            "program_abi_record_identity"
                        ) != row_identity:
                            target_attributes[
                                "program_abi_record_identity"
                            ] = row_identity
                            changed_nodes.add(target_node)
                            changed = True
            elif isinstance(data.get("expr_obj"), (
                ast.GeneratorExp, ast.ListComp, ast.SetComp,
            )):
                # The comprehension materializer is the same resident
                # sequence whose row is its exact ``elt`` edge.  Publishing
                # that edge preserves a record row as a columnar record ABI;
                # it does not create a Python generator or a parallel object.
                element = next((
                    graph_obj.nodes[parent]
                    for parent, role in parents if role == "elt"
                ), None)
                row_identity = (
                    None if element is None else
                    (element.get("attributes") or {}).get(
                        "program_abi_record_identity"
                    )
                )
                if row_identity is not None and attributes.get(
                    "program_abi_sequence_row_identity"
                ) != row_identity:
                    attributes[
                        "program_abi_sequence_row_identity"
                    ] = row_identity
                    changed_nodes.add(node_id)
                    changed = True
            elif (
                operation == "loopresult"
                and attributes.get("result_kind") == "collection"
            ):
                # Loop composition replaces the comprehension materializer
                # with this collection port.  Its ``value`` edge is the exact
                # element that is appended once per iteration, so a record
                # identity becomes the resident sequence's row identity.
                incoming = {
                    (graph_obj.nodes[parent].get("attributes") or {}).get(
                        "program_abi_record_identity"
                    )
                    for parent, role in parents
                    if role == "value"
                    and (graph_obj.nodes[parent].get("attributes") or {}).get(
                        "program_abi_record_identity"
                    ) is not None
                }
                if len(incoming) == 1:
                    row_identity = next(iter(incoming))
                    if attributes.get(
                        "program_abi_sequence_row_identity"
                    ) != row_identity:
                        attributes[
                            "program_abi_sequence_row_identity"
                        ] = row_identity
                        changed_nodes.add(node_id)
                        changed = True
            elif operation in {"phi", "boolop"}:
                for key in (
                    "program_abi_record_identity",
                    "program_abi_sequence_row_identity",
                    "program_abi_indexed_value_identity",
                ):
                    incoming = {
                        (graph_obj.nodes[parent].get("attributes") or {}).get(
                            key
                        )
                        for parent, _role in parents
                        if (graph_obj.nodes[parent].get("attributes") or {}).get(
                            key
                        ) is not None
                    }
                    if len(incoming) == 1:
                        identity = next(iter(incoming))
                        if attributes.get(key) != identity:
                            attributes[key] = identity
                            changed_nodes.add(node_id)
                            changed = True
        pending = changed_nodes | {
            consumer
            for node in changed_nodes
            for consumer in consumers[node]
        }
        guard.round(changed)

    graph_obj.graph[
        "program_abi_identity_publication_receipt"
    ] = receipt


def publish_projected_iterable_layouts(module: Any) -> None:
    """Publish sequence layouts and live lengths along exact SSA call edges.

    A projected row table is the columnar view of one authored iterable.  For
    a generator call, its source id is also the retained callsite id.  The
    callee sequence descriptor names the driver or returned materialization,
    and ``callee_input_ids`` maps that descriptor's length cell back to the
    caller's actual storage.  Recording that actual on every projected column
    keeps iteration a live-length loop without inventing another container.

    The same descriptor also owns the physical shape of every row column and
    the returned sequence identity itself.  Publish those facts through the
    call instruction's exact result and projected-column receipts.  These are
    views of the callee's storage contract, not independently inferred tensor
    objects.
    """

    functions = getattr(module, "functions", {}) or {}
    sequence_tables = getattr(module, "sequence_tables", {}) or {}
    for function_name, function in functions.items():
        occurrences = (
            *tuple(function.args),
            *tuple(
                value
                for block in function.blocks.values()
                for instruction in block.instrs
                for value in (
                    *instruction.args,
                    *((instruction.res,)
                      if instruction.res is not None else ()),
                )
            ),
        )
        calls = tuple(
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Call", "call"}
        )
        receipts = []
        for call in calls:
            callee_name = str(call.attributes.get("callee") or "")
            callee = functions.get(callee_name)
            table = sequence_tables.get(callee_name)
            if callee is None or table is None:
                continue
            returned_ids = {
                int(value.id)
                for block in callee.blocks.values()
                for instruction in block.instrs
                if str(instruction.op).casefold() in {"ret", "return"}
                for value in instruction.args
            }
            candidates = tuple(
                descriptor
                for descriptor in table.sequences.values()
                if returned_ids.intersection(map(
                    int, descriptor.column_value_ids
                ))
            )
            if len(candidates) != 1:
                driver_ids = {
                    int(row[0])
                    for row in callee.metadata.get(
                        "projected_row_tables", ()
                    )
                    if len(row) >= 1
                }
                candidates = tuple(
                    descriptor
                    for descriptor in table.sequences.values()
                    if int(descriptor.sequence_id) in driver_ids
                )
            if len(candidates) != 1:
                continue
            descriptor = candidates[0]
            callee_inputs = tuple(map(
                int, call.attributes.get("callee_input_ids", ())
            ))
            positions = tuple(
                index for index, value_id in enumerate(callee_inputs)
                if value_id == int(descriptor.length_address_id)
            )
            if len(positions) != 1 or positions[0] >= len(call.args):
                continue
            length = call.args[positions[0]]
            changed = False

            def post_view(value_id, dtype, shape, physical=None):
                """Program: a generator call whose projected row columns,
                result and row loads take the callee descriptor's layout.
                Wrong record: their occurrences were retyped and reshaped
                in place (rank in accounting only) with no ``ssa_value``
                revision.  Rule: one REVISE per value of its row, DERIVED
                from the callee's sequence descriptor cell; the occurrences
                and their accounting are views of that row."""

                from .concordance_declarations import (
                    KEEP_PHYSICAL, SEQUENCE_DESCRIPTOR as _DESCRIPTOR,
                )
                from .frame_identity_book import _frame_table_cell
                from .ssa_record_return_state import revise_ssa_value

                descriptor_cell = _frame_table_cell(
                    table, int(descriptor.sequence_id), _DESCRIPTOR,
                    function=callee,
                )
                revise_ssa_value(
                    function, int(value_id), dtype=dtype, shape=tuple(shape),
                    stage="projected_iterable_layout",
                    sources=tuple(
                        cell for cell in (descriptor_cell,) if cell is not None
                    ),
                    physical=KEEP_PHYSICAL if physical is None else physical,
                )

            def accounting_view(value, labels):
                from .frame_identity_book import (
                    _frame_accounting_from_posted, _frame_physical_fields,
                )

                return _frame_accounting_from_posted(
                    {**dict(value.accounting or {}), **labels},
                    _frame_physical_fields(function, int(value.id)),
                )

            if call.res is not None:
                result_id = int(call.res.id)
                result_contract = tuple(
                    call.attributes.get("native_result_contract", ())
                )
                result_shape = (
                    tuple(result_contract[0][2])
                    if len(result_contract) == 1
                    and len(result_contract[0]) >= 3
                    else tuple(call.res.shape or ())
                )
                result_dtype = (
                    str(result_contract[0][1])
                    if len(result_contract) == 1
                    and len(result_contract[0]) >= 2
                    and result_contract[0][1] not in {None, "", "unknown"}
                    else call.res.dtype
                )
                post_view(
                    result_id,
                    result_dtype
                    if result_dtype not in {None, "", "unknown"}
                    else call.res.dtype,
                    result_shape,
                )
                for value in occurrences:
                    if int(value.id) != result_id:
                        continue
                    value.shape = result_shape
                    if result_dtype not in {None, "", "unknown"}:
                        value.dtype = str(result_dtype)
                    changed = True
                state = {
                    "shape": tuple(result_shape), "rank": len(result_shape),
                    "dtype": str(result_dtype or "unknown"),
                }
                # The layout is read off the call's result value: the edge
                # derives from its ``ssa_value`` cell.
                from .concordance_declarations import SSA_VALUE as _SSA_VALUE
                from .ssa_record_return_state import function_scope_of

                result_cell = current_identity_book().latest_ref(
                    _SSA_VALUE, (function_scope_of(function), result_id),
                )
                record_shape_transformation(
                    shape_scope_of(callee), ("return", str(callee_name)),
                    shape_scope_of(function), result_id,
                    stage="projected_iterable_layout",
                    operation="call_result", source_state=state,
                    target_state=state, role="return",
                    source_cells=() if result_cell is None else (result_cell,),
                )
            source_receipt = call.attributes.get("plan_callsite_id")
            if source_receipt is None:
                if changed:
                    receipts.append((
                        None, callee_name,
                        int(descriptor.sequence_id), int(length.id),
                        None if call.res is None else int(call.res.id),
                    ))
                continue
            source_id = int(source_receipt)
            projected_rows = tuple(sorted(
                (
                    int(row[1]), int(row[2])
                )
                for row in function.metadata.get(
                    "projected_row_tables", ()
                )
                if len(row) >= 3 and int(row[0]) == source_id
            ))
            physical_rows = tuple(
                row for row in projected_rows if row[1] != source_id
            )
            if len(projected_rows) == len(descriptor.column_value_ids):
                descriptor_column_by_projection = tuple(
                    (projection, index)
                    for index, (projection, _value_id)
                    in enumerate(projected_rows)
                )
            elif len(physical_rows) == len(descriptor.column_value_ids):
                # A yielded record handle remains a graph identity while its
                # sibling tensors occupy physical sequence columns.  The row
                # table marks that handle by retaining the driver id itself;
                # excluding precisely that receipt aligns the remaining
                # projections with the callee's physical descriptor.
                descriptor_column_by_projection = tuple(
                    (projection, index)
                    for index, (projection, _value_id)
                    in enumerate(physical_rows)
                )
            else:
                descriptor_column_by_projection = ()

            def descriptor_column(projection: int) -> int | None:
                matches = tuple(
                    index
                    for candidate, index
                    in descriptor_column_by_projection
                    if candidate == int(projection)
                )
                return matches[0] if len(matches) == 1 else None

            # Storage-returning collection calls have no SSA result: their
            # source callsite value is the semantic sequence view.  Publish
            # the callee's leased arena and live length on that exact value.
            # A scalar/tensor return has ``call.res`` and is handled by its
            # native result contract above instead.
            if call.res is None:
                row_shape = tuple(
                    (descriptor.column_shapes or ((),))[0]
                )
                column_dtype = (
                    descriptor.column_dtypes[0]
                    if descriptor.column_dtypes else None
                )
                source_occurrence = next((
                    value for value in occurrences
                    if int(value.id) == source_id
                ), None)
                if source_occurrence is not None:
                    post_view(
                        source_id,
                        str(column_dtype)
                        if column_dtype not in {None, "", "unknown"}
                        else source_occurrence.dtype,
                        tuple(source_occurrence.shape or ()),
                        {"rank": 1 + len(row_shape)},
                    )
                for value in occurrences:
                    if int(value.id) != source_id:
                        continue
                    if column_dtype not in {None, "", "unknown"}:
                        value.dtype = str(column_dtype)
                    value.accounting = accounting_view(value, {
                        "sequence_id": int(descriptor.sequence_id),
                        "sequence_length_value_id": int(length.id),
                        "tensor_metadata_state": "dynamic",
                        "sequence_row_shape": row_shape,
                    })
                    changed = True
            for value in occurrences:
                accounting = dict(value.accounting or {})
                if int(accounting.get(
                    "projected_row_source_id", -1
                )) != source_id:
                    continue
                projection = int(
                    accounting.get("projected_row_column") or 0
                )
                column = descriptor_column(projection)
                value.accounting = {
                    **accounting,
                    "sequence_id": source_id,
                    "sequence_length_value_id": int(length.id),
                    "tensor_metadata_state": "dynamic",
                }
                if column is None:
                    continue
                row_shape = tuple(
                    (descriptor.column_shapes or tuple(
                        () for _ in descriptor.column_value_ids
                    ))[column]
                )
                column_dtype = (
                    descriptor.column_dtypes[column]
                    if column < len(descriptor.column_dtypes)
                    else value.dtype
                )
                post_view(
                    int(value.id),
                    str(column_dtype)
                    if column_dtype not in {None, "", "unknown"}
                    else value.dtype,
                    tuple(value.shape or ()),
                    {"rank": 1 + len(row_shape)},
                )
                if column_dtype not in {None, "", "unknown"}:
                    value.dtype = str(column_dtype)
                value.accounting = accounting_view(value, {
                    "sequence_row_shape": row_shape,
                })
                changed = True
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if (
                        instruction.res is None
                        or str(instruction.op).casefold() != "load"
                        or instruction.attributes.get("binding")
                        != "projected_iterable"
                        or not instruction.args
                    ):
                        continue
                    source = instruction.args[0]
                    # Projected iteration lowers as column -> GEP -> Load.
                    # The pointer is intentionally untyped; recover the
                    # column only through its unique exact producer edge.
                    producer = next((
                        candidate
                        for candidate_block in function.blocks.values()
                        for candidate in candidate_block.instrs
                        if candidate.res is not None
                        and int(candidate.res.id) == int(source.id)
                    ), None)
                    if (
                        producer is not None
                        and str(producer.op).casefold()
                        in {"getelementptr", "gep"}
                        and producer.args
                    ):
                        source = producer.args[0]
                    source_accounting = source.accounting or {}
                    if int(source_accounting.get(
                        "projected_row_source_id", -1
                    )) != source_id:
                        continue
                    projection = int(source_accounting.get(
                        "projected_row_column", 0
                    ))
                    column = descriptor_column(projection)
                    if column is None:
                        continue
                    row_shape = tuple(
                        (descriptor.column_shapes or tuple(
                            () for _ in descriptor.column_value_ids
                        ))[column]
                    )
                    column_dtype = (
                        descriptor.column_dtypes[column]
                        if column < len(descriptor.column_dtypes)
                        else instruction.res.dtype
                    )
                    target_id = int(instruction.res.id)
                    post_view(
                        target_id,
                        str(column_dtype)
                        if column_dtype not in {None, "", "unknown"}
                        else instruction.res.dtype,
                        row_shape, {"rank": len(row_shape)},
                    )
                    for value in occurrences:
                        if int(value.id) != target_id:
                            continue
                        value.shape = row_shape
                        if column_dtype not in {None, "", "unknown"}:
                            value.dtype = str(column_dtype)
                        value.accounting = accounting_view(value, {
                            "sequence_row_shape": row_shape,
                        })
                    state = {
                        "shape": row_shape, "rank": len(row_shape),
                        "dtype": str(column_dtype or "unknown"),
                    }
                    record_shape_transformation(
                        shape_scope_of(callee),
                        (
                            "sequence_column",
                            int(descriptor.sequence_id), int(column),
                        ),
                        shape_scope_of(function), target_id,
                        stage="projected_iterable_layout",
                        operation="sequence_row_column",
                        source_state=state, target_state=state,
                        role=f"column:{int(column)}",
                    )
                    changed = True
            if changed:
                receipts.append((
                    source_id, callee_name,
                    int(descriptor.sequence_id), int(length.id),
                    None if call.res is None else int(call.res.id),
                ))
        if receipts:
            function.metadata[
                "projected_iterable_layout_receipts"
            ] = tuple(receipts)


@dataclass(frozen=True, slots=True)
class Claim:
    """One record's statement about a value's identity."""

    kind: str
    key: str
    source: str

    def __setstate__(self, state):
        # Accept old dict-backed audit archives as well as slotted pickles.
        names = ("kind", "key", "source")
        values = (state[name] for name in names) if isinstance(state, dict) else state
        for name, value in zip(names, values):
            object.__setattr__(self, name, value)


@dataclass(slots=True)
class ValueRow:
    function: str
    value_id: int
    dtype: str | None = None
    is_formal: bool = False
    claims: list[Claim] = field(default_factory=list)
    definitions: list[tuple[str, int, str]] = field(default_factory=list)
    uses: list[tuple[str, int, str]] = field(default_factory=list)

    def __setstate__(self, state):
        # Default slotted state is (None, slot_dict); earlier archives used
        # the field dict directly. Lists remain the same mutable owners.
        values = state if isinstance(state, dict) else state[1]
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str
    function: str
    value_id: int | None
    detail: str

    def __setstate__(self, state):
        names = ("kind", "function", "value_id", "detail")
        values = (state[name] for name in names) if isinstance(state, dict) else state
        for name, value in zip(names, values):
            object.__setattr__(self, name, value)


_INOUT_REDEFINED = ("program_abi_mutable", "program_abi_field_written")
#: Finding kinds ``concordance_report`` counts on their own line rather than
#: in its gating first line.
# ssa-physical-unposted is the stage-3 migration worklist (accounting set, fact
# unset); retire it from this set when the last producer posts its physical fields.
_PHYSICAL_ACCOUNTING_KEYS = ("physical_dtype", "program_abi_storage",
                             "program_abi_rank", "sequence_arena", "device")


def _physical_unset(key: str, item: Any) -> bool:
    """Only ``None`` means unset, so rank 0 is a value.  ``sequence_arena`` is
    the one bool key: its writers store ``value_id in column_value_ids`` (False
    = not an arena member) and ``SSAValueFact.arena`` defaults False."""
    return item is None or (key == "sequence_arena" and item is False)


_UNSOURCED_KINDS = frozenset({"unsourced-fact", "unsourced-identity",
                              "ssa-physical-unposted"})


_IDENTITY_PAGES = frozenset(("canonical_value", "ingestion_value"))


def _canonical_identity_base(book: Any, ref: Any) -> Any:
    """The first copy of an identity cell: a ``canonical_value`` row of a
    read-scope fork derives from the row it copies (``read_scope_fork``),
    so the cell is followed to the copy that has no such parent."""
    seen = set()
    while ref not in seen:
        seen.add(ref)
        parents = [
            source for source, _stage in book.edges_into(ref)
            if source.page.name == ref.page.name
            and isinstance(source.row, tuple) and len(source.row) == 2
            and isinstance(ref.row, tuple) and len(ref.row) == 2
            and source.row[1] == ref.row[1]
        ]
        if len(parents) != 1:
            return ref
        ref = parents[0]
    return ref


def _identity_bases_of(book: Any, ref: Any, value_id: int) -> set:
    """The canonical identity cells (``_canonical_identity_base``) of
    ``value_id`` that ``ref`` derives from, walking its inbound edges up to
    and not through the identity pages.  Empty when the ancestry reaches
    none."""
    found = set()
    seen = {ref}
    frontier = [ref]
    depth = 0
    while frontier and depth < 12:
        next_frontier = []
        for cell in frontier:
            for source, _stage in book.edges_into(cell):
                if source in seen:
                    continue
                seen.add(source)
                if source.page.name in _IDENTITY_PAGES:
                    if (
                        isinstance(source.row, tuple)
                        and len(source.row) == 2
                        and source.row[1] == value_id
                    ):
                        found.add(_canonical_identity_base(book, source))
                    continue
                next_frontier.append(source)
        frontier = next_frontier
        depth += 1
    return found


def _source_class_subject_bases(book: Any, ref: Any, value_id: int) -> set:
    """The identity bases of the value a ``source_value_class`` row is ABOUT.

    Program: ``root`` calls ``hyp(a, b)`` at graph node 2; ``hyp``'s
    parameter ``b`` is also value 2 in ``hyp``.  The class row of ``hyp``'s
    ``b`` derives from the call node (a cause) as well as from ``b``'s own
    cell.  Wrong join: ``_identity_bases_of`` admitted every identity cell
    with the same integer id, so the class of ``hyp``'s ``b`` was taken as
    the class of ``root``'s call result and the call result's type ABI was
    reported as not deriving from it.  Rule: a class row keyed by scope
    ``s`` (or a specialization ``(s, 'specialization', h)``) is about the
    value in the read scope the reducer minted from that exact scope,
    ``lexical_reads:{s}``; only identity bases in that scope are its
    subject.
    """
    row_scope = ref.row[0] if isinstance(ref.row, tuple) else None
    if (
        isinstance(row_scope, tuple) and len(row_scope) == 3
        and row_scope[1] == "specialization"
    ):
        row_scope = row_scope[0]
    label = f"lexical_reads:{row_scope}"
    return {
        base for base in _identity_bases_of(book, ref, value_id)
        if isinstance(base.row, tuple) and base.row
        and isinstance(base.row[0], tuple) and base.row[0]
        and base.row[0][0] == label
    }


def scope_local_accounting_ids(value: Any) -> tuple[int, ...]:
    """The SSA value ids of the same function that ``value``'s accounting
    names (``CorrelationTable._SCOPE_LOCAL_ACCOUNTING_IDS``): its type tag,
    heap owner, optional payload, iterable target and tensor metadata.  They
    are part of ``value``'s identity, so a retained value keeps them -- every
    dead-value and dead-formal pass reads this one definition."""

    accounting = getattr(value, "accounting", None) or {}
    return tuple(
        int(accounting[key])
        for key in CorrelationTable._SCOPE_LOCAL_ACCOUNTING_IDS
        if accounting.get(key) is not None
        and not isinstance(accounting.get(key), bool)
    )


class CorrelationTable:
    """Every identity claim the module makes, keyed by (function, value id)."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, int], ValueRow] = {}
        self.sequence_roles: dict[tuple[str, int], list[tuple[int, str]]] = (
            defaultdict(list)
        )

    # ------------------------------------------------------------------ build
    def row(self, function: str, value_id: int, dtype: Any = None) -> ValueRow:
        key = (str(function), int(value_id))
        row = self.rows.get(key)
        if row is None:
            row = ValueRow(str(function), int(value_id), dtype=None)
            self.rows[key] = row
        if dtype is not None and row.dtype is None:
            row.dtype = str(dtype)
        return row

    def claim(self, function: str, value_id: int, kind: str, key: Any,
              source: str) -> None:
        self.row(function, value_id).claims.append(
            Claim(str(kind), str(key), str(source))
        )

    @classmethod
    def build(cls, module: Any) -> "CorrelationTable":
        table = cls()
        for name, function in module.functions.items():
            table._build_function(module, str(name), function)
        return table

    def _build_function(self, module: Any, name: str, function: Any) -> None:
        metadata = dict(getattr(function, "metadata", {}) or {})
        for formal in function.args:
            row = self.row(name, int(formal.id), formal.dtype)
            row.is_formal = True
            accounting = dict(formal.accounting or {})
            abi_field = accounting.get("program_abi_field")
            abi_parameter = accounting.get("program_abi_parameter")
            if abi_field is not None:
                self.claim(
                    name, formal.id, "abi-field",
                    f"{accounting.get('program_abi_record') or abi_parameter}"
                    f".{abi_field}",
                    "accounting.program_abi_field",
                )
            elif abi_parameter is not None:
                self.claim(
                    name, formal.id, "abi-parameter", abi_parameter,
                    "accounting.program_abi_parameter",
                )
            if (
                accounting.get("native_type_tag_for_value_id") is not None
                and accounting.get("source_parameter")
            ):
                self.claim(
                    name, formal.id, "abi-parameter",
                    f"{accounting['source_parameter']}.__type_tag",
                    "accounting.native_type_tag_for_value_id",
                )
            if accounting.get("program_abi_keyed_owner") is not None:
                self.claim(
                    name, formal.id, "keyed-part",
                    f"{accounting['program_abi_keyed_owner']}."
                    f"{accounting.get('program_abi_keyed_part')}",
                    "accounting.program_abi_keyed_owner",
                )
            if accounting.get("linked_call_frame_storage"):
                self.claim(
                    name, formal.id, "frame-storage",
                    f"{accounting['linked_call_frame_storage']}"
                    f"#{accounting.get('propagated_formal_id', '?')}",
                    "accounting.linked_call_frame_storage",
                )
            if accounting.get("compiler_frame_storage"):
                self.claim(
                    name, formal.id, "compiler-frame-storage",
                    accounting["compiler_frame_storage"],
                    "accounting.compiler_frame_storage",
                )
            if accounting.get("projected_row_source_id") is not None:
                self.claim(
                    name, formal.id, "projected-row",
                    f"{accounting['projected_row_source_id']}"
                    f"[{accounting.get('projected_row_column')}]",
                    "accounting.projected_row_source_id",
                )
        for label, value_id in metadata.get("parameter_names", ()) or ():
            self.claim(name, value_id, "parameter", label,
                       "metadata.parameter_names")
        for value_id, source_name, transform in (
            metadata.get("scalar_source_transforms", ()) or ()
        ):
            self.claim(
                name, value_id, "source-transform",
                f"{source_name}:{transform}",
                "metadata.scalar_source_transforms",
            )
        for entry in metadata.get("storage_formals", ()) or ():
            self.claim(name, entry["value_id"], "storage-formal",
                       entry.get("kind", "storage"), "metadata.storage_formals")
        for entry in metadata.get("closure_formals", ()) or ():
            self.claim(name, entry["value_id"], "closure-formal",
                       entry.get("name", "?"), "metadata.closure_formals")
        for entry in metadata.get("parameter_member_formals", ()) or ():
            self.claim(
                name, entry["value_id"], "member-formal",
                f"{entry.get('parameter')}{list(entry.get('path', ()))}",
                "metadata.parameter_member_formals",
            )
        aliases = metadata.get("value_aliases", ()) or ()
        alias_pairs = (
            aliases.items() if isinstance(aliases, Mapping) else aliases
        )
        for pair in alias_pairs:
            try:
                alias, target = pair
            except (TypeError, ValueError):
                continue
            self.claim(name, alias, "alias-of", int(target),
                       "metadata.value_aliases")
        for pair in metadata.get("output_identity_aliases", ()) or ():
            try:
                alias, target = pair
            except (TypeError, ValueError):
                continue
            # Output identity is semantic history, not substitutable physical
            # storage.  The target can be an edge occurrence retired after
            # its merged output is published, so applying the physical-alias
            # existence rule here reports a missing storage definition that
            # this record never claimed.  Keep the relation visible to the
            # table under its own exact kind; agreement with the authoritative
            # output page is checked separately below.
            self.claim(name, alias, "output-identity-of", int(target),
                       "metadata.output_identity_aliases")
        # A value of a named byte-layout type (a struct/union base address)
        # says so in its accounting; the claim is checked against the
        # module's struct/union tables by ``_layout_table_findings``.
        def claim_layout_type(value: Any) -> None:
            accounting = dict(getattr(value, "accounting", None) or {})
            kind = accounting.get("ssa_layout_kind")
            if kind is None:
                return
            self.claim(
                name, value.id, "layout-type",
                f"{kind}:{accounting.get('ssa_layout_identity')}",
                "accounting.ssa_layout_kind",
            )

        for formal in function.args:
            claim_layout_type(formal)
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is not None:
                    row = self.row(name, int(instruction.res.id),
                                   instruction.res.dtype)
                    row.definitions.append((block_name, index, instruction.op))
                    claim_layout_type(instruction.res)
                for argument in instruction.args:
                    argument_id = getattr(argument, "id", None)
                    if argument_id is None:
                        continue
                    self.row(name, int(argument_id)).uses.append(
                        (block_name, index, instruction.op)
                    )
        sequence_table = (getattr(module, "sequence_tables", {}) or {}).get(name)
        if sequence_table is not None:
            for descriptor in sequence_table.sequences.values():
                roles = [
                    (int(descriptor.sequence_id), "handle"),
                    *(
                        (int(column), f"column{position}")
                        for position, column in enumerate(
                            descriptor.column_value_ids
                        )
                    ),
                    (int(descriptor.length_address_id), "length"),
                    (int(descriptor.capacity_value_id), "capacity"),
                ]
                if descriptor.status_address_id is not None:
                    roles.append((int(descriptor.status_address_id), "status"))
                if descriptor.live_flags_value_id is not None:
                    roles.append(
                        (int(descriptor.live_flags_value_id), "live_flags")
                    )
                if getattr(descriptor, "heap_owner_value_id", None) is not None:
                    roles.append((int(descriptor.heap_owner_value_id), "heap_owner"))
                for value_id, role in roles:
                    self.claim(
                        name, value_id, "sequence-member",
                        f"seq{int(descriptor.sequence_id)}.{role}",
                        "sequence_table",
                    )
                    self.sequence_roles[(name, value_id)].append(
                        (int(descriptor.sequence_id), role)
                    )
        record_table = (getattr(module, "record_tables", {}) or {}).get(name)
        if record_table is not None:
            for record in record_table.records.values():
                self.claim(
                    name, int(record.record_id), "record-owner",
                    str(record.identity), "record_table",
                )
                for record_field in record.fields:
                    for value_id in record_field.value_ids:
                        self.claim(
                            name, value_id, "record-field",
                            f"{record.identity}#{record.record_id}"
                            f".{record_field.name}",
                            "record_table",
                        )

    # --------------------------------------------------------------- findings
    def findings(self, module: Any) -> list[Finding]:
        found: list[Finding] = []
        components = (getattr(module, "metadata", {}) or {}).get("ssa_composition_components", ())
        if components:
            owners = (module.metadata or {}).get("ssa_composition_function_books", {})
            expected = {}
            for component in components:
                if component.function_table is not module.function_table:
                    found.append(Finding("ssa-composition-function-owner-disagreement", "?", None,
                                         "component does not share the canonical source function table"))
                found.extend(CorrelationTable.build(component).findings(component))
                book = (component.metadata or {}).get("identity_book")
                if book is not None:
                    expected.update({name: (function.metadata.get("accepted_ssa_component") or {}).get(
                        "identity_book", book) for name, function in component.functions.items()})
            for name, book in expected.items():
                if owners.get(name) is not book:
                    found.append(Finding("ssa-composition-book-owner-disagreement", str(name), None,
                                         "composed function lost its original provenance book"))
        for name, table in (getattr(module, "record_tables", {}) or {}).items():
            sequences = (getattr(module, "sequence_tables", {}) or {}).get(name)
            for record in table.records.values():
                for record_field in record.fields:
                    if record_field.sequence_id is not None and (
                            sequences is None or sequences.by_id(record_field.sequence_id) is None):
                        found.append(Finding("record-sequence-unknown", str(name), int(record.record_id),
                            f"field {record_field.storage_identity!r} names absent sequence {record_field.sequence_id}"))
        found.extend(self._native_interface_findings(module))
        found.extend(self._region_instruction_residency_findings(module))
        found.extend(self._repository_ssa_signature_findings(module))
        found.extend(self._scope_local_accounting_findings(module))
        for name, function in module.functions.items():
            found.extend(self._function_findings(module, str(name), function))
            found.extend(
                self._undefined_operand_findings(str(name), function)
            )
            declared = self._loop_scope_findings(module, str(name), function)
            found.extend(declared)
            # The inferred check recovers the scope from branch topology; it
            # covers loops built by a path that does not declare one.  Where a
            # declaration exists it is authoritative, so do not report twice.
            if not loop_scope_declarations(identity_book(module), str(name)):
                found.extend(
                    self._stale_carried_read_findings(str(name), function)
                )
        found.extend(self._return_site_edge_findings(module))
        found.extend(self._sequence_descriptor_findings(module))
        found.extend(self._binding_kind_findings(module))
        found.extend(self._name_binding_capture_findings(module))
        found.extend(self._conditional_binding_preservation_findings(module))
        found.extend(self._source_page_declaration_findings(module))
        found.extend(self._graph_work_publication_findings(module))
        found.extend(self._tile_storage_findings(module))
        found.extend(self._source_field_identity_findings(module))
        found.extend(self._source_parameter_identity_findings(module))
        found.extend(self._source_numeric_component_findings(module))
        found.extend(self._source_numeric_intrinsic_findings(module))
        found.extend(self._source_numeric_type_dependency_findings(module))
        found.extend(self._source_numeric_operator_specialization_findings(module))
        found.extend(self._source_sequence_mutation_findings(module))
        found.extend(self._source_aggregate_layout_findings(module))
        found.extend(self._unborn_display_findings(module))
        found.extend(self._live_structure_findings(module))
        found.extend(self._ssa_publication_findings(module))
        found.extend(self._value_type_abi_findings(module))
        found.extend(self._source_callsite_activation_findings(module))
        found.extend(self._source_precision_boundary_findings(module))
        found.extend(self._source_precision_operator_findings(module))
        found.extend(self._source_precision_region_findings(module))
        found.extend(self._operator_result_type_findings(module))
        found.extend(self._planning_alias_transition_findings(module))
        found.extend(self._tensor_reduction_domain_findings(module))
        found.extend(self._callable_identity_findings(module))
        found.extend(self._post_ssa_numeric_identity_findings(module))
        found.extend(self._table_member_findings(module))
        found.extend(self._layout_table_findings(module))
        found.extend(self._operand_position_orphan_findings(module))
        # Last, so callers that show the first few findings still see the
        # per-page kinds first; ``concordance_report`` counts these two kinds
        # on their own line.
        found.extend(self._unsourced_findings(module))
        return found

    @staticmethod
    def _value_type_abi_findings(module: Any) -> list[Finding]:
        """Validate the native type discriminant as a view of value identity.

        Dense integers are deliberately not identities.  The linked
        ``value_type_abi`` row owns the canonical alternative table; every
        function-local ``value_type_tag`` must repeat that exact table and
        derive from its ABI row.  This makes an accidentally re-numbered
        subset visible before a backend can compare incompatible tags.
        """

        from .concordance_declarations import (
            SOURCE_VALUE_CLASS, UNION_ALTERNATIVE_STATE,
            VALUE_TYPE_ABI, VALUE_TYPE_TAG,
            UnionAlternativeStateFact,
            ValueTypeABIFact, ValueTypeTagFact,
            canonical_python_type_identity,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        if book is None:
            return []
        abi_page = (getattr(book, "pages", {}) or {}).get(
            VALUE_TYPE_ABI.name
        )
        tag_page = (getattr(book, "pages", {}) or {}).get(
            VALUE_TYPE_TAG.name
        )
        union_state_page = (getattr(book, "pages", {}) or {}).get(
            UNION_ALTERNATIVE_STATE.name
        )
        found: list[Finding] = []
        source_classes_by_value: dict[int, list[tuple[str, Ref]]] = {}
        source_class_page = (getattr(book, "pages", {}) or {}).get(
            SOURCE_VALUE_CLASS.name
        )
        if source_class_page is not None:
            for source_row in source_class_page.rows():
                if not (
                    isinstance(source_row, tuple)
                    and len(source_row) == 2
                ):
                    continue
                source_fact = source_class_page.latest(source_row)
                source_ref = book.latest_ref(SOURCE_VALUE_CLASS, source_row)
                if (
                    source_ref is not None
                    and isinstance(source_fact, tuple)
                    and source_fact
                ):
                    source_classes_by_value.setdefault(
                        int(source_row[1]), []
                    ).append((
                        canonical_python_type_identity(source_fact[0]),
                        source_ref,
                    ))
        abi_by_value: dict[int, list[tuple[Any, ValueTypeABIFact, Ref]]] = {}
        if abi_page is not None:
            for row in abi_page.rows():
                fact = abi_page.latest(row)
                scope, value_id = (
                    row if isinstance(row, tuple) and len(row) == 2
                    else (row, None)
                )
                problem = None
                alternatives = tuple(getattr(fact, "alternatives", ()))
                tags = tuple(getattr(fact, "tags", ()))
                physical = tuple(
                    getattr(fact, "physical_alternatives", ())
                )
                if not isinstance(fact, ValueTypeABIFact):
                    problem = f"invalid value type ABI fact {fact!r}"
                elif (
                    not alternatives
                    or alternatives != tuple(sorted(set(alternatives)))
                    or any(not isinstance(identity, str) or not identity
                           for identity in alternatives)
                ):
                    problem = (
                        "type alternatives are not unique canonical "
                        f"identities: {alternatives!r}"
                    )
                elif len(alternatives) == 1 and tags:
                    problem = "an exact type must not allocate a runtime tag"
                elif len(alternatives) > 1 and tags != tuple(
                    (identity, ordinal)
                    for ordinal, identity in enumerate(alternatives)
                ):
                    problem = (
                        "union tag table is not the dense view of its exact "
                        f"alternatives: {tags!r}"
                    )
                elif tuple(
                    getattr(arm, "identity", None) for arm in physical
                ) != alternatives:
                    problem = (
                        "type ABI semantic alternatives do not have one "
                        f"ordered physical arm each: {physical!r}"
                    )
                elif any(
                    not tuple(getattr(arm, "source_cells", ()))
                    or (
                        getattr(arm, "identity", "")
                        != "builtins.NoneType"
                        and not getattr(arm, "storage", "")
                    )
                    or (
                        getattr(arm, "identity", "")
                        == "builtins.NoneType"
                        and (
                            getattr(arm, "storage", "") != "none"
                            or getattr(arm, "dtype", "")
                            or tuple(getattr(arm, "shape", ()))
                        )
                    )
                    for arm in physical
                ):
                    problem = (
                        "type ABI has an unrepresented or malformed physical "
                        f"alternative: {physical!r}"
                    )
                ref = (
                    None if value_id is None else
                    book.latest_ref(VALUE_TYPE_ABI, row)
                )
                if problem is None and ref is not None:
                    sources = tuple(
                        source for source, _stage in book.edges_into(ref)
                    )
                    if any(
                        cell not in sources
                        for arm in physical
                        for cell in arm.source_cells
                    ):
                        problem = (
                            "type ABI physical arm does not derive from its "
                            "exact source cell"
                        )
                    if not any(
                        source.page.name in {
                            "canonical_value", "ingestion_value", "ssa_value"
                        }
                        and isinstance(source.row, tuple)
                        and len(source.row) == 2
                        and int(source.row[1]) == int(value_id)
                        for source in sources
                    ):
                        problem = "type ABI does not derive from its value identity"
                    # A class row is this value's only when both stand on
                    # one canonical identity cell: the dense id alone is
                    # not one (``root``'s formal 15 and ``execute``'s call
                    # result 15 are two values).  A class row whose
                    # ancestry reaches no identity cell is still counted.
                    abi_bases = set()
                    for source in sources:
                        if (
                            source.page.name in {
                                "canonical_value", "ingestion_value",
                                "ssa_value",
                            }
                            and isinstance(source.row, tuple)
                            and len(source.row) == 2
                            and int(source.row[1]) == int(value_id)
                        ):
                            abi_bases.add(
                                _canonical_identity_base(book, source)
                            )
                    ordinary = tuple(
                        (identity, source_ref)
                        for identity, source_ref
                        in source_classes_by_value.get(int(value_id), ())
                        if identity in alternatives
                        and (
                            (
                                class_bases := _source_class_subject_bases(
                                    book, source_ref, int(value_id),
                                )
                            ) & abi_bases
                            or (
                                not class_bases
                                and isinstance(source_ref.row, tuple)
                                and len(source_ref.row) == 2
                                and source_ref.row[0] == scope
                            )
                        )
                    )
                    if ordinary and not any(
                        source_ref in sources
                        for _identity, source_ref in ordinary
                    ):
                        problem = (
                            "ordinary type ABI does not derive directly from "
                            "its exact source_value_class cell"
                        )
                if problem is not None:
                    found.append(Finding(
                        "value-type-abi-disagreement", str(scope),
                        None if value_id is None else int(value_id), problem,
                    ))
                elif ref is not None:
                    abi_by_value.setdefault(int(value_id), []).append(
                        (scope, fact, ref)
                    )
        if union_state_page is not None:
            for row in union_state_page.rows():
                fact = union_state_page.latest(row)
                scope, value_id = (
                    row if isinstance(row, tuple) and len(row) == 2
                    else (row, None)
                )
                problem = None
                if not isinstance(fact, UnionAlternativeStateFact):
                    problem = f"invalid union alternative state {fact!r}"
                elif value_id is None:
                    problem = f"invalid union alternative state row {row!r}"
                elif len(fact.alternatives) != len(fact.states):
                    problem = (
                        "union alternative state has unequal identity/state "
                        f"arity: {fact!r}"
                    )
                else:
                    state_ref = book.latest_ref(UNION_ALTERNATIVE_STATE, row)
                    sources = set(
                        source.key for source, _stage
                        in book.edges_into(state_ref)
                    ) if state_ref is not None else set()
                    tag_ref = book.latest_ref(
                        VALUE_TYPE_TAG,
                        (scope, int(value_id), int(fact.tag_value_id)),
                    )
                    candidates = [
                        (abi, ref) for _abi_scope, abi, ref
                        in abi_by_value.get(int(value_id), ())
                        if tuple(abi.alternatives) == tuple(fact.alternatives)
                    ]
                    if len(candidates) != 1:
                        problem = (
                            "union alternative state has no unique exact "
                            "value type ABI"
                        )
                    else:
                        abi, abi_ref = candidates[0]
                        expected_states = tuple((
                            tuple(arm.shape), str(arm.dtype), int(arm.rank),
                            str(arm.metadata_state),
                            None if arm.row_shape is None
                            else tuple(arm.row_shape),
                        ) for arm in abi.physical_alternatives)
                        if tuple(fact.states) != expected_states:
                            problem = (
                                "union alternative state disagrees with its "
                                f"physical ABI: {fact.states!r} != "
                                f"{expected_states!r}"
                            )
                        elif abi_ref.key not in sources:
                            problem = (
                                "union alternative state does not derive "
                                "directly from its value type ABI"
                            )
                        elif tag_ref is None or tag_ref.key not in sources:
                            problem = (
                                "union alternative state does not derive "
                                "directly from its value type tag"
                            )
                if problem is not None:
                    found.append(Finding(
                        "union-alternative-state-disagreement", str(scope),
                        None if value_id is None else int(value_id), problem,
                    ))
        if tag_page is None:
            return found
        for row in tag_page.rows():
            fact = tag_page.latest(row)
            scope, value_id, row_tag_value_id = (
                row if isinstance(row, tuple) and len(row) == 3
                else (row, None, None)
            )
            problem = None
            if not isinstance(fact, ValueTypeTagFact):
                problem = f"invalid value type tag fact {fact!r}"
            elif value_id is None:
                problem = f"invalid value type tag row {row!r}"
            elif int(row_tag_value_id) != int(fact.tag_value_id):
                problem = (
                    "tag row identity disagrees with its fact: "
                    f"row={row_tag_value_id!r}, fact={fact.tag_value_id!r}"
                )
            else:
                tag_ref = book.latest_ref(VALUE_TYPE_TAG, row)
                sources = set(
                    source.key for source, _stage in book.edges_into(tag_ref)
                ) if tag_ref is not None else set()
                candidates = [
                    (abi, ref) for _abi_scope, abi, ref
                    in abi_by_value.get(int(value_id), ())
                    if (
                        tuple(abi.alternatives) == tuple(fact.alternatives)
                        and tuple(abi.tags) == tuple(fact.tags)
                        and ref.key in sources
                    )
                ]
                if len(candidates) != 1:
                    problem = (
                        "tag has no unique directly-derived authoritative "
                        "ABI table: "
                        f"alternatives={fact.alternatives!r}, tags={fact.tags!r}"
                    )
            if problem is not None:
                found.append(Finding(
                    "value-type-tag-disagreement", str(scope),
                    None if value_id is None else int(value_id), problem,
                ))
        return found

    @staticmethod
    def _source_page_declaration_findings(module: Any) -> list[Finding]:
        """A source Page view must resolve through its sole authored cell.

        ProcessGraph consumers retain only ``source_page_declaration_cell``;
        page name, fact class, and RowField order are read from that cell.  A
        malformed fact or an edge to another module would otherwise make a
        projected unit appear typed while pointing at a copied/local schema.
        """

        from .concordance_declarations import (
            CLASS_DECLARATION, CLASS_METHOD_DECLARATION,
            SOURCE_PAGE_FACT_CLASS,
            SOURCE_ENUM_GLYPH_TABLE, STATIC_SYMBOL_DEFINITION,
            SourcePageFactClass,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        if book is None or SOURCE_PAGE_FACT_CLASS.name not in book.pages:
            return []
        found = []
        page = book.page(SOURCE_PAGE_FACT_CLASS)
        for row in page.rows():
            fact = page.latest(row)
            cell = book.latest_ref(SOURCE_PAGE_FACT_CLASS, row)
            sources = (
                () if cell is None else book.edges_into(cell)
            )
            span_modules = {
                str(source.row[0])
                for source, _stage in sources
                if source.page.name == "source_span"
                and isinstance(source.row, tuple)
                and source.row
            }
            class_rows = tuple(
                source.row
                for source, _stage in sources
                if source.page == CLASS_DECLARATION
            )
            class_identity = str(getattr(fact, "class_identity", ""))
            fact_class_rows = tuple(
                candidate for candidate in class_rows
                if len(candidate) == 2
                and ".".join(map(str, candidate)) == class_identity
            )
            # A locally declared fact class must cite its declaration.  An
            # imported class edge legitimately names another module; when
            # present it must still agree with the authored class spelling.
            # Builtins/unadmitted external types have no CLASS_DECLARATION
            # row and retain their exact type-expression source span instead.
            class_edge_valid = (
                not isinstance(fact, SourcePageFactClass)
                or (
                    class_identity.startswith("builtins.")
                    and len(tuple(
                        source for source, _stage in sources
                        if source.page == STATIC_SYMBOL_DEFINITION
                        and tuple(source.row) == (
                            "builtins",
                            class_identity.removeprefix("builtins."),
                            "",
                        )
                        and any(
                            upstream.page.name == "source_span"
                            and str(upstream.row[0]) == str(row[0])
                            for upstream, _upstream_stage
                            in book.edges_into(source)
                        )
                    )) == 1
                )
                or len(fact_class_rows) == 1
            )
            row_schema_edges_valid = all(
                required in class_rows
                for required in (
                    ("src.compiler.identity_concordance", "RowField"),
                    (
                        "src.compiler.identity_concordance",
                        "RowFieldKind",
                    ),
                )
            )
            row_kind_enum_edge_valid = any(
                source.page == SOURCE_ENUM_GLYPH_TABLE
                and tuple(source.row) == (
                    "src.compiler.identity_concordance", "RowFieldKind",
                )
                for source, _stage in sources
            )
            factory_edge_valid = (
                len(tuple(
                    source for source, _stage in sources
                    if source.page == STATIC_SYMBOL_DEFINITION
                    and tuple(source.row) == (
                        "src.compiler.identity_concordance",
                        "Registry.declare_page",
                        "src.compiler.identity_concordance.Registry",
                    )
                    and any(
                        upstream.page == CLASS_METHOD_DECLARATION
                        and tuple(upstream.row) == (
                            "src.compiler.identity_concordance",
                            "Registry", "declare_page",
                        )
                        for upstream, _upstream_stage in book.edges_into(source)
                    )
                )) == 1
            )
            valid_fields = (
                isinstance(getattr(fact, "row_fields", None), tuple)
                and all(
                    isinstance(field, tuple)
                    and len(field) == 2
                    and all(isinstance(item, str) and item for item in field)
                    for field in fact.row_fields
                )
            )
            if (
                not isinstance(fact, SourcePageFactClass)
                or len(row) != 2
                or str(row[1]) != str(fact.page_name)
                or not str(fact.class_identity)
                or not valid_fields
                or not class_edge_valid
                or not row_schema_edges_valid
                or not row_kind_enum_edge_valid
                or not factory_edge_valid
                or not span_modules
                or span_modules != {str(row[0])}
            ):
                found.append(Finding(
                    "source-page-declaration-disagreement",
                    str(row[0]) if row else "?",
                    None,
                    "source page cell disagrees with its authored module, "
                    "page name, fact class, or RowField source edges",
                ))
        return found

    @staticmethod
    def _tile_storage_findings(module: Any) -> list[Finding]:
        """A disk-tier runtime receipt must cite this book's exact policies.

        Representation movement must not post edges by thawing a cold private
        page under pressure. Its sourced runtime receipt instead retains refs
        to the predeclared policies; audit those refs without a full-page load.
        """
        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        report = None if book is None else book.__dict__.get("_tile_offload_report")
        live = None if book is None else book.__dict__.get("_live_tile_report")
        live_findings = []
        if live is not None:
            expected_live = {
                ("compressed_tile_disk",): "true",
                ("memory_budget_bytes",): str(live["budget_bytes"]),
                ("live_tile_rows",): str(live["tile_rows"]),
                ("live_bucket_rows",): str(live["bucket_rows"]),
                ("live_resident_tiles",): str(live["hot_limit"]),
            }
            if "decoded_budget_bytes" in live:
                expected_live[("live_decoded_budget_bytes",)] = str(live["decoded_budget_bytes"])
            matched_live = set()
            for ref in live.get("policy_sources", ()):
                if (isinstance(ref, Ref) and ref.page is book.registry.pages.get("compile_policy")
                        and ref.row in expected_live
                        and book.page(ref.page).cell(ref.row, ref.column) == expected_live[ref.row]):
                    matched_live.add(ref.row)
            if matched_live != set(expected_live) or live["budget_bytes"] <= 0:
                live_findings.append(Finding("live-tile-storage-policy", "<compiler-storage>", None,
                                            "live tile offload disagrees with exact cache/disk/budget policies"))
        if report is None:
            return live_findings
        expected = {
            ("compressed_tile_disk",): "true",
            ("memory_budget_bytes",): str(report.budget_bytes),
        }
        matched = set()
        for ref in report.policy_sources:
            if not isinstance(ref, Ref) or ref.page is not book.registry.pages.get("compile_policy"):
                continue
            if ref.row in expected and book.page(ref.page).cell(ref.row, ref.column) == expected[ref.row]:
                matched.add(ref.row)
        if matched != set(expected) or report.budget_bytes <= 0:
            return live_findings + [Finding("tile-storage-policy", "<compiler-storage>", None,
                            "disk tile receipt has missing/disagreeing exact budget or disk-policy cells")]
        return live_findings

    @staticmethod
    def _graph_work_publication_findings(module: Any) -> list[Finding]:
        """Accepted worker facts must name the same region, policy and copy."""
        from .concordance_declarations import (
            GraphWorkPublication, GRAPH_WORK_PUBLICATION, DEPLOYMENT_REGION,
            COMPILE_POLICY, SCOPE_REGISTRY,
        )
        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None or "graph_work_publication" not in book.pages:
            return []
        found = []
        with book.borrowed("graph_work_publication") as work, book.borrowed("deployment_region") as regions, \
                book.borrowed("compile_policy") as policies, book.borrowed("scope_registry") as scopes:
            for row in work.rows():
                fact = work.latest(row)
                region = regions.latest(row)
                detail = None
                if not isinstance(fact, GraphWorkPublication):
                    detail = f"invalid graph work fact {fact!r}"
                elif (fact.schedule_preference not in {"asap", "alap"}
                      or not isinstance(fact.snapshot_digest, str)
                      or len(fact.snapshot_digest) != 64
                      or any(char not in "0123456789abcdef" for char in fact.snapshot_digest)
                      or type(fact.nodes) is not int or fact.nodes < 1
                      or type(fact.edges) is not int or fact.edges < 0
                      or type(fact.admission_limit) is not int or fact.admission_limit < 1
                      or not fact.provider):
                    detail = f"invalid sealed snapshot/policy {fact!r}"
                elif region is None or region.schedule_preference != fact.schedule_preference:
                    detail = f"worker preference {fact.schedule_preference!r} disagrees with region {region!r}"
                elif policies.latest(("graph_admission_limit",)) != str(fact.admission_limit):
                    detail = "worker admission disagrees with the compile policy"
                elif fact.read_scope is not None and scopes.latest(fact.read_scope) is not True:
                    detail = f"worker names an unregistered read scope {fact.read_scope!r}"
                else:
                    ref = book.latest_ref(GRAPH_WORK_PUBLICATION, row)
                    sources = {source.key for source, _stage in book.edges_into(ref)}
                    required = [book.latest_ref(DEPLOYMENT_REGION, row),
                                book.latest_ref(COMPILE_POLICY, ("graph_admission_limit",))]
                    if fact.read_scope is not None:
                        required.append(book.latest_ref(SCOPE_REGISTRY, fact.read_scope))
                    if any(cell is None or cell.key not in sources for cell in required):
                        detail = "worker publication lacks its exact region/policy/copy provenance"
                if detail is not None:
                    found.append(Finding("graph-work-publication-disagreement", str(row[0]), None, detail))
        return found

    # Accounting keys whose value is an SSA value id of the SAME function.
    # Program: ``ys = make(n)`` (list returned by a callee).  Wrong record:
    # the callee output's accounting was copied onto the caller's result, so
    # ``heap_owner_value_id`` named the callee's owner cell, a value that
    # does not exist in the caller.  Rule: these ids are scope-local; one that
    # names a value absent from its own function is a leaked foreign record.
    # ``source_value_id`` / ``carried_from_value_id`` are source-graph ids,
    # not SSA ids, and are not read here.
    _SCOPE_LOCAL_ACCOUNTING_IDS = (
        "heap_owner_value_id",
        "ssa_optional_payload_id",
        "iterable_target_value_id",
        "native_type_tag_value_id",
        "tensor_shape_value_id",
        "tensor_rank_value_id",
        "tensor_element_count_value_id",
    )

    @staticmethod
    def _scope_local_accounting_findings(module: Any) -> list[Finding]:
        found = []
        for name, function in module.functions.items():
            carried = {id(value): value for value in function.args}
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.res is not None:
                        carried[id(instruction.res)] = instruction.res
                    for argument in instruction.args:
                        carried[id(argument)] = argument
            present = {int(value.id) for value in carried.values()}
            for value in carried.values():
                accounting = value.accounting or {}
                for key in CorrelationTable._SCOPE_LOCAL_ACCOUNTING_IDS:
                    named = accounting.get(key)
                    if named is None or isinstance(named, bool):
                        continue
                    if int(named) not in present:
                        found.append(Finding(
                            "accounting-foreign-value-id", str(name), int(value.id),
                            f"accounting.{key}={int(named)} names no value of this function",
                        ))
        return found

    @staticmethod
    def _repository_ssa_signature_findings(module: Any) -> list[Finding]:
        found = []
        for name, function in module.functions.items():
            signature = function.metadata.get("repository_ssa_signature")
            if signature is None:
                continue
            represented = {int(value.id) for value in function.args}
            represented.update(int(instruction.res.id) for block in function.blocks.values()
                               for instruction in block.instrs if instruction.res is not None)
            outputs = tuple(signature.get("output_ids", ()))
            if (signature.get("schema") != "turing.repository-ssa-call-signature.v1"
                    or tuple(signature.get("argument_ids", ())) != tuple(int(value.id) for value in function.args)
                    or signature.get("source_function_reference") != function.metadata.get("source_function_reference")
                    or set(outputs) != {value for _path, value in signature.get("output_paths", ())}
                    or function.blocks and not set(outputs).issubset(represented)):
                found.append(Finding("repository-ssa-signature-disagreement", str(name), None,
                                     "composition signature disagrees with its owning SSA function"))
        return found

    @staticmethod
    def _region_instruction_residency_findings(module: Any) -> list[Finding]:
        found = []
        books = (getattr(module, "metadata", {}) or {}).get("ssa_composition_function_books", {})
        for name, function in module.functions.items():
            record = function.metadata.get("region_instruction_residency")
            if record is None:
                continue
            book = books.get(name) or identity_book(module)
            counters = ("limit", "resident_peak", "resident_instructions_peak", "snapshots", "spills", "reloads")
            if (record.get("schema") != "turing.region-instruction-residency.v1"
                    or any(type(record.get(key)) is not int or record[key] < 0 for key in counters)
                    or record["limit"] < 1 or record["resident_peak"] > record["limit"]
                    or record["resident_peak"] > record["snapshots"]
                    or book.page("compile_policy").latest(("ssa_resident_regions",)) != str(record["limit"])):
                found.append(Finding("region-instruction-residency-disagreement", str(name), None,
                                     "instruction cache receipt disagrees with its admission policy"))
        return found

    @staticmethod
    def _native_interface_findings(module: Any) -> list[Finding]:
        """A bodyless declaration is an accepted ABI view, never an empty body."""
        found = []
        for name, function in module.functions.items():
            linked = function.metadata.get("linked_piece_interface")
            if linked:
                import hashlib
                outputs = tuple(function.metadata.get("declared_native_outputs", ()))
                piece = function.metadata.get("llvm_piece") or {}
                values = (*function.args, *outputs)
                arguments = tuple(int(value.id) for value in function.args)
                if (linked.get("schema") != "turing.linked-piece-interface.v1"
                        or function.blocks
                        or tuple(linked.get("argument_ids", ())) != arguments
                        or tuple(linked.get("output_ids", ())) != tuple(int(v.id) for v in outputs)
                        or tuple(linked.get("boundary_values", ())) != tuple(
                            (int(v.id), str(v.dtype), tuple(v.shape or ())) for v in values)
                        or set(linked.get("read_only_arguments", {})) != set(arguments)
                        or not all(isinstance(v, bool) for v in linked.get("read_only_arguments", {}).values())
                        or any(tuple(linked.get(key, ())) != tuple(piece.get(key, ()))
                               for key in ("buffer_order", "extent_order"))
                        or not set(piece.get("buffer_order", ())).issubset({int(v.id) for v in values})
                        or linked.get("native_symbol") != piece.get("symbol")
                        or linked.get("llvm_ir_sha256") != hashlib.sha256(
                            str(piece.get("llvm_ir", "")).encode()).hexdigest()):
                    found.append(Finding("linked-piece-interface-disagreement", str(name), None,
                                         "declaration, effect summary and retained native ABI disagree"))
            contract = function.metadata.get("native_interface")
            if not contract:
                continue
            outputs = tuple(function.metadata.get("declared_native_outputs", ()))
            piece = function.metadata.get("llvm_piece") or {}
            values = (*function.args, *outputs)
            expected = tuple((int(v.id), str(v.dtype), tuple(v.shape or ())) for v in values)
            ids = {int(v.id) for v in values}
            arguments = tuple(int(v.id) for v in function.args)
            paths = tuple(contract.get("output_paths", ()))
            if paths and (len(dict(paths)) != len(paths)
                    or any(not isinstance(path, tuple) or not path
                           or any(not isinstance(index, int) or index < 0 for index in path)
                           for path, _value in paths)
                    or {value for _path, value in paths} != {int(v.id) for v in outputs}):
                found.append(Finding("native-result-projection-disagreement", str(name), None,
                                     "authored result paths and accepted output identities disagree"))
            if (contract.get("schema") != "turing.native-unit-interface.v1"
                    or function.blocks
                    or tuple(contract.get("argument_ids", ())) != arguments
                    or tuple(contract.get("output_ids", ())) != tuple(int(v.id) for v in outputs)
                    or tuple(contract.get("boundary_values", ())) != expected
                    or set(contract.get("read_only_arguments", {})) != set(arguments)
                    or not all(isinstance(v, bool) for v in contract.get("read_only_arguments", {}).values())
                    or not set(piece.get("buffer_order", ())).issubset(ids)
                    or any(tuple(contract.get(key, ())) != tuple(piece.get(key, ()))
                           for key in ("buffer_order", "extent_order", "buffer_shapes", "buffer_dtypes", "linked_objects"))
                    or contract.get("native_symbol") != piece.get("symbol")
                    or not piece.get("linked_objects")
                    or not contract.get("ssa_artifact_sha256")):
                found.append(Finding("native-interface-disagreement", str(name), None,
                                     "declaration, effect summary and accepted native ABI disagree"))
        return found

    @staticmethod
    def _conditional_binding_preservation_findings(module: Any) -> list[Finding]:
        """A preserved conditional source must belong to another exact binding."""
        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        owners = {}
        scopes = {
            tuple(operation.attributes['preserved_name_binding_scope'])
            for function in module.functions.values() for block in function.blocks.values()
            for operation in block.instrs
            if (operation.attributes or {}).get('preserved_name_binding_value_ids')
            and (operation.attributes or {}).get('preserved_name_binding_scope') is not None
        }
        if book is not None and "name_binding" in book.pages:
            with book.borrowed("name_binding") as names:
                # The exact declared read scope may be copy-on-read: rows()
                # lists resident rows only, whereas scope_rows resolves its
                # existing scope_origin projection without inventing aliases.
                for scope in scopes:
                    for row in names.scope_rows(scope):
                        fact = names.latest(row)
                        value_id = getattr(fact, 'value_id', None)
                        if value_id is not None:
                            owners.setdefault((row[0], int(value_id)), set()).add(row[1])
        found = []
        for symbol, function in module.functions.items():
            for block in function.blocks.values():
                for operation in block.instrs:
                    attrs = operation.attributes or {}
                    preserved = attrs.get('preserved_name_binding_value_ids', ())
                    if not preserved:
                        continue
                    scope = attrs.get('preserved_name_binding_scope')
                    scope = tuple(scope) if scope is not None else None
                    merged = attrs.get('preserved_name_binding_merge_id')
                    merged_names = owners.get((scope, merged), set())
                    for source in preserved:
                        if (operation.op == 'Phi' and merged_names
                                and owners.get((scope, source), set()) - merged_names):
                            continue
                        found.append(Finding('conditional-binding-preservation-disagreement',
                            str(symbol), int(source),
                            f'preserved source {source} has no other binding than merge {merged} in {scope!r}'))
        return found

    @staticmethod
    def _name_binding_capture_findings(module: Any) -> list[Finding]:
        """A capture connects two exact binding versions of one value."""

        from .concordance_declarations import BindingFact, BINDING_VERSION_REMOVED

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None or "name_binding_capture" not in book.pages:
            return []
        found = []
        with book.borrowed("name_binding_capture") as captures, book.borrowed("name_binding") as names:
            for row in captures.rows():
                capture = captures.latest(row)
                if (isinstance(capture, Unresolved)
                        and capture.reason == BINDING_VERSION_REMOVED):
                    target = names.latest(row)
                    if (isinstance(target, Unresolved)
                            and target.reason == BINDING_VERSION_REMOVED):
                        continue
                if (not isinstance(capture, tuple) or len(capture) != 3
                        or not isinstance(capture[0], str)
                        or not isinstance(capture[1], int) or capture[1] < 0
                        or capture[2] is not None and not isinstance(capture[2], int)):
                    detail = f"capture {row!r} has invalid source/owner {capture!r}"
                else:
                    target = names.latest(row)
                    source = names.latest((row[0], capture[0], capture[1]))
                    # The full N4 book retains explicitly unresolved bindings.
                    # Those are not BindingFacts and do not carry value_id.
                    # Report their capture as unresolved, never crash the audit
                    # or accept two missing identities as an equality.
                    if (isinstance(target, BindingFact)
                            and isinstance(source, BindingFact)
                            and target.value_id == source.value_id):
                        continue
                    detail = f"capture {row!r} connects target {target!r} to source {source!r}"
                found.append(Finding("name-binding-capture-disagreement", str(row[0]), None, detail))
        return found

    def _unsourced_worklist(
        self, module: Any,
    ) -> tuple[dict[tuple[str, str, str], int], list[tuple[str, int]]]:
        """Cells with no edge, grouped, and minted ids with no mint edge.

        Group key ``(page, stage, unit)``: ``unit`` is ``"cell"`` for a
        resolved cell on a registered page that no edge or mint row names,
        ``"raw row"`` for a row on an unregistered page written through a raw
        primitive (tagged on the unsourced page while the latch is OPEN).
        """
        book = identity_book(module)
        registered = book.registry.pages
        private = set(book.registry.private_pages) | set(_PRIVATE_PAGE_NAMES)
        # (All three private pages first: reading one back from the spill file
        # posts a receipt, which must be on the book before it is measured.)
        edge_page = book.pages.get(EDGE_PAGE.name)
        mint_page = book.pages.get(MINT_PAGE.name)
        unsourced_page = book.pages.get(UNSOURCED_PAGE.name)
        sourced: set[Any] = set()
        if edge_page is not None:
            sourced.update(row[0] for row in edge_page.rows())
        minted_with_edge: set[int] = set()
        if mint_page is not None:
            for row in mint_page.rows():
                sourced.add(row[0])
                if row[1] is not None:
                    minted_with_edge.add(int(row[1]))
        tags: dict[tuple[str, Any], str] = {}
        if unsourced_page is not None:
            for row in unsourced_page.rows():
                tags[(row[0], row[1])] = row[2]
        groups: Counter = Counter()
        #: The first few exact rows of each group, so a finding names what it
        #: counts (read by ``_unsourced_findings``; never part of the worklist).
        samples: dict[tuple[str, str, str], list[str]] = {}

        def sample(group: tuple[str, str, str], row: Any, column: Any = None):
            rows = samples.setdefault(group, [])
            if len(rows) < 3:
                rows.append(
                    render_row(row) if column is None
                    else f"{render_row(row)}@{column}"
                )

        self._unsourced_samples = samples
        # (Each page is borrowed for its scan and let go: the detector walks
        # every cell of the book and must not hold the book to do it.)
        for name in tuple(dict.keys(book.pages)):
            if name not in registered or name in private:
                continue
            with book.borrowed(name) as page:
                for (row, column), fact in page.cells.items():
                    if fact is None or isinstance(fact, Unresolved):
                        continue
                    if (name, row, column) in sourced:
                        continue
                    group = (name, tags.get((name, row), "unknown"), "cell")
                    groups[group] += 1
                    sample(group, row, column)
        for (page_name, _row), stage_name in tags.items():
            if page_name in registered:
                continue
            group = (page_name, stage_name, "raw row")
            groups[group] += 1
            sample(group, _row)
        # An imported (accepted) SSA body keeps the book that accepted it: its
        # ids were minted by Novel posts in THAT book, so their mint edges are
        # read there, not in the composing unit's book (which owns only the
        # current unit).
        owner_mints: dict[int, set[int]] = {id(book): minted_with_edge}

        def minted_in_owner(function_name: str) -> set[int]:
            function = (getattr(module, "functions", None) or {}).get(
                function_name
            )
            owner = dict(
                (getattr(function, "metadata", None) or {}).get(
                    "accepted_ssa_component"
                ) or {}
            ).get("identity_book")
            if owner is None or owner is book:
                return minted_with_edge
            found = owner_mints.get(id(owner))
            if found is None:
                found = owner_mints[id(owner)] = set()
                owner_page = owner.pages.get(MINT_PAGE.name)
                if owner_page is not None:
                    found.update(
                        int(row[1]) for row in owner_page.rows()
                        if row[1] is not None
                    )
            return found

        identities = [
            (function, value_id)
            for function, value_id in self.rows
            if has_flag(value_id, MINTED)
            and value_id not in minted_in_owner(function)
        ]
        return dict(groups), identities

    def _unsourced_findings(self, module: Any) -> list[Finding]:
        groups, identities = self._unsourced_worklist(module)
        samples = getattr(self, "_unsourced_samples", {})
        found = [
            Finding(
                "unsourced-fact", page_name, None,
                f"stage {stage_name}: {count} {unit}(s) with neither an "
                "inbound edge nor a mint edge"
                + (
                    "; e.g. " + " | ".join(
                        samples[(page_name, stage_name, unit)]
                    ) if samples.get((page_name, stage_name, unit)) else ""
                ),
            )
            for (page_name, stage_name, unit), count in sorted(groups.items())
        ]
        found.extend(
            Finding(
                "unsourced-identity", function, value_id,
                "minted id has no mint edge (minted outside a Novel post)",
            )
            for function, value_id in identities
        )
        return found

    def _layout_table_findings(self, module: Any) -> list[Finding]:
        """The struct/union tables against the values and rows that name them.

        Four checks, all read from the tables' own book pages:
        ``layout-type`` claims (a value's ``ssa_layout_kind`` /
        ``ssa_layout_identity`` accounting) must name a live row; a live
        row's nested members must name live rows; every edge on
        ``layout_supersession`` is a re-declaration (two layouts for one
        identity) and is reported with the fields that changed; a row whose
        ``layout_state`` is still ``invalidated`` was laid out from a row
        that changed underneath it and has not been re-declared.
        """

        tables = tuple(
            table for table in (
                getattr(module, "struct_table", None),
                getattr(module, "union_table", None),
            )
            if table is not None
        )
        if not tables:
            return []
        found: list[Finding] = []
        live: dict[tuple[str, int], Any] = {}
        identities: set[tuple[str, str]] = set()
        for table in tables:
            for row_id, descriptor in dict(table._rows).items():
                live[(table.kind, int(row_id))] = descriptor
                identities.add((table.kind, str(descriptor.identity)))
        for (name, value_id), row in self.rows.items():
            for claim in row.claims:
                if claim.kind != "layout-type":
                    continue
                kind, _, identity = claim.key.partition(":")
                if (kind, identity) in identities:
                    continue
                found.append(Finding(
                    "layout-type-unknown", name, int(value_id),
                    f"accounting names {kind} type {identity!r}; the module's "
                    f"{kind} table holds no live row of that identity "
                    f"(source {claim.source})",
                ))
        for (kind, row_id), descriptor in live.items():
            members = descriptor.fields if kind == "struct" else descriptor.members
            owner = next(t.owner for t in tables if t.kind == kind)
            for member in members:
                for nested_kind, nested_id in (
                    ("struct", member.struct_id), ("union", member.union_id),
                ):
                    if nested_id is None or (nested_kind, int(nested_id)) in live:
                        continue
                    found.append(Finding(
                        "layout-member-unknown", str(owner[0]), int(row_id),
                        f"{kind} {descriptor.identity!r} member {member.name!r} "
                        f"is laid out from {nested_kind} row {int(nested_id)}, "
                        f"which no live row holds",
                    ))
        for table in tables:
            for edge_row, edge_fact in table.supersessions():
                if not (isinstance(edge_fact, tuple) and len(edge_fact) == 2):
                    continue
                incumbent, replacement = edge_fact
                differing = tuple(sorted(
                    field_name for field_name in vars(replacement)
                    if getattr(incumbent, field_name, None)
                    != getattr(replacement, field_name)
                ))
                _owner, kind, target_id, source_id, stage = edge_row
                found.append(Finding(
                    "layout-redeclaration", str(table.owner[0]), int(target_id),
                    f"{kind} {replacement.identity!r} declared again at stage "
                    f"{stage!r} (row {int(source_id)} -> {int(target_id)}); "
                    f"differs in: {', '.join(differing) or 'nothing'}; "
                    + "; ".join(
                        f"{field_name}: incumbent="
                        f"{getattr(incumbent, field_name, None)!r} vs "
                        f"new={getattr(replacement, field_name)!r}"
                        for field_name in differing
                    ),
                ))
            state_page = table.book.page("layout_state")
            for state_row in state_page.scope_rows(table.owner):
                if len(state_row) != 3 or state_row[1] != table.kind:
                    continue
                fact = state_page.latest(state_row)
                if not (isinstance(fact, tuple) and fact and fact[0] == "invalidated"):
                    continue
                _owner, kind, row_id = state_row
                descriptor = live.get((kind, int(row_id)))
                found.append(Finding(
                    "layout-derivation-invalidated", str(table.owner[0]),
                    int(row_id),
                    f"{kind} row {int(row_id)} "
                    f"({getattr(descriptor, 'identity', '?')!r}) was laid out "
                    f"from {fact[1][0]} row {fact[1][1]}, which was re-declared "
                    f"at stage {fact[2]!r}; this row has not been re-declared",
                ))
        return found

    @staticmethod
    def _operand_position_orphan_findings(module: Any) -> list[Finding]:
        """Read bindings orphaned by an operand rewrite off the book.

        Page ``lexical_read_binding`` keys the binding one operand read by
        its position ``(consumer, role, ordinal)``.  ``_set_operands`` is the
        one writer of operand lists and moves those rows with the operand;
        a rewrite that bypasses it leaves the row at a position the consumer
        no longer has.  The planner records each such row on page
        ``operand_position_orphan`` when it reads the consumer's operands.
        """
        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = (getattr(book, "pages", {}) or {}).get("operand_position_orphan")
        if page is None:
            return []
        return [
            Finding(
                "operand-position-orphan",
                str(row[0]),
                int(row[1]) if isinstance(row[1], int) else None,
                f"binding {page.latest(row)!r} read at operand position "
                f"{row[2:]!r} of consumer {row[1]!r}, which that consumer "
                "no longer has -- an operand rewrite bypassed _set_operands",
            )
            for row in page.rows()
            if isinstance(row, tuple) and len(row) == 4
        ]

    @staticmethod
    def _table_member_findings(module: Any) -> list[Finding]:
        """Member rows that disagree with the descriptors they index.

        ``record_member`` / ``sequence_member`` hold, per table owner and
        value, every claim the live descriptors on ``record_descriptor`` /
        ``sequence_descriptor`` make on that value.  The two are one fact
        seen from each side; any difference is a descriptor write that
        bypassed its table.
        """

        from ..transmogrifier.ssa import (
            record_member_claims, sequence_member_roles,
            struct_member_claims, union_member_claims,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        pages = getattr(book, "pages", {}) or {}
        found: list[Finding] = []
        for descriptor_page, member_page, claims_of in (
            ("record_descriptor", "record_member", record_member_claims),
            ("sequence_descriptor", "sequence_member", sequence_member_roles),
            ("struct_descriptor", "struct_member", struct_member_claims),
            ("union_descriptor", "union_member", union_member_claims),
        ):
            descriptors = pages.get(descriptor_page)
            members = pages.get(member_page)
            if descriptors is None or members is None:
                continue
            for owner in tuple(descriptors.scopes):
                expected: dict[int, set] = {}
                for row in descriptors.scope_rows(owner):
                    for member, claims in claims_of(
                        descriptors.latest(row)
                    ).items():
                        expected.setdefault(int(member), set()).update(claims)
                recorded = {
                    int(row[1]): set(members.latest(row) or ())
                    for row in members.scope_rows(owner)
                    if members.latest(row)
                }
                for member in sorted(set(expected) | set(recorded)):
                    if expected.get(member, set()) == recorded.get(member, set()):
                        continue
                    found.append(Finding(
                        "table-member-disagreement",
                        str(owner[0]) if isinstance(owner, tuple) else str(owner),
                        int(member),
                        f"{member_page} records "
                        f"{sorted(recorded.get(member, ()), key=repr)!r}; "
                        f"{descriptor_page} implies "
                        f"{sorted(expected.get(member, ()), key=repr)!r}",
                    ))
        return found

    @staticmethod
    def _post_ssa_numeric_identity_findings(module: Any) -> list[Finding]:
        """Report disagreement in the numeric facts consumed after SSA.

        These pages are deliberately distinct because they govern distinct
        transformations: exact region feeds, the physical limb channel, and
        the descriptor forwarded by a structural one-input Phi.  The audit
        reads all three so a later pass cannot silently publish a second fact
        for the same semantic value.
        """

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        if book is None:
            return []
        specifications = (
            (
                "exact_region_feed_dtype",
                "exact-region-feed-dtype-disagreement",
                "exact region feed changed dtype",
            ),
            (
                "precision_channel_shape_concordance",
                "precision-channel-shape-disagreement",
                "Precision value changed logical shape, channel shape, or width",
            ),
            (
                "single_input_phi_descriptor_concordance",
                "single-input-phi-descriptor-disagreement",
                "single-input Phi changed source, dtype, or shape",
            ),
        )
        found: list[Finding] = []
        pages = getattr(book, "pages", {}) or {}
        for page_name, kind, detail in specifications:
            page = pages.get(page_name)
            if page is None:
                continue
            for row in page.rows():
                distinct = tuple(dict.fromkeys(
                    repr(fact) for _column, fact in page.history(row)
                ))
                if len(distinct) <= 1:
                    continue
                if (
                    page_name == "exact_region_feed_dtype"
                    and isinstance(row, tuple)
                    and len(row) == 3
                ):
                    function, value_id = str(row[1]), int(row[2])
                elif isinstance(row, tuple) and len(row) >= 2:
                    function, value_id = str(row[0]), int(row[1])
                else:
                    function, value_id = str(row), None
                found.append(Finding(
                    kind,
                    function,
                    value_id,
                    f"{detail}: {distinct!r}",
                ))
        return found

    @staticmethod
    def _source_callsite_activation_findings(module: Any) -> list[Finding]:
        """Report a source callsite resolved to multiple function identities."""

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_callsite_activation_concordance"
            )
        )
        if page is None:
            return []
        found: list[Finding] = []
        for row in page.rows():
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in page.history(row)
            ))
            if len(distinct) <= 1:
                continue
            function_name, value_id = (
                row if isinstance(row, tuple) and len(row) == 2
                else (str(row), None)
            )
            found.append(Finding(
                "source-callsite-activation-disagreement",
                str(function_name),
                None if value_id is None else int(value_id),
                "source callsite changed function-table identity across "
                f"planning stages: {distinct!r}",
            ))
        return found

    @staticmethod
    def _planning_alias_transition_findings(module: Any) -> list[Finding]:
        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "planning_alias_transition_concordance"
            )
        )
        if page is None:
            return []
        found: list[Finding] = []
        for row in page.rows():
            history = tuple(fact for _column, fact in page.history(row))
            for prior, current in zip(history, history[1:]):
                if int(tuple(prior)[1]) == int(tuple(current)[0]):
                    continue
                function_name, value_id = row
                found.append(Finding(
                    "planning-alias-transition-disagreement",
                    str(function_name), int(value_id),
                    "planning refinement history is discontinuous: "
                    f"prior={prior!r}, current={current!r}",
                ))
        return found

    @staticmethod
    def _operator_result_type_findings(module: Any) -> list[Finding]:
        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "operator_result_type_concordance"
            )
        )
        if page is None:
            return []
        found: list[Finding] = []
        for row in page.rows():
            facts = tuple(dict.fromkeys(
                fact for _column, fact in page.history(row)
            ))
            if len(facts) <= 1:
                continue
            if len(row) == 4:
                function_name, _closure_id, region_name, value_id = row
            else:
                _closure_id, region_name, value_id = row
                function_name = region_name
            found.append(Finding(
                "operator-result-type-disagreement",
                str(function_name), int(value_id),
                f"planned value acquired multiple operator result types: "
                f"region={region_name!r}, facts={facts!r}",
            ))
        return found

    @staticmethod
    def _tensor_reduction_domain_findings(module: Any) -> list[Finding]:
        """Check all-axis reduction views against their emitted kernel call."""

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "tensor_reduction_domain_concordance"
            )
        )
        if page is None:
            return []
        found: list[Finding] = []
        for row in page.rows():
            function_name, result_id = row
            history = page.history(row)
            facts = tuple(dict.fromkeys(
                fact for _column, fact in history
            ))
            if len(facts) != 1:
                found.append(Finding(
                    "tensor-reduction-domain-disagreement",
                    str(function_name), int(result_id),
                    f"all-axis reduction acquired multiple domains: {facts!r}",
                ))
                continue
            function = module.functions.get(str(function_name))
            receipts = () if function is None else tuple(
                instruction.attributes.get(
                    "tensor_reduction_domain_concordance"
                )
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
                and int(instruction.res.id) == int(result_id)
                and instruction.attributes.get(
                    "tensor_reduction_domain_concordance"
                ) is not None
            )
            if receipts != facts:
                found.append(Finding(
                    "tensor-reduction-domain-unpublished",
                    str(function_name), int(result_id),
                    f"concordance records {facts!r}, emitted call records "
                    f"{receipts!r}",
                ))
        return found

    @staticmethod
    def _loop_scope_findings(
        module: Any, name: str, function: Any,
    ) -> list[Finding]:
        """Rule on a loop against the scope it declared.

        These are the checks that no amount of shape, dominance or type
        agreement can supply, because every name involved has the same
        shape, the same dtype, and a definition that dominates every use.
        The only thing separating them is which generation they speak
        for, and that is knowable only from the declaration.
        """

        book = identity_book(module)
        declarations = loop_scope_declarations(book, name)
        if not declarations:
            return []

        definitions: dict[int, str] = {}
        resident_objects: dict[int, list[Any]] = defaultdict(list)
        for argument in function.args:
            resident_objects[int(argument.id)].append(argument)
        for block_name, block in function.blocks.items():
            for instruction in block.instrs:
                if instruction.res is not None:
                    definitions[int(instruction.res.id)] = str(block_name)
                    resident_objects[int(instruction.res.id)].append(
                        instruction.res
                    )

        found: list[Finding] = []
        for declaration in declarations:
            header, latch, exit_block = declaration["boundary"]
            inside = _blocks_between(function, header, latch)
            if not inside:
                continue
            simultaneous: dict[int, list[dict]] = defaultdict(list)
            for rebind in declaration["rebinds"]:
                simultaneous[int(rebind["outer"])].append(rebind)
            for outer, rebinds in simultaneous.items():
                if len(rebinds) < 2:
                    continue
                carried_ids = {
                    int(rebind["carried"]) for rebind in rebinds
                }
                if len(carried_ids) == len(rebinds):
                    continue
                found.append(Finding(
                    "loop-scope-simultaneous-binding-collapse",
                    name, outer,
                    f"{len(rebinds)} authored loop bindings share outer "
                    f"resident {outer} but only {len(carried_ids)} carried "
                    "identities; bindings with simultaneous future updates "
                    "must retain distinct header residents",
                ))
            for rebind in declaration["rebinds"]:
                outer = rebind["outer"]
                carried = rebind["carried"]
                inner = rebind["inner"]

                # Rule 1 -- inside the scope the outer generation is not
                # in scope.  A use of it reads the value as it was before
                # the first iteration, on every iteration.
                for block_name in sorted(inside):
                    block = function.blocks.get(block_name)
                    if block is None:
                        continue
                    for index, instruction in enumerate(block.instrs):
                        if str(instruction.op).lower() == "phi":
                            continue
                        for position, argument in enumerate(instruction.args):
                            if getattr(argument, "id", None) is None:
                                continue
                            if int(argument.id) != outer:
                                continue
                            found.append(Finding(
                                "loop-scope-outer-read", name, outer,
                                f"{block_name}#{index} {instruction.op} "
                                f"operand {position} names the outer "
                                f"generation of a value the loop rebinds; "
                                f"in scope it is {carried} (carried) or "
                                f"{inner} (inner)",
                            ))

                # Rule 2 -- the backedge must carry the declared inner
                # name.  An outlined body whose result returns through an
                # aggregate arrives under a name the parent minted after
                # the crossing; the value is right, the identity is lost.
                phi = _carried_phi(function, header, carried)
                if phi is not None and len(phi.args) == 2:
                    incoming = phi.attributes.get("incoming_blocks") or ()
                    for origin, argument in zip(incoming, phi.args):
                        if str(origin) != str(latch):
                            continue
                        if getattr(argument, "id", None) is None:
                            continue
                        if int(argument.id) == inner:
                            if any(
                                argument is resident
                                for resident in resident_objects.get(inner, ())
                            ):
                                continue
                            found.append(Finding(
                                "loop-scope-inner-private-resident",
                                name, inner,
                                f"{header} Phi {carried} takes an SSA object "
                                f"named {inner} from {latch}, but that object "
                                "is neither the function formal nor a defined "
                                "result carrying that id; loop lowering "
                                "refigured a private resident for an existing "
                                "source value",
                            ))
                            continue
                        found.append(Finding(
                            "loop-scope-latch-renamed", name, inner,
                            f"{header} Phi {carried} takes "
                            f"{int(argument.id)} from {latch}, but the "
                            f"loop declared its inner generation as "
                            f"{inner}; a transformation renamed the value "
                            "crossing the boundary without re-declaring "
                            "it",
                        ))

                # Rule 3 -- the inner generation must be defined inside
                # the scope.  A seed in the preheader is storage
                # initialization and is correct; a seed that is its ONLY
                # definition means the body never wrote the slot.
                where = definitions.get(inner)
                if where is not None and where not in inside:
                    found.append(Finding(
                        "loop-scope-inner-outside", name, inner,
                        f"the inner generation is defined in {where}, "
                        f"outside the scope ({header}..{latch}); nothing "
                        "in the body redefines it",
                    ))
        return found

    @staticmethod
    def _stale_carried_read_findings(
        name: str, function: Any,
    ) -> list[Finding]:
        """A loop body reading the value its carried Phi superseded.

        The Phi names two generations of one value: what it held on entry and
        what the latch produced.  Inside the loop only the Phi speaks for it.
        An instruction that still names the entry generation is reading the
        value as it was before the first iteration, every iteration -- the
        accumulation silently does not accumulate.
        """

        successors: dict[str, set[str]] = {}
        for block_name, block in function.blocks.items():
            targets: set[str] = set()
            for instruction in block.instrs:
                for key in ("target", "true_target", "false_target"):
                    declared = instruction.attributes.get(key)
                    if declared is not None:
                        targets.add(str(declared))
            successors[str(block_name)] = targets

        def reaches(source: str, goal: str) -> set[str]:
            """Blocks on some path from `source` to `goal`, inclusive."""

            forward: set[str] = set()
            frontier = [source]
            while frontier:
                current = frontier.pop()
                if current in forward:
                    continue
                forward.add(current)
                frontier.extend(successors.get(current, ()))
            backward: set[str] = set()
            frontier = [goal]
            while frontier:
                current = frontier.pop()
                if current in backward:
                    continue
                backward.add(current)
                for candidate, onward in successors.items():
                    if current in onward:
                        frontier.append(candidate)
            return forward & backward

        found: list[Finding] = []
        for header_name, header in function.blocks.items():
            for phi in header.instrs:
                if str(phi.op).lower() != "phi":
                    continue
                if phi.attributes.get("binding") != "loop_carried":
                    continue
                incoming = phi.attributes.get("incoming_blocks") or ()
                if len(incoming) != len(phi.args):
                    continue
                latches = [
                    str(origin) for origin in incoming
                    if str(header_name) in reaches(str(header_name), str(origin))
                ]
                if not latches:
                    continue
                body = set()
                for latch in latches:
                    body |= reaches(str(header_name), latch)
                stale = {
                    int(argument.id)
                    for origin, argument in zip(incoming, phi.args)
                    if str(origin) not in latches
                    and getattr(argument, "id", None) is not None
                }
                if not stale:
                    continue
                for block_name in sorted(body):
                    block = function.blocks.get(block_name)
                    if block is None:
                        continue
                    for index, instruction in enumerate(block.instrs):
                        if str(instruction.op).lower() == "phi":
                            continue
                        for position, argument in enumerate(instruction.args):
                            argument_id = getattr(argument, "id", None)
                            if argument_id is None:
                                continue
                            if int(argument_id) not in stale:
                                continue
                            found.append(Finding(
                                "stale-carried-read", name, int(argument_id),
                                f"{block_name}#{index} {instruction.op} operand "
                                f"{position} names the pre-loop value carried by "
                                f"{header_name} Phi {int(phi.res.id)}; inside the "
                                "loop only the Phi speaks for it",
                            ))
        return found

    @staticmethod
    def _undefined_operand_findings(
        name: str, function: Any,
    ) -> list[Finding]:
        """An operand no instruction defines and no formal supplies.

        Reading such a value yields whatever its storage happened to hold, so
        the program computes an answer from uninitialized memory instead of
        failing.  The latch incoming of a loop-carried Phi is where this hides:
        the body computes the update and stores it somewhere other than the
        slot the Phi names, and every later stage -- emission, the LLVM
        verifier, execution -- accepts the result.
        """

        found: list[Finding] = []
        formals = {int(value.id) for value in getattr(function, "args", ())}
        defined: set[int] = set()
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is not None:
                    defined.add(int(instruction.res.id))
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                for position, argument in enumerate(instruction.args):
                    argument_id = getattr(argument, "id", None)
                    if argument_id is None:
                        continue
                    argument_id = int(argument_id)
                    if argument_id in formals or argument_id in defined:
                        continue
                    found.append(Finding(
                        "operand-never-written", name, argument_id,
                        f"{block_name}#{index} {instruction.op} operand "
                        f"{position} is neither a formal nor defined by any "
                        "instruction",
                    ))
        return found

    @staticmethod
    def _callable_identity_findings(module: Any) -> list[Finding]:
        """Report a first-class callable whose exact address changed."""

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "callable_identity_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            distinct = tuple(dict.fromkeys(
                int(fact) for _column, fact in history
            ))
            if len(distinct) <= 1:
                continue
            function, value_id = (
                row if isinstance(row, tuple) and len(row) == 2
                else (str(row), None)
            )
            found.append(Finding(
                "callable-identity-disagreement",
                str(function),
                None if value_id is None else int(value_id),
                "first-class callable changed function-table address across "
                f"source stages: {distinct!r}",
            ))
        return found

    @staticmethod
    def _source_field_identity_findings(module: Any) -> list[Finding]:
        """Report a field whose source-class identity changed between reads."""

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_field_identity_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            distinct = tuple(dict.fromkeys(repr(fact) for _column, fact in history))
            if len(distinct) <= 1:
                continue
            owner, field = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, "?")
            )
            found.append(Finding(
                "source-field-identity-disagreement",
                str(owner),
                None,
                f"field {field!r} changed identity across source stages: "
                f"{distinct!r}",
            ))
        return found

    @staticmethod
    def _source_parameter_identity_findings(module: Any) -> list[Finding]:
        """Report a static parameter whose discovered identity changed."""

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_parameter_identity_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in history
            ))
            if len(distinct) <= 1:
                continue
            scope, parameter = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, "?")
            )
            found.append(Finding(
                "source-parameter-identity-disagreement",
                str(scope),
                None,
                f"parameter {parameter!r} changed identity across source "
                f"stages: {distinct!r}",
            ))
        return found

    @staticmethod
    def _source_numeric_component_findings(module: Any) -> list[Finding]:
        """Report a composite coefficient projection that changed identity."""

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_numeric_component_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in history
            ))
            if len(distinct) <= 1:
                continue
            scope, value_id = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, None)
            )
            found.append(Finding(
                "source-numeric-component-disagreement",
                str(scope),
                None if value_id is None else int(value_id),
                "numeric coefficient projection changed receiver, path, or "
                f"descriptor across source stages: {distinct!r}",
            ))
        return found

    @staticmethod
    def _source_numeric_intrinsic_findings(module: Any) -> list[Finding]:
        """Report a numeric intrinsic whose structural results changed."""

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_numeric_intrinsic_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in page.history(row)
            ))
            if len(distinct) <= 1:
                continue
            scope, value_id = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, None)
            )
            found.append(Finding(
                "source-numeric-intrinsic-disagreement",
                str(scope),
                None if value_id is None else int(value_id),
                "numeric intrinsic changed receiver or ordered component "
                f"results across source stages: {distinct!r}",
            ))
        return found

    @staticmethod
    def _source_numeric_type_dependency_findings(module: Any) -> list[Finding]:
        """Report an authored wrapper whose coefficient closure changed."""

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_numeric_type_dependency_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in page.history(row)
            ))
            if len(distinct) <= 1:
                continue
            type_name, limbs = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, None)
            )
            found.append(Finding(
                "source-numeric-type-dependency-disagreement",
                str(type_name),
                None,
                f"numeric type at width {limbs!r} acquired multiple "
                f"coefficient dependency closures: {distinct!r}",
            ))
        return found

    @staticmethod
    def _source_numeric_operator_specialization_findings(
        module: Any,
    ) -> list[Finding]:
        """Report a same-type wrapper dunder that changed its exact target."""

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_numeric_operator_specialization_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in page.history(row)
            ))
            if len(distinct) <= 1:
                continue
            scope, value_id = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, None)
            )
            found.append(Finding(
                "source-numeric-operator-specialization-disagreement",
                str(scope),
                None if value_id is None else int(value_id),
                "same-type numeric operator changed receiver, argument, "
                f"descriptor, or authored target: {distinct!r}",
            ))
        return found

    @staticmethod
    def _return_site_edge_findings(module: Any) -> list[Finding]:
        """An emitted return edge must carry its reducer site's exact slots.

        A value identity cannot identify the authored exit: three ``return
        m`` sites have equal slots and distinct control/field-state owners.
        The site is the ``return_site_cell`` the control builder stamped on
        the edge; its slots are the ``return_site_slot`` rows posted for that
        cell under the function's ``record_return_state_scope``, read from
        the module's own book (so a pickled module is checked against the
        rows it was built from, not a span-keyed copy).
        """
        from .concordance_declarations import RETURN_SITE_SLOT
        from ..common.tensors.topological_reducer import _return_slot_receipt

        module_book = identity_book(module)
        function_books = dict(
            (getattr(module, "metadata", {}) or {}).get(
                "ssa_composition_function_books", {}
            )
            or {}
        )
        found = []
        for name, function in module.functions.items():
            scope = function.metadata.get("record_return_state_scope")
            if not scope:
                continue
            # Imported SSA retains its accepting concordance owner.  The
            # composing unit's book owns only the functions lowered in that
            # unit; asking it about an imported return-site cell necessarily
            # produces a false missing receipt.  Prefer the owner carried by
            # the function, then the composition table, and use the module
            # book only for locally lowered functions.
            book = (
                dict(function.metadata.get("accepted_ssa_component") or {}).get(
                    "identity_book"
                )
                or function_books.get(str(name))
                or module_book
            )
            page = book.pages.get(RETURN_SITE_SLOT.name)
            by_site: dict = {}
            if page is not None:
                for row in page.scope_rows(tuple(scope)):
                    by_site.setdefault(row[1], []).append((row, page.latest(row)))
            receipts = {
                site: _return_slot_receipt(book, tuple(rows))
                for site, rows in by_site.items()
            }
            for block in function.blocks.values():
                if not block.instrs:
                    continue
                attrs = block.instrs[-1].attributes or {}
                slots = attrs.get("return_source_value_ids")
                if slots is None:
                    continue
                site = attrs.get("return_site_cell")
                if site not in receipts or receipts[site] != tuple(slots):
                    found.append(Finding(
                        "return-site-edge-disagreement", str(name), None,
                        f"{block.name}: return site {site!r} carries {tuple(slots)!r}; "
                        f"the reducer records {receipts.get(site)!r}",
                    ))
        return found

    @staticmethod
    def _live_structure_findings(module: Any) -> list[Finding]:
        """Audit current views separately from immutable historical evidence."""
        books = []
        main = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if main is not None:
            books.append(main)
        for function in module.functions.values():
            component = (function.metadata or {}).get("accepted_ssa_component")
            owner = None if component is None else component.get("identity_book")
            if owner is not None and all(owner is not book for book in books):
                books.append(owner)
        found = []
        for book in books:
            imported_page = book.pages.get(IMPORTED_CELL.name)
            if imported_page is not None:
                for row in imported_page.rows():
                    for column, _fact in imported_page.history(row):
                        try:
                            book._source_stamp(Ref(IMPORTED_CELL, row, column))
                        except (ValueError, KeyError) as error:
                            found.append(Finding(
                                "invalid-imported-proof", str(row), None, str(error),
                            ))
            for page_name, row in book.stale_derivations():
                found.append(Finding(
                    "stale-live-derivation", str(row), None,
                    f"{page_name} still names superseded dependency revisions",
                ))
        return found

    @staticmethod
    def _ssa_publication_findings(module: Any) -> list[Finding]:
        """Compare every resident SSA occurrence with its published value fact.

        The full bootstrap retained output facts after rewriting its Ret; the
        managed tire retained a scalar SSA fact for a returned span. Checking
        only whether an identity has a mint, or whether a derived view reads
        current revisions, misses both: an unrecorded object write changes no
        revision. This is a read-only comparison, never a late publication of
        whatever the object currently happens to contain.
        """
        from .concordance_declarations import SSA_VALUE, SSAValueFact
        from .ssa_record_return_state import function_scope_of

        main = (getattr(module, "metadata", {}) or {}).get("identity_book")
        found = []
        for name, function in module.functions.items():
            component = (function.metadata or {}).get("accepted_ssa_component")
            book = main if component is None else component.get("identity_book")
            if book is None:
                continue
            scope = function_scope_of(function)
            occurrences = {}

            accounted = {}

            def observe(value, position):
                key = (int(value.id), value.dtype, tuple(value.shape or ()))
                occurrences.setdefault(key, []).append(position)
                # Rule: the resident's accounting view of its physical fields,
                # read-only, compared below against the posted fact.
                account = dict(getattr(value, "accounting", None) or {})
                view = tuple(account.get(k) for k in (
                    "physical_dtype", "program_abi_storage", "program_abi_rank",
                    "sequence_arena", "device"))
                if any(not _physical_unset(k, item) for k, item in zip(
                        _PHYSICAL_ACCOUNTING_KEYS, view)):
                    accounted.setdefault((int(value.id), view), []).append(position)

            for index, value in enumerate(function.args):
                observe(value, ("formal", index))
            for label, block in function.blocks.items():
                for index, instruction in enumerate(block.instrs):
                    if instruction.res is not None:
                        observe(instruction.res, (str(label), index, "result"))
                    for ordinal, value in enumerate(instruction.args):
                        observe(value, (str(label), index, "operand", ordinal))
            for (value_id, dtype, shape), positions in occurrences.items():
                cell = book.latest_ref(SSA_VALUE, (scope, value_id))
                # Missing identity is already checked by the unsourced-value
                # audit. It is not permission to synthesize a publication.
                if cell is None:
                    continue
                fact = book.page(SSA_VALUE).cell(cell.row, cell.column)
                if not isinstance(fact, SSAValueFact):
                    found.append(Finding(
                        "ssa-publication-unresolved", str(name), value_id,
                        f"resident occurrence uses {cell!r} with fact {fact!r}",
                    ))
                elif (dtype, shape) != (fact.dtype, tuple(fact.shape or ())):
                    found.append(Finding(
                        "ssa-publication-type-disagreement", str(name), value_id,
                        f"{cell!r} records {(fact.dtype, tuple(fact.shape or ()))!r}; "
                        f"resident occurrence carries {(dtype, shape)!r} at "
                        f"{tuple(positions)!r}",
                    ))
            # Rule: where BOTH the accounting key and the fact field are set
            # they must agree (type-disagreement); where the accounting key is
            # set and the fact field is not, report ssa-physical-unposted --
            # the stage-3 migration worklist; reported, never refused.
            fields = (
                ("physical_dtype", "physical_dtype"),
                ("program_abi_storage", "storage"),
                ("program_abi_rank", "rank"),
                ("sequence_arena", "arena"),
                ("device", "device"),
            )
            for (value_id, view), positions in accounted.items():
                cell = book.latest_ref(SSA_VALUE, (scope, value_id))
                if cell is None:
                    continue
                fact = book.page(SSA_VALUE).cell(cell.row, cell.column)
                if not isinstance(fact, SSAValueFact):
                    continue
                for (key, field), resident in zip(fields, view):
                    posted = getattr(fact, field, None)
                    if _physical_unset(key, resident):
                        continue
                    if _physical_unset(key, posted):
                        found.append(Finding(
                            "ssa-physical-unposted", str(name), value_id,
                            f"{cell!r} has no {field}; resident accounting "
                            f"{key}={resident!r} at {tuple(positions)!r}",
                        ))
                    elif posted != resident:
                        found.append(Finding(
                            "ssa-publication-type-disagreement", str(name), value_id,
                            f"{cell!r} records {field}={posted!r}; resident "
                            f"accounting {key}={resident!r} at {tuple(positions)!r}",
                        ))
        return found

    @staticmethod
    def _source_aggregate_layout_findings(module: Any) -> list[Finding]:
        """Validate immutable reducer-owned aggregate layout facts."""

        from .concordance_declarations import (
            SOURCE_AGGREGATE_LAYOUT,
            SourceAggregateLayout,
            SourceAggregateMember,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                SOURCE_AGGREGATE_LAYOUT.name
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            facts = tuple(fact for _column, fact in history)
            distinct = tuple(dict.fromkeys(map(repr, facts)))
            fact = facts[-1]
            problem = None
            if len(distinct) != 1:
                problem = f"immutable layout changed: {distinct!r}"
            elif not isinstance(fact, SourceAggregateLayout):
                problem = f"invalid aggregate layout fact {fact!r}"
            elif fact.aggregate_kind not in {"tuple", "list", "set", "dict"}:
                problem = f"invalid aggregate kind {fact.aggregate_kind!r}"
            elif any(
                not isinstance(member, SourceAggregateMember)
                or not isinstance(member.value_id, int)
                or not isinstance(member.value_cell, Ref)
                or not member.source_cells
                or any(not isinstance(cell, Ref) for cell in member.source_cells)
                for member in fact.members
            ):
                problem = "ordered member identities/cells are incomplete"
            elif bool(fact.conditional_cell) != bool(fact.arm_layout_cells):
                problem = "conditional/control and arm layout cells disagree"
            elif fact.arm_layout_cells and (
                len(fact.arm_layout_cells) != 2
                or any(not isinstance(cell, Ref) for cell in fact.arm_layout_cells)
            ):
                problem = "conditional aggregate does not name two arm layouts"
            if problem is not None:
                scope, value_id = (
                    row if isinstance(row, tuple) and len(row) == 2
                    else (row, None)
                )
                found.append(Finding(
                    "source-aggregate-layout-disagreement", str(scope),
                    None if value_id is None else int(value_id), problem,
                ))
        return found

    @staticmethod
    def _unborn_display_findings(module: Any) -> list[Finding]:
        """Report a display whose resident length is read but never written.

        Program: ``t = (n + 1, n + 2, n * 3); len(t)`` (c260).  The
        SOURCE_AGGREGATE_LAYOUT row said three members; the module read the
        tuple's length cell, and no instruction anywhere cleared or wrote
        that sequence, so ``len(t)`` read the frame's zeroed cell and
        returned 0 with a clean audit.  The two records are the layout's
        member count and the module's writers of that sequence identity.
        """

        from .concordance_declarations import (
            SOURCE_AGGREGATE_LAYOUT,
            SourceAggregateLayout,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                SOURCE_AGGREGATE_LAYOUT.name
            )
        )
        if page is None:
            return []
        layouts: dict[int, SourceAggregateLayout] = {}
        for row in page.rows():
            if not (isinstance(row, tuple) and len(row) == 2):
                continue
            history = page.history(row)
            fact = history[-1][1] if history else None
            if isinstance(fact, SourceAggregateLayout) and fact.members:
                layouts[int(row[1])] = fact
        if not layouts:
            return []
        written: set[int] = set()
        reads: list[tuple[str, int]] = []
        for name, function in module.functions.items():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    attributes = getattr(instruction, "attributes", {}) or {}
                    sequence_id = attributes.get("sequence_id")
                    if sequence_id is not None:
                        written.add(int(sequence_id))
                    if (instruction.op == "Load"
                            and attributes.get("binding") == "ssa_sequence_length"
                            and attributes.get("sequence_value_id") is not None):
                        reads.append(
                            (str(name), int(attributes["sequence_value_id"]))
                        )
        found = []
        # Program: ``d[3] = n; xs = [1, 2] + [n, 8]; xs[2]`` (c262).  The
        # concatenation copied ``[n, 8]`` before that display's birth cleared
        # and wrote it, in one straight-line block.  The two records are the
        # birth (the display's ``ssa_sequence_clear``) and the copy that
        # names it as its source.
        for name, function in module.functions.items():
            for block in function.blocks.values():
                copied: dict[int, int] = {}
                for position, instruction in enumerate(block.instrs):
                    attributes = getattr(instruction, "attributes", {}) or {}
                    source = attributes.get("source_sequence_id")
                    if (attributes.get("ssa_sequence_operation") == "append_slice"
                            and source is not None):
                        copied.setdefault(int(source), position)
                    if (attributes.get("binding") == "ssa_sequence_clear"
                            and attributes.get("sequence_id") is not None
                            and int(attributes["sequence_id"]) in copied
                            and int(attributes["sequence_id"]) in layouts):
                        found.append(Finding(
                            "display-read-before-birth", str(name),
                            int(attributes["sequence_id"]),
                            "a slice copy reads this display before its "
                            "birth clears and writes it in the same block",
                        ))
        # Program: ``ys = [x * n for x in xs]; ys.append(n); sum(ys)``
        # (c266).  The reduction counted the comprehension's three rows (a
        # static extent) while four appends wrote the sequence.  The two
        # records are the reduced value's static extent and the module's
        # appends to that sequence identity.
        appends: dict[int, int] = {}
        reductions: list[tuple[str, int, int]] = []
        for name, function in module.functions.items():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    attributes = getattr(instruction, "attributes", {}) or {}
                    if (attributes.get("ssa_sequence_operation") == "append"
                            and attributes.get("sequence_id") is not None):
                        sequence_id = int(attributes["sequence_id"])
                        appends[sequence_id] = appends.get(sequence_id, 0) + 1
                    if instruction.op not in {"sum", "max", "min", "any", "all"}:
                        continue
                    source = instruction.args[0] if instruction.args else None
                    accounting = getattr(source, "accounting", None) or {}
                    shape = tuple(getattr(source, "shape", None) or ())
                    if (accounting.get("sequence_id") is not None
                            and len(shape) == 1
                            and isinstance(shape[0], int)):
                        reductions.append((
                            str(name), int(accounting["sequence_id"]),
                            int(shape[0]),
                        ))
        for name, sequence_id, extent in dict.fromkeys(reductions):
            if appends.get(sequence_id, 0) > extent:
                found.append(Finding(
                    "static-extent-of-grown-sequence", name, sequence_id,
                    f"a reduction reads {extent} row(s) as a static extent, "
                    f"but {appends[sequence_id]} appends write this sequence",
                ))
        for name, sequence_id in dict.fromkeys(reads):
            layout = layouts.get(sequence_id)
            if layout is None or sequence_id in written:
                continue
            found.append(Finding(
                "display-length-unborn", name, sequence_id,
                f"{layout.aggregate_kind} display has {len(layout.members)} "
                "member(s) in its source layout, but no instruction writes "
                "its resident sequence before its length is read",
            ))
        return found

    @staticmethod
    def _source_sequence_mutation_findings(module: Any) -> list[Finding]:
        """Report a source mutation whose resident sequence fact changed."""

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_sequence_mutation_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in page.history(row)
            ))
            if len(distinct) <= 1:
                continue
            scope, value_id = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, None)
            )
            found.append(Finding(
                "source-sequence-mutation-disagreement",
                str(scope),
                None if value_id is None else int(value_id),
                "source mutation changed resident sequence, operator, "
                f"arguments, policy, or mutation kind: {distinct!r}",
            ))
        return found

    @staticmethod
    def _source_precision_boundary_findings(module: Any) -> list[Finding]:
        """Report a Precision boundary whose exact source fact changed."""

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_precision_boundary_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in history
            ))
            if len(distinct) <= 1:
                continue
            scope, value_id = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, None)
            )
            found.append(Finding(
                "source-precision-boundary-disagreement",
                str(scope),
                None if value_id is None else int(value_id),
                "Precision boundary changed kind, operand, or limb width "
                f"across source stages: {distinct!r}",
            ))
        return found

    @staticmethod
    def _source_precision_region_findings(module: Any) -> list[Finding]:
        """Report an indivisible Precision region whose membership changed."""

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_precision_region_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in history
            ))
            if len(distinct) <= 1:
                continue
            scope, collapse_id = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, None)
            )
            found.append(Finding(
                "source-precision-region-disagreement",
                str(scope),
                None if collapse_id is None else int(collapse_id),
                "Precision region changed members, boundaries, or limb "
                f"width across compilation stages: {distinct!r}",
            ))
        return found

    @staticmethod
    def _source_precision_operator_findings(module: Any) -> list[Finding]:
        """Report a wide operator whose source identity changed."""

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "source_precision_operator_concordance"
            )
        )
        if page is None:
            return []
        found = []
        for row in page.rows():
            history = page.history(row)
            distinct = tuple(dict.fromkeys(
                repr(fact) for _column, fact in history
            ))
            if len(distinct) <= 1:
                continue
            scope, value_id = (
                row if isinstance(row, tuple) and len(row) == 2
                else (row, None)
            )
            found.append(Finding(
                "source-precision-operator-disagreement",
                str(scope),
                None if value_id is None else int(value_id),
                "Precision operator changed operation, receiver, class, or "
                f"limb width across source stages: {distinct!r}",
            ))
        return found

    @staticmethod
    def _binding_kind_findings(module: Any) -> list[Finding]:
        """One caller slot bound under two different kinds across callsites.

        A frame binding's kind is not a label on the slot; it selects which
        machinery may materialize it.  ``caller_storage`` can restore a slot
        a structural cleanup removed, ``caller_alias`` and ``caller_value``
        cannot.  So when two callsites name the SAME caller id for the same
        callee formal but disagree on the kind, they do not merely describe
        it differently -- one of them can supply the argument and the other
        reports ``missing_<kind>`` and refuses the call.

        Neither callsite can see this on its own: each consults its own
        private maps in its own elif order and records a locally consistent
        answer.  The disagreement only exists across the pair, which is the
        whole reason the decisions are written to a shared page.
        """
        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = (getattr(book, "pages", {}) or {}).get("argument_binding")
        if page is None:
            return []
        found: list[Finding] = []
        # Step 7 keys one row per (callee, formal, callsite); rows written
        # before it key (callee, formal, "binding") with the callsite as the
        # column.  Both are read per (callee, formal).
        formals: dict[tuple, dict[Any, dict[str, list]]] = {}
        for row in page.rows():
            if not (isinstance(row, tuple) and len(row) >= 2):
                continue
            kinds_by_source = formals.setdefault((row[0], row[1]), {})
            for callsite, fact in argument_binding_history(page, row[0], row[1], rows=(row,)):
                if not (isinstance(fact, tuple) and len(fact) == 2):
                    continue
                kind, source = fact
                if not isinstance(source, int):
                    continue
                kinds_by_source.setdefault(int(source), {}).setdefault(
                    str(kind), []
                ).append(callsite)
        for (function_name, value_id), kinds_by_source in formals.items():
            for source, by_kind in sorted(kinds_by_source.items()):
                if len(by_kind) < 2:
                    continue
                resolution = book.page(
                    "argument_binding_resolution"
                ).latest((str(function_name), int(value_id), int(source)))
                if (
                    isinstance(resolution, tuple)
                    and resolution
                    and str(resolution[0]) in by_kind
                ):
                    # The raw history remains intact, but the shared
                    # concordance has selected the one kind capable of
                    # materializing this exact source slot.
                    continue
                found.append(Finding(
                    "binding-kind-disagreement",
                    str(function_name),
                    int(value_id),
                    f"caller id {id_label(int(source))} is bound as "
                    + "; ".join(
                        f"{kind!r} at callsite(s) {sorted(columns, key=repr)}"
                        for kind, columns in sorted(by_kind.items())
                    )
                    + " -- one slot, and only some of those kinds can "
                      "materialize it",
                ))
        return found

    @staticmethod
    def _sequence_descriptor_findings(module: Any) -> list[Finding]:
        """Descriptors whose own two records contradict each other.

        A descriptor states which of its columns are keys AND what each
        column holds.  When it says column 0 is a key and also says that
        column is float64, those are two of its own claims disagreeing --
        exactly what this table exists to catch, and catchable without
        knowing which pass wrote it.

        It matters because a key is not merely imprecise as a float: a
        string key lowers to an fnv1a-**64** token and float64 carries only
        53 bits exactly, so a token above 2**53 is silently rounded and then
        never matches its own lookup.  The entry goes missing rather than
        failing loudly, which is the worst available outcome.
        """
        # The shared layout table spells opaque_ref as exact i64 storage.
        # Native identity-keyed tables legitimately compare these handles;
        # a second spelling inventory incorrectly called them float keys.
        from ..transmogrifier.dtype_layout import layout_for
        found: list[Finding] = []
        for function_name, table in (
            getattr(module, "sequence_tables", {}) or {}
        ).items():
            for sequence_id, descriptor in sorted(
                getattr(table, "sequences", {}).items()
            ):
                dtypes = tuple(getattr(descriptor, "column_dtypes", ()) or ())
                for column in getattr(descriptor, "key_columns", ()) or ():
                    if int(column) >= len(dtypes):
                        continue
                    dtype = str(dtypes[int(column)])
                    try:
                        integral = layout_for(dtype).kind in {"int", "uint", "bool"}
                    except KeyError:
                        integral = False
                    if integral:
                        continue
                    found.append(Finding(
                        "key-column-not-integral",
                        str(function_name),
                        int(sequence_id),
                        f"column {int(column)} is declared a key but holds "
                        f"{dtype!r}; a key column is an index and cannot be "
                        f"a float (dtypes={dtypes})",
                    ))
        return found

    def _identity_claims(self, row: ValueRow) -> list[Claim]:
        return [
            claim for claim in row.claims
            if claim.kind in {
                "abi-field", "abi-parameter", "keyed-part", "parameter",
                "storage-formal", "closure-formal", "member-formal",
                "frame-storage", "compiler-frame-storage", "projected-row",
                "source-transform",
            }
        ]

    @staticmethod
    def _is_authored(function: Any) -> bool:
        """Source-level function (not a planned region or generated helper).

        Regions and sequence/tensor helpers receive their formals by the
        planner's own slot convention and never carry ``parameter_names``;
        judging them by the authored-function records would only report
        the convention itself.
        """

        metadata = dict(getattr(function, "metadata", {}) or {})
        return bool(
            metadata.get("parameter_names")
            or metadata.get("authored_parameters")
        ) and not metadata.get("source_region_integral")

    def _function_findings(self, module: Any, name: str,
                           function: Any) -> list[Finding]:
        found: list[Finding] = []
        metadata = dict(getattr(function, "metadata", {}) or {})
        authored = self._is_authored(function)
        formals = {int(formal.id): formal for formal in function.args}
        rows = {
            value_id: row for (function_name, value_id), row in self.rows.items()
            if function_name == name
        }
        # multiple-definition
        for value_id, row in rows.items():
            if len(row.definitions) <= 1:
                continue
            formal = formals.get(value_id)
            accounting = dict(getattr(formal, "accounting", {}) or {})
            if formal is not None and all(
                accounting.get(key) for key in _INOUT_REDEFINED
            ):
                continue
            found.append(Finding(
                "multiple-definition", name, value_id,
                f"defined {len(row.definitions)} times: "
                f"{row.definitions[:4]!r}",
            ))
        # formal accountability and conflicts (authored functions only)
        for value_id, formal in formals.items():
            row = rows[value_id]
            identity = self._identity_claims(row)
            if not identity and not authored:
                continue
            if not identity:
                found.append(Finding(
                    "unaccounted-formal", name, value_id,
                    f"dtype={formal.dtype} accounting_keys="
                    f"{sorted(dict(formal.accounting or {}))[:6]!r}",
                ))
                continue
            abi_fields = {c.key for c in identity if c.kind == "abi-field"}
            frame = [c for c in identity if c.kind == "frame-storage"]
            accounting = dict(formal.accounting or {})
            if accounting.get("returned_record_storage") is not None:
                # Storage leased for a returned record's field carries that
                # field's identity by design; leased + field is the
                # convention here, not a conflict.
                frame = []
            if len(abi_fields) > 1 or (abi_fields and frame):
                found.append(Finding(
                    "conflicting-storage-claims", name, value_id,
                    f"claims={[(c.kind, c.key) for c in identity]!r}",
                ))
        # sequence descriptor members (authored functions only: a generated
        # helper's descriptor names the caller's cells it is handed)
        for (function_name, value_id), roles in self.sequence_roles.items():
            if function_name != name or not authored:
                continue
            row = rows.get(value_id)
            # ``sequence_id`` is the descriptor's compile-time identity.  It
            # commonly aliases column zero, but a selected nested row keeps
            # its authored graph identity while ``column_value_ids`` names the
            # row-base address produced inside the loop.  Only the physical
            # members have to be formals or instruction results.  Treating a
            # handle-only descriptor key as storage invents a definition the
            # SSA program neither has nor needs.
            physical_roles = tuple(
                role for _sequence_id, role in roles if role != "handle"
            )
            if physical_roles and (
                row is None or (not row.is_formal and not row.definitions)
            ):
                found.append(Finding(
                    "descriptor-member-unknown", name, value_id,
                    f"roles={roles!r}: not a formal, never defined",
                ))
            distinct = {(sequence_id, role.rstrip("0123456789"))
                        for sequence_id, role in roles}
            sequences = {sequence_id for sequence_id, _role in distinct}
            view_receipts = {
                int(member): set(map(int, sequence_ids))
                for member, sequence_ids, _target
                in metadata.get("concordant_sequence_views", ()) or ()
            }
            if (
                len(sequences) > 1
                and view_receipts.get(int(value_id)) == sequences
            ):
                continue
            if len(sequences) > 1 and not all(role == "heap_owner" for _, role in roles):
                found.append(Finding(
                    "descriptor-member-shared", name, value_id,
                    f"roles={roles!r}",
                ))
        if (function.metadata.get("object_context") is not None
                or function.metadata.get("native_object_context") is not None):
            from .ssa_object_context import validate_context_function, validate_native_object_requirements
            try:
                validate_context_function(module, name)
                validate_native_object_requirements(module, name)
            except (KeyError, ValueError, TypeError) as error:
                found.append(Finding("object-context-contract", name, None, str(error)))
        from .ssa_heap_allocation import validate_heap_context_projection
        for formal in function.args:
            try:
                validate_heap_context_projection(module, name, formal)
            except (KeyError, ValueError, TypeError) as error:
                found.append(Finding("heap-context-projection", name, formal.id, str(error)))
        # keyed helper operands
        described = {
            value_id for (function_name, value_id) in self.sequence_roles
            if function_name == name
        }
        keyed_parts = {
            value_id for value_id, row in rows.items()
            if any(c.kind == "keyed-part" for c in row.claims)
        }
        from .ssa_object_context import source_object_definitions
        source_definitions = (source_object_definitions(function)
            if function.metadata.get("native_object_context") is not None else {})
        object_helpers = {}
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                attributes = instruction.attributes or {}
                membership = attributes.get("structural_membership_parameter")
                if membership is not None:
                    sequence_table = module.sequence_tables.get(name)
                    descriptor = getattr(sequence_table, "sequences", {}).get(
                        int(membership["sequence_id"])
                    )
                    storage = () if descriptor is None else tuple(dict.fromkeys((
                        *descriptor.column_value_ids, descriptor.length_address_id,
                        descriptor.capacity_value_id,
                        *((descriptor.live_flags_value_id,)
                          if descriptor.live_flags_value_id is not None else ()),
                    )))
                    actual = tuple(int(value.id) for value in instruction.args)
                    source_queries = tuple(map(
                        int,
                        membership.get("source_query_ids") or (
                            (membership["source_query_id"],)
                            if membership.get("source_query_id") is not None
                            else ()
                        ),
                    ))
                    helper = module.functions.get(str(attributes.get("callee") or ""))
                    # A storage formal the helper never reads (a contains
                    # scan's capacity) is pruned from helper and call alike;
                    # the pruning pass records each one on
                    # ``pruned_callee_formal_concordance``.
                    pruned_page = identity_book(module).pages.get(
                        "pruned_callee_formal_concordance"
                    )
                    storage = tuple(
                        value_id for value_id in storage
                        if pruned_page is None or pruned_page.latest(
                            (str(attributes.get("callee") or ""), int(value_id))
                        ) is None
                    )
                    if (descriptor is None or instruction.op != "Call"
                            or attributes.get("ssa_sequence_operation") != "contains"
                            or int(membership["source_target_id"]) != int(descriptor.sequence_id)
                            or actual != (*storage, *source_queries)
                            or helper is None
                            or helper.metadata.get("ssa_sequence_operation") != "contains"
                            or len(helper.args) != len(actual)):
                        found.append(Finding("membership-parameter-contract", name,
                            None if instruction.res is None else int(instruction.res.id),
                            f"{block_name}#{index}: membership operands disagree with exact parameter descriptor"))
                if (attributes.get("source_object_constructor") is not None
                        or attributes.get("source_object_operation") is not None
                        or attributes.get("source_object_type_test") is not None
                        or attributes.get("native_collection_row") is not None):
                    from .ssa_object_context import validate_source_object_instruction
                    try:
                        validate_source_object_instruction(module, name, instruction,
                            _definitions=source_definitions, _helpers=object_helpers)
                    except (KeyError, ValueError, TypeError) as error:
                        found.append(Finding("source-object-contract", name,
                            None if instruction.res is None else int(instruction.res.id), str(error)))
                if attributes.get("opaque_object_operation") is not None:
                    from .ssa_object_operations import validate_object_call
                    try:
                        validate_object_call(module, instruction, _helpers=object_helpers)
                    except (KeyError, ValueError) as error:
                        found.append(Finding(
                            "opaque-object-contract", name,
                            None if instruction.res is None else int(instruction.res.id),
                            f"{block_name}#{index}: {error}",
                        ))
                if attributes.get("allocation_kind") == "owned_heap":
                    from .ssa_heap_allocation import validate_heap_allocation
                    try:
                        validate_heap_allocation(instruction)
                    except (KeyError, ValueError) as error:
                        found.append(Finding(
                            "heap-allocation-contract", name,
                            None if instruction.res is None else int(instruction.res.id),
                            f"{block_name}#{index}: {error}",
                        ))
                if attributes.get("native_heap_abi") == "copy_elements":
                    from .ssa_heap_allocation import validate_heap_copy
                    try:
                        validate_heap_copy(instruction)
                    except (KeyError, ValueError) as error:
                        found.append(Finding(
                            "heap-copy-contract", name, None,
                            f"{block_name}#{index}: {error}",
                        ))
                if attributes.get("keyed_lookup_owner") is None:
                    continue
                for position, argument in enumerate(instruction.args[:4]):
                    argument_id = int(argument.id)
                    if argument_id in described or argument_id in keyed_parts:
                        continue
                    found.append(Finding(
                        "helper-operand-outside-descriptor", name, argument_id,
                        f"{block_name}#{index} {attributes.get('callee')} "
                        f"operand {position} (owner "
                        f"{attributes.get('keyed_lookup_owner')!r})",
                    ))
        # duplicate storage across a call
        callee_identity: dict[str, dict[int, str]] = {}
        for callee_name, callee in module.functions.items():
            callee_identity[str(callee_name)] = {
                int(formal.id): key
                for formal in callee.args
                for key in (self._abi_or_member_key(str(callee_name), formal),)
                if key is not None
            }
        caller_by_key: dict[str, list[int]] = defaultdict(list)
        for value_id, formal in formals.items():
            key = self._abi_or_member_key(name, formal)
            if key is not None:
                caller_by_key[key].append(value_id)
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.op not in {"Call", "call"}:
                    continue
                attributes = instruction.attributes or {}
                callee_name = str(attributes.get("callee") or "")
                declared = attributes.get("callee_input_ids")
                if callee_name not in callee_identity or declared is None:
                    continue
                for callee_id, argument in zip(declared, instruction.args):
                    key = callee_identity[callee_name].get(int(callee_id))
                    if key is None:
                        continue
                    argument_id = int(argument.id)
                    accounting = dict(getattr(argument, "accounting", {}) or {})
                    if not accounting.get("linked_call_frame_storage"):
                        continue
                    owned = [
                        owner for owner in caller_by_key.get(key, ())
                        if owner != argument_id
                    ]
                    if owned:
                        found.append(Finding(
                            "duplicate-storage-across-call", name, argument_id,
                            f"{block_name}#{index} -> {callee_name} formal "
                            f"{callee_id} ({key}) fed leased storage "
                            f"{argument_id} while caller owns {owned!r}",
                        ))
        # dominance
        found.extend(self._dominance_findings(name, function, formals))
        # alias targets
        for value_id, row in rows.items():
            for claim in row.claims:
                if claim.kind != "alias-of":
                    continue
                target = rows.get(int(claim.key))
                if target is None or (
                    not target.is_formal and not target.definitions
                    # coerce_metrics returns the exact Metrics handle. Once
                    # flattened, its fields have physical definitions but the
                    # container only exists in its authoritative record table.
                    # This is not permission for a raw SSA operand to use it:
                    # the separate undefined-operand check still rejects that.
                    and not any(c.kind == "record-owner" for c in target.claims)
                ):
                    found.append(Finding(
                        "alias-target-missing", name, value_id,
                        f"alias of {claim.key} which is never defined",
                    ))
        # A private alias snapshot is allowed only as a durable copy of the
        # shared authority. This catches the exact class of failure where a
        # late pass proves an identity in ``metadata.value_aliases`` but the
        # next pass reads only ``planning_value_concordance`` (or vice versa).
        # Imported functions keep the book that accepted them; the composing
        # module's book owns only the current unit and does not duplicate the
        # imported output-identity rows.
        book = (
            dict(metadata.get("accepted_ssa_component") or {}).get(
                "identity_book"
            )
            or dict(getattr(module, "metadata", {}) or {}).get(
                "identity_book"
            )
        )
        page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "planning_value_concordance"
            )
        )
        output_page = (
            None if book is None else
            (getattr(book, "pages", {}) or {}).get(
                "output_identity_concordance"
            )
        )
        if book is not None:
            durable_aliases: list[tuple[str, Any, int, int]] = []
            local_aliases = metadata.get("value_aliases", ()) or ()
            local_pairs = (
                local_aliases.items()
                if isinstance(local_aliases, Mapping) else local_aliases
            )
            durable_aliases.extend(
                (
                    "metadata.value_aliases", page,
                    int(alias), int(target),
                )
                for alias, target in local_pairs
            )
            durable_aliases.extend(
                (
                    "metadata.output_identity_aliases",
                    output_page,
                    int(alias), int(target),
                )
                for alias, target in (
                    metadata.get("output_identity_aliases", ()) or ()
                )
            )
            for source, authority, alias, target in durable_aliases:
                concorded = (
                    None if authority is None
                    else authority.latest((name, alias))
                )
                if concorded is not None and int(concorded) == target:
                    continue
                found.append(Finding(
                    "alias-not-concorded", name, alias,
                    f"{source} says {target}, "
                    f"{('output_identity_concordance' if source.endswith('output_identity_aliases') else 'planning_value_concordance')} "
                    f"says {concorded!r}",
                ))
        return found

    def _abi_or_member_key(
        self, function: str, formal: SSAValue,
    ) -> str | None:
        # ``formal`` is an SSAValue at both call sites above.  Preserve its
        # authored accounting table as the receiver: copying it through
        # ``dict(...)`` or replacing it through ``getattr(..., {}) or {}``
        # severs the resident keyed-table identity needed by source lowering.
        # The named ABI identity entries read below are the string-valued
        # projection of the heterogeneous accounting table.  ``callsite_id``
        # is read separately from the same exact table with its integer
        # projection; keeping the projections typed prevents their lookup
        # results from escaping as opaque public formals.
        accounting: dict[str, str] = formal.accounting
        abi_field: str | None = accounting.get("program_abi_field")
        if abi_field is not None:
            # Identity is per record INSTANCE: the parameter that carries
            # the record, or the call whose result record this storage
            # holds.  Two instances of one record type in one function
            # (``metrics`` and ``coerced = coerce_metrics(metrics)``)
            # legitimately own the same field name each.
            returned_record_storage: str | None = accounting.get(
                "returned_record_storage"
            )
            instance: str | None = returned_record_storage
            if returned_record_storage is not None:
                callsite_accounting: dict[str, int] = formal.accounting
                callsite_id: int | None = callsite_accounting.get("callsite_id")
                instance = f"{returned_record_storage}@{callsite_id}"
            else:
                program_parameter: str | None = accounting.get(
                    "program_abi_parameter"
                )
                program_record: str | None = accounting.get(
                    "program_abi_record"
                )
                instance = program_parameter or program_record
            return f"{instance}.{abi_field}"
        roles: list[tuple[int, str]] | None = self.sequence_roles.get((
            function, int(formal.id),
        ))
        if roles:
            sequence_id, role = roles[0]
            # Sequence ids are compile-time identities local to one lowered
            # function.  The same integer can name unrelated residents in a
            # caller and callee (for example ``stamps`` and ``forks``).  A
            # record-backed sequence was already identified above by its
            # stable ProgramABI field; every other sequence member must keep
            # its compiler-frame owner in the comparison key.
            owner = accounting.get("compiler_frame_storage") or function
            return f"{owner}.seq{sequence_id}.{role}"
        return None

    @staticmethod
    def _dominators(function: Any) -> dict[str, set[str]]:
        blocks = list(function.blocks)
        if not blocks:
            return {}
        entry = blocks[0]
        predecessors: dict[str, set[str]] = {b: set() for b in blocks}
        for block_name, block in function.blocks.items():
            for successor in block.successors:
                if successor in predecessors:
                    predecessors[successor].add(block_name)
        reachable = {entry}
        stack = [entry]
        while stack:
            current = stack.pop()
            for successor in function.blocks[current].successors:
                if successor in predecessors and successor not in reachable:
                    reachable.add(successor)
                    stack.append(successor)
        dominators = {b: set(reachable) for b in reachable}
        dominators[entry] = {entry}
        changed = True
        while changed:
            changed = False
            for block_name in reachable:
                if block_name == entry:
                    continue
                incoming = [
                    dominators[p] for p in predecessors[block_name]
                    if p in reachable
                ]
                updated = ({block_name} | set.intersection(*incoming)
                           if incoming else {block_name})
                if updated != dominators[block_name]:
                    dominators[block_name] = updated
                    changed = True
        return dominators

    def _dominance_findings(self, name: str, function: Any,
                            formals: Mapping[int, Any]) -> list[Finding]:
        found: list[Finding] = []
        dominators = self._dominators(function)
        if not dominators:
            return found
        definition_sites: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is not None:
                    definition_sites[int(instruction.res.id)].append(
                        (block_name, index)
                    )
        for block_name, block in function.blocks.items():
            if block_name not in dominators:
                continue
            for index, instruction in enumerate(block.instrs):
                if instruction.op in {"Phi", "phi"}:
                    continue
                for argument in instruction.args:
                    argument_id = getattr(argument, "id", None)
                    if argument_id is None or int(argument_id) in formals:
                        continue
                    sites = definition_sites.get(int(argument_id))
                    if not sites:
                        continue
                    if any(
                        (site_block == block_name and site_index <= index)
                        or (site_block != block_name
                            and site_block in dominators[block_name])
                        for site_block, site_index in sites
                    ):
                        # ``site_index == index``: a region call names its
                        # own result among its operands (the out-pointer
                        # convention), which is a definition, not a read.
                        continue
                    found.append(Finding(
                        "use-not-dominated", name, int(argument_id),
                        f"{block_name}#{index} {instruction.op} reads a "
                        f"value defined at {sites[:3]!r}",
                    ))
        return found


def concordance_report(module: Any, *, limit: int = 12) -> str:
    """Build the table and render its findings as text."""

    table = CorrelationTable.build(module)
    findings = table.findings(module)
    by_kind: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        by_kind[finding.kind].append(finding)
    # The first line is the gate ``tools/audit_identity_concordance.py``
    # reads; the two ``unsourced-*`` kinds are the migration worklist and
    # are counted on their own line below, not here.
    gated = [f for f in findings if f.kind not in _UNSOURCED_KINDS]
    census = group_by_prefix(value_id for _function, value_id in table.rows)
    lines = [
        f"identity concordance: {len(table.rows)} rows across "
        f"{len(module.functions)} functions, {len(gated)} finding(s)"
    ]
    # Every row gathered under its own id group, before any finding: a
    # count per prefix says at a glance which spaces this module actually
    # uses, and a group that should be empty (``history`` ids among a
    # function's own values, say) shows up as a number rather than having
    # to be hunted for.
    if len(census) > 1 or (census and census[0].label != "legacy"):
        lines.append(
            "  id groups: "
            + ", ".join(
                f"{group.label}={len(group.value_ids)}" for group in census
            )
        )
    groups, identities = table._unsourced_worklist(module)
    lines.append(
        f"  unsourced: {sum(groups.values())} fact(s), "
        f"{len(identities)} identit(ies) -- latch "
        f"{identity_book(module).latch.name}"
    )
    for kind in sorted(by_kind):
        entries = by_kind[kind]
        lines.append(f"  [{kind}] x{len(entries)}")
        for finding in entries[:limit]:
            named = (
                "?" if finding.value_id is None
                else id_label(finding.value_id)
            )
            lines.append(
                f"     {finding.function} value {named}: "
                f"{finding.detail}"
            )
        if len(entries) > limit:
            lines.append(f"     ... {len(entries) - limit} more")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Identity pages: row x column x page, for facts a finished-module table
# cannot see.
#
# ``CorrelationTable`` above audits one finished module -- it has no notion
# of time, so it can only ever compare a value against itself, never against
# what it USED to be.  Two 2026-09-17/18 defects were exactly that: a
# correct fact computed once but never carried to the one consumer that
# needed it (an alias map read in two places, not the third that mattered),
# and a value's shape flipping A -> B -> A within a single fixed-point round
# while every step honestly reported "changed" -- a round-boundary snapshot
# necessarily reads that as no change at all, because it is none, net.
#
# A page is one pipeline stage's table.  A row is one identity -- whatever a
# page decides makes two facts "about the same thing" (a value id, an
# (function, value id) pair, ...).  A column is one round -- whatever
# "round" means on that page (a fixed-point iteration, a phase index).  A
# cell is the fact that identity held at that round.  Reading one row across
# its own columns finds an in-stage oscillation.  Reading one row's key
# across two different pages finds a cross-stage disagreement -- the shape
# of every fault above and, going forward, the general instrument for both.
# --------------------------------------------------------------------------


def _blocks_between(function: Any, header: str, latch: str) -> set[str]:
    """Blocks on some path from the declared header to the declared latch."""

    successors: dict[str, set[str]] = {}
    for block_name, block in function.blocks.items():
        targets: set[str] = set()
        for instruction in block.instrs:
            for key in ("target", "true_target", "false_target"):
                declared = instruction.attributes.get(key)
                if declared is not None:
                    targets.add(str(declared))
        successors[str(block_name)] = targets
    forward: set[str] = set()
    frontier = [str(header)]
    while frontier:
        current = frontier.pop()
        if current in forward:
            continue
        forward.add(current)
        frontier.extend(successors.get(current, ()))
    backward: set[str] = set()
    frontier = [str(latch)]
    while frontier:
        current = frontier.pop()
        if current in backward:
            continue
        backward.add(current)
        if current == str(header):
            # natural loop: the walk stops at the header, it does not
            # expand the header's predecessors (those are outside the loop)
            continue
        for candidate, onward in successors.items():
            if current in onward:
                frontier.append(candidate)
    return forward & backward


def _carried_phi(function: Any, header: str, carried_id: int) -> Any:
    """The header Phi that speaks for one declared carried generation."""

    block = function.blocks.get(str(header))
    for instruction in (block.instrs if block is not None else ()):
        if str(instruction.op).lower() != "phi":
            continue
        if instruction.res is None:
            continue
        if int(instruction.res.id) == int(carried_id):
            return instruction
    return None


# --------------------------------------------------------------------------
# The one writing api: ``IdentityBook.post`` (design:
# docs/CONCORDANCE_SINGLE_API_DESIGN_2026-09-30.md, sections 2 and 6).
#
# A post names the page it writes (a registry object, never a free string),
# the row (validated against the page's declared shape), the fact, the
# stage making the statement, and its provenance: ``Derived`` from exact
# source cells, ``Novel`` (the book mints the id and records the transform
# that produced it), or ``Unsourced`` (admitted only while the book's latch
# is OPEN, and recorded so the audit lists it).  The edge, reverse-index,
# mint and unsourced records live on four private pages named below; the
# read api on the book (``edges_into``, ``edges_out_of``, ``mint_of``,
# ``unsourced_rows``) is how a viewer reads them without knowing the names.
#
# Migration state: every write that still reaches ``IdentityPage.set``
# without coming through ``post`` is tagged ``Unsourced(RAW_PRIMITIVE)`` on
# the unsourced page while the latch is OPEN, and refused once it is CLOSED.
# --------------------------------------------------------------------------


class ConcordanceRefusal(ValueError):
    """A post (or a raw write under a CLOSED latch) the book would not admit.

    Raised at the call; nothing is recorded.
    """


class ProjectedScopeRead(LookupError):
    """A row of a node the reading scope's declared projection does not hold.

    ``fork_read_scope`` of a proper induced subgraph carries the rows of the
    copy's own nodes (``ScopeProjection``).  A read, under that forked scope,
    of a node-keyed row of a node the SOURCE graph has and the copy does not
    would otherwise miss like any absent row; it is refused instead, naming
    the scope, page and row, after an ``Unresolved(ROW_PROJECTED_OUT_OF_SCOPE)``
    row is posted at the key (derived from the scope's ``scope_origin`` cell,
    which carries the projection)."""


@dataclass(frozen=True)
class LiveScopeProjection:
    """The read-time form of a forked scope's declared ``ScopeProjection``:
    which pages are node-keyed and which source-graph nodes the scope does not
    hold.  Registered on the book (``IdentityBook.scope_projections``) by the
    writer that posts the projection, so the book can refuse a read of a node
    the relation says the scope does not hold."""

    pages: frozenset[Any]
    #: Nodes excluded by name (a caller that holds the set itself).
    excluded: frozenset[Any] = frozenset()
    #: Excluded as a COMPLEMENT: the nodes of the source graph at fork time
    #: (``universe``, one frozenset shared by every region cut from that
    #: graph -- see ``IdentityBook.register_scope_projection``) that this
    #: scope does not hold (``held``).  Spelling a region's exclusions as
    #: ``universe - held`` cost one source-sized set per region, retained
    #: for the whole compile; the complement costs the region's own nodes.
    universe: frozenset[Any] = frozenset()
    held: frozenset[Any] = frozenset()
    #: The projection of the scope this one was forked from, whose
    #: exclusions it inherits.
    inherited: "LiveScopeProjection | None" = None
    #: This level HOLDS only ``held`` of the node-keyed rows of its origin
    #: (a fork cut for a proper induced subgraph): a read-through of a node
    #: outside ``held`` finds nothing, whether or not it is also refused
    #: (``excludes``) -- the rows of ids that were nodes of neither graph
    #: (``ScopeProjection.rows_stale``) are not held.
    restricts: bool = False

    def reaches(self, node: Any) -> bool:
        """Whether a read-through of ``node``'s row under this scope may pass
        on to the scope's origin: the node is not refused and, when this
        level restricts to its own nodes, it is one of them."""
        if self.excludes(node):
            return False
        return not self.restricts or node in self.held

    def excludes(self, node: Any) -> bool:
        """Whether a read of ``node``'s rows under this scope is refused."""
        projection: LiveScopeProjection | None = self
        while projection is not None:
            if node in projection.excluded or (
                node in projection.universe and node not in projection.held
            ):
                return True
            projection = projection.inherited
        return False


class Mode(Enum):
    CONCORD = "concord"
    REVISE = "revise"


class Latch(Enum):
    OPEN = "open"
    CLOSED = "closed"


class RowFieldKind(Enum):
    SCOPE = "scope"
    VALUE_ID = "value_id"
    NAME = "name"
    INDEX = "index"
    LABEL = "label"
    PAGE_REF = "page_ref"


class _New:
    """The sentinel a ``Novel`` post carries where the minted id will go."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "NEW"

    def __reduce__(self):
        return "NEW"


NEW = _New()


def _hashable(item: Any) -> bool:
    try:
        hash(item)
    except TypeError:
        return False
    return True


@dataclass(frozen=True)
class RowField:
    name: str
    kind: RowFieldKind

    def admits(self, item: Any) -> bool:
        kind = self.kind
        if kind is RowFieldKind.SCOPE:
            return item is not None and _hashable(item)
        if kind is RowFieldKind.VALUE_ID:
            return isinstance(item, int) and not isinstance(item, bool)
        if kind is RowFieldKind.NAME:
            return isinstance(item, str)
        if kind is RowFieldKind.INDEX:
            return isinstance(item, int) and not isinstance(item, bool)
        if kind is RowFieldKind.LABEL:
            return _hashable(item)
        if kind is RowFieldKind.PAGE_REF:
            return isinstance(item, Ref)
        return False


class IdentityLogLevel(IntEnum):
    """How much of a book the compile log keeps.  A page declares the
    minimum level at which its ROWS are printed (``Page.rows_level``)."""

    OFF = 0
    SUMMARY = 1
    FACTS = 2
    FULL = 3


@dataclass(frozen=True, repr=False)
class Page:
    """A declared page: its name, its row shape and the type of its facts.

    ``rows_level`` is the minimum log level at which the page's rows are
    printed (the page's header line is printed from SUMMARY up).  It is a
    logging declaration, not part of the page's shape: it does not take part
    in equality, so declaring a page again never conflicts over it."""

    name: str
    row_fields: tuple[RowField, ...]
    # The eager implementation accepts either one class or the tuple-of-
    # classes form supported by ``isinstance``.  Native compilation lowers
    # this API-declared expectation to concorded class identities and integer
    # tags; no class object crosses a sectional ABI.
    fact_type: type | tuple[type, ...] = object
    rows_level: IdentityLogLevel = field(
        default=IdentityLogLevel.FACTS, compare=False,
    )

    def __repr__(self) -> str:
        return f"Page({self.name!r})"

    def __hash__(self) -> int:
        # The generated frozen-dataclass hash re-hashes the row-field tuple
        # (each RowField, each kind) on every call, and a Page is hashed
        # through every Ref inside every edge row and every cell key that
        # names one (a quarter of the orbital dispatch-extraction profile).
        # The fields are frozen, so the hash is computed once and kept; it
        # is the hash of exactly the compared fields, as the generated one
        # was, so equality semantics are unchanged.
        cached = self.__dict__.get("_hash")
        if cached is None:
            cached = hash((self.name, self.row_fields, self.fact_type))
            object.__setattr__(self, "_hash", cached)
        return cached

    def __getstate__(self) -> dict:
        # The cached hash is this process's (string hashes are salted per
        # process); a pickled or copied instance recomputes it.
        return {key: value for key, value in self.__dict__.items() if key != "_hash"}


@dataclass(frozen=True)
class Stage:
    name: str


@dataclass(frozen=True)
class Transform:
    name: str
    arity: int


@dataclass(frozen=True)
class Reason:
    name: str


@dataclass(frozen=True, repr=False, slots=True)
class Ref:
    """The full vector of one cell's location: page, row, column."""

    page: Page
    row: tuple
    column: int
    _hash: int | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def key(self) -> tuple[str, tuple, int]:
        """The cell's location as it is stored inside edge rows."""
        return (self.page.name, self.row, self.column)

    def __repr__(self) -> str:
        return f"Ref({self.page.name!r}, {render_row(self.row)}, {self.column})"

    def __hash__(self) -> int:
        # Same rule as ``Page.__hash__``: frozen fields, hashed once.  A Ref
        # is a cell key's member wherever a row names a cell (field-state
        # rows, edge and dependents rows), so it is hashed on every stamp.
        cached = self._hash
        if cached is None:
            cached = hash((self.page, self.row, self.column))
            object.__setattr__(self, "_hash", cached)
        return cached

    def __getstate__(self) -> dict:
        # The cached hash is this process's (string hashes are salted per
        # process); a pickled or copied instance recomputes it.
        return {"page": self.page, "row": self.row, "column": self.column}

    def __setstate__(self, state):
        # Old Ref pickles also carried a dict; ignore their process-salted hash.
        for name in ("page", "row", "column"):
            object.__setattr__(self, name, state[name])
        object.__setattr__(self, "_hash", None)


@dataclass(frozen=True)
class Derived:
    cells: tuple[Ref, ...]


@dataclass(frozen=True)
class ImportedCell:
    """An exact source edge into a digest-pinned, independently owned book.

    The owner is retained, not copied into the importing book's namespace.
    In particular equal local row/column coordinates in two units do not
    identify the same cell. The recorded fact is checked on every use.
    """

    source: Ref
    fact: Any
    owner: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class Novel:
    transform: Transform
    operands: tuple[Ref, ...]


@dataclass(frozen=True)
class Unsourced:
    reason: Reason


@dataclass(frozen=True)
class Unresolved:
    """The fact a writer posts when it looked and could not decide."""

    reason: Reason
    read: tuple[Ref, ...] = ()


@dataclass(frozen=True)
class DerivationSnapshot:
    """A live view's recipe and the exact revisions it last consumed.

    Ordinary Derived edges remain historical evidence. Only a view explicitly
    registered here follows the current revisions of its input rows. Recipes
    are import-owned functions, so the same declaration survives book copying,
    spilling and worker checkpoints without retaining compiler object closures.
    """

    recipe: Any
    arguments: tuple
    stage: Stage
    target: Ref
    reads: tuple  # (Page, row, Ref | None), including observed absence


class Registry:
    """Declared pages, stages, transforms and reasons: the closed vocabulary
    ``IdentityBook.post`` accepts.  Each name is declared once; declaring it
    again with the same shape returns the existing object, with a different
    shape is refused."""

    def __init__(self) -> None:
        self.pages: dict[str, Page] = {}
        self.stages: dict[str, Stage] = {}
        self.transforms: dict[str, Transform] = {}
        self.reasons: dict[str, Reason] = {}
        #: Pages the book writes for itself (edges, reverse index, mints,
        #: unsourced tags).  Their cells are the edges; the ``unsourced-fact``
        #: finding does not ask them for edges of their own.
        self.private_pages: set[str] = set()

    def declare_page(
        self,
        name: str,
        row_fields: Iterable[RowField],
        fact_type: type | tuple[type, ...] = object,
        *,
        private: bool = False,
        rows_level: IdentityLogLevel = IdentityLogLevel.FACTS,
    ) -> Page:
        fields = tuple(row_fields)
        if not isinstance(name, str) or not name:
            raise ConcordanceRefusal(f"page name must be a non-empty str: {name!r}")
        if not fields or not all(isinstance(item, RowField) for item in fields):
            raise ConcordanceRefusal(
                f"page {name!r}: row_fields must be a non-empty tuple of RowField"
            )
        proposed = Page(name, fields, fact_type, IdentityLogLevel(rows_level))
        existing = self.pages.get(name)
        if existing is None:
            self.pages[name] = proposed
            if private:
                self.private_pages.add(name)
            return proposed
        if existing != proposed:
            raise ConcordanceRefusal(
                f"page {name!r} already declared with a different shape: "
                f"declared={existing.row_fields!r}/{existing.fact_type!r}, "
                f"proposed={fields!r}/{fact_type!r}"
            )
        return existing

    def declare_stage(self, name: str) -> Stage:
        return self._declare(self.stages, Stage(str(name)), "stage")

    def declare_transform(self, name: str, arity: int) -> Transform:
        return self._declare(
            self.transforms, Transform(str(name), int(arity)), "transform",
        )

    def declare_reason(self, name: str) -> Reason:
        return self._declare(self.reasons, Reason(str(name)), "reason")

    @staticmethod
    def _declare(table: dict, proposed: Any, what: str) -> Any:
        existing = table.get(proposed.name)
        if existing is None:
            table[proposed.name] = proposed
            return proposed
        if existing != proposed:
            raise ConcordanceRefusal(
                f"{what} {proposed.name!r} already declared as {existing!r}, "
                f"proposed {proposed!r}"
            )
        return existing

    def page(self, name: str) -> Page:
        """The declared page named ``name``; an undeclared name is refused."""
        page = self.pages.get(name)
        if page is None:
            raise ConcordanceRefusal(f"undeclared page {name!r}")
        return page


REGISTRY = Registry()
declare_page = REGISTRY.declare_page
declare_stage = REGISTRY.declare_stage
declare_transform = REGISTRY.declare_transform
declare_reason = REGISTRY.declare_reason

_SCOPE = RowFieldKind.SCOPE
_VALUE_ID = RowFieldKind.VALUE_ID
_NAME = RowFieldKind.NAME
_INDEX = RowFieldKind.INDEX
_LABEL = RowFieldKind.LABEL

#: One edge per (target cell, source cell) a ``Derived`` post named.  Row
#: ``(target_key, source_key, stage_name)``, each key ``(page, row,
#: column)`` as ``Ref.key`` spells it; scope-first by target so
#: ``edges_into`` is one ``scope_rows`` read.
EDGE_PAGE = declare_page(
    "concordance_edge",
    (RowField("target", _SCOPE), RowField("source", _LABEL),
     RowField("stage", _NAME)),
    bool, private=True, rows_level=IdentityLogLevel.FULL,
)
#: The same edges read from their source end: row ``(source_key,
#: edge_row)`` so ``edges_out_of`` is one ``scope_rows`` read.
DEPENDENTS_PAGE = declare_page(
    "concordance_dependents",
    (RowField("source", _SCOPE), RowField("edge_row", _LABEL)),
    bool, private=True, rows_level=IdentityLogLevel.FULL,
)
LIVE_DERIVATION = declare_page(
    "live_derivation",
    (RowField("target_page", _NAME), RowField("target_row", _LABEL)),
    DerivationSnapshot,
)
IMPORTED_CELL = declare_page(
    "concordance_imported_cell",
    (RowField("artifact_sha256", _SCOPE), RowField("source_key", _LABEL)),
    ImportedCell, private=True,
)
#: One row per ``Novel`` post: ``(target_key, minted_id)`` with fact
#: ``(transform, operands)``.  ``minted_id`` is None for a root row that
#: carried no ``NEW`` (an AST source span, say): the post is then an origin
#: edge only and mints nothing.
MINT_PAGE = declare_page(
    "concordance_mint",
    (RowField("target", _SCOPE), RowField("minted_id", _LABEL)),
    tuple, private=True,
)
#: One row per unsourced statement: ``(page_name, row, stage_name)`` with
#: the ``Reason`` as fact.  Both ``Unsourced`` posts and raw-primitive
#: writes land here while the latch is OPEN.
UNSOURCED_PAGE = declare_page(
    "concordance_unsourced",
    (RowField("page", _SCOPE), RowField("row", _LABEL),
     RowField("stage", _NAME)),
    Reason, private=True,
)
_PRIVATE_PAGE_NAMES = frozenset(
    page.name for page in (EDGE_PAGE, DEPENDENTS_PAGE, MINT_PAGE, UNSOURCED_PAGE)
)

#: Pages a copy-on-read fork (``fork_read_scope``) never reads through to its
#: origin, besides the book's private pages: a scope's origin is its own fact
#: (``scope_origin``), and the planner's specialization rows are per COPY
#: (design section 7.1: a copy never carries a literal proven for another
#: caller).  This is the set ``fork_read_scope`` skipped when it copied.
READ_THROUGH_EXCLUDED_PAGES: frozenset[str] = frozenset({
    "scope_origin", "planner_specialization", "planner_tensor_descriptor",
})

#: The stage recorded for a raw write when the book knows no better
#: (``IdentityBook.active_stage`` unset), and the reason every such write
#: is tagged with.
RAW_STAGE = declare_stage("raw_primitive")
RAW_PRIMITIVE = declare_reason("raw_primitive")


def _validate_row(page: Page, row: tuple, *, allow_new: bool) -> tuple[int, ...]:
    """Check ``row`` against ``page.row_fields``; return the NEW positions."""

    row_fields: tuple[RowField, ...] = page.row_fields
    # Keep tuple subclasses accepted, as ``isinstance(row, tuple)`` did, but
    # state the exact-type fast path first.  The compiler can settle that
    # clause from the authored ``row: tuple`` schema; Python still evaluates
    # the second clause for a tuple subclass and preserves the runtime rule.
    if type(row) is not tuple and not isinstance(row, tuple):
        raise ConcordanceRefusal(
            f"{page.name}: row must be a tuple, got {type(row).__name__}"
        )
    if len(row) != len(row_fields):
        raise ConcordanceRefusal(
            f"{page.name}: row {row!r} has {len(row)} element(s); the page "
            f"declares {len(row_fields)}: "
            f"{tuple(item.name for item in row_fields)!r}"
        )
    new_positions: list[int] = []
    position: int
    declared: RowField
    item: Any
    for position, (declared, item) in enumerate(zip(row_fields, row)):
        if item is NEW:
            if declared.kind is not RowFieldKind.VALUE_ID or not allow_new:
                raise ConcordanceRefusal(
                    f"{page.name}: NEW is admissible only in a VALUE_ID field "
                    f"of a Novel post (field {declared.name!r} at {position})"
                )
            new_positions.append(position)
            continue
        if not declared.admits(item):
            raise ConcordanceRefusal(
                f"{page.name}: row element {position} ({declared.name!r}, "
                f"{declared.kind.name}) does not admit {item!r}"
            )
    return tuple(new_positions)


def _validate_fact(page: Page, fact: Any) -> None:
    if isinstance(fact, Unresolved):
        return
    if not isinstance(fact, page.fact_type):
        raise ConcordanceRefusal(
            f"{page.name}: fact {fact!r} is not a {page.fact_type!r}"
        )


#: The tables a spilled page does not hold (``identity_spill.SPILLABLE_ATTRS``).
_SPILLABLE_TABLES = frozenset((
    "cells", "stamps", "columns", "scopes", "row_columns", "column_positions",
))


@dataclass
class IdentityPage:
    """One pipeline stage's row (identity) x column (round) table of facts.

    COPY-ON-READ SCOPES.  A scope forked with ``fork_read_scope`` holds no
    copied rows: ``IdentityBook.read_through[forked] == (origin, stamp)``
    (``IdentityBook.register_read_through``; the fork's ``scope_origin`` fact
    says ``copy_on_read=True``).  A row under the fork that has no cell of its
    own READS as the origin's row did when the fork was made -- the origin's
    latest cell stamped before ``stamp`` (the snapshot the eager copy was), or,
    if the origin is itself a fork, what it reads as, to the root.  Every read
    api answers through that chain: ``latest``, ``latest_column``, ``history``
    (``((0, fact),)``, the one cell the eager copy held), ``spans``, ``cell``,
    ``stamp_at``, ``scope_rows``, ``scope_row_count``, ``alias_bindings``,
    ``IdentityBook.latest_ref`` / ``edges_into`` / ``stamp_of``.

    A WRITE under the fork (``set``, ``revise``, ``IdentityBook.post``, or a
    ``Ref`` to the unwritten row named as a source) first MATERIALISES the
    row: a cell at column 0 holding that fact, DERIVED from the origin's cell
    at stage ``read_scope_fork`` (same fact and edge as the eager copy had,
    stamped one reading before the fork's ``scope_origin`` cell as the copy
    was, ticking nothing), then the write lands as the next column.  Nothing is
    ever deleted; a fork's rows are never rewritten by the origin's later
    writes (the snapshot), and an in-place overwrite of an origin cell
    materialises its forks' rows first (``_stamp``).

    ``rows()``, ``cells``, ``stamps``, ``row_columns`` and ``scopes`` are the
    MATERIALISED book: an unwritten, unreferenced row of a fork is not a row
    -- it has no cell, no edge, no stamp -- until it is read through and
    written (or named as a source).  ``scope_rows(scope)`` is the resolved
    view (materialised rows and the origin's rows not shadowed).
    ``materialised_scope_rows(scope)`` is the book's own cells only; the
    unsourced detector and ``tools/view_identity_concordance.py`` read those,
    together with the ``scope_origin`` relation.
    """

    name: str
    cells: dict[tuple[Any, int], Any] = field(default_factory=dict)
    columns: list[int] = field(default_factory=list)
    #: Rows grouped by their first element, in first-recorded order, so the
    #: rows one scope owns (a function's table, a planning scope) are read
    #: from the page without scanning every cell.  Part of the page itself.
    scopes: dict[Any, dict[Any, None]] = field(default_factory=dict)
    #: The construction clock: one counter shared by every page of a book
    #: (``IdentityBook.page`` hands its own), ticked by each write, so the
    #: order of writes across pages is recorded, not inferred.  A page made
    #: outside a book keeps a clock of its own.
    clock: list[int] = field(default_factory=lambda: [0])
    #: ``(row, column)`` -> the clock reading when that cell was written.
    stamps: dict[tuple[Any, int], int] = field(default_factory=dict)
    #: The book this page belongs to (``IdentityBook.page`` sets it); a page
    #: made outside a book has none, so its writes cannot be tagged or
    #: latched.
    book: IdentityBook | None = field(
        default=None, repr=False, compare=False,
    )
    #: The page's own index of ``cells``, kept by ``_stamp`` (the one cell
    #: writer): each column's position in ``columns``, and each row's
    #: columns in that order.  ``history`` reads a row's cells through it
    #: instead of walking every column the page has ever had, which made
    #: every read O(all revisions on the page).  Same cells, same order.
    column_positions: dict[int, int] = field(
        default_factory=dict, repr=False, compare=False,
    )
    row_columns: dict[Any, list[int]] = field(
        default_factory=dict, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        # A page built with cells already in hand indexes them once here.
        if self.cells and not self.row_columns:
            self._reindex()
        from .live_tiles import _ACTIVE_ARENA, tile_identity_page
        arena = _ACTIVE_ARENA.get()
        if arena is not None:
            tile_identity_page(self, arena)

    def __setattr__(self, name, value):
        arena = self.__dict__.get("_live_tile_owner")
        if arena is not None and name in _SPILLABLE_TABLES:
            value = arena.manage(value)
        object.__setattr__(self, name, value)

    def __setstate__(self, state: dict) -> None:
        # A page pickled before the index existed is indexed on arrival.
        self.__dict__.update(state)
        if "row_columns" not in state or "column_positions" not in state:
            self._reindex()

    # ------------------------------------------------------------ cold spill
    # A page the book has spilled (``identity_spill``) has NO table
    # attributes: ``cells``, ``stamps``, ``columns``, ``scopes``,
    # ``row_columns`` and ``column_positions`` are deleted from the instance
    # and ``spill`` holds its ``Segment``.  Python calls ``__getattr__`` only
    # for a name the instance does not have, so a page in RAM pays nothing,
    # and the first read of a table of a spilled page -- through any api, or
    # directly -- brings the page back.

    def __getattr__(self, name: str) -> Any:
        if name in _SPILLABLE_TABLES:
            state = self.__dict__
            if state.get("spill") is not None:
                book = state.get("book")
                if book is not None:
                    from .identity_spill import PAGE

                    book._reload(self.name, (PAGE,), "access")
                    return state[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    @property
    def spilled(self) -> bool:
        """Whether this page's own tables are out of RAM (in the spill file)."""
        return self.__dict__.get("spill") is not None

    def cell_count(self) -> int:
        """The number of cells the page holds, spilled or not (a spilled page
        is counted from its segment, without reading it back; a private edge
        page counts the rows of owners that are away too)."""
        # ``spill`` and ``_detached`` are private dynamic page state; their
        # value-record contract is stated at the read boundary, as in
        # ``identity_spill._detached_ordinals``.
        from .identity_spill import Segment
        segment: Segment | None = self.__dict__.get("spill")
        count = (
            segment.cells if segment is not None else len(self.__dict__["cells"])
        )
        away: dict[str, Segment] | None = self.__dict__.get("_detached")
        if away:
            count += sum(item.cells for item in away.values())
        return count

    def row_count(self) -> int:
        """``len(self.rows())``, without reading back what is spilled."""
        from .identity_spill import Segment
        segment: Segment | None = self.__dict__.get("spill")
        count = (
            segment.rows if segment is not None
            else len(self.__dict__["row_columns"])
        )
        away: dict[str, Segment] | None = self.__dict__.get("_detached")
        if away:
            count += sum(item.rows for item in away.values())
        return count

    def __getstate__(self) -> dict:
        # A copy or a pickle of a page is of the whole page.
        if self.__dict__.get("_compressed_ram"):
            from .identity_compression import thaw_page

            thaw_page(self)
        state = self.__dict__
        book = state.get("book")
        if book is not None and (
            state.get("spill") is not None or state.get("_detached")
        ):
            book._reload_everything_of(self.name, "pickle")
        return {
            key: value for key, value in self.__dict__.items()
            if key not in ("spill", "_detached", "_borrow")
            and key != "_live_tile_owner"
        }

    def _reindex(self) -> None:
        if self.__dict__.get("_compressed_ram"):
            from .identity_compression import thaw_page

            thaw_page(self)
        self.column_positions = {
            column: position for position, column in enumerate(self.columns)
        }
        self.row_columns = {}
        for row, column in self.cells:
            if column not in self.column_positions:
                self.column_positions[column] = len(self.columns)
                self.columns.append(column)
            self.row_columns.setdefault(row, []).append(column)
        for columns in self.row_columns.values():
            columns.sort(key=self.column_positions.__getitem__)

    def _stamp(self, row: Any, column: int, fact: Any) -> None:
        """Write one cell at the current clock reading without ticking.

        ``IdentityBook.post`` writes a fact and its edges through this so
        they share one reading; it ticks the clock once afterwards.
        """
        if self.__dict__.get("_compressed_ram"):
            from .identity_compression import thaw_page

            thaw_page(self)
        positions = self.column_positions
        if len(positions) != len(self.columns):
            self._reindex()
        position = positions.get(column)
        if position is None:
            position = positions[column] = len(self.columns)
            self.columns.append(column)
        # ONE key tuple for the membership test and both tables: each extra
        # ``(row, column)`` was a tuple the size of the cell's own
        # bookkeeping, per cell of every page (the edge pages hold most of
        # the book).
        key = (row, column)
        book = self.book
        if key not in self.cells:
            columns = self.row_columns.setdefault(row, [])
            columns.append(column)
            if len(columns) > 1 and positions[columns[-2]] > position:
                columns.sort(key=positions.__getitem__)
        elif book is not None and book.__dict__.get("read_through_origins"):
            # An in-place overwrite of a cell some fork may read through to:
            # the fork's snapshot is taken (its row materialised) first.
            book._preserve_fork_snapshots(self, row, key, fact)
        self.cells[key] = fact
        stamp = None if book is None else book.__dict__.get("_stamp_override")
        self.stamps[key] = self.clock[0] if stamp is None else stamp
        if isinstance(row, tuple) and row:
            self.scopes.setdefault(row[0], {}).setdefault(row, None)

    def set(self, row: Any, column: int, fact: Any) -> None:
        """The raw primitive every pre-api write bottoms out in.

        On a book's page this is a write that names no source: while the
        book's latch is OPEN it is admitted and tagged
        ``Unsourced(RAW_PRIMITIVE)`` on the unsourced page in the same clock
        tick; once the latch is CLOSED it is refused.
        """
        book = self.book
        if book is not None:
            if book._spill is not None:
                book._ensure_owner_whole(self.name)
            if book.__dict__.get("read_through"):
                book._materialise_row(self, row)
            book._admit_raw_write(self, row)
        self._stamp(row, column, fact)
        if book is None or book.__dict__.get("_stamp_override") is None:
            self.clock[0] += 1

    # ------------------------------------------------- copy-on-read scopes
    def _fork_of_scope(self, scope: Any) -> tuple[Any, int] | None:
        """``(origin, stamp)`` when ``scope`` is a copy-on-read fork whose
        rows on THIS page read through to the origin, else None."""
        book: IdentityBook | None = self.book
        if book is None:
            return None
        forks: dict[Any, tuple[Any, int]] | None = (
            book.__dict__.get("read_through")
        )
        if not forks:
            return None
        fork: tuple[Any, int] | None = forks.get(scope)
        if fork is None:
            return None
        name = self.name
        if (
            name in READ_THROUGH_EXCLUDED_PAGES
            or name in book.registry.private_pages
        ):
            return None
        return fork

    def _reaches_origin(self, scope: Any, row: tuple) -> bool:
        """Whether the fork ``scope``'s declared projection lets a read of
        ``row`` pass on to its origin (see ``LiveScopeProjection``)."""
        projections: dict[Any, LiveScopeProjection] | None = (
            self.book.__dict__.get("scope_projections")
        )
        if not projections:
            return True
        projection: LiveScopeProjection | None = projections.get(scope)
        if projection is None or self.name not in projection.pages:
            return True
        if len(row) > 1 and isinstance(row[1], int):
            return projection.reaches(row[1])
        return True

    def _resolved(
        self, row: Any, asof: int | None = None, *, through: bool = False,
    ) -> tuple[Any, int] | None:
        """The ``(owner row, column)`` of the cell a read of ``row`` means.

        ``row``'s own latest cell; failing that (and only on a copy-on-read
        scope) the origin's row as the fork saw it -- cells stamped before
        the fork -- recursively.  ``asof`` limits the cells to those stamped
        before it.  ``through`` ignores ``row``'s own cells and starts at the
        origin hop.  None when no cell is reached.  The owner row differs
        from ``row`` exactly when the cell is virtual (a row read through)."""
        if len(self.column_positions) != len(self.columns):
            self._reindex()
        row_columns = self.row_columns
        stamps = self.stamps
        while True:
            columns = None if through else row_columns.get(row)
            through = False
            if columns:
                if asof is None:
                    return row, columns[-1]
                for column in reversed(columns):
                    if stamps[(row, column)] < asof:
                        return row, column
            fork = (
                self._fork_of_scope(row[0])
                if type(row) is tuple and row else None
            )
            if fork is None or not self._reaches_origin(row[0], row):
                return None
            origin, stamp = fork
            asof = stamp if asof is None or stamp < asof else asof
            row = (origin, *row[1:])

    def _virtual_fact(self, row: Any) -> Any:
        """The fact a copy-on-read row with no cell of its own reads as, or
        None (no origin cell, or the origin's row was withdrawn: the eager
        copy skipped a row whose fact was None)."""
        found = self._resolved(row, through=True)
        return None if found is None else self.cells[found]

    def cell(self, row: Any, column: int, default: Any = None) -> Any:
        """The fact of cell ``(row, column)``, resolved through a
        copy-on-read scope (a virtual row's one cell is column 0)."""
        key = (row, column)
        if key in self.cells:
            return self.cells[key]
        if column == 0 and not self.row_columns.get(row):
            fact = self._virtual_fact(row)
            if fact is not None:
                return fact
        return default

    def is_virtual(self, row: Any, column: int = 0) -> bool:
        """Whether ``(row, column)`` is a cell a copy-on-read read resolves
        but the book does not hold yet."""
        return (
            column == 0 and (row, column) not in self.cells
            and not self.row_columns.get(row)
            and self._virtual_fact(row) is not None
        )

    def stamp_at(self, row: Any, column: int) -> int:
        """The clock reading of cell ``(row, column)``; a virtual cell reads
        as its fork made it: one reading before the fork's ``scope_origin``
        cell, which is what its materialisation is stamped with
        (``KeyError`` if there is no such cell)."""
        key = (row, column)
        if key in self.cells:
            return self.stamps[key]
        if self.is_virtual(row, column):
            return self._fork_of_scope(row[0])[1] - 1
        raise KeyError(key)

    def holds(self, row: Any) -> bool:
        """Whether the book holds a cell of its own for ``row`` -- unlike
        ``latest`` / ``history``, which also answer for a row a copy-on-read
        scope has not written yet."""
        if len(self.column_positions) != len(self.columns):
            self._reindex()
        return bool(self.row_columns.get(row))

    def materialised_scope_rows(self, scope: Any) -> tuple[Any, ...]:
        """The rows of ``scope`` the book holds cells for, in recorded
        order.  For a scope that is not a copy-on-read fork this is
        ``scope_rows``; for a fork it is the rows written or referenced
        under it, without the origin's rows it reads through to."""
        return tuple(self.scopes.get(scope, ()))

    def scope_rows(self, scope: Any) -> tuple[Any, ...]:
        """Every row whose first element is ``scope``, in recorded order.

        On a copy-on-read fork (``fork_read_scope``) this is the RESOLVED
        view: the origin's rows as the fork saw them -- in the origin's order,
        as the eager copy listed them -- then the rows only the fork holds.
        A row the fork wrote shadows the origin's.  A row of the origin that
        reads as withdrawn (fact None), or that the fork's projection does
        not hold, is not listed, as the eager copy did not copy it."""
        own = self.scopes.get(scope)
        fork = self._fork_of_scope(scope)
        if fork is None:
            return tuple(own) if own else ()
        resolved: dict[Any, None] = {}
        row_columns = self.row_columns
        for origin_row in self.scope_rows(fork[0]):
            row = (scope, *origin_row[1:])
            if row_columns.get(row) or self._virtual_fact(row) is not None:
                resolved[row] = None
        if own:
            resolved.update(dict.fromkeys(own))
        return tuple(resolved)

    def scope_row_count(self, scope: Any) -> int:
        """How many rows ``scope`` owns (``len(scope_rows(scope))``); a plain
        scope is counted without materializing them."""
        fork = self._fork_of_scope(scope)
        if fork is None:
            return len(self.scopes.get(scope, ()))
        counts = self.book.__dict__.setdefault("_read_through_counts", {})
        key = (self.name, scope)
        virtual = counts.get(key)
        if virtual is None:
            # The origin's rows as the fork saw them are fixed once the fork
            # is made (their cells are stamped before it; a later write in
            # place materialises the fork's row first), so they are counted
            # once.
            virtual = counts[key] = sum(
                1 for origin_row in self.scope_rows(fork[0])
                if self._virtual_fact((scope, *origin_row[1:])) is not None
            )
        return virtual + sum(
            1 for row in self.scopes.get(scope, ())
            if self._virtual_fact(row) is None
        )

    def revise(self, row: Any, fact: Any) -> Any:
        """Append ``fact`` as ``row``'s next revision and return it.

        For a row whose fact legitimately grows (a record descriptor gaining
        the fields another callee projects); the full history stays on the
        page.  Identity facts that must never change use ``concord``.
        """
        entries = self.history(row)
        self.set(row, entries[-1][0] + 1 if entries else 0, fact)
        return fact

    def _missed(self, row: Any) -> None:
        """A read found no cell for ``row``.  Under a scope whose declared
        projection excludes the row's node, that is not an absent row but a
        read of a row the scope does not hold: post the ``Unresolved`` receipt
        and refuse (``ProjectedScopeRead``).  Every other miss is a plain
        miss."""

        book = self.book
        if book is None:
            return
        projections = book.__dict__.get("scope_projections")
        if not projections or type(row) is not tuple or len(row) < 2:
            return
        projection = projections.get(row[0])
        if (
            projection is None
            or self.name not in projection.pages
            or not projection.excludes(row[1])
        ):
            return
        book._refuse_projected_read(self, row)

    def latest(self, row: Any, default: Any = None) -> Any:
        """Return the most recently recorded fact for ``row``.

        The last entry of ``history(row)``, read through the row index
        directly: ``row_columns[row]`` already holds the row's columns in
        recorded order, so the answer is its last column's cell.  Building
        the whole history tuple to read one cell was the hottest read on the
        book (``_tensor_descriptor`` asks it per node per query).
        """
        if len(self.column_positions) != len(self.columns):
            self._reindex()
        columns = self.row_columns.get(row)
        if not columns:
            if type(row) is tuple and row and self._fork_of_scope(row[0]):
                fact = self._virtual_fact(row)
                if fact is not None:
                    return fact
            self._missed(row)
            return default
        return self.cells[(row, columns[-1])]

    def latest_column(self, row: Any) -> int | None:
        """The column of ``row``'s most recently recorded cell, or None --
        ``history(row)[-1][0]`` without materializing the history.  A row a
        copy-on-read scope reads through to is at column 0, as its copy was."""
        if len(self.column_positions) != len(self.columns):
            self._reindex()
        columns = self.row_columns.get(row)
        if not columns:
            if type(row) is tuple and row and self._fork_of_scope(row[0]):
                if self._virtual_fact(row) is not None:
                    return 0
            self._missed(row)
            return None
        return columns[-1]

    def concord(self, row: Any, fact: Any) -> Any:
        """Commit ``fact`` for ``row`` and return the committed fact.

        The first statement owns the row; a later stage may repeat it but may
        not replace it, so a different proposal is a disagreement, never a
        silent overwrite.  Callers use the returned fact as the decision.
        """
        incumbent = self.latest(row)
        if incumbent is None:
            self.set(row, 0, fact)
            return fact
        if incumbent != fact:
            raise ValueError(
                f"{self.name} disagreement for {row!r}: "
                f"recorded={incumbent!r}, proposed={fact!r}"
            )
        return incumbent

    def mapping(self, scope: Any, *, stage: Any = None) -> "PageMapping":
        """This page's rows under ``scope`` as a mutable mapping.

        ``mapping[key]`` is row ``(scope, key)``'s latest fact; assignment
        is a revision and deletion a ``None`` revision, so a pass that keeps
        its working state here keeps it on the book with its full history.
        On a book's page every revision is posted DERIVED from the cell it
        revises, under ``stage`` (the book's active stage when none is
        named); a key with no row cannot be revised through the mapping.
        """
        return PageMapping(self, scope, stage=stage)

    def bind_alias(self, scope: Any, alias: int, resident: int) -> None:
        """Concord one planning value occurrence with its resident identity.

        Rebinding a row appends a new column so the page remains both the live
        planning authority and the history of every decision it supplied.
        """
        row = (scope, int(alias))
        entries = self.history(row)
        column = entries[-1][0] + 1 if entries else 0
        self.set(row, column, int(resident))

    def alias_bindings(self, scope: Any) -> dict[int, int]:
        """Materialize the latest alias facts owned by one planning scope."""
        return {
            int(row[1]): int(self.latest(row))
            for row in self.scope_rows(scope)
            if (
                isinstance(row, tuple)
                and len(row) == 2
                and row[0] == scope
                and self.latest(row) is not None
            )
        }

    def resolve_alias(self, scope: Any, value_id: int) -> int:
        """Resolve one value through this page's current planning facts."""
        current = int(value_id)
        path: list[int] = []
        while True:
            target = self.latest((scope, current))
            if target is None or int(target) == current:
                return current
            if current in path:
                raise ValueError(
                    f"cyclic planning identity concordance for {scope!r}: "
                    f"{tuple((*path, current))}"
                )
            path.append(current)
            current = int(target)

    def rows(self) -> tuple[Any, ...]:
        return tuple(dict.fromkeys(row for row, _ in self.cells))

    def history(self, row: Any) -> tuple[tuple[int, Any], ...]:
        """This row's fact at every column it was recorded on, in order."""
        if len(self.column_positions) != len(self.columns):
            self._reindex()
        cells = self.cells
        columns = self.row_columns.get(row)
        if not columns:
            if type(row) is tuple and row and self._fork_of_scope(row[0]):
                fact = self._virtual_fact(row)
                if fact is not None:
                    # The one cell the eager copy of this row held.
                    return ((0, fact),)
            return ()
        return tuple((column, cells[(row, column)]) for column in columns)

    def spans(self, row: Any) -> tuple[tuple[int, int, Any], ...]:
        """This row's history collapsed to contiguous (start, end, fact) runs."""
        entries = self.history(row)
        if not entries:
            return ()
        runs: list[tuple[int, int, Any]] = []
        start_column, current_fact = entries[0]
        end_column = start_column
        for column, fact in entries[1:]:
            if fact != current_fact:
                runs.append((start_column, end_column, current_fact))
                start_column, current_fact = column, fact
            end_column = column
        runs.append((start_column, end_column, current_fact))
        return tuple(runs)

    def oscillating_rows(
        self, key: Any = None, *, old_key: Any = None,
    ) -> dict[Any, tuple[tuple[int, int, Any], ...]]:
        """Rows that left a value and later came back to it -- a round-trip,
        never a settle, and the exact shape a round-boundary-only snapshot
        cannot see (the net effect across the trip is zero).

        ``key`` extracts the comparable "resulting" value from a fact
        (default: the fact itself).  For a fact that bundles its own
        transition -- ``(old, new, ...)``, as a mutation-log style page does
        -- pass ``key=lambda fact: fact[1]``.

        That alone still misses the most common real case: fact A moves a
        value from its UNRECORDED starting point to X, fact B moves it from
        X back to that same starting point.  The visited-value sequence is
        genuinely [start, X, start] -- a real round-trip -- but only X and
        start-as-B's-new ever get compared unless the implicit start is
        counted too.  Pass ``old_key`` (extracting the "before" side of the
        SAME fact shape, e.g. ``lambda fact: fact[0]``) to prepend that
        first recorded starting value to the sequence before checking.
        """
        project = key or (lambda fact: fact)
        found = {}
        for row in self.rows():
            runs = self.spans(row)
            facts = [project(fact) for _, _, fact in runs]
            if old_key is not None and runs:
                facts = [old_key(runs[0][2]), *facts]
            if len(set(facts)) < len(facts):
                found[row] = runs
        return found


class PageMapping(MutableMapping):
    """Rows ``(scope, key)`` of one page, read and written as a mapping."""

    def __init__(
        self, page: IdentityPage, scope: Any, stage: Any = None,
    ) -> None:
        self.page = page
        self.scope = scope
        self.stage = stage

    def __getitem__(self, key: Any) -> Any:
        try:
            fact = self.page.latest((self.scope, key))
        except TypeError:  # an unhashable key names no row
            raise KeyError(key) from None
        if fact is None:
            raise KeyError(key)
        return fact

    def _revise(self, row: Any, fact: Any) -> None:
        """``row``'s next revision.  On a book's page it is posted through
        the book DERIVED from the cell it revises (the row's latest cell);
        a row with no cell is not revised here but registered by the writer
        that read it, with its cells.  A standalone page keeps its own
        ``revise``."""

        book = self.page.book
        if book is None:
            self.page.revise(row, fact)
            return
        page = book.registry.page(self.page.name)
        incumbent = book.latest_ref(page, row)
        if incumbent is None:
            raise ValueError(
                f"{self.page.name} row {row!r}: no incumbent to revise; a "
                "new row is posted by its writer with the cells it was read "
                "from"
            )
        stage = self.stage
        if stage is None:
            stage = (
                book.active_stage if book.active_stage is not None
                else RAW_STAGE
            )
        book.post(
            page, row, fact, stage=stage,
            provenance=Derived((incumbent,)), mode=Mode.REVISE,
        )

    def __setitem__(self, key: Any, value: Any) -> None:
        if value is None:
            raise ValueError(
                f"{self.page.name}: None is the removal fact; delete the row"
            )
        row = (self.scope, key)
        if self.page.latest(row) != value:
            self._revise(row, value)

    def __delitem__(self, key: Any) -> None:
        if self.page.latest((self.scope, key)) is None:
            raise KeyError(key)
        self._revise((self.scope, key), None)

    def __iter__(self):
        for row in self.page.scope_rows(self.scope):
            if self.page.latest(row) is not None:
                yield row[1]

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def __repr__(self) -> str:
        return f"PageMapping({self.page.name!r}, {self.scope!r}, {dict(self)!r})"

    def __reduce__(self):
        return (dict, (dict(self),))


class _PageTable(dict):
    """``IdentityBook.pages``: name -> ``IdentityPage``.  A plain dict (it
    keeps the dict's own C methods) until a private edge page is partial."""

    def __init__(self, book: Any, items: Iterable = ()) -> None:
        super().__init__(items)
        self.book = book


class _PartialPageTable(_PageTable):
    """``pages`` while a private edge page (``identity_spill.PARTITIONED``)
    has rows spilled pages own away from RAM: it holds every other owner's
    rows, and handing it to a caller that can walk all of its cells would
    hand over a page with rows missing.  So ``[]``, ``get``, ``values`` and
    ``items`` read the missing rows back first.  The book's own code
    (``post``, ``edges_into``, ...) reaches the live pages through
    ``dict.get(self.pages, ...)`` and asks for exactly the rows it needs.
    ``IdentityBook._refresh_page_table`` swaps the class in and out."""

    def _whole(self, page: Any) -> Any:
        if page is not None and page.__dict__.get("_detached"):
            from .identity_spill import make_whole

            make_whole(self.book, page.name, "access")
        return page

    def __getitem__(self, name: Any) -> Any:
        return self._whole(dict.__getitem__(self, name))

    def get(self, name: Any, default: Any = None) -> Any:
        page = dict.get(self, name)
        return default if page is None else self._whole(page)

    def values(self):
        # (``_whole`` may swap this table back to a plain one: bind it first.)
        whole = self._whole
        return [whole(page) for page in tuple(dict.values(self))]

    def items(self):
        whole = self._whole
        return [(name, whole(page)) for name, page in tuple(dict.items(self))]


def mint_scope(label: Any, stage: Any = None) -> tuple[str, int]:
    """A fresh scope on the active compile's book (see ``IdentityBook``)."""
    return current_identity_book().mint_scope(label, stage)


class IdentityBook:
    """Every stage's page, so one identity's claim can be read across all
    of them -- the comparison none of them makes on its own."""

    def __init__(
        self, *, detached: bool = False, registry: Registry | None = None,
    ) -> None:
        self.pages: dict[str, IdentityPage] = _PageTable(self)
        #: Shared by every page: the book's construction clock.
        self.clock: list[int] = [0]
        #: The book's spill file and its bookkeeping (``identity_spill``);
        #: None until a page is spilled.
        self._spill: Any = None
        #: Created by ``current_identity_book`` because nothing had begun a
        #: compile.  A standalone transaction owns its own book instead of
        #: joining one of these, whose facts belong to no single program.
        self.detached = bool(detached)
        #: The declared vocabulary ``post`` accepts (pages, stages,
        #: transforms, reasons).  The module registry unless a caller hands
        #: its own.
        self.registry: Registry = REGISTRY if registry is None else registry
        #: OPEN admits ``Unsourced`` posts and raw primitive writes (tagging
        #: each on the unsourced page); CLOSED refuses both.  Closing it is
        #: the proof that no writer bypasses ``post``.
        import os

        # Always CLOSED: a fact is recorded only with the cells it was read
        # from.  The attribute remains for readers (the viewer); it is not
        # a switch.
        self.latch: Latch = Latch.CLOSED
        #: When a pass sets this, raw writes made while it is set are tagged
        #: with that stage instead of ``RAW_STAGE``.
        self.active_stage: Stage | None = None
        #: Forked read scopes that hold a projection of their source's rows
        #: (``fork_read_scope`` of a proper induced subgraph), scope ->
        #: ``LiveScopeProjection``.  Consulted only when a read MISSES, so a
        #: scope without one costs nothing.
        self.scope_projections: dict[Any, LiveScopeProjection] = {}
        self._reporting_projected_read = False
        #: Copy-on-read forks (``fork_read_scope``): forked scope ->
        #: ``(origin scope, stamp)``, the clock reading of the fork's
        #: ``scope_origin`` cell.  A row under the fork with no cell of its own
        #: reads as the origin's row did before ``stamp`` (see
        #: ``IdentityPage``).  Derived from the ``scope_origin`` facts that say
        #: ``copy_on_read=True``; the relation itself is on that page.
        self.read_through: dict[Any, tuple[Any, int]] = {}
        #: origin scope -> the forks reading through to it, so an in-place
        #: overwrite of an origin cell can take the forks' snapshots first.
        self.read_through_origins: dict[Any, list[Any]] = {}
        #: While a cell is being materialised or written: the clock reading
        #: to stamp it with (the fork's), and the ``(page, row)`` whose own
        #: write must not materialise again.
        self._stamp_override: int | None = None
        self._materialising: tuple | None = None

    def __getstate__(self) -> dict:
        # A copy or a pickle of a book is of the whole book: what is spilled
        # is read back first, and the copy owns no spill file.
        if self.__dict__.get("_spill") is not None:
            self.reload_all("pickle")
        state = dict(self.__dict__)
        state["_spill"] = None
        state.pop("_tile_store", None)  # the serialized copy owns no tile file
        return state

    def __setstate__(self, state: dict) -> None:
        # A book pickled before copy-on-read scopes existed has no forks.
        self.__dict__.update(state)
        self.__dict__.setdefault("read_through", {})
        self.__dict__.setdefault("read_through_origins", {})
        self.__dict__.setdefault("_stamp_override", None)
        self.__dict__.setdefault("_materialising", None)
        self.__dict__.setdefault("_spill", None)
        if not isinstance(self.pages, _PageTable):
            self.pages = _PageTable(self, self.pages)
        else:
            self.pages.book = self
        self._refresh_page_table()

    # ------------------------------------------------------------ cold spill
    def spill_pages(self, names: Iterable[str], **kwargs: Any) -> Any:
        """Write the named pages (and the edge rows they own) out of RAM; see
        ``identity_spill.spill_pages``."""
        from .identity_spill import spill_pages

        return spill_pages(self, names, **kwargs)

    def reload_all(self, reason: str = "explicit") -> int:
        """Read every spilled page and edge partition back into RAM."""
        from .identity_spill import PARTITIONED, make_whole, restore_all

        from .identity_compression import thaw_page

        for page in dict.values(self.pages):
            thaw_page(page)
        restored = restore_all(self, reason)
        for part in PARTITIONED:
            restored += make_whole(self, part, reason)
        return restored

    def _reload(self, owner: str, parts: Any, reason: str) -> int:
        from .identity_spill import restore

        return restore(self, {owner: parts}, reason)

    def _reload_everything_of(self, name: str, reason: str) -> None:
        from .identity_spill import PARTITIONED, make_whole, restore

        restore(self, {name: None}, reason)
        if name in PARTITIONED:
            make_whole(self, name, reason)

    def _refresh_page_table(self) -> None:
        """``pages`` is the guarding table exactly while a private edge page
        is partial (see ``_PartialPageTable``)."""
        from .identity_spill import PARTITIONED

        partial = False
        for name in PARTITIONED:
            page = dict.get(self.pages, name)
            if page is not None and page.__dict__.get("_detached"):
                partial = True
                break
        self.pages.__class__ = _PartialPageTable if partial else _PageTable

    def _ensure_owner_whole(self, name: str) -> None:
        """A page is about to be written, or named as a source: whatever of
        it (its tables, the edge rows it owns) is spilled comes back first,
        so no row is ever appended beside a spilled twin of itself."""
        spill = self._spill
        if spill is not None and name in spill.detached:
            from .identity_spill import restore

            restore(self, {name: None}, "post")

    def _ensure_parts(self, owner: str, parts: Iterable[str], reason: str) -> None:
        """A read of ``owner``'s edge rows on private page(s) ``parts``."""
        spill = self._spill
        if spill is not None:
            away = spill.detached.get(owner)
            if away:
                wanted = away.intersection(parts)
                if wanted:
                    from .identity_spill import restore

                    restore(self, {owner: wanted}, reason)

    def resident_cell_count(self) -> int:
        """How many cells are in RAM right now (a spilled page holds none):
        the proxy for what the book costs, which ``cell_count`` is not."""
        return sum(
            len(page.__dict__["cells"])
            for page in dict.values(self.pages) if "cells" in page.__dict__
            and not page.__dict__.get("_compressed_ram")
            and not page.__dict__.get("_live_tile_owner")
        )

    def compress_pages(self, names: Iterable[str], **kwargs: Any) -> Any:
        """Install resident compressed tables; ordinary reads stay tile-local."""
        from .identity_compression import compress_pages

        return compress_pages(self, names, **kwargs)

    def compressed_pages(self) -> tuple[str, ...]:
        return tuple(name for name, page in dict.items(self.pages)
                     if page.__dict__.get("_compressed_ram"))

    def spilled_pages(self) -> tuple[str, ...]:
        """Pages whose own tables are in the spill file, in page order."""
        return tuple(
            name for name, page in dict.items(self.pages)
            if page.__dict__.get("spill") is not None
        )

    def borrowed(self, name: str):
        """Read one page of the book without changing the book.

        For the scanners that walk every page (the unsourced detector, the
        log): a spilled page is read into RAM for the ``with`` block and
        dropped again after it -- the segment on disk is still the page
        (nothing wrote to it: the clock and the cell count are checked) -- so
        a scan of the whole book never holds the whole book.  Posts no
        receipt: a post would change the book under the scan."""
        from contextlib import contextmanager

        @contextmanager
        def borrow():
            page = dict.get(self.pages, name)
            segment = None if page is None else page.__dict__.get("spill")
            if segment is None:
                yield page
                return
            from .identity_spill import PAGE, _load_page

            spill = self._spill
            spill.borrows += 1
            _load_page(self, name)
            spill.detached[name].discard(PAGE)
            clock = self.clock[0]
            cells = len(page.__dict__["cells"])
            try:
                yield page
            finally:
                state = page.__dict__
                if (
                    self.clock[0] == clock and "cells" in state
                    and len(state["cells"]) == cells
                ):
                    for attribute in _SPILLABLE_TABLES:
                        del state[attribute]
                    state["spill"] = segment
                    spill.detached[name].add(PAGE)
                elif not spill.detached.get(name):
                    # Something wrote while it was on loan: it stays.
                    spill.detached.pop(name, None)

        return borrow()

    def page(self, name: Any) -> IdentityPage:
        """The page named ``name`` (a str or a declared ``Page``), created
        on first mention with this book's clock.  A private edge page with
        rows spilled is made whole first (``_page`` is the live page)."""
        page = self._page(name)
        if page.__dict__.get("_detached"):
            from .identity_spill import make_whole

            make_whole(self, page.name, "access")
        return page

    def _page(self, name: Any) -> IdentityPage:
        """The live page named ``name`` (see ``page``), as it is in RAM now."""
        if isinstance(name, Page):
            name = name.name
        page = dict.get(self.pages, name)
        if page is None:
            page = self.pages[name] = IdentityPage(
                name, clock=self.clock, book=self,
            )
        elif page.book is None:
            page.book = self
        return page

    # ------------------------------------------------------------- the api
    def derive_current(
        self, page: Page, row: tuple, recipe: Any, arguments: tuple = (),
        *, stage: Stage,
    ) -> Ref:
        """Refresh an explicitly live derived view before handing it out.

        ``recipe(read, *arguments)`` reads authoritative rows through ``read``.
        Those reads are both the dependency snapshot and the Derived sources;
        there is no independently maintained invalidation graph. A source
        revision invalidates this snapshot even when its value compares equal.
        Historical provenance reads through ``cell`` remain historical.
        """
        if (not callable(recipe)
                or "<" in getattr(recipe, "__qualname__", "")
                or not getattr(recipe, "__module__", None)):
            raise ConcordanceRefusal("live derivation requires an import-owned recipe")
        key = (page.name, row)
        snapshots = self.page(LIVE_DERIVATION)
        previous = snapshots.latest(key)
        if previous is not None and (
            previous.recipe != recipe or previous.arguments != arguments
            or previous.stage != stage
        ):
            raise ConcordanceRefusal(f"live derivation recipe disagreement: {key!r}")
        active = self.__dict__.setdefault("_active_derivations", set())
        if key in active:
            raise ConcordanceRefusal(f"cyclic live derivation: {key!r}")
        active.add(key)
        try:
            if previous is not None:
                current_reads = tuple(
                    (source_page, source_row, self.current_ref(source_page, source_row))
                    for source_page, source_row, _cell in previous.reads
                )
                if (current_reads == previous.reads
                        and self.latest_ref(page, row) == previous.target):
                    return previous.target
            reads = {}

            def read(source_page, source_row, default=None):
                cell = self.current_ref(source_page, source_row)
                reads[(source_page, source_row)] = cell
                return (default if cell is None else
                        self.page(source_page).cell(cell.row, cell.column))

            def read_ref(source_page, source_row):
                read(source_page, source_row)
                return reads[(source_page, source_row)]

            read.ref = read_ref

            fact = recipe(read, *arguments)
            cells = tuple(dict.fromkeys(cell for cell in reads.values() if cell is not None))
            if not cells:
                raise ConcordanceRefusal(f"live derivation has no authoritative inputs: {key!r}")
            incumbent = self.latest_ref(page, row)
            # Even an equal answer must name its new generation when inputs
            # changed. Downstream views then observe the revision causally.
            target = self.post(
                page, row, fact, stage=stage, provenance=Derived(cells),
                mode=Mode.CONCORD if incumbent is None else Mode.REVISE,
            )
            snapshot = DerivationSnapshot(
                recipe, tuple(arguments), stage, target,
                tuple((source_page, source_row, cell)
                      for (source_page, source_row), cell in reads.items()),
            )
            self.post(
                LIVE_DERIVATION, key, snapshot, stage=stage,
                provenance=Derived((target, *cells)),
                mode=Mode.CONCORD if previous is None else Mode.REVISE,
            )
            return target
        finally:
            active.remove(key)

    def current_ref(self, page: Page, row: tuple) -> Ref | None:
        """Read a live view after refreshing its transitive dependencies."""
        snapshots = dict.get(self.pages, LIVE_DERIVATION.name)
        snapshot = None if snapshots is None else snapshots.latest((page.name, row))
        if snapshot is None:
            return self.latest_ref(page, row)
        return self.derive_current(
            page, row, snapshot.recipe, snapshot.arguments, stage=snapshot.stage,
        )

    def stale_derivations(self) -> tuple:
        """Report stale current views without rewriting historical evidence."""
        snapshots = dict.get(self.pages, LIVE_DERIVATION.name)
        if snapshots is None:
            return ()
        stale = set()
        records = tuple((row, snapshots.latest(row)) for row in snapshots.rows())
        # A transitive dependent is stale even before its direct source has
        # been recomputed. Traverse to closure; every round adds a row.
        changed = True
        while changed:
            changed = False
            for row, snapshot in records:
                if row not in stale and (self.latest_ref(
                    snapshot.target.page, snapshot.target.row,
                ) != snapshot.target or any(
                    (page.name, source_row) in stale
                    or self.latest_ref(page, source_row) != cell
                    for page, source_row, cell in snapshot.reads
                )):
                    stale.add(row)
                    changed = True
        return tuple(row for row, _snapshot in records if row in stale)

    def settle_derivations(self) -> None:
        """Settle registered live views with the compiler's bounded guard."""
        from .bounded_fixed_point import BoundedFixedPoint

        snapshots = dict.get(self.pages, LIVE_DERIVATION.name)
        if snapshots is None:
            return
        guard = BoundedFixedPoint(
            "live-concordance-derivations", len(tuple(snapshots.rows())) + 1,
            book=self,
        )
        while True:
            stale = self.stale_derivations()
            for page_name, row in stale:
                self.current_ref(self.registry.page(page_name), row)
            if not guard.round(bool(stale), changed_ids=stale,
                               state=self.stale_derivations()):
                return

    def post(
        self,
        page: Page,
        row: tuple,
        fact: Any,
        *,
        stage: Stage,
        provenance: Derived | Novel | Unsourced,
        mode: Mode,
    ) -> Ref:
        """Write one fact with its provenance; the only sanctioned writer.

        Whatever refusal or disagreement leaves ``_post`` is a failure of
        THIS claim: the exception keeps its type and message and gains the
        cells the claim stood on (``FailureClaim``).  The neighborhood is not
        walked here -- a refusal is also control flow for some callers -- but
        by ``annotate_failure_with_neighborhood`` where the failure reports.
        """
        try:
            return self._post(
                page, row, fact, stage=stage, provenance=provenance,
                mode=mode,
            )
        except ValueError as error:
            _attach_failure_claim(
                self, error, page, row, fact, stage, provenance, mode,
            )
            raise

    def _post(
        self,
        page: Page,
        row: tuple,
        fact: Any,
        *,
        stage: Stage,
        provenance: Derived | Novel | Unsourced,
        mode: Mode,
    ) -> Ref:
        """The body of ``post`` (see there for the contract).

        ``Derived``: the fact cell plus one edge per source cell on the
        edge page and its reverse index, all at one clock reading.
        ``Novel``: the row's ``NEW`` is replaced by an id minted here and
        the mint edge (transform, operands) is written beside the fact; a
        row with no ``NEW`` is a root that mints nothing and gets only the
        origin edge.
        ``Unsourced``: admitted only while the latch is OPEN, recorded on
        the unsourced page with its reason and ``stage``.

        ``Mode.CONCORD``: the first statement owns the row; a different
        fact is a disagreement (``ValueError``, as ``concord`` raises) and
        the same fact writes no cell but still records its edge.
        ``Mode.REVISE``: a ``Derived`` revision is admitted only if some
        source cell is stamped newer than the row's previous revision, or
        the revision derives from a different set of cells than the previous
        revision did (a different source is a cause).
        Every post ticks the shared clock exactly once.
        """
        registry = self.registry
        if not isinstance(page, Page) or registry.pages.get(page.name) != page:
            raise ConcordanceRefusal(f"post: undeclared page {page!r}")
        if not isinstance(stage, Stage) or registry.stages.get(stage.name) != stage:
            raise ConcordanceRefusal(f"post: undeclared stage {stage!r}")
        if not isinstance(mode, Mode):
            raise ConcordanceRefusal(f"post: mode must be a Mode, got {mode!r}")
        novel = isinstance(provenance, Novel)
        new_positions = _validate_row(page, row, allow_new=novel)
        _validate_fact(page, fact)

        sources: tuple[tuple[Ref, int], ...] = ()
        if isinstance(provenance, Derived):
            if not provenance.cells:
                raise ConcordanceRefusal(
                    f"post {page.name} {row!r}: Derived names no source cell"
                )
            sources = tuple(
                (ref, self._source_stamp(ref)) for ref in provenance.cells
            )
        elif novel:
            transform = provenance.transform
            if (
                not isinstance(transform, Transform)
                or registry.transforms.get(transform.name) != transform
            ):
                raise ConcordanceRefusal(
                    f"post {page.name}: undeclared transform {transform!r}"
                )
            if len(provenance.operands) != transform.arity:
                raise ConcordanceRefusal(
                    f"post {page.name}: transform {transform.name!r} takes "
                    f"{transform.arity} operand(s), got {len(provenance.operands)}"
                )
            # One NEW: the book mints the id.  No NEW: a root (an AST source
            # span row, say) that has no minted id; the post writes only its
            # origin edge and mints nothing.
            if len(new_positions) > 1:
                raise ConcordanceRefusal(
                    f"post {page.name} {row!r}: a Novel row carries at most one "
                    f"NEW, found {len(new_positions)}"
                )
            for operand in provenance.operands:
                self._source_stamp(operand)
        elif isinstance(provenance, Unsourced):
            # Not a provenance: a record names what it was read from or it
            # is not a record.  ``Unsourced`` survives only as the read type
            # of rows older books tagged.
            raise ConcordanceRefusal(
                f"post {page.name} {row!r}: Unsourced({provenance.reason.name})"
                " is not a provenance; derive the fact from the cell it was "
                "read from, or mint it NOVEL from its operands"
            )
        else:
            raise ConcordanceRefusal(
                f"post {page.name}: provenance must be Derived or Novel, "
                f"got {provenance!r}"
            )

        if self._spill is not None:
            self._ensure_owner_whole(page.name)
        target_page = self._page(page.name)
        minted: int | None = None
        if novel and new_positions:
            # Direct ``GLOBAL_MONOTONIC_IDS.mint()`` calls still exist
            # elsewhere; sharing the source keeps this id disjoint from them.
            minted = GLOBAL_MONOTONIC_IDS.mint()
            position = new_positions[0]
            row = row[:position] + (minted,) + row[position + 1:]

        if self.__dict__.get("read_through"):
            # A write under a copy-on-read fork lands on the row the eager
            # copy would have held: take that copy first.
            self._materialise_row(target_page, row)
        if self.__dict__.get("_materialising") == (target_page.name, row):
            # (This post IS that copy: the row has no cell of its own yet, and
            # its virtual history is the very fact being written.)
            entries = ()
        else:
            entries = target_page.history(row)
        if mode is Mode.CONCORD:
            if entries:
                column, incumbent = entries[-1]
                if incumbent != fact:
                    raise ValueError(
                        f"{page.name} disagreement for {row!r}: "
                        f"recorded={incumbent!r}, proposed={fact!r}"
                    )
                write_cell = False
                if sources:
                    # The same fact from the same cells at the same stage,
                    # again: every edge this post would write already exists
                    # (edge rows are keyed by target, source and stage), so
                    # the post would only re-stamp private edge rows and tick
                    # the clock.  No reader consults an edge row's stamp
                    # (REVISE and the selection rules compare FACT cells'
                    # stamps), and skipping a tick keeps every written cell's
                    # relative order, so the book reads the same afterwards.
                    # ``publish_program_abi_graph_identities`` re-posts every
                    # declared field's shape transformation on each of its
                    # runs; this was a third of the orbital specialization
                    # stage.
                    target_key = Ref(page, row, column).key
                    edge_cells = self._page(EDGE_PAGE.name).cells
                    if all(
                        ((target_key, source_ref.key, stage.name), 0)
                        in edge_cells
                        for source_ref, _ in sources
                    ):
                        return Ref(page, row, column)
            else:
                column, write_cell = 0, True
        else:
            if entries and sources:
                previous = max(
                    target_page.stamps[(row, column)] for column, _ in entries
                )
                # A revision has a cause when a source cell changed since the
                # previous revision, OR when it derives from a different set of
                # cells than the previous revision did: a second field write
                # derives from a second assignment's cells, which were posted
                # at ingestion (older than the first revision) yet are a new
                # source.  Same cells, none changed: no cause, refused.
                previous_ref = Ref(page, row, entries[-1][0])
                previous_sources = {
                    source.key for source, _stage in self.edges_into(previous_ref)
                }
                proposed_sources = {ref.key for ref, _stamp in sources}
                if (
                    not any(stamp > previous for _, stamp in sources)
                    and proposed_sources == previous_sources
                ):
                    raise ConcordanceRefusal(
                        f"post {page.name} {row!r}: REVISE without a changed "
                        f"source; every Derived cell is stamped at or before "
                        f"the row's previous revision ({previous}) and the "
                        f"cell set is the previous revision's"
                    )
            column = entries[-1][0] + 1 if entries else 0
            write_cell = True

        if write_cell:
            target_page._stamp(row, column, fact)
        target = Ref(page, row, column)
        target_key = target.key
        if sources:
            edge_page = self._page(EDGE_PAGE.name)
            dependents = self._page(DEPENDENTS_PAGE.name)
            for source_ref, _ in sources:
                # One key tuple for the edge row and the dependents row
                # (``Ref.key`` builds a new tuple per call).
                source_key = source_ref.key
                edge_row = (target_key, source_key, stage.name)
                edge_page._stamp(edge_row, 0, True)
                dependents._stamp((source_key, edge_row), 0, True)
        elif novel:
            self._page(MINT_PAGE.name)._stamp(
                (target_key, minted), 0,
                (provenance.transform, tuple(provenance.operands)),
            )
        else:
            self._page(UNSOURCED_PAGE.name)._stamp(
                (page.name, row, stage.name), 0, provenance.reason,
            )
        if self.__dict__.get("_stamp_override") is None:
            # (A materialised copy is written at its fork's reading and
            # ticks nothing: the order of every other write is unchanged.)
            self.clock[0] += 1
        return target

    def _source_stamp(self, ref: Any) -> int:
        """The clock reading of an existing cell named by ``ref``; a Ref to
        no cell (or to an undeclared page) is refused."""
        if not isinstance(ref, Ref):
            raise ConcordanceRefusal(f"source must be a Ref, got {ref!r}")
        if self.registry.pages.get(ref.page.name) != ref.page:
            raise ConcordanceRefusal(f"source names undeclared page {ref.page!r}")
        if self._spill is not None:
            self._ensure_owner_whole(ref.page.name)
        page = dict.get(self.pages, ref.page.name)
        if (
            page is not None and self.__dict__.get("read_through")
            and (ref.row, ref.column) not in page.cells
            and page.is_virtual(ref.row, ref.column)
        ):
            # A cell a copy-on-read scope reads through to is named as a
            # source: it exists from here on (with its edge to the origin).
            self._materialise_row(page, ref.row)
        if page is None or (ref.row, ref.column) not in page.cells:
            raise ConcordanceRefusal(f"source cell does not exist: {ref!r}")
        if ref.page == IMPORTED_CELL:
            imported = page.cells[(ref.row, ref.column)]
            imported.owner._source_stamp(imported.source)
            source_page = imported.owner.page(imported.source.page)
            if source_page.cell(imported.source.row, imported.source.column) != imported.fact:
                raise ConcordanceRefusal(f"imported source changed: {ref!r}")
        return page.stamps[(ref.row, ref.column)]

    def import_cell(self, owner, source: Ref, artifact_sha256: str) -> Ref:
        """Transport a proof, not a foreign row disguised as a local Ref.

        This is a book-owned cross-book edge, like the ordinary edge page.
        It neither mints a root nor asserts a new physical value. Accepted
        dependency loading supplies the digest it verified and retains the
        source book. Downstream Derived posts name this local edge receipt.
        """
        if owner is self:
            self._source_stamp(source)
            return source
        if (not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64
                or any(char not in "0123456789abcdef" for char in artifact_sha256)):
            raise ConcordanceRefusal("imported proof requires its verified artifact digest")
        owner._source_stamp(source)
        fact = owner.page(source.page).cell(source.row, source.column)
        row = (artifact_sha256, source.key)
        page = self.page(IMPORTED_CELL)
        previous = self.latest_ref(IMPORTED_CELL, row)
        receipt = ImportedCell(source, fact, owner)
        if previous is not None:
            self._source_stamp(previous)
            if page.cell(row, previous.column) != receipt:
                raise ConcordanceRefusal(f"artifact proof disagreement: {row!r}")
            return previous
        page._stamp(row, 0, receipt)
        self.clock[0] += 1
        return Ref(IMPORTED_CELL, row, 0)

    def _admit_raw_write(self, page: IdentityPage, row: Any) -> None:
        """Refuse a write made through a raw primitive rather than
        ``post``: a page is written only by the book, from a fact's
        sources."""
        raise ConcordanceRefusal(
            f"raw write refused: page {page.name!r} row {row!r}; write it "
            "through IdentityBook.post with the cells the fact was read from"
        )

    # -------------------------------------------------------------- reading
    def _ref_from_key(self, key: tuple) -> Ref:
        page_name, row, column = key
        return Ref(self.registry.page(page_name), row, column)

    def register_scope_projection(
        self, scope: Any, pages: Iterable[str], excluded: Iterable[Any] = (),
        *, universe: Iterable[Any] | None = None, held: Iterable[Any] = (),
        inherited: LiveScopeProjection | None = None, restricts: bool = False,
    ) -> None:
        """Hold the read-time form of ``scope``'s declared projection: reads
        of ``pages`` rows keyed by a node in ``excluded``, or by a node of
        ``universe`` that is not in ``held``, or excluded by the ``inherited``
        projection, are refused.

        ``universe`` is interned on the book: every region cut from one
        source graph states the same node set, and they share one object."""

        shared = frozenset()
        if universe is not None:
            universe = frozenset(universe)
            interned = self.__dict__.setdefault("_node_universes", {})
            shared = interned.setdefault(universe, universe)
        self.scope_projections[scope] = LiveScopeProjection(
            frozenset(pages), frozenset(excluded), shared, frozenset(held),
            inherited, restricts,
        )

    # ------------------------------------------------------ copy-on-read
    def register_read_through(self, forked: Any, origin: Any, cell: Ref) -> None:
        """Record that ``forked`` is a copy-on-read fork of ``origin``.

        ``cell`` is the fork's ``scope_origin`` cell (posted with
        ``ScopeFork(..., copy_on_read=True)``): its clock reading is the
        moment the fork was taken.  The fork posts no copied rows; a read of a
        row it has no cell for resolves through ``origin`` as of that reading
        (``IdentityPage``), and a write materialises the row first
        (``_materialise_row``)."""
        stamp = self._source_stamp(cell)
        self.read_through[forked] = (origin, stamp)
        self.read_through_origins.setdefault(origin, []).append(forked)

    def _virtual_edges_into(
        self, page: IdentityPage, ref: Ref,
    ) -> tuple[tuple[Ref, Stage], ...]:
        """The edge a not-yet-written read-through cell will have: from the
        origin's cell (as of the fork) at stage ``read_scope_fork``."""
        from .concordance_declarations import READ_SCOPE_FORK

        origin, stamp = page._fork_of_scope(ref.row[0])
        origin_row = (origin, *ref.row[1:])
        found = page._resolved(origin_row, stamp)
        declared = self.registry.pages.get(page.name)
        if found is None or declared is None:
            return ()
        owner, column = found
        return ((
            Ref(declared, origin_row, column if owner == origin_row else 0),
            READ_SCOPE_FORK,
        ),)

    def _materialise_row(self, page: IdentityPage, row: Any) -> None:
        """Give a copy-on-read row the cell the eager copy would have held.

        A no-op unless ``row`` is under a copy-on-read fork, has no cell of
        its own, and reads through to a fact.  Otherwise the row is written
        exactly as ``fork_read_scope`` wrote its copies: a cell at column 0
        with the origin's fact, DERIVED from the origin's cell (as of the
        fork) at stage ``read_scope_fork``, mode CONCORD -- or, for a page
        the registry does not declare / a fact it would not admit, through
        the raw primitive, tagged under the latch as the eager copy was.  The
        cell is stamped with the fork's clock reading and ticks nothing."""
        if type(row) is not tuple or not row:
            return
        fork = page._fork_of_scope(row[0])
        if fork is None or page.row_columns.get(row):
            return
        key = (page.name, row)
        if self.__dict__.get("_materialising") == key:
            return
        origin, stamp = fork
        if not page._reaches_origin(row[0], row):
            return
        origin_row = (origin, *row[1:])
        found = page._resolved(origin_row, stamp)
        if found is None:
            return
        if found[0] != origin_row:
            # The origin is itself a fork and has not written this row: it
            # exists there (with its own edge) before this copy derives it.
            self._materialise_row(page, origin_row)
            found = page._resolved(origin_row, stamp)
            if found is None or found[0] != origin_row:
                return
        fact = page.cells[found]
        if fact is None:
            return
        from .concordance_declarations import READ_SCOPE_FORK

        declared = self.registry.pages.get(page.name)
        previous = self.__dict__.get("_materialising")
        previous_stamp = self.__dict__.get("_stamp_override")
        self._materialising = key
        # The eager copy was posted BEFORE the fork's ``scope_origin`` cell
        # (stamp ``stamp``): one reading earlier, so a source posted at the
        # fork is newer than the copy, as it was.
        self._stamp_override = stamp - 1
        try:
            if declared is None or not (
                isinstance(fact, Unresolved)
                or isinstance(fact, declared.fact_type)
            ):
                # (Tagged under the latch with the stage active now, as the
                # eager copy's raw write was.)
                page.set(row, 0, fact)
            else:
                self.post(
                    declared, row, fact, stage=READ_SCOPE_FORK,
                    provenance=Derived((Ref(declared, origin_row, found[1]),)),
                    mode=Mode.CONCORD,
                )
        finally:
            self._materialising = previous
            self._stamp_override = previous_stamp

    def _preserve_fork_snapshots(
        self, page: IdentityPage, row: Any, key: tuple, fact: Any,
    ) -> None:
        """``page`` is about to overwrite cell ``key`` in place with ``fact``;
        every copy-on-read fork of ``row``'s scope takes its snapshot of the
        row first, so the overwrite does not change what the fork reads."""
        old = page.cells[key]
        if old is fact:
            return
        try:
            if old == fact:
                return
        except Exception:
            pass
        if type(row) is not tuple or not row:
            return
        for forked in tuple(self.read_through_origins.get(row[0], ())):
            self._materialise_row(page, (forked, *row[1:]))

    def _refuse_projected_read(self, page: IdentityPage, row: tuple) -> None:
        """Post ``Unresolved(ROW_PROJECTED_OUT_OF_SCOPE)`` at ``row`` (DERIVED
        from the scope's ``scope_origin`` cell, which carries the projection)
        and raise ``ProjectedScopeRead``."""

        from .concordance_declarations import (
            READ_SCOPE_FORK, ROW_PROJECTED_OUT_OF_SCOPE, SCOPE_ORIGIN,
        )

        if getattr(self, "_reporting_projected_read", False):
            return
        self._reporting_projected_read = True
        receipt = "an Unresolved row is posted there"
        try:
            declared = self.registry.pages.get(page.name)
            origin = self.latest_ref(SCOPE_ORIGIN, (row[0],))
            if declared is not None and origin is not None:
                self.post(
                    declared, row, Unresolved(ROW_PROJECTED_OUT_OF_SCOPE),
                    stage=READ_SCOPE_FORK, provenance=Derived((origin,)),
                    mode=Mode.CONCORD,
                )
            else:
                receipt = "no Unresolved row could be posted (undeclared page or no scope_origin)"
        except ConcordanceRefusal as refusal:
            receipt = f"the Unresolved row was refused: {refusal}"
        finally:
            self._reporting_projected_read = False
        raise ProjectedScopeRead(
            f"{page.name} row {row!r}: node {row[1]!r} is not held by read "
            f"scope {row[0]!r} (its scope_origin row declares a projection "
            f"onto the copy's own nodes); {receipt} "
            f"({ROW_PROJECTED_OUT_OF_SCOPE.name})"
        )

    def latest_ref(self, page: Page, row: tuple) -> Ref | None:
        """The Ref of ``row``'s most recent cell on ``page``, or None."""
        stored = dict.get(self.pages, page.name)
        if stored is None:
            return None
        column = stored.latest_column(row)
        if column is None:
            return None
        return Ref(page, row, column)

    def stamp_of(self, ref: Ref) -> int:
        page = dict.get(self.pages, ref.page.name)
        if (
            page is not None and self.__dict__.get("read_through")
            and page.is_virtual(ref.row, ref.column)
        ):
            return page.stamp_at(ref.row, ref.column)
        return self._source_stamp(ref)

    def edges_into(self, ref: Ref) -> tuple[tuple[Ref, Stage], ...]:
        """Every (source cell, stage) ``ref``'s cell was derived from."""
        edge_page = dict.get(self.pages, EDGE_PAGE.name)
        if edge_page is None:
            return ()
        self._ensure_parts(ref.page.name, (EDGE_PAGE.name,), "edges_into")
        page = dict.get(self.pages, ref.page.name)
        if (
            page is not None and self.__dict__.get("read_through")
            and page.is_virtual(ref.row, ref.column)
        ):
            # The cell the book has not written yet has the one edge its
            # materialisation will write: to the origin's cell, at the fork.
            return self._virtual_edges_into(page, ref)
        return tuple(
            (self._ref_from_key(row[1]), self.registry.stages[row[2]])
            for row in edge_page.scope_rows(ref.key)
        )

    def edges_out_of(self, ref: Ref) -> tuple[tuple[Ref, Stage], ...]:
        """Every (target cell, stage) derived from ``ref``'s cell."""
        dependents = dict.get(self.pages, DEPENDENTS_PAGE.name)
        if dependents is None:
            return ()
        self._ensure_parts(ref.page.name, (DEPENDENTS_PAGE.name,), "edges_out_of")
        return tuple(
            (self._ref_from_key(row[1][0]), self.registry.stages[row[1][2]])
            for row in dependents.scope_rows(ref.key)
        )

    def mint_of(self, ref: Ref) -> tuple[Transform, tuple[Ref, ...]] | None:
        """The (transform, operands) a Novel post minted ``ref``'s row
        with, or None when the cell was not posted Novel."""
        mint_page = dict.get(self.pages, MINT_PAGE.name)
        if mint_page is None:
            return None
        self._ensure_parts(ref.page.name, (MINT_PAGE.name,), "mint_of")
        for row in mint_page.scope_rows(ref.key):
            fact = mint_page.latest(row)
            if fact is not None:
                return fact
        return None

    def neighborhood(
        self,
        seeds: Iterable[Ref],
        *,
        radius: int = 2,
        max_nodes: int = 40,
        fanout: int = 6,
    ) -> "CellNeighborhood":
        """The cells within ``radius`` edges of ``seeds``, mass diffused.

        The graph is the book's own: a cell is joined to the cells it was
        derived from (``edges_into``), the cells derived from it
        (``edges_out_of``) and, for a minted row, the operands it was minted
        from (``mint_of``).  Edges are read in both directions; nothing is
        inferred.  One unit of mass starts on the seeds and each hop passes
        half of a cell's mass, divided equally, to the neighbors it keeps --
        the ``fanout`` newest by clock stamp, so a hub with thousands of
        dependents contributes its most recent ones and reports how many it
        left out.  Collection stops at ``max_nodes`` cells.  The reach is
        deliberately modest: this is the evidence at a failure, not the book.
        """
        seed_refs = tuple(dict.fromkeys(seeds))
        nodes: dict[Ref, dict] = {}
        edges: dict[tuple, None] = {}
        truncated = False
        if not seed_refs:
            return CellNeighborhood((), (), (), radius, False)
        share = 1.0 / len(seed_refs)
        frontier: dict[Ref, float] = {}
        for ref in seed_refs:
            nodes[ref] = {"hop": 0, "mass": share, "seed": True, "omitted": 0}
            frontier[ref] = share
        for hop in range(1, int(radius) + 1):
            arrivals: dict[Ref, float] = {}
            for ref, mass in frontier.items():
                found: list[tuple[Ref, tuple]] = []
                try:
                    for other, stage in self.edges_into(ref):
                        found.append((other, (other, ref, stage.name)))
                    for other, stage in self.edges_out_of(ref):
                        found.append((other, (ref, other, stage.name)))
                    minted = self.mint_of(ref)
                    if minted is not None:
                        transform, operands = minted
                        for other in operands:
                            found.append((other, (
                                other, ref, f"mint:{transform.name}",
                            )))
                except Exception as error:  # a damaged book still reports
                    nodes[ref]["omitted"] = -1
                    nodes[ref]["error"] = f"{type(error).__name__}: {error}"
                    continue

                def newest(item):
                    try:
                        return self.stamp_of(item[0])
                    except Exception:
                        return -1

                found.sort(key=newest, reverse=True)
                kept, left = found[:int(fanout)], len(found) - int(fanout)
                nodes[ref]["omitted"] = max(0, left)
                if not kept:
                    continue
                push = mass * 0.5 / len(kept)
                for other, edge in kept:
                    if other not in nodes and len(nodes) >= int(max_nodes):
                        truncated = True
                        continue
                    edges[edge] = None
                    if other in nodes:
                        nodes[other]["mass"] += push
                    else:
                        nodes[other] = {
                            "hop": hop, "mass": push, "seed": False,
                            "omitted": 0,
                        }
                        arrivals[other] = push
            frontier = arrivals
            if not frontier:
                break
        facts = {}
        for ref in nodes:
            stamp = None
            try:
                with self.borrowed(ref.page.name) as page:
                    facts[ref] = page.cells.get((ref.row, ref.column))
                stamp = self.stamp_of(ref)
            except Exception:
                pass
            nodes[ref]["stamp"] = stamp
        return CellNeighborhood(
            seed_refs,
            tuple(
                NeighborhoodNode(
                    ref=ref, hop=info["hop"], mass=info["mass"],
                    seed=info["seed"], omitted=info["omitted"],
                    stamp=info.get("stamp"),
                    fact=_truncate_fact(repr(facts.get(ref))),
                    error=info.get("error"),
                )
                for ref, info in nodes.items()
            ),
            tuple(edges), int(radius), truncated,
        )

    def project_cells(self, roots: Iterable[Ref]) -> "IdentityBook":
        """Transport the exact predecessor closure of acceptance cells.

        A persisted compiler unit carried ReturnBlock.source_cell but not
        that ingestion cell, so post_control_program refused the first
        Derived post. This is a subgraph of this book, with unchanged keys,
        facts, stamps, mint and edge rows; it is not a reconstructed history.
        The complete diagnostic book remains its separately stowed owner.
        """
        projected = IdentityBook(registry=self.registry)
        selected: dict[str, dict[tuple, Any]] = {}
        stamps: dict[str, dict[tuple, int]] = {}
        pending = list(roots)
        visited = set()

        def copy_cell(name, row, column):
            page = self.pages[name]
            key = (row, column)
            selected.setdefault(name, {})[key] = page.cells[key]
            stamps.setdefault(name, {})[key] = page.stamps[key]

        while pending:
            ref = pending.pop()
            if ref.key in visited:
                continue
            visited.add(ref.key)
            self._source_stamp(ref)  # materialise a virtual referenced cell
            copy_cell(ref.page.name, ref.row, ref.column)
            # A current derived view transports its recomputation contract,
            # not only the historical cells from which it was last computed.
            # An explicitly historical target remains historical.
            snapshot_cell = self.latest_ref(LIVE_DERIVATION, (ref.page.name, ref.row))
            if snapshot_cell is not None:
                snapshot = self.page(LIVE_DERIVATION).cell(snapshot_cell.row, snapshot_cell.column)
                if snapshot.target == ref:
                    pending.append(snapshot_cell)
            for source, stage in self.edges_into(ref):
                edge_row = (ref.key, source.key, stage.name)
                copy_cell(EDGE_PAGE.name, edge_row, 0)
                copy_cell(DEPENDENTS_PAGE.name, (source.key, edge_row), 0)
                pending.append(source)
            mint_page = dict.get(self.pages, MINT_PAGE.name)
            if mint_page is not None:
                self._ensure_parts(ref.page.name, (MINT_PAGE.name,), "unit_cell_projection")
                for row in mint_page.scope_rows(ref.key):
                    for column, fact in mint_page.history(row):
                        copy_cell(MINT_PAGE.name, row, column)
                        pending.extend(fact[1])
            unsourced_page = dict.get(self.pages, UNSOURCED_PAGE.name)
            if unsourced_page is not None:
                self._ensure_parts(ref.page.name, (UNSOURCED_PAGE.name,), "unit_cell_projection")
                for row in unsourced_page.scope_rows(ref.page.name):
                    if row[1] == ref.row:
                        for column, _fact in unsourced_page.history(row):
                            copy_cell(UNSOURCED_PAGE.name, row, column)
        projected.clock[0] = self.clock[0]
        for name, cells in selected.items():
            rows = {row for row, _column in cells}
            columns = {column for _row, column in cells}
            scopes = {scope: dict.fromkeys(row for row in self.pages[name].scope_rows(scope)
                                          if row in rows)
                      for scope in {row[0] for row in rows}}
            projected.pages[name] = IdentityPage(name, cells=cells, stamps=stamps[name],
                columns=[column for column in self.pages[name].columns if column in columns],
                scopes=scopes, clock=projected.clock, book=projected)
        return projected

    def unsourced_rows(self) -> tuple[tuple[Any, Any, Reason, Stage], ...]:
        """Every unsourced statement as (page, row, reason, stage); ``page``
        is the declared Page when the name is registered, else the name."""
        unsourced = self.pages.get(UNSOURCED_PAGE.name)
        if unsourced is None:
            return ()
        pages = self.registry.pages
        stages = self.registry.stages
        return tuple(
            (pages.get(row[0], row[0]), row[1], unsourced.latest(row),
             stages.get(row[2], Stage(row[2])))
            for row in unsourced.rows()
        )

    def mint_scope(
        self, label: Any, stage: Stage | None = None,
    ) -> tuple[str, int]:
        """A fresh scope, numbered by this book in causal order.

        Page ``scope_registry`` row ``(label, serial)`` records every scope
        minted under ``label``; the next serial is how many exist.  The
        numbering is the compile's own, never a process-wide counter.

        The row is a NOVEL root (``MINT_SCOPE``, no operands; plan 80 N5):
        the scope IS the row, so it mints no id.  ``stage`` is the caller's
        stage; a caller not yet migrated leaves it and the post is recorded
        under ``RAW_STAGE``.
        """
        from .concordance_declarations import MINT_SCOPE, SCOPE_REGISTRY

        page = self.page(SCOPE_REGISTRY)
        label = str(label)
        scope = (label, page.scope_row_count(label))
        self.post(
            SCOPE_REGISTRY, scope, True,
            stage=RAW_STAGE if stage is None else stage,
            provenance=Novel(MINT_SCOPE, ()), mode=Mode.CONCORD,
        )
        return scope

    def latest_by_page(self, row: Any) -> dict[str, Any]:
        """The final fact recorded for `row` on each page that ever saw it."""
        result = {}
        for name, page in self.pages.items():
            runs = page.spans(row)
            if runs:
                result[name] = runs[-1][2]
        return result

    def disagreements(self, row: Any) -> dict[str, Any] | None:
        """The per-page facts for `row`, if more than one distinct fact
        exists among them -- else None (the pages agree, or only one saw it)."""
        latest = self.latest_by_page(row)
        if len({repr(fact) for fact in latest.values()}) > 1:
            return latest
        return None


class _ConcurrentIdentityBook(IdentityBook):
    """The same authority while isolated deployment workers publish to it.

    A post's fact, provenance edges, reverse index and clock tick are one
    transaction. Copy-on-read materialisation uses book-wide temporary state
    and must participate in that transaction too. The serial book has no lock
    dispatch in its hot path: this capability exists only across a worker join.
    """

    def _page(self, name):
        with self._publication_lock:
            page = IdentityBook._page(self, name)
            if type(page) is IdentityPage:
                page.__class__ = _ConcurrentIdentityPage
            return page


class _ConcurrentIdentityPage(IdentityPage):
    """Page reads and writes observe the same atomic book publication."""


def _concurrent_book_operation(operation):
    from functools import wraps

    @wraps(operation)
    def locked(self, *args, **kwargs):
        with self._publication_lock:
            return operation(self, *args, **kwargs)
    return locked


for _operation in ("post", "latest_ref", "mint_scope", "_source_stamp",
                   "derive_current", "current_ref", "stale_derivations", "settle_derivations", "import_cell",
                   "_materialise_row", "_preserve_fork_snapshots", "register_read_through",
                   "register_scope_projection", "edges_into", "edges_out_of", "mint_of"):
    setattr(_ConcurrentIdentityBook, _operation,
            _concurrent_book_operation(getattr(IdentityBook, _operation)))


def _concurrent_page_operation(operation):
    from functools import wraps

    @wraps(operation)
    def locked(self, *args, **kwargs):
        with self.book._publication_lock:
            return operation(self, *args, **kwargs)
    return locked


for _operation in ("latest", "latest_column", "history", "cell", "stamp_at", "is_virtual",
                   "scope_rows", "materialised_scope_rows", "_stamp", "set", "revise", "concord"):
    setattr(_ConcurrentIdentityPage, _operation,
            _concurrent_page_operation(getattr(IdentityPage, _operation)))


def concurrent_identity_publication(book):
    """Activate atomic publication on this book until all workers have joined."""
    from contextlib import contextmanager
    from threading import RLock

    @contextmanager
    def joined():
        if isinstance(book, _ConcurrentIdentityBook):
            yield book
            return
        original_class = type(book)
        if original_class is not IdentityBook:
            raise TypeError("concurrent publication requires the authoritative IdentityBook")
        book._publication_lock = RLock()
        book.__class__ = _ConcurrentIdentityBook
        for page in dict.values(book.pages):
            if type(page) is IdentityPage:
                page.__class__ = _ConcurrentIdentityPage
        try:
            yield book
        finally:
            for page in dict.values(book.pages):
                if type(page) is _ConcurrentIdentityPage:
                    page.__class__ = IdentityPage
            book.__class__ = original_class
            del book._publication_lock
    return joined()


@dataclass(frozen=True)
class SequenceContract:
    """The physical row contract owned by one resident sequence identity."""

    policy: str
    column_count: int
    writable: bool


@dataclass(frozen=True)
class SequenceRowLayout:
    """The element layout proven for one resident sequence identity."""

    column_shapes: tuple[tuple[int, ...], ...]
    column_dtypes: tuple[str, ...]


# The fixed 5-element width is the declared ABI of this state.
def shape_transformation_state(
    descriptor: Any,
) -> tuple[tuple[int, ...], str, int, str, tuple[int, ...] | None] | None:
    """Canonical, lossless shape state carried by one transformation edge.

    ``shape=()`` is retained: rank and metadata state distinguish a scalar
    from a dynamic collection whose leading extent belongs to runtime
    sequence storage.  ``sequence_row_shape`` is likewise explicit, because
    it is the geometry of an element-to-collection transformation rather
    than the complete collection shape.
    """

    if descriptor is None:
        return None
    if isinstance(descriptor, Mapping):
        shape = tuple(map(int, descriptor.get("shape") or ()))
        row_shape = descriptor.get("sequence_row_shape")
        return (
            shape,
            str(descriptor.get("dtype") or "unknown"),
            int(descriptor.get("rank", len(shape))),
            str(descriptor.get("metadata_state") or "static"),
            (
                None if row_shape is None
                else tuple(map(int, row_shape))
            ),
        )
    if isinstance(descriptor, tuple) and len(descriptor) == 5:
        shape, dtype, rank, metadata_state, row_shape = descriptor
        return (
            tuple(map(int, shape or ())), str(dtype or "unknown"),
            int(rank), str(metadata_state or "static"),
            None if row_shape is None else tuple(map(int, row_shape)),
        )
    raise TypeError(f"unsupported shape transformation state {descriptor!r}")


def descriptor_from_shape_transformation_state(
    state: Any,
) -> dict[str, Any] | None:
    """Reconstitute the descriptor recorded by the shared concordance."""

    state = shape_transformation_state(state)
    if state is None:
        return None
    shape, dtype, rank, metadata_state, row_shape = state
    descriptor = {
        "shape": tuple(shape), "dtype": str(dtype), "rank": int(rank),
    }
    if str(metadata_state) != "static":
        descriptor["metadata_state"] = str(metadata_state)
    if row_shape is not None:
        descriptor["sequence_row_shape"] = tuple(row_shape)
    return descriptor


#: The three pages of ``record_shape_transformation``: the edge (row IS the
#: edge: target, source, stage, operator, role; fact the operator's input
#: and output states), its reverse index keyed by the source identity, and
#: the consulted projection whose fact points back at its edge.  Their rows
#: and facts are exactly what ``withdraw_superseded_shape_derivations`` and
#: ``concordant_shape_transformation_state`` read.
SHAPE_EDGE_PAGE = declare_page(
    "shape_transformation_concordance",
    (RowField("target_scope", _SCOPE), RowField("target_id", _LABEL),
     RowField("source_scope", _SCOPE), RowField("source_id", _LABEL),
     RowField("stage", _NAME), RowField("operation", _NAME),
     RowField("role", _NAME)),
    tuple,
)
SHAPE_DEPENDENTS_PAGE = declare_page(
    "shape_transformation_dependents",
    (RowField("source", _SCOPE), RowField("edge_row", _LABEL)),
    bool,
)
SHAPE_STATE_PAGE = declare_page(
    "shape_transformation_state",
    (RowField("scope", _SCOPE), RowField("value_id", _LABEL)),
    tuple,
)
#: The settled non-scalar projection consumed by legacy shape readers.  This
#: used to be an undeclared raw page even when ``record_shape_transformation``
#: had already recorded the exact graph-derived state.  In the decimal glyph
#: helper that discarded the canonical-value edge for the ten-element digit
#: tuple and left ``proven_shape`` unsourced in the accepted dependency.
PROVEN_SHAPE_PAGE = declare_page(
    "proven_shape",
    (RowField("scope", _SCOPE), RowField("value_id", _LABEL)),
    tuple,
)
PROVEN_SHAPE_STAGE = declare_stage("proven_shape")
PROVEN_SHAPE_SOURCE_NOT_ON_BOOK = declare_reason(
    "proven_shape_source_not_on_book"
)
#: A resident sequence's element layout, committed from the cells the
#: caller read it from; ``..._NOT_ON_BOOK`` is a layout with no such cell.
SEQUENCE_ROW_LAYOUT_STAGE = declare_stage("sequence_row_layout")
SEQUENCE_ROW_LAYOUT_SOURCE_NOT_ON_BOOK = declare_reason(
    "sequence_row_layout_source_not_on_book"
)
#: A shape edge whose source identity has no changed state cell on the book
#: (the source is a graph identity the caller read off the graph).
SHAPE_SOURCE_NOT_ON_BOOK = declare_reason("shape_source_not_on_book")
#: A shape state re-resolved over an edge that did not change since the
#: row's previous revision (the row was withdrawn or re-pointed in between).
SHAPE_STATE_REDERIVED = declare_reason("shape_state_rederived_over_unchanged_edge")
#: A binding-kind resolution re-posted over the same ``argument_binding`` cell
#: with a different requesting kind.
ARGUMENT_BINDING_RESOLUTION_CAUSE_NOT_ON_BOOK = declare_reason(
    "argument_binding_resolution_cause_not_on_book"
)
#: A loop scope's inner-generation transition re-posted over the same value
#: cells (the transformation that caused it left no cell).
LOOP_SCOPE_TRANSITION_CAUSE_NOT_ON_BOOK = declare_reason(
    "loop_scope_transition_cause_not_on_book"
)


def record_shape_transformation(
    source_scope: Any,
    source_id: Any,
    target_scope: Any,
    target_id: Any,
    *,
    stage: Any,
    operation: Any,
    source_state: Any,
    target_state: Any,
    role: Any = "value",
    source_cells: tuple[Ref, ...] = (),
) -> tuple[Any, ...] | None:
    """Append one exact source-to-target shape transformation to the book.

    The edge page is the graph; its monotonically increasing columns are
    compile time.  The state page is the consulted current projection of
    that graph and retains every earlier projection as row history.  No
    caller-local shape cache participates in the decision.
    """

    source_scope = _shape_key(source_scope)
    target_scope = _shape_key(target_scope)
    source = shape_transformation_state(source_state)
    target = shape_transformation_state(target_state)
    book = current_identity_book()
    # Callers still pass the stage as a label; the registry mints the Stage
    # object from it (a name already declared is returned, never redeclared).
    # The seam closes when the callers pass Stage objects themselves.
    stage_object = book.registry.declare_stage(str(stage))
    edge_row = (
        target_scope, target_id, source_scope, source_id,
        str(stage), str(operation), str(role),
    )
    edge_fact = (source, target)
    edge_ref = book.latest_ref(SHAPE_EDGE_PAGE, edge_row)
    if edge_ref is None or book.page(SHAPE_EDGE_PAGE).latest(edge_row) != edge_fact:
        # The source side of a shape edge is a graph identity, not a book
        # row (census 00, section 2).  When the book already holds a state
        # cell for that identity and it changed since this edge was last
        # written, the edge derives from it; otherwise the cause is not on
        # the book and the edge is posted unsourced, which the audit lists.
        source_ref = book.latest_ref(SHAPE_STATE_PAGE, (source_scope, source_id))
        edge_ref = _post_or_unsourced(
            book, SHAPE_EDGE_PAGE, edge_row, edge_fact, stage_object,
            tuple(dict.fromkeys((
                *source_cells, *(() if source_ref is None else (source_ref,)),
            ))),
            SHAPE_SOURCE_NOT_ON_BOOK,
        )
    # The same edge, read from its source end: which targets were derived
    # from this identity.  A row's first element is the source identity, so
    # the page answers it directly (``scope_rows``) without a side index.
    book.post(
        SHAPE_DEPENDENTS_PAGE, ((source_scope, source_id), edge_row), True,
        stage=stage_object, provenance=Derived((edge_ref,)),
        mode=Mode.CONCORD,
    )
    state_row = (target_scope, target_id)
    state_fact = ("resolved", target, edge_row)
    state_ref = book.latest_ref(SHAPE_STATE_PAGE, state_row)
    incumbent = (
        None if state_ref is None
        else book.page(SHAPE_STATE_PAGE).latest(state_row)
    )
    # One projection per identity, many edges into it.  A target derived
    # from several sources (a binary operator's lhs and rhs) receives one
    # edge per source, all carrying the SAME target state.  The edges are
    # the graph and each is on the edge and dependents pages above with its
    # own source cell; the state row is the projection and names the edge it
    # was resolved through.  A second edge that agrees with a live
    # incumbent edge corroborates that projection; it does not re-resolve
    # it.  Revising the state per edge made the two agreeing edges
    # overwrite each other on every descriptor query: the llvm_dt_system
    # air+pool lowering, dt_system_over -> run_superstep callsite, revised
    # (step_0, 42) `Pow` 4,239 times alternating lhs/rhs with the shape
    # fixed at (1,) float64, and the page grew without bound
    # (CONTINUATION_dt_compile_stall.md).  A DIFFERENT target state is not
    # a corroboration and takes the revision path below.
    if (
        isinstance(incumbent, tuple)
        and len(incumbent) == 3
        and incumbent[0] == "resolved"
        and incumbent[1] == target
        and incumbent[2] != edge_row
        and isinstance(incumbent[2], tuple)
        and tuple(incumbent[2][:2]) == state_row
    ):
        incumbent_edge = book.page(SHAPE_EDGE_PAGE).latest(incumbent[2])
        if (
            isinstance(incumbent_edge, tuple)
            and len(incumbent_edge) == 2
            and incumbent_edge[1] == target
        ):
            return target
    if state_ref is None or incumbent != state_fact:
        previous = concordant_shape_transformation_state(
            target_scope, target_id,
        )
        # The projection derives from its edge.  A re-resolution after a
        # withdrawal derives from the same edge cell but from a DIFFERENT
        # cell set than the withdrawal did, so the api admits it as a
        # revision with a cause; only a state that changes over the same
        # edge with nothing changed is causeless and is posted unsourced.
        _post_or_unsourced(
            book, SHAPE_STATE_PAGE, state_row, state_fact, stage_object,
            (edge_ref,), SHAPE_STATE_REDERIVED,
        )
        if previous != target:
            withdraw_superseded_shape_derivations(
                target_scope, target_id, target, reason=stage,
            )
    return target


def _post_or_unsourced(
    book: IdentityBook, page: Page, row: tuple, fact: Any, stage: Stage,
    cells: tuple[Ref, ...], reason: Reason,
) -> Ref:
    """Post ``fact`` DERIVED from ``cells``; when the api refuses the revision
    because nothing in ``cells`` changed and the cell set is the previous
    revision's (or there are no cells), post it ``Unsourced(reason)`` so the
    causeless statement is listed by the audit instead of hidden."""
    # ``reason`` stays in the signature for its callers; there is no
    # unsourced path: the post derives from ``cells`` or is refused, and a
    # revision the api finds causeless is refused with that cause named.
    del reason
    return book.post(
        page, row, fact, stage=stage,
        provenance=Derived(tuple(cells)), mode=Mode.REVISE,
    )


def _newer_than_row(book: IdentityBook, source: Ref, latest: Ref | None) -> bool:
    """Whether ``source``'s cell is stamped after the row ``latest`` names
    (a row with no cell yet is older than anything)."""
    if latest is None:
        return True
    return book.stamp_of(source) > book.stamp_of(latest)


def withdraw_superseded_shape_derivations(
    scope: Any, value_id: Any, state: Any, *, reason: Any,
) -> None:
    """Carry a changed shape state along every edge derived from it.

    Each edge records the source state its target was derived from.  When
    the source now says something else, that derivation is superseded: the
    target's shape state, proven extents and committed sequence row layout
    are withdrawn in the same causal step, and the withdrawal continues
    downstream.  The next descriptor query re-derives the target and appends
    a new generation.  An edge recorded without a source state carries no
    derivation claim and is left alone.
    """

    book = current_identity_book()
    dependents = book.page("shape_transformation_dependents")
    edge_page = book.page("shape_transformation_concordance")
    state_page = book.page("shape_transformation_state")
    pending = [(_shape_key(scope), value_id,
                shape_transformation_state(state))]
    visited: set[tuple[Any, Any]] = set()
    while pending:
        source_scope, source_id, current = pending.pop()
        if (source_scope, source_id) in visited:
            continue
        visited.add((source_scope, source_id))
        for dependent_row in dependents.scope_rows((source_scope, source_id)):
            edge_row = dependent_row[1]
            target_scope, target_id = edge_row[0], edge_row[1]
            if (target_scope, target_id) == (source_scope, source_id):
                # A transport edge onto the same identity is the state that
                # was just written, not a derivation downstream of it.
                continue
            edge_fact = edge_page.latest(edge_row)
            if not (isinstance(edge_fact, tuple) and len(edge_fact) == 2):
                continue
            derived_from = edge_fact[0]
            if derived_from is None or derived_from == current:
                continue
            target_fact = state_page.latest((target_scope, target_id))
            if not (
                isinstance(target_fact, tuple)
                and target_fact
                and target_fact[0] == "resolved"
            ):
                # Already withdrawn (its dependents went with it) or never
                # resolved: nothing derived from it remains to supersede.
                continue
            if isinstance(target_id, int) and not isinstance(target_id, bool):
                invalidate_proven_shape(
                    target_scope, target_id, source_id, reason,
                )
                invalidate_sequence_row_layout(
                    target_scope, target_id, source_id, reason,
                )
            else:
                invalidate_shape_transformation(
                    target_scope, target_id, source_id, reason,
                    source_scope=source_scope,
                )
            pending.append((target_scope, target_id, None))


def concordant_shape_transformation_state(
    scope: Any, value_id: Any,
) -> tuple[Any, ...] | None:
    """Read the latest causally recorded shape at one graph identity."""

    row = (_shape_key(scope), value_id)
    fact = current_identity_book().page(
        "shape_transformation_state"
    ).latest(row)
    if not (isinstance(fact, tuple) and fact and fact[0] == "resolved"):
        return None
    return shape_transformation_state(fact[1])


def invalidate_shape_transformation(
    scope: Any, value_id: Any, source_id: Any, reason: Any,
    *, source_scope: Any = None, source_cells: tuple[Ref, ...] = (),
) -> Ref | None:
    """Record that a target's prior path was superseded upstream.

    The withdrawal derives from the cell of the identity that changed: the
    source's shape state cell (``source_scope`` defaults to the target's
    scope -- a dependency inside one function).  A source with no state cell
    on the book, or one that has not changed since the target's previous
    revision, leaves the withdrawal without a cause on the book: it is
    posted ``Unsourced(SHAPE_SOURCE_NOT_ON_BOOK)`` and listed by the audit.
    """

    book = current_identity_book()
    page = book.page(SHAPE_STATE_PAGE)
    row = (_shape_key(scope), value_id)
    fact = ("invalidated", source_id, str(reason))
    if page.latest(row) != fact:
        source_ref = book.latest_ref(SHAPE_STATE_PAGE, (
            _shape_key(scope if source_scope is None else source_scope),
            source_id,
        ))
        return _post_or_unsourced(
            book, SHAPE_STATE_PAGE, row, fact,
            book.registry.declare_stage(str(reason)),
            tuple(dict.fromkeys((
                *source_cells,
                *(() if source_ref is None else (source_ref,)),
            ))),
            SHAPE_SOURCE_NOT_ON_BOOK,
        )
    return book.latest_ref(SHAPE_STATE_PAGE, row)


def committed_sequence_row_layout(
    scope: Any,
    sequence_id: int,
    *,
    page: IdentityPage | None = None,
) -> SequenceRowLayout | None:
    """Read the element layout already proven for this exact sequence."""

    scope = _shape_key(scope)
    if page is None:
        page = current_identity_book().page(
            "sequence_row_layout_concordance"
        )
    fact = page.latest((scope, int(sequence_id)))
    if fact is None or (
        isinstance(fact, tuple) and fact and fact[0] == "invalidated"
    ):
        return None
    return SequenceRowLayout(
        column_shapes=tuple(
            tuple(map(int, shape)) for shape in fact[0]
        ),
        column_dtypes=tuple(map(str, fact[1])),
    )


def invalidate_sequence_row_layout(
    scope: Any, sequence_id: int, source_id: Any, reason: Any,
) -> None:
    """Withdraw a row layout whose deriving shape state was superseded.

    The withdrawal is a row event after the layout it withdraws, so the
    re-derived layout is a new generation rather than a disagreement with
    a fact the transformation graph no longer supports.
    """

    book = current_identity_book()
    page = book.page("sequence_row_layout_concordance")
    row = (_shape_key(scope), int(sequence_id))
    incumbent = page.latest(row)
    if incumbent is None:
        return
    fact = ("invalidated", source_id, str(reason))
    if incumbent != fact:
        # The withdrawal stands on the shape state that was superseded.
        state_ref = book.latest_ref(SHAPE_STATE_PAGE, row)
        _post_or_unsourced(
            book, book.registry.page("sequence_row_layout_concordance"),
            row, fact, book.registry.declare_stage(str(reason)),
            () if state_ref is None else (state_ref,),
            SEQUENCE_ROW_LAYOUT_SOURCE_NOT_ON_BOOK,
        )


def commit_sequence_row_layout(
    scope: Any,
    sequence_id: int,
    column_shapes: Iterable[Iterable[int]],
    column_dtypes: Iterable[Any],
    *,
    source: str,
    page: IdentityPage | None = None,
    sources: Iterable[Ref] = (),
) -> SequenceRowLayout:
    """Commit an element layout reached through the transformation graph.

    The row is posted DERIVED from ``sources`` (the cells the caller read the
    layout from) and the shape-state cell of the same row; a layout with
    neither is recorded ``Unsourced``.

    This is a direct identity-book row, not an id-keyed cache.  Producers
    publish the layout on the resident sequence identity they proved, and
    consumers ask for that same row.  An empty column shape is a proven
    scalar element here; absence of the row is the only unknown shape.
    """

    scope = _shape_key(scope)
    if page is None:
        page = current_identity_book().page(
            "sequence_row_layout_concordance"
        )
    shapes = tuple(
        tuple(int(extent) for extent in shape) for shape in column_shapes
    )
    dtypes = tuple(
        _canonical_sequence_row_dtype(dtype) for dtype in column_dtypes
    )
    if len(shapes) != len(dtypes) or not shapes:
        raise ValueError(
            "sequence row layout requires one dtype for every column: "
            f"shapes={shapes!r}, dtypes={dtypes!r}"
        )
    sid = int(sequence_id)
    incumbent = committed_sequence_row_layout(scope, sid, page=page)
    if incumbent is not None and incumbent.column_shapes != shapes:
        raise ValueError(
            "sequence row layout concordance disagreement for "
            f"{scope!r} value {sid}: recorded="
            f"{incumbent.column_shapes!r}, {source} says {shapes!r}"
        )
    resolved_dtypes = dtypes
    if incumbent is not None:
        if len(incumbent.column_dtypes) != len(dtypes):
            raise ValueError(
                "sequence row layout concordance width disagreement for "
                f"{scope!r} value {sid}: recorded="
                f"{incumbent.column_dtypes!r}, {source} says {dtypes!r}"
            )
        merged: list[str] = []
        for recorded, proposed in zip(
            incumbent.column_dtypes, dtypes, strict=True,
        ):
            if (
                recorded != "unknown"
                and proposed != "unknown"
                and recorded != proposed
            ):
                raise ValueError(
                    "sequence row layout dtype disagreement for "
                    f"{scope!r} value {sid}: recorded="
                    f"{incumbent.column_dtypes!r}, {source} says {dtypes!r}"
                )
            merged.append(
                proposed if recorded == "unknown" else recorded
            )
        resolved_dtypes = tuple(merged)
    resolved = SequenceRowLayout(shapes, resolved_dtypes)
    row = (scope, sid)
    if incumbent is not None and incumbent == resolved:
        # Program: ``CorrelationTable._operator_result_type_findings``
        # (bootstrap unit 101): a second producer committed the recorded
        # layout from the same cells under another ``source`` label.  Wrong
        # record: the label is part of the fact, so the repeat was a REVISE
        # with no changed source and the book refused it.  Rule: the same
        # layout is not a revision; the incumbent cell stands.
        return resolved
    fact = (
        resolved.column_shapes,
        resolved.column_dtypes,
        str(source),
    )
    book = page.book
    if book is None:
        history = page.history(row)
        column = history[-1][0] + 1 if history else 0
        page.set(row, column, fact)
        return resolved
    _post_or_unsourced(
        book, book.registry.page("sequence_row_layout_concordance"), row,
        fact, SEQUENCE_ROW_LAYOUT_STAGE,
        tuple(dict.fromkeys(
            cell for cell in (
                *sources, book.latest_ref(SHAPE_STATE_PAGE, row),
            )
            if isinstance(cell, Ref)
        )),
        SEQUENCE_ROW_LAYOUT_SOURCE_NOT_ON_BOOK,
    )
    return resolved


def _canonical_sequence_row_dtype(dtype: Any) -> str:
    spelling = "unknown" if dtype is None else str(dtype)
    return "unknown" if spelling in {"", "None", "unknown"} else spelling


def committed_sequence_row_dtypes(
    scope: Any,
    sequence_id: int,
    *,
    page: IdentityPage | None = None,
) -> tuple[str, ...] | None:
    """Read the row dtype contract for one resident sequence identity."""

    if page is None:
        page = current_identity_book().page(
            "sequence_row_dtype_concordance"
        )
    fact = page.latest((scope, int(sequence_id)))
    if fact is None:
        return None
    return tuple(map(str, fact[0]))


def concord_sequence_row_dtypes(
    scope: Any,
    claims: Mapping[int, Iterable[Any]],
    *,
    source: str,
    page: IdentityPage | None = None,
    sources: Iterable[Ref] = (),
    stage: Any = None,
) -> tuple[str, ...]:
    """Resolve one row layout across sequence identities proven equivalent.

    ``unknown`` is absence of a claim, not a competing dtype.  A replace or
    carried-state edge proves its two arenas have the same physical row, so a
    known column on either side refines the other.  Two different known
    dtypes are a real disagreement and compilation stops here.
    """

    if page is None:
        page = current_identity_book().page(
            "sequence_row_dtype_concordance"
        )
    normalized: dict[int, tuple[str, ...]] = {}
    for sequence_id, raw_dtypes in claims.items():
        sid = int(sequence_id)
        proposed = tuple(
            _canonical_sequence_row_dtype(dtype) for dtype in raw_dtypes
        )
        incumbent = committed_sequence_row_dtypes(
            scope, sid, page=page
        )
        if incumbent is not None and len(incumbent) != len(proposed):
            raise ValueError(
                "sequence row dtype concordance width disagreement for "
                f"{scope!r} value {sid}: recorded={incumbent!r}, "
                f"{source} says {proposed!r}"
            )
        normalized[sid] = proposed if incumbent is None else tuple(
            recorded if proposed_dtype == "unknown" else proposed_dtype
            if recorded == "unknown" else recorded
            for recorded, proposed_dtype in zip(incumbent, proposed)
        )
        if incumbent is not None:
            for recorded, proposed_dtype in zip(incumbent, proposed):
                if (
                    recorded != "unknown"
                    and proposed_dtype != "unknown"
                    and recorded != proposed_dtype
                ):
                    raise ValueError(
                        "sequence row dtype concordance disagreement for "
                        f"{scope!r} value {sid}: recorded={incumbent!r}, "
                        f"{source} says {proposed!r}"
                    )
    widths = {len(dtypes) for dtypes in normalized.values()}
    if len(widths) > 1:
        raise ValueError(
            "sequence row dtype concordance cannot equate different row "
            f"widths for {scope!r} at {source}: {normalized!r}"
        )
    width = next(iter(widths), 0)
    resolved: list[str] = []
    for column in range(width):
        known = {
            dtypes[column] for dtypes in normalized.values()
            if dtypes[column] != "unknown"
        }
        if len(known) > 1:
            raise ValueError(
                "sequence row dtype concordance disagreement for "
                f"{scope!r} column {column} at {source}: {normalized!r}"
            )
        resolved.append(next(iter(known), "unknown"))
    result = tuple(resolved)
    from .concordance_declarations import CONTROL_SSA, SEQUENCE_ROW_DTYPE

    book = page.book
    # ``result`` was resolved across every row the call equates: each row's
    # fact derives from the declared sources and from the incumbent cell of
    # every row in the arena (a known column on one side refined the other).
    incumbent_cells = tuple(
        cell for cell in (
            book.latest_ref(SEQUENCE_ROW_DTYPE, (scope, int(sequence_id)))
            for sequence_id in normalized
        )
        if cell is not None
    )
    source_cells = tuple(dict.fromkeys((
        *(cell for cell in sources if isinstance(cell, Ref)),
        *incumbent_cells,
    )))
    for sequence_id in normalized:
        row = (scope, int(sequence_id))
        fact = (result, str(source))
        # No raw path: a first statement with no source is refused by the
        # book, naming the writer that read the dtypes from nothing.
        book.post(
            SEQUENCE_ROW_DTYPE, row, fact,
            stage=CONTROL_SSA if stage is None else stage,
            provenance=Derived(source_cells), mode=Mode.REVISE,
        )
    return result


def committed_sequence_contract(
    scope: Any,
    sequence_id: int,
    *,
    page: IdentityPage | None = None,
) -> SequenceContract | None:
    """Read the sequence contract already committed for this exact identity."""

    if page is None:
        page = current_identity_book().page("sequence_contract_concordance")
    fact = page.latest((scope, int(sequence_id)))
    if fact is None:
        return None
    return SequenceContract(
        policy=str(fact[0]),
        column_count=int(fact[1]),
        writable=bool(fact[2]),
    )


def commit_sequence_contract(
    scope: Any,
    sequence_id: int,
    policy: str,
    column_count: int,
    writable: bool,
    *,
    source: str,
    page: IdentityPage | None = None,
    sources: Iterable[Ref] = (),
    stage: Any = None,
) -> SequenceContract:
    """Commit or verify one resident sequence's physical row contract.

    The first source-stage fact owns policy and row width. Later compiler
    stages may repeat that contract and may prove the same storage writable,
    but they may not silently replace its policy or width. Every accepted
    statement is retained in the page history together with its source stage.
    """

    if page is None:
        page = current_identity_book().page("sequence_contract_concordance")
    row = (scope, int(sequence_id))
    proposed = SequenceContract(
        policy=str(policy),
        column_count=int(column_count),
        writable=bool(writable),
    )
    if proposed.column_count < 1:
        raise ValueError(
            f"sequence contract for {scope!r} value {sequence_id} declares "
            f"invalid column count {proposed.column_count} at {source}"
        )
    incumbent = committed_sequence_contract(scope, sequence_id, page=page)
    if incumbent is not None and (
        incumbent.policy != proposed.policy
        or incumbent.column_count != proposed.column_count
    ):
        prior = page.latest(row)
        raise ValueError(
            f"sequence contract concordance disagreement for {scope!r} "
            f"value {sequence_id}: {prior[3]} committed "
            f"{incumbent.policy}/{incumbent.column_count}, {source} says "
            f"{proposed.policy}/{proposed.column_count}"
        )
    resolved = SequenceContract(
        policy=proposed.policy,
        column_count=proposed.column_count,
        writable=bool(proposed.writable or (
            incumbent.writable if incumbent is not None else False
        )),
    )
    fact = (
        resolved.policy,
        resolved.column_count,
        resolved.writable,
        str(source),
    )
    from .concordance_declarations import (
        CONTROL_SSA, SEQUENCE_CONTRACT_CONCORDANCE,
    )

    prior_cell = page.book.latest_ref(SEQUENCE_CONTRACT_CONCORDANCE, row)
    source_cells = tuple(dict.fromkeys((
        *(cell for cell in sources if isinstance(cell, Ref)),
        *((prior_cell,) if prior_cell is not None else ()),
    )))
    # The contract derives from the cells the statement was read from and
    # the incumbent it repeats or refines; no raw path.  A first statement
    # that names no source is refused by the book at the writer.
    page.book.post(
        SEQUENCE_CONTRACT_CONCORDANCE, row, fact,
        stage=CONTROL_SSA if stage is None else stage,
        provenance=Derived(source_cells), mode=Mode.REVISE,
    )
    return resolved


# One book per top-level compile, reachable from anywhere in the call stack
# without a `module` argument -- a contextvar rather than module.metadata,
# because the module a mid-pipeline pass is building is not always the same
# object the top-level entry point will eventually return, and a crash deep
# inside one stage (exactly tonight's fault) must not lose everything
# recorded before it.  `lower_ast_source_to_ssa` opens one with
# `begin_identity_book()` and dumps it in a `finally` with
# `end_identity_book()`, success or failure, at the one small, honest cost
# of the book itself: it is only ever appended to as a side effect of work
# the pass was already doing.
_ACTIVE_IDENTITY_BOOK: contextvars.ContextVar[IdentityBook | None] = (
    contextvars.ContextVar("identity_concordance_active_book", default=None)
)


def begin_identity_book(book: IdentityBook | None = None) -> tuple[IdentityBook, contextvars.Token]:
    """Make a fresh compile book (or an explicitly resumed module book) current.

    Returns the book and a reset token.  A nested compile (one
    ``lower_ast_source_to_ssa`` invoked while another is already on the
    stack) must not clobber the outer compile's book to ``None`` when it
    finishes -- that would silently orphan everything the outer compile
    recorded before the nested one started.  The token lets ``end_identity_
    book`` restore exactly the PREVIOUS value instead of blanking it.
    """
    book = IdentityBook() if book is None else book
    token = _ACTIVE_IDENTITY_BOOK.set(book)
    return book, token


def current_identity_book() -> IdentityBook:
    """The active compile's book, creating a detached one if none is open
    (so a page write is never a hard error just because nothing called
    ``begin_identity_book`` -- it simply has nowhere to be dumped later)."""
    book = _ACTIVE_IDENTITY_BOOK.get()
    if book is None:
        book = IdentityBook(detached=True)
        _ACTIVE_IDENTITY_BOOK.set(book)
    return book


def concordant_alias_bindings(
    scope: Any,
    *ledgers: Mapping[int, int] | Iterable[tuple[int, int]],
    page: IdentityPage | None = None,
) -> dict[int, int]:
    """Return one checked alias ledger for a planning scope.

    ``planning_value_concordance`` is the shared identity authority.  Some
    finished SSA functions also retain a local ``value_aliases`` snapshot or
    an ``output_identity_aliases`` receipt because backends need the facts
    after the active compile context has closed.  A consumer must not choose
    one of those records and silently ignore the others: combine them here,
    and refuse any disagreement about the same alias.

    The page is read last only to make the authority explicit; agreement is
    required, so insertion order never decides an identity.
    """

    if page is None:
        page = current_identity_book().page("planning_value_concordance")
    sources: list[tuple[str, Iterable[tuple[int, int]]]] = []
    for index, ledger in enumerate(ledgers):
        pairs = ledger.items() if isinstance(ledger, Mapping) else ledger
        sources.append((f"ledger[{index}]", pairs))
    sources.append((page.name, page.alias_bindings(scope).items()))

    result: dict[int, int] = {}
    owners: dict[int, str] = {}
    for source, pairs in sources:
        for alias, resident in pairs:
            alias = int(alias)
            resident = int(resident)
            incumbent = result.get(alias)
            if incumbent is not None and incumbent != resident:
                raise ValueError(
                    f"identity concordance disagreement for {scope!r} "
                    f"value {alias}: {owners[alias]} says {incumbent}, "
                    f"{source} says {resident}"
                )
            result[alias] = resident
            owners.setdefault(alias, source)
    return result


def resolved_concordant_alias_bindings(
    scope: Any,
    *ledgers: Mapping[int, int] | Iterable[tuple[int, int]],
    page: IdentityPage | None = None,
) -> dict[int, int]:
    """Merge exact alias receipts and resolve every transitive chain.

    Each input ledger may own a different segment of one identity path. A
    consumer must not stop at the boundary between pages: ``a -> b`` on the
    planning page and ``b -> resident`` on the control page are one proven
    identity. Cycles remain an error and name the complete path.
    """

    sources: list[tuple[str, Iterable[tuple[int, int]]]] = []
    for index, ledger in enumerate(ledgers):
        pairs = ledger.items() if isinstance(ledger, Mapping) else ledger
        sources.append((f"ledger[{index}]", pairs))
    if page is None:
        page = current_identity_book().page("planning_value_concordance")
    sources.append((page.name, page.alias_bindings(scope).items()))

    edges: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for source, pairs in sources:
        for alias, target in pairs:
            edge = (int(target), str(source))
            if edge not in edges[int(alias)]:
                edges[int(alias)].append(edge)

    memo: dict[int, int] = {}

    def resolve(value_id: int, path: tuple[int, ...] = ()) -> int:
        value_id = int(value_id)
        if value_id in memo:
            return memo[value_id]
        if value_id in path:
            raise ValueError(
                f"cyclic planning identity concordance for {scope!r}: "
                f"{(*path, value_id)}"
            )
        targets = tuple(
            target for target, _source in edges.get(value_id, ())
            if int(target) != value_id
        )
        if not targets:
            memo[value_id] = value_id
            return value_id
        roots = {
            resolve(target, (*path, value_id)) for target in targets
        }
        if len(roots) != 1:
            claims = tuple(edges.get(value_id, ()))
            raise ValueError(
                f"identity concordance disagreement for {scope!r} value "
                f"{value_id}: claims={claims!r}, terminal residents="
                f"{tuple(sorted(roots))!r}"
            )
        root = next(iter(roots))
        memo[value_id] = root
        return root

    return {alias: resolve(alias) for alias in edges}


def end_identity_book(
    token: contextvars.Token | None = None,
) -> IdentityBook | None:
    """Detach and return the book that was active (or None if none was
    open), restoring whatever was active before it via `token` when given
    (see `begin_identity_book`) instead of unconditionally clearing to None.
    """
    book = _ACTIVE_IDENTITY_BOOK.get()
    if token is not None:
        _ACTIVE_IDENTITY_BOOK.reset(token)
    else:
        _ACTIVE_IDENTITY_BOOK.set(None)
    return book


def identity_book(module: Any) -> IdentityBook:
    """The current compile's book, also cached on ``module.metadata`` when
    available -- so code with a module in hand (an already-finished
    ``IRModule``, inspected after the fact) and code with only the ambient
    compile context (a pass mid-construction, or an exception handler with
    no module at all) read the exact same instance."""
    metadata = getattr(module, "metadata", None)
    # The compile closes its book in a ``finally`` before returning, so after
    # that point ``current_identity_book()`` mints a fresh EMPTY one.  A
    # caller holding the finished module would then read an empty book and
    # conclude the compile recorded nothing -- which is the opposite of what
    # this function promises.  The module's own attached book wins whenever
    # there is one.
    attached = None if metadata is None else metadata.get("identity_book")
    if attached is not None:
        return attached
    book = current_identity_book()
    if metadata is not None:
        metadata["identity_book"] = book
    return book


def authored_function_name(name: Any) -> str:
    """The authored name behind a lowered symbol.

    Stores are keyed by the function as written -- ``solve`` -- while a
    lowered symbol carries the module prefix, the callsite specialization
    hash and the region index.  Without stripping those, rows never line up
    and every value appears to have exactly one source.
    """

    text = str(name)
    for separator in ("__specialized_", "__planned_region"):
        if separator in text:
            text = text.split(separator)[0]
    # Split at the FIRST separator: the artifact prefix is at the front, and
    # an authored name that itself begins with an underscore makes the last
    # separator fall inside ``___``, which would eat that underscore.
    return text.split("__", 1)[-1] if "__" in text else text


def loop_scope_owner(name: Any) -> str:
    """The lowering that owns a ``loop_scope`` row: the lowered symbol with
    only its planned-region suffix removed (a region shares its owner's
    loops).

    Program: ``find(n, k)`` called twice gives two callsite
    specializations of one authored loop; each lowering mints its own
    carried value.  Wrong record: rows keyed by the AUTHORED name
    (``authored_function_name``: module prefix and specialization hash
    stripped) put both specializations' generations on one row, and the
    second declaration refused ("already holds carried ...").  Rule: a
    specialization is its own lowering, so its loop generations are its
    own rows; the key keeps the lowered symbol.
    """

    return str(name).split("__planned_region", 1)[0]


def _shape_owner_metadata(owner: Any) -> dict:
    """The metadata dict that names the shape scope of ``owner``: a
    ProcessGraph (``G.graph``), a networkx graph (``graph``), an IR function
    (``metadata``) or the metadata dict itself."""

    graph = getattr(owner, "G", None)
    if graph is not None and isinstance(getattr(graph, "graph", None), dict):
        return graph.graph
    if isinstance(getattr(owner, "graph", None), dict):
        return owner.graph
    if isinstance(getattr(owner, "metadata", None), dict):
        return owner.metadata
    if isinstance(owner, dict):
        return owner
    raise TypeError(f"no shape scope metadata on {type(owner).__name__}")


def shape_scope_of(owner: Any) -> Any:
    """The shape-proof scope of the COPY ``owner`` states shapes for.

    ``proven_shape``, ``shape_transformation_state`` and the other shape
    pages are keyed by this scope -- the book-minted scope of one graph copy
    -- and never by the authored function name: the 2x2 and the 3x3
    specialization of one function are two copies and never share a row.
    The scope is minted the first time a graph is asked and rides in the
    graph's metadata, so a copy that is only an extraction of its source
    (a function shell, a dispatch region, an SSA function lowered from it)
    states shapes in the SAME scope; a callsite specialization is a variant
    and forks its own (``fork_shape_scope``).

    An IR function that carries no scope (one no graph lowered to) is its own
    copy: its scope is its exact symbol, never parsed.
    """

    metadata = _shape_owner_metadata(owner)
    scope = metadata.get("shape_scope")
    if scope is not None:
        return tuple(scope)
    if hasattr(owner, "blocks") and hasattr(owner, "name"):
        # An IR function that no graph copy stamped.
        return ("ir_function", str(owner.name))
    from .concordance_declarations import (
        SCOPE_REGISTRY, SHAPE_SCOPE, SHAPE_SCOPE_FUNCTION,
    )

    authored = str(
        metadata.get("function_name")
        or metadata.get("qualified_name")
        or "<module>"
    )
    book = current_identity_book()
    scope = book.mint_scope(f"{authored}|shape", SHAPE_SCOPE)
    book.post(
        SHAPE_SCOPE_FUNCTION, (scope,), authored, stage=SHAPE_SCOPE,
        provenance=Derived((book.latest_ref(SCOPE_REGISTRY, scope),)),
        mode=Mode.CONCORD,
    )
    metadata["shape_scope"] = scope
    return scope


def fork_shape_scope(owner: Any, cause: str) -> Any:
    """Give a graph copy that is a VARIANT of its source its own shape scope.

    A callsite specialization states shapes the source copy does not: it is
    minted a fresh scope, whose origin (``scope_origin``) names the source
    scope it was specialized from, DERIVED from the source scope's registry
    cell.  The source's rows are never written by the variant."""

    from .concordance_declarations import (
        SCOPE_ORIGIN, SCOPE_REGISTRY, SHAPE_SCOPE, SHAPE_SCOPE_FUNCTION,
        ScopeFork,
    )

    source = shape_scope_of(owner)
    book = current_identity_book()
    authored = shape_scope_function(source)
    forked = book.mint_scope(f"{authored}|shape", SHAPE_SCOPE)
    forked_cell = book.latest_ref(SCOPE_REGISTRY, forked)
    book.post(
        SHAPE_SCOPE_FUNCTION, (forked,), authored, stage=SHAPE_SCOPE,
        provenance=Derived((forked_cell,)), mode=Mode.CONCORD,
    )
    source_cell = book.latest_ref(SCOPE_REGISTRY, source)
    book.post(
        SCOPE_ORIGIN, (forked,), ScopeFork(source, str(cause)),
        stage=SHAPE_SCOPE,
        provenance=Derived(tuple(
            cell for cell in (source_cell, forked_cell) if cell is not None
        )),
        mode=Mode.CONCORD,
    )
    _shape_owner_metadata(owner)["shape_scope"] = forked
    return forked


def shape_scope_is_variant(owner: Any) -> bool:
    """Whether ``owner``'s shape scope was forked from another copy's
    (``fork_shape_scope``): the book holds a ``scope_origin`` row for it."""

    from .concordance_declarations import SCOPE_ORIGIN

    page = current_identity_book().pages.get(SCOPE_ORIGIN.name)
    if page is None:
        return False
    return page.latest((shape_scope_of(owner),)) is not None


def shape_scope_is_variant(owner: Any) -> bool:
    """Whether ``owner``'s shape scope was forked from another copy's
    (``fork_shape_scope``): the book holds a ``scope_origin`` row for it."""

    from .concordance_declarations import SCOPE_ORIGIN

    page = current_identity_book().pages.get(SCOPE_ORIGIN.name)
    if page is None:
        return False
    return page.latest((shape_scope_of(owner),)) is not None


def shape_scope_function(scope: Any) -> str | None:
    """The authored function a shape scope states shapes for, as the book
    recorded it when the scope was minted (``None`` for a scope that is not
    a shape scope)."""

    from .concordance_declarations import SHAPE_SCOPE_FUNCTION

    page = current_identity_book().pages.get(SHAPE_SCOPE_FUNCTION.name)
    if page is None:
        return None
    return page.latest((tuple(scope) if isinstance(scope, list) else scope,))


def _shape_key(scope: Any) -> Any:
    """The row key of a shape page: the copy's scope, exactly as given.

    A scope is an identity and is used as one -- no name is parsed.  (A bare
    string is an opaque scope of its own: a caller with no copy to name.)"""

    return tuple(scope) if isinstance(scope, list) else scope


OUTER, CARRIED, INNER = 0, 1, 2

_GENERATION_NAMES = {OUTER: "outer", CARRIED: "carried", INNER: "inner"}


def _post_loop_scope_cell(
    book: "IdentityBook", row: tuple, column: int, fact: Any, cells: Any,
) -> None:
    """One generation cell of a ``loop_scope`` row: the page's COLUMN is the
    generation, so a cell is posted (REVISE, DERIVED from ``cells``) only
    when it is the row's next column; a cell already there with the same fact
    says nothing new.  Anything else is a ``ConcordanceRefusal`` naming the
    row: there is no raw write."""

    page = book.page("loop_scope")
    generation = _GENERATION_NAMES.get(column, f"column {column}")
    if (row, column) in page.cells:
        # Program: a loop declared again with a different value id for a
        # generation it already holds.  Wrong write: page.set over the cell
        # (a raw write the book refuses, which the caller swallowed).  Rule:
        # a generation already declared is revised only by a post.
        if page.cells[(row, column)] != fact:
            raise ConcordanceRefusal(
                f"loop_scope row {row!r} already holds {generation} "
                f"{page.cells[(row, column)]!r}; {fact!r} is not a post "
                "from new source cells"
            )
        return
    sources = tuple(cell for cell in dict.fromkeys(cells or ()) if cell is not None)
    # Program: a loop declared with no source cell for a generation (the
    # caller's cell lookup returned None, or no cells were passed).  Wrong
    # write: page.set of the generation, unsourced.  Rule: every generation
    # posts from the cell its value was read from, or the declaration refuses.
    if not sources:
        raise ConcordanceRefusal(
            f"loop_scope row {row!r} {generation} {fact!r} has no source "
            "cell: the value id has no posted identity cell to derive from"
        )
    if column != len(page.history(row)):
        raise ConcordanceRefusal(
            f"loop_scope row {row!r} {generation} (column {column}) is not "
            f"the next column ({len(page.history(row))}); cannot post {fact!r}"
        )
    book.post(
        book.registry.page("loop_scope"), row, fact,
        stage=book.registry.declare_stage("loop_scope_declaration"),
        provenance=Derived(sources), mode=Mode.REVISE,
    )


def declare_loop_scope(
    function: Any, loop_node_id: Any, header: str, latch: str,
    exit_block: str, rebinds: Any, *, boundary_cells: Any = (),
    rebind_cells: Any = None,
) -> None:
    """Declare one loop as a scope whose bindings evolve per iteration.

    A nested scope binds a name once for its whole extent.  A loop body
    binds it differently on every entry, so the page's COLUMN is the
    generation -- outer, carried, inner -- rather than a round.  The
    boundary is recorded with the rebinds because a transformation that
    moves code across it must be able to ask where it is, instead of
    recovering it from block names or branch topology that the
    transformation itself may have rewritten.

    ``boundary_cells``: the loop construct's cell(s), which the boundary
    derives from.  ``rebind_cells(outer, carried, inner)``: the cells of the
    three generations' values, which each generation derives from.
    """

    book = current_identity_book()
    page = book.page("loop_scope")
    scope = (loop_scope_owner(function), int(loop_node_id))
    _post_loop_scope_cell(
        book, (*scope, "boundary"), 0,
        (str(header), str(latch), str(exit_block)), boundary_cells,
    )
    rebinds = tuple(rebinds)
    outer_counts = Counter(int(rebind[0]) for rebind in rebinds)
    for ordinal, rebind in enumerate(rebinds):
        (
            outer_id, carried_id, inner_id, graph_outer, graph_inner,
            *binding_tail,
        ) = rebind
        source_bindings = tuple(binding_tail[0]) if binding_tail else ()
        # A value id is not a lexical binding identity.  When two carried
        # names share their preheader resident, retain one concordance row
        # per name/update instead of letting the later declaration overwrite
        # the earlier row.  Unique rows retain the historical key shape.
        row_key = (
            int(outer_id)
            if outer_counts[int(outer_id)] == 1 and not source_bindings
            else (
                int(outer_id),
                source_bindings or ("entry", int(ordinal)),
            )
        )
        row = (*scope, row_key)
        generation_cells = (
            () if rebind_cells is None
            else tuple(rebind_cells(
                int(outer_id), int(carried_id), int(inner_id),
            ))
        )
        # Each column is a different generation.  Deriving all three from
        # the same aggregate cell set makes the second revision correctly
        # fail the monotonic-book rule (no source changed).  Preserve the
        # callback's positional identity and source each generation from its
        # own exact SSA cell.
        generation_sources = tuple(
            (() if index >= len(generation_cells) else (
                generation_cells[index],
            ))
            for index in (OUTER, CARRIED, INNER)
        )
        _post_loop_scope_cell(
            book, row, OUTER, int(outer_id), generation_sources[OUTER],
        )
        _post_loop_scope_cell(
            book, row, CARRIED, int(carried_id), generation_sources[CARRIED],
        )
        _post_loop_scope_cell(
            book, row, INNER, int(inner_id), generation_sources[INNER],
        )
        # A row is strictly the three generations.  The graph/binding
        # annotation columns were written only by the unsourced (raw) path
        # and are gone; the binding identity is retained in the row key.


def rebind_loop_scope_inner(
    function: Any,
    loop_node_id: Any,
    declared_inner: int,
    resident_inner: int,
    reason: Any,
    *,
    cells: Any = (),
) -> None:
    """Register a transformation of the value crossing a loop backedge.

    The scope declaration preserves the authored outer/carried/inner
    generations.  Outlining or aggregate projection can subsequently mint a
    resident SSA value for that same inner generation.  Record that transition
    separately so the original declaration remains historical evidence while
    every later consumer sees the resident identity.

    ``cells``: the cells of the declared and resident values, which the
    transition derives from.
    """

    book = current_identity_book()
    row = (
        loop_scope_owner(function), int(loop_node_id),
        int(declared_inner),
    )
    fact = (int(resident_inner), str(reason))
    sources = tuple(cell for cell in dict.fromkeys(cells or ()) if cell is not None)
    # Program: a latch resident minted for a declared inner generation whose
    # ssa_value cells were not supplied.  Wrong write: page.set of the
    # transition at the next column, unsourced.  Rule: the transition posts
    # from the declared and resident cells, or it refuses by row.
    if not sources:
        raise ConcordanceRefusal(
            f"loop_scope_inner_transition row {row!r} -> {fact!r} has no "
            "source cell: the declared and resident values have no posted "
            "identity cells to derive from"
        )
    _post_or_unsourced(
        book, book.registry.page("loop_scope_inner_transition"), row,
        fact, book.registry.declare_stage("loop_scope_declaration"),
        sources, LOOP_SCOPE_TRANSITION_CAUSE_NOT_ON_BOOK,
    )


def loop_scope_declarations(book: Any, function: Any) -> list[dict]:
    """Every loop scope declared for one authored function."""

    page = book.page("loop_scope")
    wanted = loop_scope_owner(function)
    scopes: dict[int, dict] = {}
    for row in page.rows():
        if not (isinstance(row, tuple) and len(row) == 3):
            continue
        name, loop_node_id, key = row
        if name != wanted:
            continue
        record = scopes.setdefault(
            int(loop_node_id),
            {
                "loop_node_id": int(loop_node_id),
                "boundary": None,
                "rebinds": [],
            },
        )
        if key == "boundary":
            record["boundary"] = page.latest(row)
            continue
        generations = dict(page.history(row))
        if (
            OUTER in generations
            and CARRIED in generations
            and INNER in generations
        ):
            declared_inner = int(generations[INNER])
            transition = book.page("loop_scope_inner_transition").latest((
                wanted, int(loop_node_id), declared_inner,
            ))
            resident_inner = (
                int(transition[0])
                if isinstance(transition, tuple) and transition
                else declared_inner
            )
            rebind = {
                "outer": int(generations[OUTER]),
                "carried": int(generations[CARRIED]),
                "inner": resident_inner,
                "declared_inner": declared_inner,
            }
            # source_bindings come only from the row key (the annotation
            # column that once carried them is no longer written).
            if (
                isinstance(key, tuple)
                and len(key) == 2
                and isinstance(key[1], tuple)
                and not (
                    len(key[1]) == 2 and key[1][0] == "entry"
                    and isinstance(key[1][1], int)
                )
            ):
                rebind["source_bindings"] = tuple(key[1])
            record["rebinds"].append(rebind)
    return [record for record in scopes.values() if record["boundary"]]


def concord_loop_scope_latch_residents(module: Any) -> tuple[dict, ...]:
    """Concord the completed module with every declared loop generation.

    Aggregate legalization and linked-call projection happen after control
    lowering declared the authored inner generation.  At the completed-module
    seam the loop Phi is the exact authority on what now crosses the latch.
    Reconcile that resident into the concordance so the declaration tracks the
    transformation instead of leaving a locally recorded but globally unknown
    rename.  The same declaration says that an instruction inside the loop
    cannot read the preheader (outer) generation: its header Phi is the exact
    carried generation for that lexical binding.  Late-installed instructions
    are rebound here from that declaration, not from a name, dtype, or inferred
    value equivalence.
    """

    book = identity_book(module)
    receipts: list[dict] = []
    for function_name, function in getattr(module, "functions", {}).items():
        for declaration in loop_scope_declarations(book, function_name):
            header, latch, _exit = declaration["boundary"]
            loop_node_id = int(declaration["loop_node_id"])
            rebinds_by_outer: dict[int, list[dict]] = {}
            for declared_rebind in declaration["rebinds"]:
                rebinds_by_outer.setdefault(
                    int(declared_rebind["outer"]), []
                ).append(declared_rebind)
            for rebind in declaration["rebinds"]:
                carried = int(rebind["carried"])
                declared_inner = int(rebind.get(
                    "declared_inner", rebind["inner"]
                ))
                phi = _carried_phi(function, str(header), carried)
                if phi is None:
                    continue
                incoming = tuple(phi.attributes.get("incoming_blocks") or ())
                resident = next((
                    int(argument.id)
                    for predecessor, argument in zip(incoming, phi.args)
                    if str(predecessor) == str(latch)
                    and getattr(argument, "id", None) is not None
                ), None)
                if resident is None or resident == int(rebind["inner"]):
                    continue
                from .ssa_record_return_state import ssa_value_identity_cell

                rebind_loop_scope_inner(
                    function_name,
                    loop_node_id,
                    declared_inner,
                    resident,
                    "completed_module_latch_projection",
                    cells=tuple(
                        ssa_value_identity_cell(function, value_id, book=book)
                        for value_id in (declared_inner, resident)
                    ),
                )
                receipt = {
                    "function": str(function_name),
                    "loop_node_id": loop_node_id,
                    "carried": carried,
                    "declared_inner": declared_inner,
                    "resident_inner": resident,
                    "reason": "completed_module_latch_projection",
                }
                receipts.append(receipt)
                argument = next((
                    argument
                    for predecessor, argument in zip(incoming, phi.args)
                    if str(predecessor) == str(latch)
                    and int(argument.id) == resident
                ), None)
                if argument is not None:
                    argument.accounting = {
                        **dict(argument.accounting or {}),
                        "loop_scope_inner_transition": (
                            declared_inner, resident,
                            "completed_module_latch_projection",
                        ),
                    }

            # One outer value can seed several authored bindings in a
            # simultaneous update.  In that case there is no unique carried
            # generation and the concordance detector must retain the
            # collapse finding.  A unique row, however, names the exact Phi
            # that owns every in-scope read of the outer generation.
            inside = _blocks_between(function, str(header), str(latch))
            for outer, matching_rebinds in rebinds_by_outer.items():
                if len(matching_rebinds) != 1:
                    continue
                rebind = matching_rebinds[0]
                carried = int(rebind["carried"])
                if outer == carried:
                    continue
                phi = _carried_phi(function, str(header), carried)
                if phi is None or phi.res is None:
                    continue

                # Recover the declaration cell whose three generations name
                # this exact outer/carried/inner relation.  Its row key may
                # carry the authored binding identity, so match facts rather
                # than weakening that identity back to the outer value id.
                loop_page_decl = book.registry.page("loop_scope")
                loop_page = book.page(loop_page_decl)
                declaration_scope = (
                    loop_scope_owner(function_name), loop_node_id,
                )
                generation_cells = []
                for row in loop_page.scope_rows(declaration_scope[0]):
                    if not (
                        isinstance(row, tuple) and len(row) == 3
                        and row[:2] == declaration_scope
                        and row[2] != "boundary"
                    ):
                        continue
                    generations = dict(loop_page.history(row))
                    if (
                        int(generations.get(OUTER, -1)) == outer
                        and int(generations.get(CARRIED, -1)) == carried
                        and int(generations.get(INNER, -1))
                        == int(rebind.get("declared_inner", rebind["inner"]))
                    ):
                        cell = book.latest_ref(loop_page_decl, row)
                        if cell is not None:
                            generation_cells.append(cell)
                if len(generation_cells) != 1:
                    continue

                from .concordance_declarations import (
                    CONTROL_SSA_FINISH, LOOP_RESULT_USE_REBINDING,
                )
                from .ssa_record_return_state import (
                    function_scope_of, ssa_block_identity_cell,
                    ssa_value_identity_cell,
                )

                boundary_cell = book.latest_ref(
                    loop_page_decl, (*declaration_scope, "boundary"),
                )
                value_cells = tuple(
                    ssa_value_identity_cell(function, value_id, book=book)
                    for value_id in (outer, carried)
                )
                for block_name, block in function.blocks.items():
                    if str(block_name) not in inside:
                        continue
                    for instruction_index, instruction in enumerate(block.instrs):
                        if str(instruction.op).casefold() == "phi":
                            continue
                        resolved_args = list(instruction.args)
                        for argument_index, argument in enumerate(instruction.args):
                            if int(argument.id) != outer:
                                continue
                            row = (
                                function_scope_of(function), str(block_name),
                                int(instruction_index), int(argument_index),
                            )
                            sources = tuple(dict.fromkeys(cell for cell in (
                                boundary_cell,
                                generation_cells[0],
                                *value_cells,
                                ssa_block_identity_cell(
                                    function, str(block_name), book=book,
                                ),
                                ssa_block_identity_cell(
                                    function, str(header), book=book,
                                ),
                            ) if cell is not None))
                            if not sources:
                                continue
                            fact = (
                                outer, carried,
                                ("declared_loop_scope", loop_node_id,
                                 str(header), str(latch)),
                            )
                            existing = book.latest_ref(
                                LOOP_RESULT_USE_REBINDING, row,
                            )
                            if not (
                                existing is not None
                                and book.page(
                                    LOOP_RESULT_USE_REBINDING
                                ).latest(row) == fact
                                and book.edges_into(existing)
                            ):
                                book.post(
                                    LOOP_RESULT_USE_REBINDING, row, fact,
                                    stage=CONTROL_SSA_FINISH,
                                    provenance=Derived(sources),
                                    mode=Mode.REVISE,
                                )
                            resolved_args[argument_index] = phi.res
                            receipts.append({
                                "function": str(function_name),
                                "loop_node_id": loop_node_id,
                                "block": str(block_name),
                                "instruction_index": int(instruction_index),
                                "argument_index": int(argument_index),
                                "outer": outer,
                                "carried": carried,
                                "reason": "declared_loop_scope_outer_read",
                            })
                        instruction.args = resolved_args
    if receipts:
        metadata = getattr(module, "metadata", None)
        if metadata is not None:
            metadata["loop_scope_inner_reconciliations"] = tuple(receipts)
    return tuple(receipts)


def concord_compiler_frame_formals(module: Any) -> tuple[dict, ...]:
    """Account for hidden formals proved to be compiler-owned at every call.

    A structural value such as a tensor dtype can survive specialization as a
    scalar formal even when it is not an authored parameter.  If every exact
    incoming call position supplies linked frame storage, the concordance can
    classify that formal without guessing from its numerical dtype or use.
    """

    functions = getattr(module, "functions", {}) or {}
    book = identity_book(module)
    from .concordance_declarations import (
        CALL_RECORD, FORMAL_ACTUAL_OCCURRENCE, FORMAL_STORAGE_RESOLUTION,
        FRAME_BINDING, FRAME_SOURCE_CELL_ABSENT, REGION_SIGNATURE, SSA_VALUE,
    )
    # Persisted books own their vocabulary. Add this scoped relation through
    # the registry API, keeping earlier six-field rows intact and unclaimed.
    occurrence_page = book.registry.declare_page(
        FORMAL_ACTUAL_OCCURRENCE.name, FORMAL_ACTUAL_OCCURRENCE.row_fields,
        FORMAL_ACTUAL_OCCURRENCE.fact_type,
    )

    def caller_scope(function: Any) -> str:
        return str(
            function.metadata.get("tensor_shape_concordance_scope")
            or function.name
        )

    incoming_page = book.page(occurrence_page)
    for caller_name, caller in functions.items():
        # Repository-SSA composition retains imported functions beside the
        # current unit, but their identity rows remain owned by the book that
        # accepted that component.  Re-publishing an imported caller's
        # internal calls on the composing book loses the accepted SSA_VALUE
        # edges (the decimal-glyph dependency has nine planned-region feeds)
        # and manufactures raw duplicate occurrence facts.  Current-unit
        # callers still publish calls into imported callees below; ownership
        # follows the caller that contains the occurrence.
        caller_book = dict(
            caller.metadata.get("accepted_ssa_component") or {}
        ).get("identity_book")
        if caller_book is not None and caller_book is not book:
            continue
        scope = caller_scope(caller)
        for block_name, block in caller.blocks.items():
            for instruction_index, instruction in enumerate(block.instrs):
                if instruction.op not in {"Call", "call"}:
                    continue
                callee_name = str(instruction.attributes.get("callee") or "")
                callee = functions.get(callee_name)
                if callee is None or len(instruction.args) != len(callee.args):
                    continue
                for position, (formal, actual) in enumerate(zip(
                    callee.args, instruction.args,
                )):
                    # One law can be lowered repeatedly on its book. The
                    # control scope, already minted by the SSA builder, owns
                    # this call occurrence; a physical function name alone
                    # conflates its separately minted scalar/frame actuals.
                    actual_cell = book.latest_ref(
                        SSA_VALUE, (scope, int(actual.id)),
                    )
                    # A planned-region call is itself the materialization of
                    # a retained RegionSignature.  Some of its feed ids are
                    # graph identities rather than separately minted SSA
                    # values; the signature is the exact statement that this
                    # occurrence consumes that feed at this region boundary.
                    if actual_cell is None:
                        region_index = instruction.attributes.get(
                            "region_index"
                        )
                        signature_row = (
                            scope, int(region_index)
                        ) if region_index is not None else None
                        signature = (
                            None if signature_row is None
                            else book.page(REGION_SIGNATURE).latest(
                                signature_row
                            )
                        )
                        if (
                            signature is not None
                            and position < len(signature.feeds)
                            and int(signature.feeds[position])
                            == int(actual.id)
                        ):
                            actual_cell = book.latest_ref(
                                REGION_SIGNATURE, signature_row,
                            )
                    if actual_cell is None:
                        callsite_id = instruction.attributes.get(
                            "plan_callsite_id"
                        )
                        call_table = getattr(module, "call_table", None)
                        records = (
                            () if call_table is None
                            else call_table.get(str(caller_name), ())
                        )
                        matching_record = next((
                            record for record in records
                            if callsite_id is not None
                            and int(record.callsite_id) == int(callsite_id)
                            and str(record.callee_symbol or record.callee_name)
                            == str(callee_name)
                        ), None)
                        if matching_record is not None:
                            actual_cell = book.latest_ref(
                                CALL_RECORD,
                                (call_table.owner, str(caller_name)),
                            )
                    book.post(
                        occurrence_page,
                        (
                            callee_name, int(formal.id), str(caller_name),
                            str(block_name), int(instruction_index),
                            int(position), scope,
                        ),
                        int(actual.id),
                        stage=FRAME_BINDING,
                        provenance=(
                            Derived((actual_cell,)) if actual_cell is not None
                            else Unsourced(FRAME_SOURCE_CELL_ABSENT)
                        ),
                        mode=Mode.CONCORD,
                    )

    receipts: list[dict] = []
    for function_name, function in functions.items():
        metadata = function.metadata
        named = {
            int(value_id)
            for _name, value_id in metadata.get("parameter_names", ())
        }
        recorded = {
            int(item["value_id"])
            for key in ("storage_formals", "closure_formals", "member_formals")
            for item in (metadata.get(key, ()) or ())
            if isinstance(item, Mapping) and item.get("value_id") is not None
        }
        for formal in function.args:
            formal_id = int(formal.id)
            accounting = dict(formal.accounting or {})
            constructor_field_value = (
                accounting.get("record_constructor_value") is not None
                and accounting.get("program_abi_parameter") is None
            )
            if (
                formal_id in named
                or formal_id in recorded
                or accounting.get("program_abi_parameter") not in {None, ""}
                or (
                    not constructor_field_value
                    and any(
                        accounting.get(key) not in {None, ""}
                        for key in ("program_abi_field",)
                    )
                )
                or any(
                    accounting.get(key) not in {None, ""}
                    for key in (
                        "linked_call_frame_storage",
                        "compiler_frame_storage",
                        "returned_record_storage",
                    )
                )
            ):
                continue
            source_rows = tuple(
                row for row in incoming_page.scope_rows(str(function_name))
                if int(row[1]) == formal_id
                and str(row[2]) in functions
                and row[6] == caller_scope(functions[str(row[2])])
            )
            source_facts = tuple(
                incoming_page.latest(row) for row in source_rows
            )
            source_actuals = tuple(
                functions[str(row[2])].blocks[str(row[3])]
                .instrs[int(row[4])].args[int(row[5])]
                for row in source_rows
            )
            if (
                not source_facts
                or any(
                    int(actual.id) != int(fact)
                    for actual, fact in zip(source_actuals, source_facts)
                )
                or not all(
                    (actual.accounting or {}).get(
                        "linked_call_frame_storage"
                    )
                    or (actual.accounting or {}).get(
                        "compiler_frame_storage"
                    )
                    for actual in source_actuals
                )
            ):
                continue
            source_receipts = tuple(
                (str(row[2]), int(fact))
                for row, fact in zip(source_rows, source_facts)
            )
            formal.accounting = {
                **accounting,
                "compiler_frame_storage": str(function_name),
                "compiler_frame_sources": source_receipts,
            }
            storage_entry = {
                "value_id": formal_id,
                "dtype": str(formal.dtype or "unknown"),
                "shape": tuple(formal.shape or ()),
                "kind": "compiler_frame_storage",
                "sources": source_receipts,
            }
            prior = tuple(metadata.get("storage_formals", ()) or ())
            metadata["storage_formals"] = (
                *prior,
                *( () if storage_entry in prior else (storage_entry,) ),
            )
            receipt = {
                "function": str(function_name),
                "formal_id": formal_id,
                "sources": source_receipts,
                "kind": "compiler_frame_storage",
            }
            receipts.append(receipt)
            row = (str(function_name), formal_id)
            source_cells = tuple(dict.fromkeys(
                cell for cell in (
                    *(
                        book.latest_ref(occurrence_page, source_row)
                        for source_row in source_rows
                    ),
                    book.latest_ref(
                        SSA_VALUE,
                        (caller_scope(function), formal_id),
                    ),
                    book.latest_ref(FORMAL_STORAGE_RESOLUTION, row),
                )
                if cell is not None
            ))
            book.post(
                FORMAL_STORAGE_RESOLUTION,
                row,
                ("compiler_frame_storage", source_receipts),
                stage=FRAME_BINDING,
                provenance=Derived(source_cells),
                mode=Mode.REVISE,
            )
    if receipts:
        getattr(module, "metadata", {})[
            "compiler_frame_formal_reconciliations"
        ] = tuple(receipts)
    return tuple(receipts)


def concord_program_abi_frame_transitions(module: Any) -> tuple[dict, ...]:
    """Retire provisional frame leases once a formal has a ProgramABI field.

    Linked-call completion can create an anonymous caller-owned workspace
    before record-field propagation reaches that level of the call graph.  If
    the same formal is later proved to be a declared ProgramABI field, keeping
    both descriptions invents two owners for one physical value.  The field
    contract is the completed identity; the frame lease was only the means by
    which the still-anonymous value reached the function.

    This is deliberately an identity transition, not a detector exemption.
    The obsolete accounting and matching ``storage_formals`` declaration are
    removed together and the exact before/after fact is written to the shared
    book.
    """

    provisional_keys = {
        "linked_call_frame_storage",
        "propagated_formal_id",
        "restored_argument_binding",
        "split_from_result_storage",
    }
    receipts: list[dict] = []
    book = identity_book(module)
    page = book.page("program_abi_frame_transition")
    for function_name, function in (
        getattr(module, "functions", {}) or {}
    ).items():
        retired_ids: set[int] = set()
        for formal in function.args:
            accounting = dict(formal.accounting or {})
            field = accounting.get("program_abi_field")
            lease = accounting.get("linked_call_frame_storage")
            if field is None or lease in {None, ""}:
                continue
            # Returned-record slots intentionally are caller-provided output
            # storage.  Their dual role is already explicit and is not this
            # provisional-input transition.
            if accounting.get("returned_record_storage") is not None:
                continue
            formal_id = int(formal.id)
            retired = tuple(
                (key, accounting[key])
                for key in sorted(provisional_keys)
                if key in accounting
            )
            resolved = {
                key: value for key, value in accounting.items()
                if key not in provisional_keys
            }
            receipt = {
                "function": str(function_name),
                "formal_id": formal_id,
                "program_abi_record": accounting.get("program_abi_record"),
                "program_abi_field": str(field),
                "retired": retired,
                "reason": "program_abi_field_supersedes_provisional_frame_lease",
            }
            row = (str(function_name), formal_id)
            fact = (
                accounting.get("program_abi_record"), str(field), retired,
            )
            incumbent = page.latest(row)
            if incumbent is None:
                # The transition stands on the formal's own ``ssa_value``
                # cell under the function's control scope.
                from .concordance_declarations import (
                    FRAME_BINDING, FRAME_SOURCE_CELL_ABSENT,
                    PROGRAM_ABI_FRAME_TRANSITION, SSA_VALUE,
                )
                from .ssa_record_return_state import function_scope_of

                formal_cell = book.latest_ref(
                    SSA_VALUE, (function_scope_of(function), formal_id),
                )
                book.post(
                    PROGRAM_ABI_FRAME_TRANSITION, row,
                    fact, stage=FRAME_BINDING,
                    provenance=(
                        Derived((formal_cell,)) if formal_cell is not None
                        else Unsourced(FRAME_SOURCE_CELL_ABSENT)
                    ),
                    mode=Mode.CONCORD,
                )
            elif tuple(incumbent) != fact:
                raise ValueError(
                    "ProgramABI/frame transition disagreement for "
                    f"{row!r}: recorded={incumbent!r}, proposed={fact!r}"
                )
            formal.accounting = resolved
            retired_ids.add(formal_id)
            receipts.append(receipt)
        if retired_ids:
            metadata = function.metadata
            storage_formals = tuple(metadata.get("storage_formals", ()) or ())
            metadata["storage_formals"] = tuple(
                item for item in storage_formals
                if not (
                    isinstance(item, Mapping)
                    and item.get("value_id") is not None
                    and int(item["value_id"]) in retired_ids
                )
            )
    if receipts:
        getattr(module, "metadata", {})[
            "program_abi_frame_transitions"
        ] = tuple(receipts)
    return tuple(receipts)


def argument_binding_history(
    page: Any, callee_symbol: Any, formal_id: int, *, rows: Any = None,
) -> tuple[tuple[Any, Any], ...]:
    """Every ``(callsite, fact)`` recorded for one callee formal on the
    ``argument_binding`` page, in page order.

    Step 7 keys one row per ``(callee, formal, callsite)``; rows written
    before it key ``(callee, formal, "binding")`` with the callsite as the
    column.  ``rows`` restricts the read to those rows (a caller iterating
    the page); by default the formal's rows are read from the page's scope
    index.  An ``Unresolved`` fact is returned as such.
    """

    callee_symbol = str(callee_symbol)
    formal_id = int(formal_id)
    if rows is None:
        rows = tuple(
            row for row in page.scope_rows(callee_symbol)
            if len(row) == 3 and row[1] == formal_id
        )
    history: list[tuple[Any, Any]] = []
    for row in rows:
        if not (isinstance(row, tuple) and len(row) == 3):
            continue
        if str(row[0]) != callee_symbol or int(row[1]) != formal_id:
            continue
        for column, fact in page.history(row):
            history.append((column if row[2] == "binding" else row[2], fact))
    return tuple(history)


def materializing_binding_kind(
    book: Any, callee_symbol: Any, formal_id: int, source_id: int, kind: Any,
) -> str:
    """The kind that can materialize one caller slot, across every callsite.

    A frame binding's kind is not a label on the slot; it selects which
    machinery may supply the argument.  ``caller_storage`` can restore a slot
    a structural cleanup removed, ``caller_alias`` and ``caller_value``
    cannot.  Each callsite decides the kind from its own private map in its
    own order, so two callsites can name the same slot under different kinds
    and the one that cannot materialize it reports ``missing_<kind>`` and
    refuses the call -- after which the callee's output is never written and
    whatever reads it gets uninitialized memory, with no shortfall anywhere.

    The decisions are already written to the shared page precisely so this is
    answerable.  If any callsite proved the slot is caller storage, that is
    what it is, and every callsite naming it gets the kind that works.
    """

    from .concordance_declarations import ARGUMENT_BINDING
    from .concordance_declarations import ARGUMENT_BINDING_RESOLUTION

    page = book.page("argument_binding")
    callee_symbol = str(callee_symbol)
    for row in page.scope_rows(callee_symbol):
        if not (len(row) == 3 and row[1] == int(formal_id)):
            continue
        for column, fact in page.history(row):
            if not (isinstance(fact, tuple) and len(fact) == 2):
                continue
            recorded_kind, recorded_source = fact
            if not isinstance(recorded_source, int):
                continue
            if int(recorded_source) != int(source_id):
                continue
            if str(recorded_kind) == "caller_storage":
                resolution_row = (
                    callee_symbol, int(formal_id), int(source_id),
                )
                resolution = (
                    "caller_storage", str(kind), "materializing_binding_kind",
                )
                stored = book.pages.get(ARGUMENT_BINDING_RESOLUTION.name)
                if stored is None or stored.latest(resolution_row) != resolution:
                    # DERIVED from the ``argument_binding`` cell that proved
                    # the slot is caller storage.
                    _post_or_unsourced(
                        book, ARGUMENT_BINDING_RESOLUTION, resolution_row,
                        resolution,
                        book.registry.declare_stage("materializing_binding_kind"),
                        (Ref(ARGUMENT_BINDING, row, column),),
                        ARGUMENT_BINDING_RESOLUTION_CAUSE_NOT_ON_BOOK,
                    )
                return "caller_storage"
    return str(kind)


def record_proven_shape(
    function: Any, value_id: int, extents: Any, dtype: Any,
    level: int | None = 0, *, sources: Iterable[Ref] = (),
) -> None:
    """Record extents proven for one value identity at one causal level.

    Only EXTENTS are recorded.  An empty shape is both a rank-0 scalar and
    what a query returns when recovery stops, so storing it would let an
    unknown win a race against a real shape.  A deeper level supersedes a
    shallower one because it was derived from more of the program; the same
    answer arriving deeper is recorded at its own level so the page shows how
    far it has been confirmed.

    The proof is posted DERIVED from ``sources`` (the cells the caller proved
    the extents from) and the value's shape-state cell; with neither it is
    recorded ``Unsourced``.
    """

    extents = tuple(int(extent) for extent in (extents or ()))
    if not extents:
        return
    book = current_identity_book()
    page = book.page(PROVEN_SHAPE_PAGE)
    row = (_shape_key(function), int(value_id))
    sources = tuple(cell for cell in sources if isinstance(cell, Ref))
    recorded = page.history(row)
    recorded_levels = tuple(
        int(stored[3])
        if isinstance(stored, tuple) and len(stored) > 3
        and stored[0] in {"proven", "conflicting"}
        else int(column)
        for column, stored in recorded
        if not (
            isinstance(stored, tuple) and stored
            and stored[0] == "invalidated"
        )
    )
    target_level = (
        max(recorded_levels, default=0)
        if level is None else int(level)
    )
    fact = ("proven", extents, str(dtype or "float64"), target_level)
    if not recorded:
        state_ref = book.latest_ref(SHAPE_STATE_PAGE, row)
        _post_or_unsourced(
            book, PROVEN_SHAPE_PAGE, row, fact, PROVEN_SHAPE_STAGE,
            (*sources, *(() if state_ref is None else (state_ref,))),
            PROVEN_SHAPE_SOURCE_NOT_ON_BOOK,
        )
        return
    deepest = max(recorded, key=lambda entry: int(entry[0]))
    if isinstance(deepest[1], tuple) and deepest[1] and (
        deepest[1][0] == "invalidated"
    ):
        # An upstream identity changed after this proof was derived.  The next
        # descriptor query is a new proof generation, not a disagreement with
        # the invalidated fact.  Keep both events in causal order.
        # Program: a call edge re-visited by the second exact-shape
        # publication.  Wrong record: a level bump over the same single cell
        # (the query passed level 13, the row was posted at max+1 = 12, then
        # 13 over the same cell: a level change is not a cause).  Rule: the
        # row is posted at the passed level when it exceeds max+1, so the
        # next query with that level is a true repeat.
        target_level = max(recorded_levels, default=-1) + 1
        if level is not None:
            target_level = max(target_level, int(level))
        fact = ("proven", extents, str(dtype or "float64"), target_level)
        state_ref = book.latest_ref(SHAPE_STATE_PAGE, row)
        _post_or_unsourced(
            book, PROVEN_SHAPE_PAGE, row, fact, PROVEN_SHAPE_STAGE,
            (*sources, *(() if state_ref is None else (state_ref,))),
            PROVEN_SHAPE_SOURCE_NOT_ON_BOOK,
        )
        return
    deepest_level = (
        int(deepest[1][3])
        if isinstance(deepest[1], tuple) and len(deepest[1]) > 3
        else int(deepest[0])
    )
    if (
        tuple(deepest[1][1]) == extents
        and target_level < deepest_level
    ):
        # Causal depth is monotone for one proven extent.  A later caller
        # using the default level must not downgrade an answer already
        # confirmed more deeply; retain that depth and let the duplicate
        # source check below decide whether any new evidence was added.
        target_level = deepest_level
        fact = ("proven", extents, str(dtype or "float64"), target_level)
    if (
        tuple(deepest[1][1]) == extents
        or target_level > deepest_level
    ):
        # The exact shape-state cell and the caller's cells are the
        # authority for this projection.  Repeating the same assertion over
        # unchanged cells (every cell it would derive from is already an
        # edge into the row's latest cell) adds no causal information and is
        # not restated; a newer cell records an ordinary revision.
        state_ref = book.latest_ref(SHAPE_STATE_PAGE, row)
        latest_ref = book.latest_ref(PROVEN_SHAPE_PAGE, row)
        cells = (*sources, *(() if state_ref is None else (state_ref,)))
        # Program: a call edge re-visited by the second exact-shape
        # publication (_publish_exact_ssa_call_shapes) with the same four
        # cells.  Wrong record: the incumbent was ``conflicting`` and the
        # repeat posted ``proven`` over the same cells (a re-observation of
        # a conflict taken for a re-confirmation of a proof), which the book
        # refuses as a causeless REVISE.  Rule: the same extents, dtype and
        # level over cells that are already edges into the latest cell is
        # not restated, whatever the latest tag; a repeat never flips
        # ``conflicting`` to ``proven``; a NEW cell at the same level
        # re-posts ``conflicting`` with the new cell set.
        deepest_dtype = deepest[1][2] if len(deepest[1]) > 2 else None
        repeats_incumbent = (
            deepest[1][0] in {"proven", "conflicting"}
            and tuple(deepest[1][1]) == extents
            and deepest_dtype in {fact[2], str(dtype or "")}
            and target_level == deepest_level
        )
        if repeats_incumbent and latest_ref is not None and cells:
            recorded_sources = {
                source for source, _stage in book.edges_into(latest_ref)
            }
            if all(cell in recorded_sources for cell in cells):
                return
        if repeats_incumbent and deepest[1][0] == "conflicting":
            fact = ("conflicting", extents, str(dtype or ""), target_level)
        _post_or_unsourced(
            book, PROVEN_SHAPE_PAGE, row, fact, PROVEN_SHAPE_STAGE,
            cells, PROVEN_SHAPE_SOURCE_NOT_ON_BOOK,
        )
        return
    state_ref = book.latest_ref(SHAPE_STATE_PAGE, row)
    _post_or_unsourced(
        book, PROVEN_SHAPE_PAGE, row,
        ("conflicting", extents, str(dtype or ""), target_level),
        PROVEN_SHAPE_STAGE,
        (*sources, *(() if state_ref is None else (state_ref,))),
        PROVEN_SHAPE_SOURCE_NOT_ON_BOOK,
    )


def invalidate_proven_shape(
    function: Any, value_id: int, source_id: int, reason: Any,
    *, source_cells: tuple[Ref, ...] = (),
) -> None:
    """Withdraw a derived shape after one of its exact dependencies changes."""

    book = current_identity_book()
    page = book.page(PROVEN_SHAPE_PAGE)
    row = (_shape_key(function), int(value_id))
    recorded = page.history(row)
    # A transformation source may be a structured identity (a callee return,
    # a descriptor root), not only a graph value id.
    if isinstance(source_id, int) or not isinstance(source_id, tuple):
        source_id = int(source_id)
    state_ref = invalidate_shape_transformation(
        function, int(value_id), source_id, reason,
        source_cells=source_cells,
    )
    # Scalar identities have no ``proven_shape`` row: this page stores only
    # non-empty extents.  Loop continuation rewiring used to manufacture a
    # raw invalidation for every such scalar (the UTF-8 glyph decoder alone
    # made ten), even though there was no shape proof to withdraw.
    if not recorded:
        return
    cells = tuple(dict.fromkeys((
        *source_cells, *(() if state_ref is None else (state_ref,)),
    )))
    _post_or_unsourced(
        book, PROVEN_SHAPE_PAGE, row,
        ("invalidated", source_id, str(reason)),
        book.registry.declare_stage(str(reason)), cells,
        SHAPE_SOURCE_NOT_ON_BOOK,
    )


def proven_shape_contract_of(
    function: Any, value_id: int,
) -> tuple[tuple[int, ...], str] | None:
    """The shape and dtype proven for one exact value identity, or None.

    This is the question every store was answering separately.  A row that
    two derivations contradict at the same causal level answers nothing --
    concurrent and genuinely in conflict is not a fact.
    """

    page = current_identity_book().page("proven_shape")
    row = (_shape_key(function), int(value_id))
    recorded = page.history(row)
    if not recorded:
        return None
    deepest = max(recorded, key=lambda entry: int(entry[0]))[1]
    if not isinstance(deepest, tuple) or deepest[0] != "proven":
        return None
    return (
        tuple(int(extent) for extent in deepest[1]),
        str(deepest[2]),
    )


def proven_shape_of(function: Any, value_id: int) -> tuple[int, ...] | None:
    """The extents proven for this exact identity, or None."""

    contract = proven_shape_contract_of(function, value_id)
    return None if contract is None else contract[0]


def shape_store_report(book: Any, stores: Any = None) -> str:
    """Where the stores of one shape disagree, as a report.

    Kept because it turned a day of inference into three named rows: a value
    whose stores differ is a row, not a hunt.
    """

    names = tuple(stores or ("node", "linked", "ssa"))
    pages = {name: book.page(f"shape.{name}") for name in names}
    pages["proven"] = book.page("proven_shape")
    rows: set = set()
    for page in pages.values():
        rows.update(page.rows())

    def extents(name: str, row: Any):
        page = pages[name]
        if row not in set(page.rows()):
            return None
        recorded = page.history(row)
        if not recorded:
            return None
        fact = max(recorded, key=lambda entry: int(entry[0]))[1]
        if name == "proven":
            return tuple(fact[1]) if fact[0] == "proven" else None
        return tuple(fact)

    lines = []
    disagreeing = []
    for row in sorted(rows, key=str):
        present = {}
        for name in pages:
            value = extents(name, row)
            if value is not None:
                present[name] = value
        if len({tuple(value) for value in present.values()}) > 1:
            disagreeing.append((row, present))
    lines.append(
        f"shape stores: {len(disagreeing)} disagreeing of {len(rows)} "
        "value identit(ies)"
    )
    for row, present in disagreeing[:10]:
        lines.append(f"  {render_row(row)} {present}")
    for name, page in pages.items():
        lines.append(f"  store {name}: {len(page.rows())} row(s)")
    return "\n".join(lines)


def row_value_id(row: Any) -> int | None:
    """The SSA value id a page row is about, when it names one.

    Pages key rows differently -- ``(function, value id, field)`` for a
    mutation page, a bare id for a reference count -- so presentation takes
    the first integer it finds and says nothing when there is none, rather
    than guessing a position that happens to work for one page's shape.
    """
    if isinstance(row, int):
        return int(row)
    if isinstance(row, tuple):
        for item in row:
            if isinstance(item, int):
                return int(item)
    return None


#: Retained name: the latch is always closed; the variable no longer opens
#: it (a raw write or a post without a source is refused unconditionally).
LATCH_ENV = "TURING_IDENTITY_LATCH"
LOG_LEVEL_ENV = "TURING_IDENTITY_LOG_LEVEL"
LOG_PRESET_ENV = "TURING_IDENTITY_LOG_LZMA_PRESET"
#: Longest fact text a FACTS-level row keeps before the length marker.
FACTS_FACT_WIDTH = 240
_DEFAULT_LZMA_PRESET = 3


def parse_identity_log_level(value: Any) -> IdentityLogLevel | None:
    """``off|summary|facts|full`` (any case), an ``IdentityLogLevel`` or its
    int; anything else is None (the caller falls back to its default)."""
    if isinstance(value, IdentityLogLevel):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return IdentityLogLevel(value)
        except ValueError:
            return None
    if isinstance(value, str):
        return IdentityLogLevel.__members__.get(value.strip().upper())
    return None


def resolve_identity_log_level(
    explicit: Any = None, *, ok: bool,
) -> IdentityLogLevel:
    """The level this compile logs at: the caller's keyword, else the
    ``TURING_IDENTITY_LOG_LEVEL`` environment variable, else the default --
    SUMMARY for an OK compile, FULL for a FAILED one (the receipt that
    explains the failure)."""
    import os

    for candidate in (explicit, os.environ.get(LOG_LEVEL_ENV)):
        level = parse_identity_log_level(candidate)
        if level is not None:
            return level
    return IdentityLogLevel.SUMMARY if ok else IdentityLogLevel.FULL


def identity_log_lzma_preset() -> int:
    """``TURING_IDENTITY_LOG_LZMA_PRESET``: int 0-9, optional ``e`` suffix
    for extreme; default 3 (fast and near the best on measured logs)."""
    import lzma
    import os

    text = os.environ.get(LOG_PRESET_ENV, "").strip().lower()
    extreme = text.endswith("e")
    digits = text[:-1] if extreme else text
    if digits.isdigit() and 0 <= int(digits) <= 9:
        return int(digits) | (lzma.PRESET_EXTREME if extreme else 0)
    return _DEFAULT_LZMA_PRESET


def _page_rows_level(book: IdentityBook, name: str) -> IdentityLogLevel:
    declared = book.registry.pages.get(name)
    return IdentityLogLevel.FACTS if declared is None else declared.rows_level


def _truncate_fact(text: str) -> str:
    if len(text) <= FACTS_FACT_WIDTH:
        return text
    return f"{text[:FACTS_FACT_WIDTH]}...[{len(text)} chars]"


def iter_identity_book_lines(
    book: IdentityBook,
    level: IdentityLogLevel = IdentityLogLevel.FULL,
    *,
    extra_lines: Iterable[str] = (),
) -> Iterable[str]:
    """The book's log, one line at a time, pages sorted by name.

    FULL is the dense log (every row, its full span history) and is exactly
    the lines ``render_identity_book`` joins.  FACTS prints a row only on a
    page whose ``rows_level`` is at most FACTS, and then only its LAST span's
    fact, truncated.  SUMMARY prints the header, one line per page and the
    book's unsourced counts, then ``extra_lines`` (a caller's own findings).
    OFF prints nothing."""
    level = IdentityLogLevel(level)
    if level is IdentityLogLevel.OFF:
        return
    full = level is IdentityLogLevel.FULL
    yield f"identity book: {len(book.pages)} page(s)"
    if not full:
        yield f"log level: {level.name.lower()}"
    _prepare_log(book, level)
    from .shell_telemetry import compile_iter
    for page_name in compile_iter(sorted(book.pages), "identity-log-pages", label="Identity log pages"):
        page = dict.get(book.pages, page_name)
        rows_level = _page_rows_level(book, page_name)
        withheld = level < rows_level
        # (A spilled page is counted from its segment: a summary log reads
        # nothing back.)
        header = f"[{page_name}] {page.row_count()} row(s), {page.cell_count()} cell(s)"
        if not full and withheld and level >= IdentityLogLevel.FACTS:
            header += f" (rows withheld below {rows_level.name.lower()})"
        yield header
        if level < IdentityLogLevel.FACTS or withheld:
            continue
        # (A spilled page is borrowed for its rows and let go after: the log
        # of a book never holds the whole book.)
        with book.borrowed(page_name) as page:
            yield from _page_lines(page, page_name, full)
    if full:
        return
    yield f"unsourced: latch {book.latch.name}"
    from .shell_telemetry import compile_substage, compile_count, compile_count_close
    compile_substage("identity log: unsourced tally")
    tally: Counter = Counter()
    for count, (page_ref, _row, reason, stage) in enumerate(book.unsourced_rows(), 1):
        tally[(getattr(page_ref, "name", page_ref), stage.name, reason.name)] += 1
        if count % 1024 == 0:
            compile_count("identity-log-tally", count, label="Unsourced facts tallied")
    compile_count_close("identity-log-tally")
    for (page_name, stage_name, reason_name), count in sorted(tally.items()):
        yield f"  {page_name} stage={stage_name} reason={reason_name}: {count}"
    yield from extra_lines


def _prepare_log(book: IdentityBook, level: IdentityLogLevel) -> None:
    """Before a log that prints a private edge page's rows, read back the rows
    spilled pages own: the page is then whole for the whole log, and the
    receipts of the reads are posted before the first line, not under it."""
    if book.__dict__.get("_spill") is None:
        return
    from .identity_spill import PARTITIONED, make_whole

    from .shell_telemetry import compile_iter
    for part in compile_iter(PARTITIONED, "identity-log-prepare", label="Identity log spill pages prepared"):
        page = dict.get(book.pages, part)
        if page is None or not page.__dict__.get("_detached"):
            continue
        prints = (
            level >= IdentityLogLevel.FACTS
            and level >= _page_rows_level(book, part)
        )
        # (Below FULL the log ends with the unsourced tally, which reads the
        # unsourced rows of every page.)
        tally = part == UNSOURCED_PAGE.name and level is not IdentityLogLevel.FULL
        if prints or tally:
            make_whole(book, part, "log")


def _page_lines(page: IdentityPage, page_name: str, full: bool):
    """One page's row lines (the body of ``iter_identity_book_lines``)."""
    # Rows gathered under the id group they belong to, so one page's
    # entries read as the few spaces they actually span rather than as
    # one undifferentiated list.  A row whose key names no id keeps its
    # place under "unkeyed" instead of being dropped or invented into
    # a group.
    from .shell_telemetry import compile_substage, compile_count, compile_count_close
    compile_substage(f"identity log: group {page_name} rows")
    rows = page.rows()
    grouped = 0
    compile_count("identity-log-rows", 0, label="Identity log rows grouped")
    by_group: dict[str, list[Any]] = {}
    for row in rows:
        grouped += 1
        if grouped % 1024 == 0:
            compile_count("identity-log-rows", grouped)
        value_id = row_value_id(row)
        group = (
            "unkeyed" if value_id is None
            else (group_by_prefix([value_id])[0].label)
        )
        by_group.setdefault(group, []).append(row)
    compile_count("identity-log-rows", grouped)
    compile_count_close("identity-log-rows")
    compile_substage(f"identity log: write {page_name} histories")
    for group in sorted(by_group):
        group_rows = by_group[group]
        if len(by_group) > 1:
            yield f"  ({group}) {len(group_rows)} row(s)"
        for row in group_rows:
            spans = page.spans(row)
            if full:
                trail = " -> ".join(
                    f"{start}..{end}={fact}" for start, end, fact in spans
                )
                yield f"  {render_row(row)}: {trail}"
                continue
            start, end, fact = spans[-1]
            more = f" [{len(spans)} spans]" if len(spans) > 1 else ""
            yield (
                f"  {render_row(row)}: {start}..{end}="
                f"{_truncate_fact(str(fact))}{more}"
            )


def render_identity_book(book: IdentityBook) -> str:
    """Every page, every row, its full span history -- the dense log."""
    return "\n".join(iter_identity_book_lines(book, IdentityLogLevel.FULL))


def write_identity_log(
    book: Any,
    path_stem: Any,
    *,
    level: Any = IdentityLogLevel.FULL,
    kind: str = "book",
    extra_lines: Iterable[str] = (),
) -> str | None:
    """Stream ``book``'s log at ``level`` into ``<path_stem>.<kind>.log.xz``.

    The lines go through ``lzma`` as they are produced (never one big
    string), into a temporary name that is ``os.replace``d onto the final
    name only when the stream completed: a crash leaves no half-written
    file under the final name.  The preset is ``identity_log_lzma_preset()``.
    Best-effort and silent: any failure returns None, never raises.
    Returns the final path, or None when nothing was written."""
    import lzma
    import os

    temporary = None
    try:
        level = parse_identity_log_level(level)
        if book is None or level is None or level is IdentityLogLevel.OFF:
            return None
        final = f"{os.fspath(path_stem)}.{kind}.log.xz"
        temporary = f"{final}.tmp{os.getpid()}"
        from .shell_telemetry import compile_substage
        compile_substage(f"identity log: stream {level.name.lower()} checkpoint")
        with lzma.open(
            temporary, "wt", encoding="utf-8", newline="\n",
            preset=identity_log_lzma_preset(),
        ) as handle:
            for line in iter_identity_book_lines(
                book, level, extra_lines=extra_lines,
            ):
                handle.write(line)
                handle.write("\n")
            compile_substage("identity log: finalize compressed stream")
        os.replace(temporary, final)
        temporary = None
        return final
    except Exception:
        return None
    finally:
        if temporary is not None:
            try:
                os.remove(temporary)
            except OSError:
                pass


def render_row(row: Any) -> str:
    """A page row with its ids named rather than spelled out in full.

    A flagged id is a nineteen-digit number; printing it raw makes the log
    unsearchable by the serial a reader actually has in hand (from a
    traceback, say) and unreadable at a glance.  Each integer in the row is
    rendered through :func:`id_space.label`, which leaves an unflagged id
    exactly as it was -- so nothing about legacy output changes -- and
    turns a flagged one into ``minted#1000013548``.
    """
    if isinstance(row, int):
        return id_label(row)
    if isinstance(row, tuple):
        # Nested tuples (an edge row holding two cell keys, a dependents
        # row holding an edge row) are rendered the same way, so the ids
        # inside them are labelled too.
        return (
            "(" + ", ".join(
                id_label(item) if isinstance(item, int) and not isinstance(item, bool)
                else render_row(item) if isinstance(item, tuple)
                else repr(item)
                for item in row
            ) + ")"
        )
    return repr(row)


@dataclass(frozen=True)
class NeighborhoodNode:
    """One cell of a failure neighborhood: where it is, how far from the
    seeds, the diffused mass it received, and its fact as the book holds it.
    ``omitted`` counts the neighbors this cell left out (-1: its edges could
    not be read, ``error`` says why)."""

    ref: Ref
    hop: int
    mass: float
    seed: bool
    omitted: int
    stamp: int | None
    fact: str
    error: str | None = None


@dataclass(frozen=True)
class CellNeighborhood:
    """The cells and edges within ``radius`` of a failure's seed cells.

    ``edges`` are ``(source, target, kind)`` where ``kind`` is the Stage that
    derived ``target`` from ``source`` or ``mint:<transform>``."""

    seeds: tuple[Ref, ...]
    nodes: tuple[NeighborhoodNode, ...]
    edges: tuple[tuple[Ref, Ref, str], ...]
    radius: int
    truncated: bool

    def lines(self, labels: Mapping[Ref, str] | None = None) -> tuple[str, ...]:
        index = {node.ref: f"n{i}" for i, node in enumerate(self.nodes)}
        out = [
            f"neighborhood: radius {self.radius}, {len(self.nodes)} cell(s), "
            f"{len(self.edges)} edge(s)"
            + (", cell cap reached" if self.truncated else "")
        ]
        for node in sorted(self.nodes, key=lambda item: (item.hop, -item.mass)):
            role = (labels or {}).get(node.ref)
            tag = "seed" if node.seed else f"hop {node.hop}"
            if role:
                tag += f" {role}"
            more = (
                f" (+{node.omitted} neighbor(s) not shown)"
                if node.omitted > 0 else
                f" (edges unreadable: {node.error})" if node.omitted < 0
                else ""
            )
            out.append(
                f"  {index[node.ref]} [{tag}] mass={node.mass:.3f} "
                f"{node.ref.page.name} {render_row(node.ref.row)}"
                f"@{node.ref.column} stamp={node.stamp} "
                f"fact={node.fact}{more}"
            )
        for source, target, kind in self.edges:
            out.append(
                f"  {index.get(source, '?')} --{kind}--> "
                f"{index.get(target, '?')}"
            )
        return tuple(out)


@dataclass(frozen=True)
class FailureClaim:
    """A post that failed: what it claimed, and the cells it stood on.

    ``seeds`` are ``(role, Ref)``: the row's incumbent cell(s) and the
    source cells of the proposed fact.  Held by the exception; the
    neighborhood is walked only where the failure is reported."""

    page: str
    row: tuple
    mode: str
    stage: str
    proposed: str
    seeds: tuple[tuple[str, Ref], ...]


def _attach_failure_claim(
    book: IdentityBook, error: BaseException, page: Page, row: tuple,
    fact: Any, stage: Stage, provenance: Any, mode: Mode,
) -> None:
    """Record, on ``error``, the claim ``post`` was refused for."""
    try:
        seeds: list[tuple[str, Ref]] = []
        incumbent = book.latest_ref(page, row)
        if incumbent is not None:
            seeds.append(("incumbent", incumbent))
        if isinstance(provenance, Derived):
            seeds.extend(("source", ref) for ref in provenance.cells)
        elif isinstance(provenance, Novel):
            seeds.extend(("operand", ref) for ref in provenance.operands)
        claims = error.__dict__.setdefault("concordance_claims", [])
        claims.append(FailureClaim(
            page=page.name, row=row, mode=getattr(mode, "name", str(mode)),
            stage=getattr(stage, "name", str(stage)),
            proposed=_truncate_fact(repr(fact)), seeds=tuple(seeds),
        ))
    except Exception:
        pass


#: The public name for a writer that refuses a claim itself (not through
#: ``post``) to hand the claim to the failure report.
attach_failure_claim = _attach_failure_claim


def annotate_failure_with_neighborhood(
    error: BaseException, book: IdentityBook | None,
    *, radius: int = 2, max_nodes: int = 40, fanout: int = 6,
) -> tuple[str, ...]:
    """The neighborhood of every claim ``error`` (or what it was raised
    from) carries, as lines; also added to ``error`` as a note once, so the
    traceback that reports it shows it.  Best-effort: it never raises."""
    if book is None:
        return ()
    try:
        lines: list[str] = []
        seen: set[int] = set()
        pending = [error]
        while pending:
            current = pending.pop(0)
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            pending.extend((current.__cause__, current.__context__))
            for claim in current.__dict__.get("concordance_claims", ()):
                labels = {ref: role for role, ref in claim.seeds}
                hood = book.neighborhood(
                    (ref for _role, ref in claim.seeds),
                    radius=radius, max_nodes=max_nodes, fanout=fanout,
                )
                lines.append(
                    f"failed post {claim.page} {render_row(claim.row)} "
                    f"mode={claim.mode} stage={claim.stage} "
                    f"proposed={claim.proposed}"
                )
                lines.extend(hood.lines(labels))
        if lines and not error.__dict__.get("concordance_noted"):
            error.__dict__["concordance_noted"] = True
            error.add_note(chr(10).join(lines))
        return tuple(lines)
    except Exception:
        return ()
