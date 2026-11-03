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
``phi-edge-not-predecessor``
    a Phi names an incoming block that has no CFG edge to its owner.
``child-selection-not-guarded``
    a selected child view or allocation effect is outside its declared
    capacity/selection predicate's true CFG edge.
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
                        if not field.get("dtype"):
                            raise ValueError(
                                f"class field {field_name!r} declares a span "
                                f"of shape {declared_shape} but records no "
                                "element dtype"
                            )
                        declared_state = {
                            "shape": declared_shape,
                            "dtype": str(field["dtype"]),
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

    from .ssa_record_return_state import accepted_ssa_component

    functions = getattr(module, "functions", {}) or {}
    sequence_tables = getattr(module, "sequence_tables", {}) or {}
    for function_name, function in functions.items():
        if accepted_ssa_component(function):
            # A linked, digest-pinned body published its own internal call
            # edges in its owning book; the enclosing unit settles only the
            # edges it owns.
            continue
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
                    elif record_field.sequence_id is not None:
                        sequence = sequences.by_id(record_field.sequence_id)
                        from ..transmogrifier.ssa import sequence_member_roles

                        # Page.clock's provisional descriptor was retired but
                        # its field kept the old column/length/capacity IDs.
                        # A field's physical surface must belong to the SAME
                        # descriptor its sequence_id names, even when all old
                        # IDs still have historical alias receipts.
                        stale_members = sorted(
                            set(map(int, record_field.value_ids))
                            - set(sequence_member_roles(sequence))
                        )
                        if stale_members:
                            found.append(Finding(
                                "record-sequence-member-disagreement", str(name),
                                int(record.record_id),
                                f"field {record_field.storage_identity!r} names "
                                f"nonmembers {stale_members!r} of sequence "
                                f"{int(sequence.sequence_id)}",
                            ))
                        owner = sequence.heap_owner_value_id
                        if owner is not None and int(owner) not in record_field.value_ids:
                            found.append(Finding(
                                "record-sequence-owner-missing", str(name),
                                int(record.record_id),
                                f"field {record_field.storage_identity!r} omits "
                                f"sequence heap owner {int(owner)} from its physical members",
                            ))
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
        found.extend(self._source_closure_findings(module))
        found.extend(self._source_thread_region_findings(module))
        found.extend(self._control_activity_findings(module))
        found.extend(self._process_graph_fact_findings(module))
        found.extend(self._operator_safety_findings(module))
        found.extend(self._thread_operation_findings(module))
        found.extend(self._thread_table_findings(module))
        found.extend(self._synchronization_findings(module))
        found.extend(self._conditional_binding_preservation_findings(module))
        found.extend(self._source_page_declaration_findings(module))
        found.extend(self._graph_work_publication_findings(module))
        found.extend(self._tile_storage_findings(module))
        found.extend(self._source_field_identity_findings(module))
        found.extend(self._source_parameter_identity_findings(module))
        found.extend(self._source_parameter_default_findings(module))
        found.extend(self._source_callable_signature_findings(module))
        found.extend(self._source_named_tuple_layout_findings(module))
        found.extend(self._named_intrinsic_result_projection_findings(module))
        found.extend(self._source_intrinsic_admission_findings(module))
        found.extend(self._source_call_parameter_default_findings(module))
        found.extend(self._numerical_constructor_default_findings(module))
        found.extend(self._materialized_byte_row_findings(module))
        found.extend(self._annotation_class_admission_findings(module))
        found.extend(self._numerical_schema_operand_findings(module))
        found.extend(self._numerical_dtype_operand_findings(module))
        found.extend(self._source_numeric_component_findings(module))
        found.extend(self._source_numeric_specialization_findings(module))
        found.extend(self._source_numeric_intrinsic_findings(module))
        found.extend(self._source_numeric_type_dependency_findings(module))
        found.extend(self._source_numeric_operator_specialization_findings(module))
        found.extend(self._source_sequence_mutation_findings(module))
        found.extend(self._source_sequence_capacity_bound_findings(module))
        found.extend(self._runtime_result_abi_findings(module))
        found.extend(self._source_aggregate_layout_findings(module))
        found.extend(self._source_closed_aggregate_length_findings(module))
        found.extend(self._source_closed_class_alternative_findings(module))
        found.extend(self._source_class_union_native_gap_findings(module))
        found.extend(self._source_call_closed_class_alternatives_findings(module))
        found.extend(self._source_formal_closed_class_alternatives_findings(module))
        found.extend(self._loop_native_scalar_union_gap_findings(module))
        found.extend(self._source_tuple_key_native_gap_findings(module))
        found.extend(self._mapping_child_contract_findings(module))
        found.extend(self._unborn_display_findings(module))
        found.extend(self._unreturned_output_findings(module))
        found.extend(self._live_structure_findings(module))
        found.extend(self._ssa_publication_findings(module))
        found.extend(self._value_type_abi_findings(module))
        found.extend(self._ssa_abi_physical_findings(module))
        found.extend(self._source_callsite_activation_findings(module))
        found.extend(self._frame_aliased_result_findings(module))
        found.extend(self._physical_call_input_conversion_findings(module))
        found.extend(self._formal_actual_occurrence_findings(module))
        found.extend(self._source_precision_boundary_findings(module))
        found.extend(self._source_precision_operator_findings(module))
        found.extend(self._source_precision_region_findings(module))
        found.extend(self._operator_result_type_findings(module))
        found.extend(self._planning_alias_transition_findings(module))
        found.extend(self._tensor_reduction_domain_findings(module))
        found.extend(self._callable_identity_findings(module))
        found.extend(self._source_call_actual_formal_findings(module))
        found.extend(self._source_call_object_alias_findings(module))
        found.extend(self._source_call_field_effect_completion_findings(module))
        found.extend(self._source_call_activity_corridor_findings(module))
        found.extend(self._source_call_field_state_transfer_findings(module))
        found.extend(self._source_mapping_get_read_findings(module))
        found.extend(self._source_mapping_get_key_version_demand_findings(module))
        found.extend(self._source_mapping_get_copy_state_demand_findings(module))
        found.extend(self._source_guarded_mapping_payload_findings(module))
        found.extend(self._source_evidence_flush_round_findings(module))
        found.extend(self._source_guarded_callable_findings(module))
        found.extend(self._source_parameter_memory_contract_findings(module))
        found.extend(self._source_function_exit_field_state_findings(module))
        found.extend(self._source_constructor_field_effect_demand_findings(module))
        found.extend(self._source_call_tuple_member_projection_demand_findings(module))
        found.extend(self._source_method_receiver_class_demand_findings(module))
        found.extend(self._post_ssa_numeric_identity_findings(module))
        found.extend(self._constant_materialization_findings(module))
        found.extend(self._table_member_findings(module))
        found.extend(self._layout_table_findings(module))
        found.extend(self._operand_position_orphan_findings(module))
        # Last, so callers that show the first few findings still see the
        # per-page kinds first; ``concordance_report`` counts these two kinds
        # on their own line.
        found.extend(self._unsourced_findings(module))
        return found

    @staticmethod
    def _source_parameter_memory_contract_findings(module: Any) -> list[Finding]:
        """Check the one selected contract against authored and formal cells."""
        from .concordance_declarations import (
            CLASS_DECLARATION, NAME_BINDING, PARAMETER_ANNOTATION,
            SOURCE_FORMAL_ORDER, SOURCE_PARAMETER_MEMORY_CONTRACT,
        )
        from ..transmogrifier.function_table import (
            ParameterContract, ParameterStorage,
        )
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        page = book.page(SOURCE_PARAMETER_MEMORY_CONTRACT)
        found = []
        for row in page.rows():
            cell = book.latest_ref(SOURCE_PARAMETER_MEMORY_CONTRACT, row)
            fact = page.cell(cell.row, cell.column)
            contract = fact[0] if isinstance(fact, tuple) and len(fact) == 1 else None
            sources = {source for source, _stage in book.edges_into(cell)}
            order = book.latest_ref(SOURCE_FORMAL_ORDER, (row[0],))
            bindings = tuple(source for source in sources
                             if source.page == NAME_BINDING
                             and source.row[1] == row[1])
            prior = tuple(source for source in sources
                          if source.page == SOURCE_PARAMETER_MEMORY_CONTRACT)
            initial_sources = {
                source for predecessor in prior
                for source, _stage in book.edges_into(predecessor)
            }
            annotation = book.latest_ref(PARAMETER_ANNOTATION, (*row[0], row[1]))
            class_sources = tuple(source for source in initial_sources
                                  if source.page == CLASS_DECLARATION)
            matching_entries = tuple(
                entry for entry in (getattr(module, "function_table", ()) or ())
                if getattr(getattr(entry, "graph", None), "G", None) is not None
                and tuple(entry.graph.G.graph.get("authored_function_identity") or ()) == row[0]
            )
            installed = tuple(
                installed_contract
                for entry in matching_entries
                for installed_contract in entry.parameter_contracts
                if installed_contract.name == row[1]
            )
            used_inputs = tuple(
                (entry, int(node_id))
                for entry in matching_entries
                for node_id, data in entry.graph.G.nodes(data=True)
                if data.get("type") == "Input"
                and (data.get("attributes") or {}).get("binding_name") == row[1]
            )
            annotation_text = (
                book.page(PARAMETER_ANNOTATION).cell(
                    annotation.row, annotation.column,
                ) if annotation is not None else None
            )
            direct_annotation_causes = (
                {source for source, _stage in book.edges_into(annotation)}
                if annotation is not None else set()
            )
            exact_class_sources = tuple(
                source for source in class_sources
                if source.row == (row[0][0], annotation_text)
                and source in direct_annotation_causes
            )
            binding_matches = (
                len(bindings) == 1
                and len(used_inputs) == 1
                and bindings[0].row == (
                    used_inputs[0][0].graph.G.graph.get("lexical_read_scope"),
                    row[1], 0,
                )
                and book.page(NAME_BINDING).cell(
                    bindings[0].row, bindings[0].column,
                ).value_id == used_inputs[0][1]
                or len(bindings) == 0 and not used_inputs
            )
            valid = (
                isinstance(contract, ParameterContract)
                and contract.name == row[1]
                and order in sources
                and row[1] in book.page(SOURCE_FORMAL_ORDER).cell(order.row, order.column)
                if order is not None else False
            )
            valid = bool(
                valid and binding_matches and prior
                and contract.storage is ParameterStorage.RECORD
                and annotation is not None and annotation in initial_sources
                and len(exact_class_sources) == 1
                and installed == (contract,)
            )
            if not valid:
                found.append(Finding(
                    "source-parameter-memory-contract", str(row[0]), None,
                    f"parameter {row[1]!r} lost selected contract, annotation, class, order or formal binding",
                ))
        return found

    @staticmethod
    def _source_function_exit_field_state_findings(module: Any) -> list[Finding]:
        """Check that a formal's exit state names its exact cursor and exit."""
        from .concordance_declarations import (
            CLASS_FIELD_DECLARATION, PROCESS_GRAPH_OPERANDS,
            REDUCER_FIELD_STATE,
            SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_NODE,
            INGESTION_VALUE, SOURCE_FORMAL_ORDER,
            SOURCE_FUNCTION_EXIT_FIELD_STATE, SOURCE_SPAN,
            FieldState, FieldStateKind,
        )
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found = []
        page = book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE)
        for row in page.rows():
            cell = book.latest_ref(SOURCE_FUNCTION_EXIT_FIELD_STATE, row)
            fact = page.cell(cell.row, cell.column)
            if not (isinstance(fact, tuple) and len(fact) == 5):
                found.append(Finding(
                    "source-function-exit-field-state", str(row[0]), row[1],
                    "exit receipt has no formal, state, exit, assignment and flow cells",
                ))
                continue
            formal, state, exit_activity, assignment_source, flow_edge = fact
            order = book.latest_ref(SOURCE_FORMAL_ORDER, (row[0],))
            sources = {source for source, _stage in book.edges_into(cell)}
            prior_receipts = tuple(source for source in sources
                                   if source.page == SOURCE_FUNCTION_EXIT_FIELD_STATE)
            order_proved = order in sources or any(
                order in {source for source, _stage in book.edges_into(prior)}
                for prior in prior_receipts
            )
            state_fact = (
                book.page(REDUCER_FIELD_STATE).cell(state.row, state.column)
                if isinstance(state, Ref) and state.page == REDUCER_FIELD_STATE
                else None
            )
            exit_fact = (
                book.page(SOURCE_ACTIVITY_NODE).cell(
                    exit_activity.row, exit_activity.column,
                )
                if isinstance(exit_activity, Ref)
                and exit_activity.page == SOURCE_ACTIVITY_NODE else None
            )
            assignment_exit = (
                book.latest_ref(SOURCE_ACTIVITY_NODE, (
                    row[0], assignment_source, "exit",
                ))
                if isinstance(assignment_source, Ref)
                and assignment_source.page == SOURCE_SPAN else None
            )
            assignment_path = (
                assignment_source.row[2]
                if isinstance(assignment_source, Ref)
                and assignment_source.page == SOURCE_SPAN else None
            )
            sequence_proved = assignment_path == (('body', 0),)
            direct_write_proved = False
            if (isinstance(state_fact, FieldState)
                    and state_fact.kind is FieldStateKind.WRITTEN
                    and assignment_path == (('body', 0),)):
                proof_receipt = (
                    prior_receipts[0] if len(prior_receipts) == 1 else cell
                )
                proof_fact = page.cell(proof_receipt.row, proof_receipt.column)
                original_state = proof_fact[1]
                original = book.page(REDUCER_FIELD_STATE).cell(
                    original_state.row, original_state.column,
                )
                original_sources = {
                    source for source, _stage in book.edges_into(original_state)
                }
                proof_sources = {
                    source for source, _stage in book.edges_into(proof_receipt)
                }
                schemas = tuple(source for source in proof_sources
                                if source.page == CLASS_FIELD_DECLARATION)
                operands = tuple(source for source in proof_sources
                                 if source.page == PROCESS_GRAPH_OPERANDS)
                operand_fact = (
                    book.page(PROCESS_GRAPH_OPERANDS).cell(
                        operands[0].row, operands[0].column,
                    ) if len(operands) == 1 else None
                )
                target = book.latest_ref(SOURCE_SPAN, (
                    assignment_source.row[0], assignment_source.row[1],
                    (('body', 0), ('targets', 0)),
                ))
                effect = original.effect if isinstance(original, FieldState) else None
                effect_origins = (
                    (effect,) if isinstance(effect, Ref)
                    and effect.page == INGESTION_VALUE else tuple(
                        source for source, _stage in book.edges_into(effect)
                        if source.page == INGESTION_VALUE
                    ) if isinstance(effect, Ref) else ()
                )
                body_spans = tuple(
                    span_row for span_row in book.page(SOURCE_SPAN).rows()
                    if span_row[:2] == assignment_source.row[:2]
                    and len(span_row[2]) == 1
                    and span_row[2][0][0] == 'body'
                )
                direct_write_proved = (
                    len(body_spans) == 1
                    and isinstance(original, FieldState)
                    and original.kind is FieldStateKind.WRITTEN
                    and original_state in proof_sources
                    and len(schemas) == 1
                    and schemas[0].row[-1] == row[2]
                    and schemas[0] in original_sources
                    and target is not None
                    and book.page(SOURCE_SPAN).cell(
                        target.row, target.column).kind == 'Attribute'
                    and target in proof_sources
                    and len(effect_origins) == 1
                    and effect_origins[0] in proof_sources
                    and target in {
                        source for source, _stage in
                        book.edges_into(effect_origins[0])
                    }
                    and isinstance(original.value, Ref)
                    and original.value in proof_sources
                    and isinstance(formal, Ref)
                    and isinstance(state, Ref)
                    and (proof_fact[0] == formal or proof_fact[0] in {
                        source for source, _stage in book.edges_into(formal)
                    })
                    and (original_state == state or original_state in {
                        source for source, _stage in book.edges_into(state)
                    })
                    and operand_fact == (
                        (proof_fact[0], 'object'),
                        (original.value, 'value'),
                    )
                    and operands[0].row[-1] == original.effect.row[-1]
                )
            if assignment_path == (('body', 1),):
                proof_receipt = (
                    prior_receipts[0] if len(prior_receipts) == 1 else cell
                )
                proof_fact = page.cell(proof_receipt.row, proof_receipt.column)
                proof_state = proof_fact[1]
                proof_sources = {
                    source for source, _stage in book.edges_into(proof_receipt)
                }
                first_span = book.latest_ref(SOURCE_SPAN, (
                    assignment_source.row[0], assignment_source.row[1],
                    (('body', 0),),
                ))
                first_exit = (
                    book.latest_ref(SOURCE_ACTIVITY_NODE, (
                        row[0], first_span, 'exit',
                    )) if first_span is not None else None
                )
                final_enter = book.latest_ref(SOURCE_ACTIVITY_NODE, (
                    row[0], assignment_source, 'enter',
                ))
                bridge = (
                    book.latest_ref(SOURCE_ACTIVITY_EDGE, (
                        row[0], first_exit, final_enter, 'flow',
                    )) if first_exit is not None and final_enter is not None
                    else None
                )
                predecessors = tuple(
                    source for source, _stage in book.edges_into(proof_state)
                    if source.page == REDUCER_FIELD_STATE
                    and source.row == proof_state.row
                )
                prior_state = predecessors[0] if len(predecessors) == 1 else None
                first_state = (
                    book.page(REDUCER_FIELD_STATE).cell(
                        prior_state.row, prior_state.column,
                    ) if prior_state is not None else None
                )
                last_state = book.page(REDUCER_FIELD_STATE).cell(
                    proof_state.row, proof_state.column,
                )
                def exact_target(effect, position):
                    if not isinstance(effect, Ref):
                        return False
                    ingestions = (
                        (effect,) if effect.page == INGESTION_VALUE else
                        tuple(source for source, _stage in book.edges_into(effect)
                              if source.page == INGESTION_VALUE)
                    )
                    target = book.latest_ref(SOURCE_SPAN, (
                        assignment_source.row[0], assignment_source.row[1],
                        (('body', position), ('targets', 0)),
                    ))
                    return target is not None and any(
                        target in {source for source, _stage in
                                   book.edges_into(ingestion)}
                        for ingestion in ingestions
                    )
                body_spans = tuple(
                    span_row for span_row in book.page(SOURCE_SPAN).rows()
                    if span_row[:2] == assignment_source.row[:2]
                    and len(span_row[2]) == 1
                    and span_row[2][0][0] == 'body'
                )
                sequence_proved = (
                    len(body_spans) == 2
                    and first_span is not None
                    and book.page(SOURCE_SPAN).cell(
                        first_span.row, first_span.column,
                    ).kind == 'Assign'
                    and first_exit is not None and final_enter is not None
                    and bridge is not None
                    and prior_state is not None
                    and isinstance(first_state, FieldState)
                    and first_state.kind is FieldStateKind.ELEMENT_WRITTEN
                    and isinstance(last_state, FieldState)
                    and last_state.kind is FieldStateKind.ELEMENT_WRITTEN
                    and exact_target(first_state.effect, 0)
                    and exact_target(last_state.effect, 1)
                    and {first_span, first_exit, final_enter,
                         bridge, prior_state} <= proof_sources
                )
            if (not isinstance(formal, Ref)
                    or formal.row[-1] != row[1]
                    or not isinstance(state, Ref) or state.row[1:] != row[1:]
                    or not isinstance(state_fact, FieldState)
                    or not (state_fact.kind is FieldStateKind.ELEMENT_WRITTEN
                            or (state_fact.kind is FieldStateKind.WRITTEN
                                and direct_write_proved))
                    or exit_fact != ("FunctionDef", ())
                    or exit_activity.row[0] != row[0]
                    or exit_activity.row[-1] != "exit"
                    or assignment_exit is None
                    or book.page(SOURCE_SPAN).cell(
                        assignment_source.row, assignment_source.column,
                    ).kind != "Assign"
                    or not isinstance(flow_edge, Ref)
                    or flow_edge.page != SOURCE_ACTIVITY_EDGE
                    or flow_edge.row != (
                        row[0], assignment_exit, exit_activity, "flow",
                    )
                    or order is None or not order_proved
                    or not sequence_proved
                    or not {formal, state, exit_activity,
                            assignment_source, flow_edge} <= sources):
                found.append(Finding(
                    "source-function-exit-field-state", str(row[0]), row[1],
                    "exit receipt lost its formal, final write, formal order or exact assignment-to-exit flow",
                ))
        return found

    @staticmethod
    def _source_constructor_field_effect_demand_findings(module: Any) -> list[Finding]:
        """Check a missing constructor receiver at its exact class Call."""
        from .concordance_declarations import SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND
        from .source_constructor_field_demand import candidate_constructor_field_demands

        book = (getattr(module, 'metadata', {}) or {}).get('identity_book')
        table = getattr(module, 'function_table', None)
        if book is None or table is None:
            return []
        candidates = {
            row: (fact, set(causes))
            for row, fact, causes in candidate_constructor_field_demands(
                table, book=book)
        }
        found = []
        page = book.page(SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND)
        for row in page.rows():
            cell = book.latest_ref(SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND, row)
            fact = page.cell(cell.row, cell.column)
            causes = {source for source, _stage in book.edges_into(cell)}
            expected = candidates.get(row)
            if (expected is None or fact != expected[0]
                    or not expected[1] <= causes):
                found.append(Finding(
                    'source-constructor-field-effect-demand', str(row[0]),
                    row[1],
                    'constructor demand lost exact Call, class, method, address, exit or activity cause',
                ))
        return found

    @staticmethod
    def _source_call_tuple_member_projection_demand_findings(module: Any) -> list[Finding]:
        """Rejoin an unresolved callee member from exact call/layout cells."""
        from .concordance_declarations import SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND
        from .source_tuple_member_demand import candidate_tuple_member_demands

        book = (getattr(module, 'metadata', {}) or {}).get('identity_book')
        table = getattr(module, 'function_table', None)
        if book is None or table is None:
            return []
        expected = {
            row: (fact, set(causes))
            for row, fact, causes in candidate_tuple_member_demands(table, book=book)
        }
        found = []
        page = book.page(SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND)
        for row in page.rows():
            cell = book.latest_ref(SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND, row)
            fact = page.cell(cell.row, cell.column)
            sources = {source for source, _stage in book.edges_into(cell)}
            target = expected.get(row)
            if target is None or fact != target[0] or not target[1] <= sources:
                found.append(Finding(
                    'source-call-tuple-member-projection-demand', str(row[2]),
                    row[4],
                    'tuple member demand lost call, formal, layout, index or control cause',
                ))
        return found

    @staticmethod
    def _source_method_receiver_class_demand_findings(module: Any) -> list[Finding]:
        """Check a missing formal class against the exact method Call."""
        from .concordance_declarations import SOURCE_METHOD_RECEIVER_CLASS_DEMAND
        from .source_method_receiver_demand import candidate_method_receiver_demand

        book = (getattr(module, 'metadata', {}) or {}).get('identity_book')
        table = getattr(module, 'function_table', None)
        if book is None or table is None:
            return []
        by_scope = {
            tuple(entry.graph.G.graph.get('authored_function_identity') or ()): entry.graph
            for entry in table if getattr(entry, 'graph', None) is not None
        }
        found = []
        page = book.page(SOURCE_METHOD_RECEIVER_CLASS_DEMAND)
        for row in page.rows():
            cell = book.latest_ref(SOURCE_METHOD_RECEIVER_CLASS_DEMAND, row)
            fact = page.cell(cell.row, cell.column)
            sources = {source for source, _stage in book.edges_into(cell)}
            graph = by_scope.get(row[0])
            candidate = (None if graph is None else
                         candidate_method_receiver_demand(
                             graph, row[1], row[2], book=book))
            if (candidate is None or candidate[0] != row
                    or candidate[1] != fact
                    or not set(candidate[2]) <= sources):
                found.append(Finding(
                    'source-method-receiver-class-demand', str(row[0]), row[1],
                    'missing receiver class demand lost Call, Input, operand or control cause',
                ))
        return found

    @staticmethod
    def _source_evidence_flush_round_findings(module: Any) -> list[Finding]:
        """Check every whole-source round against exact input/output cells."""
        from .concordance_declarations import (
            FIXED_POINT_ROUND, SOURCE_EVIDENCE_FLUSH_ROUND,
        )
        from .source_evidence_flush import (
            _FAMILY, _apply_changes, _digest, _snapshot,
        )
        from .bounded_fixed_point import _digest as _guard_digest
        book = (getattr(module, 'metadata', {}) or {}).get('identity_book')
        if book is None:
            return []
        found = []
        page = book.page(SOURCE_EVIDENCE_FLUSH_ROUND)
        family = set(_FAMILY)
        family_names = {declaration.name for declaration in family}
        for row in page.rows():
            cell = book.latest_ref(SOURCE_EVIDENCE_FLUSH_ROUND, row)
            receipt_stamp = page.stamps[(row, cell.column)]
            later_family_write = any(
                stamp > receipt_stamp
                for declaration in family
                for stamp in book.page(declaration).stamps.values()
            ) or any(
                stamp > receipt_stamp and edge_row[0][0] in family_names
                for (edge_row, _column), stamp in book.page(EDGE_PAGE).stamps.items()
            )
            before, after, changes, guard, restarted = page.cell(
                cell.row, cell.column)
            sources = {source for source, _stage in book.edges_into(cell)}
            guard_fact = (book.page(FIXED_POINT_ROUND).cell(
                guard.row, guard.column) if guard.page == FIXED_POINT_ROUND
                else None)
            entries = tuple(entry for pair in changes for entry in pair
                            if entry is not None)
            cells = tuple(item[0] for item in entries)
            parents = tuple(source for _cell, edges in entries
                            for source, _stage in edges)
            previous = book.latest_ref(SOURCE_EVIDENCE_FLUSH_ROUND,
                (row[0], row[1] - 1)) if row[1] > 1 else None
            prior_after = (None if previous is None else
                           book.page(SOURCE_EVIDENCE_FLUSH_ROUND).cell(
                               previous.row, previous.column)[1])
            if (guard.row != ('whole-source-evidence-flush', row[0], row[1])
                    or guard_fact is None
                    or guard_fact[0] != bool(changes)
                    or guard_fact[1] != _guard_digest(after)
                    or bool(changes) != (before != after)
                    or _apply_changes(before, changes) != after
                    or any(item.page not in family for item in cells)
                    or tuple(sorted(changes, key=lambda pair: repr(
                        (pair[1] or pair[0])[0]))) != changes
                    or any(old == new for old, new in changes)
                    or any(not set(edges) <= set(book.edges_into(item))
                           for item, edges in entries)
                    or not {*cells, *parents, guard} <= sources
                    or (row[1] > 1 and not restarted and prior_after != before)
                    or (restarted and row[1] <= 1)
                    or (not later_family_write
                        and not any(next_row[0] == row[0]
                                and next_row[1] > row[1]
                                for next_row in page.rows())
                        and after != _digest(_snapshot(book)))):
                found.append(Finding('source-evidence-flush-round',
                                     str(row[0]), row[1],
                                     'whole-source round lost exact input, result or bounded guard causes'))
        return found

    @staticmethod
    def _source_guarded_mapping_payload_findings(module: Any) -> list[Finding]:
        """Rejoin each present-arm payload from its exact source cells."""
        from .concordance_declarations import (
            INGESTION_VALUE, SOURCE_GUARDED_MAPPING_PAYLOAD,
            SOURCE_MAPPING_GET_READ, SOURCE_OPERATOR_KIND, SOURCE_SPAN,
        )
        from .source_guarded_payload import candidate_facts
        book = (getattr(module, 'metadata', {}) or {}).get('identity_book')
        if book is None:
            return []
        found = []
        for row in book.page(SOURCE_OPERATOR_KIND).rows():
            cell = book.latest_ref(SOURCE_OPERATOR_KIND, row)
            operator = book.page(SOURCE_OPERATOR_KIND).cell(cell.row, cell.column)
            if not isinstance(operator, str) or not operator.startswith('compare:'):
                continue
            sources = {source for source, _stage in book.edges_into(cell)}
            ingestion = row[0]
            spans = tuple(source for source in sources if source.page == SOURCE_SPAN)
            if (ingestion.page != INGESTION_VALUE or ingestion not in sources
                    or len(spans) != 1
                    or book.page(SOURCE_SPAN).cell(spans[0].row, spans[0].column).kind != 'Compare'):
                found.append(Finding('source-compare-operator', str(row), None,
                                     'typed Compare lost its ingested occurrence'))
        for row in book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).rows():
            cell = book.latest_ref(SOURCE_GUARDED_MAPPING_PAYLOAD, row)
            facts = book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).cell(
                cell.row, cell.column)
            read = facts[0] if facts else None
            candidates = (() if read is None else
                          candidate_facts(book, row[0], int(row[1]), read))
            direct = {source for source, _stage in book.edges_into(cell)}
            if (read is None or read.page != SOURCE_MAPPING_GET_READ
                    or read.row[0:2] != row[0:2]
                    or book.latest_ref(SOURCE_MAPPING_GET_READ, read.row) != read
                    or (row, facts) not in candidates
                    or not set(facts) <= direct):
                found.append(Finding('source-guarded-mapping-payload',
                                     str(row[0]), row[2],
                                     'present payload lost its assignment, Is None guard, terminal return or arm edges'))
        return found

    @staticmethod
    def _source_guarded_callable_findings(module: Any) -> list[Finding]:
        """Check the guarded call's exact method and address source bridge."""
        import ast
        from ..common.tensors.topological_reducer import node_identity_cell
        from ..transmogrifier.function_table import function_address_identity
        from .concordance_declarations import (
            CALLABLE_IDENTITY, CANONICAL_VALUE, CLASS_DECLARATION,
            CLASS_METHOD_DECLARATION,
            FUNCTION_ADDRESS, PROCESS_GRAPH_OPERANDS,
            SOURCE_GUARDED_MAPPING_PAYLOAD, SOURCE_MAPPING_GET_READ,
            SOURCE_SPAN, SOURCE_VALUE_CLASS,
        )
        book = (getattr(module, 'metadata', {}) or {}).get('identity_book')
        if book is None:
            return []
        found = []
        for row in book.page(CALLABLE_IDENTITY).rows():
            callable_cell = book.latest_ref(CALLABLE_IDENTITY, row)
            direct = {source for source, _stage in book.edges_into(callable_cell)}
            guards = tuple(source for source in direct
                           if source.page == SOURCE_GUARDED_MAPPING_PAYLOAD)
            if not guards:
                continue
            valid = False
            if len(guards) == 1:
                guard = guards[0]
                guarded = book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).cell(
                    guard.row, guard.column)
                read = guarded[0] if guarded else None
                stored = guarded[1] if len(guarded) > 1 else None
                operands = guarded[-1] if guarded else None
                methods = tuple(source for source in direct
                                if source.page == CLASS_METHOD_DECLARATION)
                classes = tuple(source for source in direct
                                if source.page == SOURCE_VALUE_CLASS)
                addresses = tuple(source for source in direct
                                  if source.page == FUNCTION_ADDRESS)
                spans = tuple(source for source in direct
                              if source.page == SOURCE_SPAN)
                calls = tuple(source for source in direct
                              if source.page == CANONICAL_VALUE
                              and source.row[-1] == row[1])
                if (guard.row[0] == row[0] and guard.row[2] == row[1]
                        and read is not None and read.page == SOURCE_MAPPING_GET_READ
                        and stored is not None and operands is not None
                        and operands.page == PROCESS_GRAPH_OPERANDS
                        and len(methods) == len(classes) == len(addresses)
                        == len(spans) == len(calls) == 1):
                    method, class_cell, address_cell, method_span = (
                        methods[0], classes[0], addresses[0], spans[0])
                    read_fact = book.page(SOURCE_MAPPING_GET_READ).cell(
                        read.row, read.column)
                    class_fact = book.page(SOURCE_VALUE_CLASS).cell(
                        class_cell.row, class_cell.column)
                    method_fact = book.page(CLASS_METHOD_DECLARATION).cell(
                        method.row, method.column)
                    entries = tuple(
                        entry for entry in (getattr(module, 'function_table', ()) or ())
                        if entry.qualified_name == method_fact.graph_identity
                        and (entry.metadata or {}).get('method_owner') == class_fact[0]
                    )
                    wrappers = tuple(
                        entry.graph for entry in (getattr(module, 'function_table', ()) or ())
                        if getattr(entry, 'graph', None) is not None
                        and tuple(entry.graph.G.graph.get('authored_function_identity')
                                  or ()) == row[0]
                    )
                    exact_accessor = False
                    if len(wrappers) == 1 and row[1] in wrappers[0].G:
                        wrapper = wrappers[0]
                        expression = wrapper.G.nodes[row[1]].get('expr_obj')
                        if (isinstance(expression, ast.Call)
                                and isinstance(expression.func, ast.Attribute)
                                and expression.func.attr == method.row[2]
                                and node_identity_cell(wrapper, row[1], book=book)
                                == calls[0]):
                            accessor_ids = tuple(
                                int(node) for node, data in wrapper.G.nodes(data=True)
                                if data.get('expr_obj') is expression.func)
                            exact_accessor = (
                                len(accessor_ids) == 1
                                and node_identity_cell(wrapper, accessor_ids[0],
                                                       book=book) in direct)
                    declarations = tuple(source for source in direct
                                         if source.page == CLASS_DECLARATION)
                    method_sources = {source for source, _stage in
                                      book.edges_into(method)}
                    address_sources = {source for source, _stage in
                                       book.edges_into(address_cell)}
                    operand_fact = book.page(PROCESS_GRAPH_OPERANDS).cell(
                        operands.row, operands.column)
                    valid = (
                        exact_accessor
                        and len(declarations) == 1
                        and class_fact[1:] == (1, 'source class construction')
                        and class_cell.row == (read_fact[1].row[0], stored.row[-1])
                        and declarations[0].row == (read_fact[1].row[0][0],
                                                     class_fact[0])
                        and method.row[:2] == declarations[0].row
                        and method_span.row[:2] == declarations[0].row
                        and book.page(SOURCE_SPAN).cell(
                            method_span.row, method_span.column).kind
                        in ('FunctionDef', 'AsyncFunctionDef')
                        and method_span in method_sources
                        and method_span in address_sources
                        and book.latest_ref(SOURCE_GUARDED_MAPPING_PAYLOAD,
                                            guard.row) == guard
                        and len(entries) == 1
                        and address_cell.row == (
                            function_address_identity(entries[0]),)
                        and book.page(FUNCTION_ADDRESS).cell(
                            address_cell.row, address_cell.column)
                        == int(entries[0].reference.address)
                        and book.page(FUNCTION_ADDRESS).cell(
                            address_cell.row, address_cell.column)
                        == book.page(CALLABLE_IDENTITY).cell(
                            callable_cell.row, callable_cell.column)
                        and guarded[-2] == calls[0]
                        and (read_fact[7], 'operand') in operand_fact
                        and all(source in direct for source in
                                (read, stored, operands, declarations[0],
                                 method, class_cell, address_cell, method_span))
                    )
            if not valid:
                found.append(Finding(
                    'source-guarded-callable', str(row[0]), row[1],
                    'guarded callable lost its stored class, exact method span, or function address bridge',
                ))
        return found

    @staticmethod
    def _source_mapping_get_key_version_demand_findings(module: Any) -> list[Finding]:
        """A later unproved key cannot be compared to the requested key."""
        import ast
        from ..common.tensors.topological_reducer import node_identity_cell
        from .concordance_declarations import (
            PROCESS_GRAPH_OPERANDS, REDUCER_FIELD_STATE,
            SOURCE_AGGREGATE_LAYOUT, SOURCE_CALL_FIELD_STATE_TRANSFER,
            SOURCE_LITERAL_VALUE, SOURCE_MAPPING_GET_KEY_VERSION_DEMAND,
            SOURCE_SEQUENCE_MUTATION, FieldState, FieldStateKind,
        )
        book = (getattr(module, 'metadata', {}) or {}).get('identity_book')
        if book is None:
            return []
        found = []
        page = book.page(SOURCE_MAPPING_GET_KEY_VERSION_DEMAND)
        for row in page.rows():
            cell = book.latest_ref(SOURCE_MAPPING_GET_KEY_VERSION_DEMAND, row)
            fact = page.cell(cell.row, cell.column)
            if not isinstance(fact, tuple) or len(fact) != 13:
                found.append(Finding('source-mapping-get-key-version-demand',
                                     str(row[0]), row[1], 'incomplete key demand'))
                continue
            (reason, transfer, state, mutation, layout, later_key,
             read_key, observed_cell, call, base, key, operands,
             read_mutation) = fact
            transferred = (book.page(SOURCE_CALL_FIELD_STATE_TRANSFER).cell(
                transfer.row, transfer.column,
            ) if transfer.page == SOURCE_CALL_FIELD_STATE_TRANSFER else None)
            store = (book.page(SOURCE_SEQUENCE_MUTATION).cell(
                mutation.row, mutation.column,
            ) if mutation.page == SOURCE_SEQUENCE_MUTATION else None)
            get = (book.page(SOURCE_SEQUENCE_MUTATION).cell(
                read_mutation.row, read_mutation.column,
            ) if read_mutation.page == SOURCE_SEQUENCE_MUTATION else None)
            layout_fact = (book.page(SOURCE_AGGREGATE_LAYOUT).cell(
                layout.row, layout.column,
            ) if layout.page == SOURCE_AGGREGATE_LAYOUT else None)
            observed = (book.page(REDUCER_FIELD_STATE).cell(
                observed_cell.row, observed_cell.column,
            ) if observed_cell.page == REDUCER_FIELD_STATE else None)
            operand_fact = (book.page(PROCESS_GRAPH_OPERANDS).cell(
                operands.row, operands.column,
            ) if operands.page == PROCESS_GRAPH_OPERANDS else None)
            wrappers = tuple(
                entry.graph for entry in (getattr(module, 'function_table', ()) or ())
                if getattr(entry, 'graph', None) is not None
                and tuple(entry.graph.G.graph.get('authored_function_identity') or ()) == row[0]
            )
            exact_call = False
            if len(wrappers) == 1 and row[1] in wrappers[0].G:
                expression = wrappers[0].G.nodes[row[1]].get('expr_obj')
                exact_call = (
                    isinstance(expression, ast.Call)
                    and isinstance(expression.func, ast.Attribute)
                    and expression.func.attr == 'get'
                    and len(expression.args) == 1 and not expression.keywords
                    and node_identity_cell(wrappers[0], row[1], book=book) == call
                )
            direct = {source for source, _stage in book.edges_into(cell)}
            if not (
                reason == 'later_store_key_equality_unproved'
                and transfer.row[3] == row[0]
                and transfer.row[5] == row[2]
                and transferred is not None
                and transferred[6:9] == (state, mutation, observed_cell)
                and store is not None and len(store) == 5
                and store[1:3] == ('setitem', 'unique')
                and store[4] == 'mapping' and len(store[3]) == 2
                and layout_fact is not None
                and layout_fact.aggregate_kind == 'dict'
                and tuple(member.value_id for member in layout_fact.members)
                == tuple(store[3])
                and layout_fact.members[0].value_cell == later_key
                and layout.row == (state.row[0], mutation.row[1])
                and book.latest_ref(SOURCE_LITERAL_VALUE,
                    (mutation.row[0], store[3][0])) is None
                and read_key.page == SOURCE_LITERAL_VALUE
                and isinstance(book.page(SOURCE_LITERAL_VALUE).cell(
                    read_key.row, read_key.column,
                ), (str, int, float, bool, bytes))
                and isinstance(observed, FieldState)
                and observed.kind is FieldStateKind.OBSERVED
                and observed.value == base
                and operand_fact is not None
                and set(operand_fact) == {(base, 'operand'), (key, 'arg:0')}
                and get == (base.row[-1], 'unclassified_call:get',
                            'unique', (key.row[-1],), 'mapping')
                and read_key.row == (row[0], key.row[-1])
                and call.row[-1] == row[1]
                and exact_call and set(fact[1:]) <= direct
            ):
                found.append(Finding('source-mapping-get-key-version-demand',
                                     str(row[0]), row[1],
                                     'later key demand lost its transfer, key, or exact read'))
        return found

    @staticmethod
    def _source_mapping_get_copy_state_demand_findings(module: Any) -> list[Finding]:
        """A copied mapping has an exact source, but no copy state contract."""
        import ast
        from ..common.tensors.topological_reducer import node_identity_cell
        from .control_activity import binding
        from .concordance_declarations import (
            PROCESS_GRAPH_OPERANDS, REDUCER_FIELD_STATE,
            SOURCE_CALL_FIELD_STATE_TRANSFER,
            SOURCE_MAPPING_GET_COPY_STATE_DEMAND, SOURCE_SEQUENCE_MUTATION,
            FieldState, FieldStateKind,
        )
        book = (getattr(module, 'metadata', {}) or {}).get('identity_book')
        if book is None:
            return []
        found = []
        page = book.page(SOURCE_MAPPING_GET_COPY_STATE_DEMAND)
        for row in page.rows():
            cell = book.latest_ref(SOURCE_MAPPING_GET_COPY_STATE_DEMAND, row)
            fact = page.cell(cell.row, cell.column)
            if not isinstance(fact, tuple) or len(fact) != 12:
                found.append(Finding('source-mapping-get-copy-state-demand',
                                     str(row[0]), row[1], 'incomplete copy demand'))
                continue
            (reason, transfer, observed_state, observed_value,
             copy_mutation, copy_call, copy_operands, copy_binding,
             get_call, get_operands, key, get_binding) = fact
            transferred = (book.page(SOURCE_CALL_FIELD_STATE_TRANSFER).cell(
                transfer.row, transfer.column,
            ) if transfer.page == SOURCE_CALL_FIELD_STATE_TRANSFER else None)
            observed = (book.page(REDUCER_FIELD_STATE).cell(
                observed_state.row, observed_state.column,
            ) if observed_state.page == REDUCER_FIELD_STATE else None)
            mutation = (book.page(SOURCE_SEQUENCE_MUTATION).cell(
                copy_mutation.row, copy_mutation.column,
            ) if copy_mutation.page == SOURCE_SEQUENCE_MUTATION else None)
            copy_inputs = (book.page(PROCESS_GRAPH_OPERANDS).cell(
                copy_operands.row, copy_operands.column,
            ) if copy_operands.page == PROCESS_GRAPH_OPERANDS else None)
            get_inputs = (book.page(PROCESS_GRAPH_OPERANDS).cell(
                get_operands.row, get_operands.column,
            ) if get_operands.page == PROCESS_GRAPH_OPERANDS else None)
            wrappers = tuple(
                entry.graph for entry in (getattr(module, 'function_table', ()) or ())
                if getattr(entry, 'graph', None) is not None
                and tuple(entry.graph.G.graph.get('authored_function_identity') or ()) == row[0]
            )
            exact_occurrences = False
            if len(wrappers) == 1:
                wrapper = wrappers[0]
                copy_id, get_id = copy_call.row[-1], get_call.row[-1]
                if copy_id in wrapper.G and get_id in wrapper.G:
                    copy_expr = wrapper.G.nodes[copy_id].get('expr_obj')
                    get_expr = wrapper.G.nodes[get_id].get('expr_obj')
                    exact_occurrences = (
                        isinstance(copy_expr, ast.Call)
                        and isinstance(copy_expr.func, ast.Attribute)
                        and copy_expr.func.attr == 'copy'
                        and not copy_expr.args and not copy_expr.keywords
                        and isinstance(get_expr, ast.Call)
                        and isinstance(get_expr.func, ast.Attribute)
                        and get_expr.func.attr == 'get'
                        and len(get_expr.args) == 1 and not get_expr.keywords
                        and node_identity_cell(wrapper, copy_id, book=book) == copy_call
                        and node_identity_cell(wrapper, get_id, book=book) == get_call
                    )
                    if exact_occurrences:
                        try:
                            exact_occurrences = (
                                binding(wrapper, copy_id, book=book,
                                        scope=row[0])[1] == copy_binding
                                and binding(wrapper, get_id, book=book,
                                            scope=row[0])[1] == get_binding
                            )
                        except (ValueError, KeyError):
                            exact_occurrences = False
            causes = {source for source, _stage in book.edges_into(cell)}
            if not (
                reason == 'mapping_copy_alias_or_state_contract'
                and transfer.row[3] == row[0]
                and transfer.row[5] == row[2]
                and transferred is not None and transferred[8] == observed_state
                and isinstance(observed, FieldState)
                and observed.kind is FieldStateKind.OBSERVED
                and observed.value == observed_value
                and mutation == (observed_value.row[-1],
                                 'unclassified_call:copy', 'unique', (), 'mapping')
                and copy_inputs == ((observed_value, 'operand'),)
                and set(get_inputs or ()) == {(copy_call, 'operand'),
                                              (key, 'arg:0')}
                and copy_mutation.row == (row[0], copy_call.row[-1])
                and get_call.row[-1] == row[1]
                and exact_occurrences
                and set(fact[1:]) <= causes
            ):
                found.append(Finding('source-mapping-get-copy-state-demand',
                                     str(row[0]), row[1],
                                     'copy demand lost its exact transfer, operands, or source occurrences'))
        return found

    @staticmethod
    def _source_mapping_get_read_findings(module: Any) -> list[Finding]:
        """Check an exact present-key mapping read and its pure source prefix."""
        import ast
        from ..common.tensors.topological_reducer import node_identity_cell
        from .concordance_declarations import (
            PROCESS_GRAPH_OPERANDS, REDUCER_FIELD_STATE,
            SOURCE_ACTIVITY_BINDING, SOURCE_ACTIVITY_CALL,
            SOURCE_ACTIVITY_EDGE, SOURCE_AGGREGATE_LAYOUT,
            SOURCE_CALL_FIELD_STATE_TRANSFER, SOURCE_LITERAL_VALUE,
            SOURCE_MAPPING_GET_READ, SOURCE_SEQUENCE_MUTATION,
            FieldState, FieldStateKind,
        )
        from .control_activity import normal_mapping_get_prefix
        from .source_mapping_get import reaching_key_version
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        page = book.page(SOURCE_MAPPING_GET_READ)
        found = []
        for row in page.rows():
            cell = book.latest_ref(SOURCE_MAPPING_GET_READ, row)
            fact = page.cell(cell.row, cell.column)
            if not isinstance(fact, tuple) or len(fact) not in (19, 20):
                found.append(Finding("source-mapping-get-read", str(row[0]), row[1],
                                     "mapping read has no complete exact proof"))
                continue
            (transfer, store, layout, store_key, read_key, stored_value,
             observed_cell, call, method, base, key, operands,
             read_call, call_binding, activity_call, method_binding,
             method_edge, nodes, edges) = fact[:19]
            sources = {source for source, _stage in book.edges_into(cell)}
            transferred = (book.page(SOURCE_CALL_FIELD_STATE_TRANSFER).cell(
                transfer.row, transfer.column,
            ) if transfer.page == SOURCE_CALL_FIELD_STATE_TRANSFER else None)
            store_fact = (book.page(SOURCE_SEQUENCE_MUTATION).cell(
                store.row, store.column,
            ) if store.page == SOURCE_SEQUENCE_MUTATION else None)
            read_fact = (book.page(SOURCE_SEQUENCE_MUTATION).cell(
                read_call.row, read_call.column,
            ) if read_call.page == SOURCE_SEQUENCE_MUTATION else None)
            layout_fact = (book.page(SOURCE_AGGREGATE_LAYOUT).cell(
                layout.row, layout.column,
            ) if layout.page == SOURCE_AGGREGATE_LAYOUT else None)
            observed = (book.page(REDUCER_FIELD_STATE).cell(
                observed_cell.row, observed_cell.column,
            ) if observed_cell.page == REDUCER_FIELD_STATE else None)
            operand_fact = (book.page(PROCESS_GRAPH_OPERANDS).cell(
                operands.row, operands.column,
            ) if operands.page == PROCESS_GRAPH_OPERANDS else None)
            call_points = (book.page(SOURCE_ACTIVITY_BINDING).cell(
                call_binding.row, call_binding.column,
            ) if call_binding.page == SOURCE_ACTIVITY_BINDING else None)
            method_points = (book.page(SOURCE_ACTIVITY_BINDING).cell(
                method_binding.row, method_binding.column,
            ) if method_binding.page == SOURCE_ACTIVITY_BINDING else None)
            activity_fact = (book.page(SOURCE_ACTIVITY_CALL).cell(
                activity_call.row, activity_call.column,
            ) if activity_call.page == SOURCE_ACTIVITY_CALL else None)
            literal_page = book.page(SOURCE_LITERAL_VALUE)
            stored_literal = (literal_page.cell(store_key.row, store_key.column)
                              if store_key.page == SOURCE_LITERAL_VALUE else None)
            read_literal = (literal_page.cell(read_key.row, read_key.column)
                            if read_key.page == SOURCE_LITERAL_VALUE else None)
            version = (
                reaching_key_version(book, transferred, read_literal)
                if transferred is not None else None
            )
            version_cells = (() if version is None else tuple(dict.fromkeys((
                *((version[0],) if version[5] else ()),
                *(item for hop in version[5] for item in hop),
            ))))
            version_exact = (
                version is not None
                and version[1:5] == (store, layout, store_key, stored_value)
                and (
                    (len(fact) == 19 and not version[5])
                    or (len(fact) == 20
                        and fact[19] == (version[0], version[5])
                        and bool(version[5]))
                )
            )
            direct_method = (book.latest_ref(SOURCE_ACTIVITY_EDGE, (
                row[0], call_points[0], method_points[0], 'flow',
            )) if call_points is not None and method_points is not None else None)
            def ancestry(value):
                pending, seen = [value], set()
                while pending:
                    current = pending.pop()
                    if current in seen:
                        continue
                    seen.add(current)
                    pending.extend(source for source, _stage in
                                   book.edges_into(current))
                return seen
            wrappers = tuple(
                entry.graph for entry in (getattr(module, 'function_table', ()) or ())
                if getattr(entry, 'graph', None) is not None
                and tuple(entry.graph.G.graph.get('authored_function_identity') or ()) == row[0]
            )
            exact_method_occurrence = False
            if len(wrappers) == 1 and row[1] in wrappers[0].G:
                wrapper = wrappers[0]
                expression = wrapper.G.nodes[row[1]].get('expr_obj')
                if (isinstance(expression, ast.Call)
                        and isinstance(expression.func, ast.Attribute)
                        and expression.func.attr == 'get'
                        and len(expression.args) == 1
                        and not expression.keywords
                        and node_identity_cell(wrapper, row[1], book=book) == call):
                    method_ids = tuple(
                        int(node_id) for node_id, data in wrapper.G.nodes(data=True)
                        if data.get('expr_obj') is expression.func
                    )
                    exact_method_occurrence = (
                        len(method_ids) == 1
                        and node_identity_cell(
                            wrapper, method_ids[0], book=book,
                        ) == method
                    )
            valid = (
                exact_method_occurrence
                and
                transfer.row[3] == row[0] and transfer.row[5] == row[2]
                and transferred is not None and version_exact
                and transferred[8] == observed_cell
                and store_fact is not None and len(store_fact) == 5
                and store_fact[1:3] == ('setitem', 'unique')
                and store_fact[4] == 'mapping' and len(store_fact[3]) == 2
                and layout_fact is not None and layout_fact.aggregate_kind == 'dict'
                and len(layout_fact.members) == 2
                and tuple(member.value_id for member in layout_fact.members)
                == tuple(store_fact[3])
                and layout_fact.members[1].value_cell == stored_value
                and layout.row == (transferred[6].row[0], store.row[1])
                and store_key.row == (store.row[0], store_fact[3][0])
                and read_key.row == (row[0], key.row[-1])
                and stored_literal == read_literal
                and isinstance(stored_literal, (str, int, float, bool, bytes))
                and isinstance(observed, FieldState)
                and observed.kind is FieldStateKind.OBSERVED
                and observed.value == base
                and call.row[-1] == row[1]
                and operand_fact is not None
                and set(operand_fact) == {(base, 'operand'), (key, 'arg:0')}
                and read_fact == (base.row[-1], 'unclassified_call:get',
                                  'unique', (key.row[-1],), 'mapping')
                and activity_fact == (1, ())
                and call_points is not None and method_points is not None
                and activity_call.row == (row[0], call_points[0].row[1])
                and method_edge == direct_method
                and call_binding.row[1] in ancestry(call)
                and method_binding.row[1] in ancestry(method)
                and normal_mapping_get_prefix(
                    book, row[0], nodes[0], call_points[0],
                ) == (nodes, edges)
                and all(source in sources for source in (
                    transfer, store, layout, store_key, read_key, stored_value,
                    observed_cell, call, method, base, key, operands,
                    read_call, call_binding, activity_call, method_binding,
                    method_edge, *nodes, *edges,
                    *version_cells,
                ))
            )
            if not valid:
                found.append(Finding(
                    "source-mapping-get-read", str(row[0]), row[1],
                    "mapping get lost its transferred store, key, call, or pure-prefix proof",
                ))
        return found

    @staticmethod
    def _source_call_field_state_transfer_findings(module: Any) -> list[Finding]:
        """Check the source field cursor join without assigning payload class."""
        from .concordance_declarations import (
            CANONICAL_VALUE, CLASS_FIELD_DECLARATION, REDUCER_FIELD_STATE,
            SOURCE_CALL_ACTIVITY_CORRIDOR, SOURCE_CALL_FIELD_EFFECT_COMPLETION,
            SOURCE_CALL_FIELD_STATE_TRANSFER, SOURCE_CALL_OBJECT_ALIAS,
            SOURCE_FUNCTION_EXIT_FIELD_STATE, SOURCE_SEQUENCE_MUTATION,
            FieldState, FieldStateKind,
        )
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found = []
        page = book.page(SOURCE_CALL_FIELD_STATE_TRANSFER)
        def has_field_declaration_cause(state_cell: Ref, declaration: Ref) -> bool:
            current, seen = state_cell, set()
            while current not in seen and current.page == REDUCER_FIELD_STATE:
                seen.add(current)
                if current.row[-1] != declaration.row[-1]:
                    return False
                causes = {source for source, _stage in book.edges_into(current)}
                predecessors = tuple(source for source in causes
                                     if source.page == REDUCER_FIELD_STATE)
                if not predecessors:
                    return declaration in causes
                if len(predecessors) != 1:
                    return False
                current = predecessors[0]
            return False

        for row in page.rows():
            receipt = book.latest_ref(SOURCE_CALL_FIELD_STATE_TRANSFER, row)
            fact = page.cell(receipt.row, receipt.column)
            if not isinstance(fact, tuple) or len(fact) != 9:
                found.append(Finding(
                    "source-call-field-state-transfer", str(row[0]), row[1],
                    "field transfer has no complete source cell tuple",
                ))
                continue
            (completion, corridor, publisher_alias, consumer_alias,
             field_declaration, exit_receipt, written_state, mutation,
             observed_state) = fact
            completion_fact = book.page(SOURCE_CALL_FIELD_EFFECT_COMPLETION).cell(
                completion.row, completion.column,
            ) if completion.page == SOURCE_CALL_FIELD_EFFECT_COMPLETION else None
            corridor_fact = book.page(SOURCE_CALL_ACTIVITY_CORRIDOR).cell(
                corridor.row, corridor.column,
            ) if corridor.page == SOURCE_CALL_ACTIVITY_CORRIDOR else None
            publisher_fact = book.page(SOURCE_CALL_OBJECT_ALIAS).cell(
                publisher_alias.row, publisher_alias.column,
            ) if publisher_alias.page == SOURCE_CALL_OBJECT_ALIAS else None
            consumer_fact = book.page(SOURCE_CALL_OBJECT_ALIAS).cell(
                consumer_alias.row, consumer_alias.column,
            ) if consumer_alias.page == SOURCE_CALL_OBJECT_ALIAS else None
            exit_fact = book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE).cell(
                exit_receipt.row, exit_receipt.column,
            ) if exit_receipt.page == SOURCE_FUNCTION_EXIT_FIELD_STATE else None
            written = book.page(REDUCER_FIELD_STATE).cell(
                written_state.row, written_state.column,
            ) if written_state.page == REDUCER_FIELD_STATE else None
            observed = book.page(REDUCER_FIELD_STATE).cell(
                observed_state.row, observed_state.column,
            ) if observed_state.page == REDUCER_FIELD_STATE else None
            mutation_fact = book.page(SOURCE_SEQUENCE_MUTATION).cell(
                mutation.row, mutation.column,
            ) if mutation.page == SOURCE_SEQUENCE_MUTATION else None
            sources = {source for source, _stage in book.edges_into(receipt)}
            valid = (
                completion_fact is not None and corridor_fact is not None
                and publisher_fact is not None and consumer_fact is not None
                and isinstance(exit_fact, tuple) and len(exit_fact) == 5
                and isinstance(written, FieldState)
                and isinstance(observed, FieldState)
                and completion.row[:2] == row[:2]
                and completion.row[4] == row[5]
                and corridor.row == (row[0], row[1], row[2],
                                    publisher_fact[0], row[5])
                and corridor_fact[0] == completion
                and corridor_fact[1] == consumer_alias
                and publisher_alias.row == completion.row[:4]
                and consumer_alias.row == (row[0], row[2], row[3], row[4])
                and completion_fact[0] == publisher_fact[0] == consumer_fact[0]
                and completion_fact[2] == exit_receipt
                and publisher_fact[4] == consumer_fact[4]
                and field_declaration.page == CLASS_FIELD_DECLARATION
                and field_declaration.row == (*publisher_fact[4].row, row[5])
                and has_field_declaration_cause(written_state, field_declaration)
                and has_field_declaration_cause(observed_state, field_declaration)
                and exit_receipt.row == (completion.row[2], completion.row[3], row[5])
                and exit_fact[0] == completion_fact[1]
                and exit_fact[1] == written_state
                and written.kind is FieldStateKind.ELEMENT_WRITTEN
                and written.value == written.effect
                and written.effect.page == CANONICAL_VALUE
                and mutation.row == (completion.row[2], written.effect.row[-1])
                and isinstance(mutation_fact, tuple) and len(mutation_fact) == 5
                and mutation_fact[1] == "setitem"
                and mutation_fact[4] == "mapping"
                and observed_state.row == (consumer_fact[1].row[0], row[4], row[5])
                and observed.kind is FieldStateKind.OBSERVED
                and observed.value.page == CANONICAL_VALUE
                and {completion, corridor, publisher_alias, consumer_alias,
                     publisher_fact[4], field_declaration, exit_receipt,
                     written_state, mutation, observed_state} <= sources
            )
            if not valid:
                found.append(Finding(
                    "source-call-field-state-transfer", str(row[0]), row[1],
                    "field transfer lost the completed write, closed corridor, alias or observed formal field",
                ))
        return found

    @staticmethod
    def _source_call_activity_corridor_findings(module: Any) -> list[Finding]:
        """Re-enumerate every outgoing source edge of a certified interval."""
        from .concordance_declarations import (
            SOURCE_ACTIVITY_BINDING, SOURCE_CALL_ACTIVITY_CORRIDOR,
            SOURCE_CALL_FIELD_EFFECT_COMPLETION, SOURCE_CALL_OBJECT_ALIAS,
        )
        from .control_activity import normal_call_activity_corridor
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        page = book.page(SOURCE_CALL_ACTIVITY_CORRIDOR)
        found = []
        for row in page.rows():
            cell = book.latest_ref(SOURCE_CALL_ACTIVITY_CORRIDOR, row)
            (finished, alias, publisher_binding, consumer_binding,
             start, end, nodes, edges) = page.cell(cell.row, cell.column)
            causes = {source for source, _stage in book.edges_into(cell)}
            finished_fact = (book.page(SOURCE_CALL_FIELD_EFFECT_COMPLETION).cell(
                finished.row, finished.column,
            ) if finished.page == SOURCE_CALL_FIELD_EFFECT_COMPLETION else None)
            alias_fact = (book.page(SOURCE_CALL_OBJECT_ALIAS).cell(
                alias.row, alias.column,
            ) if alias.page == SOURCE_CALL_OBJECT_ALIAS else None)
            publisher_points = (book.page(SOURCE_ACTIVITY_BINDING).cell(
                publisher_binding.row, publisher_binding.column,
            ) if publisher_binding.page == SOURCE_ACTIVITY_BINDING else None)
            consumer_points = (book.page(SOURCE_ACTIVITY_BINDING).cell(
                consumer_binding.row, consumer_binding.column,
            ) if consumer_binding.page == SOURCE_ACTIVITY_BINDING else None)
            recomputed = normal_call_activity_corridor(book, row[0], start, end)
            if (finished.row[0] != row[0] or finished.row[1] != row[1]
                    or finished.row[4] != row[4]
                    or alias.row[0] != row[0] or alias.row[1] != row[2]
                    or finished_fact is None or alias_fact is None
                    or finished_fact[0] != row[3] or alias_fact[0] != row[3]
                    or finished_fact[4] != publisher_binding
                    or publisher_points is None or publisher_points[1] != start
                    or consumer_points is None or consumer_points[0] != end
                    or recomputed != (nodes, edges)
                    or any(source not in causes for source in (
                        finished, alias, publisher_binding, consumer_binding,
                        *nodes, *edges,
                    ))):
                found.append(Finding(
                    "source-call-activity-corridor", str(row[0]), row[1],
                    "normal caller interval lost its exact call, alias, or exhaustive flow proof",
                ))
        return found

    @staticmethod
    def _source_call_field_effect_completion_findings(module: Any) -> list[Finding]:
        """Check a successful callee field exit against the reached Call exit."""
        from .concordance_declarations import (
            CANONICAL_VALUE, INGESTION_VALUE, SOURCE_ACTIVITY_BINDING,
            SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_NODE,
            SOURCE_CALL_FIELD_EFFECT_COMPLETION, SOURCE_CALL_OBJECT_ALIAS,
            SOURCE_FUNCTION_EXIT_FIELD_STATE, SOURCE_FUNCTION_REACHABILITY,
        )
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found = []
        page = book.page(SOURCE_CALL_FIELD_EFFECT_COMPLETION)
        for row in page.rows():
            receipt = book.latest_ref(SOURCE_CALL_FIELD_EFFECT_COMPLETION, row)
            fact = page.cell(receipt.row, receipt.column)
            if not isinstance(fact, tuple) or len(fact) != 6:
                found.append(Finding(
                    "source-call-field-effect-completion", str(row[0]), row[1],
                    "completion receipt has no exact actual, formal and exit cells",
                ))
                continue
            actual, formal, exit_receipt, callee_exit, call_binding, call_exit = fact
            alias = book.latest_ref(SOURCE_CALL_OBJECT_ALIAS, row[:4])
            reach = book.latest_ref(SOURCE_FUNCTION_REACHABILITY, row[:3])
            alias_fact = (
                book.page(SOURCE_CALL_OBJECT_ALIAS).cell(alias.row, alias.column)
                if alias is not None else None
            )
            exit_fact = (
                book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE).cell(
                    exit_receipt.row, exit_receipt.column,
                ) if isinstance(exit_receipt, Ref)
                and exit_receipt.page == SOURCE_FUNCTION_EXIT_FIELD_STATE else None
            )
            call_cell = (
                book.latest_ref(CANONICAL_VALUE, (actual.row[0], row[1]))
                if isinstance(actual, Ref) and actual.page == CANONICAL_VALUE
                else None
            )
            call_sources = (
                tuple(source for source, _stage in book.edges_into(call_cell)
                      if source.page == INGESTION_VALUE)
                if call_cell is not None else ()
            )
            binding_fact = (
                book.page(SOURCE_ACTIVITY_BINDING).cell(
                    call_binding.row, call_binding.column,
                ) if isinstance(call_binding, Ref)
                and call_binding.page == SOURCE_ACTIVITY_BINDING else None
            )
            callee_fact = (
                book.page(SOURCE_ACTIVITY_NODE).cell(
                    callee_exit.row, callee_exit.column,
                ) if isinstance(callee_exit, Ref)
                and callee_exit.page == SOURCE_ACTIVITY_NODE else None
            )
            call_exit_fact = (
                book.page(SOURCE_ACTIVITY_NODE).cell(
                    call_exit.row, call_exit.column,
                ) if isinstance(call_exit, Ref)
                and call_exit.page == SOURCE_ACTIVITY_NODE else None
            )
            incoming = tuple(
                edge_row for edge_row in book.page(SOURCE_ACTIVITY_EDGE).rows()
                if edge_row[0] == row[2] and edge_row[2] == callee_exit
            )
            sources = {source for source, _stage in book.edges_into(receipt)}
            flow = exit_fact[4] if isinstance(exit_fact, tuple) and len(exit_fact) == 5 else None
            valid = (
                alias is not None and reach is not None
                and isinstance(alias_fact, tuple) and len(alias_fact) == 5
                and alias_fact[0] == actual
                and isinstance(alias_fact[1], Ref)
                and isinstance(formal, Ref) and formal.page == CANONICAL_VALUE
                and formal.row[-1] == row[3]
                and book.latest_ref(CANONICAL_VALUE, (
                    alias_fact[1].row[0], row[3],
                )) == formal
                and isinstance(exit_fact, tuple) and len(exit_fact) == 5
                and exit_receipt.row == (row[2], row[3], row[4])
                and exit_fact[0] == formal and exit_fact[2] == callee_exit
                and isinstance(flow, Ref) and flow.page == SOURCE_ACTIVITY_EDGE
                and len(incoming) == 1 and incoming[0] == flow.row
                and flow.row[3] == "flow"
                and callee_fact is not None and callee_fact[0] == "FunctionDef"
                and callee_exit.row[0] == row[2]
                and callee_exit.row[-1] == "exit"
                and len(call_sources) == 1
                and isinstance(call_binding, Ref)
                and call_binding.row == (row[0], call_sources[0])
                and binding_fact is not None and binding_fact[1] == call_exit
                and call_exit_fact is not None and call_exit_fact[0] == "Call"
                and call_exit.row[0] == row[0]
                and call_exit.row[-1] == "exit"
                and {reach, alias, call_cell, actual, formal,
                     exit_receipt, callee_exit, flow,
                     call_binding, call_exit} <= sources
            )
            if not valid:
                found.append(Finding(
                    "source-call-field-effect-completion", str(row[0]), row[1],
                    "completion lost its reached call, alias or successful function exit",
                ))
        return found

    @staticmethod
    def _source_call_object_alias_findings(module: Any) -> list[Finding]:
        """Check a call's record alias against the exact formal and class cells."""
        from .concordance_declarations import (
            CLASS_DECLARATION, PARAMETER_ANNOTATION, SOURCE_CALL_ACTUAL_FORMAL,
            SOURCE_CALL_OBJECT_ALIAS, SOURCE_PARAMETER_MEMORY_CONTRACT,
            SOURCE_VALUE_CLASS,
        )
        from ..transmogrifier.function_table import (
            ParameterAccess, ParameterContract, ParameterScope,
            ParameterStorage, ParameterTransfer,
        )
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found = []
        alias_page = book.page(SOURCE_CALL_OBJECT_ALIAS)
        for row in alias_page.rows():
            alias = book.latest_ref(SOURCE_CALL_OBJECT_ALIAS, row)
            actual, formal_binding, memory, actual_class, declaration = (
                alias_page.cell(alias.row, alias.column)
            )
            sources = {source for source, _stage in book.edges_into(alias)}
            binding = book.latest_ref(SOURCE_CALL_ACTUAL_FORMAL, row)
            binding_fact = (None if binding is None else
                            book.page(SOURCE_CALL_ACTUAL_FORMAL).cell(
                                binding.row, binding.column,
                            ))
            memory_fact = (None if memory.page != SOURCE_PARAMETER_MEMORY_CONTRACT
                           else book.page(SOURCE_PARAMETER_MEMORY_CONTRACT).cell(
                               memory.row, memory.column,
                           ))
            contract = (memory_fact[0] if isinstance(memory_fact, tuple)
                        and len(memory_fact) == 1 else None)
            memory_sources = ({source for source, _stage in book.edges_into(memory)}
                              if memory_fact is not None else set())
            annotation = (book.latest_ref(
                PARAMETER_ANNOTATION, (*row[2], binding_fact[2]),
            ) if binding_fact is not None else None)
            annotation_sources = ({source for source, _stage in
                                   book.edges_into(annotation)}
                                  if annotation is not None else set())
            class_fact = (book.page(SOURCE_VALUE_CLASS).cell(
                actual_class.row, actual_class.column,
            ) if actual_class.page == SOURCE_VALUE_CLASS else None)
            valid = (
                binding in sources
                and all(cell in sources for cell in (
                    actual, formal_binding, memory, actual_class, declaration,
                ))
                and binding_fact is not None
                and binding_fact[0] == actual
                and binding_fact[4] == formal_binding
                and memory.row == (row[2], binding_fact[2])
                and annotation in memory_sources
                and any(source.page == SOURCE_PARAMETER_MEMORY_CONTRACT
                        for source in memory_sources)
                and formal_binding in memory_sources
                and declaration in memory_sources
                and declaration in annotation_sources
                and declaration.page == CLASS_DECLARATION
                and declaration.row[0] == row[0][0]
                and actual_class.row == (row[0], actual.row[-1])
                and isinstance(class_fact, tuple) and len(class_fact) >= 2
                and class_fact[0] == declaration.row[1]
                and isinstance(contract, ParameterContract)
                and contract.name == binding_fact[2]
                and contract.transfer is ParameterTransfer.ALIAS
                and contract.access is ParameterAccess.INOUT
                and contract.storage is ParameterStorage.RECORD
                and contract.scope is ParameterScope.CALLER
            )
            if not valid:
                found.append(Finding(
                    "source-call-object-alias", str(row[0]), row[1],
                    "call alias lost its actual, formal contract, or class proof",
                ))
        return found

    @staticmethod
    def _source_call_actual_formal_findings(module: Any) -> list[Finding]:
        """Check source call-frame bindings against their exact graph receipts."""
        from .concordance_declarations import (
            CANONICAL_VALUE, NAME_BINDING, PROCESS_GRAPH_OPERANDS,
            SOURCE_ACTIVITY_SCOPE, SOURCE_CALL_ACTUAL_FORMAL, SOURCE_FORMAL_ORDER,
            SOURCE_FUNCTION_REACHABILITY,
        )
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found = []
        page = book.page(SOURCE_CALL_ACTUAL_FORMAL)
        for row in page.rows():
            cell = book.latest_ref(SOURCE_CALL_ACTUAL_FORMAL, row)
            actual, role, parameter, operand, formal_binding = page.cell(
                cell.row, cell.column,
            )
            sources = {source for source, _stage in book.edges_into(cell)}
            reach = book.latest_ref(SOURCE_FUNCTION_REACHABILITY, row[:3])
            order = book.latest_ref(SOURCE_FORMAL_ORDER, (row[2],))
            edges = (None if operand.page != PROCESS_GRAPH_OPERANDS else
                     book.page(PROCESS_GRAPH_OPERANDS).cell(operand.row, operand.column))
            formal_cells = tuple(
                source for source in sources
                if source.page == CANONICAL_VALUE
                and source.row[-1] == row[3]
                and (scope_cell := book.latest_ref(
                    SOURCE_ACTIVITY_SCOPE, (source.row[0],),
                )) in sources
                and book.page(SOURCE_ACTIVITY_SCOPE).cell(
                    scope_cell.row, scope_cell.column,
                ) == row[2]
            )
            formal_fact = (
                None if formal_binding.page != NAME_BINDING else
                book.page(NAME_BINDING).cell(
                    formal_binding.row, formal_binding.column,
                )
            )
            if (reach is None or order is None or reach not in sources
                    or order not in sources or actual not in sources
                    or operand not in sources or edges is None
                    or formal_binding not in sources
                    or formal_binding.row[1] != parameter
                    or formal_fact is None or not formal_fact.authored
                    or int(formal_fact.value_id) != int(row[3])
                    or not any(formal_binding.row[0] == formal.row[0]
                               for formal in formal_cells)
                    or (actual, role) not in edges
                    or parameter not in book.page(SOURCE_FORMAL_ORDER).cell(
                        order.row, order.column,
                    ) or not formal_cells):
                found.append(Finding(
                    "source-call-actual-formal", str(row[0]), int(row[1]),
                    "call-frame binding lost reachability, actual, formal, "
                    "order or operand receipt",
                ))
        return found

    @staticmethod
    def _control_activity_findings(module: Any) -> list[Finding]:
        from .concordance_declarations import (
            SOURCE_ACTIVITY_NODE, SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_BINDING, SOURCE_ACTIVITY_ARM_RELATION,
            SOURCE_LAMBDA_NORMALIZATION, SOURCE_TERMINAL_RETURN,
            SOURCE_ACTIVITY_SCOPE, SOURCE_ACTIVITY_POLICY, SOURCE_ACTIVITY_SCHEDULE, SOURCE_SPAN,
            SOURCE_RESOURCE_CONVERSION, SOURCE_RESOURCE_SCOPE, SOURCE_RESOURCE_ADMISSION_DEMAND, SOURCE_RESOURCE_CONTROL, SOURCE_PROCESS_OPERATION, SOURCE_LOOP_CONTROL_WORK,
            SOURCE_ACTIVITY_CHILD, SOURCE_ACTIVITY_CALL, SOURCE_ACTIVITY_CUT, SOURCE_CONTROL_OPERATOR,
            SOURCE_CONTROL_IR, CONTROL_PROGRAM, CONTROL_BLOCK, CONTROL_BLOCK_PLACEMENT,
            ControlBlockKind, Placement, SOURCE_LOOP_CONTROL_SITE, SOURCE_LOOP_SITE_EDGE, SOURCE_LOOP_SITE_VIEW, SOURCE_LOOP_EXIT_READ, PROCESS_GRAPH_OPERANDS, INGESTION_VALUE, SSA_CONTROL_EDGE, SSA_ARM_CONTINUATION, SSA_BLOCK,
            DEPLOYMENT_REGION_MEMBER, MemberRole,
        )
        import networkx as nx
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found, graphs = [], {}
        def value(cell):
            return book.page(cell.page).cell(cell.row, cell.column)
        for row in book.page(SOURCE_LAMBDA_NORMALIZATION).rows():
            cell = book.latest_ref(SOURCE_LAMBDA_NORMALIZATION, row)
            generated_function, generated_return = value(cell)
            sources = {source for source, _stage in book.edges_into(cell)}
            authored = row[0]
            function_sources = {
                source for source, _stage in book.edges_into(generated_function)
            }
            return_sources = {
                source for source, _stage in book.edges_into(generated_return)
            }
            function_activity = tuple(
                activity_row for activity_row in book.page(
                    SOURCE_ACTIVITY_NODE).rows()
                if activity_row[1] == generated_function
                and activity_row[2] in {'enter', 'exit'}
            )
            return_activity = tuple(
                activity_row for activity_row in book.page(
                    SOURCE_ACTIVITY_NODE).rows()
                if activity_row[1] == generated_return
                and activity_row[2] in {'enter', 'exit'}
            )
            if (authored.page != SOURCE_SPAN
                    or value(authored).kind != 'Lambda'
                    or generated_function.page != SOURCE_SPAN
                    or value(generated_function).kind != 'FunctionDef'
                    or generated_return.page != SOURCE_SPAN
                    or value(generated_return).kind != 'Return'
                    or generated_function.row[:2] != generated_return.row[:2]
                    or generated_function.row[2] != ()
                    or generated_return.row[2] != (('body', 0),)
                    or authored not in function_sources
                    or authored not in return_sources
                    or not {authored, generated_function,
                            generated_return} <= sources
                    or len(function_activity) != 2
                    or len(return_activity) != 2):
                found.append(Finding(
                    'source-lambda-normalization', str(authored.row[:2]), None,
                    'lambda wrapper lost its authored occurrence, generated '
                    'control spans or activity identities',
                ))
        for row in book.page(SOURCE_RESOURCE_CONVERSION).rows():
            cell = book.latest_ref(SOURCE_RESOURCE_CONVERSION, row)
            sources = {source for source, _stage in book.edges_into(cell)}
            if row[0].page != SOURCE_SPAN or value(row[0]).kind != 'With' or row[0] not in sources:
                found.append(Finding('resource-conversion', '<source>', None,
                                     'resource conversion lost its exact authored With occurrence'))
        for row in book.page(SOURCE_RESOURCE_SCOPE).rows():
            cell = book.latest_ref(SOURCE_RESOURCE_SCOPE, row)
            conversion, entry, cleanup = value(cell)
            sources = {source for source, _stage in book.edges_into(cell)}
            if (conversion.page != SOURCE_RESOURCE_CONVERSION or row[0].page != INGESTION_VALUE
                    or value(row[0]).type != 'Try' or entry.page != SOURCE_SPAN or cleanup.page != SOURCE_SPAN
                    or not {row[0], conversion, entry, cleanup} <= sources):
                found.append(Finding('resource-ingestion', '<source>', None,
                                     'resource scope lost its conversion and exact entry/cleanup occurrences'))
        for row in book.page(SOURCE_RESOURCE_ADMISSION_DEMAND).rows():
            cell = book.latest_ref(SOURCE_RESOURCE_ADMISSION_DEMAND, row)
            source, activity, calls = value(cell)
            sources = {item for item, _stage in book.edges_into(cell)}
            required = {source, activity}
            for call, receiver, inputs in calls:
                required.update(item for item in (call, receiver) if item is not None)
                required.update(item for _role, item in inputs if item is not None)
            if (source.page != INGESTION_VALUE or value(source).type not in {'With', 'AsyncWith'}
                    or source.row[1] != row[1] or activity.page != SOURCE_ACTIVITY_BINDING
                    or not required <= sources):
                found.append(Finding('resource-admission-demand', str(row[0]), int(row[1]),
                                     'resource demand lost its exact With, activity, call or input identities'))
        for row in book.page(SOURCE_RESOURCE_CONTROL).rows():
            cell = book.latest_ref(SOURCE_RESOURCE_CONTROL, row)
            source, control, entry, cleanup, receiver = value(cell)
            base = _canonical_identity_base(book, cell)
            sources = {item for item, _stage in book.edges_into(base)}
            conversion = value(source)[0]
            recipe = value(conversion)
            if (source.page != SOURCE_RESOURCE_SCOPE or control.page != CONTROL_BLOCK
                    or value(control).kind != ControlBlockKind.RESOURCE_SCOPE
                    or entry.page != SOURCE_PROCESS_OPERATION or cleanup.page != SOURCE_PROCESS_OPERATION
                    or value(entry)[0] != recipe[1] or value(cleanup)[0] != recipe[2]
                    or not {source, control, entry, cleanup, receiver} <= sources
                    or source not in value(control).extra):
                found.append(Finding('resource-control', str(row[0]), int(row[1]),
                                     'resource control lost its placed scope, receiver or process contracts'))
        for row in book.page(SOURCE_ACTIVITY_ARM_RELATION).rows():
            cell = book.latest_ref(SOURCE_ACTIVITY_ARM_RELATION, row)
            sources = {source for source, _stage in book.edges_into(cell)}
            if (row[2:] != ('body', 'finalbody') or value(cell) != 'resource_cleanup'
                    or not any(source.page == SOURCE_RESOURCE_CONVERSION for source in sources)):
                found.append(Finding('resource-arm-relation', str(row[0]), None,
                                     'ordered cleanup arms lack their exact resource conversion'))
        for row in book.page(SOURCE_TERMINAL_RETURN).rows():
            cell = book.latest_ref(SOURCE_TERMINAL_RETURN, row)
            enter, leave, owners = value(cell)
            sources = {source for source, _stage in book.edges_into(cell)}
            occurrence = row[1]
            protected = any(
                value(owner).kind in {'Try', 'TryStar', 'With', 'AsyncWith'}
                for owner, _arm in owners
            )
            outgoing = tuple(edge_row for edge_row in
                             book.page(SOURCE_ACTIVITY_EDGE).scope_rows(row[0])
                             if edge_row[1] == leave)
            if (occurrence.page != SOURCE_SPAN
                    or value(occurrence).kind != 'Return'
                    or enter.page != SOURCE_ACTIVITY_NODE
                    or leave.page != SOURCE_ACTIVITY_NODE
                    or enter.row != (row[0], occurrence, 'enter')
                    or leave.row != (row[0], occurrence, 'exit')
                    or value(enter) != ('Return', owners)
                    or value(leave) != ('Return', owners)
                    or protected or outgoing
                    or not {occurrence, enter, leave,
                            *(owner for owner, _arm in owners)} <= sources):
                found.append(Finding(
                    'source-terminal-return', str(row[0]), None,
                    'unprotected Return lost its exact terminal activity cut',
                ))
        for row in book.page(SOURCE_LOOP_CONTROL_WORK).rows():
            cell = book.latest_ref(SOURCE_LOOP_CONTROL_WORK, row)
            sites, operations = value(cell)
            base = _canonical_identity_base(book, cell)
            sources = {source for source, _stage in book.edges_into(base)}
            if (not sites and not operations or not {*sites, *operations} <= sources
                    or any(operation.page != SOURCE_PROCESS_OPERATION for operation in operations)):
                found.append(Finding('loop-control-work', str(row[0]), int(row[1]),
                                     'region-less loop retention lacks its exact process/exit work'))
        for row in book.page(SOURCE_ACTIVITY_NODE).rows():
            cell = book.latest_ref(SOURCE_ACTIVITY_NODE, row)
            kind, owners = value(cell)
            sources = {source for source, _stage in book.edges_into(cell)}
            if (row[1].page != SOURCE_SPAN or row[1] not in sources
                    or not isinstance(kind, str)
                    or any(owner.page != SOURCE_SPAN or not isinstance(arm, str) for owner, arm in owners)):
                found.append(Finding("activity-node-source", str(row[0]), None,
                                     "activity node lacks its exact ingested source/owner identity"))
            graphs.setdefault(row[0], nx.DiGraph()).add_node(cell)
        for row in book.page(SOURCE_ACTIVITY_EDGE).rows():
            scope, before, after, relation = row
            cell = book.latest_ref(SOURCE_ACTIVITY_EDGE, row)
            sources = {source for source, _stage in book.edges_into(cell)}
            if (before.page != SOURCE_ACTIVITY_NODE or after.page != SOURCE_ACTIVITY_NODE
                    or before.row[0] != scope or after.row[0] != scope
                    or not {before, after} <= sources):
                found.append(Finding("activity-edge-source", str(scope), None,
                                     "control edge crosses an undeclared activity scope or lacks its endpoints"))
                continue
            left, right = dict(value(before)[1]), dict(value(after)[1])
            incompatible = []
            for owner in left.keys() & right.keys():
                if left[owner] == right[owner]:
                    continue
                declaration = book.latest_ref(SOURCE_ACTIVITY_ARM_RELATION,
                    (scope, owner, left[owner], right[owner]))
                if declaration is None or declaration not in sources or value(declaration) != relation:
                    incompatible.append(owner)
            if relation != "repeat" and incompatible:
                found.append(Finding("activity-exclusive-arms", str(scope), None,
                                     "control edge orders two mutually exclusive arms"))
            if relation not in {"repeat", "unadmitted"}:
                graphs[scope].add_edge(before, after)
        for scope, graph in graphs.items():
            if not nx.is_directed_acyclic_graph(graph):
                found.append(Finding("activity-region-cycle", str(scope), None,
                                     "static activity graph has an undeclared recurrence"))
        for row in book.page(SOURCE_ACTIVITY_SCOPE).rows():
            cell = book.latest_ref(SOURCE_ACTIVITY_SCOPE, row)
            if value(cell) not in graphs:
                found.append(Finding("activity-scope-view", str(row[0]), None,
                                     "read scope names no ingested activity graph"))
        for row in book.page(SOURCE_ACTIVITY_CHILD).rows():
            cell = book.latest_ref(SOURCE_ACTIVITY_CHILD, row)
            enter, leave = value(cell)
            if (enter.row[0] != row[0] or leave.row[0] != row[0]
                    or value(row[1])[0] not in {"List", "Tuple", "Set"}
                    or not {row[1], enter, leave} <= {source for source, _stage in book.edges_into(cell)}):
                found.append(Finding("activity-display-member", str(row[0]), None,
                                     "display member lost its exact evaluation points"))
        for row in book.page(SOURCE_ACTIVITY_CALL).rows():
            cell = book.latest_ref(SOURCE_ACTIVITY_CALL, row)
            arity, names = value(cell)
            if arity < 0 or any(name is not None and not isinstance(name, str) for name in names):
                found.append(Finding("activity-call-signature", str(row[0]), None,
                                     "call arity/keyword facts are invalid"))
        for row in book.page(SOURCE_ACTIVITY_BINDING).rows():
            cell = book.latest_ref(SOURCE_ACTIVITY_BINDING, row)
            first, last = value(cell)
            sources = {source for source, _stage in book.edges_into(cell)}
            if (first.row[0] != row[0] or last.row[0] != row[0]
                    or not {row[1], first, last} <= sources):
                found.append(Finding("activity-process-binding", str(row[0]), None,
                                     "process node lost its exact entry/completion binding"))
        for row in book.page(SOURCE_ACTIVITY_SCHEDULE).rows():
            cell = book.latest_ref(SOURCE_ACTIVITY_SCHEDULE, row)
            levels = dict(value(cell)[0])
            graph = graphs[row[0]]
            sources = {source for source, _stage in book.edges_into(cell)}
            policies = [source for source in sources if source.page == SOURCE_ACTIVITY_POLICY]
            if (not any(value(policy) == row[1] for policy in policies)
                    or set(levels) != set(graph)
                    or any(levels.get(before, -1) >= levels.get(after, -1) for before, after in graph.edges)):
                found.append(Finding("activity-schedule", str(row[0]), None,
                                     "ASAP/ALAP activity arrangement violates its configured graph edges"))
        from .control_source import DispatchBlock
        for row in book.page(SOURCE_CONTROL_IR).rows():
            cell = book.latest_ref(SOURCE_CONTROL_IR, row)
            program, operators, regions, placements = value(cell)
            sources = {source for source, _stage in book.edges_into(cell)}
            if (program.page != CONTROL_PROGRAM or not {program, *operators, *regions, *placements} <= sources):
                found.append(Finding("source-control-ir", str(row[0]), None,
                                     "dual IR lost its actual control program or placed block identities"))
        for row in book.page(SOURCE_CONTROL_OPERATOR).rows():
            cell = book.latest_ref(SOURCE_CONTROL_OPERATOR, row)
            program, control, placement, block = value(cell)
            if (not isinstance(block, DispatchBlock) or block.callsite_id != row[1]
                    or control.page != CONTROL_BLOCK or value(control).kind != ControlBlockKind.DISPATCH
                    or placement.page != CONTROL_BLOCK_PLACEMENT or not isinstance(value(placement), Placement)
                    or placement.row[1] != control or block.operand_graph_cell is None
                    or block.process_operation_cell not in value(control).extra
                    or block.operand_graph_cell not in value(control).extra):
                found.append(Finding("ingested-process-control", str(row[0]), int(row[1]),
                                     "process operation lacks its control-IR block, placement or explicit operand graph"))
        for row in book.page(SOURCE_ACTIVITY_CUT).rows():
            cell = book.latest_ref(SOURCE_ACTIVITY_CUT, row)
            sources = {source for source, _stage in book.edges_into(cell)}
            if (not any(source.page == SOURCE_ACTIVITY_SCHEDULE for source in sources)
                    or not any(source.page == SOURCE_ACTIVITY_BINDING for source in sources)
                    or any(source.page != SOURCE_CONTROL_OPERATOR or source not in sources for source in value(cell))):
                found.append(Finding("activity-partition-proof", str(row[0]), int(row[1]),
                                     "region partition lacks its exact process/activity proof"))
        for row in book.page(SSA_CONTROL_EDGE).rows():
            cell = book.latest_ref(SSA_CONTROL_EDGE, row)
            scope, function, before, after, arm = row
            sources = {source for source, _stage in book.edges_into(cell)}
            if (before.page != SSA_BLOCK or after.page != SSA_BLOCK
                    or before.row[:2] != (scope, function) or after.row[:2] != (scope, function)
                    or not {before, after} <= sources or arm not in {'flow', 'body', 'orelse'}
                    or value(cell) is not True):
                found.append(Finding("ssa-control-edge", str(function), None,
                                     "SSA branch lacks its exact registered endpoints"))
        for row in book.page(SSA_ARM_CONTINUATION).rows():
            cell = book.latest_ref(SSA_ARM_CONTINUATION, row)
            scope, function, entry, tip = row
            sources = {source for source, _stage in book.edges_into(cell)}
            graph = nx.DiGraph()
            graph.add_nodes_from((entry, tip))
            graph.add_edges_from((source.row[2], source.row[3]) for source in sources
                                 if source.page == SSA_CONTROL_EDGE)
            if not {entry, tip} <= sources or value(cell) != nx.has_path(graph, entry, tip):
                found.append(Finding("ssa-arm-continuation", str(function), None,
                                     "arm fall-through disagrees with its concorded CFG snapshot"))
        for row in book.page(SOURCE_LOOP_CONTROL_SITE).rows():
            cell = book.latest_ref(SOURCE_LOOP_CONTROL_SITE, row)
            action, bindings, owner = value(cell)
            sources = {source for source, _stage in book.edges_into(cell)}
            if (row[0].page != INGESTION_VALUE or action not in {'break', 'continue'}
                    or row[0] not in sources or owner is None or owner.page != SOURCE_SPAN
                    or any(identity not in sources for _name, identity in bindings)):
                found.append(Finding("loop-site-origin", "<source>", None,
                                     "loop exit lost its exact site, lexical state or owner"))
        for row in book.page(SOURCE_LOOP_SITE_EDGE).rows():
            cell = book.latest_ref(SOURCE_LOOP_SITE_EDGE, row)
            sources = {source for source, _stage in book.edges_into(cell)}
            site = book.latest_ref(SOURCE_LOOP_CONTROL_SITE, (row[1],))
            if (site is None or site not in sources or row[0] not in sources
                    or any(identity not in sources for triple in value(cell) for identity in triple)):
                found.append(Finding("loop-site-state", "<source>", None,
                                     "loop exit lacks its exact captured state and carried identities"))
        for row in book.page(SOURCE_LOOP_SITE_VIEW).rows():
            cell = book.latest_ref(SOURCE_LOOP_SITE_VIEW, row)
            edge, triples = value(cell)
            base = _canonical_identity_base(book, cell)
            sources = {source for source, _stage in book.edges_into(base)}
            if edge.page != SOURCE_LOOP_SITE_EDGE or edge not in sources:
                found.append(Finding("loop-site-projection", str(row[0]), int(row[1]),
                                     "local loop exit state lacks its source graph edge"))
        for row in book.page(SOURCE_LOOP_EXIT_READ).rows():
            cell = book.latest_ref(SOURCE_LOOP_EXIT_READ, row)
            base = _canonical_identity_base(book, cell)
            sources = {source for source, _stage in book.edges_into(base)}
            if any(operands.page != PROCESS_GRAPH_OPERANDS or operands not in sources
                   or binding.page != SOURCE_ACTIVITY_BINDING or binding not in sources
                   for operands, binding in value(cell)):
                found.append(Finding("loop-exit-read", str(row[0]), int(row[1]),
                                     "loop exit read lacks its operand and control ownership proof"))
        regions = {}
        for row in book.page(DEPLOYMENT_REGION_MEMBER).rows():
            if book.page(DEPLOYMENT_REGION_MEMBER).latest(row) not in {MemberRole.NODE, MemberRole.OUTPUT}:
                continue
            cut = book.latest_ref(SOURCE_ACTIVITY_CUT, (row[0], row[2]))
            if cut is not None:
                regions.setdefault(row[:2], set()).add(value(cut))
        for row, cuts in regions.items():
            if len(cuts) > 1:
                found.append(Finding("activity-region-crossing", str(row[0]), int(row[1]),
                                     "atomic region spans an authored process boundary"))
        return found

    @staticmethod
    def _process_graph_fact_findings(module: Any) -> list[Finding]:
        from .concordance_declarations import (
            PROCESS_GRAPH_OPERANDS, SYNCHRONIZATION_LITERAL, SOURCE_OPERATOR_KIND,
            SOURCE_THREAD_ARGUMENT_PACK, SOURCE_THREAD_ARGUMENT,
            SSA_PROCESS_CALL, SOURCE_PROCESS_OPERATION,
        )
        from ..transmogrifier.ssa_registry import Handler
        from .concorded_literals import _INTEGER_OPERATORS
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found = []
        def ancestors(cell):
            visited, pending = set(), [cell]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(source for source, _stage in book.edges_into(current))
            return visited
        for row in book.page(PROCESS_GRAPH_OPERANDS).rows():
            cell = book.latest_ref(PROCESS_GRAPH_OPERANDS, row)
            edges = book.page(PROCESS_GRAPH_OPERANDS).cell(cell.row, cell.column)
            # Ask the exact reachability question this check needs. A region
            # copy legitimately has the whole control graph in its ancestry;
            # expanding that graph once per operand snapshot is quadratic.
            # Stop only after every required identity has actually been seen.
            required = {identity for identity, _role in edges if isinstance(identity, Ref)}
            sources, pending = {cell}, [cell]
            while pending and not required <= sources:
                current = pending.pop()
                for source, _stage in book.edges_into(current):
                    if source not in sources:
                        sources.add(source)
                        pending.append(source)
            if any(not isinstance(identity, Ref) or identity not in sources
                   or not isinstance(role, str) for identity, role in edges):
                found.append(Finding("process-operand-provenance", str(row[0]), row[1],
                                     "operand snapshot lacks its exact identity causes"))
        from .concordance_declarations import PROCESS_ESCAPE_CLASSIFICATION
        for row in book.page(PROCESS_ESCAPE_CLASSIFICATION).rows():
            cell = book.latest_ref(PROCESS_ESCAPE_CLASSIFICATION, row)
            semantic, classification = book.page(PROCESS_ESCAPE_CLASSIFICATION).latest(row)
            sources = {source for source, _stage in book.edges_into(cell)}
            operation = row[0]
            if (operation.page != SOURCE_PROCESS_OPERATION or operation not in sources
                    or classification != 'typed_graph_operation'
                    or book.page(SOURCE_PROCESS_OPERATION).cell(operation.row, operation.column)[1] != semantic):
                found.append(Finding('process-escape-classification', str(row), None,
                                     'escape analysis did not read the exact typed graph operation'))
        for row in book.page(SYNCHRONIZATION_LITERAL).rows():
            cell = book.latest_ref(SYNCHRONIZATION_LITERAL, row)
            operator, operands, _value = book.page(SYNCHRONIZATION_LITERAL).cell(cell.row, cell.column)
            sources = ancestors(cell)
            kinds = [source for source in sources if source.page == SOURCE_OPERATOR_KIND]
            if (Handler(operator) not in _INTEGER_OPERATORS
                    or not any(book.page(SOURCE_OPERATOR_KIND).cell(kind.row, kind.column) == operator
                               for kind in kinds)
                    or any(operand not in sources for operand in operands)
                    or not any(source.page == PROCESS_GRAPH_OPERANDS for source in sources)):
                found.append(Finding("synchronization-literal-provenance", str(row), None,
                                     "literal decision lacks its operator and operand graph proof"))
        from .ssa_record_return_state import ssa_value_identity_cell
        for name, function in module.functions.items():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.res is None:
                        continue
                    result = ssa_value_identity_cell(function, instruction.res.id, book=book)
                    cell = book.latest_ref(SSA_PROCESS_CALL, (result,))
                    if cell is None:
                        continue
                    call = book.page(SSA_PROCESS_CALL).cell(cell.row, cell.column)
                    source = book.page(SOURCE_PROCESS_OPERATION).cell(call.source.row, call.source.column)
                    if (call.semantic != instruction.op or source[1] != call.semantic
                            or source[2] != call.result_dtype or call.result_dtype != instruction.res.dtype
                            or call.argument_ids != tuple(value.id for value in instruction.args)):
                        found.append(Finding("ssa-process-call", str(name), instruction.res.id,
                                             "SSA process instruction disagrees with its concorded call contract"))
        for row in book.page(SOURCE_THREAD_ARGUMENT_PACK).rows():
            cell = book.latest_ref(SOURCE_THREAD_ARGUMENT_PACK, row)
            bindings = book.page(SOURCE_THREAD_ARGUMENT_PACK).cell(cell.row, cell.column)
            sources = ancestors(cell)
            if any(binding.page != SOURCE_THREAD_ARGUMENT or binding.row != (row[0], position)
                   or binding not in sources for position, binding in enumerate(bindings)):
                found.append(Finding("thread-argument-pack", str(row), None,
                                     "worker argument pack disagrees with its ordered binding cells"))
        return found

    @staticmethod
    def _operator_safety_findings(module: Any) -> list[Finding]:
        from .concordance_declarations import (
            COMPILE_POLICY, C_OPERATOR_SAFETY, SOURCE_EXCEPTION_OPERATION,
            SOURCE_PROCESS_OPERATION, C_EXCEPTION_OPERATION, SOURCE_EXCEPTION_INGESTION, SOURCE_EXCEPTION_NODE, INGESTION_VALUE, SOURCE_SPAN,
        )
        from ..transmogrifier.operator_safety import OPERATOR_FAILURE_ARMS
        from .thread_operations import PROCESS_OPERATORS
        from .emission_concordance import value_cell
        from .ssa_record_return_state import accepted_ssa_component
        from .frame_identity_book import _frame_owner_book
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found, instructions = [], {}
        def fact(cell):
            return book.page(cell.page).cell(cell.row, cell.column)
        for row in book.page(SOURCE_EXCEPTION_INGESTION).rows():
            cell = book.latest_ref(SOURCE_EXCEPTION_INGESTION, row)
            sources = {source for source, _stage in book.edges_into(cell)}
            import hashlib
            original, _converted, _checks = fact(cell)
            if (row[0].page != SOURCE_SPAN or row[0] not in sources
                    or hashlib.sha256(original.encode()).hexdigest() != fact(row[0]).dump_sha256):
                found.append(Finding("exception-ingestion-origin", "<source>", None,
                                     "exception conversion lost its exact authored definition"))
        for row in book.page(SOURCE_EXCEPTION_NODE).rows():
            cell = book.latest_ref(SOURCE_EXCEPTION_NODE, row)
            conversion = fact(cell)
            if (row[0].page != INGESTION_VALUE or conversion.page != SOURCE_EXCEPTION_INGESTION
                    or conversion not in {source for source, _stage in book.edges_into(row[0])}
                    or not {row[0], conversion} <= {source for source, _stage in book.edges_into(cell)}):
                found.append(Finding("exception-ingestion-node", "<source>", None,
                                     "generated exception node lacks its ingestion conversion"))
        for name, fn in module.functions.items():
            # A linked _read_header's ExceptionNone/Make/Raise instructions
            # retain SOURCE_EXCEPTION_OPERATION in its accepted owner book.
            # The composed caller book owns new edges, not that body's origin.
            accepted = accepted_ssa_component(fn)
            instruction_book = _frame_owner_book(fn)[0] if accepted else book
            for block in fn.blocks.values():
                for instruction in block.instrs:
                    attrs = instruction.attributes or {}
                    if instruction.op in OPERATOR_FAILURE_ARMS and "operator_safety_arms" in attrs:
                        policy = attrs.get("operator_safety_policy")
                        enabled = attrs["operator_safety_arms"]
                        if (not isinstance(policy, Ref) or policy.page != COMPILE_POLICY
                                or policy.row != ("operator_safety_arms",)
                                or fact(policy) != ("on" if enabled else "off")):
                            found.append(Finding("ssa-operator-safety-policy", name, instruction.res.id,
                                                 "division safety arms disagree with their compiler option"))
                        if enabled and (not attrs.get("effectful") or not attrs.get("ordered_effect")):
                            found.append(Finding("ssa-operator-safety-effect", name, instruction.res.id,
                                                 "division lost its ordered failure effect"))
                        cell = value_cell(book, fn, instruction.res)
                        if cell is not None: instructions[(cell, name)] = instruction
                    source = attrs.get("exception_operation_cell")
                    if source is not None:
                        data = instruction_book.page(source.page).cell(
                            source.row, source.column,
                        )
                        if (source.page != SOURCE_EXCEPTION_OPERATION or not data
                                or data[0] != attrs.get("dispatch_operation")
                                or not isinstance(data[7], Ref)
                                or data[7].page != SOURCE_PROCESS_OPERATION
                                or data[7] not in {x for x, _stage in instruction_book.edges_into(source)}
                                or (data[4] is None) != (data[5] is None)
                                or (data[4] is not None and (
                                    not isinstance(data[4], Ref)
                                    or data[4] not in {x for x, _stage in instruction_book.edges_into(source)}
                                    or not data[6]))):
                            found.append(Finding("ssa-exception-provenance", name, instruction.res.id,
                                                 "exception operation lost its process-graph origin"))
        for row in book.page(C_OPERATOR_SAFETY).rows():
            data = book.page(C_OPERATOR_SAFETY).latest(row)
            instruction = instructions.get(row)
            if (instruction is None or data != (instruction.op, tuple(a.id for a in instruction.args),
                                                instruction.failure_arms_for(book), instruction.attributes["operator_safety_policy"])):
                found.append(Finding("c-operator-safety", str(row[1]), None,
                                     "C division arms disagree with the SSA operator definition"))
        for row in book.page(C_EXCEPTION_OPERATION).rows():
            source, name, _result = row
            data = book.page(C_EXCEPTION_OPERATION).latest(row)
            function = module.functions.get(name)
            accepted = (
                function is not None and accepted_ssa_component(function)
            )
            source_book, digest = (
                _frame_owner_book(function) if accepted else (book, None)
            )
            source_fact = (
                None if source.page != SOURCE_EXCEPTION_OPERATION else
                source_book.page(SOURCE_EXCEPTION_OPERATION).cell(
                    source.row, source.column,
                )
            )
            import_ref = (
                book.latest_ref(IMPORTED_CELL, (digest, source.key))
                if source_book is not book and source_fact is not None else None
            )
            imported = (
                None if import_ref is None else
                book.page(IMPORTED_CELL).cell(import_ref.row, import_ref.column)
            )
            expected_cause = source if source_book is book else import_ref
            policy_ref = book.latest_ref(C_EXCEPTION_OPERATION, row)
            causes = (
                set() if policy_ref is None else
                {cell for cell, _stage in book.edges_into(policy_ref)}
            )
            if (source_fact is None
                    or PROCESS_OPERATORS.get(source_fact[0]) != data[0]
                    or expected_cause is None or expected_cause not in causes
                    or (source_book is not book and (
                        not isinstance(imported, ImportedCell)
                        or imported.source != source
                        or imported.fact != source_fact
                        or imported.owner is not source_book))):
                found.append(Finding("c-exception-provenance", str(name), None,
                                     "C exception operation disagrees with its source identity"))
        return found

    @staticmethod
    def _thread_table_findings(module: Any) -> list[Finding]:
        from .concordance_declarations import (
            SOURCE_THREAD_BODY, SOURCE_THREAD_ACTIVATION, SSA_THREAD_BODY,
            SOURCE_CLOSURE_CAPTURE, SOURCE_THREAD_ARGUMENT, SOURCE_AGGREGATE_LAYOUT, NAME_BINDING,
        )
        table = getattr(module, "thread_table", None)
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if table is None:
            return []
        found = []
        if table.book is not book or table.module is not module:
            found.append(Finding("thread-table-owner", "<module>", None,
                                 "thread table does not use this module's concordance"))
            return found
        def value(cell):
            return book.page(cell.page).cell(cell.row, cell.column)
        for row in book.page(SOURCE_THREAD_ARGUMENT).rows():
            activation, position = row
            actual, formal, layout_cell = book.page(SOURCE_THREAD_ARGUMENT).latest(row)
            cell = book.latest_ref(SOURCE_THREAD_ARGUMENT, row)
            sources = {source for source, _stage in book.edges_into(cell)}
            layout = value(layout_cell)
            if (activation.page != SOURCE_THREAD_ACTIVATION or formal.page != NAME_BINDING
                    or layout_cell.page != SOURCE_AGGREGATE_LAYOUT
                    or layout.aggregate_kind != "tuple" or not 0 <= position < len(layout.members)
                    or layout.members[position].value_cell != actual
                    or not {activation, actual, formal, layout_cell} <= sources):
                found.append(Finding("thread-argument-provenance", str(activation), None,
                                     "worker argument disagrees with its exact tuple member or formal binding"))
        for body, (reference, nodes, captures) in table.bodies.items():
            sources = {source for source, _stage in book.edges_into(body)}
            if not {body.row[0], *nodes, *captures} <= sources:
                found.append(Finding("thread-body-provenance", str(reference), None,
                                     "worker body lacks its address, source nodes or captures"))
            if value(body.row[0]) != reference:
                found.append(Finding("thread-body-address", str(reference), None,
                                     "worker body disagrees with its function address"))
            for capture in captures:
                if capture.page != SOURCE_CLOSURE_CAPTURE or value(capture) is None:
                    found.append(Finding("thread-body-capture", str(reference), None,
                                         "worker body has no exact closure binding"))
        from .concordance_declarations import LOWERED_SOURCE_FUNCTION
        for cell, (reference, _captures, entry) in table.ssa_bodies.items():
            body, name = cell.row
            function = module.functions.get(name)
            materialized = book.page(LOWERED_SOURCE_FUNCTION).latest((name,))
            if (body.page != SOURCE_THREAD_BODY or value(body)[0] != reference
                    or function is None or materialized is None
                    or value(materialized[0]) != reference or materialized[2] != entry
                    or entry not in function.blocks):
                found.append(Finding("thread-body-ssa", str(name), None,
                                     "worker table does not identify its actual SSA function"))
        for cell, (source, handle, arguments, targets) in table.ssa_activations.items():
            name, site = cell.row
            creates = [i for b in module.functions[name].blocks.values() for i in b.instrs
                       if i.op == "ThreadCreate" and i.attributes.get("source_callsite_id") == site]
            source_fact = value(source)
            if (source.page != SOURCE_THREAD_ACTIVATION or source_fact is None or len(creates) != 1
                    or not targets or any(t.page != SSA_THREAD_BODY or t.row[0] != source_fact[0] for t in targets)
                    or creates[0].res.id != handle or tuple(a.id for a in creates[0].args) != arguments):
                found.append(Finding("thread-activation-ssa", str(name), int(site),
                                     "thread handle or context disagrees with its body table"))
        indexed_sites = {cell.row for cell in table.ssa_activations}
        for name, function in module.functions.items():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op == "ThreadCreate" and (name, instruction.attributes.get("source_callsite_id")) not in indexed_sites:
                        found.append(Finding("thread-activation-missing", name, instruction.res.id,
                                             "SSA ThreadCreate has no thread-table row"))
        for cell, (actual, formal, actual_members, formal_members, status) in table.contexts.items():
            activation, capture, caller, worker = cell.row
            _environment, parent_binding, input_binding = value(capture)
            from .concordance_declarations import NAME_BINDING
            if (actual != book.page(NAME_BINDING).cell(parent_binding.row, parent_binding.column).value_id
                    or formal != book.page(NAME_BINDING).cell(input_binding.row, input_binding.column).value_id
                    or status not in {"live", "unused"}):
                found.append(Finding("thread-context-binding", str(caller), int(actual),
                                     "thread context disagrees with its exact source capture"))
            has_formal = bool(
                {member for member, _roles in formal_members} & {a.id for a in module.functions[worker].args}
            )
            if (status == "live") != has_formal:
                found.append(Finding("thread-context-formal", str(worker), int(formal),
                                     "thread context liveness disagrees with the worker signature"))
        from .concordance_declarations import C_THREAD_ACTIVATION, SSA_THREAD_ACTIVATION
        for row in book.page(C_THREAD_ACTIVATION).rows():
            activation = row[0]
            worker, bindings, lifetime = book.page(C_THREAD_ACTIVATION).latest(row)
            source = book.page(SSA_THREAD_ACTIVATION).cell(activation.row, activation.column)
            if (source is None or worker not in {target.row[1] for target in source[3]}
                    or worker not in module.functions
                    or tuple(binding[0] for binding in bindings) != tuple(v.id for v in module.functions[worker].args)
                    or lifetime != "join_before_owner_return"):
                found.append(Finding("c-thread-policy", str(worker), None,
                                     "C thread entry or context disagrees with its SSA table"))
        from .concordance_declarations import C_THREAD_EXTENTS
        for row in book.page(C_THREAD_EXTENTS).rows():
            caller, worker, mode, lifetime = book.page(C_THREAD_EXTENTS).latest(row)
            policy = book.page(C_THREAD_ACTIVATION).latest(row)
            if (policy is None or policy[0] != worker or row[0].row[0] != caller
                    or mode != "borrow_invocation_extents" or lifetime != policy[2]):
                found.append(Finding("c-thread-extents", str(worker), None,
                                     "worker extent context lacks its exact joined activation"))
        return found

    @staticmethod
    def _thread_operation_findings(module: Any) -> list[Finding]:
        from .thread_operations import PROCESS_OPERATORS, PROCESS_RESULT_DTYPES
        from .concordance_declarations import SOURCE_PROCESS_OPERATION
        from .ssa_record_return_state import accepted_ssa_component
        from .frame_identity_book import _frame_owner_book
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        found = []
        for name, function in module.functions.items():
            accepted = accepted_ssa_component(function)
            instruction_book = _frame_owner_book(function)[0] if accepted else book
            for block in function.blocks.values():
                for instruction in block.instrs:
                    attributes = instruction.attributes or {}
                    operation = attributes.get("dispatch_operation")
                    semantic = PROCESS_OPERATORS.get(operation)
                    from .synchronization import SYNCHRONIZATION_SPECS
                    if (instruction.op in {spec.operator for spec in SYNCHRONIZATION_SPECS.values()}
                            and attributes.get('synchronization_receipt') is not None):
                        # Composed synchronization uses SSA_SYNCHRONIZATION,
                        # not the single-instruction SSA_PROCESS_CALL contract.
                        # Its exact source/projections/effects are checked by
                        # _synchronization_findings below.
                        continue
                    if semantic is None and instruction.op not in PROCESS_RESULT_DTYPES:
                        continue
                    if instruction.op != semantic:
                        found.append(Finding("ssa-thread-operation", name, None,
                                             "source thread operation did not retain its semantic SSA operator"))
                    if instruction.res is None or instruction.res.dtype != PROCESS_RESULT_DTYPES.get(semantic):
                        found.append(Finding("ssa-thread-result", name, None,
                                             "thread operation result disagrees with its declared handle/status contract"))
                    if not attributes.get("effectful") or not attributes.get("ordered_effect"):
                        found.append(Finding("ssa-thread-effect", name, None,
                                             "thread operation lost its ordered effect contract"))
                    if book is not None:
                        cell = attributes.get("source_process_cell")
                        fact = (None if cell is None or cell.page != SOURCE_PROCESS_OPERATION else
                                instruction_book.page(SOURCE_PROCESS_OPERATION).cell(cell.row, cell.column))
                        if fact is None or fact[:3] != (operation, semantic, PROCESS_RESULT_DTYPES.get(semantic)):
                            found.append(Finding("ssa-process-provenance", name, None,
                                                 "process operation lost its exact ingestion contract"))
        return found

    @staticmethod
    def _synchronization_findings(module: Any) -> list[Finding]:
        from .concordance_declarations import SOURCE_SYNCHRONIZATION, SSA_SYNCHRONIZATION, LOCK_BOX_ENTRY, SOURCE_MUTEX_BINDING, SOURCE_PROCESS_OPERATION
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found = []
        lock_box = getattr(module, "lock_box", None)
        if lock_box is not None and lock_box.book is not book:
            found.append(Finding("ssa-lock-box-owner", "<module>", None, "lock box belongs to a different concordance"))
        for row in book.page(SOURCE_MUTEX_BINDING).rows():
            entry = book.page(SOURCE_MUTEX_BINDING).latest(row)
            if entry.page != LOCK_BOX_ENTRY or book.page(LOCK_BOX_ENTRY).cell(entry.row, entry.column) is None:
                found.append(Finding("source-mutex-binding", str(row[0]), int(row[1]), "mutex binding has no lock-box entry"))
        operators = {"MemoryBarrier", "ControlBarrier", "LatchCreate", "LatchArrive",
                     "LatchWait", "LatchSignal", "LatchReset", "CompletionWait",
                     "MutexCreate", "MutexAcquire", "MutexAcquired", "MutexToken", "MutexRelease", "MutexLocked"}
        operators.update({"WorkerToken", "BarrierCreate", "BarrierWait", "BarrierAbort", "BarrierReset"})
        operators.update({"MutexSuspend", "MutexRestore", "WaiterRegister", "WaiterSelect",
                          "WaiterRetire", "LatchWaitTimeout", "LatchDestroy", "WaitQueueCreate", "WaitQueueMutex"})
        live_projections = {}
        for name, function in module.functions.items():
            literals = {instruction.res.id: instruction.attributes.get("value")
                        for block in function.blocks.values() for instruction in block.instrs
                        if instruction.op == "Const" and instruction.res is not None}
            for block in function.blocks.values():
                previous = None
                for instruction in block.instrs:
                    if instruction.op not in operators:
                        continue
                    attributes = instruction.attributes or {}
                    entry = attributes.get("lock_box_entry")
                    if instruction.op.startswith(("Mutex", "WaitQueue", "Waiter")):
                        if entry is None or entry.page != LOCK_BOX_ENTRY or book.page(LOCK_BOX_ENTRY).cell(entry.row, entry.column) is None:
                            found.append(Finding("ssa-lock-box-binding", name, None,
                                                 "mutex operation has no authoritative token-lock entry"))
                    contract = attributes.get("synchronization_contract")
                    receipt = attributes.get("synchronization_receipt")
                    if contract is None or receipt is None:
                        found.append(Finding("ssa-sync-provenance", name, None,
                                             "synchronization operation has no source contract/SSA receipt"))
                        continue
                    live_projections[receipt] = (name, instruction)
                    fact = book.page(SSA_SYNCHRONIZATION).cell(receipt.row, receipt.column)
                    source = book.page(SOURCE_SYNCHRONIZATION).cell(contract.row, contract.column)
                    if source[0].startswith('barrier_'):
                        from .concordance_declarations import SOURCE_THREAD_BARRIER_POLICY
                        policy = book.latest_ref(SOURCE_THREAD_BARRIER_POLICY, ('threading.Barrier.wait',))
                        causes = {cause for cause, _stage in book.edges_into(contract)}
                        constants = dict(source[3])
                        if (policy is None or policy not in causes or constants.get('execution_scope') != 257
                                or source[0] == 'barrier_wait' and (
                                    constants.get('memory_scope') != 257 or constants.get('memory_semantics') != 8)):
                            found.append(Finding('ssa-host-barrier-policy', name, instruction.res.id,
                                                 'native barrier source lost its declared host object/effect policy'))
                    if source[0].startswith("condition_"):
                        identities = {
                            "condition_create": {"threading.Condition"},
                            "condition_acquire": {"threading.Condition.acquire", "threading.Condition.__enter__"},
                            "condition_release": {"threading.Condition.release", "threading.Condition.__exit__"},
                            "condition_wait": {"threading.Condition.wait"},
                            "condition_notify": {"threading.Condition.notify"},
                            "condition_notify_all": {"threading.Condition.notify_all"},
                        }
                        process_causes = [book.page(SOURCE_PROCESS_OPERATION).cell(cell.row, cell.column)
                                          for cell, _stage in book.edges_into(contract)
                                          if cell.page == SOURCE_PROCESS_OPERATION]
                        if not any(cause[0] == source[0] and cause[3] in identities[source[0]]
                                   for cause in process_causes):
                            found.append(Finding("ssa-condition-source", name, instruction.res.id,
                                                 "Condition projection lacks its exact Python process identity"))
                    if source[0] == "condition_create":
                        from ..transmogrifier.ssa import SSALockBox
                        descriptor = (None if entry is None else book.page(LOCK_BOX_ENTRY).cell(
                            entry.row, entry.column))
                        constants = dict(source[3])
                        arguments = dict(zip(instruction.arg_roles, instruction.args))
                        owned = constants["owns_mutex"]
                        invalid = (descriptor is None or
                            (owned and (instruction.op != "MutexCreate" or entry.row != (source[5],)
                                or len(descriptor) != 5 or descriptor[0] is not True
                                or descriptor[3] != SSALockBox.CONDITION_STATE_COLUMNS
                                or descriptor[4] != SSALockBox.CONDITION_INITIAL_STATE
                                or descriptor[1][0] != constants.get("execution_scope"))))
                        if not owned:
                            supplied = source[4][source[1].index("lock")]
                            invalid = invalid or instruction.op != "WaitQueueCreate" or (
                                book.page(SOURCE_MUTEX_BINDING).latest((contract.row[0], int(supplied.row[1]))) != entry)
                        if invalid or any(role not in arguments or literals.get(arguments[role].id) != constants.get(role)
                                          for role in (("execution_scope", "reentrant") if owned else ("execution_scope",))):
                            found.append(Finding("ssa-condition-state", name, instruction.res.id,
                                                 "Condition constructor disagrees with its lock-box layout/initial state"))
                    expected = (contract, instruction.op, attributes.get("synchronization_phase"),
                                tuple(instruction.arg_roles), tuple(value.id for value in instruction.args),
                                None if instruction.res is None else instruction.res.id)
                    if fact != expected or (instruction.op, attributes.get("synchronization_phase")) not in {
                        (operator, phase) for operator, phase, _semantics in source[2]
                    }:
                        found.append(Finding("ssa-sync-contract", name, None,
                                             "synchronization instruction disagrees with its exact source projection"))
                    if previous is not None and (not instruction.arg_roles or instruction.arg_roles[-1] != "after"
                                                 or instruction.args[-1].id != previous):
                        found.append(Finding("ssa-sync-order", name, None,
                                             "synchronization chain lost its preceding activity"))
                    previous = None if instruction.res is None else instruction.res.id
                    if not attributes.get("effectful") or not attributes.get("ordered_effect"):
                        found.append(Finding("ssa-sync-effect", name, previous,
                                             "synchronization operation lost its effect boundary"))
        from .concordance_declarations import SSA_SYNCHRONIZATION_PROCEDURE, SSA_CONTROL_EDGE, C_SYNCHRONIZATION_POLICY
        for row in book.page(SSA_SYNCHRONIZATION_PROCEDURE).rows():
            contract, name, entry = row
            kind, result_id, exit_cell, receipts, edges = book.page(SSA_SYNCHRONIZATION_PROCEDURE).latest(row)
            source = book.page(SOURCE_SYNCHRONIZATION).cell(contract.row, contract.column)
            actual = {(live_projections[cell][1].op,
                       live_projections[cell][1].attributes.get("synchronization_phase"))
                      for cell in receipts if cell in live_projections}
            expected = {(op, phase) for op, phase, _semantics in source[2]}
            function = module.functions.get(name)
            from .ssa_record_return_state import function_scope_of
            from .concordance_declarations import SSA_VALUE
            result_cell = (None if function is None else book.latest_ref(
                SSA_VALUE, (function_scope_of(function), result_id)))
            result_fact = (None if result_cell is None else book.page(SSA_VALUE).cell(
                result_cell.row, result_cell.column))
            # The source outcome remains a typed value identity even when its
            # unused pure Phi/None materialization is reduced away. Procedure
            # effects and CFG must survive; a discarded return value need not
            # retain a physical register/definition in the reduced body.
            expected_dtype = "bool" if kind == "condition_wait" else "none"
            if (kind != source[0] or actual != expected or any(
                    cell not in live_projections or live_projections[cell][0] != name for cell in receipts)
                    or function is None or entry.row[2] not in function.blocks
                    or exit_cell.row[2] not in function.blocks
                    or result_fact is None or result_fact.dtype != expected_dtype):
                found.append(Finding("ssa-sync-procedure", name, result_id,
                                     "synchronization body lost its exact projections/result/control extent"))
            for edge in edges:
                if (edge.page != SSA_CONTROL_EDGE or edge.row[1] != name
                        or not book.page(SSA_CONTROL_EDGE).cell(edge.row, edge.column)
                        or function is None or edge.row[3].row[2] not in function.blocks[edge.row[2].row[2]].successors):
                    found.append(Finding("ssa-sync-procedure-edge", name, result_id,
                                         "synchronization procedure disagrees with an actual CFG edge"))
        from .c_synchronization import NATIVE_PRIMITIVES
        for row in book.page(C_SYNCHRONIZATION_POLICY).rows():
            policy = book.page(C_SYNCHRONIZATION_POLICY).latest(row)
            if policy[4] != "host_synchronization":
                continue
            selected = live_projections.get(row[0])
            if selected is None or NATIVE_PRIMITIVES.get(selected[1].op) != policy[0] or policy[2] not in {4, 256, 257}:
                found.append(Finding("c-sync-policy", "<module>", None,
                                     "native synchronization selection lost its exact SSA projection"))
                continue
            instruction = selected[1]
            contract = instruction.attributes["synchronization_contract"]
            source = book.page(SOURCE_SYNCHRONIZATION).cell(contract.row, contract.column)
            constants = dict(source[3])
            order = constants.get("memory_semantics", 0) if "memory_semantics" in instruction.arg_roles else 0
            for operator, phase, semantics in source[2]:
                if (operator, phase) == (instruction.op, instruction.attributes["synchronization_phase"]) and semantics is not None:
                    order = semantics
            if policy[3] != order or policy[2] != constants.get("memory_scope", constants.get("execution_scope", 257)):
                found.append(Finding("c-sync-order", selected[0], instruction.res.id,
                                     "native synchronization changed its explicit source scope/order"))
        return found

    @staticmethod
    def _source_thread_region_findings(module: Any) -> list[Finding]:
        """Check each retained graph view against its authoritative cells."""
        from .concordance_declarations import (
            SOURCE_THREAD_REGION_MEMBER, SOURCE_THREAD_REGION_EDGE,
            SOURCE_SYNC_OPERATOR, SOURCE_THREAD_REGION_SCHEDULE,
            SOURCE_THREAD_REGION_CONTROL, SOURCE_ACTIVITY_BINDING, SOURCE_ACTIVITY_NODE,
            SOURCE_ACTIVITY_EDGE, SOURCE_CONTROL_IR, SOURCE_THREAD_BARRIER_POLICY,
            SOURCE_EXTRACTION_IDENTITY, SOURCE_CLOSURE_CALLABLE,
        )
        metadata = getattr(module, "metadata", {}) or {}
        book = metadata.get("identity_book")
        if book is None:
            return []
        found = []
        for region in metadata.get("source_thread_regions", ()):
            scope = region.dependency_graph.G.graph["thread_region_scope"]
            expected_tokens = []
            for create, start, join, _target in region.activations:
                parent_key = next(key for key in region.member_ids if key[0] is None and key[2] == create)
                create_member = region.member_ids[parent_key]
                token = book.page(SOURCE_THREAD_REGION_MEMBER).latest((scope, create_member))[0]
                expected_tokens.append(token)
                for source in (create, start, join):
                    node = region.member_ids[(None, parent_key[1], source)]
                    for graph in (region.dependency_graph, region.control_graph):
                        if graph.G.nodes[node]["attributes"].get("dispatch_worker_token") != token:
                            found.append(Finding("source-thread-worker-token", str(scope), node,
                                                 "lifecycle operation disagrees with its exact worker token"))
            for half, graph in (("dependency", region.dependency_graph), ("control", region.control_graph)):
                if graph.G.graph.get("worker_tokens") != tuple(expected_tokens):
                    found.append(Finding("source-thread-worker-token", str(scope), None,
                                         f"{half} worker identities disagree with source activations"))
                declared = {(row[2], row[3]) for row in book.page(SOURCE_THREAD_REGION_EDGE).scope_rows(scope)
                            if row[1] == half}
                declared_nodes = {row[1] for row in book.page(SOURCE_THREAD_REGION_MEMBER).scope_rows(scope)}
                if declared_nodes != set(graph.G):
                    found.append(Finding('source-thread-region-members', str(scope), None,
                                         f'{half} membership disagrees with its concorded node set'))
                if declared != set(graph.G.edges):
                    found.append(Finding("source-thread-region-edges", str(scope), None,
                                         f"{half} topology disagrees with its concorded edges"))
                for node, data in graph.G.nodes(data=True):
                    member = book.latest_ref(SOURCE_THREAD_REGION_MEMBER, (scope, node))
                    if member is None:
                        found.append(Finding("source-thread-region-member", str(scope), node,
                                             "graph operation has no concorded source membership"))
                    elif (data.get("attributes") or {}).get("worker_token") != book.page(SOURCE_THREAD_REGION_MEMBER).cell(member.row, member.column)[1]:
                        found.append(Finding("source-thread-worker-token", str(scope), node,
                                             "graph operation disagrees with its concorded worker owner"))
                    projection = (data.get("attributes") or {}).get("control_projection")
                    if projection is not None:
                        program, binding, enter, leave = book.page(SOURCE_THREAD_REGION_CONTROL).cell(projection.row, projection.column)
                        points = book.page(SOURCE_ACTIVITY_BINDING).cell(binding.row, binding.column)
                        enter_source = book.page(SOURCE_THREAD_REGION_MEMBER).latest((scope, enter))[0]
                        leave_source = book.page(SOURCE_THREAD_REGION_MEMBER).latest((scope, leave))[0]
                        causes = {source for source, _stage in book.edges_into(projection)}
                        if (program.page != SOURCE_CONTROL_IR or tuple(points) != (enter_source, leave_source)
                                or not {program, binding, member} <= causes
                                or not region.control_graph.G.has_edge(leave, node)):
                            found.append(Finding('source-thread-control-projection', str(scope), node,
                                                 'resident control ports disagree with ingested ownership'))
                    if data.get('type') == 'ControlActivity':
                        source = (data.get('attributes') or {}).get('source_activity')
                        fact = None if source is None else book.page(SOURCE_ACTIVITY_NODE).cell(source.row, source.column)
                        if (source is None or member is None
                                or source != book.page(SOURCE_THREAD_REGION_MEMBER).cell(member.row, member.column)[0]
                                or fact[1] != data['attributes'].get('control_owners')
                                or fact[0] != data['attributes'].get('source_kind')):
                            found.append(Finding('source-thread-control-owner', str(scope), node,
                                                 'copied control activity lost its source region/arm'))
                    sync = (data.get("attributes") or {}).get("sync_operator")
                    # A copied standalone memory fence has no rendezvous
                    # projection. Only the paired authored Barrier.wait facts
                    # require one shared SOURCE_SYNC_OPERATOR identity.
                    paired = sync is not None or (data.get("attributes") or {}).get(
                        "extraction_identity") == "threading.Barrier.wait"
                    if data.get("type") in {"ControlBarrier", "MemoryBarrier"} and paired:
                        other = region.control_graph if half == "dependency" else region.dependency_graph
                        if sync is None or (other.G.nodes[node].get("attributes") or {}).get("sync_operator") != sync:
                            found.append(Finding("source-sync-projection", str(scope), node,
                                                 "control/memory projections do not share one operator identity"))
                        elif book.page(SOURCE_SYNC_OPERATOR).cell(sync.row, sync.column)[3][1] != book.page(SOURCE_SYNC_OPERATOR).cell(sync.row, sync.column)[2]:
                            found.append(Finding("source-sync-scope", str(scope), node,
                                                 "barrier scope disagrees with its exact source object"))
            # An excluded recurrence must come from an actual source repeat
            # edge, not a role assigned to conceal a scheduling cycle.
            for row in book.page(SOURCE_THREAD_REGION_EDGE).scope_rows(scope):
                roles, causes = book.page(SOURCE_THREAD_REGION_EDGE).latest(row)
                roles = (roles,) if isinstance(roles, str) else roles
                if any(role in {'repeat', 'control:repeat'} for role in roles):
                    pending, seen, repeats = list(causes), set(), []
                    while pending:
                        cause = pending.pop()
                        if cause in seen:
                            continue
                        seen.add(cause)
                        if cause.page == SOURCE_ACTIVITY_EDGE and cause.row[3] == 'repeat':
                            repeats.append(cause)
                        elif cause.page == SOURCE_THREAD_REGION_EDGE:
                            pending.extend(book.page(cause.page).cell(cause.row, cause.column)[1])
                    left = book.page(SOURCE_THREAD_REGION_MEMBER).latest((scope, row[2]))[0]
                    right = book.page(SOURCE_THREAD_REGION_MEMBER).latest((scope, row[3]))[0]
                    completion = book.page(SOURCE_THREAD_REGION_CONTROL).latest((scope, row[2]))
                    if completion is not None:
                        left = book.page(SOURCE_THREAD_REGION_MEMBER).latest((scope, completion[3]))[0]
                    if not any(repeat.row[1:3] == (left, right) for repeat in repeats):
                        found.append(Finding('source-thread-repeat', str(scope), row[2],
                                             'repeat endpoints disagree with their ingested loop backedge'))
            for row in book.page(SOURCE_SYNC_OPERATOR).scope_rows(scope):
                sync = book.latest_ref(SOURCE_SYNC_OPERATOR, row)
                fact = book.page(SOURCE_SYNC_OPERATOR).latest(row)
                if fact[6] == 'source_host_participant_policy':
                    policy = book.latest_ref(SOURCE_THREAD_BARRIER_POLICY, ('threading.Barrier.wait',))
                    causes = {source for source, _stage in book.edges_into(sync)}
                    callable_proofs = {book.page(source.page).cell(source.row, source.column)
                                       if source.page == SOURCE_EXTRACTION_IDENTITY
                                       else book.page(source.page).cell(source.row, source.column)[0]
                                       for source in causes if source.page in {SOURCE_EXTRACTION_IDENTITY, SOURCE_CLOSURE_CALLABLE}}
                    if (policy is None or policy not in causes
                            or not {'threading.Barrier', 'threading.Barrier.wait'} <= callable_proofs
                            or fact[3:6] != (('barrier_participants', fact[2]),
                                            ('barrier_participant_effects', fact[2]), 8)):
                        found.append(Finding('source-thread-barrier-policy', str(scope), row[1],
                                             'host barrier lost its exact participant scope or acquire/release policy'))
            for row in book.page(SOURCE_THREAD_REGION_SCHEDULE).scope_rows(scope):
                schedules = book.page(SOURCE_THREAD_REGION_SCHEDULE).latest(row)
                for half, levels in zip(('dependency', 'control'), schedules):
                    graph = region.scheduling_graph(half)
                    levels = dict(levels)
                    if (set(levels) != set(graph)
                            or any(levels[a] >= levels[b] for a, b in graph.edges if a in levels and b in levels)):
                        found.append(Finding("source-thread-region-schedule", str(scope), None,
                                             f"{row[1]} levels violate the retained graph constraints"))
        return found

    @staticmethod
    def _source_closure_findings(module: Any) -> list[Finding]:
        """A closure capture must join the exact parent and input bindings.

        These pages are ingestion facts even when the worker has no SSA
        body yet. Audit the book itself rather than reconstructing captures
        from function names or signature positions.
        """
        from .concordance_declarations import (
            FUNCTION_ADDRESS, NAME_BINDING, SOURCE_SPAN,
            SOURCE_CLOSURE_FREE_NAMES,
            SOURCE_CLOSURE_ENVIRONMENT, SOURCE_CLOSURE_CAPTURE,
            SOURCE_CLOSURE_CALLABLE, SOURCE_EXTRACTION_IDENTITY, INGESTION_VALUE,
            SOURCE_VALUE_CLASS,
        )
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found = []

        def value(ref):
            return book.page(ref.page).cell(ref.row, ref.column)

        def check(page, row, required, detail):
            cell = book.latest_ref(page, row)
            sources = {source for source, _ in book.edges_into(cell)}
            if not set(required) <= sources:
                found.append(Finding("source-closure-provenance", str(row[0]), None, detail))

        for row in book.page(SOURCE_EXTRACTION_IDENTITY).rows():
            identity = value(book.latest_ref(SOURCE_EXTRACTION_IDENTITY, row))
            check(SOURCE_EXTRACTION_IDENTITY, row, (row[0],),
                  'external callable identity lacks its ingested source node')
            if row[0].page != INGESTION_VALUE or not isinstance(identity, str) or not identity:
                found.append(Finding('source-extraction-identity', str(row), None,
                                     'external callable identity is not an ingested classification'))
        closure_inventories = {}
        for row in book.page(SOURCE_CLOSURE_FREE_NAMES).rows():
            cell = book.latest_ref(SOURCE_CLOSURE_FREE_NAMES, row)
            fact = book.page(SOURCE_CLOSURE_FREE_NAMES).cell(
                cell.row, cell.column,
            )
            if not (
                isinstance(fact, tuple) and len(fact) == 4
                and isinstance(fact[0], Ref) and fact[0].page == SOURCE_SPAN
                and isinstance(fact[1], Ref) and fact[1].page == FUNCTION_ADDRESS
                and isinstance(fact[2], tuple)
                and isinstance(fact[3], tuple)
                and len(set(fact[2])) == len(fact[2])
                and set(fact[3]) <= set(fact[2])
                and all(isinstance(name, str) and name for name in fact[2])
            ):
                found.append(Finding(
                    "source-closure-free-names", str(row[0]), None,
                    "closure free-name inventory is not an exact definition/address view",
                ))
                continue
            check(
                SOURCE_CLOSURE_FREE_NAMES, row, fact[:2],
                "closure free-name inventory lacks its definition or function address",
            )
            closure_inventories[(int(row[1]),)] = (cell, fact)

        closure_environments = {}
        for row in book.page(SOURCE_CLOSURE_ENVIRONMENT).rows():
            parent, definition, address = book.page(SOURCE_CLOSURE_ENVIRONMENT).latest(row)
            inventory = closure_inventories.get((int(row[1]),))
            check(SOURCE_CLOSURE_ENVIRONMENT, row, (parent, definition, address),
                  "closure environment lacks its parent binding, definition or function address")
            if parent.page != NAME_BINDING or value(parent) is None:
                found.append(Finding("source-closure-parent", str(row[0]), None,
                                     "closure environment has no exact parent binding"))
            elif inventory is not None and (
                definition != inventory[1][0]
                or address != inventory[1][1]
                or str(row[2]) not in inventory[1][3]
            ):
                found.append(Finding(
                    "source-closure-environment-inventory", str(row[0]), None,
                    "closure environment disagrees with its free-name inventory",
                ))
            elif inventory is not None:
                environment_cell = book.latest_ref(
                    SOURCE_CLOSURE_ENVIRONMENT, row,
                )
                sources = {
                    source for source, _stage in book.edges_into(environment_cell)
                }
                inherited_inventory = any(
                    source.page == SOURCE_CLOSURE_ENVIRONMENT
                    and inventory[0] in {
                        predecessor
                        for predecessor, _stage in book.edges_into(source)
                    }
                    for source in sources
                )
                if inventory[0] not in sources and not inherited_inventory:
                    found.append(Finding(
                        "source-closure-environment-inventory",
                        str(row[0]), None,
                        "closure environment does not derive from its free-name inventory",
                    ))
            closure_environments.setdefault(
                (int(row[1]), str(row[2])), set(),
            ).add(book.latest_ref(SOURCE_CLOSURE_ENVIRONMENT, row))
        captured_environments = set()
        for row in book.page(SOURCE_CLOSURE_CAPTURE).rows():
            environment, parent, input_binding = book.page(SOURCE_CLOSURE_CAPTURE).latest(row)
            capture_cell = book.latest_ref(SOURCE_CLOSURE_CAPTURE, row)
            captured_environments.add(environment)
            check(SOURCE_CLOSURE_CAPTURE, row, (environment, parent, input_binding),
                  "capture does not derive from both binding identities and their environment")
            if (value(environment)[0] != parent or input_binding.row != row
                    or input_binding.page != NAME_BINDING):
                found.append(Finding("source-closure-binding", str(row[0]), None,
                                     "capture disagrees with its environment or input binding"))
            environment_address = int(environment.row[1])
            inventory = closure_inventories.get((environment_address,))
            if inventory is not None:
                parent_value = value(parent).value_id
                parent_class = book.latest_ref(
                    SOURCE_VALUE_CLASS,
                    (inventory[0].row[0], int(parent_value)),
                )
                if parent_class is not None:
                    input_value = value(input_binding).value_id
                    transfers = tuple(
                        cell
                        for class_row in book.page(SOURCE_VALUE_CLASS).rows()
                        if int(class_row[1]) == int(input_value)
                        for cell in (
                            book.latest_ref(SOURCE_VALUE_CLASS, class_row),
                        )
                        if {capture_cell, parent_class} <= {
                            source for source, _stage in book.edges_into(cell)
                        }
                    )
                    if not transfers:
                        found.append(Finding(
                            "source-closure-class-transport", str(row[0]), None,
                            "captured input does not derive its known class from the exact parent binding",
                        ))
        for (address,), (_inventory_cell, fact) in closure_inventories.items():
            for name in fact[3]:
                environments = closure_environments.get((address, name), set())
                if not environments:
                    found.append(Finding(
                        "source-closure-environment-missing", str(address), None,
                        f"transported free name {name!r} has no environment row",
                    ))
                elif not environments & captured_environments:
                    found.append(Finding(
                        "source-closure-capture-missing", str(address), None,
                        f"transported free name {name!r} has no input capture row",
                    ))
        for row in book.page(SOURCE_CLOSURE_CALLABLE).rows():
            identity, environment, parent, receiver = book.page(SOURCE_CLOSURE_CALLABLE).latest(row)
            # Canonical rows additionally derive through their ingestion row.
            cell = book.latest_ref(SOURCE_CLOSURE_CALLABLE, row)
            sources = {source for source, _ in book.edges_into(cell)}
            direct = {environment, parent, receiver} <= sources
            relabeled = any(source.page == SOURCE_CLOSURE_CALLABLE for source in sources)
            if (identity != "threading.Barrier.wait" or value(environment)[0] != parent
                    or not (direct or relabeled)):
                found.append(Finding("source-closure-callable", str(row[0]), int(row[1]),
                                     "captured callable lacks exact environment provenance"))
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
                    # Program: ``o = maybe_inc(n, 1)`` beside an unrelated
                    # caller value numbered 6, both ``None | int`` (fuzzer
                    # p0_140).  Wrong record: the candidates were keyed by
                    # value id alone, so the caller's value 6 and the
                    # callee's value 6 were two candidates for one state.
                    # Rule: value ids are scope-local; among equal-arm
                    # candidates the state's own ABI is the one it derives
                    # from.
                    if len(candidates) > 1:
                        candidates = [
                            (abi, ref) for abi, ref in candidates
                            if ref.key in sources
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
        if book is not None:
            from .ssa_record_return_state import accepted_ssa_component

            owned = []
            for attribute, rows_attribute in (
                ("sequence_tables", "sequences"),
                ("record_tables", "records"),
                ("value_union_tables", "unions"),
            ):
                for name, table in (getattr(module, attribute, {}) or {}).items():
                    function = module.functions.get(name)
                    expected = book
                    if function is not None and accepted_ssa_component(function):
                        expected = function.metadata["accepted_ssa_component"]["identity_book"]
                    if getattr(table, rows_attribute, {}):
                        owned.append((str(name), attribute, table, expected))
            for attribute, rows_attribute in (
                ("struct_table", "_rows"), ("union_table", "_rows"),
                ("call_table", None),
            ):
                table = getattr(module, attribute, None)
                rows = table if rows_attribute is None else getattr(table, rows_attribute, {})
                if rows:
                    owned.append(("<module>", attribute, table, book))
            for name, attribute, table, expected in owned:
                if getattr(table, "book", None) is not expected:
                    found.append(Finding(
                        "concordance-table-owner-disagreement", name, None,
                        f"{attribute} owner {getattr(table, 'owner', None)!r} "
                        "is a view of a different identity book than its owning module/component",
                    ))
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
        found.extend(CorrelationTable._native_heap_storage_findings(module))
        return found

    @staticmethod
    def _native_heap_storage_findings(module: Any) -> list[Finding]:
        """Native cells and resident views must agree with their descriptor.

        The real forwarded StringCore child pool kept the descriptor but
        lost optional resident heap labels. Missing labels are not missing
        authority; contradictory owner/layout labels are a disagreement.
        """
        from .structural_concordance import storage_structure
        from .concordance_declarations import SEQUENCE_DESCRIPTOR
        from .ir_sequence_tables import _storage_values

        found = []
        for function_name, table in (getattr(module, "sequence_tables", {}) or {}).items():
            function = module.functions.get(function_name)
            if function is None:
                continue
            occurrences = {}
            for value in (*function.args, *(candidate
                    for block in function.blocks.values() for instruction in block.instrs
                    for candidate in (*instruction.args, instruction.res) if candidate is not None)):
                occurrences.setdefault(int(value.id), []).append(value)
            native_claims = {}
            for descriptor in table.sequences.values():
                if descriptor.heap_owner_value_id is None:
                    continue
                sequence_id = int(descriptor.sequence_id)
                def issue(detail, member=sequence_id):
                    found.append(Finding("native-heap-storage-disagreement", str(function_name),
                                         int(member), detail))
                try:
                    view, _cell = storage_structure("sequence", table.owner, sequence_id, book=table.book)
                except ConcordanceRefusal as error:
                    issue(str(error))
                    continue
                source = table.book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner, sequence_id))
                native = dict(view.children).get("native_storage")
                if (native is None or native.kind != "product" or native.source != source
                        or native.value_id != sequence_id or native.storage != "native_storage"):
                    issue("owned sequence lacks its resolved exact descriptor native-storage view")
                    continue
                expected = {int(value.id): value for value in _storage_values(descriptor)}
                members = dict(native.children)
                if set(members) != set(expected):
                    issue(f"native members {tuple(members)!r} differ from descriptor members {tuple(expected)!r}")
                for member_id, declared in expected.items():
                    member = members.get(member_id)
                    if member is None:
                        continue
                    labels = dict(declared.accounting or {})
                    if member.value_id != member_id or member.source != source:
                        issue("native member does not retain its exact descriptor identity", member_id)
                    if (member.dtype != str(declared.dtype) or member.physical_dtype != str(declared.dtype)
                            or tuple(member.shape) != tuple(declared.shape) or member.rank != len(declared.shape)):
                        issue("native member physical type/shape differs from its descriptor storage", member_id)
                    role = ("heap_owner" if labels.get("heap_owner") else
                            "heap_cell" if labels.get("heap_element_dtype") is not None else None)
                    if role is None:
                        continue
                    claim = (role, labels.get("heap_element_dtype"), labels.get("heap_owner_value_id"))
                    prior = native_claims.setdefault(member_id, claim)
                    if prior != claim:
                        issue(f"one native identity has conflicting descriptor roles/layouts/owners: {prior!r} versus {claim!r}", member_id)
                    if (member.kind != "reference" or member.storage != role
                            or member.dtype != "ptr" or member.physical_dtype != "ptr"):
                        issue(f"native member is not its declared {role} pointer cell", member_id)
                    if role == "heap_cell":
                        children = dict(member.children)
                        element, owner = children.get("element"), children.get("owner")
                        if (element is None or element.source != source or element.value_id != member_id
                                or element.kind != "scalar" or element.rank != 0 or element.shape != ()
                                or element.dtype != labels["heap_element_dtype"]
                                or element.physical_dtype != labels["heap_element_dtype"]):
                            issue("heap cell element layout differs from its exact descriptor", member_id)
                        if (owner is None or owner.source != source or owner.storage != "heap_owner"
                                or owner.value_id != labels["heap_owner_value_id"]
                                or owner.kind != "reference" or owner.shape != (1,) or owner.rank != 1
                                or owner.dtype != "ptr" or owner.physical_dtype != "ptr"
                                or owner.value_id not in members
                                or members[owner.value_id].storage != "heap_owner"):
                            issue("heap cell context owner differs from its exact descriptor", member_id)
                    for occurrence in occurrences.get(member_id, ()):
                        resident = dict(occurrence.accounting or {})
                        mismatch = str(occurrence.dtype) != "ptr"
                        mismatch |= bool(resident.get("heap_owner")) and role != "heap_owner"
                        if resident.get("heap_element_dtype") is not None:
                            mismatch |= role != "heap_cell" or resident["heap_element_dtype"] != labels.get("heap_element_dtype")
                        if resident.get("heap_owner_value_id") is not None:
                            mismatch |= role != "heap_cell" or resident["heap_owner_value_id"] != labels.get("heap_owner_value_id")
                        if mismatch:
                            issue("resident native cell type/owner/layout disagrees with its descriptor view", member_id)
                            break
        return found

    @staticmethod
    def _formal_actual_occurrence_findings(module: Any) -> list[Finding]:
        """A changed occurrence must retain its exact control transition."""
        from .concordance_declarations import FORMAL_ACTUAL_OCCURRENCE, LOOP_RESULT_USE_REBINDING, SSA_VALUE
        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = None if book is None else book.pages.get(FORMAL_ACTUAL_OCCURRENCE.name)
        if page is None:
            return []
        found = []
        for row in page.rows():
            if len(row) != 7:
                continue
            _callee, formal, caller, block, index, position, scope = row
            history = page.history(row)
            for (old_column, old), (column, current) in zip(history, history[1:]):
                if old == current:
                    continue
                cell = Ref(FORMAL_ACTUAL_OCCURRENCE, row, column)
                causes = tuple(parent for parent, _stage in book.edges_into(cell))
                previous = Ref(FORMAL_ACTUAL_OCCURRENCE, row, old_column)
                transition_key = (scope, block, index, position)
                transitions = tuple(parent for parent in causes
                    if parent.page == LOOP_RESULT_USE_REBINDING and parent.row == transition_key)
                valid_transition = any(
                    isinstance((fact := book.page(parent.page).cell(parent.row, parent.column)), tuple)
                    and len(fact) == 3 and fact[:2] == (old, current)
                    for parent in transitions
                )
                actual_sources = tuple(parent for parent in causes
                    if parent.page == SSA_VALUE and parent.row == (scope, current))
                if previous not in causes or not valid_transition or not actual_sources:
                    found.append(Finding(
                        "formal-actual-transition-disagreement", str(caller), int(formal),
                        f"occurrence {(block, index, position)!r} changed {old!r}->{current!r} "
                        "without its prior occurrence, exact loop-use transition and actual SSA causes",
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
    def _ssa_abi_physical_findings(module: Any) -> list[Finding]:
        """Check a posted scalar physical field against its exact ABI cause."""
        from .concordance_declarations import (
            SSA_VALUE, SSAValueFact, VALUE_TYPE_ABI, ValueTypeABIFact,
        )

        book = (getattr(module, 'metadata', {}) or {}).get('identity_book')
        if book is None:
            return []
        found = []
        for row in book.page(SSA_VALUE).rows():
            cell = book.latest_ref(SSA_VALUE, row)
            fact = book.page(SSA_VALUE).cell(cell.row, cell.column)
            if (not isinstance(fact, SSAValueFact)
                    or fact.physical_dtype is None
                    or fact.storage != 'scalar' or fact.rank != 0):
                continue
            abi_sources = tuple(source for source, _stage in book.edges_into(cell)
                                if source.page == VALUE_TYPE_ABI)
            for abi_cell in abi_sources:
                abi = book.page(VALUE_TYPE_ABI).cell(
                    abi_cell.row, abi_cell.column)
                arms = (tuple(abi.physical_alternatives)
                        if isinstance(abi, ValueTypeABIFact) else ())
                views = {(arm.storage, arm.dtype, tuple(arm.shape), arm.rank)
                         for arm in arms}
                if (len(views) != 1 or next(iter(views)) !=
                        (fact.storage, fact.physical_dtype,
                         tuple(fact.shape), fact.rank)):
                    found.append(Finding(
                        'ssa-abi-physical-disagreement', str(row[0]), row[1],
                        'posted physical SSA fields disagree with their exact type ABI arm',
                    ))
        return found

    @staticmethod
    def _constant_materialization_findings(module: Any) -> list[Finding]:
        """Keep a materialized integer's exact native domain on its identity.

        ``ssa_constant_materialization`` is the incumbent record of the
        literal value and selected dtype.  Its consumers may not reclassify
        that identity from a generic Python ``int`` spelling or leave the
        corresponding SSA value at a different width.
        """

        from .concordance_declarations import (
            SSA_CONSTANT_MATERIALIZATION, SSA_VALUE,
        )
        from ..transmogrifier.dtype_layout import python_int_literal_dtype

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        if book is None:
            return []
        pages = getattr(book, "pages", {}) or {}
        materializations = pages.get(SSA_CONSTANT_MATERIALIZATION.name)
        values = pages.get(SSA_VALUE.name)
        if materializations is None or values is None:
            return []
        found: list[Finding] = []
        for row in materializations.rows():
            if not (isinstance(row, tuple) and len(row) == 4):
                continue
            scope, function_name, _block, value_id = row
            latest = materializations.latest(row)
            if not (isinstance(latest, tuple) and len(latest) == 3):
                continue
            operation, dtype, literal_repr = latest
            if str(operation).casefold() not in {"const", "constant"}:
                continue
            try:
                literal = ast.literal_eval(str(literal_repr))
            except (ValueError, SyntaxError, TypeError):
                continue
            if type(literal) is not int:
                continue
            try:
                expected_dtype = python_int_literal_dtype(literal)
            except (TypeError, OverflowError):
                expected_dtype = None
            discrepancies = []
            if expected_dtype is None or str(dtype) != expected_dtype:
                discrepancies.append(
                    f"materialization dtype={dtype!r}, exact integer domain="
                    f"{expected_dtype!r}"
                )
            value_row = (scope, int(value_id))
            value_fact = values.latest(value_row)
            if value_fact is None or str(
                getattr(value_fact, "dtype", None)
            ) != str(dtype):
                discrepancies.append(
                    f"ssa_value={value_fact!r}, materialization dtype={dtype!r}"
                )
            materialization_cell = book.latest_ref(
                SSA_CONSTANT_MATERIALIZATION, row)
            value_cell = book.latest_ref(SSA_VALUE, value_row)
            value_sources = (
                {source for source, _stage in book.edges_into(value_cell)}
                if value_cell is not None else set()
            )
            # CONTROL_CONST physical publication is a revision caused by
            # this exact literal receipt and its prior minted SSA identity.
            if materialization_cell in value_sources:
                materialization_sources = {
                    source for source, _stage in
                    book.edges_into(materialization_cell)
                }
                if (getattr(value_fact, "physical_dtype", None) != str(dtype)
                        or getattr(value_fact, "storage", None) != "scalar"
                        or getattr(value_fact, "rank", None) != 0
                        or not any(source.page == SSA_VALUE
                                   and source.row == value_row
                                   for source in materialization_sources)):
                    discrepancies.append(
                        "physical scalar publication lost its exact literal "
                        "materialization or prior SSA identity"
                    )
            function = (getattr(module, "functions", {}) or {}).get(
                str(function_name)
            )
            if function is not None:
                occurrences = tuple(
                    instruction.res
                    for block in function.blocks.values()
                    for instruction in block.instrs
                    if instruction.res is not None
                    and int(instruction.res.id) == int(value_id)
                    and str(instruction.op).casefold() in {"const", "constant"}
                    and (instruction.attributes or {}).get("value") == literal
                )
                if any(str(value.dtype) != str(dtype) for value in occurrences):
                    discrepancies.append(
                        "resident Const occurrence dtype differs from the "
                        f"materialization: {tuple(value.dtype for value in occurrences)!r}"
                    )
            if discrepancies:
                found.append(Finding(
                    "constant-materialization-domain-disagreement",
                    str(function_name), int(value_id), "; ".join(discrepancies),
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
    def _frame_aliased_result_findings(module: Any) -> list[Finding]:
        """A frame-result index is a view of one exact call's storage edge."""
        from .ssa_record_return_state import frame_aliased_result_binding
        functions = getattr(module, "functions", {}) or {}
        table = getattr(module, "call_table", None)
        book = getattr(table, "book", None) or identity_book(module)
        found = []
        for caller_name, caller in functions.items():
            records = () if table is None else table.get(str(caller_name), ())
            for block_name, block in caller.blocks.items():
                for instruction in block.instrs:
                    if instruction.op not in {"Call", "call"} or not instruction.attributes.get("result_aliases_frame"):
                        continue
                    callee_name = str(instruction.attributes.get("callee") or "")
                    callee = functions.get(callee_name)
                    matching = tuple(record for record in records
                                     if record.callsite_id == instruction.attributes.get("plan_callsite_id")
                                     and str(record.callee_symbol) == callee_name)
                    detail = None
                    try:
                        if callee is None or len(matching) != 1:
                            raise ConcordanceRefusal("frame result lacks its unique callee/call record")
                        formal, position, actual, _cell = frame_aliased_result_binding(
                            instruction, callee, book=book, record=matching[0],
                        )
                        if (instruction.attributes.get("ssa_output_formal_id") != formal
                                or instruction.attributes.get("ssa_output_argument") != position):
                            detail = (f"record binds callee formal {formal} to caller {actual.id} "
                                      f"at position {position}; emitted formal/index are "
                                      f"{instruction.attributes.get('ssa_output_formal_id')!r}/"
                                      f"{instruction.attributes.get('ssa_output_argument')!r}")
                        else:
                            # The emitted projection cites the same activation,
                            # not another call's equal-looking storage ids.
                            proof = instruction.attributes.get("ssa_output_call_record")
                            if not isinstance(proof, Ref) or proof.row[1] != str(caller_name):
                                raise ConcordanceRefusal("frame result proof belongs to another caller scope")
                            frame_aliased_result_binding(instruction, callee, book=book)
                    except ConcordanceRefusal as error:
                        detail = str(error)
                    if detail is not None:
                        found.append(Finding(
                            "frame-aliased-result-disagreement", str(caller_name),
                            None if instruction.res is None else int(instruction.res.id),
                            f"{block_name}: {detail}",
                        ))
        return found

    @staticmethod
    def _physical_call_input_conversion_findings(module: Any) -> list[Finding]:
        """A converted native argument retains its exact activation and cause."""
        from .concordance_declarations import CALL_RECORD, CALL_INPUT_CONVERSION, CELL_SET, SSA_VALUE
        from .ssa_record_return_state import ssa_value_identity_cell
        table = getattr(module, "call_table", None)
        if table is None:
            return []
        book = table.book
        functions = getattr(module, "functions", {}) or {}
        conversion_index = defaultdict(list)
        page = book.pages.get(CALL_INPUT_CONVERSION.name)
        if page is not None:
            for row in page.rows():
                for column, fact in page.history(row):
                    if len(fact) == 6:
                        conversion_index[(str(row[0]), str(row[2]), int(fact[4]))].append(
                            (Ref(CALL_INPUT_CONVERSION, row, column), fact))
        found = []
        for caller_name, records in table.items():
            caller = functions.get(str(caller_name))
            if caller is None:
                continue
            definitions = {int(ins.res.id): (str(label), index, ins)
                           for label, block in caller.blocks.items()
                           for index, ins in enumerate(block.instrs) if ins.res is not None}
            dominance = CorrelationTable._dominators(caller)
            for record in records:
                # A cloned table cites its original exact row. Follow only
                # those record revisions/cell sets, never a latest conversion
                # selected from another call sharing the same source/formal.
                pending, causes = [record.cell], set()
                while pending:
                    cell = pending.pop()
                    if cell is None or cell in causes:
                        continue
                    causes.add(cell)
                    for parent, _stage in book.edges_into(cell):
                        if parent.page in {CALL_RECORD, CELL_SET}:
                            pending.append(parent)
                        else:
                            causes.add(parent)
                conversions = conversion_index.get((str(caller_name), str(record.callee_symbol), int(record.callsite_id)), ())
                if not conversions:
                    continue
                sites = tuple((str(label), index, ins)
                              for label, block in caller.blocks.items()
                              for index, ins in enumerate(block.instrs)
                              if ins.op in {"Call", "call"}
                              and ins.attributes.get("plan_callsite_id") == record.callsite_id
                              and ins.attributes.get("callee") == record.callee_symbol)
                if not sites:
                    continue  # Existing dead-call accounting owns removed calls.
                callee = functions.get(str(record.callee_symbol))
                if len(sites) != 1 or callee is None:
                    found.append(Finding("physical-call-conversion-frame", str(caller_name), None,
                                         "converted frame has no unique current native call"))
                    continue
                block, call_index, call = sites[0]
                inputs = tuple(map(int, call.attributes.get("callee_input_ids", ())))
                frame = {int(formal): (kind, source) for formal, kind, source in record.frame_bindings}
                formals = {int(value.id): value for value in callee.args}
                for formal_id in {int(cell.row[3]) for cell, _fact in conversions}:
                    formal = formals.get(formal_id)
                    if formal is None:
                        continue  # Existing formal-pruning audit owns retirement.
                    target = (formal.accounting or {}).get("physical_dtype") or formal.dtype
                    candidates = tuple((cell, fact) for cell, fact in conversions
                                       if int(cell.row[3]) == formal_id and str(fact[2]) == str(target))
                    if not candidates:
                        continue
                    binding = frame.get(formal_id)
                    matching = tuple((cell, fact) for cell, fact in candidates
                                     if binding is not None and binding[0] == "caller_value"
                                     and int(binding[1]) == int(fact[0]))
                    detail = None
                    if len(matching) != 1 or inputs.count(formal_id) != 1:
                        detail = "physical frame discarded its exact recorded input conversion"
                    else:
                        cell, fact = matching[0]
                        converted = int(fact[0])
                        position = inputs.index(formal_id)
                        producer = definitions.get(converted)
                        value_cell = ssa_value_identity_cell(caller, converted, book=book)
                        sourced = value_cell is not None and any(
                            ref.page == SSA_VALUE and ref.row == value_cell.row for ref in causes)
                        valid = (cell in causes and position < len(call.args) and int(call.args[position].id) == converted
                                 and producer is not None and producer[2].op == "Cast"
                                 and len(producer[2].args) == 1
                                 and int(producer[2].args[0].id) == int(cell.row[1])
                                 and producer[0] == str(fact[5]) and sourced
                                 and producer[0] in dominance.get(block, set())
                                 and (producer[0] != block or producer[1] < call_index))
                        if not valid:
                            detail = "native argument disagrees with its sourced, control-owned Cast"
                    if detail is not None:
                        found.append(Finding("physical-call-conversion-frame", str(caller_name),
                                             None if call.res is None else int(call.res.id),
                                             f"callsite {record.callsite_id}, formal {formal_id}: {detail}"))
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
            retained_outers = _loop_retained_outer_reads(
                function, int(declaration["loop_node_id"]),
            )
            for rebind in declaration["rebinds"]:
                outer = rebind["outer"]
                carried = rebind["carried"]
                inner = rebind["inner"]

                # Rule 1 -- inside the scope the outer generation is not
                # in scope.  A use of it reads the value as it was before
                # the first iteration, on every iteration.  The exception is
                # an outer value a non-carried binding reads in the loop
                # (``c = b; while ...: c = c + b``): there it IS in scope.
                for block_name in sorted(() if int(outer) in retained_outers else inside):
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
                            # Program: ``h, l = step(h, l, i)`` with
                            # ``step`` returning ``low ^ byte, high``.  The
                            # updated ``l`` is the callee's bare formal, so
                            # the passthrough fold rebinds the call output
                            # to the actual -- the carried Phi of ``h`` --
                            # as a projection occurrence of that same id.
                            # Wrong record: that occurrence is a distinct
                            # object, so it read as a private resident.
                            # Rule: an occurrence the aggregate passthrough
                            # rebinding minted for a resident the function
                            # defines is that resident, not a private one.
                            if (
                                resident_objects.get(inner)
                                and (argument.accounting or {}).get(
                                    "ssa_aggregate_passthrough"
                                ) is not None
                                and (argument.accounting or {}).get(
                                    "ssa_storage_alias"
                                ) == inner
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
    def _annotation_class_admission_findings(module: Any) -> list[Finding]:
        """Imported annotation classes retain callable, declaration and field causes."""
        from .concordance_declarations import (
            PARAMETER_ANNOTATION, FUNCTION_RETURN_ANNOTATION, CLASS_DECLARATION,
            CLASS_FIELD_DECLARATION, SOURCE_SPAN, STATIC_SYMBOL_DEFINITION,
        )
        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        fields = book.pages.get(CLASS_FIELD_DECLARATION.name)
        found = []
        for authority in (PARAMETER_ANNOTATION, FUNCTION_RETURN_ANNOTATION):
            page = book.pages.get(authority.name)
            for row in (() if page is None else page.rows()):
                cell = book.latest_ref(authority, row)
                causes = tuple(source for source, _stage in book.edges_into(cell))
                callables = tuple(source for source in causes if source.page == STATIC_SYMBOL_DEFINITION)
                if not callables:
                    continue
                declarations = tuple(source for source in causes if source.page == CLASS_DECLARATION)
                spans = tuple(source for source in causes if source.page == SOURCE_SPAN and source.row[:2] == row[:2])
                problems = []
                if not declarations or not spans or any(source.row[:2] != row[:2] for source in callables):
                    problems.append("annotation lacks its exact callable, source occurrence or class declaration")
                for declaration in declarations:
                    parents = {source for source, _stage in book.edges_into(declaration)}
                    if not set(callables).issubset(parents) or not parents.intersection(spans):
                        problems.append("class admission lacks the annotation's callable and occurrence causes")
                    for field_row in (() if fields is None else fields.scope_rows(declaration.row[0])):
                        if field_row[:2] != declaration.row or fields.latest(field_row) is None:
                            continue
                        field_cell = book.latest_ref(CLASS_FIELD_DECLARATION, field_row)
                        if declaration not in {source for source, _stage in book.edges_into(field_cell)}:
                            problems.append(f"field {field_row[-1]!r} lacks its exact admitted class cause")
                if problems:
                    found.append(Finding("annotation-class-admission-disagreement", str(row[1]), None,
                                         "; ".join(dict.fromkeys(problems))))
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
    def _source_parameter_default_findings(module: Any) -> list[Finding]:
        """Check each default against its three authored source constructs."""

        from .concordance_declarations import (
            SOURCE_PARAMETER_DEFAULT, SOURCE_SPAN, SpecializationFact,
            SpecializationSource,
        )

        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        page = book.page(SOURCE_PARAMETER_DEFAULT)
        spans = book.page(SOURCE_SPAN)
        for row in page.rows():
            receipt = book.latest_ref(SOURCE_PARAMETER_DEFAULT, row)
            fact = page.latest(row)
            causes = tuple(source for source, _stage in book.edges_into(receipt))
            source_spans = tuple(source for source in causes
                                 if source.page == SOURCE_SPAN)
            valid = (
                isinstance(row, tuple) and len(row) == 2
                and isinstance(row[1], str)
                and isinstance(fact, SpecializationFact)
                and fact.source is SpecializationSource.DEFAULT
                and len(source_spans) == 3
                and len(set(source_spans)) == 3
            )
            if valid:
                by_kind = tuple(
                    (source, spans.cell(source.row, source.column).kind)
                    for source in source_spans
                )
                definitions = tuple(source for source, kind in by_kind
                                    if kind in {"FunctionDef", "AsyncFunctionDef", "Lambda"})
                arguments = tuple(source for source, kind in by_kind
                                  if kind == "arg")
                defaults = tuple(source for source in source_spans
                                 if source not in definitions and source not in arguments)
                valid = (len(definitions) == len(arguments) == len(defaults) == 1)
            if valid:
                definition, argument, default = (
                    definitions[0], arguments[0], defaults[0]
                )
                definition_kind = spans.cell(
                    definition.row, definition.column,
                ).kind
                expected_scope = (
                    definition.row if definition_kind == "Lambda"
                    else definition.row[:2]
                )
                definition_path = tuple(definition.row[2])
                valid = (
                    row[0] == expected_scope
                    and argument.row[:2] == definition.row[:2]
                    and default.row[:2] == definition.row[:2]
                    and tuple(argument.row[2])[:len(definition_path)]
                        == definition_path
                    and tuple(default.row[2])[:len(definition_path)]
                        == definition_path
                    and argument.row[2] != default.row[2]
                    and spans.cell(default.row, default.column).kind
                        not in {"arg", "FunctionDef", "AsyncFunctionDef", "Lambda"}
                )
            if not valid:
                found.append(Finding(
                    "source-parameter-default", str(row[0]), None,
                    f"parameter {row[1]!r} lacks exact function, argument, "
                    "and default-expression source causes",
                ))
        return found

    @staticmethod
    def _source_callable_signature_findings(module: Any) -> list[Finding]:
        """Check the public symbol, wrapper chain and authored formal causes."""

        from .concordance_declarations import (
            SOURCE_CALLABLE_SIGNATURE, SOURCE_PARAMETER_DEFAULT, SOURCE_SPAN,
            STATIC_SYMBOL_DEFINITION, SpecializationFact,
            SpecializationSource,
        )

        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        page = book.page(SOURCE_CALLABLE_SIGNATURE)
        for row in page.rows():
            receipt = book.latest_ref(SOURCE_CALLABLE_SIGNATURE, row)
            fact = page.latest(row)
            sources = {source for source, _stage in book.edges_into(receipt)}
            valid = (
                isinstance(row, tuple) and len(row) == 1
                and isinstance(row[0], Ref)
                and row[0].page == STATIC_SYMBOL_DEFINITION
                and isinstance(fact, tuple) and len(fact) == 3
                and row[0] in sources
            )
            if valid:
                definition, chain, formals = fact
                valid = (
                    isinstance(definition, Ref)
                    and definition.page == SOURCE_SPAN
                    and definition in sources
                    and book.page(SOURCE_SPAN).cell(
                        definition.row, definition.column,
                    ).kind in {"FunctionDef", "AsyncFunctionDef"}
                    and isinstance(chain, tuple) and bool(chain)
                    and all(isinstance(link, tuple) and len(link) == 3
                            and all(isinstance(item, str) and item
                                    for item in link)
                            for link in chain)
                    and isinstance(formals, tuple)
                )
            if valid:
                symbol_fact = book.page(STATIC_SYMBOL_DEFINITION).cell(
                    row[0].row, row[0].column,
                )
                valid = (
                    chain[0][:2] == row[0].row[:2]
                    and chain[0][2] == symbol_fact.digest
                    and chain[-1][:2] == definition.row[:2]
                )
            if valid:
                names = set()
                positional = 0
                default_scope = definition.row[:2]
                for formal in formals:
                    if not (isinstance(formal, tuple) and len(formal) == 4):
                        valid = False
                        break
                    name, kind, index, default = formal
                    if not isinstance(name, str) or not name or name in names:
                        valid = False
                        break
                    names.add(name)
                    if kind in {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"}:
                        if index != positional:
                            valid = False
                            break
                        positional += 1
                    elif kind in {"KEYWORD_ONLY", "VAR_POSITIONAL", "VAR_KEYWORD"}:
                        if index is not None:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                    if default is None:
                        continue
                    if not (
                        isinstance(default, Ref)
                        and default.page == SOURCE_PARAMETER_DEFAULT
                        and default.row == (default_scope, name)
                        and default in sources
                        and isinstance(book.page(SOURCE_PARAMETER_DEFAULT).cell(
                            default.row, default.column,
                        ), SpecializationFact)
                        and book.page(SOURCE_PARAMETER_DEFAULT).cell(
                            default.row, default.column,
                        ).source is SpecializationSource.DEFAULT
                    ):
                        valid = False
                        break
                argument_spans = tuple(
                    source for source in sources
                    if source.page == SOURCE_SPAN and source != definition
                    and source.row[:2] == definition.row[:2]
                    and book.page(SOURCE_SPAN).cell(
                        source.row, source.column,
                    ).kind == "arg"
                )
                valid = valid and len(argument_spans) == len(formals)
            if not valid:
                found.append(Finding(
                    "source-callable-signature", str(row), None,
                    "intrinsic signature lacks exact public symbol, authored "
                    "definition, wrapper digest, or formal/default causes",
                ))
        return found

    @staticmethod
    def _source_named_tuple_layout_findings(module: Any) -> list[Finding]:
        """Check a referenced NamedTuple's ordered authored field cells."""

        from .concordance_declarations import (
            CLASS_DECLARATION, CLASS_FIELD_DECLARATION,
            SOURCE_NAMED_TUPLE_LAYOUT, SOURCE_SPAN,
        )

        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        page = book.page(SOURCE_NAMED_TUPLE_LAYOUT)
        for row in page.rows():
            cell = book.latest_ref(SOURCE_NAMED_TUPLE_LAYOUT, row)
            fact = page.latest(row)
            valid = (
                isinstance(row, tuple) and len(row) == 1
                and isinstance(row[0], Ref)
                and row[0].page == CLASS_DECLARATION
                and isinstance(fact, tuple) and len(fact) == 2
                and isinstance(fact[0], Ref)
                and fact[0].page == SOURCE_SPAN
                and isinstance(fact[1], tuple) and bool(fact[1])
                and cell is not None
            )
            if valid:
                class_cell = row[0]
                base, members = fact
                direct = {source for source, _ in book.edges_into(cell)}
                class_scope = class_cell.row
                valid = (
                    class_cell in direct and base in direct
                    and base.row[:2] == class_scope
                    and len(base.row[2]) == 1
                    and base.row[2][0][0] == "bases"
                    and book.page(SOURCE_SPAN).cell(
                        base.row, base.column,
                    ).kind in {"Name", "Attribute"}
                )
            if valid:
                positions = []
                names = set()
                for member in members:
                    if not (
                        isinstance(member, tuple) and len(member) == 3
                        and isinstance(member[0], str)
                        and isinstance(member[1], Ref)
                        and member[1].page == CLASS_FIELD_DECLARATION
                        and isinstance(member[2], Ref)
                        and member[2].page == SOURCE_SPAN
                    ):
                        valid = False
                        break
                    name, field, statement = member
                    path = statement.row[2]
                    if not (
                        name not in names
                        and field.row == (*class_scope, name)
                        and field in direct and statement in direct
                        and statement in {
                            source for source, _ in book.edges_into(field)
                        }
                        and statement.row[:2] == class_scope
                        and len(path) == 1 and path[0][0] == "body"
                        and isinstance(path[0][1], int)
                        and book.page(SOURCE_SPAN).cell(
                            statement.row, statement.column,
                        ).kind == "AnnAssign"
                    ):
                        valid = False
                        break
                    names.add(name)
                    positions.append(path[0][1])
                valid = valid and positions == sorted(positions)
            if not valid:
                found.append(Finding(
                    "source-named-tuple-layout", str(row), None,
                    "named tuple layout lacks ordered authored class, base, "
                    "field or statement causes",
                ))
        return found

    @staticmethod
    def _named_intrinsic_result_projection_findings(module: Any) -> list[Finding]:
        """Check named call projections against the admitted class slot."""

        from .concordance_declarations import (
            CALL_RESULT_PROJECTION, CANONICAL_VALUE,
            CLASS_DECLARATION, FUNCTION_RETURN_ANNOTATION,
            PROCESS_GRAPH_OPERANDS, SOURCE_CALLABLE_SIGNATURE,
            SOURCE_INTRINSIC_ADMISSION, SOURCE_NAMED_TUPLE_LAYOUT,
        )
        from .concorded_literals import origin_cell

        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        page = book.page(CALL_RESULT_PROJECTION)
        for row in page.rows():
            cell = book.latest_ref(CALL_RESULT_PROJECTION, row)
            if cell is None:
                continue
            direct = {source for source, _ in book.edges_into(cell)}
            layouts = tuple(source for source in direct
                            if source.page == SOURCE_NAMED_TUPLE_LAYOUT)
            if not layouts:
                continue
            path = page.latest(row)
            valid = (
                len(layouts) == 1 and isinstance(path, tuple)
                and len(path) == 1 and isinstance(path[0], int)
                and not isinstance(path[0], bool)
            )
            if valid:
                layout = book.page(SOURCE_NAMED_TUPLE_LAYOUT).cell(
                    layouts[0].row, layouts[0].column,
                )
                members = layout[1] if isinstance(layout, tuple) and len(layout) == 2 else ()
                valid = 0 <= path[0] < len(members)
            if valid:
                selected_field = members[path[0]][1]
                operands = tuple(source for source in direct
                                 if source.page == PROCESS_GRAPH_OPERANDS)
                admissions = tuple(source for source in direct
                                   if source.page == SOURCE_INTRINSIC_ADMISSION)
                annotations = tuple(source for source in direct
                                    if source.page == FUNCTION_RETURN_ANNOTATION)
                signatures = tuple(source for source in direct
                                   if source.page == SOURCE_CALLABLE_SIGNATURE)
                canonical = tuple(source for source in direct
                                  if source.page == CANONICAL_VALUE)
                valid = (
                    selected_field in direct
                    and len(operands) == 1
                    and len(admissions) == 1
                    and len(annotations) == 1
                    and len(signatures) == 1
                    and len(canonical) >= 2
                )
            if valid:
                receiver_edges = book.page(PROCESS_GRAPH_OPERANDS).cell(
                    operands[0].row, operands[0].column,
                )
                operand_causes = {
                    source for source, _ in book.edges_into(operands[0])
                }
                receiver = receiver_edges[0][0] if (
                    isinstance(receiver_edges, tuple)
                    and len(receiver_edges) == 1
                    and isinstance(receiver_edges[0], tuple)
                    and len(receiver_edges[0]) == 2
                    and receiver_edges[0][1] == "value"
                ) else None
                projection = tuple(
                    source for source in operand_causes
                    if source.page == CANONICAL_VALUE
                    and source.row == operands[0].row
                )
                admission_fact = book.page(SOURCE_INTRINSIC_ADMISSION).cell(
                    admissions[0].row, admissions[0].column,
                )
                signature_fact = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
                    signatures[0].row, signatures[0].column,
                )
                annotation_causes = {
                    source for source, _ in book.edges_into(annotations[0])
                }
                try:
                    exact_origin = None if receiver is None else origin_cell(
                        book, receiver,
                    )
                except ValueError:
                    exact_origin = None
                valid = (
                    receiver in canonical
                    and receiver in operand_causes
                    and len(projection) == 1
                    and projection[0] in direct
                    and receiver.row[:-1] == projection[0].row[:-1]
                    and projection[0].row == operands[0].row
                    and row[1:] == (
                        receiver.row[-1], projection[0].row[-1],
                    )
                    and isinstance(admission_fact, tuple)
                    and len(admission_fact) == 3
                    and admission_fact[2] == signatures[0]
                    and admissions[0].row == (exact_origin,)
                    and isinstance(signature_fact, tuple)
                    and bool(signature_fact)
                    and isinstance(signature_fact[0], Ref)
                    and signature_fact[0].row[:2] == annotations[0].row
                    and layouts[0].row[0].page == CLASS_DECLARATION
                    and layouts[0].row[0] in annotation_causes
                )
            if not valid:
                found.append(Finding(
                    "named-intrinsic-result-projection", str(row), None,
                    "named projection lacks the declared slot, exact call "
                    "receiver, projection operand or admission causes",
                ))
        return found

    @staticmethod
    def _source_intrinsic_admission_findings(module: Any) -> list[Finding]:
        """Check accepted policy, exact source Call and public signature."""

        from .concordance_declarations import (
            INGESTION_VALUE, SOURCE_CALLABLE_SIGNATURE,
            SOURCE_EXTRACTION_IDENTITY, SOURCE_INTRINSIC_ADMISSION,
            SOURCE_SPAN, STATIC_SYMBOL_DEFINITION,
        )
        from ..common.tensors.operator_catalog import (
            CANONICAL_ABSTRACT_TENSOR_OPERATORS, canonical_operator_name,
        )

        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        page = book.page(SOURCE_INTRINSIC_ADMISSION)
        for row in page.rows():
            fact = page.latest(row)
            valid = (
                isinstance(row, tuple) and len(row) == 1
                and isinstance(row[0], Ref)
                and row[0].page == INGESTION_VALUE
                and isinstance(fact, tuple) and len(fact) == 3
            )
            if valid:
                op, decision, signature = fact
                valid = (
                    isinstance(op, str)
                    and op in CANONICAL_ABSTRACT_TENSOR_OPERATORS
                    and isinstance(decision, dict)
                    and decision.get("action") == "intrinsic"
                    and isinstance(decision.get("identity"), str)
                    and isinstance(decision.get("rule_id"), str)
                    and isinstance(signature, Ref)
                    and signature.page == SOURCE_CALLABLE_SIGNATURE
                )
            if valid:
                source_call = row[0]
                signature_fact = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
                    signature.row, signature.column,
                )
                symbol = signature.row[0]
                spans = tuple(
                    source for source, _stage in book.edges_into(source_call)
                    if source.page == SOURCE_SPAN
                    and book.page(SOURCE_SPAN).cell(
                        source.row, source.column,
                    ).kind == "Call"
                )
                extracted = book.latest_ref(
                    SOURCE_EXTRACTION_IDENTITY, (source_call,),
                )
                valid = (
                    len(spans) == 1
                    and isinstance(symbol, Ref)
                    and symbol.page == STATIC_SYMBOL_DEFINITION
                    and isinstance(signature_fact, tuple)
                    and len(signature_fact) == 3
                    and signature_fact[1][0][:2] == symbol.row[:2]
                    and decision.get("module") == symbol.row[0]
                    and decision.get("qualname") == symbol.row[1]
                    and decision.get("identity") == ".".join(symbol.row[:2])
                    and canonical_operator_name(
                        symbol.row[1].rsplit(".", 1)[-1]
                    ) == op
                    and extracted is not None
                    and book.page(SOURCE_EXTRACTION_IDENTITY).cell(
                        extracted.row, extracted.column,
                    ) == decision["identity"]
                    and source_call in {
                        source for source, _stage in book.edges_into(extracted)
                    }
                )
            if not valid:
                found.append(Finding(
                    "source-intrinsic-admission", str(row), None,
                    "intrinsic lacks exact accepted source policy, authored "
                    "Call, public wrapper signature or operator identity",
                ))
        return found

    @staticmethod
    def _source_call_parameter_default_findings(module: Any) -> list[Finding]:
        """Check an intrinsic omitted argument against its source call."""

        from .concordance_declarations import (
            PROCESS_GRAPH_OPERANDS, SOURCE_ACTIVITY_CALL,
            SOURCE_CALLABLE_SIGNATURE, SOURCE_CALL_PARAMETER_DEFAULT,
            SOURCE_PARAMETER_DEFAULT, SOURCE_SPAN, STATIC_REFERENCE_USE,
        )

        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        page = book.page(SOURCE_CALL_PARAMETER_DEFAULT)
        for row in page.rows():
            receipt = book.latest_ref(SOURCE_CALL_PARAMETER_DEFAULT, row)
            fact = page.latest(row)
            sources = {source for source, _stage in book.edges_into(receipt)}
            valid = (
                isinstance(row, tuple) and len(row) == 2
                and isinstance(row[0], Ref) and isinstance(row[1], str)
                and isinstance(fact, tuple) and len(fact) == 2
                and isinstance(fact[0], Ref)
                and fact[0].page == SOURCE_PARAMETER_DEFAULT
                and isinstance(fact[1], Ref)
                and fact[1].page == SOURCE_CALLABLE_SIGNATURE
                and {row[0], *fact} <= sources
            )
            if valid:
                default, signature = fact
                schema = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
                    signature.row, signature.column,
                )[2]
                matches = tuple(item for item in schema
                                if item[0] == row[1] and item[3] == default)
                valid = len(matches) == 1
            if valid:
                call_spans = tuple(source for source in sources
                                   if source.page == SOURCE_SPAN
                                   and book.page(SOURCE_SPAN).cell(
                                       source.row, source.column,
                                   ).kind == "Call")
                activities = tuple(source for source in sources
                                   if source.page == SOURCE_ACTIVITY_CALL)
                operands = tuple(source for source in sources
                                 if source.page == PROCESS_GRAPH_OPERANDS)
                uses = tuple(source for source in sources
                             if source.page == STATIC_REFERENCE_USE)
                valid = (len(call_spans) == len(activities) == len(operands)
                         == len(uses) == 1)
                callee_spans = tuple(
                    source for source in sources
                    if source.page == SOURCE_SPAN and source not in call_spans
                    and book.page(SOURCE_SPAN).cell(
                        source.row, source.column,
                    ).kind in {"Name", "Attribute"}
                    and source.row[:2] == call_spans[0].row[:2]
                    and source.row[2] == (*call_spans[0].row[2], ("func", -1))
                ) if valid else ()
                valid = valid and len(callee_spans) == 1
            if valid:
                call_span, activity, operand, use = (
                    call_spans[0], activities[0], operands[0], uses[0]
                )
                # The exact authored Call.func is the source occurrence of
                # this static reference. The static reference's reduction
                # cell can sit between the use and its SOURCE_SPAN cause.
                frontier = [use]
                ancestry = set()
                while frontier:
                    current = frontier.pop()
                    if current in ancestry:
                        continue
                    ancestry.add(current)
                    frontier.extend(
                        source for source, _stage in book.edges_into(current)
                        if source not in ancestry
                    )
                activity_fact = book.page(SOURCE_ACTIVITY_CALL).cell(
                    activity.row, activity.column,
                )
                name, kind, index, _default = matches[0]
                valid = (
                    activity.row[1] == call_span
                    and isinstance(activity_fact, tuple)
                    and len(activity_fact) == 2
                    and isinstance(activity_fact[0], int)
                    and isinstance(activity_fact[1], tuple)
                    and (kind == "POSITIONAL_ONLY"
                         or name not in activity_fact[1])
                    and (index is None or index >= activity_fact[0])
                    and row[0] in {
                        source for source, _stage in book.edges_into(operand)
                    }
                    and signature.row[0] in {
                        source for source, _stage in book.edges_into(use)
                    }
                    and callee_spans[0] in ancestry
                )
            if not valid:
                found.append(Finding(
                    "source-call-parameter-default", str(row), None,
                    "intrinsic call default lacks exact signature, omitted "
                    "argument inventory, or static callable provenance",
                ))
        return found

    @staticmethod
    def _numerical_schema_operand_findings(module: Any) -> list[Finding]:
        """Check that a numerical ABI exclusion has an exact schema cause."""

        from .concordance_declarations import (
            PROCESS_GRAPH_OPERANDS, SOURCE_PYTHON_IDENTITY,
            NUMERICAL_SCHEMA_OPERAND,
        )
        from ..transmogrifier.graph.python_identity_programs import (
            resolve_python_identity,
        )

        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        page = book.page(NUMERICAL_SCHEMA_OPERAND)
        for row in page.rows():
            receipt = book.latest_ref(NUMERICAL_SCHEMA_OPERAND, row)
            fact = page.latest(row)
            valid = isinstance(row, tuple) and len(row) == 3
            if valid:
                consumer, role, parent = row
                valid = (
                    isinstance(consumer, Ref) and isinstance(parent, Ref)
                    and isinstance(role, str) and isinstance(fact, tuple)
                    and len(fact) == 1 and isinstance(fact[0], Ref)
                    and fact[0].page == SOURCE_PYTHON_IDENTITY
                )
            if valid:
                identity = fact[0]
                identity_fact = book.page(SOURCE_PYTHON_IDENTITY).cell(
                    identity.row, identity.column,
                )
                valid = (
                    isinstance(identity_fact, tuple)
                    and len(identity_fact) == 4
                    and isinstance(identity_fact[0], str)
                    and isinstance(identity_fact[1], str)
                )
            if valid:
                program = resolve_python_identity(identity_fact[0])
                valid = (
                    program is not None
                    and program.object_type == "schema_type_identity"
                    and identity_fact[1] == program.kind
                    and _canonical_identity_base(book, parent) in {
                        _canonical_identity_base(book, source)
                        for source, _stage in book.edges_into(identity)
                        if source.page == parent.page
                    }
                )
            if valid:
                operand_cells = tuple(
                    source for source, _stage in book.edges_into(receipt)
                    if source.page == PROCESS_GRAPH_OPERANDS
                )
                valid = len(operand_cells) == 1 and (
                    (parent, role) in book.page(PROCESS_GRAPH_OPERANDS).cell(
                        operand_cells[0].row, operand_cells[0].column,
                    )
                    and consumer in {
                        source for source, _stage in book.edges_into(
                            operand_cells[0],
                        )
                    }
                    and {consumer, parent, identity, operand_cells[0]} <= {
                        source for source, _stage in book.edges_into(receipt)
                    }
                )
            if not valid:
                found.append(Finding(
                    "numerical-schema-operand", str(row), None,
                    "numerical ABI exclusion lacks its exact source schema "
                    "identity and operand edge",
                ))
        return found

    @staticmethod
    def _numerical_constructor_default_findings(module: Any) -> list[Finding]:
        """Check physical constructor dtype against imported SSA output stores."""

        from .concordance_declarations import (
            NUMERICAL_CONSTRUCTOR_CODE_REFERENCE,
            NUMERICAL_CONSTRUCTOR_DEFAULT, REPOSITORY_KERNEL_DEFINITION,
            REPOSITORY_KERNEL_VALUE, SOURCE_CALLABLE_SIGNATURE,
            SOURCE_CALL_PARAMETER_DEFAULT, SOURCE_INTRINSIC_ADMISSION,
            SOURCE_PARAMETER_DEFAULT, PROCESS_GRAPH_OPERANDS,
            SSA_VALUE, SpecializationFact, SpecializationSource,
        )
        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        declarations = book.page(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE)
        page = book.page(NUMERICAL_CONSTRUCTOR_DEFAULT)
        if not tuple(declarations.rows()) and not tuple(page.rows()):
            return found

        from .numerical_constructor_defaults import _stored_output_values
        from .concorded_literals import origin_cell
        from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
            c_backend_repository_ssa_reference,
        )
        from .ssa_record_return_state import function_scope_of

        code_reference = c_backend_repository_ssa_reference()
        for row in declarations.rows():
            cell = book.latest_ref(NUMERICAL_CONSTRUCTOR_CODE_REFERENCE, row)
            fact = declarations.latest(row)
            op = code_reference.operation(row[1]) if len(row) == 2 else None
            valid = (
                isinstance(row, tuple) and len(row) == 2
                and row[0] == code_reference.source_identity
                and op is not None and op.canonical_op == row[1]
                and fact == tuple(op.entrypoints)
                and len(fact) == 1
            )
            if valid:
                valid = code_reference.module.functions.get(fact[0]) is not None
            if not valid:
                found.append(Finding(
                    "numerical-constructor-code-reference", str(row), None,
                    "constructor operation lacks its exact repository SSA entrypoint",
                ))
        for row in page.rows():
            receipt = book.latest_ref(NUMERICAL_CONSTRUCTOR_DEFAULT, row)
            fact = page.latest(row)
            valid = (
                isinstance(row, tuple) and len(row) == 1
                and isinstance(row[0], Ref)
                and isinstance(fact, tuple) and len(fact) == 9
            )
            if valid:
                (op, default, admission, operand, declaration, root,
                 output, stored, dtype) = fact
                valid = (
                    isinstance(op, str) and op in {"zeros", "ones", "empty", "full"}
                    and isinstance(default, Ref)
                    and default.page == SOURCE_CALL_PARAMETER_DEFAULT
                    and isinstance(admission, Ref)
                    and admission.page == SOURCE_INTRINSIC_ADMISSION
                    and isinstance(operand, Ref)
                    and operand.page == PROCESS_GRAPH_OPERANDS
                    and isinstance(declaration, Ref)
                    and declaration.page == NUMERICAL_CONSTRUCTOR_CODE_REFERENCE
                    and isinstance(root, Ref)
                    and root.page == REPOSITORY_KERNEL_DEFINITION
                    and isinstance(output, Ref)
                    and output.page == REPOSITORY_KERNEL_VALUE
                    and isinstance(stored, tuple) and bool(stored)
                    and all(isinstance(item, Ref) and item.page == SSA_VALUE
                            for item in stored)
                    and isinstance(dtype, str)
                    and default.row[1] == "dtype"
                    and declaration.row == (code_reference.source_identity, op)
                )
            if valid:
                try:
                    valid = (origin_cell(book, default.row[0])
                             == origin_cell(book, row[0])
                             == admission.row[0])
                except ValueError:
                    valid = False
            if valid:
                causes = {source for source, _stage in book.edges_into(receipt)}
                valid = {row[0], default, admission, operand,
                         declaration, root, output, *stored} <= causes
            if valid:
                default_fact = book.page(SOURCE_CALL_PARAMETER_DEFAULT).cell(
                    default.row, default.column,
                )
                valid = (isinstance(default_fact, tuple)
                         and len(default_fact) == 2
                         and default_fact[0].page == SOURCE_PARAMETER_DEFAULT
                         and default_fact[1].page == SOURCE_CALLABLE_SIGNATURE
                         and set(default_fact) <= causes)
            if valid:
                admitted_op, decision, admitted_signature = book.page(
                    SOURCE_INTRINSIC_ADMISSION,
                ).cell(admission.row, admission.column)
                public_symbol = default_fact[1].row[0]
                valid = (
                    admitted_op == op
                    and admitted_signature == default_fact[1]
                    and isinstance(decision, dict)
                    and decision.get("action") == "intrinsic"
                    and (decision.get("parameters") or {}).get(
                        "lowering_namespace") == "abstract_tensor"
                    and decision.get("module") == public_symbol.row[0]
                    and decision.get("qualname") == public_symbol.row[1]
                    and decision.get("identity")
                        == ".".join(public_symbol.row[:2])
                    and operand == book.latest_ref(
                        PROCESS_GRAPH_OPERANDS, operand.row,
                    )
                    and row[0] in {
                        source for source, _stage in book.edges_into(operand)
                    }
                )
            if valid:
                default_value = book.page(SOURCE_PARAMETER_DEFAULT).cell(
                    default_fact[0].row, default_fact[0].column,
                )
                schema = book.page(SOURCE_CALLABLE_SIGNATURE).cell(
                    default_fact[1].row, default_fact[1].column,
                )[2]
                valid = (
                    isinstance(default_value, SpecializationFact)
                    and default_value.source == SpecializationSource.DEFAULT
                    and default_value.value is None
                    and sum(name == "dtype" and cell == default_fact[0]
                            for name, _kind, _position, cell in schema) == 1
                )
            if valid:
                position = next(index for name, _kind, index, cell in schema
                                if name == "dtype" and cell == default_fact[0])
                edges = book.page(PROCESS_GRAPH_OPERANDS).cell(
                    operand.row, operand.column,
                )
                valid = not any(
                    role == "kw:dtype"
                    or (position is not None and role == f"arg:{position}")
                    for _parent, role in edges
                )
            if valid:
                declaration_fact = declarations.cell(
                    declaration.row, declaration.column,
                )
                valid = len(declaration_fact) == 1
            if valid:
                entrypoint = declaration_fact[0]
                function = code_reference.module.functions.get(entrypoint)
                output_stores = (_stored_output_values(function)
                                 if function is not None else None)
                valid = (root.row == ("llvm_repository", entrypoint)
                         and output_stores is not None)
            if valid:
                output_value, stored_values = output_stores
                origins = function.metadata["llvm_repository_kernel"][1]
                expected_output = book.latest_ref(
                    REPOSITORY_KERNEL_VALUE,
                    (root.row, origins[int(output_value.id)][0]),
                )
                scope = function_scope_of(function)
                expected_stored = tuple(book.latest_ref(
                    SSA_VALUE, (scope, int(value.id)),
                ) for value in stored_values)
                valid = (output == expected_output
                         and stored == expected_stored
                         and all(value.dtype == dtype for value in stored_values))
            if not valid:
                found.append(Finding(
                    "numerical-constructor-default", str(row), None,
                    "physical dtype lacks exact omitted source default or native output store",
                ))
        return found

    @staticmethod
    def _materialized_byte_row_findings(module: Any) -> list[Finding]:
        """Audit only byte-iterable claims in the incumbent row dtype page."""

        from .concordance_declarations import (
            BUILTIN_ITERABLE_ELEMENT, CANONICAL_VALUE,
            PROCESS_GRAPH_OPERANDS, SEQUENCE_ROW_DTYPE,
            SOURCE_PYTHON_IDENTITY, SOURCE_VALUE_CLASS,
        )
        from ..transmogrifier.graph.python_identity_programs import (
            BUILTIN_ITERABLE_ELEMENTS,
        )
        from .concorded_literals import origin_cell
        from .planner_concordance_posts import _class_proves_value

        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        protocol_page = book.page(BUILTIN_ITERABLE_ELEMENT)
        for row in protocol_page.rows():
            fact = protocol_page.latest(row)
            if (not isinstance(row, tuple) or len(row) != 1
                    or fact != BUILTIN_ITERABLE_ELEMENTS.get(row[0])):
                found.append(Finding(
                    "builtin-iterable-element", str(row), None,
                    "builtin iterable row contract differs from ingested declaration",
                ))
        page = book.page(SEQUENCE_ROW_DTYPE)
        for row in page.rows():
            for column, fact in page.history(row):
                if not (isinstance(fact, tuple) and len(fact) == 2
                        and fact[1] == "builtin iterable materialization"):
                    continue
                receipt = Ref(SEQUENCE_ROW_DTYPE, row, column)
                causes = {source for source, _stage in book.edges_into(receipt)}
                protocols = tuple(source for source in causes
                                  if source.page == BUILTIN_ITERABLE_ELEMENT)
                identities = tuple(source for source in causes
                                   if source.page == SOURCE_PYTHON_IDENTITY)
                operands = tuple(source for source in causes
                                 if source.page == PROCESS_GRAPH_OPERANDS)
                classes = tuple(source for source in causes
                                if source.page == SOURCE_VALUE_CLASS)
                valid = (
                    len(protocols) == len(identities) == len(operands) == 1
                    and bool(classes)
                    and isinstance(row, tuple) and len(row) == 2
                    and fact[0] == ("int64",)
                    and protocol_page.cell(protocols[0].row,
                                           protocols[0].column)
                        == BUILTIN_ITERABLE_ELEMENTS.get(protocols[0].row[0])
                    and len(tuple(book.page(PROCESS_GRAPH_OPERANDS).cell(
                        operands[0].row, operands[0].column,
                    ))) >= 1
                )
                if valid:
                    identity_fact = book.page(SOURCE_PYTHON_IDENTITY).cell(
                        identities[0].row, identities[0].column,
                    )
                    edges = book.page(PROCESS_GRAPH_OPERANDS).cell(
                        operands[0].row, operands[0].column,
                    )
                    args = tuple(source for source, role in edges
                                 if role == "arg:0")
                    consumers = tuple(source for source in causes
                                      if source.page == CANONICAL_VALUE
                                      and source.row[0] == row[0]
                                      and source.row[1] == row[1])
                    valid = (
                        isinstance(identity_fact, tuple)
                        and identity_fact[0] in {"builtins.list", "builtins.tuple"}
                        and args and len(args) == 1
                        and args[0] in causes
                        and any(
                            _class_proves_value(book, class_cell, args[0])
                            and book.page(SOURCE_VALUE_CLASS).cell(
                                class_cell.row, class_cell.column,
                            )[0] == protocols[0].row[0]
                            for class_cell in classes
                        )
                        and len(consumers) == 1
                        and consumers[0] in {
                            source for source, _stage in book.edges_into(
                                operands[0],
                            )
                        }
                        and not any(role.startswith("arg:") and role != "arg:0"
                                    for _source, role in edges)
                    )
                    if valid:
                        try:
                            valid = any(
                                origin_cell(book, source)
                                == origin_cell(book, consumers[0])
                                for source, _stage in book.edges_into(
                                    identities[0],
                                ) if source.page == CANONICAL_VALUE
                            )
                        except ValueError:
                            valid = False
                if not valid:
                    found.append(Finding(
                        "materialized-byte-row", str(row), None,
                        "byte materialization row lacks exact protocol, class, "
                        "identity or operand proof",
                    ))
        return found

    @staticmethod
    def _numerical_dtype_operand_findings(module: Any) -> list[Finding]:
        """Check an omitted dtype operand against its exact literal and output."""

        from .concordance_declarations import (
            NUMERICAL_DTYPE_OPERAND, PLAN_REGION_VALUE_DTYPE,
            PROCESS_GRAPH_OPERANDS,
            PROVEN_LITERAL, SOURCE_LITERAL_VALUE, SYNCHRONIZATION_LITERAL,
        )
        from ..transmogrifier.dtype_layout import resolve as resolve_dtype

        book = (getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        page = book.page(NUMERICAL_DTYPE_OPERAND)
        for row in page.rows():
            receipt = book.latest_ref(NUMERICAL_DTYPE_OPERAND, row)
            fact = page.latest(row)
            valid = (
                isinstance(row, tuple) and len(row) == 3
                and isinstance(fact, tuple) and len(fact) == 2
            )
            if valid:
                consumer, role, parent = row
                literal_cells, dtype = fact
                valid = (
                    isinstance(consumer, Ref) and isinstance(parent, Ref)
                    and isinstance(role, str) and isinstance(dtype, str)
                    and isinstance(literal_cells, tuple) and bool(literal_cells)
                    and all(isinstance(cell, Ref) for cell in literal_cells)
                    and resolve_dtype(dtype) is not None
                    and resolve_dtype(dtype).name == dtype
                )
            if valid:
                causes = {source for source, _stage in book.edges_into(receipt)}
                operands = tuple(source for source in causes
                                 if source.page == PROCESS_GRAPH_OPERANDS)
                dtype_cells = tuple(source for source in causes
                                    if source.page == PLAN_REGION_VALUE_DTYPE)
                valid = (
                    len(operands) == 1 and len(dtype_cells) == 1
                    and {consumer, parent, *literal_cells} <= causes
                    and (parent, role) in book.page(PROCESS_GRAPH_OPERANDS).cell(
                        operands[0].row, operands[0].column,
                    )
                    and consumer in {
                        source for source, _stage in book.edges_into(operands[0])
                    }
                )
            if valid:
                settled_dtype = book.page(PLAN_REGION_VALUE_DTYPE).cell(
                    dtype_cells[0].row, dtype_cells[0].column,
                )
                valid = (
                    dtype_cells[0].row == consumer.row
                    and settled_dtype == dtype
                    and consumer in {
                        source for source, _stage in book.edges_into(dtype_cells[0])
                    }
                )
            if valid:
                admitted = {SOURCE_LITERAL_VALUE, PROVEN_LITERAL,
                            SYNCHRONIZATION_LITERAL}
                for literal_cell in literal_cells:
                    if literal_cell.page not in admitted:
                        valid = False
                        break
                    literal = book.page(literal_cell.page).cell(
                        literal_cell.row, literal_cell.column,
                    )
                    if literal_cell.page == SYNCHRONIZATION_LITERAL:
                        literal = literal[2] if isinstance(literal, tuple) and len(literal) == 3 else None
                    layout = resolve_dtype(literal) if isinstance(literal, str) else None
                    if layout is None or layout.name != dtype:
                        valid = False
                        break
                    if _canonical_identity_base(book, parent) not in {
                        _canonical_identity_base(book, source)
                        for source, _stage in book.edges_into(literal_cell)
                        if source.page == parent.page
                    }:
                        valid = False
                        break
            if not valid:
                found.append(Finding(
                    "numerical-dtype-operand", str(row), None,
                    "numerical dtype operand lacks exact source literal, "
                    "operand edge, or matching settled output dtype",
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
    def _source_numeric_specialization_findings(module: Any) -> list[Finding]:
        """Check the exact target/class causes of iterable specialization keys."""
        import hashlib
        from .concordance_declarations import (
            SOURCE_NUMERIC_SPECIALIZATION_PAGE, SOURCE_VALUE_CLASS, CELL_SET,
            FUNCTION_ADDRESS,
        )
        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        page = None if book is None else book.pages.get(SOURCE_NUMERIC_SPECIALIZATION_PAGE.name)
        if page is None:
            return []

        def unpack(cells):
            result, pending = set(), list(cells)
            while pending:
                cell = pending.pop()
                if cell in result:
                    continue
                result.add(cell)
                if cell.page == CELL_SET:
                    parents = tuple(parent for parent, _ in book.edges_into(cell))
                    fact = book.page(CELL_SET).cell(cell.row, cell.column)
                    if set(fact) != {parent.key for parent in parents}:
                        raise ValueError("CELL_SET members disagree with its exact source edges")
                    pending.extend(parents)
            return result

        found = []
        for row in page.rows():
            scope, receipt = page.latest(row)
            entries = [entry for entry in receipt if isinstance(entry, tuple)
                       and len(entry) == 2 and isinstance(entry[0], tuple)
                       and entry[0] and entry[0][0] == "iterated_class"]
            if not entries:
                continue
            try:
                digest = hashlib.sha256(repr(receipt).encode()).hexdigest()[:16]
                if row != (scope[0], digest) or scope != (row[0], "specialization", digest):
                    raise ValueError("receipt does not identify its recorded specialization")
                ref = book.latest_ref(SOURCE_NUMERIC_SPECIALIZATION_PAGE, row)
                mint = book.mint_of(ref)
                direct = () if mint is None else mint[1]
                if not any(parent.page == CELL_SET for parent in direct):
                    raise ValueError("iterable domain has no exact CELL_SET cause")
                causes = unpack(direct)
                if not any(parent.page == FUNCTION_ADDRESS for parent in causes):
                    raise ValueError("specialization has no authored function authority")
                for key, domain in entries:
                    if len(key) != 2 or not isinstance(key[1], tuple) or len(key[1]) != 3:
                        raise ValueError("iterable target is not a complete canonical Ref key")
                    targets = [cell for cell in causes if cell.key == key[1]
                               and cell.page.name in _IDENTITY_PAGES]
                    if len(targets) != 1 or _canonical_identity_base(book, targets[0]) != targets[0]:
                        raise ValueError("iterable target has no exact canonical subject cause")
                    target = targets[0]
                    if (not isinstance(domain, tuple) or len(domain) != 2
                            or not isinstance(domain[0], str) or type(domain[1]) is not int
                            or domain[1] < 1):
                        raise ValueError("invalid iterable class/precision domain")
                    matches = []
                    for cell in causes:
                        if cell.page != SOURCE_VALUE_CLASS:
                            continue
                        fact = book.page(SOURCE_VALUE_CLASS).cell(cell.row, cell.column)
                        if tuple(fact[:2]) != domain:
                            continue
                        subjects = unpack(parent for parent, _ in book.edges_into(cell))
                        if any(parent.page.name in _IDENTITY_PAGES
                               and _canonical_identity_base(book, parent) == target
                               for parent in subjects):
                            matches.append(cell)
                    if not matches:
                        raise ValueError("iterable class has no matching exact subject/class cause")
            except (ValueError, TypeError, KeyError, IndexError) as error:
                found.append(Finding("source-numeric-specialization-disagreement", str(scope), None, str(error)))
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
                from .structural_concordance import ssa_structure
                try:
                    physical, physical_cell = ssa_structure(
                        scope, value_id, shape_scope=shape_scope_of(function),
                        storage_scope=str(function.name), book=book,
                    )
                except ConcordanceRefusal as error:
                    found.append(Finding(
                        "ssa-publication-type-disagreement", str(name), value_id,
                        str(error),
                    ))
                    continue
                for (key, field), resident in zip(fields, view):
                    posted = getattr(physical, "declared_storage" if field == "storage"
                                     else field, None)
                    if field == "device":
                        posted = fact.device
                    if _physical_unset(key, resident):
                        continue
                    if _physical_unset(key, posted):
                        found.append(Finding(
                            "ssa-physical-unposted", str(name), value_id,
                            f"{physical_cell!r} has no {field}; resident accounting "
                            f"{key}={resident!r} at {tuple(positions)!r}",
                        ))
                    elif posted != resident:
                        found.append(Finding(
                            "ssa-publication-type-disagreement", str(name), value_id,
                            f"{physical_cell!r} records {field}={posted!r}; resident "
                            f"accounting {key}={resident!r} at {tuple(positions)!r}",
                        ))
        return found

    @staticmethod
    def _mapping_child_contract_findings(module: Any) -> list[Finding]:
        """Check shared child views against their exact source class/layout."""
        from .concordance_declarations import (
            CLASS_FIELD_DECLARATION, SOURCE_VALUE_CLASS,
            SOURCE_AGGREGATE_LAYOUT, SourceAggregateLayout,
        )
        metadata = dict(getattr(module, "metadata", {}) or {})
        book = metadata.get("identity_book")
        found = []
        # An ``int`` key column is the narrowest width holding the join of
        # every writer's member interval: each view cites the
        # ``source_int_range_concordance`` cells it read (none: the int64
        # range).  Re-derive that join from the cited cells.
        from .concordance_declarations import SOURCE_INT_RANGE
        from ..transmogrifier.dtype_layout import python_int_range_dtype

        int64_bounds = (-(1 << 63), (1 << 63) - 1)
        joined_key_widths: dict = {}
        if book is not None:
            joins: dict = {}
            for view in metadata.get("mapping_child_contract_evidence") or ():
                for index, cell in enumerate(view.get("key_range_cells") or ()):
                    interval = int64_bounds
                    if isinstance(cell, Ref) and cell.page == SOURCE_INT_RANGE:
                        fact = book.page(SOURCE_INT_RANGE).cell(cell.row, cell.column)
                        interval = (int(fact[0]), int(fact[1]))
                    key = (view.get("field_cell"), index)
                    low, high = joins.get(key, interval)
                    joins[key] = (min(low, interval[0]), max(high, interval[1]))
            joined_key_widths = {
                key: python_int_range_dtype(*interval)
                for key, interval in joins.items()
            }

        def int_key_dtype(view, index, cell, expected):
            if expected != "int64":
                return expected
            range_cells = tuple(view.get("key_range_cells") or ())
            if index < len(range_cells):
                range_cell = range_cells[index]
                if range_cell is not None and (
                    not isinstance(range_cell, Ref)
                    or range_cell.page != SOURCE_INT_RANGE
                    or int(range_cell.row[1]) != int(cell.row[1])
                ):
                    return None
                return joined_key_widths.get(
                    (view.get("field_cell"), index), expected,
                )
            return expected

        for view in metadata.get("mapping_child_contract_evidence") or ():
            field, scalar, key = (view.get(name) for name in (
                "field_cell", "class_cell", "key_layout_cell",
            ))
            problem = None
            # A scalar key has no aggregate layout: its one key column is
            # its own exact source class.
            scalar_key = key is None and view.get("key_width") == 1
            if book is None or any(
                not isinstance(cell, Ref) or cell.page != page
                for cell, page in ((field, CLASS_FIELD_DECLARATION),
                                   (scalar, SOURCE_VALUE_CLASS),
                                   *(() if scalar_key else ((key, SOURCE_AGGREGATE_LAYOUT),)))
            ):
                problem = "child view lacks its field/class/key-layout authority"
            elif scalar_key:
                class_fact = book.page(SOURCE_VALUE_CLASS).cell(scalar.row, scalar.column)
                expected = {"builtins.int": "int64", "builtins.float": "float64", "builtins.bool": "bool"}.get(str(class_fact[0]))
                key_cells = tuple(view.get("key_class_cells", ()))
                key_cell = key_cells[0] if len(key_cells) == 1 else None
                key_fact = (None if not isinstance(key_cell, Ref)
                            or key_cell.page != SOURCE_VALUE_CLASS
                            or key_cell.row[0] != scalar.row[0]
                            else book.page(SOURCE_VALUE_CLASS).cell(key_cell.row, key_cell.column))
                if expected != view.get("dtype") or int(class_fact[1]) != 1:
                    problem = f"child dtype {view.get('dtype')!r} disagrees with operand class {class_fact!r}"
                elif key_fact is None or int(key_fact[1]) != 1 or tuple(view.get("key_dtypes", ())) != (
                    int_key_dtype(view, 0, key_cell, {"builtins.int": "int64", "builtins.float": "float64", "builtins.bool": "bool"}.get(str(key_fact[0]))),
                ):
                    problem = "scalar key dtype disagrees with its exact key class"
            else:
                try:
                    class_fact = book.page(SOURCE_VALUE_CLASS).cell(scalar.row, scalar.column)
                    layout = book.page(SOURCE_AGGREGATE_LAYOUT).cell(key.row, key.column)
                    expected = {"builtins.int": "int64", "builtins.float": "float64", "builtins.bool": "bool"}.get(str(class_fact[0]))
                    if expected != view.get("dtype") or int(class_fact[1]) != 1:
                        problem = f"child dtype {view.get('dtype')!r} disagrees with operand class {class_fact!r}"
                    elif not isinstance(layout, SourceAggregateLayout) or layout.aggregate_kind != "tuple" or len(layout.members) != view.get("key_width"):
                        problem = "child key width disagrees with its ordered tuple layout"
                    elif len(view.get("key_class_cells", ())) != len(layout.members) or len(view.get("key_dtypes", ())) != len(layout.members):
                        problem = "child key column type view has the wrong member count"
                    else:
                        for index, (member, cell, dtype) in enumerate(zip(layout.members, view["key_class_cells"], view["key_dtypes"], strict=True)):
                            if cell is None:
                                if dtype != "unknown":
                                    problem = "unproved key member was assigned a physical dtype"
                                continue
                            if not isinstance(cell, Ref) or cell.page != SOURCE_VALUE_CLASS or int(cell.row[1]) != int(member.value_id) or cell.row[0] != scalar.row[0]:
                                problem = "key class does not name its exact ordered member"
                                break
                            key_fact = book.page(SOURCE_VALUE_CLASS).cell(cell.row, cell.column)
                            expected_key = int_key_dtype(view, index, cell, {"builtins.int": "int64", "builtins.float": "float64", "builtins.bool": "bool"}.get(str(key_fact[0]), "unknown"))
                            if dtype != expected_key:
                                problem = "key dtype disagrees with its exact member class"
                                break
                    if book.page(CLASS_FIELD_DECLARATION).cell(field.row, field.column) is None:
                        problem = "child field declaration is absent"
                except (KeyError, IndexError, TypeError, ValueError) as error:
                    problem = f"child authority cannot be read: {error}"
            if problem is not None:
                found.append(Finding("mapping-child-contract-disagreement", str(getattr(field, "row", "<source>")), None, problem))
        return found

    @staticmethod
    def _source_aggregate_layout_findings(module: Any) -> list[Finding]:
        """Validate immutable reducer-owned aggregate layout facts."""

        from .concordance_declarations import (
            CALL_RESULT_PROJECTION, CALLSITE_RETURN_MEMBER,
            PROCESS_GRAPH_OPERANDS, READ_SCOPE_FORK,
            SOURCE_AGGREGATE_LAYOUT,
            SOURCE_NAMED_TUPLE_LAYOUT,
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
            fork_projection = False
            if len(distinct) != 1 and len(history) == 2 and all(
                isinstance(item, SourceAggregateLayout) for item in facts
            ):
                inherited, projected = facts
                first = Ref(SOURCE_AGGREGATE_LAYOUT, row, history[0][0])
                last = Ref(SOURCE_AGGREGATE_LAYOUT, row, history[1][0])
                origins = tuple(
                    source for source, stage in book.edges_into(first)
                    if source.page == SOURCE_AGGREGATE_LAYOUT
                    and stage == READ_SCOPE_FORK
                )
                fork_projection = (
                    len(origins) == 1
                    and book.page(SOURCE_AGGREGATE_LAYOUT).cell(
                        origins[0].row, origins[0].column,
                    ) == inherited
                    and first in {
                        source for source, _stage in book.edges_into(last)
                    }
                    and inherited.aggregate_kind == projected.aggregate_kind
                    and tuple(member.value_id for member in inherited.members)
                    == tuple(member.value_id for member in projected.members)
                    and all(
                        new.value_cell.row == (row[0], new.value_id)
                        and (old.value_cell, READ_SCOPE_FORK)
                        in book.edges_into(new.value_cell)
                        for old, new in zip(
                            inherited.members, projected.members,
                        )
                    )
                )
            if len(distinct) != 1 and not fork_projection:
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
            if problem is None:
                named = tuple(dict.fromkeys(
                    source for member in fact.members
                    for source in member.source_cells
                    if source.page == SOURCE_NAMED_TUPLE_LAYOUT
                ))
                if named:
                    layout_cell = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, row)
                    # A later source read-scope fork can inherit this exact
                    # immutable layout without publishing a local revision.
                    # In root(values, k).topk(...).indices, the final audit
                    # saw only the child layout's direct copy edge and lost
                    # the parent cell's named class/projection causes. Follow
                    # that one declared copy chain, requiring the fact to
                    # remain identical at every hop; a changed fact must have
                    # its own audited projection revision above.
                    cause_cell = layout_cell
                    copy_seen = set()
                    copy_valid = True
                    while cause_cell not in copy_seen:
                        copy_seen.add(cause_cell)
                        causes = tuple(book.edges_into(cause_cell))
                        if len(causes) != 1 or not (
                            causes[0][0].page == SOURCE_AGGREGATE_LAYOUT
                            and causes[0][1] == READ_SCOPE_FORK
                        ):
                            break
                        parent = causes[0][0]
                        if book.page(SOURCE_AGGREGATE_LAYOUT).cell(
                            parent.row, parent.column,
                        ) != fact:
                            copy_valid = False
                            break
                        cause_cell = parent
                    else:
                        copy_valid = False
                    direct = {
                        source for source, _ in book.edges_into(cause_cell)
                    } if copy_valid else set()
                    named_fact = book.page(SOURCE_NAMED_TUPLE_LAYOUT).cell(
                        named[0].row, named[0].column,
                    ) if len(named) == 1 else None
                    declared = (
                        named_fact[1]
                        if isinstance(named_fact, tuple)
                        and len(named_fact) == 2 else ()
                    )
                    if (
                        fact.aggregate_kind != "tuple"
                        or len(named) != 1 or named[0] not in direct
                        or len(declared) != len(fact.members)
                    ):
                        problem = "named aggregate lacks its one ordered class layout"
                    else:
                        for index, member in enumerate(fact.members):
                            sources = set(member.source_cells)
                            projections = tuple(source for source in sources
                                                if source.page == CALL_RESULT_PROJECTION)
                            typed = tuple(source for source in sources
                                          if source.page == CALLSITE_RETURN_MEMBER)
                            operands = tuple(source for source in sources
                                             if source.page == PROCESS_GRAPH_OPERANDS)
                            valid = (
                                declared[index][1] in sources
                                and member.value_cell in direct
                                and sources <= direct
                                and len(projections) == 1
                                and len(typed) == 1
                                and len(operands) == 1
                            )
                            if valid:
                                valid = (
                                    book.page(CALL_RESULT_PROJECTION).cell(
                                        projections[0].row,
                                        projections[0].column,
                                    ) == (index,)
                                    and projections[0].row[1:] == (
                                        row[1], member.value_id,
                                    )
                                    and typed[0].row[1:] == (
                                        row[1], index, member.value_id,
                                    )
                                    and operands[0].row[-1] == row[1]
                                    and member.value_cell in {
                                        source for source, _ in book.edges_into(
                                            projections[0],
                                        )
                                    }
                                )
                            if not valid:
                                problem = (
                                    "named aggregate member lacks exact field, "
                                    "projection, typed member or operand causes"
                                )
                                break
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
    def _loop_native_scalar_union_gap_findings(module: Any) -> list[Finding]:
        """Check exact loop binding, owner, and distinct scalar ABI arms."""

        from .concordance_declarations import (
            CANONICAL_VALUE, LOOP_CARRIED_BINDING,
            LOOP_NATIVE_SCALAR_UNION_GAP, LOOP_UPDATE_MINIMUM_TRIPS,
            VALUE_TYPE_ABI, LoopNativeScalarUnionGap, ValueTypeABIFact,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        for row in book.page(LOOP_NATIVE_SCALAR_UNION_GAP).rows():
            cell = book.latest_ref(LOOP_NATIVE_SCALAR_UNION_GAP, row)
            fact = book.page(LOOP_NATIVE_SCALAR_UNION_GAP).cell(
                cell.row, cell.column)
            expected = (
                book.latest_ref(LOOP_CARRIED_BINDING, row),
                book.latest_ref(CANONICAL_VALUE, (row[0], row[1])),
                book.latest_ref(VALUE_TYPE_ABI, (row[0], row[3])),
                book.latest_ref(VALUE_TYPE_ABI, (row[0], row[2])),
            )
            problem = None
            if not isinstance(fact, LoopNativeScalarUnionGap):
                problem = "invalid loop scalar union gap"
            elif (
                fact.operation != "header_exit_scalar_arm_projection"
                or (fact.binding_cell, fact.loop_cell,
                    fact.initial_abi_cell, fact.updated_abi_cell) != expected
                or any(ref is None for ref in expected)
                or not set(expected) <= {
                    source for source, _stage in book.edges_into(cell)
                }
                or book.latest_ref(LOOP_UPDATE_MINIMUM_TRIPS,
                                   (row[0], row[2])) is not None
            ):
                problem = "loop scalar union gap lacks exact binding/owner/arms"
            else:
                initial = book.page(VALUE_TYPE_ABI).cell(
                    fact.initial_abi_cell.row, fact.initial_abi_cell.column)
                updated = book.page(VALUE_TYPE_ABI).cell(
                    fact.updated_abi_cell.row, fact.updated_abi_cell.column)
                if not (
                    isinstance(initial, ValueTypeABIFact)
                    and isinstance(updated, ValueTypeABIFact)
                    and len(initial.physical_alternatives) == 1
                    and len(updated.physical_alternatives) == 1
                    and initial.physical_alternatives[0].storage == "scalar"
                    and updated.physical_alternatives[0].storage == "scalar"
                    and initial.physical_alternatives[0].dtype
                    != updated.physical_alternatives[0].dtype
                    and initial.physical_alternatives[0].identity
                    != updated.physical_alternatives[0].identity
                ):
                    problem = "loop scalar union gap changed physical arms"
            if problem is not None:
                found.append(Finding(
                    "loop-native-scalar-union-gap-disagreement",
                    str(row[0]), int(row[1]), problem,
                ))
        return found

    @staticmethod
    def _source_call_closed_class_alternatives_findings(module: Any) -> list[Finding]:
        """Check the exact one-output, one-callee class-domain transport."""

        from .concordance_declarations import (
            CANONICAL_VALUE, SEMANTIC_OUTPUT,
            SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
            SOURCE_CLOSED_CLASS_ALTERNATIVES,
            SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
            SOURCE_FUNCTION_REACHABILITY, SourceClosedClassAlternatives,
        )
        from .source_evidence_resolver import (
            ClassAlternativesEvidence, require_class_alternatives,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        for row in book.page(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).rows():
            cell = book.latest_ref(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES, row)
            fact = book.page(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).cell(
                cell.row, cell.column)
            problem = None
            if not (isinstance(fact, tuple) and len(fact) == 5
                    and all(isinstance(ref, Ref) for ref in fact)):
                problem = "invalid call class-alternative transport"
            else:
                output, returned, closed, reached, call = fact
                output_fact = (
                    book.page(SEMANTIC_OUTPUT).cell(output.row, output.column)
                    if output.page == SEMANTIC_OUTPUT
                    and (output.row, output.column) in
                    book.page(SEMANTIC_OUTPUT).cells else None
                )
                closed_fact = (
                    book.page(SOURCE_CLOSED_CLASS_ALTERNATIVES).cell(
                        closed.row, closed.column)
                    if closed.page == SOURCE_CLOSED_CLASS_ALTERNATIVES
                    and (closed.row, closed.column) in
                    book.page(SOURCE_CLOSED_CLASS_ALTERNATIVES).cells else None
                )
                formal_domain = (
                    require_class_alternatives(
                        returned.row[0], int(returned.row[1]), book=book)
                    if closed.page == SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES
                    and closed.row[:2] == returned.row
                    and closed.row[2:] == row[:2]
                    else None
                )
                reached_rows = tuple(
                    candidate for candidate in
                    book.page(SOURCE_FUNCTION_REACHABILITY).rows()
                    if candidate[:2] == row[:2]
                    and book.page(SOURCE_FUNCTION_REACHABILITY).latest(
                        candidate) is True
                )
                if not (
                    output.page == SEMANTIC_OUTPUT
                    and output.row[1] == 0
                    and isinstance(output_fact, tuple)
                    and len(output_fact) == 2
                    and int(output_fact[1]) == int(row[3])
                    and returned.page == CANONICAL_VALUE
                    and returned.row[1] == row[3]
                    and returned in {source for source, _stage in
                                     book.edges_into(output)}
                    and (
                        closed.page == SOURCE_CLOSED_CLASS_ALTERNATIVES
                        and closed.row == returned.row
                        and isinstance(closed_fact,
                                       SourceClosedClassAlternatives)
                        or isinstance(formal_domain,
                                      ClassAlternativesEvidence)
                        and formal_domain.subject_cell == closed
                    )
                    and reached.page == SOURCE_FUNCTION_REACHABILITY
                    and reached.row == row[:3]
                    and reached_rows == (row[:3],)
                    and call.page == CANONICAL_VALUE
                    and call.row[1] == row[1]
                    and call in {source for source, _stage in
                                 book.edges_into(reached)}
                    and set(fact) <= {source for source, _stage in
                                      book.edges_into(cell)}
                ):
                    problem = "call class domain lacks exact output/reachability"
            if problem is not None:
                found.append(Finding(
                    "source-call-closed-class-alternatives-disagreement",
                    str(row[0]), int(row[1]), problem,
                ))
        return found

    @staticmethod
    def _source_formal_closed_class_alternatives_findings(module: Any) -> list[Finding]:
        """Check a formal class domain against its exact actual/call edges."""

        from .concordance_declarations import (
            CANONICAL_VALUE, NAME_BINDING, SEMANTIC_OUTPUT,
            SOURCE_CALL_ACTUAL_FORMAL,
            SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
            SOURCE_CLOSED_CLASS_ALTERNATIVES,
            SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        for row in book.page(SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES).rows():
            cell = book.latest_ref(SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
                                   row)
            fact = book.page(SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES).cell(
                cell.row, cell.column)
            problem = None
            if not (isinstance(fact, tuple) and len(fact) == 5
                    and all(isinstance(ref, Ref) for ref in fact)):
                problem = "invalid formal class-alternative transport"
            else:
                binding, transport, closed, output, formal_binding = fact
                binding_fact = (
                    book.page(SOURCE_CALL_ACTUAL_FORMAL).cell(
                        binding.row, binding.column)
                    if binding.page == SOURCE_CALL_ACTUAL_FORMAL and
                    (binding.row, binding.column) in
                    book.page(SOURCE_CALL_ACTUAL_FORMAL).cells else None
                )
                transport_fact = (
                    book.page(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).cell(
                        transport.row, transport.column)
                    if transport.page == SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES
                    and (transport.row, transport.column) in
                    book.page(SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES).cells
                    else None
                )
                formal_value = book.latest_ref(
                    CANONICAL_VALUE, (row[0], int(row[1])))
                if not (
                    isinstance(binding_fact, tuple)
                    and len(binding_fact) == 5
                    and isinstance(transport_fact, tuple)
                    and len(transport_fact) == 5
                    and binding.row[0] == row[2]
                    and int(binding.row[1]) == int(row[3])
                    and int(binding.row[3]) == int(row[1])
                    and binding_fact[4] == formal_binding
                    and formal_binding.page == NAME_BINDING
                    and formal_binding.row[0] == row[0]
                    and transport.row[0] == row[2]
                    and int(transport.row[1]) == int(binding_fact[0].row[1])
                    and transport_fact[0] == output
                    and transport_fact[2] == closed
                    and output.page == SEMANTIC_OUTPUT
                    and closed.page == SOURCE_CLOSED_CLASS_ALTERNATIVES
                    and formal_value is not None
                    and {binding, transport, closed, output,
                         formal_binding, formal_value,
                         binding_fact[0]} <= {
                        source for source, _stage in book.edges_into(cell)
                    }
                ):
                    problem = "formal class domain lacks exact call binding"
            if problem is not None:
                found.append(Finding(
                    "source-formal-closed-class-alternatives-disagreement",
                    str(row[0]), int(row[1]), problem,
                ))
        return found

    @staticmethod
    def _source_class_union_native_gap_findings(module: Any) -> list[Finding]:
        """Check an unlaid class merge against copied arm and source cells."""

        from .concordance_declarations import (
            CANONICAL_VALUE, PROCESS_GRAPH_OPERANDS,
            SOURCE_CLASS_UNION_NATIVE_GAP, SOURCE_CLOSED_CLASS_ALTERNATIVES,
            SOURCE_VALUE_CLASS, VALUE_TYPE_ABI, SourceClassUnionNativeGap,
            SourceClosedClassAlternatives,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        page = book.page(SOURCE_CLASS_UNION_NATIVE_GAP)
        found: list[Finding] = []
        for row in page.rows():
            cell = book.latest_ref(SOURCE_CLASS_UNION_NATIVE_GAP, row)
            fact = page.cell(cell.row, cell.column)
            sources = {source for source, _stage in book.edges_into(cell)}
            problem = None
            if not isinstance(fact, SourceClassUnionNativeGap):
                problem = "invalid native class union gap"
            elif not (
                len(fact.alternatives) == len(fact.arm_value_cells) == 2
                and isinstance(fact.closed_cell, Ref)
                and fact.closed_cell.page == SOURCE_CLOSED_CLASS_ALTERNATIVES
                and fact.closed_cell.row == row
                and isinstance(fact.operand_cell, Ref)
                and fact.operand_cell.page == PROCESS_GRAPH_OPERANDS
                and fact.operand_cell.row == row
                and all(isinstance(arm, Ref)
                        and arm.page == CANONICAL_VALUE
                        and arm.row[0] == row[0]
                        for arm in fact.arm_value_cells)
                and len(set(fact.alternatives)) == 2
                and {fact.closed_cell, fact.operand_cell,
                     *fact.arm_value_cells} <= sources
            ):
                problem = "native class union gap lacks exact copy/arm proof"
            else:
                closed = book.page(SOURCE_CLOSED_CLASS_ALTERNATIVES).cell(
                    fact.closed_cell.row, fact.closed_cell.column)
                operands = book.page(PROCESS_GRAPH_OPERANDS).cell(
                    fact.operand_cell.row, fact.operand_cell.column)
                if not isinstance(closed, SourceClosedClassAlternatives):
                    problem = "native class union gap has no closed class source"
                elif not any(
                    source.page == SOURCE_CLOSED_CLASS_ALTERNATIVES
                    and source.row[1] == row[1]
                    for source, _stage in book.edges_into(fact.closed_cell)
                ):
                    problem = "native class union gap has no source-to-copy edge"
                elif tuple((arm, role) for arm, role in operands
                           if role in {"body", "orelse"}) != (
                               (fact.arm_value_cells[0], "body"),
                               (fact.arm_value_cells[1], "orelse")):
                    problem = "native class union gap disagrees with copied operands"
                elif fact.alternatives != tuple(identity for identity, _ in
                                                closed.alternatives):
                    problem = "native class union gap changed source classes"
                else:
                    for original, arm, class_cell in zip(
                        closed.arm_value_cells, fact.arm_value_cells,
                        closed.arm_class_cells, strict=True,
                    ):
                        arm_sources = {source for source, _stage in
                                       book.edges_into(arm)}
                        class_fact = book.page(SOURCE_VALUE_CLASS).cell(
                            class_cell.row, class_cell.column)
                        if (original not in arm_sources
                                or class_fact[2] != "source class construction"
                                or book.latest_ref(VALUE_TYPE_ABI,
                                                   (row[0], arm.row[1]))
                                is not None):
                            problem = "native class union gap arm has another layout"
                            break
            if problem is not None:
                found.append(Finding(
                    "source-class-union-native-gap-disagreement",
                    str(row[0]), int(row[1]), problem,
                ))
        return found

    @staticmethod
    def _source_closed_class_alternative_findings(module: Any) -> list[Finding]:
        """Check each semantic class arm against exact control and operands."""

        from .concordance_declarations import (
            CANONICAL_VALUE, PROCESS_GRAPH_OPERANDS, SOURCE_CLOSED_CLASS_ALTERNATIVES,
            SOURCE_SPAN, SOURCE_VALUE_CLASS, SourceClosedClassAlternatives,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        page = book.page(SOURCE_CLOSED_CLASS_ALTERNATIVES)
        found: list[Finding] = []
        for row in page.rows():
            cell = book.latest_ref(SOURCE_CLOSED_CLASS_ALTERNATIVES, row)
            fact = page.cell(cell.row, cell.column)
            sources = {source for source, _stage in book.edges_into(cell)}
            problem = None
            if not isinstance(fact, SourceClosedClassAlternatives):
                problem = "invalid closed class alternatives"
            elif not (
                len(fact.alternatives) == len(fact.arm_value_cells)
                == len(fact.arm_class_cells) == 2
                and all(isinstance(arm, Ref) and arm.page == CANONICAL_VALUE
                        for arm in fact.arm_value_cells)
                and all(isinstance(class_cell, Ref)
                        and class_cell.page == SOURCE_VALUE_CLASS
                        for class_cell in fact.arm_class_cells)
                and isinstance(fact.control_cell, Ref)
                and fact.control_cell.page == SOURCE_SPAN
                and isinstance(fact.operand_cell, Ref)
                and fact.operand_cell.page == PROCESS_GRAPH_OPERANDS
                and fact.operand_cell.row == row
                and len(set(fact.alternatives)) == 2
            ):
                problem = "closed class arm/control identities are incomplete"
            elif not {
                fact.control_cell, fact.operand_cell,
                *fact.arm_value_cells, *fact.arm_class_cells,
            } <= sources:
                problem = "closed class proof edges are absent"
            else:
                operands = book.page(PROCESS_GRAPH_OPERANDS).cell(
                    fact.operand_cell.row, fact.operand_cell.column)
                body = tuple(ref for ref, role in operands if role == "body")
                orelse = tuple(ref for ref, role in operands if role == "orelse")
                test = tuple(ref for ref, role in operands if role == "test")
                if (body != (fact.arm_value_cells[0],)
                        or orelse != (fact.arm_value_cells[1],)
                        or len(test) != 1):
                    problem = "closed class arms disagree with operand roles"
                else:
                    for alternative, arm, class_cell in zip(
                        fact.alternatives, fact.arm_value_cells,
                        fact.arm_class_cells, strict=True,
                    ):
                        class_fact = book.page(SOURCE_VALUE_CLASS).cell(
                            class_cell.row, class_cell.column)
                        if (class_cell.row[1] != arm.row[1]
                                or not isinstance(class_fact, tuple)
                                or len(class_fact) < 2
                                or alternative != (class_fact[0], int(class_fact[1]))):
                            problem = "closed class alternative disagrees with arm class"
                            break
                if (problem is None
                        and book.latest_ref(SOURCE_VALUE_CLASS, row) is not None):
                    problem = "closed class merge also claims one global class"
            if problem is not None:
                found.append(Finding(
                    "source-closed-class-alternative-disagreement",
                    str(row[0]), int(row[1]), problem,
                ))
        return found

    @staticmethod
    def _source_closed_aggregate_length_findings(module: Any) -> list[Finding]:
        """Check a closed tuple length view against both immutable arm layouts."""

        from .concordance_declarations import (
            SOURCE_AGGREGATE_LAYOUT, SOURCE_CLOSED_AGGREGATE_LENGTHS,
            SourceAggregateLayout, SourceAggregateMember,
            SourceClosedAggregateLengths,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
        )
        if book is None:
            return []
        page = book.page(SOURCE_CLOSED_AGGREGATE_LENGTHS)
        found = []
        for row in page.rows():
            cell = book.latest_ref(SOURCE_CLOSED_AGGREGATE_LENGTHS, row)
            fact = page.cell(cell.row, cell.column)
            history = tuple(entry for _column, entry in page.history(row))
            sources = {source for source, _stage in book.edges_into(cell)}
            problem = None
            if len(set(map(repr, history))) != 1:
                problem = "closed length domain changed after publication"
            elif not isinstance(fact, SourceClosedAggregateLengths):
                problem = f"invalid closed length fact {fact!r}"
            elif not (
                fact.aggregate_kind == "tuple"
                and len(fact.lengths) == len(fact.arm_layout_cells) == 2
                and all(isinstance(length, int) and length >= 0
                        for length in fact.lengths)
                and len(set(fact.lengths)) == 2
                and isinstance(fact.conditional_cell, Ref)
                and all(isinstance(arm, Ref)
                        and arm.page == SOURCE_AGGREGATE_LAYOUT
                        for arm in fact.arm_layout_cells)
            ):
                problem = "closed tuple arms/control are incomplete"
            elif not {
                fact.conditional_cell, *fact.arm_layout_cells,
            } <= sources:
                problem = "closed tuple arm/control proof edges are absent"
            else:
                for expected, arm_cell in zip(
                    fact.lengths, fact.arm_layout_cells, strict=True,
                ):
                    arm = book.page(SOURCE_AGGREGATE_LAYOUT).cell(
                        arm_cell.row, arm_cell.column,
                    )
                    if not (
                        isinstance(arm, SourceAggregateLayout)
                        and arm.aggregate_kind == "tuple"
                        and len(arm.members) == expected
                        and all(
                            isinstance(member, SourceAggregateMember)
                            and isinstance(member.value_cell, Ref)
                            and member.source_cells
                            for member in arm.members
                        )
                    ):
                        problem = "closed length disagrees with an arm's members"
                        break
            if problem is not None:
                found.append(Finding(
                    "source-closed-aggregate-length-disagreement",
                    str(row[0]), int(row[1]), problem,
                ))
        return found

    @staticmethod
    def _source_tuple_key_native_gap_findings(module: Any) -> list[Finding]:
        """Check the return, call, formal and keyed-store refusal trail."""

        from .concordance_declarations import (
            SOURCE_CALL_ACTUAL_FORMAL, SOURCE_CALL_CLOSED_TUPLE_LENGTHS,
            SOURCE_CALL_RESULT_IDENTITY, SOURCE_CLOSED_AGGREGATE_LENGTHS,
            SOURCE_AGGREGATE_LAYOUT,
            SOURCE_FUNCTION_REACHABILITY, SOURCE_TUPLE_KEY_NATIVE_GAP,
            SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS,
            PROCESS_GRAPH_OPERANDS, SEMANTIC_OUTPUT, CANONICAL_VALUE,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        found: list[Finding] = []
        def existing(cell: Any) -> bool:
            return isinstance(cell, Ref) and (
                (cell.row, cell.column) in book.page(cell.page).cells
                or book.page(cell.page).is_virtual(cell.row, cell.column)
            )

        transport_page = book.page(SOURCE_CALL_CLOSED_TUPLE_LENGTHS)
        for row in transport_page.rows():
            cell = book.latest_ref(SOURCE_CALL_CLOSED_TUPLE_LENGTHS, row)
            fact = transport_page.latest(row)
            problem = None
            if not (isinstance(row, tuple) and len(row) == 4
                    and isinstance(fact, tuple) and len(fact) == 5):
                problem = "invalid call tuple-domain row"
            else:
                output, returned, domain, reached, result = fact
                output_fact = (
                    None if not existing(output) or
                    output.page != SEMANTIC_OUTPUT else
                    book.page(SEMANTIC_OUTPUT).cell(
                        output.row, output.column,
                    )
                )
                if not (
                    output_fact is not None
                    and isinstance(output_fact, tuple)
                    and len(output_fact) == 2
                    and int(output_fact[1]) == int(row[3])
                    and output.row[1] == 0
                    and book.latest_ref(SEMANTIC_OUTPUT, output.row) == output
                    and existing(returned) and returned.row[1] == row[3]
                    and existing(domain)
                    and domain.page == SOURCE_CLOSED_AGGREGATE_LENGTHS
                    and domain.row[1] == row[3]
                    and output.row[0] == domain.row[0]
                    and returned in {
                        source for source, _stage in book.edges_into(output)
                    }
                    and existing(reached)
                    and reached.page == SOURCE_FUNCTION_REACHABILITY
                    and reached.row == row[:3]
                    and existing(result)
                    and result.page == SOURCE_CALL_RESULT_IDENTITY
                    and result.row == row[:2]
                    and book.page(SOURCE_CALL_RESULT_IDENTITY).cell(
                        result.row, result.column,
                    )[1] == "builtins.tuple"
                    and {output, returned, domain, reached, result} <= {
                        source for source, _stage in book.edges_into(cell)
                    }
                ):
                    problem = "call tuple domain lacks exact output/proof edge"
            if problem:
                found.append(Finding(
                    "source-call-closed-tuple-length-disagreement",
                    str(row[0]), int(row[1]), problem,
                ))

        formal_page = book.page(SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS)
        for row in formal_page.rows():
            cell = book.latest_ref(SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS, row)
            fact = formal_page.latest(row)
            problem = None
            if not (isinstance(row, tuple) and len(row) == 4
                    and isinstance(fact, tuple) and len(fact) == 4):
                problem = "invalid formal tuple-domain row"
            else:
                binding, transport, domain, output = fact
                if not (
                    existing(binding) and binding.page == SOURCE_CALL_ACTUAL_FORMAL
                    and existing(transport)
                    and transport.page == SOURCE_CALL_CLOSED_TUPLE_LENGTHS
                    and existing(domain)
                    and domain.page == SOURCE_CLOSED_AGGREGATE_LENGTHS
                    and existing(output) and output.page == SEMANTIC_OUTPUT
                ):
                    problem = "formal tuple-domain proof cell is absent"
                else:
                    binding_fact = book.page(SOURCE_CALL_ACTUAL_FORMAL).cell(
                        binding.row, binding.column,
                    )
                    transport_fact = book.page(
                        SOURCE_CALL_CLOSED_TUPLE_LENGTHS
                    ).cell(transport.row, transport.column)
                    actual = binding_fact[0]
                    formal_value = book.latest_ref(
                        CANONICAL_VALUE, (row[0], int(row[1])),
                    )
                    if not (
                        binding.row[0] == row[2]
                        and int(binding.row[1]) == int(row[3])
                        and int(binding.row[3]) == int(row[1])
                        and binding_fact[4].row[0] == row[0]
                        and transport.row[0] == row[2]
                        and int(transport.row[1]) == int(actual.row[1])
                        and transport_fact[0] == output
                        and transport_fact[2] == domain
                        and formal_value is not None
                        and {binding, transport, domain, output,
                             formal_value, actual} <= {
                            source for source, _stage in book.edges_into(cell)
                        }
                    ):
                        problem = "formal tuple domain lacks exact call edges"
            if problem:
                found.append(Finding(
                    "source-formal-closed-tuple-length-disagreement",
                    str(row[0]), int(row[1]), problem,
                ))

        gap_page = book.page(SOURCE_TUPLE_KEY_NATIVE_GAP)
        for row in gap_page.rows():
            cell = book.latest_ref(SOURCE_TUPLE_KEY_NATIVE_GAP, row)
            fact = gap_page.latest(row)
            problem = None
            if not (isinstance(row, tuple) and len(row) == 2
                    and isinstance(fact, tuple) and len(fact) == 6):
                problem = "invalid tuple-key refusal row"
            else:
                formal_id, lengths, formal, store, operands, operation = fact
                if not (
                    existing(formal)
                    and formal.page == SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS
                    and existing(store)
                    and existing(operands)
                    and operands.page == PROCESS_GRAPH_OPERANDS
                ):
                    problem = "tuple-key refusal proof cell is absent"
                else:
                    binding, transport, domain, output = book.page(
                        SOURCE_FORMAL_CLOSED_TUPLE_LENGTHS
                    ).cell(formal.row, formal.column)
                    lengths_fact = book.page(SOURCE_CLOSED_AGGREGATE_LENGTHS).cell(
                        domain.row, domain.column,
                    )
                    graph_operands = book.page(PROCESS_GRAPH_OPERANDS).cell(
                        operands.row, operands.column,
                    )
                    if not (
                        formal.row[1] == formal_id
                        and binding.row[2] == row[0]
                        and int(binding.row[3]) == int(formal_id)
                        and store.row[1] == row[1]
                        and operands.page == PROCESS_GRAPH_OPERANDS
                        and operands.row[1] == row[1]
                        and any(
                            role == "index"
                            and isinstance(source, Ref)
                            and source.row[1] == formal_id
                            for source, role in graph_operands
                        )
                        and tuple(lengths) == tuple(lengths_fact.lengths)
                        and len(set(lengths)) > 1
                        and operation == "runtime_tuple_to_key_columns"
                        and book.latest_ref(
                            SOURCE_AGGREGATE_LAYOUT,
                            (formal.row[0], int(formal_id)),
                        ) is None
                        and {formal, store, operands} <= {
                            source for source, _stage in book.edges_into(cell)
                        }
                    ):
                        problem = "tuple-key refusal does not join exact calls and store"
            if problem:
                found.append(Finding(
                    "source-tuple-key-native-gap-disagreement",
                    str(row[0]), int(row[1]), problem,
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
                    # A heap sequence's operand is its pointer cell: its (1,)
                    # shape is the cell, and every writer reduces over the
                    # live length (E22a), so it states no static extent.
                    if (accounting.get("sequence_id") is not None
                            and accounting.get("heap_element_dtype") is None
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
    def _unreturned_output_findings(module: Any) -> list[Finding]:
        """Report a function that returns a source value from an empty Ret.

        Program: ``xs = [3, n]; xs.append(2); return xs[2]`` (corpus c310).
        The function's ``source_output_value_ids`` named the returned read,
        but its lowering never asked for that read, so the only ``Ret``
        carried no value and the caller's result was empty, with a clean
        audit.  The two records are the source outputs and the ``Ret``.
        """

        found = []
        for name, function in (getattr(module, "functions", {}) or {}).items():
            outputs = tuple(
                (getattr(function, "metadata", {}) or {}).get(
                    "source_output_value_ids"
                ) or ()
            )
            if not outputs:
                continue
            for block in (getattr(function, "blocks", {}) or {}).values():
                for instruction in block.instrs:
                    if instruction.op not in {"Ret", "ret", "Return", "return"}:
                        continue
                    if not instruction.args:
                        found.append(Finding(
                            "source-output-unreturned", name, int(outputs[0]),
                            f"the source returns {len(outputs)} value(s), "
                            "but a Ret in block "
                            f"{getattr(block, 'name', '?')} returns none",
                        ))
                        continue
                    # Program: ``return math.floor(x), int(x)`` (c922).
                    # Wrong record: the region computing ``math.floor`` was
                    # retired, and the Ret carried one value for the
                    # source's two; the root returned a 1-tuple with a
                    # clean audit.  Every source output occupies at least
                    # one Ret slot (a None, a union frame or a forwarded
                    # tuple expands to one or more), so a Ret carrying
                    # FEWER values than the source returns lost one.  More
                    # is the ABI's expansion (c306, c699, c807) and is not
                    # read here.
                    if len(instruction.args) < len(outputs) and not (
                        getattr(instruction, "attributes", None) or {}
                    ).get("fault_exit"):
                        returned = {int(value.id) for value in instruction.args}
                        missing = [
                            int(value_id) for value_id in outputs
                            if int(value_id) not in returned
                        ]
                        found.append(Finding(
                            "source-output-arity-short", name,
                            int((missing or outputs)[0]),
                            f"the source returns {len(outputs)} value(s), "
                            "but a Ret in block "
                            f"{getattr(block, 'name', '?')} returns "
                            f"{len(instruction.args)}; source output(s) "
                            f"{missing} are not among them",
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
    def _source_sequence_capacity_bound_findings(module: Any) -> list[Finding]:
        """Validate the exact bytes base, slice selector and bound causes."""

        from .concordance_declarations import (
            CANONICAL_VALUE, PROCESS_GRAPH_OPERANDS,
            SOURCE_SEQUENCE_CAPACITY_BOUND, SOURCE_VALUE_CLASS,
            SourceSequenceCapacityBound,
        )

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        page = book.page(SOURCE_SEQUENCE_CAPACITY_BOUND)
        found = []
        for row in page.rows():
            cell = book.latest_ref(SOURCE_SEQUENCE_CAPACITY_BOUND, row)
            fact = None if cell is None else page.cell(cell.row, cell.column)
            valid = isinstance(fact, SourceSequenceCapacityBound)
            if valid:
                destination = fact.destination_cell
                source = fact.source_cell
                operands = fact.operand_cell
                source_class = fact.source_class_cell
                causes = {parent for parent, _stage in book.edges_into(cell)}
                valid = (
                    destination.page == CANONICAL_VALUE
                    and destination.row == row
                    and source.page == CANONICAL_VALUE
                    and operands.page == PROCESS_GRAPH_OPERANDS
                    and operands.row == row
                    and source_class.page == SOURCE_VALUE_CLASS
                    and {destination, source, operands, source_class} <= causes
                    and source in {
                        parent for parent, _stage in book.edges_into(source_class)
                    }
                    and destination in {
                        parent for parent, _stage in book.edges_into(operands)
                    }
                )
            if valid:
                operand_fact = book.page(PROCESS_GRAPH_OPERANDS).cell(
                    operands.row, operands.column,
                )
                selector = tuple(
                    parent for parent, role in operand_fact
                    if role == "index" and parent.page == CANONICAL_VALUE
                )
                source_fact = book.page(SOURCE_VALUE_CLASS).cell(
                    source_class.row, source_class.column,
                )
                destination_fact = book.page(CANONICAL_VALUE).cell(
                    destination.row, destination.column,
                )
                selector_fact = (
                    book.page(CANONICAL_VALUE).cell(
                        selector[0].row, selector[0].column,
                    ) if len(selector) == 1 else ()
                )
                valid = (
                    (source, "base") in operand_fact
                    and len(selector) == 1
                    and isinstance(source_fact, tuple)
                    and source_fact[0] == "builtins.bytes"
                    and isinstance(destination_fact, tuple)
                    and isinstance(selector_fact, tuple)
                    and any(
                        destination_fact[index:index + 2]
                        == ("field:node.ast", "value:Subscript")
                        for index in range(len(destination_fact) - 1)
                    )
                    and any(
                        selector_fact[index:index + 2]
                        == ("field:node.ast", "value:Slice")
                        for index in range(len(selector_fact) - 1)
                    )
                )
            if not valid:
                found.append(Finding(
                    "source-sequence-capacity-bound",
                    str(row[0]), int(row[1]),
                    "bytes slice capacity lacks exact source, selector, "
                    "operand or class provenance",
                ))
        return found

    @staticmethod
    def _runtime_result_abi_findings(module: Any) -> list[Finding]:
        """Check a native result ABI against the exact source/SSA bound."""

        import hashlib

        from .concordance_declarations import (
            ArtifactFact, ArtifactPart, Backend, EMISSION_ARTIFACT,
        )
        from .ssa_storage_requirements import public_sequence_capacity_sources

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")
        if book is None:
            return []
        page = book.page(EMISSION_ARTIFACT)
        found = []
        for row in page.rows():
            if (
                not isinstance(row, tuple) or len(row) != 3
                or row[2] is not ArtifactPart.RUNTIME_ABI
            ):
                continue
            cell = book.latest_ref(EMISSION_ARTIFACT, row)
            fact = None if cell is None else page.cell(cell.row, cell.column)
            entry, backend, _part = row
            issue = lambda detail: found.append(Finding(
                "runtime-result-abi", str(entry), None, detail,
            ))
            if (
                backend is not Backend.C_MODULE
                or not isinstance(fact, ArtifactFact)
                or len(fact.location) != 1
                or not isinstance(fact.location[0], dict)
            ):
                issue("runtime ABI lacks an exact C entry and description")
                continue
            description = fact.location[0]
            payload = repr(description).encode("utf-8")
            if (
                fact.sha256 != hashlib.sha256(payload).hexdigest()
                or fact.byte_length != len(payload)
                or description.get("entry") != str(entry)
                or description.get("backend") != "c"
                or description.get("schema") != "turing-runtime-abi-v2"
            ):
                issue("runtime ABI description hash, entry or schema disagrees")
                continue
            order = tuple(
                int(buffer["value_id"])
                for buffer in description.get("buffers", ())
            )
            if len(order) != len(set(order)):
                issue("runtime ABI repeats one public buffer identity")
                continue
            buffer_index = {value_id: index for index, value_id in enumerate(order)}
            causes = {parent for parent, _stage in book.edges_into(cell)}
            order_cell = book.latest_ref(EMISSION_ARTIFACT, (
                str(entry), backend, ArtifactPart.BUFFER_ORDER,
            ))
            api_cell = book.latest_ref(EMISSION_ARTIFACT, (
                str(entry), backend, ArtifactPart.API_CONTRACT,
            ))
            # `entry_name` is an exported symbol and may differ from the
            # source function key. The exact descriptor cells cited by the
            # artifact select one owning root; a name coincidence does not.
            owners = []
            for function_name in module.functions:
                candidates = public_sequence_capacity_sources(
                    module, str(function_name),
                )
                if candidates and all(
                    proof in causes
                    for relation in candidates
                    for proof in (
                        relation.bound_cell,
                        relation.output_descriptor_cell,
                        relation.input_descriptor_cell,
                    )
                ):
                    owners.append((function_name, candidates))
            if len(owners) != 1:
                issue("runtime ABI has no unique descriptor-owned source root")
                continue
            _owner, relations = owners[0]
            expected_capacities = {}
            expected_lengths = {}
            for relation in relations:
                output = relation.output_data_id
                if output not in buffer_index or relation.input_data_id not in buffer_index:
                    continue
                if relation.output_length_id not in buffer_index:
                    continue
                expected_capacities[output] = buffer_index[relation.input_data_id]
                expected_lengths[output] = buffer_index[relation.output_length_id]
            actual_capacities = {
                int(buffer["value_id"]): buffer["capacity_from"]
                for buffer in description["buffers"]
                if "capacity_from" in buffer
            }
            actual_lengths = {
                int(buffer["value_id"]): buffer["logical_length_from"]
                for buffer in description["buffers"]
                if "logical_length_from" in buffer
            }
            actual_shapes = {
                int(buffer["value_id"]): buffer["logical_shape_from"]
                for buffer in description["buffers"]
                if "logical_shape_from" in buffer
            }
            required = {
                order_cell, api_cell,
                *(
                    proof
                    for relation in relations
                    for proof in (
                        relation.bound_cell,
                        relation.output_descriptor_cell,
                        relation.input_descriptor_cell,
                    )
                ),
            }
            if (
                None in required or not required <= causes
                or actual_capacities != expected_capacities
                or actual_lengths != expected_lengths
                or actual_shapes
                or not relations
                or page.cell(order_cell.row, order_cell.column).location != order
                or order_cell not in {
                    parent for parent, _stage in book.edges_into(api_cell)
                }
            ):
                issue("runtime ABI capacity, logical length or proof cells disagree")
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
        # Step 7 keys one row per (callee, caller, formal, callsite); older
        # rows key (callee, formal, callsite), or (callee, formal, "binding")
        # with callsite in the column. Caller-local ids and callsite ids are
        # not identities outside their caller.
        formals: dict[tuple, dict[Any, dict[str, list]]] = {}
        for row in page.rows():
            if not isinstance(row, tuple):
                continue
            if len(row) == 4:
                caller_symbol, formal_id = str(row[1]), int(row[2])
            elif len(row) == 3:
                caller_symbol, formal_id = None, int(row[1])
            else:
                continue
            kinds_by_source = formals.setdefault(
                (row[0], caller_symbol, formal_id), {},
            )
            for callsite, fact in argument_binding_history(
                page, row[0], formal_id, rows=(row,),
            ):
                if not (isinstance(fact, tuple) and len(fact) == 2):
                    continue
                kind, source = fact
                if not isinstance(source, int):
                    continue
                kinds_by_source.setdefault(int(source), {}).setdefault(
                    str(kind), []
                ).append(callsite)
        for (
            function_name, caller_symbol, value_id,
        ), kinds_by_source in formals.items():
            for source, by_kind in sorted(kinds_by_source.items()):
                if len(by_kind) < 2:
                    continue
                resolution_row = (
                    (str(function_name), caller_symbol, int(value_id), int(source))
                    if caller_symbol is not None
                    else (str(function_name), int(value_id), int(source))
                )
                resolution = book.page(
                    "argument_binding_resolution"
                ).latest(resolution_row)
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
        from .concordance_declarations import CALL_RECORD, SEQUENCE_DESCRIPTOR, RESULT_STORAGE_BINDING
        from .ssa_record_return_state import map_sequence_descriptor, map_child_pool
        from .ssa_record_return_state import ssa_value_identity_cell
        from .concordance_declarations import (
            SSA_VALUE, CELL_SET, CANONICAL_VALUE, INGESTION_VALUE, VALUE_TYPE_ABI,
            SOURCE_AGGREGATE_LAYOUT, SOURCE_VALUE_CLASS,
        )
        from ..transmogrifier.dtype_layout import resolve

        book = dict(getattr(module, "metadata", {}) or {}).get("identity_book")

        def numeric_dtype(dtype):
            layout = resolve(dtype)
            return (layout.name if layout is not None
                    and layout.kind in {"bool", "int", "uint", "float"} else None)

        def sequence_source_type(function, value_id, owner):
            """Read this SSA identity's lexical ABI, never a same-number row.

            validate_batch_envelope's static iterable names source2. Its
            adopted SSA row was int64, but its exact lexical ABI was a bool
            span. The old audit checked each page independently and missed
            the int64 Load through that iterable's concrete bool column.
            """
            start = ssa_value_identity_cell(function, int(value_id), book=owner)
            if start is None:
                return None, None, (), ()
            pending, seen, abis, lexical_abis, products = [start], set(), set(), set(), set()
            while pending:
                cell = pending.pop()
                if cell in seen:
                    continue
                seen.add(cell)
                if cell.page in {SSA_VALUE, CANONICAL_VALUE, INGESTION_VALUE}:
                    abi = owner.latest_ref(VALUE_TYPE_ABI, cell.row)
                    if abi is not None:
                        (lexical_abis if cell.page in {CANONICAL_VALUE, INGESTION_VALUE} else abis).add(abi)
                        if owner.page(SOURCE_AGGREGATE_LAYOUT).latest(cell.row) is not None:
                            products.add(abi)
                if cell.page == VALUE_TYPE_ABI:
                    abis.add(cell)
                    continue
                if cell.page in {CANONICAL_VALUE, INGESTION_VALUE}:
                    continue
                for parent, _stage in owner.edges_into(cell):
                    if (parent.page in {CELL_SET, CANONICAL_VALUE, INGESTION_VALUE, VALUE_TYPE_ABI}
                            or (parent.page == SSA_VALUE and parent.row == start.row)):
                        pending.append(parent)
            dtypes, evidence = set(), []
            for abi in lexical_abis or abis:
                # source_structure defines an aggregate as its member
                # product before considering its outer ABI carrier. That
                # carrier's integer type is not a row-element declaration.
                if abi in products:
                    continue
                fact = owner.page(VALUE_TYPE_ABI).cell(abi.row, abi.column)
                arms = tuple(getattr(fact, "physical_alternatives", ()))
                if not arms or any(arm.storage not in {"span", "sequence"} or arm.rank < 1 for arm in arms):
                    continue
                types = tuple(numeric_dtype(arm.dtype) for arm in arms)
                if all(dtype is not None for dtype in types):
                    dtypes.update(types)
                    evidence.append(abi)
            fact = owner.page(SSA_VALUE).cell(start.row, start.column)
            # Explicit physical views (including packed storage and a
            # recorded input conversion) own native representation. A
            # provisional resident dtype alone cannot override lexical ABI.
            physical = numeric_dtype(getattr(fact, "physical_dtype", None))
            return (next(iter(dtypes)) if len(dtypes) == 1 else None,
                    physical, tuple(evidence), tuple(lexical_abis or abis))

        def static_member_class(owner, abi_refs):
            claims = set()
            for abi_ref in abi_refs:
                abi = owner.page(VALUE_TYPE_ABI).cell(abi_ref.row, abi_ref.column)
                for arm in getattr(abi, "physical_alternatives", ()):
                    classes = tuple(cell for cell in arm.source_cells if cell.page == SOURCE_VALUE_CLASS)
                    for source in arm.source_cells:
                        if source.page not in {CANONICAL_VALUE, INGESTION_VALUE}:
                            continue
                        layout = owner.page(SOURCE_AGGREGATE_LAYOUT).latest(source.row)
                        if layout is None or layout.aggregate_kind not in {"list", "tuple"} or not layout.members:
                            continue
                        members = []
                        for member in layout.members:
                            anchors = {_canonical_identity_base(owner, cell)
                                       for cell in (member.value_cell, *member.source_cells)
                                       if cell.page in {CANONICAL_VALUE, INGESTION_VALUE}}
                            identities = set()
                            for cls in classes:
                                pending, seen = [cls], set()
                                while pending:
                                    cell = pending.pop()
                                    if cell in seen:
                                        continue
                                    seen.add(cell)
                                    for parent, _stage in owner.edges_into(cell):
                                        if parent.page in {CANONICAL_VALUE, INGESTION_VALUE}:
                                            if _canonical_identity_base(owner, parent) in anchors:
                                                fact = owner.page(cls.page).cell(cls.row, cls.column)
                                                identities.add(str(fact[0]))
                                        else:
                                            pending.append(parent)
                            if len(identities) != 1:
                                break
                            members.append(next(iter(identities)))
                        if len(members) == len(layout.members) and len(set(members)) == 1:
                            claims.add(members[0])
            return next(iter(claims)) if len(claims) == 1 else None

        def column_and_load_findings(function_name, table):
            function = (getattr(module, "functions", {}) or {}).get(function_name)
            if function is None:
                return []
            owner = table.book
            issues, column_types = [], {}
            for descriptor in table.sequences.values():
                for column, dtype in zip(descriptor.column_value_ids, descriptor.column_dtypes):
                    declared = numeric_dtype(dtype)
                    if declared is not None:
                        column_types.setdefault(int(column), set()).add(declared)
                # A scalar-row ABI describes one payload column. Multi-column
                # records/unions retain their existing per-member audits.
                if len(descriptor.column_value_ids) != 1:
                    continue
                column = int(descriptor.column_value_ids[0])
                lexical, physical, sources, _abis = sequence_source_type(function, column, owner)
                if not sources and int(descriptor.sequence_id) != column:
                    lexical, physical, sources, _abis = sequence_source_type(function, descriptor.sequence_id, owner)
                expected = {dtype for dtype in (lexical, physical) if dtype is not None}
                declared = numeric_dtype(descriptor.column_dtypes[0]) if descriptor.column_dtypes else None
                if sources and expected and declared is not None and declared not in expected:
                    issues.append(Finding(
                        "sequence-column-abi-disagreement", str(function_name), int(descriptor.sequence_id),
                        f"column {column} descriptor={declared}, exact incoming ABI/native views={tuple(sorted(expected))}; sources={sources!r}",
                    ))
            definitions = {int(ins.res.id): ins for block in function.blocks.values()
                           for ins in block.instrs if ins.res is not None}
            for label, block in function.blocks.items():
                for instruction in block.instrs:
                    if instruction.op not in {"Load", "load"} or instruction.res is None or len(instruction.args) != 1:
                        continue
                    address = definitions.get(int(instruction.args[0].id))
                    if address is None or address.op not in {"GetElementPtr", "getelementptr"} or not address.args:
                        continue
                    base = address.args[0]
                    producer = definitions.get(int(base.id))
                    expected, evidence = None, ()
                    if (producer is not None and producer.op in {"Const", "const"}
                            and producer.attributes.get("binding") == "static_iterable"
                            and producer.attributes.get("source_value_id") is not None):
                        lexical, _source_physical, evidence, abis = sequence_source_type(
                            function, int(producer.attributes["source_value_id"]), owner)
                        materialized = ssa_value_identity_cell(function, int(base.id), book=owner)
                        fact = None if materialized is None else owner.page(SSA_VALUE).cell(materialized.row, materialized.column)
                        expected = numeric_dtype(getattr(fact, "physical_dtype", None)) or lexical
                        kind = "static-iterable-load-type-disagreement"
                        member_class = static_member_class(owner, abis)
                        _dtype, _physical, _sources, result_abis = sequence_source_type(function, instruction.res.id, owner)
                        alternatives = {identity for cell in result_abis
                                        for identity in owner.page(VALUE_TYPE_ABI).cell(cell.row, cell.column).alternatives}
                        materialized = dict(function.metadata.get("native_value_type_abi", {}) or {}).get(int(instruction.res.id))
                        materialized_alternatives = tuple(getattr(materialized, "alternatives", ()))
                        disagreement = (alternatives and member_class not in alternatives) or (
                            materialized_alternatives and member_class not in materialized_alternatives)
                        if member_class is not None and disagreement:
                            issues.append(Finding(
                                "static-iterable-element-class-disagreement", str(function_name), int(instruction.res.id),
                                f"{label}: exact aggregate members are {member_class}, current loop ABI={tuple(sorted(alternatives))}, materialized loop ABI={materialized_alternatives}; source ABI={abis!r}",
                            ))
                    elif len(column_types.get(int(base.id), ())) == 1:
                        expected = next(iter(column_types[int(base.id)]))
                        kind = "sequence-column-load-type-disagreement"
                    else:
                        continue
                    actual = numeric_dtype(instruction.res.dtype)
                    if expected is not None and actual is not None and expected != actual:
                        issues.append(Finding(
                            kind, str(function_name), int(instruction.res.id),
                            f"{label}: Load {actual} from column {int(base.id)} whose exact native element is {expected}; sources={evidence!r}",
                        ))
            return issues

        def returned_frame_findings(function_name, table, sequence_id, descriptor):
            if book is None:
                return []
            current = book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner, int(sequence_id)))
            if current is None:
                return []
            # Revisions retain the original derivation. Follow this view's
            # own descriptor history/cell sets, stopping at the exact source
            # descriptor and call-record authorities rather than following
            # unrelated value ancestry or schema names.
            pending, seen, calls, sources = [current], set(), set(), set()
            while pending:
                cell = pending.pop()
                if cell in seen:
                    continue
                seen.add(cell)
                for parent, _stage in book.edges_into(cell):
                    if parent.page == CALL_RECORD:
                        calls.add(parent.row)
                    elif parent.page == SEQUENCE_DESCRIPTOR:
                        if parent.row == current.row:
                            pending.append(parent)
                        else:
                            sources.add((book, parent.row))
                    elif parent.page.name == "imported_cell":
                        receipt = book.page(parent.page).cell(parent.row, parent.column)
                        if receipt.source.page == SEQUENCE_DESCRIPTOR:
                            sources.add((receipt.owner, receipt.source.row))
                    elif parent.page.name == "cell_set":
                        pending.append(parent)
            issues = []
            for row in calls:
                for record in book.page(CALL_RECORD).latest(row) or ():
                    if record.caller != function_name:
                        continue
                    source_table = (getattr(module, "sequence_tables", {}) or {}).get(record.callee_symbol)
                    if source_table is None:
                        continue
                    for source_id, target_id in record.result_bindings:
                        if (int(target_id) != int(sequence_id)
                                or (source_table.book, (source_table.owner, int(source_id))) not in sources):
                            continue
                        source = source_table.by_id(int(source_id))
                        if source is None:
                            continue
                        frame = {}
                        contradictory = False
                        for formal, kind, actual in record.frame_bindings:
                            if kind in {"caller_value", "caller_alias", "caller_storage"}:
                                prior = frame.setdefault(int(formal), int(actual))
                                contradictory |= prior != int(actual)
                        result_page = book.pages.get(RESULT_STORAGE_BINDING.name)
                        for binding_row in (() if result_page is None else result_page.rows()):
                            if binding_row[:2] != (function_name, int(record.callsite_id)):
                                continue
                            ref = result_page.latest(binding_row)
                            if not isinstance(ref, Ref) or ref.page.name != "ssa_value":
                                continue
                            prior = frame.setdefault(int(binding_row[2]), int(ref.row[1]))
                            contradictory |= prior != int(ref.row[1])
                        try:
                            expected = map_sequence_descriptor(
                                source, frame, sequence_id=int(sequence_id),
                                child_table_pool=map_child_pool(source.child_table_pool, frame),
                            )
                        except KeyError as error:
                            issues.append(Finding(
                                "returned-sequence-frame-disagreement", str(function_name), int(sequence_id),
                                f"callsite {record.callsite_id} has no binding for sequence member {error}",
                            ))
                            continue
                        if contradictory or descriptor != expected:
                            issues.append(Finding(
                                "returned-sequence-frame-disagreement", str(function_name), int(sequence_id),
                                f"callsite {record.callsite_id} sequence view differs from its exact callee frame: "
                                f"observed={descriptor!r}; expected={expected!r}",
                            ))
            return issues

        found: list[Finding] = []
        for function_name, table in (
            getattr(module, "sequence_tables", {}) or {}
        ).items():
            found.extend(column_and_load_findings(function_name, table))
            for sequence_id, descriptor in sorted(
                getattr(table, "sequences", {}).items()
            ):
                found.extend(returned_frame_findings(function_name, table, sequence_id, descriptor))
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

        frames_by_slot: dict[int, set] | None = None

        def call_frames_of(value_id):
            """The call frames of this caller whose storage ``value_id`` is,
            read from the frame linker's rows."""

            nonlocal frames_by_slot
            if frames_by_slot is None:
                from .concordance_declarations import (
                    ARGUMENT_BINDING, FRAME_LEASE,
                )

                frames_by_slot = defaultdict(set)
                # The caller's rows are in the book that owns it: an
                # accepted dependency component's, else the module's.
                component = (
                    function.metadata or {}
                ).get("accepted_ssa_component") or {}
                book = (
                    component.get("identity_book")
                    or (getattr(module, "metadata", {}) or {}).get(
                        "identity_book"
                    )
                    or current_identity_book()
                )
                lease_page = book.pages.get(FRAME_LEASE.name)
                for row in () if lease_page is None else lease_page.rows():
                    fact = lease_page.latest(row)
                    if (
                        str(row[0]) == str(name)
                        and isinstance(fact, tuple) and len(fact) == 3
                    ):
                        frames_by_slot[int(row[1])].add(
                            (str(fact[1]), fact[0])
                        )
                binding_page = book.pages.get(ARGUMENT_BINDING.name)
                for row in () if binding_page is None else binding_page.rows():
                    if len(row) != 4 or str(row[1]) != str(name):
                        continue
                    fact = binding_page.latest(row)
                    if (
                        isinstance(fact, tuple) and len(fact) == 2
                        and fact[0] == "caller_storage"
                        and isinstance(fact[1], int)
                    ):
                        frames_by_slot[int(fact[1])].add(
                            (str(row[0]), row[3])
                        )
            return frozenset(frames_by_slot.get(int(value_id), ()))

        def sequence_roles_of(symbol, value_id):
            table = (getattr(module, "sequence_tables", {}) or {}).get(symbol)
            return frozenset(
                role for _sequence_id, role in (
                    () if table is None else table.member_claims(value_id)
                )
            )

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
                    formal_roles = sequence_roles_of(callee_name, int(callee_id))
                    lease_frames = call_frames_of(argument_id)
                    owned = []
                    for owner in caller_by_key.get(key, ()):
                        if owner == argument_id:
                            continue
                        # Program: ``env, t = batch(n, 2)`` in the root
                        # and again in ``one(n)``, which the root calls.
                        # Wrong record: the root's storage for its own
                        # ``batch`` frame and the lease it hands ``one``
                        # for ``one``'s inner ``batch`` frame share the
                        # compiler-frame key, and the lease was reported
                        # as duplicate storage.  Rule: the frame linker's
                        # records name the call frame each slot is storage
                        # of -- ``frame_lease_link`` for a lease,
                        # ``argument_binding`` (``caller_storage``) for the
                        # caller's own frame storage -- and slots of two
                        # different call frames are two instances; a slot
                        # no record names keeps the check.
                        owner_frames = call_frames_of(owner)
                        if (
                            owner_frames and lease_frames
                            and owner_frames.isdisjoint(lease_frames)
                        ):
                            continue
                        owner_roles = sequence_roles_of(name, owner)
                        # c631: the callee's operational status lease is
                        # not the caller's column/length/capacity storage.
                        # The same field identity alone erased those roles
                        # and falsely reported a duplicate. Compare the
                        # descriptor-owned roles when both endpoints state
                        # them; absent role evidence keeps the old check.
                        if (
                            formal_roles and owner_roles
                            and formal_roles.isdisjoint(owner_roles)
                        ):
                            continue
                        owned.append(owner)
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
            # Program: ``return 0, statuses, tokens`` bound as ``env,
            # statuses, tokens = batch(n, 2)``.  Wrong record: the owner was
            # the callee's frame but the id was the caller's own sequence
            # id, so the caller's sequence 7 (the callee's 4) keyed as the
            # callee's sequence 7 (the caller's 8) and one correct binding
            # was reported as duplicate storage.  Rule: the id is read in
            # the owner's namespace -- the frame's own sequence id that
            # the accounting records beside its owner.
            frame_sequence_id = accounting.get("compiler_frame_sequence_id")
            if owner != function and frame_sequence_id is not None:
                sequence_id = int(frame_sequence_id)
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
        predecessors: dict[str, set[str]] = defaultdict(set)
        for predecessor_name, predecessor in function.blocks.items():
            for successor in predecessor.successors:
                predecessors[str(successor)].add(str(predecessor_name))
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is not None:
                    definition_sites[int(instruction.res.id)].append(
                        (block_name, index)
                    )
        definitions = {
            int(instruction.res.id): instruction
            for block in function.blocks.values() for instruction in block.instrs
            if instruction.res is not None
        }
        selection_validities = {
            int(instruction.attributes["source_value_id"]): int(instruction.res.id)
            for instruction in definitions.values()
            if instruction.attributes.get("binding") == "selected_child_validity"
        }

        def implies_guard(condition_id: int, guard_id: int, seen=None) -> bool:
            if condition_id == guard_id:
                return True
            visited = set() if seen is None else seen
            if condition_id in visited:
                return False
            visited.add(condition_id)
            producer = definitions.get(condition_id)
            # A true conjunction entails each of its explicit Boolean inputs.
            return bool(producer is not None and producer.op == "LAnd"
                        and str(producer.res.dtype) == "bool" and any(
                implies_guard(int(argument.id), guard_id, visited)
                for argument in producer.args
            ))

        def guarded_by_true_edge(block_name: str, guard_id: int) -> bool:
            owner_dominators = dominators.get(block_name, set())
            for candidate in owner_dominators:
                for branch in function.blocks[candidate].instrs:
                    if branch.op != "CondBr" or not branch.args:
                        continue
                    target = str(branch.attributes.get("true_target"))
                    if target in owner_dominators and implies_guard(int(branch.args[0].id), guard_id):
                        return True
            return False
        for block_name, block in function.blocks.items():
            if block_name not in dominators:
                continue
            for index, instruction in enumerate(block.instrs):
                guard_id = instruction.attributes.get("child_selection_guard_value_id")
                if guard_id is not None:
                    producer = definitions.get(int(guard_id))
                    guard_value = producer.res if producer is not None else formals.get(int(guard_id))
                    if guard_value is None or str(guard_value.dtype) != "bool":
                        found.append(Finding("child-selection-not-guarded", name,
                            None if instruction.res is None else int(instruction.res.id),
                            f"{block_name}#{index} guard {int(guard_id)} is not an existing Boolean SSA value"))
                if instruction.attributes.get("binding") == "selected_child_row_view":
                    source_id = int(instruction.attributes["source_value_id"])
                    expected = selection_validities.get(source_id)
                    if expected is not None and guard_id != expected:
                        found.append(Finding("child-selection-not-guarded", name,
                            int(instruction.res.id),
                            f"selected view {source_id} claims guard {guard_id!r}, but its selection owns {expected}"))
                if guard_id is not None and not guarded_by_true_edge(block_name, int(guard_id)):
                    found.append(Finding("child-selection-not-guarded", name,
                        None if instruction.res is None else int(instruction.res.id),
                        f"{block_name}#{index} {instruction.op} is outside guard {int(guard_id)}'s true edge"))
                if instruction.op in {"Phi", "phi"}:
                    incoming = tuple(map(str, instruction.attributes.get("incoming_blocks") or ()))
                    invalid = tuple(edge for edge in incoming if edge not in predecessors[str(block_name)])
                    if invalid:
                        found.append(Finding(
                            "phi-edge-not-predecessor", name,
                            None if instruction.res is None else int(instruction.res.id),
                            f"{block_name}#{index} Phi names {invalid!r}; actual CFG predecessors are {tuple(sorted(predecessors[str(block_name)]))!r}",
                        ))
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

    def to_failure_mapping(self) -> dict[str, Any]:
        """Expose retained diagnostic artifacts through the product's existing hook."""
        return dict(getattr(self, "compiler_failure_artifacts", {}))


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
    edge_reads: tuple = ()  # (exact historical Ref, incoming Refs)
    dependent_reads: tuple = ()  # (exact Ref, authority Page, outgoing Refs)


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
                current_edges = tuple(
                    (source, tuple(parent for parent, _stage in self.edges_into(source)))
                    for source, _parents in getattr(previous, "edge_reads", ())
                )
                current_dependents = tuple(
                    (source, authority, tuple(dict.fromkeys(
                        child for child, _stage in self.edges_out_of(source)
                        if child.page == authority
                    )))
                    for source, authority, _children in getattr(previous, "dependent_reads", ())
                )
                if (current_reads == previous.reads
                        and current_edges == getattr(previous, "edge_reads", ())
                        and current_dependents == getattr(previous, "dependent_reads", ())
                        and self.latest_ref(page, row) == previous.target):
                    return previous.target
            reads = {}
            edge_reads = {}
            dependent_reads = {}

            def read(source_page, source_row, default=None):
                cell = self.current_ref(source_page, source_row)
                reads[(source_page, source_row)] = cell
                return (default if cell is None else
                        self.page(source_page).cell(cell.row, cell.column))

            def read_ref(source_page, source_row):
                read(source_page, source_row)
                return reads[(source_page, source_row)]

            read.ref = read_ref

            def read_sources(source_page, source_row=None):
                """The current claim's exact incoming identities.

                A concordance recipe must follow the recorded trajectory,
                rather than reconstruct a value from an id or dtype. Reading
                the owning row also tracks a changed generation; the recipe
                reads any selected authority through this same reader.
                """
                if isinstance(source_page, Ref):
                    source = source_page
                    read(source.page, source.row)
                else:
                    source = read_ref(source_page, source_row)
                if source is None:
                    return ()
                parents = tuple(
                    parent for parent, _source_stage in self.edges_into(source)
                )
                edge_reads[source] = parents
                return parents

            read.sources = read_sources

            def read_dependents(source, authority):
                """Exact outgoing claims on a declared authority page.

                Selecting the authority is essential: the derived view's own
                publication must not become one of its inputs merely because
                it also cites the source. No scope or numeric id is rebuilt.
                """
                if not isinstance(source, Ref) or not isinstance(authority, Page):
                    raise ConcordanceRefusal("dependent read requires a source Ref and authority Page")
                read(source.page, source.row)
                children = tuple(dict.fromkeys(
                    child for child, _source_stage in self.edges_out_of(source)
                    if child.page == authority
                ))
                dependent_reads[(source, authority)] = children
                return children

            read.dependents = read_dependents

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
                tuple(edge_reads.items()),
                tuple((source, authority, children)
                      for (source, authority), children in dependent_reads.items()),
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
                ) or any(
                    tuple(parent for parent, _stage in self.edges_into(source)) != parents
                    for source, parents in getattr(snapshot, "edge_reads", ())
                ) or any(
                    tuple(dict.fromkeys(child for child, _stage in self.edges_out_of(source)
                                        if child.page == authority)) != children
                    for source, authority, children in getattr(snapshot, "dependent_reads", ())
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


def concordant_shape_transformation_sources(
    scope: Any, value_id: Any, *, operation: Any | None = None,
    role: Any | None = None,
) -> tuple[tuple[Any, ...], ...]:
    """Return the exact live input edges recorded for one shape value.

    Structural specialization may fold an input expression and retire its
    mutable graph operand position.  The transformation edge remains the
    authoritative correlation between the result and that input identity;
    readers must consult it instead of reconstructing the input from a
    surviving name, position, or source coordinate.
    """

    target_scope = _shape_key(scope)
    operation_name = None if operation is None else str(operation).casefold()
    role_name = None if role is None else str(role).casefold()
    page = current_identity_book().page(SHAPE_EDGE_PAGE)
    return tuple(
        edge_row
        for edge_row in page.scope_rows(target_scope)
        if edge_row[1] == value_id
        and (
            operation_name is None
            or str(edge_row[5]).casefold() == operation_name
        )
        and (
            role_name is None
            or str(edge_row[6]).casefold() == role_name
        )
        and isinstance(page.latest(edge_row), tuple)
        and len(page.latest(edge_row)) == 2
    )


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
    incumbent = page.latest(row)
    if incumbent != fact:
        source_ref = book.latest_ref(SHAPE_STATE_PAGE, (
            _shape_key(scope if source_scope is None else source_scope),
            source_id,
        ))
        try:
            return _post_or_unsourced(
                book, SHAPE_STATE_PAGE, row, fact,
                book.registry.declare_stage(str(reason)),
                tuple(dict.fromkeys((
                    *source_cells,
                    *(() if source_ref is None else (source_ref,)),
                ))),
                SHAPE_SOURCE_NOT_ON_BOOK,
            )
        except ConcordanceRefusal:
            # Program: ``s = f"a@{n}"; return len(s)`` (corpus c57; also
            # c02, c58, c62).  Lane D, D3a.  The descriptor invalidation
            # cone withdrew the glyph helpers' rows a third time, under a
            # different reason, from the same unchanged identity cell, and
            # the book refused the revision.  The caller swallowed every
            # exception, so this looked like a lost withdrawal that left a
            # stale ``proven_shape``.  Observed instead: each refused row's
            # latest fact was already a withdrawal, and no ``proven_shape``
            # row existed.  Rule: a row whose latest fact is already a
            # withdrawal has nothing to withdraw, and the refusal of a
            # second, causeless withdrawal is that fact, not a failure.
            # Every other refusal stays loud.
            if (
                isinstance(incumbent, tuple)
                and incumbent
                and incumbent[0] == "invalidated"
            ):
                return book.latest_ref(SHAPE_STATE_PAGE, row)
            raise
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


def _loop_retained_outer_reads(function: Any, loop_node_id: int) -> set[int]:
    """The preheader values a non-carried binding reads inside loop
    ``loop_node_id`` (the control builder's ``loop_retained_outer_reads``)."""

    return {
        int(value_id)
        for retained_loop, value_id in (
            (getattr(function, "metadata", None) or {}).get(
                "loop_retained_outer_reads", ()
            ) or ()
        )
        if int(retained_loop) == int(loop_node_id)
    }


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
            # Program: ``b = k + 2; c = b; while c < 40: c = c + b``.  Wrong
            # record: ``b``'s read of the preheader value (3) inside the loop
            # was rebound to ``c``'s header Phi, so the body added ``c + c``
            # (64 for 40).  Rule: when the control builder resolved a
            # non-carried binding's in-loop read to the outer value, that
            # outer value is still read in scope and is not rebound.
            retained_outers = _loop_retained_outer_reads(function, loop_node_id)
            for outer, matching_rebinds in rebinds_by_outer.items():
                if len(matching_rebinds) != 1 or outer in retained_outers:
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
                    occurrence_row = (
                        callee_name, int(formal.id), str(caller_name),
                        str(block_name), int(instruction_index), int(position), scope,
                    )
                    occurrence_sources = () if actual_cell is None else (actual_cell,)
                    occurrence_mode = Mode.CONCORD
                    previous = book.latest_ref(occurrence_page, occurrence_row)
                    if previous is not None and incoming_page.latest(occurrence_row) != int(actual.id):
                        # The storage-effects bootstrap's targets span is
                        # rebound to its control-owned loop header Phi after
                        # the first formal/actual census. The transition is
                        # already authoritative; retain this occurrence as
                        # its derived current view, not an immutable old id.
                        from .concordance_declarations import LOOP_RESULT_USE_REBINDING
                        transition_row = (scope, str(block_name), int(instruction_index), int(position))
                        transition_ref = book.latest_ref(LOOP_RESULT_USE_REBINDING, transition_row)
                        transition = book.page(LOOP_RESULT_USE_REBINDING).latest(transition_row)
                        if (actual_cell is not None and actual_cell.page == SSA_VALUE
                                and transition_ref is not None and isinstance(transition, tuple)
                                and len(transition) == 3
                                and transition[0] == incoming_page.latest(occurrence_row)
                                and transition[1] == int(actual.id)):
                            occurrence_mode = Mode.REVISE
                            occurrence_sources = (previous, transition_ref, actual_cell)
                    book.post(
                        occurrence_page, occurrence_row, int(actual.id), stage=FRAME_BINDING,
                        provenance=(Derived(occurrence_sources) if occurrence_sources
                                    else Unsourced(FRAME_SOURCE_CELL_ABSENT)),
                        mode=occurrence_mode,
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
    caller_symbol: Any = None,
) -> tuple[tuple[Any, Any], ...]:
    """Every caller-qualified ``(callsite, fact)`` for one callee formal.

    Current rows key ``(callee, caller, formal, callsite)`` and return
    ``((caller, callsite), fact)``. Older step-7 rows key
    ``(callee, formal, callsite)``; pre-step-7 rows key
    ``(callee, formal, "binding")`` with callsite in the column. ``rows``
    restricts the read to those rows; an ``Unresolved`` fact is preserved.
    """

    callee_symbol = str(callee_symbol)
    formal_id = int(formal_id)
    caller_filter = None if caller_symbol is None else str(caller_symbol)
    if rows is None:
        rows = tuple(
            row for row in page.scope_rows(callee_symbol)
            if (
                len(row) == 4 and int(row[2]) == formal_id
                or len(row) == 3 and int(row[1]) == formal_id
            )
        )
    history: list[tuple[Any, Any]] = []
    for row in rows:
        if not isinstance(row, tuple):
            continue
        if str(row[0]) != callee_symbol:
            continue
        if len(row) == 4:
            if int(row[2]) != formal_id:
                continue
            row_caller = str(row[1])
            if caller_filter is not None and row_caller != caller_filter:
                continue
            callsite = (row_caller, row[3])
        elif len(row) == 3:
            if int(row[1]) != formal_id:
                continue
            row_caller = None
            callsite = None if row[2] == "binding" else row[2]
        else:
            continue
        for column, fact in page.history(row):
            history.append((column if row[2] == "binding" else callsite, fact))
    return tuple(history)


def materializing_binding_kind(
    book: Any, callee_symbol: Any, formal_id: int, source_id: int, kind: Any,
    *, caller_symbol: Any = None,
) -> str:
    """The kind that can materialize one caller slot across its callsites.

    A frame binding's kind is not a label on the slot; it selects which
    machinery may supply the argument.  ``caller_storage`` can restore a slot
    a structural cleanup removed, ``caller_alias`` and ``caller_value``
    cannot.  Each callsite decides the kind from its own private map in its
    own order, so two callsites can name the same slot under different kinds
    and the one that cannot materialize it reports ``missing_<kind>`` and
    refuses the call -- after which the callee's output is never written and
    whatever reads it gets uninitialized memory, with no shortfall anywhere.

    Current rows include caller identity because callsite IDs and local value
    IDs are scoped to that caller. If any callsite in this caller proved the
    slot is caller storage, every callsite in this caller naming that exact
    slot gets the kind that works.
    """

    from .concordance_declarations import ARGUMENT_BINDING
    from .concordance_declarations import ARGUMENT_BINDING_RESOLUTION

    page = book.page("argument_binding")
    callee_symbol = str(callee_symbol)
    caller_filter = None if caller_symbol is None else str(caller_symbol)
    for row in page.scope_rows(callee_symbol):
        if len(row) == 4:
            if int(row[2]) != int(formal_id):
                continue
            row_caller = str(row[1])
            if caller_filter is not None and row_caller != caller_filter:
                continue
        elif len(row) == 3:
            if caller_filter is not None and len(page.row_fields) == 4:
                # Current pages cannot safely join a caller-blind legacy
                # row to a caller-qualified physical frame.
                continue
            if int(row[1]) != int(formal_id):
                continue
            row_caller = None
        else:
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
                    (callee_symbol, row_caller, int(formal_id), int(source_id))
                    if row_caller is not None
                    else (callee_symbol, int(formal_id), int(source_id))
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
    if not dtype:
        raise ValueError(
            f"proven shape for value {int(value_id)} of "
            f"{_shape_key(function)!r} (extents {extents}) was posted with "
            "no dtype; the caller never recorded the element type, so the "
            "proof refuses to invent one"
        )
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
    fact = ("proven", extents, str(dtype), target_level)
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
        fact = ("proven", extents, str(dtype), target_level)
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
        fact = ("proven", extents, str(dtype), target_level)
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
    # Program: the managed balloon tire's ``np.maximum(..., 0.0)`` at
    # ``balloon_tire_vector_step`` line 372.  Lane D, D3a follow-up.  The
    # call-result cone visits a value twice from one unchanged source, and
    # the second visit re-posted the identical withdrawal with the identical
    # cell set, which the book refuses (REVISE without a changed source).
    # The cone's old blanket try swallowed that; removing it (#798) stopped
    # the tire at deployment select.  Rule: a row whose latest fact is
    # already this same withdrawal has nothing left to withdraw.  Any other
    # prior state still posts, and a refusal there stays loud.
    proposed = ("invalidated", source_id, str(reason))
    if max(recorded, key=lambda entry: int(entry[0]))[1] == proposed:
        return
    cells = tuple(dict.fromkeys((
        *source_cells, *(() if state_ref is None else (state_ref,)),
    )))
    _post_or_unsourced(
        book, PROVEN_SHAPE_PAGE, row, proposed,
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
