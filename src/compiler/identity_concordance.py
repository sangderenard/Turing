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
    a recorded value alias points at a value the function never defines.
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
    changed = True
    while changed:
        changed = False
        for node_id, data in graph_obj.nodes(data=True):
            attributes = data.setdefault("attributes", {})
            operation = str(
                data.get("op") or data.get("type") or ""
            ).casefold()
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
                            changed = True
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
                record_shape_transformation(
                    shape_scope_of(callee), ("return", str(callee_name)),
                    shape_scope_of(function), result_id,
                    stage="projected_iterable_layout",
                    operation="call_result", source_state=state,
                    target_state=state, role="return",
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
                for value in occurrences:
                    if int(value.id) != source_id:
                        continue
                    if column_dtype not in {None, "", "unknown"}:
                        value.dtype = str(column_dtype)
                    value.accounting = {
                        **dict(value.accounting or {}),
                        "sequence_id": int(descriptor.sequence_id),
                        "sequence_length_value_id": int(length.id),
                        "tensor_metadata_state": "dynamic",
                        "sequence_row_shape": row_shape,
                        "program_abi_rank": 1 + len(row_shape),
                    }
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
                if column_dtype not in {None, "", "unknown"}:
                    value.dtype = str(column_dtype)
                value.accounting = {
                    **value.accounting,
                    "sequence_row_shape": row_shape,
                    "program_abi_rank": 1 + len(row_shape),
                }
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
                    for value in occurrences:
                        if int(value.id) != target_id:
                            continue
                        value.shape = row_shape
                        if column_dtype not in {None, "", "unknown"}:
                            value.dtype = str(column_dtype)
                        value.accounting = {
                            **dict(value.accounting or {}),
                            "sequence_row_shape": row_shape,
                            "program_abi_rank": len(row_shape),
                        }
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


@dataclass(frozen=True)
class Claim:
    """One record's statement about a value's identity."""

    kind: str
    key: str
    source: str


@dataclass
class ValueRow:
    function: str
    value_id: int
    dtype: str | None = None
    is_formal: bool = False
    claims: list[Claim] = field(default_factory=list)
    definitions: list[tuple[str, int, str]] = field(default_factory=list)
    uses: list[tuple[str, int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Finding:
    kind: str
    function: str
    value_id: int | None
    detail: str


_INOUT_REDEFINED = ("program_abi_mutable", "program_abi_field_written")
#: Finding kinds ``concordance_report`` counts on their own line rather than
#: in its gating first line.
_UNSOURCED_KINDS = frozenset({"unsourced-fact", "unsourced-identity"})


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
        found.extend(self._source_field_identity_findings(module))
        found.extend(self._source_parameter_identity_findings(module))
        found.extend(self._source_numeric_component_findings(module))
        found.extend(self._source_numeric_intrinsic_findings(module))
        found.extend(self._source_numeric_type_dependency_findings(module))
        found.extend(self._source_numeric_operator_specialization_findings(module))
        found.extend(self._source_sequence_mutation_findings(module))
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
        sourced: set[Any] = set()
        edge_page = book.pages.get(EDGE_PAGE.name)
        if edge_page is not None:
            sourced.update(row[0] for row in edge_page.rows())
        minted_with_edge: set[int] = set()
        mint_page = book.pages.get(MINT_PAGE.name)
        if mint_page is not None:
            for row in mint_page.rows():
                sourced.add(row[0])
                if row[1] is not None:
                    minted_with_edge.add(int(row[1]))
        tags: dict[tuple[str, Any], str] = {}
        unsourced_page = book.pages.get(UNSOURCED_PAGE.name)
        if unsourced_page is not None:
            for row in unsourced_page.rows():
                tags[(row[0], row[1])] = row[2]
        groups: Counter = Counter()
        for name, page in book.pages.items():
            if name not in registered or name in private:
                continue
            for (row, column), fact in page.cells.items():
                if fact is None or isinstance(fact, Unresolved):
                    continue
                if (name, row, column) in sourced:
                    continue
                groups[(name, tags.get((name, row), "unknown"), "cell")] += 1
        for (page_name, _row), stage_name in tags.items():
            if page_name in registered:
                continue
            groups[(page_name, stage_name, "raw row")] += 1
        identities = [
            (function, value_id)
            for function, value_id in self.rows
            if has_flag(value_id, MINTED) and value_id not in minted_with_edge
        ]
        return dict(groups), identities

    def _unsourced_findings(self, module: Any) -> list[Finding]:
        groups, identities = self._unsourced_worklist(module)
        found = [
            Finding(
                "unsourced-fact", page_name, None,
                f"stage {stage_name}: {count} {unit}(s) with neither an "
                "inbound edge nor a mint edge",
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

        book = identity_book(module)
        page = book.pages.get(RETURN_SITE_SLOT.name)
        found = []
        for name, function in module.functions.items():
            scope = function.metadata.get("record_return_state_scope")
            if not scope:
                continue
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
        integral = {
            "int", "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64", "bool",
        }
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
                    if dtype in integral:
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
            if row is None or (not row.is_formal and not row.definitions):
                found.append(Finding(
                    "descriptor-member-unknown", name, value_id,
                    f"roles={roles!r}: not a formal, never defined",
                ))
            distinct = {(sequence_id, role.rstrip("0123456789"))
                        for sequence_id, role in roles}
            sequences = {sequence_id for sequence_id, _role in distinct}
            if len(sequences) > 1:
                found.append(Finding(
                    "descriptor-member-shared", name, value_id,
                    f"roles={roles!r}",
                ))
        # keyed helper operands
        described = {
            value_id for (function_name, value_id) in self.sequence_roles
            if function_name == name
        }
        keyed_parts = {
            value_id for value_id, row in rows.items()
            if any(c.kind == "keyed-part" for c in row.claims)
        }
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                attributes = instruction.attributes or {}
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
                ):
                    found.append(Finding(
                        "alias-target-missing", name, value_id,
                        f"alias of {claim.key} which is never defined",
                    ))
        # A private alias snapshot is allowed only as a durable copy of the
        # shared authority. This catches the exact class of failure where a
        # late pass proves an identity in ``metadata.value_aliases`` but the
        # next pass reads only ``planning_value_concordance`` (or vice versa).
        book = dict(getattr(module, "metadata", {}) or {}).get(
            "identity_book"
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

    def _abi_or_member_key(self, function: str, formal: Any) -> str | None:
        accounting = dict(getattr(formal, "accounting", {}) or {})
        abi_field = accounting.get("program_abi_field")
        if abi_field is not None:
            # Identity is per record INSTANCE: the parameter that carries
            # the record, or the call whose result record this storage
            # holds.  Two instances of one record type in one function
            # (``metrics`` and ``coerced = coerce_metrics(metrics)``)
            # legitimately own the same field name each.
            instance = accounting.get("returned_record_storage")
            if instance is not None:
                instance = f"{instance}@{accounting.get('callsite_id')}"
            else:
                instance = accounting.get("program_abi_parameter") or accounting.get("program_abi_record")
            return f"{instance}.{abi_field}"
        roles = self.sequence_roles.get((function, int(formal.id)))
        if roles:
            sequence_id, role = roles[0]
            return f"seq{sequence_id}.{role}"
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
    fact_type: Any = object
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


@dataclass(frozen=True, repr=False)
class Ref:
    """The full vector of one cell's location: page, row, column."""

    page: Page
    row: tuple
    column: int

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
        cached = self.__dict__.get("_hash")
        if cached is None:
            cached = hash((self.page, self.row, self.column))
            object.__setattr__(self, "_hash", cached)
        return cached

    def __getstate__(self) -> dict:
        # The cached hash is this process's (string hashes are salted per
        # process); a pickled or copied instance recomputes it.
        return {key: value for key, value in self.__dict__.items() if key != "_hash"}


@dataclass(frozen=True)
class Derived:
    cells: tuple[Ref, ...]


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
        fact_type: Any = object,
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

#: The stage recorded for a raw write when the book knows no better
#: (``IdentityBook.active_stage`` unset), and the reason every such write
#: is tagged with.
RAW_STAGE = declare_stage("raw_primitive")
RAW_PRIMITIVE = declare_reason("raw_primitive")


def _validate_row(page: Page, row: Any, *, allow_new: bool) -> tuple[int, ...]:
    """Check ``row`` against ``page.row_fields``; return the NEW positions."""

    if not isinstance(row, tuple):
        raise ConcordanceRefusal(
            f"{page.name}: row must be a tuple, got {type(row).__name__}"
        )
    if len(row) != len(page.row_fields):
        raise ConcordanceRefusal(
            f"{page.name}: row {row!r} has {len(row)} element(s); the page "
            f"declares {len(page.row_fields)}: "
            f"{tuple(item.name for item in page.row_fields)!r}"
        )
    new_positions: list[int] = []
    for position, (declared, item) in enumerate(zip(page.row_fields, row)):
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


@dataclass
class IdentityPage:
    """One pipeline stage's row (identity) x column (round) table of facts."""

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
    book: Any = field(default=None, repr=False, compare=False)
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

    def __setstate__(self, state: dict) -> None:
        # A page pickled before the index existed is indexed on arrival.
        self.__dict__.update(state)
        if "row_columns" not in state or "column_positions" not in state:
            self._reindex()

    def _reindex(self) -> None:
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
        positions = self.column_positions
        if len(positions) != len(self.columns):
            self._reindex()
        position = positions.get(column)
        if position is None:
            position = positions[column] = len(self.columns)
            self.columns.append(column)
        if (row, column) not in self.cells:
            columns = self.row_columns.setdefault(row, [])
            columns.append(column)
            if len(columns) > 1 and positions[columns[-2]] > position:
                columns.sort(key=positions.__getitem__)
        self.cells[(row, column)] = fact
        self.stamps[(row, column)] = self.clock[0]
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
            book._admit_raw_write(self, row)
        self._stamp(row, column, fact)
        self.clock[0] += 1

    def scope_rows(self, scope: Any) -> tuple[Any, ...]:
        """Every row whose first element is ``scope``, in recorded order."""
        return tuple(self.scopes.get(scope, ()))

    def scope_row_count(self, scope: Any) -> int:
        """How many rows ``scope`` owns, without materializing them."""
        return len(self.scopes.get(scope, ()))

    def revise(self, row: Any, fact: Any) -> Any:
        """Append ``fact`` as ``row``'s next revision and return it.

        For a row whose fact legitimately grows (a record descriptor gaining
        the fields another callee projects); the full history stays on the
        page.  Identity facts that must never change use ``concord``.
        """
        entries = self.history(row)
        self.set(row, entries[-1][0] + 1 if entries else 0, fact)
        return fact

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
            return default
        return self.cells[(row, columns[-1])]

    def latest_column(self, row: Any) -> int | None:
        """The column of ``row``'s most recently recorded cell, or None --
        ``history(row)[-1][0]`` without materializing the history."""
        if len(self.column_positions) != len(self.columns):
            self._reindex()
        columns = self.row_columns.get(row)
        return columns[-1] if columns else None

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

    def mapping(self, scope: Any) -> "PageMapping":
        """This page's rows under ``scope`` as a mutable mapping.

        ``mapping[key]`` is row ``(scope, key)``'s latest fact; assignment
        is a revision and deletion a ``None`` revision, so a pass that keeps
        its working state here keeps it on the book with its full history.
        """
        return PageMapping(self, scope)

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
            for row in self.rows()
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
        return tuple(
            (column, cells[(row, column)])
            for column in self.row_columns.get(row, ())
        )

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

    def __init__(self, page: IdentityPage, scope: Any) -> None:
        self.page = page
        self.scope = scope

    def __getitem__(self, key: Any) -> Any:
        try:
            fact = self.page.latest((self.scope, key))
        except TypeError:  # an unhashable key names no row
            raise KeyError(key) from None
        if fact is None:
            raise KeyError(key)
        return fact

    def __setitem__(self, key: Any, value: Any) -> None:
        if value is None:
            raise ValueError(
                f"{self.page.name}: None is the removal fact; delete the row"
            )
        row = (self.scope, key)
        if self.page.latest(row) != value:
            self.page.revise(row, value)

    def __delitem__(self, key: Any) -> None:
        if self.page.latest((self.scope, key)) is None:
            raise KeyError(key)
        self.page.revise((self.scope, key), None)

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


def mint_scope(label: Any, stage: Any = None) -> tuple[str, int]:
    """A fresh scope on the active compile's book (see ``IdentityBook``)."""
    return current_identity_book().mint_scope(label, stage)


class IdentityBook:
    """Every stage's page, so one identity's claim can be read across all
    of them -- the comparison none of them makes on its own."""

    def __init__(
        self, *, detached: bool = False, registry: Registry | None = None,
    ) -> None:
        self.pages: dict[str, IdentityPage] = {}
        #: Shared by every page: the book's construction clock.
        self.clock: list[int] = [0]
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
        self.latch: Latch = Latch.OPEN
        #: When a pass sets this, raw writes made while it is set are tagged
        #: with that stage instead of ``RAW_STAGE``.
        self.active_stage: Stage | None = None

    def page(self, name: Any) -> IdentityPage:
        """The page named ``name`` (a str or a declared ``Page``), created
        on first mention with this book's clock."""
        if isinstance(name, Page):
            name = name.name
        page = self.pages.get(name)
        if page is None:
            page = self.pages[name] = IdentityPage(
                name, clock=self.clock, book=self,
            )
        elif page.book is None:
            page.book = self
        return page

    # ------------------------------------------------------------- the api
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
            if self.latch is not Latch.OPEN:
                raise ConcordanceRefusal(
                    f"post {page.name} {row!r}: Unsourced({provenance.reason.name})"
                    " refused; the latch is CLOSED"
                )
            reason = provenance.reason
            if (
                not isinstance(reason, Reason)
                or registry.reasons.get(reason.name) != reason
            ):
                raise ConcordanceRefusal(
                    f"post {page.name}: undeclared reason {reason!r}"
                )
        else:
            raise ConcordanceRefusal(
                f"post {page.name}: provenance must be Derived, Novel or "
                f"Unsourced, got {provenance!r}"
            )

        target_page = self.page(page)
        minted: int | None = None
        if novel and new_positions:
            # Direct ``GLOBAL_MONOTONIC_IDS.mint()`` calls still exist
            # elsewhere; sharing the source keeps this id disjoint from them.
            minted = GLOBAL_MONOTONIC_IDS.mint()
            position = new_positions[0]
            row = row[:position] + (minted,) + row[position + 1:]

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
                    edge_cells = self.page(EDGE_PAGE).cells
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
            edge_page = self.page(EDGE_PAGE)
            dependents = self.page(DEPENDENTS_PAGE)
            for source_ref, _ in sources:
                edge_row = (target_key, source_ref.key, stage.name)
                edge_page._stamp(edge_row, 0, True)
                dependents._stamp((source_ref.key, edge_row), 0, True)
        elif novel:
            self.page(MINT_PAGE)._stamp(
                (target_key, minted), 0,
                (provenance.transform, tuple(provenance.operands)),
            )
        else:
            self.page(UNSOURCED_PAGE)._stamp(
                (page.name, row, stage.name), 0, provenance.reason,
            )
        self.clock[0] += 1
        return target

    def _source_stamp(self, ref: Any) -> int:
        """The clock reading of an existing cell named by ``ref``; a Ref to
        no cell (or to an undeclared page) is refused."""
        if not isinstance(ref, Ref):
            raise ConcordanceRefusal(f"source must be a Ref, got {ref!r}")
        if self.registry.pages.get(ref.page.name) != ref.page:
            raise ConcordanceRefusal(f"source names undeclared page {ref.page!r}")
        page = self.pages.get(ref.page.name)
        if page is None or (ref.row, ref.column) not in page.cells:
            raise ConcordanceRefusal(f"source cell does not exist: {ref!r}")
        return page.stamps[(ref.row, ref.column)]

    def _admit_raw_write(self, page: IdentityPage, row: Any) -> None:
        """Tag (OPEN) or refuse (CLOSED) a write made through a raw
        primitive rather than ``post``."""
        if self.latch is not Latch.OPEN:
            raise ConcordanceRefusal(
                f"raw write refused under a CLOSED latch: page {page.name!r} "
                f"row {row!r}; write it through IdentityBook.post"
            )
        stage = self.active_stage if self.active_stage is not None else RAW_STAGE
        self.page(UNSOURCED_PAGE)._stamp(
            (page.name, row, stage.name), 0, RAW_PRIMITIVE,
        )

    # -------------------------------------------------------------- reading
    def _ref_from_key(self, key: tuple) -> Ref:
        page_name, row, column = key
        return Ref(self.registry.page(page_name), row, column)

    def latest_ref(self, page: Page, row: tuple) -> Ref | None:
        """The Ref of ``row``'s most recent cell on ``page``, or None."""
        stored = self.pages.get(page.name)
        if stored is None:
            return None
        column = stored.latest_column(row)
        if column is None:
            return None
        return Ref(page, row, column)

    def stamp_of(self, ref: Ref) -> int:
        return self._source_stamp(ref)

    def edges_into(self, ref: Ref) -> tuple[tuple[Ref, Stage], ...]:
        """Every (source cell, stage) ``ref``'s cell was derived from."""
        edge_page = self.pages.get(EDGE_PAGE.name)
        if edge_page is None:
            return ()
        return tuple(
            (self._ref_from_key(row[1]), self.registry.stages[row[2]])
            for row in edge_page.scope_rows(ref.key)
        )

    def edges_out_of(self, ref: Ref) -> tuple[tuple[Ref, Stage], ...]:
        """Every (target cell, stage) derived from ``ref``'s cell."""
        dependents = self.pages.get(DEPENDENTS_PAGE.name)
        if dependents is None:
            return ()
        return tuple(
            (self._ref_from_key(row[1][0]), self.registry.stages[row[1][2]])
            for row in dependents.scope_rows(ref.key)
        )

    def mint_of(self, ref: Ref) -> tuple[Transform, tuple[Ref, ...]] | None:
        """The (transform, operands) a Novel post minted ``ref``'s row
        with, or None when the cell was not posted Novel."""
        mint_page = self.pages.get(MINT_PAGE.name)
        if mint_page is None:
            return None
        for row in mint_page.scope_rows(ref.key):
            fact = mint_page.latest(row)
            if fact is not None:
                return fact
        return None

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


def shape_transformation_state(descriptor: Any) -> tuple[Any, ...] | None:
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
#: A shape edge whose source identity has no changed state cell on the book
#: (the source is a graph identity the caller read off the graph).
SHAPE_SOURCE_NOT_ON_BOOK = declare_reason("shape_source_not_on_book")
#: A shape state re-resolved over an edge that did not change since the
#: row's previous revision (the row was withdrawn or re-pointed in between).
SHAPE_STATE_REDERIVED = declare_reason("shape_state_rederived_over_unchanged_edge")
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
    if cells:
        try:
            return book.post(
                page, row, fact, stage=stage,
                provenance=Derived(cells), mode=Mode.REVISE,
            )
        except ConcordanceRefusal:
            pass
    return book.post(
        page, row, fact, stage=stage,
        provenance=Unsourced(reason), mode=Mode.REVISE,
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
    *, source_scope: Any = None,
) -> None:
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
        _post_or_unsourced(
            book, SHAPE_STATE_PAGE, row, fact,
            book.registry.declare_stage(str(reason)),
            () if source_ref is None else (source_ref,),
            SHAPE_SOURCE_NOT_ON_BOOK,
        )


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

    page = current_identity_book().page("sequence_row_layout_concordance")
    row = (_shape_key(scope), int(sequence_id))
    incumbent = page.latest(row)
    if incumbent is None:
        return
    fact = ("invalidated", source_id, str(reason))
    if incumbent != fact:
        page.revise(row, fact)


def commit_sequence_row_layout(
    scope: Any,
    sequence_id: int,
    column_shapes: Iterable[Iterable[int]],
    column_dtypes: Iterable[Any],
    *,
    source: str,
    page: IdentityPage | None = None,
) -> SequenceRowLayout:
    """Commit an element layout reached through the transformation graph.

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
    history = page.history(row)
    column = history[-1][0] + 1 if history else 0
    page.set(row, column, (
        resolved.column_shapes,
        resolved.column_dtypes,
        str(source),
    ))
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
    for sequence_id in normalized:
        row = (scope, int(sequence_id))
        history = page.history(row)
        column = history[-1][0] + 1 if history else 0
        page.set(row, column, (result, str(source)))
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
    history = page.history(row)
    column = history[-1][0] + 1 if history else 0
    page.set(row, column, (
        resolved.policy,
        resolved.column_count,
        resolved.writable,
        str(source),
    ))
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
    says nothing new; anything else is written as before, raw."""

    page = book.page("loop_scope")
    if (row, column) in page.cells:
        if page.cells[(row, column)] != fact:
            page.set(row, column, fact)
        return
    sources = tuple(cell for cell in dict.fromkeys(cells or ()) if cell is not None)
    if sources and column == len(page.history(row)):
        book.post(
            book.registry.page("loop_scope"), row, fact,
            stage=book.registry.declare_stage("loop_scope_declaration"),
            provenance=Derived(sources), mode=Mode.REVISE,
        )
    else:
        page.set(row, column, fact)


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
    scope = (authored_function_name(function), int(loop_node_id))
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
            if outer_counts[int(outer_id)] == 1
            else (
                int(outer_id),
                source_bindings or ("entry", int(ordinal)),
            )
        )
        row = (*scope, row_key)
        cells = (
            () if rebind_cells is None
            else tuple(rebind_cells(
                int(outer_id), int(carried_id), int(inner_id),
            ))
        )
        _post_loop_scope_cell(book, row, OUTER, int(outer_id), cells)
        _post_loop_scope_cell(book, row, CARRIED, int(carried_id), cells)
        _post_loop_scope_cell(book, row, INNER, int(inner_id), cells)
        _post_loop_scope_cell(
            book, row, INNER + 1,
            ("graph", int(graph_outer), int(graph_inner)), cells,
        )
        if source_bindings:
            _post_loop_scope_cell(
                book, row, INNER + 2, ("bindings", source_bindings), cells,
            )


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
    page = book.page("loop_scope_inner_transition")
    row = (
        authored_function_name(function), int(loop_node_id),
        int(declared_inner),
    )
    fact = (int(resident_inner), str(reason))
    sources = tuple(cell for cell in dict.fromkeys(cells or ()) if cell is not None)
    if sources:
        _post_or_unsourced(
            book, book.registry.page("loop_scope_inner_transition"), row,
            fact, book.registry.declare_stage("loop_scope_declaration"),
            sources, LOOP_SCOPE_TRANSITION_CAUSE_NOT_ON_BOOK,
        )
        return
    history = page.history(row)
    column = history[-1][0] + 1 if history else 0
    page.set(row, column, fact)


def loop_scope_declarations(book: Any, function: Any) -> list[dict]:
    """Every loop scope declared for one authored function."""

    page = book.page("loop_scope")
    wanted = authored_function_name(function)
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
            binding_identity = generations.get(INNER + 2)
            if (
                isinstance(binding_identity, tuple)
                and len(binding_identity) == 2
                and binding_identity[0] == "bindings"
            ):
                rebind["source_bindings"] = tuple(binding_identity[1])
            record["rebinds"].append(rebind)
    return [record for record in scopes.values() if record["boundary"]]


def concord_loop_scope_latch_residents(module: Any) -> tuple[dict, ...]:
    """Publish the final resident identity on every transformed backedge.

    Aggregate legalization and linked-call projection happen after control
    lowering declared the authored inner generation.  At the completed-module
    seam the loop Phi is the exact authority on what now crosses the latch.
    Reconcile that resident into the concordance so the declaration tracks the
    transformation instead of leaving a locally recorded but globally unknown
    rename.
    """

    book = identity_book(module)
    receipts: list[dict] = []
    for function_name, function in getattr(module, "functions", {}).items():
        for declaration in loop_scope_declarations(book, function_name):
            header, latch, _exit = declaration["boundary"]
            loop_node_id = int(declaration["loop_node_id"])
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
        FORMAL_ACTUAL_OCCURRENCE, FRAME_BINDING, FRAME_SOURCE_CELL_ABSENT, SSA_VALUE,
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
            page = book.page("formal_storage_resolution")
            row = (str(function_name), formal_id)
            history = page.history(row)
            column = history[-1][0] + 1 if history else 0
            page.set(row, column, (
                "compiler_frame_storage", source_receipts,
            ))
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
                page.set(row, 0, fact)
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

    page = book.page("argument_binding")
    for _callsite, fact in argument_binding_history(
        page, str(callee_symbol), int(formal_id),
    ):
        if not (isinstance(fact, tuple) and len(fact) == 2):
            continue
        recorded_kind, recorded_source = fact
        if not isinstance(recorded_source, int):
            continue
        if int(recorded_source) != int(source_id):
            continue
        if str(recorded_kind) == "caller_storage":
            resolution_page = book.page("argument_binding_resolution")
            resolution_row = (
                str(callee_symbol), int(formal_id), int(source_id),
            )
            history = resolution_page.history(resolution_row)
            column = history[-1][0] + 1 if history else 0
            resolution_page.set(resolution_row, column, (
                "caller_storage", str(kind),
                "materializing_binding_kind",
            ))
            return "caller_storage"
    return str(kind)


def record_proven_shape(
    function: Any, value_id: int, extents: Any, dtype: Any,
    level: int | None = 0,
) -> None:
    """Record extents proven for one value identity at one causal level.

    Only EXTENTS are recorded.  An empty shape is both a rank-0 scalar and
    what a query returns when recovery stops, so storing it would let an
    unknown win a race against a real shape.  A deeper level supersedes a
    shallower one because it was derived from more of the program; the same
    answer arriving deeper is recorded at its own level so the page shows how
    far it has been confirmed.
    """

    extents = tuple(int(extent) for extent in (extents or ()))
    if not extents:
        return
    page = current_identity_book().page("proven_shape")
    row = (_shape_key(function), int(value_id))
    recorded = page.history(row)
    fact = ("proven", extents, str(dtype or "float64"))
    target_level = (
        max((int(column) for column, _fact in recorded), default=0)
        if level is None else int(level)
    )
    if not recorded:
        page.set(row, target_level, fact)
        return
    deepest = max(recorded, key=lambda entry: int(entry[0]))
    if isinstance(deepest[1], tuple) and deepest[1] and (
        deepest[1][0] == "invalidated"
    ):
        # An upstream identity changed after this proof was derived.  The next
        # descriptor query is a new proof generation, not a disagreement with
        # the invalidated fact.  Keep both events in causal order.
        page.set(row, int(deepest[0]) + 1, fact)
        return
    if (
        tuple(deepest[1][1]) == extents
        or target_level > int(deepest[0])
    ):
        page.set(row, target_level, fact)
        return
    page.set(
        row, target_level,
        ("conflicting", extents, str(dtype or "")),
    )


def invalidate_proven_shape(
    function: Any, value_id: int, source_id: int, reason: Any,
) -> None:
    """Withdraw a derived shape after one of its exact dependencies changes."""

    page = current_identity_book().page("proven_shape")
    row = (_shape_key(function), int(value_id))
    recorded = page.history(row)
    column = max(
        (int(existing) for existing, _fact in recorded), default=-1,
    ) + 1
    # A transformation source may be a structured identity (a callee return,
    # a descriptor root), not only a graph value id.
    if isinstance(source_id, int) or not isinstance(source_id, tuple):
        source_id = int(source_id)
    page.set(
        row, column,
        ("invalidated", source_id, str(reason)),
    )
    invalidate_shape_transformation(
        function, int(value_id), source_id, reason,
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
    for page_name in sorted(book.pages):
        page = book.pages[page_name]
        rows = page.rows()
        rows_level = _page_rows_level(book, page_name)
        withheld = level < rows_level
        header = f"[{page_name}] {len(rows)} row(s), {len(page.cells)} cell(s)"
        if not full and withheld and level >= IdentityLogLevel.FACTS:
            header += f" (rows withheld below {rows_level.name.lower()})"
        yield header
        if level < IdentityLogLevel.FACTS or withheld:
            continue
        # Rows gathered under the id group they belong to, so one page's
        # entries read as the few spaces they actually span rather than as
        # one undifferentiated list.  A row whose key names no id keeps its
        # place under "unkeyed" instead of being dropped or invented into
        # a group.
        by_group: dict[str, list[Any]] = {}
        for row in rows:
            value_id = row_value_id(row)
            group = (
                "unkeyed" if value_id is None
                else (group_by_prefix([value_id])[0].label)
            )
            by_group.setdefault(group, []).append(row)
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
    if full:
        return
    yield f"unsourced: latch {book.latch.name}"
    tally: Counter = Counter()
    for page_ref, _row, reason, stage in book.unsourced_rows():
        tally[(getattr(page_ref, "name", page_ref), stage.name, reason.name)] += 1
    for (page_name, stage_name, reason_name), count in sorted(tally.items()):
        yield f"  {page_name} stage={stage_name} reason={reason_name}: {count}"
    yield from extra_lines


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
        with lzma.open(
            temporary, "wt", encoding="utf-8", newline="\n",
            preset=identity_log_lzma_preset(),
        ) as handle:
            for line in iter_identity_book_lines(
                book, level, extra_lines=extra_lines,
            ):
                handle.write(line)
                handle.write("\n")
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
