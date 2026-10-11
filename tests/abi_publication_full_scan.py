"""Frozen pre-worklist publication, used only as the parity oracle."""
from __future__ import annotations

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
