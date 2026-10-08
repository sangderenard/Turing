"""Sequence expression ops: augmented assignment, concatenation, append, prepend and bit-pack operations, value kinds and source transforms (relocated from fortran_c_shell)"""

from __future__ import annotations
import ast
import importlib
import inspect
import os
import sys
from typing import (
    Any,
    Iterable,
    Mapping,
)

from .resident_sequence_materialization import _loop_carried_storage_aliases


def _sequence_augassign_ops(graph_obj: Any) -> tuple[tuple[int, int, int], ...]:
    """Return proven ``sequence += sequence`` storage mutations.

    The graph's identity history correlates each lexical spelling with its
    successive SSA versions.  Resolve the AugAssign destination back to the
    resident aggregate producer bearing that spelling, and do the same for the
    source.  No operation-name inference is involved: both endpoints must be
    graph nodes already carrying an aggregate storage contract.
    """

    sequence_kinds = {"list", "set", "dict", "bytes", "bytearray"}
    identity = dict(graph_obj.graph.get("identity_table") or {})
    resident_by_value: dict[int, int] = {}
    for _name, history in identity.items():
        residents = tuple(
            int(value_id)
            for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident

    # Conditional/loop joins preserve the authored storage identity when every
    # incoming value already denotes that same resident arena.  Carry that
    # alias through Phi/LoopResult nodes to later AugAssign occurrences; never
    # choose an arbitrary branch when the residents differ or are unresolved.
    changed = True
    while changed:
        changed = False
        for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
            if int(node_id) in resident_by_value:
                continue
            data = graph_obj.nodes[node_id]
            if str(data.get("type") or data.get("op")).lower() not in {
                "phi", "loopresult"
            }:
                continue
            incoming = {
                resident_by_value[int(parent)]
                for parent, role in (data.get("parents") or ())
                if str(role) in {"body", "orelse", "initial", "updated", "value"}
                and int(parent) in resident_by_value
            }
            unresolved = any(
                str(role) in {"body", "orelse", "initial", "updated", "value"}
                and int(parent) not in resident_by_value
                for parent, role in (data.get("parents") or ())
            )
            if len(incoming) == 1 and not unresolved:
                resident_by_value[int(node_id)] = incoming.pop()
                changed = True

    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.AugAssign)
            and isinstance(expression.op, ast.Add)
            and isinstance(expression.target, ast.Name)
        ):
            continue
        parents = {
            str(role): int(parent) for parent, role in (data.get("parents") or ())
        }
        result_id = int(data.get("value_id", node_id))
        destination_id = resident_by_value.get(parents.get("lhs", -1))
        source_id = resident_by_value.get(parents.get("rhs", -1))
        if destination_id is None or source_id is None:
            continue
        operations.append((result_id, destination_id, source_id))
    return tuple(operations)


def _sequence_concat_ops(
    graph_obj: Any,
    *,
    call_result_kinds: Mapping[int, str] | None = None,
    structural_aliases: Mapping[int, tuple[int, str]] | None = None,
) -> tuple[
    tuple[tuple[int, int, int, str, int | None, int | None], ...],
    tuple[tuple[int, int], ...],
    dict[int, int],
]:
    """Recognize value-producing sequence ``+`` without numericizing it.

    Python's ``bytes + bytes`` and ``list + list`` allocate a new logical
    sequence; they are not elementwise arithmetic.  ProcessGraph deliberately
    keeps the authored ``Add`` spelling, so this identity proves sequence
    semantics from the operands' aggregate contracts (including pursued-call
    result contracts) and returns both concat operations and immutable
    materialization aliases such as ``bytes([value]) -> [value]``.
    """

    sequence_kinds = {"list", "bytes", "bytearray"}
    resident_by_value: dict[int, int] = {}
    kind_by_value: dict[int, str] = {}
    singleton_by_resident: dict[int, int] = {}
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        kind = str(attributes.get("aggregate_kind") or "")
        if kind not in sequence_kinds:
            continue
        value_id = int(data.get("value_id", node_id))
        if attributes.get("producer_kind") == "aggregate_materialization":
            source_nodes = tuple(
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role).startswith("arg:") and parent in graph_obj
            )
            sources = tuple(
                int(graph_obj.nodes[parent].get("value_id", parent))
                for parent in source_nodes
            )
            if len(sources) == 1:
                resident_id = resident_by_value.get(
                    sources[0], sources[0]
                )
                resident_by_value[value_id] = resident_id
                kind_by_value[value_id] = kind
                # ``bytes([opcode])`` is an immutable view of a singleton
                # authored aggregate.  The materializer may carry the leaf
                # ledger even when its transient list source does not; retain
                # that exact scalar identity on the physical resident so
                # concat lowering emits append_scalar and does not demand a
                # sequence descriptor for a one-byte temporary.
                source_expression = (
                    graph_obj.nodes[source_nodes[0]].get("expr_obj")
                    if len(source_nodes) == 1 else None
                )
                source_attributes = (
                    graph_obj.nodes[source_nodes[0]].get("attributes") or {}
                    if len(source_nodes) == 1 else {}
                )
                literal_singleton = (
                    isinstance(source_expression, (ast.List, ast.Tuple, ast.Set))
                    and len(source_expression.elts) == 1
                ) or (
                    source_attributes.get("aggregate_kind") in {
                        "list", "tuple", "set"
                    }
                    and len(tuple(source_attributes.get(
                        "aggregate_leaf_value_ids", ()
                    ))) == 1
                )
                if os.environ.get("TURING_DEBUG_SEQUENCE_CONCAT"):
                    print(
                        "DEBUGMATERIALIZE "
                        f"value={value_id} kind={kind!r} sources={sources!r} "
                        f"source_expr={ast.dump(source_expression, include_attributes=False) if isinstance(source_expression, ast.AST) else None!r} "
                        f"source_kind={source_attributes.get('aggregate_kind')!r} "
                        f"source_type={graph_obj.nodes[source_nodes[0]].get('type')!r} "
                        f"source_constant={graph_obj.nodes[source_nodes[0]].get('constant')!r} "
                        f"source_attributes={source_attributes!r} "
                        f"source_leaves={tuple(source_attributes.get('aggregate_leaf_value_ids', ()))!r} "
                        f"materializer_leaves={tuple(attributes.get('aggregate_leaf_value_ids', ()))!r}",
                        file=sys.stderr,
                    )
                # Prefer the authored singleton aggregate's leaf ledger.  A
                # bytes materializer records its direct argument (the list)
                # as its leaf, while the list records the actual scalar.  The
                # latter is what append_scalar must consume.  Fall back to the
                # materializer ledger only for frontends which omit source
                # aggregate metadata.
                leaves = (
                    tuple(map(int, source_attributes.get(
                        "aggregate_leaf_value_ids", ()
                    )))
                    if literal_singleton else ()
                )
                if not leaves and literal_singleton:
                    leaves = tuple(map(int, attributes.get(
                        "aggregate_leaf_value_ids", ()
                    )))
                if len(leaves) == 1:
                    singleton_by_resident[int(resident_id)] = leaves[0]
                continue
        resident_by_value[value_id] = value_id
        kind_by_value[value_id] = kind
        leaves = tuple(map(int, attributes.get(
            "aggregate_leaf_value_ids", ()
        )))
        if len(leaves) == 1:
            singleton_by_resident[value_id] = leaves[0]

    for value_id, kind in dict(call_result_kinds or {}).items():
        if str(kind) not in sequence_kinds:
            continue
        resident_by_value[int(value_id)] = int(value_id)
        kind_by_value[int(value_id)] = str(kind)
    for value_id, (resident_id, kind) in dict(
        structural_aliases or {}
    ).items():
        if str(kind) not in sequence_kinds:
            continue
        resident_by_value[int(value_id)] = int(resident_id)
        resident_by_value.setdefault(int(resident_id), int(resident_id))
        kind_by_value[int(value_id)] = str(kind)
        kind_by_value.setdefault(int(resident_id), str(kind))
    # IndexedStore and its loop-result versions are deterministic names for
    # the same resident arena.  Retain that correlation while recognizing
    # structural concatenations across control-versioned sequence names.
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

    operations: list[
        tuple[int, int, int, str, int | None, int | None]
    ] = []
    # Deterministic ids are source ordered, so inner concatenations precede
    # the outer expression that consumes them.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if str(data.get("type") or data.get("op") or "") != "Add":
            continue
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        lhs_node = parents.get("lhs")
        rhs_node = parents.get("rhs")
        if lhs_node is None or rhs_node is None:
            continue
        lhs_value = int(graph_obj.nodes[lhs_node].get("value_id", lhs_node))
        rhs_value = int(graph_obj.nodes[rhs_node].get("value_id", rhs_node))
        lhs = resident_by_value.get(lhs_value)
        rhs = resident_by_value.get(rhs_value)
        if lhs is None or rhs is None:
            if os.environ.get("TURING_DEBUG_SEQUENCE_CONCAT"):
                print(
                    "DEBUGSEQMISS "
                    f"fn={graph_obj.graph.get('function_name')} "
                    f"result={int(data.get('value_id', node_id))} "
                    f"lhs_value={lhs_value} lhs_resident={lhs} "
                    f"rhs_value={rhs_value} rhs_resident={rhs} "
                    f"source={ast.unparse(data.get('expr_obj')) if isinstance(data.get('expr_obj'), ast.AST) else None!r}",
                    file=sys.stderr,
                )
            continue
        lhs_kind = kind_by_value.get(lhs_value, kind_by_value.get(lhs))
        rhs_kind = kind_by_value.get(rhs_value, kind_by_value.get(rhs))
        if lhs_kind != rhs_kind:
            if os.environ.get("TURING_DEBUG_SEQUENCE_CONCAT"):
                print(
                    "DEBUGSEQMISS "
                    f"fn={graph_obj.graph.get('function_name')} "
                    f"result={int(data.get('value_id', node_id))} "
                    f"lhs_kind={lhs_kind!r} rhs_kind={rhs_kind!r} "
                    f"source={ast.unparse(data.get('expr_obj')) if isinstance(data.get('expr_obj'), ast.AST) else None!r}",
                    file=sys.stderr,
                )
            continue
        result_id = int(data.get("value_id", node_id))
        result_kind = str(lhs_kind)
        operations.append((
            result_id, int(lhs), int(rhs), result_kind,
            singleton_by_resident.get(int(lhs)),
            singleton_by_resident.get(int(rhs)),
        ))
        if os.environ.get("TURING_DEBUG_SEQUENCE_CONCAT"):
            print(
                "DEBUGSEQ "
                f"fn={graph_obj.graph.get('function_name')} "
                f"result={result_id} lhs={int(lhs)} rhs={int(rhs)} "
                f"lhs_scalar={singleton_by_resident.get(int(lhs))} "
                f"rhs_scalar={singleton_by_resident.get(int(rhs))} "
                f"source={ast.unparse(data.get('expr_obj')) if isinstance(data.get('expr_obj'), ast.AST) else None!r}",
                file=sys.stderr,
            )
        resident_by_value[result_id] = result_id
        kind_by_value[result_id] = result_kind

    aliases = tuple(sorted(
        (int(value_id), int(resident_id))
        for value_id, resident_id in resident_by_value.items()
        if int(value_id) != int(resident_id)
    ))
    singleton_materializations = {
        int(value_id): int(singleton_by_resident[int(resident_id)])
        for value_id, resident_id in aliases
        if int(resident_id) in singleton_by_resident
    }
    return tuple(operations), aliases, singleton_materializations


def _sequence_value_kinds(
    graph_obj: Any,
    *,
    return_sequence_kind_by_reference: Mapping[int, str] | None = None,
) -> dict[int, str]:
    """Resolve sequence semantics through calls and structural expressions.

    A pursued callee's result is represented in its caller by the callee's
    return value identity, not by another aggregate-construction node. Merely
    scanning nodes carrying ``aggregate_kind`` therefore loses a sequence at
    exactly the boundary where an ordinary helper returns it. Resolve direct
    aggregate facts, known call returns, materialization aliases, and sequence
    concatenations as one backend-independent structural closure.
    """

    sequence_kinds = {"list", "bytes", "bytearray"}
    kinds: dict[int, str] = {}
    for node_id, data in graph_obj.nodes(data=True):
        attributes = data.get("attributes") or {}
        kind = str(attributes.get("aggregate_kind") or "")
        if kind not in sequence_kinds:
            continue
        value_id = int(data.get("value_id", node_id))
        kinds[value_id] = kind
        if attributes.get("producer_kind") != "aggregate_materialization":
            continue
        sources = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (data.get("parents") or ())
            if str(role).startswith("arg:") and parent in graph_obj
        )
        if len(sources) == 1:
            kinds[sources[0]] = kind

    call_result_kinds: dict[int, str] = {}
    return_kinds = dict(return_sequence_kind_by_reference or {})
    for node_id, data in graph_obj.nodes(data=True):
        attributes = data.get("attributes") or {}
        reference = attributes.get(
            "callee_ref", attributes.get("method_ref")
        )
        if reference is None:
            continue
        kind = return_kinds.get(int(reference))
        if kind in sequence_kinds:
            value_id = int(data.get("value_id", node_id))
            call_result_kinds[value_id] = str(kind)
            kinds[value_id] = str(kind)

    structural_aliases = {
        int(result_id): (int(source_id), "bytes")
        for result_id, source_id, _source_name, _transform
        in (
            *_utf8_encode_aliases(graph_obj),
            *(
                record for record in _bytes_join_source_transforms(graph_obj)
                if str(record[3]) == "join_bytes"
            ),
        )
    }
    concatenations, aliases, _singletons = _sequence_concat_ops(
        graph_obj,
        call_result_kinds=call_result_kinds,
        structural_aliases=structural_aliases,
    )
    for result_id, _lhs, _rhs, kind, _lhs_scalar, _rhs_scalar in concatenations:
        kinds[int(result_id)] = str(kind)
    for alias_id, resident_id in aliases:
        resident_kind = kinds.get(int(resident_id))
        if resident_kind is not None:
            kinds[int(alias_id)] = resident_kind
    return kinds


def _utf8_encode_aliases(
    graph_obj: Any,
) -> tuple[tuple[int, int, str, str], ...]:
    """Return exact ``str.encode('utf-8')`` source-boundary aliases.

    Encoding is an ABI transform: the authored caller still accepts ``str``;
    the native function receives its deterministic UTF-8 byte sequence.  Only
    the explicit UTF-8 spelling is admitted, so locale/default-codec behavior
    is never inferred.
    """

    aliases = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if str(data.get("op") or data.get("type") or "").casefold() != "encode":
            continue
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        operand_node = parents.get("operand")
        encoding_node = parents.get("arg:0")
        if operand_node is None or encoding_node is None:
            continue
        operand = graph_obj.nodes.get(operand_node, {})
        encoding = graph_obj.nodes.get(encoding_node, {})
        operand_attributes = operand.get("attributes") or {}
        encoding_value = (encoding.get("attributes") or {}).get(
            "value", encoding.get("constant")
        )
        source_name = operand_attributes.get("binding_name")
        if (
            operand_attributes.get("binding_kind") != "parameter"
            or source_name is None
            or str(encoding_value).casefold().replace("_", "-") != "utf-8"
        ):
            continue
        aliases.append((
            int(data.get("value_id", node_id)),
            int(operand.get("value_id", operand_node)),
            str(source_name),
            "utf8",
        ))
    return tuple(aliases)


def _bytes_join_source_transforms(
    graph_obj: Any,
) -> tuple[tuple[int, int, str, str], ...]:
    """Describe ``b''.join(list(parameter))`` as native byte views.

    The list view carries authored row count while the join view carries the
    flattened bytes.  Both name the same authored iterable so a wrapper can
    materialize it exactly once and publish the two deterministic ABI views.
    """

    transforms = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if str(data.get("op") or data.get("type") or "").casefold() != "join":
            continue
        parents = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        separator_id = parents.get("operand")
        materialized_id = parents.get("arg:0")
        if separator_id is None or materialized_id is None:
            continue
        separator = graph_obj.nodes.get(separator_id, {})
        separator_value = (separator.get("attributes") or {}).get(
            "value", separator.get("constant")
        )
        materialized = graph_obj.nodes.get(materialized_id, {})
        materialized_attributes = materialized.get("attributes") or {}
        if (
            separator_value != b""
            or materialized_attributes.get("producer_kind")
            != "aggregate_materialization"
            or materialized_attributes.get("aggregate_kind") != "list"
            or materialized_attributes.get("static_python_reference") != "list"
        ):
            continue
        sources = tuple(map(
            int, materialized_attributes.get("materialized_source_value_ids", ())
        ))
        if len(sources) != 1:
            continue
        source = graph_obj.nodes.get(sources[0], {})
        source_name = (source.get("attributes") or {}).get("binding_name")
        if source_name is None:
            continue
        transforms.extend((
            (
                int(materialized.get("value_id", materialized_id)),
                int(materialized.get("value_id", materialized_id)),
                str(source_name),
                "row_count",
            ),
            (
                int(data.get("value_id", node_id)),
                int(data.get("value_id", node_id)),
                str(source_name),
                "join_bytes",
            ),
        ))
    return tuple(transforms)


def _scalar_source_transforms(
    graph_obj: Any,
    sequence_transforms: Iterable[tuple[int, int, str, str]],
) -> tuple[tuple[int, str, str], ...]:
    """Describe scalar ABI projections such as source-backed ``len``.

    A linked call can legitimately consume a scalar whose producer is a
    structural sequence operation omitted from the numerical SSA region.
    Retaining only that scalar as an anonymous function argument leaves a
    native wrapper no exact way to construct it.  Record the authored source
    and projection at the compilation boundary instead.
    """

    transformed_values = {
        int(result_id): (
            str(source_name),
            (
                "utf8_length" if str(transform) == "utf8"
                else "materialized_length" if str(transform) == "row_count"
                else "sequence_length"
            ),
        )
        for result_id, _source_id, source_name, transform
        in sequence_transforms
    }
    records: list[tuple[int, str, str]] = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
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
            if str(role).startswith("arg:") and int(parent) in graph_obj
        )
        if len(arguments) != 1:
            continue
        argument = graph_obj.nodes[arguments[0]]
        argument_id = int(argument.get("value_id", arguments[0]))
        source = transformed_values.get(argument_id)
        if source is None:
            attributes = argument.get("attributes") or {}
            source_name = attributes.get("binding_name")
            if (
                attributes.get("binding_kind") == "parameter"
                and source_name is not None
            ):
                source = (str(source_name), "sequence_length")
            elif (
                attributes.get("producer_kind")
                == "aggregate_materialization"
            ):
                materialized_sources = tuple(map(
                    int, attributes.get("materialized_source_value_ids", ())
                ))
                if len(materialized_sources) == 1:
                    source_data = next((
                        candidate
                        for candidate_id, candidate in graph_obj.nodes(data=True)
                        if int(candidate.get("value_id", candidate_id))
                        == materialized_sources[0]
                    ), None)
                    source_attributes = (
                        {} if source_data is None
                        else source_data.get("attributes") or {}
                    )
                    source_name = source_attributes.get("binding_name")
                    if source_name is not None:
                        source = (
                            str(source_name), "materialized_length"
                        )
        if source is not None:
            records.append((
                int(data.get("value_id", node_id)), source[0], source[1]
            ))
    return tuple(records)


def _sequence_append_fill_ops(
    graph_obj: Any,
) -> tuple[
    tuple[int, int, int | float | bool | None, int, int], ...
]:
    """Return exact ``resident += literal_sequence * runtime_count`` rows."""

    sequence_kinds = {"list", "bytes", "bytearray"}
    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.AugAssign)
            and isinstance(expression.op, ast.Add)
        ):
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        lhs_id = by_role.get("lhs")
        rhs_id = by_role.get("rhs")
        if lhs_id is None or rhs_id is None:
            continue
        lhs = graph_obj.nodes.get(lhs_id, {})
        rhs = graph_obj.nodes.get(rhs_id, {})
        if (lhs.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in sequence_kinds:
            continue
        if str(rhs.get("type") or rhs.get("op")) not in {"Mul", "Mult"}:
            continue
        operands = {
            str(role): int(parent)
            for parent, role in (rhs.get("parents") or ())
        }
        literal_id = operands.get("lhs")
        count_id = operands.get("rhs")
        if literal_id is None or count_id is None:
            continue
        literal_data = graph_obj.nodes.get(literal_id, {})
        literal = (literal_data.get("attributes") or {}).get(
            "value", literal_data.get("constant")
        )
        if isinstance(literal, (bytes, bytearray)) and len(literal) == 1:
            literal = int(literal[0])
        elif not isinstance(literal, (bool, int, float)) and literal is not None:
            continue
        operations.append((
            int(data.get("value_id", node_id)),
            int(lhs.get("value_id", lhs_id)),
            literal,
            int(graph_obj.nodes[count_id].get("value_id", count_id)),
            int(rhs.get("value_id", rhs_id)),
        ))
    return tuple(operations)


def _sequence_append_slice_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Return exact ``resident += resident[lower:upper]`` mutations.

    Only an authored unit-stride slice with explicit lower and upper values is
    admitted here.  The SSA helper performs Python-compatible bound clipping;
    other slice forms remain in the ordinary lowering ledger.
    """

    sequence_kinds = {"list", "bytes", "bytearray"}

    def _sequence_like(node: Mapping[str, Any]) -> bool:
        attributes = node.get("attributes") or {}
        if attributes.get("aggregate_kind") in sequence_kinds:
            return True
        # A structurally specialized sequence presents as its captured
        # initial value: a Constant whose payload is itself a sequence
        # object (re's ``out``/``tail`` arrive as Constant [] with
        # structural_specialization=True while their sequence descriptors
        # live on under the same value ids).
        return (
            str(node.get("type") or node.get("op")) == "Constant"
            and isinstance(
                attributes.get("value"), (list, bytes, bytearray)
            )
        )

    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.AugAssign)
            and isinstance(expression.op, ast.Add)
        ):
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        lhs_id = by_role.get("lhs")
        rhs_id = by_role.get("rhs")
        if lhs_id is None or rhs_id is None:
            continue
        lhs = graph_obj.nodes.get(lhs_id, {})
        rhs = graph_obj.nodes.get(rhs_id, {})
        if str(rhs.get("type") or rhs.get("op")) != "Indexed":
            # Whole-sequence extend: ``resident += other_resident`` (re's
            # ``out += tail``). Same helper, bounds spanning the whole
            # source (the lowering clips into [0, source_length], so the
            # emitter passes 0 and a beyond-length constant). No extra
            # expression node exists to suppress, hence the None slice id.
            if _sequence_like(lhs) and _sequence_like(rhs):
                operations.append((
                    int(data.get("value_id", node_id)),
                    int(lhs.get("value_id", lhs_id)),
                    int(rhs.get("value_id", rhs_id)),
                    None,
                    None,
                    None,
                ))
            continue
        if (lhs.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in sequence_kinds:
            continue
        indexed = {
            str(role): int(parent)
            for parent, role in (rhs.get("parents") or ())
        }
        source_id = indexed.get("base")
        slice_id = indexed.get("index")
        if source_id is None or slice_id is None:
            continue
        source = graph_obj.nodes.get(source_id, {})
        slice_data = graph_obj.nodes.get(slice_id, {})
        if (source.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in sequence_kinds:
            continue
        if str(slice_data.get("type") or slice_data.get("op")) != "Slice":
            continue
        bounds = {
            str(role): int(parent)
            for parent, role in (slice_data.get("parents") or ())
        }
        lower_id = bounds.get("lower")
        upper_id = bounds.get("upper")
        if lower_id is None or upper_id is None or "step" in bounds:
            continue
        operations.append((
            int(data.get("value_id", node_id)),
            int(lhs.get("value_id", lhs_id)),
            int(source.get("value_id", source_id)),
            int(graph_obj.nodes[lower_id].get("value_id", lower_id)),
            int(graph_obj.nodes[upper_id].get("value_id", upper_id)),
            int(rhs.get("value_id", rhs_id)),
        ))
    return tuple(operations)


def _sequence_bit_pack_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, tuple[int, ...]], ...]:
    """Recognize complete translate/reverse/fixed-bit-word materialization.

    This is the structural meaning of ``_mk_bitmap``: a 0/1 byte arena is
    packed little-endian into fixed-width words.  Recognition follows the AST
    dataflow (translate -> reverse slice -> fixed slice -> int(base=2) ->
    list comprehension), never the function name.
    """

    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        result = graph_obj.nodes[node_id]
        if (result.get("attributes") or {}).get(
            "aggregate_kind"
        ) != "list":
            continue
        expression = result.get("expr_obj")
        if not isinstance(expression, ast.ListComp) or len(expression.generators) != 1:
            continue
        element_parents = tuple(
            int(parent) for parent, role in (result.get("parents") or ())
            if str(role) == "elt" and parent in graph_obj
        )
        if len(element_parents) != 1:
            continue
        conversion = graph_obj.nodes[element_parents[0]]
        conversion_expression = conversion.get("expr_obj")
        if not (
            isinstance(conversion_expression, ast.Call)
            and len(conversion_expression.args) == 2
            and isinstance(conversion_expression.args[1], ast.Constant)
            and conversion_expression.args[1].value == 2
        ):
            continue
        conversion_roles = {
            str(role): int(parent)
            for parent, role in (conversion.get("parents") or ())
        }
        slice_value_id = conversion_roles.get("arg:0")
        if slice_value_id is None or slice_value_id not in graph_obj:
            continue
        sliced = graph_obj.nodes[slice_value_id]
        sliced_roles = {
            str(role): int(parent)
            for parent, role in (sliced.get("parents") or ())
        }
        reversed_id = sliced_roles.get("base")
        inner_slice_id = sliced_roles.get("index")
        if reversed_id is None or inner_slice_id is None:
            continue
        reversed_value = graph_obj.nodes[reversed_id]
        reversed_roles = {
            str(role): int(parent)
            for parent, role in (reversed_value.get("parents") or ())
        }
        translated_id = reversed_roles.get("base")
        reverse_slice_id = reversed_roles.get("index")
        if translated_id is None or reverse_slice_id is None:
            continue
        translated = graph_obj.nodes[translated_id]
        translated_roles = {
            str(role): int(parent)
            for parent, role in (translated.get("parents") or ())
        }
        source_id = translated_roles.get("operand")
        if source_id is None:
            continue
        source = graph_obj.nodes[source_id]
        if (source.get("attributes") or {}).get(
            "aggregate_kind"
        ) not in {"bytes", "bytearray", "list"}:
            continue
        reverse_slice = graph_obj.nodes[reverse_slice_id]
        reverse_step = tuple(
            int(parent) for parent, role in (reverse_slice.get("parents") or ())
            if str(role) == "step"
        )
        if len(reverse_step) != 1:
            continue
        step_expression = graph_obj.nodes[reverse_step[0]].get("expr_obj")
        if not (
            isinstance(step_expression, ast.UnaryOp)
            and isinstance(step_expression.op, ast.USub)
            and isinstance(step_expression.operand, ast.Constant)
            and step_expression.operand.value == 1
        ):
            continue
        inner_slice = graph_obj.nodes[inner_slice_id]
        lower_ids = tuple(
            int(parent) for parent, role in (inner_slice.get("parents") or ())
            if str(role) == "lower"
        )
        if len(lower_ids) != 1:
            continue
        lower = graph_obj.nodes[lower_ids[0]]
        lower_roles = {
            str(role): int(parent)
            for parent, role in (lower.get("parents") or ())
        }
        width_node_id = lower_roles.get("rhs")
        if width_node_id is None:
            continue
        width_id = int(graph_obj.nodes[width_node_id].get(
            "value_id", width_node_id
        ))
        consumed = tuple(dict.fromkeys((
            int(translated.get("value_id", translated_id)),
            int(reversed_value.get("value_id", reversed_id)),
            int(lower.get("value_id", lower_ids[0])),
            int(sliced.get("value_id", slice_value_id)),
            int(result.get("value_id", node_id)),
        )))
        operations.append((
            int(result.get("value_id", node_id)),
            int(source.get("value_id", source_id)),
            width_id,
            consumed,
        ))
    return tuple(operations)


def _sequence_prepend_concat_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Recognize ``[scalar] + sequence`` consumed by ``base[0:0] = ...``."""

    sequence_kinds = {"list", "bytearray"}
    resident_by_value: dict[int, int] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        residents = tuple(
            int(value_id) for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident
    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        store = graph_obj.nodes[node_id]
        if str(store.get("type") or store.get("op")) != "IndexedStore":
            continue
        roles = {
            str(role): int(parent)
            for parent, role in (store.get("parents") or ())
        }
        base_id = roles.get("base")
        index_id = roles.get("index")
        value_id = roles.get("value")
        if None in (base_id, index_id, value_id):
            continue
        base = graph_obj.nodes[base_id]
        index = graph_obj.nodes[index_id]
        concatenation = graph_obj.nodes[value_id]
        resident_id = resident_by_value.get(
            int(base.get("value_id", base_id))
        )
        if resident_id is None:
            continue
        bounds = {
            str(role): int(parent)
            for parent, role in (index.get("parents") or ())
        }
        if set(bounds) != {"lower", "upper"}:
            continue
        bound_values = []
        for bound_id in bounds.values():
            bound_data = graph_obj.nodes[bound_id]
            bound_values.append((bound_data.get("attributes") or {}).get(
                "value", bound_data.get("constant")
            ))
        if bound_values != [0, 0]:
            continue
        if str(concatenation.get("type") or concatenation.get("op")) != "Add":
            continue
        concat_roles = {
            str(role): int(parent)
            for parent, role in (concatenation.get("parents") or ())
        }
        singleton_id = concat_roles.get("lhs")
        tail_id = concat_roles.get("rhs")
        if singleton_id is None or tail_id is None:
            continue
        singleton = graph_obj.nodes[singleton_id]
        leaves = tuple((singleton.get("attributes") or {}).get(
            "aggregate_leaf_value_ids", ()
        ))
        if (
            (singleton.get("attributes") or {}).get("aggregate_kind") != "list"
            or len(leaves) != 1
        ):
            continue
        operations.append((
            int(store.get("value_id", node_id)),
            int(resident_id),
            int(leaves[0]),
            int(concatenation.get("value_id", value_id)),
            int(graph_obj.nodes[tail_id].get("value_id", tail_id)),
        ))
    return tuple(operations)


def _sequence_prepend_packed_call_ops(
    graph_obj: Any,
) -> tuple[tuple[int, int, int, int, int, int, int], ...]:
    """Correlate prefix splice with the pursued byte-packing call edge."""

    sequence_kinds = {"list", "bytes", "bytearray"}
    resident_by_value: dict[int, int] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        residents = tuple(
            int(value_id) for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident
    operations = []
    for (
        store_result_id, destination_id, prefix_id, concat_result_id,
        tail_result_id,
    ) in _sequence_prepend_concat_ops(graph_obj):
        tail_node = next((
            (node_id, data)
            for node_id, data in graph_obj.nodes(data=True)
            if int(data.get("value_id", node_id)) == int(tail_result_id)
        ), None)
        if tail_node is None:
            continue
        _tail_node_id, tail = tail_node
        attributes = tail.get("attributes") or {}
        if attributes.get("callee_ref") is None:
            continue
        arguments = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (tail.get("parents") or ())
            if str(role).startswith(("arg:", "kw:"))
        )
        if len(arguments) != 1:
            continue
        source_id = resident_by_value.get(int(arguments[0]))
        if source_id is None:
            continue
        operations.append((
            int(store_result_id), int(destination_id), int(prefix_id),
            int(source_id), int(concat_result_id), int(tail_result_id),
            int(_tail_node_id),
        ))
    return tuple(dict.fromkeys(operations))


def _sequence_inplace_bit_pack_call_ops(
    graph: Any,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Return pursued calls whose callee is a structural bit-pack program."""

    graph_obj = graph.G
    function_table = getattr(graph, "function_table", None)
    if function_table is None:
        return ()
    sequence_kinds = {"list", "bytes", "bytearray"}
    resident_by_value: dict[int, int] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        residents = tuple(
            int(value_id) for value_id in history
            if int(value_id) in graph_obj
            and (graph_obj.nodes[int(value_id)].get("attributes") or {}).get(
                "aggregate_kind"
            ) in sequence_kinds
        )
        if not residents:
            continue
        resident = residents[0]
        for value_id in history:
            resident_by_value[int(value_id)] = resident
    operations = []
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        call = graph_obj.nodes[node_id]
        attributes = call.get("attributes") or {}
        reference = attributes.get("callee_ref")
        if reference is None:
            continue
        try:
            callee = function_table.entry(int(reference)).graph
        except (KeyError, TypeError, ValueError):
            continue
        if callee is None:
            continue
        contracts = _sequence_bit_pack_ops(callee.G)
        if len(contracts) != 1:
            continue
        _callee_destination, _callee_source, _width_id, _consumed = contracts[0]
        arguments = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (call.get("parents") or ())
            if str(role).startswith(("arg:", "kw:"))
        )
        if len(arguments) != 1:
            continue
        resident_id = resident_by_value.get(int(arguments[0]))
        if resident_id is None:
            continue
        entry = function_table.entry(int(reference))
        callable_object = getattr(entry, "python_callable", None)
        if callable_object is None and "." in str(entry.qualified_name):
            parts = str(entry.qualified_name).split(".")
            for split in range(len(parts) - 1, 0, -1):
                try:
                    candidate = importlib.import_module(".".join(parts[:split]))
                except ImportError:
                    continue
                try:
                    for attribute in parts[split:]:
                        candidate = getattr(candidate, attribute)
                except AttributeError:
                    continue
                callable_object = candidate
                break
        if callable_object is None:
            continue
        try:
            signature = inspect.signature(callable_object)
        except (TypeError, ValueError):
            continue
        parameters = tuple(signature.parameters.values())
        if len(parameters) < 2:
            continue
        width_default = parameters[1].default
        if not isinstance(width_default, int) or width_default <= 0:
            continue
        operations.append((
            int(call.get("value_id", node_id)),
            int(resident_id),
            int(width_default),
            int(reference),
            int(node_id),
        ))
    return tuple(dict.fromkeys(operations))
