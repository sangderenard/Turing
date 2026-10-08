"""Field slot recovery: the per-field slot operation contract recovered from the class surface graph (relocated from fortran_c_shell)"""

from __future__ import annotations
import ast
from typing import (
    Any,
    Mapping,
)

from .authored_record_abi import (
    _authored_row_layout,
    _graph_sequence_record_abi,
    _sequence_record_binding_value_ids,
    _sequence_record_storage_columns,
)
from .authored_parameter_abi import (
    _authored_sequence_annotation_contract,
    _current_authored_parameter_annotations,
)
from .resident_sequence_materialization import (
    _loop_carried_storage_aliases,
    _sequence_append_call_mutations,
    _source_mapping_mutations,
)


def _field_slot_ops(
    graph_obj: Any,
    *,
    contract_scope: str | None = None,
    retained_storage_identities: frozenset[str] = frozenset(),
    keyed_table_fields: frozenset[str] = frozenset(),
    keyed_writable_fields: frozenset[str] = frozenset(),
):
    """Recover a method's instance-field accesses as slot loads and stores.

    A class's field layout is declared once (``class_table[class]['fields']``),
    giving each field a fixed slot. ``self`` is that field arena. This reads the
    process graph's field-op nodes and returns, for one method:

    * ``self_value_id`` -- the value id of the ``self`` arena, or ``None``;
    * ``field_ops`` -- ``(kind, value_id, slot)`` for every field access in the
      graph's own schedule order, ``kind`` being ``"read"`` (a ``GetAttr``, whose
      ``value_id`` is the result the method already consumes) or ``"write"`` (a
      ``setattr``, whose ``value_id`` is the stored source). Keeping reads and
      writes in one ordered list preserves their interleaving, so a store and a
      later read of one slot stay in the order the source wrote them;
    * ``field_count`` -- the arena length, so ``self`` is a sized array.
    """

    class_table = dict(graph_obj.graph.get("class_table") or {})
    owner = graph_obj.graph.get("method_owner")
    record = (
        class_table.get(owner)
        if owner in class_table
        else (next(iter(class_table.values())) if len(class_table) == 1 else None)
    )
    declared_self_record = dict(
        (graph_obj.graph.get("parameter_record_abi") or {}).get("self") or {}
    )
    declared_fields = tuple(dict(declared_self_record.get("fields") or {}))
    # Once a method receiver has a ProgramABI record, that declaration is its
    # physical layout and identity. Class navigation supplies method bodies;
    # its descriptive field catalogue must not widen or reorder ABI storage.
    fields = declared_fields or tuple((record or {}).get("fields") or ())
    declared_identity = declared_self_record.get("identity")
    record_identity = (
        str(declared_identity) if declared_identity is not None else owner
    )
    slot_of = {name: index for index, name in enumerate(fields)}

    identity = dict(graph_obj.graph.get("identity_table") or {})
    sequence_contract_scope = str(
        contract_scope
        or graph_obj.graph.get("qualified_name")
        or graph_obj.graph.get("function_name")
        or "<anonymous>"
    )
    from .identity_concordance import (
        commit_sequence_contract,
        committed_sequence_contract,
    )
    self_history = identity.get("self") or ()
    self_value_id = int(self_history[-1]) if self_history else None

    # Order field ops by SOURCE order (node id), not the data-dependency
    # schedule. Memory ordering between a write and a later read of the same
    # field is a real dependency the AST wrote but the graph does not carry as a
    # data edge, so a topological sort is free to float the read ahead of the
    # write. Nodes are created in source order, so their ids preserve the order
    # the programmer wrote -- which is the order the memory operations must run.
    field_ops: list[tuple[str, int, int]] = []
    # One reducer WRITTEN cell per ``"write"`` op, in the same order: the
    # write's identity on the book, which control lowering published its
    # in-arm version under (``ssa_field_version``).
    write_state_cells: list[Any] = []
    const_sources: dict[int, Any] = {}
    sequence_initializations: list[tuple[int, str, int]] = []
    sequence_declarations: list[tuple[int, str, int, bool]] = []
    sequence_memberships: list[tuple[int, int, int, bool]] = []
    table_lookups: list[tuple[int, int | tuple[int, ...], int]] = []
    table_stores: list[tuple[int, int | tuple[int, ...], int, int]] = []
    table_deletions: list[
        tuple[int, int | tuple[int, ...], int | None, str]
    ] = []
    tombstone_sequence_ids: set[int] = set()
    retained_sequence_ids: set[int] = set()
    nested_sequence_ids: set[int] = set()
    nested_tensor_sequence_ids: set[int] = set()
    nested_record_fields: dict[int, tuple[str, int]] = {}
    field_aliases = dict(graph_obj.graph.get("class_field_aliases") or {})
    field_aggregate_kinds = dict(
        graph_obj.graph.get("class_field_aggregate_kinds") or {}
    )
    # A dataclass annotation is also an exact physical record contract.  Keep
    # its span kind available to the class-field coordinator so a read such as
    # ``self.locals`` becomes the declared sequence arena, never a scalar load
    # from the receiver slot vector.  Explicit frontend aggregate evidence
    # wins when both are present.
    declared_span_kinds = {
        str(field_name): str(field.get("aggregate_kind"))
        for field_name, field in dict(
            declared_self_record.get("fields") or {}
        ).items()
        if str(field.get("storage") or "") == "span"
        and field.get("aggregate_kind") is not None
    }
    declared_parameter_table_kinds = {
        str(field_name): "list"
        for parameter_record in dict(
            graph_obj.graph.get("parameter_record_abi") or {}
        ).values()
        for field_name, field in dict(
            parameter_record.get("fields") or {}
        ).items()
        if str(field.get("storage") or "") == "table"
    }
    field_aggregate_kinds = {
        **declared_span_kinds,
        **declared_parameter_table_kinds,
        **field_aggregate_kinds,
    }
    field_value_aggregate_kinds = dict(
        graph_obj.graph.get("class_field_value_aggregate_kinds") or {}
    )

    def node_operation(data: Mapping[str, Any]) -> str:
        """Canonical ProcessGraph operation spelling for field analysis.

        Authored graph nodes carry the semantic class in ``type`` and often a
        lower-case executable spelling in ``op``.  Preferring ``op`` without
        normalizing it made ``type=GetAttr, op=getattr`` invisible to the
        whole-object resolver, exactly where object fields should become
        native record storage.
        """

        return str(data.get("op") or data.get("type") or "").casefold()

    def canonical_field(name: str) -> str:
        seen: set[str] = set()
        current = str(name)
        while current in field_aliases and current not in seen:
            seen.add(current)
            current = str(field_aliases[current])
        return current

    # A method may mention only an alias (NetworkX ``_succ``) while its record
    # storage is authored under the canonical field (``_adj``). Correlate all
    # such GetAttr occurrences to one resident sequence ID; this is storage
    # aliasing, not value copying or object dispatch.
    field_sequence_ids: dict[str, int] = {}
    lexical_sequence_ids: dict[tuple[str, str], int] = {}
    inferred_nested_table_bases: dict[int, int] = {}
    aggregate_reads: list[tuple[str, str, int]] = []

    def fixed_row_width(annotation: Any) -> int | None:
        """Read a fixed tuple row width from an authored container annotation."""

        layout = _authored_row_layout(annotation, dict(
            (graph_obj.graph.get("program_abi") or {}).get("records") or {}
        ))
        if layout is None:
            return None
        return sum(width for _kind, _identity, width in layout)

    annotated_row_widths: dict[int, int] = {}
    for binding_name, annotation in (
        *dict(graph_obj.graph.get("type_annotations") or {}).items(),
        # A parameter's authored annotation declares its row exactly as a
        # local annotation does; ``failures: list[tuple[float, Metrics,
        # tuple[str, ...]]] | None`` is the physical row of ``failures``.
        *_current_authored_parameter_annotations(graph_obj).items(),
    ):
        width = fixed_row_width(annotation)
        if width is None:
            continue
        for value_id in identity.get(str(binding_name), ()):
            annotated_row_widths[int(value_id)] = int(width)
    for parameter_name, annotation in (
        _current_authored_parameter_annotations(graph_obj).items()
    ):
        contract = _authored_sequence_annotation_contract(annotation)
        history = tuple(map(int, identity.get(str(parameter_name), ())))
        if contract is None or not history:
            continue
        policy, column_count, writable, _dtypes = contract
        commit_sequence_contract(
            sequence_contract_scope,
            int(history[0]),
            policy,
            int(column_count),
            bool(writable),
            source=f"authored parameter {parameter_name}: {annotation}",
        )
    for binding_name, record in _graph_sequence_record_abi(graph_obj).items():
        width = len(_sequence_record_storage_columns(record))
        binding_value_ids = set(_sequence_record_binding_value_ids(
            graph_obj, binding_name,
        ))
        for value_id in binding_value_ids:
            annotated_row_widths[int(value_id)] = int(width)
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "getattr":
            continue
        attribute = (data.get("attributes") or {}).get("attribute")
        if field_aggregate_kinds.get(str(attribute)) not in {
            "list", "set", "dict", "tuple", "bytes", "bytearray"
        }:
            continue
        aggregate_reads.append((
            str(attribute), canonical_field(str(attribute)),
            int(data.get("value_id", node_id)),
        ))
    for attribute, canonical, result_id in aggregate_reads:
        if attribute == canonical:
            field_sequence_ids[canonical] = result_id
    for _attribute, canonical, result_id in aggregate_reads:
        field_sequence_ids.setdefault(canonical, result_id)
    # Table fields on ordinary record parameters and on rows selected from a
    # Sequence[Record] are resident sequences too. They are not method-self
    # slots, so declare them from the exact receiver ABI instead of routing
    # them through `slot_of` below.
    sequence_record_by_value: dict[int, Mapping[str, Any]] = {}
    for binding_name, sequence_record in _graph_sequence_record_abi(
        graph_obj
    ).items():
        binding_value_ids = set(_sequence_record_binding_value_ids(
            graph_obj, binding_name,
        ))
        for value_id in binding_value_ids:
            sequence_record_by_value[int(value_id)] = sequence_record
    parameter_record_by_value = {
        int(value_id): record
        for parameter_name, record in dict(
            graph_obj.graph.get("parameter_record_abi") or {}
        ).items()
        for value_id in identity.get(str(parameter_name), ())
    }
    for node_id, data in graph_obj.nodes(data=True):
        if node_operation(data) != "getattr":
            continue
        attribute = str((data.get("attributes") or {}).get("attribute") or "")
        receiver_id = next((
            int(parent)
            for parent, role in data.get("parents") or ()
            if str(role) in {"value", "object", "base", "receiver"}
            and int(parent) in graph_obj
        ), None)
        if receiver_id is None:
            continue
        field_record = parameter_record_by_value.get(receiver_id)
        receiver = graph_obj.nodes[receiver_id]
        if field_record is None and node_operation(receiver) in {
            "indexed", "load",
        }:
            row_source = next((
                int(parent)
                for parent, role in receiver.get("parents") or ()
                if str(role) == "base" and int(parent) in graph_obj
            ), None)
            if row_source is not None:
                field_record = sequence_record_by_value.get(row_source)
        field = (
            None if field_record is None
            else dict(field_record.get("fields") or {}).get(attribute)
        )
        if not isinstance(field, Mapping) or str(
            field.get("storage") or ""
        ) != "table":
            continue
        sequence_declarations.append((
            int(data.get("value_id", node_id)),
            "duplicates",
            len(tuple(field.get("columns") or ())),
            bool(field.get("mutable", False)),
        ))
    # A contract-declared keyed field is a lookup table too, but it is a
    # program-ABI record field, not a class-field aggregate, so it must not
    # enter ``field_sequence_ids`` (that registry engages the object-field
    # arena machinery).  Same identity convention: the GetAttr's own value id
    # names the table, one canonical id per field.
    keyed_field_sequence_ids: dict[str, int] = {}
    if keyed_table_fields:
        for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
            data = graph_obj.nodes[node_id]
            if node_operation(data) != "getattr":
                continue
            attribute = str(
                (data.get("attributes") or {}).get("attribute") or ""
            )
            if attribute not in keyed_table_fields:
                continue
            sequence_id = keyed_field_sequence_ids.setdefault(
                attribute, int(data.get("value_id", node_id))
            )
            receiver_id = next((
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) in {"value", "object", "base", "receiver"}
                and int(parent) in graph_obj
            ), None)
            receiver_record = (
                None if receiver_id is None
                else parameter_record_by_value.get(receiver_id)
            )
            receiver_field = (
                None if receiver_record is None
                else dict(receiver_record.get("fields") or {}).get(attribute)
            )
            if bool(
                (data.get("attributes") or {}).get(
                    "mapping_value_tensor", False
                )
                or (
                    isinstance(receiver_field, Mapping)
                    and receiver_field.get("value_tensor")
                )
            ):
                # The exact Program-ABI receiver field (and, when retained,
                # the reducer's equivalent node receipt) declares
                # ``Mapping[..., Tensor]``. Carry that same structural fact
                # into the sequence contract: the outer value column stores
                # child-row handles and the child pool owns the tensor spans.
                # Treating it as the scalar-value table is the point where
                # tensor identity otherwise disappears before SSA lowering.
                nested_sequence_ids.add(int(sequence_id))
                nested_tensor_sequence_ids.add(int(sequence_id))
    # Runtime aggregates captured from a lexical/module binding are resident
    # storage exactly like aggregate fields, but they have no record slot.
    # Correlate all normalized occurrences by their authored binding identity;
    # never inspect or serialize the bound Python collection's contents.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        aggregate_kind = attributes.get("aggregate_kind")
        binding_name = attributes.get("binding_name")
        binding_kind = attributes.get("binding_kind")
        if (
            aggregate_kind not in {
                "list", "set", "dict", "tuple", "bytes", "bytearray"
            }
            or binding_name is None
            or binding_kind not in {"parameter", "closure", "external"}
        ):
            continue
        identity_key = (str(binding_kind), str(binding_name))
        sequence_id = lexical_sequence_ids.setdefault(
            identity_key, int(data.get("value_id", node_id))
        )
        committed = committed_sequence_contract(
            sequence_contract_scope, sequence_id
        )
        policy = (
            committed.policy if committed is not None else
            "unique" if aggregate_kind in {"set", "dict"} else "duplicates"
        )
        column_count = (
            committed.column_count if committed is not None else
            2 if aggregate_kind == "dict" else
            max(1, annotated_row_widths.get(int(sequence_id), 1))
        )
        writable = (
            committed.writable if committed is not None else
            aggregate_kind not in {"tuple", "bytes"}
        )
        sequence_declarations.append((
            sequence_id,
            policy,
            column_count,
            writable,
        ))
        commit_sequence_contract(
            sequence_contract_scope,
            sequence_id,
            policy,
            column_count,
            writable,
            source=(
                f"runtime aggregate {binding_kind}.{binding_name} "
                f"viewed as {aggregate_kind}"
            ),
        )
        storage_identity = f"{binding_kind}.{binding_name}"
        if storage_identity in retained_storage_identities:
            retained_sequence_ids.add(sequence_id)

    # An authored homogeneous collection annotation is sufficient physical
    # ABI evidence even when the Input node itself has no Python aggregate
    # value attached. This matters for detached compilation: Sequence[str]
    # is a token arena, not an untyped scalar that happens to be indexed.
    declared_sequence_ids = {
        int(sequence_id)
        for sequence_id, _policy, _columns, _writable
        in sequence_declarations
    }
    for parameter_name, annotation in (
        _current_authored_parameter_annotations(graph_obj).items()
    ):
        contract = _authored_sequence_annotation_contract(annotation)
        history = tuple(map(int, identity.get(str(parameter_name), ())))
        if contract is None or not history:
            continue
        sequence_id = int(history[0])
        if sequence_id in declared_sequence_ids:
            continue
        policy, column_count, writable, _dtypes = contract
        lexical_sequence_ids.setdefault(
            ("parameter", str(parameter_name)), sequence_id
        )
        sequence_declarations.append((
            sequence_id, policy, int(column_count), bool(writable),
        ))
        declared_sequence_ids.add(sequence_id)
        retained_sequence_ids.add(sequence_id)

    # Source-derived ``Sequence[Record]`` metadata survives graph extraction
    # independently of any one Input wrapper. Declare its caller-owned
    # columnar arena explicitly so later field projections share one physical
    # descriptor instead of minting anonymous fallback columns.
    for parameter_name, record in _graph_sequence_record_abi(graph_obj).items():
        history = _sequence_record_binding_value_ids(
            graph_obj, parameter_name,
        )
        sequence_record_fields = tuple(dict(record.get("fields") or {}))
        if not history or not sequence_record_fields:
            continue
        sequence_id = int(history[0])
        lexical_sequence_ids.setdefault(
            ("parameter", str(parameter_name)), sequence_id
        )
        sequence_declarations.append((
            sequence_id,
            "duplicates",
            len(_sequence_record_storage_columns(record)),
            bool(record.get("mutable", False)),
        ))

    # An unannotated parameter can still state a complete aggregate contract
    # through authored operations.  ``key in p`` plus ``p[key]`` whose result
    # is iterated proves a keyed table whose values are child sequences.  This
    # derives storage from graph structure; it does not inspect parameter
    # contents or infer from a method's spelling.
    iterated_value_ids = {
        int(graph_obj.nodes[parent].get("value_id", parent))
        for node_id in graph_obj.nodes()
        for data in (graph_obj.nodes[node_id],)
        if (data.get("op") or data.get("type")) == "For"
        for parent, role in (data.get("parents") or ())
        if str(role) == "iterable" and parent in graph_obj
    }
    inferred_by_identity: dict[tuple[str, str], list[int]] = {}
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "indexed":
            continue
        result_id = int(data.get("value_id", node_id))
        if result_id not in iterated_value_ids:
            continue
        base_nodes = tuple(
            int(parent) for parent, role in (data.get("parents") or ())
            if str(role) == "base" and parent in graph_obj
        )
        if len(base_nodes) != 1:
            continue
        base_data = graph_obj.nodes[base_nodes[0]]
        base_attributes = base_data.get("attributes") or {}
        if base_attributes.get("binding_kind") != "parameter":
            continue
        binding_name = base_attributes.get("binding_name")
        if binding_name is None:
            continue
        inferred_by_identity.setdefault(
            ("parameter", str(binding_name)), []
        ).append(int(base_data.get("value_id", base_nodes[0])))
    for identity_key, observed_ids in inferred_by_identity.items():
        history = tuple(map(int, identity.get(identity_key[1], ())))
        sequence_id = int(history[0] if history else observed_ids[0])
        if sequence_id in declared_sequence_ids:
            # An authored list/set/dict/sequence annotation already owns the
            # physical policy and row width.  Iterating ``factors[1:]`` is a
            # list slice, not evidence that ``factors`` is an unannotated
            # keyed table.  Structural inference may fill an absent contract;
            # it may never reinterpret a committed one.
            continue
        lexical_sequence_ids.setdefault(identity_key, sequence_id)
        sequence_declarations.append((sequence_id, "unique", 2, False))
        nested_sequence_ids.add(sequence_id)
        for value_id in (*history, *observed_ids):
            inferred_nested_table_bases[int(value_id)] = sequence_id

    # Locally constructed aggregates are resident storage too.  Previously the
    # record extractor declared only parameters, captures and object fields;
    # a local ``out = []`` or ``charmap = bytearray(256)`` therefore reached a
    # planned region as a shapeless scalar unless append/add happened to create
    # a descriptor as a side effect.  The source graph already owns the exact
    # producer identity, storage policy and writability, so publish that same
    # identity in the method sequence table.  Tuple values remain record rows
    # and are handled by projected-row/record lowering rather than pretending
    # the heterogeneous record itself is one homogeneous arena.
    singleton_structural_scalar_ids = {
        int(leaf_ids[0])
        for _candidate_id, candidate in graph_obj.nodes(data=True)
        for leaf_ids in (tuple(map(
            int,
            (candidate.get("attributes") or {}).get(
                "aggregate_leaf_value_ids", ()
            ),
        )),)
        if len(leaf_ids) == 1
    }
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        aggregate_kind = attributes.get("aggregate_kind")
        snapshot = (aggregate_kind == "tuple" and
                    attributes.get("producer_kind") == "aggregate_materialization")
        if not snapshot and aggregate_kind not in {
            "list", "set", "dict", "bytes", "bytearray"
        }:
            continue
        if (
            aggregate_kind == "bytes"
            and attributes.get("producer_kind")
            == "aggregate_materialization"
        ):
            # ``bytes(local_sequence)`` is a result view over the local arena,
            # not a second independently allocated sequence.  Root authority
            # below maps it to its sole source only after that relation is
            # proven; declaring another arena here fabricates an ABI input.
            continue
        sequence_id = int(data.get("value_id", node_id))
        leaf_ids = tuple(map(int, attributes.get(
            "aggregate_leaf_value_ids", ()
        )))
        if (
            len(leaf_ids) == 1 and sequence_id == leaf_ids[0]
            or sequence_id in singleton_structural_scalar_ids
        ):
            # A singleton literal/wrapper may deliberately reuse its sole
            # scalar leaf as a reducible structural identity.  It is consumed
            # by concat lowering as ``append_scalar`` and owns no arena of its
            # own. Declaring that scalar id as a sequence turns ordinary
            # parameters such as ``section_id`` into rank-1 arrays.
            continue
        sequence_declarations.append((
            sequence_id,
            "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
            (
                (
                    int(attributes.get("mapping_key_column_count", 1)) + 1
                ) if aggregate_kind == "dict"
                else max(
                    int(attributes.get("sequence_column_count", 1)),
                    annotated_row_widths.get(sequence_id, 1),
                )
            ),
            snapshot or bool(attributes.get(
                "sequence_writable", aggregate_kind != "bytes"
            )),
        ))
        mapping_items = attributes.get("compile_time_mapping_items")
        if (aggregate_kind == "dict" and mapping_items is not None
                and not isinstance(data.get("expr_obj"), ast.Dict)):
            from .string_table import string_token

            encoded_rows = tuple(
                (
                    string_token(key) if isinstance(key, str) else key,
                    string_token(item) if isinstance(item, str) else item,
                )
                for key, item in tuple(mapping_items)
            )
            sequence_initializations.append((
                sequence_id,
                f"literal_table={encoded_rows!r}",
                2,
            ))
        if attributes.get("mapping_value_aggregate_kind") in {
            "list", "set", "dict", "bytearray"
        }:
            nested_sequence_ids.add(sequence_id)
    mapping_storage_aliases = _loop_carried_storage_aliases(graph_obj)
    declared_mapping_ids = {
        int(sequence_id) for sequence_id, policy, columns, _writable in sequence_declarations
        if policy == "unique" and columns >= 2
    }
    lexical_mapping_effects = {
        int(mutation.effect_node_id) for mutation in _source_mapping_mutations(graph_obj)
        if mutation.operator == "update"
    }

    def table_sequence(base_id: int) -> tuple[int | None, str | None]:
        base_id = int(mapping_storage_aliases.get(int(base_id), base_id))
        # A defensive ``mapping or {}`` keeps the mapping's exact contents:
        # the empty fallback is selected only when the mapping is already
        # empty.  Treat that expression as the left mapping for every keyed
        # query, including ``get``. Membership lowering already applied this
        # identity rule locally; keeping it here makes all table consumers use
        # one graph rewrite and lets the normal lookup path retain its default.
        for _ in range(4):
            if base_id not in graph_obj:
                break
            base_data = graph_obj.nodes[base_id]
            expression = base_data.get("expr_obj")
            if not (
                isinstance(expression, ast.BoolOp)
                and isinstance(expression.op, ast.Or)
            ):
                break
            operands = {
                str(role): int(parent)
                for parent, role in (base_data.get("parents") or ())
            }
            left_id = operands.get("value:0")
            right_id = operands.get("value:1")
            if (
                left_id is None
                or right_id is None
                or right_id not in graph_obj
            ):
                break
            right_data = graph_obj.nodes[right_id]
            right_expression = right_data.get("expr_obj")
            if not (
                isinstance(right_expression, ast.Dict)
                and not right_expression.keys
                and not right_expression.values
            ):
                break
            base_id = int(mapping_storage_aliases.get(left_id, left_id))
        if base_id not in graph_obj:
            return None, None
        base_data = graph_obj.nodes[base_id]
        inferred_sequence_id = inferred_nested_table_bases.get(
            int(base_data.get("value_id", base_id))
        )
        if inferred_sequence_id is not None:
            return inferred_sequence_id, "parameter.inferred_nested_table"
        attributes = base_data.get("attributes") or {}
        if attributes.get("compile_time_mapping_items") is not None:
            return (
                int(base_data.get("value_id", base_id)),
                f"constant.{attributes.get('binding_name', base_id)}",
            )
        field_name = attributes.get("attribute")
        if field_aggregate_kinds.get(str(field_name)) == "dict":
            canonical = canonical_field(str(field_name))
            return (
                field_sequence_ids[canonical],
                f"{owner}.{canonical}",
            )
        if str(field_name) in keyed_field_sequence_ids:
            return (
                keyed_field_sequence_ids[str(field_name)],
                f"keyed.{field_name}",
            )
        if attributes.get("aggregate_kind") != "dict":
            return None, None
        binding_name = attributes.get("binding_name")
        binding_kind = attributes.get("binding_kind")
        identity_key = (str(binding_kind), str(binding_name))
        if binding_name is None or identity_key not in lexical_sequence_ids:
            value_id = int(base_data.get("value_id", base_id))
            if value_id in declared_mapping_ids:
                return value_id, f"materialized-mapping:{value_id}"
            return None, None
        return (
            lexical_sequence_ids[identity_key],
            f"{binding_kind}.{binding_name}",
        )

    def authored_index_values(data: Any) -> int | tuple[int, ...] | None:
        values = tuple(
            int(graph_obj.nodes[parent].get("value_id", parent))
            for parent, role in (data.get("parents") or ())
            if str(role) == "index" and parent in graph_obj
        )
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    def deletes_first_live_key(data: Any) -> bool:
        expression = data.get("expr_obj")
        if not isinstance(expression, ast.Subscript):
            return False
        key = expression.slice
        return (
            isinstance(key, ast.Call)
            and isinstance(key.func, ast.Name)
            and key.func.id == "next"
            and len(key.args) == 1
            and isinstance(key.args[0], ast.Call)
            and isinstance(key.args[0].func, ast.Name)
            and key.args[0].func.id == "iter"
            and len(key.args[0].args) == 1
            and ast.dump(key.args[0].args[0], include_attributes=False)
            == ast.dump(expression.value, include_attributes=False)
        )
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        node_type = node_operation(data)
        attribute = (data.get("attributes") or {}).get("attribute")
        canonical_attribute = (
            canonical_field(str(attribute)) if attribute is not None else None
        )
        if canonical_attribute is None or canonical_attribute not in slot_of:
            continue
        if node_type == "getattr":
            result_id = data.get("value_id", node_id)
            field_ops.append((
                "read", int(result_id), slot_of[canonical_attribute]
            ))
            field_value_id = int(result_id)
            aggregate_kind = field_aggregate_kinds.get(str(attribute))
            if aggregate_kind in {
                "list", "set", "dict", "tuple", "bytes", "bytearray"
            }:
                sequence_id = field_sequence_ids[canonical_field(str(attribute))]
                if int(result_id) == sequence_id:
                    declared_field = dict(
                        dict(declared_self_record.get("fields") or {}).get(
                            canonical_field(str(attribute))
                        ) or {}
                    )
                    sequence_writable = (
                        bool(declared_field.get("mutable"))
                        if "mutable" in declared_field
                        else aggregate_kind not in {"tuple", "bytes"}
                    )
                    sequence_declarations.append((
                        sequence_id,
                        "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
                        (
                            2 if aggregate_kind == "dict" else
                            max(1, annotated_row_widths.get(
                                int(sequence_id), 1,
                            ))
                        ),
                        sequence_writable,
                    ))
                storage_identity = f"{owner}.{canonical_field(str(attribute))}"
                if storage_identity in retained_storage_identities:
                    retained_sequence_ids.add(sequence_id)
                if field_value_aggregate_kinds.get(str(attribute)) == "dict":
                    nested_sequence_ids.add(sequence_id)
        elif node_type == "setattr":
            source_parent = next(
                (
                    parent
                    for parent, role in (data.get("parents") or ())
                    if str(role) == "value"
                ),
                None,
            )
            if source_parent is None:
                continue
            source_data = graph_obj.nodes[source_parent]
            source_id = source_data.get("value_id", source_parent)
            field_value_id = int(source_id)
            field_ops.append((
                "write", int(source_id), slot_of[canonical_attribute]
            ))
            write_state_cells.append(
                (data.get("attributes") or {}).get("field_state_cell")
            )
            source_attributes = source_data.get("attributes") or {}
            if node_operation(source_data) == "staticreference":
                reference_identity = source_attributes.get(
                    "static_python_reference"
                )
                if reference_identity is not None:
                    from .string_table import string_token

                    reference_name = str(reference_identity)
                    const_sources[int(source_id)] = {
                        "ssa_reference_identity": reference_name,
                        "reference_kind": "static-python",
                        "reference_handle": string_token(
                            "\x00turing.reference.static-python\x00"
                            + reference_name
                        ),
                        "host_resident": True,
                    }
            # ``class_ref`` is the constructor spelling retained on a direct
            # construction.  Call/result specialization can replace that
            # spelling with ``result_class_ref`` while preserving the same
            # authored value identity.  Both facts say that the value stored
            # in this field is the record instance; consulting only the former
            # degraded a specialized ``Metrics`` write to an untyped scalar
            # receiver slot.
            nested_class_identity = (
                source_attributes.get("result_class_ref")
                or source_attributes.get("class_ref")
            )
            if nested_class_identity is not None:
                nested_record_fields[slot_of[canonical_attribute]] = (
                    str(nested_class_identity), int(source_id)
                )
            aggregate_kind = (
                source_attributes.get("aggregate_kind")
                or field_aggregate_kinds.get(str(attribute))
            )
            if (
                aggregate_kind in {
                    "list", "set", "dict", "tuple", "bytearray",
                }
                and attribute not in field_aliases
            ):
                # A tuple assignment replaces an object field with one
                # indexable resident just as a list assignment does.  Tuple
                # immutability governs element mutation; it does not turn the
                # whole value into an opaque reference.  Publishing the write
                # source as the field's sequence identity lets reads,
                # iteration and call-frame forwarding derive views of that
                # same resident instead of materializing an opaque slot beside
                # a second sequence object.
                sequence_initializations.append((
                    int(source_id),
                    "unique" if aggregate_kind in {"set", "dict"} else "duplicates",
                    2 if aggregate_kind == "dict" else 1,
                ))
                storage_identity = f"{owner}.{canonical_field(str(attribute))}"
                if storage_identity in retained_storage_identities:
                    retained_sequence_ids.add(int(source_id))
                if field_value_aggregate_kinds.get(str(attribute)) == "dict":
                    nested_sequence_ids.add(int(source_id))
            # A constant field write (``self.x = None`` / ``5`` / ``"s"``) has
            # no producer in the control body, so carry the constant value; the
            # injection materialises it before the store (None becomes the
            # absence sentinel via the tokenizer).
            if node_operation(source_data) in {"const", "constant"}:
                attrs = source_data.get("attributes") or {}
                const_sources[int(source_id)] = attrs.get(
                    "value", source_data.get("constant")
                )
        else:
            continue
        # A receiver field the ProgramABI declares as a nested record is that
        # record on every access, read or write: its identity is the
        # declaration's, never a scalar slot.
        declared_field = dict(
            dict(declared_self_record.get("fields") or {}).get(
                canonical_attribute
            ) or {}
        )
        if str(declared_field.get("storage") or "") == "record":
            nested_schema = str(declared_field.get("record") or "")
            nested_receipt = dict(
                dict(
                    (graph_obj.graph.get("program_abi") or {}).get("records")
                    or {}
                ).get(nested_schema) or {}
            )
            from .identity_concordance import current_identity_book

            nested_identity = current_identity_book().page(
                "receiver_nested_record_field_concordance"
            ).concord(
                (sequence_contract_scope, str(canonical_attribute)),
                str(nested_receipt.get("identity") or nested_schema),
            )
            nested_record_fields.setdefault(
                slot_of[canonical_attribute],
                (nested_identity, field_value_id),
            )
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Compare)
            and len(expression.ops) == 1
            and isinstance(expression.ops[0], (ast.In, ast.NotIn))
        ):
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        query_id = by_role.get("lhs")
        field_id = by_role.get("rhs")
        if query_id is None or field_id is None or field_id not in graph_obj:
            continue
        field_data = graph_obj.nodes[field_id]
        # ``x in (obj.field or {})`` (or a local alias of that expression,
        # e.g. ``channels = metrics.error_channels or {}; ... "k" in
        # channels``) is a real, common defensive idiom, not a different
        # shape of membership test: the ``or`` only ever supplies a fallback
        # when the field itself is falsy, so the field identity for
        # membership purposes is its LEFT operand. Without this, the field
        # lookup below sees a BoolOp node with no ``attribute``/
        # ``aggregate_kind`` of its own and silently fails to recognize the
        # membership test at all (diagnosed against the real managed-tire
        # compile: `_no_exchange_observed`'s `"power_w" in channels` never
        # materialized a value because of exactly this).
        for _ in range(4):
            boolop_expression = field_data.get("expr_obj")
            if not (
                isinstance(boolop_expression, ast.BoolOp)
                and isinstance(boolop_expression.op, ast.Or)
            ):
                break
            left_id = next((
                int(parent)
                for parent, role in (field_data.get("parents") or ())
                if str(role) == "value:0"
            ), None)
            if left_id is None or left_id not in graph_obj:
                break
            field_id = left_id
            field_data = graph_obj.nodes[field_id]
        field_attributes = field_data.get("attributes") or {}
        field_name = field_attributes.get("attribute")
        sequence_id = None
        if field_aggregate_kinds.get(str(field_name)) in {"dict", "set"}:
            sequence_id = field_sequence_ids[canonical_field(str(field_name))]
        elif field_attributes.get("aggregate_kind") in {"dict", "set"}:
            identity_key = (
                str(field_attributes.get("binding_kind")),
                str(field_attributes.get("binding_name")),
            )
            sequence_id = lexical_sequence_ids.get(identity_key)
        elif int(field_data.get("value_id", field_id)) in (
            inferred_nested_table_bases
        ):
            sequence_id = inferred_nested_table_bases[
                int(field_data.get("value_id", field_id))
            ]
        if sequence_id is None:
            continue
        sequence_memberships.append((
            int(data.get("value_id", node_id)),
            int(graph_obj.nodes[query_id].get("value_id", query_id)),
            int(sequence_id),
            isinstance(expression.ops[0], ast.NotIn),
        ))
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "indexed":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("base")
        key_id = authored_index_values(data)
        if base_id is None or key_id is None or base_id not in graph_obj:
            continue
        sequence_id, _storage_identity = table_sequence(base_id)
        if sequence_id is None:
            continue
        table_lookups.append((
            int(data.get("value_id", node_id)),
            key_id,
            int(sequence_id),
        ))
    # ``d.get(key, default)`` is the same lookup ``d[key]`` is -- the key's
    # token walked against the table -- differing only in what the absent
    # branch yields.  Recognising only ``indexed`` left ``get`` unclaimed, so
    # its result crossed every backend as a producerless argument.  The
    # authored default rides beside the lookup by result id.
    table_lookup_defaults: dict[int, Any] = {}
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "get":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("operand") or by_role.get("value")
        key_id = by_role.get("arg:0")
        if base_id is None or key_id is None or base_id not in graph_obj:
            continue
        sequence_id, _storage_identity = table_sequence(base_id)
        if sequence_id is None:
            continue
        result_id = int(data.get("value_id", node_id))
        key_id = int(graph_obj.nodes[key_id].get("value_id", key_id))
        table_lookups.append((result_id, key_id, int(sequence_id)))
        default_node = by_role.get("arg:1")
        if default_node is not None and default_node in graph_obj:
            default_data = graph_obj.nodes[default_node]
            literal = default_data.get("constant")
            if literal is None:
                literal = (
                    default_data.get("attributes") or {}
                ).get("value")
            if isinstance(literal, (int, float)) and not isinstance(
                literal, bool
            ):
                table_lookup_defaults[result_id] = float(literal)
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "indexedstore":
            continue
        if int(data.get("value_id", node_id)) in lexical_mapping_effects:
            # These writes are emitted at their authored control position.
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("base")
        key_id = authored_index_values(data)
        value_id = by_role.get("value")
        if None in (base_id, key_id, value_id) or base_id not in graph_obj:
            continue
        sequence_id, _storage_identity = table_sequence(base_id)
        if sequence_id is None:
            continue
        table_stores.append((
            int(data.get("value_id", node_id)),
            key_id,
            int(graph_obj.nodes[value_id].get("value_id", value_id)),
            int(sequence_id),
        ))
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if str(data.get("op") or data.get("type")).lower() != "delitem":
            continue
        by_role = {
            str(role): int(parent)
            for parent, role in (data.get("parents") or ())
        }
        base_id = by_role.get("base")
        key_id = (() if deletes_first_live_key(data)
                  else authored_index_values(data))
        if base_id is None or key_id is None or base_id not in graph_obj:
            continue
        sequence_value_id, storage_identity = table_sequence(base_id)
        if storage_identity is None:
            storage_identity = f"nested-table-value:{base_id}"
        table_deletions.append((
            int(data.get("value_id", node_id)),
            key_id,
            sequence_value_id,
            storage_identity,
        ))
    # A retained mapping ``pop`` is lowered by the control mutation identity,
    # not by this field-op table.  It nevertheless changes the table's
    # physical ABI: lookup-and-delete requires the same live-flag/tombstone
    # storage as authored ``del table[key]``.  Survey that structural fact
    # separately so it cannot accidentally emit a second deletion.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        if node_operation(data) != "pop":
            continue
        receiver_nodes = tuple(
            int(parent)
            for parent, role in (data.get("parents") or ())
            if str(role) in {"operand", "value", "object", "base", "receiver"}
            and parent in graph_obj
        )
        if not receiver_nodes:
            func_nodes = tuple(
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role) == "func" and parent in graph_obj
            )
            if len(func_nodes) != 1:
                continue
            func_data = graph_obj.nodes[func_nodes[0]]
            func_attributes = func_data.get("attributes") or {}
            if (
                node_operation(func_data) != "getattr"
                or str(func_attributes.get("attribute") or "") != "pop"
            ):
                continue
            receiver_nodes = tuple(
                int(parent)
                for parent, role in (func_data.get("parents") or ())
                if str(role) in {"value", "object", "base", "receiver"}
                and parent in graph_obj
            )
        if len(receiver_nodes) != 1:
            continue
        receiver_data = graph_obj.nodes[receiver_nodes[0]]
        receiver_attributes = receiver_data.get("attributes") or {}
        if receiver_attributes.get("aggregate_kind") == "dict":
            tombstone_sequence_ids.add(int(
                receiver_data.get("value_id", receiver_nodes[0])
            ))
        sequence_id, _storage_identity = table_sequence(receiver_nodes[0])
        if sequence_id is not None:
            tombstone_sequence_ids.add(int(sequence_id))
    # Runtime sequence replication (``[x] * count``) is a fill of resident
    # caller storage, not numerical multiplication and not a Python literal.
    for node_id in sorted(graph_obj.nodes(), key=lambda value: int(value)):
        data = graph_obj.nodes[node_id]
        attributes = data.get("attributes") or {}
        if attributes.get("producer_kind") != "sequence_replication":
            continue
        parents = tuple(data.get("parents") or ())
        sequence_parents = tuple(
            int(parent) for parent, role in parents
            if str(role) in {"lhs", "rhs"}
            and parent in graph_obj
            and (graph_obj.nodes[parent].get("attributes") or {}).get(
                "aggregate_kind"
            ) in {"list", "tuple"}
        )
        fill_ids = tuple(
            int(leaf_id)
            for parent in sequence_parents
            for leaf_id in (
                graph_obj.nodes[parent].get("attributes") or {}
            ).get("aggregate_leaf_value_ids", ())
        )
        count_ids = tuple(
            int(parent) for parent, role in parents
            if str(role) in {"lhs", "rhs"}
            and int(parent) not in sequence_parents
        )
        if len(fill_ids) != 1 or len(count_ids) != 1:
            continue
        fill_data = graph_obj.nodes.get(fill_ids[0], {})
        literal = (fill_data.get("attributes") or {}).get(
            "value", fill_data.get("constant")
        )
        if literal is not None and not isinstance(literal, (bool, int, float)):
            continue
        result_id = int(data.get("value_id", node_id))
        sequence_initializations.append((
            result_id,
            f"fill={literal!r};count={count_ids[0]}",
            1,
        ))
    # An unannotated local can still have a fixed-width row contract when the
    # authored mutation supplies a tuple record.  The mutation extractor has
    # already expanded that tuple into its exact leaf identities.  Reconcile
    # those leaves with the resident declaration before control lowering so a
    # list initially spelled ``[]`` is not frozen as a one-column sequence and
    # then rejected when ``append((left, right))`` reaches it.  Conflicting row
    # widths remain unresolved and are refused later; choosing one would invent
    # an ABI that the source does not state.
    declared_sequence_ids = {
        int(sequence_id) for sequence_id, *_rest in sequence_declarations
    }
    resident_by_value = {
        int(sequence_id): int(sequence_id)
        for sequence_id in declared_sequence_ids
    }
    for history in identity.values():
        history_ids = tuple(map(int, history))
        residents = tuple(
            value_id for value_id in history_ids
            if value_id in declared_sequence_ids
        )
        if len(set(residents)) != 1:
            continue
        resident = residents[0]
        resident_by_value.update({
            value_id: resident for value_id in history_ids
        })
    observed_row_widths: dict[int, set[int]] = {}
    for mutation in _sequence_append_call_mutations(graph_obj):
        if mutation.argument_kind != "row":
            continue
        resident = resident_by_value.get(int(mutation.sequence_value_id))
        width = len(tuple(mutation.argument_value_ids))
        if resident is not None and width > 1:
            observed_row_widths.setdefault(int(resident), set()).add(width)
    inferred_row_widths = {
        sequence_id: next(iter(widths))
        for sequence_id, widths in observed_row_widths.items()
        if len(widths) == 1
    }
    sequence_declarations = [
        (
            sequence_id,
            policy,
            max(column_count, inferred_row_widths.get(sequence_id, 1)),
            writable,
        )
        for sequence_id, policy, column_count, writable
        in sequence_declarations
    ]

    key_width_by_sequence: dict[int, int] = {}
    for _effect_or_result, key_ids, sequence_id, *_rest in (
        *table_lookups, *table_stores, *table_deletions
    ):
        if sequence_id is None:
            continue
        key_width_by_sequence[int(sequence_id)] = max(
            key_width_by_sequence.get(int(sequence_id), 1),
            len(key_ids) if isinstance(key_ids, tuple) else 1,
        )
    sequence_declarations = [
        (
            sequence_id,
            policy,
            (
                max(
                    column_count,
                    key_width_by_sequence.get(sequence_id, 1) + 1,
                )
                if policy == "unique" and column_count > 1
                else column_count
            ),
            writable,
        )
        for sequence_id, policy, column_count, writable
        in sequence_declarations
    ]
    # A record-field dict is a table exactly as a local one is, but declare it
    # only where a table operation actually addresses it: a declaration
    # materializes anonymous descriptor storage into the frame, and doing that
    # in functions that merely ITERATE the mapping displaced their public-span
    # correlation for every unrelated rank-2 field.
    field_table_ids = {
        int(sequence_id)
        for sequence_id in (
            *field_sequence_ids.values(),
            *keyed_field_sequence_ids.values(),
        )
    }
    referenced_table_ids = {
        int(sequence_id)
        for _result, _query, sequence_id in table_lookups
    } | {
        int(sequence_id)
        for _effect, _key, _value, sequence_id in table_stores
    }
    declared_ids = {
        int(sequence_id) for sequence_id, *_rest in sequence_declarations
    }
    # A contract-declared keyed field is writable exactly when the contract
    # says so (``mutable: true``): its store helper is refused as a read-only
    # destination otherwise.
    writable_keyed_table_ids = {
        int(sequence_id)
        for field_name, sequence_id in keyed_field_sequence_ids.items()
        if str(field_name) in keyed_writable_fields
    }
    sequence_declarations.extend(
        (int(sequence_id), "unique", 2, int(sequence_id) in writable_keyed_table_ids)
        for sequence_id in sorted(field_table_ids & referenced_table_ids)
        if int(sequence_id) not in declared_ids
    )
    for sequence_id, policy, column_count, writable in sequence_declarations:
        commit_sequence_contract(
            sequence_contract_scope,
            int(sequence_id),
            str(policy),
            int(column_count),
            bool(writable),
            source="shell sequence declaration",
        )
    graph_obj.graph["field_write_state_cells"] = tuple(write_state_cells)
    return (
        self_value_id,
        tuple(field_ops),
        const_sources,
        len(fields),
        fields,
        record_identity,
        tuple(dict.fromkeys(sequence_initializations)),
        tuple(
            (slot_of[alias], slot_of[target])
            for alias, target in field_aliases.items()
            if alias in slot_of and target in slot_of
        ),
        tuple(dict.fromkeys(sequence_declarations)),
        tuple(dict.fromkeys(sequence_memberships)),
        tuple(dict.fromkeys(table_lookups)),
        dict(table_lookup_defaults),
        tuple(dict.fromkeys(table_stores)),
        tuple(dict.fromkeys(table_deletions)),
        tuple(sorted(retained_sequence_ids)),
        tuple(sorted(nested_sequence_ids)),
        tuple(sorted(nested_tensor_sequence_ids)),
        tuple(sorted(
            (slot, identity, value_id)
            for slot, (identity, value_id) in nested_record_fields.items()
        )),
        tuple(sorted(tombstone_sequence_ids)),
    )
