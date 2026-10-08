"""Sequence row projection: nested row projections, record sequence projection bindings, sequence row operations (relocated from fortran_c_shell)"""

from __future__ import annotations
import ast
from typing import (
    Any,
    Iterable,
    Mapping,
)
from dataclasses import replace

from .authored_record_abi import (
    _graph_sequence_record_abi,
    _sequence_record_binding_value_ids,
)
from .resident_sequence_materialization import _loop_carried_storage_aliases


def _nested_row_projection_ops(
    graph_obj: Any, control: Any,
) -> tuple[tuple[int, int, int, str], ...]:
    """Find fixed-column reads from a projected/destructured loop row."""

    aliases_by_value: dict[int, frozenset[int]] = {}
    for history in (graph_obj.graph.get("identity_table") or {}).values():
        values = frozenset(map(int, history))
        for value_id in values:
            aliases_by_value[int(value_id)] = values
    operations = []
    for _iterable, target_id, induction, _projection in (
        getattr(control, "projected_iterable_bindings", ())
    ):
        aliases = aliases_by_value.get(
            int(target_id), frozenset((int(target_id),))
        )
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("type")) not in {"Indexed", "indexed"}:
                continue
            roles = {
                str(role): int(parent)
                for parent, role in (data.get("parents") or ())
            }
            base = roles.get("base")
            index = roles.get("index")
            if base not in aliases or index is None or index not in graph_obj:
                continue
            index_data = graph_obj.nodes[index]
            expression = index_data.get("expr_obj")
            literal = (
                expression.value
                if isinstance(expression, ast.Constant)
                else (index_data.get("attributes") or {}).get(
                    "value", index_data.get("value")
                )
            )
            if not isinstance(literal, int) or isinstance(literal, bool):
                continue
            operations.append((
                int(target_id), int(literal),
                int(data.get("value_id", node_id)), str(induction),
            ))
    return tuple(dict.fromkeys(operations))


def _rewrite_optional_row_handle_none_predicate(
    expression: Any, handle_ids: Iterable[int],
) -> Any:
    """Compare optional record row handles with their physical -1 sentinel."""

    if expression is None:
        return None
    handles = set(map(int, handle_ids))
    operands = tuple(
        _rewrite_optional_row_handle_none_predicate(operand, handles)
        for operand in expression.operands
    )
    rewritten = replace(expression, operands=operands)
    if rewritten.op not in {"eq", "ne"} or len(operands) != 2:
        return rewritten
    for handle_index, none_index in ((0, 1), (1, 0)):
        handle = operands[handle_index]
        none = operands[none_index]
        if (
            handle.value_id is None
            or int(handle.value_id) not in handles
            or none.op != "const"
            or none.literal is not None
        ):
            continue
        replaced = list(operands)
        replaced[none_index] = replace(none, literal=-1)
        return replace(rewritten, operands=tuple(replaced))
    return rewritten


def _record_sequence_projection_bindings(
    graph_obj: Any, control: Any,
) -> tuple[
    Any,
    tuple[tuple[int, int, str, object], ...],
    tuple[tuple[int, str, str], ...],
]:
    """Project fields of ``Sequence[Record]`` rows at their loop binding.

    The authored parameter is a columnar resident sequence. A lexical loop
    target names the current row, while each ``target.field`` names one exact
    column load at the same induction. This preserves the ordinary intuitive
    source and gives downstream regions/calls real in-loop producers instead
    of promoting field values to anonymous public inputs.
    """

    records = _graph_sequence_record_abi(graph_obj)
    identities = graph_obj.graph.get("identity_table") or {}
    aliases_by_value: dict[int, frozenset[int]] = {}
    for history in identities.values():
        aliases = frozenset(map(int, history))
        for value_id in aliases:
            aliases_by_value[int(value_id)] = aliases
    parameter_by_sequence: dict[int, tuple[str, Mapping[str, Any]]] = {}
    for parameter_name, record in records.items():
        for value_id in _sequence_record_binding_value_ids(
            graph_obj, parameter_name,
        ):
            parameter_by_sequence[int(value_id)] = (
                str(parameter_name), record,
            )
    row_loop_bindings = tuple(dict.fromkeys((
        *tuple(getattr(control, "iterable_bindings", ())),
        *tuple(
            (int(iterable_id), int(target_id), str(induction))
            for iterable_id, target_id, induction, projection
            in getattr(control, "projected_iterable_bindings", ())
            if projection is None
        ),
    )))
    bindings: list[tuple[int, int, str, object]] = []
    fields: list[tuple[int, str, str]] = []
    direct_rows: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for iterable_id, target_id, _induction in row_loop_bindings:
        selected = parameter_by_sequence.get(int(iterable_id))
        if selected is None:
            iterable_aliases = aliases_by_value.get(
                int(iterable_id), frozenset((int(iterable_id),))
            )
            selected = next((
                record for candidate, record in parameter_by_sequence.items()
                if candidate in iterable_aliases
            ), None)
        if selected is not None:
            _parameter_name, record = selected
            direct_rows[int(target_id)] = (int(iterable_id), record)

    # A filtered comprehension over a record sequence stores integer row
    # handles into its derived resident sequence.  Recover that relationship
    # from the exact mutation argument identity; no Python row object is
    # reconstructed and no name spelling participates in the correlation.
    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock,
        SequenceMutationBlock, SequenceQueryBlock, WhileBlock,
    )

    mutations = []
    queries = []

    def gather_mutations(block):
        if isinstance(block, SequenceMutationBlock):
            mutations.append(block.mutation)
        elif isinstance(block, SequenceQueryBlock):
            queries.append(block)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                gather_mutations(child)
        elif isinstance(block, ConditionalBlock):
            gather_mutations(block.body)
            if block.orelse is not None:
                gather_mutations(block.orelse)
        elif isinstance(block, (LoopBlock, WhileBlock)):
            mutations.extend(block.sequence_mutations)
            if isinstance(block, WhileBlock):
                gather_mutations(block.condition)
            gather_mutations(block.body)
        elif isinstance(block, CallBlock):
            gather_mutations(block.callee)

    gather_mutations(control.root)
    derived_sequences: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for mutation in mutations:
        if str(mutation.operator) not in {"append", "add"}:
            continue
        arguments = set(map(int, mutation.argument_value_ids))
        for target_id, origin in direct_rows.items():
            target_aliases = aliases_by_value.get(
                int(target_id), frozenset((int(target_id),))
            )
            if arguments.intersection(target_aliases):
                derived_sequences[int(mutation.sequence_value_id)] = origin
                effect_id = int(mutation.effect_node_id)
                if effect_id in graph_obj:
                    # Hierarchical composition namespaces resident storage,
                    # while a later authored loop still names the source
                    # materializer result.  The mutation's retained effect
                    # node is their exact correlation receipt.
                    derived_sequences[int(graph_obj.nodes[effect_id].get(
                        "value_id", effect_id
                    ))] = origin

    optional_projections: dict[
        int, tuple[tuple[int, int, int, int, str], ...]
    ] = {}
    # ``mapping.pop(key, None)`` has the same physical optional-handle ABI as
    # ``first_or_default``: its lookup result is -1 when absent.  Register the
    # exact effect identity even when no parameter-record projection exists so
    # an authored ``result is None`` predicate is rewritten consistently.
    for mutation in mutations:
        if (
            str(mutation.operator) == "pop"
            and mutation.argument_kind == "mapping_pop_default_none"
        ):
            optional_projections[int(mutation.effect_node_id)] = ()
    optional_query_calls: set[int] = set()
    for query in queries:
        if str(query.operation) != "first_or_default":
            continue
        origin = derived_sequences.get(int(query.sequence_value_id))
        if origin is None:
            continue
        origin_sequence_id, record = origin
        receivers = {
            int(query.result_value_id), *map(int, query.result_alias_ids)
        }
        physical_columns = {}
        physical_column = 0
        for field_name, receipt in dict(record.get("fields") or {}).items():
            if bool(receipt.get("optional")):
                physical_column += 1
            physical_columns[str(field_name)] = (
                physical_column, str(receipt.get("dtype") or "unknown")
            )
            physical_column += 1
        projected = []
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("op") or data.get("type") or "").casefold() != "getattr":
                continue
            attribute = str(
                (data.get("attributes") or {}).get("attribute") or ""
            )
            if attribute not in physical_columns or not any(
                int(graph_obj.nodes[parent].get("value_id", parent)) in receivers
                and str(role) in {"value", "object", "base", "receiver"}
                for parent, role in (data.get("parents") or ())
                if parent in graph_obj
            ):
                continue
            column, dtype = physical_columns[attribute]
            result_id = int(data.get("value_id", node_id))
            projected.append((
                int(origin_sequence_id), int(query.result_value_id),
                result_id, int(column), str(dtype),
            ))
            fields.append((result_id, attribute, str(dtype)))
        if projected:
            optional_projections[int(query.result_value_id)] = tuple(projected)
            if query.source_call_node_id is not None:
                optional_query_calls.add(int(query.source_call_node_id))

    for iterable_id, target_id, induction in row_loop_bindings:
        selected = parameter_by_sequence.get(int(iterable_id))
        if selected is None:
            iterable_aliases = aliases_by_value.get(
                int(iterable_id), frozenset((int(iterable_id),))
            )
            selected = next((
                record for candidate, record in parameter_by_sequence.items()
                if candidate in iterable_aliases
            ), None)
        derived = False
        if selected is None:
            selected = derived_sequences.get(int(iterable_id))
            derived = selected is not None
        if selected is None:
            continue
        if derived:
            origin_sequence_id, record = selected
        else:
            _parameter_name, record = selected
            origin_sequence_id = int(iterable_id)
        field_contracts = tuple(dict(record.get("fields") or {}).items())
        field_columns = {}
        physical_column = 0
        for field_name, field_receipt in field_contracts:
            if bool(field_receipt.get("optional")):
                # Presence precedes value.  Attribute access selects the value
                # column; an explicit ``is None`` field test can consume the
                # adjacent presence column without overloading its payload.
                physical_column += 1
            field_columns[str(field_name)] = physical_column
            physical_column += 1
        # Loop targets with the same authored spelling are distinct SSA
        # identities (two separate ``for row in ...`` bindings are the common
        # case).  The graph's GetAttr receiver already carries the exact target
        # value; consulting the scope-free name history here conflates those
        # loops and places a later field load in the earlier coordinator.
        target_aliases = frozenset((int(target_id),))
        if not derived:
            # The conceptual row itself is a stable integer handle.  Field
            # values are loaded from the source columns below; retaining a
            # second flat-element load for the row would confuse column zero
            # with the record identity when a comprehension republishes it.
            bindings.append((
                int(origin_sequence_id), int(target_id), str(induction),
                "induction",
            ))
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("op") or data.get("type") or "").casefold() != "getattr":
                continue
            attribute = str(
                (data.get("attributes") or {}).get("attribute") or ""
            )
            if attribute not in field_columns or not any(
                int(graph_obj.nodes[parent].get("value_id", parent))
                in target_aliases
                and str(role) in {"value", "object", "base", "receiver"}
                for parent, role in (data.get("parents") or ())
                if parent in graph_obj
            ):
                continue
            result_id = int(data.get("value_id", node_id))
            receipt = dict(next(
                field_receipt
                for field_name, field_receipt in field_contracts
                if str(field_name) == attribute
            ))
            bindings.append((
                int(origin_sequence_id), result_id, str(induction),
                (
                    (
                        "column_at_value", int(field_columns[attribute]),
                        int(target_id),
                    )
                    if derived else int(field_columns[attribute])
                ),
            ))
            fields.append((
                result_id, attribute, str(receipt.get("dtype") or "unknown")
            ))
    def expression_values(expression):
        if expression is None:
            return frozenset()
        return frozenset((
            *((int(expression.value_id),) if expression.value_id is not None else ()),
            *(value for operand in expression.operands for value in expression_values(operand)),
        ))

    def attach_optional(block):
        if isinstance(block, SequenceQueryBlock):
            return replace(
                block,
                row_handle=(
                    block.source_call_node_id is not None
                    and int(block.source_call_node_id) in optional_query_calls
                ),
            )
        if isinstance(block, SequenceBlock):
            return replace(
                block,
                blocks=tuple(attach_optional(child) for child in block.blocks),
            )
        if isinstance(block, ConditionalBlock):
            body = attach_optional(block.body)
            orelse = (
                None if block.orelse is None
                else attach_optional(block.orelse)
            )
            predicate_expression = (
                _rewrite_optional_row_handle_none_predicate(
                    block.predicate_expression, optional_projections
                )
            )
            values = expression_values(predicate_expression)
            entry = tuple(
                projection
                for handle_id, projections in optional_projections.items()
                if int(handle_id) in values
                and predicate_expression is not None
                and predicate_expression.op == "ne"
                for projection in projections
            )
            return replace(
                block, body=body, orelse=orelse,
                predicate_expression=predicate_expression,
                entry_record_projections=tuple(dict.fromkeys((
                    *block.entry_record_projections, *entry,
                ))),
            )
        if isinstance(block, LoopBlock):
            return replace(block, body=attach_optional(block.body))
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=attach_optional(block.condition),
                body=attach_optional(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=attach_optional(block.callee))
        return block

    control = replace(control, root=attach_optional(control.root))
    return (
        control,
        tuple(dict.fromkeys(bindings)),
        tuple(dict.fromkeys(fields)),
    )


def _sequence_row_operations(
    graph_obj: Any,
    sequence_declarations: Iterable[tuple[int, str, int, bool]],
) -> tuple[tuple[Any, ...], ...]:
    """Recover positional reads/writes of fixed-width sequence rows."""

    widths = {
        int(sequence_id): int(column_count)
        for sequence_id, _policy, column_count, _writable
        in sequence_declarations
        if int(column_count) > 1
    }
    if not widths:
        return ()
    identities = graph_obj.graph.get("identity_table") or {}
    resident_by_value = {
        int(value_id): int(sequence_id)
        for sequence_id in widths
        for history in identities.values()
        if int(sequence_id) in set(map(int, history))
        for value_id in history
    }
    resident_by_value.update({sequence_id: sequence_id for sequence_id in widths})
    changed = True
    aliases = _loop_carried_storage_aliases(graph_obj)
    while changed:
        changed = False
        for alias_id, source_id in aliases.items():
            resident = resident_by_value.get(int(source_id))
            if resident is not None and resident_by_value.get(int(alias_id)) != resident:
                resident_by_value[int(alias_id)] = resident
                changed = True
        for node_id, data in graph_obj.nodes(data=True):
            if str(data.get("op") or data.get("type") or "").casefold() not in {
                "phi", "loopresult", "loopexit",
            }:
                continue
            residents = {
                resident_by_value[int(graph_obj.nodes[parent].get(
                    "value_id", parent
                ))]
                for parent, role in data.get("parents") or ()
                if str(role) not in {"control", "test"}
                and parent in graph_obj
                and int(graph_obj.nodes[parent].get(
                    "value_id", parent
                )) in resident_by_value
            }
            if len(residents) == 1:
                value_id = int(data.get("value_id", node_id))
                resident = next(iter(residents))
                if resident_by_value.get(value_id) != resident:
                    resident_by_value[value_id] = resident
                    changed = True

    def roles(data: Mapping[str, Any]) -> dict[str, int]:
        return {
            str(role): int(parent)
            for parent, role in data.get("parents") or ()
            if parent in graph_obj
        }

    def value_id(node_id: int) -> int:
        return int(graph_obj.nodes[node_id].get("value_id", node_id))

    def integer_literal(node_id: int) -> int | None:
        data = graph_obj.nodes[node_id]
        expression = data.get("expr_obj")
        literal = (
            expression.value if isinstance(expression, ast.Constant)
            else (data.get("attributes") or {}).get(
                "value", data.get("constant")
            )
        )
        return (
            int(literal)
            if isinstance(literal, int) and not isinstance(literal, bool)
            else None
        )

    operations: list[tuple[Any, ...]] = []
    for node_id, data in sorted(graph_obj.nodes(data=True)):
        kind = str(data.get("op") or data.get("type") or "").casefold()
        node_roles = roles(data)
        if kind == "indexed":
            outer_id = node_roles.get("base")
            column_node = node_roles.get("index")
            if outer_id is None or column_node is None:
                continue
            outer = graph_obj.nodes[outer_id]
            if str(outer.get("op") or outer.get("type") or "").casefold() != "indexed":
                continue
            outer_roles = roles(outer)
            base_node = outer_roles.get("base")
            row_index_node = outer_roles.get("index")
            if base_node is None or row_index_node is None:
                continue
            sequence_id = resident_by_value.get(value_id(base_node))
            column = integer_literal(column_node)
            if (
                sequence_id is None or column is None or column < 0
                or column >= widths[sequence_id]
            ):
                continue
            operations.append((
                "load", value_id(node_id), int(sequence_id),
                value_id(row_index_node), integer_literal(row_index_node),
                int(column), value_id(outer_id),
            ))
        elif kind == "indexedstore":
            base_node = node_roles.get("base")
            row_index_node = node_roles.get("index")
            row_value_node = node_roles.get("value")
            if base_node is None or row_index_node is None or row_value_node is None:
                continue
            sequence_id = resident_by_value.get(value_id(base_node))
            row_attributes = graph_obj.nodes[row_value_node].get("attributes") or {}
            leaves = tuple(map(
                int, row_attributes.get("aggregate_leaf_value_ids", ())
            ))
            if (
                sequence_id is None
                or row_attributes.get("aggregate_kind") != "tuple"
                or len(leaves) != widths[sequence_id]
            ):
                continue
            operations.append((
                "store", value_id(node_id), int(sequence_id),
                value_id(row_index_node), integer_literal(row_index_node),
                leaves, value_id(row_value_node),
            ))
    return tuple(operations)
