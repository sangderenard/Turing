"""Exact source proof for a read of one transferred keyed mapping entry."""

import ast

from .concordance_declarations import (
    PROCESS_GRAPH_OPERANDS, REDUCER_FIELD_STATE, REDUCTION,
    SOURCE_ACTIVITY_CALL, SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_NODE,
    SOURCE_AGGREGATE_LAYOUT, SOURCE_CALL_FIELD_STATE_TRANSFER,
    SOURCE_LITERAL_VALUE, SOURCE_MAPPING_GET_COPY_STATE_DEMAND,
    SOURCE_MAPPING_GET_KEY_VERSION_DEMAND, SOURCE_MAPPING_GET_READ,
    SOURCE_SEQUENCE_MUTATION,
    FieldState, FieldStateKind,
)
from .control_activity import binding, normal_mapping_get_prefix
from .identity_concordance import Derived, Mode, Ref, current_identity_book


def reaching_key_version(book, transfer_fact, read_key_value):
    """Read the exact final-to-earlier field revisions for one keyed get."""
    final_state, final_mutation = transfer_fact[6:8]
    if (final_state.page != REDUCER_FIELD_STATE
            or final_mutation.page != SOURCE_SEQUENCE_MUTATION):
        return None
    publisher_scope = final_mutation.row[0]
    state_page = book.page(REDUCER_FIELD_STATE)
    mutation_page = book.page(SOURCE_SEQUENCE_MUTATION)
    layout_page = book.page(SOURCE_AGGREGATE_LAYOUT)
    literal_page = book.page(SOURCE_LITERAL_VALUE)
    current, hops, visited = final_state, [], set()
    while current not in visited:
        visited.add(current)
        state = state_page.cell(current.row, current.column)
        if not isinstance(state, FieldState):
            return None
        if state.kind is FieldStateKind.OBSERVED:
            return None
        if (state.kind is not FieldStateKind.ELEMENT_WRITTEN
                or not isinstance(state.effect, Ref)):
            return None
        effect_id = int(state.effect.row[-1])
        mutation = book.latest_ref(
            SOURCE_SEQUENCE_MUTATION, (publisher_scope, effect_id),
        )
        mutation_fact = (mutation_page.cell(mutation.row, mutation.column)
                         if mutation is not None else None)
        if (mutation_fact is None or len(mutation_fact) != 5
                or mutation_fact[1:3] != ('setitem', 'unique')
                or mutation_fact[4] != 'mapping'
                or len(mutation_fact[3]) != 2):
            return None
        key_id, value_id = map(int, mutation_fact[3])
        layout = book.latest_ref(SOURCE_AGGREGATE_LAYOUT,
                                 (current.row[0], effect_id))
        layout_fact = (layout_page.cell(layout.row, layout.column)
                       if layout is not None else None)
        if (layout_fact is None or layout_fact.aggregate_kind != 'dict'
                or len(layout_fact.members) != 2
                or tuple(member.value_id for member in layout_fact.members)
                != (key_id, value_id)):
            return None
        key = book.latest_ref(SOURCE_LITERAL_VALUE, (publisher_scope, key_id))
        key_value = (literal_page.cell(key.row, key.column)
                     if key is not None else None)
        if not isinstance(key_value, (str, int, float, bool, bytes)):
            return None
        if not hops and mutation != final_mutation:
            return None
        if key_value == read_key_value:
            return (current, mutation, layout, key,
                    layout_fact.members[1].value_cell, tuple(hops))
        # A later equal key overwrites the earlier version. Known unequal
        # literal keys preserve it; unknown key equality proves no version.
        origin = tuple(source for source, _stage in book.edges_into(current)
                       if source.page == REDUCER_FIELD_STATE
                       and source.row != current.row)
        if len(origin) != 1:
            return None
        predecessor = tuple(
            source for source, _stage in book.edges_into(origin[0])
            if source.page == REDUCER_FIELD_STATE
            and source.row == origin[0].row
        )
        if len(predecessor) != 1:
            return None
        prior = predecessor[0]
        canonical = tuple(
            Ref(REDUCER_FIELD_STATE, current.row, column)
            for column, _fact in state_page.history(current.row)
            if prior in {source for source, _stage in book.edges_into(
                Ref(REDUCER_FIELD_STATE, current.row, column))}
        )
        if len(canonical) != 1:
            return None
        hops.append((current, origin[0], prior, canonical[0],
                     mutation, layout, key))
        current = canonical[0]
    return None


def post_mapping_get_reads(function_table, *, book=None):
    """Publish a present-key read contract without typing the optional result."""
    from ..common.tensors.topological_reducer import node_identity_cell

    book = current_identity_book() if book is None else book
    transfers = book.page(SOURCE_CALL_FIELD_STATE_TRANSFER)
    entries = tuple(entry for entry in function_table
                    if getattr(entry, 'graph', None) is not None)
    by_scope = {
        tuple(entry.graph.G.graph.get('authored_function_identity') or ()): entry.graph
        for entry in entries
    }
    for transfer_row in transfers.rows():
        consumer_scope = transfer_row[3]
        wrapper = by_scope.get(consumer_scope)
        if wrapper is None:
            continue
        transfer = book.latest_ref(SOURCE_CALL_FIELD_STATE_TRANSFER, transfer_row)
        transfer_fact = transfers.cell(transfer.row, transfer.column)
        _completion, _corridor, _publisher_alias, _consumer_alias, _field, _exit, written_state, store_mutation, observed_state = transfer_fact
        if store_mutation.page != SOURCE_SEQUENCE_MUTATION:
            continue
        store_fact = book.page(SOURCE_SEQUENCE_MUTATION).cell(
            store_mutation.row, store_mutation.column,
        )
        if (len(store_fact) != 5 or store_fact[1:3] != ('setitem', 'unique')
                or store_fact[4] != 'mapping' or len(store_fact[3]) != 2):
            continue
        store_key_id, store_value_id = map(int, store_fact[3])
        layout = book.latest_ref(
            SOURCE_AGGREGATE_LAYOUT,
            (written_state.row[0], int(store_mutation.row[1])),
        )
        if layout is None:
            continue
        layout_fact = book.page(SOURCE_AGGREGATE_LAYOUT).cell(
            layout.row, layout.column,
        )
        if (layout_fact.aggregate_kind != 'dict'
                or len(layout_fact.members) != 2
                or tuple(member.value_id for member in layout_fact.members)
                != (store_key_id, store_value_id)):
            continue
        stored_key = book.latest_ref(
            SOURCE_LITERAL_VALUE, (store_mutation.row[0], store_key_id),
        )
        observed = book.page(REDUCER_FIELD_STATE).cell(
            observed_state.row, observed_state.column,
        ) if observed_state.page == REDUCER_FIELD_STATE else None
        if not isinstance(observed, FieldState) or observed.kind is not FieldStateKind.OBSERVED:
            continue
        for call_id, data in wrapper.G.nodes(data=True):
            expression = data.get('expr_obj')
            if not (isinstance(expression, ast.Call)
                    and isinstance(expression.func, ast.Attribute)
                    and expression.func.attr == 'get'
                    and len(expression.args) == 1 and not expression.keywords):
                continue
            call_id = int(call_id)
            roles = dict((str(role), int(parent)) for parent, role in
                         data.get('parents') or ())
            if set(roles) != {'operand', 'arg:0'}:
                continue
            base_id, key_id = roles['operand'], roles['arg:0']
            call_cell = node_identity_cell(wrapper, call_id, book=book)
            base_cell = node_identity_cell(wrapper, base_id, book=book)
            key_cell = node_identity_cell(wrapper, key_id, book=book)
            if (call_cell is None or base_cell != observed.value
                    or key_cell is None):
                continue
            operand = book.latest_ref(
                PROCESS_GRAPH_OPERANDS,
                (wrapper.G.graph.get('operand_position_scope'), call_id),
            )
            if operand is None or set(book.page(PROCESS_GRAPH_OPERANDS).cell(
                    operand.row, operand.column)) != {
                        (base_cell, 'operand'), (key_cell, 'arg:0'),
                    }:
                continue
            read_key = book.latest_ref(
                SOURCE_LITERAL_VALUE, (consumer_scope, key_id),
            )
            if read_key is None:
                continue
            read_key_value = book.page(SOURCE_LITERAL_VALUE).cell(
                read_key.row, read_key.column,
            )
            read_mutation = book.latest_ref(
                SOURCE_SEQUENCE_MUTATION, (consumer_scope, call_id),
            )
            read_fact = (book.page(SOURCE_SEQUENCE_MUTATION).cell(
                read_mutation.row, read_mutation.column,
            ) if read_mutation is not None else None)
            if read_fact != (base_id, 'unclassified_call:get', 'unique',
                             (key_id,), 'mapping'):
                continue
            version = reaching_key_version(book, transfer_fact, read_key_value)
            if version is None:
                if stored_key is None:
                    causes = (
                        transfer, written_state, store_mutation, layout,
                        layout_fact.members[0].value_cell, read_key,
                        observed_state, call_cell, base_cell, key_cell,
                        operand, read_mutation,
                    )
                    book.post(
                        SOURCE_MAPPING_GET_KEY_VERSION_DEMAND,
                        (consumer_scope, call_id, str(transfer_row[5])),
                        ('later_store_key_equality_unproved', *causes),
                        stage=REDUCTION, provenance=Derived(causes),
                        mode=Mode.CONCORD,
                    )
                continue
            (selected_state, store_mutation, layout, stored_key,
             stored_value, version_hops) = version
            layout_fact = book.page(SOURCE_AGGREGATE_LAYOUT).cell(
                layout.row, layout.column,
            )
            try:
                call_points, call_binding = binding(
                    wrapper, call_id, book=book, scope=consumer_scope,
                )
            except (ValueError, KeyError):
                continue
            if call_points is None:
                continue
            call_enter = call_points[0]
            activity_call = book.latest_ref(
                SOURCE_ACTIVITY_CALL, (consumer_scope, call_enter.row[1]),
            )
            if (activity_call is None or book.page(SOURCE_ACTIVITY_CALL).cell(
                    activity_call.row, activity_call.column) != (1, ())):
                continue
            method_ids = tuple(
                int(node) for node, member in wrapper.G.nodes(data=True)
                if member.get('expr_obj') is expression.func
            )
            if len(method_ids) != 1:
                continue
            method_cell = node_identity_cell(wrapper, method_ids[0], book=book)
            method_points, method_binding = binding(
                wrapper, method_ids[0], book=book, scope=consumer_scope,
            )
            method_edge = (None if method_points is None else book.latest_ref(
                SOURCE_ACTIVITY_EDGE,
                (consumer_scope, call_enter, method_points[0], 'flow'),
            ))
            if method_cell is None or method_edge is None or method_binding is None:
                continue
            starts = tuple(
                book.latest_ref(SOURCE_ACTIVITY_NODE, row)
                for row in book.page(SOURCE_ACTIVITY_NODE).scope_rows(consumer_scope)
                if row[2] == 'enter'
                and book.page(SOURCE_ACTIVITY_NODE).latest(row) == ('FunctionDef', ())
            )
            prefixes = tuple(
                prefix for start in starts
                if (prefix := normal_mapping_get_prefix(
                    book, consumer_scope, start, call_enter)) is not None
            )
            if len(prefixes) != 1:
                continue
            prefix_nodes, prefix_edges = prefixes[0]
            version_proof = (
                ((selected_state, version_hops),) if version_hops else ()
            )
            version_cells = tuple(dict.fromkeys((
                *((selected_state,) if version_hops else ()),
                *(cell for hop in version_hops for cell in hop),
            )))
            book.post(
                SOURCE_MAPPING_GET_READ,
                (consumer_scope, call_id, str(transfer_row[5])),
                (transfer, store_mutation, layout, stored_key, read_key,
                 stored_value, observed_state,
                 call_cell, method_cell, base_cell, key_cell, operand,
                 read_mutation, call_binding, activity_call, method_binding,
                 method_edge, prefix_nodes, prefix_edges, *version_proof),
                stage=REDUCTION,
                provenance=Derived((
                    transfer, store_mutation, layout, stored_key, read_key,
                    stored_value, observed_state,
                    call_cell, method_cell, base_cell, key_cell, operand,
                    read_mutation, call_binding, activity_call,
                    method_binding, method_edge, *prefix_nodes, *prefix_edges,
                    *version_cells,
                )), mode=Mode.CONCORD,
            )
    _post_copy_state_demands(transfers, by_scope, book=book)


def _post_copy_state_demands(transfers, by_scope, *, book):
    """Name the missing copy contract at the exact transferred field read."""
    from ..common.tensors.topological_reducer import node_identity_cell

    for transfer_row in transfers.rows():
        consumer_scope = transfer_row[3]
        wrapper = by_scope.get(consumer_scope)
        if wrapper is None:
            continue
        transfer = book.latest_ref(SOURCE_CALL_FIELD_STATE_TRANSFER, transfer_row)
        observed_state = transfers.cell(transfer.row, transfer.column)[8]
        observed = book.page(REDUCER_FIELD_STATE).cell(
            observed_state.row, observed_state.column,
        ) if observed_state.page == REDUCER_FIELD_STATE else None
        if not isinstance(observed, FieldState) or observed.kind is not FieldStateKind.OBSERVED:
            continue
        for copy_id, copy_data in wrapper.G.nodes(data=True):
            copy_expr = copy_data.get('expr_obj')
            if not (isinstance(copy_expr, ast.Call)
                    and isinstance(copy_expr.func, ast.Attribute)
                    and copy_expr.func.attr == 'copy'
                    and not copy_expr.args and not copy_expr.keywords):
                continue
            copy_id = int(copy_id)
            copy_cell = node_identity_cell(wrapper, copy_id, book=book)
            copy_operands = book.latest_ref(PROCESS_GRAPH_OPERANDS, (
                wrapper.G.graph.get('operand_position_scope'), copy_id,
            ))
            if (copy_operands is None or book.page(PROCESS_GRAPH_OPERANDS).cell(
                    copy_operands.row, copy_operands.column,
            ) != ((observed.value, 'operand'),)):
                continue
            copy_mutation = book.latest_ref(
                SOURCE_SEQUENCE_MUTATION, (consumer_scope, copy_id),
            )
            if (copy_mutation is None or book.page(SOURCE_SEQUENCE_MUTATION).cell(
                    copy_mutation.row, copy_mutation.column,
            ) != (observed.value.row[-1], 'unclassified_call:copy',
                  'unique', (), 'mapping')):
                continue
            try:
                _copy_points, copy_binding = binding(
                    wrapper, copy_id, book=book, scope=consumer_scope,
                )
            except (ValueError, KeyError):
                continue
            if copy_binding is None:
                continue
            for get_id, get_data in wrapper.G.nodes(data=True):
                get_expr = get_data.get('expr_obj')
                if not (isinstance(get_expr, ast.Call)
                        and isinstance(get_expr.func, ast.Attribute)
                        and get_expr.func.attr == 'get'
                        and len(get_expr.args) == 1 and not get_expr.keywords):
                    continue
                get_id = int(get_id)
                get_cell = node_identity_cell(wrapper, get_id, book=book)
                get_operands = book.latest_ref(PROCESS_GRAPH_OPERANDS, (
                    wrapper.G.graph.get('operand_position_scope'), get_id,
                ))
                key_cells = (() if get_operands is None else tuple(
                    cell for cell, role in book.page(PROCESS_GRAPH_OPERANDS).cell(
                        get_operands.row, get_operands.column,
                    ) if role == 'arg:0'
                ))
                if (get_operands is None or len(key_cells) != 1
                        or set(book.page(PROCESS_GRAPH_OPERANDS).cell(
                            get_operands.row, get_operands.column,
                        )) != {(copy_cell, 'operand'), (key_cells[0], 'arg:0')}):
                    continue
                try:
                    _get_points, get_binding = binding(
                        wrapper, get_id, book=book, scope=consumer_scope,
                    )
                except (ValueError, KeyError):
                    continue
                if get_binding is None:
                    continue
                row = (consumer_scope, get_id, str(transfer_row[5]))
                if book.latest_ref(SOURCE_MAPPING_GET_READ, row) is not None:
                    continue
                causes = (transfer, observed_state, observed.value,
                          copy_mutation, copy_cell, copy_operands,
                          copy_binding, get_cell, get_operands,
                          key_cells[0], get_binding)
                book.post(
                    SOURCE_MAPPING_GET_COPY_STATE_DEMAND, row,
                    ('mapping_copy_alias_or_state_contract', *causes),
                    stage=REDUCTION, provenance=Derived(causes),
                    mode=Mode.CONCORD,
                )
