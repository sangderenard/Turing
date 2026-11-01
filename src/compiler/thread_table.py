"""Exact source worker definitions, activation contexts, and their SSA bodies.

A create site describes a runtime instance factory, not a fixed native lane.
Bodies are shared by function identity; captures retain storage identity.
"""
from .concordance_declarations import (
    SOURCE_THREAD_BODY, SOURCE_THREAD_ACTIVATION, SOURCE_THREAD_CAPTURE_USE, THREAD_CAPTURE_CONTRACT,
    SOURCE_CLOSURE_CAPTURE, SOURCE_VALUE_CLASS, NAME_BINDING, FUNCTION_ADDRESS,
    SSA_THREAD_BODY, SSA_THREAD_ACTIVATION, CONTROL_PROGRAM_BUILD, FRAME_BINDING,
    SSA_THREAD_CONTEXT, THREAD_CONTEXT_BINDING, THREAD_CAPTURE_COLUMNS,
    SOURCE_THREAD_ARGUMENT, SOURCE_AGGREGATE_LAYOUT, SOURCE_FORMAL_ORDER, SOURCE_THREAD_ARGUMENT_PACK,
)
from .identity_concordance import current_identity_book, Derived, Mode


def ingest_thread_table(source_graph):
    from ..common.tensors.topological_reducer import node_identity_cell, _source_numeric_scope
    from .process_graph_facts import operation as process_operation, local_operands
    from .concordance_declarations import CALLABLE_IDENTITY
    from ..transmogrifier.function_table import function_address_identity
    book = current_identity_book()
    entries = {int(e.reference.address): e for e in source_graph.function_table
               if getattr(e, "graph", None) is not None}
    for parent_ref, parent_entry in entries.items():
        parent = parent_entry.graph
        for create, data in parent.G.nodes(data=True):
            operation, process_cell = process_operation(parent, int(create))
            if operation is None or operation[0] != "thread_create":
                continue
            operands, operand_cell = local_operands(parent, int(create))
            targets = [int(n) for n, role in operands if role in {"kw:target", "arg:1"}]
            if len(targets) != 1:
                raise ValueError("thread table requires one exact target operand")
            callable_cell = book.latest_ref(CALLABLE_IDENTITY, (_source_numeric_scope(parent), targets[0]))
            if callable_cell is None:
                raise ValueError("thread table target has no concorded callable identity")
            target = book.page(CALLABLE_IDENTITY).cell(callable_cell.row, callable_cell.column)
            if target not in entries:
                raise ValueError("thread table target has no ingested source body")
            entry = entries[target]
            worker = entry.graph
            address = book.latest_ref(FUNCTION_ADDRESS, (function_address_identity(entry),))
            if address is None:
                raise ValueError("thread table body has no exact function address")
            scope = worker.G.graph["lexical_read_scope"]
            captures = tuple(book.latest_ref(SOURCE_CLOSURE_CAPTURE, row)
                             for row in book.page(SOURCE_CLOSURE_CAPTURE).scope_rows(scope))
            nodes = tuple(node_identity_cell(worker, int(n)) for n in worker.G)
            body = book.post(SOURCE_THREAD_BODY, (address,), (int(target), nodes, captures),
                             stage=CONTROL_PROGRAM_BUILD,
                             provenance=Derived((address, *nodes, *captures)), mode=Mode.CONCORD)
            create_cell = node_identity_cell(parent, int(create))
            boundaries = {"thread_start": [], "thread_join": []}
            boundary_causes = []
            for child in parent.G:
                child_fact, child_cell = process_operation(parent, int(child))
                child_op = None if child_fact is None else child_fact[0]
                boundary = 'thread_join' if child_op == 'sync_completion_wait' else child_op
                if boundary in boundaries:
                    child_operands, edge_cell = local_operands(parent, int(child))
                    if (int(create), "arg:0") in child_operands:
                        boundaries[boundary].append(node_identity_cell(parent, int(child)))
                        boundary_causes.extend((child_cell, edge_cell))
            operand_cells = tuple((node_identity_cell(parent, int(n)), str(role)) for n, role in operands)
            starts, joins = tuple(boundaries["thread_start"]), tuple(boundaries["thread_join"])
            activation = book.post(SOURCE_THREAD_ACTIVATION, (create_cell,),
                (body, parent_ref, operand_cells, starts, joins, tuple(operation[4])),
                stage=CONTROL_PROGRAM_BUILD,
                provenance=Derived((create_cell, process_cell, operand_cell, callable_cell, body, *(c for c, _ in operand_cells), *starts, *joins, *boundary_causes)),
                mode=Mode.CONCORD)
            data.setdefault("attributes", {})["source_thread_activation"] = activation
            packs = [int(n) for n, role in operands if role == "kw:args"]
            if len(packs) == 1 and not any(role == "kw:kwargs" for _n, role in operands):
                layout_cell = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, (parent.G.graph["lexical_read_scope"], packs[0]))
                layout = None if layout_cell is None else book.page(SOURCE_AGGREGATE_LAYOUT).cell(layout_cell.row, layout_cell.column)
                formal_order = book.latest_ref(SOURCE_FORMAL_ORDER, (_source_numeric_scope(worker),))
                if formal_order is None:
                    raise ValueError("worker signature has no ingested formal order")
                parameters = book.page(SOURCE_FORMAL_ORDER).cell(formal_order.row, formal_order.column)
                if layout is not None and layout.aggregate_kind == "tuple" and len(layout.members) == len(parameters):
                    bindings = []
                    for position, (member, parameter) in enumerate(zip(layout.members, parameters, strict=True)):
                        formal = book.latest_ref(NAME_BINDING, (scope, parameter, 0))
                        if formal is None:
                            raise ValueError("thread argument has no exact worker formal binding")
                        bindings.append(book.post(SOURCE_THREAD_ARGUMENT, (activation, position),
                            (member.value_cell, formal, layout_cell), stage=CONTROL_PROGRAM_BUILD,
                            provenance=Derived((activation, member.value_cell, formal, layout_cell, *member.source_cells)),
                            mode=Mode.CONCORD))
                    book.post(SOURCE_THREAD_ARGUMENT_PACK, (activation,), tuple(bindings),
                              stage=CONTROL_PROGRAM_BUILD,
                              provenance=Derived((activation, layout_cell, formal_order, *bindings)), mode=Mode.CONCORD)
                    data["attributes"]["source_thread_argument_bindings"] = tuple(bindings)
            _capture_sequence_contracts(worker, captures, entries)
            from .process_graph_facts import indexed_write_bases
            written, write_cells = indexed_write_bases(worker)
            from ..common.tensors.topological_reducer import _set_operands
            graphs = {e.graph.G.graph["lexical_read_scope"]: e.graph for e in entries.values()}
            captured_operands = []
            for index, capture in enumerate(captures):
                _environment, binding, input_binding = book.page(SOURCE_CLOSURE_CAPTURE).cell(capture.row, capture.column)
                storage_graph = graphs[binding.row[0]]
                if storage_graph is not parent:
                    raise ValueError("thread capture requires an enclosing-context storage binding")
                actual = book.page(NAME_BINDING).cell(binding.row, binding.column).value_id
                formal = book.page(NAME_BINDING).cell(input_binding.row, input_binding.column).value_id
                actual_cell = node_identity_cell(parent, int(actual))
                formal_cell = node_identity_cell(worker, int(formal))
                use = book.post(SOURCE_THREAD_CAPTURE_USE, (activation, capture),
                    (actual_cell, formal_cell, int(formal) in written),
                    stage=CONTROL_PROGRAM_BUILD,
                    provenance=Derived((activation, capture, binding, input_binding, actual_cell, formal_cell, *nodes, *write_cells)),
                    mode=Mode.CONCORD)
                actual_attrs = parent.G.nodes[actual].setdefault("attributes", {})
                actual_attrs["source_thread_capture_uses"] = tuple(dict.fromkeys((
                    *actual_attrs.get("source_thread_capture_uses", ()), use,
                )))
                captured_operands.append((int(actual), f"capture:{index}"))
            if captured_operands:
                _set_operands(parent, int(create), (*operands, *captured_operands), cause=THREAD_CONTEXT_BINDING)


def _capture_sequence_contracts(worker, captures, entries):
    """Propagate element TYPE proofs, never capture values or scalar copies.

    shared=[0]; worker: shared[0]+=1. The closure input names the parent's
    storage through SOURCE_CLOSURE_CAPTURE. Without its element-class proof,
    planning the already-ingested worker refused Indexed's dtype. Read that
    exact binding; do not find another object by the spelling 'shared'.
    """
    from ..common.tensors.topological_reducer import _source_numeric_scope
    book = current_identity_book()
    by_scope = {e.graph.G.graph.get("lexical_read_scope"): e.graph for e in entries.values()}
    for capture in captures:
        _environment, parent_binding, input_binding = book.page(SOURCE_CLOSURE_CAPTURE).cell(capture.row, capture.column)
        parent = by_scope.get(parent_binding.row[0])
        if parent is None:
            raise ValueError("thread capture parent graph is absent")
        actual = book.page(NAME_BINDING).cell(parent_binding.row, parent_binding.column).value_id
        formal = book.page(NAME_BINDING).cell(input_binding.row, input_binding.column).value_id
        layout_cell = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, (parent.G.graph["lexical_read_scope"], int(actual)))
        layout = None if layout_cell is None else book.page(SOURCE_AGGREGATE_LAYOUT).cell(layout_cell.row, layout_cell.column)
        if layout is None or layout.aggregate_kind not in {"list", "tuple"}:
            row = (_source_numeric_scope(parent.G), int(actual))
            fact = book.page(SOURCE_VALUE_CLASS).latest(row)
            source = book.latest_ref(SOURCE_VALUE_CLASS, row)
            if source is not None and isinstance(fact, tuple) and fact[0] in {
                "builtins.int", "builtins.float", "builtins.bool",
            }:
                contract = book.post(THREAD_CAPTURE_CONTRACT, (capture,), (fact[0], (source,)),
                    stage=CONTROL_PROGRAM_BUILD,
                    provenance=Derived((capture, parent_binding, input_binding, source)), mode=Mode.CONCORD)
                attrs = worker.G.nodes[formal].setdefault("attributes", {})
                attrs["result_class_ref"] = fact[0]
                attrs["thread_capture_contract"] = contract
            continue
        leaves = tuple(member.value_id for member in layout.members)
        classes, sources = [], [layout_cell]
        for leaf in leaves:
            row = (_source_numeric_scope(parent.G), int(leaf))
            fact = book.page(SOURCE_VALUE_CLASS).latest(row)
            cell = book.latest_ref(SOURCE_VALUE_CLASS, row)
            if cell is None or not isinstance(fact, tuple):
                break
            classes.append(fact[0]); sources.append(cell)
        else:
            if classes and len(set(classes)) == 1:
                contract = book.post(THREAD_CAPTURE_CONTRACT, (capture,), (classes[0], tuple(sources)),
                    stage=CONTROL_PROGRAM_BUILD,
                    provenance=Derived((capture, parent_binding, input_binding, *sources)), mode=Mode.CONCORD)
                attrs = worker.G.nodes[formal].setdefault("attributes", {})
                attrs["sequence_element_class_ref"] = classes[0]
                attrs["callsite_sequence_element_class_cells"] = (contract,)
                attrs["thread_capture_contract"] = contract
                # Use the planner's existing writer-checked fixed point for
                # the physical row contract as well. The sequence builder
                # reads columns, whereas region planning reads element class.
                from .glsl_deployment_strategy import _planned_scalar_dtypes
                dtype = _planned_scalar_dtypes(worker).get(int(formal))
                if dtype is not None:
                    from ..common.tensors.topological_reducer import node_identity_cell
                    columns = book.post(THREAD_CAPTURE_COLUMNS, (contract,), (dtype,),
                        stage=CONTROL_PROGRAM_BUILD,
                        provenance=Derived((contract, *(node_identity_cell(worker, int(n)) for n in worker.G))),
                        mode=Mode.CONCORD)
                    attrs["sequence_column_dtypes"] = (dtype,)
                    attrs["thread_capture_columns"] = columns


def publish_ssa_thread_table(module, book):
    """Join the source body identity to actual lowered functions and handles."""
    from ..transmogrifier.ssa import SSAThreadTable
    from .ssa_record_return_state import ssa_value_identity_cell
    table = SSAThreadTable(module=module, book=book)
    module.thread_table = table
    from .concordance_declarations import LOWERED_SOURCE_FUNCTION
    by_reference = {}
    for row in book.page(LOWERED_SOURCE_FUNCTION).rows():
        address, variant, entry_block, captures = book.page(LOWERED_SOURCE_FUNCTION).latest(row)
        reference = book.page(FUNCTION_ADDRESS).cell(address.row, address.column)
        if not variant and row[0] in module.functions:
            by_reference.setdefault(int(reference), []).append((row[0], module.functions[row[0]]))
    body_cells = {}
    for body, fact in table.bodies.items():
        reference = fact[0]
        for name, function in by_reference.get(reference, ()):
            function_cell = book.latest_ref(LOWERED_SOURCE_FUNCTION, (name,))
            _address, _variant, entry_block, capture_bindings = book.page(LOWERED_SOURCE_FUNCTION).cell(
                function_cell.row, function_cell.column)
            captures = tuple((binding.row[1], book.page(NAME_BINDING).cell(binding.row, binding.column).value_id)
                             for binding in capture_bindings)
            signature = tuple(ssa_value_identity_cell(function, value.id, book=book) for value in function.args)
            if any(cell is None for cell in signature):
                raise ValueError("worker signature has an unconcorded SSA formal")
            cell = book.post(SSA_THREAD_BODY, (body, name), (reference, captures, entry_block),
                stage=FRAME_BINDING, provenance=Derived((body, function_cell, *signature)), mode=Mode.CONCORD)
            body_cells.setdefault(body, []).append(cell)
    for name, function in module.functions.items():
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op != "ThreadCreate":
                    continue
                from .concordance_declarations import SSA_PROCESS_CALL
                from dataclasses import replace
                result_cell = ssa_value_identity_cell(function, instruction.res.id, book=book)
                call_cell = book.latest_ref(SSA_PROCESS_CALL, (result_cell,))
                if call_cell is None:
                    raise ValueError("SSA ThreadCreate has no concorded operand contract")
                call = book.page(SSA_PROCESS_CALL).cell(call_cell.row, call_cell.column)
                source = call.activation
                if source is None:
                    raise ValueError("SSA ThreadCreate has no thread-table activation")
                body = book.page(SOURCE_THREAD_ACTIVATION).cell(source.row, source.column)[0]
                targets = tuple(body_cells.get(body, ()))
                if not targets:
                    raise ValueError("thread table worker body was not lowered to SSA")
                capture_ids = call.capture_value_ids
                captures = book.page(SOURCE_THREAD_BODY).cell(body.row, body.column)[2]
                if len(capture_ids) != len(captures):
                    raise ValueError("thread context operands disagree with the source capture table")
                live_ids = []
                for capture, actual in zip(captures, capture_ids, strict=True):
                    _environment, _parent, binding = book.page(SOURCE_CLOSURE_CAPTURE).cell(capture.row, capture.column)
                    formal = int(book.page(NAME_BINDING).cell(binding.row, binding.column).value_id)
                    used = False
                    for target in targets:
                        worker_name = target.row[1]
                        worker = module.functions[worker_name]
                        actual_members = _context_members(module, name, int(actual))
                        formal_members = _context_members(module, worker_name, formal)
                        live = bool({value.id for value in worker.args} & {member for member, _roles in formal_members})
                        used |= live
                        # A context unused by the lowered body has no physical
                        # ABI operands. Keep the capture identity in the table;
                        # do not manufacture a value for an empty record.
                        book.post(SSA_THREAD_CONTEXT, (source, capture, name, worker_name),
                            (int(actual), formal, actual_members, formal_members, "live" if live else "unused"),
                            stage=FRAME_BINDING, provenance=Derived((source, capture, target)), mode=Mode.CONCORD)
                    if used:
                        live_ids.append(int(actual))
                if len(live_ids) != len(capture_ids):
                    runtime_count = len(instruction.args) - len(capture_ids)
                    arguments = [*instruction.args[:runtime_count],
                                 *(value for value in instruction.args[runtime_count:] if value.id in live_ids)]
                    context_cells = tuple(book.latest_ref(SSA_THREAD_CONTEXT, row)
                        for row in book.page(SSA_THREAD_CONTEXT).rows() if row[0] == source and row[2] == name)
                    call = replace(call, capture_value_ids=tuple(live_ids),
                                   argument_ids=tuple(value.id for value in arguments))
                    book.post(SSA_PROCESS_CALL, (result_cell,), call,
                              stage=FRAME_BINDING, provenance=Derived((call_cell, *context_cells)), mode=Mode.REVISE)
                    instruction.args = arguments
                    instruction.attributes["capture_value_ids"] = tuple(live_ids)
                book.post(SSA_THREAD_ACTIVATION, (name, call.callsite),
                    (source, instruction.res.id, tuple(a.id for a in instruction.args), targets),
                    stage=FRAME_BINDING, provenance=Derived((source, *targets)), mode=Mode.CONCORD)


def _context_members(module, function, value_id):
    from ..transmogrifier.ssa import sequence_member_roles, record_member_claims
    sequences = module.sequence_tables.get(function)
    records = module.record_tables.get(function)
    claims = sequence_member_roles(None if sequences is None else sequences.sequences.get(value_id))
    record_claims = record_member_claims(None if records is None else records.records.get(value_id))
    for member, roles in record_claims.items():
        claims.setdefault(member, set()).update(roles)
    if not claims:
        claims = {int(value_id): set()}
    return tuple((int(member), tuple(sorted(roles, key=repr))) for member, roles in sorted(claims.items()))


def publish_source_function(function, graph, function_table):
    """The function materialization writer binds the actual SSA symbol."""
    from ..transmogrifier.function_table import function_address_identity
    from .concordance_declarations import LOWERED_SOURCE_FUNCTION, SCOPE_ORIGIN, CONTROL_PROGRAM, SHELL
    from .identity_concordance import shape_scope_of
    from .ssa_record_return_state import ssa_value_identity_cell, ssa_block_identity_cell
    book = current_identity_book()
    target = getattr(graph, 'G', graph)
    reference = target.graph.get('function_ref')
    if reference is None:
        return
    entry = function_table.entry(int(reference))
    address = book.latest_ref(FUNCTION_ADDRESS, (function_address_identity(entry),))
    if address is None:
        raise ValueError('materialized source function has no exact address cell')
    scope = target.graph['lexical_read_scope']
    captures = tuple(book.page(SOURCE_CLOSURE_CAPTURE).latest(row)[2]
                     for row in book.page(SOURCE_CLOSURE_CAPTURE).scope_rows(scope))
    variant = book.latest_ref(SCOPE_ORIGIN, (shape_scope_of(graph),))
    signature = tuple(ssa_value_identity_cell(function, value.id, book=book) for value in function.args)
    if any(cell is None for cell in signature):
        raise ValueError('materialized function has an unconcorded formal')
    entry_block = str(function.metadata.get('entry_block') or 'entry')
    if entry_block not in function.blocks:
        raise ValueError('materialized function entry block is absent')
    program = book.latest_ref(CONTROL_PROGRAM, (scope, SHELL))
    entry_cell = ssa_block_identity_cell(function, entry_block, book=book)
    if program is None or entry_cell is None:
        raise ValueError('source function lacks its concorded control-program/SSA entry relationship')
    pending, ancestors = [entry_cell], set()
    while pending:
        cell = pending.pop()
        if cell in ancestors:
            continue
        ancestors.add(cell)
        pending.extend(source for source, _stage in book.edges_into(cell))
    if program not in ancestors:
        raise ValueError('materialized SSA entry belongs to a different source control program')
    book.post(LOWERED_SOURCE_FUNCTION, (function.name,),
              (address, variant is not None, entry_block, captures), stage=FRAME_BINDING,
              provenance=Derived((address, program, entry_cell, *captures, *signature, *(() if variant is None else (variant,)))),
              mode=Mode.CONCORD)
