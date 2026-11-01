"""Materialize process operations in control IR at the dual-IR boundary.

No source syntax is read here. Ingestion has already declared activity,
operation identity and exact dataflow operand edges. Numerical carving and
backend lowering consume these control blocks; they do not create them.
"""
from .concordance_declarations import (
    SOURCE_CONTROL_IR, SOURCE_CONTROL_OPERATOR, SOURCE_ACTIVITY_NODE,
    SOURCE_ACTIVITY_SCOPE, SOURCE_THREAD_ACTIVATION, SOURCE_THREAD_BODY,
    SOURCE_THREAD_ARGUMENT_PACK, SOURCE_DISPATCH_TARGET,
    CONTROL_PROGRAM_BUILD, CONTROL_BLOCK_PLACEMENT, SOURCE_EXCEPTION_OPERATION, SOURCE_RESOURCE_CONTROL,
)
from .identity_concordance import Derived, Mode, current_identity_book


def ingest_source_control_ir(source_graph, preference=None):
    for entry in source_graph.function_table:
        graph = getattr(entry, 'graph', None)
        if graph is not None:
            ingest_function_control(graph, preference)


def ingest_function_control(graph, preference=None):
    from .control_activity import (ActivitySchedule, activity_scope, binding, _ingested_kind,
                                   configure_activity_schedule)
    from .process_graph_facts import operation, local_operands, thread_activation, resource_scope
    from dataclasses import replace
    from .control_source import (ControlProgram, SourceRegionBlock, SequenceBlock, DispatchBlock, ResourceScopeBlock,
                                 post_control_program, control_block_cell)
    from .synchronization import concord_source_synchronization
    from ..transmogrifier.graph.edge_roles import ordered_arguments, keyword_argument_name
    book = current_identity_book()
    scope = graph.G.graph['lexical_read_scope']
    if book.latest_ref(SOURCE_CONTROL_IR, (scope,)) is not None:
        return
    source_scope = activity_scope(graph, book=book)
    configured = preference if preference is not None else (graph.G.graph.get('loop_settings') or {}).get(
        'schedule_preference', 'alap')
    configure_activity_schedule(source_scope, str(configured).lower(), book=book)
    activity = ActivitySchedule(graph, book=book)
    controls, owner_nodes, points_by_node = {}, {}, {}
    region_kinds = {'If', 'IfExp', 'For', 'AsyncFor', 'While', 'BoolOp', 'comprehension', 'Try'}
    for node in graph.G:
        points, _binding_cell = binding(graph, int(node), book=book)
        if points is None:
            continue
        points_by_node[int(node)] = points
        kind, owners = book.page(SOURCE_ACTIVITY_NODE).cell(points[0].row, points[0].column)
        if kind in region_kinds and _ingested_kind(graph, node, book=book) == kind:
            owner_nodes[points[0].row[1]] = int(node)
            controls[int(node)] = (kind, owners, points)
    entries, blocks, causes = {(): []}, {}, [activity.cell]

    def prefix_for(owners):
        prefix = ()
        for occurrence, arm in owners:
            if occurrence not in owner_nodes:
                raise ValueError('source control owner has no exact process-graph identity')
            owner = owner_nodes[occurrence]
            prefix = (*prefix, (owner, str(arm)))
            entries.setdefault(prefix, [])
        return prefix

    for node, (kind, owners, points) in controls.items():
        prefix = prefix_for(owners)
        phase = 'select' if kind in {'If', 'IfExp'} else 'enter'
        entries[prefix].append((activity.position(node, phase), node, 'region'))
        declared_arms = ({'If': ('body', 'orelse'), 'IfExp': ('body', 'orelse'),
                          'For': ('iter', 'body', 'orelse'), 'AsyncFor': ('iter', 'body', 'orelse'),
                          'While': ('test', 'body', 'orelse'), 'BoolOp': ('body', 'orelse'),
                          'comprehension': ('iter', 'body', 'test'), 'Try': ('body', 'finalbody')})[kind]
        for arm in declared_arms:
            entries.setdefault((*prefix, (node, arm)), [])
    for node in graph.G:
        fact, process_cell = operation(graph, int(node))
        if fact is None:
            continue
        if any(not (kind in {'If', 'Try'} or kind in {'For', 'While'} and arm == 'body'
                    or kind == 'While' and arm == 'test') for kind, _line, arm in fact[4]):
            raise ValueError('process operation has an unadmitted source control region')
        points, binding_cell = binding(graph, int(node), book=book)
        if points is None:
            raise ValueError('process operation has no ingested control activity')
        _kind, owners = book.page(SOURCE_ACTIVITY_NODE).cell(points[0].row, points[0].column)
        prefix = prefix_for(owners)
        parents, operands_cell = local_operands(graph, int(node))
        positional = tuple(ordered_arguments(parents))
        keywords = tuple((keyword_argument_name(role), parent) for parent, role in parents
                         if keyword_argument_name(role) is not None)
        activation = thread_activation(graph, int(node))
        target_cell = None
        if activation is not None:
            body = book.page(SOURCE_THREAD_ACTIVATION).cell(activation.row, activation.column)[0]
            reference = book.page(SOURCE_THREAD_BODY).cell(body.row, body.column)[0]
            target_cell = book.post(SOURCE_DISPATCH_TARGET, (scope, int(node)), reference,
                stage=CONTROL_PROGRAM_BUILD, provenance=Derived((activation, body, body.row[0], process_cell)),
                mode=Mode.CONCORD)
        pack = None if activation is None else book.latest_ref(SOURCE_THREAD_ARGUMENT_PACK, (activation,))
        pack_bindings = () if pack is None else book.page(SOURCE_THREAD_ARGUMENT_PACK).cell(pack.row, pack.column)
        synchronization = None
        if fact[0].startswith(('sync_', 'condition_', 'barrier_')):
            synchronization = concord_source_synchronization(
                graph, int(node), fact[0], (*positional, *(parent for _name, parent in keywords)),
                tuple(name for name, _parent in keywords))
        block = DispatchBlock(
            callsite_id=int(node), operation=fact[0], argument_value_ids=positional,
            keyword_argument_value_ids=keywords, result_value_id=int(node), result_dtype=fact[2],
            target_function_cell=target_cell, synchronization_cell=synchronization,
            process_operation_cell=process_cell, thread_activation_cell=activation,
            capture_argument_value_ids=tuple(parent for parent, role in parents if role.startswith('capture:')),
            worker_argument_bindings=pack_bindings, worker_argument_pack_expanded=pack is not None,
            operand_graph_cell=operands_cell,
            exception_operation_cell=book.latest_ref(SOURCE_EXCEPTION_OPERATION, (process_cell.row[0],)),
        )
        blocks[int(node)] = block
        entries[prefix].append((activity.position(node), int(node), 'operator'))
        causes.extend(cell for cell in (process_cell, binding_cell, operands_cell, synchronization, target_cell, pack)
                      if cell is not None)

    regions, resource_records = [], {}
    def build(prefix=()):
        children = []
        for _position, node, category in sorted(entries.get(prefix, ()), key=lambda item: item[0]):
            if category == 'operator':
                children.append(blocks[node])
            else:
                kind, _owners, points = controls[node]
                arms = tuple((arm, build(path)) for path in entries
                             if len(path) == len(prefix) + 1 and path[:-1] == prefix and path[-1][0] == node
                             for arm in (path[-1][1],))
                header_operands = local_operands(graph, node)[1]
                if kind == 'Try':
                    contract, resource_cell = resource_scope(graph, node, book=book)
                    if resource_cell is None:
                        raise ValueError('Try has no ingested resource control contract')
                    _conversion, entry_occurrence, cleanup_occurrence = contract
                    operators = {points_by_node[operator][0].row[1]: operator for operator in blocks}
                    entry_node, cleanup_node = operators.get(entry_occurrence), operators.get(cleanup_occurrence)
                    if entry_node is None or cleanup_node is None:
                        raise ValueError('resource entry/cleanup lost their exact process identities')
                    entry, cleanup = blocks[entry_node], blocks[cleanup_node]
                    from .concordance_declarations import SOURCE_RESOURCE_CONVERSION
                    recipe = book.page(SOURCE_RESOURCE_CONVERSION).cell(_conversion.row, _conversion.column)
                    if (entry.operation != recipe[1] or cleanup.operation != recipe[2]
                            or len(entry.argument_value_ids) != 1
                            or entry.argument_value_ids != cleanup.argument_value_ids):
                        raise ValueError('resource entry and cleanup do not consume the same captured receiver')
                    cleanup = replace(cleanup, result_value_id=None)
                    blocks[cleanup_node] = cleanup
                    region = ResourceScopeBlock(dict(arms)['body'], (cleanup,), node, resource_cell)
                    resource_records[node] = (resource_cell, entry, cleanup, entry.argument_value_ids[0])
                    causes.extend((resource_cell, entry.process_operation_cell, cleanup.process_operation_cell))
                else:
                    region = SourceRegionBlock(kind, node, points[0], points[1], arms, header_operands)
                regions.append(region)
                children.append(region)
        return SequenceBlock(tuple(children))
    program = ControlProgram(build())
    program_cell = post_control_program(graph, program, stage=CONTROL_PROGRAM_BUILD,
                                        cause=tuple(dict.fromkeys(causes)), label='ingested_process_control')
    operator_cells, region_cells, placements = [], [], []
    for node, block in blocks.items():
        control = control_block_cell(book, scope, block)
        placement = book.latest_ref(CONTROL_BLOCK_PLACEMENT, (scope, control))
        if control is None or placement is None:
            raise ValueError('ingested process operation has no placed control IR block')
        operator_cells.append(book.post(SOURCE_CONTROL_OPERATOR, (scope, node),
            (program_cell, control, placement, block), stage=CONTROL_PROGRAM_BUILD,
            provenance=Derived((program_cell, control, placement, block.process_operation_cell,
                                block.operand_graph_cell)), mode=Mode.CONCORD))
        placements.append(placement)
    for block in regions:
        control = control_block_cell(book, scope, block)
        if control is None:
            raise ValueError('ingested control region has no control IR identity')
        region_cells.append(control)
        placements.append(book.latest_ref(CONTROL_BLOCK_PLACEMENT, (scope, control)))
        if isinstance(block, ResourceScopeBlock):
            from ..common.tensors.topological_reducer import node_identity_cell
            resource_cell, entry, cleanup, receiver = resource_records[block.source_scope_id]
            receiver_cell = node_identity_cell(graph, receiver)
            book.post(SOURCE_RESOURCE_CONTROL, (scope, block.source_scope_id),
                      (resource_cell, control, entry.process_operation_cell, cleanup.process_operation_cell, receiver_cell),
                      stage=CONTROL_PROGRAM_BUILD,
                      provenance=Derived((resource_cell, control, entry.process_operation_cell,
                                          cleanup.process_operation_cell, receiver_cell)), mode=Mode.CONCORD)
    book.post(SOURCE_CONTROL_IR, (scope,),
              (program_cell, tuple(operator_cells), tuple(region_cells), tuple(placements)),
              stage=CONTROL_PROGRAM_BUILD,
              provenance=Derived((program_cell, *operator_cells, *region_cells, *placements)), mode=Mode.CONCORD)


def source_operator(graph, node, *, book=None):
    book = current_identity_book() if book is None else book
    scope = graph.G.graph['lexical_read_scope']
    cell = book.latest_ref(SOURCE_CONTROL_OPERATOR, (scope, int(node)))
    if cell is None:
        raise ValueError('process operation did not become control IR at ingestion')
    return book.page(SOURCE_CONTROL_OPERATOR).cell(cell.row, cell.column), cell
