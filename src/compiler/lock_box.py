"""Concorded mutex bindings for the SSA token-lock box.

The runtime key is the mutex handle, never its spelling or factory site.
These descriptors prepare state layout and ownership policy; native table
allocation and atomic token transitions belong to lowering.
"""

from .concordance_declarations import (
    LOCK_BOX_ENTRY, SOURCE_MUTEX_BINDING, SOURCE_CLOSURE_CAPTURE, NAME_BINDING,
    SOURCE_SYNCHRONIZATION, CONTROL_PROGRAM_BUILD,
)
from .identity_concordance import current_identity_book, Derived, Mode


def bind_mutex(graph, value, seen=()):
    from ..common.tensors.topological_reducer import node_identity_cell
    from ..transmogrifier.graph.edge_roles import ordered_arguments, keyword_argument_name
    from .synchronization import concord_source_synchronization
    from ..transmogrifier.ssa import SSALockBox
    from .process_graph_facts import operation as process_operation, local_operands

    book = current_identity_book()
    scope = graph.G.graph["lexical_read_scope"]
    row = (scope, int(value))
    existing = book.page(SOURCE_MUTEX_BINDING).latest(row)
    if existing is not None:
        return existing
    source = node_identity_cell(graph, value)
    if source in seen:
        raise ValueError("mutex binding has cyclic capture provenance")
    operation_fact, operation_cell = process_operation(graph, int(value))
    operation = None if operation_fact is None else operation_fact[0]
    causes = [source, *(() if operation_cell is None else (operation_cell,))]
    entry = None
    captures = []
    for capture_row in book.page(SOURCE_CLOSURE_CAPTURE).scope_rows(scope):
        capture = book.page(SOURCE_CLOSURE_CAPTURE).latest(capture_row)
        input_binding = capture[2]
        binding = book.page(NAME_BINDING).cell(input_binding.row, input_binding.column)
        if int(binding.value_id) == int(value):
            captures.append((capture_row, capture))
    if len(captures) > 1:
        raise ValueError("mutex input has ambiguous concorded closure bindings")
    if captures:
        capture_row, capture = captures[0]
        parent_binding = capture[1]
        parent = next((item.graph for item in graph.function_table
                       if getattr(item, "graph", None) is not None
                       and item.graph.G.graph.get("lexical_read_scope") == parent_binding.row[0]), None)
        if parent is None:
            raise ValueError("mutex closure has no exact parent graph")
        parent_value = book.page(NAME_BINDING).cell(parent_binding.row, parent_binding.column).value_id
        entry = bind_mutex(parent, parent_value, (*seen, source))
        causes.extend((book.latest_ref(SOURCE_CLOSURE_CAPTURE, capture_row), parent_binding, entry))
    if entry is None:
        reentrant = None
        execution_scope = ("object", source)
        if operation in {"sync_mutex_create", "condition_create"}:
            parents, edge_cell = local_operands(graph, int(value))
            causes.append(edge_cell)
            keywords = tuple((parent, keyword_argument_name(role)) for parent, role in parents
                             if keyword_argument_name(role) is not None)
            contract = concord_source_synchronization(
                graph, value, operation,
                (*ordered_arguments(parents), *(parent for parent, _name in keywords)),
                tuple(name for _parent, name in keywords),
            )
            constants = dict(book.page(SOURCE_SYNCHRONIZATION).cell(contract.row, contract.column)[3])
            reentrant = constants.get("reentrant")
            execution_scope = (constants["execution_scope"], source)
            causes.append(contract)
            if operation == "condition_create" and not constants["owns_mutex"]:
                # Distinct Condition identity/queue, exact same authored lock.
                supplied = (*ordered_arguments(parents), *(parent for parent, _name in keywords))
                entry = bind_mutex(graph, supplied[0], (*seen, source))
                descriptor = book.page(LOCK_BOX_ENTRY).cell(entry.row, entry.column)
                if len(descriptor) != 4 or type(descriptor[0]) is not bool:
                    raise ValueError("Condition lock requires an ingested Lock or RLock instance")
                causes.append(entry)
        condition = operation == "condition_create"
        layout = SSALockBox.CONDITION_STATE_COLUMNS if condition else SSALockBox.STATE_COLUMNS
        entry = entry or book.post(
            LOCK_BOX_ENTRY, (source,),
            (reentrant, execution_scope, "mutex_handle", layout,
             *((SSALockBox.CONDITION_INITIAL_STATE,) if condition else ())),
            stage=CONTROL_PROGRAM_BUILD, provenance=Derived(tuple(causes)), mode=Mode.CONCORD,
        )
        causes.append(entry)
    book.post(SOURCE_MUTEX_BINDING, row, entry, stage=CONTROL_PROGRAM_BUILD,
              provenance=Derived(tuple(causes)), mode=Mode.CONCORD)
    return entry


def ingest_mutex_bindings(source_graph):
    """Catalogue every exact factory/use, including captured source workers."""
    from ..transmogrifier.graph.edge_roles import ordered_arguments, keyword_argument_name
    from .process_graph_facts import operation as process_operation, local_operands
    graphs = [entry.graph for entry in source_graph.function_table if getattr(entry, "graph", None) is not None]
    for graph in graphs:
        for node, data in graph.G.nodes(data=True):
            fact, _cell = process_operation(graph, int(node))
            operation = "" if fact is None else fact[0]
            if operation in {"sync_mutex_create", "condition_create"}:
                bind_mutex(graph, node)
    for graph in graphs:
        for node, data in graph.G.nodes(data=True):
            fact, _cell = process_operation(graph, int(node))
            operation = "" if fact is None else fact[0]
            if ((operation.startswith("sync_mutex_") and operation != "sync_mutex_create")
                    or operation.startswith("condition_") and operation != "condition_create"):
                parents, _edge_cell = local_operands(graph, int(node))
                positional = tuple(ordered_arguments(parents))
                receiver = positional[0] if positional else next((parent for parent, role in parents
                           if keyword_argument_name(role) == "mutex"), None)
                if receiver is None:
                    raise ValueError("mutex operation has no exact receiver operand")
                bind_mutex(graph, receiver)
