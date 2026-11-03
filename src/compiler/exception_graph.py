"""Native exception admission is decided from ingested node identities.

No AST evaluation, source-coordinate matching, or source rewrite supplies
exception control after conversion. Unsupported source nodes retain a typed,
concorded refusal until a real exceptional-control graph is available.
"""
import ast

from .concordance_declarations import (
    CALLABLE_IDENTITY, CANONICAL_VALUE, FUNCTION_ADDRESS, INGESTION_VALUE,
    NATIVE_EXCEPTION_CONTROL_GAP, REDUCTION, SOURCE_ACTIVITY_BINDING,
    SOURCE_PYTHON_IDENTITY,
    SOURCE_RESOURCE_ADMISSION_DEMAND, SOURCE_RESOURCE_SCOPE,
)
from .identity_concordance import Derived, Mode, Ref, current_identity_book


class MissingResourceControlTopology(ValueError):
    pass


class MissingExceptionControlTopology(ValueError):
    def __init__(self, cell, kind):
        self.cell = cell
        super().__init__(f"native {kind} requires concorded exceptional control and cleanup edges")


def record_resource_admission_demand(graph, node):
    """Retain a With's exact source, control, call and input identities."""
    from ..common.tensors.topological_reducer import _source_numeric_scope, node_identity_cell
    from .control_activity import binding

    if node not in graph.G:
        return None
    statement = graph.G.nodes[node].get("expr_obj")
    if not isinstance(statement, (ast.With, ast.AsyncWith)):
        return None
    book = current_identity_book()
    scope = _source_numeric_scope(graph)
    source = node_identity_cell(graph, node)
    _points, activity = binding(graph, node, book=book, scope=scope)
    if source is None or activity is None:
        raise ValueError("resource demand has no source or control activity identity")
    calls = []
    causes = [source, activity]
    for item in statement.items:
        context = item.context_expr
        call = node_identity_cell(graph, id(context)) if id(context) in graph.G else None
        inputs = tuple(
            (str(role), node_identity_cell(graph, int(parent)))
            for parent, role in (
                graph.G.nodes[id(context)].get("parents") or ()
                if id(context) in graph.G else ()
            )
            if int(parent) in graph.G
        )
        receiver = next((cell for role, cell in inputs if role == "operand"), None)
        calls.append((call, receiver, inputs))
        causes.extend(cell for cell in (call, receiver, *(value for _role, value in inputs))
                      if cell is not None)
    demand = book.post(
        SOURCE_RESOURCE_ADMISSION_DEMAND, (scope, int(node)),
        (source, activity, tuple(calls)), stage=REDUCTION,
        provenance=Derived(tuple(dict.fromkeys(causes))), mode=Mode.CONCORD,
    )
    # The local callable can already be proven for a direct factory result.
    # Publish that exact identity while the context call remains resident;
    # later whole-source settlement may add proofs, but never needs the
    # syntactic With node as a substitute for the retained demand.
    for item in statement.items:
        _concord_context_call_identity(graph, item.context_expr)
    return demand


def finalize_resource_admission_demands(source_graph):
    """Resolve every recorded With before an executable control program exists."""
    from ..common.tensors.topological_reducer import _source_numeric_scope
    from .source_evidence_resolver import CallableEvidence, UnresolvedSourceDemand, require_callable

    book = current_identity_book()
    graphs = tuple(
        entry.graph for entry in source_graph.function_table
        if getattr(entry, "graph", None) is not None
    )
    by_scope = {_source_numeric_scope(graph): graph for graph in graphs}
    refusals = []
    for row in sorted(book.page(SOURCE_RESOURCE_ADMISSION_DEMAND).rows(), key=repr):
        demand = book.latest_ref(SOURCE_RESOURCE_ADMISSION_DEMAND, row)
        graph = by_scope.get(row[0])
        source, _activity, _calls = book.page(SOURCE_RESOURCE_ADMISSION_DEMAND).cell(
            demand.row, demand.column,
        )
        resolutions = []
        call_identity_causes = []
        identity_gaps = []
        for call, _receiver, _inputs in _calls:
            # Program: the guarded `book.borrowed(name)` call has canonical
            # value 11 after lexical relabel, while its resource demand still
            # cites the earlier INGESTION_VALUE keyed by AST identity.
            # The canonical row's provenance is the exact identity bridge.
            canonical_call, identity_gap = canonical_resource_call(
                book, graph, call,
            )
            if canonical_call is None:
                identity_gaps.append(identity_gap)
                continue
            call_identity_causes.append(canonical_call)
            resolutions.append(require_callable(
                graph, row[0], int(canonical_call.row[1]), book=book,
            ))
        missing = (
            *tuple(dict.fromkeys(identity_gaps)),
            *tuple(dict.fromkeys(
                item.missing for item in resolutions
                if isinstance(item, UnresolvedSourceDemand)
            )),
            *tuple(dict.fromkeys(
                item.control_scope.missing
                if isinstance(item.control_scope, UnresolvedSourceDemand)
                else 'invalid_control_scope'
                for item in resolutions
                if isinstance(item, CallableEvidence)
                and (not isinstance(item.control_scope, Ref)
                     or item.control_scope.page != SOURCE_ACTIVITY_BINDING
                     or item.control_scope != _activity)
            )),
            *(("missing_call_value",) if not resolutions and not identity_gaps else ()),
            "resource_enter", "resource_exit", "cleanup",
        )
        gap = book.post(
            NATIVE_EXCEPTION_CONTROL_GAP, (source,),
            ("With", missing), stage=REDUCTION,
            provenance=Derived((source, demand, *call_identity_causes, *(
                item.callable_cell for item in resolutions
                if isinstance(item, CallableEvidence)
            ))),
            mode=Mode.CONCORD,
        )
        refusals.append(gap)
    if refusals:
        raise MissingResourceControlTopology(
            "native resource scope requires concorded entry, exit and cleanup edges: "
            + ", ".join(map(str, refusals))
        )


def canonical_resource_call(book, graph, call):
    """Read one exact INGESTION_VALUE→CANONICAL_VALUE call bridge."""
    read_scope = (None if graph is None else
                  graph.G.graph.get('lexical_read_scope'))
    candidates = (() if call is None or read_scope is None else tuple(
        book.latest_ref(CANONICAL_VALUE, row)
        for row in book.page(CANONICAL_VALUE).scope_rows(read_scope)
        if call in {source for source, _stage in book.edges_into(
            book.latest_ref(CANONICAL_VALUE, row))}
    ))
    if len(candidates) != 1:
        return (None, 'missing_canonical_call_identity' if not candidates
                else 'ambiguous_canonical_call_identity')
    return candidates[0], None


def _concord_context_call_identity(graph, context):
    """Publish the exact already-resolved identity of one resource call.

    Method resolution owns discovery.  This reader accepts only the callable
    cell it wrote, the FUNCTION_ADDRESS cell that caused it, the exact proof
    cells that selected that callable, and the graph identities of the
    ordered inputs.  A direct receiver contributes its SOURCE_VALUE_CLASS;
    an optional receiver contributes its guarded annotation/control cells.
    Source spellings and literal argument values are intentionally absent.
    """

    from ..common.tensors.topological_reducer import (
        _source_numeric_scope, node_identity_cell,
    )

    if not isinstance(context, ast.Call) or id(context) not in graph.G:
        return None
    book = current_identity_book()
    scope = _source_numeric_scope(graph)
    call_id = int(id(context))
    call_cell = node_identity_cell(graph, call_id)
    callable_cell = book.latest_ref(CALLABLE_IDENTITY, (scope, call_id))
    if call_cell is None or callable_cell is None:
        return None
    address = book.page(CALLABLE_IDENTITY).cell(
        callable_cell.row, callable_cell.column,
    )
    address_cells = tuple(
        parent for parent, _stage in book.edges_into(callable_cell)
        if parent.page == FUNCTION_ADDRESS
        and book.page(FUNCTION_ADDRESS).cell(parent.row, parent.column)
        == address
    )
    if len(address_cells) != 1:
        return None
    call_data = graph.G.nodes[call_id]
    inputs = tuple(
        (int(parent), str(role))
        for parent, role in call_data.get("parents") or ()
        if int(parent) in graph.G
    )
    receivers = tuple(parent for parent, role in inputs if role == "operand")
    if len(receivers) != 1:
        return None
    callable_sources = tuple(
        parent for parent, _stage in book.edges_into(callable_cell)
    )
    callable_proof_cells = tuple(dict.fromkeys(
        parent for parent in callable_sources
        if parent != call_cell and parent not in address_cells
    ))
    if not callable_proof_cells:
        return None
    input_cells = tuple(
        (role, node_identity_cell(graph, parent))
        for parent, role in inputs
    )
    if any(input_cell is None for _role, input_cell in input_cells):
        return None
    descriptor = (
        "concorded_context_call",
        callable_cell,
        address_cells[0],
        callable_proof_cells,
        input_cells,
    )
    row = (scope, call_id)
    incumbent = book.page(SOURCE_PYTHON_IDENTITY).latest(row)
    if incumbent is not None and incumbent != descriptor:
        return None
    if incumbent is None:
        book.post(
            SOURCE_PYTHON_IDENTITY, row, descriptor,
            stage=REDUCTION,
            provenance=Derived(tuple(dict.fromkeys((
                call_cell, callable_cell, address_cells[0],
                *callable_proof_cells,
                *(input_cell for _role, input_cell in input_cells),
            )))),
            mode=Mode.CONCORD,
        )
    graph.G.nodes[call_id].setdefault("attributes", {})[
        "source_python_identity_row"
    ] = row
    return book.latest_ref(SOURCE_PYTHON_IDENTITY, row)


def require_exception_control(graph, node):
    from ..common.tensors.topological_reducer import node_identity_cell
    if node not in graph.G:
        return
    book = current_identity_book()
    cell = node_identity_cell(graph, node)
    # This admission check runs while source statements are converted. At
    # this point their identities are the already-posted INGESTION_VALUEs.
    if cell.page != INGESTION_VALUE:
        return
    fact = book.page(INGESTION_VALUE).cell(cell.row, cell.column)
    if fact.type == "With":
        gap = book.post(NATIVE_EXCEPTION_CONTROL_GAP, (cell,),
                        (fact.type, ("resource_enter", "resource_exit", "cleanup")),
                        stage=REDUCTION, provenance=Derived((cell,)), mode=Mode.CONCORD)
        raise MissingResourceControlTopology(
            f"native resource scope requires concorded entry, exit and cleanup edges: {gap}")
    if fact.type == "Try" and book.latest_ref(SOURCE_RESOURCE_SCOPE, (cell,)) is not None:
        # Its source conversion declares entry/body/cleanup control arms;
        # it is admitted as a ResourceScopeBlock, not a Python exception call.
        return
    if fact.type not in {"Try", "TryStar", "Raise"}:
        return
    gap = book.post(NATIVE_EXCEPTION_CONTROL_GAP, (cell,),
                    (fact.type, ("exceptional_successor", "handler", "cleanup")),
                    stage=REDUCTION, provenance=Derived((cell,)), mode=Mode.CONCORD)
    raise MissingExceptionControlTopology(gap, fact.type)
