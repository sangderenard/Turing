"""Control-activity graph readers. No source AST is retained or interpreted."""
import hashlib

from .concordance_declarations import (
    SOURCE_ACTIVITY_NODE, SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_BINDING,
    SOURCE_ACTIVITY_POLICY, SOURCE_ACTIVITY_SCHEDULE,
    SOURCE_ACTIVITY_CHILD, SOURCE_ACTIVITY_CALL, SOURCE_ACTIVITY_SCOPE,
    CANONICAL_VALUE, INGESTION_VALUE, CONTROL_PROGRAM_BUILD,
)
from .identity_concordance import Derived, Mode, current_identity_book


def activity_scope(graph, *, book=None):
    book = current_identity_book() if book is None else book
    target = getattr(graph, 'G', graph)
    read_scope = target.graph.get('lexical_read_scope')
    if read_scope is None:
        raise ValueError('process graph has no declared control read scope')
    cell = book.latest_ref(SOURCE_ACTIVITY_SCOPE, (read_scope,))
    if cell is None:
        raise ValueError('process read scope has no concorded source activity graph')
    return book.page(SOURCE_ACTIVITY_SCOPE).cell(cell.row, cell.column)


def binding(graph, node, *, book=None, scope=None):
    """Read exact activity ownership; ingestion may supply its declared scope.

    Before function-graph folding publishes a lexical read scope, the ingestion
    writer already holds the scope passed to ingest_control_activity. This is
    that same key, not a name/operand-based reconstruction of ownership.
    """
    from ..common.tensors.topological_reducer import node_identity_cell
    book = current_identity_book() if book is None else book
    scope = activity_scope(graph, book=book) if scope is None else scope
    pending, seen, found = [node_identity_cell(graph, int(node), book=book)], set(), set()
    while pending:
        identity = pending.pop()
        if identity in seen:
            continue
        seen.add(identity)
        cell = book.latest_ref(SOURCE_ACTIVITY_BINDING, (scope, identity))
        if cell is not None:
            found.add(cell)
        elif identity.page == CANONICAL_VALUE:
            pending.extend(source for source, _stage in book.edges_into(identity)
                           if source.page in {CANONICAL_VALUE, INGESTION_VALUE})
    if not found:
        return None, None
    contracts = {book.page(SOURCE_ACTIVITY_BINDING).cell(cell.row, cell.column) for cell in found}
    if len(contracts) != 1:
        raise ValueError('process node has ambiguous control-activity ownership')
    cell = next(iter(found))
    return next(iter(contracts)), cell


def memberships(graph, *, book=None):
    """Project exact owner/arm edges onto this graph's resident identities."""
    book = current_identity_book() if book is None else book
    target = getattr(graph, 'G', graph)
    bindings = {int(node): binding(graph, int(node), book=book)[0] for node in target}
    # A control owner may have been removed by folding, but a live guard
    # requires its exact retained canonical identity, not a same-line node.
    owners = {}
    for node, contract in bindings.items():
        if contract is not None:
            fact = book.page(SOURCE_ACTIVITY_NODE).cell(contract[0].row, contract[0].column)
            if _ingested_kind(graph, node, book=book) == fact[0]:
                owners.setdefault(contract[0].row[1], set()).add(node)
    result = {}
    for node, contract in bindings.items():
        if contract is None:
            continue
        fact = book.page(SOURCE_ACTIVITY_NODE).cell(contract[0].row, contract[0].column)
        guards = []
        for occurrence, arm in fact[1]:
            candidates = owners.get(occurrence, ())
            if len(candidates) > 1:
                raise ValueError('control owner has multiple resident identities')
            if candidates:
                guards.append((next(iter(candidates)), arm))
        if guards:
            result[node] = frozenset(guards)
    return result


class ActivitySchedule:
    """An existing scheduler's arrangement of the ingested activity graph."""
    def __init__(self, graph, preference=None, *, book=None):
        from .graph_work import DependencySnapshot, schedule_dependency_snapshot
        self.book = book = current_identity_book() if book is None else book
        self.scope = scope = activity_scope(graph, book=book)
        policy = (book.latest_ref(SOURCE_ACTIVITY_POLICY, (scope,)) if preference is None else
                  configure_activity_schedule(scope, preference, book=book))
        if policy is None:
            raise ValueError('control activity has no concorded scheduling configuration')
        configured = book.page(SOURCE_ACTIVITY_POLICY).cell(policy.row, policy.column)
        if preference is not None and preference != configured:
            raise ValueError('activity schedule disagrees with its configured policy')
        nodes = tuple(book.latest_ref(SOURCE_ACTIVITY_NODE, row)
                      for row in book.page(SOURCE_ACTIVITY_NODE).scope_rows(scope))
        indices = {cell: index for index, cell in enumerate(nodes)}
        edges, edge_cells = [], []
        parents, children = {i: [] for i in indices.values()}, {i: [] for i in indices.values()}
        for row in book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope):
            _scope, before, after, relation = row
            edge_cells.append(book.latest_ref(SOURCE_ACTIVITY_EDGE, row))
            if relation in {'repeat', 'unadmitted'}:
                continue
            a, b = indices[before], indices[after]
            edges.append((a, b)); parents[b].append((a, relation)); children[a].append((b, relation))
        records = tuple((i, tuple(parents[i]), tuple(children[i])) for i in indices.values())
        digest = hashlib.sha256(repr((nodes, tuple(edges), configured)).encode()).hexdigest()
        proposal = schedule_dependency_snapshot(DependencySnapshot(records, tuple(edges), configured, digest))
        levels = {cell: proposal.levels[index] for cell, index in indices.items()}
        target = getattr(graph, 'G', graph)
        view = book.latest_ref(SOURCE_ACTIVITY_SCOPE, (target.graph['lexical_read_scope'],))
        binding_cells = tuple(cell for node in target for _contract, cell in (binding(graph, int(node), book=book),)
                              if cell is not None)
        projection_cells = tuple(book.latest_ref(page, row)
                                 for page in (SOURCE_ACTIVITY_CHILD, SOURCE_ACTIVITY_CALL)
                                 for row in book.page(page).scope_rows(scope))
        self.cell = book.post(SOURCE_ACTIVITY_SCHEDULE, (scope, configured),
                             (tuple(levels.items()), proposal.provider), stage=CONTROL_PROGRAM_BUILD,
                             provenance=Derived(tuple(dict.fromkeys((policy, view, *nodes, *edge_cells, *binding_cells, *projection_cells)))), mode=Mode.CONCORD)
        self.graph = graph

    @property
    def levels(self):
        return dict(self.book.page(SOURCE_ACTIVITY_SCHEDULE).cell(self.cell.row, self.cell.column)[0])

    def position(self, node, phase='exit'):
        contract, _cell = binding(self.graph, int(node), book=self.book)
        if contract is None:
            raise ValueError(f'process node {node} lacks an ingested evaluation point')
        point = contract[0] if phase == 'enter' else contract[1]
        if phase not in {'enter', 'exit'}:
            selected = self.book.latest_ref(SOURCE_ACTIVITY_NODE, (*point.row[:2], phase))
            if selected is None:
                raise ValueError(f'process node {node} has no {phase} control point')
            point = selected
        return (int(self.levels[point]), 0, 0)

    def member_position(self, node, ordinal):
        contract, _cell = binding(self.graph, int(node), book=self.book)
        if contract is None:
            raise ValueError('display member has no exact activity owner')
        cell = self.book.latest_ref(SOURCE_ACTIVITY_CHILD, (self.scope, contract[0], int(ordinal)))
        if cell is None:
            raise ValueError('display member has no ingested evaluation point')
        _enter, leave = self.book.page(SOURCE_ACTIVITY_CHILD).cell(cell.row, cell.column)
        return (int(self.levels[leave]), 0, 0)

    def kind(self, node):
        contract, _cell = binding(self.graph, int(node), book=self.book)
        if contract is None:
            return None
        return self.book.page(SOURCE_ACTIVITY_NODE).cell(contract[0].row, contract[0].column)[0]

    def call_signature(self, node):
        contract, _cell = binding(self.graph, int(node), book=self.book)
        if contract is None:
            return None
        point = contract[0]
        return self.book.page(SOURCE_ACTIVITY_CALL).latest(point.row[:2])


def _ingested_kind(graph, node, *, book=None):
    from ..common.tensors.topological_reducer import node_identity_cell
    book = current_identity_book() if book is None else book
    pending, seen, kinds = [node_identity_cell(graph, int(node), book=book)], set(), set()
    while pending:
        cell = pending.pop()
        if cell in seen:
            continue
        seen.add(cell)
        if cell.page == INGESTION_VALUE:
            kinds.add(book.page(INGESTION_VALUE).cell(cell.row, cell.column).type)
        elif cell.page == CANONICAL_VALUE:
            pending.extend(source for source, _stage in book.edges_into(cell)
                           if source.page in {CANONICAL_VALUE, INGESTION_VALUE})
    return next(iter(kinds)) if len(kinds) == 1 else None


def conditional_memberships(graph, *, book=None):
    """The branch planner reads the ingested ownership edges, not syntax."""
    target = getattr(graph, 'G', graph)
    if not target:
        # A selected unit's empty definition catalogue has no operations
        # or conditional ownership. Its function bodies retain their scopes.
        return {}
    book = current_identity_book() if book is None else book
    scope = target.graph.get('lexical_read_scope')
    view = None if scope is None else book.latest_ref(SOURCE_ACTIVITY_SCOPE, (scope,))
    if view is None:
        # A module catalogue retains definitions. It does not execute the
        # bodies nested under those definitions as module-level control.
        if any(_ingested_kind(graph, int(node), book=book) == 'Module' for node in target):
            return {}
        raise ValueError('conditional ownership requires ingested control activity')
    result = {}
    for node, guards in memberships(graph, book=book).items():
        kept = frozenset((owner, arm) for owner, arm in guards
                         if _ingested_kind(graph, owner, book=book) in {'If', 'IfExp', 'BoolOp', 'Try', 'TryStar'})
        if kept:
            result[node] = kept
    return result


def configure_activity_schedule(scope, preference, *, book=None):
    """The policy writer used by deployment configuration and graph queries."""
    from .concordance_declarations import POLICY_DECLARATION
    from .identity_concordance import Novel
    book = current_identity_book() if book is None else book
    if preference not in {'asap', 'alap'}:
        raise ValueError('activity schedule preference must be asap or alap')
    previous = book.latest_ref(SOURCE_ACTIVITY_POLICY, (scope,))
    if previous is not None and book.page(SOURCE_ACTIVITY_POLICY).cell(previous.row, previous.column) == preference:
        return previous
    return book.post(SOURCE_ACTIVITY_POLICY, (scope,), preference, stage=CONTROL_PROGRAM_BUILD,
                     provenance=Novel(POLICY_DECLARATION, ()), mode=Mode.REVISE)


def process_partition_frontiers(graph, nodes):
    """Existing atomic regions cannot span a retained process operation.

    The predicate is reachability in the ingested activity graph, not line
    order or coincident storage. It adds a partition constraint, no new
    synchronization or comparative edge between worker bodies.
    """
    import networkx as nx
    from .process_graph_facts import operation
    from .source_control_ir import source_operator
    from .concordance_declarations import SOURCE_ACTIVITY_CUT, PLANNER_REGION_CARVE
    from ..common.tensors.topological_reducer import node_identity_cell
    book = current_identity_book()
    processes = [(int(node), source_operator(graph, int(node), book=book)[1]) for node in graph.G
                 for fact, _cell in (operation(graph, int(node)),) if fact is not None]
    if not processes:
        return {int(node): () for node in nodes}
    schedule = ActivitySchedule(graph, book=book)
    topology = nx.DiGraph()
    topology.add_nodes_from(schedule.levels)
    for scope, before, after, relation in book.page(SOURCE_ACTIVITY_EDGE).scope_rows(schedule.scope):
        if relation not in {'repeat', 'unadmitted'}:
            topology.add_edge(before, after)
    frontiers = []
    for process, source in processes:
        points, cell = binding(graph, process, book=book)
        if points is None:
            raise ValueError('process boundary has no exact activity binding')
        frontiers.append((source, cell, nx.descendants(topology, points[1])))
    planning_scope = graph.G.graph.get('planning_scope')
    if planning_scope is None:
        raise ValueError('activity partition requires the incumbent planning scope')
    result = {}
    for node in nodes:
        points, cell = binding(graph, int(node), book=book)
        if points is None:
            raise ValueError(f'eligible process node {node} lacks an activity partition proof')
        completed = tuple(source for source, _binding, following in frontiers if points[1] in following)
        cut = book.post(SOURCE_ACTIVITY_CUT, (planning_scope, int(node)), completed,
                        stage=PLANNER_REGION_CARVE,
                        provenance=Derived((schedule.cell, node_identity_cell(graph, node, book=book), cell,
                                            *(source for source, _binding, _following in frontiers),
                                            *(binding_cell for _source, binding_cell, _following in frontiers))),
                        mode=Mode.CONCORD)
        result[int(node)] = book.page(SOURCE_ACTIVITY_CUT).cell(cut.row, cut.column)
    return result
