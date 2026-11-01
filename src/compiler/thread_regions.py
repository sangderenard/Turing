"""Source-owned run-to-join graphs, before structural specialization.

An activation qualifies a use of the worker definition; it does not clone
the definition or allocate another captured object. The graphs below are
ingestion products, not backend deployment lanes or executable programs.
"""

from dataclasses import dataclass
import networkx as nx

from .concordance_declarations import (
    NAME_BINDING, SOURCE_THREAD_REGION,
    SOURCE_THREAD_ACTIVATION, SOURCE_THREAD_BODY, SOURCE_THREAD_CAPTURE_USE,
    SOURCE_THREAD_ARGUMENT,
    SOURCE_THREAD_REGION_MEMBER, SOURCE_THREAD_REGION_EDGE,
    SOURCE_THREAD_REGION_SCHEDULE, THREAD_REGION_BUILD, CONTROL_PROGRAM_BUILD,
    SOURCE_SYNC_OPERATOR, SOURCE_CONTROL_RENDEZVOUS, SOURCE_THREAD_BARRIER_POLICY,
    CONTROL_BLOCK, ControlBlockKind, ControlBlockFact,
    SOURCE_THREAD_REGION_GAP, SOURCE_THREAD_REGION_CONTROL,
    SOURCE_ACTIVITY_NODE, SOURCE_ACTIVITY_EDGE, SOURCE_CONTROL_IR,
)
from .identity_concordance import Derived, Mode, current_identity_book


def _control_projection(graph, book):
    """Read the existing control graph including regions and repeat edges."""
    from .control_activity import activity_scope, binding
    scope = activity_scope(graph, book=book)
    nodes = tuple(book.latest_ref(SOURCE_ACTIVITY_NODE, row)
                  for row in book.page(SOURCE_ACTIVITY_NODE).scope_rows(scope))
    edges = tuple((row[1], row[2], row[3], book.latest_ref(SOURCE_ACTIVITY_EDGE, row))
                  for row in book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope))
    bindings = tuple((int(node), *binding(graph, int(node), book=book)) for node in graph.G)
    control = book.latest_ref(SOURCE_CONTROL_IR, (graph.G.graph['lexical_read_scope'],))
    if control is None:
        raise MissingThreadControlTopology('thread region requires ingested control IR')
    causes = (control, *nodes, *(cell for _a, _b, _role, cell in edges),
              *(cell for _node, _points, cell in bindings if cell is not None))
    return nodes, edges, bindings, control, causes


class MissingThreadControlTopology(ValueError):
    pass


@dataclass
class ThreadRegion:
    """A simultaneous extent with its surrounding parent boundary context.

    ``activations`` defines the extent. Parent setup/continuation nodes stay
    in the graphs to retain every edge crossing those start/join boundaries.
    """
    dependency_graph: object
    control_graph: object
    activations: tuple  # (create, start, join, worker function ref), source-local ids
    member_ids: dict  # (activation create or None, function ref, source node) -> region node
    cell: object
    book: object

    @property
    def worker_tokens(self):
        """Exact authored activation identities for later native binding."""
        return self.dependency_graph.G.graph["worker_tokens"]

    def worker_nodes(self, worker_token):
        """Select one worker from authoritative membership, including captures."""
        if worker_token not in self.worker_tokens:
            raise KeyError(worker_token)
        scope = self.dependency_graph.G.graph["thread_region_scope"]
        page = self.book.page(SOURCE_THREAD_REGION_MEMBER)
        return tuple(row[1] for row in page.scope_rows(scope)
                     if page.latest(row)[1] == worker_token
                     and page.latest(row)[0].page != SOURCE_ACTIVITY_NODE)

    def worker_activity_nodes(self, worker_token):
        """The exact activity ports, including nested regions, for this use."""
        if worker_token not in self.worker_tokens:
            raise KeyError(worker_token)
        scope = self.control_graph.G.graph["thread_region_scope"]
        page = self.book.page(SOURCE_THREAD_REGION_MEMBER)
        return tuple(row[1] for row in page.scope_rows(scope)
                     if page.latest(row)[1] == worker_token
                     and page.latest(row)[0].page == SOURCE_ACTIVITY_NODE)

    def scheduling_graph(self, half):
        """One static region traversal; retain declared recurrence in the IR.

        This is the same contract as ActivitySchedule: a repeat edge denotes
        the next iteration, not a strict same-iteration schedule inequality.
        No other edge is removed to make a schedule possible.
        """
        if half not in {"dependency", "control"}:
            raise ValueError('thread region half must be dependency or control')
        graph = getattr(self, half + '_graph').G
        scope = self.control_graph.G.graph['thread_region_scope']
        page = self.book.page(SOURCE_THREAD_REGION_EDGE)
        def same_iteration(a, b):
            roles = page.latest((scope, half, a, b))[0]
            roles = (roles,) if isinstance(roles, str) else roles
            return any(role not in {'repeat', 'control:repeat', 'unadmitted', 'control:unadmitted'} for role in roles)
        view = nx.subgraph_view(graph, filter_edge=same_iteration)
        arranged = nx.DiGraph(view)
        excluded = {'repeat', 'control:repeat', 'unadmitted', 'control:unadmitted'}
        # ILPScheduler reads parents/children as well as NetworkX topology.
        # All representations of this scheduling view must have the same
        # declared iteration cut; the authoritative IR remains untouched.
        for node, data in arranged.nodes(data=True):
            data['parents'] = [(parent, role) for parent, role in graph.nodes[node].get('parents', ())
                               if view.has_edge(parent, node) and role not in excluded]
            data['children'] = [(child, role) for child, role in graph.nodes[node].get('children', ())
                                if view.has_edge(node, child) and role not in excluded]
        return arranged

    def worker_boundary_edges(self, worker_token, half="dependency"):
        """Retain shared storage, activation, join and synchronization crossings."""
        if half not in {"dependency", "control"}:
            raise ValueError("thread region half must be 'dependency' or 'control'")
        nodes = set((*self.worker_nodes(worker_token), *self.worker_activity_nodes(worker_token)))
        scope = self.dependency_graph.G.graph["thread_region_scope"]
        page = self.book.page(SOURCE_THREAD_REGION_EDGE)
        return tuple((row[2], row[3], page.latest(row)[0])
                     for row in page.scope_rows(scope)
                     if row[1] == half and ((row[2] in nodes) != (row[3] in nodes)))

    def schedule(self, preference):
        """Use the existing dependency scheduler on both actual region halves."""
        from .native_compiler_accelerators import dependency_levels, lexicographical_topological_order
        if preference not in {"asap", "alap"}:
            raise ValueError("thread region preference must be 'asap' or 'alap'")
        book = self.book
        scope = self.dependency_graph.G.graph["thread_region_scope"]
        if any(book.page(SOURCE_SYNC_OPERATOR).latest(row)[6] != 'source_host_participant_policy'
               for row in book.page(SOURCE_SYNC_OPERATOR).scope_rows(scope)):
            raise ValueError("thread region memory-barrier policy is unresolved at its authored wait")
        gaps = tuple(book.page(SOURCE_THREAD_REGION_GAP).latest(row)
                     for row in book.page(SOURCE_THREAD_REGION_GAP).scope_rows(scope))
        if gaps:
            raise ValueError(f"thread region has unresolved topology: {gaps[0][0]}")
        levels = []
        for half in ('dependency', 'control'):
            graph = self.scheduling_graph(half)
            order = lexicographical_topological_order(graph, key=int)
            levels.append(dependency_levels(graph, order=order, preference=preference))
        causes = [self.cell]
        for row in book.page(SOURCE_THREAD_REGION_MEMBER).scope_rows(scope):
            causes.append(book.latest_ref(SOURCE_THREAD_REGION_MEMBER, row))
        for row in book.page(SOURCE_THREAD_REGION_EDGE).scope_rows(scope):
            causes.append(book.latest_ref(SOURCE_THREAD_REGION_EDGE, row))
        book.post(
            SOURCE_THREAD_REGION_SCHEDULE, (scope, preference),
            tuple(tuple(sorted(result.items())) for result in levels),
            stage=CONTROL_PROGRAM_BUILD, provenance=Derived(tuple(causes)), mode=Mode.CONCORD,
        )
        return tuple(levels)


def ingest_thread_regions(source_graph):
    """Copy dependency/control subgraphs with exact activation qualification.

    Structured arms, loop repeat edges and source control block identities
    remain in the graph. Static scheduling uses ActivitySchedule's recurrence
    contract, never a linearization of mutually exclusive arms.
    """
    from ..transmogrifier.graph.graph_express2 import ProcessGraph
    from ..common.tensors.topological_reducer import (
        node_identity_cell, post_derived_node_identity, _set_operands,
    )
    book = current_identity_book()
    from .process_graph_facts import local_operands, local_value
    from .concorded_literals import origin_cell
    from .concordance_declarations import SOURCE_CLOSURE_CALLABLE, SOURCE_EXTRACTION_IDENTITY
    from ..common.tensors.topological_reducer import _source_numeric_scope

    def callable_identity(graph, node):
        captured = book.latest_ref(SOURCE_CLOSURE_CALLABLE, (_source_numeric_scope(graph), int(node)))
        if captured is not None:
            return book.page(SOURCE_CLOSURE_CALLABLE).cell(captured.row, captured.column)[0], captured
        origin = origin_cell(book, node_identity_cell(graph, int(node)))
        extracted = book.latest_ref(SOURCE_EXTRACTION_IDENTITY, (origin,))
        return (None, None) if extracted is None else (book.page(extracted.page).cell(extracted.row, extracted.column), extracted)

    table = source_graph.function_table
    control_views = {}

    def control_view(graph):
        key = graph.G.graph["lexical_read_scope"]
        if key not in control_views:
            control_views[key] = _control_projection(graph, book)
        return control_views[key]
    regions = []

    def capture_uses(create_cell):
        page = book.page(SOURCE_THREAD_CAPTURE_USE)
        return tuple((book.latest_ref(SOURCE_THREAD_CAPTURE_USE, row), page.latest(row))
                     for row in page.rows() if row[0].row == (create_cell,))
    entries = {int(entry.reference.address): entry.graph for entry in table
               if getattr(entry, "graph", None) is not None}
    for parent_ref, parent in entries.items():
        G = parent.G
        if not any(book.page(SOURCE_THREAD_ACTIVATION).latest(row)[1] == parent_ref
                   for row in book.page(SOURCE_THREAD_ACTIVATION).rows()):
            continue
        scope = G.graph["lexical_read_scope"]
        paths = []
        reason = None
        activation_cells = []
        control_causes = list(control_view(parent)[4])
        for row in book.page(SOURCE_THREAD_ACTIVATION).rows():
            activation = book.latest_ref(SOURCE_THREAD_ACTIVATION, row)
            body, owner, operands, starts, joins, owners = book.page(SOURCE_THREAD_ACTIVATION).cell(activation.row, activation.column)
            if owner != parent_ref:
                continue
            activation_cells.append(activation)
            target = book.page(SOURCE_THREAD_BODY).cell(body.row, body.column)[0]
            create = int(row[0].row[1])
            if len(starts) != 1 or not joins or target not in entries:
                reason = "unresolved activation handle or target"
                break
            worker = entries[target]
            control_causes.extend(control_view(worker)[4])
            paths.append((create, int(starts[0].row[1]), int(joins[0].row[1]), int(target)))
        sources = tuple(dict.fromkeys((*activation_cells, *control_causes)))
        cell = book.post(SOURCE_THREAD_REGION, (scope,),
                         (tuple(paths), reason or "source_graph"),
                         stage=CONTROL_PROGRAM_BUILD, provenance=Derived(sources), mode=Mode.CONCORD)
        if reason:
            G.graph["thread_regions"] = ()
            continue
        dependency = ProcessGraph(materialize_memory=False)
        control = ProcessGraph(materialize_memory=False)
        worker_tokens = tuple(node_identity_cell(parent, create) for create, _, _, _ in paths)
        for half, graph in (("dependency", dependency), ("control", control)):
            graph.G.graph.update(
                ingestion_value_scope=(scope, "thread_region", half),
                thread_region_scope=scope,
                worker_tokens=worker_tokens,
            )
        members = {}
        member_cells = {}
        originals = {}
        synchronizations = {}
        barrier_groups = {}
        for activation, function_ref, graph in (
            (None, parent_ref, parent),
            *((create, target, entries[target]) for create, _, _, target in paths),
        ):
            for n, data in graph.G.nodes(data=True):
                rid = len(members)
                members[(activation, function_ref, int(n))] = rid
                original = node_identity_cell(graph, int(n))
                activation_cell = None if activation is None else node_identity_cell(parent, activation)
                causes = tuple(c for c in (cell, original, activation_cell) if c is not None)
                member_cells[rid] = book.post(
                    SOURCE_THREAD_REGION_MEMBER, (scope, rid),
                    (original, activation_cell, function_ref),
                    stage=CONTROL_PROGRAM_BUILD, provenance=Derived(causes), mode=Mode.CONCORD,
                )
                originals[rid] = (activation, function_ref, int(n), data)
                sync_cell = None
                identity, callable_cell = callable_identity(graph, n)
                if identity == "threading.Barrier.wait":
                    wait_operands, wait_operands_cell = local_operands(graph, n)
                    receivers = [source for source, role in wait_operands if role in {'operand', 'arg:0'}]
                    receiver_causes = []
                    barrier_node = None
                    if len(receivers) == 1:
                        receiver = receivers[0]
                        if activation is None:
                            barrier_node = receiver
                        else:
                            matches = [(use, actual) for use, (actual, formal, _written)
                                       in capture_uses(activation_cell) if local_value(graph, formal) == receiver]
                            if len(matches) == 1:
                                barrier_node = local_value(parent, matches[0][1])
                                receiver_causes.append(matches[0][0])
                    barrier_identity, barrier_identity_cell = (None, None) if barrier_node is None else callable_identity(parent, barrier_node)
                    if barrier_identity == 'threading.Barrier':
                        barrier_cell = node_identity_cell(parent, barrier_node)
                        # Python's host Barrier has release/acquire memory
                        # synchronization for the participating paths. This
                        # source policy selects their effects through this
                        # exact object; it supplies no GPU/global scope mask.
                        from .synchronization import concord_host_barrier_policy
                        from .concordance_declarations import SOURCE_SYNCHRONIZATION
                        policy = concord_host_barrier_policy()
                        native_contract = book.latest_ref(SOURCE_SYNCHRONIZATION, (graph.G.graph['lexical_read_scope'], int(n)))
                        selector, memory_selector, semantics = book.page(SOURCE_THREAD_BARRIER_POLICY).cell(policy.row, policy.column)
                        sync_cell = book.post(
                            SOURCE_SYNC_OPERATOR, (scope, rid),
                            (original, activation_cell, barrier_cell,
                             (selector, barrier_cell), (memory_selector, barrier_cell), semantics,
                             'source_host_participant_policy'),
                            stage=CONTROL_PROGRAM_BUILD,
                            provenance=Derived((*causes, barrier_cell, policy, callable_cell, barrier_identity_cell,
                                                wait_operands_cell, *receiver_causes,
                                                *((native_contract,) if native_contract is not None else ()))), mode=Mode.CONCORD,
                        )
                        synchronizations[rid] = sync_cell
                        barrier_groups.setdefault(barrier_node, []).append(rid)
                        book.post(
                            CONTROL_BLOCK,
                            ((scope, "thread_region", "control"), ControlBlockKind.CONTROL_BARRIER, sync_cell),
                            ControlBlockFact(ControlBlockKind.CONTROL_BARRIER, None, (),
                                             (barrier_cell,), (), original,
                                             (sync_cell, ("barrier_participants", barrier_cell))),
                            stage=CONTROL_PROGRAM_BUILD,
                            provenance=Derived((sync_cell, original, barrier_cell)), mode=Mode.CONCORD,
                        )
                for projection in (dependency, control):
                    node_type = ("ControlBarrier" if projection is control else "MemoryBarrier") if sync_cell else data.get("type")
                    copied = {key: value for key, value in data.items()
                              if key not in {"expr_obj", "parents", "children", "attributes", "type"}}
                    copied.update(type=node_type, parents=[], children=[],
                                  attributes={**(data.get("attributes") or {}),
                                              "thread_region_member": member_cells[rid],
                                              "worker_token": activation_cell,
                                              "sync_operator": sync_cell})
                    projection.G.add_node(rid, **copied)
                    post_derived_node_identity(projection, rid, (*causes, *((sync_cell,) if sync_cell else ())), stage=CONTROL_PROGRAM_BUILD)

        activity_members = {}
        for activation, function_ref, graph in (
            (None, parent_ref, parent),
            *((create, target, entries[target]) for create, _, _, target in paths),
        ):
            token = None if activation is None else node_identity_cell(parent, activation)
            for point in control_view(graph)[0]:
                rid = len(member_cells)
                activity_members[(activation, function_ref, point)] = rid
                member_cells[rid] = book.post(SOURCE_THREAD_REGION_MEMBER, (scope, rid),
                    (point, token, function_ref), stage=CONTROL_PROGRAM_BUILD,
                    provenance=Derived(tuple(c for c in (cell, point, token) if c is not None)), mode=Mode.CONCORD)
                kind, owners = book.page(SOURCE_ACTIVITY_NODE).cell(point.row, point.column)
                for projection in (dependency, control):
                    projection.G.add_node(rid, type="ControlActivity", parents=[], children=[],
                        attributes={"thread_region_member": member_cells[rid], "worker_token": token,
                                    "source_activity": point, "source_kind": kind, "control_owners": owners})
                    post_derived_node_identity(projection, rid, (cell, member_cells[rid], point), stage=CONTROL_PROGRAM_BUILD)

        edge_inputs = {"dependency": {}, "control": {}}

        def edge(half, source, target, role, causes):
            if source == target:
                raise ValueError("thread region constraint is self-dependent")
            causes = tuple(dict.fromkeys((cell, member_cells[source], member_cells[target], *causes)))
            row = (scope, half, source, target)
            previous = book.latest_ref(SOURCE_THREAD_REGION_EDGE, row)
            mode = Mode.CONCORD
            roles = role
            if previous is not None:
                prior_roles, prior_causes = book.page(SOURCE_THREAD_REGION_EDGE).cell(previous.row, previous.column)
                prior_roles = (prior_roles,) if isinstance(prior_roles, str) else prior_roles
                merged = tuple(dict.fromkeys((*prior_roles, role)))
                roles = merged[0] if len(merged) == 1 else merged
                causes = tuple(dict.fromkeys((*prior_causes, *causes, previous)))
                mode = Mode.REVISE
            book.post(SOURCE_THREAD_REGION_EDGE, row, (roles, causes), stage=CONTROL_PROGRAM_BUILD,
                      provenance=Derived(causes), mode=mode)
            if (source, role) not in edge_inputs[half].setdefault(target, []):
                edge_inputs[half][target].append((source, role))

        # Preserve the real dependency edges, once for the parent and once
        # per activation's use of the common worker body.
        for activation, function_ref, graph in (
            (None, parent_ref, parent),
            *((create, target, entries[target]) for create, _, _, target in paths),
        ):
            for target in graph.G:
                operands, operand_cell = local_operands(graph, int(target))
                for source, role in operands:
                    edge("dependency", members[(activation, function_ref, source)],
                         members[(activation, function_ref, target)], role, (operand_cell,))
            _points, activity_edges, bindings, program, _causes = control_view(graph)
            completed = {}
            for node, ports, binding_cell in bindings:
                if ports is None:
                    continue
                rid = members[(activation, function_ref, node)]
                enter, leave = (activity_members[(activation, function_ref, point)] for point in ports)
                projection = book.post(SOURCE_THREAD_REGION_CONTROL, (scope, rid),
                    (program, binding_cell, enter, leave), stage=CONTROL_PROGRAM_BUILD,
                    provenance=Derived((cell, program, binding_cell, member_cells[rid],
                                        member_cells[enter], member_cells[leave])), mode=Mode.CONCORD)
                for view in (dependency, control):
                    view.G.nodes[rid]['attributes']['control_projection'] = projection
                edge('control', leave, rid, 'evaluation_completion', (projection,))
                completed.setdefault(ports[1], []).append((rid, projection))
            for before, after, relation, source_edge in activity_edges:
                a = activity_members[(activation, function_ref, before)]
                b = activity_members[(activation, function_ref, after)]
                # A resident operation occupies its source completion port.
                # Every successor waits for it, without ordering independent
                # operations that share that port. Repeat retains its edge kind.
                producers = completed.get(before, ((a, source_edge),))
                for producer, projection in producers:
                    edge('control', producer, b, relation, (source_edge, projection))
                if relation == 'unadmitted':
                    book.post(SOURCE_THREAD_REGION_GAP, (scope, a),
                        ('source control topology is unadmitted', source_edge), stage=CONTROL_PROGRAM_BUILD,
                        provenance=Derived((cell, source_edge)), mode=Mode.CONCORD)

        for create, start, join, target in paths:
            worker = entries[target]
            worker_token = node_identity_cell(parent, create)
            # These operations execute on the parent path, while naming the
            # same exact worker activation that owns the body occurrences.
            # The source region's lifecycle fact already authorizes this view.
            for operation in (create, start, join):
                operation_id = members[(None, parent_ref, operation)]
                for projection in (dependency, control):
                    projection.G.nodes[operation_id]["attributes"]["dispatch_worker_token"] = worker_token
            activation = book.latest_ref(SOURCE_THREAD_ACTIVATION, (worker_token,))
            join_cells = book.page(SOURCE_THREAD_ACTIVATION).cell(activation.row, activation.column)[4]
            join_ids = [members[(None, parent_ref, local_value(parent, identity))] for identity in join_cells]
            for join_id in join_ids:
                for projection in (dependency, control):
                    projection.G.nodes[join_id]['attributes']['dispatch_worker_token'] = worker_token
            start_id = members[(None, parent_ref, start)]
            worker_ids = [members[(create, target, n)] for n in worker.G]
            worker_ids.extend(rid for (use, ref, _point), rid in activity_members.items()
                              if use == create and ref == target)
            for rid in worker_ids:
                # Completion depends on every operation in this activation;
                # no edge to another activation is introduced here.
                edge("control", start_id, rid, "activation", ())
                for join_id in join_ids:
                    edge("control", rid, join_id, "completion", (activation, *join_cells))
            resolved = set()
            for use, (actual, formal, _written) in capture_uses(worker_token):
                source, destination = local_value(parent, actual), local_value(worker, formal)
                edge("dependency", members[(None, parent_ref, source)],
                     members[(create, target, destination)], "capture", (use, actual, formal))
                resolved.add(destination)
            activation = book.latest_ref(SOURCE_THREAD_ACTIVATION, (worker_token,))
            for row in book.page(SOURCE_THREAD_ARGUMENT).rows():
                if row[0] != activation:
                    continue
                argument = book.latest_ref(SOURCE_THREAD_ARGUMENT, row)
                actual, formal_binding, layout = book.page(SOURCE_THREAD_ARGUMENT).latest(row)
                formal = book.page(NAME_BINDING).cell(formal_binding.row, formal_binding.column).value_id
                edge("dependency", members[(None, parent_ref, local_value(parent, actual))],
                     members[(create, target, formal)], "argument", (argument, actual, formal_binding, layout))
                resolved.add(formal)
            # An Input may also be local loop state. Only the exact authored
            # signature and closure table declare activation boundary inputs.
            from .concordance_declarations import SOURCE_FORMAL_ORDER, SOURCE_CLOSURE_CAPTURE
            signature = book.latest_ref(SOURCE_FORMAL_ORDER, (_source_numeric_scope(worker),))
            if signature is None:
                raise MissingThreadControlTopology('worker has no ingested signature')
            worker_scope = worker.G.graph['lexical_read_scope']
            required = {}
            for parameter in book.page(SOURCE_FORMAL_ORDER).cell(signature.row, signature.column):
                formal = book.latest_ref(NAME_BINDING, (worker_scope, parameter, 0))
                if formal is None:
                    raise MissingThreadControlTopology('worker signature has no exact formal binding')
                required[book.page(NAME_BINDING).cell(formal.row, formal.column).value_id] = (signature, formal)
            body = book.page(SOURCE_THREAD_ACTIVATION).cell(activation.row, activation.column)[0]
            for capture in book.page(SOURCE_THREAD_BODY).cell(body.row, body.column)[2]:
                formal = book.page(SOURCE_CLOSURE_CAPTURE).cell(capture.row, capture.column)[2]
                required[book.page(NAME_BINDING).cell(formal.row, formal.column).value_id] = (body, capture, formal)
            for n, proofs in required.items():
                if n not in resolved:
                    source_cell = node_identity_cell(worker, n)
                    book.post(SOURCE_THREAD_REGION_GAP, (scope, members[(create, target, n)]),
                              ("activation input binding is unresolved", source_cell),
                              stage=CONTROL_PROGRAM_BUILD,
                              provenance=Derived((cell, source_cell, activation, *proofs)), mode=Mode.CONCORD)

        for barrier_node, waits in barrier_groups.items():
            from .concorded_literals import literal
            from .process_graph_facts import local_operands
            constructor_operands, constructor_cell = local_operands(parent, barrier_node)
            party_nodes = [n for n, role in constructor_operands if role in {'arg:0', 'kw:parties', 'kw:participants'}]
            parties, party_causes = literal(parent, party_nodes[0]) if len(party_nodes) == 1 else (None, ())
            # Static occurrences in one path form phases by existing
            # reachability, not source position or node numbering. Repeated
            # loop epochs remain represented by their retained repeat edges.
            topology = nx.DiGraph()
            topology.add_nodes_from(member_cells)
            topology.add_edges_from((a, b) for b, inputs in edge_inputs['control'].items()
                                    for a, role in inputs if role not in {'repeat', 'unadmitted'})
            by_path = {}
            for wait in waits:
                by_path.setdefault(originals[wait][0], []).append(wait)
            ordered = {}
            for path, occurrences in by_path.items():
                ranks = {wait: sum(nx.has_path(topology, previous, wait)
                                  for previous in occurrences if previous != wait) for wait in occurrences}
                if len(set(ranks.values())) != len(occurrences):
                    break
                ordered[path] = tuple(sorted(occurrences, key=ranks.__getitem__))
            counts = {len(occurrences) for occurrences in ordered.values()}
            if (type(parties) is not int or parties != len(by_path) or len(ordered) != len(by_path)
                    or len(counts) != 1):
                for wait in waits:
                    book.post(SOURCE_THREAD_REGION_GAP, (scope, wait),
                        ("barrier participant set or phase is unresolved", synchronizations[wait]),
                        stage=CONTROL_PROGRAM_BUILD,
                        provenance=Derived((cell, synchronizations[wait], constructor_cell, *party_causes)), mode=Mode.CONCORD)
                continue
            for phase in range(next(iter(counts))):
                waits = [occurrences[phase] for occurrences in ordered.values()]
                barrier_cell = node_identity_cell(parent, barrier_node)
                party_node = party_nodes[0]
                party_cell = node_identity_cell(parent, party_node)
                sync_cells = tuple(synchronizations[rid] for rid in waits)
                meeting = book.post(
                    SOURCE_CONTROL_RENDEZVOUS, (scope, barrier_cell, phase), (sync_cells, parties, party_cell),
                    stage=CONTROL_PROGRAM_BUILD, provenance=Derived((barrier_cell, party_cell, constructor_cell, *party_causes, *sync_cells)), mode=Mode.CONCORD,
                )
                release = len(member_cells)
                member_cells[release] = book.post(
                    SOURCE_THREAD_REGION_MEMBER, (scope, release), (barrier_cell, None, parent_ref),
                    stage=CONTROL_PROGRAM_BUILD, provenance=Derived((meeting, barrier_cell)), mode=Mode.CONCORD,
                )
                for projection in (dependency, control):
                    projection.G.add_node(release, type="ControlBarrierRelease", op="rendezvous",
                                          label="barrier completion", parents=[], children=[],
                                          attributes={"rendezvous": meeting, "worker_token": None})
                    post_derived_node_identity(projection, release, (meeting, barrier_cell), stage=CONTROL_PROGRAM_BUILD)
                following = {target for target, inputs in edge_inputs["control"].items()
                             if any(source in waits for source, _role in inputs)}
                dependent = {target for target, inputs in edge_inputs["dependency"].items()
                             if any(source in waits for source, _role in inputs)}
                for wait in waits:
                    edge("control", wait, release, "barrier_arrival", (meeting,))
                for target in following:
                    edge("control", release, target, "barrier_completion", (meeting,))
                for target in dependent - following:
                    edge("dependency", release, target, "control:barrier_completion", (meeting,))
        # Control constraints participate in dependency scheduling too. They
        # retain their explicit control cause; they are never memory fences.
        existing = {(source, target) for target, inputs in edge_inputs["dependency"].items()
                    for source, _role in inputs}
        for target, inputs in tuple(edge_inputs["control"].items()):
            for source, role in inputs:
                if (source, target) not in existing:
                    cause = book.latest_ref(SOURCE_THREAD_REGION_EDGE, (scope, "control", source, target))
                    edge("dependency", source, target, "control:" + role, (cause,))
                    existing.add((source, target))
        for half, projection in (("dependency", dependency), ("control", control)):
            for target, inputs in edge_inputs[half].items():
                _set_operands(projection, target, inputs, cause=THREAD_REGION_BUILD)
        region = ThreadRegion(dependency, control, tuple(paths), members, cell, book)
        regions.append(region)
        G.graph["thread_regions"] = (region,)
    source_graph.G.graph["thread_regions"] = tuple(regions)
    return tuple(regions)


def require_thread_regions(graph, book):
    scope = graph.G.graph['lexical_read_scope']
    row = book.page(SOURCE_THREAD_REGION).latest((scope,))
    if row is None or row[1] != 'source_graph':
        raise MissingThreadControlTopology('thread region requires ingested concorded control topology')
    return graph.G.graph['thread_regions']
