"""Source-driven dual-IR contract for thread regions and barrier operations.

The expected failures name missing ingestion facts. No test constructs a
stand-in SSA program or executes a worker on a Python thread.
"""

import pytest

from src.compiler.concordance_declarations import (
    CONTROL_BLOCK, CONTROL_BLOCK_PLACEMENT, NAME_BINDING,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from tests.test_python_two_threads_shared_location import SOURCE


CONTRACT = ExtractionContract("extraction_contracts/program_extraction.yaml")
BARRIER_SOURCE = """\
import threading

def root():
    gate = threading.Barrier(2)
    def worker():
        gate.wait()
    first = threading.Thread(target=worker)
    first.start()
    gate.wait()
    first.join()
    return 1
"""
SAME_NAME_SOURCE = """\
import threading

class Other:
    def wait(self):
        return 3

def root():
    gate = threading.Barrier(1)
    other = Other()
    x = other.wait()
    y = gate.wait()
    return x + y
"""


def _compile(source):
    captured = []
    module, _, _ = lower_ast_source_to_ssa(
        source, "root", extraction_contract=CONTRACT,
        resolved_process_graph_sink=captured.append, progress_bars=False,
    )
    graphs = {
        entry.graph.G.graph.get("function_name"): entry.graph
        for entry in captured[0].function_table
        if getattr(entry, "graph", None) is not None
    }
    return module, graphs


def _operations(graph, operation):
    return tuple(
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if ((data.get("attributes") or {}).get("dispatch_operation") or {}).get(
            "operation"
        ) == operation
    )


@pytest.fixture(scope="module")
def thread_dual_ir():
    return _compile(SOURCE)


@pytest.fixture(scope="module")
def barrier_dual_ir():
    return _compile(BARRIER_SOURCE)


def test_actual_dual_ir_has_two_handle_paths_one_target_and_worker_effects(thread_dual_ir):
    _, graphs = thread_dual_ir
    root, worker = graphs["root"].G, graphs["worker"].G
    creates = _operations(graphs["root"], "thread_create")
    starts = _operations(graphs["root"], "thread_start")
    joins = _operations(graphs["root"], "thread_join")
    assert len(creates) == len(starts) == len(joins) == 2

    targets = tuple(
        parent for create in creates
        for parent, _, edge in root.in_edges(create, data=True)
        if edge.get("role") == "kw:target"
    )
    assert len(targets) == 2 and targets[0] == targets[1]
    target = root.nodes[targets[0]]
    assert target["type"] == "StaticReference"
    assert (target.get("attributes") or {}).get(
        "first_class_function_ref"
    ) == worker.graph["function_ref"]

    for create in creates:
        receivers = {
            child for _, child, edge in root.out_edges(create, data=True)
            if edge.get("role") == "arg:0"
        }
        assert len(receivers & set(starts)) == 1
        assert len(receivers & set(joins)) == 1

    closure_inputs = tuple(
        node_id for node_id, data in worker.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_kind") == "closure"
        and (data.get("attributes") or {}).get("binding_name") == "shared"
    )
    assert len(closure_inputs) == 1
    reads = [n for n, d in worker.nodes(data=True) if d.get("type") == "Indexed"]
    stores = [n for n, d in worker.nodes(data=True) if d.get("type") == "IndexedStore"]
    assert len(reads) == len(stores) == 1
    assert all(
        worker.has_edge(closure_inputs[0], node)
        and worker.edges[closure_inputs[0], node]["role"] == "base"
        for node in (*reads, *stores)
    )


def test_actual_barrier_constructor_and_wait_are_distinct_graph_nodes(barrier_dual_ir):
    _, graphs = barrier_dual_ir
    root, worker = graphs["root"].G, graphs["worker"].G
    constructors = [
        n for n, d in root.nodes(data=True)
        if d.get("type") == "BarrierCreate"
        and (d.get("attributes") or {}).get("extraction_identity") == "threading.Barrier"
    ]
    root_waits = [
        n for n, d in root.nodes(data=True)
        if (d.get("attributes") or {}).get("extraction_identity")
        == "threading.Barrier.wait"
    ]
    worker_waits = [n for n, d in worker.nodes(data=True) if d.get("type") == "BarrierWait"]
    assert len(constructors) == len(root_waits) == len(worker_waits) == 1
    assert root.has_edge(constructors[0], root_waits[0])
    assert root.edges[constructors[0], root_waits[0]]["role"] == "arg:0"
    closure = next(
        n for n, d in worker.nodes(data=True)
        if d.get("type") == "Input"
        and (d.get("attributes") or {}).get("binding_kind") == "closure"
        and (d.get("attributes") or {}).get("binding_name") == "gate"
    )
    assert worker.has_edge(closure, worker_waits[0])
    assert worker.edges[closure, worker_waits[0]]["role"] == "arg:0"


def test_same_spelled_wait_is_not_the_barrier_operation():
    captured = []
    lower_ast_source_to_ssa(
        SAME_NAME_SOURCE, "root", extraction_contract=CONTRACT,
        stop_after_compilation_unit_plan=True,
        resolved_process_graph_sink=captured.append, progress_bars=False,
    )
    root = next(
        entry.graph.G for entry in captured[0].function_table
        if getattr(entry, "graph", None) is not None
        and entry.graph.G.graph.get("function_name") == "root"
    )
    waits = {
        (data.get("source_span") or {}).get("line"):
            (data.get("attributes") or {}).get("extraction_identity")
        for _, data in root.nodes(data=True)
        if data.get("type") in {"wait", "BarrierWait"}
    }
    assert waits == {10: None, 11: "threading.Barrier.wait"}


def test_actual_dual_ir_concords_capture_to_the_parent_binding(thread_dual_ir):
    module, graphs = thread_dual_ir
    book = module.metadata["identity_book"]
    root_scope = tuple(graphs["root"].G.graph["lexical_read_scope"])
    worker_scope = tuple(graphs["worker"].G.graph["lexical_read_scope"])
    root_binding = book.latest_ref(NAME_BINDING, (root_scope, "shared", 0))
    worker_binding = book.latest_ref(NAME_BINDING, (worker_scope, "shared", 0))
    assert root_binding is not None and worker_binding is not None
    root_allocations = [
        node_id for node_id, data in graphs["root"].G.nodes(data=True)
        if data.get("type") == "List"
    ]
    worker_inputs = [
        node_id for node_id, data in graphs["worker"].G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_kind") == "closure"
        and (data.get("attributes") or {}).get("binding_name") == "shared"
    ]
    assert len(root_allocations) == len(worker_inputs) == 1
    assert book.page(NAME_BINDING).latest(
        root_binding.row
    ).value_id == root_allocations[0]
    assert book.page(NAME_BINDING).latest(
        worker_binding.row
    ).value_id == worker_inputs[0]

    # The relationship may live on any capture/closure page, but its receipt
    # must derive from both exact binding cells, not their common spelling.
    joined = []
    for name in book.pages:
        if "capture" not in name and "closure" not in name:
            continue
        page = book.page(name)
        for row in page.rows():
            fact_cell = book.latest_ref(page, row)
            sources = {source for source, _ in book.edges_into(fact_cell)}
            if {root_binding, worker_binding} <= sources:
                joined.append(fact_cell)
    assert joined, "no capture/closure fact derives from both binding identities"


def test_start_join_boundary_is_present_in_the_actual_control_graph(thread_dual_ir):
    from src.compiler.concordance_declarations import SOURCE_PROCESS_OPERATION
    from src.compiler.identity_concordance import Ref
    module, _ = thread_dual_ir
    book = module.metadata["identity_book"]
    blocks = book.page(CONTROL_BLOCK)
    lifecycle = {}
    for row in blocks.rows():
        fact = blocks.latest(row)
        if fact.kind.value != "dispatch" or not fact.extra:
            continue
        if fact.extra[0] not in {"thread_start", "thread_join"}:
            continue
        block_cell = book.latest_ref(CONTROL_BLOCK, row)
        assert book.latest_ref(CONTROL_BLOCK_PLACEMENT, (row[0], block_cell)) is not None
        sources = [cell for cell in fact.extra
                   if isinstance(cell, Ref) and cell.page == SOURCE_PROCESS_OPERATION]
        assert len(sources) == 1
        view = lifecycle.setdefault(row[0], {})
        assert sources[0] not in view, "one control view duplicated an authored operation"
        view[sources[0]] = fact.extra[0]
    assert lifecycle
    for view in lifecycle.values():
        assert sorted(view.values()) == [
            "thread_join", "thread_join", "thread_start", "thread_start",
        ]
    # Ingestion and planned control are views of the same four operations.
    # Counting all book rows as four would conflate those explicit scopes.
    assert len({frozenset(view) for view in lifecycle.values()}) == 1


def test_worker_barrier_wait_keeps_the_exact_authored_operation(barrier_dual_ir):
    _, graphs = barrier_dual_ir
    worker_waits = [
        data for _, data in graphs["worker"].G.nodes(data=True)
        if data.get("type") == "BarrierWait"
    ]
    assert len(worker_waits) == 1
    assert (worker_waits[0].get("attributes") or {}).get(
        "extraction_identity"
    ) == "threading.Barrier.wait"


def test_source_run_to_join_region_schedules_both_activations_without_ordering_them(thread_dual_ir):
    import networkx as nx
    module, graphs = thread_dual_ir
    (region,) = graphs["root"].G.graph["thread_regions"]
    assert len(region.activations) == 2
    worker_ref = graphs["worker"].G.graph["function_ref"]
    root_ref = graphs["root"].G.graph["function_ref"]
    paths = []
    assert len(set(region.worker_tokens)) == 2
    for token, (create, start, join, target) in zip(region.worker_tokens, region.activations):
        assert target == worker_ref
        nodes = {region.member_ids[(create, target, n)] for n in graphs["worker"].G}
        assert nodes == set(region.worker_nodes(token))
        assert token.row[1] == create
        paths.append(nodes)
        start_id = region.member_ids[(None, root_ref, start)]
        join_id = region.member_ids[(None, root_ref, join)]
        assert all(region.control_graph.G.has_edge(start_id, n)
                   and region.control_graph.G.has_edge(n, join_id) for n in nodes)
        for graph in (region.dependency_graph, region.control_graph):
            assert all(graph.G.nodes[n]["attributes"]["worker_token"] == token for n in nodes)
            assert graph.G.nodes[start_id]["attributes"]["dispatch_worker_token"] == token
            assert graph.G.nodes[join_id]["attributes"]["dispatch_worker_token"] == token
        crossings = region.worker_boundary_edges(token)
        assert any(role == "capture" and source not in nodes and target in nodes
                   for source, target, role in crossings)
        assert region.worker_boundary_edges(token, "control")
    allocation = next(n for n, d in graphs["root"].G.nodes(data=True) if d.get("type") == "List")
    assert region.dependency_graph.G.nodes[region.member_ids[(None, root_ref, allocation)]]["attributes"]["worker_token"] is None
    before = tuple(tuple(g.G.edges) for g in (region.dependency_graph, region.control_graph))
    for preference in ("asap", "alap"):
        levels = region.schedule(preference)
        for graph, schedule in zip((region.dependency_graph, region.control_graph), levels):
            assert all(schedule[a] < schedule[b] for a, b in graph.G.edges)
            assert not any(nx.has_path(graph.G, a, b) or nx.has_path(graph.G, b, a)
                           for a in paths[0] for b in paths[1])
    assert before == tuple(tuple(g.G.edges) for g in (region.dependency_graph, region.control_graph))
    from src.compiler.identity_concordance import CorrelationTable
    assert CorrelationTable._source_closure_findings(module) == []
    assert CorrelationTable._source_thread_region_findings(module) == []


def test_captured_same_spelled_method_stays_unrelated():
    source = """\
import threading
class Other:
    def wait(self):
        return 3
def root():
    gate = Other()
    def worker():
        gate.wait()
    first = threading.Thread(target=worker)
    first.start()
    first.join()
    return 1
"""
    _, graphs = _compile(source)
    waits = [d for _, d in graphs["worker"].G.nodes(data=True) if d.get("type") == "wait"]
    assert len(waits) == 1
    assert (waits[0].get("attributes") or {}).get("extraction_identity") is None
    (region,) = graphs["root"].G.graph["thread_regions"]
    assert not any(d.get("type") == "MemoryBarrier" for _, d in region.dependency_graph.G.nodes(data=True))


def test_captured_barrier_alias_uses_the_exact_parent_value():
    source = BARRIER_SOURCE.replace(
        "    def worker():\n        gate.wait()",
        "    alias = gate\n    def worker():\n        alias.wait()",
    )
    _, graphs = _compile(source)
    waits = [d for _, d in graphs["worker"].G.nodes(data=True) if d.get("type") == "BarrierWait"]
    assert len(waits) == 1
    assert waits[0]["attributes"]["extraction_identity"] == "threading.Barrier.wait"


def test_barrier_wait_projects_into_both_dual_ir_graphs(barrier_dual_ir):
    module, graphs = barrier_dual_ir
    book = module.metadata["identity_book"]
    control_facts = tuple(
        book.page(CONTROL_BLOCK).latest(row)
        for row in book.page(CONTROL_BLOCK).rows()
    )
    assert any(fact.kind.value == "control_barrier" for fact in control_facts)
    (region,) = graphs["root"].G.graph["thread_regions"]
    from src.compiler.concordance_declarations import SOURCE_SYNC_OPERATOR
    controls = {n: d["attributes"]["sync_operator"]
                for n, d in region.control_graph.G.nodes(data=True)
                if d.get("type") == "ControlBarrier"}
    memories = {n: d["attributes"]["sync_operator"]
                for n, d in region.dependency_graph.G.nodes(data=True)
                if d.get("type") == "MemoryBarrier"}
    assert len(controls) == 2 and controls == memories
    for sync in controls.values():
        fact = book.page(SOURCE_SYNC_OPERATOR).cell(sync.row, sync.column)
        assert fact[3] == ("barrier_participants", fact[2])
        assert fact[4:7] == (("barrier_participant_effects", fact[2]), 8, "source_host_participant_policy")
    releases = [n for n, d in region.control_graph.G.nodes(data=True)
                if d.get("type") == "ControlBarrierRelease"]
    assert len(releases) == 1
    assert all(region.control_graph.G.has_edge(wait, releases[0]) for wait in controls)
    (token,) = region.worker_tokens
    assert any(target == releases[0] and role == "barrier_arrival"
               for _source, target, role in region.worker_boundary_edges(token, "control"))
    for preference in ('asap', 'alap'):
        for half, levels in zip(('dependency', 'control'), region.schedule(preference)):
            assert all(levels[a] < levels[b] for a, b in region.scheduling_graph(half).edges)


def test_region_copies_source_dependencies_and_records_their_operands(thread_dual_ir):
    from src.compiler.concordance_declarations import SOURCE_THREAD_REGION_EDGE, PROCESS_GRAPH_OPERANDS
    module, graphs = thread_dual_ir
    book = module.metadata['identity_book']
    (region,) = graphs['root'].G.graph['thread_regions']
    scope = region.dependency_graph.G.graph['thread_region_scope']
    views = [(None, graphs['root'])] + [(create, graphs['worker'])
                                      for create, _start, _join, _target in region.activations]
    for activation, graph in views:
        reference = graph.G.graph['function_ref']
        for source, target, data in graph.G.edges(data=True):
            a = region.member_ids[(activation, reference, source)]
            b = region.member_ids[(activation, reference, target)]
            assert region.dependency_graph.G.has_edge(a, b)
            role, causes = book.page(SOURCE_THREAD_REGION_EDGE).latest((scope, 'dependency', a, b))
            assert role == data.get('role', 'value')
            assert any(cause.page == PROCESS_GRAPH_OPERANDS for cause in causes)


def test_authored_join_before_next_start_retains_comparative_order():
    import networkx as nx
    source = SOURCE.replace('    second.start()\n    first.join()',
                            '    first.join()\n    second.start()')
    _, graphs = _compile(source)
    (region,) = graphs['root'].G.graph['thread_regions']
    first, second = region.worker_tokens
    for graph in (region.control_graph, region.dependency_graph):
        assert all(nx.has_path(graph.G, a, b)
                   for a in region.worker_nodes(first) for b in region.worker_nodes(second))
    for preference in ('asap', 'alap'):
        for graph, levels in zip((region.dependency_graph, region.control_graph), region.schedule(preference)):
            assert all(levels[a] < levels[b] for a, b in graph.G.edges)
