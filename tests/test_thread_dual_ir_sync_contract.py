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
        if d.get("type") == "Barrier"
        and (d.get("attributes") or {}).get("extraction_identity") == "threading.Barrier"
    ]
    root_waits = [
        n for n, d in root.nodes(data=True)
        if (d.get("attributes") or {}).get("extraction_identity")
        == "threading.Barrier.wait"
    ]
    worker_waits = [n for n, d in worker.nodes(data=True) if d.get("type") == "wait"]
    assert len(constructors) == len(root_waits) == len(worker_waits) == 1
    assert root.has_edge(constructors[0], root_waits[0])
    assert root.edges[constructors[0], root_waits[0]]["role"] == "operand"
    closure = next(
        n for n, d in worker.nodes(data=True)
        if d.get("type") == "Input"
        and (d.get("attributes") or {}).get("binding_kind") == "closure"
        and (d.get("attributes") or {}).get("binding_name") == "gate"
    )
    assert worker.has_edge(closure, worker_waits[0])
    assert worker.edges[closure, worker_waits[0]]["role"] == "operand"


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
        if data.get("type") == "wait"
    }
    assert waits == {10: None, 11: "threading.Barrier.wait"}


@pytest.mark.xfail(strict=True, reason="the nested worker's captured binding has no exact parent-binding concordance")
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


@pytest.mark.xfail(strict=True, reason="thread starts and joins are not placed as concorded control-graph blocks")
def test_start_join_boundary_is_present_in_the_actual_control_graph(thread_dual_ir):
    module, _ = thread_dual_ir
    book = module.metadata["identity_book"]
    blocks = book.page(CONTROL_BLOCK)
    lifecycle = []
    for row in blocks.rows():
        fact = blocks.latest(row)
        if fact.kind.value != "dispatch" or not fact.extra:
            continue
        if fact.extra[0] not in {"thread_start", "thread_join"}:
            continue
        block_cell = book.latest_ref(CONTROL_BLOCK, row)
        assert book.latest_ref(
            CONTROL_BLOCK_PLACEMENT, (row[0], block_cell)
        ) is not None
        lifecycle.append(fact.extra[0])
    assert sorted(lifecycle) == [
        "thread_join", "thread_join", "thread_start", "thread_start",
    ]


@pytest.mark.xfail(strict=True, reason="the worker's barrier wait loses its callable identity through the closure")
def test_worker_barrier_wait_keeps_the_exact_authored_operation(barrier_dual_ir):
    _, graphs = barrier_dual_ir
    worker_waits = [
        data for _, data in graphs["worker"].G.nodes(data=True)
        if data.get("type") == "wait"
    ]
    assert len(worker_waits) == 1
    assert (worker_waits[0].get("attributes") or {}).get(
        "extraction_identity"
    ) == "threading.Barrier.wait"


@pytest.mark.xfail(strict=True, reason="the real dual IR has no paired control and memory barrier projections")
def test_barrier_wait_projects_into_both_dual_ir_graphs(barrier_dual_ir):
    module, graphs = barrier_dual_ir
    book = module.metadata["identity_book"]
    control_facts = tuple(
        book.page(CONTROL_BLOCK).latest(row)
        for row in book.page(CONTROL_BLOCK).rows()
    )
    assert any(fact.kind.value == "control_barrier" for fact in control_facts)
    assert any(
        data.get("type") == "MemoryBarrier"
        for graph in graphs.values()
        for _, data in graph.G.nodes(data=True)
    )
