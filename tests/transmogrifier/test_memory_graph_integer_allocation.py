import ast
import contextlib
import ctypes
import io
import networkx as nx
import pickle

from src.transmogrifier.graph.memory_graph import BitTensorMemoryGraph
from src.transmogrifier.graph.memory_graph.helpers import EdgeEntry, NodeEntry
from src.transmogrifier.graph.memory_graph.helpers.integer_region_allocator import (
    IntegerRegionAllocator,
    allocation_quantum,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.compiler.ssa_builder import process_graph_to_ssa_instrs
from src.compiler.process_graph_fusion import extract_clean_process_subgraph
from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology


def test_integer_allocator_uses_exact_aligned_disjoint_intervals():
    allocator = IntegerRegionAllocator(2048, reserved_prefix=128)
    node = allocator.allocate("node", 512, alignment=512)
    edge = allocator.allocate("edge", 128, alignment=128)

    assert node == 512
    assert edge == 128
    assert allocation_quantum((128, 512, 8)) == 512
    assert allocator.offsets("node") == (512,)
    assert allocator.offsets("edge") == (128,)


def test_integer_allocator_reuses_freed_interval_without_moving_neighbor():
    allocator = IntegerRegionAllocator(2048, reserved_prefix=128)
    first = allocator.allocate("edge", 128, alignment=128)
    neighbor = allocator.allocate("node", 512, alignment=512)
    allocator.release(first, label="edge")

    replacement = allocator.allocate("edge", 128, alignment=128)
    assert replacement == first
    assert allocator.offsets("node") == (neighbor,)


def test_memory_graph_exposes_networkx_contract_and_preserves_attributes():
    memory = BitTensorMemoryGraph(size=4096)
    graph = memory.G
    marker = object()

    assert not isinstance(graph, nx.DiGraph)

    graph.graph["marker"] = marker
    graph.add_node(10, payload=marker, children=set())
    graph.add_node(20, label="target")
    graph.add_edge(10, 20, extra={"flow"})

    assert graph.graph["marker"] is marker
    assert graph.nodes[10]["payload"] is marker
    graph.nodes[10]["children"].add(20)
    assert graph.nodes[10]["children"] == {20}
    assert graph.has_edge(10, 20)
    assert list(graph.in_edges(20, data=True))[0][2]["extra"] == {"flow"}
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.copy(as_view=False).nodes[10]["payload"] is marker

    node_records = memory.hard_memory.allocator.allocations("node")
    edge_records = memory.hard_memory.allocator.allocations("edge")
    assert len(node_records) == 2
    assert len(edge_records) == 1
    assert all(record.start % ctypes.sizeof(NodeEntry) == 0 for record in node_records)
    assert all(record.start % ctypes.sizeof(EdgeEntry) == 0 for record in edge_records)


def test_edge_record_retains_graph_endpoints():
    edge = EdgeEntry(123, 456)
    assert edge.src_ptr == 123
    assert edge.dst_ptr == 456


def test_meta_record_allocation_uses_integer_arena():
    memory = BitTensorMemoryGraph(size=4096)
    memory.add_meta_parent(77)
    allocations = memory.hard_memory.allocator.allocations("parent")
    assert len(allocations) == 1


def test_materialized_process_graph_supports_real_ast_ingestion():
    graph = ProcessGraph(materialize_memory=True)
    graph.build_from_ast("""
def kernel(x, y):
    value = x + y
    return value
""")

    assert [data["op"] for _, data in graph.G.nodes(data=True)] == [
        "input", "input", "add", "return",
    ]
    assert graph.G.number_of_edges() > 0
    assert len(graph.MG.hard_memory.allocator.allocations("node")) == 4
    assert [instruction.op for instruction in process_graph_to_ssa_instrs(graph)] == [
        "input", "input", "add", "return",
    ]


def test_deferred_layout_process_graph_has_no_networkx_storage_bypass():
    graph = ProcessGraph(materialize_memory=False)

    assert graph.materialize_memory is False
    assert graph.MG is graph.G.bit_tensor_memory_graph
    assert not isinstance(graph.G, nx.DiGraph)


@pytest.mark.parametrize("materialize_memory", (False, True))
def test_process_subgraph_keeps_its_own_memory_graph(materialize_memory):
    graph = ProcessGraph(materialize_memory=materialize_memory)
    graph.build_from_ast("""
def kernel(x, y):
    return x + y
""")

    extracted = extract_clean_process_subgraph(graph, graph.G)

    assert extracted.MG is extracted.G.bit_tensor_memory_graph
    assert extracted.MG is not graph.MG
    assert extracted.G.number_of_nodes() == graph.G.number_of_nodes()


@pytest.mark.parametrize("materialize_memory", (False, True))
def test_canonical_relabel_keeps_function_memory_storage(materialize_memory):
    graph = ProcessGraph(materialize_memory=materialize_memory)
    module = ast.parse("def kernel(x, y):\n    return x + y\n")
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
        reduce_abstract_tensor_topology(graph)

    entry = next(
        entry
        for entry in graph.function_table._entries.values()
        if entry.qualified_name == "kernel"
    )
    function = entry.graph
    assert not isinstance(function.G, nx.DiGraph)
    assert function.MG is function.G.bit_tensor_memory_graph
    assert function.G.graph["canonical_value_ids"] is True
    assert len(function.MG.hard_memory.allocator.allocations("node")) == len(
        function.G
    )


def test_bulk_networkx_operations_allocate_and_release_exact_records():
    memory = BitTensorMemoryGraph(size=4096)
    graph = memory.G
    graph.add_nodes_from(((1, {"kind": "a"}), (2, {"kind": "b"})))
    graph.add_edges_from(((1, 2, {"role": "value"}),))

    assert graph.nodes[1]["kind"] == "a"
    assert graph.edges[1, 2]["role"] == "value"
    assert len(memory.hard_memory.allocator.allocations("node")) == 2
    assert len(memory.hard_memory.allocator.allocations("edge")) == 1

    graph.remove_node(1)
    assert len(memory.hard_memory.allocator.allocations("node")) == 1
    assert memory.hard_memory.allocator.allocations("edge") == ()

    graph.add_nodes_from((3, 4))
    graph.add_edges_from(((2, 3), (3, 4)))
    graph.remove_edges_from(((2, 3),))
    assert len(memory.hard_memory.allocator.allocations("edge")) == 1
    graph.clear()
    assert memory.hard_memory.allocator.allocations() == ()


def test_networkx_algorithms_treat_memory_graph_as_the_graph_protocol():
    memory = BitTensorMemoryGraph(size=0)
    graph = memory.G
    graph.add_edges_from(((0, 1), (0, 2), (1, 3), (2, 3)))

    assert list(nx.lexicographical_topological_sort(graph)) == [0, 1, 2, 3]
    assert nx.ancestors(graph, 3) == {0, 1, 2}
    assert nx.descendants(graph, 0) == {1, 2, 3}
    assert nx.is_directed_acyclic_graph(graph)

    relabeled = nx.relabel_nodes(graph, {0: 10}, copy=True)
    assert not isinstance(relabeled, nx.DiGraph)
    assert relabeled.has_edge(10, 1)
    assert len(
        relabeled.bit_tensor_memory_graph.hard_memory.allocator.allocations(
            "node"
        )
    ) == len(relabeled)


def test_arena_growth_keeps_existing_record_offsets_and_payloads():
    memory = BitTensorMemoryGraph(size=4096)
    graph = memory.G
    for node in range(12):
        graph.add_node(node, ordinal=node)

    offsets = memory.hard_memory.allocator.offsets("node")
    assert len(offsets) == 12
    assert offsets[:7] == tuple(range(512, 4096, 512))
    assert offsets[7] >= 4096
    assert [graph.nodes[node]["ordinal"] for node in range(12)] == list(range(12))
    assert memory.get_node(0).node_id == 0


def test_memory_graph_spore_round_trip_preserves_bytes_and_can_grow():
    memory = BitTensorMemoryGraph(size=0)
    memory.G.add_edge(1, 2, role="value")
    original_bytes = bytes(memory.hard_memory.data[: memory.hard_memory.size])
    original_allocations = memory.hard_memory.allocator.allocations()

    restored = pickle.loads(pickle.dumps(memory))

    assert bytes(restored.hard_memory.data[: restored.hard_memory.size]) == original_bytes
    assert restored.hard_memory.allocator.allocations() == original_allocations
    assert restored.G.bit_tensor_memory_graph is restored
    assert restored.hard_memory.graph is restored
    assert restored.G.edges[1, 2]["role"] == "value"
    restored.G.add_edge(2, 3, role="continued")
    assert restored.G.edges[2, 3]["role"] == "continued"
