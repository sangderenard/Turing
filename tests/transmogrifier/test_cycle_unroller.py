import networkx as nx

from src.transmogrifier.cycle_unroller import unroll_self_edges


def test_unroll_self_edges_basic():
    g = nx.DiGraph()
    g.add_node("A", color="red")
    g.add_node("B")
    g.add_edge("A", "A", weight=1)
    g.add_edge("B", "A", weight=2)
    g.add_edge("A", "B", weight=3)

    unrolled = unroll_self_edges(g)

    assert set(unrolled.nodes) == {"A_v0", "A_v1", "B"}
    assert not unrolled.has_edge("A_v1", "A_v1")
    assert unrolled.has_edge("A_v0", "A_v1")
    assert unrolled.edges["A_v0", "A_v1"]["weight"] == 1
    assert unrolled.has_edge("B", "A_v0")
    assert unrolled.edges["B", "A_v0"]["weight"] == 2
    assert unrolled.has_edge("A_v1", "B")
    assert unrolled.edges["A_v1", "B"]["weight"] == 3
    assert unrolled.nodes["A_v0"]["color"] == "red"
    assert unrolled.nodes["A_v0"]["source"] == "A"
    assert unrolled.nodes["A_v1"]["version"] == 1


def test_unroll_multiple_self_loops():
    g = nx.DiGraph()
    g.add_edge("X", "X")
    g.add_edge("X", "Y")
    g.add_edge("Y", "X")
    g.add_edge("Y", "Y")

    unrolled = unroll_self_edges(g)

    expected_nodes = {"X_v0", "X_v1", "Y_v0", "Y_v1"}
    assert set(unrolled.nodes) == expected_nodes
    assert unrolled.has_edge("X_v0", "X_v1")
    assert unrolled.has_edge("Y_v0", "Y_v1")
    assert unrolled.has_edge("X_v1", "Y_v0")
    assert unrolled.has_edge("Y_v1", "X_v0")
