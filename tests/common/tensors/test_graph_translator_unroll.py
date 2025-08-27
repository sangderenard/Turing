import networkx as nx

from src.common.tensors.graph_translator import GraphTranslator


def test_graph_translator_handles_self_loops():
    trace: list[str] = []
    g = nx.DiGraph()
    g.add_node("A", op=lambda: trace.append("A"))
    g.add_node("B", op=lambda: trace.append("B"))
    g.add_edge("A", "A")
    g.add_edge("A", "B")

    trans = GraphTranslator(g)
    order = trans.schedule()
    trans.execute()

    assert trace == ["A", "B"]
    assert order == ["A", "B"]
    assert g.nodes["A"]["level"] < g.nodes["B"]["level"]
    assert {g.nodes[n]["level"] for n in g} == {0, 1}

