import networkx as nx

from src.common.tensors.graph_translator import GraphTranslator


def test_schedule_handles_existing_parent_child_attrs() -> None:
    g = nx.DiGraph()
    g.add_node("a", op=lambda: None, parents=[("p", "dep")], children=[("c", "dep")])
    g.add_node(
        "b", op=lambda: None, parents=[("q", "dep")], children=[("d", "dep")], label="x"
    )
    g.add_edge("a", "b")

    translator = GraphTranslator(g)
    order = translator.schedule()
    assert order == ["a", "b"]
