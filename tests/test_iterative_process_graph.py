import ast

from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_structural_ast_ingestion_exceeds_python_recursion_limit():
    expression = ast.Name(id="leaf", ctx=ast.Load())
    for _ in range(1500):
        expression = ast.UnaryOp(op=ast.USub(), operand=expression)

    graph = ProcessGraph(materialize_memory=False)
    root = graph.build_graph(expression)

    assert root == id(expression)
    assert root in graph.roots
    assert sum(
        isinstance(data.get("expr_obj"), ast.UnaryOp)
        for _, data in graph.G.nodes(data=True)
    ) == 1500
    assert any(
        isinstance(data.get("expr_obj"), ast.Name)
        for _, data in graph.G.nodes(data=True)
    )


def test_structural_ingestion_terminates_an_identity_cycle():
    class CyclicNode:
        def __init__(self):
            self.args = []

    cyclic = CyclicNode()
    cyclic.args.append(cyclic)

    graph = ProcessGraph(materialize_memory=False)
    root = graph.build_graph(cyclic)

    assert root == id(cyclic)
    assert graph.G.number_of_nodes() == 1
    assert graph.G.has_edge(root, root)
