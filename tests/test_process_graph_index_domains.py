import ast

from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _indexed_graph(label, *, node_type, constant=None):
    graph = ProcessGraph(materialize_memory=False)
    graph.G.clear()
    common = {
        "domain_node": None,
        "store_id": None,
        "parents": [],
        "children": [(99, "result")],
        "attributes": {},
    }
    graph.G.add_node(
        1,
        **common,
        label=label,
        type=node_type,
        expr_obj=label,
        constant=constant,
    )
    graph.G.add_node(
        2,
        **{**common, "parents": [(1, "index")]},
        label="array",
        type="IndexedBase",
        expr_obj=None,
    )
    graph.G.add_edge(1, 2)
    return graph


def test_symbolic_index_domain_is_dynamic_instead_of_float_coerced():
    label = ast.Name(id="index", ctx=ast.Load())
    graph = _indexed_graph(label, node_type="Name")

    graph.finalize_graph_with_outputs()

    assert graph.G.nodes[2]["domain_shape"] == "dynamic"
    assert graph.G.nodes[2]["index_symbols"] == [(label, "Name")]


def test_integral_constant_index_domain_retains_static_extent():
    graph = _indexed_graph("not-a-number", node_type="Constant", constant=3)

    graph.finalize_graph_with_outputs()

    assert graph.G.nodes[2]["domain_shape"] == (4,)
