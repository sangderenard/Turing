from src.compiler.bitops_process_graph import expand_bitops_process_graph
from src.compiler.nodus_graph_ir import process_graph_to_nodus_graph_ir
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _graph(source):
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(source)
    return graph


def test_process_graph_serializes_as_nodus_tensor_tools():
    graph = _graph(
        """
def kernel(x, y):
    return (x + y) * 3
"""
    )
    source = process_graph_to_nodus_graph_ir(graph)

    assert 'tensor_node("add")' in source
    assert 'tensor_node("mul")' in source
    assert 'tensor_node("return")' in source
    assert 'tensor_input(' in source
    assert 'tensor_output(' in source
    assert 'connect(' in source
    assert '"process.constant", 3' in source


def test_nodus_export_keeps_bitbit_quanta_accounting():
    graph = expand_bitops_process_graph(
        _graph(
            """
def kernel(x, y):
    return x ^ y
"""
        ),
        bit_width=8,
    )
    source = process_graph_to_nodus_graph_ir(graph)
    assert '"bitbit.quanta", 8' in source
    assert '"bitbit.bitsforbits", 1' in source


def test_nodus_export_uses_shared_canonical_comparison_names():
    source = process_graph_to_nodus_graph_ir(
        _graph(
            """
def kernel(x, y):
    return x > y
"""
        )
    )
    assert 'tensor_node("greater")' in source
    assert 'tensor_node("gt")' not in source
