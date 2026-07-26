from src.compiler.ast_process_graph import ast_to_process_graph
from src.compiler.bitops_process_graph import expand_bitops_process_graph
from src.compiler.nodus_graph_ir import process_graph_to_nodus_graph_ir


def test_process_graph_serializes_as_nodus_tensor_tools():
    graph = ast_to_process_graph(
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
        ast_to_process_graph(
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
