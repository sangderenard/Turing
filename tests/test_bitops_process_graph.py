from src.compiler.bitops_process_graph import expand_bitops_process_graph
from src.compiler.ssa_builder import process_graph_to_ssa_instrs
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _graph(source):
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(source)
    return graph


def _ops(graph):
    import networkx as nx

    return [graph.G.nodes[node]["op"] for node in nx.topological_sort(graph.G)]


def test_bitxor_expands_to_the_real_turing_nand_algebra():
    source = _graph(
        """
def kernel(x, y):
    return x ^ y
"""
    )
    lowered = expand_bitops_process_graph(source, bit_width=4)
    ops = _ops(lowered)

    assert "bitxor" not in ops
    assert "nand" in ops
    assert "return" in ops
    assert all(
        data["control"].get("lowered_by") == "bitops"
        for _, data in lowered.G.nodes(data=True)
        if data["op"] == "nand"
    )
    nand_payload = next(
        data
        for _, data in lowered.G.nodes(data=True)
        if data["op"] == "nand"
    )
    assert nand_payload["bit_quanta"]["quanta"] == 4
    assert nand_payload["bit_quanta"]["bits_per_quantum"] == 1
    assert len(nand_payload["bit_quanta"]["source_nodes"]) == 2


def test_partial_expansion_is_explicit_and_ssa_consumable():
    source = _graph(
        """
def kernel(x, y, n):
    z = (x + y) * 3
    if z > n:
        z = z ^ n
    return z
"""
    )
    lowered = expand_bitops_process_graph(source, bit_width=4)
    ops = _ops(lowered)

    assert "add" not in ops
    assert "mul" not in ops
    assert "bitxor" not in ops
    assert "gt" in ops
    gt = next(
        data
        for _, data in lowered.G.nodes(data=True)
        if data["op"] == "gt"
    )
    assert gt["attributes"]["bitops_status"] == "unexpanded"

    instrs = process_graph_to_ssa_instrs(lowered, schedule="asap")
    assert any(instr.op == "nand" for instr in instrs)
    assert any(instr.op == "gt" for instr in instrs)
