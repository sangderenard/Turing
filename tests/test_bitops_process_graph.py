from src.compiler.ast_process_graph import ast_to_process_graph
from src.compiler.bitops_process_graph import expand_bitops_process_graph
from src.compiler.ssa_builder import process_graph_to_ssa_instrs


def _ops(graph):
    return [data["process_op"].op for _, data in graph.G.nodes(data=True)]


def test_bitxor_expands_to_the_real_turing_nand_algebra():
    source = ast_to_process_graph(
        """
def kernel(x, y):
    return x ^ y
"""
    )
    lowered = expand_bitops_process_graph(source, bit_width=4)
    ops = _ops(lowered)

    assert "bitxor" not in ops
    assert "nand" in ops
    assert ops[-1] == "return"
    assert all(
        data["process_op"].control.get("lowered_by") == "bitops"
        for _, data in lowered.G.nodes(data=True)
        if data["process_op"].op == "nand"
    )


def test_partial_expansion_is_explicit_and_ssa_consumable():
    source = ast_to_process_graph(
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
        data["process_op"]
        for _, data in lowered.G.nodes(data=True)
        if data["process_op"].op == "gt"
    )
    assert gt.attributes["bitops_status"] == "unexpanded"

    instrs = process_graph_to_ssa_instrs(lowered, schedule="asap")
    assert any(instr.op == "nand" for instr in instrs)
    assert any(instr.op == "gt" for instr in instrs)
