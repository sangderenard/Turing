from src.common.tensors.numpy_backend import NumPyTensorOperations
from src.compiler.abstract_tensor_bitops import (
    AbstractTensorBitOpsTranslator,
    abstract_tensor_bitops_factory,
)
from src.compiler.bitops_process_graph import expand_bitops_process_graph
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _like():
    return NumPyTensorOperations._tensor_from_list([0])


def test_turing_calculus_uses_abstract_tensor_as_opaque_carrier():
    translator = AbstractTensorBitOpsTranslator(4, like=_like())

    left = translator.bits_from_int(0b1011)
    right = translator.bits_from_int(0b0110)

    assert translator.int_from_bits(
        translator.apply_bits("bitand", left, right)
    ) == 0b0010
    assert translator.int_from_bits(
        translator.apply_bits("bitxor", left, right)
    ) == 0b1101
    assert translator.int_from_bits(
        translator.apply_bits("add", left, right)
    ) == 0b0001
    assert {
        node.op for node in translator.graph.nodes
    } <= {
        "nand",
        "sigma_L",
        "sigma_R",
        "concat",
        "slice",
        "mu",
        "length",
        "zeros",
    }


def test_ast_process_graph_can_expand_bitops_through_abstract_tensor():
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(
        """
def kernel(x, y):
    return (x + y) ^ 3
"""
    )

    lowered = expand_bitops_process_graph(
        graph,
        bit_width=4,
        translator_factory=abstract_tensor_bitops_factory(_like()),
    )
    ops = [data["op"] for _, data in lowered.G.nodes(data=True)]

    assert "add" not in ops
    assert "bitxor" not in ops
    assert "nand" in ops
    assert "return" in ops
