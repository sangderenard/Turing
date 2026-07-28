import contextlib
import io
from types import SimpleNamespace

import networkx as nx

from src.common.tensors.numpy_backend import NumPyTensorOperations
from src.compiler.process_graph_callable import make_process_graph_callable


def _elementwise_graph():
    graph = nx.DiGraph()
    graph.add_node(1, type="Input", label="x", parents=[])
    graph.add_node(2, type="sin", label="sin(x)", parents=[(1, "arg0")])
    graph.add_node(3, type="Store", label="result", parents=[(2, "result")])
    return SimpleNamespace(graph=graph, G=graph, levels={1: 0, 2: 1, 3: 2})


def test_process_graph_callable_executes_and_captures_forward_path():
    with contextlib.redirect_stdout(io.StringIO()):
        callable_graph = make_process_graph_callable(_elementwise_graph())

    value = NumPyTensorOperations.tensor([0.0, 1.0])
    (result,) = callable_graph(floatx=value)
    captured = callable_graph.capture_forward(floatx=value)

    assert result.tolist() == value.sin().tolist()
    assert [step.op_name for step in captured.program.steps] == ["sin"]
    assert captured.program.outputs.keys() == {"result"}
    assert tuple(captured.feeds.values()) == (value,)
