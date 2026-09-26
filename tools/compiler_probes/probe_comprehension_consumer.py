import ast
import contextlib
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.loop_composer import LoopBackendCapabilities, LoopComposer
from src.transmogrifier.graph.graph_express2 import ProcessGraph


source = '''
def kernel(nodes, rows):
    node_index = {name: index for index, name in enumerate(nodes)}
    result = []
    for name, value in rows:
        index = node_index[name]
        result.append(index + value)
    return result
'''
graph = ProcessGraph(materialize_memory=False)
with contextlib.redirect_stdout(io.StringIO()):
    graph.build_from_ast(ast.parse(source))
reduce_abstract_tensor_topology(graph)
graph = graph.function_table.entry("kernel").graph
composer = LoopComposer(LoopBackendCapabilities(
    backend="glsl", native_for=True, native_while=True,
    dynamic_bounds=True, unroll_limit=8,
))
for plan in composer.discover(graph):
    expression = graph.G.nodes[plan.loop.node_id].get("expr_obj")
    print("LOOP", plan.loop.node_id, ast.unparse(expression))
    print("NODE", graph.G.nodes[plan.loop.node_id])
    for node_id in plan.loop.body_nodes:
        data = graph.G.nodes[node_id]
        expression = data.get("expr_obj")
        print("BODY", node_id, data.get("type"),
              ast.unparse(expression) if isinstance(expression, ast.AST) else None,
              data.get("parents"), data.get("source_span"))
for node_id, data in graph.G.nodes(data=True):
    expression = data.get("expr_obj")
    if isinstance(expression, ast.DictComp):
        print("DICTCOMP", node_id, data)
