import ast
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.compiler.loop_composer import LoopComposer, LoopBackendCapabilities


def test_pruned_specialized_arm_cannot_resurrect_return_from_shared_slot():
    tree = ast.parse('''
while True:
    if not rollback:
        return metrics, dt * 3
    return metrics, dt * 2
''')
    loop = tree.body[0]
    dead = loop.body[0].body[0].value
    live = loop.body[1].value
    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(0, expr_obj=loop.test, type='Constant', constant=True, parents=[])
    graph.G.add_node(1, type='Input', label='metrics', parents=[])
    graph.G.add_node(2, expr_obj=live.elts[1], type='Mul', parents=[])
    graph.G.add_node(4, expr_obj=loop, type='While', parents=[(0,'condition')])
    graph.roots = [2]
    key = lambda expr: (expr.lineno, expr.col_offset, expr.end_lineno, expr.end_col_offset)
    graph.G.graph['return_slot_values'] = {key(dead): (1,99), key(live): (1,2)}
    graph.G.graph['planner_specializations'] = {'rollback': True}
    description = LoopComposer(LoopBackendCapabilities(backend='c',native_while=True)).describe(graph,4)
    assert len(description.return_controls) == 1
