import ast
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.compiler.control_source import SequenceBlock, StatementBlock, ConditionalBlock, WhileBlock, LoopBlock, LoopControlBlock

g, regions, reductions, loops, conditionals, nesting, result, kwargs = pickle.loads(Path('build/step-overlay.pkl').read_bytes())
print('REGIONS', regions, 'NESTING', nesting, 'DEPS', kwargs)
def walk(block, indent=''):
    if isinstance(block, SequenceBlock):
        for child in block.blocks:
            walk(child, indent)
    elif isinstance(block, StatementBlock):
        print(indent, block.lines)
    elif isinstance(block, ConditionalBlock):
        data = g.nodes.get(block.source_node_id, {})
        expr = data.get('expr_obj')
        print(indent, 'IF', block.source_node_id, block.predicate_value_id,
              ast.unparse(expr.test) if isinstance(expr,ast.If) else expr)
        walk(block.body, indent+'  ')
        if block.orelse:
            print(indent, 'ELSE'); walk(block.orelse, indent+'  ')
        print(indent, 'ENDIF')
    elif isinstance(block, (WhileBlock, LoopBlock)):
        print(indent, type(block).__name__, block.source_loop_node_id)
        walk(block.body, indent+'  ')
        print(indent, 'ENDLOOP')
    else:
        print(indent, block)
walk(result.root)
for n, data in g.nodes(data=True):
    expr = data.get('expr_obj')
    if isinstance(expr, ast.Return):
        print('RETURN SOURCE', n, getattr(expr,'lineno',None), ast.unparse(expr), data.get('parents'))
