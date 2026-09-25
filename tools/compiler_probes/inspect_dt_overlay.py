import ast
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
g, regions, reductions, loops, conditionals, nesting, result = pickle.loads(
    Path('build/dt-overlay.pkl').read_bytes())
print('REGIONS', regions, 'NESTING', nesting)
for label, programs in [('LOOP', loops), ('IF', conditionals)]:
    for p in programs:
        print(label, p)
for n, data in g.nodes(data=True):
    expression = data.get('expr_obj')
    if isinstance(expression, ast.AST):
        source = ast.unparse(expression)
        if ('minimum' in source and isinstance(expression, ast.Call)) or isinstance(expression, ast.While):
            print('NODE', n, 'SOURCE', source[:250], 'LINE', getattr(expression,'lineno',None))
            print('DATA', {k:v for k,v in data.items() if k != 'expr_obj'})
print('RESULT', result)
