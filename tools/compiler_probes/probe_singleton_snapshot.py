import runpy
import numpy as np

lower = runpy.run_path('tests/test_aggregate_call_identity.py')['_lower']
source = '''
def capture(array):
    return (array.copy(),)
def recover(saved):
    return saved[0]
def tick(array):
    saved = capture(array)
    return recover(saved)
'''
module, outputs, exports = lower(source, 'singleton_snapshot', {'array': np.zeros((2, 3))})
print('OUTPUTS', outputs, flush=True)
for name, function in module.functions.items():
    if name.startswith('singleton_snapshot'):
        print(name, [(a.id, a.shape, a.dtype) for a in function.args], flush=True)
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op in ('Call', 'Ret'):
                    print(instruction, flush=True)
