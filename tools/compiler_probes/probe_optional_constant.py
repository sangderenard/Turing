from pathlib import Path
from src.compiler import precompile_to_ssa as p
import traceback
SSAValue = p.SSAValue
class TracedValue(SSAValue):
    @property
    def dtype(self): return self.__dict__['dtype']
    @dtype.setter
    def dtype(self, value):
        if self.__dict__.get('dtype') != value:
            print('DTYPE CHANGE', self.id, self.__dict__.get('dtype'), value, flush=True)
            traceback.print_stack(limit=5)
        self.__dict__['dtype'] = value
original = p._materialize_control_constants
def inspect(function, constants, **kwargs):
    if function.name.endswith('__root'):
        print('CONSTANT CONTRACT', constants, kwargs, flush=True)
    result = original(function, constants, **kwargs)
    if function.name.endswith('__root'):
        print('CONSTANTS', [(i.res.id, i.res.dtype, i.attributes) for b in function.blocks.values() for i in b.instrs if i.res and i.op == 'Const'], flush=True)
        for b in function.blocks.values():
            for i in b.instrs:
                if i.res and i.op == 'Const' and i.res.dtype == 'ptr': i.res.__class__ = TracedValue
    return result
p._materialize_control_constants = inspect
exec(compile(Path('tools/compiler_probes/probe_authored_snapshot.py').read_text(), 'probe', 'exec'))
