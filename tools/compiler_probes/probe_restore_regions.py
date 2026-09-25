import inspect
import runpy
from src.compiler import precompile_to_ssa as lowering

original = lowering.lower_control_sections_to_ssa
def capture(control, **kwargs):
    if kwargs.get('control_name', '').endswith('__restore'):
        shell = inspect.currentframe().f_back.f_locals['shell']
        print('RESTORE CONTROL', repr(control), flush=True)
        for index, subgraph in enumerate(shell.dispatch_subgraphs):
            print('REGION', index,
                  {key: subgraph.G.graph.get(key) for key in ('deployment_nodes', 'deployment_inputs', 'deployment_outputs')}, flush=True)
    return original(control, **kwargs)
lowering.lower_control_sections_to_ssa = capture
runpy.run_module('build.probe_authored_snapshot', run_name='__main__')
