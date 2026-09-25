import runpy
import inspect
from pathlib import Path
from src.compiler import glsl_deployment_strategy as strategy
from src.compiler import ssa_c_backend as backend
from src.compiler import precompile_to_ssa as lowering
from src.compiler import tensor_ssa_lowering as tensor_lowering
original_tensor_lower = tensor_lowering.lower_tensor_calls_to_repository_ssa
def before_tensor(module, *args, **kwargs):
    functions = getattr(module, 'functions', module)
    if isinstance(functions, dict):
        for function in functions.values():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op == 'IndexedStore':
                        print('BEFORE TENSOR', repr(instruction), flush=True)
    return original_tensor_lower(module, *args, **kwargs)
tensor_lowering.lower_tensor_calls_to_repository_ssa = before_tensor
original_lower = lowering.lower_control_sections_to_ssa
def before_lower(control, **kwargs):
    frame = inspect.currentframe().f_back
    graph = frame.f_locals.get('graph_obj')
    if graph is not None:
        for key, data in graph.nodes(data=True):
            if data.get('type') == 'IndexedStore':
                print('PRE SSA STORE', key, data.get('attributes'), data.get('parents'), flush=True)
    return original_lower(control, **kwargs)
lowering.lower_control_sections_to_ssa = before_lower

original = strategy._structural_region_program_from_subgraph
def capture(graph):
    for node_id in graph.G.graph.get('deployment_nodes', ()):
        data = graph.G.nodes[node_id]
        print('REGION NODE', node_id, data.get('op'), data.get('attributes'), flush=True)
    return original(graph)
strategy._structural_region_program_from_subgraph = capture
original_emit = backend.emit_ssa_module_to_c
def emit(module, entry, **kwargs):
    for name, function in module.functions.items():
        for block in function.blocks.values():
            for instruction in block.instrs:
                print('SSA', name, instruction.op, instruction.attributes, flush=True)
    return original_emit(module, entry, **kwargs)
backend.emit_ssa_module_to_c = emit
runpy.run_path('tests/test_native_record_span_restore.py')['test_record_span_restore_writes_full_slice'](Path('build'))
