from src.compiler import fortran_c_shell as shell
lower_ast_source_to_ssa = shell.lower_ast_source_to_ssa
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.vehicle_python_compilation import (
    BalloonTireManagedState, balloon_tire_managed_extraction_contract,
)
from src.common.dt_system.dt_scaler import Metrics, coerce_metrics
import numpy as np

material = BalloonTireManagedState.__new__(BalloonTireManagedState)
for name in ('inputs', 'state', 'output', 'wheel_input_indices', 'rest',
             'face_vertices', 'face_rest', 'face_scatter', 'bending_incidence',
             'bending_scatter', 'bending_weight', 'vertex_area', 'bead_mask',
             'face_material', 'telemetry'):
    setattr(material, name, np.zeros((1,), dtype=np.float64))
policy = balloon_tire_managed_extraction_contract(material)
abi = policy.program_abi.receipt()
abi['bindings'] = [*abi['bindings'], {
    'function': 'root', 'parameter': 'state',
    'record': 'BalloonTireManagedState',
}]
policy = policy.with_program_abi(abi)
original_field_slots = shell._field_slot_ops
def inspect_field_slots(graph, **kwargs):
    result = original_field_slots(graph, **kwargs)
    raw = getattr(graph, 'G', graph)
    if raw.graph.get('function_name') in {'restore', 'copy_shallow'}:
        print('FIELD_SLOTS', raw.graph.get('function_name'),
              raw.graph.get('method_owner'), result[:6],
              raw.graph.get('class_table'))
    return result
shell._field_slot_ops = inspect_field_slots
original_gate = shell._full_native_link_failures
def inspect_gate(*args, **kwargs):
    failures = original_gate(*args, **kwargs)
    module = kwargs.get('module')
    if module is not None:
        for name, function in module.functions.items():
            if name.endswith(('__restore', '__copy_shallow')):
                print('FORMALS', name,
                      [(v.id, v.dtype, v.shape, v.accounting) for v in function.args],
                      function.metadata.get('parameter_names'),
                      function.metadata.get('authored_parameters'))
    print('GATE', failures)
    return ()
shell._full_native_link_failures = inspect_gate
source = '''
def root(state):
    saved = state.copy_shallow()
    state.restore(saved)
    return state.state
'''
module, outputs, exports = lower_ast_source_to_ssa(
    source, 'root', name='retained_metrics',
    python_bindings={'coerce_metrics': coerce_metrics, 'Metrics': Metrics},
    extraction_contract=policy,
    retain=(BalloonTireManagedState,), runtime_closure_only=True,
)
print('LOWERED', tuple(module.functions))
