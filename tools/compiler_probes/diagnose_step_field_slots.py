import runpy
from src.compiler import fortran_c_shell as shell

original = shell._field_slot_ops
def inspect(graph, **kwargs):
    result = original(graph, **kwargs)
    raw = getattr(graph, 'G', graph)
    if raw.graph.get('function_name') in {'restore', 'copy_shallow', 'update_dt_max'}:
        print('FIELD_SLOTS', raw.graph.get('function_name'),
              'owner=', raw.graph.get('method_owner'),
              'fields=', result[4],
              'ops=', result[1],
              'parameter_records=', tuple((raw.graph.get('parameter_record_abi') or {})),
              flush=True)
    return result
shell._field_slot_ops = inspect
runpy.run_module('tools.repro_step_with_dt_control_used', run_name='__main__')
