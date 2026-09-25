import runpy
from pathlib import Path
import pickle
from src.compiler import fortran_c_shell as shell
original = shell._full_native_link_failures

def inspect(*args, **kwargs):
    module = kwargs['module']
    for name, function in module.functions.items():
        if 'inner_recurrence__specialized' in name and '__planned_region' not in name:
            print('SIGNATURE', name, [(v.id, v.shape, v.accounting) for v in function.args], flush=True)
            for key in ('parameter_names', 'value_names', 'carried_port_values', 'storage_formals'):
                print(key, function.metadata.get(key), flush=True)
    return original(*args, **kwargs)

shell._full_native_link_failures = inspect
scope = runpy.run_path('tests/test_aggregate_call_identity.py')
try:
    scope['_lower'](scope['HISTORY_SOURCE'], 'history_projection', scope['HISTORY_FEEDS'])
except Exception as error:
    print(type(error).__name__, str(error))
