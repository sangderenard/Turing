import pickle
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
destination = Path('build/managed_dt_return_order_20260905')
destination.mkdir(parents=True, exist_ok=True)
sys.argv = ['tools/build_balloon_tire_native.py', '--managed-dt', '--batch-size', '8',
            '--optimization', 'O0', '--frames', '0',
            '--window-duration', str(2**-20), '--dt-initial', str(2**-20),
            '--output', str(destination)]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except Exception as error:
    trace = error.__traceback__
    while trace is not None:
        values = trace.tb_frame.f_locals
        if isinstance(values.get('all_functions'), dict):
            from src.transmogrifier.ssa import IRModule
            module = IRModule(
                values['all_functions'],
                tensor_tables=values.get('all_tensor_tables', {}),
                sequence_tables=values.get('all_sequence_tables', {}),
                record_tables=values.get('all_record_tables', {}),
                reference_tables=values.get('all_reference_tables', {}),
                call_table=values.get('call_records', {}),
                metadata={'diagnostic_incomplete': True, 'failure': str(error)},
            )
            try:
                (destination / 'failed-link-ssa.pkl').write_bytes(
                    pickle.dumps((module, {}, ()), protocol=5))
                print('Saved incomplete link state:', destination / 'failed-link-ssa.pkl', flush=True)
            except Exception as save_error:
                print('Could not save incomplete link state:', save_error, flush=True)
            break
        trace = trace.tb_next
    raise
