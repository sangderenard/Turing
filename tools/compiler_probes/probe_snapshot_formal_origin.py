import runpy
import sys

seen = set()

def trace(frame, event, arg):
    if frame.f_code.co_filename.endswith('fortran_c_shell.py') and event == 'line':
        for name in ('function', 'caller', 'callee'):
            value = frame.f_locals.get(name)
            symbol = str(getattr(value, 'name', ''))
            if not symbol.endswith('__root'):
                continue
            ids = tuple(int(item.id) for item in getattr(value, 'args', ()))
            key = (symbol, ids)
            if key not in seen:
                seen.add(key)
                print('FORMALS', frame.f_code.co_name, frame.f_lineno, name, symbol, ids, flush=True)
    return trace

sys.settrace(trace)
try:
    runpy.run_module('build.probe_authored_snapshot', run_name='__main__')
finally:
    sys.settrace(None)
