import json
from pathlib import Path
import re
import subprocess
import sys

directory = Path('build/managed_dt_atomic_control_order_20260905').resolve()
source = (directory / 'balloon_tire_managed_native_c.c').read_text()
lines = ['#include <stdio.h>']
current = None
condition = 'trace_this <= 3 || !(trace_this & (trace_this - 1))'
for line in source.splitlines():
    match = re.match(r'(?:static|TURING_EXPORT) void (\w+)\(.*\) \{$', line)
    if match and (match.group(1).endswith('__balloon_tire_managed_advance') or
                  match.group(1).endswith('__run_superstep__specialized_b0c1d57d7251') or
                  match.group(1).endswith('__step_with_dt_control_used__specialized_d8d399b621bf')):
        current = match.group(1)
        lines.extend([line, '    static unsigned long long trace_count = 0;',
                      '    unsigned long long trace_this = ++trace_count;',
                      f'    if ({condition}) fprintf(stderr, "ENTER {current} #%llu\\n", trace_this);'])
        if current.endswith('__balloon_tire_managed_advance'):
            lines.append('    fprintf(stderr, "ADVANCE dt=%.17g\\n", *(double*)v0);')
        if current.endswith('__step_with_dt_control_used__specialized_d8d399b621bf'):
            lines.append('    fprintf(stderr, "STEP dt=%.17g\\n", *(double*)v9);')
        continue
    if current and line.strip() == 'return;':
        if current.endswith('__step_with_dt_control_used__specialized_d8d399b621bf'):
            lines.append('    fprintf(stderr, "STEP RETURN next=%.17g used=%.17g\\n", *(double*)out1538, *(double*)out1539);')
        lines.append(f'    if ({condition}) fprintf(stderr, "EXIT {current} #%llu\\n", trace_this);')
    lines.append(line)
    if current and current.endswith('__run_superstep__specialized_b0c1d57d7251') and line.strip() == 'L_impl_while_header: (void)0;':
        lines.append('    fprintf(stderr, "LOOP total=%.17g cap=%.17g iters=%.17g condition=%u\\n", t356, t354, t353, (unsigned)t357);')
    if line == '}':
        current = None
(directory / 'trace_module.c').write_text('\n'.join(lines))
host = (directory / 'balloon_tire_managed_native_c_host.c').read_text()
(directory / 'trace_host.c').write_text(host.replace('final-outputs.bin', 'trace-final-outputs.bin'))
exe = directory / 'trace_managed.exe'
subprocess.run([sys.executable, '-m', 'ziglang', 'cc', '-O0', '-g', '-std=c11',
                '-o', str(exe), str(directory / 'trace_module.c'), str(directory / 'trace_host.c')],
               check=True, timeout=120)
print('Running instrumented C diagnostic for at most 20 seconds', flush=True)
with (directory / 'trace-native.log').open('w') as log:
    try:
        result = subprocess.run([str(exe), '1'], stdout=log, stderr=subprocess.STDOUT, timeout=20)
        status = {'returncode': result.returncode, 'timeout': False}
    except subprocess.TimeoutExpired:
        status = {'returncode': None, 'timeout': True}
(directory / 'trace-status.json').write_text(json.dumps(status))
print(status, flush=True)
