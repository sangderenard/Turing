"""Designated short end-to-end program for the complete provenance backend.

Reuses guarded_splice from the existing Python control regression corpus:
while/break, independent guards, mutable nested rows, starred tuple assembly,
enumerate, destructuring and repeated native calls with all guard combinations.
The existing C test verifies exact results/cardinality and concordance.
It does not claim complete payload or Python-language coverage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROGRAM = "tests/test_pythonic_control_torture.py::test_pythonic_control_native_effects[c-guarded_splice]"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('auto', 'python', 'native'), default='auto')
    parser.add_argument('--output', type=Path, default=Path('build/provenance-acceptance'))
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(TURING_UNROLL_DEPTH='0', TURING_INLINE_DEPTH='0', TURING_CUSTOM_VERSION_DEPTH='0')
    if args.backend == 'python':
        environment['TURING_PROVENANCE_SOURCE_ONLY'] = '1'
    elif args.backend == 'native':
        environment.pop('TURING_PROVENANCE_SOURCE_ONLY', None)
    worker = r'''
import json, sys
from src.compiler.compiler_bootstrap_runtime import provenance_core_state
state = provenance_core_state()
print("PROVENANCE_BACKEND " + json.dumps(state, sort_keys=True), flush=True)
if sys.argv[1] != "auto" and state["backend"] != sys.argv[1]:
    raise SystemExit("requested provenance backend is unavailable")
import pytest
raise SystemExit(pytest.main(["-q", "-s", sys.argv[2], "--tb=short"]))
'''
    started = time.perf_counter()
    # Each run is one fresh interpreter. No arbitrary timeout or Python
    # execution fallback masks failed native output verification.
    with (destination / 'program.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-c', worker, args.backend, PROGRAM],
                                cwd=repository, env=environment, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    backend = None
    with (destination / 'program.log').open(encoding='utf-8') as log:
        for line in log:
            if line.startswith('PROVENANCE_BACKEND '):
                backend = json.loads(line.removeprefix('PROVENANCE_BACKEND '))
                break
    peak = None
    if sys.platform != 'win32':
        import resource
        # Fresh worker plus its descendants; maximum process RSS, not a
        # simultaneous aggregate. Linux reports KiB, macOS reports bytes.
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if sys.platform != 'darwin':
            peak *= 1024
    receipt = {'schema': 'turing.provenance-acceptance.v1', 'program': PROGRAM,
               'backend': backend, 'status': 'passed' if result.returncode == 0 else 'failed',
               'exit_code': result.returncode, 'wall_seconds': elapsed,
               'maximum_child_process_rss_bytes': peak,
               'expansion_policy': {'unroll_depth': 0, 'inline_depth': 0, 'custom_version_depth': 0}}
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
