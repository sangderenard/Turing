"""Compare a built managed DT executable with its eager authored window.

No compilation is performed. Both executions are bounded subprocesses and run
sequentially. The initial native buffer file must match the recreated fixture
before either result can count as parity evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def eager_worker(directory: Path, frames: int) -> None:
    setup_started = time.perf_counter()
    print('Eager setup started', flush=True)
    import numpy as np
    from src.common.tensors import AbstractTensor
    from src.compiler.vehicle_python_compilation import (
        Metrics, run_superstep, _abstract_tensor_stage_callable,
    )
    from src.compiler.vehicle_balloon_tire import balloon_tire_symbolic_compilations

    source, entrypoint, feeds = pickle.loads((directory / 'parity-inputs.pkl').read_bytes())
    # These are the same symbolic laws in their AbstractTensor Python stage;
    # no optional eager native-kernel insertion is consulted here.
    namespace = {'AbstractTensor': AbstractTensor, 'Metrics': Metrics,
                 'run_superstep': run_superstep}
    namespace.update({name: _abstract_tensor_stage_callable(compilation, name)
                      for name, compilation in balloon_tire_symbolic_compilations().items()})
    with AbstractTensor.use_backend('numpy'):
        material = feeds['material']
        for name, value in vars(material).items():
            if isinstance(value, np.ndarray):
                setattr(material, name, AbstractTensor.tensor(value.copy(), dtype=str(value.dtype)))
        exec(compile(source, '<managed-dt-eager-source>', 'exec'), namespace)
        print(f'Eager setup complete: {time.perf_counter() - setup_started:.6f}s', flush=True)
        stepping_started = time.perf_counter()
        for _ in range(frames):
            returned = namespace[entrypoint](**feeds)
        print(f'Eager stepping complete: {time.perf_counter() - stepping_started:.6f}s', flush=True)
        returned = tuple(value.numpy() if isinstance(value, AbstractTensor) else value
                         for value in returned)
        for name, value in vars(material).items():
            if isinstance(value, AbstractTensor):
                setattr(material, name, value.numpy())
        # Scalar controller state can also acquire a tensor-valued result.
        for owner in (feeds['controller'], feeds['targets']):
            for name, value in vars(owner).items():
                if isinstance(value, AbstractTensor):
                    setattr(owner, name, value.item())
    (directory / 'parity-eager-feeds.pkl').write_bytes(pickle.dumps((feeds, returned)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--rtol', type=float, default=1e-8)
    parser.add_argument('--atol', type=float, default=1e-10)
    parser.add_argument('--eager-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.frames < 1 or args.timeout <= 0:
        parser.error('frames and timeout must be positive')
    if args.eager_worker:
        eager_worker(directory, args.frames)
        return 0

    import numpy as np
    from src.compiler.ssa_c_backend import _numpy_dtype
    from src.compiler.vehicle_python_compilation import (
        VehiclePythonSSALowering, balloon_tire_managed_python_compilation_inputs,
        _managed_native_feeds_by_id,
    )

    manifest = json.loads((directory / 'balloon_tire_managed.manifest.json').read_text())
    module, outputs, exports = pickle.loads((directory / 'repository-ssa.pkl').read_bytes())
    root = next(name for name in module.functions if name.endswith('__balloon_tire_managed_window'))
    lowered = VehiclePythonSSALowering(module, root, outputs, exports)
    inputs = balloon_tire_managed_python_compilation_inputs(
        manifest['batch_size'], window_duration=manifest['window_duration'],
        dt_initial=manifest['dt_initial'],
    )
    initial = _managed_native_feeds_by_id(lowered, inputs.feeds)
    for entry in manifest['buffers']:
        if entry.get('role') == 'output':
            initial[entry['value_id']] = np.zeros(entry['shape'], dtype=_numpy_dtype(entry['dtype']))

    def read_buffers(path):
        data = path.read_bytes()
        offset, result = 0, {}
        for entry in manifest['buffers']:
            dtype = np.dtype(_numpy_dtype(entry['dtype']))
            count = entry['element_count']
            result[entry['value_id']] = np.frombuffer(data, dtype=dtype, count=count, offset=offset).copy()
            offset += count * dtype.itemsize
        if offset != len(data):
            raise ValueError(f'{path.name}: unexpected byte length {len(data)} != {offset}')
        return result

    before = read_buffers(directory / 'initial-state.bin')
    for value_id, value in before.items():
        np.testing.assert_array_equal(value, np.asarray(initial[value_id], dtype=value.dtype).reshape(-1),
                                      err_msg=f'initial fixture mismatch at SSA {value_id}')
    (directory / 'parity-inputs.pkl').write_bytes(pickle.dumps((inputs.source, inputs.entrypoint, inputs.feeds)))
    executable = directory / (manifest['entrypoint'] + ('.exe' if sys.platform == 'win32' else ''))
    final_path = directory / 'final-outputs.bin'
    if final_path.exists():
        final_path.unlink()  # Do not accept a previous run's output after failure.
    executions = []
    for label, command in (
        ('native', [str(executable), str(args.frames)]),
        ('eager', [sys.executable, str(Path(__file__).resolve()), str(directory),
                   '--frames', str(args.frames), '--eager-worker']),
    ):
        print(f'Running {label}, timeout={args.timeout}s', flush=True)
        try:
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired as error:
            (directory / f'parity-{label}.log').write_text(f'TIMEOUT after {args.timeout}s\n{error}', encoding='utf-8')
            executions.append({'mode': label, 'timeout': True, 'returncode': None})
            continue
        (directory / f'parity-{label}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        executions.append({'mode': label, 'timeout': False, 'returncode': result.returncode})
    if any(item['timeout'] or item['returncode'] != 0 for item in executions):
        (directory / 'managed-dt-parity.json').write_text(json.dumps({
            'passed': False, 'frames': args.frames, 'optimization': manifest['optimization'],
            'executions': executions, 'comparisons': [],
        }, indent=2), encoding='utf-8')
        print('Execution incomplete; see managed-dt-parity.json and parity logs', flush=True)
        return 1
    native = read_buffers(final_path)
    eager_feeds, returned = pickle.loads((directory / 'parity-eager-feeds.pkl').read_bytes())
    eager = _managed_native_feeds_by_id(lowered, eager_feeds)
    for entry in manifest['buffers']:
        if entry.get('role') == 'output':
            eager[entry['value_id']] = returned[entry['return_index']]
    comparisons = []
    failed_buffers = {}
    for entry in manifest['buffers']:
        value_id = entry['value_id']
        actual = native[value_id]
        expected = np.asarray(eager[value_id], dtype=actual.dtype).reshape(-1)
        matches = actual.shape == expected.shape and (
            np.array_equal(actual, expected) if actual.dtype.kind in 'biu'
            else np.allclose(actual, expected, rtol=args.rtol, atol=args.atol, equal_nan=False)
        )
        comparison = {'name': entry['name'], 'value_id': value_id, 'matches': bool(matches)}
        comparisons.append(comparison)
        if not matches:
            print(f'MISMATCH {entry["name"]} (SSA {value_id})', flush=True)
            failed_buffers[f'native_{value_id}'] = actual
            failed_buffers[f'eager_{value_id}'] = expected
            comparison['native_shape'] = list(actual.shape)
            comparison['eager_shape'] = list(expected.shape)
            if actual.shape == expected.shape:
                close = (actual == expected) if actual.dtype.kind in 'biu' else np.isclose(
                    actual, expected, rtol=args.rtol, atol=args.atol, equal_nan=False)
                comparison['first_mismatch_indices'] = np.flatnonzero(~close)[:8].tolist()
    passed = all(item['matches'] for item in comparisons)
    (directory / 'managed-dt-parity.json').write_text(json.dumps({
        'passed': passed, 'frames': args.frames, 'rtol': args.rtol, 'atol': args.atol,
        'optimization': manifest['optimization'],
        'executions': executions,
        'scope': 'managed DT and balloon tire, same initial fixture; not full vehicle validator',
        'comparisons': comparisons,
    }, indent=2), encoding='utf-8')
    if failed_buffers:
        np.savez(directory / 'managed-dt-parity-mismatches.npz', **failed_buffers)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
