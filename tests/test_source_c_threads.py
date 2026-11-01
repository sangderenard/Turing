"""Authored Python threads compiled into standalone C executables."""
import subprocess
import numpy as np
import pytest
from tests.test_source_ssa_synchronization import compile_source
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.identity_concordance import concordance_report

CASES = {
    'captured_storage': '''import threading
def root():
    shared = [0]
    def worker():
        shared[0] = 23
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    t.join()
    return shared[0]
''',
    'two_workers': '''import threading
def root():
    shared = [0, 0]
    def first():
        shared[0] = 23
    def second():
        shared[1] = 41
    a = threading.Thread(target=first)
    b = threading.Thread(target=second)
    a.start()
    b.start()
    a.join()
    b.join()
    return shared[0] * 100 + shared[1]
''',
    'worker_arguments': '''import threading
def root():
    shared = [0]
    def worker(value: int):
        shared[0] = value + 1
    t = threading.Thread(target=worker, args=(17,))
    t.start()
    t.join()
    return shared[0]
''',
    'create_only': '''import threading
def root():
    shared = [7]
    def worker():
        shared[0] = 99
    t = threading.Thread(target=worker)
    return shared[0]
''',
}


@pytest.mark.parametrize('case', CASES)
def test_authored_threads_run_in_a_standalone_c_shell(tmp_path, case):
    source = CASES[case]
    namespace = {}
    exec(source, namespace)
    expected = namespace['root']()
    module = compile_source(source)
    artifact = emit_ssa_module_to_c(module, 'root__root')
    assert artifact.complete, artifact.shortfalls
    assert artifact.thread_runtime_required
    executable = artifact.compile_standalone(tmp_path / case, {})
    process = subprocess.run([str(executable.executable_path), '2', '1'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    offset = 0
    got = []
    from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
    raw = executable.final_outputs_path.read_bytes()
    for dtype, shape, role in zip(artifact.buffer_dtypes, artifact.buffer_shapes, artifact.buffer_roles, strict=True):
        count = int(np.prod(shape)) if shape else 1
        data = np.frombuffer(raw, dtype=c_lane_numpy_dtype(dtype), count=count, offset=offset)
        offset += data.nbytes
        if role == 'output':
            got.extend(data.tolist())
    assert got == [expected]
    assert '0 finding(s)' in concordance_report(module)


def test_native_backend_refuses_an_escaping_captured_activation():
    source = CASES['captured_storage'].replace('    t.join()\n', '')
    artifact = emit_ssa_module_to_c(compile_source(source), 'root__root')
    assert not artifact.complete
    assert any('outlive its owner' in item.reason for item in artifact.shortfalls)


def test_worker_error_is_a_packet_and_join_does_not_raise_into_parent(tmp_path):
    source = '''import threading
def root(divisor: int):
    shared = [0]
    def worker():
        shared[0] = 12 // divisor
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return shared[4]
'''
    module = compile_source(source)
    root = module.functions['root__root']
    parameters = dict(root.metadata['parameter_names'])
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    executable = artifact.compile_standalone(tmp_path / 'worker_packet',
        {parameters['divisor']: np.array([0], dtype=np.int64)})
    process = subprocess.run([str(executable.executable_path), '1', '0'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    # The parent reaches its own failing access after join. That active error
    # is separate from the worker's reported ZeroDivisionError packet.
    assert process.returncode == 7, process.stderr
    assert 'code=71 worker=0' in process.stderr
    assert 'type=builtins.IndexError' in process.stderr
    assert not executable.final_outputs_path.exists()


def test_worker_error_reaches_the_standalone_shell_with_worker_identity(tmp_path):
    source = '''import threading
def root(divisor: int):
    shared = [0]
    def worker():
        shared[0] = 12 // divisor
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return 5
'''
    module = compile_source(source)
    parameters = dict(module.functions['root__root'].metadata['parameter_names'])
    artifact = emit_ssa_module_to_c(module, 'root__root')
    assert artifact.complete, artifact.shortfalls
    executable = artifact.compile_standalone(tmp_path / 'worker_fault',
        {parameters['divisor']: np.array([0], dtype=np.int64)})
    process = subprocess.run([str(executable.executable_path), '1', '0'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 7, process.stderr
    assert 'code=88 worker=1' in process.stderr
    assert 'function=root__worker type=builtins.ZeroDivisionError' in process.stderr
    assert not executable.final_outputs_path.exists()


@pytest.mark.parametrize('case', ['double_start', 'join_before_start'])
def test_thread_lifecycle_errors_are_reported_without_invalid_native_handle_use(tmp_path, case):
    source = CASES['captured_storage']
    if case == 'double_start':
        source = source.replace('    t.start()\n', '    t.start()\n    t.start()\n')
    else:
        source = source.replace('    t.start()\n', '    t.join()\n    t.start()\n')
    artifact = emit_ssa_module_to_c(compile_source(source), 'root__root')
    assert artifact.complete, artifact.shortfalls
    executable = artifact.compile_standalone(tmp_path / case, {})
    process = subprocess.run([str(executable.executable_path), '1', '0'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 7, process.stderr
    assert 'code=97' in process.stderr
    assert 'type=builtins.RuntimeError' in process.stderr
    assert not executable.final_outputs_path.exists()


def test_thread_factory_loop_copies_the_current_worker_argument(tmp_path):
    source = '''import threading
def root():
    shared = [0, 0, 0]
    def worker(index: int):
        shared[index] = index + 11
    i = 0
    while i < 3:
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        t.join()
        i += 1
    return shared[0] * 10000 + shared[1] * 100 + shared[2]
'''
    namespace = {}
    exec(source, namespace)
    expected = namespace['root']()
    artifact = emit_ssa_module_to_c(compile_source(source), 'root__root')
    assert artifact.complete, artifact.shortfalls
    executable = artifact.compile_standalone(tmp_path / 'factory_loop', {})
    process = subprocess.run([str(executable.executable_path), '1', '0'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
    assert np.fromfile(executable.final_outputs_path, dtype=np.int64).tolist() == [expected]
