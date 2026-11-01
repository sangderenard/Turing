"""Real source synchronization -> SSA -> standalone compiled C execution."""
import subprocess
import sys
import numpy as np
import pytest
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.identity_concordance import concordance_report
from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
from tests.test_source_ssa_synchronization import CONTRACT

PREFIX = '''from src.compiler.synchronization import (
 Scope, MemorySemantics, latch_create, latch_arrive, latch_wait, latch_signal,
 latch_reset, control_barrier, control_memory_barrier, completion_wait)
'''
CASES = {
 'retained_epochs': PREFIX + '''def root():
    gate = latch_create(2, 0, Scope.OBJECT)
    latch_arrive(gate, 0, 1)
    latch_arrive(gate, 0, 1)
    latch_wait(gate, 0)
    latch_reset(gate, 0, 7, 1)
    latch_wait(gate, 0)
    latch_arrive(gate, 7, 1)
    latch_wait(gate, 7)
    latch_reset(gate, 7, 9, 0)
    latch_wait(gate, 0)
    latch_wait(gate, 7)
    latch_wait(gate, 9)
    return 17
''',
 'two_workers': PREFIX + '''import threading
def root():
    shared = [0, 0]
    gate = latch_create(2, 0, Scope.OBJECT)
    def first():
        shared[0] = 23
        control_memory_barrier(gate, Scope.OBJECT, Scope.OBJECT, MemorySemantics.ACQUIRE_RELEASE, 0, 2, shared)
        shared[0] = shared[1] + 10
    def second():
        shared[1] = 41
        control_memory_barrier(gate, Scope.OBJECT, Scope.OBJECT, MemorySemantics.ACQUIRE_RELEASE, 0, 2, shared)
    a = threading.Thread(target=first)
    b = threading.Thread(target=second)
    a.start()
    b.start()
    completion_wait(a, Scope.OBJECT, MemorySemantics.ACQUIRE, shared)
    completion_wait(b, Scope.OBJECT, MemorySemantics.ACQUIRE, shared)
    return shared[0] * 100 + shared[1]
''',
 'invalid_reset': PREFIX + '''def root():
    gate = latch_create(1, 0, Scope.OBJECT)
    try:
        latch_reset(gate, 0, 1, 2)
    except RuntimeError:
        return 17
    return 9
''',
}
EXPECTED = {'retained_epochs':17,'two_workers':5141,'invalid_reset':17}

CASES['python_barrier'] = '''import threading
def root():
    gate = threading.Barrier(2)
    shared = [0, 0, 0]
    def worker():
        shared[0] = 23
        shared[2] = gate.wait()
        shared[0] = shared[1] + 10
    task = threading.Thread(target=worker)
    task.start()
    shared[1] = 41
    index = gate.wait()
    task.join()
    return shared[0] * 100 + shared[1] + index + shared[2]
'''
EXPECTED['python_barrier'] = 5142
CASES['cyclic_python_barrier'] = '''import threading
def root():
    gate = threading.Barrier(parties=2)
    shared = [0, 0, 0]
    def worker():
        for i in range(8):
            shared[0] = i + 1
            gate.wait()
            shared[2] = shared[0] + shared[1]
            gate.wait()
    task = threading.Thread(target=worker)
    task.start()
    for i in range(8):
        shared[1] = (i + 1) * 10
        gate.wait()
        gate.wait()
    task.join()
    return shared[2]
'''
EXPECTED['cyclic_python_barrier'] = 88
CASES['broken_reset_abort'] = '''import threading
def root():
    gate = threading.Barrier(2, timeout=0.001)
    try:
        gate.wait()
    except threading.BrokenBarrierError:
        gate.reset()
        gate.abort()
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            return 17
    return 9
'''
EXPECTED['broken_reset_abort'] = 17


CASES['old_waiter_after_reset'] = PREFIX + """import threading
def root():
    shared = [0]
    gate = latch_create(2, 0, Scope.OBJECT)
    def worker():
        latch_wait(gate, 0)
        shared[0] = 17
    task = threading.Thread(target=worker)
    task.start()
    latch_arrive(gate, 0, 2)
    latch_reset(gate, 0, 1, 0)
    task.join()
    return shared[0]
"""
EXPECTED['old_waiter_after_reset'] = 17
CASES['control_only_single_participant'] = PREFIX + """def root():
    gate = latch_create(1, 0, Scope.OBJECT)
    control_barrier(gate, Scope.OBJECT, 0, 1)
    latch_reset(gate, 0, 1, 1)
    control_barrier(gate, Scope.OBJECT, 1, 1)
    return 17
"""
EXPECTED['control_only_single_participant'] = 17

CASES['explicit_none_barrier'] = """import threading
def root():
    gate = threading.Barrier(1, action=None, timeout=None)
    return gate.wait()
"""
EXPECTED['explicit_none_barrier'] = 0
CASES['invalid_dynamic_arrival'] = PREFIX + """def root(count: int):
    gate = latch_create(1, 0, Scope.OBJECT)
    try:
        latch_arrive(gate, 0, count)
    except RuntimeError:
        return 17
    return 9
"""
EXPECTED['invalid_dynamic_arrival'] = 17
INPUTS = {'invalid_dynamic_arrival': {'count': -1}}

CASES['try_barrier_result'] = """import threading
from threading import BrokenBarrierError as Broken
def root():
    try:
        gate = threading.Barrier(1)
        index = gate.wait()
        return index + 17
    except Broken:
        return 9
"""
EXPECTED['try_barrier_result'] = 17

@pytest.mark.parametrize('preference',['asap','alap'])
@pytest.mark.parametrize('case', CASES)
def test_synchronization_runs_as_standalone_c(tmp_path,case,preference):
    module, _, _ = lower_ast_source_to_ssa(CASES[case], 'root', extraction_contract=CONTRACT,
        schedule_preference=preference, progress=lambda _: None, progress_bars=False)
    if case in {'python_barrier', 'cyclic_python_barrier', 'broken_reset_abort', 'explicit_none_barrier', 'try_barrier_result'}:
        native = [instruction for function in module.functions.values() for block in function.blocks.values()
                  for instruction in block.instrs if instruction.op in {'BarrierCreate', 'BarrierWait'}]
        assert {instruction.op for instruction in native} == {'BarrierCreate', 'BarrierWait'}
        assert all(instruction.attributes.get('synchronization_receipt') for instruction in native)
    artifact = emit_ssa_module_to_c(module,'root__root')
    assert artifact.complete, artifact.shortfalls
    parameters = dict(module.functions['root__root'].metadata['parameter_names'])
    inputs = {parameters[name]: np.array([value], dtype=np.int64) for name, value in INPUTS.get(case, {}).items()}
    executable=artifact.compile_standalone(tmp_path/case,inputs)
    process=subprocess.run([str(executable.executable_path),'3','1'],cwd=executable.directory,
        capture_output=True,text=True,timeout=15)
    assert process.returncode==0,process.stderr
    raw,offset,outputs=executable.final_outputs_path.read_bytes(),0,[]
    for dtype,shape,role in zip(artifact.buffer_dtypes,artifact.buffer_shapes,artifact.buffer_roles,strict=True):
        count=int(np.prod(shape)) if shape else 1
        values=np.frombuffer(raw,dtype=c_lane_numpy_dtype(dtype),count=count,offset=offset)
        offset+=values.nbytes
        if role=='output':outputs.extend(values.tolist())
    expected = EXPECTED[case]
    if not CASES[case].startswith(PREFIX):
        reference = subprocess.run([sys.executable, '-c', CASES[case] + '\nprint(root())'],
                                   capture_output=True, text=True, timeout=15)
        assert reference.returncode == 0, reference.stderr
        assert int(reference.stdout.strip()) == expected
    assert outputs==[expected]
    assert '0 finding(s)' in concordance_report(module)



def test_completion_wait_preserves_worker_error_bus(tmp_path):
    source = PREFIX + """import threading
def root(divisor: int):
    shared = [0]
    def worker():
        shared[0] = 12 // divisor
    task = threading.Thread(target=worker)
    task.start()
    completion_wait(task, Scope.OBJECT, MemorySemantics.ACQUIRE, shared)
    return 5
"""
    module, _, _ = lower_ast_source_to_ssa(source, 'root', extraction_contract=CONTRACT,
                                          progress=lambda _: None, progress_bars=False)
    artifact = emit_ssa_module_to_c(module, 'root__root')
    assert artifact.complete, artifact.shortfalls
    parameter = dict(module.functions['root__root'].metadata['parameter_names'])['divisor']
    executable = artifact.compile_standalone(tmp_path / 'worker_error', {parameter: np.array([0], dtype=np.int64)})
    process = subprocess.run([str(executable.executable_path), '1', '0'], cwd=executable.directory,
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 7, process.stderr
    assert 'worker=1' in process.stderr and 'type=builtins.ZeroDivisionError' in process.stderr
    assert not executable.final_outputs_path.exists()
    assert '0 finding(s)' in concordance_report(module)

@pytest.mark.parametrize('scope,reason', [('Scope.INVOCATION', 'exactly one participant'),
                                         ('Scope.WORKGROUP', 'does not support scope')])
def test_native_barrier_scope_cannot_be_silently_widened(scope, reason):
    source = PREFIX + f'''def root():
    gate = latch_create(2, 0, {scope})
    control_barrier(gate, {scope}, 0, 2)
    return 1
'''
    module, _, _ = lower_ast_source_to_ssa(source, 'root', extraction_contract=CONTRACT,
                                          progress=lambda _: None, progress_bars=False)
    artifact = emit_ssa_module_to_c(module, 'root__root')
    assert not artifact.complete
    assert any(reason in shortfall.reason for shortfall in artifact.shortfalls)
