"""Real ingested control plus native compute sections, linked AOT and run out of process."""
import subprocess
import sys
import numpy as np
import pytest
from src.compiler.c_shell_sections import CShellControl
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.compiler.ssa_fortran_backend import emit_module, fortran_compiler
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report
from src.compiler.concordance_declarations import C_SHELL_SECTION
from tests.test_source_ssa_synchronization import CONTRACT

SOURCE = '''import threading
def compute(x: float) -> float:
    return x * 2.0 + 1.0
def root(value: float, divisor: int):
    gate = threading.Barrier(2)
    shared = [0.0]
    def worker():
        for i in range(1000):
            base: float = compute(value)
            try:
                result = 10 // divisor
                shared[0] = base + result
            except ZeroDivisionError:
                shared[0] = base
            finally:
                gate.wait()
    task = threading.Thread(target=worker)
    task.start()
    for i in range(1000):
        gate.wait()
    task.join()
    return shared[0]
'''


def ingest(source=SOURCE, preference='asap'):
    return lower_ast_source_to_ssa(source, 'root', extraction_contract=CONTRACT,
        schedule_preference=preference, progress=lambda _:None, progress_bars=False)[0]


def numerical_sections(module):
    # Select the actual ingested numerical region by its source owner and body,
    # not a hand-built approximation of the computation.
    return tuple(name for name, fn in module.functions.items()
                 if '__compute' in name and any(i.op == 'Mul' for b in fn.blocks.values() for i in b.instrs))


@pytest.mark.parametrize('backend', ['llvm', 'fortran'])
@pytest.mark.parametrize('preference', ['asap', 'alap'])
def test_aot_c_control_with_native_compute_sections(tmp_path, backend, preference):
    if backend == 'fortran' and fortran_compiler() is None:
        pytest.skip('AOT Fortran compiler unavailable')
    module = ingest(preference=preference)
    names = numerical_sections(module)
    assert names
    approval = CShellControl(names, tmp_path / 'sections')
    before = {name: tuple(fn.blocks) for name, fn in module.functions.items()}
    if backend == 'llvm':
        artifact = emit_ssa_function_to_llvm(module, 'root__root', c_shell_control=approval)
        assert len(artifact.linked_llvm) == len(names)
    else:
        artifact = emit_module(module, name='mixed_fortran', entrypoint='root__root', c_shell_control=approval)
        assert artifact.linked_objects
    assert artifact.complete, artifact.shortfalls
    assert artifact.thread_runtime_required
    assert before == {name: tuple(fn.blocks) for name, fn in module.functions.items()}
    rows = module.metadata['identity_book'].page(C_SHELL_SECTION).rows()
    assert {(r[1], r[2]) for r in rows} == {(name, backend) for name in names}
    assert all('__c_section' in str(s) for s in artifact.linked_llvm) if backend == 'llvm' else True
    parameters = dict(module.functions['root__root'].metadata['parameter_names'])
    for divisor, expected in [(0,41.0), (2,46.0)]:
        exe = artifact.compile_standalone(tmp_path / f'run{divisor}', {
            parameters['value']:np.array([20.0]), parameters['divisor']:np.array([divisor],dtype=np.int64)})
        if sys.platform == 'linux':
            from tests.test_source_native_thread_counts import run_observing_threads
            run_observing_threads(exe,2,frames=2,warmup=1)
        else:
            run = subprocess.run([str(exe.executable_path),'2','1'],cwd=exe.directory,
                                 capture_output=True,text=True,timeout=20)
            assert run.returncode == 0, run.stderr
        outputs=[]
        offset=0
        from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
        raw=exe.final_outputs_path.read_bytes()
        for dtype,shape,role in zip(artifact.buffer_dtypes,artifact.buffer_shapes,artifact.buffer_roles,strict=True):
            count=int(np.prod(shape)) if shape else 1
            values=np.frombuffer(raw,dtype=c_lane_numpy_dtype(dtype),count=count,offset=offset)
            offset+=values.nbytes
            if role=='output':outputs.extend(values.tolist())
        assert outputs == [expected]
    assert '0 finding(s)' in concordance_report(module)
    assert 'unsourced: 0 fact(s)' in concordance_report(module)


def test_c_control_requires_explicit_sections_and_keeps_effects_out():
    module=ingest()
    with pytest.raises(TypeError,match='CShellControl'):
        emit_ssa_function_to_llvm(module,'root__root',c_shell_control=True)
    with pytest.raises(ValueError,match='non-entry'):
        emit_ssa_function_to_llvm(module,'root__root',c_shell_control=CShellControl(('root__root',)))
    worker=next(n for n in module.functions if n.endswith('__worker'))
    with pytest.raises(ValueError,match='compute section|aggregate/alias'):
        emit_ssa_function_to_llvm(module,'root__root',c_shell_control=CShellControl((worker,)))
