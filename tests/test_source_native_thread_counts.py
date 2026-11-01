"""OS-observed thread counts in standalone source-compiled applications."""
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pytest
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from tests.test_source_ssa_synchronization import CONTRACT

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='native OS observer currently uses Linux /proc tasks')


def run_observing_threads(executable, expected, *, timeout=30, frames=1, warmup=0):
    """Observe the actual child, without changing or injecting its executable."""
    process = subprocess.Popen([str(executable.executable_path), str(frames), str(warmup)],
        cwd=executable.directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    counts = []
    deadline=time.monotonic()+timeout
    try:
        while process.poll() is None:
            if time.monotonic()>deadline:
                raise TimeoutError('standalone native thread observation timed out')
            try:
                count=len(tuple((Path('/proc')/str(process.pid)/'task').iterdir()))
                if count: counts.append(count)
            except FileNotFoundError:
                pass
            time.sleep(0.001)
        stdout,stderr=process.communicate(timeout=2)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode==0,(stdout,stderr)
    assert counts and max(counts)==expected, f'expected {expected} native tasks; observed {sorted(set(counts))}'
    import json
    (executable.directory/'thread-observation.json').write_text(json.dumps({
        'expected_total':expected,'max_observed':max(counts),
        'observed_counts':sorted(set(counts)),'sample_count':len(counts)},indent=2)+'\n')
    return counts


EXPLICIT = '''import threading
def root(iterations: int):
    shared = [0.0, 0.0]
    gate = threading.Barrier(3)
    def first():
        gate.wait()
        total = 0.0
        for i in range(iterations):
            total = total + 1.0
        shared[0] = total
    def second():
        gate.wait()
        total = 0.0
        for i in range(iterations):
            total = total + 1.0
        shared[1] = total
    a = threading.Thread(target=first)
    b = threading.Thread(target=second)
    a.start()
    b.start()
    gate.wait()
    a.join()
    b.join()
    return shared[0] + shared[1]
'''


def test_two_authored_workers_exist_as_native_threads(tmp_path):
    module,_,_=lower_ast_source_to_ssa(EXPLICIT,'root',extraction_contract=CONTRACT,
        progress=lambda _:None,progress_bars=False)
    artifact=emit_ssa_module_to_c(module,'root__root')
    assert artifact.complete,artifact.shortfalls
    assert not artifact.pool_required
    parameter=dict(module.functions['root__root'].metadata['parameter_names'])['iterations']
    executable=artifact.compile_standalone(tmp_path/'explicit', {parameter:np.array([50_000_000],dtype=np.int64)})
    # Two authored workers, plus the parent main thread waiting at the joins.
    run_observing_threads(executable,3)
    assert output_values(artifact,executable)==[100_000_000.0]


def output_values(artifact,executable,*,include_inout=False):
    from src.transmogrifier.dtype_layout import c_lane_numpy_dtype
    raw=executable.final_outputs_path.read_bytes()
    offset=0;outputs=[]
    for dtype,shape,role in zip(artifact.buffer_dtypes,artifact.buffer_shapes,artifact.buffer_roles,strict=True):
        values=np.frombuffer(raw,dtype=c_lane_numpy_dtype(dtype),count=int(np.prod(shape)) if shape else 1,offset=offset)
        offset+=values.nbytes
        if role=='output' or include_inout and role=='inout':outputs.extend(values.tolist())
    return outputs


PASSIVE = """def root(xs):
    for i in range(1000000):
        xs[i] = i * 2.0
    return xs
"""


@pytest.mark.parametrize('preference', ['asap','alap'])
@pytest.mark.parametrize('deployment,cores', [('serial',2),('auto',2),('auto',1)])
def test_deployment_discovers_two_threads_without_authored_threads(tmp_path,deployment,cores,preference):
    from dataclasses import replace
    from src.compiler.work_contract import active_contract,scoped_work_contract
    from src.compiler.identity_concordance import concordance_report
    contract=CONTRACT.with_program_abi({'records':{},'bindings':[],'values':[
        {'function':'root','parameter':'xs','storage':'span','dtype':'float64',
         'rank':1,'shape':[1000000],'python_type':'numpy.ndarray'}]})
    module,_,_=lower_ast_source_to_ssa(PASSIVE,'root',extraction_contract=contract,
        schedule_preference=preference,progress=lambda _:None,progress_bars=False)
    assert not any(i.op.startswith('Thread') for f in module.functions.values() for b in f.blocks.values() for i in b.instrs)
    with scoped_work_contract(replace(active_contract(),deployment=deployment)):
        artifact=emit_ssa_module_to_c(module,'root__root',deployment_cores=cores)
    assert artifact.complete,artifact.shortfalls
    pooled=deployment=='auto' and cores>1
    assert artifact.pool_required is pooled
    assert bool(artifact.pooled_regions) is pooled
    if deployment=='auto':
        region,=module.deployment_table['root__root']
        assert region.schedule_preference==preference
        assert region.source_loop_node_id is not None
    assert not artifact.thread_runtime_required
    xs=dict(module.functions['root__root'].metadata['parameter_names'])['xs']
    executable=artifact.compile_standalone(tmp_path/deployment,{xs:np.zeros(1000000)})
    # One parked pool worker plus the caller = two total native threads.
    # The serial control case observes exactly one, using identical source.
    run_observing_threads(executable,2 if pooled else 1)
    np.testing.assert_array_equal(output_values(artifact,executable,include_inout=True),np.arange(1000000)*2.0)
    report=concordance_report(module)
    assert '0 finding(s)' in report,report
    assert 'unsourced: 0 fact(s)' in report,report


def test_passive_discovery_does_not_parallelize_shared_reads(tmp_path):
    from dataclasses import replace
    from src.compiler.work_contract import active_contract,scoped_work_contract
    source=PASSIVE.replace('xs[i] = i * 2.0','xs[i] = xs[0] + i * 2.0')
    contract=CONTRACT.with_program_abi({'records':{},'bindings':[],'values':[
        {'function':'root','parameter':'xs','storage':'span','dtype':'float64',
         'rank':1,'shape':[1000000],'python_type':'numpy.ndarray'}]})
    module,_,_=lower_ast_source_to_ssa(source,'root',extraction_contract=contract,
        progress=lambda _:None,progress_bars=False)
    with scoped_work_contract(replace(active_contract(),deployment='auto')):
        artifact=emit_ssa_module_to_c(module,'root__root',deployment_cores=2)
    assert artifact.complete,artifact.shortfalls
    assert not artifact.pool_required
    xs=dict(module.functions['root__root'].metadata['parameter_names'])['xs']
    initial=np.zeros(1000000);initial[0]=7.0
    executable=artifact.compile_standalone(tmp_path/'shared_read',{xs:initial})
    run_observing_threads(executable,1)
    np.testing.assert_array_equal(output_values(artifact,executable,include_inout=True),7.0+np.arange(1000000)*2.0)
