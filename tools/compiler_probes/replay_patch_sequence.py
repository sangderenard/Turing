import os,pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.compiler import fortran_c_shell as shell
from src.compiler.project_compilation_product import _dump_resolved_process_graph
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
print('PID',os.getpid(),flush=True)
out=Path('build/patch_sequence_replay_v6'); out.mkdir(exist_ok=True)
graph=pickle.loads(Path('build/patch_sequence_diagnostic_v3/resolved-process-graph.pkl').read_bytes())
original=shell._class_surface_ssa_program
def capture(*args,**kwargs):
    from joblib.externals import cloudpickle
    saved_kwargs = {key: value for key, value in kwargs.items() if key != 'progress'}
    with (out/'pre-frame-link.pkl').open('wb') as stream:
        cloudpickle.dump((args,saved_kwargs),stream)
    print('SAVED PRE-FRAME LINK',flush=True)
    return original(*args,**kwargs)
shell._class_surface_ssa_program=capture
def progress(message):
    print(message,flush=True)
    if message == 'ssa-source: validating resolved ProcessGraph call topology':
        from joblib.externals import cloudpickle
        frame = sys._getframe(1)
        while frame is not None and 'deployment' not in frame.f_locals:
            frame = frame.f_back
        if frame is not None:
            with (out/'post-instantiation.pkl').open('wb') as stream:
                cloudpickle.dump(frame.f_locals['deployment'], stream)
            print('SAVED POST-INSTANTIATION',flush=True)
module,outputs,exports=shell._lower_resolved_process_graph_deployment(graph,'balloon_tire_managed_window',name='balloon_tire_managed_python',tensor_ssa_reference=c_backend_repository_ssa_reference(),progress=progress)
with (out/'repository-ssa.pkl').open('wb') as stream: pickle.dump((module,outputs,exports),stream)
print('LOWERING COMPLETE',len(module.functions),flush=True)
from src.compiler.ssa_self_check import run_all
for finding in run_all(module): print(finding,flush=True)
