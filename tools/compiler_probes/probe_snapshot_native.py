import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from src.compiler.ssa_c_backend import emit_ssa_module_to_c

module = pickle.loads(Path('build/authored-snapshot-failed.pkl').read_bytes())
entry = 'snapshot_authored__root'
root = module.functions[entry]
fields = {argument.accounting['program_abi_field']: argument.id
          for argument in root.args if argument.accounting.get('program_abi_field')}
rollback = dict(root.metadata['parameter_names'])['rollback']
artifact = emit_ssa_module_to_c(module, entry)
assert artifact.complete, artifact.shortfalls
Path('build/snapshot-authored.c').write_text(artifact.source, encoding='utf-8')
work = Path(tempfile.mkdtemp(prefix='snapshot_native_'))
artifact.compile(work / 'snapshot')
payload = work / 'artifact.pkl'
payload.write_bytes(pickle.dumps((artifact, fields, rollback)))
probe = subprocess.run([sys.executable, '-c', '''
import pickle, sys
import numpy as np
with open(sys.argv[1], 'rb') as stream:
    artifact, fields, rollback = pickle.load(stream)
for enabled in (False, True):
    feeds = {fields['state']: np.array([2.0, 3.0]),
             fields['telemetry']: np.array([10.0, 20.0]),
             rollback: np.array([enabled], dtype=np.bool_)}
    result = artifact.prepare_execution(feeds).run()
    print(enabled, {name: np.asarray(result.buffers[value]).tolist() for name, value in fields.items()}, flush=True)
''', str(payload)], capture_output=True, text=True, timeout=20)
print(probe.stdout, probe.stderr, flush=True)
assert probe.returncode == 0, probe.returncode
