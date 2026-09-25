import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.compiler import vehicle_python_compilation as vehicle
from src.compiler.ssa_c_backend import emit_ssa_to_c

source_directory = Path('build/managed_dt_forwarded_marker_20260905')
directory = Path(sys.argv[1]) if len(sys.argv) > 1 else source_directory
directory.mkdir(parents=True, exist_ok=True)
snapshot = (source_directory / 'repository-ssa.pkl').read_bytes()
if directory != source_directory:
    (directory / 'repository-ssa.pkl').write_bytes(snapshot)
module, outputs, exports = pickle.loads(snapshot)
root = next(name for name in module.functions if name.endswith('__balloon_tire_managed_window'))
lowered = vehicle.VehiclePythonSSALowering(module, root, outputs, exports)
print('Loading unchanged complete source-lowered SSA', flush=True)
artifact_path = source_directory / 'native-artifact.pkl'
artifact = (pickle.loads(artifact_path.read_bytes()) if artifact_path.exists() and '--reemit' not in sys.argv else
            emit_ssa_to_c(module, root, entry_name='balloon_tire_managed_native_c'))
assert artifact.complete, artifact.shortfalls
(directory / 'native-artifact.pkl').write_bytes(pickle.dumps(artifact))
vehicle.emit_balloon_tire_managed_python_c = lambda **kwargs: (lowered, artifact)
print('Compiling native program through corrected wrapper', flush=True)
options = ({'window_duration': 2.0**-20, 'dt_initial': 2.0**-20}
           if directory != source_directory else {})
executable = vehicle.compile_balloon_tire_managed_python_native(directory, batch_size=8, optimization='O0', **options)
print(executable, flush=True)
