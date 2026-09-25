import pickle
import numpy as np
from pathlib import Path
artifact,fields,result_id,source,rollback_id=pickle.loads(Path(r'C:\Users\alber\AppData\Local\Temp\pytest-of-alber\pytest-3293\test_native_snapshot_restores_0\artifact.pkl').read_bytes())
for flag in (False,True):
    r=artifact.prepare_execution({fields['state']:np.array([2.,3.]),fields['telemetry']:np.array([10.,20.]),rollback_id:np.array([flag],dtype=np.bool_)}).run()
    print(flag,r.buffers[result_id],r.buffers[fields['state']],r.buffers[fields['telemetry']])
