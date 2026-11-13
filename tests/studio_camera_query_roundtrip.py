"""CPU serialization roundtrip against the real matching Spectral producer.
Run with STUDIO_SPECTRAL_TEST_ROOT set; no GL imports or dispatch.
"""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, os.environ['STUDIO_SPECTRAL_TEST_ROOT'])
import numpy as np
from camera_designer.studio_scene_responses import SceneResponseStore
from camera_designer.studio_response_lineage import NativeResponseLineage

lineage = NativeResponseLineage(0)
s = SceneResponseStore(scene_revision='a'*64, memory_budget_bytes=1048576,
    lineage_provider=lambda: lineage.snapshot(), bounds=[[0,0,0],[4,1,1]], dimensions=[4,1,1])
a = np.zeros((2,128), '<u4')
b = np.zeros((2,128), '<u4')
a[:,4] = [2,0xffffffff]; a[:,5] = 0xfffffffe; a[:,6] = 3
a[:,8:11] = np.array([-1,.5,.5],np.float32).view(np.uint32)
a[:,11] = np.array([3,np.nan],np.float32).view(np.uint32)
a[:,12:15] = np.array([2,0,0],np.float32).view(np.uint32)
a[:,32] = 4; a[:,40] = 7; a[:,48] = 0x80000001; a[:,80] = 0x40000002
a[:,116:119] = np.array([999,999,999],np.float32).view(np.uint32)
a[:,126] = 2
b[:,0:3] = [0,0x80000000,0x7fc12345]; b[1,96] = 7; b[1,108] = 0x15
s.consume_camera_pairs(dict(version='native-camera-intervals-v1',response_schema='native-camera-response-copies-v1',
    complete_dispatch_observation=True,application_ready=False,row_count=2,
    scene_revision='a'*64,stream_id='cpu-pairs',sequence=0,source_sha256='b'*64,transport_scene_sha256='c'*64),a,b)
m,p = s.snapshot()
fixture=dict(manifest=m,payload=base64.b64encode(p).decode(),cameraRhsWords=np.concatenate([a,b],axis=1).ravel().tolist(),queryIndex=True,expectedQueryReferences=4)
subprocess.run(['node',str(Path(__file__).with_name('meowdoku_camera_pairs_check.mjs')),'--stdin'],
    input=json.dumps(fixture),text=True,check=True)
