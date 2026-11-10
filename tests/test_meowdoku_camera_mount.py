"""Real native camera geometry supplies the isolated support contract."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from examples.meowdoku.marble_camera_mount import camera_mount_source

ROOT=Path(__file__).resolve().parents[1]
SPECTRAL=ROOT.parent/'spectral-analyzer'

def test_native_camera_housing_mount(tmp_path):
    if not (SPECTRAL/'camera_designer/camera_frame.py').exists():
        pytest.skip('Canonical native camera frame unavailable')
    camera={'eye_m':[.3,.4,.1],'target_m':[0,0,.18], 'aperture_radius_m':.005,
            'aperture_blades':8,'aperture_rotation_radians':.2}
    before=json.dumps(camera)
    record=camera_mount_source(camera,SPECTRAL)
    assert json.dumps(camera)==before
    assert record['schema']=='native-camera-housing-mount.v1'
    assert record['triangle_count']>1000
    from camera_designer.camera_preset import thin_lens_housing_preset
    from camera_designer.camera_frame import thin_lens_housing_frame
    from camera_designer.scene_builder import build_gpu_scene
    preset=thin_lens_housing_preset(camera)
    expected=build_gpu_scene(preset,[],scene_object=None,wavelengths_um=[.450,.550,.650],
                            camera_frame=thin_lens_housing_frame(preset,camera['eye_m'],camera['target_m']))
    vertices=np.asarray(record['housingVerticesM'],dtype='<f4')
    assert np.array_equal(vertices,np.asarray(expected[3]).reshape(-1,3))
    assert hashlib.sha256(vertices.tobytes()).hexdigest()==record['geometry_sha256']
    file=tmp_path/'native_camera_housing.json';file.write_text(json.dumps(record),encoding='utf-8')
    result=subprocess.run(['node','tests/meowdoku_marble_camera_mount_check.mjs','--native-source',str(file)],
                          cwd=ROOT,text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'exact native housing/pose source' in result.stdout

def test_camera_mount_export_requires_explicit_pose():
    for camera in ({},{'eye_m':[0,0,0],'target_m':[0,0,0]},
                   {'eye_m':[0,float('nan'),0],'target_m':[0,0,1]}):
        with pytest.raises(ValueError):camera_mount_source(camera,SPECTRAL)
