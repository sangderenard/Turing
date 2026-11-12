"""CPU-only admission of the native sensor movement controls."""
import math
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
from meowdoku.marble_photo import _camera_config, _camera_movements, _CAMERA_POSE_FIELDS, PhotoError


def test_native_movements_survive_camera_admission():
    movements=dict(tilt_shift=[.3,-.2],lens_tilt=[.1,-.15])
    result=_camera_config(movements,256,ROOT)
    for key,value in movements.items():
        assert result[key]==value
        assert key in _CAMERA_POSE_FIELDS
    assert _camera_movements({})==dict(tilt_shift=[0.,0.],lens_tilt=[0.,0.])


@pytest.mark.parametrize('key',['tilt_shift','lens_tilt'])
@pytest.mark.parametrize('value',[[True,0],[float('nan'),0],[float('inf'),0],[2,0],[0],None,'0,0'])
def test_malformed_or_unbounded_movements_are_refused(key,value):
    with pytest.raises(PhotoError):_camera_movements({key:value})
