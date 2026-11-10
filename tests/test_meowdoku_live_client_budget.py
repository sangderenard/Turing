"""No high-resolution admission on the strength of a diagnostic/client flag."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from meowdoku.marble_live import live_settings
from meowdoku.marble_photo import PhotoError

@pytest.mark.parametrize('edge',[613,768,1024,2048,4096])
@pytest.mark.parametrize('claims',[{}, {'client_participation':True}, {'client_work_sharing':True}])
def test_excess_resolution_requires_actual_engine_capability(edge,claims):
    with pytest.raises(PhotoError,match='enabled client intersection sharing'):
        live_settings(claims,{'resolution':edge})

def test_existing_live_resolution_remains_longest_edge_pixels():
    config=live_settings({'aspect':2},{'resolution':512})
    assert (config['width'],config['height'])==(512,256)

def test_requested_boundary_is_not_silently_unlocked_as_resolution():
    with pytest.raises(PhotoError):live_settings({}, {'resolution':612})
