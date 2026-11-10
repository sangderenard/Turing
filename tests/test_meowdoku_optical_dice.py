import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples'))
from meowdoku.marble_region_media import saved_region_media
from meowdoku.optical_topology import centered_insphere_radius

def test_actual_die_region_admission_and_conservative_containment(tmp_path):
    target=tmp_path/'die.json'
    subprocess.run(['node',str(ROOT/'tests/meowdoku_optical_dice_check.mjs'),str(target)],cwd=ROOT,check=True,capture_output=True)
    value=json.loads(target.read_text());saved=value['saved'];faces=value['faces']
    vertices=np.asarray([f[:3] for f in faces]);ids=np.asarray([f[3] for f in faces]);pbr=np.asarray(saved['materials']['tables']['pbr']['data']).reshape(-1,16)
    regions=saved_region_media(saved,vertices,ids,pbr)['regions'];host=regions[0]
    shell=vertices[host['start']:host['start']+host['count']]
    radius=centered_insphere_radius(shell)
    assert 0<radius<1
    assert np.linalg.norm(vertices[:host['start']],axis=2).max()<radius
    escaped=vertices.copy();escaped[:host['start']]*=2
    with pytest.raises(ValueError,match='insphere'):saved_region_media(saved,escaped,ids,pbr)
    with pytest.raises(ValueError,match='origin'):centered_insphere_radius(shell+np.array([4,0,0]))
