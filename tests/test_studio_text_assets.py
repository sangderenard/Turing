import numpy as np
import pytest
from examples.meowdoku.studio_text_assets import build_text_assets,write_text_assets


def test_curve_cookie_stays_distinct_from_explicit_mesh_bake(tmp_path):
    asset=build_text_assets('O',height_m=.02,glyph=True)
    assert any(len(curve)>2 for loop in asset['cookie']['contours'] for curve in loop)
    mesh=asset['glyph']
    assert mesh['kind']=='tessellated-glyph-mesh'
    assert np.ptp(mesh['vertices'][:,:,1])==pytest.approx(.02)
    assert mesh['vertices'].shape[0]>0
    assert mesh['normals'].shape==(len(mesh['vertices']),3)
    assert np.isfinite(mesh['normals']).all()
    paths=write_text_assets(tmp_path,text='O',height_m=.02,glyph=True)
    with np.load(paths['glyph']) as saved:
        assert np.array_equal(saved['vertices'],mesh['vertices'])
    assert asset['cookie']['source']['font']==mesh['font']
