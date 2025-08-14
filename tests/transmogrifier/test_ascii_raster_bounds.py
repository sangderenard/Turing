import numpy as np
from types import SimpleNamespace

from src.cells.softbody.demo.run_numpy_demo import _rasterize_ascii_numpy
from src.cells.cellsim.mechanics.softbody0d import Softbody0DProvider, SoftbodyProviderCfg
from src.cells.cellsim.data.state import Cell, Bath


def test_rasterize_ascii_1d_no_error():
    n = 4
    xs = np.linspace(0.2, 0.8, n + 1)
    X = np.stack([xs, np.full(n + 1, 0.5), np.zeros(n + 1)], axis=1)
    F = np.column_stack([np.arange(n), np.arange(1, n + 1)])
    cell = SimpleNamespace(X=X, faces=F, organelles=[])
    h = SimpleNamespace(cells=[cell])
    api = SimpleNamespace(cells=[SimpleNamespace(n={}, internal_pressure=0.0)])
    chars, rgb = _rasterize_ascii_numpy(
        h, api, 10, 5, render_mode="edges", face_stride=1, draw_points=True
    )
    assert chars.shape == (5, 10)
    assert rgb.shape == (5, 10, 3)


def test_provider_1d_bounds_match_bath():
    cells = [Cell(V=1.0, n={"Imp": 0.0})]
    bath = Bath(V=1.0, n={"Na": 0.0})
    prov = Softbody0DProvider(SoftbodyProviderCfg(dim=1))
    prov.sync(cells, bath)
    h = prov._h
    params = prov._params
    assert np.allclose(h.box_min, np.array(params.bath_min))
    assert np.allclose(h.box_max, np.array(params.bath_max))
    ys = np.unique(np.concatenate([c.X[:, 1] for c in h.cells]))
    zs = np.unique(np.concatenate([c.X[:, 2] for c in h.cells]))
    assert ys.size == 1
    assert zs.size == 1
