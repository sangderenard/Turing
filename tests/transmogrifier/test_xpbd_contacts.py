import numpy as np

from src.transmogrifier.softbody.engine.params import EngineParams
from src.transmogrifier.softbody.engine.xpbd_core import XPBDSolver


def test_project_contacts_returns_lambda_and_resolves_penetration():
    params = EngineParams()
    params.contact_compliance = 0.0
    solver = XPBDSolver(params)

    X = np.array([[0.0, -0.1, 0.0]], dtype=np.float64)
    invm = np.array([1.0], dtype=np.float64)
    faces = np.empty((0, 3), dtype=np.int32)

    def vol(*args, **kwargs):
        return 0.0

    def vol_grads(*args, **kwargs):
        return np.empty((0, 3))

    contacts = {
        "indices": np.array([0], dtype=np.int32),
        "normals": np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
        "depth": np.array([-0.1], dtype=np.float64),
        "lamb": np.array([0.0], dtype=np.float64),
    }

    lamb = solver.project(
        {},
        X,
        invm,
        faces,
        vol,
        vol_grads,
        1.0,
        1,
        np.array([-1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        contacts=contacts,
    )

    assert np.allclose(X[0], [0.0, 0.0, 0.0])
    assert np.allclose(lamb, [0.1])
