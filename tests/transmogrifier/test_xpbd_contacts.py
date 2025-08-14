import numpy as np

from src.cells.softbody.engine.params import EngineParams
from src.cells.softbody.engine.xpbd_core import XPBDSolver


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


def test_build_contacts_nd():
    params = EngineParams()
    solver = XPBDSolver(params)

    X2 = np.array([[0.0, 0.0], [1.5, 0.0], [0.0, -1.5]], dtype=np.float64)
    idx2, normals2, depth2 = solver.build_contacts(X2, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert idx2.shape[0] == 2
    assert normals2.shape == (2, 2)
    assert np.all(depth2 < 0)

    X1 = np.array([[0.0], [2.0], [-2.0]], dtype=np.float64)
    idx1, normals1, depth1 = solver.build_contacts(X1, np.array([-1.0]), np.array([1.0]))
    assert idx1.shape[0] == 2
    assert normals1.shape == (2, 1)
    assert np.all(depth1 < 0)
