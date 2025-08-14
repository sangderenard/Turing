import pytest

from src.cells.softbody.demo.run_numpy_demo import make_cellsim_backend, step_cellsim


def test_step_cellsim_exposes_bath_state():
    api, provider = make_cellsim_backend(
        cell_vols=[10.0],
        cell_imps=[100.0],
        cell_elastic_k=[0.1],
        bath_na=10.0,
        bath_cl=10.0,
        bath_pressure=10.0,
        bath_volume_factor=5.0,
        substeps=1,
        dt_provider=1e-10,
    )
    dt = 1e-10
    dt = step_cellsim(api, dt)
    state = getattr(api, "last_bath_state", None)
    assert state is not None
    for key in ("pressure", "temperature", "viscosity"):
        assert key in state
        assert state[key] == pytest.approx(getattr(api.bath, key))
