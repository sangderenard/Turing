import pytest

from src.cells.bath.make_sph import make_sph
from src.cells.bath.make_mac import make_mac
from src.cells.bath.make_hybrid import make_hybrid
@pytest.mark.parametrize(
    "factory, kwargs",
    [
        (make_sph, {"resolution": 1}),
        (make_mac, {"resolution": 1}),
        (make_hybrid, {"resolution": 1, "n_particles": 1}),
    ],
)
def test_step_with_controller_available(factory, kwargs):
    fluid = factory(**kwargs)
    assert callable(getattr(fluid, "step_with_controller", None))
