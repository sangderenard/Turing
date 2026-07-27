import numpy as np

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.riemann.demo_driven_chladni_plate import (
    advance_modal_plate,
)


def test_modal_plate_step_is_vectorized_abstract_tensor_dynamics():
    eigenvalues = AT.tensor((-1.0, -2.0, -4.0))
    displacement = AT.zeros((3,))
    velocity = AT.zeros((3,))
    projection = AT.tensor((1.0, 0.5, 0.25))
    displacement, velocity = advance_modal_plate(
        displacement,
        velocity,
        eigenvalues,
        projection,
        time_value=np.pi / 2.0,
        dt=0.01,
        drive_frequency=1.0,
        drive_strength=2.0,
        damping=0.01,
        frequency_scale=1.0,
    )
    assert isinstance(displacement, AT)
    assert isinstance(velocity, AT)
    np.testing.assert_allclose(
        velocity.tolist(), (0.02, 0.01, 0.005), rtol=1e-6
    )
    np.testing.assert_allclose(
        displacement.tolist(), (0.0002, 0.0001, 0.00005), rtol=1e-6
    )
