import numpy as np
import pytest

from src.common.tensors.accelerator_backends.demo_mandelbrot_fusion import (
    animated_camera,
    capture_mandelbrot,
    capture_parametric_mandelbrot,
    capture_parametric_mandelbrot_encoder,
    dream_parameters,
    normalized_plane,
    parametric_mandelbrot_escape,
)
from src.common.tensors.accelerator_backends.demo_mandelbrot_fusion import (
    mandelbrot_escape,
)
from src.common.tensors.numpy_backend import NumPyTensorOperations as NT


def test_animated_camera_produces_visible_nonrepeating_variety():
    center = complex(-0.743643887, 0.131825904)
    span = 0.004
    samples = [animated_camera(center, span, phase) for phase in np.linspace(0, 8, 33)]
    centers = np.asarray([value[0] for value in samples])
    spans = np.asarray([value[1] for value in samples])

    assert np.ptp(centers.real) > span
    assert np.ptp(centers.imag) > 0.5 * span
    assert spans.max() / spans.min() > 8.0


@pytest.mark.parametrize(
    "capture",
    (
        lambda: capture_mandelbrot(
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            4,
        ),
        lambda: capture_parametric_mandelbrot(4),
        lambda: capture_parametric_mandelbrot_encoder(4),
    ),
)
def test_execution_tape_capture_is_disabled(capture):
    with pytest.raises(RuntimeError, match="AST -> ProcessGraph"):
        capture()


def test_zero_family_mix_is_exactly_the_mandelbrot_recurrence():
    unit_x, unit_y = normalized_plane(24, 16)
    center = complex(-0.72, 0.1)
    span = 2.4
    x = NT.tensor(unit_x)
    y = NT.tensor(unit_y)
    scalar = lambda value: NT.tensor(np.asarray([value], dtype=np.float32))
    mixed = parametric_mandelbrot_escape(
        x,
        y,
        scalar(center.real),
        scalar(center.imag),
        scalar(span),
        scalar(0.0),
        scalar(-0.4),
        scalar(0.6),
        12,
    )
    expected = mandelbrot_escape(
        scalar(center.real) + x * scalar(span),
        scalar(center.imag) + y * scalar(span),
        12,
    )
    np.testing.assert_array_equal(mixed.tolist(), expected.tolist())


def test_dream_path_preserves_a_detailed_mandelbrot_chart():
    center = complex(-0.743643887, 0.131825904)
    for travel in np.linspace(0.0, 20.0, 41):
        for bass, low_mid, high_mid in (
            (0.0, 0.0, 0.0),
            (0.2, 0.8, 0.4),
            (1.0, 1.0, 1.0),
        ):
            view_center, view_span, family_mix, julia_c = dream_parameters(
                center,
                0.004,
                float(travel),
                bass=bass,
                low_mid=low_mid,
                high_mid=high_mid,
                reaction=0.35,
                zoom_rate=0.0,
            )
            assert np.isfinite(
                (view_center.real, view_center.imag, view_span,
                 julia_c.real, julia_c.imag)
            ).all()
            assert view_span > 0.0
            assert 0.04 <= family_mix <= 0.22

            # The recurrence's blended constant plane is exactly the intended
            # Mandelbrot camera chart despite the nonzero family mix.
            recovered_center = (
                (1.0 - family_mix) * view_center
                + family_mix * julia_c
            )
            recovered_span = (1.0 - family_mix) * view_span
            expected_center, expected_span = animated_camera(
                center, 0.004, float(travel)
            )
            expected_span *= np.exp(0.35 * 0.08 * (0.5 - bass))
            assert abs(recovered_center - expected_center) < 1e-12
            assert abs(recovered_span - expected_span) < 1e-12
