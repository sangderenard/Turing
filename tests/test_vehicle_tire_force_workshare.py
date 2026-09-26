from __future__ import annotations

import numpy as np
import pytest

from src.compiler.vehicle_tire_force_workshare import (
    TireForceReferenceWorkShare,
    TireForceWorkShareConfig,
    compile_tire_force_workshare_c,
    compile_tire_force_workshare_wasm,
)


SCALE = np.asarray([12000.0, 12000.0, 12000.0, 4000.0, 4000.0, 4000.0])


def test_workshare_starts_exact_then_reduces_reference_duty_after_real_low_loss_trials():
    controller = TireForceReferenceWorkShare(
        output_scale=SCALE,
        config=TireForceWorkShareConfig(
            loss_ema_rate=0.5,
            alpha_fall_rate=0.25,
            minimum_reference_alpha=0.05,
            maximum_trial_interval=16,
        ),
    )
    prediction = np.zeros(6)
    calls = 0

    def exact():
        nonlocal calls
        calls += 1
        return np.zeros(6)

    first, used, loss = controller.step(prediction, exact)
    assert used and loss == pytest.approx(0.0)
    np.testing.assert_allclose(first, 0.0)
    for _ in range(80):
        controller.step(prediction, exact)
    assert controller.state.alpha < 0.2
    assert calls < 55  # it no longer pays for the teacher every live step
    assert controller.state.steps_since_reference < 16


@pytest.mark.parametrize("novelty_name", [
    "plastic_activity", "contact_novelty", "thermodynamic_novelty",
])
def test_plastic_contact_or_thermodynamic_novelty_forces_exact_authority(novelty_name):
    controller = TireForceReferenceWorkShare(output_scale=SCALE)
    controller.state.alpha = 0.03
    controller.state.reference_trials = 10
    controller.state.trial_phase = 0.0
    predicted = np.asarray([9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    reference = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mixed, used, _loss = controller.step(
        predicted, lambda: reference, **{novelty_name: 1.0},
    )
    assert used
    np.testing.assert_allclose(mixed, reference)
    assert controller.state.last_effective_alpha == pytest.approx(1.0)


def test_one_alpha_can_work_share_the_four_wheel_inference_batch():
    controller = TireForceReferenceWorkShare(output_scale=SCALE)
    controller.state.alpha = 0.25
    predicted = np.zeros((4, 6))
    reference = np.ones((4, 6))
    np.testing.assert_allclose(controller.mix(predicted, reference), 0.25)


def test_large_trial_error_raises_exact_share_faster_than_clean_trials_lower_it():
    config = TireForceWorkShareConfig(alpha_rise_rate=0.8, alpha_fall_rate=0.05)
    controller = TireForceReferenceWorkShare(output_scale=SCALE, config=config)
    controller.state.alpha = 0.1
    controller.state.normalized_loss_ema = 0.0
    controller.observe_trial(np.ones(6) * SCALE, np.zeros(6))
    raised = controller.state.alpha
    assert raised > 0.75
    controller.observe_trial(np.zeros(6), np.zeros(6))
    assert controller.state.alpha > raised - 0.06


def test_compiled_symbolic_alpha_transition_matches_python_update(tmp_path):
    config = TireForceWorkShareConfig()
    controller = TireForceReferenceWorkShare(output_scale=SCALE, config=config)
    controller.state.alpha = 0.63
    controller.state.normalized_loss_ema = 0.02
    trial_loss = 0.004
    artifact = compile_tire_force_workshare_c().compile(tmp_path)
    inputs = {
        "previous_alpha": controller.state.alpha,
        "previous_loss_ema": controller.state.normalized_loss_ema,
        "trial_loss": trial_loss,
        "trial_performed": 1.0,
        "low_loss": config.low_normalized_loss,
        "high_loss": config.high_normalized_loss,
        "loss_ema_rate": config.loss_ema_rate,
        "alpha_rise_rate": config.alpha_rise_rate,
        "alpha_fall_rate": config.alpha_fall_rate,
        "minimum_reference_alpha": config.minimum_reference_alpha,
        "plastic_activity": 0.0,
        "contact_novelty": 0.0,
        "thermodynamic_novelty": 0.0,
    }
    compiled = dict(zip(artifact.output_names, artifact.run(inputs)))
    residual = np.sqrt(trial_loss) * SCALE
    controller.observe_trial(residual, np.zeros(6))
    assert compiled["next_loss_ema"] == pytest.approx(controller.state.normalized_loss_ema)
    assert compiled["next_alpha"] == pytest.approx(controller.state.alpha)


def test_same_workshare_equation_lowers_to_browser_wasm():
    assert compile_tire_force_workshare_wasm().complete
