from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.common.tensors.abstract_nn.demo_perforated_multifuel_engine import (
    FUEL_PROFILES, NAMED_TRANSITION_REGIMES, TRANSITION_REGIMES,
    _flatten_numeric, _load_engine_toy, _named_regime_for_episode,
    _station_shaft_load, _transition_context, engine_parameter_mapping,
    episode_epoch_batches, episode_holdout_mask, make_engine_union_data,
    load_engine_dataset, make_multifuel_engine_data,
    sample_episode_batch_indices, save_engine_dataset,
    physical_state_mapping,
)
from src.common.tensors.abstract_nn.demo_engine_dendrite_live import (
    LiveEngineControls, LiveShadowTeacher, NetworkEngineRollout,
)


@dataclass
class _Nested:
    gain: float = 2.5
    switches: tuple[bool, ...] = (True, False)
    calibration: dict[str, float] = field(default_factory=lambda: {"a": 3.0})


def test_numeric_parameter_flattening_keeps_every_nested_leaf():
    result = {}
    _flatten_numeric(_Nested(), "engine", result)
    assert result == {
        "engine.gain": 2.5, "engine.switches[0]": 1.0,
        "engine.switches[1]": 0.0, "engine.calibration[a]": 3.0,
    }


def test_large_episode_counts_never_dispatch_random_coverage_as_script():
    assert "random-coverage" not in NAMED_TRANSITION_REGIMES
    for episode in range(512):
        regime_id, regime = _named_regime_for_episode(episode)
        assert regime in NAMED_TRANSITION_REGIMES
        assert TRANSITION_REGIMES[regime_id] == regime


def test_real_multifuel_engine_exposes_complete_numeric_spec_and_fuels():
    _, get = _load_engine_toy()
    engine = get("ldt465-multifuel-deuce")
    parameters = engine_parameter_mapping(engine)
    assert len(parameters) > 140
    assert parameters["engine.displacement_l"] == 7.62
    assert any(name.startswith("engine.ecu.") for name in parameters)
    assert set(FUEL_PROFILES) == set(engine.fuel_compatibility)


def test_idle_load_adjustment_is_owned_by_general_ecu():
    _, get = _load_engine_toy()
    from ecu import EngineControlUnit
    engine = get("ldt465-multifuel-deuce")
    ecu = EngineControlUnit()
    assert ecu.idle_load_feedforward_frac(engine, 0.0) == 0.0
    assert ecu.idle_load_feedforward_frac(engine, 5_000.0) > 0.0


def test_external_test_stand_inventory_is_visible_to_training():
    EngineCycleSim, get = _load_engine_toy()
    sim = EngineCycleSim(get("agt1500-abrams-turbine"))
    values = physical_state_mapping(sim.state)
    assert values["state.test_stand_cooling.reservoir_l"] > 0.0
    assert values["state.test_stand_cooling.reservoir_temp_k"] == 293.15
    assert values["state.test_stand_cooling.pump_rated_flow_l_min"] > 0.0


def test_station_load_uses_matching_power_signal_and_shaft_reaction():
    @dataclass
    class State:
        rpm: float = 600.0

    class Sim:
        state = State()
        known_accessory_shaft_load_w = 0.0
        brake_load_nm = 0.0

        @staticmethod
        def _current_gear_ratio():
            return 2.0

    sim = Sim()
    _station_shaft_load(sim, 4_000.0)
    transfer_omega = 600.0 / 2.0 * 2.0 * np.pi / 60.0
    assert sim.known_accessory_shaft_load_w == 4_000.0
    assert np.isclose(sim.brake_load_nm * transfer_omega, 4_000.0)


def test_transition_context_records_history_without_recurrent_runtime():
    current = {"state.rpm": 650.0, "state.current_torque_nm": 80.0}
    previous = {"state.rpm": 625.0, "state.current_torque_nm": 70.0}
    controls = {"control.throttle": 0.2, "control.brake_load_nm": 10.0}
    old_controls = {"control.throttle": 0.1, "control.brake_load_nm": 8.0}
    context = _transition_context(current, previous, controls, old_controls)
    assert context["transition.history_valid"] == 1.0
    assert context["transition.delta.rpm"] == 25.0
    assert np.isclose(context["transition.control_delta.throttle"], 0.1)
    assert context["transition.previous_control.brake_load_nm"] == 8.0


def test_real_dataset_contains_independent_transition_episodes(tmp_path):
    dataset = make_multifuel_engine_data(
        samples_per_fuel=10, episodes_per_fuel=4,
        random_coverage_samples=10, seed=12)
    assert dataset.features.shape[0] == len(FUEL_PROFILES) * 10 + 10
    assert np.unique(dataset.episode_ids).size == len(FUEL_PROFILES) * 5
    assert "transition.delta.rpm" in dataset.feature_names
    assert "transition.previous_control.known_accessory_shaft_load_w" in dataset.feature_names
    assert "control.starter_signal" in dataset.feature_names
    assert "control.ignition_enabled" in dataset.feature_names
    assert dataset.idle_feedforward_rows > 0
    random_regime = TRANSITION_REGIMES.index("random-coverage")
    assert np.count_nonzero(dataset.regime_ids == random_regime) == 10
    throttle = dataset.features[:, dataset.feature_names.index("control.throttle")]
    assert np.ptp(throttle[dataset.regime_ids == random_regime]) > 0.1

    held_out = episode_holdout_mask(dataset, seed=19)
    for episode in np.unique(dataset.episode_ids):
        rows = dataset.episode_ids == episode
        assert np.all(held_out[rows] == held_out[rows][0])

    train = np.flatnonzero(~held_out)
    chosen = sample_episode_batch_indices(
        np.random.default_rng(4), train, dataset.episode_ids, batch_size=5)
    assert np.unique(dataset.episode_ids[chosen]).size == 5

    scheduled = list(episode_epoch_batches(
        np.random.default_rng(8), train, dataset.episode_ids, batch_size=7))
    valid_rows = np.concatenate([
        indices[weights.astype(bool)] for indices, weights in scheduled
    ])
    assert sorted(valid_rows.tolist()) == sorted(train.tolist())
    assert all(np.unique(dataset.episode_ids[indices[weights.astype(bool)]]).size
               == int(weights.sum()) for indices, weights in scheduled)
    restored = load_engine_dataset(save_engine_dataset(
        tmp_path / "capture.npz", dataset))
    assert restored.feature_names == dataset.feature_names
    assert restored.engine_names == dataset.engine_names
    np.testing.assert_array_equal(restored.features, dataset.features)
    np.testing.assert_array_equal(restored.targets, dataset.targets)


def test_automatic_capture_neutral_does_not_create_extreme_shaft_speed(monkeypatch):
    EngineCycleSim, _ = _load_engine_toy()
    original_step = EngineCycleSim.step
    neutral_steps = 0

    def checked_step(sim, dt):
        nonlocal neutral_steps
        if sim.gear_index == 0:
            neutral_steps += 1
            # Neutral disconnects the shafts; it cannot multiply idle speed
            # into billions of rad/s while preparing a capture episode.
            assert abs(sim._load_omega) < 1e6
        return original_step(sim, dt)

    monkeypatch.setattr(EngineCycleSim, "step", checked_step)
    # Camry's allocation/seed in the default 29-engine, 3,000-row union.
    dataset = make_multifuel_engine_data(
        samples_per_fuel=256, random_coverage_samples=104, seed=3747,
        engine_identity="toyota-3sfe-camry-1990")
    assert neutral_steps > 0
    assert np.isfinite(dataset.features).all()
    assert np.isfinite(dataset.targets).all()


def test_live_rollout_reports_execution_performance():
    rollout = object.__new__(NetworkEngineRollout)
    rollout.compiled_ns = deque((2_000_000, 4_000_000), maxlen=2048)
    rollout.transition_ns = deque((3_000_000, 7_000_000), maxlen=2048)
    rollout.deadline_misses = 1
    rollout.timed_transitions = 2
    rollout.divergence_count = 0
    runtime = rollout.performance_snapshot()
    assert runtime["compiled_mean_ms"] == 3.0
    assert runtime["compiled_transitions_per_second"] == 1000.0 / 3.0
    assert runtime["end_to_end_mean_ms"] == 5.0
    assert runtime["realtime_factor"] == 1.0
    assert runtime["deadline_miss_fraction"] == 0.5


def test_live_shadow_emits_real_transition_for_explicit_start_and_ignition_controls():
    dataset = make_multifuel_engine_data(
        samples_per_fuel=10, episodes_per_fuel=4, seed=21)
    feature_mean = dataset.features.mean(axis=0)
    feature_scale = dataset.features.std(axis=0)
    feature_scale[feature_scale < 1e-10] = 1.0
    target_mean = dataset.targets.mean(axis=0)
    target_scale = dataset.targets.std(axis=0)
    target_scale[target_scale < 1e-10] = 1.0
    preprocessing = {
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
    }
    teacher = LiveShadowTeacher(preprocessing)
    x, y = teacher.step(LiveEngineControls(
        throttle=0.35, brake_load_nm=20.0, electrical_load_frac=0.5,
        gear_index=1, clutch_frac=0.25, starter_signal=True,
        ignition_enabled=False))
    assert x.shape == (dataset.features.shape[1],)
    assert y.shape == (dataset.targets.shape[1],)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert teacher.sim.state.ignition_cut


def test_engine_union_abi_pads_distinct_profiles_and_preserves_selector():
    identities = (
        "ldt465-multifuel-deuce", "dual-motor-ev-reference",
        "agt1500-abrams-turbine")
    dataset = make_engine_union_data(
        named_samples_per_fuel=10, episodes_per_fuel=4,
        random_coverage_samples=0, seed=31,
        engine_identities=identities)
    assert dataset.engine_names == identities
    assert set(np.unique(dataset.engine_ids)) == {0, 1, 2}
    for engine_id, identity in enumerate(identities):
        column = dataset.feature_names.index(f"engine_profile.{identity}")
        rows = dataset.engine_ids == engine_id
        assert np.all(dataset.features[rows, column] == 1.0)
        assert np.all(dataset.features[~rows, column] == 0.0)
    assert "state.rpm" in dataset.target_names
