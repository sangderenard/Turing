"""Distil the real ``engine_toy`` multifuel engine into AbstractTensor models.

The teacher is ``engine_toy.engine_cycle_sim.EngineCycleSim``. Inputs contain
the complete numeric engine specification, every physical state scalar, all
operator/load controls, fuel identity, and the ECU-owned idle-load feedforward
command. Targets are the complete physical state delta. Large component damage
ledgers are represented by health statistics instead of thousands of graph
bookkeeping cells.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, fields, is_dataclass
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from ..abstraction import AbstractTensor as AT
from ..autograd import GradTape, autograd
from .core import Linear
from .demo_perforated_regression import _parameter_count, _train_phase
from .perforated import PerforatedLinear


ENGINE_IDENTITY = "ldt465-multifuel-deuce"
FUEL_PROFILES = (
    "ultra-low-sulfur-diesel", "kerosene", "crude-oil",
    "vegetable-oil", "pump-gasoline-93",
)
LOAD_LEVELS = (0.0, 0.10, 0.28, 0.52, 0.78)
LOAD_BANDS = ("idle", "light", "medium", "heavy", "overload")
TRANSITION_REGIMES = (
    "starting", "idling", "idle-recovery", "idle-load-compensation",
    "load-recovery", "high-end", "upshift", "downshift",
    "random-coverage",
)
NAMED_TRANSITION_REGIMES = TRANSITION_REGIMES[:-1]
MACHINE_OUTPUT_NAMES = (
    "state.current_torque_nm", "state.torque_rms_nm", "state.power_kw",
    "state.power_rms_kw", "state.accessory_drag_nm",
    "state.supercharger_drag_nm", "state.compression_brake_torque_nm",
    "state.dyno_torque_nm", "state.dyno_absorbed_kw",
)
_LEDGER_FIELDS = frozenset({"part_damage", "wear", "absent_parts"})
_TRANSITION_STATE_NAMES = (
    "state.rpm", "state.current_torque_nm", "state.torque_rms_nm",
    "state.power_rms_kw", "state.manifold_pressure_frac",
    "state.turbo_spool_frac", "state.boost_frac", "state.real_fire_hz",
    "state.accessory_drag_nm", "state.ac_compressor_load_w",
    "state.battery_voltage", "state.alternator_output_frac",
    "state.coolant_temp_k", "state.oil_temp_k", "state.exhaust_temp_k",
    "state.dyno_rpm", "state.dyno_absorbed_kw", "state.dyno_torque_nm",
    "state.coupling_slipping", "state.stalled",
)
_CONTROL_NAMES = (
    "control.throttle", "control.brake_load_nm",
    "control.electrical_load_frac",
    "control.known_accessory_shaft_load_w",
    "control.idle_load_feedforward_frac", "control.clutch_frac",
    "control.gear_index", "control.starter_signal",
    "control.ignition_enabled",
)


@dataclass(frozen=True)
class EngineDataset:
    features: np.ndarray
    targets: np.ndarray
    load_fractions: np.ndarray
    fuel_ids: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    idle_feedforward_rows: int
    episode_ids: np.ndarray
    transition_indices: np.ndarray
    regime_ids: np.ndarray
    engine_ids: np.ndarray
    engine_names: tuple[str, ...]


@dataclass(frozen=True)
class EngineRun:
    seed: int
    ordinary_mse: float
    perforated_mse: float
    ordinary_by_load: tuple[float, ...]
    perforated_by_load: tuple[float, ...]


@dataclass(frozen=True)
class MultiFuelResult:
    runs: tuple[EngineRun, ...]
    load_bands: tuple[str, ...]
    ordinary_parameters: int
    perforated_parameters: int
    train_samples: int
    test_samples: int
    input_parameters: int
    output_parameters: int
    idle_feedforward_rows: int
    backend: str

    @property
    def ordinary_mean_mse(self) -> float:
        return float(np.mean([run.ordinary_mse for run in self.runs]))

    @property
    def perforated_mean_mse(self) -> float:
        return float(np.mean([run.perforated_mse for run in self.runs]))

    @property
    def improvement_ratio(self) -> float:
        return self.ordinary_mean_mse / max(self.perforated_mean_mse, 1e-30)


def _engine_toy_path(path: str | Path | None = None) -> Path:
    candidate = (Path(path) if path else Path(__file__).resolve().parents[5] / "engine_toy").resolve()
    if not (candidate / "engine_cycle_sim.py").is_file():
        raise FileNotFoundError("engine_toy not found; pass --engine-toy-path")
    return candidate


def _load_engine_toy(path: str | Path | None = None):
    root = str(_engine_toy_path(path))
    if root not in sys.path:
        sys.path.insert(0, root)
    from engine_test_stand import EngineTestStand
    from engines import get
    return EngineTestStand, get


def _load_engine_roster(path: str | Path | None = None):
    EngineCycleSim, get = _load_engine_toy(path)
    from engines import CATALOGUE
    identities = tuple(engine.identity for engine in CATALOGUE)
    return EngineCycleSim, get, identities


def _flatten_numeric(value: Any, prefix: str, output: dict[str, float], *,
                     aggregate_ledgers: bool = False) -> None:
    """Flatten every finite numeric leaf using stable semantic paths."""
    if isinstance(value, (bool, int, float, np.number)):
        number = float(value)
        if math.isfinite(number):
            output[prefix] = number
        return
    if is_dataclass(value):
        for descriptor in fields(value):
            child = getattr(value, descriptor.name)
            path = f"{prefix}.{descriptor.name}" if prefix else descriptor.name
            if aggregate_ledgers and descriptor.name in _LEDGER_FIELDS:
                _flatten_ledger(child, path, output)
            else:
                _flatten_numeric(child, path, output,
                                 aggregate_ledgers=aggregate_ledgers)
        return
    if isinstance(value, Mapping):
        for key, child in sorted(value.items(), key=lambda pair: str(pair[0])):
            _flatten_numeric(child, f"{prefix}[{key}]", output,
                             aggregate_ledgers=aggregate_ledgers)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _flatten_numeric(child, f"{prefix}[{index}]", output,
                             aggregate_ledgers=aggregate_ledgers)


def _flatten_ledger(value: Any, prefix: str, output: dict[str, float]) -> None:
    leaves: dict[str, float] = {}
    _flatten_numeric(value, prefix, leaves)
    numbers = np.fromiter(leaves.values(), dtype=np.float64)
    output[f"{prefix}.leaf_count"] = float(numbers.size)
    output[f"{prefix}.nonzero_count"] = float(np.count_nonzero(numbers))
    output[f"{prefix}.mean"] = float(numbers.mean()) if numbers.size else 0.0
    output[f"{prefix}.max"] = float(numbers.max()) if numbers.size else 0.0


def engine_parameter_mapping(engine: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    _flatten_numeric(engine, "engine", result)
    return result


def physical_state_mapping(state: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    _flatten_numeric(state, "state", result, aggregate_ledgers=True)
    # Bench services are external to the engine, but their finite inventory
    # affects its next transition and must be visible to the learner.
    _flatten_numeric(getattr(state, "test_stand_cooling", None),
                     "state.test_stand_cooling", result)
    return result


def _rows_to_matrix(rows: Sequence[Mapping[str, float]], names: Sequence[str]):
    return np.asarray([[row.get(name, 0.0) for name in names] for row in rows],
                      dtype=np.float64)


def _station_shaft_load(sim: Any, shaft_load_w: float) -> None:
    """Apply one owned shaft-power load exactly once, as the station does.

    The power value is the ECU's advance knowledge of the accessory load.  Its
    mechanical reaction enters separately through the existing load shaft as
    P/omega; using an unrelated fraction of peak engine torque made the former
    demo's feedforward signal disagree with the load actually being applied.
    """
    shaft_load_w = max(0.0, float(shaft_load_w))
    ratio = max(float(sim._current_gear_ratio()), 1e-9)
    transfer_omega = max(
        float(sim.state.rpm) / ratio * 2.0 * math.pi / 60.0, 1e-6)
    sim.known_accessory_shaft_load_w = shaft_load_w
    sim.brake_load_nm = shaft_load_w / transfer_omega


def _transition_context(current_state: Mapping[str, float],
                        previous_state: Mapping[str, float] | None,
                        controls: Mapping[str, float],
                        previous_controls: Mapping[str, float] | None) -> dict[str, float]:
    """Small finite-memory surface; keeps the deployed network feed-forward."""
    valid = previous_state is not None and previous_controls is not None
    result = {"transition.history_valid": float(valid)}
    for name in _TRANSITION_STATE_NAMES:
        current = float(current_state.get(name, 0.0))
        previous = float(previous_state.get(name, current)) if valid else current
        result[f"transition.delta.{name.removeprefix('state.')}"] = current - previous
    for name in _CONTROL_NAMES:
        current = float(controls.get(name, 0.0))
        previous = float(previous_controls.get(name, current)) if valid else current
        suffix = name.removeprefix("control.")
        result[f"transition.previous_control.{suffix}"] = previous
        result[f"transition.control_delta.{suffix}"] = current - previous
    return result


def episode_holdout_mask(dataset: EngineDataset, seed: int,
                         fraction: float = 0.2) -> np.ndarray:
    """Hold out complete trajectories so neighboring ticks cannot leak."""
    rng = np.random.default_rng(seed)
    held_out = np.zeros(len(dataset.episode_ids), dtype=bool)
    for fuel_id in range(len(FUEL_PROFILES)):
        rows = np.flatnonzero(dataset.fuel_ids == fuel_id)
        episodes = np.unique(dataset.episode_ids[rows])
        rng.shuffle(episodes)
        count = max(1, int(math.ceil(len(episodes) * fraction)))
        held_out[np.isin(dataset.episode_ids, episodes[:count])] = True
    return held_out


def sample_episode_batch_indices(rng: np.random.Generator,
                                 candidate_indices: np.ndarray,
                                 episode_ids: np.ndarray,
                                 batch_size: int) -> np.ndarray:
    """Choose distinct trajectories first, then a transition within each."""
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("cannot sample an empty transition set")
    available = np.unique(episode_ids[candidates])
    chosen: list[int] = []
    while len(chosen) < batch_size:
        order = rng.permutation(available)
        for episode in order:
            rows = candidates[episode_ids[candidates] == episode]
            chosen.append(int(rng.choice(rows)))
            if len(chosen) == batch_size:
                break
    return np.asarray(chosen, dtype=np.int64)


def episode_epoch_batches(rng: np.random.Generator,
                          candidate_indices: np.ndarray,
                          episode_ids: np.ndarray,
                          batch_size: int):
    """Yield one complete pass, with every transition weighted exactly once."""
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.size == 0 or batch_size < 1:
        raise ValueError("a pass needs transitions and a positive batch size")
    queues = {
        int(episode): list(map(int, rng.permutation(
            candidates[episode_ids[candidates] == episode])))
        for episode in np.unique(episode_ids[candidates])
    }
    while any(queues.values()):
        active = np.asarray([episode for episode, rows in queues.items() if rows],
                            dtype=np.int64)
        rng.shuffle(active)
        batch = [queues[int(episode)].pop() for episode in active[:batch_size]]
        valid = len(batch)
        if valid < batch_size:
            batch.extend([batch[-1]] * (batch_size - valid))
        weights = np.zeros(batch_size, dtype=np.float64)
        weights[:valid] = 1.0
        yield np.asarray(batch, dtype=np.int64), weights


def _regime_command(regime: str, progress: float, player_noise: float,
                    environment_noise: float) -> tuple[float, float, float, int | None, float]:
    """Return load, throttle, electrical load, gear, and clutch commands.

    Noise perturbs genuine operator/environment commands. It is never added to
    captured engine state or targets.
    """
    electrical = float(np.clip(0.24 + environment_noise, 0.0, 1.0))
    if regime == "starting":
        return 0.0, float(np.clip(0.18 + player_noise, 0.08, 0.32)), electrical, 0, 0.0
    if regime == "idling":
        return 0.0, 0.0, electrical, None, 1.0
    if regime == "idle-recovery":
        load = 0.28 if progress < 0.38 else 0.0
        return load, 0.0, electrical, None, 1.0
    if regime == "idle-load-compensation":
        load = 0.10 if progress < 0.35 else 0.28
        return load, 0.0, electrical, None, 1.0
    if regime == "load-recovery":
        load = 0.52 if progress < 0.52 else 0.10
        throttle = 0.48 if progress < 0.52 else 0.14
        return load, float(np.clip(throttle + player_noise, 0.09, 0.72)), electrical, None, 1.0
    if regime == "high-end":
        return 0.78, float(np.clip(0.88 + player_noise, 0.70, 1.0)), electrical, None, 1.0
    if regime == "upshift":
        # Disengage, select the next ratio, and re-engage around midpoint.
        if progress < 0.42:
            gear, clutch = 2, 1.0
        elif progress < 0.50:
            gear, clutch = 2, 0.0
        elif progress < 0.58:
            gear, clutch = 3, 0.35
        else:
            gear, clutch = 3, 1.0
        return 0.28, float(np.clip(0.48 + player_noise, 0.25, 0.70)), electrical, gear, clutch
    if regime == "downshift":
        if progress < 0.42:
            gear, clutch = 4, 1.0
        elif progress < 0.50:
            gear, clutch = 4, 0.0
        elif progress < 0.58:
            gear, clutch = 3, 0.35
        else:
            gear, clutch = 3, 1.0
        return 0.28, float(np.clip(0.36 + player_noise, 0.18, 0.62)), electrical, gear, clutch
    raise ValueError(f"unknown engine transition regime {regime!r}")


def _named_regime_for_episode(episode_index: int) -> tuple[int, str]:
    """Cycle only scripted regimes; randomized coverage has its own sampler."""
    name = NAMED_TRANSITION_REGIMES[
        episode_index % len(NAMED_TRANSITION_REGIMES)]
    return TRANSITION_REGIMES.index(name), name


def make_multifuel_engine_data(samples_per_fuel: int = 256, seed: int = 1729,
                               *, episodes_per_fuel: int = 8,
                               random_coverage_samples: int = 0,
                               engine_identity: str = ENGINE_IDENTITY,
                               engine_toy_path: str | Path | None = None) -> EngineDataset:
    """Collect independent, finite-memory LDT-465 transition trajectories."""
    if samples_per_fuel < 10:
        raise ValueError("samples_per_fuel must be at least ten")
    if episodes_per_fuel < 2:
        raise ValueError("episodes_per_fuel must be at least two")
    if random_coverage_samples < 0:
        raise ValueError("random_coverage_samples must be non-negative")
    EngineCycleSim, get_engine = _load_engine_toy(engine_toy_path)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    engine = get_engine(engine_identity)
    fuel_profiles = tuple(engine.fuel_compatibility) or ("none",)
    engine_parameters = engine_parameter_mapping(engine)
    idle_power_w = max(engine.peak_torque_nm * engine.idle_rpm * 2.0 * math.pi / 60.0, 1.0)
    feature_rows, target_rows = [], []
    load_fractions, fuel_ids, episode_ids, transition_indices, regime_ids = (
        [], [], [], [], [])
    engine_ids = []
    feedforward_rows = 0
    random_episode_cursor = len(fuel_profiles) * episodes_per_fuel

    def record_transition(sim, *, fuel_id: int, episode_id: int,
                          local_index: int, regime_id: int,
                          engine_id: int = 0,
                          load_fraction: float, applied_load_w: float,
                          throttle: float, electrical_load: float,
                          external_brake_nm: float = 0.0,
                          starter_signal: float = 0.0,
                          ignition_enabled: float = 1.0,
                          previous_state=None, previous_controls=None):
        nonlocal feedforward_rows
        sim.throttle = throttle
        sim.electrical_load_frac = electrical_load
        _station_shaft_load(sim, applied_load_w)
        sim.brake_load_nm += max(0.0, external_brake_nm)
        if starter_signal > 0.5:
            sim.engage_starter()
        sim.state.ignition_cut = ignition_enabled <= 0.5
        before = physical_state_mapping(sim.state)
        known_total_w = (sim.state.ac_compressor_load_w
                         + sim.electrical.reading.alternator_shaft_load_w
                         + applied_load_w)
        feedforward = sim.ecu.idle_load_feedforward_frac(
            sim.engine, known_total_w)
        if throttle < 0.08 and feedforward > 0.0:
            feedforward_rows += 1
        controls = {
            "control.throttle": throttle,
            "control.brake_load_nm": sim.brake_load_nm,
            "control.electrical_load_frac": sim.electrical_load_frac,
            "control.known_accessory_shaft_load_w": applied_load_w,
            "control.idle_load_feedforward_frac": feedforward,
            "control.clutch_frac": sim.clutch_frac,
            "control.gear_index": float(sim.gear_index),
            "control.starter_signal": float(starter_signal),
            "control.ignition_enabled": float(ignition_enabled),
        }
        controls.update({f"fuel.{name}": float(index == fuel_id)
                         for index, name in enumerate(fuel_profiles)})
        context = _transition_context(
            before, previous_state, controls, previous_controls)
        feature_rows.append({**engine_parameters, **before, **controls,
                             **context})
        sim.step(0.005)
        after = physical_state_mapping(sim.state)
        target_rows.append({name: after.get(name, 0.0) - value
                            for name, value in before.items()})
        load_fractions.append(load_fraction)
        fuel_ids.append(fuel_id)
        episode_ids.append(episode_id)
        transition_indices.append(local_index)
        regime_ids.append(regime_id)
        engine_ids.append(engine_id)
        return before, controls

    for fuel_id, fuel in enumerate(fuel_profiles):
        episode_count = min(episodes_per_fuel, samples_per_fuel // 2)
        lengths = np.full(episode_count, samples_per_fuel // episode_count,
                          dtype=np.int64)
        lengths[:samples_per_fuel % episode_count] += 1
        captured_operating_states: dict[str, Any] = {}
        if episode_count > TRANSITION_REGIMES.index("high-end"):
            capture = EngineCycleSim(get_engine(engine_identity))
            capture.start()
            if fuel != "none":
                capture.fuel_choice = fuel
            capture.gear_index = 0
            capture.clutch_frac = 0.0
            capture.throttle = 0.88
            shift_rpm = max(1_200.0, capture.engine.idle_rpm * 1.8)
            high_rpm = capture.engine.power_peak_rpm * 0.92
            for _ in range(100):
                capture.step(0.05)
                if ("shift" not in captured_operating_states
                        and capture.state.rpm >= shift_rpm):
                    captured_operating_states["shift"] = copy.deepcopy(capture)
                if capture.state.rpm >= high_rpm:
                    captured_operating_states["high-end"] = copy.deepcopy(capture)
                    break
            # A damaged or unusual fuel may not reach the requested threshold;
            # its highest genuine state is still preferable to a fabricated one.
            captured_operating_states.setdefault("shift", copy.deepcopy(capture))
            captured_operating_states.setdefault("high-end", copy.deepcopy(capture))
        for local_episode, episode_length in enumerate(lengths):
            regime_id, regime = _named_regime_for_episode(local_episode)
            if regime == "high-end":
                sim = copy.deepcopy(captured_operating_states["high-end"])
            elif regime in ("upshift", "downshift"):
                sim = copy.deepcopy(captured_operating_states["shift"])
            else:
                sim = EngineCycleSim(get_engine(engine_identity))
                if fuel != "none":
                    sim.fuel_choice = fuel
            if regime == "starting":
                sim.engage_starter()
            elif regime not in ("high-end", "upshift", "downshift"):
                sim.start()
            # Match the station's safe service-shaft engagement: select its
            # direct/high service gear, synchronize the unloaded shaft, and
            # ramp the existing clutch before presenting an accessory load.
            initial_gear = (
                2 if regime == "upshift" else 4 if regime == "downshift"
                else 0 if regime == "starting"
                else len(sim.engine.transmission.gear_ratios))
            sim.gear_index = initial_gear
            sim.clutch_frac = 0.0
            ratio = sim._current_gear_ratio()
            # Neutral has no shaft connection; preserve the load's speed.
            if ratio > 0.0:
                sim._load_omega = sim._omega / ratio
            if regime != "starting":
                for engagement in range(4):
                    sim.clutch_frac = (engagement + 1) / 4.0
                    _station_shaft_load(sim, 0.0)
                    sim.step(0.005)

            episode_id = fuel_id * episodes_per_fuel + local_episode
            previous_state = previous_controls = None
            applied_load_w = 0.0
            player_noise = environment_noise = 0.0
            for local_index in range(int(episode_length)):
                progress = local_index / max(int(episode_length) - 1, 1)
                player_noise = 0.82 * player_noise + float(rng.normal(0.0, 0.025))
                environment_noise = (0.94 * environment_noise
                                     + float(rng.normal(0.0, 0.018)))
                (load_fraction, throttle, electrical_load, gear,
                 clutch) = _regime_command(
                    regime, progress, player_noise, environment_noise)
                if gear is not None:
                    sim.gear_index = gear
                sim.clutch_frac = clutch
                target_load_w = load_fraction * idle_power_w
                applied_load_w += 0.35 * (target_load_w - applied_load_w)
                previous_state, previous_controls = record_transition(
                    sim, fuel_id=fuel_id, episode_id=episode_id,
                    local_index=local_index, regime_id=regime_id,
                    load_fraction=load_fraction,
                    applied_load_w=applied_load_w, throttle=throttle,
                    electrical_load=electrical_load,
                    starter_signal=float(regime == "starting"),
                    ignition_enabled=1.0,
                    previous_state=previous_state,
                    previous_controls=previous_controls)

        # Additional broad trajectories remain fully teacher-generated. Each
        # episode has independently sampled initial/command conditions, while
        # low-pass command motion preserves realistic temporal continuity.
        fuel_random_samples = (random_coverage_samples // len(fuel_profiles)
                               + int(fuel_id < random_coverage_samples
                                     % len(fuel_profiles)))
        if fuel_random_samples:
            random_episode_count = min(
                12, max(1, math.ceil(fuel_random_samples / 40)))
            random_lengths = np.full(
                random_episode_count,
                fuel_random_samples // random_episode_count,
                dtype=np.int64)
            random_lengths[:fuel_random_samples % random_episode_count] += 1
            random_regime_id = TRANSITION_REGIMES.index("random-coverage")
            for episode_length in random_lengths:
                sim = EngineCycleSim(get_engine(engine_identity))
                if fuel != "none":
                    sim.fuel_choice = fuel
                if rng.random() < 0.14:
                    sim.engage_starter()
                    sim.gear_index = 0
                    sim.clutch_frac = 0.0
                else:
                    sim.start()
                    sim.gear_index = int(rng.integers(
                        0, len(sim.engine.transmission.gear_ratios) + 1))
                    sim.clutch_frac = 0.0
                    ratio = sim._current_gear_ratio()
                    if ratio > 0.0:
                        sim._load_omega = sim._omega / ratio
                    for engagement in range(4):
                        sim.clutch_frac = (engagement + 1) / 4.0
                        _station_shaft_load(sim, 0.0)
                        sim.step(0.005)

                    # Arrive at varied initial states through genuine engine
                    # motion, including unloaded revs and partially loaded
                    # operation; no state variable is fabricated.
                    warmup_throttle = float(rng.uniform(0.0, 1.0))
                    warmup_load = float(rng.uniform(0.0, 0.75))
                    for _ in range(int(rng.integers(0, 25))):
                        sim.throttle = warmup_throttle
                        sim.electrical_load_frac = float(rng.uniform(0.0, 1.0))
                        _station_shaft_load(sim, warmup_load * idle_power_w)
                        sim.step(0.02)

                episode_id = random_episode_cursor
                random_episode_cursor += 1
                previous_state = previous_controls = None
                throttle = float(rng.uniform(0.0, 1.0))
                load_fraction = float(rng.uniform(0.0, 0.82))
                electrical_load = float(rng.uniform(0.0, 1.0))
                applied_load_w = 0.0
                external_brake_nm = 0.0
                shift_release = 0
                command_hold = 0
                ignition_enabled = 1.0
                starter_signal = 0.0
                for local_index in range(int(episode_length)):
                    if command_hold <= 0:
                        command_hold = int(rng.integers(4, 13))
                        draw = rng.random()
                        throttle_target = (0.0 if draw < 0.18 else
                                           1.0 if draw > 0.86 else
                                           float(rng.uniform(0.0, 1.0)))
                        load_target = (0.0 if rng.random() < 0.22 else
                                       float(rng.uniform(0.0, 0.85)))
                        electrical_target = float(rng.uniform(0.0, 1.0))
                        external_brake_target = (0.0 if rng.random() < 0.35
                                                 else float(rng.uniform(
                                                     0.0,
                                                     sim.engine.peak_torque_nm)))
                        if rng.random() < 0.08:
                            ignition_enabled = 1.0 - ignition_enabled
                        starter_signal = float(
                            ignition_enabled > 0.5 and rng.random() < 0.08)
                        if rng.random() < 0.16:
                            sim.gear_index = int(rng.integers(
                                0, len(sim.engine.transmission.gear_ratios) + 1))
                            shift_release = int(rng.integers(2, 6))
                    command_hold -= 1
                    throttle += 0.28 * (throttle_target - throttle)
                    load_fraction += 0.20 * (load_target - load_fraction)
                    electrical_load += 0.18 * (
                        electrical_target - electrical_load)
                    external_brake_nm += 0.22 * (
                        external_brake_target - external_brake_nm)
                    if shift_release:
                        sim.clutch_frac = 0.0
                        shift_release -= 1
                    else:
                        sim.clutch_frac += 0.32 * (1.0 - sim.clutch_frac)
                    target_load_w = load_fraction * idle_power_w
                    applied_load_w += 0.25 * (
                        target_load_w - applied_load_w)
                    previous_state, previous_controls = record_transition(
                        sim, fuel_id=fuel_id, episode_id=episode_id,
                        local_index=local_index,
                        regime_id=random_regime_id,
                        load_fraction=load_fraction,
                        applied_load_w=applied_load_w,
                        throttle=float(np.clip(throttle, 0.0, 1.0)),
                        electrical_load=float(np.clip(
                            electrical_load, 0.0, 1.0)),
                        external_brake_nm=external_brake_nm,
                        starter_signal=starter_signal,
                        ignition_enabled=ignition_enabled,
                        previous_state=previous_state,
                        previous_controls=previous_controls)

    feature_names = tuple(sorted(set().union(*(row.keys() for row in feature_rows))))
    target_names = tuple(sorted(set().union(*(row.keys() for row in target_rows))))
    return EngineDataset(
        _rows_to_matrix(feature_rows, feature_names),
        _rows_to_matrix(target_rows, target_names),
        np.asarray(load_fractions), np.asarray(fuel_ids, dtype=np.int64),
        feature_names, target_names, feedforward_rows,
        np.asarray(episode_ids, dtype=np.int64),
        np.asarray(transition_indices, dtype=np.int64),
        np.asarray(regime_ids, dtype=np.int64),
        np.asarray(engine_ids, dtype=np.int64),
        (engine_identity,),
    )


def combine_engine_datasets(datasets: Sequence[EngineDataset]) -> EngineDataset:
    """Pad heterogeneous engine captures into one stable union ABI."""
    if not datasets:
        raise ValueError("at least one engine dataset is required")
    engine_names = tuple(name for dataset in datasets
                         for name in dataset.engine_names)
    feature_names = tuple(sorted(
        set().union(*(dataset.feature_names for dataset in datasets))
        | {f"engine_profile.{name}" for name in engine_names}))
    target_names = tuple(sorted(
        set().union(*(dataset.target_names for dataset in datasets))))
    features, targets = [], []
    load_fractions, fuel_ids, episode_ids = [], [], []
    transition_indices, regime_ids, engine_ids = [], [], []
    episode_cursor = 0
    for engine_id, dataset in enumerate(datasets):
        feature_lookup = {name: index
                          for index, name in enumerate(dataset.feature_names)}
        target_lookup = {name: index
                         for index, name in enumerate(dataset.target_names)}
        x = np.zeros((len(dataset.features), len(feature_names)), dtype=np.float64)
        y = np.zeros((len(dataset.targets), len(target_names)), dtype=np.float64)
        for column, name in enumerate(feature_names):
            if name in feature_lookup:
                x[:, column] = dataset.features[:, feature_lookup[name]]
            elif name == f"engine_profile.{dataset.engine_names[0]}":
                x[:, column] = 1.0
        for column, name in enumerate(target_names):
            if name in target_lookup:
                y[:, column] = dataset.targets[:, target_lookup[name]]
        local_episodes = np.unique(dataset.episode_ids)
        episode_map = {int(value): episode_cursor + index
                       for index, value in enumerate(local_episodes)}
        features.append(x)
        targets.append(y)
        load_fractions.append(dataset.load_fractions)
        fuel_ids.append(dataset.fuel_ids)
        episode_ids.append(np.asarray([
            episode_map[int(value)] for value in dataset.episode_ids
        ], dtype=np.int64))
        transition_indices.append(dataset.transition_indices)
        regime_ids.append(dataset.regime_ids)
        engine_ids.append(np.full(len(x), engine_id, dtype=np.int64))
        episode_cursor += len(local_episodes)
    return EngineDataset(
        np.concatenate(features), np.concatenate(targets),
        np.concatenate(load_fractions), np.concatenate(fuel_ids),
        feature_names, target_names,
        sum(dataset.idle_feedforward_rows for dataset in datasets),
        np.concatenate(episode_ids), np.concatenate(transition_indices),
        np.concatenate(regime_ids), np.concatenate(engine_ids), engine_names)


def save_engine_dataset(path: str | Path, dataset: EngineDataset) -> Path:
    """Persist a captured union ABI without pickle-dependent objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez(
        temporary,
        features=dataset.features, targets=dataset.targets,
        load_fractions=dataset.load_fractions, fuel_ids=dataset.fuel_ids,
        feature_names=np.asarray(dataset.feature_names),
        target_names=np.asarray(dataset.target_names),
        idle_feedforward_rows=np.asarray(dataset.idle_feedforward_rows),
        episode_ids=dataset.episode_ids,
        transition_indices=dataset.transition_indices,
        regime_ids=dataset.regime_ids, engine_ids=dataset.engine_ids,
        engine_names=np.asarray(dataset.engine_names))
    temporary.replace(path)
    return path


def load_engine_dataset(path: str | Path) -> EngineDataset:
    """Load a dataset written by :func:`save_engine_dataset`."""
    with np.load(Path(path), allow_pickle=False) as stored:
        return EngineDataset(
            stored["features"], stored["targets"], stored["load_fractions"],
            stored["fuel_ids"], tuple(map(str, stored["feature_names"])),
            tuple(map(str, stored["target_names"])),
            int(stored["idle_feedforward_rows"]), stored["episode_ids"],
            stored["transition_indices"], stored["regime_ids"],
            stored["engine_ids"], tuple(map(str, stored["engine_names"])))


def make_engine_union_data(*, random_coverage_samples: int = 3_000,
                           named_samples_per_fuel: int = 16,
                           episodes_per_fuel: int = 8,
                           seed: int = 1729,
                           engine_identities: Sequence[str] | None = None,
                           engine_toy_path: str | Path | None = None) -> EngineDataset:
    """Balanced all-profile capture compiled into a padded union ABI."""
    _, _, roster = _load_engine_roster(engine_toy_path)
    identities = tuple(engine_identities) if engine_identities else roster
    if not identities:
        raise ValueError("engine_identities must not be empty")
    per_engine = np.full(
        len(identities), random_coverage_samples // len(identities),
        dtype=np.int64)
    per_engine[:random_coverage_samples % len(identities)] += 1
    datasets = [
        make_multifuel_engine_data(
            samples_per_fuel=named_samples_per_fuel,
            episodes_per_fuel=episodes_per_fuel,
            random_coverage_samples=int(per_engine[index]),
            engine_identity=identity, seed=seed + index * 1009,
            engine_toy_path=engine_toy_path)
        for index, identity in enumerate(identities)
    ]
    return combine_engine_datasets(datasets)


def _predict(model, x: AT) -> np.ndarray:
    with autograd.no_grad():
        return np.asarray(model.forward(x).tolist(), dtype=np.float64)


def _standardize(train: np.ndarray, test: np.ndarray, *, range_floor: float = 0.0,
                 return_statistics: bool = False):
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    if range_floor:
        observed = np.max(np.abs(np.concatenate((train, test), axis=0)),
                          axis=0, keepdims=True)
        scale = np.maximum(scale, range_floor * observed)
    scale[scale < 1e-10] = 1.0
    standardized = ((train - mean) / scale, (test - mean) / scale)
    if return_statistics:
        return *standardized, mean, scale
    return standardized


def _automatic_backend() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "torch-cuda"
    except Exception:
        pass
    return "numpy"


def run_multifuel_comparison(*, seeds: Sequence[int] = (1729, 1730, 1731),
                             samples_per_fuel: int = 256,
                             episodes_per_fuel: int = 8,
                             random_coverage_samples: int = 0,
                             dendrites_per_neuron: int = 2, steps: int = 180,
                             learning_rate: float = 0.012, backend: str = "auto",
                             engine_toy_path: str | Path | None = None) -> MultiFuelResult:
    """Jointly train every model parameter on the full engine transition."""
    dataset = make_multifuel_engine_data(
        samples_per_fuel, int(seeds[0]),
        episodes_per_fuel=episodes_per_fuel,
        random_coverage_samples=random_coverage_samples,
        engine_toy_path=engine_toy_path)
    test_mask = episode_holdout_mask(dataset, int(seeds[0]) + 91)
    train_mask = ~test_mask
    x_train, x_test = _standardize(dataset.features[train_mask],
                                   dataset.features[test_mask])
    y_train, y_test = _standardize(dataset.targets[train_mask],
                                   dataset.targets[test_mask], range_floor=0.05)
    # A machine-replacement ABI cannot make ports disappear merely because a
    # particular capture held one field constant. Predict the complete numeric
    # state delta, including every torque/power output, under one stable shape.
    active_outputs = np.ones(dataset.targets.shape[1], dtype=bool)
    y_train, y_test = y_train[:, active_outputs], y_test[:, active_outputs]
    backend = _automatic_backend() if backend == "auto" else backend
    selected_backend = "torch" if backend == "torch-cuda" else backend
    device = "cuda" if backend == "torch-cuda" else "cpu"
    previous_tape = autograd.tape
    runs = []
    ordinary_parameters = perforated_parameters = 0
    try:
        with AT.use_backend(selected_backend, device):
            x_train_at = AT.tensor(x_train, dtype="float64")
            y_train_at = AT.tensor(y_train, dtype="float64")
            x_test_at = AT.tensor(x_test, dtype="float64")
            for seed in seeds:
                autograd.tape = GradTape()
                random.seed(int(seed))
                ordinary = Linear(x_train.shape[1], y_train.shape[1],
                                  like=x_train_at, init="xavier")
                random.seed(int(seed))
                perforated = PerforatedLinear(
                    x_train.shape[1], y_train.shape[1], like=x_train_at,
                    dendrites_per_neuron=dendrites_per_neuron,
                    init="xavier", active=True)
                _train_phase(ordinary, ordinary.parameters(), x_train_at, y_train_at,
                             steps=steps, learning_rate=learning_rate)
                _train_phase(perforated, perforated.parameters(), x_train_at, y_train_at,
                             steps=steps, learning_rate=learning_rate)
                ordinary_prediction = _predict(ordinary, x_test_at)
                perforated_prediction = _predict(perforated, x_test_at)
                ordinary_error = np.mean((ordinary_prediction - y_test) ** 2, axis=1)
                perforated_error = np.mean((perforated_prediction - y_test) ** 2, axis=1)
                test_loads = dataset.load_fractions[test_mask]

                def by_load(error):
                    return tuple(float(error[np.isclose(test_loads, load)].mean())
                                 for load in LOAD_LEVELS)

                runs.append(EngineRun(
                    int(seed), float(ordinary_error.mean()), float(perforated_error.mean()),
                    by_load(ordinary_error), by_load(perforated_error)))
                ordinary_parameters = _parameter_count(ordinary.parameters())
                perforated_parameters = _parameter_count(perforated.parameters())
    finally:
        autograd.tape = previous_tape
    return MultiFuelResult(
        tuple(runs), LOAD_BANDS, ordinary_parameters, perforated_parameters,
        int(train_mask.sum()), int(test_mask.sum()), x_train.shape[1],
        int(active_outputs.sum()), dataset.idle_feedforward_rows, backend)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-fuel", type=int, default=256)
    parser.add_argument("--episodes-per-fuel", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 1730, 1731])
    parser.add_argument("--dendrites", type=int, default=2)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.012)
    parser.add_argument("--backend", choices=("auto", "numpy", "torch", "torch-cuda"), default="auto")
    parser.add_argument("--engine-toy-path")
    args = parser.parse_args()
    result = run_multifuel_comparison(
        seeds=args.seeds, samples_per_fuel=args.samples_per_fuel,
        episodes_per_fuel=args.episodes_per_fuel,
        dendrites_per_neuron=args.dendrites, steps=args.steps,
        learning_rate=args.learning_rate, backend=args.backend,
        engine_toy_path=args.engine_toy_path)
    print(f"backend={result.backend}; {result.train_samples} train / {result.test_samples} held out; "
          f"full surface={result.input_parameters} inputs / {result.output_parameters} moving outputs; "
          f"idle-feedforward rows={result.idle_feedforward_rows}")
    print(f"trainable parameters: ordinary={result.ordinary_parameters}; "
          f"perforated={result.perforated_parameters} (all jointly Adam-trained)")
    for run in result.runs:
        print(f"seed {run.seed}: ordinary={run.ordinary_mse:.7g} "
              f"perforated={run.perforated_mse:.7g} "
              f"improvement={run.ordinary_mse / max(run.perforated_mse, 1e-30):.2f}x")
    print(f"mean normalized held-out MSE: ordinary={result.ordinary_mean_mse:.7g}; "
          f"perforated={result.perforated_mean_mse:.7g}; "
          f"improvement={result.improvement_ratio:.2f}x")
    ordinary_bands = np.mean([run.ordinary_by_load for run in result.runs], axis=0)
    perforated_bands = np.mean([run.perforated_by_load for run in result.runs], axis=0)
    for name, ordinary, perforated in zip(result.load_bands, ordinary_bands, perforated_bands):
        print(f"  {name:8s} ordinary={ordinary:.7g} perforated={perforated:.7g}")


if __name__ == "__main__":
    main()
