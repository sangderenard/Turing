"""Live OpenGL/audio display for the compiled multifuel engine learner.

Interactive mode opens an OpenGL dot field and plays the engine's existing
stereo synthesizer. ``--headless`` runs the identical compiled learner without
a window and writes a final PNG plus WAV.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import wave

import numpy as np

from .demo_perforated_multifuel_engine import (
    ENGINE_IDENTITY, FUEL_PROFILES, LOAD_LEVELS, MACHINE_OUTPUT_NAMES,
    TRANSITION_REGIMES,
    _load_engine_toy,
    _standardize, _station_shaft_load, _transition_context,
    engine_parameter_mapping, episode_epoch_batches, episode_holdout_mask,
    load_engine_dataset, make_engine_union_data, physical_state_mapping,
    save_engine_dataset,
)
from ....compiler.compiled_perforated_adam import CompiledPerforatedAdam


# Production-oriented defaults. Development/smoke runs can still override
# every value on the command line without changing the compiled ABI.
DEFAULT_EPOCHS = 32
DEFAULT_PASSES_PER_EPOCH = 1
DEFAULT_SAMPLES_PER_FUEL = 256
DEFAULT_EPISODES_PER_FUEL = 256
DEFAULT_RANDOM_COVERAGE_SAMPLES = 3_000
DEFAULT_BATCH_SIZE = 64
DEFAULT_BRANCHES = 2
DEFAULT_LEARNING_RATE = 0.002
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_MAX_GLOBAL_GRADIENT_NORM = 1.0
DEFAULT_ADAPTIVE_SAMPLE_REFRESH = True
DEFAULT_REFRESH_GAP_GROWTH = 0.02
DEFAULT_REFRESH_NAMED_SAMPLES_PER_FUEL = 64
DEFAULT_REFRESH_RANDOM_COVERAGE_SAMPLES = 750
DEFAULT_MAX_SAMPLE_REFRESHES = 4


@dataclass(frozen=True)
class LiveRunArtifacts:
    image_path: Path
    audio_path: Path | None
    contract_path: Path
    metrics_path: Path
    initial_loss: float
    final_loss: float
    best_validation_loss: float


@dataclass
class LiveEngineControls:
    """Operator/system commands shared by the network and shadow teacher."""

    throttle: float = 0.0
    brake_load_nm: float = 0.0
    electrical_load_frac: float = 0.24
    known_accessory_shaft_load_w: float = 0.0
    gear_index: int = 0
    clutch_frac: float = 0.0
    starter_signal: bool = False
    ignition_enabled: bool = True
    fuel_choice: str = FUEL_PROFILES[0]


class NetworkEngineRollout:
    """Closed-loop engine state advanced only by compiled network weights."""

    def __init__(self, learner: CompiledPerforatedAdam, preprocessing: dict,
                 engine_toy_path=None,
                 engine_identity: str = ENGINE_IDENTITY) -> None:
        EngineCycleSim, get_engine = _load_engine_toy(engine_toy_path)
        self._EngineCycleSim = EngineCycleSim
        self._get_engine = get_engine
        self.engine_identity = engine_identity
        seed = EngineCycleSim(get_engine(engine_identity))
        seed.start()
        self.learner = learner
        self.engine = seed.engine
        self.ecu = seed.ecu
        self._seed_state = copy.deepcopy(seed.state)
        self._seed_values = physical_state_mapping(seed.state)
        self.engine_parameters = engine_parameter_mapping(seed.engine)
        self.feature_names = tuple(preprocessing["feature_names"])
        self.fuel_names = tuple(name.removeprefix("fuel.")
                                for name in self.feature_names
                                if name.startswith("fuel."))
        self.feature_mean = np.asarray(preprocessing["feature_mean"])
        self.feature_scale = np.asarray(preprocessing["feature_scale"])
        self.target_names = tuple(preprocessing["target_names"])
        self.target_mean = np.asarray(preprocessing["target_mean"])
        self.target_scale = np.asarray(preprocessing["target_scale"])
        self.engine_names = tuple(preprocessing.get(
            "engine_names", (engine_identity,)))
        self.compiled_ns = deque(maxlen=2048)
        self.transition_ns = deque(maxlen=2048)
        self.deadline_misses = 0
        self.timed_transitions = 0
        self.divergence_count = 0
        self.reset()
        self.brake_load_nm = 0.0
        self.damage_events = []
        self._drivetrain = None

    def set_engine(self, engine_identity: str) -> None:
        if engine_identity not in self.engine_names:
            raise ValueError(f"engine profile {engine_identity!r} is not trained")
        seed = self._EngineCycleSim(self._get_engine(engine_identity))
        seed.start()
        self.engine_identity = engine_identity
        self.engine = seed.engine
        self.ecu = seed.ecu
        self._seed_state = copy.deepcopy(seed.state)
        self._seed_values = physical_state_mapping(seed.state)
        self.engine_parameters = engine_parameter_mapping(seed.engine)
        self.reset()

    def reset(self) -> None:
        self.state = copy.deepcopy(self._seed_state)
        self.state_values = dict(self._seed_values)
        self.previous_state = None
        self.previous_controls = None
        self.throttle = 0.0
        self.diverged = False

    def performance_snapshot(self) -> dict[str, float | int]:
        """Runtime measurements for the actual user-driven network path."""
        compiled_ms = np.asarray(self.compiled_ns, dtype=np.float64) / 1e6
        transition_ms = np.asarray(self.transition_ns, dtype=np.float64) / 1e6
        mean_compiled = float(compiled_ms.mean()) if compiled_ms.size else 0.0
        mean_transition = float(transition_ms.mean()) if transition_ms.size else 0.0
        return {
            "sample_count": self.timed_transitions,
            "compiled_mean_ms": mean_compiled,
            "compiled_p95_ms": (float(np.percentile(compiled_ms, 95.0))
                                if compiled_ms.size else 0.0),
            "compiled_transitions_per_second": (
                1000.0 / mean_compiled if mean_compiled > 0.0 else 0.0),
            "end_to_end_mean_ms": mean_transition,
            "end_to_end_transitions_per_second": (
                1000.0 / mean_transition if mean_transition > 0.0 else 0.0),
            "realtime_factor": (5.0 / mean_transition
                                if mean_transition > 0.0 else 0.0),
            "deadline_miss_fraction": (
                self.deadline_misses / self.timed_transitions
                if self.timed_transitions else 0.0),
            "divergence_count": self.divergence_count,
        }

    def _mark_diverged(self) -> None:
        if not self.diverged:
            self.divergence_count += 1
        self.diverged = True

    @property
    def rpm(self) -> float:
        return float(self.state_values.get("state.rpm", 0.0))

    def drain_damage_events(self):
        events, self.damage_events = self.damage_events, []
        return events

    def _publish_audio_state(self) -> None:
        for name, value in self.state_values.items():
            suffix = name.removeprefix("state.")
            if "." not in suffix and "[" not in suffix and hasattr(self.state, suffix):
                setattr(self.state, suffix, value if math.isfinite(value) else 0.0)
        # Protect the synthesizer only; the unbounded learned value remains in
        # state_values and therefore remains visible to drift/error reporting.
        self.state.rpm = float(np.clip(self.rpm, 0.0, self.engine.redline_rpm * 1.5))

    def correct_from_sim(self, sim) -> None:
        """Re-anchor visible learned state to an occasional teacher snapshot."""
        self.state = copy.deepcopy(sim.state)
        self.correct_from_values(physical_state_mapping(sim.state))

    def correct_from_values(self, values: dict[str, float]) -> None:
        """Re-anchor without making the real-time thread own a simulator."""
        self.state_values = dict(values)
        self.previous_state = None
        self.previous_controls = None
        self.diverged = False
        self._publish_audio_state()

    def step(self, command: LiveEngineControls | float) -> None:
        if self.diverged:
            return
        transition_started = time.perf_counter_ns()
        if isinstance(command, (float, int)):
            compatible_fuels = tuple(self.engine.fuel_compatibility)
            command = LiveEngineControls(
                throttle=float(command),
                gear_index=len(self.engine.transmission.gear_ratios),
                clutch_frac=1.0,
                fuel_choice=(compatible_fuels[0]
                             if compatible_fuels else "none"))
        self.throttle = float(np.clip(command.throttle, 0.0, 1.0))
        known_load_w = max(0.0, float(command.known_accessory_shaft_load_w))
        known_total_w = max(
            0.0, self.state_values.get("state.ac_compressor_load_w", 0.0)
            + known_load_w)
        feedforward = self.ecu.idle_load_feedforward_frac(
            self.engine, known_total_w)
        controls = {
            "control.throttle": self.throttle,
            "control.brake_load_nm": max(0.0, float(command.brake_load_nm)),
            "control.electrical_load_frac": float(np.clip(
                command.electrical_load_frac, 0.0, 3.0)),
            "control.known_accessory_shaft_load_w": known_load_w,
            "control.idle_load_feedforward_frac": feedforward,
            "control.clutch_frac": float(np.clip(command.clutch_frac, 0.0, 1.0)),
            "control.gear_index": float(command.gear_index),
            "control.starter_signal": float(command.starter_signal),
            "control.ignition_enabled": float(command.ignition_enabled),
        }
        controls.update({f"fuel.{name}": float(name == command.fuel_choice)
                         for name in self.fuel_names})
        context = _transition_context(
            self.state_values, self.previous_state, controls,
            self.previous_controls)
        profile = {f"engine_profile.{name}": float(name == self.engine_identity)
                   for name in self.engine_names}
        row = {**self.engine_parameters, **profile,
               **self.state_values, **controls,
               **context}
        raw = np.asarray([row.get(name, 0.0) for name in self.feature_names])
        normalized = (raw - self.feature_mean) / self.feature_scale
        if not np.all(np.isfinite(normalized)):
            self._mark_diverged()
            return
        batch = self.learner.shapes["x"][0]
        compiled_started = time.perf_counter_ns()
        prediction = self.learner.forward(np.tile(normalized, (batch, 1)))[0]
        self.compiled_ns.append(time.perf_counter_ns() - compiled_started)
        delta = prediction * self.target_scale + self.target_mean
        before = dict(self.state_values)
        updated = dict(self.state_values)
        for name, change in zip(self.target_names, delta):
            updated[name] = updated.get(name, 0.0) + float(change)
        if (not all(math.isfinite(value) for value in updated.values())
                or abs(updated.get("state.rpm", 0.0)) > self.engine.redline_rpm * 20.0):
            self._mark_diverged()
            return
        self.state_values = updated
        self.previous_state = before
        self.previous_controls = controls
        self._publish_audio_state()
        elapsed = time.perf_counter_ns() - transition_started
        self.transition_ns.append(elapsed)
        self.timed_transitions += 1
        if elapsed > 5_000_000:
            self.deadline_misses += 1


class LiveShadowTeacher:
    """Real simulator receiving the exact live controls and yielding examples."""

    def __init__(self, preprocessing: dict, engine_toy_path=None) -> None:
        EngineCycleSim, get_engine = _load_engine_toy(engine_toy_path)
        self._EngineCycleSim = EngineCycleSim
        self._get_engine = get_engine
        self.engine_names = tuple(preprocessing.get(
            "engine_names", (ENGINE_IDENTITY,)))
        self.engine_identity = self.engine_names[0]
        self.sim = EngineCycleSim(get_engine(self.engine_identity))
        self.sim.start()
        compatible = tuple(self.sim.engine.fuel_compatibility)
        if compatible:
            self.sim.fuel_choice = compatible[0]
        self.engine_parameters = engine_parameter_mapping(self.sim.engine)
        self.feature_names = tuple(preprocessing["feature_names"])
        self.fuel_names = tuple(name.removeprefix("fuel.")
                                for name in self.feature_names
                                if name.startswith("fuel."))
        self.feature_mean = np.asarray(preprocessing["feature_mean"])
        self.feature_scale = np.asarray(preprocessing["feature_scale"])
        self.target_names = tuple(preprocessing["target_names"])
        self.target_mean = np.asarray(preprocessing["target_mean"])
        self.target_scale = np.asarray(preprocessing["target_scale"])
        self.previous_state = None
        self.previous_controls = None

    def set_engine(self, engine_identity: str) -> None:
        if engine_identity not in self.engine_names:
            raise ValueError(f"engine profile {engine_identity!r} is not trained")
        self.engine_identity = engine_identity
        self.sim = self._EngineCycleSim(self._get_engine(engine_identity))
        self.sim.start()
        compatible = tuple(self.sim.engine.fuel_compatibility)
        if compatible:
            self.sim.fuel_choice = compatible[0]
        self.engine_parameters = engine_parameter_mapping(self.sim.engine)
        self.previous_state = None
        self.previous_controls = None

    def step(self, command: LiveEngineControls) -> tuple[np.ndarray, np.ndarray]:
        sim = self.sim
        sim.fuel_choice = command.fuel_choice
        sim.throttle = float(np.clip(command.throttle, 0.0, 1.0))
        sim.electrical_load_frac = float(np.clip(
            command.electrical_load_frac, 0.0, 3.0))
        sim.gear_index = int(np.clip(
            command.gear_index, -1,
            len(sim.engine.transmission.gear_ratios)))
        sim.clutch_frac = float(np.clip(command.clutch_frac, 0.0, 1.0))
        _station_shaft_load(sim, command.known_accessory_shaft_load_w)
        sim.brake_load_nm += max(0.0, float(command.brake_load_nm))
        if command.starter_signal:
            sim.engage_starter()
        sim.state.ignition_cut = not command.ignition_enabled
        before = physical_state_mapping(sim.state)
        known_total_w = (sim.state.ac_compressor_load_w
                         + sim.electrical.reading.alternator_shaft_load_w
                         + max(0.0, command.known_accessory_shaft_load_w))
        controls = {
            "control.throttle": sim.throttle,
            "control.brake_load_nm": sim.brake_load_nm,
            "control.electrical_load_frac": sim.electrical_load_frac,
            "control.known_accessory_shaft_load_w": max(
                0.0, command.known_accessory_shaft_load_w),
            "control.idle_load_feedforward_frac": (
                sim.ecu.idle_load_feedforward_frac(sim.engine, known_total_w)),
            "control.clutch_frac": sim.clutch_frac,
            "control.gear_index": float(sim.gear_index),
            "control.starter_signal": float(command.starter_signal),
            "control.ignition_enabled": float(command.ignition_enabled),
        }
        controls.update({f"fuel.{name}": float(name == sim.fuel_choice)
                         for name in self.fuel_names})
        context = _transition_context(
            before, self.previous_state, controls, self.previous_controls)
        profile = {f"engine_profile.{name}": float(name == self.engine_identity)
                   for name in self.engine_names}
        row = {**self.engine_parameters, **profile,
               **before, **controls, **context}
        raw_x = np.asarray([row.get(name, 0.0) for name in self.feature_names])
        sim.step(0.005)
        after = physical_state_mapping(sim.state)
        raw_y = np.asarray([after.get(name, 0.0) - before.get(name, 0.0)
                            for name in self.target_names])
        self.previous_state, self.previous_controls = before, controls
        return ((raw_x - self.feature_mean) / self.feature_scale,
                (raw_y - self.target_mean) / self.target_scale)

    def drift_from(self, rollout: NetworkEngineRollout) -> dict[str, float]:
        return self.drift_from_values(
            rollout, physical_state_mapping(self.sim.state))

    @staticmethod
    def drift_from_values(rollout: NetworkEngineRollout,
                          actual: dict[str, float]) -> dict[str, float]:
        names = rollout.target_names
        errors = np.asarray([
            rollout.state_values.get(name, 0.0) - actual.get(name, 0.0)
            for name in names
        ])
        scales = np.asarray([
            rollout.feature_scale[rollout.feature_names.index(name)]
            if name in rollout.feature_names else 1.0 for name in names
        ])
        return {
            "normalized_rmse": float(np.sqrt(np.mean((errors / scales) ** 2))),
            "rpm_absolute": abs(rollout.rpm - float(actual.get("state.rpm", 0.0))),
            "torque_absolute": abs(
                rollout.state_values.get("state.current_torque_nm", 0.0)
                - actual.get("state.current_torque_nm", 0.0)),
        }


def _copy_network_parameters(source: CompiledPerforatedAdam,
                             destination: CompiledPerforatedAdam) -> None:
    for name in source.parameter_names:
        destination.parameters[name][...] = source.parameters[name]


def evaluate_rollout_error(learner: CompiledPerforatedAdam, dataset,
                           preprocessing: dict, test_mask: np.ndarray) -> dict:
    """Run held-out episodes closed-loop and report accumulated state drift."""
    feature_names = tuple(preprocessing["feature_names"])
    target_names = tuple(preprocessing["target_names"])
    feature_lookup = {name: index for index, name in enumerate(feature_names)}
    target_lookup = {name: index for index, name in enumerate(dataset.target_names)}
    feature_mean = np.asarray(preprocessing["feature_mean"])
    feature_scale = np.asarray(preprocessing["feature_scale"])
    target_mean = np.asarray(preprocessing["target_mean"])
    target_scale = np.asarray(preprocessing["target_scale"])
    state_names = tuple(name for name in feature_names if name.startswith("state."))
    control_names = tuple(name for name in feature_names if name.startswith("control."))
    squared_normalized, absolute, final_absolute = [], {}, {}
    regime_squared: dict[str, list[float]] = {name: [] for name in TRANSITION_REGIMES}
    for episode in np.unique(dataset.episode_ids[test_mask]):
        rows = np.flatnonzero(test_mask & (dataset.episode_ids == episode))
        rows = rows[np.argsort(dataset.transition_indices[rows])]
        predicted = {
            name: float(dataset.features[rows[0], feature_lookup[name]])
            for name in state_names
        }
        previous_state = previous_controls = None
        last_errors = {}
        for row_index in rows:
            raw = dataset.features[row_index].copy()
            controls = {name: float(raw[feature_lookup[name]])
                        for name in control_names}
            context = _transition_context(
                predicted, previous_state, controls, previous_controls)
            for name, value in predicted.items():
                raw[feature_lookup[name]] = value
            for name, value in context.items():
                raw[feature_lookup[name]] = value
            normalized = (raw - feature_mean) / feature_scale
            batch = learner.shapes["x"][0]
            prediction = learner.forward(np.tile(normalized, (batch, 1)))[0]
            delta = prediction * target_scale + target_mean
            before = dict(predicted)
            for name, change in zip(target_names, delta):
                predicted[name] = predicted.get(name, 0.0) + float(change)
            actual = {
                name: float(dataset.features[row_index, feature_lookup[name]])
                for name in state_names
            }
            for name in target_names:
                actual[name] = actual.get(name, 0.0) + float(
                    dataset.targets[row_index, target_lookup[name]])
            errors = {name: predicted[name] - actual[name]
                      for name in target_names}
            normalized_errors = np.asarray([
                errors[name] / max(float(feature_scale[feature_lookup[name]]), 1e-12)
                for name in target_names
            ])
            mean_square = float(np.mean(normalized_errors * normalized_errors))
            squared_normalized.append(mean_square)
            regime = TRANSITION_REGIMES[int(dataset.regime_ids[row_index])]
            regime_squared[regime].append(mean_square)
            for name, error in errors.items():
                absolute.setdefault(name, []).append(abs(error))
            last_errors = errors
            previous_state, previous_controls = before, controls
        for name, error in last_errors.items():
            final_absolute.setdefault(name, []).append(abs(error))
    field_summary = {
        name: {
            "mean_absolute": float(np.mean(values)),
            "final_absolute": float(np.mean(final_absolute.get(name, values[-1:]))),
            "normalized_final": float(np.mean(final_absolute.get(name, values[-1:]))
                                      / max(float(feature_scale[feature_lookup[name]]), 1e-12)),
        }
        for name, values in absolute.items()
    }
    worst = sorted(field_summary, key=lambda name:
                   field_summary[name]["normalized_final"], reverse=True)[:12]
    return {
        "normalized_rmse": float(math.sqrt(np.mean(squared_normalized))),
        "by_regime_normalized_rmse": {
            name: (float(math.sqrt(np.mean(values))) if values else None)
            for name, values in regime_squared.items()
        },
        "key_state_error": {
            name: field_summary[name] for name in (
                "state.rpm", "state.current_torque_nm",
                "state.manifold_pressure_frac", "state.coolant_temp_k",
                "state.oil_temp_k") if name in field_summary
        },
        "largest_final_normalized_errors": {
            name: field_summary[name] for name in worst
        },
    }


def _prepare(samples_per_fuel: int, seed: int, engine_toy_path=None,
             episodes_per_fuel: int = 8,
             random_coverage_samples: int = 3_000,
             engine_identities: tuple[str, ...] | None = None,
             dataset_cache_dir: str | Path | None = None):
    cache_path = None
    if dataset_cache_dir is not None:
        _load_engine_toy(engine_toy_path)
        fingerprint = {
            "schema": 1, "samples_per_fuel": samples_per_fuel,
            "seed": seed, "episodes_per_fuel": episodes_per_fuel,
            "random_coverage_samples": random_coverage_samples,
            "engine_identities": engine_identities or ("all",),
        }
        digest = hashlib.sha256(json.dumps(
            fingerprint, sort_keys=True).encode("utf-8"))
        engine_toy_root = Path(
            sys.modules["engine_cycle_sim"].__file__).resolve().parent
        source_paths = (
            Path(__file__).resolve().with_name(
                "demo_perforated_multifuel_engine.py"),
            *sorted(engine_toy_root.glob("*.py")),
        )
        for source_path in source_paths:
            stat = source_path.stat()
            digest.update(f"{source_path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        cache_path = Path(dataset_cache_dir) / f"engine-union-{digest.hexdigest()[:20]}.npz"
    if cache_path is not None and cache_path.is_file():
        print(f"[dataset cache hit] {cache_path}", flush=True)
        dataset = load_engine_dataset(cache_path)
    else:
        print("[dataset capture] collecting real simulator transitions...", flush=True)
        dataset = make_engine_union_data(
            named_samples_per_fuel=samples_per_fuel, seed=seed,
            episodes_per_fuel=episodes_per_fuel,
            random_coverage_samples=random_coverage_samples,
            engine_identities=engine_identities,
            engine_toy_path=engine_toy_path)
        if cache_path is not None:
            save_engine_dataset(cache_path, dataset)
            print(f"[dataset cached] {cache_path}", flush=True)
    print(f"[dataset ready] {len(dataset.features)} transitions, "
          f"{len(dataset.engine_names)} engines, "
          f"{len(dataset.feature_names)} -> {len(dataset.target_names)}",
          flush=True)
    test = episode_holdout_mask(dataset, seed + 91)
    train = ~test
    x_train, x_test, x_mean, x_scale = _standardize(
        dataset.features[train], dataset.features[test], return_statistics=True)
    y_train, y_test, y_mean, y_scale = _standardize(
        dataset.targets[train], dataset.targets[test], range_floor=0.05,
        return_statistics=True)
    active = np.ones(dataset.targets.shape[1], dtype=bool)
    preprocessing = {
        "feature_names": list(dataset.feature_names),
        "target_names": [name for name, enabled in zip(dataset.target_names, active)
                         if enabled],
        "feature_mean": x_mean.reshape(-1).tolist(),
        "feature_scale": x_scale.reshape(-1).tolist(),
        "target_mean": y_mean[:, active].reshape(-1).tolist(),
        "target_scale": y_scale[:, active].reshape(-1).tolist(),
        "engine_names": list(dataset.engine_names),
    }
    return (x_train, y_train[:, active], x_test, y_test[:, active], dataset,
            active, dataset.episode_ids[train], preprocessing)


def _attach_engine_transition_contract(learner: CompiledPerforatedAdam,
                                       preprocessing: dict,
                                       episodes_per_fuel: int,
                                       random_coverage_samples: int) -> None:
    path = (learner.training_cycle.manifest_path
            if learner.training_cycle is not None
            else learner.compiled.manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["engine_transition"] = {
        "schema": "turing.multifuel-engine-transition",
        "version": 1,
        "physics_dt_seconds": 0.005,
        "prediction": "normalized-next-state-delta",
        "history": "current-state-plus-compact-one-transition-context",
        "batching": "distinct-episode-first",
        "episodes_per_fuel": episodes_per_fuel,
        "random_coverage_samples": random_coverage_samples,
        "engine_profiles": preprocessing["engine_names"],
        "state_update": "next_state[name] = current_state[name] + denormalized_prediction[name]",
        "machine_outputs": [name for name in MACHINE_OUTPUT_NAMES
                            if name in preprocessing["target_names"]],
        **preprocessing,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _pad_batch(values: np.ndarray, indices: np.ndarray, batch: int) -> np.ndarray:
    selected = values[indices]
    if len(selected) < batch:
        selected = np.concatenate((selected, np.repeat(selected[-1:], batch - len(selected), axis=0)))
    return np.ascontiguousarray(selected)


def _train_with_dropout(learner: CompiledPerforatedAdam, x: np.ndarray,
                        y: np.ndarray, rng: np.random.Generator,
                        feature_names, *, sample_weight=None,
                        information_dropout: float = 0.02,
                        branch_dropout: float = 0.02):
    """Train robustness to union-ABI absences and perforated branch loss."""
    if not 0.0 <= information_dropout < 1.0:
        raise ValueError("information_dropout must be in [0, 1)")
    if not 0.0 <= branch_dropout < 1.0:
        raise ValueError("branch_dropout must be in [0, 1)")
    x = np.asarray(x, dtype=np.float64).copy()
    if information_dropout > 0.0:
        protected = np.asarray([
            name.startswith(("control.", "fuel.", "engine_profile."))
            for name in feature_names
        ])
        missing = rng.random(x.shape) < information_dropout
        missing[:, protected] = False
        x[missing] = 0.0  # standardized mean is the missing-information value
    branches = learner.compiled.contract.dendrites_per_neuron
    out_dim = learner.shapes["base_bias"][1]
    mask = np.ones((1, out_dim * branches), dtype=np.float64)
    if branch_dropout > 0.0:
        mask = (rng.random(mask.shape) >= branch_dropout).astype(np.float64)
        per_output = mask.reshape(out_dim, branches)
        empty = np.flatnonzero(per_output.sum(axis=1) == 0.0)
        if empty.size:
            per_output[empty, rng.integers(0, branches, size=len(empty))] = 1.0
    learner.set_branch_authority(mask)
    try:
        return learner.step(np.ascontiguousarray(x), y,
                            sample_weight=sample_weight)
    finally:
        learner.set_branch_authority(np.ones_like(mask))


def _training_schedule(rng: np.random.Generator, episode_ids: np.ndarray,
                       batch_size: int, epochs: int, passes_per_epoch: int):
    if epochs < 1 or passes_per_epoch < 1:
        raise ValueError("epochs and passes_per_epoch must be positive")
    rows = np.arange(len(episode_ids), dtype=np.int64)
    schedule = []
    for epoch in range(epochs):
        epoch_batches = []
        for pass_index in range(passes_per_epoch):
            epoch_batches.extend(
                (pass_index, indices, weights)
                for indices, weights in episode_epoch_batches(
                    rng, rows, episode_ids, batch_size))
        for position, (pass_index, indices, weights) in enumerate(epoch_batches):
            schedule.append((epoch, pass_index, indices, weights,
                             position + 1 == len(epoch_batches)))
    return schedule


def _training_cycle_banks(
    x: np.ndarray,
    y: np.ndarray,
    schedule,
    rng: np.random.Generator,
    feature_names,
    *,
    batch_size: int,
    branches: int,
    information_dropout: float,
    branch_dropout: float,
):
    """Materialize the changing inputs consumed by one native LLVM cycle."""
    if not 0.0 <= information_dropout < 1.0:
        raise ValueError("information_dropout must be in [0, 1)")
    if not 0.0 <= branch_dropout < 1.0:
        raise ValueError("branch_dropout must be in [0, 1)")
    protected = np.asarray([
        name.startswith(("control.", "fuel.", "engine_profile."))
        for name in feature_names
    ])
    x_bank, y_bank, weight_bank, mask_bank = [], [], [], []
    width = y.shape[1] * branches
    for _epoch, _pass_index, indices, weights, _epoch_end in schedule:
        batch_x = _pad_batch(x, indices, batch_size).copy()
        if information_dropout > 0.0:
            missing = rng.random(batch_x.shape) < information_dropout
            missing[:, protected] = False
            batch_x[missing] = 0.0
        mask = np.ones((1, width), dtype=np.float64)
        if branch_dropout > 0.0:
            mask = (rng.random(mask.shape) >= branch_dropout).astype(np.float64)
            per_output = mask.reshape(y.shape[1], branches)
            empty = np.flatnonzero(per_output.sum(axis=1) == 0.0)
            if empty.size:
                per_output[empty, rng.integers(
                    0, branches, size=len(empty))] = 1.0
        x_bank.append(batch_x)
        y_bank.append(_pad_batch(y, indices, batch_size))
        weight_bank.append(np.asarray(weights, dtype=np.float64).reshape(batch_size, 1))
        mask_bank.append(mask)
    return tuple(np.ascontiguousarray(np.stack(bank)) for bank in (
        x_bank, y_bank, weight_bank, mask_bank))


def _capture_refresh_training_pool(
    preprocessing: dict,
    active: np.ndarray,
    *,
    seed: int,
    named_samples_per_fuel: int,
    episodes_per_fuel: int,
    random_coverage_samples: int,
    engine_toy_path=None,
):
    """Capture fresh real-simulator programs and normalize to the fixed ABI."""
    fresh = make_engine_union_data(
        named_samples_per_fuel=named_samples_per_fuel,
        seed=seed,
        episodes_per_fuel=episodes_per_fuel,
        random_coverage_samples=random_coverage_samples,
        engine_identities=tuple(preprocessing["engine_names"]),
        engine_toy_path=engine_toy_path,
    )
    if list(fresh.feature_names) != preprocessing["feature_names"]:
        raise RuntimeError("refreshed simulator features changed the union ABI")
    selected_targets = [
        name for name, enabled in zip(fresh.target_names, active) if enabled
    ]
    if selected_targets != preprocessing["target_names"]:
        raise RuntimeError("refreshed simulator targets changed the union ABI")
    feature_mean = np.asarray(preprocessing["feature_mean"])
    feature_scale = np.asarray(preprocessing["feature_scale"])
    target_mean = np.asarray(preprocessing["target_mean"])
    target_scale = np.asarray(preprocessing["target_scale"])
    x = (np.asarray(fresh.features) - feature_mean) / feature_scale
    y = (np.asarray(fresh.targets)[:, active] - target_mean) / target_scale
    return (np.ascontiguousarray(x), np.ascontiguousarray(y),
            np.asarray(fresh.episode_ids, dtype=np.int64))


def _network_geometry(learner: CompiledPerforatedAdam, width: int, height: int):
    activity = learner.dendrite_activity()
    normalized = activity / max(float(np.percentile(activity, 95)), 1e-12)
    normalized = np.clip(normalized, 0.0, 1.5)
    branches = learner.compiled.contract.dendrites_per_neuron
    out_dim = learner.shapes["base_bias"][1]
    center = np.asarray((width * 0.44, height * 0.50))
    outer = min(width * 0.33, height * 0.38)
    hubs, dots = [], []
    for output in range(out_dim):
        angle = 2.0 * math.pi * output / out_dim - math.pi / 2.0
        hub = center + outer * np.asarray((math.cos(angle), math.sin(angle)))
        hubs.append(hub)
        for branch in range(branches):
            index = output * branches + branch
            tangent = np.asarray((-math.sin(angle), math.cos(angle)))
            radial = np.asarray((math.cos(angle), math.sin(angle)))
            spread = (branch - (branches - 1) / 2.0) * 14.0
            inward = 55.0 + 80.0 * min(normalized[index], 1.0)
            dot = hub - radial * inward + tangent * spread
            dots.append((dot, hub, float(normalized[index])))
    return np.asarray(hubs), dots


def _draw_headless(path: Path, learner: CompiledPerforatedAdam,
                   history: list[float], metrics: dict) -> None:
    from PIL import Image, ImageDraw, ImageFont
    width, height = 1280, 720
    image = Image.new("RGB", (width, height), (10, 13, 19))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    hubs, dots = _network_geometry(learner, width, height)
    for dot, hub, strength in dots:
        alpha = int(25 + 90 * min(strength, 1.0))
        draw.line((*dot, *hub), fill=(58, 95, 126, alpha), width=1)
    for dot, _hub, strength in dots:
        radius = 2.0 + 5.0 * min(strength, 1.0)
        color = (65, int(145 + 70 * min(strength, 1.0)), 255, 220)
        draw.ellipse((dot[0] - radius, dot[1] - radius,
                      dot[0] + radius, dot[1] + radius), fill=color)
    for hub in hubs:
        draw.ellipse((hub[0] - 2, hub[1] - 2, hub[0] + 2, hub[1] + 2),
                     fill=(255, 173, 75, 210))
    draw.text((24, 20), "COMPILED DENDRITE ENGINE LEARNER", font=font,
              fill=(230, 239, 248, 255))
    runtime = metrics.get("runtime_performance", {})
    labels = (
        f"step {metrics['step']} / {metrics['steps']}",
        f"epoch {metrics['epoch']} / {metrics['epochs']}  "
        f"pass {metrics['pass_in_epoch']} / {metrics['passes_per_epoch']}",
        f"train loss  {metrics['loss']:.6g}",
        f"gradient L2 {metrics['gradient_norm']:.5g}",
        f"neural-weight L2 {metrics['parameter_norm']:.5g}",
        f"best validation {metrics['best_validation_loss']:.6g}",
        f"compiled batch-1 {runtime.get('compiled_mean_ms', 0.0):.3f} ms  "
        f"{runtime.get('compiled_transitions_per_second', 0.0):.1f} transitions/s",
        f"end-to-end {runtime.get('end_to_end_mean_ms', 0.0):.3f} ms  "
        f"realtime x{runtime.get('realtime_factor', 0.0):.2f}",
        f"engine {ENGINE_IDENTITY}",
        f"fuel {metrics['fuel']}",
        ("network rollout DIVERGED (guarded)"
         if metrics.get("network_rollout_diverged") else
         f"load {metrics['load']:.0%}   rpm {metrics['rpm']:.1f}"),
        "LLVM forward + ProcessGraph VJP / Adam",
    )
    for index, label in enumerate(labels):
        draw.text((24, 62 + index * 23), label, font=small,
                  fill=(184, 201, 216, 255))
    x0, y0, x1, y1 = 860, 490, 1245, 675
    draw.rectangle((x0, y0, x1, y1), outline=(65, 78, 92, 255), width=1)
    if len(history) > 1:
        logged_raw = np.log10(np.maximum(history, 1e-12))
        # Thousands of stochastic minibatch points otherwise form a solid
        # orange block. Bin to the pixel budget and plot the median tendency.
        chunks = np.array_split(logged_raw, min(240, len(logged_raw)))
        logged = np.asarray([np.median(chunk) for chunk in chunks])
        validation = np.asarray(metrics.get("validation_history", ()), dtype=float)
        validation_log = (np.log10(np.maximum(validation[:, 1], 1e-12))
                          if validation.size else np.zeros(0))
        domain = np.concatenate((logged, validation_log))
        low, high = float(domain.min()), float(domain.max())
        span = max(high - low, 1e-9)
        points = [(x0 + i * (x1 - x0) / (len(logged) - 1),
                   y1 - (value - low) / span * (y1 - y0))
                  for i, value in enumerate(logged)]
        draw.line(points, fill=(255, 173, 75, 255), width=2)
        if validation.size:
            validation_points = [
                (x0 + (step / metrics["steps"]) * (x1 - x0),
                 y1 - (value - low) / span * (y1 - y0))
                for (step, _), value in zip(validation, validation_log)
            ]
            draw.line(validation_points, fill=(89, 214, 151, 255), width=2)
    draw.text((x0, y0 - 22), "loss, log scale: train / validation", font=small,
              fill=(184, 201, 216, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _render_audio_frame(audio_state, synth, n_frames: int):
    snap = audio_state.snapshot()
    (engine, rpm, throttle, load_frac, knock_active, knock_intensity,
     misfire_active, boost_frac, turbo_spool_frac, wastegate_flutter,
     surge_active, backfire_active, backfire_kind, backfire_strength,
     real_fire_hz, intake_flow_demand_frac, knock_ring_hz, exhaust_temp_k,
     preignition_active, preignition_intensity, compression_brake_active,
     extras) = snap
    return synth.render_stereo(
        engine, rpm, throttle, load_frac, n_frames,
        knock_active=knock_active, knock_intensity=knock_intensity,
        misfire_active=misfire_active, boost_frac=boost_frac,
        turbo_spool_frac=turbo_spool_frac,
        wastegate_flutter=wastegate_flutter, surge_active=surge_active,
        backfire_active=backfire_active, backfire_kind=backfire_kind,
        backfire_strength=backfire_strength, real_fire_hz=real_fire_hz,
        intake_flow_demand_frac=intake_flow_demand_frac,
        knock_ring_hz=knock_ring_hz, exhaust_temp_k=exhaust_temp_k,
        preignition_active=preignition_active,
        preignition_intensity=preignition_intensity,
        compression_brake_active=compression_brake_active, **extras)


def _write_wav(path: Path, left: list[np.ndarray], right: list[np.ndarray],
               sample_rate: int) -> None:
    stereo = np.column_stack((np.concatenate(left), np.concatenate(right)))
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def _drive_engine(sim, step: int, steps: int):
    engine = sim.engine
    phase = min(len(LOAD_LEVELS) - 1, step * len(LOAD_LEVELS) // max(steps, 1))
    load = LOAD_LEVELS[phase]
    fuel_index = min(len(FUEL_PROFILES) - 1,
                     step * len(FUEL_PROFILES) // max(steps, 1))
    sim.fuel_choice = FUEL_PROFILES[fuel_index]
    idle_row = load <= LOAD_LEVELS[2] and step % 3 != 2
    sim.throttle = 0.0 if idle_row else min(1.0, 0.10 + 0.72 * load)
    idle_power = engine.peak_torque_nm * engine.idle_rpm * 2.0 * math.pi / 60.0
    sim.electrical_load_frac = min(1.0, 0.15 + 0.75 * load)
    sim.gear_index = len(engine.transmission.gear_ratios)
    ratio = sim._current_gear_ratio()
    if sim._load_omega == 0.0 and ratio > 0.0:
        sim._load_omega = sim._omega / ratio
    sim.clutch_frac = min(1.0, (step + 1) / 30.0)
    _station_shaft_load(sim, load * idle_power * sim.clutch_frac)
    return load


def _validation_loss(learner: CompiledPerforatedAdam, x: np.ndarray,
                     y: np.ndarray, batch_size: int) -> float:
    squared_error = 0.0
    count = 0
    for start in range(0, len(x), batch_size):
        stop = min(len(x), start + batch_size)
        indices = np.arange(start, stop)
        prediction = learner.forward(_pad_batch(x, indices, batch_size))
        error = prediction[:stop - start] - y[start:stop]
        squared_error += float(np.sum(error * error))
        count += error.size
    return squared_error / max(count, 1)


def run_headless(*, output_dir: str | Path, epochs: int = DEFAULT_EPOCHS,
                 passes_per_epoch: int = DEFAULT_PASSES_PER_EPOCH,
                 samples_per_fuel: int = DEFAULT_SAMPLES_PER_FUEL,
                 episodes_per_fuel: int = DEFAULT_EPISODES_PER_FUEL,
                 random_coverage_samples: int = DEFAULT_RANDOM_COVERAGE_SAMPLES,
                 engine_identities: tuple[str, ...] | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 branches: int = DEFAULT_BRANCHES, seed: int = 1729,
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 audio: bool = True,
                 gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
                 max_global_gradient_norm: float = DEFAULT_MAX_GLOBAL_GRADIENT_NORM,
                 adaptive_sample_refresh: bool = DEFAULT_ADAPTIVE_SAMPLE_REFRESH,
                 refresh_gap_growth: float = DEFAULT_REFRESH_GAP_GROWTH,
                 refresh_named_samples_per_fuel: int = DEFAULT_REFRESH_NAMED_SAMPLES_PER_FUEL,
                 refresh_random_coverage_samples: int = DEFAULT_REFRESH_RANDOM_COVERAGE_SAMPLES,
                 max_sample_refreshes: int = DEFAULT_MAX_SAMPLE_REFRESHES,
                 information_dropout: float = 0.02,
                 branch_dropout: float = 0.02,
                 use_cache: bool = True,
                 engine_toy_path=None,
                 max_audio_seconds: float = 20.0) -> LiveRunArtifacts:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (x_train, y_train, x_test, y_test, dataset, active,
     train_episode_ids, preprocessing) = _prepare(
        samples_per_fuel, seed, engine_toy_path, episodes_per_fuel,
        random_coverage_samples, engine_identities,
        output_dir / "dataset-cache" if use_cache else None)
    rng = np.random.default_rng(seed + 7)
    bank_schedule = _training_schedule(
        rng, train_episode_ids, batch_size, 1, 1)
    cycle_length = len(bank_schedule)
    steps = cycle_length * epochs * passes_per_epoch
    print("[LLVM] preparing training artifact...", flush=True)
    learner = CompiledPerforatedAdam.compile(
        output_dir / "compiled", batch=batch_size, in_dim=x_train.shape[1],
        out_dim=y_train.shape[1], dendrites_per_neuron=branches,
        seed=seed, learning_rate=learning_rate, cycle_length=cycle_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_global_gradient_norm=max_global_gradient_norm,
        use_cache=use_cache)
    print(f"[LLVM {'cache hit' if learner.training_cycle.cache_hit else 'compiled'}] "
          f"{cycle_length}-batch bank / {steps}-motion training cycle ready",
          flush=True)
    _attach_engine_transition_contract(
        learner, preprocessing, episodes_per_fuel, random_coverage_samples)
    print("[LLVM] preparing batch-1 rollout artifact...", flush=True)
    rollout_learner = CompiledPerforatedAdam.compile(
        output_dir / "compiled-rollout", batch=1, in_dim=x_train.shape[1],
        out_dim=y_train.shape[1], dendrites_per_neuron=branches,
        seed=seed, learning_rate=learning_rate, use_cache=use_cache)
    print(f"[LLVM {'cache hit' if rollout_learner.compiled.forward.key == 'cached' else 'compiled'}] "
          "batch-1 rollout ready", flush=True)
    _attach_engine_transition_contract(
        rollout_learner, preprocessing, episodes_per_fuel,
        random_coverage_samples)
    _copy_network_parameters(learner, rollout_learner)
    rollout = NetworkEngineRollout(
        rollout_learner, preprocessing, engine_toy_path,
        engine_identity=preprocessing["engine_names"][0])
    engine_root = str(Path(sys.modules["engine_cycle_sim"].__file__).parent)
    if engine_root not in sys.path:
        sys.path.insert(0, engine_root)
    from audio_stream import LiveAudioState
    from engine_sound import BLOCK_SIZE, SAMPLE_RATE, EngineSoundSynth
    audio_state = LiveAudioState(rollout.engine)
    synth = EngineSoundSynth(SAMPLE_RATE)
    left_blocks, right_blocks = [], []
    history: list[float] = []
    validation_history = []
    initial_validation = _validation_loss(
        learner, x_test, y_test, batch_size)
    pool_x = x_train
    pool_y = y_train
    pool_episode_ids = np.asarray(train_episode_ids, dtype=np.int64)
    initial_training_loss = _validation_loss(
        learner, pool_x, pool_y, batch_size)
    previous_training_loss = initial_training_loss
    previous_gap = initial_validation - initial_training_loss
    best_validation = initial_validation
    best_step = 0
    best_parameters = {name: value.copy()
                       for name, value in learner.parameters.items()}
    audio_stride = max(1, int(math.ceil(
        steps / max(max_audio_seconds * 20.0, 1.0))))
    banks = _training_cycle_banks(
        x_train, y_train, bank_schedule, rng, preprocessing["feature_names"],
        batch_size=batch_size, branches=branches,
        information_dropout=information_dropout,
        branch_dropout=branch_dropout)
    history = [initial_training_loss]
    training_seconds = 0.0
    sample_refreshes = []
    result = None
    final_validation = initial_validation
    for epoch_index in range(epochs):
        epoch_steps = cycle_length * passes_per_epoch
        print(f"[LLVM] epoch {epoch_index + 1}/{epochs}: executing "
              f"{epoch_steps} motions...", flush=True)
        training_started = time.perf_counter()
        result, native_history = learner.run_cycle(*banks, steps=epoch_steps)
        training_seconds += time.perf_counter() - training_started
        history.extend(native_history.tolist())
        training_loss = _validation_loss(
            learner, pool_x, pool_y, batch_size)
        final_validation = _validation_loss(
            learner, x_test, y_test, batch_size)
        completed_steps = (epoch_index + 1) * epoch_steps
        validation_history.append((completed_steps, final_validation))
        if final_validation < best_validation:
            best_validation = final_validation
            best_step = completed_steps
            best_parameters = {name: value.copy()
                               for name, value in learner.parameters.items()}
        gap = final_validation - training_loss
        gap_growth = gap - previous_gap
        should_refresh = (
            adaptive_sample_refresh
            and len(sample_refreshes) < max_sample_refreshes
            and training_loss < previous_training_loss
            and training_loss < final_validation
            and gap_growth >= refresh_gap_growth
        )
        if should_refresh:
            refresh_index = len(sample_refreshes) + 1
            scale = refresh_index
            print(f"[sample refresh {refresh_index}] validation gap grew by "
                  f"{gap_growth:.5g}; capturing new simulator programs...",
                  flush=True)
            fresh_x, fresh_y, fresh_episodes = _capture_refresh_training_pool(
                preprocessing, active,
                seed=seed + 10_000 * refresh_index,
                named_samples_per_fuel=(
                    refresh_named_samples_per_fuel * scale),
                episodes_per_fuel=episodes_per_fuel,
                random_coverage_samples=(
                    refresh_random_coverage_samples * scale),
                engine_toy_path=engine_toy_path,
            )
            episode_offset = int(pool_episode_ids.max(initial=-1)) + 1
            pool_x = np.concatenate((pool_x, fresh_x), axis=0)
            pool_y = np.concatenate((pool_y, fresh_y), axis=0)
            pool_episode_ids = np.concatenate((
                pool_episode_ids, fresh_episodes + episode_offset))
            refreshed_schedule = _training_schedule(
                rng, pool_episode_ids, batch_size, 1, 1)[:cycle_length]
            banks = _training_cycle_banks(
                pool_x, pool_y, refreshed_schedule, rng,
                preprocessing["feature_names"], batch_size=batch_size,
                branches=branches, information_dropout=information_dropout,
                branch_dropout=branch_dropout)
            sample_refreshes.append({
                "epoch": epoch_index + 1,
                "gap_growth": gap_growth,
                "new_samples": len(fresh_x),
                "pool_samples": len(pool_x),
            })
        previous_training_loss = training_loss
        previous_gap = gap
    print(f"[LLVM] {epochs} native epochs completed in "
          f"{training_seconds:.3f}s", flush=True)
    assert result is not None
    _copy_network_parameters(learner, rollout_learner)
    for step in range(steps):
        throttle = float(np.clip(
            0.5 + 0.46 * math.sin(2.0 * math.pi * step / max(steps, 1)),
            0.0, 1.0))
        for _ in range(10):
            rollout.step(throttle)
        audio_state.set_from_sim(rollout)
        if audio and step % audio_stride == 0:
            # Headless rendering has no wall clock to keep the sound alive.
            # Give every visual/training step 1/20 second of phase-continuous
            # synthesis, matching the cadence of a readable live display.
            audio_frames = max(BLOCK_SIZE, SAMPLE_RATE // 20)
            left, right = _render_audio_frame(audio_state, synth, audio_frames)
            left_blocks.append(left)
            right_blocks.append(right)
        if ((step + 1) % (cycle_length * passes_per_epoch) == 0
                and step + 1 < steps):
            rollout.reset()
    metrics = {
        "step": steps, "steps": steps, "loss": result.loss,
        "epoch": epochs, "epochs": epochs,
        "pass_in_epoch": passes_per_epoch,
        "passes_per_epoch": passes_per_epoch,
        "gradient_norm": result.gradient_norm,
        "parameter_norm": result.parameter_norm,
        "fuel": FUEL_PROFILES[0], "load": 0.0, "rpm": rollout.rpm,
        "throttle": throttle,
        "network_rollout_diverged": rollout.diverged,
        "validation_loss": final_validation,
        "best_validation_loss": best_validation,
        "best_step": best_step,
        "learning_rate": learner.learning_rate,
        "native_training_seconds": training_seconds,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_global_gradient_norm": max_global_gradient_norm,
        "adaptive_sample_refresh": adaptive_sample_refresh,
        "sample_refreshes": sample_refreshes,
        "final_sample_pool_size": len(pool_x),
    }
    for name, value in best_parameters.items():
        learner.parameters[name][...] = value
    _copy_network_parameters(learner, rollout_learner)
    metrics["runtime_performance"] = rollout.performance_snapshot()
    metrics["parameter_norm"] = sum(
        float(np.sum(value * value)) for value in learner.parameters.values()
    ) ** 0.5
    metrics["validation_history"] = validation_history
    image_path = output_dir / "engine-dendrite-training.png"
    _draw_headless(image_path, learner, history, metrics)
    audio_path = output_dir / "engine-dendrite-training.wav" if audio else None
    if audio_path is not None:
        _write_wav(audio_path, left_blocks, right_blocks, SAMPLE_RATE)
    metrics_path = output_dir / "metrics.json"
    weights_path = output_dir / "best-compiled-learner-weights.npz"
    np.savez(
        weights_path, **learner.parameters, dendrite_route=learner.route,
        feature_mean=np.asarray(preprocessing["feature_mean"]),
        feature_scale=np.asarray(preprocessing["feature_scale"]),
        target_mean=np.asarray(preprocessing["target_mean"]),
        target_scale=np.asarray(preprocessing["target_scale"]),
        feature_names=np.asarray(preprocessing["feature_names"]),
        target_names=np.asarray(preprocessing["target_names"]),
    )
    metrics_path.write_text(json.dumps({
        **metrics,
        "initial_loss": history[0], "final_loss": history[-1],
        "loss_history": history,
        "validation_history": validation_history,
        "initial_validation_loss": initial_validation,
        "best_validation_loss": best_validation,
        "best_step": best_step,
        "feature_count": int(dataset.features.shape[1]),
        "episode_count": int(np.unique(dataset.episode_ids).size),
        "train_episode_count": int(np.unique(train_episode_ids).size),
        "moving_output_count": int(active.sum()),
        "compiled_contract": str(learner.training_cycle.manifest_path.resolve()),
        "audio": str(audio_path) if audio_path else None,
        "image": str(image_path),
        "weights": str(weights_path),
    }, indent=2) + "\n", encoding="utf-8")
    return LiveRunArtifacts(
        image_path, audio_path, learner.training_cycle.manifest_path.resolve(),
        metrics_path, history[0], history[-1], best_validation)


def run_interactive(*, output_dir: str | Path, epochs: int = DEFAULT_EPOCHS,
                    passes_per_epoch: int = DEFAULT_PASSES_PER_EPOCH,
                    samples_per_fuel: int = DEFAULT_SAMPLES_PER_FUEL,
                    episodes_per_fuel: int = DEFAULT_EPISODES_PER_FUEL,
                    random_coverage_samples: int = DEFAULT_RANDOM_COVERAGE_SAMPLES,
                    engine_identities: tuple[str, ...] | None = None,
                    batch_size: int = DEFAULT_BATCH_SIZE,
                    branches: int = DEFAULT_BRANCHES, seed: int = 1729,
                    learning_rate: float = DEFAULT_LEARNING_RATE,
                    audio: bool = True,
                    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
                    max_global_gradient_norm: float = DEFAULT_MAX_GLOBAL_GRADIENT_NORM,
                    adaptive_sample_refresh: bool = DEFAULT_ADAPTIVE_SAMPLE_REFRESH,
                    refresh_gap_growth: float = DEFAULT_REFRESH_GAP_GROWTH,
                    refresh_named_samples_per_fuel: int = DEFAULT_REFRESH_NAMED_SAMPLES_PER_FUEL,
                    refresh_random_coverage_samples: int = DEFAULT_REFRESH_RANDOM_COVERAGE_SAMPLES,
                    max_sample_refreshes: int = DEFAULT_MAX_SAMPLE_REFRESHES,
                    information_dropout: float = 0.02,
                    branch_dropout: float = 0.02,
                    use_cache: bool = True,
                    engine_toy_path=None,
                    shadow_correction_seconds: float = 2.0) -> None:
    """Open the real OpenGL/audio display; Escape or window close exits."""
    import pygame
    from OpenGL.GL import (
        GL_BLEND, GL_COLOR_BUFFER_BIT, GL_LINES, GL_ONE_MINUS_SRC_ALPHA,
        GL_POINTS, GL_PROJECTION, GL_MODELVIEW, GL_RGBA, GL_SRC_ALPHA,
        GL_UNSIGNED_BYTE, glBegin, glBlendFunc, glClear, glClearColor,
        glColor4f, glDrawPixels, glEnable, glEnd, glLoadIdentity,
        glMatrixMode, glOrtho, glPointSize, glVertex2f, glViewport,
        glWindowPos2d,
    )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (x_train, y_train, x_test, y_test, dataset, active,
     train_episode_ids, preprocessing) = _prepare(
        samples_per_fuel, seed, engine_toy_path, episodes_per_fuel,
        random_coverage_samples, engine_identities,
        output_dir / "dataset-cache" if use_cache else None)
    rng = np.random.default_rng(seed + 7)
    bank_schedule = _training_schedule(
        rng, train_episode_ids, batch_size, 1, 1)
    cycle_length = len(bank_schedule)
    steps = cycle_length * epochs * passes_per_epoch
    print("[LLVM] preparing training artifact...", flush=True)
    learner = CompiledPerforatedAdam.compile(
        output_dir / "compiled", batch=batch_size, in_dim=x_train.shape[1],
        out_dim=y_train.shape[1], dendrites_per_neuron=branches,
        seed=seed, learning_rate=learning_rate, cycle_length=cycle_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_global_gradient_norm=max_global_gradient_norm,
        use_cache=use_cache)
    print(f"[LLVM {'cache hit' if learner.training_cycle.cache_hit else 'compiled'}] "
          f"{cycle_length}-batch bank / {steps}-motion training cycle ready",
          flush=True)
    _attach_engine_transition_contract(
        learner, preprocessing, episodes_per_fuel, random_coverage_samples)
    print("[LLVM] preparing batch-1 rollout artifact...", flush=True)
    rollout_learner = CompiledPerforatedAdam.compile(
        output_dir / "compiled-rollout", batch=1, in_dim=x_train.shape[1],
        out_dim=y_train.shape[1], dendrites_per_neuron=branches,
        seed=seed, learning_rate=learning_rate, use_cache=use_cache)
    print(f"[LLVM {'cache hit' if rollout_learner.compiled.forward.key == 'cached' else 'compiled'}] "
          "batch-1 rollout ready", flush=True)
    _attach_engine_transition_contract(
        rollout_learner, preprocessing, episodes_per_fuel,
        random_coverage_samples)
    _copy_network_parameters(learner, rollout_learner)
    rollout = NetworkEngineRollout(
        rollout_learner, preprocessing, engine_toy_path,
        engine_identity=preprocessing["engine_names"][0])
    shadow = LiveShadowTeacher(preprocessing, engine_toy_path)
    from audio_stream import AudioStreamer, LiveAudioState
    from engine_sound import BLOCK_SIZE, SAMPLE_RATE
    audio_state = LiveAudioState(rollout.engine)
    audio_state.set_from_sim(rollout)
    streamer = stream = None
    if audio:
        import sounddevice as sd
        streamer = AudioStreamer(audio_state, SAMPLE_RATE)
        streamer.start()
        def callback(outdata, frames, _time, _status):
            left, right = streamer.pull(frames)
            outdata[:, 0], outdata[:, 1] = left, right
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2,
                                 dtype="float32", blocksize=BLOCK_SIZE,
                                 callback=callback)
        stream.start()
    width, height = 1280, 720
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption("Compiled dendrite engine learner")
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()
    training_banks = _training_cycle_banks(
        x_train, y_train, bank_schedule, rng, preprocessing["feature_names"],
        batch_size=batch_size, branches=branches,
        information_dropout=information_dropout,
        branch_dropout=branch_dropout)
    initial_training_loss = _validation_loss(
        learner, x_train, y_train, batch_size)
    initial_validation = _validation_loss(
        learner, x_test, y_test, batch_size)
    adaptive_state = {
        "pool_x": x_train,
        "pool_y": y_train,
        "pool_episode_ids": np.asarray(train_episode_ids, dtype=np.int64),
        "banks": training_banks,
        "previous_training_loss": initial_training_loss,
        "previous_gap": initial_validation - initial_training_loss,
        "sample_refreshes": [],
    }
    history = []
    step = 0
    running = True
    result = None
    live_result = None
    compatible_fuels = tuple(rollout.engine.fuel_compatibility)
    command = LiveEngineControls(
        fuel_choice=(compatible_fuels[0] if compatible_fuels else "none"))
    replay_x: deque[np.ndarray] = deque(maxlen=8192)
    replay_y: deque[np.ndarray] = deque(maxlen=8192)
    correction_count = 0
    live_drift = shadow.drift_from(rollout)
    latest_shadow_values = physical_state_mapping(shadow.sim.state)
    epochs_submitted = 0
    epochs_completed = 0
    training_complete = False
    epoch = pass_index = 0
    scheduler_drops = 0
    tick_accumulator = 0.0
    last_correction_time = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=1,
                                  thread_name_prefix="engine-teacher")
    future = None
    pending_engine_identity = None

    def background_update(epoch_index: int | None,
                          command_snapshot: LiveEngineControls):
        offline_result = None
        offline_history = None
        metadata = None
        if epoch_index is not None:
            offline_result, offline_history = learner.run_cycle(
                *adaptive_state["banks"],
                steps=cycle_length * passes_per_epoch)
            training_loss = _validation_loss(
                learner, adaptive_state["pool_x"],
                adaptive_state["pool_y"], batch_size)
            validation_loss = _validation_loss(
                learner, x_test, y_test, batch_size)
            gap = validation_loss - training_loss
            gap_growth = gap - adaptive_state["previous_gap"]
            should_refresh = (
                adaptive_sample_refresh
                and len(adaptive_state["sample_refreshes"])
                    < max_sample_refreshes
                and training_loss < adaptive_state["previous_training_loss"]
                and training_loss < validation_loss
                and gap_growth >= refresh_gap_growth
            )
            if should_refresh:
                refresh_index = len(adaptive_state["sample_refreshes"]) + 1
                fresh_x, fresh_y, fresh_episodes = _capture_refresh_training_pool(
                    preprocessing, active,
                    seed=seed + 10_000 * refresh_index,
                    named_samples_per_fuel=(
                        refresh_named_samples_per_fuel * refresh_index),
                    episodes_per_fuel=episodes_per_fuel,
                    random_coverage_samples=(
                        refresh_random_coverage_samples * refresh_index),
                    engine_toy_path=engine_toy_path,
                )
                offset = int(adaptive_state["pool_episode_ids"].max(
                    initial=-1)) + 1
                adaptive_state["pool_x"] = np.concatenate((
                    adaptive_state["pool_x"], fresh_x), axis=0)
                adaptive_state["pool_y"] = np.concatenate((
                    adaptive_state["pool_y"], fresh_y), axis=0)
                adaptive_state["pool_episode_ids"] = np.concatenate((
                    adaptive_state["pool_episode_ids"],
                    fresh_episodes + offset))
                refreshed_schedule = _training_schedule(
                    rng, adaptive_state["pool_episode_ids"],
                    batch_size, 1, 1)[:cycle_length]
                adaptive_state["banks"] = _training_cycle_banks(
                    adaptive_state["pool_x"], adaptive_state["pool_y"],
                    refreshed_schedule, rng, preprocessing["feature_names"],
                    batch_size=batch_size, branches=branches,
                    information_dropout=information_dropout,
                    branch_dropout=branch_dropout)
                adaptive_state["sample_refreshes"].append({
                    "epoch": epoch_index + 1,
                    "gap_growth": gap_growth,
                    "new_samples": len(fresh_x),
                    "pool_samples": len(adaptive_state["pool_x"]),
                })
            adaptive_state["previous_training_loss"] = training_loss
            adaptive_state["previous_gap"] = gap
            metadata = (epoch_index, passes_per_epoch - 1, True)
        live_x, live_y = shadow.step(command_snapshot)
        replay_x.append(live_x)
        replay_y.append(live_y)
        replay_result = None
        if len(replay_x) >= batch_size:
            chosen = rng.choice(len(replay_x), size=batch_size, replace=False)
            adaptive_state["banks"][0][0] = np.stack(
                [replay_x[i] for i in chosen])
            adaptive_state["banks"][1][0] = np.stack(
                [replay_y[i] for i in chosen])
            adaptive_state["banks"][2][0].fill(1.0)
            replay_result, _replay_history = learner.run_cycle(
                *adaptive_state["banks"], steps=1)
        parameters = {name: value.copy()
                      for name, value in learner.parameters.items()}
        return {
            "offline_result": offline_result,
            "offline_history": offline_history,
            "replay_result": replay_result,
            "metadata": metadata,
            "parameters": parameters,
            "shadow_values": physical_state_mapping(shadow.sim.state),
        }
    try:
        while running:
            frame_seconds = clock.tick(60) / 1000.0
            tick_accumulator += min(frame_seconds, 0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_k:
                    command.ignition_enabled = not command.ignition_enabled
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_PERIOD:
                    command.gear_index = min(
                        len(rollout.engine.transmission.gear_ratios),
                        command.gear_index + 1)
                    command.clutch_frac = 0.0
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_COMMA:
                    command.gear_index = max(-1, command.gear_index - 1)
                    command.clutch_frac = 0.0
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    fuels = tuple(rollout.engine.fuel_compatibility)
                    if fuels:
                        fuel_index = (fuels.index(command.fuel_choice)
                                      if command.fuel_choice in fuels else -1)
                        command.fuel_choice = fuels[(fuel_index + 1) % len(fuels)]
                elif (event.type == pygame.KEYDOWN
                      and event.key in (pygame.K_LEFTBRACKET,
                                        pygame.K_RIGHTBRACKET)):
                    profiles = rollout.engine_names
                    profile_index = profiles.index(rollout.engine_identity)
                    direction = (-1 if event.key == pygame.K_LEFTBRACKET else 1)
                    pending_engine_identity = profiles[
                        (profile_index + direction) % len(profiles)]
            if future is not None and future.done():
                update = future.result()
                future = None
                for name, value in update["parameters"].items():
                    rollout_learner.parameters[name][...] = value
                latest_shadow_values = update["shadow_values"]
                if update["offline_result"] is not None:
                    result = update["offline_result"]
                    history.extend(update["offline_history"].tolist())
                    step += cycle_length * passes_per_epoch
                    epochs_completed += 1
                    training_complete = epochs_completed >= epochs
                if update["replay_result"] is not None:
                    live_result = update["replay_result"]
                if update["metadata"] is not None:
                    epoch, pass_index, _epoch_end = update["metadata"]
            if pending_engine_identity is not None and future is None:
                rollout.set_engine(pending_engine_identity)
                shadow.set_engine(pending_engine_identity)
                fuels = tuple(rollout.engine.fuel_compatibility)
                command = LiveEngineControls(
                    fuel_choice=(fuels[0] if fuels else "none"))
                replay_x.clear(); replay_y.clear()
                latest_shadow_values = physical_state_mapping(shadow.sim.state)
                last_correction_time = time.monotonic()
                pending_engine_identity = None
            if future is None:
                epoch_to_run = None
                if epochs_submitted < epochs:
                    epoch_to_run = epochs_submitted
                    epochs_submitted += 1
                future = executor.submit(
                    background_update, epoch_to_run, replace(command))
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                command.throttle += 0.035
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                command.throttle -= 0.035
            if keys[pygame.K_SPACE]:
                command.throttle = 0.0
            if keys[pygame.K_e]:
                command.brake_load_nm += rollout.engine.peak_torque_nm * 0.025
            if keys[pygame.K_d]:
                command.brake_load_nm -= rollout.engine.peak_torque_nm * 0.025
            if keys[pygame.K_x]:
                command.electrical_load_frac += 0.04
            if keys[pygame.K_z]:
                command.electrical_load_frac -= 0.04
            if keys[pygame.K_v]:
                command.clutch_frac += 0.08
            if keys[pygame.K_c]:
                command.clutch_frac -= 0.08
            command.starter_signal = bool(keys[pygame.K_g])
            command.throttle = float(np.clip(command.throttle, 0.0, 1.0))
            command.brake_load_nm = float(np.clip(
                command.brake_load_nm, 0.0,
                rollout.engine.peak_torque_nm * 1.3))
            command.electrical_load_frac = float(np.clip(
                command.electrical_load_frac, 0.0, 3.0))
            command.clutch_frac = float(np.clip(command.clutch_frac, 0.0, 1.0))
            ticks = 0
            while tick_accumulator >= 0.005 and ticks < 8:
                rollout.step(command)
                tick_accumulator -= 0.005
                ticks += 1
            if tick_accumulator >= 0.005:
                scheduler_drops += int(tick_accumulator / 0.005)
                tick_accumulator %= 0.005
            live_drift = LiveShadowTeacher.drift_from_values(
                rollout, latest_shadow_values)
            now = time.monotonic()
            correction_due = (shadow_correction_seconds > 0.0
                              and now - last_correction_time
                              >= shadow_correction_seconds)
            if correction_due or rollout.diverged:
                rollout.correct_from_values(latest_shadow_values)
                last_correction_time = now
                correction_count += 1
            audio_state.set_from_sim(rollout)
            glViewport(0, 0, width, height)
            glClearColor(10 / 255, 13 / 255, 19 / 255, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, width, height, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            hubs, dots = _network_geometry(rollout_learner, width, height)
            glBegin(GL_LINES)
            for dot, hub, strength in dots:
                glColor4f(0.23, 0.42, 0.58, 0.10 + 0.35 * min(strength, 1.0))
                glVertex2f(*dot); glVertex2f(*hub)
            glEnd()
            glPointSize(5.0); glBegin(GL_POINTS)
            for dot, _hub, strength in dots:
                glColor4f(0.25, 0.58 + 0.25 * min(strength, 1.0), 1.0, 0.9)
                glVertex2f(*dot)
            glColor4f(1.0, 0.68, 0.29, 0.9)
            for hub in hubs: glVertex2f(*hub)
            glEnd()
            runtime = rollout.performance_snapshot()
            lines = [
                "COMPILED DENDRITE ENGINE LEARNER",
                f"epoch {epoch + 1}/{epochs} pass {pass_index + 1}/{passes_per_epoch}  "
                f"step {step}/{steps}   loss "
                f"{result.loss:.6g}" if result else "background training starting",
                (f"live teacher replay loss {live_result.loss:.6g}  "
                 f"buffer {len(replay_x)}" if live_result else
                 f"live teacher replay warming {len(replay_x)}/{batch_size}"),
                (f"gradient {result.gradient_norm:.5g}   neural weights {result.parameter_norm:.5g}"
                 if result else "neural weights awaiting first background update"),
                f"{rollout.engine_identity}  network audio throttle {command.throttle:.0%} rpm {rollout.rpm:.1f}  "
                f"gear {command.gear_index} clutch {command.clutch_frac:.0%}",
                f"brake {command.brake_load_nm:.0f} Nm electrical {command.electrical_load_frac:.0%}  "
                f"ignition {'ON' if command.ignition_enabled else 'OFF'}  {command.fuel_choice}",
                "[/] engine  W/S throttle  E/D brake  ,/. gear  C/V clutch  Z/X electrical  G starter  K ignition  F fuel",
                f"live shadow drift {live_drift['normalized_rmse']:.4g}  "
                f"rpm {live_drift['rpm_absolute']:.1f}  torque {live_drift['torque_absolute']:.1f}  "
                f"corrections {correction_count}",
                f"compiled batch-1 {runtime['compiled_mean_ms']:.3f} ms mean / "
                f"{runtime['compiled_p95_ms']:.3f} ms p95 / "
                f"{runtime['compiled_transitions_per_second']:.1f} transitions/s",
                f"end-to-end {runtime['end_to_end_mean_ms']:.3f} ms / "
                f"realtime x{runtime['realtime_factor']:.2f} / "
                f"deadline misses {runtime['deadline_miss_fraction']:.1%} / scheduler drops {scheduler_drops}",
                ("network rollout DIVERGED; correcting from teacher"
                 if rollout.diverged else "network rollout finite"),
                "LLVM forward + ProcessGraph VJP / Adam",
            ]
            for index, line in enumerate(lines):
                surface = font.render(line, True, (220, 232, 242))
                pixels = pygame.image.tostring(surface, "RGBA", True)
                glWindowPos2d(20, height - 28 - index * 23)
                glDrawPixels(surface.get_width(), surface.get_height(), GL_RGBA,
                             GL_UNSIGNED_BYTE, pixels)
            pygame.display.flip()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if stream is not None: stream.stop(); stream.close()
        if streamer is not None: streamer.stop()
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", default=".turing-cache/engine-dendrite-live")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--passes-per-epoch", type=int, default=DEFAULT_PASSES_PER_EPOCH)
    parser.add_argument("--samples-per-fuel", type=int, default=DEFAULT_SAMPLES_PER_FUEL)
    parser.add_argument("--episodes-per-fuel", type=int, default=DEFAULT_EPISODES_PER_FUEL)
    parser.add_argument("--random-coverage-samples", type=int,
                        default=DEFAULT_RANDOM_COVERAGE_SAMPLES)
    parser.add_argument(
        "--engine-profiles", nargs="+", default=["all"],
        help="catalogue engine identities, or 'all' for the union ABI")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--branches", type=int, default=DEFAULT_BRANCHES)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--learning-rate", type=float,
                        default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--gradient-accumulation-steps", type=int,
                        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--max-global-gradient-norm", type=float,
                        default=DEFAULT_MAX_GLOBAL_GRADIENT_NORM)
    parser.add_argument("--no-adaptive-sample-refresh", action="store_true")
    parser.add_argument("--refresh-gap-growth", type=float,
                        default=DEFAULT_REFRESH_GAP_GROWTH)
    parser.add_argument("--refresh-named-samples-per-fuel", type=int,
                        default=DEFAULT_REFRESH_NAMED_SAMPLES_PER_FUEL)
    parser.add_argument("--refresh-random-coverage-samples", type=int,
                        default=DEFAULT_REFRESH_RANDOM_COVERAGE_SAMPLES)
    parser.add_argument("--max-sample-refreshes", type=int,
                        default=DEFAULT_MAX_SAMPLE_REFRESHES)
    parser.add_argument("--information-dropout", type=float, default=0.02)
    parser.add_argument("--branch-dropout", type=float, default=0.02)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--engine-toy-path")
    parser.add_argument("--max-audio-seconds", type=float, default=20.0)
    parser.add_argument("--shadow-correction-seconds", type=float, default=2.0)
    args = parser.parse_args()
    engine_identities = (None if args.engine_profiles == ["all"]
                         else tuple(args.engine_profiles))
    kwargs = dict(
        output_dir=args.output_dir, epochs=args.epochs,
        passes_per_epoch=args.passes_per_epoch,
        samples_per_fuel=args.samples_per_fuel, batch_size=args.batch_size,
        episodes_per_fuel=args.episodes_per_fuel,
        random_coverage_samples=args.random_coverage_samples,
        engine_identities=engine_identities,
        branches=args.branches, seed=args.seed,
        learning_rate=args.learning_rate, audio=not args.no_audio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_global_gradient_norm=args.max_global_gradient_norm,
        adaptive_sample_refresh=not args.no_adaptive_sample_refresh,
        refresh_gap_growth=args.refresh_gap_growth,
        refresh_named_samples_per_fuel=args.refresh_named_samples_per_fuel,
        refresh_random_coverage_samples=args.refresh_random_coverage_samples,
        max_sample_refreshes=args.max_sample_refreshes,
        information_dropout=args.information_dropout,
        branch_dropout=args.branch_dropout,
        use_cache=not args.no_cache,
        engine_toy_path=args.engine_toy_path)
    if args.headless:
        result = run_headless(
            **kwargs, max_audio_seconds=args.max_audio_seconds)
        print(json.dumps({
            "image": str(result.image_path),
            "audio": str(result.audio_path) if result.audio_path else None,
            "contract": str(result.contract_path),
            "metrics": str(result.metrics_path),
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "best_validation_loss": result.best_validation_loss,
        }, indent=2))
    else:
        run_interactive(
            **kwargs,
            shadow_correction_seconds=args.shadow_correction_seconds)


if __name__ == "__main__":
    main()
