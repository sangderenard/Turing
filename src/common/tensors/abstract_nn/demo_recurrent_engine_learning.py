"""Train the compiled recurrent perforated cell on continuous real engines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ....compiler.perforated_recurrent_llvm import (
    CompiledPerforatedRecurrent,
    compile_recurrent_adam_trajectory,
    run_recurrent_adam_trajectory,
)
from .demo_perforated_multifuel_engine import (
    _load_engine_toy,
    _station_shaft_load,
    _transition_context,
    engine_parameter_mapping,
    physical_state_mapping,
)


# Do not expose the host-orchestrated prototype as the engine demo.  This is
# enabled only after whole-trajectory recurrence and Adam execute correctly in
# the single native LLVM entry.
FULL_LLVM_RECURRENT_TRAINING_READY = True


def capture_continuous_engine_run(*, engine_identity: str, steps: int,
                                  dt_seconds: float, seed: int,
                                  engine_toy_path=None):
    """Capture one uninterrupted start/load/shift/WOT/recovery engine run."""
    if steps < 64:
        raise ValueError("a continuous engine run requires at least 64 transitions")
    EngineCycleSim, get_engine = _load_engine_toy(engine_toy_path)
    sim = EngineCycleSim(get_engine(engine_identity))
    rng = np.random.default_rng(seed)
    parameters = engine_parameter_mapping(sim.engine)
    fuel_profiles = tuple(sim.engine.fuel_compatibility) or ("none",)
    if fuel_profiles[0] != "none":
        sim.fuel_choice = fuel_profiles[0]
    idle_power_w = max(
        sim.engine.peak_torque_nm * sim.engine.idle_rpm * 2.0 * math.pi / 60.0,
        1.0)
    rows, targets = [], []
    previous_state = previous_controls = None
    applied_load_w = 0.0
    throttle = 0.0
    top_gear = len(sim.engine.transmission.gear_ratios)
    for index in range(steps):
        progress = index / max(steps - 1, 1)
        starter = float(progress < 0.10)
        ignition = float(progress >= 0.025)
        if starter:
            sim.engage_starter()
        if progress < 0.20:
            target_throttle, target_load, gear = 0.0, 0.0, 0
        elif progress < 0.34:
            target_throttle, target_load, gear = 0.10, 0.58, top_gear
        elif progress < 0.52:
            phase = (progress - 0.34) / 0.18
            target_throttle, target_load = 0.18 + 0.82 * phase, 0.30
            gear = max(1, min(top_gear, 1 + int(phase * top_gear)))
        elif progress < 0.72:
            phase = (progress - 0.52) / 0.20
            target_throttle, target_load = 1.0, 0.08
            gear = max(1, min(top_gear, 1 + int(phase * top_gear)))
        elif progress < 0.84:
            phase = (progress - 0.72) / 0.12
            target_throttle, target_load = 0.42, 0.48
            gear = max(1, min(top_gear, top_gear - int(phase * top_gear)))
        else:
            target_throttle, target_load, gear = 0.0, 0.72, top_gear
        throttle += 0.22 * (target_throttle - throttle)
        throttle = float(np.clip(throttle + rng.normal(0.0, 0.006), 0.0, 1.0))
        applied_load_w += 0.16 * (target_load * idle_power_w - applied_load_w)
        if gear != sim.gear_index:
            sim.clutch_frac = max(0.0, sim.clutch_frac - 0.45)
            if sim.clutch_frac <= 0.05:
                sim.gear_index = gear
        else:
            sim.clutch_frac += 0.24 * (1.0 - sim.clutch_frac)
        ratio = sim._current_gear_ratio()
        if sim.clutch_frac <= 0.05 and ratio > 0.0:
            sim._load_omega = sim._omega / ratio
        sim.throttle = throttle
        sim.electrical_load_frac = float(
            np.clip(0.35 + 0.3 * math.sin(index * 0.071 + seed), 0.0, 1.0))
        _station_shaft_load(sim, applied_load_w)
        sim.state.ignition_cut = ignition <= 0.5
        before = physical_state_mapping(sim.state)
        feedforward = sim.ecu.idle_load_feedforward_frac(
            sim.engine,
            sim.state.ac_compressor_load_w
            + sim.electrical.reading.alternator_shaft_load_w
            + applied_load_w)
        controls = {
            "control.dt_seconds": dt_seconds,
            "control.throttle": throttle,
            "control.brake_load_nm": sim.brake_load_nm,
            "control.electrical_load_frac": sim.electrical_load_frac,
            "control.known_accessory_shaft_load_w": applied_load_w,
            "control.idle_load_feedforward_frac": feedforward,
            "control.clutch_frac": sim.clutch_frac,
            "control.gear_index": float(sim.gear_index),
            "control.starter_signal": starter,
            "control.ignition_enabled": ignition,
        }
        controls.update({f"fuel.{name}": float(name == fuel_profiles[0])
                         for name in fuel_profiles})
        context = _transition_context(
            before, previous_state, controls, previous_controls)
        rows.append({**parameters, **before, **controls, **context})
        sim.step(dt_seconds)
        after = physical_state_mapping(sim.state)
        targets.append({name: after.get(name, 0.0) - value
                        for name, value in before.items()})
        previous_state, previous_controls = before, controls
    feature_names = tuple(sorted(set().union(*(row.keys() for row in rows))))
    target_names = tuple(sorted(set().union(*(row.keys() for row in targets))))
    x = np.asarray([[row.get(name, 0.0) for name in feature_names] for row in rows])
    y = np.asarray([[row.get(name, 0.0) for name in target_names] for row in targets])
    return x, y, feature_names, target_names


def _standardize(train: np.ndarray, validation: np.ndarray, *, range_floor=0.0):
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    observed = np.max(np.abs(np.concatenate((train, validation))), axis=0, keepdims=True)
    scale = np.maximum(scale, range_floor * observed)
    scale[scale < 1e-10] = 1.0
    return (train - mean) / scale, (validation - mean) / scale, mean, scale


def closed_loop_rollout_mse(learner, raw_x: np.ndarray, raw_y: np.ndarray,
                            feature_names, target_names,
                            x_mean: np.ndarray, x_scale: np.ndarray,
                            y_mean: np.ndarray, y_scale: np.ndarray) -> float:
    """Measure accumulated state error while the network owns engine state."""
    feature_lookup = {name: index for index, name in enumerate(feature_names)}
    target_lookup = {name: index for index, name in enumerate(target_names)}
    state_names = tuple(name for name in target_names if name in feature_lookup)
    control_names = tuple(name for name in feature_names
                          if name.startswith("control."))
    predicted = {name: float(raw_x[0, feature_lookup[name]])
                 for name in state_names}
    previous_state = previous_controls = None
    hidden = np.zeros(learner.shapes["hidden"])
    squared = []
    for row_index, source in enumerate(raw_x):
        row = source.copy()
        controls = {name: float(row[feature_lookup[name]])
                    for name in control_names}
        context = _transition_context(
            predicted, previous_state, controls, previous_controls)
        for name, value in predicted.items():
            row[feature_lookup[name]] = value
        for name, value in context.items():
            if name in feature_lookup:
                row[feature_lookup[name]] = value
        normalized = (row.reshape(1, -1) - x_mean) / x_scale
        step = learner.forward(normalized, hidden)
        delta = step.prediction * y_scale + y_mean
        before = dict(predicted)
        for name in state_names:
            predicted[name] += float(delta[0, target_lookup[name]])
            actual_next = (float(source[feature_lookup[name]])
                           + float(raw_y[row_index, target_lookup[name]]))
            scale = max(float(x_scale[0, feature_lookup[name]]), 1e-12)
            squared.append(((predicted[name] - actual_next) / scale) ** 2)
        hidden = step.hidden
        previous_state, previous_controls = before, controls
    return float(np.mean(squared))


def run_learning(*, output_dir: str | Path, engine_identity: str,
                 steps: int = 320, dt_seconds: float = 0.05,
                 hidden_dim: int = 24, epochs: int = 32,
                 learning_rate: float = 0.00035,
                 training_runs: int = 16, validation_runs: int = 4,
                 seed: int = 1729, engine_toy_path=None):
    output_dir = Path(output_dir)
    if training_runs < 2 or validation_runs < 1:
        raise ValueError("use at least two training and one validation experience")
    capture_dir = output_dir / "capture-cache"
    capture_dir.mkdir(parents=True, exist_ok=True)

    def capture(run_seed: int):
        key = f"{engine_identity}-{steps}-{dt_seconds:.9g}-{run_seed}.npz"
        path = capture_dir / key
        if path.is_file():
            stored = np.load(path, allow_pickle=False)
            print(f"[capture cache hit] seed={run_seed}", flush=True)
            return (stored["x"], stored["y"],
                    tuple(map(str, stored["feature_names"])),
                    tuple(map(str, stored["target_names"])))
        print(f"[capture real simulator] seed={run_seed}", flush=True)
        result = capture_continuous_engine_run(
            engine_identity=engine_identity, steps=steps, dt_seconds=dt_seconds,
            seed=run_seed, engine_toy_path=engine_toy_path)
        np.savez_compressed(path, x=result[0], y=result[1],
                            feature_names=np.asarray(result[2]),
                            target_names=np.asarray(result[3]))
        return result

    all_runs = [capture(seed + index)
                for index in range(training_runs + validation_runs)]
    abi = all_runs[0][2:]
    if any(run[2:] != abi for run in all_runs[1:]):
        raise RuntimeError("continuous engine experiences produced incompatible ABIs")
    training_raw = all_runs[:training_runs]
    validation_raw = all_runs[training_runs:]
    train_x_all = np.concatenate([run[0] for run in training_raw])
    valid_x_all = np.concatenate([run[0] for run in validation_raw])
    train_y_all = np.concatenate([run[1] for run in training_raw])
    valid_y_all = np.concatenate([run[1] for run in validation_raw])
    x_train_all, x_valid_all, x_mean, x_scale = _standardize(
        train_x_all, valid_x_all)
    y_train_all, y_valid_all, y_mean, y_scale = _standardize(
        train_y_all, valid_y_all, range_floor=0.05)
    x_training = tuple(np.split(x_train_all, training_runs))
    x_validation = tuple(np.split(x_valid_all, validation_runs))
    y_training = tuple(np.split(y_train_all, training_runs))
    y_validation = tuple(np.split(y_valid_all, validation_runs))
    feature_names, target_names = abi
    feature_lookup = {name: index for index, name in enumerate(feature_names)}
    state_columns = np.asarray([feature_lookup[name] for name in target_names], dtype=np.int64)
    state_scale = x_scale[:, state_columns]
    delta_to_state_scale = y_scale / state_scale
    delta_to_state_bias = y_mean / state_scale

    def normalized_next_states(raw_x, raw_y):
        current = raw_x[:, state_columns]
        return (current + raw_y - x_mean[:, state_columns]) / state_scale

    train_states = tuple(normalized_next_states(run[0], run[1])
                         for run in training_raw)
    valid_states = tuple(normalized_next_states(run[0], run[1])
                         for run in validation_raw)

    state_route = np.zeros((len(target_names), len(feature_names)))
    for state_index, feature_column in enumerate(state_columns):
        state_route[state_index, feature_column] = 1.0
    exogenous_bank = np.stack(x_training)
    exogenous_bank[:, :, state_columns] = 0.0
    target_state_bank = np.stack(train_states)
    initial_state_bank = np.stack([
        experience[0, state_columns] for experience in x_training
    ])
    print(f"[compile full LLVM cycle] {training_runs} experiences x {steps} "
          f"transitions; {x_train_all.shape[1]} -> {hidden_dim} -> "
          f"{y_train_all.shape[1]}", flush=True)
    native_cycle = compile_recurrent_adam_trajectory(
        output_dir / "llvm-training", in_dim=x_train_all.shape[1],
        hidden_dim=hidden_dim, out_dim=y_train_all.shape[1],
        trajectory_steps=steps, experience_count=training_runs,
        dendrites_per_hidden=2, max_global_gradient_norm=0.25)
    # The same compiled cell is the deployment/evaluation surface. It starts
    # from the identical deterministic parameters used by the native cycle.
    learner = CompiledPerforatedRecurrent.compile(
        output_dir / "llvm-inference", in_dim=x_train_all.shape[1],
        hidden_dim=hidden_dim, out_dim=y_train_all.shape[1], seed=seed,
        learning_rate=learning_rate, max_gradient_norm=0.25)

    def closed_loop_mean(xs, states):
        losses = []
        for experience, targets in zip(xs, states):
            predicted = learner.rollout_closed_loop(
                experience, experience[0:1, state_columns],
                state_columns=state_columns,
                delta_to_state_scale=delta_to_state_scale,
                delta_to_state_bias=delta_to_state_bias)[0]
            losses.append(np.mean((predicted - targets) ** 2))
        return float(np.mean(losses))

    initial_closed_loop = closed_loop_mean(x_validation, valid_states)
    initial_training = closed_loop_mean(x_training, train_states)
    initial_validation = float(np.mean([
        np.mean((learner.rollout(x)[0] - y) ** 2)
        for x, y in zip(x_validation, y_validation)]))
    total_transitions = training_runs * steps
    print(f"[native train] one LLVM call; {epochs} epochs x {training_runs} "
          f"complete experiences ({total_transitions} transitions/epoch)",
          flush=True)
    native_result = run_recurrent_adam_trajectory(
        native_cycle, exogenous=exogenous_bank,
        target_states=target_state_bank,
        initial_states=initial_state_bank, state_route=state_route,
        delta_to_state_scale=delta_to_state_scale,
        delta_to_state_bias=delta_to_state_bias, epochs=epochs, seed=seed,
        learning_rate=learning_rate)
    for name, value in native_result.parameters.items():
        learner.parameters[name][...] = value
    final_closed_loop = closed_loop_mean(x_validation, valid_states)
    final_training = closed_loop_mean(x_training, train_states)
    final_validation = float(np.mean([
        np.mean((learner.rollout(x)[0] - y) ** 2)
        for x, y in zip(x_validation, y_validation)]))
    zero_hidden = np.zeros(learner.shapes["hidden"])
    reset_prediction = np.asarray([
        learner.forward(row.reshape(1, -1), zero_hidden).prediction[0]
        for row in x_valid_all
    ])
    reset_validation = float(np.mean((reset_prediction - y_valid_all) ** 2))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "recurrent-engine-weights.npz", **learner.parameters,
        feature_names=np.asarray(feature_names), target_names=np.asarray(target_names),
        feature_mean=x_mean, feature_scale=x_scale,
        target_mean=y_mean, target_scale=y_scale)
    metrics = {
        "engine": engine_identity, "trajectory_seconds": steps * dt_seconds,
        "transitions_per_trajectory": steps, "hidden_dim": hidden_dim,
        "training_experiences": training_runs,
        "validation_experiences": validation_runs,
        "training_transitions_per_epoch": total_transitions,
        "epochs": epochs, "training_initial_mse": initial_training,
        "training_final_mse": final_training,
        "learning_rate": learning_rate,
        "validation_initial_mse": initial_validation,
        "validation_final_mse": final_validation,
        "closed_loop_initial_mse": initial_closed_loop,
        "closed_loop_final_mse": final_closed_loop,
        "native_final_experience_loss_bank": native_result.loss_bank.tolist(),
        "native_adam_updates": native_result.iteration,
        "native_clipped_gradient_norm": native_result.gradient_norm,
        "validation_hidden_reset_mse": reset_validation,
        "cycle_owner": "single-native-llvm-entry",
        "vjp": "compiled-process-graph", "tape_autograd": False,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def main() -> None:
    if not FULL_LLVM_RECURRENT_TRAINING_READY:
        raise SystemExit(
            "recurrent engine training is withheld: the whole-trajectory LLVM "
            "cycle compiles but has not passed finite native execution"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="build/recurrent-engine-learning")
    parser.add_argument("--engine", default="ldt465-multifuel-deuce")
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--training-runs", type=int, default=16)
    parser.add_argument("--validation-runs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--engine-toy-path")
    args = parser.parse_args()
    run_learning(output_dir=args.output_dir, engine_identity=args.engine,
                 steps=args.steps, dt_seconds=args.dt, hidden_dim=args.hidden,
                 epochs=args.epochs, learning_rate=args.learning_rate,
                 training_runs=args.training_runs,
                 validation_runs=args.validation_runs,
                 seed=args.seed,
                 engine_toy_path=args.engine_toy_path)


if __name__ == "__main__":
    main()
