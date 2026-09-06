"""Train and profile the balloon-tire operator against its native authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compiler.vehicle_tire_authority import (
    build_tire_authority_definition,
    open_native_tire_authority,
    write_native_tire_authority,
)
from src.compiler.vehicle_tire_force_network import (
    TireForceNetworkSpec,
    TireForceNetworkTrainer,
    build_tire_force_training_graphs,
    compile_tire_force_forward_native,
)
from src.compiler.vehicle_tire_operator_experiment import (
    CompiledTireTeacher,
    ExactBalloonAnchorTrialGenerator,
    fit_linear_tire_operator,
    fit_linear_tire_state_operator,
    normalized_error,
    prepare_native_operator_execution,
    profile_native_operator,
    train_tire_operator,
    write_linear_tire_gpu_artifact,
    write_linear_tire_state_gpu_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build/native_tire_operator_trial"))
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--profile-iterations", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(1908)
    np.random.seed(1908)

    authority_definition = build_tire_authority_definition()
    authority_dir = Path("build/vehicle_tire_authority")
    try:
        written = open_native_tire_authority(authority_dir, definition=authority_definition)
    except (FileNotFoundError, ValueError):
        written = write_native_tire_authority(authority_dir, definition=authority_definition)
    teacher = CompiledTireTeacher(written)
    topology = authority_definition.manifest["topology"]
    spec = TireForceNetworkSpec(
        circumferential_segments=topology["circumferential_segments"],
        section_segments=topology["section_segments"],
        hidden_channels=6,
        latent_width=8,
        batch_size=4,
    )
    generator = ExactBalloonAnchorTrialGenerator(teacher, spec)
    trainer = TireForceNetworkTrainer(spec, lr=1.8e-3)

    print("[1/5] fitting the full-state linear GPU candidate", flush=True)
    linear, linear_corpus = fit_linear_tire_operator(
        generator, batches=256, ridge=1.0e-8,
    )
    linear_gpu = write_linear_tire_gpu_artifact(linear, spec, args.output / "linear_gpu")
    linear_state = fit_linear_tire_state_operator(linear_corpus, spec)
    linear_state_gpu = write_linear_tire_state_gpu_artifact(
        linear_state, spec, args.output / "linear_state_gpu",
    )

    print("[2/5] generating native-authority trials and training AbstractNN Adam", flush=True)
    training = train_tire_operator(trainer, generator, steps=args.steps)
    validation = training.pop("validation_batches")
    batch = validation[0]

    linear_errors = [normalized_error(
        linear.predict(item.features, spec), item.target_operator_output, spec.output_scale,
    ) for item in validation]
    state_errors = [normalized_error(
        linear_state.predict(item.features, spec), item.target_membrane_state,
        linear_state.output_scale(),
    ) for item in validation]
    linear_gpu_errors = []
    for item in validation:
        halo = spec.halo
        matrix = item.features[
            :, :, halo:halo + spec.circumferential_segments,
            halo:halo + spec.section_segments,
        ].reshape(spec.batch_size, -1).astype(np.float32)
        augmented = np.concatenate(
            (matrix, np.ones((spec.batch_size, 1), dtype=np.float32)), axis=1,
        )
        prediction = (
            augmented @ linear.weights.astype(np.float32)
        ).astype(np.float64) * spec.output_scale
        linear_gpu_errors.append(normalized_error(
            prediction, item.target_operator_output, spec.output_scale,
        ))

    print("[3/5] lowering the trained operator's repository SSA to native LLVM", flush=True)
    graphs = build_tire_force_training_graphs(spec)
    native = compile_tire_force_forward_native(graphs, args.output / "operator")
    execution = prepare_native_operator_execution(native, graphs, trainer, batch)
    execution.run()
    native_prediction = (
        np.asarray(execution.buffers[native.output_value_ids[0]], dtype=np.float64)
        * spec.output_scale
    )
    python_prediction = trainer.predict(batch.features)
    parity_error = float(np.max(np.abs(native_prediction - python_prediction)))

    print("[4/5] profiling native learned and exact-authority paths", flush=True)
    native_profile = profile_native_operator(
        execution, native.output_value_ids[0], iterations=args.profile_iterations,
    )
    exact_iterations = max(20, args.profile_iterations // 10)
    started = perf_counter()
    exact = None
    for _ in range(exact_iterations):
        exact = generator.reevaluate_targets(batch)
    exact_elapsed = perf_counter() - started
    assert exact is not None

    controller = trainer.create_reference_workshare()
    mixed_iterations = 256
    reference_calls = 0
    linear_full_prediction = linear.predict(batch.features, spec)
    linear_prediction = linear_full_prediction[:, :6]
    thermodynamic_reconstruction_rmse = normalized_error(
        linear_full_prediction[:, 6:], batch.target_thermodynamic_state,
        spec.output_scale[6:],
    )
    thermodynamic_novelty = float(np.clip(
        thermodynamic_reconstruction_rmse / 0.02, 0.0, 1.0,
    ))
    started = perf_counter()
    mixed = None
    for _ in range(mixed_iterations):
        mixed, did_reference, _loss = controller.step(
            linear_prediction,
            lambda: generator.reevaluate_targets(batch)[:, :6],
            thermodynamic_novelty=thermodynamic_novelty,
        )
        reference_calls += int(did_reference)
    mixed_elapsed = perf_counter() - started

    report = {
        "schema": "turing.native-tire-operator-trial.v1",
        "authority_digest": authority_definition.digest,
        "authority_library": str(written.library_path),
        "operator_library": str(native.artifact.library_path),
        "scope": {
            "trial_kind": "progressive anchored whole-balloon/contact curriculum",
            "teacher_boundary": "compiled persistent skin plus measured equal-opposite anchor wrench",
            "release_order": ["terrain-excited-clamped", "rim-rotation", "hub-translation"],
            "topology_vertices": topology["vertex_count"],
            "topology_faces": topology["face_count"],
            "feature_channels": spec.input_channels,
        },
        "training": training,
        "validation": {
            "linear_normalized_rmse": float(np.mean(linear_errors)),
            "linear_gpu_float32_normalized_rmse": float(np.mean(linear_gpu_errors)),
            "linear_state_transition_normalized_rmse": float(np.mean(state_errors)),
            "native_normalized_rmse": normalized_error(
                native_prediction, batch.target_operator_output, spec.output_scale,
            ),
            "python_native_max_abs_wrench_difference": parity_error,
        },
        "profile": {
            "linear_gpu": linear_gpu,
            "linear_state_gpu": linear_state_gpu,
            "learned_native": native_profile,
            "exact_compiled_balloon_appendage": {
                "iterations": exact_iterations,
                "seconds": exact_elapsed,
                "microseconds_per_four_wheel_batch": 1e6 * exact_elapsed / exact_iterations,
            },
            "adaptive_linear_exact_mix": {
                "iterations": mixed_iterations,
                "seconds": mixed_elapsed,
                "microseconds_per_four_wheel_batch": 1e6 * mixed_elapsed / mixed_iterations,
                "timing_scope": "host scheduler plus exact CPU trial calls; GPU dispatch excluded",
                "reference_trials": reference_calls,
                "reference_duty_fraction": reference_calls / mixed_iterations,
                "final_alpha": controller.state.alpha,
                "final_loss_ema": controller.state.normalized_loss_ema,
                "thermodynamic_reconstruction_rmse": thermodynamic_reconstruction_rmse,
                "thermodynamic_novelty": thermodynamic_novelty,
                "mixed_normalized_rmse": normalized_error(
                    np.asarray(mixed), batch.target_hub_wrench, spec.hub_wrench_scale,
                ),
            },
        },
    }
    report_path = args.output / "trial_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("[5/5] wrote", report_path.resolve(), flush=True)
    print(json.dumps({
        "training": training,
        "validation": report["validation"],
        "profile": report["profile"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
