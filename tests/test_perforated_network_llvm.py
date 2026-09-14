import json

import numpy as np
import pytest

from src.compiler.perforated_network_llvm import (
    compile_perforated_adam_chunk,
    compile_perforated_network,
)
from src.compiler.compiled_perforated_adam import CompiledPerforatedAdam
from src.compiler.ssa_llvm_backend import prepare_artifact_execution


def test_perforated_network_compiles_and_executes_forward_and_vjp(tmp_path):
    compiled = compile_perforated_network(
        tmp_path, batch=2, in_dim=2, out_dim=2, dendrites_per_neuron=2
    )
    ids = {port.name: port.value_id for port in compiled.contract.inputs}
    values = {
        ids["x"]: np.asarray([[0.2, -0.4], [0.7, 0.1]]),
        ids["base_weight"]: np.asarray([[0.5, 0.2], [-0.25, 0.6]]),
        ids["base_bias"]: np.asarray([[0.1, -0.05]]),
        ids["dendrite_weight"]: np.asarray(
            [[0.3, -0.6, 0.1, 0.7], [0.8, 0.2, -0.4, 0.3]]
        ),
        ids["dendrite_bias"]: np.asarray([[0.05, -0.1, 0.02, 0.08]]),
        ids["dendrite_gain"]: np.asarray([[0.4, -0.35, 0.2, 0.3]]),
        ids["dendrite_route"]: np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        ids["dendrite_mask"]: np.ones((1, 4)),
        ids["network_authority"]: np.ones((2, 2)),
        ids["simulator_delta"]: np.zeros((2, 2)),
    }
    forward = prepare_artifact_execution(compiled.forward.artifact, values).run()
    x = values[ids["x"]]
    branches = np.tanh(
        x @ values[ids["dendrite_weight"]]
        + values[ids["dendrite_bias"]]
    )
    prediction = (
        x @ values[ids["base_weight"]]
        + values[ids["base_bias"]]
        + np.sum(
            branches.reshape(2, 2, 2)
            * values[ids["dendrite_gain"]].reshape(1, 2, 2),
            axis=2,
        )
    )
    np.testing.assert_allclose(
        forward.buffers[compiled.forward.output_value_ids[0]], prediction
    )

    seed = np.asarray([[0.7, -0.1], [-0.2, 0.4]])
    vjp = prepare_artifact_execution(
        compiled.forward_vjp.artifact,
        {**values, compiled.contract.prediction_adjoint.value_id: seed},
    ).run()
    gradient_ids = {
        port.name.removeprefix("grad_"): port.value_id
        for port in compiled.contract.gradients
    }
    reverse_shapes = dict(zip(
        compiled.forward_vjp.artifact.buffer_order,
        compiled.forward_vjp.artifact.buffer_shapes,
    ))
    assert all(
        reverse_shapes[port.value_id] == port.shape
        for port in compiled.contract.inputs
    )
    np.testing.assert_allclose(vjp.buffers[gradient_ids["base_weight"]], x.T @ seed)
    np.testing.assert_allclose(
        vjp.buffers[gradient_ids["base_bias"]], seed.sum(axis=0, keepdims=True)
    )
    routed_seed = np.repeat(seed, 2, axis=1)
    flat_branches = branches.reshape(2, 4)
    flat_activation_gradient = (
        routed_seed
        * values[ids["dendrite_gain"]]
        * (1.0 - flat_branches ** 2)
    )
    np.testing.assert_allclose(
        vjp.buffers[gradient_ids["dendrite_weight"]],
        x.T @ flat_activation_gradient,
    )
    np.testing.assert_allclose(
        vjp.buffers[gradient_ids["dendrite_bias"]],
        flat_activation_gradient.sum(axis=0, keepdims=True),
    )
    np.testing.assert_allclose(
        vjp.buffers[gradient_ids["dendrite_gain"]],
        (routed_seed * flat_branches).sum(axis=0, keepdims=True),
    )

    manifest = json.loads(compiled.manifest_path.read_text(encoding="utf-8"))
    assert manifest["execution"] == {
        "backward_packaging": "combined-forward-vjp",
        "backward_source": "process-graph-generator",
        "forward_source": "isolated-abstract-tensor-ssa",
        "tape_autograd": False,
    }
    assert compiled.forward.artifact.library_path.is_file()
    assert compiled.forward_vjp.artifact.library_path.is_file()

    cached = compile_perforated_network(
        tmp_path, batch=2, in_dim=2, out_dim=2, dendrites_per_neuron=2
    )
    assert cached.forward.key == "cached"
    cached_forward = prepare_artifact_execution(
        cached.forward.artifact, values).run()
    np.testing.assert_allclose(
        cached_forward.buffers[cached.forward.output_value_ids[0]], prediction)

    learner = CompiledPerforatedAdam(compiled, seed=7, learning_rate=0.01)
    before = {name: value.copy() for name, value in learner.parameters.items()}
    trained = learner.step(values[ids["x"]], np.zeros((2, 2)))
    assert np.isfinite(trained.loss)
    assert trained.gradient_norm > 0.0
    assert all(not np.array_equal(before[name], learner.parameters[name])
               for name in learner.parameter_names)
    fallback = np.asarray([[9.0, 8.0], [7.0, 6.0]])
    learner.set_output_authority(
        np.asarray([[1.0, 0.0], [1.0, 0.0]]), fallback)
    hybrid = learner.forward(values[ids["x"]])
    np.testing.assert_allclose(hybrid[:, 1], fallback[:, 1])


def test_singleton_batch_publishes_root_gradient_shapes(tmp_path):
    compiled = compile_perforated_network(
        tmp_path / "singleton", batch=1, in_dim=3, out_dim=2,
        dendrites_per_neuron=2,
    )
    expected = {
        "grad_base_weight": (3, 2),
        "grad_base_bias": (1, 2),
        "grad_dendrite_weight": (3, 4),
        "grad_dendrite_bias": (1, 4),
        "grad_dendrite_gain": (1, 4),
    }
    shapes = dict(zip(
        compiled.forward_vjp.artifact.buffer_order,
        compiled.forward_vjp.artifact.buffer_shapes,
    ))
    assert {port.name: shapes[port.value_id]
            for port in compiled.contract.gradients} == expected
    learner = CompiledPerforatedAdam(compiled, seed=11)
    result = learner.step(np.asarray([[0.2, -0.1, 0.7]]),
                          np.asarray([[0.3, -0.4]]))
    assert np.isfinite(result.loss)
    assert result.gradient_norm > 0.0


def test_compiled_adam_can_mask_fixed_batch_padding(tmp_path):
    learner = CompiledPerforatedAdam.compile(
        tmp_path / "masked", batch=2, in_dim=2, out_dim=1,
        dendrites_per_neuron=1, seed=13,
    )
    x = np.asarray([[0.25, -0.5], [99.0, 99.0]])
    target = np.asarray([[0.4], [-99.0]])
    result = learner.step(x, target, sample_weight=np.asarray([1.0, 0.0]))
    expected = float((result.prediction[0, 0] - target[0, 0]) ** 2)
    assert np.isclose(result.loss, expected)
    assert result.gradient_norm > 0.0


def test_native_adam_chunk_cycles_distinct_batches_and_matches_reference(tmp_path):
    compiled = compile_perforated_adam_chunk(
        tmp_path / "adam_chunk", batch=2, in_dim=2, out_dim=1,
        cycle_length=3, dendrites_per_neuron=2,
        gradient_accumulation_steps=2, max_global_gradient_norm=0.15,
    )
    assert compiled.cache_hit is False
    cached = compile_perforated_adam_chunk(
        tmp_path / "adam_chunk", batch=2, in_dim=2, out_dim=1,
        cycle_length=3, dendrites_per_neuron=2,
        gradient_accumulation_steps=2, max_global_gradient_norm=0.15,
    )
    assert cached.cache_hit is True
    assert cached.optimizer_state_value_ids == compiled.optimizer_state_value_ids
    ids = compiled.input_value_ids
    x = np.asarray([
        [[.2, -.4], [.7, .1]],
        [[-.3, .8], [.5, -.6]],
        [[.9, .2], [-.1, -.7]],
    ])
    target = np.asarray([
        [[.3], [-.2]], [[.5], [.1]], [[-.4], [.6]],
    ])
    sample_weight = np.asarray([
        [[1.], [1.]], [[1.], [.5]], [[1.], [0.]],
    ])
    loss_scale = np.stack([
        np.full((2, 1), 1.0 / weights.sum())
        for weights in sample_weight
    ])
    parameters = {
        "base_weight": np.asarray([[.2], [-.1]]),
        "base_bias": np.asarray([[.05]]),
        "dendrite_weight": np.asarray([[.3, -.2], [.1, .25]]),
        "dendrite_bias": np.asarray([[.02, -.03]]),
        "dendrite_gain": np.asarray([[.08, .06]]),
    }
    reference = {name: value.copy() for name, value in parameters.items()}
    first = {name: np.zeros_like(value) for name, value in parameters.items()}
    second = {name: np.zeros_like(value) for name, value in parameters.items()}
    state = compiled.optimizer_state_value_ids
    values = {
        ids["x"]: x,
        ids["target"]: target,
        ids["sample_weight"]: sample_weight,
        ids["loss_scale"]: loss_scale,
        ids["dendrite_route"]: np.ones((2, 1)),
        ids["dendrite_mask"]: np.ones((3, 1, 2)),
        ids["network_authority"]: np.ones((2, 1)),
        ids["simulator_delta"]: np.zeros((2, 1)),
        state["steps"]: np.asarray(5, dtype=np.int32),
        state["learning_rate"]: np.asarray(.01),
        state["beta1"]: np.asarray(.9),
        state["beta2"]: np.asarray(.999),
        state["epsilon"]: np.asarray(1e-8),
        state["beta1_power"]: np.asarray(1.0),
        state["beta2_power"]: np.asarray(1.0),
        state["iteration"]: np.asarray(0, dtype=np.int32),
    }
    for name, value in parameters.items():
        parameter_id = compiled.parameter_value_ids[name]
        values[parameter_id] = value.copy()
        values[state["first_moment"][parameter_id]] = np.zeros_like(value)
        values[state["second_moment"][parameter_id]] = np.zeros_like(value)

    execution = prepare_artifact_execution(compiled.artifact, values).run()

    beta1_power = beta2_power = 1.0
    final_loss = 0.0
    route = np.ones((2, 1))
    for group_start in range(0, 5, 2):
        accumulated = {name: np.zeros_like(value)
                       for name, value in reference.items()}
        group_stop = min(group_start + 2, 5)
        for motion in range(group_start, group_stop):
            index = motion % 3
            branches = np.tanh(
                x[index] @ reference["dendrite_weight"]
                + reference["dendrite_bias"])
            prediction = (
                x[index] @ reference["base_weight"]
                + reference["base_bias"]
                + (reference["dendrite_gain"] * branches) @ route
            )
            error = prediction - target[index]
            final_loss = float(np.sum(
                error * error * sample_weight[index] * loss_scale[index]))
            seed = 2.0 * error * sample_weight[index] * loss_scale[index]
            routed_seed = seed @ route.T
            branch_gradient = (
                routed_seed * reference["dendrite_gain"]
                * (1.0 - branches * branches)
            )
            gradients = {
                "base_weight": x[index].T @ seed,
                "base_bias": seed.sum(axis=0, keepdims=True),
                "dendrite_weight": x[index].T @ branch_gradient,
                "dendrite_bias": branch_gradient.sum(axis=0, keepdims=True),
                "dendrite_gain": (routed_seed * branches).sum(
                    axis=0, keepdims=True),
            }
            for name in accumulated:
                accumulated[name] += gradients[name]
        gradients = {name: value / (group_stop - group_start)
                     for name, value in accumulated.items()}
        gradient_norm = sum(float(np.sum(value * value))
                            for value in gradients.values()) ** 0.5
        clip_scale = min(1.0, 0.15 / (gradient_norm + 1e-9))
        gradients = {name: value * clip_scale
                     for name, value in gradients.items()}
        beta1_power *= .9
        beta2_power *= .999
        for name in reference:
            first[name] = .9 * first[name] + .1 * gradients[name]
            second[name] = .999 * second[name] + .001 * gradients[name] ** 2
            reference[name] -= .01 * (
                first[name] / (1.0 - beta1_power)
            ) / (np.sqrt(second[name] / (1.0 - beta2_power)) + 1e-8)

    for name in reference:
        parameter_id = compiled.parameter_value_ids[name]
        np.testing.assert_allclose(
            execution.buffers[parameter_id], reference[name],
            rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(
            execution.buffers[state["first_moment"][parameter_id]], first[name],
            rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(
            execution.buffers[state["second_moment"][parameter_id]], second[name],
            rtol=1e-12, atol=1e-13)
    assert int(execution.buffers[state["iteration"]]) == 3
    assert float(execution.buffers[state["beta1_power"]]) == pytest.approx(beta1_power)
    assert float(execution.buffers[state["beta2_power"]]) == pytest.approx(beta2_power)
    assert float(execution.buffers[compiled.output_value_ids["loss_0"]][1]) == pytest.approx(final_loss)
    assert float(execution.buffers[state["gradient_norm"]]) == pytest.approx(
        gradient_norm)
    assert float(execution.buffers[state["clipped_gradient_norm"]]) == pytest.approx(
        min(gradient_norm, 0.15), rel=1e-8)
    assert json.loads(compiled.manifest_path.read_text())["execution"] == {
        "changing_minibatches": True,
        "cycle_owner": "native-llvm-entry",
        "motion": "combined-forward-loss-process-graph-vjp",
        "optimizer": "adam",
        "tape_autograd": False,
    }
