import json

import numpy as np

from src.compiler.perforated_network_llvm import compile_perforated_network
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
