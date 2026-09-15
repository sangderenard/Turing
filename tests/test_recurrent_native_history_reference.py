"""Native trajectory reverse/update versus independent finite differences."""
import numpy as np

from src.compiler import perforated_recurrent_llvm as recurrent


def test_native_history_gradients_and_adam_match_finite_differences(tmp_path, monkeypatch):
    compiled = recurrent.compile_recurrent_adam_trajectory(
        tmp_path, in_dim=3, hidden_dim=2, out_dim=2,
        trajectory_steps=3, experience_count=2, max_global_gradient_norm=0.05,
    )
    captured = {}
    prepare = recurrent.prepare_artifact_execution

    def capture(artifact, values):
        captured["inputs"] = {key: value.copy() for key, value in values.items()}
        captured["execution"] = prepare(artifact, values)
        return captured["execution"]

    monkeypatch.setattr(recurrent, "prepare_artifact_execution", capture)
    exogenous = np.array([
        [[0., 0., .7], [0., 0., -.2], [0., 0., .3]],
        [[0., 0., -.4], [0., 0., .5], [0., 0., -.1]],
    ])
    targets = np.array([
        [[.2, -.1], [.3, .15], [-.2, .4]],
        [[-.3, .1], [.2, -.2], [.1, .3]],
    ])
    initial = np.array([[.1, -.05], [-.1, .03]])
    route = np.array([[1., 0., 0.], [0., 1., 0.]])
    scale = np.array([[.8, 1.2]])
    bias = np.array([[.01, -.02]])
    result = recurrent.run_recurrent_adam_trajectory(
        compiled, exogenous=exogenous, target_states=targets,
        initial_states=initial, state_route=route,
        delta_to_state_scale=scale, delta_to_state_bias=bias,
        epochs=2, seed=1729, learning_rate=.001,
    )
    inputs = captured["inputs"]
    parameters = {name: inputs[value_id].copy()
                  for name, value_id in compiled.parameter_value_ids.items()}
    dendrite_route = inputs[compiled.input_value_ids["dendrite_route"]]

    def loss(experience):
        state = initial[experience:experience + 1].copy()
        hidden = np.zeros((1, 2))
        total = 0.0
        p = parameters
        for step in range(3):
            x = exogenous[experience, step:step + 1] + state @ route
            branches = np.tanh(x @ p["dendrite_input_weight"]
                               + hidden @ p["dendrite_recurrent_weight"]
                               + p["dendrite_bias"])
            candidate = np.tanh(x @ p["input_weight"]
                                + hidden @ p["recurrent_weight"]
                                + p["hidden_bias"]
                                + (branches * p["dendrite_gain"]) @ dendrite_route)
            hidden = .35 * hidden + .65 * candidate
            prediction = (x @ p["direct_weight"]
                          + hidden @ p["output_weight"] + p["output_bias"])
            state = state + prediction * scale + bias
            total += np.sum((state - targets[experience, step]) ** 2) / 6
        return total

    first = {name: np.zeros_like(value) for name, value in parameters.items()}
    second = {name: np.zeros_like(value) for name, value in parameters.items()}
    epsilon = 2e-6
    for epoch in range(1, 3):
        losses = np.array([loss(experience) for experience in range(2)])
        gradients = {}
        for name, parameter in parameters.items():
            gradient = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                above = np.mean([loss(experience) for experience in range(2)])
                parameter[index] = original - epsilon
                below = np.mean([loss(experience) for experience in range(2)])
                parameter[index] = original
                gradient[index] = (above - below) / (2 * epsilon)
            gradients[name] = gradient
        norm = np.sqrt(sum(np.sum(value ** 2) for value in gradients.values()))
        factor = min(1., .05 / norm)
        for name, parameter in parameters.items():
            gradient = gradients[name] * factor
            first[name] = .9 * first[name] + .1 * gradient
            second[name] = .999 * second[name] + .001 * gradient ** 2
            parameter -= .001 * (first[name] / (1 - .9 ** epoch)) / (
                np.sqrt(second[name] / (1 - .999 ** epoch)) + 1e-8)

    assert result.iteration == 2
    np.testing.assert_allclose(result.loss_bank, losses, rtol=2e-7, atol=2e-9)
    buffers = captured["execution"].buffers
    optimizer = compiled.optimizer_state_value_ids
    for name, parameter in parameters.items():
        value_id = compiled.parameter_value_ids[name]
        np.testing.assert_allclose(result.parameters[name], parameter,
                                   rtol=2e-6, atol=2e-8, err_msg=name)
        np.testing.assert_allclose(buffers[optimizer["first_moment"][value_id]],
                                   first[name], rtol=2e-6, atol=2e-9, err_msg=name)
        np.testing.assert_allclose(buffers[optimizer["second_moment"][value_id]],
                                   second[name], rtol=4e-6, atol=2e-11, err_msg=name)
