import numpy as np

from src.compiler.perforated_recurrent_llvm import (
    CompiledPerforatedRecurrent,
    compile_recurrent_adam_trajectory,
    run_recurrent_adam_trajectory,
)


def test_compiled_recurrent_vjp_learns_across_a_complete_trajectory(tmp_path):
    x = np.asarray([
        [0.0, 0.3], [0.8, 0.1], [0.0, 0.2], [0.0, 0.4],
        [0.6, 0.2], [0.0, 0.1], [0.0, 0.5], [0.2, 0.3],
    ])
    memory = 0.0
    target = []
    for row in x:
        memory = 0.72 * memory + row[0]
        target.append([memory + 0.1 * row[1]])
    target = np.asarray(target)
    learner = CompiledPerforatedRecurrent.compile(
        tmp_path, in_dim=2, hidden_dim=5, out_dim=1,
        dendrites_per_hidden=2, seed=7, learning_rate=0.02)
    result = learner.train_trajectories(((x, target),), epochs=16, seed=11)
    assert result.final_rollout_mse < result.initial_rollout_mse * 0.35
    assert learner.reverse_artifact.saved_binding_count > 0
    assert learner.gradient_ids["hidden"] > 0


def test_whole_trajectory_recurrence_and_adam_run_in_one_llvm_call(tmp_path):
    compiled = compile_recurrent_adam_trajectory(
        tmp_path, in_dim=3, hidden_dim=4, out_dim=2,
        trajectory_steps=2, experience_count=2, dendrites_per_hidden=2)
    exogenous = np.zeros((2, 2, 3))
    exogenous[:, :, 2] = ((0.1, 0.2), (0.08, 0.16))
    targets = np.asarray((
        ((0.1, 0.0), (0.2, 0.1)),
        ((0.08, 0.0), (0.16, 0.08)),
    ))
    route = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    arguments = dict(
        exogenous=exogenous, target_states=targets,
        initial_states=np.zeros((2, 2)), state_route=route,
        delta_to_state_scale=np.ones((1, 2)),
        delta_to_state_bias=np.zeros((1, 2)), learning_rate=0.001,
        seed=1729)
    first = run_recurrent_adam_trajectory(compiled, epochs=1, **arguments)
    trained = run_recurrent_adam_trajectory(compiled, epochs=6, **arguments)
    assert trained.iteration == 6
    assert np.isfinite(trained.loss_bank).all()
    assert all(np.isfinite(value).all() for value in trained.parameters.values())
    assert trained.loss_bank.mean() < first.loss_bank.mean()
    assert 0.0 < trained.gradient_norm <= 0.25 + 1e-12
