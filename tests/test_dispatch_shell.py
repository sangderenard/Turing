import numpy as np
import pytest

from src.common.tensors.accelerator_backends.dispatch_shell import dispatch_plan


@pytest.fixture
def plan():
    return dispatch_plan(modules=4, backends=3, stages=2, feature_dim=5)


def test_cube_dimensions_are_reported(plan):
    assert (plan.modules, plan.backends, plan.stages) == (4, 3, 2)
    assert plan.observations.shape == (4, 3, 2)
    assert plan.counts.shape == (4, 3, 2)
    assert plan.decisions.shape == (4, 2)


def test_graph_round_trips_as_coo_edge_index(plan):
    """The graph must come back in the layout a GNN already consumes."""

    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int32)
    plan.set_edges(edges)

    recovered = plan.edge_index
    assert recovered.shape == (2, 4)
    assert recovered.dtype == np.int32
    assert np.array_equal(recovered, edges)


def test_edges_outside_the_module_range_are_rejected(plan):
    with pytest.raises(ValueError):
        plan.set_edges(np.array([[0], [99]]))


def test_node_features_round_trip_as_float32_matrix(plan):
    for module in range(plan.modules):
        plan.set_features(module, np.arange(5) + module)

    features = plan.features
    assert features.shape == (4, 5)
    assert features.dtype == np.float32
    assert np.array_equal(features[2], np.arange(5, dtype=np.float32) + 2)


def test_unmeasured_cells_are_nan_not_zero(plan):
    """An unexplored option must never look like a free one."""

    observations = plan.observations
    assert np.all(np.isnan(observations))

    plan.observe(1, 2, 0, 0.5)
    assert not np.isnan(plan.observations[1, 2, 0])
    assert np.isnan(plan.observations[1, 2, 1])


def test_observations_accumulate_as_means(plan):
    plan.observe(0, 1, 0, 0.004)
    plan.observe(0, 1, 0, 0.006)

    assert plan.observations[0, 1, 0] == pytest.approx(0.005)
    assert plan.counts[0, 1, 0] == 2


def test_score_normalises_by_apparent_work_and_backend_weight(plan):
    plan.set_work(0, 1000.0)
    plan.observe(0, 0, 0, 0.010)
    plan.observe(0, 1, 0, 0.010)
    plan.set_backend_weights([1.0, 0.5, 1.0])

    # Same measured time, half the weight, so half the score.
    assert plan.score(0, 1, 0) == pytest.approx(plan.score(0, 0, 0) / 2)
    # Work normalisation makes differently sized modules comparable.
    assert plan.score(0, 0, 0) == pytest.approx(0.010 / 1000.0)


def test_select_best_uses_weights_and_ignores_unmeasured(plan):
    plan.set_work(0, 1.0)
    plan.observe(0, 0, 0, 0.001)
    plan.observe(0, 2, 1, 0.004)
    plan.set_backend_weights([1.0, 1.0, 0.1])

    # Backend 2 is slower but weighted down far enough to win; backend 1 is
    # unmeasured and must not be selected on the strength of being absent.
    assert plan.select_best(0) == 2
    assert plan.decisions[0].tolist() == [2, 1]


def test_python_can_implant_a_whole_decision_vector(plan):
    wanted = np.array([[0, 0], [1, 1], [2, 0], [1, 1]], dtype=np.int32)
    plan.implant(wanted)
    assert np.array_equal(plan.decisions, wanted)


def test_implant_rejects_out_of_range_decisions(plan):
    with pytest.raises(IndexError):
        plan.set_decision(0, 99, 0)


def test_launch_routes_through_the_decision_and_records_itself(plan):
    plan.set_decision(2, 1, 0)
    entered = []
    callback = plan.callback(lambda ctx, dev: (entered.append(1), 1)[1])

    result = plan.launch(2, callback)

    assert result["status"] == 1
    assert result["backend"] == 1
    assert entered == [1]
    # The launch observed itself into the cell its decision named.
    assert plan.counts[2, 1, 0] == 1
    assert plan.observations[2, 1, 0] > 0.0
    # And nowhere else.
    assert plan.counts.sum() == 1


def test_launch_without_a_decision_still_runs_but_records_nothing(plan):
    callback = plan.callback(lambda ctx, dev: 1)
    result = plan.launch(3, callback)

    assert result["status"] == 1
    assert result["backend"] == -1
    assert plan.counts.sum() == 0


def test_device_time_is_kept_separate_from_shell_time(plan):
    plan.set_decision(0, 0, 0)

    def gpu_like(context, device_ns):
        device_ns[0] = 5_000_000
        return 1

    result = plan.launch(0, plan.callback(gpu_like))
    assert result["device_ns"] == 5_000_000
    assert plan.device_observations[0, 0, 0] == pytest.approx(0.005, rel=1e-6)
