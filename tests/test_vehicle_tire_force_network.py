from __future__ import annotations

import random

import numpy as np
import pytest

from src.compiler.vehicle_tire_force_network import (
    HUB_WRENCH_NAMES,
    TIRE_OPERATOR_OUTPUT_NAMES,
    TireForceNetworkSpec,
    TireForceNetworkTrainer,
    build_tire_force_training_graphs,
    pack_tire_force_features,
    reduce_teacher_bead_wrenches,
    tire_force_feature_names,
)


def _small_spec() -> TireForceNetworkSpec:
    return TireForceNetworkSpec(
        circumferential_segments=6,
        section_segments=6,
        history_frames=3,
        temporal_order=2,
        vehicle_state_width=4,
        hidden_channels=3,
        latent_width=4,
        batch_size=2,
    )


def test_feature_packer_keeps_temporal_orders_and_periodic_tire_topology():
    spec = _small_spec()
    b, t, u, v = 2, 3, 6, 6
    membrane = np.zeros((b, t, u, v, 6))
    terrain = np.zeros((b, t, u, v, 10))
    membrane[:, 0, :, :, 0] = 0.0
    membrane[:, 1, :, :, 0] = 1.0
    membrane[:, 2, :, :, 0] = 4.0
    membrane[:, :, -1, -1, 1] = 9.0
    rest = np.zeros((u, v, 3))
    vehicle = np.arange(b * 4, dtype=np.float64).reshape(b, 4)
    packed = pack_tire_force_features(
        spec,
        membrane_history=membrane,
        terrain_history=terrain,
        rest_skin_position=rest,
        tire_thermodynamic_state=np.zeros((b, 3)),
        vehicle_state=vehicle,
        dt=1.0,
    )
    assert packed.shape == spec.input_shape
    base = spec.base_field_width
    center = spec.halo
    assert packed[0, 0, center, center] == pytest.approx(4.0)
    assert packed[0, base, center, center] == pytest.approx(3.0 / 600.0)
    assert packed[0, 2 * base, center, center] == pytest.approx(2.0 / 600.0 ** 2)
    # The upper-left halo samples the wrapped lower-right field.
    assert packed[0, 1, spec.halo - 1, spec.halo - 1] == pytest.approx(9.0)
    assert len(tire_force_feature_names(spec)) == spec.input_channels
    assert {"gas_pressure_pa", "volume_ratio", "gas_temperature_k"} <= set(
        tire_force_feature_names(spec)
    )


def test_teacher_target_is_the_exact_sum_of_compiled_bead_rim_wrenches():
    forces = np.asarray([[[1.0, 2.0, 3.0], [4.0, -2.0, 1.0]]])
    moments = np.asarray([[[5.0, 0.0, -1.0], [-2.0, 3.0, 4.0]]])
    target = reduce_teacher_bead_wrenches(forces, moments)
    np.testing.assert_allclose(target, [[5.0, 0.0, 4.0, 3.0, 3.0, 3.0]])


def test_forward_and_functional_adam_are_abstract_tensor_graphs_without_required_reverse():
    spec = _small_spec()
    graphs = build_tire_force_training_graphs(spec)
    assert graphs.backward is None
    assert len(graphs.forward_graph.roots) == 2  # prediction and scalar loss
    assert len(graphs.adam_graph.roots) == 3 * len(graphs.parameter_names) + 1
    assert set(graphs.parameter_names) == {
        "conv0.weight", "conv0.bias", "conv1.weight", "conv1.bias",
        "dense0.weight", "dense0.bias", "dense1.weight", "dense1.bias",
    }
    assert graphs.manifest["stored_backward_graph_required"] is False
    assert graphs.manifest["output"] == list(TIRE_OPERATOR_OUTPUT_NAMES)
    assert set(graphs.forward_input_value_ids) == {
        "surface_state", "target_tire_operator_normalized", "tire_operator_loss_weight",
        *graphs.parameter_names,
    }
    assert graphs.forward_output_value_ids == tuple(map(int, graphs.forward_graph.roots))


def test_static_reverse_remains_an_explicit_opt_in_audit_product():
    graphs = build_tire_force_training_graphs(_small_spec(), include_static_backward=True)
    assert graphs.backward is not None
    assert graphs.backward.packaging == "independent"
    assert len(graphs.backward.adjoint.gradient_value_ids) == len(graphs.parameter_names)


def test_pythonic_abstract_nn_trainer_updates_the_same_model_and_reduces_loss():
    random.seed(3)
    np.random.seed(3)
    spec = _small_spec()
    trainer = TireForceNetworkTrainer(spec, lr=1e-3)
    features = np.zeros(spec.input_shape)
    target = np.zeros((spec.batch_size, len(TIRE_OPERATOR_OUTPUT_NAMES)))
    first = trainer.train_batch(features, target)
    second = trainer.train_batch(features, target)
    assert second < first
    assert trainer.predict(features).shape == target.shape
