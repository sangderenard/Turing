from __future__ import annotations

import numpy as np

from src.compiler.vehicle_tire_local_acceleration_network import (
    TireLocalNetworkSpec,
    build_tire_local_training_graphs,
    pack_tire_local_features,
    teacher_local_acceleration,
    tire_local_feature_names,
    tire_signed_incidence,
)


def _spec() -> TireLocalNetworkSpec:
    return TireLocalNetworkSpec(
        circumferential_segments=6, section_segments=6,
        history_frames=2, temporal_order=1, batch_size=2,
    )


def test_local_feature_abi_contains_no_terrain_or_vehicle_state():
    spec = _spec();names = tire_local_feature_names(spec)
    assert not any("terrain" in name or "chassis" in name or "engine" in name for name in names)
    assert {"d0.external_force_x_n", "linear_acceleration_x_m_s2",
            "skin_thickness_m", "gas_pressure_pa"} <= set(names)
    b, t, u, v = 2, 2, 6, 6
    packed = pack_tire_local_features(
        spec,
        membrane_history=np.zeros((b, t, u, v, 6)),
        boundary_force_history=np.zeros((b, t, u, v, 4)),
        rest_skin_position=np.zeros((u, v, 3)),
        tire_thermodynamic_state=np.zeros((b, 3)),
        hub_local_motion=np.zeros((b, 9)),
        material_state=np.zeros((b, 6)), dt=1 / 600,
    )
    assert packed.shape == spec.input_shape


def test_teacher_is_per_vertex_acceleration_not_hub_wrench():
    now = np.zeros((2, 6, 6, 3));nxt = now.copy();nxt[..., 1] = .5
    target = teacher_local_acceleration(now, nxt, .25)
    assert target.shape == (2, 36, 3)
    np.testing.assert_allclose(target[:, :, 1], 2.0)


def test_signed_incidence_is_the_authoritative_skin_graph():
    spec = _spec();incidence = tire_signed_incidence(spec)
    assert incidence.shape == (spec.batch_size, spec.edge_count, spec.vertex_count)
    np.testing.assert_allclose(incidence.sum(axis=2), 0.0)
    assert np.all(np.count_nonzero(incidence[0], axis=1) == 2)


def test_local_operator_builds_one_abstract_tensor_graph_over_the_batch():
    spec = _spec();graphs = build_tire_local_training_graphs(spec)
    assert graphs.manifest["output_shape"] == list(spec.output_shape)
    assert graphs.manifest["surface_free_mode"].startswith("all external")
    assert graphs.manifest["hub_wrench"].startswith("emergent")
    assert graphs.manifest["edge_count"] == spec.edge_count
    assert "signed_skin_incidence" in graphs.forward_input_value_ids
    assert set(graphs.parameter_names) == {
        "edge_law.weight", "node_law.weight", "node_law.bias",
    }
    assert len(graphs.forward_graph.roots) == 2
