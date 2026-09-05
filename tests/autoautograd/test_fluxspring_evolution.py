import numpy as np

from src.common.tensors.autoautograd.fluxspring import (
    EvolutionEdge,
    EvolutionNode,
    MultiNetworkFluxSpring,
)


def test_multiple_networks_bud_from_live_source_and_grow():
    physics = MultiNetworkFluxSpring(seed=3, growth_seconds=0.5)
    source = EvolutionNode("source", "ingestion", "input", "process-graph")
    target = EvolutionNode("target", "ssa", "operation", "ssa")
    physics.synchronize((source,), ())
    for _ in range(5):
        physics.step(0.02)
    source_position = physics.positions[0].copy()

    physics.synchronize((source, target), ())
    assert not np.array_equal(physics.positions[1], source_position)
    physics.synchronize(
        (source, target),
        (EvolutionEdge("source", "target", "handoff"),),
    )

    assert np.array_equal(physics.positions[1], physics.positions[0])
    assert physics.age[1] == 0.0
    initial_size = physics.frame().sizes[1]
    for _ in range(8):
        physics.step(0.02)
    frame = physics.frame()
    assert frame.sizes[1] > initial_size
    assert set(physics._network_order) == {"ingestion", "ssa"}


def test_empty_evolution_waits_for_its_first_network():
    physics = MultiNetworkFluxSpring()

    physics.synchronize((), ())
    physics.step(1.0 / 60.0)

    frame = physics.frame()
    assert frame.positions.shape == (0, 3)
    assert physics._system is None


def test_simulated_fluxspring_edges_publish_live_activation():
    physics = MultiNetworkFluxSpring(seed=7)
    physics.synchronize(
        (
            EvolutionNode("input", "network-a", "input"),
            EvolutionNode("result", "network-a", "operation"),
        ),
        (EvolutionEdge("input", "result", "data"),),
    )

    for _ in range(10):
        physics.step(0.02)

    frame = physics.frame()
    assert physics._spec is not None
    assert frame.edge_activation.shape == (1,)
    assert np.isfinite(frame.edge_activation).all()
    assert abs(float(frame.edge_activation[0])) > 0.0
    assert frame.edge_colors[0, 3] > 0.22


def test_visual_haze_is_observer_state_not_a_force_term():
    physics = MultiNetworkFluxSpring(seed=5)
    physics.synchronize(
        (EvolutionNode("source", "program"), EvolutionNode("child", "program")),
        (EvolutionEdge("source", "child"),),
    )
    before_positions = physics.positions.copy()
    before_velocities = physics.velocities.copy()

    physics.set_visual_haze({0: 0.8, 1: -0.4})

    assert physics.visual_haze == {0: 0.8, 1: -0.4}
    assert physics._system.visual_haze == physics.visual_haze
    assert np.array_equal(physics.positions, before_positions)
    assert np.array_equal(physics.velocities, before_velocities)
