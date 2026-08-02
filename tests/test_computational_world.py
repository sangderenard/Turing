import pytest
import ast
import contextlib
import io
from pathlib import Path

from src.common.dt_system.state_table import StateTable
from src.common.dt_system.time_runtime import TimeWindowRequest
from src.computational_world import (
    ComputationalWorld,
    ComputationalWorldState,
    ProvenanceRecord,
    WorldBoundaryEvent,
    WorldStatusBatch,
    WorldTickLease,
    BoundSpringParameters,
    install_bound_spring,
    append_bound_spring,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.compiler.evolution_metagraph import EvolutionMetaGraph


def _world():
    state = ComputationalWorldState.with_player()
    table = StateTable()
    world = ComputationalWorld(state)
    lease = WorldTickLease(world, state, table)
    return state, table, world, lease


def test_inactive_lease_does_not_harvest_or_advance_world():
    state, table, _world_engine, lease = _world()
    harvested = []
    request = TimeWindowRequest(1, 0, 0.0, 0.1, 0.05)

    report = lease.advance_from_shell(
        request,
        lambda: harvested.append(True) or WorldStatusBatch(),
    )

    assert report is None
    assert harvested == []
    assert lease.runtime.current_time == 0.0
    assert state.managed_time.tolist() == [0.0]
    assert state.position.tolist() == [[0.0, 0.0, 0.0]]
    assert table.store == {}


def test_active_lease_lands_boundaries_and_commits_player_and_provenance():
    state, table, _world_engine, lease = _world()
    lease.set_active(True)
    record0 = ProvenanceRecord(
        0, "process-graph:0", "7", "component-spawn",
        artifact_reference="src/world.py:7", captured_ns=900,
    )
    record1 = ProvenanceRecord(
        1, "ssa:1", "12", "component-handoff",
        artifact_reference="src/world.py:7", captured_ns=100,
    )
    request = TimeWindowRequest(
        4, 0, 0.0, 0.3, 0.2, event_times=(0.1, 0.25),
    )

    report = lease.advance_from_shell(
        request,
        lambda: WorldStatusBatch(
            player_intent=(2.0, 0.0, 0.0),
            boundary_events=(
                WorldBoundaryEvent(0.1, (record0,)),
                WorldBoundaryEvent(0.25, (record1,)),
            ),
        ),
    )

    assert report is not None and report.exact_landing
    assert report.result.landed_boundaries == pytest.approx((0.1, 0.25))
    assert state.managed_time.tolist() == pytest.approx([0.3])
    assert state.position.tolist()[0] == pytest.approx([0.6, 0.0, 0.0])
    assert state.provenance_cursor.tolist() == [1]
    assert state.provenance_records == (record0, record1)
    assert state.artifact_references == ("src/world.py:7",)
    # captured_ns ordering never controls admission ordering.
    assert record1.captured_ns < record0.captured_ns
    assert table.get("world", "managed", "time").tolist() == pytest.approx([0.3])


def test_rejected_commit_restores_every_world_field_and_publishes_nothing():
    state, table, _world_engine, lease = _world()
    lease.set_active(True)
    record = ProvenanceRecord(0, "graph:0", "n", "component-spawn")
    batch = WorldStatusBatch(
        player_intent=(1.0, 0.0, 0.0),
        boundary_events=(WorldBoundaryEvent(0.05, (record,)),),
    )
    request = TimeWindowRequest(
        2, 0, 0.0, 0.1, 0.1, event_times=(0.05,),
    )

    with pytest.raises(RuntimeError, match="commit gate rejected"):
        lease.advance_from_shell(
            request,
            lambda: batch,
            commit_gate=lambda _request, _result: False,
        )

    assert state.position.tolist() == [[0.0, 0.0, 0.0]]
    assert state.velocity.tolist() == [[0.0, 0.0, 0.0]]
    assert state.managed_time.tolist() == [0.0]
    assert state.provenance_cursor.tolist() == [-1]
    assert state.provenance_records == ()
    assert state.pending_status == (batch,)
    assert table.store == {}

    report = lease.retry_pending(request)
    assert report.exact_landing
    assert state.position.tolist()[0] == pytest.approx([0.1, 0.0, 0.0])
    assert state.provenance_records == (record,)
    assert state.pending_status == ()


def test_bound_spring_physics_uses_managed_steps_and_existing_force_equations():
    state = ComputationalWorldState.empty()
    params = BoundSpringParameters(
        c_repulse=0.0,
        growth_rate=0.0,
        relax_rate=0.0,
        cycle_period=0.1,
        boundary_radius=100.0,
    )
    install_bound_spring(
        state,
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0, 1),),
        parameters=params,
    )
    # Stretch after rest-length capture so the legacy Hooke force pulls inward.
    state.spring_position = type(state.spring_position).tensor(
        [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype="float32"
    )
    table = StateTable()
    world = ComputationalWorld(state, spring_parameters=params)
    lease = WorldTickLease(world, state, table)
    lease.set_active(True)

    report = lease.advance_from_shell(
        TimeWindowRequest(1, 0, 0.0, 0.01, 0.01),
        WorldStatusBatch,
    )

    assert report.exact_landing
    positions = state.spring_position.tolist()
    assert positions[0][0] > -1.5
    assert positions[1][0] < 1.5
    assert table.get("engine", "spring", "position") is state.spring_position


def test_spring_activation_boundary_is_subdivided_only_by_dt_system():
    state = ComputationalWorldState.empty()
    params = BoundSpringParameters(
        c_repulse=0.0,
        growth_rate=0.0,
        relax_rate=0.0,
        cycle_period=0.01,
        boundary_radius=100.0,
    )
    install_bound_spring(
        state,
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0, 1),),
        edge_level_mask=((True,), (False,)),
        edge_type_mask=((False,), (True,)),
        edge_role_mask=((False,), (False,)),
        node_level_mask=((True, True), (False, False)),
        node_type_mask=((False, False), (True, True)),
        node_role_mask=((False, False), (False, False)),
        parameters=params,
    )
    table = StateTable()
    world = ComputationalWorld(state, spring_parameters=params)
    lease = WorldTickLease(world, state, table)
    lease.set_active(True)

    report = lease.advance_from_shell(
        TimeWindowRequest(1, 0, 0.0, 0.02, 0.02),
        WorldStatusBatch,
    )

    assert report.exact_landing
    assert report.result.rejected_attempts >= 1
    assert report.result.accepted_dts == pytest.approx((0.01, 0.01))
    assert state.spring_group_index.tolist() == [0]


def test_sparse_voxel_object_component_and_relationship_records():
    state = ComputationalWorldState.empty()
    player = state.spawn_entity(1, (0.0, 1.0, 0.0))
    terminal = state.spawn_entity(2, (2.0, 1.0, 0.0), flags=4)
    edge = state.connect_entities(player, terminal, 7, features=(0.5,))
    voxel = state.set_voxel((3, -2, 8), 11, features=(0.25, 0.75))
    component = state.attach_component(terminal, 5, features=(1.0,))

    assert (player, terminal) == (0, 1)
    assert state.edge_index.tolist() == [[0], [1]]
    assert state.edge_kind.tolist()[edge] == 7
    assert state.edge_state.tolist()[edge] == pytest.approx([0.5])
    assert state.occupied_block_coord.tolist()[voxel] == [3, -2, 8]
    assert state.component_entity.tolist()[component] == terminal
    state.validate_sparse_shapes()


def test_real_python_world_class_is_first_class_ast_state_machine_source():
    source_path = Path("src/computational_world/engine.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(tree)

    plans = graph.G.graph["state_machine_controls"]
    assert graph.G.graph["state_machine_control_shortfalls"] == ()
    plan = next(plan for plan in plans if plan.class_name == "ComputationalWorld")
    assert plan.state_field == "phase"
    assert plan.case_methods == ((0, "advance_world"),)


def test_evolution_metagraph_is_harvested_only_during_active_game_mode():
    state, _table, _world_engine, lease = _world()
    metagraph = EvolutionMetaGraph()
    graph_ref = metagraph.open_graph("process-graph", "source")
    metagraph.component(graph_ref, 7, label="add", kind="operation")
    calls = []
    original_snapshot = metagraph.snapshot

    def counted_snapshot():
        calls.append(True)
        return original_snapshot()

    metagraph.snapshot = counted_snapshot
    request = TimeWindowRequest(
        1, 0, 0.0, 0.1, 0.1, event_times=(0.05,),
    )

    assert lease.advance_from_evolution(
        request, metagraph, max_records_per_boundary=2
    ) is None
    assert calls == []
    assert state.provenance_cursor.tolist() == [-1]

    lease.set_active(True)
    report = lease.advance_from_evolution(
        request, metagraph, max_records_per_boundary=2
    )
    assert report.exact_landing
    assert calls == [True]
    assert state.provenance_cursor.tolist() == [1]
    assert [record.kind for record in state.provenance_records] == [
        "graph-open", "component-spawn"
    ]


def test_bound_spring_state_machine_holds_multiple_independent_networks():
    state = ComputationalWorldState.empty()
    params = BoundSpringParameters(
        c_repulse=1.0,
        growth_rate=0.0,
        relax_rate=0.0,
        cycle_period=0.1,
        boundary_radius=100.0,
    )
    install_bound_spring(state, ((-10.0, 0.0, 0.0),), (), parameters=params)
    second = append_bound_spring(
        state, ((10.0, 0.0, 0.0),), (), parameters=params
    )

    assert second == 1
    assert state.spring_node_network.tolist() == [0, 1]
    assert state.spring_boundary_center.tolist() == [
        [-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]
    ]
    world = ComputationalWorld(state, spring_parameters=params)
    lease = WorldTickLease(world, state, StateTable())
    lease.set_active(True)
    lease.advance_from_shell(
        TimeWindowRequest(1, 0, 0.0, 0.01, 0.01),
        WorldStatusBatch,
    )
    # One-node networks have no self-repulsion and do not repel each other.
    assert state.spring_velocity.tolist() == [
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    ]
