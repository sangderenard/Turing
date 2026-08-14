import numpy as np
import pytest
import threading
import time
from types import SimpleNamespace

from src.compiler.evolution_metagraph import (
    EvolutionComponentRef,
    EvolutionMetaGraph,
    record_evolution,
    record_fused_program_evolution,
)
from src.compiler.control_source import (
    ControlProgram,
    LoopBlock,
    StatementBlock,
)
from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.precompile_to_ssa import (
    lower_control_program_to_ssa,
    lower_fused_program_to_ssa,
)
from src.compiler.autogenesis import compile_source_autogenesis
from src.rendering.precompiled_graph import (
    ExpansionEmergencyClamp,
    ExpansionLimitExceeded,
    EvolutionVisualProjector,
    LiveEvolutionEventBuffer,
    SpringVisualSimulation,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_metagraph_records_consumption_and_spawn_without_owning_ir():
    meta = EvolutionMetaGraph()
    ingestion = meta.open_graph("process-graph", "ingestion")
    precompile = meta.open_graph("fused-program", "numeric region")
    source = meta.component(ingestion, 7, label="multiply", kind="operation")
    target = meta.component(
        precompile,
        7,
        label="mul",
        kind="operation",
        consumes=(source,),
    )
    meta.handoff(target, (source,), transformation="precompile")

    snapshot = meta.snapshot()
    assert len(snapshot.graphs) == 2
    assert {event.kind for event in snapshot.events} >= {
        "graph-open", "component-spawn", "component-handoff"
    }
    handoff = next(event for event in snapshot.events if event.kind == "component-handoff")
    assert handoff.component == EvolutionComponentRef(precompile.id, "7")
    assert handoff.sources == (EvolutionComponentRef(ingestion.id, "7"),)


def test_process_graph_updates_optional_metagraph_in_the_hot_ingestion_loop():
    with record_evolution() as meta:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast("""
def kernel(x, y):
    return (x + y) * 3
""")

    snapshot = meta.snapshot()
    process_graph = next(item for item in snapshot.graphs if item.stage == "process-graph")
    spawned = [
        event for event in snapshot.events
        if event.kind == "component-spawn" and event.graph == process_graph
    ]
    links = [
        event for event in snapshot.events
        if event.kind == "component-link" and event.graph == process_graph
    ]
    assert len(spawned) == graph.G.number_of_nodes()
    assert len(links) == graph.G.number_of_edges()
    assert any(event.detail["kind"] == "add" for event in spawned)


def test_exact_value_identity_spawns_precompile_then_ssa_geometry():
    program = FusedProgram(
        version=1,
        feeds={0, 1},
        steps=[
            OpStep(0, "add", [0, 1], {}, 2),
            OpStep(1, "sin", [2], {}, 3),
        ],
        outputs={"result": 3},
    )
    with record_evolution() as meta:
        source = meta.open_graph("process-graph", "semantic ingestion")
        for value_id, label in ((0, "x"), (1, "y"), (2, "add"), (3, "sin")):
            meta.component(source, value_id, label=label, kind="operation")
        precompile = record_fused_program_evolution(program, source_graph=source)
        function, shortfalls = lower_fused_program_to_ssa(program)

    assert not shortfalls
    snapshot = meta.snapshot()
    ssa = meta.graph_for_artifact(function)
    assert precompile is not None and ssa is not None
    handoffs = [event for event in snapshot.events if event.kind == "component-handoff"]
    assert any(
        event.sources == (EvolutionComponentRef(source.id, "2"),)
        and event.component == EvolutionComponentRef(precompile.id, "2")
        for event in handoffs
    )
    assert any(
        event.sources == (EvolutionComponentRef(precompile.id, "2"),)
        and event.component == EvolutionComponentRef(ssa.id, "2")
        for event in handoffs
    )


def test_control_blocks_are_consumed_as_ssa_instructions_in_the_hot_loop():
    body = StatementBlock(("__scheduled_region_7__",))
    loop = LoopBlock("i", "0", "3", "1", body)
    program = ControlProgram(loop, region_indices=(7,))

    with record_evolution() as meta:
        function, shortfalls = lower_control_program_to_ssa(
            program,
            first_value_id=100,
        )

    assert shortfalls == ()
    control_graph = meta.graph_for_artifact(program)
    ssa_graph = meta.graph_for_artifact(function)
    loop_component = meta.component_for_artifact(loop)
    body_component = meta.component_for_artifact(body)
    assert control_graph is not None and control_graph.stage == "control-ir"
    assert ssa_graph is not None and ssa_graph.stage == "ssa"
    assert loop_component is not None and body_component is not None

    handoffs = [
        event
        for event in meta.snapshot().events
        if event.kind == "component-handoff"
        and event.detail.get("transformation") == "control-ir-to-ssa"
    ]
    assert any(event.sources == (loop_component,) for event in handoffs)
    assert any(event.sources == (body_component,) for event in handoffs)
    assert all(event.component.graph_id == ssa_graph.id for event in handoffs)


def test_handoff_geometry_spawns_at_source_then_moves_to_stage_anchor():
    meta = EvolutionMetaGraph()
    source_graph = meta.open_graph("process-graph", "source")
    target_graph = meta.open_graph("ssa", "target")
    source = meta.component(source_graph, 1, label="add", kind="operation")
    target = meta.component(target_graph, 1, label="Add", kind="instruction")
    meta.handoff(target, (source,), transformation="lower")

    projector = EvolutionVisualProjector()
    simulation = SpringVisualSimulation(projector.graph())
    target_key = f"{target.graph_id}/{target.local_id}"
    source_key = f"{source.graph_id}/{source.local_id}"
    for event in meta.snapshot().events:
        simulation.replace_graph(projector.apply(event))

    index = {key: i for i, key in enumerate(simulation._keys)}
    assert np.array_equal(
        simulation.positions[index[target_key]],
        simulation.positions[index[source_key]],
    )
    before = simulation.positions[index[target_key]].copy()
    for _ in range(30):
        simulation.step(1.0 / 60.0)
    assert not np.array_equal(simulation.positions[index[target_key]], before)


def test_class_branch_growth_reports_without_perturbing_spring_physics():
    meta = EvolutionMetaGraph()
    source_graph = meta.open_graph("process-graph", "source")
    lowered_graph = meta.open_graph("ssa", "lowered")
    class_ref = meta.component(
        source_graph,
        "class",
        label="LeakyLayer",
        kind="ClassDef",
        attributes={
            "source_class": "LeakyLayer",
            "source_scope": ("LeakyLayer",),
            "boundary_rule": "python.LeakyLayer.stopgap",
        },
    )
    previous = class_ref
    for index in range(12):
        target = meta.component(
            lowered_graph,
            index,
            label=f"expanded_{index}",
            kind="instruction",
        )
        meta.handoff(target, (previous,), transformation="expand")
        previous = target

    projector = EvolutionVisualProjector()
    simulation = SpringVisualSimulation(projector.graph())
    for event in meta.snapshot().events:
        simulation.replace_graph(projector.apply(event))
    before_positions = simulation.positions.copy()
    before_velocities = simulation.velocities.copy()
    simulation.set_haze(projector.haze_scores())

    report = projector.report()
    assert report.hotspots[0].owner == "class LeakyLayer"
    assert report.hotspots[0].node_count == 13
    assert report.hotspots[0].depth == 12
    assert report.hotspots[0].height == 12
    assert report.hotspots[0].stages == (("process-graph", 1), ("ssa", 12))
    assert report.hotspots[0].boundaries == (
        "python.LeakyLayer.stopgap",
    )
    assert np.array_equal(simulation.positions, before_positions)
    assert np.array_equal(simulation.velocities, before_velocities)
    class_key = f"{class_ref.graph_id}/{class_ref.local_id}"
    class_index = simulation._keys.index(class_key)
    assert simulation.fluxspring.visual_haze[class_index] > 0.0


def test_emergency_growth_clamp_aborts_the_event_thread_with_culprit():
    meta = EvolutionMetaGraph()
    guard = ExpansionEmergencyClamp(
        max_depth=1,
        max_height=1,
        max_nodes_per_branch=100,
        check_interval=1,
    )
    meta.subscribe(guard)
    source_graph = meta.open_graph("process-graph", "source")
    target_graph = meta.open_graph("ssa", "target")
    root = meta.component(
        source_graph,
        "class",
        label="UnboundedLayer",
        kind="ClassDef",
        attributes={"source_class": "UnboundedLayer"},
    )
    first = meta.component(target_graph, 1, label="one", kind="instruction")
    meta.handoff(first, (root,), transformation="expand")
    second = meta.component(target_graph, 2, label="two", kind="instruction")

    with pytest.raises(ExpansionLimitExceeded, match="class UnboundedLayer") as caught:
        meta.handoff(second, (first,), transformation="expand")
    assert caught.value.depth == 2
    assert caught.value.height == 2


def test_live_event_buffer_preserves_order_and_backpressures_compiler():
    stream = LiveEvolutionEventBuffer(max_backlog=1)
    stream.activate()
    stream.publish("node")
    published_edge = threading.Event()

    def publish_edge() -> None:
        stream.publish("edge")
        published_edge.set()

    worker = threading.Thread(target=publish_edge)
    worker.start()
    time.sleep(0.05)
    assert not published_edge.is_set()
    assert stream.pop() == "node"
    assert published_edge.wait(1.0)
    assert stream.pop() == "edge"
    stream.close()
    worker.join(timeout=1.0)


def test_live_event_buffer_repairs_replay_startup_race_by_compiler_sequence():
    stream = LiveEvolutionEventBuffer(max_backlog=4)
    stream.publish(SimpleNamespace(sequence=1, kind="component-spawn"))
    stream.publish(SimpleNamespace(sequence=0, kind="graph-open"))
    stream.activate()

    assert stream.pop().sequence == 0
    assert stream.pop().sequence == 1
    stream.close()


def test_projector_reveals_node_then_edge_on_their_exact_events():
    meta = EvolutionMetaGraph()
    graph = meta.open_graph("process-graph", "source")
    first = meta.component(graph, "first", label="first", kind="Name")
    second = meta.component(graph, "second", label="second", kind="Call")
    meta.relationship(graph, first, second, role="argument")

    projector = EvolutionVisualProjector()
    snapshots = [projector.apply(event) for event in meta.snapshot().events]

    assert [len(snapshot.nodes) for snapshot in snapshots] == [0, 1, 2, 2]
    assert [len(snapshot.edges) for snapshot in snapshots] == [0, 0, 0, 1]


def test_autogenesis_exposes_compiler_run_before_aot_source_graph():
    meta = EvolutionMetaGraph()
    compile_source_autogenesis(
        "def kernel():\n    return 1\n",
        "kernel",
        {},
        metagraph=meta,
        final_target=None,
    )
    snapshot = meta.snapshot()
    run_graph = next(graph for graph in snapshot.graphs if graph.stage == "compiler-run")
    run_events = [event for event in snapshot.events if event.graph == run_graph]

    assert run_events[0].kind == "graph-open"
    assert run_events[1].kind == "component-spawn"
    assert run_events[1].detail["label"] == "compile requested"
    assert any(
        event.kind == "component-link"
        and event.detail.get("role") == "phase-order"
        for event in run_events
    )


def test_isolated_compiler_events_replay_without_renumbering():
    compiler = EvolutionMetaGraph()
    graph = compiler.open_graph("process-graph", "isolated")
    compiler.component(graph, "node", label="node", kind="Name")
    visual = EvolutionMetaGraph()
    delivered = []
    visual.subscribe(delivered.append)

    for event in compiler.snapshot().events:
        visual.ingest_event(event)

    assert [event.sequence for event in delivered] == [0, 1]
    assert visual.snapshot().graphs == (graph,)
    assert visual.snapshot().components[0].label == "node"
