import numpy as np

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
from src.rendering.precompiled_graph import (
    EvolutionVisualProjector,
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
