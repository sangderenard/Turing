import ast
import queue
import threading

import numpy as np
import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.nodus_graph_ir import process_graph_to_nodus_graph_ir
from src.rendering.precompiled_graph import (
    FluxSpringVisualSimulation,
    SpringVisualSimulation,
    VisualEdge,
    VisualGraph,
    VisualGraphDelta,
    VisualNode,
    _projection_update_process,
    visual_graph_from_ir,
)
from src.rendering.gpu_resident_bound_spring import (
    GpuResidentBoundSpringSimulation,
)
from src.rendering.opengl_render.fluxspring_shader import load_fluxspring_graph_shaders
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.graph.graph_express2 import _annotate_visual_source_owners
from src.transmogrifier.ssa import BasicBlock, Function, IRModule, Instr, SSAValue


def _fused_program():
    return FusedProgram(
        version=1,
        feeds={1, 2},
        steps=[
            OpStep(0, "add", [1, 2], {}, 3),
            OpStep(1, "sin", [3], {}, 4),
        ],
        outputs={"result": 4},
    )


def test_short_sympy_compile_publishes_final_visual_census():
    from src.compiler.autogenesis import compile_sympy_autogenesis

    result = compile_sympy_autogenesis(
        "Eq(zeta(s),2**s*pi**(s-1)*sin(pi*s/2)*gamma(1-s)*zeta(1-s))"
    )
    requests = queue.Queue()
    publications = queue.Queue()
    requests.put(tuple(result.metagraph.snapshot().events))
    requests.put(None)

    _projection_update_process(
        requests, publications, threading.Event(), top_k=8, report_hz=1.0,
    )

    delta, _last_event, report, report_changed = publications.get_nowait()
    assert report_changed
    # Telemetry retains the compiler-phase event in the compiler census even
    # though the physical visual projection deliberately omits that marker.
    assert report.node_count == len(delta.nodes_added) + 1
    assert report.edge_count == len(delta.edges_added)
    assert report.node_count > 20
    assert report.edge_count > 0
    assert all(node.kind != "compiler-phase" for node in delta.nodes_added)
    process_nodes = [node for node in delta.nodes_added if node.group == "process-graph"]
    ssa_nodes = [node for node in delta.nodes_added if node.group == "ssa"]
    package_nodes = [node for node in delta.nodes_added if node.group == "ir-package"]
    assert process_nodes and all(node.schedule_group is not None for node in process_nodes)
    assert ssa_nodes and all(node.schedule_group is not None for node in ssa_nodes)
    assert package_nodes and all(node.schedule_group is None for node in package_nodes)
    assert all(node.state == "finalized" for node in delta.nodes_added)


def test_fused_program_and_wrappers_share_visual_graph_contract():
    graph = visual_graph_from_ir(_fused_program())
    assert graph.source_kind == "fused-program"
    assert {node.label for node in graph.nodes} >= {"feed 1", "add", "sin", "result"}
    assert {(edge.source, edge.target) for edge in graph.edges} >= {
        ("1", "3"),
        ("2", "3"),
        ("3", "4"),
        ("4", "output:result"),
    }

    class Wrapper:
        compiled_shell_program = _fused_program()

    assert visual_graph_from_ir(Wrapper()) == visual_graph_from_ir(Wrapper.compiled_shell_program)


def test_ssa_module_uses_values_and_control_edges_without_new_ir():
    x = SSAValue(1)
    y = SSAValue(2)
    total = SSAValue(3)
    result = SSAValue(4)
    module = IRModule(functions={
        "kernel": Function(
            "kernel",
            [x, y],
            {
                "entry": BasicBlock(
                    "entry",
                    [Instr("add", [x, y], total)],
                    ["exit"],
                ),
                "exit": BasicBlock("exit", [Instr("sin", [total], result)]),
            },
        )
    })
    graph = visual_graph_from_ir(module)
    assert graph.source_kind == "ssa"
    assert {node.label for node in graph.nodes} >= {"add", "sin"}
    assert any(edge.role == "control" for edge in graph.edges)


def test_process_graph_accessor_publishes_lock_safe_live_revisions():
    graph = ProcessGraph(materialize_memory=False)
    accessor = graph.graph_accessor()
    initial = accessor.snapshot()
    revisions = []
    unsubscribe = accessor.subscribe(revisions.append, replay=False)

    worker = threading.Thread(target=lambda: graph.build_from_ast("""
def kernel(x, y):
    return (x + y) * 3
"""))
    worker.start()
    changed = accessor.wait_for_change(initial.revision, timeout=2.0)
    worker.join(timeout=5.0)
    unsubscribe()
    final = accessor.snapshot()

    assert changed.revision > initial.revision
    assert final.revision >= changed.revision
    assert final.graph.number_of_nodes() > 0
    assert revisions
    assert [item.revision for item in revisions] == sorted(
        item.revision for item in revisions
    )
    assert revisions[-1].graph.number_of_nodes() == final.graph.number_of_nodes()
    assert visual_graph_from_ir(accessor).revision == final.revision


def test_nodus_graph_ir_returns_to_same_visual_surface():
    process = ProcessGraph(materialize_memory=False)
    process.build_from_ast("""
def kernel(x, y):
    return x + y
""")
    graph = visual_graph_from_ir(process_graph_to_nodus_graph_ir(process))
    assert graph.source_kind == "nodus-graph-ir"
    assert any(node.label == "add" for node in graph.nodes)
    assert graph.edges


def test_fluxspring_surface_keeps_live_edges_colors_and_motion_headless():
    simulation = FluxSpringVisualSimulation(visual_graph_from_ir(_fused_program()))
    before = simulation.positions.copy()
    simulation.step(1.0 / 60.0)
    first = simulation.layers(0.0)
    first_activation = simulation.fluxspring.edge_activation.copy()
    simulation.step(1.0 / 60.0)
    second = simulation.layers(1.0)

    assert not np.array_equal(before, simulation.positions)
    assert first["lines"].positions.shape[0] == len(simulation.graph.edges) * 2
    assert first["points"].colors.shape == (len(simulation.graph.nodes), 4)
    assert not np.array_equal(first["points"].positions, second["points"].positions)
    assert simulation.fluxspring._spec is not None
    assert first_activation.shape == (len(simulation.graph.edges),)


def test_gpu_resident_surface_preserves_free_physics_on_growth():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA-resident spring test requires a CUDA device")
    nodes = (
        VisualNode("source", "source", network="compiler"),
        VisualNode("target", "target", network="ssa"),
    )
    graph = VisualGraph(
        nodes,
        (VisualEdge("source", "target", "handoff"),),
        "evolution-metagraph",
    )
    simulation = GpuResidentBoundSpringSimulation(
        graph, node_capacity=32, edge_capacity=64, observation_hz=240
    )
    initial = simulation._position_pages[simulation._current_page][:2].clone()
    for _ in range(120):
        simulation.step(1.0 / 240.0)
    torch.cuda.synchronize()
    positions = simulation._position_pages[simulation._current_page][:2]
    velocities = simulation._velocity_pages[simulation._current_page][:2]
    assert not torch.allclose(positions, initial)
    assert velocities.norm().item() > 0.0

    rest_length = simulation._base_rest[0].clone()
    velocity = velocities.clone()
    resident_pages = tuple(simulation._position_pages)
    simulation.replace_graph(VisualGraph(
        nodes + (VisualNode("child", "child", network="ssa"),),
        graph.edges + (VisualEdge("target", "child", "data"),),
        graph.source_kind,
    ))

    torch.cuda.synchronize()
    topology_ready = simulation._topology_publication[7]
    assert topology_ready is not None
    assert tuple(simulation._position_pages) == resident_pages
    assert simulation.node_count == 3
    assert simulation.edge_count == 2
    assert torch.equal(simulation._base_rest[0], rest_length)
    assert torch.allclose(
        simulation._velocity_pages[simulation._current_page][:2], velocity
    )
    simulation.step(1.0 / 240.0)
    torch.cuda.synchronize()
    assert simulation._applied_topology_ready_event is topology_ready


def test_velocity_normalized_dt_uses_previous_state_max_without_reset():
    graph = VisualGraph(
        (VisualNode("node", "node", network="physics"),),
        (),
        "velocity-normalization",
    )
    simulation = GpuResidentBoundSpringSimulation(
        graph,
        node_capacity=4,
        edge_capacity=4,
        device="cpu",
        target_max_velocity=1.0,
        min_dt_scale=0.1,
        max_dt_scale=2.0,
    )
    simulation._velocity_pages[simulation._current_page][0, 0] = 10.0

    simulation.step(0.25)

    assert float(simulation._previous_max_velocity) == 10.0
    assert float(simulation._dt_scale) == pytest.approx(0.1)
    assert float(simulation._effective_dt) == pytest.approx(0.025)


def test_resident_storage_grows_without_resetting_existing_state():
    nodes = tuple(VisualNode(key, key, network="math") for key in "abcde")
    initial = VisualGraph(
        nodes[:2],
        (VisualEdge("a", "b", "data"),),
        "growth-test",
    )
    simulation = GpuResidentBoundSpringSimulation(
        initial,
        node_capacity=2,
        edge_capacity=1,
        device="cpu",
    )
    for _ in range(8):
        simulation.step(1.0 / 240.0)
    old_positions = [page[:2].clone() for page in simulation._position_pages]
    old_velocities = [page[:2].clone() for page in simulation._velocity_pages]
    old_video = simulation.video
    simulation.replace_graph(VisualGraph(
        nodes,
        (
            VisualEdge("a", "b", "data"),
            VisualEdge("b", "c", "data"),
            VisualEdge("c", "d", "data"),
            VisualEdge("d", "e", "data"),
        ),
        "growth-test",
    ))

    assert simulation.node_capacity >= 5
    assert simulation.edge_capacity >= 4
    assert simulation.video is not old_video
    assert old_video in simulation._retired_video_buffers
    for page, expected in zip(simulation._position_pages, old_positions):
        assert simulation.torch.equal(page[:2], expected)
    for page, expected in zip(simulation._velocity_pages, old_velocities):
        assert simulation.torch.equal(page[:2], expected)


def test_resident_topology_and_physics_reject_split_thread_ownership():
    graph = VisualGraph(
        (VisualNode("a", "a", network="math"),),
        (),
        "ownership-test",
    )
    simulation = GpuResidentBoundSpringSimulation(
        graph,
        node_capacity=4,
        edge_capacity=4,
        device="cpu",
    )
    simulation.bind_mutation_owner()
    failures = []

    def mutate_from_another_thread():
        try:
            simulation.step(1.0 / 240.0)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=mutate_from_another_thread)
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "must share one owning thread" in str(failures[0])


def test_delta_growth_seeds_new_nodes_from_incident_live_geometry():
    graph = VisualGraph(
        (
            VisualNode("left", "left", network="base"),
            VisualNode("right", "right", network="base"),
        ),
        (),
        "evolution-metagraph",
        1,
    )
    simulation = GpuResidentBoundSpringSimulation(
        graph, node_capacity=16, edge_capacity=16, device="cpu"
    )
    for page in simulation._position_pages:
        page[0] = simulation.torch.tensor([0.0, 0.0, 0.0])
        page[1] = simulation.torch.tensor([2.0, 0.0, 0.0])
    simulation.apply_delta(VisualGraphDelta(
        2,
        "evolution-metagraph",
        (VisualNode("middle", "middle", network="grown"),),
        (),
        (
            VisualEdge("left", "middle", "data"),
            VisualEdge("right", "middle", "data"),
        ),
    ))

    position = simulation._position_pages[simulation._current_page][2].numpy()
    assert position[0] == pytest.approx(1.0, abs=0.3)
    assert simulation._base_rest[:2].numpy().tolist() == pytest.approx([1.05, 1.05])


def test_gpu_physics_does_not_wait_for_unconsumed_video_pages():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA-resident spring test requires a CUDA device")
    graph = VisualGraph(
        (
            VisualNode("source", "source", network="compiler"),
            VisualNode("target", "target", network="ssa"),
        ),
        (VisualEdge("source", "target", "handoff"),),
        "evolution-metagraph",
    )
    simulation = GpuResidentBoundSpringSimulation(
        graph, node_capacity=32, edge_capacity=64, observation_hz=1_000_000
    )

    # Fill both observation pages without ever running the video consumer.
    for _ in range(2):
        simulation._next_observation = 0.0
        simulation.step(1.0 / 240.0)
    torch.cuda.synchronize()
    simulation._poll_video_copies()
    assert all(page.ready for page in simulation.video.frames)

    # A saturated display mailbox drops every further observation.  Physics
    # must continue to enqueue and complete steps without freeing either page.
    before = simulation.steps
    for _ in range(240):
        simulation._next_observation = 0.0
        simulation.step(1.0 / 240.0)
    torch.cuda.synchronize()
    assert simulation.steps == before + 240
    assert all(page.ready for page in simulation.video.frames)


def test_compiler_schedule_pulse_physically_contracts_active_gpu_edges():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA-resident spring test requires a CUDA device")
    nodes = (
        VisualNode("a", "a", kind="op", schedule_group=0),
        VisualNode("b", "b", kind="op", schedule_group=0),
        VisualNode("c", "c", kind="op", schedule_group=0),
        VisualNode("d", "d", kind="op", schedule_group=1),
    )
    graph = VisualGraph(
        nodes,
        (
            VisualEdge("a", "b", "data"),
            VisualEdge("a", "c", "data"),
            VisualEdge("b", "d", "control"),
            VisualEdge("c", "d", "control"),
        ),
        "schedule-test",
    )
    simulation = GpuResidentBoundSpringSimulation(
        graph,
        node_capacity=32,
        edge_capacity=64,
        damping=0.902,
        edge_pulse_mode="compiler-schedule",
        schedule_hz=1.0,
    )
    for _ in range(120):
        simulation.step(1.0 / 240.0)
    torch.cuda.synchronize()
    assert simulation.damping == pytest.approx(0.902)
    assert simulation.active_schedule_group == 0
    assert simulation.schedule_pulse > 0.99
    assert torch.all(simulation._rest_state[:2] < simulation._base_rest[:2])
    assert torch.allclose(
        simulation._rest_state[2:4], simulation._base_rest[2:4]
    )


def test_compiled_execution_frames_override_colliding_local_dag_groups():
    graph = VisualGraph(
        (
            VisualNode("root", "root", schedule_group=0, execution_groups=(2,)),
            VisualNode("left", "left", schedule_group=0, execution_groups=(4,)),
            VisualNode("right", "right", schedule_group=0, execution_groups=(7,)),
        ),
        (
            VisualEdge("root", "left", "control-next"),
            VisualEdge("left", "right", "control-next"),
        ),
        "compiled-execution",
    )
    simulation = GpuResidentBoundSpringSimulation(
        graph,
        node_capacity=8,
        edge_capacity=8,
        device="cpu",
    )

    assert simulation.execution_plan_active
    assert simulation.schedule_group_count == 8
    assert simulation._edge_schedule_group_host[:2].tolist() == [4, 7]


def test_contraction_strength_and_response_have_independent_time_scales():
    graph = VisualGraph(
        (
            VisualNode("a", "a", schedule_group=0),
            VisualNode("b", "b", schedule_group=0),
        ),
        (VisualEdge("a", "b", "data"),),
        "contraction-envelope",
    )

    def evolved(response_hz):
        simulation = GpuResidentBoundSpringSimulation(
            graph,
            node_capacity=4,
            edge_capacity=4,
            device="cpu",
            schedule_hz=1.0,
            schedule_contraction=0.75,
            contraction_response_hz=response_hz,
            target_max_velocity=0.0,
        )
        for _ in range(120):
            simulation.step(1.0 / 240.0)
        return simulation

    slow = evolved(1.0)
    fast = evolved(12.0)
    assert fast.schedule_pulse > 0.99
    assert fast.schedule_contraction == pytest.approx(0.75)
    assert fast._rest_state[0] < slow._rest_state[0]


def test_frequency_normalized_yank_survives_120_hz_two_tick_envelope():
    graph = VisualGraph(
        (
            VisualNode("a", "a", schedule_group=0),
            VisualNode("b", "b", schedule_group=0),
        ),
        (VisualEdge("a", "b", "data"),),
        "yank-envelope",
    )

    def speed(yank):
        simulation = GpuResidentBoundSpringSimulation(
            graph,
            node_capacity=4,
            edge_capacity=4,
            device="cpu",
            schedule_hz=120.0,
            schedule_contraction=0.8,
            contraction_response_hz=1000.0,
            schedule_yank_impulse=yank,
            target_max_velocity=0.0,
        )
        for _ in range(2):
            simulation.step(1.0 / 240.0)
        return float(
            simulation._velocity_pages[simulation._current_page][:2].norm()
        )

    assert speed(0.06) > speed(0.0)


def test_active_edges_contract_in_geometry_despite_sampled_repulsion():
    import torch

    node_count = 200
    graph = VisualGraph(
        tuple(
            VisualNode(str(index), str(index), kind="op", schedule_group=0)
            for index in range(node_count)
        ),
        tuple(
            VisualEdge(str(index), str(index + 1), "data")
            for index in range(node_count - 1)
        ),
        "contraction-geometry",
    )
    simulation = GpuResidentBoundSpringSimulation(
        graph,
        node_capacity=256,
        edge_capacity=256,
        device="cpu",
        damping=0.0,
        schedule_hz=1.0,
        schedule_contraction=0.95,
        contraction_response_hz=1000.0,
        schedule_yank_impulse=1.0,
        target_max_velocity=0.0,
    )
    endpoints = simulation._edge_index[:simulation.edge_count]

    def mean_edge_length():
        positions = simulation._position_pages[simulation._current_page]
        return float(
            (positions[endpoints[:, 0]] - positions[endpoints[:, 1]])
            .norm(dim=1)
            .mean()
        )

    initial = mean_edge_length()
    for _ in range(240):
        simulation.step(1.0 / 480.0)

    assert simulation.schedule_pulse > 0.99
    assert mean_edge_length() < initial

def test_reusable_surface_loads_the_original_fluxspring_demo_shader():
    vertex, fragment = load_fluxspring_graph_shaders()
    assert "layout (location = 0) in vec3 in_pos" in vertex
    assert "layout (location = 2) in float in_size" in vertex
    assert "gl_PointSize = in_size" in vertex
    assert "dot(p, p) > 1.0" in fragment


def test_source_class_ownership_reaches_observer_node_metadata():
    tree = ast.parse("""
class LeakyLayer:
    def forward(self, value):
        return value + 1
""")
    _annotate_visual_source_owners(tree)
    class_node = tree.body[0]
    add_node = class_node.body[0].body[0].value
    graph = ProcessGraph(materialize_memory=False)
    class_id, _ = graph.ensure_node(class_node)
    add_id, _ = graph.ensure_node(add_node)

    assert graph.G.nodes[class_id]["source_class"] == "LeakyLayer"
    assert graph.G.nodes[class_id]["source_scope"] == ("LeakyLayer",)
    assert graph.G.nodes[add_id]["source_class"] == "LeakyLayer"
    assert graph.G.nodes[add_id]["source_scope"] == (
        "LeakyLayer", "forward"
    )
