import threading

import numpy as np

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.nodus_graph_ir import process_graph_to_nodus_graph_ir
from src.rendering.precompiled_graph import (
    FluxSpringVisualSimulation,
    SpringVisualSimulation,
    visual_graph_from_ir,
)
from src.rendering.opengl_render.fluxspring_shader import load_fluxspring_graph_shaders
from src.transmogrifier.graph.graph_express2 import ProcessGraph
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


def test_reusable_surface_loads_the_original_fluxspring_demo_shader():
    vertex, fragment = load_fluxspring_graph_shaders()
    assert "layout (location = 0) in vec3 in_pos" in vertex
    assert "layout (location = 2) in float in_size" in vertex
    assert "gl_PointSize = in_size" in vertex
    assert "dot(p, p) > 1.0" in fragment
