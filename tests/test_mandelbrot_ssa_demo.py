from __future__ import annotations

from types import SimpleNamespace

from src.common.tensors.accelerator_backends import demo_mandelbrot_ssa
from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.compiler.control_source import ControlProgram, StatementBlock


class _Deployment:
    source_node_count = 12
    primitive_count = 3
    dispatch_count = 1
    function_shells = {}
    callsite_function_shells = {}
    module_shell = None

    def __init__(self):
        self.compiled_shell_program = SimpleNamespace(
            program=FusedProgram(
                version=1,
                feeds={0},
                steps=[OpStep(0, "neg", [0], {}, 1)],
                outputs={"result": 1},
                meta={
                    0: Meta((4,), "float32", "glsl"),
                    1: Meta((4,), "float32", "glsl"),
                },
            ),
            stages=(),
        )
        self.shell_control_program = ControlProgram(
            StatementBlock(("__scheduled_region_0__",)),
            region_indices=(0,),
        )
        self.compiled = False
        self.captured = False
        self.released = False

    def compile_process_graph(self):
        self.compiled = True

    def capture_fused_programs(self, feeds):
        assert feeds["width"] == 8
        assert feeds["height"] == 8
        assert feeds["iterations"] == 2
        self.captured = True

    def release(self):
        self.released = True


def test_complete_demo_uses_existing_optimized_precompile_and_control(
    monkeypatch,
):
    deployment = _Deployment()
    monkeypatch.setattr(
        demo_mandelbrot_ssa,
        "require_gl_context",
        lambda: None,
    )
    monkeypatch.setattr(
        demo_mandelbrot_ssa,
        "build_parametric_mandelbrot_glsl_deployment",
        lambda *args, **kwargs: (deployment, object()),
    )

    audit = demo_mandelbrot_ssa.audit_complete_mandelbrot_ssa(
        width=8,
        height=8,
        iterations=2,
        frame_count=2,
    )

    assert deployment.compiled
    assert deployment.captured
    assert deployment.released
    assert audit.source_nodes == 12
    assert audit.scheduled_nodes == 3
    assert audit.dispatch_regions == 1
    assert audit.precompile_steps == 1
    assert audit.lowering.complete
    assert set(audit.lowering.module.functions) == {
        "mandelbrot_recording_numerical",
        "mandelbrot_recording_control",
    }


def test_demo_feeds_cover_a_frame_batch():
    feeds = demo_mandelbrot_ssa.mandelbrot_recording_feeds(
        16,
        8,
        3,
        4,
    )

    assert feeds["unit_x"].shape == (8 * 16,)
    assert feeds["unit_y"].shape == (8 * 16,)
    for name in (
        "center_x",
        "center_y",
        "span",
        "family_mix",
        "julia_x",
        "julia_y",
        "palette_phase",
        "color_drive",
    ):
        assert feeds[name].shape == (4,)
