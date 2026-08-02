from __future__ import annotations

import contextlib
import io

import numpy as np

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends.demo_mandelbrot_fusion import (
    build_parametric_mandelbrot_glsl_deployment,
    normalized_plane,
)
from src.common.tensors.accelerator_backends.mandelbrot_encoder_program import (
    mandelbrot_frame_program,
)
from src.compiler.glsl_deployment_strategy import _walk_planned_shells


def test_composed_mandelbrot_child_matches_numpy_with_planned_glsl_phases():
    width, height, iterations = 16, 8, 4
    unit_x, unit_y = normalized_plane(width, height)
    feeds = {
        "unit_x": unit_x,
        "unit_y": unit_y,
        "center_x": np.asarray((-0.74,), dtype=np.float32),
        "center_y": np.asarray((0.13,), dtype=np.float32),
        "span": np.asarray((0.004,), dtype=np.float32),
        "family_mix": np.asarray((0.0,), dtype=np.float32),
        "julia_x": np.asarray((0.0,), dtype=np.float32),
        "julia_y": np.asarray((0.0,), dtype=np.float32),
        "palette_phase": np.asarray((0.0,), dtype=np.float32),
        "color_drive": np.asarray((0.5,), dtype=np.float32),
        "width": width,
        "height": height,
        "iterations": iterations,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        deployment, _ = build_parametric_mandelbrot_glsl_deployment(
            iterations
        )
        deployment.compile_process_graph()

    child = next(
        shell
        for shell in _walk_planned_shells(deployment)
        if shell.process_graph.G.graph.get("function_name")
        == "mandelbrot_frame_program"
        and "@callsite-" not in shell.profile_path
    )
    try:
        # This test owns the composed numerical child.  Capturing the complete
        # recording/JPEG parent first is both unrelated and intentionally
        # rejected by its own truthful hierarchical-ABI completeness gate.
        child.capture_fused_programs(feeds)
        glsl = child.execute_named(feeds)
        installed = child.installed_control_shell
        assert installed.artifact.device_resident
        assert installed.artifact.contiguous_plan.dispatch_count == 3
        # Contiguation still records the three host-dispatch phases that would
        # be required by a C shell.  A GLSL shell executes those phases inside
        # one resident workgroup program and therefore has one physical launch.
        assert installed.last_dispatches == 1
        with AbstractTensor.use_backend("numpy"):
            numpy_frames, numpy_counts = mandelbrot_frame_program(
                *(
                    AbstractTensor.tensor(feeds[name])
                    for name in (
                        "unit_x",
                        "unit_y",
                        "center_x",
                        "center_y",
                        "span",
                        "family_mix",
                        "julia_x",
                        "julia_y",
                        "palette_phase",
                        "color_drive",
                    )
                ),
                width=width,
                height=height,
                iterations=iterations,
            )
        np.testing.assert_allclose(
            glsl["frames"].numpy(),
            numpy_frames.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            glsl["counts"].numpy(),
            numpy_counts.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
    finally:
        deployment.release()
