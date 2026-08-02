from __future__ import annotations

import contextlib
import inspect
import io

import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.compiler.backend_sources import collect_backend_sources
from src.computational_world import bound_spring_stretch_force


def _feeds():
    return {
        "edge_displacement": AT.get_tensor(
            np.asarray([[-3.0, 0.0, 0.0]], dtype=np.float32)
        ),
        "source_incidence": AT.get_tensor(
            np.asarray([[1.0], [0.0]], dtype=np.float32)
        ),
        "target_incidence": AT.get_tensor(
            np.asarray([[0.0], [1.0]], dtype=np.float32)
        ),
        "rest_length": AT.get_tensor(
            np.asarray([2.0], dtype=np.float32)
        ),
        "k_stretch": AT.get_tensor(np.asarray(8.0, dtype=np.float32)),
    }


def test_first_class_python_spring_kernel_matches_legacy_hooke_direction():
    force = bound_spring_stretch_force(**_feeds())

    assert np.asarray(force.tolist()) == pytest.approx(
        np.asarray([[8.0, 0.0, 0.0], [-8.0, 0.0, 0.0]]),
        abs=1.0e-5,
    )


def test_first_class_python_spring_kernel_uses_existing_backend_surfaces():
    # Compile the same function the live state machine calls. There is no
    # shadow DSL kernel and no invented SSA operation in this qualification.
    source = inspect.getsource(bound_spring_stretch_force)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        aot = compile_ast_aot(
            source,
            "bound_spring_stretch_force",
            _feeds(),
            precompile_only=True,
        )
        emitted = collect_backend_sources(
            aot,
            numerical_name="bound_spring_stretch_force",
            control_name="bound_spring_stretch_force_control",
        )

    by_language = {item.language: item for item in emitted.sources}
    assert by_language["ssa"].available
    assert by_language["fortran"].available
    assert by_language["spirv"].available
    # These are explicit capability shortfalls in the existing tables. They
    # remain visible for error reporting instead of being retained as source.
    assert not by_language["glsl"].available
    assert "sum" in by_language["glsl"].reason
    assert not by_language["webgl"].available
    assert "sum" in by_language["webgl"].reason
