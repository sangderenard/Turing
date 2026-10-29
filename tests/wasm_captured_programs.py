"""Wasm test programs built by the real capture, not written by hand.

A ``FusedProgram`` assembled from ``OpStep`` literals has no producer, so it
carries no dtype record and the backend refuses it by name. These helpers run
the source through the AOT front end and project its public numeric program,
so every feed and value carries the record the capture writes.
"""

from __future__ import annotations

import ast

import numpy as np

from src.common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    project_public_numerical_program,
)

SAMPLE_FLOATS = np.asarray([1.0, 5.0, -2.0, 0.5])
SAMPLE_OTHER_FLOATS = np.asarray([3.0, 1.0, 4.0, -0.25])


def captured_program(source: str, **feeds):
    """Capture ``source``'s single function with ``feeds`` as runtime inputs."""

    (function,) = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    ]
    compilation = compile_ast_aot(
        source, function.name, feeds, precompile_only=True,
        remove_loops=True, mutable_parameters=tuple(feeds),
    )
    return project_public_numerical_program(compilation)


def captured_floats(source: str, *names: str):
    """Capture ``source`` with float64 sample feeds named ``names``."""

    samples = (SAMPLE_FLOATS, SAMPLE_OTHER_FLOATS)
    return captured_program(
        source, **{name: samples[index % 2] for index, name in enumerate(names)}
    )
