from __future__ import annotations

import numpy as np
import pytest

from src.compiler.control_source import (
    ControlProgram,
    ControlTarget,
    LoopBlock,
    RegionCode,
    SequenceBlock,
    StatementBlock,
    compile_python_shell,
)
from src.compiler.evaluation_patterns import (
    EvaluationPatternMap,
    EvaluationVariantKey,
)
from src.compiler.benchmark_control_shell_variants import run_profile_matrix


def _recurrent_tensor_shell(backend: str):
    logical = ControlProgram(
        SequenceBlock((
            StatementBlock(("state = AbstractTensor.tensor(values)",)),
            LoopBlock(
                "iteration",
                "0",
                "steps",
                "1",
                StatementBlock(("__scheduled_region_0__",)),
            ),
            StatementBlock(("return state",)),
        )),
        region_indices=(0,),
    )
    return compile_python_shell(
        logical,
        (RegionCode(
            0,
            ControlTarget.PYTHON,
            StatementBlock((
                "state = (state * scale + bias).tanh()",
            )),
        ),),
        function_name="recurrent_tensor_problem",
        parameters=("values", "steps", "scale", "bias"),
        abstract_tensor_backend=backend,
    )


def _coupled_tensor_shell(backend: str):
    logical = ControlProgram(
        SequenceBlock((
            StatementBlock((
                "left = AbstractTensor.tensor(values)",
                "right = AbstractTensor.tensor(values[::-1].copy())",
            )),
            LoopBlock(
                "iteration",
                "0",
                "steps",
                "1",
                StatementBlock(("__scheduled_region_0__",)),
            ),
            StatementBlock(("return left + right",)),
        )),
        region_indices=(0,),
    )
    return compile_python_shell(
        logical,
        (RegionCode(
            0,
            ControlTarget.PYTHON,
            StatementBlock((
                "next_left = left * 0.75 + right * 0.25",
                "right = right * 0.6 - left * 0.1",
                "left = next_left",
            )),
        ),),
        function_name="coupled_tensor_problem",
        parameters=("values", "steps"),
        abstract_tensor_backend=backend,
    )


@pytest.mark.parametrize(
    ("object_key", "factory", "arguments"),
    (
        (
            "nonlinear_recurrence",
            _recurrent_tensor_shell,
            (np.linspace(-2.0, 2.0, 257, dtype=np.float32), 12, 0.91, 0.07),
        ),
        (
            "coupled_recurrence",
            _coupled_tensor_shell,
            (np.linspace(-1.0, 1.0, 384, dtype=np.float32), 10),
        ),
    ),
)
def test_profiles_complex_abstract_tensor_python_interiors_across_backends(
    object_key,
    factory,
    arguments,
):
    patterns = EvaluationPatternMap()
    signature = tuple(
        (tuple(value.shape), value.dtype.str)
        if isinstance(value, np.ndarray)
        else type(value).__name__
        for value in arguments
    )
    keys = {}
    for backend in ("pure_python", "numpy", "c"):
        key = EvaluationVariantKey(
            object_key,
            signature,
            ControlTarget.PYTHON,
            backend,
        )
        keys[backend] = key
        patterns.register(key, lambda backend=backend: factory(backend))

    numpy_result = patterns.profile(
        keys["numpy"],
        arguments,
        warmups=1,
        repeats=2,
    )
    c_result = patterns.profile(
        keys["c"],
        arguments,
        warmups=1,
        repeats=2,
    )
    pure_result = patterns.profile(
        keys["pure_python"],
        arguments,
        warmups=1,
        repeats=2,
    )

    def host_array(value):
        payload = getattr(value, "data", value)
        if hasattr(payload, "tolist"):
            return np.asarray(payload.tolist())
        return np.asarray(value.numpy())

    np.testing.assert_allclose(
        host_array(c_result),
        host_array(numpy_result),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        host_array(pure_result),
        host_array(numpy_result),
        rtol=2e-5,
        atol=2e-5,
    )
    rows = patterns.profile_rows()
    assert len(rows) == 3
    assert all(row["compile_ns"] is not None for row in rows)
    assert all(len(row["samples_ns"]) == 2 for row in rows)
    assert all(row["median_ns"] > 0 for row in rows)


def test_hot_swap_retains_lazy_variants_and_profile_history():
    patterns = EvaluationPatternMap()
    created = []
    numpy_key = EvaluationVariantKey(
        "object", "shape", ControlTarget.PYTHON, "numpy"
    )
    c_key = EvaluationVariantKey(
        "object", "shape", ControlTarget.PYTHON, "c"
    )
    patterns.register(
        numpy_key,
        lambda: created.append("numpy") or (lambda value: value + 1),
    )
    patterns.register(
        c_key,
        lambda: created.append("c") or (lambda value: value + 2),
    )

    assert created == []
    assert patterns.select(numpy_key)(3) == 4
    assert patterns.active("object", "shape")(4) == 5
    assert created == ["numpy"]
    assert patterns.select(c_key)(3) == 5
    assert patterns.active("object", "shape")(4) == 6
    assert created == ["numpy", "c"]
    assert patterns.resolve(numpy_key)(5) == 6
    assert created == ["numpy", "c"]


def test_profile_matrix_reports_backend_failure_without_aborting():
    rows = run_profile_matrix(size=32, steps=2, repeats=1)

    assert len(rows) == 8
    assert sum(row["status"] == "passed" for row in rows) == 6
    glsl = [
        row for row in rows if row["shell_language"] == "glsl"
    ]
    assert len(glsl) == 2
    assert all(row["status"] == "failed" for row in glsl)
    assert all("GLSLFusedProgramNetwork" in row["error"] for row in glsl)
