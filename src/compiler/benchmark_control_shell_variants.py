"""Profile equivalent complex control shells without hiding failed variants."""

from __future__ import annotations

from typing import Any

import numpy as np

from .control_source import (
    ControlProgram,
    ControlTarget,
    LoopBlock,
    RegionCode,
    SequenceBlock,
    StatementBlock,
    compile_cffi_shell,
    compile_python_shell,
)
from .evaluation_patterns import (
    EvaluationPatternMap,
    EvaluationVariantKey,
)


def _python_recurrence(backend: str):
    logical = ControlProgram(
        SequenceBlock((
            StatementBlock(("state = AbstractTensor.tensor(values)",)),
            LoopBlock(
                "iteration", "0", "steps", "1",
                StatementBlock(("__scheduled_region_0__",)),
            ),
            StatementBlock(("return state",)),
        )),
        (0,),
    )
    return compile_python_shell(
        logical,
        (RegionCode(
            0,
            ControlTarget.PYTHON,
            StatementBlock(("state = (state * scale + bias).tanh()",)),
        ),),
        function_name="python_recurrence",
        parameters=("values", "steps", "scale", "bias"),
        abstract_tensor_backend=backend,
    )


def _python_coupled(backend: str):
    logical = ControlProgram(
        SequenceBlock((
            StatementBlock((
                "left = AbstractTensor.tensor(values)",
                "right = AbstractTensor.tensor(values[::-1].copy())",
            )),
            LoopBlock(
                "iteration", "0", "steps", "1",
                StatementBlock(("__scheduled_region_0__",)),
            ),
            StatementBlock(("return left + right",)),
        )),
        (0,),
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
        function_name="python_coupled",
        parameters=("values", "steps"),
        abstract_tensor_backend=backend,
    )


def _c_recurrence():
    logical = ControlProgram(
        LoopBlock(
            "iteration", "0", "steps", "1",
            StatementBlock(("__scheduled_region_0__",)),
        ),
        (0,),
    )
    compiled = compile_cffi_shell(
        logical,
        (RegionCode(
            0,
            ControlTarget.C,
            StatementBlock((
                "for (int element = 0; element < count; ++element) {",
                "    state[element] = tanhf(state[element] * scale + bias);",
                "}",
            )),
        ),),
        function_name="c_recurrence",
        parameters=(
            "float *state", "int count", "int steps",
            "float scale", "float bias",
        ),
        c_declaration=(
            "void c_recurrence(float *state, int count, int steps, "
            "float scale, float bias);"
        ),
        preamble="#include <math.h>",
    )

    def execute(values, steps, scale, bias):
        array = np.asarray(values, dtype=np.float32).copy()
        pointer = compiled.ffi.cast("float *", array.ctypes.data)
        compiled(pointer, array.size, steps, scale, bias)
        return array

    return execute


def _c_coupled():
    logical = ControlProgram(
        SequenceBlock((
            LoopBlock(
                "iteration", "0", "steps", "1",
                StatementBlock(("__scheduled_region_0__",)),
            ),
            StatementBlock((
                "for (int element = 0; element < count; ++element) {",
                "    left[element] += right[element];",
                "}",
            )),
        )),
        (0,),
    )
    compiled = compile_cffi_shell(
        logical,
        (RegionCode(
            0,
            ControlTarget.C,
            StatementBlock((
                "for (int element = 0; element < count; ++element) {",
                "    float old_left = left[element];",
                "    left[element] = old_left * 0.75f + right[element] * 0.25f;",
                "    right[element] = right[element] * 0.6f - old_left * 0.1f;",
                "}",
            )),
        ),),
        function_name="c_coupled",
        parameters=("float *left", "float *right", "int count", "int steps"),
        c_declaration=(
            "void c_coupled(float *left, float *right, int count, int steps);"
        ),
    )

    def execute(values, steps):
        left = np.asarray(values, dtype=np.float32).copy()
        right = left[::-1].copy()
        compiled(
            compiled.ffi.cast("float *", left.ctypes.data),
            compiled.ffi.cast("float *", right.ctypes.data),
            left.size,
            steps,
        )
        return left

    return execute


def _unavailable_glsl_shell():
    raise RuntimeError(
        "composed GLSL control source exists, but deployment installation "
        "still selects GLSLFusedProgramNetwork instead of an executable "
        "composed-control artifact"
    )


def run_profile_matrix(
    *,
    size: int = 4096,
    steps: int = 24,
    repeats: int = 5,
) -> tuple[dict[str, Any], ...]:
    patterns = EvaluationPatternMap()
    problems = {
        "nonlinear_recurrence": (
            (
                np.linspace(-2.0, 2.0, size, dtype=np.float32),
                steps,
                0.91,
                0.07,
            ),
            _python_recurrence,
            _c_recurrence,
        ),
        "coupled_recurrence": (
            (np.linspace(-1.0, 1.0, size, dtype=np.float32), steps),
            _python_coupled,
            _c_coupled,
        ),
    }
    for name, (arguments, python_factory, c_factory) in problems.items():
        signature = (size, steps)
        variants = (
            (ControlTarget.PYTHON, "numpy", lambda f=python_factory: f("numpy")),
            (ControlTarget.PYTHON, "c", lambda f=python_factory: f("c")),
            (ControlTarget.C, "cffi", c_factory),
            (ControlTarget.GLSL, "glsl", _unavailable_glsl_shell),
        )
        for language, backend, factory in variants:
            key = EvaluationVariantKey(name, signature, language, backend)
            patterns.register(key, factory)
            patterns.profile_attempt(
                key,
                arguments,
                warmups=1,
                repeats=repeats,
            )
    return patterns.profile_rows()


def format_profile_matrix(rows) -> str:
    lines = []
    for row in rows:
        identity = (
            f"{row['object_key']:22} "
            f"{row['shell_language']}/{row['interior_backend']}"
        )
        if row["status"] == "passed":
            lines.append(
                f"{identity:42} PASS "
                f"compile={row['compile_ns'] / 1e6:.3f}ms "
                f"median={row['median_ns'] / 1e6:.3f}ms"
            )
        else:
            lines.append(f"{identity:42} FAIL {row['error']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_profile_matrix(run_profile_matrix()))


__all__ = ["format_profile_matrix", "run_profile_matrix"]
