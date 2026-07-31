"""Whole-tape GLSL torture execution through the common profiled C shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifact_cache import (
    CachedArtifact,
    RepositoryArtifactCache,
    implementation_digest,
)
from .c_primitive_program import compile_recorded_fused_tape
from .glsl_backend import (
    GLChunk,
    compile_captured_fused_program,
    execute_captured_fused_program,
)
from .profiled_c_shell import CLaunchProfile, profiled_c_shell
from .tensor_torture import CapturedTortureCase


@dataclass(frozen=True)
class GLSLJITExecution:
    outputs: Mapping[str, np.ndarray]
    profile: CLaunchProfile
    cache_hit: bool


def _glsl_implementation_digest() -> str:
    directory = Path(__file__).resolve().parent
    return implementation_digest(
        (
            directory / "glsl_backend.py",
            directory / "c_primitive_program.py",
            directory / "glsl_jit_backend.py",
            directory.parents[0] / "fused_ir.py",
        )
    )


class GLSLJITProgram:
    def __init__(
        self,
        *,
        captured: Any,
        input_ids: Mapping[str, int],
        source_artifact: CachedArtifact,
    ):
        self.captured = captured
        self.input_ids = dict(input_ids)
        self.source_artifact = source_artifact

    def execute(
        self,
        inputs: Mapping[str, Any],
        *,
        profiler: Any | None = None,
        profile_path: str = "torture/glsl",
    ) -> GLSLJITExecution:
        from OpenGL import GL
        import ctypes

        missing = set(self.input_ids) - set(inputs)
        if missing:
            raise ValueError(f"missing GLSL JIT feeds: {sorted(missing)}")
        runtime_chunks = {
            value_id: GLChunk.from_numpy(
                np.ascontiguousarray(inputs[name], dtype=np.float32)
            )
            for name, value_id in self.input_ids.items()
        }
        shell = profiled_c_shell()
        result_holder: dict[str, Any] = {}

        @shell.callback
        def dispatch(_context, device_ns):
            query = None
            try:
                query = int(
                    np.asarray(GL.glGenQueries(1)).reshape(-1)[0]
                )
                GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)
                result_holder["outputs"] = execute_captured_fused_program(
                    self.captured,
                    runtime_chunks,
                )
                GL.glEndQuery(GL.GL_TIME_ELAPSED)
                elapsed = ctypes.c_uint64()
                GL.glGetQueryObjectui64v(
                    query,
                    GL.GL_QUERY_RESULT,
                    ctypes.byref(elapsed),
                )
                device_ns[0] = int(elapsed.value)
                return 1
            except BaseException as error:
                result_holder["error"] = error
                try:
                    GL.glEndQuery(GL.GL_TIME_ELAPSED)
                except Exception:
                    pass
                return 0
            finally:
                if query is not None:
                    try:
                        GL.glDeleteQueries(1, (query,))
                    except Exception:
                        pass

        token = (
            profiler.begin_shell(profile_path)
            if profiler is not None
            else None
        )
        profile = shell.launch(dispatch)
        if profiler is not None:
            shell.record(
                profiler,
                profile,
                path=profile_path,
                label="glsl-compute",
            )
            profiler.end_shell(profile_path, token)
        if "error" in result_holder:
            raise result_holder["error"]
        if profile.status != 1:
            raise RuntimeError(
                f"GLSL compute closure returned status {profile.status}"
            )
        outputs = {
            name: chunk.numpy()
            for name, chunk in result_holder["outputs"].items()
        }
        return GLSLJITExecution(
            outputs=outputs,
            profile=profile,
            cache_hit=self.source_artifact.hit,
        )


def compile_torture_case_to_glsl(
    captured_case: CapturedTortureCase,
    *,
    cache: RepositoryArtifactCache | None = None,
) -> GLSLJITProgram:
    """Compile every recorded stage with graph optimization disabled."""

    cache = cache or RepositoryArtifactCache()
    captured = compile_recorded_fused_tape(
        captured_case.tape,
        outputs=captured_case.outputs,
        strict_outputs=True,
    )
    source = compile_captured_fused_program(captured)
    input_ids = {
        name: id(value) for name, value in captured_case.inputs.items()
    }
    record = {
        "case": captured_case.case.semantic_record(),
        "compiler": "turing-captured-glsl",
        "implementation": _glsl_implementation_digest(),
        "optimization": False,
    }
    source_artifact = cache.load("glsl", record, suffix=".comp.glsl")
    if source_artifact is None:
        source_artifact = cache.store(
            "glsl",
            record,
            source,
            suffix=".comp.glsl",
            metadata={
                "input_names": list(input_ids),
                "stage_count": len(captured.execution_programs),
            },
        )
    return GLSLJITProgram(
        captured=captured,
        input_ids=input_ids,
        source_artifact=source_artifact,
    )


__all__ = [
    "GLSLJITExecution",
    "GLSLJITProgram",
    "compile_torture_case_to_glsl",
]
