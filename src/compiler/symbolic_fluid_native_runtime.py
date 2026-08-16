"""Run the authored SymPy fluid function through its emitted LLVM artifact.

This module is an ABI adapter, not a second implementation of the fluid
equations.  Every cell update enters the repository-SSA function produced by
``compile_symbolic_fluid_step`` and compiled ahead of time by the LLVM lane.
The surrounding managed-time functions are loaded from the same source string
submitted to the AOT compiler, so interpreted demonstrations and compilation
attempts exercise one orchestration program.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..common.dt_system.dt_controller import run_superstep
from ..common.dt_system.dt_scaler import Metrics
from .ssa_llvm_backend import (
    LLVMExecution,
    LLVMFunctionArtifact,
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from .symbolic_equation_compiler import SymbolicEquationCompilation
from .symbolic_fluid_dt import SYMBOLIC_FLUID_DT_SOURCE
from .symbolic_fluid_model import compile_symbolic_fluid_step


@dataclass(slots=True)
class NativeSymbolicFluidStep:
    """Callable positional ABI for one compiled scalar fluid stencil."""

    compilation: SymbolicEquationCompilation
    artifact: LLVMFunctionArtifact
    execution: LLVMExecution
    input_names: tuple[str, ...]
    input_ids: tuple[int, ...]
    output_names: tuple[str, ...]
    output_ids: tuple[int, ...]

    def __call__(self, *values: Any) -> tuple[float, ...]:
        if len(values) != len(self.input_ids):
            raise TypeError(
                f"{self.artifact.name} expects {len(self.input_ids)} arguments, "
                f"received {len(values)}"
            )
        for value_id, value in zip(self.input_ids, values):
            self.execution.buffers[value_id][...] = value
        self.execution.run()
        return tuple(
            float(self.execution.buffers[value_id])
            for value_id in self.output_ids
        )


def compile_native_symbolic_fluid_step(
    directory: str | Path | None = None,
) -> NativeSymbolicFluidStep:
    """Compile the pure-SymPy stencil to LLVM and bind its buffer ABI."""

    compilation = compile_symbolic_fluid_step()
    artifact = emit_ssa_function_to_llvm(
        compilation.module, compilation.function.name,
    )
    if not artifact.complete:
        raise RuntimeError(
            "symbolic fluid LLVM emission has shortfalls: "
            + "; ".join(item.reason for item in artifact.shortfalls)
        )
    compile_artifact(
        artifact,
        directory=None if directory is None else Path(directory),
    )
    input_names = tuple(compilation.function.metadata["argument_names"])
    output_names = tuple(compilation.function.metadata["output_names"])
    input_ids = tuple(compilation.input_ids[name] for name in input_names)
    output_ids = tuple(compilation.output_ids[name] for name in output_names)
    execution = prepare_artifact_execution(
        artifact, {value_id: 0.0 for value_id in input_ids},
    )
    return NativeSymbolicFluidStep(
        compilation,
        artifact,
        execution,
        input_names,
        input_ids,
        output_names,
        output_ids,
    )


def load_symbolic_fluid_managed_functions(
    native_step: Callable[..., tuple[float, ...]],
) -> dict[str, Callable[..., Any]]:
    """Load the exact managed-dt source with its numerical call bound native."""

    namespace: dict[str, Any] = {
        "Metrics": Metrics,
        "run_superstep": run_superstep,
        "symbolic_fluid_step": native_step,
    }
    exec(SYMBOLIC_FLUID_DT_SOURCE, namespace)
    return {
        "symbolic_fluid_advance": namespace["symbolic_fluid_advance"],
        "symbolic_fluid_frame": namespace["symbolic_fluid_frame"],
    }


__all__ = [
    "NativeSymbolicFluidStep",
    "compile_native_symbolic_fluid_step",
    "load_symbolic_fluid_managed_functions",
]
