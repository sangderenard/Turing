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
    _inputs: tuple = ()
    _outputs: tuple = ()
    _entry: Any = None
    _arena: Any = None
    _in_slots: Any = None
    _out_slots: Any = None
    _pointers: Any = None
    _extents: Any = None

    def __post_init__(self) -> None:
        # The buffer a name refers to never changes, so resolving it on every
        # call was a dictionary lookup per argument per cell -- forty-one of
        # them, plus the indirection through ``run`` and ``entry``, for a
        # stencil that is already native. Bind them once.
        self._inputs = tuple(
            self.execution.buffers[value_id] for value_id in self.input_ids
        )
        self._outputs = tuple(
            self.execution.buffers[value_id] for value_id in self.output_ids
        )
        self._entry = self.artifact.entry()
        self._pointers = self.execution.pointers
        self._extents = self.execution.extents
        # Every scalar of one dtype shares an arena, so the whole argument set
        # moves in one indexed write instead of one write per argument.
        index = self.execution.scalar_index
        arenas = {
            self.execution.buffers[value_id].dtype
            for value_id in (*self.input_ids, *self.output_ids)
        }
        self._arena = None
        if len(arenas) == 1 and all(
            value_id in index
            for value_id in (*self.input_ids, *self.output_ids)
        ):
            import numpy as _np

            self._arena = self.execution.scalar_arena[next(iter(arenas))]
            self._in_slots = _np.array(
                [index[value_id] for value_id in self.input_ids], dtype=_np.intp
            )
            self._out_slots = _np.array(
                [index[value_id] for value_id in self.output_ids], dtype=_np.intp
            )

    def __call__(self, *values: Any) -> tuple[float, ...]:
        if len(values) != len(self._inputs):
            raise TypeError(
                f"{self.artifact.name} expects {len(self.input_ids)} arguments, "
                f"received {len(values)}"
            )
        if self._arena is None:
            for buffer, value in zip(self._inputs, values):
                buffer[()] = value
            self._entry(self._pointers, self._extents)
            return tuple(buffer[()] for buffer in self._outputs)
        self._arena[self._in_slots] = values
        self._entry(self._pointers, self._extents)
        return tuple(self._arena[self._out_slots].tolist())


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
