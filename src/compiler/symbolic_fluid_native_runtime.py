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


@dataclass(slots=True)
class NativeSymbolicFluidAdvance:
    """The whole compiled grid traversal: ``advance(state, dt)``.

    Every binding is by NAME through the dual IR's port record --
    ``program_abi_parameter``/``program_abi_field`` accounting on arguments,
    ``parameter_value_abi`` for ``dt``, ``source_output_value_ids`` +
    ``record_return_layouts`` + per-value accounting for the return -- never
    by position.  The traversal, the stencil, the reductions and the mass
    balance all execute inside one native artifact; nothing is interpreted.
    """

    artifact: LLVMFunctionArtifact
    function: Any
    _module_functions: Any = None
    outputs_record: Any = None
    execution: LLVMExecution | None = None
    _state_feeds: tuple = ()          # (value_id, field, rank)
    _written: tuple = ()              # (value_id, field, rank)
    _dt_ids: tuple = ()
    _ok_id: int = -1
    _metrics_parts: Any = None

    def _bind(self, state: Any, dt: float) -> None:
        feeds: dict[int, Any] = {}
        state_feeds: list[tuple[int, str, int]] = []
        written: list[tuple[int, str, int]] = []
        dt_ids: list[int] = []
        for argument in self.function.args:
            accounting = dict(argument.accounting or {})
            parameter = accounting.get("program_abi_parameter")
            field = accounting.get("program_abi_field")
            rank = int(accounting.get("program_abi_rank") or 0)
            value_id = int(argument.id)
            if parameter == "state" and field is not None:
                source = getattr(state, str(field))
                feeds[value_id] = (
                    np.asarray(source, dtype=np.float64)
                    if rank else float(source)
                )
                state_feeds.append((value_id, str(field), rank))
                if accounting.get("program_abi_field_written"):
                    written.append((value_id, str(field), rank))
            elif parameter == "dt":
                feeds[value_id] = float(dt)
                dt_ids.append(value_id)
        if not dt_ids:
            # ``dt`` may carry no record accounting of its own, but the
            # stencil this function calls names every formal
            # (``argument_names`` is that function's declared port record),
            # and the operand feeding the stencil's ``dt`` formal IS this
            # function's dt.
            dt_ids.extend(self._named_stencil_operand("dt"))
            argument_ids = {int(argument.id) for argument in self.function.args}
            dt_ids[:] = [
                value_id for value_id in dict.fromkeys(dt_ids)
                if value_id in argument_ids
            ]
            for value_id in dt_ids:
                feeds[value_id] = float(dt)
        if not dt_ids:
            raise RuntimeError(
                "the advance artifact declares no bindable dt parameter"
            )
        # The authored source derives its loop bounds from the grid shape
        # (``height_count = state.height.shape[0]``); shape extraction is
        # coordinator work, so the counts arrive as named arguments.
        named_arguments = {
            str(name): int(value_id)
            for name, value_id in (
                self.function.metadata.get("value_names") or ()
            )
        }
        argument_ids = {int(argument.id) for argument in self.function.args}
        shape = np.asarray(state.height).shape
        for name, extent in (
            ("height_count", int(shape[0])),
            ("width_count", int(shape[1])),
        ):
            value_id = named_arguments.get(name)
            if value_id is not None and value_id in argument_ids:
                feeds[value_id] = int(extent)
        self.execution = prepare_artifact_execution(self.artifact, feeds)
        self._state_feeds = tuple(state_feeds)
        self._written = tuple(written)
        self._dt_ids = tuple(dt_ids)
        # The lowering RETURNS the gathered I/O record -- the authored
        # return, in order, one value per component.  Consume it; do not
        # reconstruct it from metadata plus guesses.
        record = tuple(self.outputs_record or ())
        if len(record) < 2:
            raise RuntimeError(
                "the lowering outputs record for the advance is required; "
                "pass the second element of lower_ast_source_to_ssa's result"
            )
        self._ok_id = int(record[0].id)
        import dataclasses as _dataclasses

        metric_fields = [
            field.name
            for field in _dataclasses.fields(Metrics)
            if field.name != "error_channels"
        ]
        components = record[1:]
        if len(components) > len(metric_fields):
            raise RuntimeError(
                f"outputs record has {len(components)} Metrics components "
                f"for {len(metric_fields)} contract fields"
            )
        by_kwarg = {
            name: int(value.id)
            for name, value in zip(metric_fields, components)
        }
        names = dict(
            (str(name), int(value_id))
            for name, value_id in self.function.metadata.get(
                "value_names", ()
            )
        )
        channels = {
            "height_positivity": names.get("max_height_violation"),
            "tracer_bounds": names.get("max_tracer_violation"),
        }
        self._metrics_parts = (by_kwarg, channels)

    def _named_stencil_operand(self, wanted: str) -> list[int]:
        """Operand ids feeding the stencil formal named ``wanted``."""

        module_functions = getattr(self, "_module_functions", None) or {}
        found: list[int] = []
        for block in self.function.blocks.values():
            for instruction in block.instrs:
                callee_name = str(
                    (instruction.attributes or {}).get("callee") or ""
                )
                callee = module_functions.get(callee_name)
                if callee is None:
                    continue
                names = tuple(
                    callee.metadata.get("argument_names") or ()
                )
                if wanted in names:
                    position = names.index(wanted)
                else:
                    # A linked function names its values through the
                    # ``value_names`` record instead: (name, value id).
                    named_ids = {
                        str(name): int(value_id)
                        for name, value_id in (
                            callee.metadata.get("value_names") or ()
                        )
                    }
                    if wanted not in named_ids:
                        continue
                    formal_ids = [
                        int(argument.id) for argument in callee.args
                    ]
                    if named_ids[wanted] not in formal_ids:
                        continue
                    position = formal_ids.index(named_ids[wanted])
                if position < len(instruction.args):
                    found.append(int(instruction.args[position].id))
        return found

    def _accounting_of(self, value_id: int) -> dict:
        for block in self.function.blocks.values():
            for instruction in block.instrs:
                for value in (
                    *instruction.args,
                    *((instruction.res,)
                      if instruction.res is not None else ()),
                ):
                    if int(value.id) == int(value_id) and value.accounting:
                        return dict(value.accounting)
        for argument in self.function.args:
            if int(argument.id) == int(value_id):
                return dict(argument.accounting or {})
        return {}

    def _read(self, value_id: int | None) -> float:
        if value_id is None or self.execution is None:
            return 0.0
        stored = self.execution.buffers.get(int(value_id))
        if stored is None:
            return 0.0
        return float(np.asarray(stored).reshape(-1)[0])

    def __call__(self, state: Any, dt: Any) -> tuple[bool, Metrics]:
        dt = float(dt)
        if self.execution is None:
            self._bind(state, dt)
        buffers = self.execution.buffers
        for value_id, field, rank in self._state_feeds:
            source = getattr(state, field)
            if rank:
                np.asarray(buffers[value_id]).reshape(
                    np.asarray(source).shape
                )[...] = source
            else:
                buffers[value_id][...] = float(source)
        for value_id in self._dt_ids:
            buffers[value_id][...] = dt
        self.execution.run()
        for value_id, field, rank in self._written:
            target = getattr(state, field)
            if rank:
                np.asarray(target)[...] = np.asarray(
                    buffers[value_id]
                ).reshape(np.asarray(target).shape)
            else:
                setattr(state, field, self._read(value_id))
        by_kwarg, channels = self._metrics_parts
        ok = bool(np.asarray(buffers.get(self._ok_id, 0)).reshape(-1)[0])
        metrics = Metrics(
            max_vel=self._read(by_kwarg.get("max_vel")),
            max_flux=self._read(by_kwarg.get("max_flux")),
            div_inf=self._read(by_kwarg.get("div_inf")),
            mass_err=self._read(by_kwarg.get("mass_err")),
            dt_limit=self._read(by_kwarg.get("dt_limit")),
            error_channels={
                name: self._read(value_id)
                for name, value_id in channels.items()
            },
        )
        return ok, metrics


def compile_native_symbolic_fluid_advance(
    build_directory: str | Path | None = None,
) -> NativeSymbolicFluidAdvance:
    """Compile the whole authored traversal -- never exec it.

    A cached ``control_repository_ssa.pkl`` under ``build_directory`` is
    loaded; otherwise the same whole-program lowering the direct-control
    worker performs runs here (~30 s once).
    """

    import pickle

    module = None
    outputs_record = None
    if build_directory is not None:
        cached = Path(build_directory) / "control_repository_ssa.pkl"
        if cached.is_file():
            with cached.open("rb") as stream:
                module, lowering_outputs, _exports = pickle.load(stream)
            outputs_record = lowering_outputs.get(
                "symbolic_fluid_control__symbolic_fluid_advance"
            )
    if module is None:
        from .fortran_c_shell import lower_ast_source_to_ssa

        symbolic = compile_symbolic_fluid_step()
        module, lowering_outputs, _exports = lower_ast_source_to_ssa(
            SYMBOLIC_FLUID_DT_SOURCE,
            "symbolic_fluid_frame",
            python_bindings={
                "Metrics": Metrics,
                "run_superstep": run_superstep,
            },
            linked_process_graphs={
                "symbolic_fluid_step": symbolic.process_graph,
            },
            name="symbolic_fluid_control",
            runtime_closure_only=True,
            extraction_contract=(
                Path(__file__).resolve().parents[2]
                / "extraction_contracts"
                / "program_extraction.yaml"
            ),
        )
    name = "symbolic_fluid_control__symbolic_fluid_advance"
    if outputs_record is None:
        outputs_record = lowering_outputs.get(name)
    artifact = emit_ssa_function_to_llvm(module, name)
    if not artifact.complete:
        raise RuntimeError(
            "symbolic fluid advance emission has shortfalls: "
            + "; ".join(item.reason for item in artifact.shortfalls)
        )
    compile_artifact(
        artifact,
        directory=(
            None if build_directory is None
            else Path(build_directory) / "advance-native"
        ),
    )
    return NativeSymbolicFluidAdvance(
        artifact, module.functions[name], dict(module.functions),
        outputs_record,
    )


def load_symbolic_fluid_managed_functions(
    build_directory: str | Path | None = None,
) -> dict[str, Callable[..., Any]]:
    """The managed-dt functions, every one of them compiled.

    The exec of ``SYMBOLIC_FLUID_DT_SOURCE`` is gone: the traversal that
    interpreted the grid loop cell by cell -- and forced per-cell scalar
    marshalling across the ABI -- now IS the artifact.  ``run_superstep``
    remains the Python controller the demo drives; its ``advance`` callable
    is native end to end.
    """

    advance = compile_native_symbolic_fluid_advance(build_directory)

    def symbolic_fluid_frame(
        state: Any,
        targets: Any,
        controller: Any,
        frame_duration: float,
        dt_initial: float,
    ) -> tuple:
        _advanced, dt_next, _metrics = run_superstep(
            state,
            float(frame_duration),
            float(dt_initial),
            state.dx,
            targets,
            controller,
            advance,
            max_iters=256,
        )
        return (
            state.height,
            state.momentum_x,
            state.momentum_y,
            state.tracer,
            dt_next,
            state.last_wave_speed,
            state.last_height_violation,
            state.last_tracer_violation,
        )

    return {
        "symbolic_fluid_advance": advance,
        "symbolic_fluid_frame": symbolic_fluid_frame,
    }


__all__ = [
    "NativeSymbolicFluidAdvance",
    "NativeSymbolicFluidStep",
    "compile_native_symbolic_fluid_advance",
    "compile_native_symbolic_fluid_step",
    "load_symbolic_fluid_managed_functions",
]
