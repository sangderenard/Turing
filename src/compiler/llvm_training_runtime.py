"""Thin runtime coordination for graph-autograd LLVM training motions.

This module does not introduce a compiler or an optimizer IR.  It packages
the existing ProcessGraph reverse pass, repository SSA lowering, LLVM emitter,
and native SGD loop behind named parameter groups.  Each group receives its
own native entry point over the same complete forward/loss/backward motion;
selecting a group therefore changes only which declared parameters are stepped.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class NativeParameterGroup:
    name: str
    parameter_ids: tuple[int, ...]
    learning_rate: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("native parameter group requires a name")
        if not self.parameter_ids:
            raise ValueError(f"parameter group {self.name!r} is empty")
        if len(set(map(int, self.parameter_ids))) != len(self.parameter_ids):
            raise ValueError(f"parameter group {self.name!r} repeats a parameter")
        if self.learning_rate <= 0:
            raise ValueError(f"parameter group {self.name!r} learning rate must be positive")


@dataclass(frozen=True)
class NativeTrainingSchedule:
    name: str
    outputs: Mapping[str, int]
    groups: tuple[NativeParameterGroup, ...]
    artifacts: Mapping[str, Any]
    saved_binding_count: int

    def group(self, name: str) -> NativeParameterGroup:
        selected = next(
            (group for group in self.groups if group.name == str(name)), None,
        )
        if selected is None:
            available = tuple(group.name for group in self.groups)
            raise KeyError(
                f"unknown parameter group {name!r}; available={available!r}"
            )
        return selected


@dataclass(frozen=True)
class NativeGraphReverse:
    """One fully compiled forward/VJP artifact with an explicit seed ABI.

    This is the publication unit for a method reverse.  ``artifact`` is a
    loaded-library-capable LLVM artifact, not a serialized ProcessGraph and
    not a Python backward callback.  The value-id maps are retained so an
    outer object bundle can expose a stable binding contract without learning
    details of repository SSA or the LLVM emitter.
    """

    name: str
    output_value_ids: tuple[int, ...]
    gradient_value_ids: Mapping[int, int]
    seed_value_ids: Mapping[int, int]
    artifact: Any
    saved_binding_count: int


def compile_native_graph_reverse(
    output: Any,
    *,
    bindings: Mapping[str, Any],
    wrt_value_ids: Sequence[int],
    name: str,
    directory: Path,
    unit_output_seed: bool = False,
) -> NativeGraphReverse:
    """Graph-invert and natively compile an AbstractTensor method.

    By default each forward output receives a runtime upstream-gradient
    parameter, making the product a genuine parametric VJP.  Scalar loss
    callers may request ``unit_output_seed=True``; standard-object publication
    should keep the default so the reverse ABI remains generally reusable.
    """

    from .process_graph_autograd import (
        abstract_tensor_program_to_process_graph,
        compile_process_graph_backward,
        lower_training_motion_to_repository_ssa,
    )
    from .ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm

    wrt = tuple(map(int, wrt_value_ids))
    if not wrt:
        raise ValueError("compiled graph reverse requires at least one wrt value")
    if len(set(wrt)) != len(wrt):
        raise ValueError(f"compiled graph reverse repeats wrt values: {wrt!r}")

    forward = abstract_tensor_program_to_process_graph(output, bindings=bindings)
    product = compile_process_graph_backward(
        forward,
        wrt=wrt,
        packaging="combined",
        unit_loss_seed=bool(unit_output_seed),
    )
    if product.motion is None:
        raise RuntimeError("combined graph-autograd request produced no motion")
    lowering = lower_training_motion_to_repository_ssa(
        product.motion,
        function_name=str(name),
    )
    if lowering.shortfalls:
        raise RuntimeError(
            f"{name} repository-SSA shortfalls: {lowering.shortfalls!r}"
        )
    emitted = emit_ssa_function_to_llvm(
        lowering.module,
        lowering.function_name,
        entry_name=lowering.function_name,
    )
    if emitted.shortfalls:
        raise RuntimeError(f"{name} LLVM shortfalls: {emitted.shortfalls!r}")
    artifact = compile_artifact(emitted, directory=Path(directory))
    if artifact.library_path is None:
        raise RuntimeError(f"{name} reverse compilation produced no native library")
    return NativeGraphReverse(
        name=str(name),
        output_value_ids=tuple(map(int, product.adjoint.output_value_ids)),
        gradient_value_ids=dict(product.motion.gradient_value_ids),
        seed_value_ids=dict(product.motion.seed_value_ids),
        artifact=artifact,
        saved_binding_count=len(product.adjoint.saved_value_contracts),
    )


def compile_native_training_schedule(
    output: Any,
    *,
    bindings: Mapping[str, Any],
    parameter_groups: Sequence[NativeParameterGroup],
    name: str,
    directory: Path,
    observed_outputs: Mapping[str, int] | None = None,
) -> NativeTrainingSchedule:
    """Compile one complete motion and one native stepping entry per group."""

    from .process_graph_autograd import (
        abstract_tensor_program_to_process_graph,
        compile_process_graph_backward,
        lower_training_motion_to_repository_ssa,
    )
    from .ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
        with_native_sgd_loop,
    )

    groups = tuple(parameter_groups)
    names = tuple(group.name for group in groups)
    if len(set(names)) != len(names):
        raise ValueError(f"parameter group names must be unique: {names!r}")
    parameter_ids = tuple(dict.fromkeys(
        int(parameter_id)
        for group in groups
        for parameter_id in group.parameter_ids
    ))
    forward = abstract_tensor_program_to_process_graph(output, bindings=bindings)
    product = compile_process_graph_backward(
        forward, wrt=parameter_ids, packaging="combined",
    )
    if product.motion is None:
        raise RuntimeError("combined graph-autograd request produced no motion")
    lowering = lower_training_motion_to_repository_ssa(
        product.motion,
        function_name=str(name),
        observed_outputs=observed_outputs,
    )
    if lowering.shortfalls:
        raise RuntimeError(
            f"{name} repository-SSA shortfalls: {lowering.shortfalls!r}"
        )
    base = emit_ssa_function_to_llvm(
        lowering.module,
        lowering.function_name,
        entry_name=lowering.function_name,
    )
    if base.shortfalls:
        raise RuntimeError(f"{name} LLVM shortfalls: {base.shortfalls!r}")

    artifacts = {}
    directory = Path(directory)
    for group in groups:
        stepped = with_native_sgd_loop(
            base,
            parameter_gradient_pairs=tuple(
                (
                    int(parameter_id),
                    int(lowering.outputs[f"grad_{int(parameter_id)}"]),
                )
                for parameter_id in group.parameter_ids
            ),
            entry_name=f"{name}__step__{group.name}",
        )
        artifacts[group.name] = compile_artifact(
            stepped, directory=directory / group.name,
        )
    return NativeTrainingSchedule(
        name=str(name),
        outputs=dict(lowering.outputs),
        groups=groups,
        artifacts=artifacts,
        saved_binding_count=len(product.adjoint.saved_value_contracts),
    )


def run_parameter_group(
    schedule: NativeTrainingSchedule,
    group_name: str,
    buffers: MutableMapping[int, Any],
    *,
    steps: int = 1,
    learning_rate: float | None = None,
):
    """Run a selected native group and carry its parameter state forward."""

    from .ssa_llvm_backend import prepare_artifact_execution

    group = schedule.group(group_name)
    artifact = schedule.artifacts[group.name]
    execution = prepare_artifact_execution(artifact, {
        **dict(buffers),
        artifact.training_steps_value_id: int(steps),
        artifact.learning_rate_value_id: float(
            group.learning_rate if learning_rate is None else learning_rate
        ),
    }).run()
    for parameter_id in group.parameter_ids:
        buffers[int(parameter_id)] = execution.buffers[int(parameter_id)].copy()
    return execution


def native_artifact_record(artifact: Any) -> dict[str, Any]:
    if artifact.library_path is None:
        raise RuntimeError("cannot manifest an uncompiled LLVM artifact")
    return {
        "name": artifact.name,
        "library_path": str(Path(artifact.library_path).resolve()),
        "buffer_order": list(artifact.buffer_order),
        "buffer_shapes": [list(shape) for shape in artifact.buffer_shapes],
        "buffer_dtypes": list(artifact.buffer_dtypes),
        "extent_order": [list(item) for item in artifact.extent_order],
        "needs_text_sink": bool(artifact.needs_text_sink),
        "training_steps_value_id": artifact.training_steps_value_id,
        "learning_rate_value_id": artifact.learning_rate_value_id,
    }


def save_training_schedule(schedule: NativeTrainingSchedule, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": schedule.name,
        "outputs": dict(schedule.outputs),
        "saved_binding_count": int(schedule.saved_binding_count),
        "groups": [
            {
                "name": group.name,
                "parameter_ids": list(group.parameter_ids),
                "learning_rate": group.learning_rate,
                "artifact": native_artifact_record(schedule.artifacts[group.name]),
            }
            for group in schedule.groups
        ],
    }, indent=2, sort_keys=True), encoding="utf-8")


def load_training_schedule(path: Path) -> NativeTrainingSchedule:
    from .ssa_llvm_backend import LLVMFunctionArtifact

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = tuple(
        NativeParameterGroup(
            str(record["name"]),
            tuple(map(int, record["parameter_ids"])),
            float(record["learning_rate"]),
        )
        for record in payload["groups"]
    )
    artifacts = {}
    for record in payload["groups"]:
        data = record["artifact"]
        artifacts[str(record["name"])] = LLVMFunctionArtifact(
            name=str(data["name"]),
            llvm_ir="",
            buffer_order=tuple(map(int, data["buffer_order"])),
            buffer_shapes=tuple(
                tuple(shape) for shape in data["buffer_shapes"]
            ),
            extent_order=tuple(tuple(item) for item in data["extent_order"]),
            shortfalls=(),
            buffer_dtypes=tuple(map(str, data["buffer_dtypes"])),
            needs_text_sink=bool(data["needs_text_sink"]),
            library_path=Path(data["library_path"]),
            training_steps_value_id=data["training_steps_value_id"],
            learning_rate_value_id=data["learning_rate_value_id"],
        )
    return NativeTrainingSchedule(
        name=str(payload["name"]),
        outputs={str(key): int(value) for key, value in payload["outputs"].items()},
        groups=groups,
        artifacts=artifacts,
        saved_binding_count=int(payload["saved_binding_count"]),
    )


__all__ = [
    "NativeParameterGroup",
    "NativeTrainingSchedule",
    "compile_native_training_schedule",
    "load_training_schedule",
    "run_parameter_group",
    "save_training_schedule",
]
