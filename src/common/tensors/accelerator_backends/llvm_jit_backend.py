"""LLVM compute compilation and execution through the shared profiled C shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np

from .artifact_cache import (
    CachedArtifact,
    RepositoryArtifactCache,
    implementation_digest,
)
from ..abstraction import tensor_identity
from .c_backend_llvm_ssa import lower_abstract_tensor_tape_to_llvm_ssa
from .profiled_c_shell import CLaunchProfile, profiled_c_shell
from .tensor_torture import CapturedTortureCase


class LLVMJITShortfall(RuntimeError):
    pass


@dataclass(frozen=True)
class LLVMOutputSpec:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class LLVMJITExecution:
    outputs: Mapping[str, np.ndarray]
    profile: CLaunchProfile
    cache_hit: bool


def _llvm_implementation_digest() -> str:
    directory = Path(__file__).resolve().parent
    return implementation_digest(
        (
            directory / "c_backend_llvm_ssa.py",
            directory / "llvm_signal_math.py",
            directory / "llvm_jit_backend.py",
            directory / "c_backend" / "ctensor_ops.c",
            directory / "c_backend" / "ctensor_ops.h",
        )
    )


def _initialize_llvm():
    from llvmlite import binding as llvm

    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    return llvm


def _shell_adapter_ir(
    *,
    function_name: str,
    shell_name: str,
    argument_count: int,
) -> str:
    lines = [
        f"define i32 @{shell_name}(ptr %context, ptr %device_ns) {{",
        "entry:",
    ]
    arguments = []
    for index in range(argument_count):
        lines.extend(
            (
                f"  %argument.address.{index} = getelementptr ptr, "
                f"ptr %context, i64 {index}",
                f"  %argument.{index} = load ptr, "
                f"ptr %argument.address.{index}, align 8",
            )
        )
        arguments.append(f"ptr %argument.{index}")
    lines.extend(
        (
            f"  call void @{function_name}({', '.join(arguments)})",
            "  store i64 0, ptr %device_ns, align 8",
            "  ret i32 1",
            "}",
            "",
        )
    )
    return "\n".join(lines)


class LLVMJITProgram:
    """Owning MCJIT program with the uniform C-shell callback ABI."""

    def __init__(
        self,
        *,
        llvm_ir: str,
        function_name: str,
        shell_name: str,
        feed_names: tuple[str, ...],
        outputs: tuple[LLVMOutputSpec, ...],
        workspace_sizes: tuple[int, ...],
        cache_artifact: CachedArtifact,
        _llvm_profile: Any | None = None,
    ):
        from ....compiler.llvm_optimizing_pipeline import (
            harden_ir,
            run_pipeline,
            target_machine as build_target_machine,
            tuned_host_profile,
        )

        # Codegen used to be create_target_machine(opt=0) against the default
        # triple with no pass pipeline, which is scalar generic code.  The tuned
        # profile names the host CPU, runs -O3, asserts noalias on the kernel
        # buffers, and selects the vector width the host actually executes at
        # full rate.  Measured on bdver2 this is ~4x the previous throughput and
        # reaches parity with gfortran -O3 -march=native.
        profile = _llvm_profile or tuned_host_profile()
        llvm = _initialize_llvm()
        module = llvm.parse_assembly(harden_ir(llvm_ir, profile))
        module.verify()
        target_machine = build_target_machine(profile)
        if profile.opt > 0:
            run_pipeline(module, target_machine, profile)
        engine = llvm.create_mcjit_compiler(module, target_machine)
        engine.finalize_object()
        engine.run_static_constructors()
        shell_address = int(engine.get_function_address(shell_name))
        if not shell_address:
            raise RuntimeError(f"LLVM JIT did not expose {shell_name!r}")
        self.llvm_ir = llvm_ir
        self.function_name = function_name
        self.shell_name = shell_name
        self.feed_names = feed_names
        self.output_specs = outputs
        self.workspace_sizes = workspace_sizes
        self.cache_artifact = cache_artifact
        self._module = module
        self._engine = engine
        self._target_machine = target_machine
        self._shell_address = shell_address

    def execute(
        self,
        inputs: Mapping[str, Any],
        *,
        profiler: Any | None = None,
        profile_path: str = "torture/llvm",
    ) -> LLVMJITExecution:
        missing = set(self.feed_names) - set(inputs)
        if missing:
            raise ValueError(f"missing LLVM JIT feeds: {sorted(missing)}")
        feed_arrays = [
            np.ascontiguousarray(inputs[name], dtype=np.float64)
            for name in self.feed_names
        ]
        output_arrays = [
            np.empty(spec.shape or (), dtype=np.float64)
            for spec in self.output_specs
        ]
        workspace_arrays = [
            np.empty(size, dtype=np.float64)
            for size in self.workspace_sizes
        ]
        arrays = [*feed_arrays, *output_arrays, *workspace_arrays]
        shell = profiled_c_shell()
        context = shell.ffi.new(
            "void *[]",
            [
                shell.ffi.cast("void *", int(array.ctypes.data))
                for array in arrays
            ],
        )
        token = (
            profiler.begin_shell(profile_path)
            if profiler is not None
            else None
        )
        profile = shell.launch(self._shell_address, context)
        if profiler is not None:
            shell.record(
                profiler,
                profile,
                path=profile_path,
                label=self.function_name,
            )
            profiler.end_shell(profile_path, token)
        if profile.status != 1:
            raise RuntimeError(
                f"LLVM compute closure returned status {profile.status}"
            )
        return LLVMJITExecution(
            outputs={
                spec.name: array
                for spec, array in zip(self.output_specs, output_arrays)
            },
            profile=profile,
            cache_hit=self.cache_artifact.hit,
        )


def compile_torture_case_to_llvm(
    captured: CapturedTortureCase,
    *,
    trig_solver: str = "lut",
    trig_epsilon: float | None = None,
    cache: RepositoryArtifactCache | None = None,
    llvm_profile: Any | None = None,
) -> LLVMJITProgram:
    """Compile one complete case.

    Codegen runs the tuned host profile by default: -O3, the host CPU named,
    noalias asserted on kernel buffers, and the vector width the host executes
    at full rate.  Pass ``llvm_profile=REFERENCE_PROFILE`` for the previous
    unoptimized behaviour, which remains useful as a differential reference.
    """

    from ....compiler.llvm_optimizing_pipeline import tuned_host_profile

    cache = cache or RepositoryArtifactCache()
    resolved_profile = llvm_profile or tuned_host_profile()
    function_name = "turing_torture_compute"
    shell_name = "turing_torture_c_shell_entry"
    record = {
        "case": captured.case.semantic_record(),
        "compiler": "turing-direct-c-kernel-llvm",
        "implementation": _llvm_implementation_digest(),
        # The optimization settings are part of the artifact identity.  A bare
        # False here would let an opt=0 module be served from cache for a tuned
        # request, silently returning code the caller did not ask for.
        "optimization": {
            "opt": resolved_profile.opt,
            "host_cpu": resolved_profile.use_host_cpu,
            "noalias": resolved_profile.annotate_noalias,
            "fast_math": resolved_profile.fast_math,
            "vector_width": resolved_profile.prefer_vector_width,
        },
        "trig_solver": str(trig_solver),
        "trig_epsilon": trig_epsilon,
        "machine": platform.machine(),
    }
    cached = cache.load("llvm", record, suffix=".ll")
    if cached is not None:
        metadata = dict(cached.manifest["metadata"])
        feed_names = tuple(metadata["feed_names"])
        outputs = tuple(
            LLVMOutputSpec(
                name=item["name"],
                shape=tuple(item["shape"]),
            )
            for item in metadata["outputs"]
        )
        workspace_sizes = tuple(metadata.get("workspace_sizes", ()))
        return LLVMJITProgram(
            llvm_ir=cached.source,
            function_name=function_name,
            shell_name=shell_name,
            feed_names=feed_names,
            outputs=outputs,
            workspace_sizes=workspace_sizes,
            cache_artifact=cached,
        )

    lowered = lower_abstract_tensor_tape_to_llvm_ssa(
        captured.tape,
        captured.outputs,
        function_name=function_name,
        trig_solver=trig_solver,
        trig_epsilon=trig_epsilon,
    )
    if lowered.shortfalls:
        details = "; ".join(
            f"{shortfall.operation}: {shortfall.reason}"
            for shortfall in lowered.shortfalls
        )
        raise LLVMJITShortfall(
            f"{captured.case.name} cannot lower completely to LLVM: {details}"
        )
    # Keyed by the same token the lowering reports, which is the tape's own
    # ``tensor_identity`` rather than a memory address.
    input_names_by_id = {
        tensor_identity(value): name for name, value in captured.inputs.items()
    }
    try:
        feed_names = tuple(
            input_names_by_id[value_id] for value_id in lowered.feed_ids
        )
    except KeyError as error:
        raise LLVMJITShortfall(
            "LLVM lowering exposed a feed that is not a declared torture-case "
            f"input: {error.args[0]}"
        ) from error
    outputs = tuple(
        LLVMOutputSpec(name=name, shape=tuple(value.shape))
        for name, value in captured.outputs.items()
    )
    workspace_sizes = tuple(lowered.workspace_sizes)
    argument_count = (
        len(feed_names) + len(outputs) + len(workspace_sizes)
    )
    llvm_ir = (
        lowered.llvm_ir.rstrip()
        + "\n\n"
        + _shell_adapter_ir(
            function_name=function_name,
            shell_name=shell_name,
            argument_count=argument_count,
        )
    )
    llvm = _initialize_llvm()
    module = llvm.parse_assembly(llvm_ir)
    module.verify()
    stored = cache.store(
        "llvm",
        record,
        llvm_ir,
        suffix=".ll",
        metadata={
            "feed_names": list(feed_names),
            "outputs": [
                {"name": spec.name, "shape": list(spec.shape)}
                for spec in outputs
            ],
            "workspace_sizes": list(workspace_sizes),
        },
    )
    return LLVMJITProgram(
        llvm_ir=llvm_ir,
        function_name=function_name,
        shell_name=shell_name,
        feed_names=feed_names,
        outputs=outputs,
        workspace_sizes=workspace_sizes,
        cache_artifact=stored,
        _llvm_profile=resolved_profile,
    )


__all__ = [
    "LLVMJITExecution",
    "LLVMJITProgram",
    "LLVMJITShortfall",
    "LLVMOutputSpec",
    "compile_torture_case_to_llvm",
]
