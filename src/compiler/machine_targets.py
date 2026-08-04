"""One place to ask "which machine-level targets can serve this program?".

Every low-level backend in this repository answers the same three questions
in its own vocabulary: what can you express, what do you emit, and can you
assemble it here. Callers currently learn that by importing a specific
backend and finding out -- which is how the Mandelbrot Fortran route spent a
long time compiling a program that was silently one unrolled iteration, and
how a caller reaches for WebAssembly and only then discovers it has no
``exp``.

This hub makes those answers data. A target declares its capabilities up
front, so a caller can choose *before* compiling, and every target returns
the same ``TargetArtifact`` -- source, shortfalls, and the API descriptor
that says how to call the result (``compiled_program_api.py``).

The distinction that matters here is between emitting and assembling.
Emission is pure text generation and always available; assembly needs a
toolchain that may not be installed. ``available()`` reports the second
without affecting the first, so a target is never quietly skipped for
lacking a compiler -- ``emit`` still works and says so.

WebAssembly was the first member, and it is deliberately the simplest one:
it consumes ``FusedProgram``, the only intermediary with no control flow,
which is exactly what makes it a translation rather than a compiler (see
``fused_program_wasm_backend`` for why WebAssembly's structured control flow
makes the SSA route a much larger problem).

The hub also registers desktop GLSL, Fortran-through-SSA, and WebGL 2.  C and
LLVM are inventoried here but not registered as ``FusedProgram`` printers:
their current JIT entry points consume captured autograd tapes directly.

WebGL 2 is registered but deprecated (``TargetCapabilities.deprecated``):
its fragment-raster ABI predates browser compute shaders.  ``get_target``
and ``emit`` redirect plain ``"webgl"`` lookups to ``"webgpu"`` unless the
caller passes ``allow_deprecated=True``. ``"webgpu"`` emits WGSL compute
shaders from the shared SSA and records its launch shape in API metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..common.tensors.fused_ir import FusedProgram


class TargetUnavailable(RuntimeError):
    """A target was asked to assemble without the toolchain to do it."""


@dataclass(frozen=True)
class TargetCapabilities:
    """What a target can express, stated rather than discovered by failing."""

    name: str
    # Which intermediary it consumes. "fused_program" is the flat numeric IR;
    # "ssa" is the full control-flow module.
    consumes: str
    # The file extension of what emit() produces.
    emits: str
    # Whether the target has native control flow for a loop-bearing program,
    # or only handles the straight-line numeric region.
    control_flow: bool
    # Whether the target can execute rank-bearing tensor operations directly.
    # A non-native target (rank-1 ABI like WASM's run(count, ...)) needs every
    # tensor op unrolled to scalar form before the graph is flattened.
    tensor_native: bool = True
    # A closed operation vocabulary when the target has one. ``None`` is for
    # a target whose acceptance depends on richer IR than an operation name.
    # Closed sets make unknown operations fail selection instead of appearing
    # supported merely because they were absent from a blacklist.
    supported_operations: frozenset[str] | None = None
    # Operations the target cannot express at all. A caller can check this
    # against a program's steps before paying for emission.
    unsupported_operations: frozenset[str] = frozenset()
    # The external tool needed to turn emitted text into a binary, if any.
    assembler: str | None = None
    note: str | None = None
    # Which pipeline stage (shader_stages.STAGES key) this target's emitted
    # text runs as, for targets that are a real GPU shading language.
    # ``None`` for targets with no stage concept at all (Fortran, WASM, C).
    stage: str | None = None
    # A deprecated target is still registered and still emits when asked
    # explicitly (``allow_deprecated=True``), but a caller who names it
    # without that override gets redirected to ``redirect_to`` instead.
    # This is data, like everything else on this dataclass, so deprecating
    # a target is a registration change, not a call-site hunt.
    deprecated: bool = False
    deprecated_reason: str | None = None
    redirect_to: str | None = None

    def supports(self, operations: set[str] | frozenset[str]) -> bool:
        used = frozenset(operations)
        if self.supported_operations is not None and not (
            used <= self.supported_operations
        ):
            return False
        return not bool(used & self.unsupported_operations)


@dataclass(frozen=True)
class BackendOperatorInventory:
    """Read-only view of the finite list owned by an existing backend."""

    backend: str
    consumes: str
    operations: frozenset[str]
    sources: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class TargetArtifact:
    """What every target returns, in the same shape."""

    target: str
    name: str
    source: str
    complete: bool
    shortfalls: tuple[str, ...] = ()
    api: Any = None
    extension: str = ".txt"
    module: Any = None

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.name}{self.extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source, encoding="utf-8")
        if self.api is not None:
            self.api.write(path.with_suffix(".api.yaml"))
        return path

    def shortfall_report(self) -> str:
        if self.complete:
            return f"{self.target}: no shortfalls"
        return f"{self.target} shortfalls:\n" + "\n".join(
            "- " + line for line in self.shortfalls
        )


class MachineTarget(Protocol):
    capabilities: TargetCapabilities

    def emit(self, program: FusedProgram, *, name: str) -> TargetArtifact: ...

    def available(self) -> bool: ...


_REGISTRY: dict[str, MachineTarget] = {}


def register_target(target: MachineTarget) -> None:
    _REGISTRY[target.capabilities.name] = target


def targets() -> Mapping[str, MachineTarget]:
    return dict(_REGISTRY)


def capabilities() -> tuple[TargetCapabilities, ...]:
    return tuple(t.capabilities for t in _REGISTRY.values())


def get_target(name: str, *, allow_deprecated: bool = False) -> MachineTarget:
    """Look up a registered target, honoring deprecation redirects.

    A target marked ``deprecated`` on its capabilities is not removed from
    the registry -- a caller can still reach it with
    ``allow_deprecated=True`` -- but the default lookup silently hands back
    ``redirect_to`` instead, so new call sites land on the replacement
    without every existing caller needing to know a redirect happened.
    """

    target = _REGISTRY.get(name)
    if target is None:
        raise KeyError(
            f"unknown machine target {name!r}; one of {sorted(_REGISTRY)}"
        )
    if target.capabilities.deprecated and not allow_deprecated:
        redirect_name = target.capabilities.redirect_to
        redirected = _REGISTRY.get(redirect_name) if redirect_name else None
        if redirected is not None:
            return redirected
        raise TargetUnavailable(
            f"target {name!r} is deprecated "
            f"({target.capabilities.deprecated_reason or 'no reason given'}) "
            "and has no available replacement registered; pass "
            "allow_deprecated=True to use it anyway"
        )
    return target


def targets_for(program: FusedProgram) -> tuple[str, ...]:
    """Which registered targets can express every step of ``program``.

    Answered from declared capabilities, without emitting anything.
    """

    used = {step.op_name for step in program.steps}
    return tuple(
        name
        for name, target in _REGISTRY.items()
        if target.capabilities.supports(used)
    )


def emit(
    program: FusedProgram,
    target: str,
    *,
    name: str = "program",
    allow_deprecated: bool = False,
) -> TargetArtifact:
    return get_target(target, allow_deprecated=allow_deprecated).emit(
        program, name=name
    )


def operator_inventories() -> tuple[BackendOperatorInventory, ...]:
    """Inventory C, GLSL, Fortran, LLVM and WebGL without copying their lists."""

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_dispatch_operations,
        covered_operations,
    )
    from ..common.tensors.accelerator_backends.glsl_backend import GLSL_OPS
    from .fused_program_webgl_backend import supported_operations as webgl_ops
    from .ssa_fortran_backend import supported_tensor_operations
    from .ssa_webgpu_backend import supported_tensor_operations as webgpu_ops

    return (
        BackendOperatorInventory(
            backend="c",
            consumes="captured_tape",
            operations=c_dispatch_operations(),
            sources=(
                "src/common/tensors/accelerator_backends/c_backend.py:"
                "CAbstractTensor._apply_operator__.binary_codes/unary_codes",
            ),
            note="scalar opcode dispatch; structural kernels have separate entry points",
        ),
        BackendOperatorInventory(
            backend="glsl",
            consumes="fused_program",
            operations=GLSL_OPS,
            sources=(
                "src/common/tensors/accelerator_backends/glsl_backend.py:"
                "_BINARY/_UNARY/GLSL_OPS",
            ),
            note="desktop OpenGL 4.3 compute expression surface",
        ),
        BackendOperatorInventory(
            backend="fortran",
            consumes="ssa",
            operations=supported_tensor_operations(),
            sources=(
                "src/compiler/ssa_fortran_backend.py:"
                "_BINARY/_UNARY/_REDUCTION/_SHAPE_ONLY",
            ),
            note="tensor operations only; SSA control instructions are additional",
        ),
        BackendOperatorInventory(
            backend="llvm",
            consumes="captured_tape",
            operations=covered_operations(),
            sources=(
                "src/common/tensors/accelerator_backends/"
                "c_backend_llvm_ssa.py:TRANSLATIONS",
            ),
            note="authored C-kernel to LLVM-SSA correspondences",
        ),
        BackendOperatorInventory(
            backend="webgl",
            consumes="fused_program",
            operations=webgl_ops(),
            sources=(
                "src/common/tensors/accelerator_backends/glsl_backend.py:"
                "GLSL_FLOAT_SCALAR_OPS",
                "src/compiler/fused_program_webgl_backend.py:"
                "supported_operations",
            ),
            note=(
                "WebGL 2 fragment-raster ABI; no compute shaders or SSBOs. "
                "Deprecated -- get_target/emit redirect to webgpu unless "
                "called with allow_deprecated=True."
            ),
        ),
        BackendOperatorInventory(
            backend="webgpu",
            consumes="fused_program",
            operations=webgpu_ops(),
            sources=(
                "src/compiler/ssa_webgpu_backend.py:"
                "_BINARY/_UNARY/_REDUCTION/_SHAPE_ONLY",
            ),
            note="WebGPU compute shader emitted directly from shared SSA",
        ),
    )


# --- WebAssembly -----------------------------------------------------------


class _WasmTarget:
    """WAT text from the flat numeric IR. The hub's first member."""

    def __init__(self):
        from .fused_program_wasm_backend import (
            _BINARY_INSTRUCTION,
            _COMPARISON_INSTRUCTION,
            _LUT_OPS,
            _NO_WASM_INSTRUCTION,
            _UNARY_INSTRUCTION,
        )

        supported = frozenset(
            set(_BINARY_INSTRUCTION)
            | set(_COMPARISON_INSTRUCTION)
            | set(_LUT_OPS)
            | set(_UNARY_INSTRUCTION)
            | {"tensor_from_list"}
        )

        self.capabilities = TargetCapabilities(
            name="wasm",
            consumes="fused_program",
            emits=".wat",
            # The only loop it writes is the elementwise walk over the
            # extent; it does not lower a program's own control flow.
            control_flow=False,
            # Rank-1 run(count, ...) ABI: tensor ops must be unrolled to scalar
            # form at ProcessGraph intake, before flattening.
            tensor_native=False,
            supported_operations=supported,
            unsupported_operations=frozenset(_NO_WASM_INSTRUCTION),
            assembler="wat2wasm",
            note=(
                "WebAssembly has no transcendental instructions, so exp/log/"
                "pow and the trigonometric family cannot be expressed; they "
                "are reported rather than approximated"
            ),
        )

    def available(self) -> bool:
        from .fused_program_wasm_backend import wat_assembler

        return wat_assembler() is not None

    def emit(self, program: FusedProgram, *, name: str = "program") -> TargetArtifact:
        from .fused_program_wasm_backend import emit_wasm_module

        module = emit_wasm_module(program, name=name)
        return TargetArtifact(
            target="wasm",
            name=name,
            source=module.source,
            complete=module.complete,
            shortfalls=tuple(s.format() for s in module.shortfalls),
            api=module.api,
            extension=".wat",
            module=module,
        )

    def assemble(self, artifact: TargetArtifact, *, directory: str | Path | None = None) -> Path:
        from .fused_program_wasm_backend import compile_wat

        if not self.available():
            raise TargetUnavailable(
                "wat2wasm is not installed; emission does not need it, "
                "assembly does"
            )
        return compile_wat(artifact.module, directory=directory)


register_target(_WasmTarget())


# --- desktop GLSL compute --------------------------------------------------


class _GLSLTarget:
    def __init__(self):
        from ..common.tensors.accelerator_backends.glsl_backend import GLSL_OPS

        self.capabilities = TargetCapabilities(
            name="glsl",
            consumes="fused_program",
            emits=".comp.glsl",
            control_flow=False,
            supported_operations=GLSL_OPS,
            assembler="OpenGL 4.3 driver",
            note="desktop compute shader with std430 SSBO bindings",
            stage="compute",
        )

    def available(self) -> bool:
        from ..common.tensors.accelerator_backends.glsl_backend import (
            gl_context_info,
        )

        return gl_context_info() is not None

    def emit(self, program: FusedProgram, *, name: str = "program") -> TargetArtifact:
        from ..common.tensors.accelerator_backends.glsl_backend import (
            emit_multi_output_program_source,
            emit_program_source,
        )
        from .compiled_program_api import CompiledProgramAPI, EntryPoint

        try:
            source = (
                emit_multi_output_program_source(program)
                if len(program.outputs) > 1
                else emit_program_source(program)
            )
            shortfalls: tuple[str, ...] = ()
        except Exception as error:
            source = ""
            shortfalls = (f"{type(error).__name__}: {error}",)
        api = CompiledProgramAPI(
            module=name,
            language="glsl",
            entry="main",
            entry_points=(EntryPoint("main", "main", "numerical"),),
            metadata={
                "execution_model": "compute",
                "minimum_opengl": "4.3",
                "storage": "std430 SSBO",
            },
        )
        return TargetArtifact(
            target="glsl",
            name=name,
            source=source,
            complete=not shortfalls,
            shortfalls=shortfalls,
            api=api,
            extension=".comp.glsl",
        )


# --- Fortran ---------------------------------------------------------------


class _FortranTarget:
    def __init__(self):
        from .ssa_fortran_backend import supported_tensor_operations

        self.capabilities = TargetCapabilities(
            name="fortran",
            consumes="fused_program->ssa",
            emits=".f90",
            control_flow=True,
            supported_operations=supported_tensor_operations(),
            assembler="gfortran/ifx/ifort/flang/lfortran",
            note="FusedProgram is first lowered to the shared repository SSA",
        )

    def available(self) -> bool:
        from .ssa_fortran_backend import fortran_compiler

        return fortran_compiler() is not None

    def emit(self, program: FusedProgram, *, name: str = "program") -> TargetArtifact:
        from ..transmogrifier.ssa import IRModule
        from .precompile_to_ssa import lower_fused_program_to_ssa
        from .ssa_fortran_backend import emit_module

        function, lowering_shortfalls = lower_fused_program_to_ssa(
            program, function_name=name,
        )
        returns = tuple(
            instruction.args
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        )
        output_values = returns[-1] if returns else ()
        module = emit_module(
            IRModule({name: function}),
            # A Fortran module and a procedure contained by that module may
            # not share an identifier. Keep the public procedure/ABI name and
            # give only the enclosing compilation unit a distinct suffix.
            name=f"{name}_fortran",
            outputs={name: output_values},
        )
        shortfalls = tuple(
            f"{item.domain}:{item.name} at {item.location}: {item.reason}"
            for item in lowering_shortfalls
        ) + tuple(item.format() for item in module.shortfalls)
        return TargetArtifact(
            target="fortran",
            name=name,
            source=module.source,
            complete=not shortfalls,
            shortfalls=shortfalls,
            api=module.api,
            extension=".f90",
            module=module,
        )


# --- WebGL 2 ---------------------------------------------------------------


class _WebGLTarget:
    def __init__(self):
        from .fused_program_webgl_backend import supported_operations

        self.capabilities = TargetCapabilities(
            name="webgl",
            consumes="fused_program",
            emits=".frag.glsl",
            control_flow=False,
            supported_operations=supported_operations(),
            assembler="WebGL 2 browser driver",
            note=(
                "fragment-raster target: sampler2D inputs and up to four "
                "float color-attachment outputs"
            ),
            deprecated=True,
            deprecated_reason=(
                "WebGL 2's fragment-raster ABI (no compute shaders, no "
                "SSBOs, values smuggled through sampler2D/color attachments) "
                "is a workaround for a browser that had no compute path. "
                "WebGPU has one. New callers should target webgpu; webgl "
                "stays registered and fully functional for callers that "
                "pass allow_deprecated=True, since browsers without WebGPU "
                "support still exist."
            ),
            redirect_to="webgpu",
            stage="fragment",
        )

    def available(self) -> bool:
        # Emission has no toolchain dependency. Runtime availability is a
        # property of the browser/graphics context that receives the artifact.
        return True

    def emit(self, program: FusedProgram, *, name: str = "program") -> TargetArtifact:
        from .fused_program_webgl_backend import emit_webgl_fragment_module

        module = emit_webgl_fragment_module(program, name=name)
        return TargetArtifact(
            target="webgl",
            name=name,
            source=module.source,
            complete=module.complete,
            shortfalls=tuple(item.format() for item in module.shortfalls),
            api=module.api,
            extension=".frag.glsl",
            module=module,
        )


# --- WebGPU compute ---------------------------------------------------------


class _WebGPUTarget:
    """WGSL compute text from a FusedProgram lowered through shared SSA."""

    def __init__(self):
        from .ssa_webgpu_backend import supported_tensor_operations

        self.capabilities = TargetCapabilities(
            name="webgpu",
            consumes="fused_program->ssa",
            emits=".wgsl",
            control_flow=True,
            supported_operations=supported_tensor_operations(),
            unsupported_operations=frozenset(),
            assembler="WebGPU browser driver",
            note="FusedProgram is lowered to shared SSA and emitted as WGSL compute",
            stage="compute",
        )

    def available(self) -> bool:
        return True

    def emit(self, program: FusedProgram, *, name: str = "program") -> TargetArtifact:
        from ..transmogrifier.ssa import IRModule
        from .precompile_to_ssa import lower_fused_program_to_ssa
        from .ssa_webgpu_backend import emit_module

        function, lowering_shortfalls = lower_fused_program_to_ssa(
            program, function_name=name,
        )
        returns = tuple(
            instruction.args
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        )
        output_values = returns[-1] if returns else ()
        count = max(
            (int(size) for value in function.args for size in value.shape),
            default=1,
        )
        module = emit_module(
            IRModule({name: function}), name=name,
            outputs={name: output_values}, count=count,
        )
        shortfalls = tuple(
            f"{item.domain}:{item.name} at {item.location}: {item.reason}"
            for item in lowering_shortfalls
        ) + tuple(item.format() for item in module.shortfalls)
        return TargetArtifact(
            target="webgpu",
            name=name,
            source=module.source,
            complete=not shortfalls,
            shortfalls=shortfalls,
            api=module.api,
            extension=".wgsl",
            module=module,
        )


register_target(_GLSLTarget())
register_target(_FortranTarget())
register_target(_WebGLTarget())
register_target(_WebGPUTarget())


__all__ = [
    "BackendOperatorInventory",
    "MachineTarget",
    "TargetArtifact",
    "TargetCapabilities",
    "TargetUnavailable",
    "capabilities",
    "emit",
    "get_target",
    "operator_inventories",
    "register_target",
    "targets",
    "targets_for",
]
