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

WebAssembly is the first member, and it is deliberately the simplest one:
it consumes ``FusedProgram``, the only intermediary with no control flow,
which is exactly what makes it a translation rather than a compiler (see
``fused_program_wasm_backend`` for why WebAssembly's structured control flow
makes the SSA route a much larger problem).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

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
    # Operations the target cannot express at all. A caller can check this
    # against a program's steps before paying for emission.
    unsupported_operations: frozenset[str] = frozenset()
    # The external tool needed to turn emitted text into a binary, if any.
    assembler: str | None = None
    note: str | None = None


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


def get_target(name: str) -> MachineTarget:
    target = _REGISTRY.get(name)
    if target is None:
        raise KeyError(
            f"unknown machine target {name!r}; one of {sorted(_REGISTRY)}"
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
        if not (used & target.capabilities.unsupported_operations)
    )


def emit(program: FusedProgram, target: str, *, name: str = "program") -> TargetArtifact:
    return get_target(target).emit(program, name=name)


# --- WebAssembly -----------------------------------------------------------


class _WasmTarget:
    """WAT text from the flat numeric IR. The hub's first member."""

    def __init__(self):
        from .fused_program_wasm_backend import _NO_WASM_INSTRUCTION

        self.capabilities = TargetCapabilities(
            name="wasm",
            consumes="fused_program",
            emits=".wat",
            # The only loop it writes is the elementwise walk over the
            # extent; it does not lower a program's own control flow.
            control_flow=False,
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


__all__ = [
    "MachineTarget",
    "TargetArtifact",
    "TargetCapabilities",
    "TargetUnavailable",
    "capabilities",
    "emit",
    "get_target",
    "register_target",
    "targets",
    "targets_for",
]
