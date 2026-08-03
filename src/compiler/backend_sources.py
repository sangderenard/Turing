"""Emit one AOT compilation through every backend, and say what came out.

The point is not any single language. It is that one ordinary Python
function, ingested as an AST, becomes a ProcessGraph, is planned and lowered
once, and then comes out the other side as Fortran, as SSA, as SPIR-V, as
GLSL, as WebAssembly, and as plain NumPy or PyTorch -- all from the same
compilation, not from six re-implementations that have to be kept in step by
hand.

The per-backend artifacts handled here are internal projections of a Python
compilation that has already passed through AST and ProcessGraph ingestion.
This module is not a collection of compiler entrypoints.  In particular, a
backend accepting a normalized numerical ``FusedProgram`` does not mean the
Python recompiler accepts only numerical functions; it means this late stage
is emitting the numerical regions selected by the complete compilation.

Everything here starts from the AOT compilation
(``aot_compile.compile_ast_aot`` -> ``lower_precompile_and_control_to_ssa``).
That is the shared stage: the SSA module is what the language backends
consume, so a source produced here is genuinely the same program as its
neighbours in the tab beside it.

A backend that cannot serve a particular program is recorded as such, with
the reason it gave. That is the honest outcome and it is worth showing: the
set of languages a program *can* reach is a real property of the program,
and a tab that quietly went missing would hide it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BackendSource:
    """One backend's attempt at one program."""

    language: str
    # A short label for a tab.
    title: str
    source: str = ""
    available: bool = True
    reason: str = ""
    # Rough syntax family, so a page can style it without guessing.
    highlight: str = "text"
    # The in-memory compiler artifact, when the emitter has one.  This is not
    # serialized into the page; the bundle builder uses it to compile and
    # verify the exact source that it publishes.
    artifact: Any = field(default=None, repr=False, compare=False)
    # Deployment role is separate from source language.  A browser shader is
    # only allowed to take over the generated page when the bundle explicitly
    # publishes it as the shader surface.
    role: str = ""

    @property
    def lines(self) -> int:
        return self.source.count("\n") + 1 if self.source else 0


@dataclass(frozen=True)
class BackendSourceSet:
    sources: tuple[BackendSource, ...] = ()

    def available(self) -> tuple[BackendSource, ...]:
        return tuple(s for s in self.sources if s.available)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sources": [
                {
                    "language": s.language,
                    "title": s.title,
                    "source": s.source,
                    "available": s.available,
                    "reason": s.reason,
                    "highlight": s.highlight,
                    "lines": s.lines,
                    **({"role": s.role} if s.role else {}),
                }
                for s in self.sources
            ]
        }


def normalized_program(program: Any) -> Any:
    """Drop dead steps and fold one-element constants into their consumers.

    An AOT-compiled program records every value the observation produced and
    keeps its literals as recorded constructor steps (``tensor_from_list``),
    because turning one back into a Python number would contradict
    AbstractTensor's own type decision. That is right for the capture, but a
    language backend then sees a "tensor" that is a single number, and most
    of them have no lowering for one -- which is why GLSL and the Python
    dialects refused a program that is otherwise entirely within their
    vocabulary.

    Folding is done once here rather than in each backend, so every backend
    sees the same normalized program and none of them has to grow a private
    rule about constants.
    """

    from ..common.tensors.fused_ir import FusedProgram

    steps = list(getattr(program, "steps", ()))
    constants: dict[int, float] = {}
    for step in steps:
        if step.op_name != "tensor_from_list":
            continue
        values = step.attrs.get("values")
        while isinstance(values, (list, tuple)) and len(values) == 1:
            values = values[0]
        if isinstance(values, bool):
            constants[step.result_id] = float(values)
        elif isinstance(values, (int, float)):
            constants[step.result_id] = float(values)

    rewritten = []
    for step in steps:
        if step.op_name == "tensor_from_list" and step.result_id in constants:
            continue
        inputs = list(step.input_ids)
        attrs = dict(step.attrs)
        if len(inputs) == 2 and inputs[1] in constants and "right_scalar" not in attrs:
            attrs["right_scalar"] = constants[inputs[1]]
            inputs = [inputs[0]]
        elif len(inputs) == 2 and inputs[0] in constants and "right_scalar" not in attrs:
            attrs["right_scalar"] = constants[inputs[0]]
            attrs["reverse"] = True
            inputs = [inputs[1]]
        if any(value in constants for value in inputs):
            # A constant reaching a unary op has no scalar form; leave the
            # step alone rather than inventing one.
            continue
        rewritten.append(type(step)(
            step_id=step.step_id, op_name=step.op_name, input_ids=inputs,
            attrs=attrs, result_id=step.result_id,
            mode_sensitive=step.mode_sensitive, level=step.level,
        ))

    # Meta dtypes arrive spelled however the observing backend spelled them
    # ("torch.float64"), and a backend that looks the name up in a table --
    # SPIR-V does -- fails on the qualified form. Normalize once here.
    meta = program.meta
    if meta:
        normalized_meta = {}
        for value_id, entry in meta.items():
            dtype = getattr(entry, "dtype", None)
            if isinstance(dtype, str) and "." in dtype:
                entry = type(entry)(
                    shape=entry.shape, dtype=dtype.rsplit(".", 1)[-1],
                    device=entry.device,
                )
            normalized_meta[value_id] = entry
        meta = normalized_meta

    folded = FusedProgram(
        version=program.version, feeds=set(program.feeds), steps=rewritten,
        outputs=dict(program.outputs), state_in=program.state_in,
        meta=meta, extras=program.extras,
    )
    from .fused_program_wasm_backend import required_steps

    live = required_steps(folded)
    return FusedProgram(
        version=folded.version, feeds=set(folded.feeds), steps=live,
        outputs=dict(folded.outputs), state_in=folded.state_in,
        meta=folded.meta, extras=folded.extras,
    )


def _ssa_module(lowering: Any, names: Sequence[str]):
    from ..transmogrifier.ssa import IRModule

    present = [n for n in names if n in lowering.module.functions]
    return IRModule(
        {n: lowering.module.functions[n] for n in present},
        recursion_table={
            name: dict(lowering.module.recursion_table[name])
            for name in present
            if name in lowering.module.recursion_table
        },
    ), present


def _returned_values(function: Any) -> tuple[Any, ...]:
    """Return the SSA values carried by the function's return instruction."""

    returns = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op).casefold() == "ret"
    ]
    if len(returns) != 1:
        raise ValueError(
            f"{function.name} must have exactly one return instruction; "
            f"found {len(returns)}"
        )
    return tuple(returns[0].args)


def collect_backend_sources(
    aot: Any,
    *,
    numerical_name: str = "kernel",
    control_name: str = "kernel_control",
    channel: Any = None,
    wasm_source: str = "",
    program: Any = None,
) -> BackendSourceSet:
    """Emit internal products of ``aot`` through every backend that takes them.

    ``aot`` is an ``AOTCompilation`` from a ``precompile_only=True`` run, so
    the numeric and control IR are backend-agnostic at this point and no
    backend has been privileged by getting there first.

    This function is downstream of the Python compiler frontend.  The local
    ``program`` variable below is only its normalized numerical member, not a
    replacement input language and not the application's compilation path.
    """

    def note(message: str, **detail: Any) -> None:
        if channel is not None:
            channel.log(message, path="sources", **detail)

    def failed(message: str, **detail: Any) -> None:
        if channel is not None:
            channel.error(message, path="sources", **detail)

    collected: list[BackendSource] = []
    # A captured program names every observed value as an output, including
    # ones nothing reads. Dead-step pruning cannot remove those -- they are
    # outputs -- so a caller that knows which result it actually wants says
    # so, and the rest becomes prunable.
    # INTERNAL NUMERICAL MEMBER: Python AST ingestion and control/map planning
    # have already happened.  Do not move this extraction outward into an
    # application entrypoint or describe it as the Python compiler's scope.
    raw = program if program is not None else getattr(
        aot.compiled_shell_program, "program", aot.compiled_shell_program
    )
    program = normalized_program(raw)
    note("normalized numeric program",
         steps_before=len(getattr(raw, "steps", ())),
         steps_after=len(program.steps))

    # --- the shared SSA stage ------------------------------------------
    lowering = None
    try:
        from .precompile_to_ssa import lower_precompile_and_control_to_ssa

        lowering = lower_precompile_and_control_to_ssa(
            aot.compiled_shell_program,
            aot.shell_control_program,
            region_programs=aot.region_programs,
            numerical_name=numerical_name,
            control_name=control_name,
            identity_table=getattr(aot, "identity_table", {}),
            function_outputs=tuple(getattr(aot, "function_outputs", ())),
            function_parameters=tuple(getattr(aot, "function_parameters", ())),
        )
        if not lowering.complete:
            failed("SSA lowering incomplete", detail=lowering.shortfall_report()[:400])
            lowering = None
        else:
            note("SSA lowering complete",
                 functions=len(lowering.module.functions))
    except Exception as error:  # pragma: no cover - depends on the program
        failed(f"SSA lowering raised: {type(error).__name__}: {error}")
        lowering = None

    if lowering is not None:
        try:
            text = "\n\n".join(
                str(lowering.module.functions[name])
                for name in (numerical_name, control_name)
                if name in lowering.module.functions
            )
            collected.append(BackendSource(
                language="ssa", title="SSA", source=text, highlight="llvm",
            ))
        except Exception as error:
            collected.append(BackendSource(
                language="ssa", title="SSA", available=False,
                reason=f"{type(error).__name__}: {error}",
            ))

    # --- Fortran --------------------------------------------------------
    if lowering is None:
        collected.append(BackendSource(
            language="fortran", title="Fortran", available=False,
            reason="needs the SSA stage, which did not complete",
        ))
    else:
        try:
            from .ssa_fortran_backend import emit_module as fortran_emit

            names = [numerical_name, control_name] + [
                n for n in lowering.module.functions
                if n.startswith("numerical_region_")
            ]
            module, present = _ssa_module(lowering, names)
            emitted = fortran_emit(
                module,
                name="kernel_fortran",
                outputs={
                    function_name: _returned_values(function)
                    for function_name, function in module.functions.items()
                },
            )
            collected.append(BackendSource(
                language="fortran", title="Fortran",
                source=emitted.source,
                available=emitted.complete,
                reason="" if emitted.complete else emitted.shortfalls[0].format(),
                highlight="fortran",
                artifact=emitted,
            ))
            note("Fortran emitted", functions=len(present),
                 complete=emitted.complete)
        except Exception as error:
            collected.append(BackendSource(
                language="fortran", title="Fortran", available=False,
                reason=f"{type(error).__name__}: {error}",
            ))
            failed(f"Fortran emission raised: {type(error).__name__}: {error}")

    # --- SPIR-V ---------------------------------------------------------
    if lowering is None:
        collected.append(BackendSource(
            language="spirv", title="SPIR-V", available=False,
            reason="needs the SSA stage, which did not complete",
        ))
    else:
        try:
            from .ssa_spirv_backend import emit_module as spirv_emit

            module, _present = _ssa_module(lowering, [numerical_name])
            # SSA values carry whatever dtype spelling the observing backend
            # used ("torch.float64"), and SPIR-V looks the name up in a
            # table. Normalize on the module handed to it rather than
            # teaching every backend the qualified form.
            for function in module.functions.values():
                for value in list(function.args):
                    dtype = getattr(value, "dtype", None)
                    if isinstance(dtype, str) and "." in dtype:
                        try:
                            value.dtype = dtype.rsplit(".", 1)[-1]
                        except Exception:
                            pass
                for block in function.blocks.values():
                    for instruction in block.instrs:
                        result = getattr(instruction, "res", None)
                        dtype = getattr(result, "dtype", None)
                        if isinstance(dtype, str) and "." in dtype:
                            try:
                                result.dtype = dtype.rsplit(".", 1)[-1]
                            except Exception:
                                pass
            emitted = spirv_emit(module)
            text = getattr(emitted, "assembly", None) or getattr(
                emitted, "source", None
            ) or str(emitted)
            collected.append(BackendSource(
                language="spirv", title="SPIR-V", source=str(text),
                highlight="llvm",
            ))
            note("SPIR-V emitted")
        except Exception as error:
            collected.append(BackendSource(
                language="spirv", title="SPIR-V", available=False,
                reason=f"{type(error).__name__}: {error}",
            ))

    # --- GLSL -----------------------------------------------------------
    try:
        from ..common.tensors.accelerator_backends.glsl_backend import (
            emit_multi_output_program_source,
            emit_program_source,
        )

        # A same-shape multi-output region is one shader writing several
        # SSBOs while sharing every intermediate -- the backend has done this
        # since the four-plane Mandelbrot dispatch. Only the single-output
        # entry point refuses it, so pick the door that matches the program
        # rather than reporting the backend cannot serve it.
        emit = (
            emit_multi_output_program_source
            if len(program.outputs) > 1
            else emit_program_source
        )

        collected.append(BackendSource(
            language="glsl", title="GLSL",
            source=emit(program), highlight="c",
        ))
        note("GLSL emitted")
    except Exception as error:
        collected.append(BackendSource(
            language="glsl", title="GLSL", available=False,
            reason=f"{type(error).__name__}: {error}",
        ))

    # --- WebGL 2 --------------------------------------------------------
    # WebGL shares GLSL scalar expressions, not desktop compute storage.
    # Its fragment-raster adapter therefore gets its own source tab and an
    # honest shortfall when the program needs a non-browser-native layout.
    try:
        from .fused_program_webgl_backend import emit_webgl_fragment_module

        emitted = emit_webgl_fragment_module(program, name=numerical_name)
        collected.append(BackendSource(
            language="webgl",
            title="WebGL 2",
            source=emitted.source,
            available=emitted.complete,
            reason=(
                "" if emitted.complete
                else emitted.shortfalls[0].format()
            ),
            highlight="c",
            role="shader-surface",
        ))
        note("WebGL emitted", complete=emitted.complete)
    except Exception as error:
        collected.append(BackendSource(
            language="webgl", title="WebGL 2", available=False,
            reason=f"{type(error).__name__}: {error}",
        ))

    # --- WebAssembly ----------------------------------------------------
    if wasm_source:
        collected.append(BackendSource(
            language="wat", title="WebAssembly", source=wasm_source,
            highlight="lisp",
        ))

    # --- Python dialects -------------------------------------------------
    try:
        from ..common.tensors.fused_ir import ordered_feed_ids
        from .fused_program_python_backend import compile_single_region_python

        feed_names = {
            feed_id: f"feed{index}"
            for index, feed_id in enumerate(ordered_feed_ids(program))
        }
        for dialect, title in (
            ("numpy", "NumPy"),
            ("torch", "PyTorch"),
            ("abstract_tensor", "AbstractTensor"),
        ):
            try:
                compiled = compile_single_region_python(
                    program, feed_names, dialect=dialect,
                    function_name="kernel",
                )
                collected.append(BackendSource(
                    language=dialect, title=title, source=compiled.source,
                    highlight="python",
                ))
            except Exception as error:
                collected.append(BackendSource(
                    language=dialect, title=title, available=False,
                    reason=f"{type(error).__name__}: {error}",
                ))
        note("Python dialects emitted")
    except Exception as error:
        failed(f"Python emission raised: {type(error).__name__}: {error}")

    served = sum(1 for s in collected if s.available)
    note(f"{served} of {len(collected)} backends served this program")
    return BackendSourceSet(tuple(collected))


__all__ = ["BackendSource", "BackendSourceSet", "collect_backend_sources"]
