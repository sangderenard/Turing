"""Adapt existing repository SSA to the established WebGL fragment emitter.

This module does not add an intermediate operator vocabulary.  It accepts only
existing ``Handler`` instructions, converts the scalar subset to the existing
``FusedProgram`` semantic names, and lets ``fused_program_webgl_backend`` own
GLSL ES spelling and raster ABI details.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.tensors.accelerator_backends.glsl_backend import (
    GLSL_FLOAT_SCALAR_OPS,
)
from ..common.tensors.fused_ir import FusedProgram, OpStep
from ..transmogrifier.ssa import Function
from ..transmogrifier.ssa_registry import Handler
from .fused_program_webgl_backend import (
    WebGLFragmentModule,
    WebGLShortfall,
    emit_webgl_fragment_module,
)
from .precompile_ssa_validator import PRECOMPILE_TO_SSA


_SSA_TO_CANONICAL: dict[Handler, str] = {}
for _spelling, _handler in PRECOMPILE_TO_SSA.items():
    _SSA_TO_CANONICAL.setdefault(_handler, _spelling)


class SSAWebGLEmissionError(ValueError):
    """Repository SSA still contains a non-WebGL compiler dialect."""


@dataclass(frozen=True, order=True)
class SSAWebGLShortfall:
    instruction_index: int
    operation: str
    reason: str


@dataclass(frozen=True)
class SSAFusedProgramResult:
    program: FusedProgram
    shortfalls: tuple[SSAWebGLShortfall, ...]

    @property
    def complete(self) -> bool:
        return not self.shortfalls


def _precision_section_shortfalls(
    function: Function, module=None,
) -> tuple[SSAWebGLShortfall, ...]:
    """Refuse precision sections BEFORE the fused path can swallow them.

    GLSL ES 3.0 has no fma() and no contraction control, so "webgl" claims
    nothing in ``BACKEND_PRECISION_CAPABILITIES`` and a section reaching this
    lane is unhostable. Refusing has to happen here: the underlying
    FusedProgram emitter substitutes ``float x = 0.0; // unsupported`` for an
    unknown op, and a section's expansion is plain Add/Sub/Mul that IS in the
    scalar surface -- it would emit "successfully", the driver would remain
    free to fold the error terms (which are algebraically zero), and the
    residual would come back zero with nothing announcing the failure.

    When the caller holds the IRModule the function came from, the pipeline
    receipt on its metadata is authoritative and ``precision_backend_
    shortfalls`` reads the contracts from it. This entry point is often
    handed a bare Function, though (autogenesis does exactly that), so there
    is a fallback with no receipt to lean on: the ``precision_section``
    attribute stamped on every section instruction is, by ir_identities' own
    words, "the one thing about the section that must reach a destination",
    and it is enough to reconstruct the obligations -- every section demands
    SECTION_ISOLATION, and FMA_MANDATORY exactly when an explicit ``Fma``
    survives in the marked run.
    """

    from .ir_identities import (
        BACKEND_PRECISION_CAPABILITIES,
        FMA_MANDATORY,
        PRECISION_SECTION_ATTRIBUTE,
        SECTION_ISOLATION,
        precision_backend_shortfalls,
    )

    if module is not None:
        return tuple(
            SSAWebGLShortfall(
                -1, "precision_section",
                "backend cannot honour precision obligations "
                + repr(item["missing"]),
            )
            for item in precision_backend_shortfalls(
                module, "webgl", (function.name,)
            )
        )
    marked = [
        instruction
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get(PRECISION_SECTION_ATTRIBUTE)
    ]
    if not marked:
        return ()
    required = {SECTION_ISOLATION}
    if any(str(instruction.op) == "Fma" for instruction in marked):
        required.add(FMA_MANDATORY)
    missing = tuple(sorted(
        required - set(BACKEND_PRECISION_CAPABILITIES.get("webgl", ()))
    ))
    if not missing:
        return ()
    return (SSAWebGLShortfall(
        -1, "precision_section",
        "backend cannot honour precision obligations " + repr(missing),
    ),)


def ssa_function_to_fused_program(
    function: Function, *, module=None,
) -> SSAFusedProgramResult:
    """Select the existing float-scalar FusedProgram subset from one function.

    ``module`` is optional and only consulted for the precision-pipeline
    receipt on its metadata -- see ``_precision_section_shortfalls``.
    """

    from .machine_dialect_ssa import (
        format_machine_dialect_occurrences,
        module_machine_dialect_occurrences,
    )

    machine_residuals = module_machine_dialect_occurrences(
        {function.name: function}
    )
    if machine_residuals:
        raise SSAWebGLEmissionError(
            "WebGL emission requires legalized repository SSA; retained "
            "machine-state SSA remains: "
            + format_machine_dialect_occurrences(machine_residuals)
        )

    precision_shortfalls = _precision_section_shortfalls(function, module)

    from .evolution_metagraph import (
        EvolutionComponentRef,
        active_evolution_metagraph,
    )

    evolution = active_evolution_metagraph()
    source_graph = (
        None if evolution is None else evolution.graph_for_artifact(function)
    )
    adapter_graph = (
        None
        if evolution is None
        else evolution.open_graph("backend-adapter:webgl", function.name)
    )

    def record_value(value_id: int, label: str, kind: str) -> None:
        if evolution is None or adapter_graph is None:
            return
        source = (
            None
            if source_graph is None
            else EvolutionComponentRef(source_graph.id, str(value_id))
        )
        sources = (source,) if source is not None and evolution.has_component(source) else ()
        target = evolution.component(
            adapter_graph,
            value_id,
            label=label,
            kind=kind,
            attributes={"granularity": "exact-value"},
            consumes=sources,
        )
        if sources:
            evolution.handoff(
                target,
                sources,
                transformation="ssa-to-webgl-adapter",
                detail={"granularity": "exact-value"},
            )

    for argument in function.args:
        record_value(argument.id, argument.name(), "feed")

    if set(function.blocks) != {"entry"}:
        program = FusedProgram(
            version=1,
            feeds={value.id for value in function.args},
            steps=[],
            outputs={},
        )
        if evolution is not None and adapter_graph is not None:
            evolution.bind_artifact(program, adapter_graph)
            evolution.close_graph(adapter_graph)
        return SSAFusedProgramResult(program, precision_shortfalls + (
            SSAWebGLShortfall(
                -1,
                "control_flow",
                "WebGL fragment emission requires one straight-line SSA entry block",
            ),
        ))

    steps: list[OpStep] = []
    outputs: dict[str, int] = {}
    shortfalls: list[SSAWebGLShortfall] = list(precision_shortfalls)
    available = {value.id for value in function.args}
    instructions = function.blocks["entry"].instrs
    for index, instruction in enumerate(instructions):
        try:
            handler = Handler(instruction.op)
        except ValueError:
            try:
                handler = Handler[instruction.op]
            except KeyError:
                shortfalls.append(SSAWebGLShortfall(
                    index,
                    instruction.op,
                    "instruction is not in the existing SSA Handler set",
                ))
                continue

        if handler is Handler.Ret:
            outputs = {
                f"result_{output_index}": value.id
                for output_index, value in enumerate(instruction.args)
            }
            continue
        if instruction.res is None:
            shortfalls.append(SSAWebGLShortfall(
                index, handler.value, "value instruction has no SSA result",
            ))
            continue

        attrs = dict(instruction.attributes)
        if handler is Handler.Const:
            if "value" not in attrs:
                shortfalls.append(SSAWebGLShortfall(
                    index, handler.value, "constant has no value attribute",
                ))
                continue
            operation = "tensor_from_list"
            attrs = {"values": attrs["value"]}
        elif handler is Handler.Call:
            # Numeric SSA calls retain both the implementation kernel and the
            # canonical tensor operation. WebGL consumes the latter through
            # the existing GLSL expression table; it must not mistake a C
            # kernel spelling such as ``binary_double`` for a new operator.
            tensor_operation = str(attrs.get("tensor_operation", ""))
            operation = (
                tensor_operation
                if tensor_operation in GLSL_FLOAT_SCALAR_OPS
                else str(attrs.get("callee", ""))
            )
            if operation not in GLSL_FLOAT_SCALAR_OPS:
                shortfalls.append(SSAWebGLShortfall(
                    index,
                    handler.value,
                    f"callee {operation!r} is not in the existing WebGL scalar surface",
                ))
                continue
            attrs = {
                key: attrs[key]
                for key in ("right_scalar", "left_scalar", "reverse")
                if key in attrs
            }
        else:
            operation = _SSA_TO_CANONICAL.get(handler, "")
            if operation not in GLSL_FLOAT_SCALAR_OPS:
                shortfalls.append(SSAWebGLShortfall(
                    index,
                    handler.value,
                    "existing SSA instruction has no WebGL float-scalar lowering",
                ))
                continue
            attrs = {}

        missing = [value.id for value in instruction.args if value.id not in available]
        if missing:
            shortfalls.append(SSAWebGLShortfall(
                index,
                handler.value,
                "operands have no emitted producer: "
                + ", ".join(map(str, missing)),
            ))
            continue
        steps.append(OpStep(
            step_id=len(steps),
            op_name=operation,
            input_ids=[value.id for value in instruction.args],
            attrs=attrs,
            result_id=instruction.res.id,
        ))
        record_value(instruction.res.id, operation, "operation")
        if evolution is not None and adapter_graph is not None:
            target = EvolutionComponentRef(adapter_graph.id, str(instruction.res.id))
            for position, argument in enumerate(instruction.args):
                source = EvolutionComponentRef(adapter_graph.id, str(argument.id))
                if evolution.has_component(source):
                    evolution.relationship(
                        adapter_graph,
                        source,
                        target,
                        role=f"arg{position}",
                    )
        available.add(instruction.res.id)

    for name, value_id in tuple(outputs.items()):
        if value_id not in available:
            shortfalls.append(SSAWebGLShortfall(
                -1, "output", f"value {value_id} has no emitted producer",
            ))

    program = FusedProgram(
            version=1,
            feeds={value.id for value in function.args},
            steps=steps,
            outputs=outputs,
    )
    if evolution is not None and adapter_graph is not None:
        evolution.bind_artifact(program, adapter_graph)
        evolution.close_graph(adapter_graph)
    return SSAFusedProgramResult(program, tuple(shortfalls))


def emit_ssa_webgl_fragment_module(
    function: Function,
    *,
    name: str | None = None,
    module=None,
) -> WebGLFragmentModule:
    """Emit WebGL from repository SSA and retain ordinary named shortfalls.

    ``module`` is optional and only consulted for the precision-pipeline
    receipt on its metadata -- see ``_precision_section_shortfalls``.
    """

    lowering = ssa_function_to_fused_program(function, module=module)
    emitted = emit_webgl_fragment_module(
        lowering.program, name=name or function.name
    )
    conversion_shortfalls = tuple(
        WebGLShortfall(
            item.instruction_index,
            item.operation,
            item.reason,
        )
        for item in lowering.shortfalls
    )
    result = WebGLFragmentModule(
        name=emitted.name,
        source=emitted.source,
        shortfalls=conversion_shortfalls + emitted.shortfalls,
        api=emitted.api,
        vertex_source=emitted.vertex_source,
        io_layout=emitted.io_layout,
        component_abi=emitted.component_abi,
    )
    from .evolution_metagraph import (
        EvolutionComponentRef,
        active_evolution_metagraph,
    )

    evolution = active_evolution_metagraph()
    if evolution is not None:
        adapter_graph = evolution.graph_for_artifact(lowering.program)
        backend_graph = evolution.open_graph("backend:webgl", result.name)
        values = [
            (value_id, f"feed {value_id}", "sampler")
            for value_id in sorted(lowering.program.feeds)
        ] + [
            (step.result_id, step.op_name, "glsl-expression")
            for step in lowering.program.steps
        ]
        for value_id, label, kind in values:
            source = (
                None
                if adapter_graph is None
                else EvolutionComponentRef(adapter_graph.id, str(value_id))
            )
            sources = (
                (source,)
                if source is not None and evolution.has_component(source)
                else ()
            )
            target = evolution.component(
                backend_graph,
                value_id,
                label=label,
                kind=kind,
                attributes={
                    "generated_symbol": f"v_{value_id}",
                    "granularity": "exact-value",
                },
                consumes=sources,
            )
            if sources:
                evolution.handoff(
                    target,
                    sources,
                    transformation="webgl-adapter-to-glsl-es",
                    detail={"granularity": "exact-value"},
                )
        evolution.bind_artifact(result, backend_graph)
        evolution.close_graph(backend_graph)
    return result


__all__ = [
    "SSAWebGLEmissionError",
    "SSAFusedProgramResult",
    "SSAWebGLShortfall",
    "emit_ssa_webgl_fragment_module",
    "ssa_function_to_fused_program",
]
