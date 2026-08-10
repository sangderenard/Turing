"""Record one real compiler run as a lineage-preserving autogenesis graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .evolution_metagraph import (
    EvolutionComponentRef,
    EvolutionMetaGraph,
    record_evolution,
    record_fused_program_evolution,
)


@dataclass(frozen=True, slots=True)
class AutogenesisCompilation:
    metagraph: EvolutionMetaGraph
    aot: Any
    ssa: Any
    final_artifact: Any = None


def _record_planned_process_graph(
    metagraph: EvolutionMetaGraph,
    process_graph: Any,
    *,
    label: str,
):
    """Record an already-planned semantic graph and its source-span lineage."""

    graph = metagraph.open_graph("process-graph", label)
    existing = metagraph.snapshot().components

    def span_of(data: Mapping[str, Any]):
        span = data.get("source_span")
        if span:
            return dict(span)
        expression = data.get("expr_obj")
        if expression is not None and hasattr(expression, "lineno"):
            return {
                "line": getattr(expression, "lineno", None),
                "column": getattr(expression, "col_offset", None),
                "end_line": getattr(expression, "end_lineno", None),
                "end_column": getattr(expression, "end_col_offset", None),
            }
        return None

    for node_id, data in process_graph.G.nodes(data=True):
        span = span_of(data)
        sources = tuple(
            component.ref
            for component in existing
            if component.ref.graph_id != graph.id
            and component.attributes.get("source_span")
            and dict(component.attributes["source_span"]) == span
        ) if span else ()
        target = metagraph.component(
            graph,
            node_id,
            label=str(data.get("label") or data.get("op") or node_id),
            kind=str(data.get("type") or data.get("op") or "operation"),
            attributes={"source_span": span},
            consumes=sources,
        )
        if sources:
            metagraph.handoff(
                target,
                sources,
                transformation="ingestion-to-semantic-process-graph",
            )
    for source, target, data in process_graph.G.edges(data=True):
        metagraph.relationship(
            graph,
            EvolutionComponentRef(graph.id, str(source)),
            EvolutionComponentRef(graph.id, str(target)),
            role=str(data.get("role") or "data"),
        )
    metagraph.close_graph(graph)
    return graph


def compile_source_autogenesis(
    source: str,
    entrypoint: str,
    feeds: Mapping[str, Any],
    *,
    final_target: str | None = "webgl",
    metagraph: EvolutionMetaGraph | None = None,
) -> AutogenesisCompilation:
    """Compile once while recording exact cross-IR component lineage."""

    from ..common.tensors.accelerator_backends.aot_compile import compile_ast_aot
    from .glsl_deployment_strategy import _walk_planned_shells
    from .precompile_to_ssa import lower_precompile_and_control_to_ssa

    metagraph = metagraph or EvolutionMetaGraph()
    region_evolution: dict[int, Any] = {}
    with record_evolution(metagraph):
        aot = compile_ast_aot(
            source,
            entrypoint,
            dict(feeds),
            precompile_only=True,
        )

        # Region programs have already been remapped to their owning semantic
        # ProcessGraph value IDs. Record that exact correspondence before SSA
        # consumes the same program objects.
        for shell in _walk_planned_shells(aot.deployment):
            process_graph = getattr(shell, "process_graph", None)
            source_graph = getattr(
                process_graph,
                "_evolution_graph",
                None,
            )
            for region_index, captured in (
                getattr(shell, "captured_region_programs", {}) or {}
            ).items():
                program = getattr(captured, "program", captured)
                program_ids = {
                    *map(int, program.feeds),
                    *(int(step.result_id) for step in program.steps),
                }
                if (
                    source_graph is None
                    or not all(metagraph.has_component(EvolutionComponentRef(
                        source_graph.id, str(value_id)
                    )) for value_id in program_ids)
                ):
                    source_graph = _record_planned_process_graph(
                        metagraph,
                        process_graph,
                        label=(
                            "planned semantic "
                            + str(process_graph.G.graph.get("function_name") or entrypoint)
                        ),
                    )
                region_evolution[int(region_index)] = record_fused_program_evolution(
                    program,
                    source_graph=source_graph,
                    label=f"numeric region {int(region_index)}",
                )

        lowering = lower_precompile_and_control_to_ssa(
            aot.compiled_shell_program,
            aot.shell_control_program,
            region_programs=dict(aot.region_programs),
            hierarchy_plan=getattr(aot, "hierarchy_plan", None),
            numerical_name=entrypoint,
            control_name=f"{entrypoint}_control",
        )

        # The package graph is intentionally coarse until a packaging format
        # exposes line/record correspondence. It truthfully consumes complete
        # SSA functions rather than inventing component-level identities.
        package_graph = metagraph.open_graph("ir-package", f"{entrypoint} package")
        for function_name, function in lowering.module.functions.items():
            ssa_graph = metagraph.graph_for_artifact(function)
            if ssa_graph is None:
                continue
            sources = tuple(
                component.ref
                for component in metagraph.snapshot().components
                if component.ref.graph_id == ssa_graph.id
            )
            target = metagraph.component(
                package_graph,
                function_name,
                label=function_name,
                kind="function",
                attributes={"granularity": "whole-function"},
                consumes=sources,
            )
            if sources:
                metagraph.handoff(
                    target,
                    sources,
                    transformation="ssa-to-package",
                    detail={"granularity": "whole-function"},
                )
        metagraph.close_graph(package_graph)

        final_artifact = None
        if final_target and aot.region_programs:
            region_index, program = next(iter(sorted(aot.region_programs.items())))
            program = getattr(program, "program", program)
            region_function = lowering.module.functions.get(
                f"numerical_region_{int(region_index)}"
            )
            if final_target == "webgl" and region_function is not None:
                from .ssa_webgl_backend import emit_ssa_webgl_fragment_module

                final_artifact = emit_ssa_webgl_fragment_module(
                    region_function,
                    name=f"{entrypoint}_{final_target}",
                )
            else:
                from .machine_targets import emit

                final_artifact = emit(
                    program,
                    final_target,
                    name=f"{entrypoint}_{final_target}",
                )
                # Other registered targets currently consume FusedProgram,
                # not SSA. Keep that boundary honest until they expose an SSA
                # adapter comparable to WebGL's.
                source_graph = metagraph.graph_for_artifact(program)
                backend_graph = metagraph.open_graph(
                    f"backend:{final_target}",
                    f"{final_target} artifact",
                )
                sources = () if source_graph is None else tuple(
                    component.ref
                    for component in metagraph.snapshot().components
                    if component.ref.graph_id == source_graph.id
                )
                target = metagraph.component(
                    backend_graph,
                    "artifact",
                    label=str(getattr(final_artifact, "name", final_target)),
                    kind=str(final_target),
                    attributes={
                        "complete": bool(getattr(final_artifact, "complete", True)),
                        "granularity": "whole-artifact",
                    },
                    consumes=sources,
                )
                if sources:
                    metagraph.handoff(
                        target,
                        sources,
                        transformation=f"precompile-to-{final_target}",
                        detail={"granularity": "whole-artifact"},
                    )
                metagraph.bind_artifact(final_artifact, backend_graph)
                metagraph.close_graph(backend_graph)

    return AutogenesisCompilation(
        metagraph=metagraph,
        aot=aot,
        ssa=lowering,
        final_artifact=final_artifact,
    )


__all__ = ["AutogenesisCompilation", "compile_source_autogenesis"]
