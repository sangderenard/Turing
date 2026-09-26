"""Record one real compiler run as a lineage-preserving autogenesis graph."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .evolution_metagraph import (
    EvolutionComponentRef,
    EvolutionMetaGraph,
    extend_compiled_execution_lineage,
    record_compiled_execution_evolution,
    record_evolution,
    record_fused_program_evolution,
)


@dataclass(frozen=True, slots=True)
class AutogenesisCompilation:
    metagraph: EvolutionMetaGraph
    aot: Any
    ssa: Any
    final_artifact: Any = None
    final_artifacts: Mapping[int, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RepositorySSALowering:
    """Uniform autogenesis result for direct whole-source SSA lowering."""

    module: Any
    outputs: Mapping[str, Any]
    exports: tuple[str, ...]
    complete: bool = True


@dataclass(frozen=True, slots=True)
class SympySSALowering:
    """Direct canonical ProcessGraph-to-SSA result for a SymPy expression."""

    instructions: tuple[Any, ...]
    complete: bool = True


def compile_sympy_autogenesis(
    expression_text: str,
    *,
    metagraph: EvolutionMetaGraph | None = None,
    schedule: str = "asap",
    final_target: str | None = None,
    pi_solver: str = "literal",
    pi_epsilon: float | None = None,
) -> AutogenesisCompilation:
    """Ingest a SymPy string directly and record its repository SSA handoff.

    This path never parses Python AST and never manufactures an authored
    function wrapper. SymPy nodes enter the canonical ProcessGraph translator,
    then the expression graph's own scheduler orders repository SSA values.
    """

    import sympy

    from .evolution_metagraph import EvolutionComponentRef
    from .bounded_constants import materialize_pi
    from .ssa_builder import process_graph_to_ssa_instrs
    from ..transmogrifier.graph.graph_express2 import ProcessGraph

    if schedule not in {"asap", "alap"}:
        raise ValueError("SymPy ProcessGraph schedule must be 'asap' or 'alap'")
    if final_target is not None:
        raise ValueError(
            "direct SymPy autogenesis currently stops at repository SSA"
        )
    metagraph = metagraph or EvolutionMetaGraph()
    run_graph = metagraph.open_graph(
        "compiler-run", "sympy:expression live compile"
    )
    metagraph.component(
        run_graph,
        0,
        label="SymPy expression requested",
        kind="compiler-phase",
        attributes={"source_language": "sympy", "source_scope": ()},
    )
    with record_evolution(metagraph):
        expression = sympy.sympify(str(expression_text), evaluate=False)
        graph = ProcessGraph(
            materialize_memory=False,
            source_language="sympy",
        )
        graph.build_from_expression(expression)
        pi_materialization = materialize_pi(pi_solver, pi_epsilon)
        for _node_id, node_data in graph.G.nodes(data=True):
            if node_data.get("op") != "Pi":
                continue
            if pi_materialization.value is None:
                raise ValueError(
                    "SymPy expression contains pi but the selected pi "
                    "solver rejects materialization"
                )
            node_data.setdefault("attributes", {}).update(
                pi_materialization.contract()
            )
        instructions = tuple(
            process_graph_to_ssa_instrs(graph, schedule=schedule)
        )
        # SymPy ingestion intentionally writes canonical nodes directly. Make
        # those existing nodes visible to the same evolution recorder used by
        # AST ingestion after scheduling has installed any storage nodes. This
        # is observation, not a second graph build, and guarantees ProcessGraph
        # and direct SSA counts describe the same finalized graph.
        for node_id, data in graph.G.nodes(data=True):
            graph.observe_evolution_node(node_id, data)
        for node_id, data in graph.G.nodes(data=True):
            for parent_id, role in data.get("parents") or ():
                graph.observe_evolution_edge(parent_id, node_id, role)
        if graph._evolution_graph is not None:
            metagraph.close_graph(graph._evolution_graph)

        ssa_graph = metagraph.open_graph("ssa", "SymPy expression SSA")
        ssa_components: dict[int, EvolutionComponentRef] = {}
        for ordinal, instruction in enumerate(instructions):
            # ``Instr`` names its result ``res``. Reading ``result`` always
            # missed, so ``value_id`` silently fell back to the loop ordinal:
            # every SSA component was keyed by position and handed off to
            # whichever ProcessGraph node happened to share that number, which
            # is only the right node by coincidence. The value id is the
            # compiler's cross-representation identity -- it is what lets a
            # colour authored over the ProcessGraph follow the same value into
            # SSA -- so keying on anything else silently reroutes the lineage.
            result = getattr(instruction, "res", None)
            value_id = int(getattr(result, "id", ordinal))
            process_source = EvolutionComponentRef(
                graph._evolution_graph.id, str(value_id)
            )
            consumes = (
                (process_source,)
                if metagraph.has_component(process_source)
                else ()
            )
            target = metagraph.component(
                ssa_graph,
                value_id,
                label=str(instruction.op),
                kind="instruction",
                attributes={
                    "source_language": "sympy",
                    "schedule": schedule,
                    **{
                        key: instruction.attributes[key]
                        for key in (
                            "schedule_level", "schedule_group", "schedule_method"
                        )
                        if key in instruction.attributes
                    },
                },
                consumes=consumes,
            )
            ssa_components[value_id] = target
            if consumes:
                metagraph.handoff(
                    target,
                    consumes,
                    transformation="sympy-process-graph-to-ssa",
                )
            for argument, role in zip(
                instruction.args,
                getattr(instruction, "arg_roles", ()),
            ):
                source = ssa_components.get(int(argument.id))
                if source is not None:
                    metagraph.relationship(
                        ssa_graph, source, target, role=str(role or "data")
                    )
        metagraph.close_graph(ssa_graph)
        package_graph = metagraph.open_graph(
            "ir-package", "SymPy expression package"
        )
        package = metagraph.component(
            package_graph,
            "sympy_expression",
            label="sympy_expression",
            kind="function",
            attributes={"granularity": "whole-expression"},
            consumes=tuple(ssa_components.values()),
        )
        if ssa_components:
            metagraph.handoff(
                package,
                tuple(ssa_components.values()),
                transformation="ssa-to-package",
                detail={"granularity": "whole-expression"},
            )
        metagraph.close_graph(package_graph)
        metagraph.close_graph(run_graph)

    return AutogenesisCompilation(
        metagraph=metagraph,
        aot=graph,
        ssa=SympySSALowering(instructions),
        final_artifact=None,
    )


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
            attributes={
                "source_span": span,
                "source_scope": tuple(data.get("source_scope") or ()),
                "source_class": data.get("source_class"),
            },
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
    retained_levels = dict(getattr(process_graph, "levels", {}) or {})
    if retained_levels and set(retained_levels) == set(process_graph.G):
        # This graph was already planned by its owning compiler shell. Mirror
        # the retained result like DualIR mirrors an already-produced program;
        # do not invoke a scheduler while recording its evolution surface.
        metagraph.finalize_schedule(
            graph,
            retained_levels,
            method=str(
                process_graph.G.graph.get("schedule_preference") or "retained"
            ),
            order="dependency",
        )
    metagraph.close_graph(graph)
    return graph


def compile_source_autogenesis(
    source: str,
    entrypoint: str | None,
    feeds: Mapping[str, Any],
    *,
    final_target: str | None = "webgl",
    metagraph: EvolutionMetaGraph | None = None,
    boundary_namespace: Any = None,
    source_language: str = "python",
    extraction_contract: Any = None,
) -> AutogenesisCompilation:
    """Compile once while recording exact cross-IR component lineage."""

    from .compiler_entrypoints import warn_legacy_source_compiler

    warn_legacy_source_compiler("compile_source_autogenesis")

    metagraph = metagraph or EvolutionMetaGraph()
    compile_name = entrypoint or "whole_source"
    run_graph = metagraph.open_graph(
        "compiler-run", f"{source_language}:{compile_name} live compile"
    )
    progress_ordinal = 0
    previous_progress = metagraph.component(
        run_graph,
        progress_ordinal,
        label="compile requested",
        kind="compiler-phase",
        attributes={
            "source_scope": (() if entrypoint is None else (entrypoint,)),
            "source_language": source_language,
        },
    )

    def record_progress(message: str) -> None:
        nonlocal progress_ordinal, previous_progress
        progress_ordinal += 1
        current = metagraph.component(
            run_graph,
            progress_ordinal,
            label=str(message),
            kind="compiler-phase",
            attributes={
                "source_scope": (() if entrypoint is None else (entrypoint,)),
                "source_language": source_language,
            },
        )
        metagraph.relationship(
            run_graph,
            previous_progress,
            current,
            role="phase-order",
        )
        previous_progress = current

    record_progress("loading AOT compiler")
    from ..common.tensors.accelerator_backends.aot_compile import compile_ast_aot
    record_progress("loading deployment planner")
    from .glsl_deployment_strategy import _walk_planned_shells
    record_progress("loading SSA lowering")
    from .precompile_to_ssa import lower_precompile_and_control_to_ssa

    if entrypoint is None:
        record_progress("whole-source SSA compile starting")
        from .fortran_c_shell import lower_ast_source_to_ssa

        with record_evolution(metagraph):
            try:
                module, outputs, exports = lower_ast_source_to_ssa(
                    source,
                    None,
                    name=compile_name,
                    runtime_closure_only=False,
                    progress=record_progress,
                    boundary_namespace=boundary_namespace,
                    source_language=source_language,
                    extraction_contract=extraction_contract,
                )
            finally:
                metagraph.close_graph(run_graph)
        if final_target is not None:
            raise ValueError(
                "whole-source autogenesis currently stops at repository SSA; "
                "select an entrypoint before requesting a final backend target"
            )
        package_graph = metagraph.open_graph(
            "ir-package", f"{compile_name} package"
        )
        for function_name, function in module.functions.items():
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
        return AutogenesisCompilation(
            metagraph=metagraph,
            aot=None,
            ssa=RepositorySSALowering(module, outputs, tuple(exports)),
            final_artifact=None,
        )

    record_progress("AOT compile starting")

    region_evolution: dict[int, Any] = {}
    with record_evolution(metagraph):
        try:
            aot = compile_ast_aot(
                source,
                entrypoint,
                dict(feeds),
                precompile_only=True,
                boundary_namespace=boundary_namespace,
                source_language=source_language,
                extraction_contract=extraction_contract,
                progress=record_progress,
            )
        finally:
            metagraph.close_graph(run_graph)

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

        # ``aot.region_programs`` is the selected, remapped region table that
        # the accepted ControlProgram and SSA lowering actually consume.
        # Planned shells use local region indices which may collide, and their
        # captured objects may precede hierarchical remapping. Record the
        # accepted table explicitly and use only it for execution membership.
        recorded_snapshot = metagraph.snapshot()
        process_graph_refs = tuple(
            graph_ref for graph_ref in recorded_snapshot.graphs
            if graph_ref.stage == "process-graph"
        )
        process_ids = {
            graph_ref.id: {
                component.ref.local_id
                for component in recorded_snapshot.components
                if component.ref.graph_id == graph_ref.id
            }
            for graph_ref in process_graph_refs
        }

        def accepted_source_graph(program: Any):
            value_ids = {
                *map(str, getattr(program, "feeds", ()) or ()),
                *(
                    str(step.result_id)
                    for step in getattr(program, "steps", ()) or ()
                ),
            }
            owners = [
                graph_ref for graph_ref in process_graph_refs
                if value_ids and value_ids <= process_ids[graph_ref.id]
            ]
            if not owners:
                return None
            smallest = min(len(process_ids[item.id]) for item in owners)
            closest = [
                item for item in owners
                if len(process_ids[item.id]) == smallest
            ]
            # Exact value IDs are the compiler's cross-IR identity; the
            # smallest containing graph is their prepared compartment. Refuse
            # tied owners rather than joining unrelated local ID domains.
            return closest[0] if len(closest) == 1 else None

        accepted_region_evolution = {}
        for region_index, captured in sorted(aot.region_programs.items()):
            program = getattr(captured, "program", captured)
            accepted_region_evolution[int(region_index)] = (
                record_fused_program_evolution(
                    program,
                    source_graph=accepted_source_graph(program),
                    label=f"accepted numeric region {int(region_index)}",
                )
            )

        # The accepted compiled product is the planner-owned ControlProgram
        # composed with its deployment regions and exact numeric dispatches.
        # Publish that structure only after every region graph exists, so the
        # observer can connect control, lanes, loops, and instruction bodies
        # without flattening ProcessGraph levels into a counterfeit program.
        execution_graph = record_compiled_execution_evolution(
            aot.shell_control_program,
            region_graphs=accepted_region_evolution,
            region_programs=dict(aot.region_programs),
            label=f"{entrypoint} accepted execution",
        )

        lowering = lower_precompile_and_control_to_ssa(
            aot.compiled_shell_program,
            aot.shell_control_program,
            region_programs=dict(aot.region_programs),
            hierarchy_plan=getattr(aot, "hierarchy_plan", None),
            numerical_name=entrypoint,
            control_name=f"{entrypoint}_control",
        )
        navigation = getattr(aot, "class_navigation", None)
        navigation_mapping = (
            navigation.to_mapping()
            if hasattr(navigation, "to_mapping")
            else dict(navigation or {})
        )
        if navigation_mapping.get("classes"):
            # A class-bearing source is represented authoritatively by the
            # non-projecting class-surface module: reachable methods, scheduled
            # operator regions, physical field tables, and explicit call-frame
            # status. The numerical projection remains available on ``aot``
            # and has already contributed its evolution graph, but merging its
            # duplicate raw GetAttr regions back into repository SSA would
            # undo object legalization and represent the source twice.
            from .fortran_c_shell import lower_class_surface_to_ssa

            object_module, _object_outputs, _object_exports = (
                lower_class_surface_to_ssa(aot, entrypoint)
            )
            lowering = replace(
                lowering,
                module=object_module,
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
        final_artifacts: dict[int, Any] = {}
        if final_target and aot.region_programs:
            for region_index, captured in sorted(aot.region_programs.items()):
                program = getattr(captured, "program", captured)
                region_function = lowering.module.functions.get(
                    f"numerical_region_{int(region_index)}"
                )
                artifact_name = (
                    f"{entrypoint}_{final_target}_region_{int(region_index)}"
                )
                if final_target == "webgl" and region_function is not None:
                    from .ssa_webgl_backend import emit_ssa_webgl_fragment_module

                    artifact = emit_ssa_webgl_fragment_module(
                        region_function,
                        name=artifact_name,
                    )
                else:
                    from .machine_targets import emit

                    artifact = emit(
                        program,
                        final_target,
                        name=artifact_name,
                    )
                final_artifacts[int(region_index)] = artifact
                if final_artifact is None:
                    # Compatibility for callers that historically requested
                    # the first region. ``final_artifacts`` is the complete
                    # accepted backend product.
                    final_artifact = artifact
                # Other registered targets currently consume FusedProgram,
                # not SSA. Keep that boundary honest until they expose an SSA
                # adapter comparable to WebGL's.
                if final_target == "webgl" and region_function is not None:
                    continue
                source_graph = metagraph.graph_for_artifact(program)
                backend_graph = metagraph.open_graph(
                    f"backend:{final_target}",
                    f"{final_target} region {int(region_index)} artifact",
                )
                sources = () if source_graph is None else tuple(
                    component.ref
                    for component in metagraph.snapshot().components
                    if component.ref.graph_id == source_graph.id
                )
                target = metagraph.component(
                    backend_graph,
                    f"region_{int(region_index)}",
                    label=str(getattr(artifact, "name", final_target)),
                    kind=str(final_target),
                    attributes={
                        "complete": bool(getattr(artifact, "complete", True)),
                        "granularity": "dispatch-region",
                        "region_index": int(region_index),
                    },
                    consumes=sources,
                )
                if sources:
                    metagraph.handoff(
                        target,
                        sources,
                        transformation=f"precompile-to-{final_target}",
                        detail={
                            "granularity": "dispatch-region",
                            "region_index": int(region_index),
                        },
                    )
                metagraph.bind_artifact(artifact, backend_graph)
                metagraph.close_graph(backend_graph)

        # SSA, package, and backend graphs are produced after the control plan.
        # Propagate frame membership only through explicit representation
        # handoffs so the final display covers the complete accepted product
        # without deriving execution from presentation topology.
        extend_compiled_execution_lineage(execution_graph)

    return AutogenesisCompilation(
        metagraph=metagraph,
        aot=aot,
        ssa=lowering,
        final_artifact=final_artifact,
        final_artifacts=final_artifacts,
    )


__all__ = [
    "AutogenesisCompilation",
    "RepositorySSALowering",
    "SympySSALowering",
    "compile_source_autogenesis",
    "compile_sympy_autogenesis",
]
