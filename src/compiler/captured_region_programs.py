"""Captured region programs: storage identity and feed aliasing of a captured tape, remapping of captured programs onto planned ids, planned-step wiring and the projection of captured programs onto regions."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Iterable, Mapping

from ..common.tensors.abstraction import AbstractTensor
from ..common.tensors.accelerator_backends.c_primitive_program import CapturedFusedProgram
from ..common.tensors.fused_ir import (
    FusedProgram,
    Meta,
    OpStep,
)
from .planned_shell_tree import _walk_planned_shells


def _captured_storage_meta(values: dict[int, Any]) -> dict[int, Meta]:
    """Extract arena storage contracts without consuming numerical values."""

    result = {}
    for value_id, value in values.items():
        if isinstance(value, AbstractTensor):
            result[int(value_id)] = Meta(
                shape=tuple(int(size) for size in value.shape),
                dtype=str(value.dtype),
                device="glsl",
            )
        elif isinstance(value, bool):
            result[int(value_id)] = Meta((), "bool", "glsl")
        elif isinstance(value, int):
            result[int(value_id)] = Meta((), "int32", "glsl")
        elif isinstance(value, float):
            result[int(value_id)] = Meta((), "float32", "glsl")
    return result


def _compiler_input_name(label: str) -> str:
    """Mirror the established GraphDeepCompiler input naming convention."""

    import re

    name = str(label).strip()
    lowered = name.lower()
    match = re.match(r"([a-zA-Z]+)[_\d]*$", lowered)
    root = match.group(1) if match else lowered
    if root in {"i", "j", "k", "l", "m", "n"}:
        prefix = "int"
    elif (
        lowered.startswith("num")
        or lowered.endswith("idx")
        or lowered.isdigit()
    ):
        prefix = "int"
    elif lowered.startswith("is_") or lowered.startswith("has_"):
        prefix = "bool"
    else:
        prefix = "float"
    return f"{prefix}{name}"


def _capture_storage_identity(value: Any):
    """Return the exact resident range represented by a capture wrapper."""

    storage = (
        value
        if getattr(value, "_storage", None) is not None
        else getattr(value, "data", None)
    )
    if storage is None:
        return None
    physical = getattr(storage, "_storage", None)
    if physical is not None:
        return (
            id(physical),
            getattr(storage, "_offset", None),
            getattr(storage, "_count", None),
            str(getattr(storage, "_dtype", None)),
        )
    return ("object", id(storage))


def _remap_captured_program(
    captured: CapturedFusedProgram,
    *,
    feed_ids: dict[int, int],
    output_ids: tuple[int, ...],
) -> CapturedFusedProgram:
    """Replace transient tensor identities with ProcessGraph boundary IDs."""

    program = captured.program
    captured_outputs = tuple(program.outputs.values())
    if len(captured_outputs) != len(output_ids):
        raise ValueError(
            "captured terminal count does not match planned outputs"
        )
    id_map = dict(feed_ids)
    id_map.update(zip(captured_outputs, output_ids))
    remap = lambda value_id: id_map.get(value_id, value_id)
    captured_feed_meta = _captured_storage_meta({
        int(value_id): value
        for value_id, value in captured.feeds.items()
    })
    def remap_program(source, *, public_outputs=False):
        target = FusedProgram(
            version=source.version,
            feeds={remap(value_id) for value_id in source.feeds},
            steps=[
                OpStep(
                    step_id=step.step_id,
                    op_name=step.op_name,
                    input_ids=[
                        remap(value_id) for value_id in step.input_ids
                    ],
                    attrs=dict(step.attrs),
                    result_id=remap(step.result_id),
                    mode_sensitive=step.mode_sensitive,
                    level=step.level,
                )
                for step in source.steps
            ],
            outputs=(
                {
                    f"value_{output_id}": output_id
                    for output_id in output_ids
                }
                if public_outputs
                else {
                    name: remap(value_id)
                    for name, value_id in source.outputs.items()
                }
            ),
            state_in=(
                None
                if source.state_in is None
                else {remap(value_id) for value_id in source.state_in}
            ),
            meta={
                **{
                    remap(value_id): meta
                    for value_id, meta in captured_feed_meta.items()
                    if value_id in source.feeds
                },
                **{
                    remap(value_id): meta
                    for value_id, meta in (source.meta or {}).items()
                },
            },
            extras=source.extras,
        )
        if hasattr(source, "glsl_linear_output_shape"):
            target.glsl_linear_output_shape = tuple(
                source.glsl_linear_output_shape
            )
        return target

    remapped = remap_program(program, public_outputs=True)
    return CapturedFusedProgram(
        remapped,
        # Captured tensors are evidence used to reveal the operation sequence;
        # they are not part of the compiled program.  Runtime data enters only
        # through the remapped feed IDs.  Retaining numeric payloads here would
        # compile an observed instance instead of the program.
        {},
        tuple(
            remap_program(stage)
            for stage in captured.stages
        ),
    )


def _remap_captured_all_ids(
    captured: CapturedFusedProgram,
    id_map: Mapping[int, int],
    *,
    source_closure_id: int | None = None,
) -> CapturedFusedProgram:
    """Namespace every captured IR identity for hierarchical composition."""

    def remap_program(source):
        remap = lambda value_id: int(
            id_map.get(int(value_id), int(value_id))
        )
        extras = copy.deepcopy(dict(source.extras or {}))
        if source_closure_id is not None:
            extras["source_closure_id"] = int(source_closure_id)
        for table_name in (
            "ssa_identity_tokens",
            "capture_feed_origins",
            "descriptor_derivations",
        ):
            table = extras.get(table_name)
            if not isinstance(table, Mapping):
                continue
            extras[table_name] = {
                remap(value_id): value
                for value_id, value in table.items()
            }
        source_value_ids = extras.get("source_value_ids")
        if isinstance(source_value_ids, Mapping):
            extras["source_value_ids"] = {
                remap(value_id): int(source_id)
                for value_id, source_id in source_value_ids.items()
            }
        derivations = extras.get("descriptor_derivations")
        if isinstance(derivations, Mapping):
            extras["descriptor_derivations"] = {
                int(value_id): {
                    **dict(receipt),
                    **(
                        {"source_id": remap(receipt["source_id"])}
                        if isinstance(receipt, Mapping)
                        and receipt.get("source_id") is not None else {}
                    ),
                    **(
                        {"input_ids": tuple(
                            remap(item) for item in receipt["input_ids"]
                        )}
                        if isinstance(receipt, Mapping)
                        and receipt.get("input_ids") is not None else {}
                    ),
                }
                for value_id, receipt in derivations.items()
            }
        target = FusedProgram(
            version=source.version,
            feeds={remap(value_id) for value_id in source.feeds},
            steps=[
                OpStep(
                    step.step_id,
                    step.op_name,
                    [remap(value_id) for value_id in step.input_ids],
                    dict(step.attrs),
                    remap(step.result_id),
                    step.mode_sensitive,
                    step.level,
                )
                for step in source.steps
            ],
            outputs={
                name: remap(value_id)
                for name, value_id in source.outputs.items()
            },
            state_in=(
                None
                if source.state_in is None
                else {remap(value_id) for value_id in source.state_in}
            ),
            meta={
                remap(value_id): replace(
                    meta,
                    source_id=(
                        None if meta.source_id is None
                        else remap(meta.source_id)
                    ),
                    shape_source_ids=(
                        None if meta.shape_source_ids is None else tuple(
                            None if item is None else remap(item)
                            for item in meta.shape_source_ids
                        )
                    ),
                )
                for value_id, meta in (source.meta or {}).items()
            },
            extras=extras,
        )
        if hasattr(source, "glsl_linear_output_shape"):
            target.glsl_linear_output_shape = tuple(
                source.glsl_linear_output_shape
            )
        return target

    return CapturedFusedProgram(
        remap_program(captured.program),
        {},
        tuple(remap_program(stage) for stage in captured.stages),
    )


def _wire_planned_step_inputs(
    captured: CapturedFusedProgram,
    step_inputs: Mapping[int, tuple[tuple[int, int], ...]],
) -> CapturedFusedProgram:
    """Replace primitive operand IDs with their planned PG input positions.

    ``step_inputs`` is emitted synchronously by GraphDeepCompiler for one
    operation invocation.  Keys are primitive result occurrence IDs and
    values are the ProcessGraph parent IDs already present in the compartment.
    No payload, object, storage, or tape traversal participates here.
    """

    def rewrite(source: FusedProgram) -> FusedProgram:
        steps = []
        for step in source.steps:
            planned = step_inputs.get(int(step.result_id))
            inputs = (
                tuple(int(graph_id) for _primitive_id, graph_id in planned)
                if planned is not None
                and len(planned) == len(step.input_ids)
                else tuple(map(int, step.input_ids))
            )
            steps.append(OpStep(
                step.step_id,
                step.op_name,
                inputs,
                dict(step.attrs),
                int(step.result_id),
                step.mode_sensitive,
                step.level,
            ))
        produced = {int(step.result_id) for step in steps}
        feeds = {
            int(value_id)
            for step in steps
            for value_id in step.input_ids
            if int(value_id) not in produced
        }
        meta = dict(source.meta or {})
        for positional_inputs in step_inputs.values():
            for primitive_id, graph_id in positional_inputs:
                if int(primitive_id) in meta:
                    meta.setdefault(int(graph_id), meta[int(primitive_id)])
        return FusedProgram(
            version=source.version,
            feeds=feeds,
            steps=steps,
            outputs=dict(source.outputs),
            state_in=source.state_in,
            meta=meta,
            extras=source.extras,
        )

    return CapturedFusedProgram(
        rewrite(captured.program),
        {},
        tuple(rewrite(stage) for stage in captured.stages),
    )


def _collapse_planned_collection_materializations(
    captured: CapturedFusedProgram,
    aliases: Mapping[int, int],
) -> CapturedFusedProgram:
    """Replace observed aggregate leaves with one planner collection input.

    The materialization remains a real numerical producer with its own result
    ID.  Only its discovery-time variadic inputs are replaced; the loop plan's
    resident result is the single runtime input.
    """

    aliases = {
        int(result_id): int(collection_id)
        for result_id, collection_id in aliases.items()
    }
    if not aliases:
        return captured

    def rewrite(source: FusedProgram) -> FusedProgram:
        metadata = dict(source.meta or {})
        steps = [
            OpStep(
                step.step_id,
                step.op_name,
                (
                    [aliases[int(step.result_id)]]
                    if int(step.result_id) in aliases
                    else list(step.input_ids)
                ),
                dict(step.attrs),
                int(step.result_id),
                step.mode_sensitive,
                step.level,
            )
            for step in source.steps
        ]
        for step in source.steps:
            collection_id = aliases.get(int(step.result_id))
            result_meta = metadata.get(int(step.result_id))
            if collection_id is not None and result_meta is not None:
                metadata.setdefault(int(collection_id), result_meta)
        produced = {int(step.result_id) for step in steps}
        feeds = {
            int(value_id)
            for step in steps
            for value_id in step.input_ids
            if int(value_id) not in produced
        }
        target = FusedProgram(
            version=source.version,
            feeds=feeds,
            steps=steps,
            outputs={
                name: int(value_id)
                for name, value_id in source.outputs.items()
            },
            state_in=(
                None
                if source.state_in is None
                else {int(value_id) for value_id in source.state_in}
            ),
            meta=metadata,
            extras=source.extras,
        )
        if hasattr(source, "glsl_linear_output_shape"):
            target.glsl_linear_output_shape = tuple(
                source.glsl_linear_output_shape
            )
        return target

    full = rewrite(captured.program)
    stages = tuple(
        rewritten
        for stage in captured.stages
        if (rewritten := rewrite(stage)).steps
    )
    return CapturedFusedProgram(full, {}, stages)


def _project_captured_program(
    captured: CapturedFusedProgram,
    *,
    output_ids: tuple[int, ...],
    boundary_ids: tuple[int, ...],
    allowed_result_ids: tuple[int, ...] | None = None,
) -> CapturedFusedProgram:
    """Project a planner region from an already-lowered shell program.

    This function operates only on backend-neutral IR.  It deliberately has
    no access to a GradTape or captured tensor payload, so requesting another
    region cannot rediscover, replay, or recursively lower the Python
    execution.  ``boundary_ids`` are planner cut points: dependencies stop
    there even if another region in the complete shell produces that value.
    """

    outputs = tuple(int(value) for value in output_ids)
    boundaries = {int(value) for value in boundary_ids}
    allowed_results = (
        None
        if allowed_result_ids is None
        else {int(value) for value in allowed_result_ids}
    )
    synthetic_results = {
        int(value_id)
        for program in (captured.program, *captured.execution_programs)
        for value_id in (
            (program.extras or {}).get("synthetic_result_ids", ())
        )
    }
    all_steps = [
        step
        for program in captured.execution_programs
        for step in program.steps
    ]
    all_meta = {
        int(value_id): meta
        for program in (captured.program, *captured.execution_programs)
        for value_id, meta in (program.meta or {}).items()
    }
    producers = {int(step.result_id): step for step in all_steps}
    available_feeds = {
        int(value_id)
        for program in (captured.program, *captured.execution_programs)
        for value_id in program.feeds
    }
    missing_outputs = tuple(
        value_id for value_id in outputs
        if value_id not in producers and value_id not in available_feeds
    )
    if missing_outputs:
        raise ValueError(
            "planner region outputs are absent from the once-lowered shell "
            f"program: missing={missing_outputs!r}; "
            f"available={tuple(sorted(producers))!r}"
        )
    # A value may be consumed again inside the stage that creates it and also
    # cross a later stage/region boundary.  `produced - consumed` alone is
    # therefore not liveness: it drops such values from the stage ABI and the
    # later consumer is misclassified as an external shell input.  The
    # once-lowered program already states every cross-stage feed, while the
    # planner states every region output; retain both sets as explicit
    # materialization boundaries.
    cross_stage_feeds = {
        int(value_id)
        for source in captured.execution_programs
        for value_id in source.feeds
    }
    required_live_outs = set(outputs) | cross_stage_feeds
    needed = set(outputs)
    selected: set[int] = set()
    pending = [(value_id, True) for value_id in outputs]
    while pending:
        value_id, is_region_output = pending.pop()
        value_id = int(value_id)
        if (
            (value_id in boundaries and not is_region_output)
            or (
                allowed_results is not None
                and value_id not in allowed_results
                and value_id not in synthetic_results
                and not is_region_output
            )
            or value_id in selected
        ):
            continue
        step = producers.get(value_id)
        if step is None:
            continue
        selected.add(value_id)
        for input_id in step.input_ids:
            input_id = int(input_id)
            needed.add(input_id)
            if (
                input_id not in boundaries
                and (
                    allowed_results is None
                    or input_id in allowed_results
                    or input_id in synthetic_results
                )
            ):
                pending.append((input_id, False))

    def project(source: FusedProgram, *, public: bool = False):
        steps = [
            OpStep(
                step.step_id,
                step.op_name,
                list(step.input_ids),
                dict(step.attrs),
                step.result_id,
                step.mode_sensitive,
                step.level,
            )
            for step in source.steps
            if int(step.result_id) in selected
        ]
        produced = {int(step.result_id) for step in steps}
        consumed = {
            int(input_id)
            for step in steps
            for input_id in step.input_ids
        }
        stage_terminals = (
            (produced - consumed)
            | (produced & required_live_outs)
        )
        feeds = {
            int(input_id)
            for step in steps
            for input_id in step.input_ids
            if int(input_id) not in produced
        }
        if public:
            feeds.update(
                value_id
                for value_id in outputs
                if value_id not in produced
            )
        target = FusedProgram(
            version=source.version,
            feeds=feeds,
            steps=steps,
            outputs=(
                {f"value_{value_id}": value_id for value_id in outputs}
                if public
                else {
                    f"value_{value_id}": value_id
                    for value_id in sorted(stage_terminals)
                }
            ),
            state_in=(
                None
                if source.state_in is None
                else {
                    int(value_id)
                    for value_id in source.state_in
                    if int(value_id) in needed
                }
            ),
            meta={
                value_id: (
                    (source.meta or {}).get(value_id)
                    or all_meta[value_id]
                )
                for value_id in (
                    needed | produced | feeds | set(outputs)
                )
                if (
                    value_id in (source.meta or {})
                    or value_id in all_meta
                )
            },
            extras=source.extras,
        )
        if hasattr(source, "glsl_linear_output_shape"):
            target.glsl_linear_output_shape = tuple(
                source.glsl_linear_output_shape
            )
        return target

    manifest = project(captured.program, public=True)
    stages = tuple(
        projected
        for source in captured.execution_programs
        if (projected := project(source)).steps
    )
    return CapturedFusedProgram(manifest, {}, stages)


def _resolve_binding_name(shell: Any, captured: Any, feed_id: int):
    """Which source parameter a feed came from, if the capture knows.

    Object identity is tried first and is exact when it survives. When the
    value was rewrapped on its way to the tape -- which is the usual case --
    the resident range underneath is unchanged, so the storage recorded at
    binding time still identifies it.
    """

    # The shell that recorded the binding is not always the shell that
    # lowers the program: a function's inputs are bound in its own shell,
    # while the owner is the one that builds the origins. Search the whole
    # planned tree rather than assuming they are the same object.
    value = (getattr(captured, "feeds", None) or {}).get(feed_id)
    storage = _capture_storage_identity(value) if value is not None else None
    candidates = [shell]
    try:
        candidates.extend(_walk_planned_shells(
            shell, include_function_registry=False
        ))
    except Exception:
        pass
    for candidate in candidates:
        names = getattr(candidate, "_capture_input_names", None) or {}
        direct = names.get(feed_id)
        if direct:
            return direct
        for aggregate_map in (
            getattr(candidate, "compiled_aggregate_feed_paths", ()) or ()
        ):
            for capture_id, graph_input_id, path in aggregate_map:
                if int(capture_id) != int(feed_id):
                    continue
                graph = getattr(candidate, "process_graph", None)
                if graph is None or int(graph_input_id) not in graph.G:
                    continue
                root = _compiler_input_name(
                    graph.G.nodes[int(graph_input_id)]["label"]
                )
                suffix = "".join(
                    f".{part}" if isinstance(part, str)
                    and part.isidentifier()
                    else f"[{part!r}]"
                    for part in path
                )
                return root + suffix
    if storage is None:
        return None
    for candidate in candidates:
        by_storage = getattr(candidate, "_capture_input_storage", None) or {}
        found = by_storage.get(storage)
        if found:
            return found
    return None


def _capture_feed_aliases(
    captured: CapturedFusedProgram,
    feed_ids: dict[int, int],
) -> dict[int, int]:
    """Correlate capture wrappers that share one resident boundary value."""

    remapped = dict(feed_ids)

    anchors = {
        feed_id: (
            captured.feeds[feed_id],
            _capture_storage_identity(captured.feeds[feed_id]),
        )
        for feed_id in feed_ids
        if feed_id in captured.feeds
    }
    for candidate_id, candidate in captured.feeds.items():
        if candidate_id in remapped:
            continue
        candidate_storage = _capture_storage_identity(candidate)
        for anchor_id, (anchor, anchor_storage) in anchors.items():
            if (
                candidate is anchor
                or (
                    candidate_storage is not None
                    and candidate_storage == anchor_storage
                )
            ):
                remapped[candidate_id] = feed_ids[anchor_id]
                break
        if candidate_id in remapped:
            continue
        origin = (
            (captured.program.extras or {})
            .get("capture_feed_origins", {})
            .get(candidate_id)
        )
        if origin is not None and origin.get("op") is not None:
            # A recorded result that becomes a later stage feed is an
            # on-device program value.  Similar shape/dtype does not make it
            # an alias of a source boundary—doing so can rewrite a loop result
            # back to its initial input and erase the operation.
            continue
        metadata = captured.program.meta or {}
        candidate_meta = metadata.get(candidate_id)
        compatible = [
            anchor_id
            for anchor_id in anchors
            if metadata.get(anchor_id) == candidate_meta
        ]
        # A non-recorded capture root with one uniquely compatible boundary
        # descriptor is another wrapper for that boundary value.  Shape/dtype
        # matching is deliberately not used when two graph inputs are
        # compatible; that ambiguity must remain an error.
        if len(compatible) == 1:
            remapped[candidate_id] = feed_ids[compatible[0]]
    return remapped


def _unique_runtime_feed_aliases(
    program: FusedProgram,
    missing_feed_ids: Iterable[int],
    runtime_feeds: dict[int, Any],
) -> dict[int, AbstractTensor]:
    """Resolve an orphan feed only when one runtime tensor can match it."""

    leaves: list[AbstractTensor] = []

    def collect(value):
        if isinstance(value, AbstractTensor):
            leaves.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    for value in runtime_feeds.values():
        collect(value)

    resolved = {}
    metadata = program.meta or {}
    for feed_id in missing_feed_ids:
        meta = metadata.get(feed_id)
        if meta is None:
            continue
        compatible = [
            value
            for value in leaves
            if tuple(value.shape) == tuple(meta.shape or ())
            and str(value.dtype).lower().rsplit(".", 1)[-1]
            == str(meta.dtype).lower().rsplit(".", 1)[-1]
        ]
        if len(compatible) == 1:
            resolved[feed_id] = compatible[0]
    return resolved
