"""Compose per-closure control programs through a typed hierarchy plan."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Mapping

from .control_source import (
    CallBlock,
    ControlBlock,
    ControlExpression,
    ControlProgram,
    ControlUniform,
    LoopControlBlock,
    LoopBlock,
    ParallelDeployment,
    RecursionRegion,
    SequenceBlock,
    StateMachineTick,
    StatementBlock,
    StreamPublishBlock,
    ValidationBlock,
    WhileBlock,
)
from .hierarchical_plan import (
    HierarchyValueTable,
    PlanCall,
    PlanClosure,
)


def _region_marker(block: StatementBlock) -> int | None:
    if len(block.lines) != 1:
        return None
    line = block.lines[0]
    prefix = "__scheduled_region_"
    if not line.startswith(prefix) or not line.endswith("__"):
        return None
    return int(line[len(prefix):-2])


def _present_loop_ids(block: ControlBlock) -> "frozenset[int]":
    """Node ids of every ``LoopBlock`` actually present in ``block``.

    A loop scheduled by ``glsl_deployment_strategy`` can be fused away
    (its iteration space folded into a surviving outer loop) without being
    marked in the evaporated-loop bookkeeping the hierarchy plan consulted
    when it recorded a call's ``enclosing_loop_ids``. That id then names a
    loop this closure's own control program never materializes. Parsing
    the ``iteration_{node_id}`` induction convention out of the tree that
    actually exists is the only way to tell which of a call's recorded
    enclosing loops survived scheduling.
    """

    found: set[int] = set()

    def walk(b: ControlBlock) -> None:
        if isinstance(b, LoopBlock):
            name = str(b.induction)
            if name.startswith("iteration_"):
                try:
                    found.add(int(name[len("iteration_"):]))
                except ValueError:
                    pass
            walk(b.body)
        elif isinstance(b, WhileBlock):
            if b.source_loop_node_id is not None:
                found.add(int(b.source_loop_node_id))
            walk(b.condition)
            walk(b.body)
        elif isinstance(b, SequenceBlock):
            for child in b.blocks:
                walk(child)
        elif isinstance(b, StateMachineTick):
            for _value, body in b.cases:
                walk(body)
            if b.default is not None:
                walk(b.default)
        elif isinstance(b, ParallelDeployment):
            for lane in b.lanes:
                walk(lane)
        elif isinstance(b, CallBlock):
            walk(b.callee)

    walk(block)
    return frozenset(found)


@dataclass(frozen=True)
class HierarchicalControl:
    program: ControlProgram
    # (closure id, local region id, composed region id)
    region_correlations: tuple[tuple[int, int, int], ...]


def compose_hierarchical_control(
    hierarchy: PlanClosure,
    controls: Mapping[int, ControlProgram],
    values: HierarchyValueTable,
) -> HierarchicalControl:
    """Inline typed call control while preserving planner-owned loop scopes."""

    region_correlations: list[tuple[int, int, int]] = []
    next_region = 0
    # ``assign_hierarchy_ids`` only harvests (closure, local) keys from
    # PlanLine/PlanCall value ids -- the tensor dataflow surface known at
    # plan time. Control-flow-only endpoints (a loop induction/predicate,
    # a recursion-region member, a value alias) are discovered later, while
    # composing the control program itself, so they can reference a local id
    # the value table never assigned a global identity to. That is a real
    # gap in the table's harvest, not a stray or corrupted value -- give it
    # a stable per-endpoint identity here rather than raising past this
    # otherwise-complete composition.
    next_synthetic_value = 1 + max(
        (global_id for _, _, global_id in values.correlations),
        default=-1,
    )
    synthetic_value_ids: dict[tuple[int, int], int] = {}

    def value_in(closure_id: int, local_id: int) -> int:
        closure_id = int(closure_id)
        local_id = int(local_id)
        try:
            return values.global_id(closure_id, local_id)
        except KeyError:
            nonlocal next_synthetic_value
            endpoint = (closure_id, local_id)
            synthetic = synthetic_value_ids.get(endpoint)
            if synthetic is None:
                synthetic = next_synthetic_value
                next_synthetic_value += 1
                synthetic_value_ids[endpoint] = synthetic
            return synthetic

    all_uniforms: list[ControlUniform] = []
    all_aliases: list[tuple[int, int]] = []
    all_iterables: list[tuple[int, int, str]] = []
    all_static_iterables: list[
        tuple[int, int, str, tuple[object, ...]]
    ] = []
    all_collections: list[tuple[int, int, str, int]] = []
    all_closure_iterables: list[
        tuple[int, int, str, tuple[int, ...]]
    ] = []
    all_projected_iterables: list[tuple[int, int, str, object]] = []
    all_recursion_regions: list[RecursionRegion] = []
    all_deployment_regions = []
    next_recursion_region = 0
    next_deployment_region = 0

    def compose(closure: PlanClosure) -> ControlProgram:
        nonlocal next_region, next_recursion_region
        nonlocal next_deployment_region
        closure_id = int(closure.closure_id)
        local = controls[closure_id]
        region_map = {}
        for local_region in local.region_indices:
            region_map[int(local_region)] = next_region
            region_correlations.append(
                (closure_id, int(local_region), next_region)
            )
            next_region += 1

        def value(local_id: int) -> int:
            return value_in(closure_id, local_id)

        recursion_region_map = {}
        for region in local.recursion_regions:
            global_region_id = next_recursion_region
            next_recursion_region += 1
            recursion_region_map[int(region.region_id)] = global_region_id
            edge = lambda item: (value(item[0]), value(item[1]), item[2])
            all_recursion_regions.append(RecursionRegion(
                region_id=global_region_id,
                kind=region.kind,
                lower_as=region.lower_as,
                members=tuple(value(member) for member in region.members),
                control_ir=region.control_ir,
                control_members=tuple(
                    value(member) for member in region.control_members
                ),
                incoming=tuple(edge(item) for item in region.incoming),
                outgoing=tuple(edge(item) for item in region.outgoing),
                feedback=tuple(edge(item) for item in region.feedback),
            ))

        for deployment in local.deployment_regions:
            global_deployment_id = next_deployment_region
            next_deployment_region += 1
            all_deployment_regions.append(replace(
                deployment,
                region_id=global_deployment_id,
                lanes=tuple(
                    replace(
                        lane,
                        region_indices=tuple(
                            region_map[int(region_index)]
                            for region_index in lane.region_indices
                            if int(region_index) in region_map
                        ),
                        value_ids=tuple(
                            value(value_id) for value_id in lane.value_ids
                        ),
                    )
                    for lane in deployment.lanes
                ),
                recursion_region_id=(
                    None
                    if deployment.recursion_region_id is None
                    else recursion_region_map[
                        int(deployment.recursion_region_id)
                    ]
                ),
            ))

        induction_names: dict[str, str] = {}

        def global_induction(local_name: str) -> str:
            local_name = str(local_name)
            return induction_names.setdefault(
                local_name,
                f"{local_name}_closure_{closure_id}",
            )

        uniform_names: dict[str, str] = {}
        for uniform in local.uniforms:
            global_value_id = value(uniform.value_id)
            # Control uniforms are closure-local before composition.  Their
            # source names therefore collide routinely (every function can
            # have a local node 1).  Once values are globally namespaced, the
            # textual shader symbol must follow that global identity as well;
            # preserving the local spelling emits duplicate GLSL declarations
            # for unrelated values.
            global_name = f"u_control_{global_value_id}"
            uniform_names[uniform.name] = global_name
            all_uniforms.append(ControlUniform(
                global_name,
                global_value_id,
                uniform.dtype,
            ))

        def rename_control_text(source: str) -> str:
            result = str(source)
            for local_name, global_name in uniform_names.items():
                result = re.sub(
                    rf"\b{re.escape(local_name)}\b",
                    global_name,
                    result,
                )
            for local_name, global_name in induction_names.items():
                result = re.sub(
                    rf"\b{re.escape(local_name)}\b",
                    global_name,
                    result,
                )
            # Iterable extents are value identities just as surely as control
            # uniforms are.  A closure-local marker must not survive hierarchy
            # composition while its iterable binding is moved into the global
            # value namespace: that would leave the loop condition referring
            # to a different (and usually nonexistent) identity than the
            # binding which supplies its extent.
            local_iterable_ids = {
                int(iterable)
                for iterable, _target, _induction
                in local.iterable_bindings
            }
            local_iterable_ids.update(
                int(iterable)
                for iterable, _target, _induction, _source
                in local.static_iterable_bindings
            )
            local_iterable_ids.update(
                int(iterable)
                for iterable, _target, _induction, _sources
                in local.closure_iterable_bindings
            )
            local_iterable_ids.update(
                int(iterable)
                for iterable, _target, _induction, _projection
                in local.projected_iterable_bindings
            )
            for local_iterable_id in local_iterable_ids:
                local_marker = (
                    f"__iterable_extent_{local_iterable_id}__"
                )
                global_marker = (
                    f"__iterable_extent_{value(local_iterable_id)}__"
                )
                result = result.replace(local_marker, global_marker)
            return result
        # Hierarchy identity reduction can legitimately prove that both sides
        # of a planner alias are the same canonical resident value.  Such an
        # alias has already done its job; retaining ``x -> x`` would fabricate
        # a cycle in the emitted shell alias table.
        all_aliases.extend(
            (global_updated, global_initial)
            for updated, initial in local.value_aliases
            for global_updated, global_initial in ((
                value(updated),
                value(initial),
            ),)
            if global_updated != global_initial
        )
        all_iterables.extend(
            (value(iterable), value(target), global_induction(induction))
            for iterable, target, induction in local.iterable_bindings
        )
        all_static_iterables.extend(
            (
                value(iterable),
                value(target),
                global_induction(induction),
                source,
            )
            for iterable, target, induction, source
            in local.static_iterable_bindings
        )
        all_collections.extend(
            (
                value(source),
                value(collection),
                global_induction(induction),
                start,
            )
            for source, collection, induction, start
            in local.collection_bindings
        )
        all_closure_iterables.extend(
            (
                value(iterable),
                value(target),
                global_induction(induction),
                tuple(value(source) for source in sources),
            )
            for iterable, target, induction, sources
            in local.closure_iterable_bindings
        )
        all_projected_iterables.extend(
            (
                value(iterable),
                value(target),
                global_induction(induction),
                projection,
            )
            for iterable, target, induction, projection
            in local.projected_iterable_bindings
        )

        calls_before: dict[int, list[CallBlock]] = {}
        calls_after: list[CallBlock] = []
        pending: list[tuple[CallBlock, tuple[int, ...]]] = []
        for item in closure.items:
            if isinstance(item, PlanCall):
                child = compose(item.callee)
                pending.append((
                    CallBlock(
                        item.callsite_id,
                        child.root,
                        tuple(
                            (
                                value(caller),
                                value_in(
                                    item.callee.closure_id, callee
                                ),
                            )
                            for caller, callee in item.argument_bindings
                        ),
                        tuple(
                            (
                                value_in(
                                    item.callee.closure_id, callee
                                ),
                                value(caller),
                            )
                            for callee, caller in item.result_bindings
                        ),
                    ),
                    tuple(int(loop_id) for loop_id in item.enclosing_loop_ids),
                ))
                continue
            if isinstance(item, PlanClosure) and item.name.startswith(
                "region_"
            ):
                local_region = int(item.name.split("_", 1)[1])
                if local_region not in region_map:
                    # Static/structural region projection removes the marker,
                    # not source calls pending before it.  Carry those calls
                    # forward until a retained runtime region supplies a real
                    # insertion point; if none does, calls_after below keeps
                    # them at the closure boundary.
                    continue
                if pending:
                    calls_before.setdefault(local_region, []).extend(
                        call for call, _loop_ids in pending
                    )
                    pending = []
        calls_at_loop_end: dict[int, list[CallBlock]] = {}
        surviving_loop_ids = _present_loop_ids(local.root)
        for call, loop_ids in pending:
            # ``loop_ids`` is outermost-first; a loop can be fused into a
            # surviving enclosing loop during GLSL scheduling without being
            # marked evaporated, so the innermost id the plan recorded may
            # name a LoopBlock this closure's control program no longer has.
            # Anchor to the innermost enclosing loop that actually survived;
            # only fall back to the closure boundary (like the no-loop case
            # above) if none of them did.
            target_loop_id = next(
                (
                    loop_id
                    for loop_id in reversed(loop_ids)
                    if loop_id in surviving_loop_ids
                ),
                None,
            )
            if target_loop_id is not None:
                calls_at_loop_end.setdefault(target_loop_id, []).append(call)
            else:
                calls_after.append(call)
        remaining_calls_after = list(calls_after)

        def rewrite(block: ControlBlock) -> ControlBlock:
            nonlocal remaining_calls_after
            def rewrite_expression(expression):
                if expression is None:
                    return None
                return ControlExpression(
                    expression.op,
                    tuple(
                        rewrite_expression(operand)
                        for operand in expression.operands
                    ),
                    None if expression.value_id is None
                    else value(expression.value_id),
                    expression.literal,
                )
            def rewrite_mutation(mutation):
                return replace(
                    mutation,
                    sequence_value_id=value(mutation.sequence_value_id),
                    argument_value_ids=tuple(
                        value(value_id)
                        for value_id in mutation.argument_value_ids
                    ),
                    effect_node_id=value(mutation.effect_node_id),
                    predicate_expression=rewrite_expression(
                        mutation.predicate_expression
                    ),
                    argument_expressions=tuple(
                        rewrite_expression(expression)
                        for expression in mutation.argument_expressions
                    ),
                )
            if isinstance(block, StatementBlock):
                local_region = _region_marker(block)
                if local_region is None:
                    return StatementBlock(tuple(
                        rename_control_text(line) for line in block.lines
                    ))
                marker = StatementBlock((
                    f"__scheduled_region_{region_map[local_region]}__",
                ))
                before = tuple(calls_before.get(local_region, ()))
                return (
                    marker
                    if not before
                    else SequenceBlock((*before, marker))
                )
            if isinstance(block, SequenceBlock):
                return SequenceBlock(tuple(
                    rewrite(child) for child in block.blocks
                ))
            if isinstance(block, LoopBlock):
                body = rewrite(block.body)
                loop_id = next((
                    candidate
                    for candidate in calls_at_loop_end
                    if str(block.induction) == f"iteration_{candidate}"
                ), None)
                if loop_id is not None:
                    body = SequenceBlock((
                        body,
                        *calls_at_loop_end.pop(loop_id),
                    ))
                return LoopBlock(
                    global_induction(block.induction),
                    rename_control_text(block.start),
                    rename_control_text(block.stop),
                    rename_control_text(block.step),
                    body,
                    carried_aliases=tuple(
                        (value(updated), value(initial))
                        for updated, initial in block.carried_aliases
                    ),
                    result_ports=tuple(
                        (value(port), value(initial), value(updated))
                        for port, initial, updated in block.result_ports
                    ),
                    carried_seeds=tuple(
                        (value(initial), literal)
                        for initial, literal in block.carried_seeds
                    ),
                    parallel_iterations=block.parallel_iterations,
                    dispatch_shell=block.dispatch_shell,
                    recursion_region_id=(
                        None
                        if block.recursion_region_id is None
                        else recursion_region_map[
                            int(block.recursion_region_id)
                        ]
                    ),
                    schedule_preference=block.schedule_preference,
                    sequence_mutations=tuple(
                        rewrite_mutation(mutation)
                        for mutation in block.sequence_mutations
                    ),
                    comparison=block.comparison,
                    terminal_controls=tuple(
                        rewrite(terminal)
                        for terminal in block.terminal_controls
                    ),
                )
            if isinstance(block, WhileBlock):
                body = rewrite(block.body)
                loop_id = block.source_loop_node_id
                if loop_id is not None and int(loop_id) in calls_at_loop_end:
                    body = SequenceBlock((
                        body,
                        *calls_at_loop_end.pop(int(loop_id)),
                    ))
                return WhileBlock(
                    value(block.predicate_value_id),
                    rewrite(block.condition),
                    body,
                    carried_aliases=tuple(
                        (value(updated), value(initial))
                        for updated, initial in block.carried_aliases
                    ),
                    result_ports=tuple(
                        (value(port), value(initial), value(updated))
                        for port, initial, updated in block.result_ports
                    ),
                    carried_seeds=tuple(
                        (value(initial), literal)
                        for initial, literal in block.carried_seeds
                    ),
                    recursion_region_id=(
                        None if block.recursion_region_id is None
                        else recursion_region_map[int(block.recursion_region_id)]
                    ),
                    predicate_expression=rewrite_expression(
                        block.predicate_expression
                    ),
                    sequence_mutations=tuple(
                        rewrite_mutation(mutation)
                        for mutation in block.sequence_mutations
                    ),
                    source_loop_node_id=block.source_loop_node_id,
                    terminal_controls=tuple(
                        rewrite(terminal)
                        for terminal in block.terminal_controls
                    ),
                )
            if isinstance(block, LoopControlBlock):
                return LoopControlBlock(
                    block.action,
                    None if block.predicate_value_id is None
                    else value(block.predicate_value_id),
                    block.expect_true,
                    rewrite_expression(block.predicate_expression),
                    block.source_action,
                )
            if isinstance(block, StateMachineTick):
                return StateMachineTick(
                    rename_control_text(block.state),
                    tuple(
                        (rename_control_text(case), rewrite(body))
                        for case, body in block.cases
                    ),
                    None if block.default is None else rewrite(block.default),
                )
            if isinstance(block, ParallelDeployment):
                return ParallelDeployment(
                    tuple(rewrite(lane) for lane in block.lanes),
                    block.schedule_preference,
                )
            if isinstance(block, CallBlock):
                return CallBlock(
                    block.callsite_id,
                    rewrite(block.callee),
                    block.argument_bindings,
                    block.result_bindings,
                )
            if isinstance(block, ValidationBlock):
                return ValidationBlock(
                    value(block.predicate_value_id),
                    block.error_code,
                    block.expect_true,
                    rewrite_expression(block.predicate_expression),
                    block.extraction_identity,
                )
            if isinstance(block, StreamPublishBlock):
                publication = StreamPublishBlock(
                    # Stream identity follows the globally namespaced published
                    # SSA value.  Local stream number zero is common in every
                    # closure and cannot survive whole-program composition.
                    value(block.value_id),
                    value(block.value_id),
                    (
                        None
                        if block.count_value_id is None
                        else value(block.count_value_id)
                    ),
                    (
                        None
                        if block.predicate_value_id is None
                        else value(block.predicate_value_id)
                    ),
                    block.final,
                )
                if remaining_calls_after:
                    # A call may be the final numerical action in a lexical
                    # loop, followed only by a structural publication.  There
                    # is then no later region marker at which to insert it.
                    # Appending the call outside the rewritten root executes
                    # the publication first (and, for loops, outside its
                    # scope), exposing the zero-initialized result.  The first
                    # following publication is the real control anchor: keep
                    # all still-pending calls immediately before it, inside
                    # the same loop/state-machine compartment.
                    before = tuple(remaining_calls_after)
                    remaining_calls_after = []
                    return SequenceBlock((*before, publication))
                return publication
            raise TypeError(type(block).__name__)

        root = rewrite(local.root)
        if calls_at_loop_end:
            raise ValueError(
                "planned calls reference enclosing loops absent from "
                f"closure control: {tuple(sorted(calls_at_loop_end))!r}"
            )
        if remaining_calls_after:
            root = SequenceBlock((root, *remaining_calls_after))
        return ControlProgram(
            root=root,
            region_indices=tuple(
                region_map[region] for region in local.region_indices
            ),
            recursion_regions=tuple(
                region
                for region in all_recursion_regions
                if region.region_id in recursion_region_map.values()
            ),
        )

    program = compose(hierarchy)
    ordered_regions: list[int] = []

    def collect_regions(block: ControlBlock) -> None:
        if isinstance(block, StatementBlock):
            region = _region_marker(block)
            if region is not None:
                ordered_regions.append(region)
            return
        if isinstance(block, SequenceBlock):
            for child in block.blocks:
                collect_regions(child)
            return
        if isinstance(block, LoopBlock):
            collect_regions(block.body)
            return
        if isinstance(block, WhileBlock):
            collect_regions(block.condition)
            collect_regions(block.body)
            return
        if isinstance(block, LoopControlBlock):
            return
        if isinstance(block, StateMachineTick):
            for _case, body in block.cases:
                collect_regions(body)
            if block.default is not None:
                collect_regions(block.default)
            return
        if isinstance(block, ParallelDeployment):
            for lane in block.lanes:
                collect_regions(lane)
            return
        if isinstance(block, CallBlock):
            collect_regions(block.callee)
            return
        if isinstance(block, ValidationBlock):
            return
        if isinstance(block, StreamPublishBlock):
            return
        raise TypeError(type(block).__name__)

    collect_regions(program.root)
    program = replace(
        program,
        region_indices=tuple(ordered_regions),
        uniforms=tuple(dict.fromkeys(all_uniforms)),
        value_aliases=tuple(dict.fromkeys(all_aliases)),
        iterable_bindings=tuple(dict.fromkeys(all_iterables)),
        static_iterable_bindings=tuple(
            dict.fromkeys(all_static_iterables)
        ),
        collection_bindings=tuple(dict.fromkeys(all_collections)),
        closure_iterable_bindings=tuple(
            dict.fromkeys(all_closure_iterables)
        ),
        projected_iterable_bindings=tuple(
            dict.fromkeys(all_projected_iterables)
        ),
        recursion_regions=tuple(dict.fromkeys(all_recursion_regions)),
        deployment_regions=tuple(all_deployment_regions),
    )
    return HierarchicalControl(program, tuple(region_correlations))


__all__ = ["HierarchicalControl", "compose_hierarchical_control"]
